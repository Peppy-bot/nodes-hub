//! The depth frames into the grid. One subscription receives every camera's
//! frames, each stamped with its arrival on the node's clock and handed to its
//! sensor's intake as that sensor's newest frame. An intake takes its newest
//! frame once it is free: it holds the frame's header to the camera's
//! alignment and frame, places the camera by a lookup of the frame the header
//! names in the robot frame, and the robot frame by a lookup of it in odom,
//! both at the header's timestamp, as sensor_mounts asks a consumer to place a
//! measurement, waiting up to the lookup timeout for the robot's tree; then it
//! inserts the frame's points into the grid, centered on the base there while
//! a goal runs or the release burst after it streams, which notes the frame's
//! arrival for the stop. A frame the tree cannot place is dropped and logged
//! with the lookup's failure kind, unknown_frame among them.

use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use control_core::geometry::{Point, Pose};
use control_core::latched_log::LatchedLog;
use control_core::node_clock::NodeClock;
use control_core::receiving::receive;
use control_core::robot_tree::RobotTree;
use control_core::sync::lock_ignoring_poison;
use frame_tree::{LookupError, LookupTime, Stamp};
use occupancy_map::depth_points;
use peppygen::NodeRunner;
use peppygen::consumed_topics::cameras::depth_stream;
use tokio::sync::watch;
use tracing::error;

use crate::cameras::{Camera, Sensors};
use crate::clock;
use crate::follow::{odometry_frame, robot_frame};
use crate::grid::GridSlot;

/// One frame and its arrival (ns) on the node's clock.
#[derive(Clone, Debug)]
pub struct Arrived {
    pub arrived_ns: i64,
    pub frame: Arc<depth_stream::Message>,
}

/// Receive every camera's depth frames until the node cancels, handing each
/// to `newest`, by the sensor that sent it, stamped with its arrival on
/// `clock`.
pub async fn run_listener(
    runner: Arc<NodeRunner>,
    clock: NodeClock,
    sensors: Arc<Sensors>,
    newest: Vec<watch::Sender<Option<Arrived>>>,
) {
    let mut subscription = match depth_stream::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => return error!("depth_stream subscribe: {e}"),
    };
    receive(
        "depth_stream",
        runner.cancellation_token().cancelled(),
        async move || subscription.next().await,
        async move |(producer, frame): (peppygen::ProducerRef, depth_stream::Message)| {
            if let Some(source) = sensors.source_of(&producer) {
                newest[source].send_replace(Some(Arrived {
                    arrived_ns: clock::now_ns(&clock),
                    frame: Arc::new(frame),
                }));
            }
            ControlFlow::Continue(())
        },
    )
    .await;
}

/// The message of a lookup that failed, led by its failure kind, as
/// sensor_mounts asks a consumer to log a frame it cannot place.
fn unplaced(what: &str, error: &LookupError) -> String {
    format!("{}: {what}: {error}", error.failure_kind())
}

/// The intake of the sensor at `source`.
pub struct Intake {
    pub source: usize,
    pub sensors: Arc<Sensors>,
    pub tree: Arc<RobotTree>,
    pub grid: Arc<Mutex<GridSlot>>,
    pub lookup_timeout: Duration,
}

impl Intake {
    /// Insert `arrived`, a frame of `camera`, into the grid, or why it is
    /// dropped.
    async fn take(&self, camera: &Camera, arrived: &Arrived) -> Result<(), String> {
        let frame = &arrived.frame;
        let header = &frame.header;
        camera
            .depth
            .admits_frame(&header.align_mode, &header.frame)
            .map_err(|e| e.to_string())?;
        let stamp = Stamp::from_time(header.timestamp)
            .map_err(|e| format!("the frame's timestamp: {e}"))?;
        let at = LookupTime::At(stamp);
        let base = self
            .tree
            .lookup_within(&odometry_frame(), &robot_frame(), at, self.lookup_timeout)
            .await
            .map_err(|e| unplaced("the robot frame at the frame's timestamp", &e))?
            .placement;
        let base = Point::parse(base.position())
            .map(|position| Pose::new(position, base.orientation()))
            .and_then(Pose::planar)
            .map_err(|e| format!("the robot frame in odom at the frame's timestamp: {e}"))?;
        let color_in_robot = self
            .tree
            .lookup_within(&robot_frame(), &camera.depth.frame, at, self.lookup_timeout)
            .await
            .map_err(|e| unplaced("the camera's frame at the frame's timestamp", &e))?
            .placement
            .transform();
        let sensor_on_base = camera.depth.depth_in_robot(&color_in_robot);
        tokio::task::block_in_place(|| {
            let points = depth_points(
                &frame.data,
                &camera.depth.lens,
                camera.depth_unit.get(),
                camera.depth.depth_model,
            )
            .map_err(|e| e.to_string())?;
            let mut slot = lock_ignoring_poison(&self.grid);
            let grid = slot
                .grid_for_reading(base)
                .map_err(|e| format!("the grid centered on the base: {e}"))?
                .ok_or("the grid waits for describe_base's footprint")?;
            grid.insert_depth_frame(
                self.source,
                &points,
                base,
                sensor_on_base,
                camera.range,
                arrived.arrived_ns,
            )
            .map_err(|e| e.to_string())
        })
    }

    /// Take the newest frame of `frames` into the grid each time one arrives
    /// and the intake is free, until the node cancels.
    pub async fn run(self, mut frames: watch::Receiver<Option<Arrived>>) {
        let name = self.sensors.names()[self.source].to_owned();
        let dropped = LatchedLog::warn(format!("camera {name} frame intake"));
        while frames.changed().await.is_ok() {
            let Some(arrived) = frames.borrow_and_update().clone() else {
                continue;
            };
            self.handle(&arrived, &dropped).await;
        }
    }

    /// Take `arrived` into the grid, or log and record why it is dropped.
    async fn handle(&self, arrived: &Arrived, dropped: &LatchedLog) {
        let Some(camera) = self.sensors.ready(self.source) else {
            return dropped.failure("a frame before the camera's description");
        };
        match self.take(&camera, arrived).await {
            Ok(()) => {
                dropped.success();
                self.sensors.frame_handled(self.source, None);
            }
            Err(reason) => {
                dropped.failure(format!("dropping a frame: {reason}"));
                self.sensors.frame_handled(self.source, Some(reason));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::SystemTime;

    use camera_core::depth_camera::{DepthExtrinsics, DepthIntrinsics, read_depth_camera};
    use control_core::node_clock::NodeClock;
    use control_core::positive_finite::PositiveFinite;
    use frame_tree::{MovingMessage, Publisher, StaticMessage, TransformRecord};
    use navigation_core::grid::LETHAL;

    use crate::cameras::tests::camera_slot;
    use crate::cameras::{DepthUnit, SensorState};
    use crate::follow::ROBOT_FRAME;
    use crate::grid::tests::grid;

    const CAMERA_FRAME: &str = "front_color_optical_frame";

    fn at(seconds_value: f64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs_f64(seconds_value)
    }

    /// A depth camera of 4 by 4 pixels, its depth frame its colour frame.
    fn camera() -> Camera {
        let depth = read_depth_camera(
            &DepthIntrinsics {
                width: 4,
                height: 4,
                fx: 8.0,
                fy: 8.0,
                cx: 1.5,
                cy: 1.5,
                distortion_model: "none",
                distortion: &[],
                depth_model: "z",
                align_mode: "none",
            },
            &DepthExtrinsics {
                frame: CAMERA_FRAME,
                position: [0.0; 3],
                orientation: [0.0, 0.0, 0.0, 1.0],
                align_mode: "none",
            },
        )
        .expect("a depth camera");
        Camera {
            depth,
            depth_unit: DepthUnit::parse(0.001).expect("a positive depth unit"),
            range: PositiveFinite::parse(4.0).unwrap(),
        }
    }

    /// The tree of a camera 0.3 m above the robot frame's origin, looking
    /// along +x, and of the robot frame at odom's origin from 10 s to 11 s.
    fn tree() -> Arc<RobotTree> {
        let tree = RobotTree::new(Duration::from_secs(10), NodeClock::wall()).expect("a window");
        let publisher = |instance: &str| Publisher {
            core_node: "core".to_owned(),
            instance_id: instance.to_owned(),
        };
        tree.take_static(
            &publisher("initializer"),
            &StaticMessage {
                transforms: vec![TransformRecord::new(
                    ROBOT_FRAME,
                    CAMERA_FRAME,
                    [0.0, 0.0, 0.3],
                    [-0.5, 0.5, -0.5, 0.5],
                )],
            },
        )
        .expect("the camera's mount");
        for stamp in [10.0, 11.0] {
            tree.take_moving(
                &publisher("backbone"),
                &MovingMessage {
                    timestamp: at(stamp),
                    transforms: vec![TransformRecord::new(
                        "odom",
                        ROBOT_FRAME,
                        [0.0; 3],
                        [0.0, 0.0, 0.0, 1.0],
                    )],
                    start_timestamp: at(1.0),
                },
            )
            .expect("the odometry");
        }
        Arc::new(tree)
    }

    /// A depth frame of the camera at `stamp_s`, naming `frame`, every sample
    /// 1.02 m deep.
    fn frame(frame: &str, stamp_s: f64) -> Arrived {
        Arrived {
            arrived_ns: 0,
            frame: Arc::new(depth_stream::Message {
                header: depth_stream::MessageHeader {
                    frame: frame.to_owned(),
                    timestamp: at(stamp_s),
                    capture_index: 0,
                    align_mode: "none".to_owned(),
                },
                encoding: "z16".to_owned(),
                width: 4,
                height: 4,
                data: 1020_u16.to_le_bytes().repeat(16),
            }),
        }
    }

    fn intake() -> Intake {
        let sensors = Sensors::new(vec![camera_slot("front")]);
        sensors.set(0, SensorState::Ready(Arc::new(camera())));
        let mut slot = GridSlot::default();
        slot.set(grid());
        Intake {
            source: 0,
            sensors: Arc::new(sensors),
            tree: tree(),
            grid: Arc::new(Mutex::new(slot)),
            lookup_timeout: Duration::ZERO,
        }
    }

    // A frame placed by the tree at its timestamp marks the wall it sees 1 m
    // ahead of the base; one whose header names another frame, and one
    // stamped where the odometry holds no transform, are dropped with why.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_frame_placed_at_its_timestamp_marks_what_the_camera_sees() {
        let intake = intake();
        let camera = camera();
        intake
            .take(&camera, &frame(CAMERA_FRAME, 10.5))
            .await
            .expect("a frame the tree places");
        let lethal: Vec<[f64; 2]> = {
            let mut slot = lock_ignoring_poison(&intake.grid);
            let occupancy = slot
                .grid_mut()
                .expect("a grid")
                .occupancy()
                .expect("an occupancy");
            let geometry = occupancy.geometry();
            geometry
                .cells()
                .filter(|&cell| occupancy.cost(cell) == LETHAL)
                .map(|cell| geometry.center_of(cell))
                .collect()
        };
        // The four columns of rays, 1.02 m deep, stand 0.064 m and 0.191 m to
        // each side of the camera's axis: the cells 1 m to 1.05 m ahead whose
        // centers lie 0.075 m and 0.175 m to each side.
        let expected = [
            [1.025, -0.175],
            [1.025, -0.075],
            [1.025, 0.075],
            [1.025, 0.175],
        ];
        assert_eq!(lethal.len(), expected.len(), "{lethal:?}");
        for (cell, center) in lethal.iter().zip(expected) {
            assert!(
                (cell[0] - center[0]).abs() < 1e-9 && (cell[1] - center[1]).abs() < 1e-9,
                "{lethal:?}"
            );
        }
        let other = intake
            .take(&camera, &frame("rear_color_optical_frame", 10.5))
            .await;
        assert!(other.is_err_and(|why| why.contains("rear_color_optical_frame")));
        let late = intake.take(&camera, &frame(CAMERA_FRAME, 12.0)).await;
        assert!(
            matches!(&late, Err(why) if why.starts_with("extrapolation: ")),
            "{late:?}"
        );
    }

    // A camera whose frames arrive and are dropped is waited for as such,
    // with why, and as a reading once a frame of it goes into the grid.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_camera_whose_frames_are_dropped_is_waited_for_with_why() {
        let intake = intake();
        let dropped = LatchedLog::warn("the test's intake");
        intake
            .handle(&frame("rear_color_optical_frame", 10.5), &dropped)
            .await;
        let waits = intake.sensors.waits(&[0], 0.5);
        assert!(
            waits.contains("arrive and are dropped") && waits.contains("rear_color_optical_frame"),
            "{waits}"
        );
        intake.handle(&frame(CAMERA_FRAME, 10.5), &dropped).await;
        assert_eq!(
            intake.sensors.waits(&[0], 0.5),
            "a reading of the camera front within source_timeout_s 0.5 s"
        );
    }
}
