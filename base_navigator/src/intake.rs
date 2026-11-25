//! The readings into the grid: each depth camera's frames and each 2-D lidar's scans. One
//! subscription a slot receives the readings of every sensor of that slot, each stamped with
//! its arrival on the node's clock and handed to its sensor's intake as that sensor's newest
//! reading. An intake takes its newest reading once it is free and places it by the robot's
//! tree, as sensor_mounts asks a consumer to place a measurement, each lookup waiting up to
//! the lookup timeout for the tree:
//!
//! - a depth frame, held to the camera's alignment and frame: the camera by a lookup of the
//!   frame the header names in the robot frame, and the robot frame by a lookup of it in odom,
//!   both at the header's timestamp;
//! - a scan, whose header must name the frame describe_sensors gives the lidar: the robot
//!   frame by a lookup of it in odom at the scan's first ray, its timestamp, and at its last
//!   ray, before the first for a lidar that sweeps clockwise, and the lidar by a lookup of its
//!   frame in the robot frame at the first ray; navigation_core places each ray where the
//!   lidar stood at the ray's time, and the scan's own farthest range bounds its rays.
//!
//! The intake then inserts the reading into the grid, centered on the base where it stood at
//! the reading's time, a scan's first ray, while a goal runs or the release burst after it
//! streams; the grid notes the reading's arrival for the stop. A reading the tree cannot place
//! goes into none and is logged with the lookup's failure kind, unknown_frame among them, and
//! its sensor records why.

use std::collections::BTreeMap;
use std::fmt::Display;
use std::future::Future;
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use camera_core::laser_scan::{LaserScan, Sweep};
use control_core::latched_log::LatchedLog;
use control_core::node_clock::NodeClock;
use control_core::planar_pose::PlanarPose;
use control_core::positive_finite::PositiveFinite;
use control_core::receiving::receive;
use control_core::robot_tree::RobotTree;
use control_core::sync::lock_ignoring_poison;
use control_core::time::nanoseconds;
use frame_tree::{FrameName, LookupTime, Stamp, Transform};
use occupancy_map::depth_points;
use peppygen::consumed_topics::cameras::depth_stream;
use peppygen::consumed_topics::lidars::scan_stream;
use peppygen::{NodeRunner, ProducerRef};
use tokio::sync::watch;
use tracing::error;

use crate::cameras::Camera;
use crate::clock;
use crate::follow::{robot_frame, robot_in_odom, unplaced};
use crate::grid::{GridSlot, LiveGrid};
use crate::sensors::Sensors;

/// One reading and its arrival (ns) on the node's clock.
#[derive(Clone)]
pub struct Arrived<M> {
    pub arrived_ns: i64,
    pub reading: Arc<M>,
}

/// Where the newest reading of each sensor of one slot goes, by the sensor's source.
pub type Newest<M> = BTreeMap<usize, watch::Sender<Option<Arrived<M>>>>;

/// Receive the readings `next` gives until `cancelled` resolves or the subscription of
/// `topic` closes, handing each to `newest` by the sensor that sent it, stamped with its
/// arrival on `clock`.
async fn hand_out<M, E: Display>(
    topic: &str,
    cancelled: impl Future<Output = ()>,
    clock: NodeClock,
    sensors: Arc<Sensors>,
    newest: Newest<M>,
    next: impl AsyncFnMut() -> Result<Option<(ProducerRef, M)>, E>,
) {
    receive(
        topic,
        cancelled,
        next,
        async move |(producer, reading): (ProducerRef, M)| {
            let sender = sensors
                .source_of(&producer)
                .and_then(|source| newest.get(&source));
            if let Some(sender) = sender {
                sender.send_replace(Some(Arrived {
                    arrived_ns: clock::now_ns(&clock),
                    reading: Arc::new(reading),
                }));
            }
            ControlFlow::Continue(())
        },
    )
    .await;
}

/// Receive every camera's depth frames until the node cancels, handing each to `newest`, by
/// the camera that sent it, stamped with its arrival on `clock`.
pub async fn run_depth_listener(
    runner: Arc<NodeRunner>,
    clock: NodeClock,
    sensors: Arc<Sensors>,
    newest: Newest<depth_stream::Message>,
) {
    let mut subscription = match depth_stream::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => return error!("depth_stream subscribe: {e}"),
    };
    hand_out(
        "depth_stream",
        runner.cancellation_token().cancelled(),
        clock,
        sensors,
        newest,
        async move || subscription.next().await,
    )
    .await;
}

/// Receive every lidar's scans until the node cancels, handing each to `newest`, by the
/// lidar that sent it, stamped with its arrival on `clock`.
pub async fn run_scan_listener(
    runner: Arc<NodeRunner>,
    clock: NodeClock,
    sensors: Arc<Sensors>,
    newest: Newest<scan_stream::Message>,
) {
    let mut subscription = match scan_stream::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => return error!("scan_stream subscribe: {e}"),
    };
    hand_out(
        "scan_stream",
        runner.cancellation_token().cancelled(),
        clock,
        sensors,
        newest,
        async move || subscription.next().await,
    )
    .await;
}

/// The newest reading `readings` holds once one arrives that the intake has not taken; none
/// once the listener is gone.
async fn next_newest<M: Clone>(
    readings: &mut watch::Receiver<Option<Arrived<M>>>,
) -> Option<Arrived<M>> {
    loop {
        readings.changed().await.ok()?;
        let newest = readings.borrow_and_update().clone();
        if newest.is_some() {
            return newest;
        }
    }
}

/// The stamp `offset_s` after `stamp`, before it for an offset below 0; none outside the
/// times a stamp holds.
fn stamp_after(stamp: Stamp, offset_s: f64) -> Option<Stamp> {
    stamp
        .as_nanos()
        .checked_add_signed(nanoseconds(offset_s))
        .map(Stamp::from_nanos)
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
    /// The robot frame in odom at `stamp`, on the floor plane; `when` names the time in a
    /// refusal.
    async fn base_at(&self, stamp: Stamp, when: &str) -> Result<PlanarPose, String> {
        robot_in_odom(&self.tree, stamp, self.lookup_timeout, when).await
    }

    /// `frame` in the robot frame at `stamp`; `what` names the lookup in a refusal.
    async fn in_robot_frame(
        &self,
        frame: &FrameName,
        stamp: Stamp,
        what: &str,
    ) -> Result<Transform, String> {
        let found = self
            .tree
            .lookup_within(
                &robot_frame(),
                frame,
                LookupTime::At(stamp),
                self.lookup_timeout,
            )
            .await
            .map_err(|e| unplaced(what, &e))?;
        Ok(found.placement.transform())
    }

    /// Insert into the grid, by `insert`, a reading taken with the base at `base` in odom,
    /// the grid centered on the base there while it rolls.
    fn insert_reading(
        &self,
        base: PlanarPose,
        insert: impl FnOnce(&mut LiveGrid) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut slot = lock_ignoring_poison(&self.grid);
        let grid = slot
            .grid_for_reading(base)
            .map_err(|e| format!("the grid centered on the base: {e}"))?
            .ok_or("the grid waits for describe_base's footprint")?;
        insert(grid)
    }

    /// Insert `arrived`, a frame of `camera`, into the grid, or why it is
    /// dropped.
    async fn take_depth_frame(
        &self,
        camera: &Camera,
        arrived: &Arrived<depth_stream::Message>,
    ) -> Result<(), String> {
        let frame = &arrived.reading;
        let header = &frame.header;
        camera
            .depth
            .admits_frame(&header.align_mode, &header.frame)
            .map_err(|e| e.to_string())?;
        let stamp = Stamp::from_time(header.timestamp)
            .map_err(|e| format!("the frame's timestamp: {e}"))?;
        let base = self.base_at(stamp, "the frame's timestamp").await?;
        let color_in_robot = self
            .in_robot_frame(
                &camera.depth.frame,
                stamp,
                "the camera's frame at the frame's timestamp",
            )
            .await?;
        let sensor_on_base = camera.depth.depth_in_robot(&color_in_robot);
        tokio::task::block_in_place(|| {
            let points = depth_points(
                &frame.data,
                &camera.depth.lens,
                camera.depth_unit.get(),
                camera.depth.depth_model,
            )
            .map_err(|e| e.to_string())?;
            self.insert_reading(base, |grid| {
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
        })
    }

    /// Insert `arrived`, a scan of the lidar to which describe_sensors gives the frame
    /// `lidar`, into the grid, or why it is dropped.
    async fn take_scan(
        &self,
        lidar: &FrameName,
        arrived: &Arrived<scan_stream::Message>,
    ) -> Result<(), String> {
        let message = &arrived.reading;
        let header = &message.header;
        if header.frame != lidar.as_str() {
            return Err(format!(
                "the scan names the frame {:?}, where sensor_mounts' describe_sensors gives \
                 the lidar the frame {:?}: run the lidar under the name of the sensor whose \
                 frame its scans name",
                header.frame,
                lidar.as_str()
            ));
        }
        let scan = LaserScan::parse(
            message.min_angle_rad,
            message.angle_increment_rad,
            message.time_increment_s,
            message.min_range_m,
            message.max_range_m,
            &message.ranges,
        )
        .map_err(|e| e.to_string())?;
        let first_ray =
            Stamp::from_time(header.timestamp).map_err(|e| format!("the scan's timestamp: {e}"))?;
        let last_ray = stamp_after(first_ray, scan.sweep_s()).ok_or_else(|| {
            format!(
                "the scan's last ray, {} s after its timestamp {first_ray}, lies outside the \
                 times a stamp holds",
                scan.sweep_s()
            )
        })?;
        let base = Sweep {
            first_ray: self.base_at(first_ray, "the scan's first ray").await?,
            last_ray: self.base_at(last_ray, "the scan's last ray").await?,
        };
        let lidar_on_base = self
            .in_robot_frame(
                lidar,
                first_ray,
                "the lidar's frame at the scan's first ray",
            )
            .await?;
        let range = PositiveFinite::parse(f64::from(scan.max_range_m()))
            .expect("a scan's max_range_m lies above its min_range_m, 0 or above, and below 10000");
        tokio::task::block_in_place(|| {
            self.insert_reading(base.first_ray, |grid| {
                grid.insert_laser_scan(
                    self.source,
                    &scan,
                    base,
                    lidar_on_base,
                    range,
                    arrived.arrived_ns,
                )
                .map_err(|e| e.to_string())
            })
        })
    }

    /// The log of the readings this intake drops, by its sensor's noun and name.
    fn dropped_log(&self) -> LatchedLog {
        let slot = &self.sensors.slots()[self.source];
        LatchedLog::warn(format!(
            "{} {} {} intake",
            slot.kind.class().noun,
            slot.name,
            slot.kind.reading()
        ))
    }

    /// Log and record what came of a reading: `taken` went into the grid, or why it is
    /// dropped.
    fn record(&self, taken: Result<(), String>, dropped: &LatchedLog) {
        match taken {
            Ok(()) => {
                dropped.success();
                self.sensors.reading_handled(self.source, None);
            }
            Err(reason) => {
                let reading = self.sensors.slots()[self.source].kind.reading();
                dropped.failure(format!("dropping a {reading}: {reason}"));
                self.sensors.reading_handled(self.source, Some(reason));
            }
        }
    }

    /// Take `arrived`, a depth frame, into the grid, or log and record why it is dropped.
    async fn handle_depth_frame(
        &self,
        arrived: &Arrived<depth_stream::Message>,
        dropped: &LatchedLog,
    ) {
        let Some(camera) = self.sensors.camera(self.source) else {
            return dropped.failure("a frame before the camera's description");
        };
        self.record(self.take_depth_frame(&camera, arrived).await, dropped);
    }

    /// Take `arrived`, a scan, into the grid, or log and record why it is dropped.
    async fn handle_scan(&self, arrived: &Arrived<scan_stream::Message>, dropped: &LatchedLog) {
        let Some(lidar) = self.sensors.lidar(self.source) else {
            return dropped.failure("a scan before describe_sensors gives the lidar's frame");
        };
        self.record(self.take_scan(&lidar, arrived).await, dropped);
    }

    /// Take the newest frame of `frames` into the grid each time one arrives and the intake
    /// is free, until the node cancels.
    pub async fn run_depth_frames(
        self,
        mut frames: watch::Receiver<Option<Arrived<depth_stream::Message>>>,
    ) {
        let dropped = self.dropped_log();
        while let Some(arrived) = next_newest(&mut frames).await {
            self.handle_depth_frame(&arrived, &dropped).await;
        }
    }

    /// Take the newest scan of `scans` into the grid each time one arrives and the intake is
    /// free, until the node cancels.
    pub async fn run_scans(
        self,
        mut scans: watch::Receiver<Option<Arrived<scan_stream::Message>>>,
    ) {
        let dropped = self.dropped_log();
        while let Some(arrived) = next_newest(&mut scans).await {
            self.handle_scan(&arrived, &dropped).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeSet;
    use std::time::SystemTime;

    use camera_core::depth_camera::{DepthExtrinsics, DepthIntrinsics, read_depth_camera};
    use control_core::node_clock::NodeClock;
    use frame_tree::arithmetic::quaternion_of_yaw;
    use frame_tree::{MovingMessage, Publisher, StaticMessage, TransformRecord};
    use navigation_core::grid::{CostGrid, LETHAL};

    use crate::cameras::DepthUnit;
    use crate::follow::ROBOT_FRAME;
    use crate::grid::tests::grid;
    use crate::sensors::tests::{camera_slot, lidar_slot, sensors_of};
    use crate::sensors::{Described, SensorSlot, SensorState};

    const CAMERA_FRAME: &str = "front_color_optical_frame";

    const LIDAR_FRAME: &str = "lidar_frame";

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

    fn publisher(instance: &str) -> Publisher {
        Publisher {
            core_node: "core".to_owned(),
            instance_id: instance.to_owned(),
        }
    }

    /// The tree of `mount`, a sensor's frame fixed to the robot frame, and of the robot
    /// frame at odom's origin turned by each heading (rad) of `odometry` at its time (s).
    fn tree(mount: TransformRecord, odometry: &[(f64, f64)]) -> Arc<RobotTree> {
        let tree = RobotTree::new(Duration::from_secs(10), NodeClock::wall()).expect("a window");
        tree.take_static(
            &publisher("initializer"),
            &StaticMessage {
                transforms: vec![mount],
            },
        )
        .expect("the sensor's mount");
        for &(stamp_s, yaw) in odometry {
            tree.take_moving(
                &publisher("backbone"),
                &MovingMessage {
                    timestamp: at(stamp_s),
                    transforms: vec![TransformRecord::new(
                        "odom",
                        ROBOT_FRAME,
                        [0.0; 3],
                        quaternion_of_yaw(yaw),
                    )],
                    start_timestamp: at(1.0),
                },
            )
            .expect("the odometry");
        }
        Arc::new(tree)
    }

    /// The intake of `slot`, the one sensor, described as `described`, on the grid of the
    /// tests' parameters around odom's origin, placed by `tree`.
    fn intake_of(slot: SensorSlot, described: Described, tree: Arc<RobotTree>) -> Intake {
        let sensors = sensors_of(vec![slot]);
        sensors.set(0, SensorState::Ready(described));
        let mut grid_slot = GridSlot::default();
        grid_slot.set(grid());
        Intake {
            source: 0,
            sensors: Arc::new(sensors),
            tree,
            grid: Arc::new(Mutex::new(grid_slot)),
            lookup_timeout: Duration::ZERO,
        }
    }

    /// The occupancy of the intake's grid as it stands.
    fn occupancy(intake: &Intake) -> Arc<CostGrid> {
        lock_ignoring_poison(&intake.grid)
            .grid_mut()
            .expect("a grid")
            .occupancy(None)
            .expect("an occupancy")
    }

    /// The intake of a camera 0.3 m above the robot frame's origin, looking along +x, the
    /// robot frame at odom's origin from 10 s to 11 s.
    fn intake() -> Intake {
        let mount = TransformRecord::new(
            ROBOT_FRAME,
            CAMERA_FRAME,
            [0.0, 0.0, 0.3],
            [-0.5, 0.5, -0.5, 0.5],
        );
        intake_of(
            camera_slot("front"),
            Described::DepthCamera(Arc::new(camera())),
            tree(mount, &[(10.0, 0.0), (11.0, 0.0)]),
        )
    }

    /// A depth frame of the camera at `stamp_s`, naming `frame`, every sample
    /// 1.02 m deep.
    fn frame(frame: &str, stamp_s: f64) -> Arrived<depth_stream::Message> {
        Arrived {
            arrived_ns: 0,
            reading: Arc::new(depth_stream::Message {
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

    // A frame placed by the tree at its timestamp marks the wall it sees 1 m
    // ahead of the base; one whose header names another frame, and one
    // stamped where the odometry holds no transform, are dropped with why.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_frame_placed_at_its_timestamp_marks_what_the_camera_sees() {
        let intake = intake();
        let camera = camera();
        intake
            .take_depth_frame(&camera, &frame(CAMERA_FRAME, 10.5))
            .await
            .expect("a frame the tree places");
        let lethal: Vec<[f64; 2]> = {
            let occupancy = occupancy(&intake);
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
            .take_depth_frame(&camera, &frame("rear_color_optical_frame", 10.5))
            .await;
        assert!(other.is_err_and(|why| why.contains("rear_color_optical_frame")));
        let late = intake
            .take_depth_frame(&camera, &frame(CAMERA_FRAME, 12.0))
            .await;
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
            .handle_depth_frame(&frame("rear_color_optical_frame", 10.5), &dropped)
            .await;
        let waits = intake.sensors.waits(&[0], 0.5);
        assert!(
            waits.contains("arrive and are dropped") && waits.contains("rear_color_optical_frame"),
            "{waits}"
        );
        intake
            .handle_depth_frame(&frame(CAMERA_FRAME, 10.5), &dropped)
            .await;
        assert_eq!(
            intake.sensors.waits(&[0], 0.5),
            "a reading of the depth camera front within source_timeout_s 0.5 s"
        );
    }

    /// The wall's distance (m) along odom's x from the lidar: its cells stand 1 m to 1.05 m
    /// ahead, and every ray of [`wall_scan`] meets it at least 2 mm from a cell's edge.
    const WALL_X: f64 = 1.012;

    /// The scan's rays: the first one's angle (rad), the step from one to the next (rad), and
    /// how many.
    const FIRST_RAY_RAD: f32 = -0.47;
    const RAY_STEP_RAD: f32 = 0.05;
    const RAYS: usize = 21;

    /// The time from one ray to the next (s), 2^-8 s, so the sweep's 20 steps take
    /// 0.078125 s, whole nanoseconds.
    const RAY_TIME_S: f32 = 0.003_906_25;

    /// How far the base turns in place while the lidar sweeps (rad), counterclockwise.
    const SWEEP_TURN_RAD: f64 = 0.5;

    /// When the sweep starts (s): the odometry holds the robot frame at odom's origin
    /// heading +x then, and turned [`SWEEP_TURN_RAD`] when the sweep ends.
    const SWEEP_STARTS_S: f64 = 10.0;

    /// How long the sweep takes (s): `RAYS - 1` steps of [`RAY_TIME_S`].
    fn sweep_s() -> f64 {
        f64::from(RAY_TIME_S) * (RAYS - 1) as f64
    }

    /// The intake of a lidar 0.32 m above the robot frame's origin, heading along it, while
    /// the base turns [`SWEEP_TURN_RAD`] in place over one sweep.
    fn lidar_intake() -> Intake {
        let mount = TransformRecord::new(
            ROBOT_FRAME,
            LIDAR_FRAME,
            [0.0, 0.0, 0.32],
            [0.0, 0.0, 0.0, 1.0],
        );
        let odometry = [
            (SWEEP_STARTS_S, 0.0),
            (SWEEP_STARTS_S + sweep_s(), SWEEP_TURN_RAD),
        ];
        intake_of(
            lidar_slot("lidar"),
            Described::Lidar(FrameName::parse(LIDAR_FRAME).expect("a frame")),
            tree(mount, &odometry),
        )
    }

    /// Ray `index`'s direction in odom (rad): its angle in the lidar's frame turned by the
    /// base's heading at the ray's time. Counterclockwise the rays come in the order the
    /// lidar measures them, ray 0 at the sweep's start; clockwise in the reverse order, ray
    /// 0 at its end.
    fn pointing(index: usize, counterclockwise: bool) -> f64 {
        let share = index as f64 / (RAYS - 1) as f64;
        let measured = if counterclockwise { share } else { 1.0 - share };
        f64::from(FIRST_RAY_RAD)
            + index as f64 * f64::from(RAY_STEP_RAD)
            + SWEEP_TURN_RAD * measured
    }

    /// The scan the lidar measures of the wall [`WALL_X`] ahead along odom's x while the
    /// base turns, swept counterclockwise or clockwise, its header naming `frame`.
    fn wall_scan(counterclockwise: bool, frame: &str) -> Arrived<scan_stream::Message> {
        let (stamp_s, time_increment_s) = if counterclockwise {
            (SWEEP_STARTS_S, RAY_TIME_S)
        } else {
            (SWEEP_STARTS_S + sweep_s(), -RAY_TIME_S)
        };
        let ranges = (0..RAYS)
            .map(|index| (WALL_X / pointing(index, counterclockwise).cos()) as f32)
            .collect();
        Arrived {
            arrived_ns: 0,
            reading: Arc::new(scan_stream::Message {
                header: scan_stream::MessageHeader {
                    timestamp: at(stamp_s),
                    frame: frame.to_owned(),
                    capture_index: 0,
                },
                min_angle_rad: FIRST_RAY_RAD,
                angle_increment_rad: RAY_STEP_RAD,
                time_increment_s,
                min_range_m: 0.05,
                max_range_m: 8.0,
                ranges,
                intensities: Vec::new(),
            }),
        }
    }

    // A scan swept while the base turns 0.5 rad between its first ray and its last marks
    // the cells of the wall each ray met, placed where the lidar stood at the ray's time,
    // whether the lidar swept counterclockwise, its last ray after its first, or clockwise,
    // its last ray before its first.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_scan_swept_while_the_base_turns_marks_the_wall_where_it_stands() {
        let lidar = FrameName::parse(LIDAR_FRAME).expect("a frame");
        for counterclockwise in [true, false] {
            let intake = lidar_intake();
            intake
                .take_scan(&lidar, &wall_scan(counterclockwise, LIDAR_FRAME))
                .await
                .unwrap_or_else(|why| panic!("counterclockwise {counterclockwise}: {why}"));
            let occupancy = occupancy(&intake);
            let geometry = occupancy.geometry();
            let lethal: BTreeSet<(u32, u32)> = geometry
                .cells()
                .filter(|&cell| occupancy.cost(cell) == LETHAL)
                .map(|cell| (cell.x(), cell.y()))
                .collect();
            let wall: BTreeSet<(u32, u32)> = (0..RAYS)
                .map(|index| {
                    let hit = [WALL_X, WALL_X * pointing(index, counterclockwise).tan()];
                    let cell = geometry.cell_of(hit).expect("the wall lies on the grid");
                    (cell.x(), cell.y())
                })
                .collect();
            assert_eq!(lethal, wall, "counterclockwise {counterclockwise}");
        }
    }

    // A scan whose header names a frame other than the lidar's is dropped naming both, and
    // the lidar is waited for with why; once a scan in its frame goes into the grid, it is
    // waited for as a reading.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_scan_naming_another_frame_is_dropped_naming_both() {
        let intake = lidar_intake();
        let dropped = LatchedLog::warn("the test's intake");
        intake
            .handle_scan(&wall_scan(true, "rear_lidar_frame"), &dropped)
            .await;
        let waits = intake.sensors.waits(&[0], 0.5);
        assert!(
            waits.contains("its scans arrive and are dropped")
                && waits.contains("\"rear_lidar_frame\"")
                && waits.contains("\"lidar_frame\""),
            "{waits}"
        );
        intake
            .handle_scan(&wall_scan(true, LIDAR_FRAME), &dropped)
            .await;
        assert_eq!(
            intake.sensors.waits(&[0], 0.5),
            "a reading of the lidar lidar within source_timeout_s 0.5 s"
        );
    }
}
