//! The rig the harness tests drive: room_mapper on the harness's simulation clock, a mock
//! base that drives a drawn room, its base_states the odometry, and a mock lidar that
//! ray-casts the room from the base, 0.3 m above base_footprint.

#![allow(dead_code)]

use std::cell::Cell;
use std::f64::consts::PI;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use control_core::fault::Fault;
use peppygen::Parameters;
use peppygen::consumed_services::base::describe_base;
use peppygen::consumed_services::sensor_mounts::describe_sensors;
use peppygen::consumed_services::transform_lookup::lookup_transform;
use peppygen::consumed_topics::base::base_states;
use peppygen::consumed_topics::lidar::scan_stream;
use peppygen::fixtures::exposed_services::room_map::{
    finish_mapping, get_map_state, load_map, set_initial_pose, start_mapping,
};
use peppygen::fixtures::harness::{Config, Harness, Mocks};
use peppylib::testing::HarnessClock;

/// How long the wire may take for any one exchange.
pub const WIRE: Duration = Duration::from_secs(10);
/// The guard that ends a test whose node hangs: no test waits this long when it works.
pub const HANG: Duration = Duration::from_secs(120);
pub const NANOS_PER_SECOND: f64 = 1e9;
/// The lidar's frame, which sensor_mounts names a lidar fixed to the base.
pub const LIDAR_FRAME: &str = "lidar_frame";
/// The room: a rectangle about the origin, its half width and half depth (m), and a box
/// in its north-east quarter, its corners (m), so no place of the room looks like another.
pub const HALF_WIDTH: f64 = 4.0;
pub const HALF_DEPTH: f64 = 2.5;
pub const BOX: [[f64; 2]; 2] = [[1.5, 0.8], [2.5, 1.6]];
/// The lidar's rays: all around, at once.
pub const RAYS: usize = 360;
pub const MAX_RANGE_M: f32 = 12.0;

/// A lidar's fan as scan_stream carries it.
#[derive(Clone, Copy, Debug)]
pub struct Fan {
    pub min_angle_rad: f32,
    pub angle_increment_rad: f32,
    pub time_increment_s: f32,
    pub min_range_m: f32,
    pub max_range_m: f32,
}

/// The drawn room's lidar: [`RAYS`] rays all around, measured at once.
pub const DRAWN_FAN: Fan = Fan {
    min_angle_rad: -std::f32::consts::PI,
    angle_increment_rad: (2.0 * PI / RAYS as f64) as f32,
    time_increment_s: 0.0,
    min_range_m: 0.05,
    max_range_m: MAX_RANGE_M,
};
/// The scans of a loop around the box: 9, 5, 9 and 5 steps of at most 0.6 m.
pub const LOOP_SCANS: u32 = 28;

pub fn at(seconds: f64) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs_f64(seconds)
}

/// The node of `tests/parameters.json`, saving its maps in a directory of its own.
pub fn config(maps: &std::path::Path) -> Config {
    let mut parameters: Parameters = serde_json::from_str(include_str!("../parameters.json"))
        .expect("tests/parameters.json gives every parameter");
    parameters.maps_directory = maps.display().to_string();
    Config {
        parameters: Some(parameters),
        clock: HarnessClock::Consumer,
        ..Config::default()
    }
}

/// The node booted over the harness on the maps of `maps`: its mock base answers
/// describe_base `answers` times with `odometry_source`, and sensor_mounts names the lidar
/// carried by `carried_by`; with the fault that ends the node.
async fn boot(
    maps: &std::path::Path,
    odometry_source: &str,
    carried_by: &str,
    answers: usize,
) -> (Harness, Mocks, Fault) {
    let fault = Fault::default();
    let raised = fault.clone();
    let (harness, mocks) = Harness::start_with(config(maps), move |params, runner| {
        room_mapper::setup(params, runner, raised)
    })
    .await
    .expect("the node starts");
    for _ in 0..answers {
        mocks
            .deps
            .base
            .describe_base
            .enqueue_response(base_answer(odometry_source))
            .expect("a describe_base answer");
    }
    mocks
        .deps
        .sensor_mounts
        .describe_sensors
        .enqueue_response(sensors_answer(carried_by))
        .expect("a describe_sensors answer");
    (harness, mocks, fault)
}

/// A directory of maps that no other test writes.
pub fn maps_directory(label: &str) -> std::path::PathBuf {
    let directory = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("room_mapper_{label}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    directory
}

/// The base: a 0.6 m square, its odometry `source`.
pub fn base_answer(source: &str) -> describe_base::ResponseData {
    let corner = |x: f64, y: f64| describe_base::ResponseDataFootprintPointsItem {
        position: [x, y, 0.0],
    };
    describe_base::ResponseData {
        base_name: "test_base".into(),
        odometry_source: source.into(),
        linear_limits: describe_base::ResponseDataLinearLimits {
            min_velocity: [-0.5, 0.0],
            max_velocity: [0.5, 0.0],
            max_acceleration: [1.0, 0.0],
            max_deceleration: [1.0, 0.0],
            deadband: [0.0, 0.0],
            rest_velocity: [0.01, 0.0],
        },
        angular_limits: describe_base::ResponseDataAngularLimits {
            min_rate: -1.0,
            max_rate: 1.0,
            max_acceleration: 1.0,
            max_deceleration: 1.0,
            deadband: 0.0,
            rest_rate: 0.01,
        },
        footprint: describe_base::ResponseDataFootprint {
            frame: "base_footprint".into(),
            timestamp: None,
            points: vec![
                corner(-0.3, -0.3),
                corner(0.3, -0.3),
                corner(0.3, 0.3),
                corner(-0.3, 0.3),
            ],
        },
    }
}

pub fn sensors_answer(carried_by: &str) -> describe_sensors::ResponseData {
    describe_sensors::ResponseData {
        robot_frame: "base_footprint".into(),
        sensor_names: vec!["lidar".into()],
        sensor_kinds: vec!["lidar_2d".into()],
        sensor_frames: vec![LIDAR_FRAME.into()],
        carried_by: vec![carried_by.into()],
    }
}

/// The lookup's answer placing `child_frame` at `position` and `orientation` in
/// base_footprint.
pub fn lookup_answer(
    child_frame: &str,
    position: [f64; 3],
    orientation: [f64; 4],
) -> lookup_transform::ResponseData {
    lookup_transform::ResponseData {
        frame: "base_footprint".into(),
        child_frame: child_frame.into(),
        timestamp: None,
        position,
        orientation,
    }
}

/// The lidar 0.3 m above base_footprint, level, as the lookup answers it.
pub fn mount_answer() -> lookup_transform::ResponseData {
    lookup_answer(LIDAR_FRAME, [0.0, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0])
}

/// The robot and its sensors as a test drives them.
pub struct Rig {
    pub harness: Harness,
    pub mocks: Mocks,
    /// Where the node saves its maps.
    pub maps: std::path::PathBuf,
    /// The time (s) of the simulation clock.
    pub now_s: Cell<f64>,
    /// The odometry start (s, wall clock) and the base's true pose.
    pub odometry_start_s: Cell<f64>,
    pub pose: Cell<[f64; 3]>,
    /// The bearings (rad, from the base's heading) of rays that a pole on the base's own
    /// body ends at 0.2 m, inside the padded footprint; empty for none.
    pub own_body: Cell<Option<[f64; 2]>>,
    /// Where odom's origin lies in the room: base_states give the base's pose from it.
    pub odometry_offset: Cell<[f64; 3]>,
    pub captures: Cell<u32>,
    /// The fault that ends the node, with why.
    pub fault: Fault,
}

impl Rig {
    /// A rig whose base has mapped the room's loop as `lab` and loaded it, its base where
    /// the loop ended, with no estimate.
    pub async fn mapped_and_loaded(label: &str) -> Rig {
        let rig = Rig::start(label, "measured", "").await;
        rig.mocks
            .deps
            .transform_lookup
            .lookup_transform
            .enqueue_response(mount_answer())
            .expect("a lookup answer");
        rig.start_mapping("lab").await.expect("mapping starts");
        rig.drive_a_loop().await;
        finish_mapping::poll(&rig.harness, WIRE)
            .await
            .expect("the map is saved");
        load_map::poll(
            &rig.harness,
            &load_map::RequestData {
                map_name: "lab".into(),
            },
            WIRE,
        )
        .await
        .expect("the map loads");
        rig
    }

    /// A set_initial_pose request: a target in `frame` at the base's position moved by
    /// `offset` (m) along x, at the base's heading, searched within `radius_m` and 0.3 rad.
    pub fn target(&self, frame: &str, offset: f64, radius_m: f64) -> set_initial_pose::RequestData {
        let [x, y, yaw] = self.pose.get();
        let (sin, cos) = (yaw / 2.0).sin_cos();
        set_initial_pose::RequestData {
            target: peppygen::exposed_services::room_map::set_initial_pose::SetInitialPoseTarget {
                frame: frame.into(),
                timestamp: None,
                position: [x + offset, y, 0.0],
                orientation: [0.0, 0.0, sin, cos],
            },
            search_radius_m: radius_m,
            search_angle_rad: 0.3,
        }
    }

    /// `request`'s answer, a scan every 50 ms of the wall clock until it comes. The base is
    /// measured first, as a running base's base_states keep it heard.
    pub async fn set_initial_pose(
        &self,
        request: &set_initial_pose::RequestData,
    ) -> Result<set_initial_pose::Response, set_initial_pose::Failure> {
        self.measure_base().await;
        let pending = set_initial_pose::poll(&self.harness, request, WIRE);
        tokio::pin!(pending);
        let answer = loop {
            tokio::select! {
                answer = &mut pending => break answer,
                () = tokio::time::sleep(Duration::from_millis(50)) => self.scan().await,
            }
        };
        answer.map_err(|error| set_initial_pose::Failure::of(&error).expect("an error reply"))
    }

    pub async fn start(label: &str, odometry_source: &str, carried_by: &str) -> Rig {
        Rig::start_with(label, odometry_source, carried_by, 8).await
    }

    /// A rig whose mock base answers describe_base `answers` times.
    pub async fn start_with(
        label: &str,
        odometry_source: &str,
        carried_by: &str,
        answers: usize,
    ) -> Rig {
        let _ = tracing_subscriber::fmt().with_test_writer().try_init();
        let maps = maps_directory(label);
        let (harness, mocks, fault) = boot(&maps, odometry_source, carried_by, answers).await;
        Rig {
            harness,
            mocks,
            maps,
            now_s: Cell::new(1.0),
            odometry_start_s: Cell::new(1_000.0),
            pose: Cell::new([-2.0, -1.0, 0.0]),
            own_body: Cell::new(None),
            odometry_offset: Cell::new([0.0; 3]),
            captures: Cell::new(0),
            fault,
        }
    }

    /// The node stopped and started again on the same maps, the robot as it stands, its
    /// mock base answering describe_base `answers` times with `odometry_source`.
    pub async fn restart(self, odometry_source: &str, carried_by: &str, answers: usize) -> Rig {
        self.harness.shutdown().await.expect("the node stops");
        let (harness, mocks, fault) = boot(&self.maps, odometry_source, carried_by, answers).await;
        harness
            .clock
            .tick((self.now_s.get() * NANOS_PER_SECOND) as u64)
            .await
            .expect("the clock ticks");
        Rig {
            harness,
            mocks,
            fault,
            ..self
        }
    }

    pub async fn tick(&self, seconds: f64) {
        self.tick_to(self.now_s.get() + seconds).await;
    }

    /// The clock at `seconds` (s).
    pub async fn tick_to(&self, seconds: f64) {
        self.now_s.set(seconds);
        self.harness
            .clock
            .tick((seconds * NANOS_PER_SECOND) as u64)
            .await
            .expect("the clock ticks");
    }

    /// base_states of the base at its pose, at the clock's time.
    pub async fn measure_base(&self) {
        let ([tx, ty, tyaw], [ox, oy, oyaw]) = (self.pose.get(), self.odometry_offset.get());
        let (sin_o, cos_o) = oyaw.sin_cos();
        let (dx, dy) = (tx - ox, ty - oy);
        let odometry = [
            cos_o * dx + sin_o * dy,
            -sin_o * dx + cos_o * dy,
            tyaw - oyaw,
        ];
        self.publish_base_state(self.now_s.get(), odometry).await;
    }

    /// base_states placing the base at `[x, y, yaw]` of the odometry, stamped `at_s` (s).
    pub async fn publish_base_state(&self, at_s: f64, [x, y, yaw]: [f64; 3]) {
        let (sin, cos) = (yaw / 2.0).sin_cos();
        let state = base_states::Message {
            timestamp: at(at_s),
            odometry_start_timestamp: at(self.odometry_start_s.get()),
            frame: "odom".into(),
            child_frame: "base_footprint".into(),
            pose: base_states::MessagePose {
                position: [x, y, 0.0],
                orientation: [0.0, 0.0, sin, cos],
            },
            twist: base_states::MessageTwist {
                linear_velocity: [0.0; 3],
                angular_velocity: [0.0; 3],
            },
            mode: "moving".into(),
            parts_within_footprint: false,
        };
        self.mocks
            .deps
            .base
            .base_states
            .publish(&state)
            .await
            .expect("base_states is published");
    }

    /// A scan of `fan` with `ranges`, stamped `at_s` (s), its capture counted.
    pub async fn publish_scan(&self, at_s: f64, fan: &Fan, ranges: Vec<f32>) {
        self.captures.set(self.captures.get() + 1);
        let scan = scan_stream::Message {
            header: scan_stream::MessageHeader {
                timestamp: at(at_s),
                frame: LIDAR_FRAME.into(),
                capture_index: self.captures.get(),
            },
            min_angle_rad: fan.min_angle_rad,
            angle_increment_rad: fan.angle_increment_rad,
            time_increment_s: fan.time_increment_s,
            min_range_m: fan.min_range_m,
            max_range_m: fan.max_range_m,
            ranges,
            intensities: Vec::new(),
        };
        self.mocks
            .deps
            .lidar
            .scan_stream
            .publish(&scan)
            .await
            .expect("a scan is published");
    }

    /// A scan of the room from the base, at the clock's time, enclosed by base_states
    /// before and after it.
    pub async fn scan(&self) {
        self.measure_base().await;
        self.tick(0.05).await;
        self.measure_base().await;
        let [x, y, yaw] = self.pose.get();
        let ranges = (0..RAYS)
            .map(|ray| {
                let bearing = -PI + ray as f64 * f64::from(DRAWN_FAN.angle_increment_rad);
                let on_the_body = self
                    .own_body
                    .get()
                    .is_some_and(|[from, to]| (from..=to).contains(&bearing));
                if on_the_body {
                    0.2
                } else {
                    range_from([x, y], yaw + bearing)
                }
            })
            .collect();
        self.publish_scan(self.now_s.get() - 0.025, &DRAWN_FAN, ranges)
            .await;
        self.tick(0.05).await;
    }

    /// Scans until the lidar's place on the base is known and the base is heard, so
    /// start_mapping takes a map; then that start_mapping's answer.
    pub async fn start_mapping(&self, name: &str) -> Result<(), start_mapping::Failure> {
        let request = start_mapping::RequestData {
            map_name: name.into(),
        };
        for _ in 0..50 {
            self.scan().await;
            self.measure_base().await;
            match start_mapping::poll(&self.harness, &request, WIRE).await {
                Err(error) => {
                    let failure = start_mapping::Failure::of(&error).expect("an error reply");
                    if failure.kind != start_mapping::FailureKind::ResourceUnavailable {
                        return Err(failure);
                    }
                }
                Ok(()) => return Ok(()),
            }
        }
        panic!("the lidar's place stays unknown");
    }

    /// Waits until the mapping has read `scans` scans, as get_map_state counts them,
    /// however long the host takes; a hang ends at [`HANG`]. base_states and scans may
    /// drop on the wire (sensor_data), so while it waits the base is measured again, which
    /// encloses a scan whose closing measurement dropped, and every 25th poll the lidar
    /// scans again from where the base stands, which stands in for a scan that dropped.
    pub async fn wait_for_scans(&self, scans: u32) {
        let read = async {
            for poll in 1_u32.. {
                let state = get_map_state::poll(&self.harness, WIRE)
                    .await
                    .expect("a state");
                if state.map.is_some_and(|map| map.scans >= scans) {
                    return;
                }
                if poll.is_multiple_of(25) {
                    self.scan().await;
                } else {
                    self.measure_base().await;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        };
        tokio::time::timeout(HANG, read)
            .await
            .unwrap_or_else(|_| panic!("the mapping read no {scans} scans before the hang guard"));
    }

    /// The base driven around the box in steps of 0.6 m, a scan at each, each read by the
    /// mapping before the base drives on: 28 scans.
    pub async fn drive_a_loop(&self) {
        let mut scanned = 0;
        let corners: [[f64; 2]; 5] = [
            [-2.0, -1.0],
            [3.0, -1.0],
            [3.0, 1.8],
            [-2.0, 1.8],
            [-2.0, -1.0],
        ];
        for leg in corners.windows(2) {
            let ([x0, y0], [x1, y1]) = (leg[0], leg[1]);
            let yaw = (y1 - y0).atan2(x1 - x0);
            let steps = ((x1 - x0).hypot(y1 - y0) / 0.6).ceil() as usize;
            for step in 1..=steps {
                let ratio = step as f64 / steps as f64;
                self.pose
                    .set([x0 + ratio * (x1 - x0), y0 + ratio * (y1 - y0), yaw]);
                self.tick(1.0).await;
                self.scan().await;
                scanned += 1;
                self.wait_for_scans(scanned).await;
            }
        }
    }
}

/// The range (m) from `origin` along `heading` to the room's walls or its box, or the
/// no-return mark past the lidar's reach.
/// The walls of the room and of its box, each a segment.
fn walls() -> [[[f64; 2]; 2]; 8] {
    let [[bx0, by0], [bx1, by1]] = BOX;
    [
        [[-HALF_WIDTH, -HALF_DEPTH], [HALF_WIDTH, -HALF_DEPTH]],
        [[HALF_WIDTH, -HALF_DEPTH], [HALF_WIDTH, HALF_DEPTH]],
        [[HALF_WIDTH, HALF_DEPTH], [-HALF_WIDTH, HALF_DEPTH]],
        [[-HALF_WIDTH, HALF_DEPTH], [-HALF_WIDTH, -HALF_DEPTH]],
        [[bx0, by0], [bx1, by0]],
        [[bx1, by0], [bx1, by1]],
        [[bx1, by1], [bx0, by1]],
        [[bx0, by1], [bx0, by0]],
    ]
}

/// The distance (m) from `point` to the nearest wall of the room or its box.
pub fn distance_to_walls(point: [f64; 2]) -> f64 {
    walls()
        .iter()
        .map(|[a, b]| {
            let edge = [b[0] - a[0], b[1] - a[1]];
            let along = ((point[0] - a[0]) * edge[0] + (point[1] - a[1]) * edge[1])
                / (edge[0] * edge[0] + edge[1] * edge[1]);
            let t = along.clamp(0.0, 1.0);
            (point[0] - a[0] - t * edge[0]).hypot(point[1] - a[1] - t * edge[1])
        })
        .fold(f64::INFINITY, f64::min)
}

pub fn range_from(origin: [f64; 2], heading: f64) -> f32 {
    let direction = [heading.cos(), heading.sin()];
    let nearest = walls()
        .iter()
        .filter_map(|[a, b]| {
            let edge = [b[0] - a[0], b[1] - a[1]];
            let denominator = direction[0] * edge[1] - direction[1] * edge[0];
            if denominator.abs() < 1e-12 {
                return None;
            }
            let offset = [a[0] - origin[0], a[1] - origin[1]];
            let t = (offset[0] * edge[1] - offset[1] * edge[0]) / denominator;
            let u = (offset[0] * direction[1] - offset[1] * direction[0]) / denominator;
            (t > 0.0 && (0.0..=1.0).contains(&u)).then_some(t)
        })
        .fold(f64::INFINITY, f64::min);
    if nearest < f64::from(MAX_RANGE_M) {
        nearest as f32
    } else {
        camera_core::laser_scan::NO_RETURN_MARK_M
    }
}

/// The first message of `next` stamped after `after`, its stamp read by `stamp`.
pub async fn first_after<T>(
    mut next: impl AsyncFnMut() -> Option<T>,
    stamp: impl Fn(&T) -> SystemTime,
    after: SystemTime,
) -> T {
    loop {
        let message = tokio::time::timeout(WIRE, next())
            .await
            .expect("a message within the wire's time")
            .expect("the topic stays open");
        if stamp(&message) > after {
            return message;
        }
    }
}
