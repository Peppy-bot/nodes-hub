//! The rig the navigator's harness tests drive, through the node's wire
//! alone: drive_link's pair, the four actions and clear_grid, and the
//! simulated clock the test ticks. The node boots in-process, its plans
//! starting at once or each held until the test lets it start
//! (`base_navigator::setup_with`). The base is played by a mock of
//! drive_link's follower that executes the node's setpoints through
//! base_state's limiter ([`mock_base`]) and reports each step's drive_states
//! with the transform odom to base_footprint of the same stamp, as botshare's
//! backbone does; the robot's initializer by a publisher of each sensor's
//! fixed frame on static_transforms and by describe_sensors' answer;
//! base_state by describe_base's answer; each depth camera by a pinhole that
//! renders the scene's boxes, tables, slabs and drops from where the base
//! stands; each 2-D lidar, which the boot binds only when a test asks, by a
//! fan whose rays each render the scene from where the base stood at the ray's
//! time, between where it stood at the step before and where it stands
//! ([`scene`]); and room_map, whose link the boot leaves vacant unless a test
//! asks, by a fake that gives maps of boxes, a map_poses after each report of
//! the base, and get_map_state's answer as the test sets them ([`room_map`]).
//!
//! Each step moves the base by the velocity it executes, publishes the
//! odometry, each camera's frame of the step's time and each lidar's scan whose
//! sweep ends at that time, and ticks the clock to that time. While a goal
//! runs, the step waits for the goal's feedback of that step, or for the end of
//! its feedback, which the node gives at the step that ends the goal: the rig
//! and the node step together. A goal in map waits, the clock standing still,
//! for the fake room_map's answer of get_map_state ([`Rig::send_stepping`]).
//! The node stamps each setpoint with its step's time, and the rig keeps each
//! by its stamp. The base commands a setpoint from the step after it arrives;
//! a step of a spin, a drive_on_heading or a back_up, each of which streams a
//! setpoint, waits for it. A setpoint that arrives proves that every setpoint
//! the node streamed before it arrived, since the node streams them in order
//! on one publisher.
//!
//! The wire orders none of these against the clock's tick: the node's taking
//! of a step's drive_states, depth frames and scans, a plan finishing, a goal
//! reaching the node's control loop after its acceptance, and a navigation's
//! setpoint reaching the base before the rig's next step. The tests assert
//! what holds whichever step each of these lands on. The clock can lag the
//! messages by a fixed time, and can leap: the inputs of the steps it leaps
//! over arrive while it stands still.

pub mod log;
pub mod mock_base;
pub mod room_map;
pub mod scene;

use std::collections::{BTreeMap, VecDeque};
use std::f64::consts::PI;
use std::fmt::Display;
use std::num::NonZeroUsize;
use std::ops::RangeBounds;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use base_navigator::{PlanGate, Planner};
use control_core::fault::Fault;
use control_core::time::nanoseconds;
use control_core::twist::PlanarTwist;
use frame_tree::Stamp;
use frame_tree::arithmetic::{Quaternion, product, quaternion_of_yaw};
use peppygen::FeedbackBound;
use peppygen::consumed_services::base_state::describe_base::{
    ResponseDataAngularLimits, ResponseDataFootprint, ResponseDataFootprintPointsItem,
    ResponseDataLinearLimits,
};
use peppygen::consumed_services::geometry::get_depth_to_color_extrinsics::ResponseDataDepthToColor;
use peppygen::consumed_topics::cameras::depth_stream::{self, MessageHeader};
use peppygen::consumed_topics::lidars::scan_stream;
use peppygen::exposed_actions::base_navigation::navigate_to_pose::GoalTarget;
use peppygen::fixtures::exposed_actions::base_navigation::{
    back_up, drive_on_heading, navigate_to_pose, spin,
};
use peppygen::fixtures::exposed_services::base_navigation::clear_grid;
use peppygen::fixtures::harness::{Config, Harness, Mocks};
use peppygen::mock::deps::{base_state, cameras, geometry, lidars, sensor_mounts, transforms};
use peppygen::mock::pairings::base;
use peppygen::paired_topics::base::drive_states::{
    MessageAngularLimits, MessageLinearLimits, MessagePose, MessageTwist,
};
use peppylib::testing::{HarnessClock, MockClock};
use tokio::sync::Mutex;

use self::log::logged_with;
use control_core::drive_link::FollowState;
use mock_base::{Limits, MockBase, Planar};
use room_map::{FakeMap, FakeRoomMap, MAP_FRAME, RoomState};
use scene::{DepthLens, LidarLens, Pose3, Scene, TurnedBox};

/// One step of the node's controller and of the rig (ns): 20 Hz.
pub const STEP_NS: i64 = 50_000_000;

/// One step (s).
pub const DT: f64 = 0.05;

/// The time the rig's first step reads (ns): a whole step.
pub const START_NS: i64 = 1_000_000_000;

/// When the base's odometry started (ns), unless a test starts it over.
pub const ODOMETRY_STARTED_AT_NS: i64 = 500_000_000;

/// base_state's frames, as the node places by them.
pub const ODOMETRY_FRAME: &str = "odom";
pub const ROBOT_FRAME: &str = "base_footprint";

/// The base's part name, as describe_base answers it.
pub const BASE_NAME: &str = "base";

/// The publishers of the robot's tree the rig binds on transforms: the base's
/// backbone, which moves odom to base_footprint, and the robot's initializer,
/// which fixes each sensor's frame to the robot frame.
pub const BACKBONE: &str = "backbone";
pub const INITIALIZER: &str = "initializer";

/// The base the tests drive: a differential base of 0.5 m/s each way and
/// 1.5 rad/s, whose turn slows faster than it speeds up along x, so the stop
/// projects time_before_collision_s, 1 s: the source timeout's 0.5 s and the
/// 0.5 s either axis takes to rest from its fastest velocity.
pub const LIMITS: Limits = Limits {
    max_linear_m_s: 0.5,
    linear_acceleration_m_s2: 1.0,
    linear_deceleration_m_s2: 1.0,
    linear_deadband_m_s: 0.02,
    max_angular_rad_s: 1.5,
    angular_acceleration_rad_s2: 3.0,
    angular_deceleration_rad_s2: 3.0,
    angular_deadband_rad_s: 0.05,
    rest_velocity_m_s: 0.01,
    rest_rate_rad_s: 0.01,
};

/// The base's footprint, 0.44 m long and 0.40 m wide about the robot frame's
/// origin, counterclockwise seen from above.
pub const FOOTPRINT: [[f64; 2]; 4] = [[0.22, -0.2], [0.22, 0.2], [-0.22, 0.2], [-0.22, -0.2]];

/// The half of the grid's side in tests/parameters.json (m): a goal's end lies
/// within this of the base along odom's x and y.
pub fn grid_half_side_m() -> f64 {
    params().grid.side_m / 2.0
}

/// The lens of the tests' cameras: 120 x 90 pixels, a 90 degree horizontal
/// field, millimeter counts, rendered to 8 m.
pub const LENS: DepthLens = DepthLens {
    width: 120,
    height: 90,
    fx: 60.0,
    fy: 60.0,
    cx: 59.5,
    cy: 44.5,
    unit_m: 0.001,
    max_depth_m: 8.0,
};

/// The alignment of the cameras' geometry and of each frame's header.
pub const ALIGNED: &str = "depth_to_color";

/// rgbd_camera, the kind describe_sensors names each camera of the grid.
pub const DEPTH_CAMERA: &str = "rgbd_camera";

/// lidar_2d, the kind describe_sensors names each lidar of the grid.
pub const LIDAR_2D: &str = "lidar_2d";

/// The turn from a camera's body, x ahead, y left and z up, to its colour
/// optical frame, x right, y down and z along the view.
const BODY_TO_OPTICAL: Quaternion = [-0.5, 0.5, -0.5, 0.5];

/// A camera of the rig: its name, where the base carries it, its body turned
/// by `yaw_rad` about the vertical, then pitched down by `pitch_rad`, and its
/// lens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mount {
    pub name: &'static str,
    pub position: [f64; 3],
    pub yaw_rad: f64,
    pub pitch_rad: f64,
    pub lens: DepthLens,
}

impl Mount {
    /// The camera's colour optical frame, as describe_sensors names it and as
    /// each of its frames' headers names it.
    pub fn frame(&self) -> String {
        format!("{}_color_optical_frame", self.name)
    }

    /// The camera's colour optical frame in the robot frame.
    pub fn pose(&self) -> Pose3 {
        let (yaw_sin, yaw_cos) = (self.yaw_rad / 2.0).sin_cos();
        let (pitch_sin, pitch_cos) = (self.pitch_rad / 2.0).sin_cos();
        let body = product(
            [0.0, 0.0, yaw_sin, yaw_cos],
            [0.0, pitch_sin, 0.0, pitch_cos],
        );
        Pose3 {
            position: self.position,
            orientation: product(body, BODY_TO_OPTICAL),
        }
    }
}

/// How far each side camera looks down (rad): 10 degrees.
const SIDE_PITCH_RAD: f64 = 10.0 * PI / 180.0;

/// The camera at the base's front edge, 0.30 m up, pitched 10 degrees down.
pub const FRONT: Mount = Mount {
    name: "front",
    position: [0.225, 0.0, 0.3],
    yaw_rad: 0.0,
    pitch_rad: SIDE_PITCH_RAD,
    lens: LENS,
};

/// The camera at the base's left edge, looking left.
pub const LEFT: Mount = Mount {
    name: "left",
    position: [0.0, 0.205, 0.3],
    yaw_rad: PI / 2.0,
    pitch_rad: SIDE_PITCH_RAD,
    lens: LENS,
};

/// The camera at the base's back edge, looking back.
pub const BACK: Mount = Mount {
    name: "back",
    position: [-0.225, 0.0, 0.3],
    yaw_rad: PI,
    pitch_rad: SIDE_PITCH_RAD,
    lens: LENS,
};

/// The camera at the base's right edge, looking right.
pub const RIGHT: Mount = Mount {
    name: "right",
    position: [0.0, -0.205, 0.3],
    yaw_rad: -PI / 2.0,
    pitch_rad: SIDE_PITCH_RAD,
    lens: LENS,
};

/// Four cameras, one on each edge, so whatever the base's heading one of them
/// sees what lies ahead of where it started.
pub const RING: [Mount; 4] = [FRONT, LEFT, BACK, RIGHT];

/// A camera on top of the base, over its origin, 0.5 m up, pitched 30
/// degrees down, so its view holds the base's own deck.
pub const TOP: Mount = Mount {
    name: "top",
    position: [0.0, 0.0, 0.5],
    yaw_rad: 0.0,
    pitch_rad: 30.0 * PI / 180.0,
    lens: LENS,
};

/// The fan of the tests' lidars: 360 rays a turn, one a degree from -pi,
/// swept counterclockwise in 35.9 ms, within a step, measuring from 0.05 m to
/// 8 m.
pub const LIDAR_LENS: LidarLens = LidarLens {
    min_angle_rad: -PI as f32,
    angle_increment_rad: (2.0 * PI / 360.0) as f32,
    rays: 360,
    time_increment_s: 1e-4,
    min_range_m: 0.05,
    max_range_m: 8.0,
};

/// A 2-D lidar of the rig: its name, where the base carries it, its sensor
/// frame turned by `yaw_rad` about the vertical, its fan, and the part that
/// carries it as describe_sensors names it, empty for a lidar fixed to the
/// base.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LidarMount {
    pub name: &'static str,
    pub position: [f64; 3],
    pub yaw_rad: f64,
    pub lens: LidarLens,
    pub carried_by: &'static str,
}

impl LidarMount {
    /// The lidar's sensor frame, as describe_sensors names it and as each of
    /// its scans' headers names it.
    pub fn frame(&self) -> String {
        format!("{}_frame", self.name)
    }

    /// The lidar's sensor frame in the robot frame.
    pub fn pose(&self) -> Pose3 {
        Pose3 {
            position: self.position,
            orientation: quaternion_of_yaw(self.yaw_rad),
        }
    }
}

/// The lidar over the base's origin, 0.3 m up, heading along the base, fixed
/// to the base.
pub const LIDAR: LidarMount = LidarMount {
    name: "lidar",
    position: [0.0, 0.0, 0.3],
    yaw_rad: 0.0,
    lens: LIDAR_LENS,
    carried_by: "",
};

/// The base's deck as the top camera sees it, its half sizes in the robot
/// frame: 0.01 m inside the footprint's outline on each side, 0.3 m tall.
pub const DECK_INSIDE_HALF: [f64; 3] = [0.21, 0.19, 0.15];

/// The deck reaching the footprint's outline itself, which the padded
/// footprint, 0.01 m wider, holds within its band.
pub const DECK_AT_OUTLINE_HALF: [f64; 3] = [0.22, 0.2, 0.15];

/// How long a test waits for what the node answers once its time has come:
/// a bound on a test that fails, never a pass condition.
pub const ANSWER_BUDGET: Duration = Duration::from_secs(30);

/// How many steps a goal waits for the node to know the base and every
/// sensor: a bound on a test that fails, never a pass condition.
pub const READY_STEPS: usize = 100;

/// How many steps a drive takes at most before a test calls it stuck: 60 s of
/// the clock, a bound on a test that fails, never a pass condition.
pub const DRIVE_STEPS: usize = 1200;

/// How far (rad) the spin of [`Rig::fence`] turns.
pub const FENCE_TURN_RAD: f64 = 0.2;

/// How many feedback messages a goal's handle keeps unread: every step of
/// the longest drive the tests run.
pub const FEEDBACK_KEPT: FeedbackBound =
    FeedbackBound::new(NonZeroUsize::new(65_536).expect("65536 is not zero"));

/// The navigator the tests run, `tests/parameters.json`, which the unit tests
/// read too: a 5 m grid of 0.05 m voxels, Nav2's defaults and botshare's
/// controller.
pub fn params() -> peppygen::Parameters {
    serde_json::from_str(include_str!("../parameters.json"))
        .expect("tests/parameters.json gives every parameter")
}

/// A time of the rig's clock (ns) as a wire time.
pub fn wire(time_ns: i64) -> SystemTime {
    Stamp::from_nanos(u64::try_from(time_ns).expect("a time of the rig's clock")).time()
}

/// A wire time as nanoseconds of the rig's clock.
pub fn ns(time: SystemTime) -> i64 {
    let stamp = Stamp::from_time(time).expect("a time a stamp holds");
    i64::try_from(stamp.as_nanos()).expect("a time of the rig's clock")
}

/// `future`, which `what` names, within [`ANSWER_BUDGET`].
pub async fn within<T>(what: impl Display, future: impl Future<Output = T>) -> T {
    tokio::time::timeout(ANSWER_BUDGET, future)
        .await
        .unwrap_or_else(|_| panic!("the rig waited {ANSWER_BUDGET:?} for {what}"))
}

/// The linear and angular limits of a base of `limits`, as describe_base
/// gives them and drive_states carries them, written once here because the
/// two generated messages carry distinct types with identical fields.
macro_rules! limits_of {
    ($limits:expr, $linear:ident, $angular:ident) => {{
        let limits: Limits = $limits;
        (
            $linear {
                min_velocity: [-limits.max_linear_m_s, 0.0],
                max_velocity: [limits.max_linear_m_s, 0.0],
                max_acceleration: [limits.linear_acceleration_m_s2, 0.0],
                max_deceleration: [limits.linear_deceleration_m_s2, 0.0],
                deadband: [limits.linear_deadband_m_s, 0.0],
                rest_velocity: [limits.rest_velocity_m_s, limits.rest_velocity_m_s],
            },
            $angular {
                min_rate: -limits.max_angular_rad_s,
                max_rate: limits.max_angular_rad_s,
                max_acceleration: limits.angular_acceleration_rad_s2,
                max_deceleration: limits.angular_deceleration_rad_s2,
                deadband: limits.angular_deadband_rad_s,
                rest_rate: limits.rest_rate_rad_s,
            },
        )
    }};
}

/// describe_base's answer for a base of `limits` whose odometry is
/// `odometry_source`, named [`BASE_NAME`], of [`FOOTPRINT`].
pub fn base_answer(
    limits: Limits,
    odometry_source: &str,
) -> base_state::describe_base::ResponseData {
    let (linear_limits, angular_limits) =
        limits_of!(limits, ResponseDataLinearLimits, ResponseDataAngularLimits);
    base_state::describe_base::ResponseData {
        base_name: BASE_NAME.to_string(),
        odometry_source: odometry_source.to_string(),
        linear_limits,
        angular_limits,
        footprint: ResponseDataFootprint {
            frame: ROBOT_FRAME.to_string(),
            timestamp: None,
            points: FOOTPRINT
                .iter()
                .map(|&[x, y]| ResponseDataFootprintPointsItem {
                    position: [x, y, 0.0],
                })
                .collect(),
        },
    }
}

/// describe_base's answer for the rig's base, its odometry measured.
pub fn measured() -> base_state::describe_base::ResponseData {
    base_answer(LIMITS, "measured")
}

/// [`LIMITS`] with both axes slowing at `deceleration` (m/s^2 and rad/s^2).
pub fn decelerating(deceleration: f64) -> Limits {
    Limits {
        linear_deceleration_m_s2: deceleration,
        angular_deceleration_rad_s2: deceleration,
        ..LIMITS
    }
}

/// A target at (`x`, `y`) in odom, heading `yaw`, at the latest time.
pub fn in_odom(x: f64, y: f64, yaw: f64) -> GoalTarget {
    GoalTarget {
        frame: Some(ODOMETRY_FRAME.to_string()),
        timestamp: None,
        position: [x, y, 0.0],
        orientation: quaternion_of_yaw(yaw),
    }
}

/// A navigation to (`x`, `y`) in odom heading +x, within 0.1 m and 0.2 rad,
/// at the base's top speed.
pub fn goal(x: f64, y: f64) -> navigate_to_pose::GoalRequestData {
    navigate_to_pose::GoalRequestData {
        target: in_odom(x, y, 0.0),
        position_tolerance_m: 0.1,
        orientation_tolerance_rad: 0.2,
        max_speed_m_s: 0.0,
    }
}

/// A spin of `angle_rad`, within `timeout_s` (0: no bound).
pub fn spin_by(angle_rad: f64, timeout_s: f64) -> spin::GoalRequestData {
    spin::GoalRequestData {
        angle_rad,
        timeout_s,
    }
}

/// A drive_on_heading of `distance_m` at `speed_m_s`, within `timeout_s`.
pub fn ahead(distance_m: f64, speed_m_s: f64, timeout_s: f64) -> drive_on_heading::GoalRequestData {
    drive_on_heading::GoalRequestData {
        distance_m,
        speed_m_s,
        timeout_s,
    }
}

/// A back_up of `distance_m` at `speed_m_s`, within `timeout_s`.
pub fn back(distance_m: f64, speed_m_s: f64, timeout_s: f64) -> back_up::GoalRequestData {
    back_up::GoalRequestData {
        distance_m,
        speed_m_s,
        timeout_s,
    }
}

/// One setpoint the node streamed.
pub type Setpoint = base::drive_setpoints::Message;

/// The forward speed of `setpoint` (m/s).
pub fn forward(setpoint: &Setpoint) -> f64 {
    setpoint.linear_velocity[0]
}

/// The turn rate of `setpoint` (rad/s), counterclockwise.
pub fn turn(setpoint: &Setpoint) -> f64 {
    setpoint.angular_velocity[2]
}

/// The norm of `setpoint`'s linear_velocity (m/s).
pub fn speed(setpoint: &Setpoint) -> f64 {
    setpoint.linear_velocity[0].hypot(setpoint.linear_velocity[1])
}

pub fn is_zero(setpoint: &Setpoint) -> bool {
    setpoint.linear_velocity == [0.0; 3] && setpoint.angular_velocity == [0.0; 3]
}

pub fn turns_in_place(setpoint: &Setpoint) -> bool {
    forward(setpoint) == 0.0 && turn(setpoint) != 0.0
}

/// A goal's feedback of one step, by its action.
#[derive(Clone, Debug)]
pub enum Feedback {
    Navigate(navigate_to_pose::Feedback),
    Spin(spin::Feedback),
    DriveOnHeading(drive_on_heading::Feedback),
    BackUp(back_up::Feedback),
}

impl Feedback {
    /// Whether the step that gave this feedback streamed a setpoint: each
    /// step of a spin, a drive_on_heading or a back_up that runs on does.
    fn streams(&self) -> bool {
        !matches!(self, Self::Navigate(_))
    }
}

/// One step of the node, as the rig saw it.
#[derive(Clone, Debug)]
pub struct StepSeen {
    /// The step's time on the node's clock (ns), which stamps its setpoint.
    pub at_ns: i64,
    /// The feedback the goal in flight gave at this step, when it ran on.
    pub feedback: Option<Feedback>,
    /// Whether the goal in flight ended at this step.
    pub goal_ended: bool,
}

impl StepSeen {
    /// The feedback of the navigation in flight at this step.
    pub fn navigation(&self) -> Option<&navigate_to_pose::Feedback> {
        match &self.feedback {
            Some(Feedback::Navigate(feedback)) => Some(feedback),
            _ => None,
        }
    }
}

/// The handle of a goal the rig sent, which the rig and the test share.
pub type Shared<H> = Arc<Mutex<H>>;

/// A goal the rig sent: whether the node accepted it, the failure it refused
/// it with, and its handle.
pub struct Goal<H, F> {
    pub accepted: bool,
    pub refusal: Option<F>,
    pub handle: Shared<H>,
}

/// The goal in flight, by its action, whose feedback each step reads.
enum InFlight {
    Navigate(Shared<navigate_to_pose::GoalHandle>),
    Spin(Shared<spin::GoalHandle>),
    DriveOnHeading(Shared<drive_on_heading::GoalHandle>),
    BackUp(Shared<back_up::GoalHandle>),
}

/// The feedback of the next step of the goal of `$handle`, `$variant` of
/// [`Feedback`], or none once the node ended the goal's feedback. Written
/// once here because the four generated handles carry distinct types with an
/// identical surface.
macro_rules! next_step {
    ($handle:expr, $variant:ident) => {{
        let next = $handle.lock().await.on_next_feedback().await;
        match next {
            Ok(feedback) => Some(Feedback::$variant(feedback)),
            Err(peppygen::Error::ActionFeedbackChannelClosed) => None,
            Err(e) => panic!("the goal's feedback: {e}"),
        }
    }};
}

impl InFlight {
    /// The goal's feedback of its next step, or none once it ended.
    async fn next_step(&self) -> Option<Feedback> {
        match self {
            Self::Navigate(handle) => next_step!(handle, Navigate),
            Self::Spin(handle) => next_step!(handle, Spin),
            Self::DriveOnHeading(handle) => next_step!(handle, DriveOnHeading),
            Self::BackUp(handle) => next_step!(handle, BackUp),
        }
    }
}

/// What the base reported at a step, published once the odometry's lag
/// passed, and where the robot frame truly stood then.
#[derive(Clone, Copy, Debug)]
pub struct OdometryReport {
    stamp_ns: i64,
    start_ns: i64,
    truth: Planar,
    odometry: Planar,
    velocity: PlanarTwist,
    mode: &'static str,
    parts_within_footprint: bool,
    follow_state: FollowState,
}

/// The mocks the steps publish on and read from, and the ones that answered
/// at the start, kept so the node keeps its producers.
struct Wire {
    base: base::Mock,
    backbone: transforms::Mock,
    cameras: Vec<cameras::Mock>,
    lidars: Vec<lidars::Mock>,
    _initializer: transforms::Mock,
    _sensor_mounts: sensor_mounts::Mock,
    _geometry: Vec<geometry::Mock>,
}

/// Everything the rig plays but the harness: the base, the scene, the
/// cameras and the lidars, the clock's time, and what the node streamed.
pub struct World {
    /// The time of the rig's newest step (ns); the node's clock reads it less
    /// [`World::clock_lag_ns`].
    pub now_ns: i64,
    pub base: MockBase,
    pub scene: Scene,
    pub mounts: Vec<Mount>,
    pub lidars: Vec<LidarMount>,
    /// Whether the cameras publish a frame each step.
    pub cameras_on: bool,
    /// Whether the lidars publish a scan each step.
    pub lidars_on: bool,
    /// The frame each scan's header names in place of its lidar's frame,
    /// when a test sets one.
    pub scan_frame: Option<&'static str>,
    /// Whether the base publishes drive_states and its odometry's transform
    /// each step.
    pub base_reports: bool,
    /// The half sizes of the base's deck, which the sensors then render as a
    /// box on the base's footprint.
    pub deck_half: Option<[f64; 3]>,
    /// How far (rad) the base's body rests turned nose-up about its own y
    /// axis, which the sensors render from; the odometry and the sensors'
    /// fixed frames hold the body level.
    pub body_pitch_up_rad: f64,
    /// How many steps after its stamp the base's report goes out: how far the
    /// odometry lags the sensors' readings.
    pub odometry_lag_steps: usize,
    /// How far (ns) the node's clock reads behind the messages' stamps.
    pub clock_lag_ns: i64,
    /// The gate of the node's planner, when its plans are held.
    gate: Option<PlanGate>,
    /// base_state's mock, which answers describe_base.
    pub base_state: base_state::Mock,
    /// The fake room_map, while the rig binds the room_map link.
    pub room: Option<FakeRoomMap>,
    /// Every setpoint the node streamed that arrived, by its stamp on the
    /// node's clock (ns).
    streamed: BTreeMap<i64, Setpoint>,
    /// Where each drive_states the base published placed the robot frame in
    /// odom, and the velocity it reported, by its stamp (ns).
    reported: BTreeMap<i64, (Planar, PlanarTwist)>,
    wire: Wire,
    /// The goal in flight, whose feedback each step reads.
    in_flight: Option<InFlight>,
    reports: VecDeque<OdometryReport>,
    frames_sent: u64,
    scans_sent: u64,
    /// Where the robot frame stood in the world at the step before the
    /// newest, from which the lidars' sweeps of the newest step start.
    truth_before: Planar,
}

impl World {
    /// Let `count` more held plans start.
    pub fn release_plans(&mut self, count: usize) {
        let gate = self.gate.as_ref().expect("the node's plans are held");
        gate.release(count);
    }

    /// How many plans the node asked of its held planner.
    pub fn plans_asked(&self) -> usize {
        self.gate
            .as_ref()
            .expect("the node's plans are held")
            .asked()
    }

    /// The node's clock reading at the rig's time `rig_ns`.
    pub fn node_ns(&self, rig_ns: i64) -> i64 {
        rig_ns - self.clock_lag_ns
    }

    /// The setpoint stamped `at_ns` that arrived, if one did.
    pub fn setpoint_at(&self, at_ns: i64) -> Option<&Setpoint> {
        self.streamed.get(&at_ns)
    }

    /// The setpoints that arrived stamped within `stamps`, by stamp.
    pub fn setpoints_in(&self, stamps: impl RangeBounds<i64>) -> Vec<(i64, Setpoint)> {
        self.streamed
            .range(stamps)
            .map(|(at_ns, setpoint)| (*at_ns, setpoint.clone()))
            .collect()
    }

    /// Where the drive_states stamped `stamp` placed the robot frame in odom,
    /// and the velocity it reported.
    pub fn reported_at(&self, stamp: SystemTime) -> (Planar, PlanarTwist) {
        *self
            .reported
            .get(&ns(stamp))
            .unwrap_or_else(|| panic!("no drive_states stamped {stamp:?}"))
    }

    /// The base's report of the newest step.
    pub fn report(&self) -> OdometryReport {
        OdometryReport {
            stamp_ns: self.now_ns,
            start_ns: self.base.odometry_start_ns,
            truth: self.base.truth,
            odometry: self.base.odometry,
            velocity: self.base.velocity,
            mode: self.base.mode(),
            parts_within_footprint: self.base.parts_within_footprint,
            follow_state: self.base.follow_state,
        }
    }

    /// Publish `report`: the transform odom to base_footprint of its stamp,
    /// then its drive_states.
    async fn publish_report(&mut self, report: &OdometryReport) {
        self.publish_odometry_transform(report).await;
        self.publish_drive_states(report).await;
    }

    /// Publish the backbone's transform odom to base_footprint of `report`'s
    /// stamp, in the odometry it started.
    pub async fn publish_odometry_transform(&self, report: &OdometryReport) {
        let position = [report.odometry.x, report.odometry.y, 0.0];
        let orientation = quaternion_of_yaw(report.odometry.yaw);
        self.wire
            .backbone
            .transforms
            .publish(&transforms::transforms::Message {
                timestamp: wire(report.stamp_ns),
                transforms: vec![
                    peppygen::consumed_topics::transforms::transforms::MessageTransformsItem {
                        frame: ODOMETRY_FRAME.to_string(),
                        child_frame: ROBOT_FRAME.to_string(),
                        position,
                        orientation,
                    },
                ],
                start_timestamp: wire(report.start_ns),
            })
            .await
            .expect("the backbone publishes the odometry's transform");
    }

    /// Publish `report`'s drive_states, kept by its stamp.
    pub async fn publish_drive_states(&mut self, report: &OdometryReport) {
        let position = [report.odometry.x, report.odometry.y, 0.0];
        let orientation = quaternion_of_yaw(report.odometry.yaw);
        let velocity = report.velocity;
        let (linear_limits, angular_limits) = limits_of!(
            self.base.limits(),
            MessageLinearLimits,
            MessageAngularLimits
        );
        self.wire
            .base
            .drive_states
            .publish(&base::drive_states::Message {
                timestamp: wire(report.stamp_ns),
                odometry_start_timestamp: wire(report.start_ns),
                frame: ODOMETRY_FRAME.to_string(),
                child_frame: ROBOT_FRAME.to_string(),
                pose: MessagePose {
                    position,
                    orientation,
                },
                twist: MessageTwist {
                    linear_velocity: [velocity.linear_x(), velocity.linear_y(), 0.0],
                    angular_velocity: [0.0, 0.0, velocity.angular_z()],
                },
                linear_limits,
                angular_limits,
                mode: report.mode.to_string(),
                parts_within_footprint: report.parts_within_footprint,
                follow_state: report.follow_state.as_str().to_string(),
            })
            .await
            .expect("the base publishes drive_states");
        self.reported
            .insert(report.stamp_ns, (report.odometry, report.velocity));
    }

    /// The base's body in the world where the robot frame stands at `truth`,
    /// resting turned nose-up by [`World::body_pitch_up_rad`].
    fn body(&self, truth: Planar) -> Pose3 {
        let (pitch_sin, pitch_cos) = (-self.body_pitch_up_rad / 2.0).sin_cos();
        Pose3::planar(truth.x, truth.y, truth.yaw).compose(&Pose3 {
            position: [0.0; 3],
            orientation: [0.0, pitch_sin, 0.0, pitch_cos],
        })
    }

    /// The base's deck where the robot frame stands at `truth`, when the
    /// sensors render one.
    fn deck(&self, truth: Planar) -> Vec<TurnedBox> {
        self.deck_half
            .map(|half| TurnedBox {
                center: [truth.x, truth.y, half[2]],
                half,
                yaw: truth.yaw,
            })
            .into_iter()
            .collect()
    }

    /// Publish each camera's frame of the newest step, rendered from where the
    /// base stands in the world.
    async fn publish_frames(&mut self) {
        let truth = self.base.truth;
        let body = self.body(truth);
        let deck = self.deck(truth);
        let mut sent = self.frames_sent;
        for (mount, camera) in self.mounts.iter().zip(&self.wire.cameras) {
            let lens = mount.lens;
            let data = lens.render(&body.compose(&mount.pose()), &self.scene, &deck);
            sent += 1;
            camera
                .depth_stream
                .publish(&depth_stream::Message {
                    header: MessageHeader {
                        timestamp: wire(self.now_ns),
                        frame: mount.frame(),
                        capture_index: sent.try_into().expect("a capture index"),
                        align_mode: ALIGNED.to_string(),
                    },
                    encoding: "z16".to_string(),
                    width: lens.width,
                    height: lens.height,
                    data,
                })
                .await
                .expect("the camera publishes its frame");
        }
        self.frames_sent = sent;
    }

    /// Publish each lidar's scan of the newest step, its sweep ending at the
    /// step's time: each ray rendered from where the base stood at the ray's
    /// time, between where it stood at the step before and where it stands.
    async fn publish_scans(&mut self) {
        let step_start_ns = self.now_ns - STEP_NS;
        let mut sent = self.scans_sent;
        for (mount, lidar) in self.lidars.iter().zip(&self.wire.lidars) {
            let lens = mount.lens;
            let sweep_ns = nanoseconds(lens.time_s(lens.rays - 1));
            let first_ray_ns = self.now_ns - sweep_ns.max(0);
            let ranges = (0..lens.rays)
                .map(|index| {
                    let ray_ns = first_ray_ns + nanoseconds(lens.time_s(index));
                    let share = (ray_ns - step_start_ns) as f64 / STEP_NS as f64;
                    let truth = self.truth_before.toward(self.base.truth, share);
                    let sensor = self.body(truth).compose(&mount.pose());
                    let (sin, cos) = lens.angle_rad(index).sin_cos();
                    let direction = sensor.turn([cos, sin, 0.0]);
                    lens.range(&self.scene, &self.deck(truth), sensor.position, direction)
                })
                .collect();
            sent += 1;
            lidar
                .scan_stream
                .publish(&scan_stream::Message {
                    header: scan_stream::MessageHeader {
                        timestamp: wire(first_ray_ns),
                        frame: self.scan_frame.map_or_else(|| mount.frame(), str::to_owned),
                        capture_index: sent.try_into().expect("a capture index"),
                    },
                    min_angle_rad: lens.min_angle_rad,
                    angle_increment_rad: lens.angle_increment_rad,
                    time_increment_s: lens.time_increment_s,
                    min_range_m: lens.min_range_m,
                    max_range_m: lens.max_range_m,
                    ranges,
                    intensities: Vec::new(),
                })
                .await
                .expect("the lidar publishes its scan");
        }
        self.scans_sent = sent;
    }

    /// Publish the inputs of the newest step: the base's report once its lag
    /// passed, with room_map's map_poses after it, then each camera's frame
    /// and each lidar's scan.
    async fn publish_inputs(&mut self) {
        if self.base_reports {
            self.reports.push_back(self.report());
        }
        while self.reports.len() > self.odometry_lag_steps {
            let report = self.reports.pop_front().expect("a report waits");
            self.publish_report(&report).await;
            if let Some(room) = &self.room {
                room.follow_report(wire(report.stamp_ns), report.truth)
                    .await;
            }
        }
        if self.cameras_on {
            self.publish_frames().await;
        }
        if self.lidars_on {
            self.publish_scans().await;
        }
    }

    /// Keep `setpoint` by its stamp, and hand it to the base, which commands
    /// it from the next step.
    fn take(&mut self, setpoint: Setpoint) -> i64 {
        assert_eq!(setpoint.frame, None, "a setpoint in the robot frame");
        let stamp = setpoint
            .timestamp
            .expect("a setpoint stamped with its step");
        let velocity = PlanarTwist::new(
            setpoint.linear_velocity[0],
            setpoint.linear_velocity[1],
            setpoint.angular_velocity[2],
        )
        .expect("a finite setpoint");
        self.base.take(velocity, self.now_ns);
        let at_ns = ns(stamp);
        assert!(
            self.streamed.insert(at_ns, setpoint).is_none(),
            "one setpoint a step, at {at_ns} ns"
        );
        at_ns
    }

    /// Take every setpoint that has arrived.
    async fn take_arrived(&mut self) {
        loop {
            let next = tokio::select! {
                biased;
                next = self.wire.base.drive_setpoints.next() => next,
                () = std::future::ready(()) => return,
            };
            let setpoint = next
                .expect("drive_setpoints receive")
                .expect("drive_setpoints open");
            self.take(setpoint);
        }
    }

    /// Take each setpoint as it arrives until one stamped at or after `at_ns`
    /// arrived: every setpoint stamped before it has arrived then. Returns
    /// that setpoint's stamp.
    pub async fn until_setpoint_from(&mut self, at_ns: i64) -> i64 {
        if let Some((&stamp, _)) = self.streamed.range(at_ns..).next() {
            return stamp;
        }
        loop {
            let next = within(
                format!("a setpoint stamped at or after {at_ns} ns"),
                self.wire.base.drive_setpoints.next(),
            )
            .await
            .expect("drive_setpoints receive")
            .expect("drive_setpoints open");
            let stamp = self.take(next);
            if stamp >= at_ns {
                return stamp;
            }
        }
    }

    /// Tick the node's clock to the newest step's time and see the node's
    /// step of that time: the feedback, or the end, of the goal in flight,
    /// and the setpoints that arrived, the one of a step that streams one
    /// waited for.
    pub async fn tick_and_see(&mut self, clock: &MockClock) -> StepSeen {
        let at_ns = self.node_ns(self.now_ns);
        clock
            .tick(u64::try_from(at_ns).expect("a time after the clock's origin"))
            .await
            .expect("the clock ticks");
        let (feedback, goal_ended) = match &self.in_flight {
            Some(goal) => match within("the goal's step", goal.next_step()).await {
                Some(feedback) => (Some(feedback), false),
                None => (None, true),
            },
            None => (None, false),
        };
        if goal_ended {
            self.in_flight = None;
        }
        if feedback.as_ref().is_some_and(Feedback::streams) {
            self.until_setpoint_from(at_ns).await;
        }
        self.take_arrived().await;
        StepSeen {
            at_ns,
            feedback,
            goal_ended,
        }
    }

    /// The first half of a step: the base moves to the next step's time.
    pub fn advance(&mut self) {
        self.now_ns += STEP_NS;
        self.truth_before = self.base.truth;
        self.base.advance(self.now_ns);
    }

    /// The second half of a step: the base's inputs of the newest step's time
    /// go out, the clock ticks, and the node's step of the time is seen.
    pub async fn publish_and_see(&mut self, clock: &MockClock) -> StepSeen {
        self.publish_inputs().await;
        self.tick_and_see(clock).await
    }

    /// One step: the base moves, its inputs go out, the clock ticks, and the
    /// node's step of the time is seen.
    pub async fn step(&mut self, clock: &MockClock) -> StepSeen {
        self.advance();
        self.publish_and_see(clock).await
    }

    /// The clock leaps over `steps` steps: the base moves and its inputs go
    /// out at each step's time while the node's clock stands still, then the
    /// clock ticks once to the last step's time.
    pub async fn leap(&mut self, steps: usize, clock: &MockClock) -> StepSeen {
        assert!(steps > 0, "a leap of one step or more");
        for _ in 1..steps {
            self.advance();
            self.publish_inputs().await;
        }
        self.step(clock).await
    }
}

/// How the node runs its plans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plans {
    /// Each plan starts once it is asked.
    AtOnce,
    /// Each plan waits until the test releases it.
    Held,
}

/// The node, booted, and what its setup was handed.
struct Booted {
    harness: Harness,
    mocks: Mocks,
    gate: Option<PlanGate>,
}

/// The node booted on the harness's simulated clock with `params`, its
/// cameras the mocks named after `mounts` in both camera slots, its lidars the
/// mocks named after `lidars`, the robot's tree bound to the backbone and the
/// initializer, its room_map link bound to the mock when `room_map` and vacant
/// otherwise, and its planner running `plans`.
async fn boot(
    params: peppygen::Parameters,
    mounts: &[Mount],
    lidars: &[LidarMount],
    room_map: bool,
    plans: Plans,
) -> Booted {
    let names: Vec<String> = mounts.iter().map(|mount| mount.name.to_string()).collect();
    let lidar_names = lidars.iter().map(|lidar| lidar.name.to_string()).collect();
    let fault = Fault::default();
    let (planner, gate) = match plans {
        Plans::AtOnce => (Planner::default(), None),
        Plans::Held => {
            let (planner, gate) = Planner::gated();
            (planner, Some(gate))
        }
    };
    let (harness, mocks) = Harness::start_with(
        Config {
            parameters: Some(params),
            clock: HarnessClock::Consumer,
            cameras_instance_ids: names.clone(),
            geometry_instance_ids: names,
            lidars_instance_ids: lidar_names,
            room_map_vacant: !room_map,
            transforms_instance_ids: vec![BACKBONE.to_string(), INITIALIZER.to_string()],
            ..Config::default()
        },
        move |params, runner| base_navigator::setup_with(params, runner, fault, planner),
    )
    .await
    .expect("the harness starts the node");
    Booted {
        harness,
        mocks,
        gate,
    }
}

/// describe_sensors' answer: each camera of `mounts` a depth camera fixed to
/// the base at its colour optical frame, then each lidar of `lidars` a lidar
/// at its sensor frame, carried by the part it names.
pub fn sensors_answer(
    mounts: &[Mount],
    lidars: &[LidarMount],
) -> sensor_mounts::describe_sensors::ResponseData {
    let sensors: Vec<(&str, &str, String, &str)> = mounts
        .iter()
        .map(|mount| (mount.name, DEPTH_CAMERA, mount.frame(), ""))
        .chain(
            lidars
                .iter()
                .map(|lidar| (lidar.name, LIDAR_2D, lidar.frame(), lidar.carried_by)),
        )
        .collect();
    sensor_mounts::describe_sensors::ResponseData {
        robot_frame: ROBOT_FRAME.to_string(),
        sensor_names: sensors.iter().map(|s| s.0.to_string()).collect(),
        sensor_kinds: sensors.iter().map(|s| s.1.to_string()).collect(),
        sensor_frames: sensors.iter().map(|s| s.2.clone()).collect(),
        carried_by: sensors.iter().map(|s| s.3.to_string()).collect(),
    }
}

/// Script each camera's description, each answered once: its lens and its
/// depth optical frame, the colour one, and its stream.
fn describe_cameras(mounts: &[Mount], cameras: &[cameras::Mock], geometry: &[geometry::Mock]) {
    for ((mount, camera), lens) in mounts.iter().zip(cameras).zip(geometry) {
        lens.get_depth_intrinsics
            .enqueue_response(geometry::get_depth_intrinsics::ResponseData {
                width: mount.lens.width,
                height: mount.lens.height,
                fx: mount.lens.fx,
                fy: mount.lens.fy,
                cx: mount.lens.cx,
                cy: mount.lens.cy,
                distortion_model: "none".to_string(),
                distortion: Vec::new(),
                depth_model: "z".to_string(),
                min_depth_m: 0.05,
                max_depth_m: mount.lens.max_depth_m,
                align_mode: ALIGNED.to_string(),
            })
            .expect("the geometry takes its answer");
        lens.get_depth_to_color_extrinsics
            .enqueue_response(geometry::get_depth_to_color_extrinsics::ResponseData {
                align_mode: ALIGNED.to_string(),
                depth_to_color: ResponseDataDepthToColor {
                    frame: mount.frame(),
                    position: [0.0; 3],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                },
            })
            .expect("the geometry takes its answer");
        camera
            .depth_stream_info
            .enqueue_response(cameras::depth_stream_info::ResponseData {
                width: mount.lens.width,
                height: mount.lens.height,
                frames_per_second: 20,
                encoding: "z16".to_string(),
                depth_unit: mount.lens.unit_m,
            })
            .expect("the camera takes its answer");
    }
}

/// Publish each camera's colour optical frame and each lidar's sensor frame
/// fixed to the robot frame, as the robot's initializer publishes the fixed
/// joints of the robot's model.
async fn fix_the_sensors(initializer: &transforms::Mock, mounts: &[Mount], lidars: &[LidarMount]) {
    let fixed = mounts
        .iter()
        .map(|mount| (mount.frame(), mount.pose()))
        .chain(lidars.iter().map(|lidar| (lidar.frame(), lidar.pose())))
        .map(|(frame, pose)| {
            peppygen::consumed_topics::transforms::static_transforms::MessageTransformsItem {
                frame: ROBOT_FRAME.to_string(),
                child_frame: frame,
                position: pose.position,
                orientation: pose.orientation,
            }
        })
        .collect();
    initializer
        .static_transforms
        .publish(&transforms::static_transforms::Message { transforms: fixed })
        .await
        .expect("the initializer publishes the sensors' frames");
}

pub struct Rig {
    pub harness: Harness,
    pub world: World,
}

/// The senders of goals to an action, written once here because the four
/// generated action modules carry distinct types with an identical surface.
/// [`sender!`] sends a goal now, with the clock standing still; a goal the
/// node accepts is the goal in flight, its `$variant` of [`InFlight`].
macro_rules! sender {
    ($send:ident, $action:ident, $variant:ident) => {
        /// Send `goal` now; the clock stands still while the node answers.
        pub async fn $send(
            &mut self,
            goal: &$action::GoalRequestData,
        ) -> Goal<$action::GoalHandle, $action::Failure> {
            let sent = $action::send_goal(&self.harness, goal, FEEDBACK_KEPT, ANSWER_BUDGET);
            let handle = within(stringify!($action), sent)
                .await
                .expect(concat!(stringify!($action), " answers"));
            self.world.sent(handle, InFlight::$variant)
        }
    };
}

/// A sender of goals to `$action` once the node knows the base and a reading
/// of every sensor, through the [`sender!`] `$send`.
macro_rules! ready_sender {
    ($send_when_ready:ident, $send:ident, $action:ident) => {
        /// Send `goal` once the node knows the base and a reading of every
        /// sensor: while it refuses the goal with `resource_unavailable`, step
        /// and send it again. The base rests meanwhile, so the step the goal
        /// is accepted at changes nothing of the world it drives in.
        pub async fn $send_when_ready(
            &mut self,
            goal: &$action::GoalRequestData,
        ) -> Goal<$action::GoalHandle, $action::Failure> {
            for _ in 0..READY_STEPS {
                let sent = self.$send(goal).await;
                match &sent.refusal {
                    Some(refusal) if refusal.kind == $action::FailureKind::ResourceUnavailable => {
                        self.step().await;
                    }
                    _ => return sent,
                }
            }
            panic!("the node waits {READY_STEPS} steps for the base or a sensor");
        }
    };
}

/// A goal's answer as the rig keeps it, written once here because the four
/// generated handles carry distinct types with an identical surface: the
/// handle shared with the test, and, once the node accepted the goal, the
/// goal in flight.
macro_rules! sent {
    ($($action:ident),+) => {
        $(
            impl From<$action::GoalHandle> for Goal<$action::GoalHandle, $action::Failure> {
                fn from(mut handle: $action::GoalHandle) -> Self {
                    Self {
                        accepted: handle.accepted,
                        refusal: handle.refusal.take(),
                        handle: Arc::new(Mutex::new(handle)),
                    }
                }
            }
        )+
    };
}

sent!(navigate_to_pose, spin, drive_on_heading, back_up);

impl World {
    /// The goal of `handle`, which the node answered: the goal in flight,
    /// `in_flight` of its handle, once the node accepted it.
    fn sent<H, F>(&mut self, handle: H, in_flight: impl FnOnce(Shared<H>) -> InFlight) -> Goal<H, F>
    where
        Goal<H, F>: From<H>,
    {
        let goal = Goal::from(handle);
        if goal.accepted {
            self.in_flight = Some(in_flight(goal.handle.clone()));
        }
        goal
    }
}

impl Rig {
    /// The node with `params` and the cameras of `mounts` before `scene`: the
    /// robot's sensors and the cameras' descriptions scripted, the cameras'
    /// frames fixed on the robot's tree, each plan starting at once, and the
    /// clock at a step before [`START_NS`], before the base's first state and
    /// before any answer of describe_base.
    pub async fn booted(params: peppygen::Parameters, scene: Scene, mounts: &[Mount]) -> Self {
        Self::booted_with(params, scene, mounts, &[], 0, Plans::AtOnce).await
    }

    /// [`Rig::booted`] with the lidars of `lidars` too, on a clock that reads
    /// `clock_lag_ns` behind the messages' stamps from its first tick on, the
    /// planner running `plans`, the room_map link vacant.
    pub async fn booted_with(
        params: peppygen::Parameters,
        scene: Scene,
        mounts: &[Mount],
        lidars: &[LidarMount],
        clock_lag_ns: i64,
        plans: Plans,
    ) -> Self {
        Self::booted_in(params, scene, mounts, lidars, clock_lag_ns, plans, None).await
    }

    /// [`Rig::booted_with`] with the room_map link bound to a fake that gives
    /// `room`, vacant for none.
    async fn booted_in(
        params: peppygen::Parameters,
        scene: Scene,
        mounts: &[Mount],
        lidars: &[LidarMount],
        clock_lag_ns: i64,
        plans: Plans,
        room: Option<RoomState>,
    ) -> Self {
        let Booted {
            harness,
            mocks,
            gate,
        } = boot(params, mounts, lidars, room.is_some(), plans).await;
        let Mocks { deps, pairings, .. } = mocks;
        let room = room.map(|state| {
            let mock = deps.room_map.expect("the harness binds the room_map mock");
            FakeRoomMap::new(mock, state)
        });
        deps.sensor_mounts
            .describe_sensors
            .enqueue_response(sensors_answer(mounts, lidars))
            .expect("sensor_mounts takes its answer");
        describe_cameras(mounts, &deps.cameras, &deps.geometry);
        let [backbone, initializer] = <[_; 2]>::try_from(deps.transforms)
            .unwrap_or_else(|_| panic!("the rig binds the backbone and the initializer"));
        fix_the_sensors(&initializer, mounts, lidars).await;
        let mut world = World {
            now_ns: START_NS - STEP_NS,
            base: MockBase::new(LIMITS, ODOMETRY_STARTED_AT_NS),
            scene,
            mounts: mounts.to_vec(),
            lidars: lidars.to_vec(),
            cameras_on: true,
            lidars_on: true,
            scan_frame: None,
            base_reports: true,
            deck_half: None,
            body_pitch_up_rad: 0.0,
            odometry_lag_steps: 0,
            clock_lag_ns,
            gate,
            base_state: deps.base_state,
            room,
            streamed: BTreeMap::new(),
            reported: BTreeMap::new(),
            wire: Wire {
                base: pairings.base,
                backbone,
                cameras: deps.cameras,
                lidars: deps.lidars,
                _initializer: initializer,
                _sensor_mounts: deps.sensor_mounts,
                _geometry: deps.geometry,
            },
            in_flight: None,
            reports: VecDeque::new(),
            frames_sent: 0,
            scans_sent: 0,
            truth_before: Planar::default(),
        };
        // The node's clock reads a time before the base's first state arrives.
        world.tick_and_see(&harness.clock).await;
        Self { harness, world }
    }

    /// [`Rig::booted`], then the base's first step with describe_base
    /// answering `answer`, every plan starting at once.
    pub async fn start_with(
        params: peppygen::Parameters,
        scene: Scene,
        mounts: &[Mount],
        answer: base_state::describe_base::ResponseData,
    ) -> Self {
        Self::start_with_lidars(params, scene, mounts, &[], answer).await
    }

    /// [`Rig::start_with`] with the lidars of `lidars` too.
    pub async fn start_with_lidars(
        params: peppygen::Parameters,
        scene: Scene,
        mounts: &[Mount],
        lidars: &[LidarMount],
        answer: base_state::describe_base::ResponseData,
    ) -> Self {
        let mut rig = Self::booted_with(params, scene, mounts, lidars, 0, Plans::AtOnce).await;
        rig.first_step(answer).await;
        rig
    }

    /// [`Rig::start_with`] for a base of `limits`, its odometry measured.
    pub async fn start_with_limits(
        params: peppygen::Parameters,
        scene: Scene,
        mounts: &[Mount],
        limits: Limits,
    ) -> Self {
        let mut rig = Self::booted(params, scene, mounts).await;
        rig.world.base = MockBase::new(limits, ODOMETRY_STARTED_AT_NS);
        rig.first_step(base_answer(limits, "measured")).await;
        rig
    }

    /// The node with the cameras of `mounts` and the lidars of `lidars` before
    /// `scene`, its room_map link bound to a fake that gives `room`, its plans
    /// running as `plans` says, after the base's first step with describe_base
    /// answering for the rig's base.
    pub async fn start_in_room(
        scene: Scene,
        mounts: &[Mount],
        lidars: &[LidarMount],
        plans: Plans,
        room: RoomState,
    ) -> Self {
        Self::start_in_room_with(params(), scene, mounts, lidars, plans, room).await
    }

    /// [`Rig::start_in_room`] with `params`.
    pub async fn start_in_room_with(
        params: peppygen::Parameters,
        scene: Scene,
        mounts: &[Mount],
        lidars: &[LidarMount],
        plans: Plans,
        room: RoomState,
    ) -> Self {
        let mut rig = Self::booted_in(params, scene, mounts, lidars, 0, plans, Some(room)).await;
        rig.first_step(measured()).await;
        rig
    }

    /// [`Rig::start`] with the plans held: no plan starts until the test
    /// releases it.
    pub async fn start_held(scene: Scene) -> Self {
        let mut rig = Self::booted_with(params(), scene, &[FRONT], &[], 0, Plans::Held).await;
        rig.first_step(measured()).await;
        rig
    }

    /// [`Rig::start`] on a clock that reads `clock_lag_ns` behind the
    /// messages' stamps.
    pub async fn start_lagging(scene: Scene, clock_lag_ns: i64) -> Self {
        let mut rig =
            Self::booted_with(params(), scene, &[FRONT], &[], clock_lag_ns, Plans::AtOnce).await;
        rig.first_step(measured()).await;
        rig
    }

    /// The base's first step, describe_base answering `answer`.
    async fn first_step(&mut self, answer: base_state::describe_base::ResponseData) {
        self.answer_describe_base(answer);
        self.step().await;
    }

    /// Step until the node knows the base and a reading of every sensor, and
    /// leave no goal in flight: while drive_states report follower_goal, a
    /// goal is refused with `precondition_failed` once every earlier check
    /// passes, and with `resource_unavailable` before.
    pub async fn ready(&mut self) {
        let follows = self.world.base.follow_state;
        self.world.base.follow_state = FollowState::FollowerGoal;
        self.step().await;
        let probe = goal(0.5, 0.0);
        let mut known = false;
        for _ in 0..READY_STEPS {
            let refusal = self.refusal(&probe).await;
            match refusal.kind {
                navigate_to_pose::FailureKind::ResourceUnavailable => {
                    self.step().await;
                }
                navigate_to_pose::FailureKind::PreconditionFailed => {
                    known = true;
                    break;
                }
                other => panic!("the readiness probe is refused with {other:?}: {refusal:?}"),
            }
        }
        assert!(
            known,
            "the node waits {READY_STEPS} steps for the base or a sensor"
        );
        self.world.base.follow_state = follows;
        self.step().await;
    }

    /// The node with the front camera before `scene`, the rig's base
    /// measuring its odometry.
    pub async fn start(scene: Scene) -> Self {
        Self::start_with(params(), scene, &[FRONT], measured()).await
    }

    /// The fake room_map of a rig that binds the room_map link.
    pub fn room(&self) -> &FakeRoomMap {
        self.world
            .room
            .as_ref()
            .expect("the rig binds the room_map link")
    }

    /// Publish `map`'s grid on map_grids, stamped with the rig's time.
    pub async fn publish_grid(&self, map: &FakeMap) {
        self.room().publish_grid(map, wire(self.world.now_ns)).await;
    }

    /// Step until the node logged a message that holds each of `parts`:
    /// a test reads the node's log from [`log::keep_the_log`] on.
    pub async fn until_logged(&mut self, parts: &[&str]) {
        for _ in 0..READY_STEPS {
            if !logged_with(parts).is_empty() {
                return;
            }
            self.step().await;
        }
        panic!("the node logged no {parts:?} in {READY_STEPS} steps");
    }

    /// Answer the node's next describe_base with `answer`.
    pub fn answer_describe_base(&self, answer: base_state::describe_base::ResponseData) {
        self.world
            .base_state
            .describe_base
            .enqueue_response(answer)
            .expect("base_state takes its answer");
    }

    pub async fn step(&mut self) -> StepSeen {
        self.world.step(&self.harness.clock).await
    }

    /// `count` steps.
    pub async fn steps(&mut self, count: usize) -> Vec<StepSeen> {
        let mut seen = Vec::with_capacity(count);
        for _ in 0..count {
            seen.push(self.step().await);
        }
        seen
    }

    /// The clock leaps over `steps` steps, the base's and the sensors' inputs
    /// arriving meanwhile.
    pub async fn leap(&mut self, steps: usize) -> StepSeen {
        self.world.leap(steps, &self.harness.clock).await
    }

    sender!(send, navigate_to_pose, Navigate);
    ready_sender!(send_when_ready, send, navigate_to_pose);
    sender!(send_spin, spin, Spin);
    ready_sender!(send_spin_when_ready, send_spin, spin);
    sender!(send_drive, drive_on_heading, DriveOnHeading);
    ready_sender!(send_drive_when_ready, send_drive, drive_on_heading);
    sender!(send_back_up, back_up, BackUp);

    /// Send `goal`, stepping the rig while the node has not answered: a goal
    /// whose checks wait on the node's clock is answered once the clock reaches
    /// the end of the wait. A goal in map against a fake room_map that answers
    /// get_map_state first waits, the clock standing still, until the fake
    /// answered an ask after the send, so every map_poses of the steps arrives
    /// after the node asked, whatever the host's speed.
    pub async fn send_stepping(
        &mut self,
        goal: &navigate_to_pose::GoalRequestData,
    ) -> Goal<navigate_to_pose::GoalHandle, navigate_to_pose::Failure> {
        let in_map = goal.target.frame.as_deref() == Some(MAP_FRAME);
        let room = self.world.room.as_ref().filter(|_| in_map);
        let mut answered = room.filter(|room| room.answers()).map(|room| {
            let mut answered = room.answered();
            answered.borrow_and_update();
            answered
        });
        let sent = navigate_to_pose::send_goal(&self.harness, goal, FEEDBACK_KEPT, ANSWER_BUDGET);
        tokio::pin!(sent);
        if let Some(answered) = answered.as_mut() {
            let first = within("the fake's answer of get_map_state, or the goal's", async {
                tokio::select! {
                    handle = &mut sent => Some(handle),
                    changed = answered.changed() => {
                        changed.expect("the fake counts its answers while it lives");
                        None
                    }
                }
            })
            .await;
            if let Some(handle) = first {
                let handle = handle.expect("navigate_to_pose answers");
                return self.world.sent(handle, InFlight::Navigate);
            }
        }
        for _ in 0..READY_STEPS {
            let answered = tokio::select! {
                biased;
                handle = &mut sent => Some(handle),
                () = std::future::ready(()) => None,
            };
            if let Some(handle) = answered {
                let handle = handle.expect("navigate_to_pose answers");
                return self.world.sent(handle, InFlight::Navigate);
            }
            self.world.step(&self.harness.clock).await;
        }
        panic!("the node did not answer the goal in {READY_STEPS} steps");
    }

    /// The failure `goal` is refused with now.
    pub async fn refusal(
        &mut self,
        goal: &navigate_to_pose::GoalRequestData,
    ) -> navigate_to_pose::Failure {
        self.send(goal)
            .await
            .refusal
            .unwrap_or_else(|| panic!("the node accepted {goal:?}"))
    }

    /// The failure `goal` is refused with once `holds` holds of it, stepping
    /// while it does not: a refusal that names what the node has read by then.
    pub async fn refusal_once(
        &mut self,
        goal: &navigate_to_pose::GoalRequestData,
        mut holds: impl FnMut(&navigate_to_pose::Failure) -> bool,
    ) -> navigate_to_pose::Failure {
        for _ in 0..READY_STEPS {
            let refusal = self.refusal(goal).await;
            if holds(&refusal) {
                return refusal;
            }
            self.step().await;
        }
        panic!("no refusal of {goal:?} in {READY_STEPS} steps held");
    }

    /// The failure `goal` is refused with, stepping while the node waits on
    /// its clock to answer.
    pub async fn refusal_stepping(
        &mut self,
        goal: &navigate_to_pose::GoalRequestData,
    ) -> navigate_to_pose::Failure {
        self.send_stepping(goal)
            .await
            .refusal
            .unwrap_or_else(|| panic!("the node accepted {goal:?}"))
    }

    /// Step `steps` times, the goal in flight running on through each.
    pub async fn run_on(&mut self, steps: usize) -> Vec<StepSeen> {
        let seen = self.steps(steps).await;
        if let Some(ended) = seen.iter().position(|step| step.goal_ended) {
            panic!("the goal ended early, at step {ended} of {steps}");
        }
        seen
    }

    /// Step until the goal in flight ends, the step that ended it included.
    pub async fn run_until_ended(&mut self) -> Vec<StepSeen> {
        let mut seen = Vec::new();
        for _ in 0..DRIVE_STEPS {
            let step = self.step().await;
            let ended = step.goal_ended;
            seen.push(step);
            if ended {
                return seen;
            }
        }
        panic!("the goal ran {DRIVE_STEPS} steps without an end");
    }

    /// Step the goal in flight until `holds` holds of the base, at most
    /// [`DRIVE_STEPS`] steps.
    pub async fn run_until(&mut self, mut holds: impl FnMut(&MockBase) -> bool) -> Vec<StepSeen> {
        let mut seen = Vec::new();
        for _ in 0..DRIVE_STEPS {
            if holds(&self.world.base) {
                return seen;
            }
            seen.extend(self.run_on(1).await);
        }
        panic!("the base did not reach the state in {DRIVE_STEPS} steps");
    }

    /// Step until the goal in flight ends, checking `each` of the base after
    /// every step.
    pub async fn run_checking(&mut self, mut each: impl FnMut(&MockBase)) -> Vec<StepSeen> {
        let mut seen = Vec::new();
        for _ in 0..DRIVE_STEPS {
            let step = self.step().await;
            each(&self.world.base);
            let ended = step.goal_ended;
            seen.push(step);
            if ended {
                return seen;
            }
        }
        panic!("the goal ran {DRIVE_STEPS} steps without an end");
    }

    /// Step the navigation in flight until `stop` holds of a step's feedback:
    /// the steps, the one `stop` held of last.
    pub async fn run_until_feedback(
        &mut self,
        mut stop: impl FnMut(&navigate_to_pose::Feedback) -> bool,
    ) -> Vec<StepSeen> {
        let mut seen = Vec::new();
        for _ in 0..DRIVE_STEPS {
            let step = self.step().await;
            assert!(!step.goal_ended, "the navigation ended early");
            let stops = stop(step.navigation().expect("a navigation's feedback"));
            seen.push(step);
            if stops {
                return seen;
            }
        }
        panic!("no feedback in {DRIVE_STEPS} steps held");
    }

    /// The setpoints streamed from the step at `ended_ns` on, at which a goal
    /// ended: the release burst after it, if any. The rig steps 10 steps,
    /// twice the steps of a burst at 50 ms, then sends a spin of
    /// [`FENCE_TURN_RAD`], whose first setpoint proves that every setpoint
    /// streamed before it arrived. The spin then runs.
    pub async fn burst_after(&mut self, ended_ns: i64) -> Vec<(i64, Setpoint)> {
        self.steps(10).await;
        let fence_ns = self.fence().await;
        self.world.setpoints_in(ended_ns..fence_ns)
    }

    /// Send a spin of [`FENCE_TURN_RAD`] once the node takes goals and step
    /// until its first setpoint arrives: every setpoint the node streamed
    /// before it arrived then. Returns that setpoint's stamp. The spin then
    /// runs.
    pub async fn fence(&mut self) -> i64 {
        let fence = self
            .send_spin_when_ready(&spin_by(FENCE_TURN_RAD, 0.0))
            .await;
        assert!(fence.accepted, "the fence spin: {:?}", fence.refusal);
        let first = self.step().await;
        assert!(first.feedback.is_some(), "the fence spin turns");
        first.at_ns
    }

    pub async fn clear_grid(&self) {
        within("clear_grid", clear_grid::poll(&self.harness, ANSWER_BUDGET))
            .await
            .expect("clear_grid answers");
    }

    pub async fn shutdown(self) {
        self.harness
            .shutdown()
            .await
            .expect("the node shuts down cleanly");
    }
}

/// One reading of the ending of a goal of the action `$action`, off its
/// result, written once here because the four generated action modules carry
/// distinct types with an identical surface.
macro_rules! outcome {
    (result_of, $action:ident) => {
        /// The result of `goal`, which has ended.
        pub async fn result_of(goal: &Sent) -> $action::ResultResponse {
            let handle = goal.handle.lock().await;
            within("the goal's result", handle.get_result(ANSWER_BUDGET))
                .await
                .expect("the goal ends with a result")
        }
    };
    (completed, $action:ident) => {
        pub fn completed(result: &$action::ResultResponse) -> &$action::ResultData {
            match &result.outcome {
                $action::ResultOutcome::Completed(data) => data,
                other => panic!("the goal did not complete: {other:?}"),
            }
        }
    };
    (failure, $action:ident) => {
        pub fn failure(result: &$action::ResultResponse) -> &$action::Failure {
            match &result.outcome {
                $action::ResultOutcome::Failed(failure) => failure,
                other => panic!("the goal did not fail: {other:?}"),
            }
        }
    };
    (cancelled, $action:ident) => {
        /// The reason and the result of a goal that ended cancelled.
        pub fn cancelled(result: &$action::ResultResponse) -> (&str, &$action::ResultData) {
            match &result.outcome {
                $action::ResultOutcome::Cancelled { reason, data } => (reason, data),
                other => panic!("the goal did not end cancelled: {other:?}"),
            }
        }
    };
    (cancel, $action:ident) => {
        /// Cancel `goal`, which the node signals at once.
        pub async fn cancel(goal: &Sent) {
            let handle = goal.handle.lock().await;
            let state = within("the cancel", handle.cancel_goal(ANSWER_BUDGET))
                .await
                .expect("the cancel answers");
            assert_eq!(state, $action::CancelState::Signalled);
        }
    };
}

/// The readings `$part` of the endings of the goals of `$action`.
macro_rules! outcomes {
    ($action:ident: $($part:ident),+) => {
        pub mod $action {
            use super::super::*;

            /// A goal of the action, as the rig sent it.
            pub type Sent = Goal<$action::GoalHandle, $action::Failure>;

            $(outcome!($part, $action);)+
        }
    };
}

/// The endings of each action's goals, each reading the tests take.
pub mod outcome {
    outcomes!(navigate_to_pose: result_of, completed, failure, cancelled, cancel);
    outcomes!(spin: result_of, completed, failure, cancelled, cancel);
    outcomes!(drive_on_heading: result_of, completed);
    outcomes!(back_up: result_of, completed);
}

/// The base's footprint outline in the world where `base` stands, 20 points
/// along each edge.
pub fn outline(base: &MockBase) -> Vec<[f64; 2]> {
    let Planar { x, y, yaw } = base.truth;
    let (sin, cos) = yaw.sin_cos();
    let place = |[px, py]: [f64; 2]| [x + cos * px - sin * py, y + sin * px + cos * py];
    (0..FOOTPRINT.len())
        .flat_map(|i| {
            let (a, b) = (FOOTPRINT[i], FOOTPRINT[(i + 1) % FOOTPRINT.len()]);
            (0..20).map(move |k| {
                let t = f64::from(k) / 20.0;
                [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
            })
        })
        .map(place)
        .collect()
}

/// Whether the footprint's outline where `base` stands meets `obstacle`'s
/// outline on the floor.
pub fn touches(base: &MockBase, obstacle: &scene::Box3) -> bool {
    outline(base).iter().any(|point| obstacle.covers(*point))
}

/// The yaw (rad) of a result's or feedback's orientation.
pub fn yaw_of(orientation: [f64; 4]) -> f64 {
    frame_tree::arithmetic::yaw_of_quaternion(orientation)
}
