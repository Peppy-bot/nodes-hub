//! Every subscription the backbone holds. (Peppy vocabulary throughout: a
//! pairing *slot* delivers one direction of its pairing's two one-way *streams*
//! from its one *peer*; a *subscription* receives a stream, live and never
//! replayed, and delivers nothing while the slot is unpaired.)
//!
//! The streams: the leading node's per-limb setpoints (the upstream pairings'
//! command direction), both paired arms' measured joint state, both paired
//! grippers' measured aperture, and the runtime governor controls. Each
//! listener holds one subscription and keeps the latest well-formed message in
//! a watch channel the coordinator reads every tick. One held subscription per
//! stream means no re-subscribe gap, so a message is never dropped between
//! receives.

use std::fmt::Display;
use std::future::{Future, ready};
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use control_core::geometry::{Pose, Stamped};
use control_core::receiving::{Tagged, receive, tagged};
use control_core::robot_tree::Placer;
use control_core::stream_window::StreamWindow;
use control_core::throttle::{LOG_WINDOW, Throttle};
use peppygen::NodeRunner;
use peppygen::consumed_topics::collision_ctrl::governor_control as collision_ctrl_governor_control;
use peppygen::paired_topics::{
    leader_left_arm, leader_left_arm_pose, leader_left_gripper, leader_right_arm,
    leader_right_arm_pose, leader_right_gripper, left_arm, left_gripper, right_arm, right_gripper,
};
use peppylib::runtime::CancellationToken;
use robot_description::{ArmId, PerArm};
use srs_model::JointVec;
use tokio::sync::watch;
use tracing::{error, warn};

use crate::robot::{ARMS, LIMBS};
use crate::upstream::Upstream;

/// The latest commanded opening for one gripper, fed by the upstream
/// pairing's `gripper_setpoints`: the raw wire opening fraction, clamped into
/// `[0, 1]` by the coordinator as it applies it, plus the commanded effort
/// cap (`None` when the wire carried no preference).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GripperCommand {
    pub opening: f64,
    pub max_effort: Option<f64>,
}

/// The latest measured joint state for one arm, fed by the joint_link
/// pairing's `joint_states` back-channel. The backbone anchors trajectories and
/// governor queries on position; the velocities ride along for the upstream
/// state relay only. `timestamp` is the time of the measurement, as its
/// follower stamps it on the clock of transforms: the time of each pose the
/// backbone reports from these joints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmState {
    pub positions: JointVec,
    pub velocities: JointVec,
    pub timestamp: SystemTime,
}

/// The latest measured state of one gripper: the opening clamped at ingestion
/// into `[0, 1]` (0 = closed, 1 = fully open), so downstream consumers never
/// see an out-of-range placement, plus the measured effort and the follower's
/// effort ceiling, which ride along for the upstream state relay only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GripperState {
    pub fraction: f64,
    pub effort: f64,
    pub max_effort: f64,
}

/// Parses one paired arm's inbound joint_states payload. This backbone drives
/// fixed 7-joint arms whose followers always measure velocity, so a message
/// must carry exactly [`ARM_DOF`] positions with matching velocities,
/// all finite; the generic contract's empty-velocities form is deliberately
/// rejected here rather than half-accepted.
fn parse_joint_state(
    timestamp: SystemTime,
    positions: Vec<f64>,
    velocities: Vec<f64>,
) -> Result<ArmState, &'static str> {
    let finite = positions
        .iter()
        .chain(velocities.iter())
        .all(|v| v.is_finite());
    let Ok(positions) = JointVec::try_from(positions) else {
        return Err("a non-arm joint count");
    };
    let Ok(velocities) = JointVec::try_from(velocities) else {
        return Err("a velocity count that does not match its joints");
    };
    if !finite {
        return Err("non-finite values");
    }
    Ok(ArmState {
        positions,
        velocities,
        timestamp,
    })
}

/// Parses one commanded arm setpoint: exactly [`ARM_DOF`] finite
/// positions. The wire's velocities and efforts are not parsed: the backbone
/// plans its own velocity shaping over the governed position stream.
fn parse_joint_command(positions: Vec<f64>) -> Result<Upstream, &'static str> {
    let finite = positions.iter().all(|v| v.is_finite());
    let Ok(positions) = JointVec::try_from(positions) else {
        return Err("a non-arm joint count");
    };
    if !finite {
        return Err("non-finite values");
    }
    Ok(Upstream::Joints(positions))
}

/// One commanded end-effector pose of pose_link, field for field.
struct PoseSetpoint {
    frame: Option<String>,
    timestamp: Option<SystemTime>,
    position: [f64; 3],
    orientation: [f64; 4],
}

/// Parses one commanded end-effector pose into the geometry types, in its
/// frame or, with none, in the robot frame, and places it in the robot frame
/// by the robot's tree, as pose_link places a setpoint; a pose no contract
/// takes, as one whose orientation is no unit quaternion,
/// and one the tree cannot place are refused, and the listener drops them.
async fn place_pose_command(
    placer: Arc<Placer>,
    setpoint: PoseSetpoint,
) -> Result<Upstream, String> {
    let stamped = Pose::parse(setpoint.position, setpoint.orientation)
        .and_then(|pose| {
            let frame = placer.frame_of(setpoint.frame.as_deref());
            Stamped::parse(frame, setpoint.timestamp, pose)
        })
        .map_err(|e| format!("a pose no contract takes: {e}"))?;
    let placed = placer
        .place(&stamped)
        .await
        .map_err(|e| format!("a pose the robot's tree cannot place: {e}"))?;
    Ok(Upstream::Pose(placed.value.into()))
}

/// Parses one commanded gripper setpoint: finite fields, a non-negative effort
/// cap, with the wire's 0 (no preference) parsed to `None`. The opening is
/// clamped later by the coordinator, matching the arm command path.
fn parse_gripper_command(opening: f64, max_effort: f64) -> Result<GripperCommand, &'static str> {
    if !opening.is_finite() || !max_effort.is_finite() {
        return Err("non-finite values");
    }
    if max_effort < 0.0 {
        return Err("a negative max_effort");
    }
    Ok(GripperCommand {
        opening,
        max_effort: (max_effort > 0.0).then_some(max_effort),
    })
}

/// Parses one paired gripper's inbound measured state: finite fields, the
/// fraction clamped into the contract's `[0, 1]`, the ceiling non-negative.
fn parse_gripper_state(
    opening: f64,
    effort: f64,
    max_effort: f64,
) -> Result<GripperState, &'static str> {
    if !opening.is_finite() || !effort.is_finite() || !max_effort.is_finite() {
        return Err("non-finite values");
    }
    if max_effort < 0.0 {
        return Err("a negative effort ceiling");
    }
    Ok(GripperState {
        fraction: opening.clamp(0.0, 1.0),
        effort,
        max_effort,
    })
}

/// The arms the `left_*` and the `right_*` pairing slots serve, in that order:
/// the first and the second arm, as the robot's limb names place `left_arm`
/// and `right_arm`.
fn slot_arms() -> [ArmId; ARMS] {
    PerArm::from_fn(|arm| arm).into_array()
}

/// Subscribe both of a pairing's slots, naming the stream if either fails.
/// `None` ends the listener: a stream that cannot be subscribed never delivers.
async fn subscribe_pair<L, R>(
    what: &'static str,
    left: impl Future<Output = peppygen::Result<L>>,
    right: impl Future<Output = peppygen::Result<R>>,
) -> Option<(L, R)> {
    match tokio::join!(left, right) {
        (Ok(l), Ok(r)) => Some((l, r)),
        (l, r) => {
            error!("{what} subscribe: left {:?}, right {:?}", l.err(), r.err());
            None
        }
    }
}

/// A leader command younger than this means a leader is streaming the limb:
/// the limb is the leader's, and a goal on it is refused until the stream
/// stops; a limb whose leader has stopped streaming holds its setpoint, its
/// last command dropped. The window is the SO-101 backbone's leader deadman and the
/// headset's release burst, both a quarter second: a refusal lasts as long as
/// a follower would still follow the stream, and the burst after a grip is
/// released reads as one unbroken stream. It covers the gaps of a stream at
/// 4 Hz or faster.
pub const LEADER_STREAMING_WINDOW: Duration = Duration::from_millis(250);

/// A leader's stream of one limb, on the monotonic clock: live within
/// [`LEADER_STREAMING_WINDOW`] of its newest command.
const LEADER_STREAM: StreamWindow<Duration> = StreamWindow::new(LEADER_STREAMING_WINDOW);

/// When a leader last streamed a well-formed command for one limb. Written by
/// the limb's listener, read by the goal handlers, which refuse a goal on a
/// limb a leader is streaming, and by the coordinator, which follows a command
/// only while its leader is streaming.
#[derive(Clone, Default)]
pub struct LeaderActivity {
    last: Arc<Mutex<Option<Instant>>>,
}

impl LeaderActivity {
    /// A leader that streams however long a test takes: its last command is
    /// stamped an hour ahead, so no test host is slow enough to end the
    /// stream.
    #[cfg(test)]
    pub fn streaming_always() -> Self {
        let ahead = Instant::now() + Duration::from_secs(3600);
        Self {
            last: Arc::new(Mutex::new(Some(ahead))),
        }
    }

    /// A leader command arrived now.
    pub fn note(&self) {
        *self.last.lock().expect("stamp lock poisoned") = Some(Instant::now());
    }

    /// Whether a leader command arrived within [`LEADER_STREAMING_WINDOW`].
    pub fn streaming(&self) -> bool {
        let last = *self.last.lock().expect("stamp lock poisoned");
        LEADER_STREAM.streams(last, Instant::now())
    }
}

/// Receive a paired stream until `cancelled` resolves or a subscription
/// `next_message` receives closes, keeping the latest well-formed message per
/// arm: `next_message` gives the next message tagged with its arm, by a
/// select over both arms' subscriptions, so whichever arm delivers next wins
/// and the other stays queued in its own subscription, or from one arm's
/// subscription alone ([`follow_pose_setpoints`]).
///
/// This is the whole receive policy, in one place: a well-formed message
/// replaces that arm's latest and, on a leader's stream, notes the leader's
/// activity, and a malformed one is dropped with a throttled reason and
/// drives no arm. Receive errors take control_core's receive policy. Each
/// generated slot has its own message type and the subscriptions share no
/// trait, so the select stays with its listener; everything after it is here.
async fn follow_pair<Raw, Parsed, Reason: Display, Parsing>(
    what: &'static str,
    cancelled: impl Future<Output = ()>,
    mut next_message: impl AsyncFnMut() -> Result<Option<(ArmId, Raw)>, Tagged<ArmId, peppygen::Error>>,
    parse: impl Fn(Raw) -> Parsing,
    latest: PerArm<watch::Sender<Option<Parsed>>, ARMS>,
    activity: Option<PerArm<LeaderActivity, ARMS>>,
) where
    Parsing: Future<Output = Result<Parsed, Reason>>,
{
    let mut reject_warn = Throttle::new(LOG_WINDOW);
    let stream = what.to_owned();
    let named_errors = async move || {
        next_message()
            .await
            .map_err(|Tagged { tag, error }| Tagged {
                tag: LIMBS.arm_name(tag),
                error,
            })
    };
    receive(what, cancelled, named_errors, async move |(arm, raw)| {
        match parse(raw).await {
            Ok(parsed) => {
                latest[arm].send_replace(Some(parsed));
                if let Some(activity) = &activity {
                    activity[arm].note();
                }
            }
            Err(reason) => {
                if reject_warn.admit() {
                    warn!(
                        "{stream}: dropping {} message with {reason}",
                        LIMBS.arm_name(arm)
                    );
                }
            }
        }
        ControlFlow::Continue(())
    })
    .await;
}

/// Receive both arms' upstream pose streams forever, keeping the latest
/// well-formed pose per arm, placed in the robot frame by `placer`. The
/// Cartesian mirror of [`run_joint_command_listener`]; `upstream_mode` spawns
/// exactly one of the two.
pub async fn run_pose_command_listener(
    runner: Arc<NodeRunner>,
    placer: Arc<Placer>,
    latest: PerArm<watch::Sender<Option<Upstream>>, ARMS>,
    activity: PerArm<LeaderActivity, ARMS>,
) {
    const WHAT: &str = "upstream pose_setpoints";
    let [left_arm, right_arm] = slot_arms();
    let Some((mut left, mut right)) = subscribe_pair(
        WHAT,
        leader_left_arm_pose::pose_setpoints::subscribe(&runner),
        leader_right_arm_pose::pose_setpoints::subscribe(&runner),
    )
    .await
    else {
        return;
    };
    follow_pose_setpoints(
        WHAT,
        runner.cancellation_token(),
        async move || {
            tagged(
                left_arm,
                left.next().await.map(|m| {
                    m.map(|(_, msg)| PoseSetpoint {
                        frame: msg.frame,
                        timestamp: msg.timestamp,
                        position: msg.position,
                        orientation: msg.orientation,
                    })
                }),
            )
        },
        async move || {
            tagged(
                right_arm,
                right.next().await.map(|m| {
                    m.map(|(_, msg)| PoseSetpoint {
                        frame: msg.frame,
                        timestamp: msg.timestamp,
                        position: msg.position,
                        orientation: msg.orientation,
                    })
                }),
            )
        },
        placer,
        latest,
        activity,
    )
    .await;
}

/// Follow both arms' pose setpoints as [`follow_pair`] follows a stream, each
/// arm's on a loop of its own, until `stopped` cancels or either arm's
/// subscription closes: `next_left` and `next_right` receive the next
/// setpoint of each arm, and `placer` places it, waiting up to its lookup
/// timeout for the tree, which holds back only the setpoints of that arm.
async fn follow_pose_setpoints(
    what: &'static str,
    stopped: &CancellationToken,
    next_left: impl AsyncFnMut()
        -> Result<Option<(ArmId, PoseSetpoint)>, Tagged<ArmId, peppygen::Error>>,
    next_right: impl AsyncFnMut()
        -> Result<Option<(ArmId, PoseSetpoint)>, Tagged<ArmId, peppygen::Error>>,
    placer: Arc<Placer>,
    latest: PerArm<watch::Sender<Option<Upstream>>, ARMS>,
    activity: PerArm<LeaderActivity, ARMS>,
) {
    let left_placer = placer.clone();
    tokio::select! {
        () = follow_pair(
            what,
            stopped.cancelled(),
            next_left,
            move |setpoint| place_pose_command(left_placer.clone(), setpoint),
            latest.clone(),
            Some(activity.clone()),
        ) => {}
        () = follow_pair(
            what,
            stopped.cancelled(),
            next_right,
            move |setpoint| place_pose_command(placer.clone(), setpoint),
            latest,
            Some(activity),
        ) => {}
    }
}

/// Receive both upstream arm setpoint streams forever (the leading node's
/// joint_link command direction), keeping the latest well-formed message per
/// arm. The slot IS the arm (a pairing delivers only its one peer), so there
/// is no id demux. Whichever slot delivers next wins the select; the other
/// stays queued in its own subscription, so neither arm can starve the other.
pub async fn run_joint_command_listener(
    runner: Arc<NodeRunner>,
    latest: PerArm<watch::Sender<Option<Upstream>>, ARMS>,
    activity: PerArm<LeaderActivity, ARMS>,
) {
    const WHAT: &str = "upstream joint_setpoints";
    let [left_arm, right_arm] = slot_arms();
    let Some((mut left, mut right)) = subscribe_pair(
        WHAT,
        leader_left_arm::joint_setpoints::subscribe(&runner),
        leader_right_arm::joint_setpoints::subscribe(&runner),
    )
    .await
    else {
        return;
    };
    follow_pair(
        WHAT,
        runner.cancellation_token().cancelled(),
        async move || {
            tokio::select! {
                r = left.next() => tagged(left_arm, r.map(|m| m.map(|(_, msg)| msg.positions))),
                r = right.next() => tagged(right_arm, r.map(|m| m.map(|(_, msg)| msg.positions))),
            }
        },
        |positions| ready(parse_joint_command(positions)),
        latest,
        Some(activity),
    )
    .await;
}

/// Receive both upstream gripper setpoint streams forever, keeping the latest
/// well-formed command per arm. The 1-DOF mirror of
/// [`run_joint_command_listener`].
pub async fn run_gripper_command_listener(
    runner: Arc<NodeRunner>,
    latest: PerArm<watch::Sender<Option<GripperCommand>>, ARMS>,
    activity: PerArm<LeaderActivity, ARMS>,
) {
    const WHAT: &str = "upstream gripper_setpoints";
    let [left_arm, right_arm] = slot_arms();
    let Some((mut left, mut right)) = subscribe_pair(
        WHAT,
        leader_left_gripper::gripper_setpoints::subscribe(&runner),
        leader_right_gripper::gripper_setpoints::subscribe(&runner),
    )
    .await
    else {
        return;
    };
    follow_pair(
        WHAT,
        runner.cancellation_token().cancelled(),
        async move || tokio::select! {
            r = left.next() => tagged(left_arm, r.map(|m| m.map(|(_, msg)| (msg.opening, msg.max_effort)))),
            r = right.next() => tagged(right_arm, r.map(|m| m.map(|(_, msg)| (msg.opening, msg.max_effort)))),
        },
        |(opening, max_effort)| ready(parse_gripper_command(opening, max_effort)),
        latest,
        Some(activity),
    )
    .await;
}

/// Receive both paired arms' measured state forever (the joint_link
/// back-channel), keeping the latest per arm. The governor anchors only on its
/// exclusive command-loop peers: a stray broadcast producer cannot pose as an
/// arm. Non-finite states are dropped so the coordinator never anchors a
/// trajectory or a governor query on a bad measurement.
pub async fn run_joint_state_listener(
    runner: Arc<NodeRunner>,
    latest: PerArm<watch::Sender<Option<ArmState>>, ARMS>,
) {
    const WHAT: &str = "joint_states";
    let [left_arm, right_arm] = slot_arms();
    let Some((mut left, mut right)) = subscribe_pair(
        WHAT,
        left_arm::joint_states::subscribe(&runner),
        right_arm::joint_states::subscribe(&runner),
    )
    .await
    else {
        return;
    };
    follow_pair(
        WHAT,
        runner.cancellation_token().cancelled(),
        async move || tokio::select! {
            r = left.next() => tagged(left_arm, r.map(|m| m.map(|(_, msg)| (msg.timestamp, msg.positions, msg.velocities)))),
            r = right.next() => tagged(right_arm, r.map(|m| m.map(|(_, msg)| (msg.timestamp, msg.positions, msg.velocities)))),
        },
        |(timestamp, positions, velocities)| ready(parse_joint_state(timestamp, positions, velocities)),
        latest,
        None,
    )
    .await;
}

/// Receive both paired grippers' measured aperture forever (the gripper_link
/// back-channel), keeping the latest opening fraction per arm. The collision
/// model reads finger positions only from these exclusive command-loop peers,
/// so a stray broadcast producer cannot spoof the modeled fingers.
pub async fn run_gripper_state_listener(
    runner: Arc<NodeRunner>,
    latest: PerArm<watch::Sender<Option<GripperState>>, ARMS>,
) {
    const WHAT: &str = "gripper_states";
    let [left_arm, right_arm] = slot_arms();
    let Some((mut left, mut right)) = subscribe_pair(
        WHAT,
        left_gripper::gripper_states::subscribe(&runner),
        right_gripper::gripper_states::subscribe(&runner),
    )
    .await
    else {
        return;
    };
    follow_pair(
        WHAT,
        runner.cancellation_token().cancelled(),
        async move || tokio::select! {
            r = left.next() => tagged(left_arm, r.map(|m| m.map(|(_, msg)| (msg.opening, msg.effort, msg.max_effort)))),
            r = right.next() => tagged(right_arm, r.map(|m| m.map(|(_, msg)| (msg.opening, msg.effort, msg.max_effort)))),
        },
        |(opening, effort, max_effort)| ready(parse_gripper_state(opening, effort, max_effort)),
        latest,
        None,
    )
    .await;
}

/// The backbone's runtime governor controls from the commander: the on/off toggle plus
/// the live-tunable band and stream speed cap. Raw values; the governor and
/// planners validate them as they apply (an invalid band or speed is ignored,
/// keeping the last good one), so the toggle still takes effect regardless.
#[derive(Clone, Copy)]
pub struct GovernorConfig {
    pub enabled: bool,
    pub d_stop: f64,
    pub d_safe: f64,
    pub max_stream_ee_velocity_m_s: f64,
    pub max_gripper_rate_frac_s: f64,
}

/// Receive the `governor_control` stream forever, mirroring the latest governor
/// config into `config`. The producer re-publishes periodically, so the lossy QoS
/// cannot strand the backbone in a stale state.
pub async fn run_governor_config_listener(
    runner: Arc<NodeRunner>,
    config: watch::Sender<GovernorConfig>,
) {
    let mut subscription = match collision_ctrl_governor_control::subscribe(&runner).await {
        Ok(s) => s,
        Err(e) => return error!("governor_control subscribe: {e}"),
    };
    receive(
        "governor_control",
        runner.cancellation_token().cancelled(),
        async move || subscription.next().await,
        async move |(_, msg)| {
            config.send_replace(GovernorConfig {
                enabled: msg.collision_governor_enabled,
                d_stop: msg.d_stop,
                d_safe: msg.d_safe,
                max_stream_ee_velocity_m_s: msg.max_stream_ee_velocity_m_s,
                max_gripper_rate_frac_s: msg.max_gripper_rate_frac_s,
            });
            ControlFlow::Continue(())
        },
    )
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    use control_core::geometry::pose_arrays;
    use control_core::node_clock::NodeClock;
    use control_core::robot_tree::RobotTree;
    use frame_tree::{FrameName, MovingMessage, Publisher, StaticMessage, TransformRecord};
    use srs_model::ARM_DOF;
    use srs_model::nalgebra::{Isometry3, Vector3};

    use crate::test_support::{LEFT, RIGHT, measured_at};

    #[test]
    fn a_leader_streams_a_limb_for_the_window_after_its_last_command() {
        let activity = LeaderActivity::default();
        assert!(!activity.streaming());
        activity.note();
        assert!(activity.streaming());
        std::thread::sleep(LEADER_STREAMING_WINDOW + Duration::from_millis(20));
        assert!(!activity.streaming());
    }

    #[test]
    fn joint_state_requires_arm_dof_positions_with_matching_velocities() {
        let state =
            parse_joint_state(measured_at(), vec![0.1; ARM_DOF], vec![0.0; ARM_DOF]).unwrap();
        assert_eq!(state.positions, [0.1; ARM_DOF]);
        assert_eq!(state.timestamp, measured_at());
        assert!(
            parse_joint_state(
                measured_at(),
                vec![0.1; ARM_DOF - 1],
                vec![0.0; ARM_DOF - 1]
            )
            .is_err()
        );
        assert!(
            parse_joint_state(
                measured_at(),
                vec![0.1; ARM_DOF + 1],
                vec![0.0; ARM_DOF + 1]
            )
            .is_err()
        );
        assert!(parse_joint_state(measured_at(), vec![], vec![]).is_err());
        assert!(
            parse_joint_state(measured_at(), vec![0.1; ARM_DOF], vec![0.0; ARM_DOF - 1]).is_err()
        );
    }

    #[test]
    fn joint_state_rejects_the_contracts_empty_velocities_form() {
        // Deliberate strictness: this backbone's followers always measure
        // velocity, so the generic contract's unmeasured (empty) form is
        // treated as malformed instead of half-accepted.
        assert_eq!(
            parse_joint_state(measured_at(), vec![0.1; ARM_DOF], vec![]),
            Err("a velocity count that does not match its joints")
        );
    }

    #[test]
    fn joint_state_rejects_non_finite_values() {
        let mut positions = vec![0.1; ARM_DOF];
        positions[3] = f64::NAN;
        assert_eq!(
            parse_joint_state(measured_at(), positions, vec![0.0; ARM_DOF]),
            Err("non-finite values")
        );
        let mut velocities = vec![0.0; ARM_DOF];
        velocities[6] = f64::INFINITY;
        assert_eq!(
            parse_joint_state(measured_at(), vec![0.1; ARM_DOF], velocities),
            Err("non-finite values")
        );
    }

    fn frame(name: &str) -> FrameName {
        FrameName::parse(name).expect("a frame name")
    }

    /// A placer whose tree holds the frame `mount` fixed 1 m above the robot
    /// frame, `base_link`, and that waits `wait` for the tree.
    fn placer_waiting(wait: Duration) -> Placer {
        let tree = RobotTree::new(Duration::from_secs(10), NodeClock::wall()).expect("a window");
        let fixed = StaticMessage {
            transforms: vec![TransformRecord {
                frame: "base_link".to_string(),
                child_frame: "mount".to_string(),
                position: [0.0, 0.0, 1.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
            }],
        };
        let initializer = Publisher {
            core_node: "core".to_string(),
            instance_id: "init_inst".to_string(),
        };
        tree.take_static(&initializer, &fixed)
            .expect("a fixed transform");
        Placer::new(Arc::new(tree), frame("base_link"), wait)
    }

    /// A placer whose tree holds the frame `mount` fixed 1 m above the robot
    /// frame, `base_link`, and that waits for nothing.
    fn placer() -> Placer {
        placer_waiting(Duration::ZERO)
    }

    /// The robot frame of the tests' base standing at `x, y` (m) on the floor
    /// of odom, turned by `yaw` (rad) about +z.
    fn base_in_odom(x: f64, y: f64, yaw: f64) -> Isometry3<f64> {
        Isometry3::new(Vector3::new(x, y, 0.0), Vector3::new(0.0, 0.0, yaw))
    }

    /// A placer whose tree holds the robot frame, `base_link`, on a base that
    /// odom places at `base_then` at `then` and at `base_now` at `now`, the
    /// latest time it holds, on a clock that reads `now`, and that waits for
    /// nothing.
    fn placer_on_a_moving_base(
        then: SystemTime,
        base_then: &Isometry3<f64>,
        now: SystemTime,
        base_now: &Isometry3<f64>,
    ) -> Placer {
        let (clock, feed) = NodeClock::simulation();
        let now_ns = now
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("a time after the epoch")
            .as_nanos();
        feed.tick(u64::try_from(now_ns).expect("a time within u64 nanoseconds"));
        let tree = RobotTree::new(Duration::from_secs(10), clock).expect("a window");
        let odometry = Publisher {
            core_node: "core".to_string(),
            instance_id: "base_inst".to_string(),
        };
        for (timestamp, base) in [(then, base_then), (now, base_now)] {
            let (position, orientation) = pose_arrays(base);
            let measured = MovingMessage {
                timestamp,
                transforms: vec![TransformRecord {
                    frame: "odom".to_string(),
                    child_frame: "base_link".to_string(),
                    position,
                    orientation,
                }],
                start_timestamp: then,
            };
            tree.take_moving(&odometry, &measured)
                .expect("a transform of the base");
        }
        Placer::new(Arc::new(tree), frame("base_link"), Duration::ZERO)
    }

    /// A setpoint in `frame` at `timestamp`, or at the latest time for none, at
    /// `position` with `orientation`.
    fn setpoint(
        frame: &str,
        timestamp: Option<SystemTime>,
        position: [f64; 3],
        orientation: [f64; 4],
    ) -> PoseSetpoint {
        PoseSetpoint {
            frame: Some(frame.to_string()),
            timestamp,
            position,
            orientation,
        }
    }

    /// The pose `placer` places a setpoint at.
    async fn placed_by(
        placer: &Arc<Placer>,
        setpoint: PoseSetpoint,
    ) -> Result<Isometry3<f64>, String> {
        match place_pose_command(placer.clone(), setpoint).await? {
            Upstream::Pose(pose) => Ok(pose),
            Upstream::Joints(_) => panic!("the pose placement produced a joint command"),
        }
    }

    /// The pose a setpoint the placer takes is placed at.
    async fn placed(setpoint: PoseSetpoint) -> Result<Isometry3<f64>, String> {
        placed_by(&Arc::new(placer()), setpoint).await
    }

    // A setpoint in the robot frame of a robot whose robot frame is the root of
    // its tree is placed as it comes at any time; one in a frame of the tree is
    // placed by that frame.
    #[tokio::test]
    async fn a_pose_setpoint_is_placed_in_the_robot_frame_by_its_frame() {
        let identity = [0.0, 0.0, 0.0, 1.0];
        let long_ago = Some(SystemTime::now() - Duration::from_secs(3600));
        let in_robot = placed(setpoint("base_link", long_ago, [0.4, -0.1, 0.3], identity))
            .await
            .expect("a robot-frame setpoint");
        assert_eq!(in_robot.translation.vector, Vector3::new(0.4, -0.1, 0.3));
        let on_mount = placed(setpoint("mount", None, [0.4, -0.1, 0.3], identity))
            .await
            .expect("a setpoint in a frame of the tree");
        assert_eq!(on_mount.translation.vector, Vector3::new(0.4, -0.1, 1.3));
    }

    // A setpoint in the robot frame at a time names where the robot frame
    // stood then, as pose_link places one: on a base that moved since, it is
    // placed where that place stands in the robot frame now. One with no
    // timestamp names where the robot frame stands at the latest time. A
    // setpoint with no frame lies in the robot frame.
    #[tokio::test]
    async fn a_robot_frame_setpoint_at_a_time_is_placed_where_the_robot_frame_stood_then() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let then = now - Duration::from_secs(2);
        let base_then = base_in_odom(-0.05, 0.02, -0.1);
        let base_now = base_in_odom(0.10, 0.05, 0.2);
        let placer = Arc::new(placer_on_a_moving_base(then, &base_then, now, &base_now));
        let pose = Isometry3::new(Vector3::new(0.4, -0.1, 0.3), Vector3::new(0.0, 0.0, 0.3));
        let (position, orientation) = pose_arrays(&pose);

        let stood_then = placed_by(
            &placer,
            setpoint("base_link", Some(then), position, orientation),
        )
        .await
        .expect("a robot-frame setpoint at a time the tree holds");
        let expected = base_now.inverse() * base_then * pose;
        assert!(
            (stood_then.translation.vector - expected.translation.vector).norm() < 1e-12,
            "{stood_then} placed, {expected} expected"
        );
        assert!(stood_then.rotation.angle_to(&expected.rotation) < 1e-12);
        assert!(
            (stood_then.translation.vector - pose.translation.vector).norm() > 0.05,
            "the base moves far enough between the two times to tell the places apart"
        );
        let no_frame = PoseSetpoint {
            frame: None,
            ..setpoint("", Some(then), position, orientation)
        };
        assert_eq!(placed_by(&placer, no_frame).await, Ok(stood_then));

        let stands_now = placed_by(&placer, setpoint("base_link", None, position, orientation))
            .await
            .expect("a robot-frame setpoint at the latest time");
        assert!((stands_now.translation.vector - pose.translation.vector).norm() < 1e-12);
        assert!(stands_now.rotation.angle_to(&pose.rotation) < 1e-12);
    }

    // A setpoint that waits for the tree holds back only its own arm: the
    // other arm's setpoint, which comes after it, is placed while it waits.
    // Tokio's clock is paused and moves only while every task waits, so the
    // left arm's wait and the bound on the right arm's placement end on it
    // alone, the bound first.
    #[tokio::test(start_paused = true)]
    async fn a_setpoint_that_waits_for_the_tree_holds_back_only_its_own_arm() {
        type Received = Result<Option<(ArmId, PoseSetpoint)>, Tagged<ArmId, peppygen::Error>>;
        let identity = [0.0, 0.0, 0.0, 1.0];
        let placer = Arc::new(placer_waiting(Duration::from_secs(3600)));
        let latest = PerArm::from_fn(|_| watch::Sender::new(None));
        let mut right_latest = latest[RIGHT].subscribe();
        let left_taken = tokio::sync::Notify::new();
        let mut left_sent = false;
        let next_left = async || -> Received {
            if left_sent {
                return std::future::pending().await;
            }
            left_sent = true;
            left_taken.notify_one();
            Ok(Some((LEFT, setpoint("nowhere", None, [0.0; 3], identity))))
        };
        let mut right_sent = false;
        let next_right = async || -> Received {
            if right_sent {
                return std::future::pending().await;
            }
            left_taken.notified().await;
            right_sent = true;
            Ok(Some((
                RIGHT,
                setpoint("base_link", None, [0.4, -0.1, 0.3], identity),
            )))
        };
        let stopped = CancellationToken::new();
        let following = follow_pose_setpoints(
            "pose_setpoints",
            &stopped,
            next_left,
            next_right,
            placer,
            latest.clone(),
            PerArm::from_fn(|_| LeaderActivity::default()),
        );
        tokio::select! {
            () = following => panic!("the setpoints stopped being followed"),
            changed = tokio::time::timeout(Duration::from_secs(10), right_latest.changed()) => {
                changed
                    .expect("the right arm's setpoint is placed while the left arm's waits")
                    .expect("the right arm's latest is open");
            }
        }
        assert!(matches!(*right_latest.borrow(), Some(Upstream::Pose(_))));
        assert_eq!(*latest[LEFT].borrow(), None);
    }

    #[tokio::test]
    #[expect(
        clippy::approx_constant,
        reason = "a quarter turn typed to four places, as a caller types one"
    )]
    async fn a_pose_setpoint_typed_near_unit_is_normalized() {
        let typed = setpoint("base_link", None, [0.0; 3], [0.0, 0.0, 0.7071, 0.7071]);
        let pose = placed(typed)
            .await
            .expect("an orientation within tf2's tolerance");
        assert!((pose.rotation.quaternion().norm() - 1.0).abs() < 1e-12);
        assert!((pose.rotation.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-3);
    }

    // A setpoint no contract takes and one the tree cannot place are dropped,
    // the reason naming why.
    #[tokio::test]
    async fn a_pose_setpoint_off_unit_or_in_a_frame_the_tree_lacks_is_dropped() {
        let latest = None;
        let refused = |orientation| placed(setpoint("base_link", latest, [0.0; 3], orientation));
        for orientation in [[0.0, 0.0, 0.0, 1.006], [0.0; 4], [0.0, f64::NAN, 0.0, 1.0]] {
            let reason = refused(orientation).await.expect_err("no unit quaternion");
            assert!(reason.starts_with("a pose no contract takes: "), "{reason}");
        }
        let reason = placed(setpoint("odom", latest, [0.0; 3], [0.0, 0.0, 0.0, 1.0]))
            .await
            .expect_err("a frame the tree does not hold");
        assert!(
            reason.starts_with("a pose the robot's tree cannot place: no transform of the tree names the frame odom"),
            "{reason}"
        );
        let reason = placed(setpoint("", latest, [0.0; 3], [0.0, 0.0, 0.0, 1.0]))
            .await
            .expect_err("no frame");
        assert!(reason.starts_with("a pose no contract takes: "), "{reason}");
    }

    #[test]
    fn gripper_state_clamps_into_the_contract_range() {
        assert_eq!(parse_gripper_state(0.5, 0.0, 0.0).unwrap().fraction, 0.5);
        assert_eq!(parse_gripper_state(-0.2, 0.0, 0.0).unwrap().fraction, 0.0);
        assert_eq!(parse_gripper_state(1.7, 0.0, 0.0).unwrap().fraction, 1.0);
        assert!(parse_gripper_state(f64::NAN, 0.0, 0.0).is_err());
    }

    #[test]
    fn gripper_state_retains_effort_and_ceiling() {
        let state = parse_gripper_state(0.5, -0.4, 1.5).unwrap();
        assert_eq!(state.effort, -0.4);
        assert_eq!(state.max_effort, 1.5);
        assert!(parse_gripper_state(0.5, f64::NAN, 1.5).is_err());
        assert_eq!(
            parse_gripper_state(0.5, 0.0, -1.0),
            Err("a negative effort ceiling")
        );
    }

    #[test]
    fn joint_command_requires_arm_dof_finite_positions() {
        assert_eq!(
            parse_joint_command(vec![0.2; ARM_DOF]),
            Ok(Upstream::Joints([0.2; ARM_DOF]))
        );
        assert!(parse_joint_command(vec![0.2; ARM_DOF - 1]).is_err());
        let mut positions = vec![0.2; ARM_DOF];
        positions[0] = f64::NAN;
        assert_eq!(parse_joint_command(positions), Err("non-finite values"));
    }

    #[test]
    fn gripper_command_parses_the_effort_cap() {
        let command = parse_gripper_command(0.5, 1.5).unwrap();
        assert_eq!(command.opening, 0.5);
        assert_eq!(command.max_effort, Some(1.5));
        // The wire's 0 means no preference, not a zero-force cap.
        assert_eq!(parse_gripper_command(0.5, 0.0).unwrap().max_effort, None);
    }

    #[test]
    fn gripper_command_rejects_malformed_fields() {
        assert_eq!(
            parse_gripper_command(f64::NAN, 0.0),
            Err("non-finite values")
        );
        assert_eq!(
            parse_gripper_command(0.5, f64::INFINITY),
            Err("non-finite values")
        );
        assert_eq!(
            parse_gripper_command(0.5, -1.0),
            Err("a negative max_effort")
        );
    }
}
