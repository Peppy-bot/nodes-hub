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

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use peppygen::NodeRunner;
use peppygen::consumed_topics::collision_ctrl::governor_control as collision_ctrl_governor_control;
use peppygen::paired_topics::{
    leader_left_arm, leader_left_arm_pose, leader_left_gripper, leader_right_arm,
    leader_right_arm_pose, leader_right_gripper, left_arm_link, left_gripper_link, right_arm_link,
    right_gripper_link,
};
use tokio::sync::watch;
use tracing::{error, warn};

use crate::types::{JointVec, Side, pose_from_wire};
use crate::upstream::Upstream;

/// At most one dropped-message warning per stream (and one build/timestamp error
/// per publisher) in this window, so a misrouted producer or a stalled clock
/// is visible in the log without flooding it at the stream rate.
pub(crate) const THROTTLED_WARN_PERIOD: Duration = Duration::from_secs(1);

/// Pause after a receive error before retrying, so a persistently broken
/// subscription cannot spin the listener at full CPU or flood the log at the stream
/// rate. Transient errors still recover; a genuinely dead stream just idles.
const RECEIVE_ERROR_BACKOFF: Duration = Duration::from_millis(100);

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
/// state relay only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmState {
    pub positions: JointVec,
    pub velocities: JointVec,
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

/// Run `emit` at most once per [`THROTTLED_WARN_PERIOD`] per `last` state.
pub(crate) fn warn_throttled(last: &mut Option<Instant>, emit: impl FnOnce()) {
    let now = Instant::now();
    if last.is_none_or(|t| now.duration_since(t) >= THROTTLED_WARN_PERIOD) {
        emit();
        *last = Some(now);
    }
}

/// Parses one paired arm's inbound joint_states payload. This backbone drives
/// fixed 7-joint arms whose followers always measure velocity, so a message
/// must carry exactly [`crate::types::ARM_DOF`] positions with matching velocities,
/// all finite; the generic contract's empty-velocities form is deliberately
/// rejected here rather than half-accepted.
fn parse_joint_state(positions: Vec<f64>, velocities: Vec<f64>) -> Result<ArmState, &'static str> {
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
    })
}

/// Parses one commanded arm setpoint: exactly [`crate::types::ARM_DOF`] finite
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

/// Parses one commanded end-effector pose through the shared wire decoder
/// ([`pose_from_wire`]): finite components, quaternion normalized or refused.
fn parse_pose_command(position: [f64; 3], orientation: [f64; 4]) -> Result<Upstream, &'static str> {
    pose_from_wire(position, orientation).map(Upstream::Pose)
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

/// Apply one received message to its side's watch, and report whether the
/// listener should keep going.
///
/// This is the whole receive policy, in one place: a well-formed message
/// replaces that side's latest, a malformed one is dropped with a throttled
/// reason rather than driving an arm, a receive error backs off instead of
/// spinning, and a closed subscription ends the listener (the node is shutting
/// down). Each generated slot has its own message type and the subscriptions
/// share no trait, so the select that produced `received` stays with its
/// listener; everything after it is here.
async fn accept<Raw, Parsed>(
    what: &'static str,
    side: Side,
    received: peppygen::Result<Option<Raw>>,
    parse: impl Fn(Raw) -> Result<Parsed, &'static str>,
    latest: &[watch::Sender<Option<Parsed>>; 2],
    last_reject_warn: &mut Option<Instant>,
) -> bool {
    match received {
        Ok(Some(raw)) => match parse(raw) {
            Ok(parsed) => {
                latest[side.index()].send_replace(Some(parsed));
            }
            Err(reason) => warn_throttled(last_reject_warn, || {
                warn!("{what}: dropping {} message with {reason}", side.label());
            }),
        },
        Ok(None) => return false,
        Err(e) => {
            error!("{what} receive ({}): {e}", side.label());
            tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
        }
    }
    true
}

/// Receive both arms' upstream pose streams forever, keeping the latest
/// well-formed pose per side. The Cartesian mirror of
/// [`run_joint_command_listener`]; `upstream_mode` spawns exactly one of the
/// two.
pub async fn run_pose_command_listener(
    runner: Arc<NodeRunner>,
    latest: [watch::Sender<Option<Upstream>>; 2],
) {
    const WHAT: &str = "upstream pose_setpoints";
    let Some((mut left, mut right)) = subscribe_pair(
        WHAT,
        leader_left_arm_pose::pose_setpoints::subscribe(&runner),
        leader_right_arm_pose::pose_setpoints::subscribe(&runner),
    )
    .await
    else {
        return;
    };
    let mut warned = None;
    loop {
        let (side, received) = tokio::select! {
            r = left.next() => (Side::Left, r.map(|m| m.map(|(_, msg)| (msg.position, msg.orientation)))),
            r = right.next() => (Side::Right, r.map(|m| m.map(|(_, msg)| (msg.position, msg.orientation)))),
        };
        let applied = accept(
            WHAT,
            side,
            received,
            |(position, orientation)| parse_pose_command(position, orientation),
            &latest,
            &mut warned,
        )
        .await;
        if !applied {
            return;
        }
    }
}

/// Receive both upstream arm setpoint streams forever (the leading node's
/// joint_link command direction), keeping the latest well-formed message per
/// side. The slot IS the side (a pairing delivers only its one peer), so there
/// is no id demux. Whichever slot delivers next wins the select; the other
/// stays queued in its own subscription, so neither side can starve the other.
pub async fn run_joint_command_listener(
    runner: Arc<NodeRunner>,
    latest: [watch::Sender<Option<Upstream>>; 2],
) {
    const WHAT: &str = "upstream joint_setpoints";
    let Some((mut left, mut right)) = subscribe_pair(
        WHAT,
        leader_left_arm::joint_setpoints::subscribe(&runner),
        leader_right_arm::joint_setpoints::subscribe(&runner),
    )
    .await
    else {
        return;
    };
    let mut warned = None;
    loop {
        let (side, received) = tokio::select! {
            r = left.next() => (Side::Left, r.map(|m| m.map(|(_, msg)| msg.positions))),
            r = right.next() => (Side::Right, r.map(|m| m.map(|(_, msg)| msg.positions))),
        };
        let applied = accept(
            WHAT,
            side,
            received,
            parse_joint_command,
            &latest,
            &mut warned,
        )
        .await;
        if !applied {
            return;
        }
    }
}

/// Receive both upstream gripper setpoint streams forever, keeping the latest
/// well-formed command per side. The 1-DOF mirror of
/// [`run_joint_command_listener`].
pub async fn run_gripper_command_listener(
    runner: Arc<NodeRunner>,
    latest: [watch::Sender<Option<GripperCommand>>; 2],
) {
    const WHAT: &str = "upstream gripper_setpoints";
    let Some((mut left, mut right)) = subscribe_pair(
        WHAT,
        leader_left_gripper::gripper_setpoints::subscribe(&runner),
        leader_right_gripper::gripper_setpoints::subscribe(&runner),
    )
    .await
    else {
        return;
    };
    let mut warned = None;
    loop {
        let (side, received) = tokio::select! {
            r = left.next() => (Side::Left, r.map(|m| m.map(|(_, msg)| (msg.opening, msg.max_effort)))),
            r = right.next() => (Side::Right, r.map(|m| m.map(|(_, msg)| (msg.opening, msg.max_effort)))),
        };
        let applied = accept(
            WHAT,
            side,
            received,
            |(opening, max_effort)| parse_gripper_command(opening, max_effort),
            &latest,
            &mut warned,
        )
        .await;
        if !applied {
            return;
        }
    }
}

/// Receive both paired arms' measured state forever (the joint_link
/// back-channel), keeping the latest per side. The governor anchors only on its
/// exclusive command-loop peers: a stray broadcast producer cannot pose as an
/// arm. Non-finite states are dropped so the coordinator never anchors a
/// trajectory or a governor query on a bad measurement.
pub async fn run_joint_state_listener(
    runner: Arc<NodeRunner>,
    latest: [watch::Sender<Option<ArmState>>; 2],
) {
    const WHAT: &str = "joint_states";
    let Some((mut left, mut right)) = subscribe_pair(
        WHAT,
        left_arm_link::joint_states::subscribe(&runner),
        right_arm_link::joint_states::subscribe(&runner),
    )
    .await
    else {
        return;
    };
    let mut warned = None;
    loop {
        let (side, received) = tokio::select! {
            r = left.next() => (Side::Left, r.map(|m| m.map(|(_, msg)| (msg.positions, msg.velocities)))),
            r = right.next() => (Side::Right, r.map(|m| m.map(|(_, msg)| (msg.positions, msg.velocities)))),
        };
        let applied = accept(
            WHAT,
            side,
            received,
            |(positions, velocities)| parse_joint_state(positions, velocities),
            &latest,
            &mut warned,
        )
        .await;
        if !applied {
            return;
        }
    }
}

/// Receive both paired grippers' measured aperture forever (the gripper_link
/// back-channel), keeping the latest opening fraction per side. The collision
/// model reads finger positions only from these exclusive command-loop peers,
/// so a stray broadcast producer cannot spoof the modeled fingers.
pub async fn run_gripper_state_listener(
    runner: Arc<NodeRunner>,
    latest: [watch::Sender<Option<GripperState>>; 2],
) {
    const WHAT: &str = "gripper_states";
    let Some((mut left, mut right)) = subscribe_pair(
        WHAT,
        left_gripper_link::gripper_states::subscribe(&runner),
        right_gripper_link::gripper_states::subscribe(&runner),
    )
    .await
    else {
        return;
    };
    let mut warned = None;
    loop {
        let (side, received) = tokio::select! {
            r = left.next() => (Side::Left, r.map(|m| m.map(|(_, msg)| (msg.opening, msg.effort, msg.max_effort)))),
            r = right.next() => (Side::Right, r.map(|m| m.map(|(_, msg)| (msg.opening, msg.effort, msg.max_effort)))),
        };
        let applied = accept(
            WHAT,
            side,
            received,
            |(opening, effort, max_effort)| parse_gripper_state(opening, effort, max_effort),
            &latest,
            &mut warned,
        )
        .await;
        if !applied {
            return;
        }
    }
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
    pub max_ee_velocity_m_s: f64,
    pub max_gripper_rate_frac_s: f64,
}

/// Receive the `governor_control` stream forever, mirroring the latest governor
/// config into `config`. The producer re-publishes periodically, so the lossy QoS
/// cannot strand the backbone in a stale state.
pub async fn run_governor_config_listener(
    runner: Arc<NodeRunner>,
    config: watch::Sender<GovernorConfig>,
) {
    let mut sub = match collision_ctrl_governor_control::subscribe(&runner).await {
        Ok(s) => s,
        Err(e) => return error!("governor_control subscribe: {e}"),
    };
    loop {
        match sub.next().await {
            Ok(Some((_, msg))) => {
                config.send_replace(GovernorConfig {
                    enabled: msg.collision_governor_enabled,
                    d_stop: msg.d_stop,
                    d_safe: msg.d_safe,
                    max_ee_velocity_m_s: msg.max_ee_velocity_m_s,
                    max_gripper_rate_frac_s: msg.max_gripper_rate_frac_s,
                });
            }
            Ok(None) => return,
            Err(e) => {
                error!("governor_control receive: {e}");
                tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::ARM_DOF;
    use srs_model::nalgebra::{Isometry3, Vector3};

    #[test]
    fn joint_state_requires_arm_dof_positions_with_matching_velocities() {
        let state = parse_joint_state(vec![0.1; ARM_DOF], vec![0.0; ARM_DOF]).unwrap();
        assert_eq!(state.positions, [0.1; ARM_DOF]);
        assert!(parse_joint_state(vec![0.1; ARM_DOF - 1], vec![0.0; ARM_DOF - 1]).is_err());
        assert!(parse_joint_state(vec![0.1; ARM_DOF + 1], vec![0.0; ARM_DOF + 1]).is_err());
        assert!(parse_joint_state(vec![], vec![]).is_err());
        assert!(parse_joint_state(vec![0.1; ARM_DOF], vec![0.0; ARM_DOF - 1]).is_err());
    }

    #[test]
    fn joint_state_rejects_the_contracts_empty_velocities_form() {
        // Deliberate strictness: this backbone's followers always measure
        // velocity, so the generic contract's unmeasured (empty) form is
        // treated as malformed instead of half-accepted.
        assert_eq!(
            parse_joint_state(vec![0.1; ARM_DOF], vec![]),
            Err("a velocity count that does not match its joints")
        );
    }

    #[test]
    fn joint_state_rejects_non_finite_values() {
        let mut positions = vec![0.1; ARM_DOF];
        positions[3] = f64::NAN;
        assert_eq!(
            parse_joint_state(positions, vec![0.0; ARM_DOF]),
            Err("non-finite values")
        );
        let mut velocities = vec![0.0; ARM_DOF];
        velocities[6] = f64::INFINITY;
        assert_eq!(
            parse_joint_state(vec![0.1; ARM_DOF], velocities),
            Err("non-finite values")
        );
    }

    /// Extract the pose from an `Upstream` the parser was expected to accept.
    fn parsed_pose(
        position: [f64; 3],
        orientation: [f64; 4],
    ) -> Result<Isometry3<f64>, &'static str> {
        match parse_pose_command(position, orientation)? {
            Upstream::Pose(pose) => Ok(pose),
            Upstream::Joints(_) => panic!("the pose parser produced a joint command"),
        }
    }

    #[test]
    fn pose_command_reads_the_wire_as_scalar_last() {
        // A quarter turn about +Z, written the way the wire writes it. Reading
        // it scalar-first instead would silently produce a different attitude,
        // so pin the recovered axis and angle rather than the four components.
        let half = std::f64::consts::FRAC_PI_4;
        let pose = parsed_pose([0.4, -0.1, 0.3], [0.0, 0.0, half.sin(), half.cos()]).unwrap();
        assert_eq!(pose.translation.vector, Vector3::new(0.4, -0.1, 0.3));
        let (axis, angle) = pose.rotation.axis_angle().unwrap();
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((axis.into_inner() - Vector3::z()).norm() < 1e-12);
    }

    #[test]
    fn pose_command_normalizes_a_non_unit_quaternion() {
        // Four independent floats reach us; only unit ones name a rotation.
        let pose = parsed_pose([0.0; 3], [0.0, 0.0, 0.0, 5.0]).unwrap();
        assert!((pose.rotation.quaternion().norm() - 1.0).abs() < 1e-12);
        assert!(pose.rotation.angle() < 1e-12);
    }

    #[test]
    fn pose_command_rejects_the_shapes_the_governor_could_not_survive() {
        // A non-finite component reaching the governor faults the whole tick,
        // and a zero quaternion names no rotation to substitute for.
        assert_eq!(
            parse_pose_command([f64::NAN, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            Err("non-finite values")
        );
        assert_eq!(
            parse_pose_command([0.0; 3], [0.0, f64::INFINITY, 0.0, 1.0]),
            Err("non-finite values")
        );
        assert_eq!(
            parse_pose_command([0.0; 3], [0.0, 0.0, 0.0, 0.0]),
            Err("an unnormalizable orientation")
        );
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
