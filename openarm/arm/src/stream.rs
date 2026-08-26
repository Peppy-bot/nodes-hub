//! Stream plumbing for the real-arm follower: a listener that keeps the latest
//! governed setpoint from the paired backbone (the `backbone` slot of joint_link)
//! in a watch channel for the control loop, and a publisher that emits the
//! measured joint state at a fixed rate to the paired backbone on the pairing's
//! `joint_states` (the exclusive command loop the governor anchors on; any
//! monitor observes the pairing). All run for the life of the node.

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use control_core::pacer::Pacer;
use control_core::throttle::Throttle;
use peppygen::NodeRunner;
use peppygen::paired_topics::backbone;
use tokio::sync::watch;
use tracing::{error, warn};

use crate::health::HealthSample;
use crate::{ARM_DOF, JointVec};

/// At most one reject warning in this window, so a persistently malformed
/// setpoint stream is visible in the log without flooding it at the stream
/// rate. Every bad message still clears the target; only the warn throttles.
const REJECT_WARN_PERIOD: Duration = Duration::from_secs(1);

/// Warn once per failure burst instead of once per tick: one line when a
/// publish starts failing, silence while it keeps failing, re-armed by the
/// first success.
pub(crate) struct LatchedWarn {
    context: &'static str,
    failing: bool,
}

impl LatchedWarn {
    pub(crate) fn new(context: &'static str) -> Self {
        Self {
            context,
            failing: false,
        }
    }

    pub(crate) fn failure(&mut self, error: &str) {
        if !self.failing {
            self.failing = true;
            warn!("{} failing, suppressing repeats: {error}", self.context);
        }
    }

    pub(crate) fn success(&mut self) {
        self.failing = false;
    }
}

/// Capture timestamp from the daemon-resolved clock (sim time under a simulated
/// clock), so consumers age samples on the same timeline they read. Errors
/// until the clock delivers its first tick.
pub(crate) fn capture_timestamp() -> Result<SystemTime, String> {
    let ns = peppygen::clock::now_ns().map_err(|e| format!("clock not ready: {e}"))?;
    Ok(UNIX_EPOCH + Duration::from_nanos(ns))
}

/// The latest governed setpoint for this arm: the position/velocity the MIT loop
/// tracks. Produced by the backbone, already collision-governed and rate-limited.
#[derive(Clone, Copy)]
pub struct GovernedSetpoint {
    pub q_des: JointVec,
    pub dq_des: JointVec,
}

/// Measured joint state the control loop publishes each tick, as wire arrays.
/// Torques feed the pairing's `joint_states` efforts. `captured` is the
/// tick's identity: the publisher skips a value it already published, so a
/// wedged control loop goes silent on the wire instead of flowing as fresh.
#[derive(Clone, Copy)]
pub struct MeasuredState {
    pub positions: JointVec,
    pub velocities: JointVec,
    pub torques: JointVec,
    pub captured: Instant,
}

/// The control loop's connections to the stream tasks: the inbound governed
/// setpoint (latest) and the outbound measured-state and health channels
/// feeding the publishers.
pub struct StreamWiring {
    pub governed: watch::Receiver<Option<GovernedSetpoint>>,
    pub measured: watch::Sender<Option<MeasuredState>>,
    pub health: watch::Sender<Option<HealthSample>>,
}

/// Receive the paired backbone's `joint_setpoints` forever, folding each message
/// into `latest` via [`apply_setpoint`]. The slot delivers only the paired peer's
/// messages, so there is no arm_id filter; subscribing while unpaired is legal
/// (the slot stays silent until a backbone pairs). One held subscription, looped: no
/// re-subscribe gap, so a setpoint is never dropped between receives.
pub async fn run_governed_setpoint_listener(
    runner: Arc<NodeRunner>,
    latest: watch::Sender<Option<GovernedSetpoint>>,
) {
    let mut sub = match backbone::joint_setpoints::subscribe(&runner).await {
        Ok(s) => s,
        Err(e) => return error!("joint_setpoints subscribe: {e}"),
    };
    let mut reject_warn = Throttle::new(REJECT_WARN_PERIOD);
    loop {
        let msg = match sub.next().await {
            Ok(Some((_, msg))) => msg,
            Ok(None) => return, // subscription closed: node shutting down
            Err(e) => {
                error!("joint_setpoints receive: {e}");
                continue;
            }
        };
        apply_setpoint(&latest, &mut reject_warn, &msg);
    }
}

/// Fold one received setpoint into the watch: publish it when it parses,
/// otherwise clear the target (so the control loop holds the measured pose
/// rather than tracking a stale target) and warn with the reason.
fn apply_setpoint(
    latest: &watch::Sender<Option<GovernedSetpoint>>,
    reject_warn: &mut Throttle,
    msg: &backbone::joint_setpoints::Message,
) {
    match parse_setpoint(msg) {
        Ok(setpoint) => {
            latest.send_replace(Some(setpoint));
        }
        Err(reason) => {
            if reject_warn.admit() {
                warn!("joint_setpoints: clearing target: {reason}");
            }
            latest.send_replace(None);
        }
    }
}

/// Parse a wire setpoint into a governed target: exactly [`ARM_DOF`] positions
/// and velocities, all finite, and no efforts. Inbound efforts are rejected
/// outright because an ungoverned torque feedforward would bypass the
/// backbone's collision governor.
fn parse_setpoint(msg: &backbone::joint_setpoints::Message) -> Result<GovernedSetpoint, String> {
    let q_des: JointVec = msg
        .positions
        .as_slice()
        .try_into()
        .map_err(|_| format!("expected {ARM_DOF} positions, got {}", msg.positions.len()))?;
    let dq_des: JointVec = msg.velocities.as_slice().try_into().map_err(|_| {
        format!(
            "expected {ARM_DOF} velocities, got {}",
            msg.velocities.len()
        )
    })?;
    if !msg.efforts.is_empty() {
        return Err(format!(
            "rejected {} efforts (torque feedforward would bypass the governor)",
            msg.efforts.len()
        ));
    }
    if !q_des.iter().chain(dq_des.iter()).all(|v| v.is_finite()) {
        return Err("non-finite values".to_string());
    }
    Ok(GovernedSetpoint { q_des, dq_des })
}

/// Emit the measured joint state at a fixed rate, forever, to the paired
/// backbone on the pairing's `joint_states` (the command loop's state input;
/// monitors observe the pairing). The watch starts empty and is first filled
/// by the control loop's first tick, so nothing is published before a real
/// measurement exists. A tick the control loop has not refreshed is not
/// republished, and the loop exits if the control task drops the sender: a
/// wedged or dead control loop goes silent on the wire either way, rather
/// than flowing stamped-fresh.
pub async fn run_state_publisher(
    runner: Arc<NodeRunner>,
    period: Duration,
    mut measured: watch::Receiver<Option<MeasuredState>>,
) {
    let peer_pub = match backbone::joint_states::declare_publisher(&runner).await {
        Ok(p) => p,
        Err(e) => return error!("declare paired joint_states publisher: {e}"),
    };
    if measured.wait_for(Option::is_some).await.is_err() {
        return; // control loop gone: node is shutting down
    }
    let mut pacer =
        Pacer::new(period).expect("state publish period is non-zero (derives from state_rate_hz)");
    let mut peer_warn = LatchedWarn::new("paired state publish");
    let mut last_published: Option<Instant> = None;
    loop {
        if measured.has_changed().is_err() {
            return;
        }
        let m = (*measured.borrow()).expect("gated on first measurement");
        if last_published == Some(m.captured) {
            pacer.pace().await;
            continue;
        }
        last_published = Some(m.captured);
        let peer_result = async {
            let joints = backbone::joint_states::build_message(
                capture_timestamp()?,
                m.positions.to_vec(),
                m.velocities.to_vec(),
                m.torques.to_vec(),
            )
            .map_err(|e| e.to_string())?;
            peer_pub.publish(joints).await.map_err(|e| e.to_string())?;
            Ok::<(), String>(())
        }
        .await;
        match peer_result {
            Ok(()) => peer_warn.success(),
            Err(e) => peer_warn.failure(&e),
        }
        pacer.pace().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setpoint_msg(
        positions: Vec<f64>,
        velocities: Vec<f64>,
        efforts: Vec<f64>,
    ) -> backbone::joint_setpoints::Message {
        backbone::joint_setpoints::Message {
            timestamp: SystemTime::now(),
            positions,
            velocities,
            efforts,
        }
    }

    fn valid_msg() -> backbone::joint_setpoints::Message {
        setpoint_msg(vec![0.1; ARM_DOF], vec![0.0; ARM_DOF], vec![])
    }

    #[test]
    fn valid_setpoint_publishes_the_target() {
        let (tx, rx) = watch::channel::<Option<GovernedSetpoint>>(None);
        apply_setpoint(&tx, &mut Throttle::new(REJECT_WARN_PERIOD), &valid_msg());
        let target = rx.borrow().expect("valid 7/7/empty setpoint is published");
        assert_eq!(target.q_des, [0.1; ARM_DOF]);
        assert_eq!(target.dq_des, [0.0; ARM_DOF]);
    }

    #[test]
    fn non_finite_setpoint_clears_the_target() {
        let (tx, rx) = watch::channel::<Option<GovernedSetpoint>>(None);
        apply_setpoint(&tx, &mut Throttle::new(REJECT_WARN_PERIOD), &valid_msg());
        assert!(rx.borrow().is_some(), "valid setpoint should be published");
        // A non-finite value in either positions or velocities clears the target so
        // the control loop holds; a valid setpoint after a clear republishes.
        let mut bad_pos = vec![0.0; ARM_DOF];
        bad_pos[0] = f64::NAN;
        apply_setpoint(
            &tx,
            &mut Throttle::new(REJECT_WARN_PERIOD),
            &setpoint_msg(bad_pos, vec![0.0; ARM_DOF], vec![]),
        );
        assert!(
            rx.borrow().is_none(),
            "non-finite position must clear the target"
        );

        apply_setpoint(&tx, &mut Throttle::new(REJECT_WARN_PERIOD), &valid_msg());
        assert!(
            rx.borrow().is_some(),
            "valid setpoint must republish after a clear"
        );
        let mut bad_vel = vec![0.0; ARM_DOF];
        bad_vel[3] = f64::INFINITY;
        apply_setpoint(
            &tx,
            &mut Throttle::new(REJECT_WARN_PERIOD),
            &setpoint_msg(vec![0.1; ARM_DOF], bad_vel, vec![]),
        );
        assert!(
            rx.borrow().is_none(),
            "non-finite velocity must clear the target"
        );
    }

    #[test]
    fn dimension_mismatch_clears_the_target() {
        let (tx, rx) = watch::channel::<Option<GovernedSetpoint>>(None);
        apply_setpoint(&tx, &mut Throttle::new(REJECT_WARN_PERIOD), &valid_msg());
        assert!(rx.borrow().is_some(), "valid setpoint should be published");
        apply_setpoint(
            &tx,
            &mut Throttle::new(REJECT_WARN_PERIOD),
            &setpoint_msg(vec![0.1; ARM_DOF - 1], vec![0.0; ARM_DOF], vec![]),
        );
        assert!(
            rx.borrow().is_none(),
            "short positions must clear the target"
        );

        apply_setpoint(&tx, &mut Throttle::new(REJECT_WARN_PERIOD), &valid_msg());
        assert!(rx.borrow().is_some());
        apply_setpoint(
            &tx,
            &mut Throttle::new(REJECT_WARN_PERIOD),
            &setpoint_msg(vec![0.1; ARM_DOF], vec![0.0; ARM_DOF + 1], vec![]),
        );
        assert!(
            rx.borrow().is_none(),
            "long velocities must clear the target"
        );
    }

    #[test]
    fn non_empty_efforts_clear_the_target() {
        let (tx, rx) = watch::channel::<Option<GovernedSetpoint>>(None);
        apply_setpoint(&tx, &mut Throttle::new(REJECT_WARN_PERIOD), &valid_msg());
        assert!(rx.borrow().is_some(), "valid setpoint should be published");
        // Effort feedforward is ungoverned torque, so any efforts entry is
        // rejected even when positions and velocities are well-formed.
        apply_setpoint(
            &tx,
            &mut Throttle::new(REJECT_WARN_PERIOD),
            &setpoint_msg(vec![0.1; ARM_DOF], vec![0.0; ARM_DOF], vec![0.5]),
        );
        assert!(
            rx.borrow().is_none(),
            "non-empty efforts must clear the target"
        );
    }
}
