//! The real arm's control: a single task that owns the motors and runs an MIT
//! control loop following the backbone's governed setpoint. The bimanual coordination
//! backbone (openarm_backbone) owns all trajectory generation, stream following, and
//! collision governing, and streams the resolved (q_des, dq_des) per arm; this
//! loop adds only the realtime feedforward (gravity/Coriolis/friction the backbone
//! cannot compute remotely) and a final clamp to the joint limits, then commands
//! the motors. There is no mode state machine and no streaming logic here.
//!
//! On shutdown the loop disables the motors and lets the arm go limp. It does not
//! park to a pose: a collision-aware return-to-home is the backbone's job (it sees both
//! arms), and a local straight-line joint path would be collision-blind and could
//! command the two arms into each other.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use openarm_description::ARM_DOF;
use peppygen::NodeRunner;
use peppylib::runtime::CancellationToken;
use srs_model::{JointVec, Limit};
use tokio::sync::oneshot;
use tracing::{error, info};

use control_core::motor_health::{NOT_DRIVING_ESCALATE_AFTER, Ratings};
use control_core::pacer::Pacer;
use control_core::sync::lock_ignoring_poison;
use control_core::time;

use crate::drive;
use crate::friction;
use crate::health::Monitor;
use crate::stream::{GovernedSetpoint, MeasuredState, StreamWiring};
use openarm_can::{ArmCan, CanErrorThrottle};

/// Set when the loop stops on a hard fault, so main can exit non-zero after
/// the shutdown hooks have run and the daemon records the instance as failed
/// rather than finished.
pub static HARD_FAULT: AtomicBool = AtomicBool::new(false);

const CONTEXT: &str = "arm control";

/// All-zero joint vector: the desired velocity sent while holding a pose.
const ZERO: JointVec = [0.0; ARM_DOF];

/// Settle time after disabling the motors on shutdown, before draining ACK
/// frames. Mirrors the ROS2 v10_simple_hardware on_deactivate sleep.
const POST_DISABLE_SLEEP: Duration = Duration::from_millis(100);

pub struct ControlConfig {
    pub kp: JointVec,
    pub kd: JointVec,
    pub cycle_period: Duration,
    pub recv_timeout_us: u32,
    /// Joint position limits, parsed from the URDF, the final guard clamp applied
    /// to every governed setpoint before it reaches the motors.
    pub limits: [Limit; ARM_DOF],
    /// Per-joint torque ratings the health filters judge against, read from
    /// the motors at bring-up so a joint configured to trip below its
    /// datasheet peak still warns before it cuts out.
    pub ratings: [Ratings; ARM_DOF],
}

/// Spawn the arm's control and supervise it. `run_control` is the sole motor
/// writer, so a fire-and-forget spawn would let a panic pass unobserved, leaving
/// the node up with the motors uncontrolled and no control loop. The supervisor
/// watches the control task and, if it ever stops, cancels the node so it restarts
/// under supervision instead of serving blind. `shutdown_tx` is signalled once a
/// clean stop has disabled the motors, so the shutdown hook in `node` can then
/// release the lock.
pub fn spawn(
    runner: &NodeRunner,
    arm: Arc<Mutex<ArmCan>>,
    cfg: ControlConfig,
    model: srs_model::Arm,
    wiring: StreamWiring,
    shutdown_tx: oneshot::Sender<()>,
    started_tx: oneshot::Sender<()>,
) {
    let token = runner.cancellation_token().clone();
    let control = tokio::spawn(run_control(
        arm.clone(),
        cfg,
        model,
        wiring,
        token.clone(),
        shutdown_tx,
        started_tx,
    ));
    tokio::spawn(supervise(control, arm, token));
}

/// Watch the sole motor-writer task. A clean stop returns `Ok` after the loop has
/// already disabled the motors; a panic returns `Err` with the motors still live,
/// so disable them here. Either way cancel the node: on a clean stop the token is
/// already cancelled (idempotent), and on a panic this converts a silently dead
/// control loop into a node restart rather than an arm left with no controller.
async fn supervise(
    control: tokio::task::JoinHandle<()>,
    arm: Arc<Mutex<ArmCan>>,
    token: CancellationToken,
) {
    if let Err(join_error) = control.await {
        error!(%join_error, "control loop terminated unexpectedly; disabling motors");
        disable_motors(&arm);
        HARD_FAULT.store(true, Ordering::SeqCst);
    }
    token.cancel();
}

/// The single motor-owning control loop. Each tick reads the measured state,
/// computes the feedforward, reports the measured state upward, and commands the
/// motors to the latest governed setpoint (clamped), holding the measured pose
/// until the first governed setpoint arrives or whenever the stream is empty. On
/// cancellation it disables the motors and signals `shutdown_tx`.
async fn run_control(
    arm: Arc<Mutex<ArmCan>>,
    cfg: ControlConfig,
    mut model: srs_model::Arm,
    wiring: StreamWiring,
    token: CancellationToken,
    shutdown_tx: oneshot::Sender<()>,
    started_tx: oneshot::Sender<()>,
) {
    let mut pacer =
        Pacer::new(cfg.cycle_period).expect("main asserts the period is non-zero before spawn");
    let mut can_errors = CanErrorThrottle::new();
    let mut monitor = Monitor::new(cfg.ratings, cfg.cycle_period);
    let escalate_after_ticks = time::ticks_within(NOT_DRIVING_ESCALATE_AFTER, cfg.cycle_period);
    let mut not_driving = [0u32; ARM_DOF];
    info!("control loop started (MIT follower of governed setpoints, in-process feedforward)");
    // Readiness gates on this: the loop is entered, not merely spawned, and
    // never acknowledged into a shutdown already in progress.
    if token.is_cancelled() {
        return;
    }
    let _ = started_tx.send(());
    loop {
        let (state, read) = read_state(&arm, cfg.recv_timeout_us);
        let (q, qdot) = (state.positions, state.velocities);
        let ff_tau = feedforward(&mut model, &q, &qdot);
        wiring.measured.send_replace(Some(MeasuredState {
            positions: q,
            velocities: qdot,
            torques: state.torques,
            captured: Instant::now(),
        }));
        let silent = monitor.observe(&state, &wiring.health);

        match drive::assess(
            &state.statuses,
            &silent,
            &mut not_driving,
            escalate_after_ticks,
        ) {
            drive::Verdict::Continue => {}
            drive::Verdict::Reenable(_) => reenable_motors(&arm, &token),
            drive::Verdict::HardFault(reason) => {
                // The faulted motor is already limp; six joints still driving
                // gravity feedforward around it is unpredictable, so stop the
                // node without commanding this tick. The post-loop path below
                // disables the motors, and is_ready drops via the HARD_FAULT
                // latch.
                error!("{reason}; stopping node");
                HARD_FAULT.store(true, Ordering::SeqCst);
                token.cancel();
                break;
            }
        }

        // Follow the latest governed setpoint; hold the measured pose (zero
        // desired velocity) until the backbone's stream is live, so the arm never
        // lunges before the backbone is up.
        let (q_des, dq_des) = match *wiring.governed.borrow() {
            Some(GovernedSetpoint { q_des, dq_des }) => {
                clamp_setpoint_to_limits(&q_des, &dq_des, &cfg.limits)
            }
            None => (q, ZERO),
        };

        // A driver fault costs this tick's frames, not the loop: the motors
        // hold their last commanded pose, the next tick re-sends an absolute
        // command, and the disable that stopping would imply travels over the
        // same socket that just failed. One verdict per tick, first error wins,
        // so a persistent fault logs as one throttled burst.
        let sent = command(&arm, &cfg, &ff_tau, &q_des, &dq_des, &token);
        match read.and(sent) {
            Ok(()) => can_errors.success(CONTEXT),
            Err(e) => can_errors.failure(CONTEXT, &e),
        }
        // Biased so a cancelled token always wins over an already-due (overrun)
        // tick: on shutdown break out and disable the motors below.
        tokio::select! {
            biased;
            _ = token.cancelled() => break,
            _ = pacer.pace() => {}
        }
    }

    // Cancelled: disable the motors and let the arm go limp. A graceful, collision-
    // aware park is the backbone's responsibility (it sees both arms); the arm on its own
    // must not drive to a fixed pose, because a collision-blind straight joint path
    // could command the two arms into each other. The shutdown hook in `node`
    // awaits `shutdown_tx` so the lock is released only after the motors are off.
    info!("control loop stopping: disabling motors");
    disable_motors(&arm);
    tokio::time::sleep(POST_DISABLE_SLEEP).await;
    {
        let mut a = lock_ignoring_poison(&arm);
        if let Err(e) = a.recv_all(cfg.recv_timeout_us) {
            error!("drain disable replies: {e}");
        }
    }
    // A dropped receiver (the shutdown hook in `node` has exited) is fine;
    // nothing to do.
    let _ = shutdown_tx.send(());
    info!("control loop stopped (motors disabled)");
}

/// Re-enable all motors after an unexpected Disabled report. Addressed to
/// every motor: enable is idempotent, and one send covers every joint that
/// is concurrently disabled.
fn reenable_motors(arm: &Mutex<ArmCan>, token: &CancellationToken) {
    let mut a = lock_ignoring_poison(arm);
    // Checked under the lock the disable hook shares: cancellation precedes
    // the hooks, so a blip recovery can never re-energise motors the hook
    // just disabled.
    if token.is_cancelled() {
        return;
    }
    if let Err(e) = a.enable_all() {
        error!("re-enable motors: {e}");
    }
}

/// One tick of rigid-body feedforward: gravity and Coriolis from the posed chain
/// (carrying the distal gripper payload) plus locally computed friction, so the
/// PD term only corrects residual error.
fn feedforward(model: &mut srs_model::Arm, q: &JointVec, qdot: &JointVec) -> JointVec {
    let posed = model.at(q);
    let gravity = posed.gravity_torques();
    let coriolis = posed.coriolis_torques(qdot);
    let friction = friction::torques(&friction::NOMINAL, qdot);
    std::array::from_fn(|i| gravity[i] + coriolis[i] + friction[i])
}

/// Command the motors once: this tick's feedforward plus PD to the governed
/// position/velocity.
fn command(
    arm: &Mutex<ArmCan>,
    cfg: &ControlConfig,
    ff_tau: &JointVec,
    q_des: &JointVec,
    dq_des: &JointVec,
    token: &CancellationToken,
) -> openarm_can::Result<()> {
    let mut a = lock_ignoring_poison(arm);
    // Checked under the lock the disable hook shares, so a tick straddling
    // cancellation cannot command motors the hook is about to disable; the
    // loop observes the token at its next select and stops.
    if token.is_cancelled() {
        return Ok(());
    }
    a.mit_control(&cfg.kp, &cfg.kd, q_des, dq_des, ff_tau)
}

/// Clamp a governed setpoint into the joint position limits, the final guard
/// before the motors. A joint whose target is pinned at a limit also has its
/// desired velocity zeroed when that velocity points further past the stop, so the
/// MIT controller is never commanded to drive outward through a hard limit (the
/// `kd * (dq_des - qdot)` term cannot add outward torque at the wall). Inward
/// (recovering) velocity is preserved.
fn clamp_setpoint_to_limits(
    q: &JointVec,
    dq: &JointVec,
    limits: &[Limit; ARM_DOF],
) -> (JointVec, JointVec) {
    let q_clamped: JointVec = std::array::from_fn(|i| q[i].clamp(limits[i].lo, limits[i].hi));
    let dq_clamped: JointVec = std::array::from_fn(|i| {
        let driving_below = q[i] <= limits[i].lo && dq[i] < 0.0;
        let driving_above = q[i] >= limits[i].hi && dq[i] > 0.0;
        if driving_below || driving_above {
            0.0
        } else {
            dq[i]
        }
    });
    (q_clamped, dq_clamped)
}

/// Disable all motors so the arm goes limp. Recovers a poisoned lock (unwrap into
/// the inner guard) so the disable runs even if the control loop panicked holding
/// it, since going limp is the safe failure state.
fn disable_motors(arm: &Mutex<ArmCan>) {
    let mut a = lock_ignoring_poison(arm);
    if let Err(e) = a.disable_all() {
        error!("disable motors: {e}");
    }
}

/// Read the measured joint state once, returning the driver cache and the
/// pass's outcome. A failed pass leaves the cache at its previous reading.
///
/// Nothing is solicited here: a MIT command frame makes the motor reply with
/// its state, and the loop commands every tick unconditionally, so the
/// previous tick's command is this tick's request. Bring-up seeds the cache
/// with one explicit pass, and a motor that stops replying is caught by the
/// staleness window.
fn read_state(
    arm: &Mutex<ArmCan>,
    recv_timeout_us: u32,
) -> (openarm_can::ArmState, openarm_can::Result<()>) {
    let mut a = lock_ignoring_poison(arm);
    let received = a.recv_all(recv_timeout_us);
    (a.get_state(), received)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_limits() -> [Limit; ARM_DOF] {
        std::array::from_fn(|_| Limit { lo: -1.0, hi: 1.0 })
    }

    #[test]
    fn clamps_position_and_zeros_outward_velocity_at_a_limit() {
        // Joint 0: target above hi with outward (+) velocity. Joint 1: target below
        // lo with outward (-) velocity. Both clamp to the limit and zero the velocity.
        let q = [2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let dq = [0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (qc, dqc) = clamp_setpoint_to_limits(&q, &dq, &unit_limits());
        assert_eq!(qc[0], 1.0);
        assert_eq!(qc[1], -1.0);
        assert_eq!(dqc[0], 0.0, "outward velocity at the upper stop is zeroed");
        assert_eq!(dqc[1], 0.0, "outward velocity at the lower stop is zeroed");
    }

    #[test]
    fn preserves_inward_recovery_velocity_past_a_limit() {
        // Past the limits but velocity points back toward range: keep it so the arm
        // can recover rather than being pinned outside.
        let q = [2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let dq = [-0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (_, dqc) = clamp_setpoint_to_limits(&q, &dq, &unit_limits());
        assert_eq!(dqc[0], -0.5, "inward velocity above hi is preserved");
        assert_eq!(dqc[1], 0.5, "inward velocity below lo is preserved");
    }

    #[test]
    fn leaves_in_range_setpoints_untouched() {
        let q = [0.5, -0.5, 0.0, 0.3, -0.3, 0.1, -0.1];
        let dq = [0.4, -0.4, 0.2, -0.2, 0.1, -0.1, 0.0];
        let (qc, dqc) = clamp_setpoint_to_limits(&q, &dq, &unit_limits());
        assert_eq!(qc, q, "in-range positions are unchanged");
        assert_eq!(dqc, dq, "in-range velocities are unchanged");
    }

    #[test]
    fn zeros_outward_velocity_exactly_at_a_limit() {
        // q exactly on a stop is already at the wall, so outward velocity is
        // zeroed there too: the MIT `kd` term must add no outward torque.
        let q = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let dq = [0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (_, dqc) = clamp_setpoint_to_limits(&q, &dq, &unit_limits());
        assert_eq!(
            dqc[0], 0.0,
            "outward velocity exactly at the upper stop is zeroed"
        );
        assert_eq!(
            dqc[1], 0.0,
            "outward velocity exactly at the lower stop is zeroed"
        );
    }

    #[test]
    fn the_not_driving_escalation_spans_a_hundred_ticks_at_100_hz() {
        assert_eq!(
            time::ticks_within(NOT_DRIVING_ESCALATE_AFTER, Duration::from_millis(10)),
            100
        );
    }
}
