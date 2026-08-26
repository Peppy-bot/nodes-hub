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

use peppygen::NodeRunner;
use peppylib::runtime::CancellationToken;
use srs_model::Limit;
use tokio::sync::oneshot;
use tracing::{error, info, warn};

use control_core::motor_health::{NOT_DRIVING_ESCALATE_AFTER, Ratings};
use control_core::pacer::Pacer;

use crate::friction;
use crate::health::Monitor;
use crate::stream::{GovernedSetpoint, MeasuredState, StreamWiring};
use crate::{ARM_DOF, JointVec};
use openarm_can::{ArmCan, CanErrorThrottle, MotorStatus};

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

#[derive(Clone)]
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
/// clean stop has disabled the motors, so main.rs can then release the lock.
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
    let escalate_after_ticks = ticks_within(NOT_DRIVING_ESCALATE_AFTER, cfg.cycle_period);
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
        let stale = monitor.observe(&state, &wiring);

        match assess(
            &state.statuses,
            &stale,
            &mut not_driving,
            escalate_after_ticks,
        ) {
            Verdict::Continue => {}
            Verdict::Reenable(_) => reenable_motors(&arm, &token),
            Verdict::HardFault(reason) => {
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
    // could command the two arms into each other. main.rs awaits `shutdown_tx` so
    // the lock is released only after the motors are off.
    info!("control loop stopping: disabling motors");
    disable_motors(&arm);
    tokio::time::sleep(POST_DISABLE_SLEEP).await;
    {
        let mut a = arm.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = a.recv_all(cfg.recv_timeout_us) {
            error!("drain disable replies: {e}");
        }
    }
    // A dropped receiver (main.rs already exited) is fine; nothing to do.
    let _ = shutdown_tx.send(());
    info!("control loop stopped (motors disabled)");
}

/// What one tick's motor statuses require of the loop.
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Continue,
    /// The named joint (first of any) reports Disabled without a fault: send
    /// a re-enable this tick.
    Reenable(usize),
    HardFault(String),
}

/// Classify the tick's statuses and staleness against the per-joint
/// not-driving streaks, which this advances (warning at each joint's onset).
///
/// A fault stops the node immediately, from a fresh frame or the cache: a
/// faulted motor latches until power-cycled, so a cached fault is still the
/// joint's condition. Any other joint that is not driving gets a re-enable
/// each tick it freshly reports Disabled, and escalates to a hard fault
/// after `escalate_after_ticks` consecutive not-driving ticks.
///
/// "Not driving" covers three states, because they are the same thing to a
/// held load: the motor reports Disabled, reports a status nibble this
/// driver does not recognise (such a frame cannot confirm the motor is
/// driving), or has nothing current to report (stale, or never heard from).
///
/// The streaks are per joint, so escalation names the joint that ran out of
/// patience, and a flapping joint's good ticks reset only its own streak.
fn assess(
    statuses: &[MotorStatus; ARM_DOF],
    stale: &[bool; ARM_DOF],
    not_driving: &mut [u32; ARM_DOF],
    escalate_after_ticks: u32,
) -> Verdict {
    let fault = statuses
        .iter()
        .enumerate()
        .find_map(|(i, s)| s.fault().map(|kind| (i, kind)));
    if let Some((joint, kind)) = fault {
        return Verdict::HardFault(format!("motor j{} fault: {}", joint + 1, kind.name()));
    }

    for (joint, streak) in not_driving.iter_mut().enumerate() {
        let (driving, how) = drive_condition(statuses[joint], stale[joint]);
        *streak = if driving { 0 } else { *streak + 1 };
        if *streak == 1 {
            warn!("motor j{} {how}: not driving", joint + 1);
        }
    }

    let escalated = (0..ARM_DOF).find(|&j| not_driving[j] >= escalate_after_ticks);
    if let Some(joint) = escalated {
        let (_, how) = drive_condition(statuses[joint], stale[joint]);
        return Verdict::HardFault(format!(
            "motor j{} not driving for {} consecutive ticks, currently {how}",
            joint + 1,
            not_driving[joint]
        ));
    }

    // Re-enabling is the answer to a fresh Disabled report; a silent motor is
    // not listening and an unrecognised status never said the motor was
    // disabled, so those only run out the escalation window.
    match (0..ARM_DOF).find(|&j| !stale[j] && statuses[j] == MotorStatus::Disabled) {
        Some(joint) => Verdict::Reenable(joint),
        None => Verdict::Continue,
    }
}

/// Whether one joint is confirmed driving this tick, with the word for its
/// state when it is not.
fn drive_condition(status: MotorStatus, stale: bool) -> (bool, &'static str) {
    if stale || status == MotorStatus::Unreported {
        return (false, "silent");
    }
    match status {
        MotorStatus::Disabled => (false, "disabled"),
        MotorStatus::Unknown(_) => (false, "in an unrecognised status"),
        _ => (true, "driving"),
    }
}

/// Whole ticks covering `window`, at least one.
pub(crate) fn ticks_within(window: Duration, cycle_period: Duration) -> u32 {
    window.as_nanos().div_ceil(cycle_period.as_nanos()).max(1) as u32
}

/// Re-enable all motors after an unexpected Disabled report. Addressed to
/// every motor: enable is idempotent, and one send covers every joint that
/// is concurrently disabled.
fn reenable_motors(arm: &Mutex<ArmCan>, token: &CancellationToken) {
    let mut a = arm.lock().unwrap_or_else(|e| e.into_inner());
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
    let mut a = arm.lock().unwrap_or_else(|e| e.into_inner());
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
    let mut a = arm.lock().unwrap_or_else(|e| e.into_inner());
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
    let mut a = arm.lock().unwrap_or_else(|e| e.into_inner());
    let received = a.recv_all(recv_timeout_us);
    (a.get_state(), received)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openarm_can::FaultKind;

    fn unit_limits() -> [Limit; ARM_DOF] {
        std::array::from_fn(|_| Limit { lo: -1.0, hi: 1.0 })
    }

    const FRESH: [bool; ARM_DOF] = [false; ARM_DOF];

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

    /// assess with fresh streaks and every joint's state current.
    fn verdict(statuses: &[MotorStatus; ARM_DOF], escalate_after: u32) -> Verdict {
        let mut streaks = [0u32; ARM_DOF];
        assess(statuses, &FRESH, &mut streaks, escalate_after)
    }

    #[test]
    fn assess_verdicts_by_status_table() {
        let all = |s: MotorStatus| -> [MotorStatus; ARM_DOF] { [s; ARM_DOF] };
        let one = |i: usize, s: MotorStatus| -> [MotorStatus; ARM_DOF] {
            let mut v = all(MotorStatus::Enabled);
            v[i] = s;
            v
        };
        assert_eq!(verdict(&all(MotorStatus::Enabled), 100), Verdict::Continue);
        // One frame with an undefined nibble is noise; only persistence
        // escalates, and it is no grounds for a re-enable.
        assert_eq!(
            verdict(&one(2, MotorStatus::Unknown(0x3)), 100),
            Verdict::Continue
        );
        assert_eq!(
            verdict(&one(4, MotorStatus::Disabled), 100),
            Verdict::Reenable(4)
        );
        // A fault outranks a concurrent disabled joint.
        let mut mixed = one(1, MotorStatus::Disabled);
        mixed[5] = MotorStatus::Fault(FaultKind::Overload);
        assert!(matches!(verdict(&mixed, 100), Verdict::HardFault(_)));
    }

    #[test]
    fn a_cached_fault_on_a_stale_joint_still_stops_the_node() {
        // A faulted motor latches until power-cycled, so the fault is the
        // joint's condition even when its frames have since stopped.
        let mut statuses = [MotorStatus::Enabled; ARM_DOF];
        statuses[3] = MotorStatus::Fault(FaultKind::Overload);
        let mut stale = FRESH;
        stale[3] = true;
        let mut streaks = [0u32; ARM_DOF];
        assert!(matches!(
            assess(&statuses, &stale, &mut streaks, 100),
            Verdict::HardFault(_)
        ));
    }

    #[test]
    fn a_stale_joint_escalates_whatever_its_cache_says() {
        // Past the stale window the cached Enabled is not a measurement; the
        // joint is silent and runs out the escalation window like any other.
        let mut stale = FRESH;
        stale[4] = true;
        let statuses = [MotorStatus::Enabled; ARM_DOF];
        let mut streaks = [0u32; ARM_DOF];
        for tick in 1..100 {
            assert_eq!(
                assess(&statuses, &stale, &mut streaks, 100),
                Verdict::Continue,
                "tick {tick}"
            );
        }
        match assess(&statuses, &stale, &mut streaks, 100) {
            Verdict::HardFault(reason) => {
                assert!(reason.contains("j5"), "{reason}");
                assert!(reason.contains("silent"), "{reason}");
            }
            other => panic!("a stale joint must escalate, got {other:?}"),
        }
    }

    #[test]
    fn a_motor_that_goes_silent_under_load_escalates_to_a_hard_fault() {
        // A silent joint hangs limp while six others drive around it; its
        // streak runs the escalation window out and the node stops.
        let mut streaks = [0u32; ARM_DOF];
        let mut statuses = [MotorStatus::Enabled; ARM_DOF];
        statuses[4] = MotorStatus::Unreported;
        for tick in 1..100 {
            assert_eq!(
                assess(&statuses, &FRESH, &mut streaks, 100),
                Verdict::Continue,
                "tick {tick}"
            );
        }
        let escalated = assess(&statuses, &FRESH, &mut streaks, 100);
        match escalated {
            Verdict::HardFault(reason) => {
                assert!(reason.contains("j5"), "{reason}");
                assert!(reason.contains("silent"), "{reason}");
            }
            other => panic!("a silent motor must escalate, got {other:?}"),
        }
    }

    #[test]
    fn a_motor_stuck_in_an_unrecognised_status_escalates_without_reenables() {
        // Frames are arriving but their status nibble is undefined, so the
        // motor cannot be confirmed driving. It never said Disabled, so no
        // re-enable; it runs out the window like a silent one.
        let mut streaks = [0u32; ARM_DOF];
        let mut statuses = [MotorStatus::Enabled; ARM_DOF];
        statuses[3] = MotorStatus::Unknown(0x7);
        for tick in 1..10 {
            assert_eq!(
                assess(&statuses, &FRESH, &mut streaks, 10),
                Verdict::Continue,
                "tick {tick}"
            );
        }
        let escalated = assess(&statuses, &FRESH, &mut streaks, 10);
        match escalated {
            Verdict::HardFault(reason) => {
                assert!(reason.contains("j4"), "{reason}");
                assert!(reason.contains("unrecognised"), "{reason}");
            }
            other => panic!("a stuck unrecognised status must escalate, got {other:?}"),
        }
    }

    #[test]
    fn a_flapping_joint_still_escalates() {
        // The streak is per joint and reset only by its own driving ticks, so
        // a joint limp half the time still runs out its window however the
        // not-driving states alternate.
        let mut streaks = [0u32; ARM_DOF];
        let mut disabled = [MotorStatus::Enabled; ARM_DOF];
        disabled[3] = MotorStatus::Disabled;
        let mut silent = [MotorStatus::Enabled; ARM_DOF];
        silent[3] = MotorStatus::Unreported;
        for _ in 0..200 {
            let v = assess(&disabled, &FRESH, &mut streaks, 10);
            if let Verdict::HardFault(reason) = v {
                assert!(reason.contains("j4"), "{reason}");
                return;
            }
            let v = assess(&silent, &FRESH, &mut streaks, 10);
            if let Verdict::HardFault(reason) = v {
                assert!(reason.contains("j4"), "{reason}");
                return;
            }
        }
        panic!("a joint that is never driving must escalate however it flaps");
    }

    #[test]
    fn escalation_names_the_joint_that_ran_out_not_the_first_one_seen() {
        // j2 spends most of the window disabled and recovers; j5 disables
        // later and escalates only when its own streak runs out.
        let mut streaks = [0u32; ARM_DOF];
        let mut early = [MotorStatus::Enabled; ARM_DOF];
        early[1] = MotorStatus::Disabled;
        for _ in 0..50 {
            assess(&early, &FRESH, &mut streaks, 100);
        }
        let mut late = [MotorStatus::Enabled; ARM_DOF];
        late[4] = MotorStatus::Disabled;
        for _ in 0..99 {
            assert_eq!(
                assess(&late, &FRESH, &mut streaks, 100),
                Verdict::Reenable(4)
            );
        }
        match assess(&late, &FRESH, &mut streaks, 100) {
            Verdict::HardFault(reason) => assert!(reason.contains("j5"), "{reason}"),
            other => panic!("expected j5 to escalate, got {other:?}"),
        }
    }

    #[test]
    fn a_recovered_joint_starts_its_patience_over() {
        let mut streaks = [0u32; ARM_DOF];
        let mut disabled = [MotorStatus::Enabled; ARM_DOF];
        disabled[0] = MotorStatus::Disabled;
        for _ in 0..99 {
            assess(&disabled, &FRESH, &mut streaks, 100);
        }
        assert_eq!(
            assess(&[MotorStatus::Enabled; ARM_DOF], &FRESH, &mut streaks, 100),
            Verdict::Continue
        );
        assert_eq!(streaks[0], 0, "recovery clears that joint's streak");
    }

    #[test]
    fn a_stale_disabled_cache_gets_no_reenable() {
        // A stale joint's cached Disabled is not a fresh report; it is
        // silent, and re-enabling answers only a motor that just said so.
        let mut statuses = [MotorStatus::Enabled; ARM_DOF];
        statuses[2] = MotorStatus::Disabled;
        let mut stale = FRESH;
        stale[2] = true;
        let mut streaks = [0u32; ARM_DOF];
        assert_eq!(
            assess(&statuses, &stale, &mut streaks, 100),
            Verdict::Continue
        );
    }

    #[test]
    fn ticks_within_covers_the_window_exactly_and_never_hits_zero() {
        assert_eq!(
            ticks_within(NOT_DRIVING_ESCALATE_AFTER, Duration::from_millis(10)),
            100
        );
        assert_eq!(
            ticks_within(Duration::from_secs(1), Duration::from_secs(5)),
            1
        );
        assert_eq!(
            ticks_within(Duration::from_millis(500), Duration::from_millis(200)),
            3
        );
    }
}
