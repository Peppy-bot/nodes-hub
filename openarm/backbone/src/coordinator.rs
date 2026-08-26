//! The bimanual coordination loop. Every tick it advances both arms' planners
//! and both grippers to candidate setpoints, governs the whole step
//! against the self-collision model in one call (arms and grippers are one
//! governed configuration), and publishes the governed per-arm setpoints and
//! per-gripper gripper fractions. One loop owns the governor (the single collision
//! model), both planners, and the backbone-executed gripper moves, so everything is
//! always governed together against a consistent configuration, and the
//! governed result is fed back so the next tick chases from where each DOF was
//! actually allowed to go.
//!
//! An arm whose follower has stopped delivering state is frozen at its held
//! setpoint and its wire goes silent, and the first delivery after the gap
//! re-anchors that setpoint on the measured pose, so a follower that restarts
//! is never handed a target that drifted while nobody could see the arm.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use peppygen::NodeRunner;
use peppygen::exposed_actions::limb_motion::move_gripper;
use peppylib::runtime::CancellationToken;
use tokio::sync::{mpsc, watch};
use tracing::{error, info, warn};

use control_core::filters::LowPassFilter;
use control_core::pacer::Pacer;

use crate::arm_pair::ArmPair;
use crate::chase::rate_limited;
use crate::governor::{GovState, Governor, Guard};
use crate::liveness::{self, Admission, Liveness};
use crate::motion::{MOTION_TIMEOUT_FACTOR, MoveBudget};
use crate::planner::{self, BusyGuard, Goal, Planner};
use crate::publish::{LimbStateSnapshot, Publishers};
use crate::streams::{ArmState, GovernorConfig, GripperCommand, GripperState};
use crate::types::{ARM_DOF, JointVec, Side};
use crate::upstream::{Upstream, UpstreamMode};

/// How long [`seed_all`] waits for an arm's first measured state before warning that
/// the backbone is still blocked, so a silent arm is visible in the log instead of an
/// indefinite quiet stall.
const SEED_WAIT_WARN_PERIOD: Duration = Duration::from_secs(2);

/// One arm's inbound channels into the coordinator: the leading node's arm
/// command stream, its gripper opening command stream, the measured arm state,
/// the measured gripper opening, the accepted-goal queues, and the single-flight
/// busy flags (one for arm moves, one for gripper moves).
///
/// The two command streams are held as their `watch::Sender`, not a receiver: the
/// coordinator both reads the latest (`borrow`) and clears it (`send_replace`)
/// while a move runs on that side, so a setpoint still in flight when the move was
/// fired cannot re-target the arm (or snap the grippers) when the move ends. The
/// stream listener holds a clone of the same sender and fills it.
pub struct ArmChannels {
    pub command: watch::Sender<Option<Upstream>>,
    pub gripper_command: watch::Sender<Option<GripperCommand>>,
    pub measured: watch::Receiver<Option<ArmState>>,
    pub gripper: watch::Receiver<Option<GripperState>>,
    pub goals: mpsc::Receiver<Goal>,
    pub busy: Arc<AtomicBool>,
    pub gripper_goals: mpsc::Receiver<GripperGoal>,
    pub gripper_busy: Arc<AtomicBool>,
}

/// The coordinator's run parameters. A *leading node* that stops streaming gets
/// no deadman: joints mode holds the last governed setpoint where it is, and
/// pose mode keeps converging to the last received pose, rate-capped and
/// governed (the planner documents that choice). A *follower* that stops
/// delivering is the opposite case and does have one, keyed to the control
/// period rather than configured here (see [`crate::liveness`]).
pub struct RunConfig {
    pub cycle_period: Duration,
    /// Cutoff (Hz) for the low-pass on each published desired velocity. `dq` is a
    /// per-tick position difference scaled by `1/dt`, so it amplifies any setpoint noise
    /// by the control rate; filtering it keeps the arm's Kd term from buzzing on a noisy
    /// stream without touching the desired position.
    pub velocity_filter_cutoff_hz: f64,
    /// Which upstream kind this instance follows. Joints mode has no pose_link
    /// peer, so the upstream relay skips the per-tick FK and pose publish.
    pub upstream_mode: UpstreamMode,
}

/// An accepted `move_gripper` goal handed to the coordinator, which executes it
/// through the same per-tick governing as everything else (the gripper analog of
/// [`Goal`] for the arms). The opening is the validated goal fraction; the
/// effort cap is validated non-negative, `None` when the goal carried no
/// preference.
pub struct GripperGoal {
    pub opening: f64,
    pub max_effort: Option<f64>,
    pub ctx: move_gripper::GoalContext,
}

impl GripperGoal {
    /// Complete unstarted, `success: false`. The busy flag is the caller's
    /// concern.
    pub async fn refuse(self, reason: &str, reported_frac: f64) {
        if let Err(e) = self
            .ctx
            .complete(false, reason.to_string(), reported_frac, 0.0)
            .await
        {
            error!("move_gripper refuse: {e}");
        }
    }
}

/// A backbone-executed gripper move in flight: the opening chases `target_frac`
/// through the governor until the governed chase lands on the target, the goal
/// is cancelled, or the move overruns its budget (a governed clamp short of the
/// target ends here). Like the arm's trajectory tiers, completion is graded on
/// the commanded motion, not the measured grippers; the result reports the measured
/// opening and the caller judges it. The busy guard releases the side's
/// single-flight slot on any exit.
struct GripperMove {
    target_frac: f64,
    max_effort: Option<f64>,
    ctx: move_gripper::GoalContext,
    started: Instant,
    /// Nominal chase duration; the runtime aborts once the move runs past
    /// `MOTION_TIMEOUT_FACTOR` times this, exactly as the arm servo does.
    /// Re-budgeted by [`MoveBudget::after_rate_change`] when the operator slows
    /// the opening rate mid-move.
    budget: MoveBudget,
    _busy: BusyGuard,
}

/// Run the coordination loop. Holds the governor and both planners. Runs until
/// the node's cancellation token fires; returns `Err` if a publisher cannot be
/// declared at bringup. Any return takes the node down (the supervisor in `main`
/// treats it as fatal).
pub async fn run(
    runner: Arc<NodeRunner>,
    mut governor: Governor,
    mut planners: ArmPair<Planner>,
    mut channels: ArmPair<ArmChannels>,
    governor_config: watch::Receiver<GovernorConfig>,
    config: RunConfig,
    token: CancellationToken,
) -> peppygen::Result<()> {
    let RunConfig {
        cycle_period,
        velocity_filter_cutoff_hz,
        upstream_mode,
    } = config;
    let publishers = Publishers::declare(&runner).await?;

    // Hold each arm's real pose, not a neutral zero: wait for the first measured
    // state from both arms and seed the held setpoints there before publishing.
    if seed_all(&mut channels, &mut planners).await.is_err() {
        return Ok(());
    }
    info!("bimanual backbone: both arms reporting; governed streaming begins");

    // A gripper's latest measured gripper fraction. `seed_all` gated on each
    // side's first reading and the watch never reverts to `None`, so the read
    // is infallible from here on.
    let gripper_fraction = |gripper: &watch::Receiver<Option<GripperState>>| {
        gripper
            .borrow()
            .map(|g| g.fraction)
            .expect("seed gated on the first gripper opening")
    };
    // Track the last governed opening fraction per gripper: the governed
    // configuration's `prev`. Anchored on the measured grippers (here and whenever a
    // side idles) so governing always ramps from where the fingers really are;
    // the opening rate is read from the governor (its single owner) rather than
    // carried here.
    let mut governed_grippers = ArmPair::new(
        gripper_fraction(&channels.left.gripper),
        gripper_fraction(&channels.right.gripper),
    );
    // In-flight backbone-executed gripper moves, one single-flight slot per side.
    let mut gripper_moves: ArmPair<Option<GripperMove>> = ArmPair::new(None, None);

    let dt = cycle_period.as_secs_f64();
    // One low-pass per joint per arm, smoothing the published desired velocity. `main`
    // validates `0 < cutoff < Nyquist` at bringup, a strict superset of what `from_cutoff`
    // rejects, so construction cannot fail: build one filter and copy it per joint (the
    // same bringup-invariant pattern as `Pacer::new(...).expect(...)`).
    let filter = LowPassFilter::from_cutoff(velocity_filter_cutoff_hz, dt)
        .expect("velocity_filter_cutoff_hz is bringup-validated in (0, Nyquist)");
    let mut dq_filters = ArmPair::new([filter; ARM_DOF], [filter; ARM_DOF]);
    // The proximity readout is for human eyes, so publish it at ~20 Hz rather than
    // the control rate: one extra distance query every `readout_every` ticks.
    let readout_every = (0.05 / dt).round().max(1.0) as u64;
    let mut tick: u64 = 0;
    let mut pacer = Pacer::new(cycle_period).expect("control_rate_hz is asserted > 0 at startup");
    // Both arms are seeded from their first measurement above, which is the
    // anchor a recovery would re-establish, so they start live.
    let stale_limit = liveness::stale_limit(cycle_period);
    let (mut arm_liveness, mut gripper_liveness) = {
        let seeded = Instant::now();
        (
            ArmPair::new(Liveness::seeded(seeded), Liveness::seeded(seeded)),
            ArmPair::new(Liveness::seeded(seeded), Liveness::seeded(seeded)),
        )
    };
    // The grippers chase their target at the gripper rate exactly as the planner
    // velocity-limits the arm candidates; an idle side chases nowhere.
    let chase_gripper = |prev_frac: f64, target: Option<GripperTarget>, rate: f64| -> f64 {
        rate_limited(prev_frac, target.map_or(prev_frac, |t| t.frac), rate, dt)
    };
    loop {
        consume_streams_of_busy_sides(&channels);
        apply_controls(&mut governor, &mut planners, *governor_config.borrow());
        // Re-read every tick: the operator retunes this live, and the chase, the
        // move budget and the governor's own clamp must agree within a tick or a
        // move budgeted at one rate is driven at another and times out short.
        let gripper_rate = governor.max_gripper_rate_frac_s();
        let now = Instant::now();

        let arm_admission = admit_arms(&mut arm_liveness, &channels, now, stale_limit);
        let gripper_admission =
            admit_grippers(&mut gripper_liveness, &mut channels, now, stale_limit);
        let arm_ticks = advance_arms(&mut channels, &mut planners, arm_admission, now).await;
        let arm_candidate = ArmPair::new(arm_ticks.left.candidate, arm_ticks.right.candidate);
        let hands = ArmPair::new(arm_ticks.left.streamed_hand, arm_ticks.right.streamed_hand);
        let measured_grippers = ArmPair::new(
            gripper_fraction(&channels.left.gripper),
            gripper_fraction(&channels.right.gripper),
        );
        service_gripper_moves(
            &mut gripper_moves,
            &mut channels,
            governed_grippers,
            measured_grippers,
            gripper_rate,
            now,
        )
        .await;

        // Resolve each gripper's target for this tick: an in-flight move owns the
        // opening; otherwise the latest leader command drives it; otherwise the
        // side idles (never commanded, or unpaired), silent on the wire with the
        // governed opening re-anchored on the measured grippers.
        let targets = ArmPair::new(
            gripper_target(&gripper_moves.left, &channels.left),
            gripper_target(&gripper_moves.right, &channels.right),
        );
        if targets.left.is_none() {
            governed_grippers.left = measured_grippers.left;
        }
        if targets.right.is_none() {
            governed_grippers.right = measured_grippers.right;
        }

        // One governed configuration: the last published setpoints and grippers
        // as `prev`, the rate-limited chases as the candidate. The governor
        // throttles, holds, scans and monitors everything through one barrier.
        let prev = GovState::new(
            ArmPair::new(planners.left.setpoint(), planners.right.setpoint()),
            governed_grippers,
        );
        let cand = GovState::new(
            arm_candidate,
            ArmPair::new(
                chase_gripper(prev.grippers.left, targets.left, gripper_rate),
                chase_gripper(prev.grippers.right, targets.right, gripper_rate),
            ),
        );
        let measured = measured_config(&channels, &prev, measured_grippers);
        let governed = governor.govern(&prev, &cand, &measured, &hands, dt);
        governed_grippers = governed.grippers;

        // Publish one governed setpoint per arm on its pairing slot; the slot
        // scopes the stream to its paired arm, so the message names no side.
        for (planner, filters, wire, prev_q, governed_q, admission) in [
            (
                &mut planners.left,
                &mut dq_filters.left,
                &publishers.arm_setpoints.left,
                prev.arms.left,
                governed.arms.left,
                arm_admission.left,
            ),
            (
                &mut planners.right,
                &mut dq_filters.right,
                &publishers.arm_setpoints.right,
                prev.arms.right,
                governed.arms.right,
                arm_admission.right,
            ),
        ] {
            // Desired velocity is the per-tick position delta; low-pass it per joint so a
            // noisy stream does not drive the arm's Kd term into buzz. The published
            // position (`governed_q`) is untouched, so tracking is unaffected.
            let dq = filtered_velocity(filters, &governed_q, &prev_q, dt);
            // Every side commits (identity for a stale one, whose candidate was
            // the frozen setpoint), but a stale side stays silent so its
            // follower holds its own last setpoint rather than tracking one the
            // backbone can no longer vouch for.
            planner.commit(governed_q);
            if admission == Admission::Stale {
                continue;
            }
            wire.send(&governed_q, &dq).await;
        }

        // Publish each active side's governed opening fraction on its pairing
        // slot (the slot scopes the stream to its paired gripper, so the message
        // names no side); an idle side stays silent and its gripper holds its
        // opening.
        for (wire, gripper_frac, target) in [
            (
                &publishers.gripper_setpoints.left,
                governed_grippers.left,
                targets.left,
            ),
            (
                &publishers.gripper_setpoints.right,
                governed_grippers.right,
                targets.right,
            ),
        ] {
            if let Some(target) = target {
                wire.send(gripper_frac, target.max_effort).await;
            }
        }

        relay_upstream(
            &publishers,
            &channels,
            &mut planners,
            arm_admission,
            gripper_admission,
            upstream_mode,
        )
        .await;

        // Operator readouts (rate-limited), sharing one cadence: the proximity
        // feed and the whole-robot state snapshot.
        if tick.is_multiple_of(readout_every) {
            // The nearest checked pair's signed distance and link names, live
            // regardless of the governor state, plus the governor's current
            // disposition of the commanded motion.
            if let Some(p) = governor.proximity(&prev) {
                let guard = governor.guard();
                publishers
                    .send_status(
                        p.distance,
                        p.link_a,
                        p.link_b,
                        guard == Guard::Throttling,
                        guard == Guard::Stopped,
                    )
                    .await;
            }
            // Published only while every limb has a live measurement, so a
            // reader never sees a snapshot the backbone has stopped vouching
            // for; a silent stretch reads as staleness on the consumer's side.
            if let Some(snapshot) =
                limb_state_snapshot(&channels, &mut planners, arm_admission, gripper_admission)
            {
                publishers.send_limb_states(&snapshot).await;
            }
        }
        tick += 1;
        tokio::select! {
            _ = token.cancelled() => return Ok(()),
            _ = pacer.pace() => {}
        }
    }
}

/// Wipe the streamed command of any side running a discrete move.
///
/// A setpoint still in flight when the move was fired (the leading node streams
/// at the control rate, so one is almost always queued) would otherwise survive in
/// the watch and re-target the arm, or snap the grippers, the moment the move ends
/// and Follow resumes. Streaming and discrete moves are mutually exclusive per
/// side, so this never drops a command the operator still wants.
fn consume_streams_of_busy_sides(channels: &ArmPair<ArmChannels>) {
    for ch in [&channels.left, &channels.right] {
        if ch.busy.load(Ordering::Acquire) {
            ch.command.send_replace(None);
        }
        if ch.gripper_busy.load(Ordering::Acquire) {
            ch.gripper_command.send_replace(None);
        }
    }
}

/// Apply the commander's latest runtime controls. Cheap no-ops when unchanged;
/// an invalid band or speed is rejected by the setter, keeping the last good
/// value, so a malformed control message cannot disarm the governor.
fn apply_controls(governor: &mut Governor, planners: &mut ArmPair<Planner>, cfg: GovernorConfig) {
    governor.set_enabled(cfg.enabled);
    governor.set_band(cfg.d_stop, cfg.d_safe);
    // The cap lives twice by design: the governor limits streamed hands with
    // it, the planners budget planned moves at admission and step streamed
    // poses with it.
    governor.set_ee_cap(cfg.max_ee_velocity_m_s);
    governor.set_gripper_rate(cfg.max_gripper_rate_frac_s);
    planners.left.set_max_ee_velocity(cfg.max_ee_velocity_m_s);
    planners.right.set_max_ee_velocity(cfg.max_ee_velocity_m_s);
}

/// Judge each arm's follower before commanding it.
///
/// A follower that has stopped delivering cannot be vouched for: its limb
/// freezes at the held setpoint and its wire goes silent, so the follower holds
/// its own last setpoint instead of tracking one that keeps advancing on the
/// operator's stream while the real arm drifts. The first delivery back
/// re-anchors the held setpoint on the measured pose, so the limb never steps
/// by the drift it accumulated unseen.
fn admit_arms(
    liveness: &mut ArmPair<Liveness>,
    channels: &ArmPair<ArmChannels>,
    now: Instant,
    stale_limit: Duration,
) -> ArmPair<Admission> {
    ArmPair::new(
        liveness.left.admit(
            channels.left.measured.has_changed().unwrap_or(false),
            now,
            stale_limit,
        ),
        liveness.right.admit(
            channels.right.measured.has_changed().unwrap_or(false),
            now,
            stale_limit,
        ),
    )
}

/// Judge each gripper follower's delivery, the opening analog of
/// [`admit_arms`] over each gripper's own pairing.
///
/// Gates the upstream relay only: a gripper that has stopped delivering must
/// not have its last aperture republished under a fresh timestamp, which would show
/// the leading node a live-looking back-channel. The governed opening still
/// streams down, because a held gripper holds where the operator put it rather
/// than drifting away unseen the way an uncommanded arm does.
fn admit_grippers(
    liveness: &mut ArmPair<Liveness>,
    channels: &mut ArmPair<ArmChannels>,
    now: Instant,
    stale_limit: Duration,
) -> ArmPair<Admission> {
    // Read the flag, then mark the watch seen, so the next tick asks about that
    // tick's delivery rather than every delivery since the loop began. Nothing
    // else updates this watch; the other readers only borrow.
    let delivered_left = channels.left.gripper.has_changed().unwrap_or(false);
    let _ = channels.left.gripper.borrow_and_update();
    let delivered_right = channels.right.gripper.has_changed().unwrap_or(false);
    let _ = channels.right.gripper.borrow_and_update();
    ArmPair::new(
        liveness.left.admit(delivered_left, now, stale_limit),
        liveness.right.admit(delivered_right, now, stale_limit),
    )
}

/// Advance both planners to this tick's candidate setpoints and hand bases.
async fn advance_arms(
    channels: &mut ArmPair<ArmChannels>,
    planners: &mut ArmPair<Planner>,
    admission: ArmPair<Admission>,
    now: Instant,
) -> ArmPair<planner::Tick> {
    ArmPair::new(
        tick_arm(&mut channels.left, &mut planners.left, admission.left, now).await,
        tick_arm(
            &mut channels.right,
            &mut planners.right,
            admission.right,
            now,
        )
        .await,
    )
}

/// Service both sides' backbone-executed gripper moves: admit a queued goal
/// into a free side, and complete an in-flight move on the chase landing,
/// cancellation, or a budget overrun. `governed` is last tick's opening, which
/// is also the chase base a newly admitted goal budgets from.
async fn service_gripper_moves(
    moves: &mut ArmPair<Option<GripperMove>>,
    channels: &mut ArmPair<ArmChannels>,
    governed: ArmPair<f64>,
    measured: ArmPair<f64>,
    gripper_rate_frac_s: f64,
    now: Instant,
) {
    service_gripper_move(
        &mut moves.left,
        &mut channels.left,
        governed.left,
        measured.left,
        gripper_rate_frac_s,
        now,
    )
    .await;
    service_gripper_move(
        &mut moves.right,
        &mut channels.right,
        governed.right,
        measured.right,
        gripper_rate_frac_s,
        now,
    )
    .await;
}

/// The real configuration the governor's measured-state monitor judges against.
/// An arm falls back to its held setpoint if a measurement is momentarily
/// absent (only before the first state, which `seed_all` already gated on), so
/// a gap never reads as a breach.
fn measured_config(
    channels: &ArmPair<ArmChannels>,
    prev: &GovState,
    grippers: ArmPair<f64>,
) -> GovState {
    let positions = |ch: &ArmChannels, held: JointVec| {
        ch.measured.borrow().as_ref().map_or(held, |m| m.positions)
    };
    GovState::new(
        ArmPair::new(
            positions(&channels.left, prev.arms.left),
            positions(&channels.right, prev.arms.right),
        ),
        grippers,
    )
}

/// A stale side reads as nothing to report, whatever the measurement is.
fn live<T>(admission: Admission, read: impl FnOnce() -> Option<T>) -> Option<T> {
    (admission != Admission::Stale).then(read).flatten()
}

/// Gather one whole-robot snapshot for the limb_state readout: every limb's
/// live measurement, the arms' grasp-point poses FK'd from those joints. Any
/// stale or missing limb yields `None`: a partial robot is not a snapshot.
fn limb_state_snapshot(
    channels: &ArmPair<ArmChannels>,
    planners: &mut ArmPair<Planner>,
    arm_admission: ArmPair<Admission>,
    gripper_admission: ArmPair<Admission>,
) -> Option<LimbStateSnapshot> {
    let left = live(arm_admission.left, || *channels.left.measured.borrow())?;
    let right = live(arm_admission.right, || *channels.right.measured.borrow())?;
    let left_gripper = live(gripper_admission.left, || *channels.left.gripper.borrow())?;
    let right_gripper = live(gripper_admission.right, || *channels.right.gripper.borrow())?;
    Some(LimbStateSnapshot {
        joints: ArmPair::new(left.positions, right.positions),
        poses: ArmPair::new(
            planners.left.ee_pose_world(&left.positions),
            planners.right.ee_pose_world(&right.positions),
        ),
        openings: ArmPair::new(left_gripper.fraction, right_gripper.fraction),
    })
}

/// Relay every limb's measured state up its leader pairing slot, so the
/// leading node sees the same back-channel a follower gives the backbone. A
/// stale side's arm relay goes silent with its setpoint stream:
/// republishing a frozen measurement under a fresh timestamp would show
/// the leading node a live-looking limb the backbone has stopped vouching
/// for. Each watch is read out before its send, so no borrow guard is held
/// across an await.
async fn relay_upstream(
    publishers: &Publishers,
    channels: &ArmPair<ArmChannels>,
    planners: &mut ArmPair<Planner>,
    arm_admission: ArmPair<Admission>,
    gripper_admission: ArmPair<Admission>,
    upstream_mode: UpstreamMode,
) {
    let arms = ArmPair::new(
        live(arm_admission.left, || *channels.left.measured.borrow()),
        live(arm_admission.right, || *channels.right.measured.borrow()),
    );
    let grippers = ArmPair::new(
        live(gripper_admission.left, || *channels.left.gripper.borrow()),
        live(gripper_admission.right, || *channels.right.gripper.borrow()),
    );
    // In pose mode the arm back-channel also carries the same measurement as
    // the end-effector pose, FK'd here; joints mode has no pose_link peer, so
    // that work is skipped.
    for (wire, pose_wire, planner, measured) in [
        (
            &publishers.arm_states.left,
            &publishers.arm_pose_states.left,
            &mut planners.left,
            arms.left,
        ),
        (
            &publishers.arm_states.right,
            &publishers.arm_pose_states.right,
            &mut planners.right,
            arms.right,
        ),
    ] {
        if let Some(m) = measured {
            wire.send(&m.positions, &m.velocities).await;
            if upstream_mode == UpstreamMode::Pose {
                pose_wire.send(&planner.ee_pose_world(&m.positions)).await;
            }
        }
    }
    for (wire, measured) in [
        (&publishers.gripper_states.left, grippers.left),
        (&publishers.gripper_states.right, grippers.right),
    ] {
        if let Some(g) = measured {
            wire.send(&g).await;
        }
    }
}

/// All senders on the measured-state channel dropped (its only producer is the
/// state listener task), so no measurement will ever arrive: seeding is abandoned.
struct Shutdown;

/// Reason completing every goal refused while the followers are still silent.
const SEED_REFUSAL: &str = "the follower has not reported its first state yet";

/// Refusal for goals reaching an arm whose follower stream has gone stale.
const STALE_REFUSAL: &str = "the follower stopped reporting";

/// Wait for both arms' first measured states and both grippers' first
/// gripper fractions, then seed each planner's held setpoint from its measured pose
/// (clamped into the joint limits). One wait over all four goal queues: a
/// goal accepted for EITHER side before its follower reports is refused with
/// its busy claim released, so a silent side cannot strand the other side's
/// goals. Warns periodically while a stream stays silent; `Err(Shutdown)` if
/// a channel closes first.
async fn seed_all(
    channels: &mut ArmPair<ArmChannels>,
    planners: &mut ArmPair<Planner>,
) -> Result<(), Shutdown> {
    loop {
        tokio::select! {
            firsts = async {
                wait_for_first(&mut channels.left.measured, Side::Left, "arm measured state")
                    .await?;
                wait_for_first(&mut channels.left.gripper, Side::Left, "gripper opening").await?;
                wait_for_first(&mut channels.right.measured, Side::Right, "arm measured state")
                    .await?;
                wait_for_first(&mut channels.right.gripper, Side::Right, "gripper opening").await
            } => {
                firsts?;
                break;
            }
            Some(goal) = channels.left.goals.recv() => {
                refuse_seed_arm_goal(goal, &channels.left, &mut planners.left).await;
            }
            Some(goal) = channels.right.goals.recv() => {
                refuse_seed_arm_goal(goal, &channels.right, &mut planners.right).await;
            }
            Some(goal) = channels.left.gripper_goals.recv() => {
                refuse_seed_gripper_goal(goal, &channels.left).await;
            }
            Some(goal) = channels.right.gripper_goals.recv() => {
                refuse_seed_gripper_goal(goal, &channels.right).await;
            }
        }
    }
    for (channels, planner) in [
        (&channels.left, &mut planners.left),
        (&channels.right, &mut planners.right),
    ] {
        let q0 = channels
            .measured
            .borrow()
            .expect("gated on first state")
            .positions;
        planner.seed_from_measured(q0);
    }
    Ok(())
}

/// Refuse one arm goal during the seed wait, reporting the measured pose when
/// one already arrived and releasing the goal's busy claim.
async fn refuse_seed_arm_goal(goal: Goal, channels: &ArmChannels, planner: &mut Planner) {
    let measured = *channels.measured.borrow();
    let _release = BusyGuard(channels.busy.clone());
    let reported = measured.map_or_else(|| planner.setpoint(), |m| m.positions);
    goal.refuse(SEED_REFUSAL, reported, planner).await;
}

/// Refuse one gripper goal during the seed wait, reporting the measured
/// opening when one already arrived and releasing the goal's busy claim.
async fn refuse_seed_gripper_goal(goal: GripperGoal, channels: &ArmChannels) {
    let _release = BusyGuard(channels.gripper_busy.clone());
    let reported = (*channels.gripper.borrow()).map_or(0.0, |g| g.fraction);
    goal.refuse(SEED_REFUSAL, reported).await;
}

/// Block until `latest` holds its first value, warning every
/// [`SEED_WAIT_WARN_PERIOD`] while `what` stays silent; `Err(Shutdown)` if the
/// channel closes first (its listener task died).
async fn wait_for_first<T>(
    latest: &mut watch::Receiver<Option<T>>,
    side: Side,
    what: &str,
) -> Result<(), Shutdown> {
    loop {
        match tokio::time::timeout(SEED_WAIT_WARN_PERIOD, latest.wait_for(Option::is_some)).await {
            Ok(Ok(_)) => return Ok(()),
            Ok(Err(_)) => {
                error!(
                    "{} {what} channel closed before its first value",
                    side.label()
                );
                return Err(Shutdown);
            }
            Err(_) => warn!(
                "{} {what} not reported yet; backbone waiting to stream",
                side.label()
            ),
        }
    }
}

/// Advance one arm's planner to its candidate setpoint for this tick: anchor on the
/// measured pose (or the held setpoint if no measurement yet), feed the latest
/// leader command, and admit any pending move goal.
async fn tick_arm(
    channels: &mut ArmChannels,
    planner: &mut Planner,
    admission: Admission,
    now: Instant,
) -> planner::Tick {
    let measured_q = match *channels.measured.borrow_and_update() {
        Some(s) => s.positions,
        None => planner.setpoint(),
    };
    // A stale limb holds exactly where it was last governed. Advancing the
    // planner would walk the setpoint away from an arm nobody can see, and a
    // held setpoint needs no hand basis: there is no streamed motion to cap.
    // A move in that darkness can neither progress nor be verified, so it and
    // any queued goal fail here, freeing the claim each holds: a ready share
    // that kept its claim would wedge the ready action for good.
    if admission == Admission::Stale {
        planner.abort_active(STALE_REFUSAL, measured_q, now).await;
        while let Ok(goal) = channels.goals.try_recv() {
            let _release = BusyGuard(channels.busy.clone());
            goal.refuse(STALE_REFUSAL, measured_q, planner).await;
        }
        return planner::Tick {
            candidate: planner.setpoint(),
            streamed_hand: None,
        };
    }
    // First delivery after a gap: the held setpoint is now fiction, so adopt
    // the measured pose before advancing. The per-joint velocity limit in the
    // chase then walks it back to the operator's command instead of the
    // follower stepping the whole divergence in one tick.
    if admission == Admission::Reanchor {
        warn!("arm follower stream recovered; re-anchoring on the measured pose");
        planner.seed_from_measured(measured_q);
    }
    // Read out so no watch borrow guard is held across the await below.
    let command = *channels.command.borrow();
    planner
        .tick(
            measured_q,
            command,
            &mut channels.goals,
            &channels.busy,
            now,
        )
        .await
}

/// Landing threshold for the governed chase, in opening fraction. Purely
/// numerical: the rate-limited chase lands on its target up to IEEE rounding
/// residue, and the governor passes an unthrottled candidate through
/// bit-exact, so anything past this is a real clamp. Nanometer-scale gripper
/// travel, orders of magnitude below actuator resolution; goal satisfaction
/// is the caller's judgment from the reported `final_opening`.
const GRIPPER_LANDED_FRAC: f64 = 1e-9;

/// Nominal duration (s) of a gripper move admitted with the chase at
/// `governed_frac`: the commanded travel at the opening rate. The gripper
/// analog of the arm servo's plan-time rollout, graded by the same
/// [`motion_timed_out`] rule.
fn gripper_move_budget_s(governed_frac: f64, target_frac: f64, gripper_rate_frac_s: f64) -> f64 {
    (target_frac - governed_frac).abs() / gripper_rate_frac_s
}

/// Admit a queued gripper goal into a free side and drive an in-flight move to
/// its terminal: the governed chase landing on the target completes it,
/// cancellation ends it, and overrunning the budget sized at admission fails it
/// (a collision-governed clamp short of the target lands here, so the message
/// says so). Every terminal reports the measured grippers as `final_opening`; like
/// the arm's post-move reached check, judging that against the goal belongs to
/// the caller. The busy slot releases with the move on every path.
async fn service_gripper_move(
    mv: &mut Option<GripperMove>,
    channels: &mut ArmChannels,
    governed_frac: f64,
    measured_frac: f64,
    gripper_rate_frac_s: f64,
    now: Instant,
) {
    // Drain fully: every queued goal is answered, never left parked.
    while let Ok(goal) = channels.gripper_goals.try_recv() {
        if mv.is_none() {
            *mv = Some(GripperMove {
                target_frac: goal.opening,
                max_effort: goal.max_effort,
                ctx: goal.ctx,
                started: now,
                budget: MoveBudget::new(
                    gripper_move_budget_s(governed_frac, goal.opening, gripper_rate_frac_s),
                    gripper_rate_frac_s,
                ),
                _busy: BusyGuard(channels.gripper_busy.clone()),
            });
        } else {
            goal.refuse("another gripper move is in flight", measured_frac)
                .await;
        }
    }
    let Some(m) = mv.as_mut() else { return };
    let elapsed_s = now.duration_since(m.started).as_secs_f64();
    m.budget = m.budget.after_rate_change(elapsed_s, gripper_rate_frac_s);
    let m = &*m;
    let landed = (governed_frac - m.target_frac).abs() <= GRIPPER_LANDED_FRAC;
    let (success, message, cancelled) = if m.ctx.is_cancelled() {
        (false, "goal cancelled".to_string(), true)
    } else if landed {
        (true, "move complete".to_string(), false)
    } else if m.budget.timed_out(elapsed_s) {
        (
            false,
            format!(
                "overran {MOTION_TIMEOUT_FACTOR:.0}x its {:.1}s nominal travel, short of the target (a collision-governed clamp ends here)",
                m.budget.seconds()
            ),
            false,
        )
    } else {
        return;
    };
    let m = mv.take().expect("in-flight move checked above");
    let result = if cancelled {
        m.ctx
            .complete_cancelled(success, message, measured_frac, elapsed_s)
            .await
    } else {
        m.ctx
            .complete(success, message, measured_frac, elapsed_s)
            .await
    };
    if let Err(e) = result {
        error!("move_gripper complete: {e}");
    }
}

/// One side's resolved tick command: the opening fraction to chase and the
/// effort cap to relay on the pairing setpoint (`None` = no preference, sent
/// as the wire's 0, leaving the follower's configured ceiling in charge).
#[derive(Clone, Copy, Debug, PartialEq)]
struct GripperTarget {
    frac: f64,
    max_effort: Option<f64>,
}

/// The side's target for this tick: an in-flight backbone-executed
/// move owns it; otherwise the leading node's streamed command drives it;
/// otherwise `None` (idle: before any command, or on an unpaired side).
fn gripper_target(mv: &Option<GripperMove>, channels: &ArmChannels) -> Option<GripperTarget> {
    if let Some(m) = mv {
        return Some(GripperTarget {
            frac: m.target_frac,
            max_effort: m.max_effort,
        });
    }
    follow_gripper_target(&channels.gripper_command.borrow().clone())
}

/// Resolve the streamed target: the latest leader command with its opening
/// clamped into `[0, 1]`, or `None` when none has arrived. The stream is
/// paired to one producer, so there is nothing to arbitrate; a stopped
/// producer just leaves the last opening in place, held by the gripper.
fn follow_gripper_target(command: &Option<GripperCommand>) -> Option<GripperTarget> {
    command.as_ref().map(|c| GripperTarget {
        frac: c.opening.clamp(0.0, 1.0),
        max_effort: c.max_effort,
    })
}

/// The published desired velocity: per joint, the tick's position delta scaled to a rate
/// and low-passed. `filters` carries the per-joint state across ticks, so the smoothing is
/// over time, not within a tick. Only the velocity is shaped; the position (`governed_q`)
/// is published as-is.
fn filtered_velocity(
    filters: &mut [LowPassFilter; ARM_DOF],
    governed_q: &JointVec,
    prev_q: &JointVec,
    dt: f64,
) -> JointVec {
    std::array::from_fn(|j| filters[j].filter((governed_q[j] - prev_q[j]) / dt))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmd(opening: f64) -> Option<GripperCommand> {
        Some(GripperCommand {
            opening,
            max_effort: None,
        })
    }

    fn target(frac: f64) -> Option<GripperTarget> {
        Some(GripperTarget {
            frac,
            max_effort: None,
        })
    }

    const DT: f64 = 0.01;
    // The node default; well below the 50 Hz Nyquist so it actually attenuates.
    const CUTOFF_HZ: f64 = 15.0;

    fn dq_filters() -> [LowPassFilter; ARM_DOF] {
        std::array::from_fn(|_| LowPassFilter::from_cutoff(CUTOFF_HZ, DT).unwrap())
    }

    #[test]
    fn filtered_velocity_differentiates_position_into_a_rate() {
        // A steady position delta of 0.01 rad/tick at 100 Hz is a 1 rad/s velocity; the
        // first tick seeds on that value (no startup transient).
        let mut filters = dq_filters();
        let prev = [0.0; ARM_DOF];
        let q = [0.01; ARM_DOF];
        let dq = filtered_velocity(&mut filters, &q, &prev, DT);
        assert!(
            dq.iter().all(|v| (v - 1.0).abs() < 1e-12),
            "delta/dt is the rate"
        );
    }

    #[test]
    fn filtered_velocity_attenuates_a_noisy_stream() {
        // A jittering position (alternating +/-) makes the raw per-tick velocity swing
        // by +/- (2*amp/dt); the low-pass carries state across ticks and damps it.
        let mut filters = dq_filters();
        let amp = 0.001;
        let mut prev = [0.0; ARM_DOF];
        let mut worst_raw: f64 = 0.0;
        let mut worst_filtered: f64 = 0.0;
        for k in 0..200 {
            let q = [if k % 2 == 0 { amp } else { -amp }; ARM_DOF];
            let raw = (q[0] - prev[0]) / DT;
            let filtered = filtered_velocity(&mut filters, &q, &prev, DT)[0];
            if k > 1 {
                worst_raw = worst_raw.max(raw.abs());
                worst_filtered = worst_filtered.max(filtered.abs());
            }
            prev = q;
        }
        assert!(
            worst_filtered < worst_raw * 0.5,
            "the low-pass more than halves the jitter amplitude ({worst_filtered} vs {worst_raw})"
        );
    }

    #[test]
    fn follow_clamps_the_wire_fraction() {
        // In-range passes through; past-open and negative commands clamp into
        // [0, 1] at this boundary.
        assert_eq!(follow_gripper_target(&cmd(0.5)), target(0.5));
        assert_eq!(follow_gripper_target(&cmd(1.5)), target(1.0));
        assert_eq!(follow_gripper_target(&cmd(-0.5)), target(0.0));
    }

    #[test]
    fn follow_relays_the_effort_cap_unchanged() {
        let command = Some(GripperCommand {
            opening: 0.5,
            max_effort: Some(1.5),
        });
        assert_eq!(
            follow_gripper_target(&command),
            Some(GripperTarget {
                frac: 0.5,
                max_effort: Some(1.5),
            })
        );
    }

    #[test]
    fn follow_stays_idle_without_a_command() {
        assert_eq!(follow_gripper_target(&None), None);
    }

    #[test]
    fn a_consumed_command_holds_the_move_endpoint_until_a_newer_one() {
        // The gripper twin of the arm handoff: an accepted move_gripper clears
        // the side's command watch (`send_replace(None)` in the handler), so the
        // gripper follows nothing new and holds the move's endpoint until a
        // command that arrives after the clear. Locks the contract; the handler
        // performing the clear is covered by the live regression.
        let (tx, rx) = watch::channel(None);

        tx.send_replace(cmd(0.6));
        assert_eq!(
            follow_gripper_target(&rx.borrow()),
            target(0.6),
            "a live streamed opening is followed"
        );

        tx.send_replace(None);
        assert_eq!(
            follow_gripper_target(&rx.borrow()),
            None,
            "a consumed command leaves the gripper holding the move endpoint"
        );

        tx.send_replace(cmd(0.3));
        assert_eq!(
            follow_gripper_target(&rx.borrow()),
            target(0.3),
            "an opening after the move resumes following"
        );
    }

    // The gripper budget mirrors the arm servo's rollout: the commanded travel
    // at the opening rate, so a long move earns a long leash and a short one
    // stays tight.
    #[test]
    fn gripper_budget_is_the_commanded_travel_at_the_opening_rate() {
        const RATE: f64 = 3.0;
        assert_eq!(gripper_move_budget_s(0.0, 1.0, RATE), 1.0 / RATE);
        // Binary-exact travel (0.25) so the equality is exact.
        assert_eq!(gripper_move_budget_s(0.5, 0.75, 2.0), 0.125);
        // Direction of travel does not matter.
        assert_eq!(
            gripper_move_budget_s(0.8, 0.2, RATE),
            gripper_move_budget_s(0.2, 0.8, RATE)
        );
    }

    #[test]
    fn gripper_move_times_out_at_the_shared_factor_over_budget() {
        // A clamped full-travel move fails once it overruns 2x its budget,
        // exactly as the arm servo grades its rollout.
        let budget = MoveBudget::new(gripper_move_budget_s(0.0, 1.0, 3.0), 3.0);
        let nominal = budget.seconds();
        assert!(!budget.timed_out(nominal * MOTION_TIMEOUT_FACTOR - 0.01));
        assert!(budget.timed_out(nominal * MOTION_TIMEOUT_FACTOR + 0.01));
    }

    // The chase's landing arithmetic (`prev + (t - prev)`) can leave IEEE
    // rounding residue; the landing threshold absorbs it so a finished chase
    // cannot dangle one ulp short of terminal.
    #[test]
    fn landing_threshold_absorbs_chase_rounding_residue() {
        let target: f64 = 0.7;
        let mut governed: f64 = 0.13;
        for _ in 0..1000 {
            let step = (target - governed).clamp(-0.03, 0.03);
            governed += step;
        }
        assert!((governed - target).abs() <= GRIPPER_LANDED_FRAC);
    }

    #[tokio::test]
    async fn a_stale_tick_refuses_queued_goals_instead_of_parking_them() {
        use crate::planner::{JointReply, PlanConfig, ReadyOutcome, ReadyReply};
        use crate::servo::EeCaps;

        const TEST_PERIOD: Duration = Duration::from_millis(10);
        let version = openarm_description::HardwareVersion::V1;
        let model = crate::arm_model(version, openarm_description::Side::Left)
            .expect("build arm from the bundled URDF");
        let limits = model.limits();
        let mut planner = Planner::new(
            Side::Left,
            model,
            PlanConfig {
                cycle_period: TEST_PERIOD,
                smoothing: crate::servo::smoothing_for(TEST_PERIOD).unwrap(),
                max_joint_velocity_rad_s: [10.0; ARM_DOF],
                ee: EeCaps {
                    linear_m_s: 1.0,
                    angular_rad_s: 0.8,
                },
                limits,
            },
        );
        let held = [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];
        planner.commit(held);

        let (command, _command_rx) = watch::channel(None);
        let (gripper_command, _gripper_command_rx) = watch::channel(None);
        let (_measured_tx, measured) = watch::channel(None);
        let (_gripper_tx, gripper) = watch::channel(None);
        let (goal_tx, goals) = mpsc::channel(2);
        let (_gripper_goal_tx, gripper_goals) = mpsc::channel(1);
        let busy = Arc::new(AtomicBool::new(true)); // claimed at accept
        let mut channels = ArmChannels {
            command,
            gripper_command,
            measured,
            gripper,
            goals,
            busy: busy.clone(),
            gripper_goals,
            gripper_busy: Arc::new(AtomicBool::new(false)),
        };
        let (done_tx, mut done_rx) = mpsc::channel::<ReadyOutcome>(1);
        goal_tx
            .send(Goal::Joint {
                target: held,
                duration_s: 1.0,
                reply: JointReply::Ready(ReadyReply {
                    done_tx,
                    cancelled: Arc::new(AtomicBool::new(false)),
                }),
            })
            .await
            .expect("queue the goal");

        let tick = tick_arm(
            &mut channels,
            &mut planner,
            Admission::Stale,
            Instant::now(),
        )
        .await;

        let outcome = done_rx.recv().await.expect("a refused goal must report");
        assert!(!outcome.success);
        assert!(outcome.message.contains("stopped reporting"));
        assert!(!busy.load(Ordering::Acquire), "refusal releases the claim");
        assert_eq!(tick.candidate, planner.setpoint(), "a stale side holds");
        assert!(tick.streamed_hand.is_none(), "a stale side streams nothing");
    }
}
