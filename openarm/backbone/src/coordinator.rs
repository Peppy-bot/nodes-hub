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
//!
//! At the start of a tick, the loop also answers the requests of the
//! actions and services ([`CoordinatorRequest`]):
//! - a stop ends every move in flight on both sides;
//! - a plan check asks one arm's planner whether a Cartesian goal has a plan
//!   from where it stands;
//! - a posture goal that completes or ends cancelled asks for the grasp pose
//!   of each arm.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use peppygen::NodeRunner;
use peppygen::exposed_actions::limb_motion::move_gripper;
use peppylib::runtime::CancellationToken;
use srs_model::nalgebra::Isometry3;
use tokio::sync::{mpsc, oneshot, watch};
use tracing::{error, info, warn};

use control_core::filters::LowPassFilter;
use control_core::pacer::Pacer;

use crate::camera_mounts::MeasuredGrasps;
use crate::failure::{Failure, FailureKind};
use crate::governor::{GovState, Governor, Guard};
use crate::liveness::{Admission, Cadence, CadenceChange, Liveness};
use crate::planner::{self, BusyGuard, Goal, Measurement, PlanCheck, Planner, Unmeasured};
use crate::publish::{LimbStateSnapshot, Publishers};
use crate::side::{ArmPair, Side};
use crate::streams::{ArmState, GovernorConfig, GripperCommand, GripperState, LeaderActivity};
use crate::types::{ARM_DOF, ArmMoveRequest, JointVec};
use crate::upstream::{Upstream, UpstreamMode};
use control_core::chase::rate_limited;

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
    /// When the arm's and the gripper's leaders last streamed a well-formed
    /// command. A command is followed only while its leader is streaming.
    pub leader: LeaderActivity,
    pub gripper_leader: LeaderActivity,
    pub measured: watch::Receiver<Option<ArmState>>,
    pub gripper: watch::Receiver<Option<GripperState>>,
    pub goals: mpsc::Receiver<Goal>,
    pub busy: Arc<AtomicBool>,
    pub gripper_goals: mpsc::Receiver<GripperGoal>,
    pub gripper_busy: Arc<AtomicBool>,
    /// Whether the arm's and the gripper's followers deliver fresh state,
    /// written every tick for the goal handlers, which refuse a goal on a
    /// limb whose follower is stale or has not reported yet.
    pub fresh: Arc<AtomicBool>,
    pub gripper_fresh: Arc<AtomicBool>,
}

/// The coordinator's run parameters. A *leading node* that stops streaming is
/// held off by the leader window ([`crate::streams::LEADER_STREAMING_WINDOW`]):
/// its last command is dropped and the limb holds its governed setpoint. A
/// *follower* that stops delivering has its own limit, `stale_limit` (see
/// [`crate::liveness`]).
pub struct RunConfig {
    pub cycle_period: Duration,
    /// Silence a follower is allowed before its limb is frozen.
    pub stale_limit: Duration,
    /// The rate the followers were launched to report at; a follower running
    /// well under it is warned about ahead of the stale limit.
    pub follower_state_rate_hz: u32,
    /// Cutoff (Hz) for the low-pass on each published desired velocity. `dq` is a
    /// per-tick position difference scaled by `1/dt`, so it amplifies any setpoint noise
    /// by the control rate; filtering it keeps the arm's Kd term from buzzing on a noisy
    /// stream without touching the desired position.
    pub velocity_filter_cutoff_hz: f64,
    /// Which upstream kind this instance follows. Joints mode has no pose_link
    /// peer, so the upstream relay skips the per-tick FK and pose publish.
    pub upstream_mode: UpstreamMode,
}

/// The grasp pose of one arm in the robot frame, from the joints it
/// measured, or why it has no fresh measurement of them.
pub type MeasuredGrasp = Result<Isometry3<f64>, Unmeasured>;

/// What the limb_motion services and the posture moves ask of the loop,
/// answered on the reply channel each carries. A reply that the asker no
/// longer waits for is dropped.
pub enum CoordinatorRequest {
    /// End every planned move in flight on both sides, answering the names of
    /// the limbs whose move was ended.
    Stop {
        reason: String,
        reply: oneshot::Sender<Vec<String>>,
    },
    /// Whether a `move_arm` goal with these fields has a plan from where the
    /// arm stands, or the refusal the goal would get.
    CheckArmMove {
        request: ArmMoveRequest,
        reply: oneshot::Sender<Result<PlanCheck, Failure>>,
    },
    /// The grasp pose of each arm, measured when the loop serves the
    /// request: what a posture result gives when its goal completes or ends
    /// cancelled.
    MeasureGrasps {
        reply: oneshot::Sender<ArmPair<MeasuredGrasp>>,
    },
}

/// The reason every goal a stop ends is cancelled for: the stop, and the
/// reason of the stop request.
pub fn stop_reason(reason: &str) -> String {
    if reason.is_empty() {
        return "stopped".to_string();
    }
    format!("stopped: {reason}")
}

/// An accepted `move_gripper` goal handed to the coordinator, which executes it
/// through the same per-tick governing as everything else (the gripper analog of
/// [`Goal`] for the arms). The opening is the validated goal fraction; the
/// effort cap is validated non-negative, `None` when the goal carried no
/// preference.
pub struct GripperGoal {
    pub opening: f64,
    pub max_effort: Option<f64>,
    /// The longest the move waits, after the planned end of the commanded
    /// travel, for the jaws to stand still.
    pub settle_timeout: Duration,
    pub ctx: move_gripper::GoalContext,
}

impl GripperGoal {
    /// Fail unstarted with `failure`, with no result. The busy flag is the
    /// caller's concern.
    pub async fn fail(self, failure: Failure) {
        fail_gripper_goal(&self.ctx, failure).await;
    }

    /// End unstarted as cancelled by the stop service, for `reason`, which
    /// names the stop, reporting `reported_frac`. The busy flag is the
    /// caller's concern.
    pub async fn stop(self, reason: &str, reported_frac: f64) {
        if let Err(e) = self
            .ctx
            .complete_cancelled(reason, false, reported_frac, 0.0)
            .await
        {
            error!("move_gripper end: {e}");
        }
    }
}

/// Fail a gripper goal with `failure`, with no result.
async fn fail_gripper_goal(ctx: &move_gripper::GoalContext, failure: Failure) {
    if let Err(e) = ctx.fail(failure.kind.into(), failure.message).await {
        error!("move_gripper end: {e}");
    }
}

/// A backbone-executed gripper move in flight: the motion its terminal is
/// decided from, the effort cap it relays, and the goal it answers. The move
/// holds the side's single-flight slot until it ends, so a second gripper goal
/// on the side is refused until then; the busy guard releases the slot on any
/// exit.
struct GripperMove {
    motion: GripperMotion,
    max_effort: Option<f64>,
    ctx: move_gripper::GoalContext,
    _busy: BusyGuard,
}

/// What a gripper move's terminal is decided from (see
/// [`gripper_move_terminal`]): the target, when it was admitted, the planned
/// end of the commanded travel, the goal's settle timeout, and the run of
/// openings the jaws are judged still on. The commanded opening ramps to the
/// target through the governor at the opening rate the operator sets, re-read
/// every tick, and holds it while the jaws follow.
#[derive(Clone, Copy, Debug)]
struct GripperMotion {
    target_frac: f64,
    admitted: Instant,
    /// The first tick of the commanded travel's current rest, on the target
    /// or held by the collision guard. `None` while it travels.
    planned_end: Option<Instant>,
    settle_timeout: Duration,
    /// The latest run of openings the follower delivered since admission
    /// that lie within [`GRIPPER_STILL_BAND`] of one another. `None` before
    /// the first.
    still: Option<StillRun>,
}

/// A run of measured openings that all lie within [`GRIPPER_STILL_BAND`] of
/// one another: the lowest and the highest of them, and when the first was
/// measured.
#[derive(Clone, Copy, Debug)]
struct StillRun {
    low: f64,
    high: f64,
    since: Instant,
}

impl StillRun {
    fn starting(frac: f64, now: Instant) -> Self {
        Self {
            low: frac,
            high: frac,
            since: now,
        }
    }

    /// This run with `frac` measured at `now` added, or a new run from it
    /// when it lies farther than the band from an opening of this one.
    fn extended(self, frac: f64, now: Instant) -> Self {
        let (low, high) = (self.low.min(frac), self.high.max(frac));
        if high - low > GRIPPER_STILL_BAND {
            return Self::starting(frac, now);
        }
        Self { low, high, ..self }
    }
}

/// The jaws stand still once every opening measured over
/// [`GRIPPER_STILL_WINDOW`] lies within this fraction of the jaw travel of
/// the others: the band the limb_motion contract states.
const GRIPPER_STILL_BAND: f64 = 0.01;

/// How long the measured openings must stay within [`GRIPPER_STILL_BAND`] of
/// one another for the jaws to stand still, as the limb_motion contract
/// states.
const GRIPPER_STILL_WINDOW: Duration = Duration::from_millis(250);

impl GripperMotion {
    /// A move to `target_frac` admitted at `now`.
    fn new(target_frac: f64, settle_timeout: Duration, now: Instant) -> Self {
        Self {
            target_frac,
            admitted: now,
            planned_end: None,
            settle_timeout,
            still: None,
        }
    }

    /// Note the opening the governor published last tick, and return the
    /// planned end while the commanded travel rests: on each tick after
    /// admission on which the commanded opening stands on the target, or the
    /// collision guard held it where it stood. The planned
    /// end is the first tick of the current rest. The travel moving on clears
    /// it, with the openings judged during the rest, so a hold the guard lets
    /// go of leaves nothing behind. Whatever the opening rate, and however the
    /// operator retunes it during the move, the travel rests only there.
    fn travel_end(&mut self, commanded: CommandedOpening, now: Instant) -> Option<Instant> {
        let rests = now > self.admitted && (commanded.frac == self.target_frac || commanded.held);
        let rest_start = self.planned_end.unwrap_or(now);
        self.planned_end = rests.then_some(rest_start);
        if !rests {
            self.still = None;
        }
        self.planned_end
    }

    /// Add one opening the follower delivered this tick to the run, and
    /// whether the jaws stand still: the openings of the run lie within
    /// [`GRIPPER_STILL_BAND`] of one another, and the first of them is
    /// [`GRIPPER_STILL_WINDOW`] old.
    fn stands_still(&mut self, frac: f64, now: Instant) -> bool {
        let run = self.still.map_or_else(
            || StillRun::starting(frac, now),
            |run| run.extended(frac, now),
        );
        self.still = Some(run);
        now.duration_since(run.since) >= GRIPPER_STILL_WINDOW
    }
}

/// A gripper follower's measured opening as one tick sees it: the latest
/// opening, and whether the follower delivered it since the last tick.
#[derive(Clone, Copy, Debug)]
struct MeasuredOpening {
    frac: f64,
    fresh: bool,
}

/// A gripper's commanded opening as one tick sees it: the opening the
/// governor published last tick, and whether its collision guard held the
/// opening where it stood then.
#[derive(Clone, Copy, Debug)]
struct CommandedOpening {
    frac: f64,
    held: bool,
}

/// The largest part of the chase's step the governor may let through for the
/// guard to count as holding the opening: a hold leaves float dust, orders of
/// magnitude under any throttle, which still moves the opening and is travel.
const GUARD_HOLD_STEP_FRAC: f64 = 0.01;

impl CommandedOpening {
    /// An idle side's opening, re-anchored on the measured gripper.
    fn idle(frac: f64) -> Self {
        Self { frac, held: false }
    }

    /// The opening the governor published, `governed`, from the chase's step
    /// from `prev` to `chased`: held when the governor let through under
    /// [`GUARD_HOLD_STEP_FRAC`] of that step.
    fn published(governed: f64, prev: f64, chased: f64) -> Self {
        Self {
            frac: governed,
            held: (governed - prev).abs() < GUARD_HOLD_STEP_FRAC * (chased - prev).abs(),
        }
    }
}

/// How a gripper move ended.
#[derive(Debug, PartialEq)]
enum GripperTerminal {
    /// The measured opening stood still, on the target or short of it where
    /// an object or the effort cap stopped the jaws.
    Settled,
    /// The settle timeout passed before the jaws stood still; the move
    /// completes where they stand, with `settled` false.
    SettleTimedOut,
    /// The follower stopped reporting; the move fails with
    /// `resource_unavailable`.
    FollowerStale,
    /// The goal was cancelled, for this reason: by its caller, or by the
    /// stop service, whose reason names the stop.
    Cancelled(String),
}

/// Run the coordination loop. Holds the governor and both planners. Runs until
/// the node's cancellation token fires; returns `Err` if a publisher cannot be
/// declared at bringup. Any return takes the node down (the supervisor in `main`
/// treats it as fatal).
#[expect(
    clippy::too_many_arguments,
    reason = "one input per producer the loop reads, wired once in node.rs"
)]
pub async fn run(
    runner: Arc<NodeRunner>,
    mut governor: Governor,
    mut planners: ArmPair<Planner>,
    mut channels: ArmPair<ArmChannels>,
    mut requests: mpsc::Receiver<CoordinatorRequest>,
    governor_config: watch::Receiver<GovernorConfig>,
    measured_grasps: watch::Sender<Option<MeasuredGrasps>>,
    config: RunConfig,
    token: CancellationToken,
) -> peppygen::Result<()> {
    let RunConfig {
        cycle_period,
        stale_limit,
        follower_state_rate_hz,
        velocity_filter_cutoff_hz,
        upstream_mode,
    } = config;
    let publishers = Publishers::declare(&runner, measured_grasps).await?;

    // Hold each arm's real pose, not a neutral zero: wait for the first measured
    // state from both arms and seed the held setpoints there before publishing.
    if seed_all(&mut channels, &mut planners, &mut requests)
        .await
        .is_err()
    {
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
    // Track the opening the governor published last per gripper, and whether
    // it changed the chase's step: the governed configuration's `prev`, and
    // what a gripper move reads its commanded travel from. Anchored on the
    // measured grippers (here and whenever a side idles) so governing always
    // ramps from where the fingers really are; the opening rate is read from
    // the governor, its single owner.
    let mut commanded_grippers =
        ArmPair::from_fn(|side| CommandedOpening::idle(gripper_fraction(&channels[side].gripper)));
    // In-flight backbone-executed gripper moves, one single-flight slot per side.
    let mut gripper_moves: ArmPair<Option<GripperMove>> = ArmPair::from_fn(|_| None);

    let dt = cycle_period.as_secs_f64();
    // One low-pass per joint per arm, smoothing the published desired velocity. `main`
    // validates `0 < cutoff < Nyquist` at bringup, a strict superset of what `from_cutoff`
    // rejects, so construction cannot fail: build one filter and copy it per joint (the
    // same bringup-invariant pattern as `Pacer::new(...).expect(...)`).
    let filter = LowPassFilter::from_cutoff(velocity_filter_cutoff_hz, dt)
        .expect("velocity_filter_cutoff_hz is bringup-validated in (0, Nyquist)");
    let mut dq_filters = ArmPair::from_fn(|_| [filter; ARM_DOF]);
    // The proximity readout is for human eyes, so publish it at ~20 Hz rather than
    // the control rate: one extra distance query every `readout_every` ticks.
    let readout_every = (0.05 / dt).round().max(1.0) as u64;
    let mut tick: u64 = 0;
    let mut pacer = Pacer::new(cycle_period).expect("control_rate_hz is asserted > 0 at startup");
    // Both arms are seeded from their first measurement above, which is the
    // anchor a recovery would re-establish, so they start live.
    let (mut arm_liveness, mut gripper_liveness, mut arm_cadence, mut gripper_cadence) = {
        let seeded = Instant::now();
        let cadence = || Cadence::new(follower_state_rate_hz, seeded);
        (
            ArmPair::from_fn(|_| Liveness::seeded(seeded)),
            ArmPair::from_fn(|_| Liveness::seeded(seeded)),
            ArmPair::from_fn(|_| cadence()),
            ArmPair::from_fn(|_| cadence()),
        )
    };
    // The grippers chase their target at the gripper rate exactly as the planner
    // velocity-limits the arm candidates; an idle side chases nowhere.
    // The chase lands on the target exactly, so a gripper move sees its
    // commanded travel end by equality.
    let chase_gripper = |prev_frac: f64, target: Option<GripperTarget>, rate: f64| -> f64 {
        let target_frac = target.map_or(prev_frac, |t| t.frac);
        if (target_frac - prev_frac).abs() <= rate * dt {
            return target_frac;
        }
        rate_limited(prev_frac, target_frac, rate, dt)
    };
    loop {
        consume_streams_of_busy_sides(&channels);
        apply_controls(&mut governor, &mut planners, *governor_config.borrow());
        // Re-read every tick: the operator retunes this live, and the chase and
        // the governor's own clamp agree on it within the tick.
        let gripper_rate = governor.max_gripper_rate_frac_s();
        let now = Instant::now();

        let arm_admission = admit_arms(
            &mut arm_liveness,
            &mut arm_cadence,
            &channels,
            now,
            stale_limit,
        );
        let gripper_delivered = take_gripper_deliveries(&mut channels);
        let gripper_admission = admit_grippers(
            &mut gripper_liveness,
            &mut gripper_cadence,
            gripper_delivered,
            now,
            stale_limit,
        );
        for side in Side::BOTH {
            let arm = &channels[side];
            let fresh = |admission: Admission| admission != Admission::Stale;
            arm.fresh
                .store(fresh(arm_admission[side]), Ordering::Release);
            arm.gripper_fresh
                .store(fresh(gripper_admission[side]), Ordering::Release);
        }
        // The services are answered ahead of the tick's advance: a stop ends
        // every move before the planners would step it once more.
        while let Ok(request) = requests.try_recv() {
            serve_request(
                request,
                &mut channels,
                &mut planners,
                arm_admission,
                &mut gripper_moves,
                &gripper_fraction,
                now,
            )
            .await;
        }
        let arm_ticks = advance_arms(&mut channels, &mut planners, arm_admission, now).await;
        let (arm_candidate, hands) = arm_ticks
            .map(|_, tick| (tick.candidate, tick.streamed_hand))
            .unzip();
        let measured_grippers = ArmPair::from_fn(|side| gripper_fraction(&channels[side].gripper));
        service_gripper_moves(
            &mut gripper_moves,
            &mut channels,
            commanded_grippers,
            ArmPair::from_fn(|side| MeasuredOpening {
                frac: measured_grippers[side],
                fresh: gripper_delivered[side],
            }),
            gripper_admission,
            now,
        )
        .await;

        // Resolve each gripper's target for this tick: an in-flight move owns the
        // opening; otherwise the latest leader command drives it; otherwise the
        // side idles (never commanded, or unpaired), silent on the wire with the
        // governed opening re-anchored on the measured grippers.
        let targets =
            ArmPair::from_fn(|side| gripper_target(&gripper_moves[side], &channels[side]));
        for side in Side::BOTH {
            if targets[side].is_none() {
                commanded_grippers[side] = CommandedOpening::idle(measured_grippers[side]);
            }
        }

        // One governed configuration: the last published setpoints and grippers
        // as `prev`, the rate-limited chases as the candidate. The governor
        // throttles, holds, scans and monitors everything through one barrier.
        let prev = GovState::new(
            ArmPair::from_fn(|side| planners[side].setpoint()),
            ArmPair::from_fn(|side| commanded_grippers[side].frac),
        );
        let cand = GovState::new(
            arm_candidate,
            ArmPair::from_fn(|side| {
                chase_gripper(prev.grippers[side], targets[side], gripper_rate)
            }),
        );
        let measured = measured_config(&channels, &prev, measured_grippers);
        let governed = governor.govern(&prev, &cand, &measured, &hands, dt);
        commanded_grippers = ArmPair::from_fn(|side| {
            CommandedOpening::published(
                governed.grippers[side],
                prev.grippers[side],
                cand.grippers[side],
            )
        });

        // Publish one governed setpoint per arm on its pairing slot; the slot
        // scopes the stream to its paired arm, so the message names no side.
        for side in Side::BOTH {
            let governed_q = governed.arms[side];
            // Desired velocity is the per-tick position delta; low-pass it per joint so a
            // noisy stream does not drive the arm's Kd term into buzz. The published
            // position (`governed_q`) is untouched, so tracking is unaffected.
            let dq = filtered_velocity(&mut dq_filters[side], &governed_q, &prev.arms[side], dt);
            // Every side commits (identity for a stale one, whose candidate was
            // the frozen setpoint), but a stale side stays silent so its
            // follower holds its own last setpoint rather than tracking one the
            // backbone can no longer vouch for.
            planners[side].commit(governed_q);
            if arm_admission[side] == Admission::Stale {
                continue;
            }
            publishers.arm_setpoints[side].send(&governed_q, &dq).await;
        }

        // Publish each active side's governed opening fraction on its pairing
        // slot (the slot scopes the stream to its paired gripper, so the message
        // names no side); an idle side stays silent and its gripper holds its
        // opening.
        for side in Side::BOTH {
            if let Some(target) = targets[side] {
                publishers.gripper_setpoints[side]
                    .send(commanded_grippers[side].frac, target.max_effort)
                    .await;
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

/// Answer one request:
/// - A stop ends the moves in flight on both sides and the goals admitted
///   but not started. The planner ends an arm's move; the gripper's
///   terminal ends a gripper's move.
/// - Each goal the stop ends is cancelled, for a reason that names the
///   stop. The answer names the limbs whose move was in flight. The
///   setpoints hold where they were last governed.
/// - A check asks the named arm's planner and moves nothing.
/// - A grasp request answers the grasp pose of each arm.
///
/// The results and the grasp poses come from each arm's
/// [`seeded_measurement`] under `arm_admission`.
async fn serve_request(
    request: CoordinatorRequest,
    channels: &mut ArmPair<ArmChannels>,
    planners: &mut ArmPair<Planner>,
    arm_admission: ArmPair<Admission>,
    gripper_moves: &mut ArmPair<Option<GripperMove>>,
    gripper_fraction: &impl Fn(&watch::Receiver<Option<GripperState>>) -> f64,
    now: Instant,
) {
    match request {
        CoordinatorRequest::Stop { reason, reply } => {
            let reason = stop_reason(&reason);
            info!("stop requested: {reason}");
            let mut stopped = Vec::new();
            for side in Side::BOTH {
                let arm = &mut channels[side];
                let planner = &mut planners[side];
                let measured = seeded_measurement(arm, arm_admission[side]);
                if planner.stop_active(&reason, measured, now).await {
                    stopped.push(side.arm_name().to_string());
                }
                while let Ok(goal) = arm.goals.try_recv() {
                    let _release = BusyGuard(arm.busy.clone());
                    goal.stop(&reason, measured, planner).await;
                }
                let measured_frac = gripper_fraction(&arm.gripper);
                if let Some(m) = gripper_moves[side].take() {
                    let elapsed_s = now.duration_since(m.motion.admitted).as_secs_f64();
                    end_gripper_move(
                        m,
                        GripperTerminal::Cancelled(reason.clone()),
                        measured_frac,
                        elapsed_s,
                    )
                    .await;
                    stopped.push(side.gripper_name().to_string());
                }
                while let Ok(goal) = arm.gripper_goals.try_recv() {
                    let _release = BusyGuard(arm.gripper_busy.clone());
                    goal.stop(&reason, measured_frac).await;
                }
            }
            let _ = reply.send(stopped);
        }
        CoordinatorRequest::CheckArmMove { request, reply } => {
            let _ = reply.send(planners[request.side].check_cartesian(
                &request.target,
                request.tolerance,
                request.duration,
            ));
        }
        CoordinatorRequest::MeasureGrasps { reply } => {
            let _ = reply.send(measured_grasps(channels, planners, arm_admission));
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
    for side in Side::BOTH {
        let ch = &channels[side];
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
    for side in Side::BOTH {
        planners[side].set_max_ee_velocity(cfg.max_ee_velocity_m_s);
    }
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
    cadence: &mut ArmPair<Cadence>,
    channels: &ArmPair<ArmChannels>,
    now: Instant,
    stale_limit: Duration,
) -> ArmPair<Admission> {
    ArmPair::from_fn(|side| {
        let delivered = channels[side].measured.has_changed().unwrap_or(false);
        report_cadence(side, "arm", cadence[side].observe(delivered, now));
        liveness[side].admit(delivered, now, stale_limit)
    })
}

/// Log the delivery cadence of the follower of `side`'s `limb` crossing the
/// threshold under its declared rate: the signal that the launcher's
/// follower_state_rate_hz is a promise the follower is not keeping, ahead of
/// the stale limit freezing the limb.
fn report_cadence(side: Side, limb: &str, change: Option<CadenceChange>) {
    let side = side.label();
    match change {
        Some(CadenceChange::Degraded {
            delivered,
            expected,
        }) => warn!(
            "{side} {limb} follower delivered {delivered} states in the last second, \
             under its declared {expected}"
        ),
        Some(CadenceChange::Recovered {
            delivered,
            expected,
        }) => {
            info!("{side} {limb} follower delivery back to {delivered} of its declared {expected}")
        }
        None => {}
    }
}

/// Whether each gripper follower delivered an opening since the last tick.
fn take_gripper_deliveries(channels: &mut ArmPair<ArmChannels>) -> ArmPair<bool> {
    // Read the flag, then mark the watch seen, so the next tick asks about that
    // tick's delivery rather than every delivery since the loop began. Nothing
    // else updates this watch; the other readers only borrow.
    let take = |gripper: &mut watch::Receiver<Option<GripperState>>| {
        let delivered = gripper.has_changed().unwrap_or(false);
        let _ = gripper.borrow_and_update();
        delivered
    };
    ArmPair::from_fn(|side| take(&mut channels[side].gripper))
}

/// Judge each gripper follower's delivery, the opening analog of
/// [`admit_arms`] over each gripper's own pairing.
///
/// Gates the upstream relay and a gripper move: a gripper that
/// has stopped delivering must not have its last aperture republished under a
/// fresh timestamp, which would show the leading node a live-looking
/// back-channel, and must not pass for a gripper standing still. The governed
/// opening still streams down, because a held gripper holds where the operator
/// put it rather than drifting away unseen the way an uncommanded arm does.
fn admit_grippers(
    liveness: &mut ArmPair<Liveness>,
    cadence: &mut ArmPair<Cadence>,
    delivered: ArmPair<bool>,
    now: Instant,
    stale_limit: Duration,
) -> ArmPair<Admission> {
    ArmPair::from_fn(|side| {
        report_cadence(side, "gripper", cadence[side].observe(delivered[side], now));
        liveness[side].admit(delivered[side], now, stale_limit)
    })
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
/// into a free side, and end an in-flight move once its jaws stand still or
/// its settle timeout passes, on cancellation, or on a follower gone stale.
/// `commanded` is the opening the governor published last tick and whether
/// the guard held it: what a move reads its commanded travel from.
async fn service_gripper_moves(
    moves: &mut ArmPair<Option<GripperMove>>,
    channels: &mut ArmPair<ArmChannels>,
    commanded: ArmPair<CommandedOpening>,
    measured: ArmPair<MeasuredOpening>,
    admission: ArmPair<Admission>,
    now: Instant,
) {
    for side in Side::BOTH {
        service_gripper_move(
            &mut moves[side],
            &mut channels[side],
            commanded[side],
            measured[side],
            admission[side],
            now,
        )
        .await;
    }
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
    GovState::new(
        ArmPair::from_fn(|side| {
            channels[side]
                .measured
                .borrow()
                .as_ref()
                .map_or(prev.arms[side], |m| m.positions)
        }),
        grippers,
    )
}

/// A stale side reads as nothing to report, whatever the measurement is.
fn live<T>(admission: Admission, read: impl FnOnce() -> Option<T>) -> Option<T> {
    (admission != Admission::Stale).then(read).flatten()
}

/// Every limb's live measurement, as [`live`] reads it: the joint state of
/// each arm and the gripper state of each gripper.
fn live_limbs(
    channels: &ArmPair<ArmChannels>,
    arm_admission: ArmPair<Admission>,
    gripper_admission: ArmPair<Admission>,
) -> (ArmPair<Option<ArmState>>, ArmPair<Option<GripperState>>) {
    (
        ArmPair::from_fn(|side| live(arm_admission[side], || *channels[side].measured.borrow())),
        ArmPair::from_fn(|side| live(gripper_admission[side], || *channels[side].gripper.borrow())),
    )
}

/// The joints an arm's move results report after the seed: its latest
/// measurement, or [`Unmeasured::Stale`] for a stale side, as [`live`]
/// reads it. The seed gated on every arm's first measurement, so a live
/// side always holds one.
fn seeded_measurement(channels: &ArmChannels, admission: Admission) -> Measurement {
    if admission == Admission::Stale {
        return Err(Unmeasured::Stale);
    }
    Ok(channels
        .measured
        .borrow()
        .expect("seed gated on the first state")
        .positions)
}

/// The grasp pose of each arm after the seed, at the joints of its
/// [`seeded_measurement`] under `arm_admission`, or why it has none.
fn measured_grasps(
    channels: &ArmPair<ArmChannels>,
    planners: &mut ArmPair<Planner>,
    arm_admission: ArmPair<Admission>,
) -> ArmPair<MeasuredGrasp> {
    ArmPair::from_fn(|side| {
        seeded_measurement(&channels[side], arm_admission[side])
            .map(|q| planners[side].world_ee_pose(&q))
    })
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
    let (arms, grippers) = live_limbs(channels, arm_admission, gripper_admission);
    let (arms, grippers) = (arms.transpose()?, grippers.transpose()?);
    Some(LimbStateSnapshot {
        joints: arms.map(|_, arm| arm.positions),
        poses: arms.map(|side, arm| planners[side].world_ee_pose(&arm.positions)),
        openings: grippers.map(|_, gripper| gripper.fraction),
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
    let (arms, grippers) = live_limbs(channels, arm_admission, gripper_admission);
    // In pose mode the arm back-channel also carries the same measurement as
    // the end-effector pose, FK'd here; joints mode has no pose_link peer, so
    // that work is skipped.
    for side in Side::BOTH {
        if let Some(m) = arms[side] {
            publishers.arm_states[side]
                .send(&m.positions, &m.velocities)
                .await;
            if upstream_mode == UpstreamMode::Pose {
                let pose = planners[side].world_ee_pose(&m.positions);
                publishers.arm_pose_states[side].send(&pose).await;
            }
        }
    }
    for side in Side::BOTH {
        if let Some(g) = grippers[side] {
            publishers.gripper_states[side].send(&g).await;
        }
    }
}

/// All senders on the measured-state channel dropped (its only producer is the
/// state listener task), so no measurement will ever arrive: seeding is abandoned.
struct Shutdown;

/// The failure of every goal that reaches the coordinator while the
/// followers are still silent, and the refusal of a plan check then.
fn seed_failure() -> Failure {
    Failure::new(
        FailureKind::ResourceUnavailable,
        "the follower has not reported its first state yet",
    )
}

/// The failure of a goal that reaches a limb whose follower stream has gone
/// stale, and of a move whose follower goes stale while it runs.
fn stale_failure() -> Failure {
    Failure::new(
        FailureKind::ResourceUnavailable,
        "the follower stopped reporting",
    )
}

/// Wait for both arms' first measured states and both grippers' first
/// gripper fractions, then seed each planner's held setpoint from its measured pose
/// (clamped into the joint limits). One wait over all four goal queues: a
/// goal accepted for EITHER side before its follower reports fails with
/// [`seed_failure`], its busy claim released, so a silent side cannot strand
/// the other side's goals. Warns periodically while a stream stays silent; `Err(Shutdown)` if
/// a channel closes first.
async fn seed_all(
    channels: &mut ArmPair<ArmChannels>,
    planners: &mut ArmPair<Planner>,
    requests: &mut mpsc::Receiver<CoordinatorRequest>,
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
                let _release = BusyGuard(channels.left.busy.clone());
                goal.fail(seed_failure(), &planners.left).await;
            }
            Some(goal) = channels.right.goals.recv() => {
                let _release = BusyGuard(channels.right.busy.clone());
                goal.fail(seed_failure(), &planners.right).await;
            }
            Some(goal) = channels.left.gripper_goals.recv() => {
                let _release = BusyGuard(channels.left.gripper_busy.clone());
                goal.fail(seed_failure()).await;
            }
            Some(goal) = channels.right.gripper_goals.recv() => {
                let _release = BusyGuard(channels.right.gripper_busy.clone());
                goal.fail(seed_failure()).await;
            }
            // Nothing moves before the seed, so a stop stops nothing and a
            // check has no held pose to plan from. The seed judges no
            // liveness, so no measurement is known to be fresh: a grasp
            // request gets no pose for either arm.
            Some(request) = requests.recv() => match request {
                CoordinatorRequest::Stop { reply, .. } => {
                    let _ = reply.send(Vec::new());
                }
                CoordinatorRequest::CheckArmMove { reply, .. } => {
                    let _ = reply.send(Err(seed_failure()));
                }
                CoordinatorRequest::MeasureGrasps { reply } => {
                    let unseeded = Err(Unmeasured::NotYet);
                    let _ = reply.send(ArmPair::from_fn(|_| unseeded));
                }
            }
        }
    }
    for side in Side::BOTH {
        let q0 = channels[side]
            .measured
            .borrow()
            .expect("gated on first state")
            .positions;
        planners[side].seed_from_measured(q0);
    }
    Ok(())
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
/// measured pose, feed the latest leader command, and admit any pending move
/// goal. Runs only after the seed, which gated on the arm's first measurement.
async fn tick_arm(
    channels: &mut ArmChannels,
    planner: &mut Planner,
    admission: Admission,
    now: Instant,
) -> planner::Tick {
    // A stale limb holds exactly where it was last governed. Advancing the
    // planner would walk the setpoint away from an arm nobody can see, and a
    // held setpoint needs no hand basis: there is no streamed motion to cap.
    // A move in that darkness can neither progress nor be verified, so it and
    // any queued goal fail here with no result, freeing the claim each holds:
    // a ready share that kept its claim would wedge the ready action for good.
    if admission == Admission::Stale {
        planner.fail_active(stale_failure()).await;
        while let Ok(goal) = channels.goals.try_recv() {
            let _release = BusyGuard(channels.busy.clone());
            goal.fail(stale_failure(), planner).await;
        }
        return planner::Tick {
            candidate: planner.setpoint(),
            streamed_hand: None,
        };
    }
    let measured_q = channels
        .measured
        .borrow_and_update()
        .expect("seed gated on the first state")
        .positions;
    // First delivery after a gap: the held setpoint is now fiction, so adopt
    // the measured pose before advancing. The per-joint velocity limit in the
    // chase then walks it back to the operator's command instead of the
    // follower stepping the whole divergence in one tick.
    if admission == Admission::Reanchor {
        warn!("arm follower stream recovered; re-anchoring on the measured pose");
        planner.seed_from_measured(measured_q);
    }
    // Read out so no watch borrow guard is held across the await below. A
    // leader that has stopped streaming leaves nothing to follow: its last
    // command is dropped and the limb holds its setpoint.
    let command = channels
        .leader
        .streaming()
        .then(|| *channels.command.borrow())
        .flatten();
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

/// Decide whether a gripper move ends this tick, and how. No I/O and no
/// clock: everything the decision reads is passed in, and the only state it
/// writes is the move's own planned end and still run.
///
/// A cancel, `cancel` giving its reason, ends the move ahead of any other
/// verdict of the same tick, and a follower gone stale fails it: the move
/// can neither progress nor be verified. Nothing else is judged while the commanded travel runs
/// ([`GripperMotion::travel_end`], fed `commanded`, the opening the governor
/// published last tick). From the planned end on the move completes as soon
/// as the openings delivered since it stand still, and at the latest once its
/// settle timeout has passed, where the jaws stand: a gripper move never fails
/// on its settle, since a grasp is expected to stop short of the target. An
/// opening that merely stands since an earlier tick is no evidence that the
/// jaws stopped.
fn gripper_move_terminal(
    motion: &mut GripperMotion,
    commanded: CommandedOpening,
    measured: MeasuredOpening,
    admission: Admission,
    cancel: Option<String>,
    now: Instant,
) -> Option<GripperTerminal> {
    if let Some(reason) = cancel {
        return Some(GripperTerminal::Cancelled(reason));
    }
    if admission == Admission::Stale {
        return Some(GripperTerminal::FollowerStale);
    }
    let planned_end = motion.travel_end(commanded, now)?;
    if measured.fresh && motion.stands_still(measured.frac, now) {
        return Some(GripperTerminal::Settled);
    }
    let since_end = now.duration_since(planned_end);
    (since_end >= motion.settle_timeout).then_some(GripperTerminal::SettleTimedOut)
}

/// Admit a queued gripper goal into a free side, failing it while the
/// follower is stale or another gripper move is in flight, and end an
/// in-flight move on the terminal [`gripper_move_terminal`] decides.
/// `commanded` is the opening the governor published last tick and whether
/// the guard held it. A terminal with a result reports the opening measured
/// that tick as `final_opening` and the time since admission as
/// `action_time`. The busy slot releases with the move on every path.
async fn service_gripper_move(
    mv: &mut Option<GripperMove>,
    channels: &mut ArmChannels,
    commanded: CommandedOpening,
    measured: MeasuredOpening,
    admission: Admission,
    now: Instant,
) {
    // Drain fully: every queued goal is answered, never left parked.
    while let Ok(goal) = channels.gripper_goals.try_recv() {
        if admission == Admission::Stale {
            goal.fail(stale_failure()).await;
        } else if mv.is_none() {
            *mv = Some(GripperMove {
                motion: GripperMotion::new(goal.opening, goal.settle_timeout, now),
                max_effort: goal.max_effort,
                ctx: goal.ctx,
                _busy: BusyGuard(channels.gripper_busy.clone()),
            });
        } else {
            let failure = Failure::new(
                FailureKind::PreconditionFailed,
                "another gripper move is in flight",
            );
            goal.fail(failure).await;
        }
    }
    let Some(m) = mv.as_mut() else { return };
    let cancel = m.ctx.cancel_cause().map(|cause| cause.reason().to_string());
    let Some(terminal) =
        gripper_move_terminal(&mut m.motion, commanded, measured, admission, cancel, now)
    else {
        return;
    };
    let m = mv.take().expect("in-flight move checked above");
    let elapsed_s = now.duration_since(m.motion.admitted).as_secs_f64();
    end_gripper_move(m, terminal, measured.frac, elapsed_s).await;
}

/// End a gripper move's goal on `terminal`. A goal that completes or ends
/// cancelled reports `measured_frac` as `final_opening` and `elapsed_s` as
/// `action_time`, and `settled` true only when the jaws stood still; a
/// follower gone stale fails the goal with no result. The busy slot releases
/// with the move.
async fn end_gripper_move(
    m: GripperMove,
    terminal: GripperTerminal,
    measured_frac: f64,
    elapsed_s: f64,
) {
    let ended = match terminal {
        GripperTerminal::Settled => m.ctx.complete(true, measured_frac, elapsed_s).await,
        GripperTerminal::SettleTimedOut => m.ctx.complete(false, measured_frac, elapsed_s).await,
        GripperTerminal::FollowerStale => {
            return fail_gripper_goal(&m.ctx, stale_failure()).await;
        }
        GripperTerminal::Cancelled(reason) => {
            m.ctx
                .complete_cancelled(reason, false, measured_frac, elapsed_s)
                .await
        }
    };
    if let Err(e) = ended {
        error!("move_gripper end: {e}");
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
/// move owns it, through its chase and its settle alike, so a settling gripper
/// keeps being commanded to the move's target; otherwise the leading node's
/// streamed command drives it; otherwise `None` (idle: before any command, or
/// on an unpaired side).
fn gripper_target(mv: &Option<GripperMove>, channels: &ArmChannels) -> Option<GripperTarget> {
    if let Some(m) = mv {
        return Some(GripperTarget {
            frac: m.motion.target_frac,
            max_effort: m.max_effort,
        });
    }
    channels
        .gripper_leader
        .streaming()
        .then(|| follow_gripper_target(&channels.gripper_command.borrow().clone()))
        .flatten()
}

/// Resolve the streamed target: the latest leader command with its opening
/// clamped into `[0, 1]`, or `None` when none has arrived. The stream is
/// paired to one producer, so there is nothing to arbitrate; the caller hands
/// a stopped producer's command in as `None`, which leaves the gripper holding
/// its setpoint.
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
    use crate::planner::Outcome;

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

    // The gripper move's terminal decision. Every instant is handed to the
    // decision as an offset from one anchor, so no test waits on a clock.

    /// The target of the moves below: a half close from fully open.
    const MOVE_TARGET: f64 = 0.5;
    /// The settle timeout of the moves below.
    const MOVE_TIMEOUT: Duration = Duration::from_secs(2);

    fn ms(millis: u64) -> Duration {
        Duration::from_millis(millis)
    }

    fn fresh(frac: f64) -> MeasuredOpening {
        MeasuredOpening { frac, fresh: true }
    }

    fn stale_cached(frac: f64) -> MeasuredOpening {
        MeasuredOpening { frac, fresh: false }
    }

    /// The governor published `frac`, the chase's own step.
    fn free(frac: f64) -> CommandedOpening {
        CommandedOpening { frac, held: false }
    }

    /// The governor held the opening at `frac` against the chase's step.
    fn held(frac: f64) -> CommandedOpening {
        CommandedOpening { frac, held: true }
    }

    #[test]
    fn the_guard_holds_an_opening_it_lets_creep_and_throttles_one_it_lets_travel() {
        let (prev, chased) = (0.94, 0.88);
        assert!(!CommandedOpening::published(chased, prev, chased).held);
        assert!(
            !CommandedOpening::published(0.91, prev, chased).held,
            "a throttle is travel"
        );
        assert!(CommandedOpening::published(prev, prev, chased).held);
        assert!(
            CommandedOpening::published(prev - 4e-9, prev, chased).held,
            "a creep is a hold"
        );
        assert!(
            !CommandedOpening::published(prev, prev, prev).held,
            "no step asked for"
        );
    }

    /// A move to [`MOVE_TARGET`] whose commanded travel came to rest at
    /// `planned_end`.
    fn motion_ending_at(planned_end: Instant) -> GripperMotion {
        GripperMotion {
            planned_end: Some(planned_end),
            ..GripperMotion::new(MOVE_TARGET, MOVE_TIMEOUT, planned_end - ms(100))
        }
    }

    /// The verdict on `motion` at `now` with the command on the target, the
    /// follower live and the goal not cancelled.
    fn judge(
        motion: &mut GripperMotion,
        measured: MeasuredOpening,
        now: Instant,
    ) -> Option<GripperTerminal> {
        gripper_move_terminal(
            motion,
            free(MOVE_TARGET),
            measured,
            Admission::Live,
            None,
            now,
        )
    }

    #[test]
    fn the_commanded_travel_ends_the_tick_the_commanded_opening_lands() {
        // Admitted fully open; the operator slows the rate mid-move, so the
        // steps shrink, and the travel ends only when the target is reached.
        let start = Instant::now();
        let mut motion = GripperMotion::new(MOVE_TARGET, MOVE_TIMEOUT, start);
        assert_eq!(
            motion.travel_end(free(1.0), start),
            None,
            "the admission tick"
        );
        for (tick, commanded) in [0.94, 0.88, 0.87, 0.86, 0.85].into_iter().enumerate() {
            let now = start + ms(10) * (tick as u32 + 1);
            assert_eq!(
                motion.travel_end(free(commanded), now),
                None,
                "still travelling at {commanded}"
            );
        }
        let landed = start + ms(60);
        assert_eq!(motion.travel_end(free(MOVE_TARGET), landed), Some(landed));
        assert_eq!(
            motion.travel_end(free(MOVE_TARGET), landed + ms(10)),
            Some(landed)
        );
    }

    #[test]
    fn a_commanded_opening_the_guard_holds_rests_until_the_guard_lets_go() {
        let start = Instant::now();
        let mut motion = GripperMotion::new(MOVE_TARGET, MOVE_TIMEOUT, start);
        assert_eq!(motion.travel_end(free(0.94), start + ms(10)), None);
        let held_at = start + ms(20);
        assert_eq!(motion.travel_end(held(0.94), held_at), Some(held_at));
        assert_eq!(
            motion.travel_end(held(0.94), held_at + ms(10)),
            Some(held_at),
            "the rest goes on"
        );
        assert_eq!(
            motion.travel_end(held(0.94 - 1e-9), held_at + ms(20)),
            Some(held_at),
            "a creeping hold is a rest"
        );
        assert_eq!(
            motion.travel_end(free(0.88), held_at + ms(30)),
            None,
            "the travel moved on"
        );
        let landed = held_at + ms(40);
        assert_eq!(motion.travel_end(free(MOVE_TARGET), landed), Some(landed));
    }

    #[test]
    fn a_move_with_no_travel_ends_it_on_the_tick_after_admission() {
        let start = Instant::now();
        let mut motion = GripperMotion::new(MOVE_TARGET, MOVE_TIMEOUT, start);
        assert_eq!(
            motion.travel_end(free(MOVE_TARGET), start),
            None,
            "the admission tick"
        );
        assert_eq!(
            motion.travel_end(free(MOVE_TARGET), start + ms(10)),
            Some(start + ms(10))
        );
    }

    #[test]
    fn jaws_standing_still_while_the_command_travels_are_not_judged() {
        let start = Instant::now();
        let mut motion = GripperMotion::new(MOVE_TARGET, MOVE_TIMEOUT, start);
        for tick in 1..=40u32 {
            let commanded = 1.0 - 0.01 * f64::from(tick);
            assert_eq!(
                gripper_move_terminal(
                    &mut motion,
                    free(commanded),
                    fresh(0.9),
                    Admission::Live,
                    None,
                    start + ms(10) * tick
                ),
                None
            );
        }
        assert!(
            motion.still.is_none(),
            "the still run starts at the planned end"
        );
    }

    #[test]
    fn a_hold_the_guard_lets_go_of_mid_travel_leaves_nothing_behind() {
        // The guard holds the command for two ticks mid-travel with the jaws
        // standing at 0.9; once it lets go, nothing judged then counts.
        let start = Instant::now();
        let mut motion = GripperMotion::new(MOVE_TARGET, MOVE_TIMEOUT, start);
        let ends = |m: &mut GripperMotion, commanded, now| {
            gripper_move_terminal(m, commanded, fresh(0.9), Admission::Live, None, now)
        };
        assert_eq!(ends(&mut motion, free(0.94), start + ms(10)), None);
        assert_eq!(ends(&mut motion, held(0.94), start + ms(20)), None);
        assert_eq!(ends(&mut motion, held(0.94), start + ms(30)), None);
        assert!(motion.still.is_some(), "judged while the guard held it");
        assert_eq!(ends(&mut motion, free(0.88), start + ms(40)), None);
        assert_eq!(motion.planned_end, None);
        assert!(motion.still.is_none());
        // Landed on the target, the still run starts over: the move settles
        // a window after landing, never on the hold's openings.
        let landed = start + ms(50);
        assert_eq!(ends(&mut motion, free(MOVE_TARGET), landed), None);
        assert_eq!(
            ends(
                &mut motion,
                free(MOVE_TARGET),
                landed + GRIPPER_STILL_WINDOW - ms(1)
            ),
            None
        );
        assert_eq!(
            ends(
                &mut motion,
                free(MOVE_TARGET),
                landed + GRIPPER_STILL_WINDOW
            ),
            Some(GripperTerminal::Settled)
        );
    }

    #[test]
    fn jaws_standing_still_for_the_window_from_the_planned_end_settle_the_move() {
        let end = Instant::now();
        let mut motion = motion_ending_at(end);
        assert_eq!(judge(&mut motion, fresh(0.5), end), None);
        assert_eq!(
            judge(&mut motion, fresh(0.5), end + GRIPPER_STILL_WINDOW - ms(1)),
            None
        );
        assert_eq!(
            judge(&mut motion, fresh(0.5), end + GRIPPER_STILL_WINDOW),
            Some(GripperTerminal::Settled)
        );
    }

    #[test]
    fn jaws_stopped_short_by_an_item_settle_the_move_where_they_stand() {
        // An object holds the jaws at 0.8 on their way to the half close.
        let end = Instant::now();
        let mut motion = motion_ending_at(end);
        assert_eq!(judge(&mut motion, fresh(0.8), end), None);
        assert_eq!(
            judge(&mut motion, fresh(0.8), end + GRIPPER_STILL_WINDOW),
            Some(GripperTerminal::Settled)
        );
    }

    #[test]
    fn a_grasp_that_creeps_inside_the_band_settles_the_move() {
        // A can slipping in the jaws after contact: 0.0034 of the travel per
        // 0.25 s, measured on an openarm_v2, inside the band.
        let end = Instant::now();
        let mut motion = motion_ending_at(end);
        assert_eq!(judge(&mut motion, fresh(0.6), end), None);
        assert_eq!(judge(&mut motion, fresh(0.5983), end + ms(125)), None);
        assert_eq!(
            judge(&mut motion, fresh(0.5966), end + GRIPPER_STILL_WINDOW),
            Some(GripperTerminal::Settled)
        );
    }

    #[test]
    fn a_change_larger_than_the_band_restarts_the_window() {
        let end = Instant::now();
        let mut motion = motion_ending_at(end);
        assert_eq!(judge(&mut motion, fresh(0.7), end), None);
        let moved = 0.7 - GRIPPER_STILL_BAND * 1.5;
        let restart = end + ms(200);
        assert_eq!(judge(&mut motion, fresh(moved), restart), None);
        assert_eq!(
            judge(&mut motion, fresh(moved), end + GRIPPER_STILL_WINDOW),
            None
        );
        assert_eq!(
            judge(&mut motion, fresh(moved), restart + GRIPPER_STILL_WINDOW),
            Some(GripperTerminal::Settled)
        );
    }

    #[test]
    fn openings_spread_wider_than_the_band_within_the_window_are_not_still() {
        // Each opening lies within the band of the first, yet the lowest and
        // the highest lie farther apart than the band: the jaws moved that much.
        let end = Instant::now();
        let mut motion = motion_ending_at(end);
        assert_eq!(judge(&mut motion, fresh(0.7), end), None);
        assert_eq!(
            judge(
                &mut motion,
                fresh(0.7 + 0.8 * GRIPPER_STILL_BAND),
                end + ms(100)
            ),
            None
        );
        let low = 0.7 - 0.8 * GRIPPER_STILL_BAND;
        let restart = end + ms(200);
        assert_eq!(judge(&mut motion, fresh(low), restart), None);
        assert_eq!(
            judge(&mut motion, fresh(low), end + GRIPPER_STILL_WINDOW),
            None,
            "the run restarted on the low opening"
        );
        assert_eq!(
            judge(&mut motion, fresh(low), restart + GRIPPER_STILL_WINDOW),
            Some(GripperTerminal::Settled)
        );
    }

    #[test]
    fn an_opening_not_delivered_this_tick_never_completes_the_window() {
        let end = Instant::now();
        let mut motion = motion_ending_at(end);
        assert_eq!(judge(&mut motion, fresh(0.5), end), None);
        assert_eq!(
            judge(&mut motion, stale_cached(0.5), end + GRIPPER_STILL_WINDOW),
            None,
            "the follower delivered nothing new"
        );
        assert_eq!(
            judge(&mut motion, fresh(0.5), end + GRIPPER_STILL_WINDOW + ms(10)),
            Some(GripperTerminal::Settled)
        );
    }

    #[test]
    fn jaws_not_still_by_the_settle_timeout_complete_the_move_where_they_stand() {
        // The jaws close 0.02 every 100 ms and never stand still.
        let end = Instant::now();
        let mut motion = motion_ending_at(end);
        for step in 0..20u32 {
            assert_eq!(
                judge(
                    &mut motion,
                    fresh(0.9 - 0.02 * f64::from(step)),
                    end + ms(100) * step
                ),
                None
            );
        }
        assert_eq!(
            judge(&mut motion, fresh(0.5), end + MOVE_TIMEOUT),
            Some(GripperTerminal::SettleTimedOut)
        );
    }

    #[test]
    fn a_cancel_ends_the_move_cancelled_for_its_reason_ahead_of_any_verdict() {
        let start = Instant::now();
        let reason = "cancelled by its caller";
        let cancelled = Some(GripperTerminal::Cancelled(reason.to_string()));
        let mut travelling = GripperMotion::new(MOVE_TARGET, MOVE_TIMEOUT, start);
        let ends = |m: &mut GripperMotion, commanded, now| {
            let cancel = Some(reason.to_string());
            gripper_move_terminal(m, commanded, fresh(0.8), Admission::Live, cancel, now)
        };
        assert_eq!(ends(&mut travelling, free(0.94), start + ms(10)), cancelled);
        let end = Instant::now();
        let mut settling = motion_ending_at(end);
        assert_eq!(
            ends(&mut settling, free(MOVE_TARGET), end + MOVE_TIMEOUT),
            cancelled
        );
    }

    #[test]
    fn a_follower_gone_stale_fails_the_move_before_and_after_the_planned_end() {
        let start = Instant::now();
        let fails = |m: &mut GripperMotion, commanded, now| {
            gripper_move_terminal(m, commanded, fresh(0.9), Admission::Stale, None, now)
        };
        let mut travelling = GripperMotion::new(MOVE_TARGET, MOVE_TIMEOUT, start);
        assert_eq!(
            fails(&mut travelling, free(0.94), start + ms(10)),
            Some(GripperTerminal::FollowerStale)
        );
        let end = Instant::now();
        let mut settling = motion_ending_at(end);
        assert_eq!(
            fails(&mut settling, free(MOVE_TARGET), end + ms(10)),
            Some(GripperTerminal::FollowerStale)
        );
    }

    /// A planner of the `side` arm of v1 hardware, holding `held`.
    fn planner_holding(side: Side, held: JointVec) -> Planner {
        use crate::planner::PlanConfig;
        use srs_model::chain_kinematics::EeCaps;

        const TEST_PERIOD: Duration = Duration::from_millis(10);
        let version = openarm_description::HardwareVersion::V1;
        let model =
            crate::arm_model(version, side.model()).expect("build arm from the bundled URDF");
        let limits = model.limits();
        let mut planner = Planner::new(
            side,
            model,
            PlanConfig {
                cycle_period: TEST_PERIOD,
                smoothing: srs_model::chain_kinematics::command_smoothing(TEST_PERIOD).unwrap(),
                max_joint_velocity_rad_s: [10.0; ARM_DOF],
                max_joint_acceleration_rad_s2: crate::test_support::A_MAX,
                ee: EeCaps {
                    linear_m_s: 1.0,
                    angular_rad_s: 0.8,
                },
                limits,
            },
        );
        planner.commit(held);
        planner
    }

    /// One arm's channels, with its busy slot claimed as at a goal's
    /// accept. The test drives them through the two senders it also gets:
    /// the measured state and the goal queue.
    fn claimed_arm_channels() -> (
        ArmChannels,
        watch::Sender<Option<ArmState>>,
        mpsc::Sender<Goal>,
    ) {
        let (command, _) = watch::channel(None);
        let (gripper_command, _) = watch::channel(None);
        let (measured_tx, measured) = watch::channel(None);
        let (_, gripper) = watch::channel(None);
        let (goal_tx, goals) = mpsc::channel(2);
        let (_, gripper_goals) = mpsc::channel(1);
        let channels = ArmChannels {
            command,
            gripper_command,
            leader: LeaderActivity::default(),
            gripper_leader: LeaderActivity::default(),
            measured,
            gripper,
            goals,
            busy: Arc::new(AtomicBool::new(true)),
            gripper_goals,
            gripper_busy: Arc::new(AtomicBool::new(false)),
            fresh: Arc::new(AtomicBool::new(false)),
            gripper_fresh: Arc::new(AtomicBool::new(false)),
        };
        (channels, measured_tx, goal_tx)
    }

    /// The measured state of an arm at `positions`, standing still.
    fn standing_at(positions: JointVec) -> Option<ArmState> {
        Some(ArmState {
            positions,
            velocities: [0.0; ARM_DOF],
        })
    }

    #[tokio::test]
    async fn a_stale_tick_fails_queued_goals_instead_of_parking_them() {
        let held = [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];
        let mut planner = planner_holding(Side::Left, held);
        let (mut channels, _measured_tx, goal_tx) = claimed_arm_channels();
        let (done_tx, mut done_rx) = mpsc::channel::<Outcome>(1);
        goal_tx
            .send(Goal::posture_share(held, done_tx))
            .await
            .expect("queue the goal");

        let tick = tick_arm(
            &mut channels,
            &mut planner,
            Admission::Stale,
            Instant::now(),
        )
        .await;

        let outcome = done_rx.recv().await.expect("a failed goal must report");
        assert_eq!(outcome, Outcome::Failed(stale_failure().labelled("left")));
        assert!(
            !channels.busy.load(Ordering::Acquire),
            "the failure releases the claim"
        );
        assert_eq!(tick.candidate, planner.setpoint(), "a stale side holds");
        assert!(tick.streamed_hand.is_none(), "a stale side streams nothing");
    }

    /// A leader's command is followed only while the leader streams: once it
    /// stops, the arm holds its setpoint.
    #[tokio::test]
    async fn a_command_from_a_leader_that_stopped_streaming_is_held_not_followed() {
        let held = [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];
        let mut planner = planner_holding(Side::Left, held);
        let (mut channels, measured_tx, _goal_tx) = claimed_arm_channels();
        measured_tx.send_replace(standing_at(held));
        let target = [0.3, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];
        channels
            .command
            .send_replace(Some(Upstream::Joints(target)));

        let silent = tick_arm(&mut channels, &mut planner, Admission::Live, Instant::now()).await;
        assert_eq!(silent.candidate, held, "no leader streaming: the arm holds");

        channels.leader.note();
        let streaming =
            tick_arm(&mut channels, &mut planner, Admission::Live, Instant::now()).await;
        assert!(
            streaming.candidate[0] > held[0],
            "a streaming leader's command is chased: {:?}",
            streaming.candidate
        );
    }

    /// A posture share in flight when its follower goes stale fails at the
    /// first stale tick and frees its arm.
    #[tokio::test]
    async fn a_stale_tick_ends_the_posture_share_in_flight() {
        let held = [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];
        let mut planner = planner_holding(Side::Left, held);
        let (mut channels, measured_tx, goal_tx) = claimed_arm_channels();
        measured_tx.send_replace(standing_at(held));
        let (done_tx, mut done_rx) = mpsc::channel::<Outcome>(1);
        goal_tx
            .send(Goal::posture_share([0.3; ARM_DOF], done_tx))
            .await
            .expect("queue the goal");

        let start = Instant::now();
        tick_arm(&mut channels, &mut planner, Admission::Live, start).await;
        assert!(done_rx.try_recv().is_err(), "the share is in flight");
        assert!(channels.busy.load(Ordering::Acquire));

        tick_arm(
            &mut channels,
            &mut planner,
            Admission::Stale,
            start + Duration::from_millis(100),
        )
        .await;
        let outcome = done_rx.try_recv().expect("the stale tick ends the share");
        assert_eq!(outcome, Outcome::Failed(stale_failure().labelled("left")));
        assert!(
            !channels.busy.load(Ordering::Acquire),
            "the failure releases the claim"
        );
    }

    /// A grasp request measures each arm when the loop serves it, not when
    /// the arm's posture share ended:
    /// - the left share runs its time out, then the left follower goes
    ///   stale, so the left arm has no fresh measurement;
    /// - the right share fails on a stale follower, then the right follower
    ///   recovers, so the right arm has the grasp pose of its latest joints.
    #[tokio::test]
    async fn a_grasp_request_measures_each_arm_when_it_is_served() {
        let held = [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];
        let (left, left_measured, left_goals) = claimed_arm_channels();
        let (right, right_measured, right_goals) = claimed_arm_channels();
        let mut channels = ArmPair::new(left, right);
        let mut planners = ArmPair::new(
            planner_holding(Side::Left, held),
            planner_holding(Side::Right, held),
        );
        let (done_tx, mut done_rx) = mpsc::channel::<Outcome>(2);
        for (measured, goals) in [
            (&left_measured, &left_goals),
            (&right_measured, &right_goals),
        ] {
            measured.send_replace(standing_at(held));
            goals
                .send(Goal::posture_share([0.3; ARM_DOF], done_tx.clone()))
                .await
                .expect("queue the share");
        }

        let start = Instant::now();
        for side in Side::BOTH {
            let (arm, planner) = (&mut channels[side], &mut planners[side]);
            tick_arm(arm, planner, Admission::Live, start).await;
        }
        assert!(done_rx.try_recv().is_err(), "both shares are in flight");
        let late = start + Duration::from_secs(3600);
        tick_arm(
            &mut channels.left,
            &mut planners.left,
            Admission::Live,
            late,
        )
        .await;
        tick_arm(
            &mut channels.right,
            &mut planners.right,
            Admission::Stale,
            late,
        )
        .await;
        let left_end = done_rx.try_recv().expect("the left share ends");
        assert_eq!(left_end, Outcome::Complete);
        let right_end = done_rx.try_recv().expect("the right share ends");
        assert_eq!(
            right_end,
            Outcome::Failed(stale_failure().labelled("right"))
        );

        let recovered = [0.2, -0.6, 0.1, 1.0, 0.0, 0.0, 0.0];
        right_measured.send_replace(standing_at(recovered));
        let (reply, answer) = oneshot::channel();
        serve_request(
            CoordinatorRequest::MeasureGrasps { reply },
            &mut channels,
            &mut planners,
            ArmPair::new(Admission::Stale, Admission::Reanchor),
            &mut ArmPair::new(None, None),
            &|_: &watch::Receiver<Option<GripperState>>| 0.0,
            late,
        )
        .await;
        let grasps = answer.await.expect("the loop answers the request");
        assert_eq!(grasps.left, Err(Unmeasured::Stale));
        assert_eq!(grasps.right, Ok(planners.right.world_ee_pose(&recovered)));
    }

    /// A stop ends every move in flight, whoever started it, and every goal
    /// admitted but not started yet: each ends cancelled for a reason that
    /// names the stop and the reason of its request. The answer names the
    /// limbs whose move was in flight, and both arms are free again.
    #[tokio::test]
    async fn a_stop_ends_each_move_cancelled_for_the_stops_reason() {
        let held = [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];
        let (left, left_measured, left_goals) = claimed_arm_channels();
        let (right, right_measured, right_goals) = claimed_arm_channels();
        let mut channels = ArmPair::new(left, right);
        let mut planners = ArmPair::new(
            planner_holding(Side::Left, held),
            planner_holding(Side::Right, held),
        );
        left_measured.send_replace(standing_at(held));
        right_measured.send_replace(standing_at(held));
        let (running_tx, mut running) = mpsc::channel::<Outcome>(1);
        left_goals
            .send(Goal::posture_share([0.3; ARM_DOF], running_tx))
            .await
            .expect("queue the left share");
        let start = Instant::now();
        tick_arm(
            &mut channels.left,
            &mut planners.left,
            Admission::Live,
            start,
        )
        .await;
        assert!(running.try_recv().is_err(), "the left share is in flight");
        let (queued_tx, mut queued) = mpsc::channel::<Outcome>(1);
        right_goals
            .send(Goal::posture_share([0.3; ARM_DOF], queued_tx))
            .await
            .expect("queue the right share");

        let (reply, answer) = oneshot::channel();
        serve_request(
            CoordinatorRequest::Stop {
                reason: "the operator asked".to_string(),
                reply,
            },
            &mut channels,
            &mut planners,
            ArmPair::new(Admission::Live, Admission::Live),
            &mut ArmPair::new(None, None),
            &|_: &watch::Receiver<Option<GripperState>>| 0.0,
            start,
        )
        .await;
        assert_eq!(
            answer.await.expect("the loop answers the stop"),
            ["left_arm"]
        );
        let stopped = Outcome::Cancelled("stopped: the operator asked".to_string());
        assert_eq!(
            running.try_recv().expect("the move in flight ends"),
            stopped
        );
        assert_eq!(queued.try_recv().expect("the queued goal ends"), stopped);
        assert!(!channels.left.busy.load(Ordering::Acquire));
        assert!(!channels.right.busy.load(Ordering::Acquire));
    }

    #[test]
    fn the_stop_reason_names_the_stop_and_the_reason_of_its_request() {
        assert_eq!(
            stop_reason("the operator asked"),
            "stopped: the operator asked"
        );
        assert_eq!(stop_reason(""), "stopped");
    }

    /// During the seed wait, a grasp request gets no pose for either arm,
    /// also when both arms have measured their joints. Here the grippers
    /// stay silent, so the seed never ends.
    #[tokio::test]
    async fn a_grasp_request_during_the_seed_wait_gets_no_pose() {
        let held = [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];
        let (mut left, left_measured, _left_goals) = claimed_arm_channels();
        let (mut right, right_measured, _right_goals) = claimed_arm_channels();
        let (_left_gripper, silent) = watch::channel(None);
        left.gripper = silent;
        let (_right_gripper, silent) = watch::channel(None);
        right.gripper = silent;
        left_measured.send_replace(standing_at(held));
        right_measured.send_replace(standing_at(held));
        let mut channels = ArmPair::new(left, right);
        let mut planners = ArmPair::new(
            planner_holding(Side::Left, held),
            planner_holding(Side::Right, held),
        );
        let (requests_tx, mut requests) = mpsc::channel(1);
        let (reply, answer) = oneshot::channel();
        requests_tx
            .send(CoordinatorRequest::MeasureGrasps { reply })
            .await
            .expect("queue the request");

        tokio::select! {
            _ = seed_all(&mut channels, &mut planners, &mut requests) => {
                panic!("the seed ended while the grippers are silent");
            }
            grasps = answer => {
                let grasps = grasps.expect("the seed wait answers the request");
                assert_eq!(grasps.left, Err(Unmeasured::NotYet));
                assert_eq!(grasps.right, Err(Unmeasured::NotYet));
            }
        }
    }

    /// After the seed, a side's results report its latest measurement while
    /// it delivers, and no measurement once it is stale, whatever it last
    /// measured.
    #[test]
    fn a_stale_side_reports_no_measurement_and_a_live_side_its_latest() {
        let (channels, measured_tx, _goal_tx) = claimed_arm_channels();
        let latest = [0.1, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];
        measured_tx.send_replace(standing_at(latest));
        assert_eq!(seeded_measurement(&channels, Admission::Live), Ok(latest));
        assert_eq!(
            seeded_measurement(&channels, Admission::Reanchor),
            Ok(latest)
        );
        assert_eq!(
            seeded_measurement(&channels, Admission::Stale),
            Err(Unmeasured::Stale)
        );
    }
}
