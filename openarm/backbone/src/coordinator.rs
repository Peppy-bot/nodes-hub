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
//! - the path of an execute_joint_path goal starts on every arm it moves at
//!   once, or fails before any arm moves, and is judged beside the loop
//!   ([`crate::path_checks`]), its arms held until the judgment comes;
//! - a posture goal that completes or ends cancelled asks for the grasp pose
//!   of each arm.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use peppygen::NodeRunner;
use peppylib::runtime::CancellationToken;
use srs_model::JointVec;
use srs_model::nalgebra::Isometry3;
use tokio::sync::{mpsc, oneshot, watch};
use tracing::{error, info, warn};

use control_core::filters::LowPassFilter;
use control_core::pacer::Pacer;
use openarm_description::{ARM_DOF, ArmPair, Side};

use crate::camera_mounts::MeasuredGrasps;
use crate::failure::{Failure, FailureKind, stale_failure};
use crate::governor::{GovState, Governor, Guard, GuardAt, PathArms};
use crate::gripper_move::{
    CommandedOpening, GripperGoal, GripperMove, GripperTerminal, MeasuredOpening, end_gripper_move,
    service_gripper_move,
};
use crate::liveness::{Admission, Cadence, CadenceChange, Liveness};
use crate::move_slot::MoveSlot;
use crate::path_checks::PathChecks;
use crate::planner::{
    self, ArmMoveRequest, Goal, Measurement, PathGoal, PathJudgment, PlanCheck, Planner,
    Unmeasured, chase_together, commit_both, start_path,
};
use crate::publish::{LimbStateSnapshot, Publishers};
use crate::streams::{ArmState, GovernorConfig, GripperCommand, GripperState, LeaderActivity};
use crate::upstream::{Upstream, UpstreamMode};
use control_core::chase::rate_limited;

/// How long [`seed_all`] waits for an arm's first measured state before warning that
/// the backbone is still blocked, so a silent arm is visible in the log instead of an
/// indefinite quiet stall.
const SEED_WAIT_WARN_PERIOD: Duration = Duration::from_secs(2);

/// One arm's inbound channels into the coordinator: the leading node's arm
/// command stream, its gripper opening command stream, the measured arm state,
/// the measured gripper opening, the accepted-goal queues, and the move slots
/// (one for arm moves, one for gripper moves).
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
    pub move_slot: MoveSlot,
    pub gripper_goals: mpsc::Receiver<GripperGoal>,
    pub gripper_move_slot: MoveSlot,
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
pub type GraspMeasurement = Result<Isometry3<f64>, Unmeasured>;

/// What the limb_motion services, the posture moves and execute_joint_path
/// ask of the loop, answered on the reply channel each carries. A reply that
/// the asker no longer waits for is dropped.
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
        reply: oneshot::Sender<ArmPair<GraspMeasurement>>,
    },
    /// Start the path of an execute_joint_path goal on every arm it moves at
    /// once, answering that the loop took it, or the failure it ended with
    /// before any arm moved. Its shares report how they end, the failure of
    /// the path's judgment among them, and a stop that comes before the path
    /// starts ends each of them.
    StartPath {
        goal: PathGoal,
        started: oneshot::Sender<Result<(), Failure>>,
    },
}

/// The reason every goal a stop ends is cancelled for: the stop, and the
/// reason of the stop request.
fn stop_reason(reason: &str) -> String {
    if reason.is_empty() {
        return "stopped".to_string();
    }
    format!("stopped: {reason}")
}

/// Run the coordination loop. Holds the governor and both planners. Runs until
/// the node's cancellation token fires; returns `Err` if a publisher cannot be
/// declared at bringup. Any return takes the node down (the supervisor in `node.rs`
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
    mut path_checks: PathChecks,
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
    // In-flight backbone-executed gripper moves, at most one per side.
    let mut gripper_moves: ArmPair<Option<GripperMove>> = ArmPair::from_fn(|_| None);
    // The effort cap of the last setpoint each gripper got, `None` for no
    // preference and before the first setpoint: what a move that keeps the
    // effort of its gripper relays.
    let mut sent_efforts: ArmPair<Option<f64>> = ArmPair::from_fn(|_| None);

    let dt = cycle_period.as_secs_f64();
    // One low-pass per joint per arm, smoothing the published desired velocity. `node`
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

        let gripper_delivered = take_gripper_deliveries(&mut channels);
        let gripper_admission = admit_grippers(
            &mut gripper_liveness,
            &mut gripper_cadence,
            gripper_delivered,
            now,
            stale_limit,
        );
        let tick_state = TickState {
            arm_liveness: &mut arm_liveness,
            arm_cadence: &mut arm_cadence,
            channels: &mut channels,
            planners: &mut planners,
            commanded_grippers: &commanded_grippers,
            requests: &mut requests,
            gripper_moves: &mut gripper_moves,
            path_checks: &mut path_checks,
        };
        let arm_admission = admit_and_serve(
            tick_state,
            &gripper_fraction,
            |held| governor.guard_at(held),
            now,
            stale_limit,
        )
        .await;
        for side in Side::BOTH {
            let arm = &channels[side];
            let fresh = |admission: Admission| admission != Admission::Stale;
            arm.fresh
                .store(fresh(arm_admission[side]), Ordering::Release);
            arm.gripper_fresh
                .store(fresh(gripper_admission[side]), Ordering::Release);
        }
        let arm_ticks = advance_arms(&mut channels, &mut planners, arm_admission, now).await;
        let paths = paths_of(ArmPair::from_fn(|side| arm_ticks[side].on_path));
        let (arm_candidate, hands) = arm_ticks
            .map(|_, tick| (tick.candidate, tick.streamed_hand))
            .unzip();
        let arm_candidate = chase_together(&planners, arm_candidate);
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
        let targets = ArmPair::from_fn(|side| {
            gripper_target(&gripper_moves[side], &channels[side], sent_efforts[side])
        });
        for side in Side::BOTH {
            if targets[side].is_none() {
                commanded_grippers[side] = CommandedOpening::idle(measured_grippers[side]);
            }
        }

        // One governed configuration: the last published setpoints and grippers
        // as `prev`, the rate-limited chases as the candidate. The governor
        // throttles, holds, scans and monitors everything through one barrier.
        let prev = last_governed(&planners, &commanded_grippers);
        let cand = GovState::new(
            arm_candidate,
            ArmPair::from_fn(|side| {
                chase_gripper(prev.grippers[side], targets[side], gripper_rate)
            }),
        );
        let measured = measured_config(&channels, &prev, measured_grippers);
        let governed = governor.govern_along(&prev, &cand, &measured, &hands, paths, dt);
        commanded_grippers = ArmPair::from_fn(|side| {
            CommandedOpening::published(
                governed.grippers[side],
                prev.grippers[side],
                cand.grippers[side],
            )
        });

        // Every side commits (identity for a stale one, whose candidate was
        // the frozen setpoint), and a stale side stays silent, so its
        // follower holds its own last setpoint.
        commit_both(&mut planners, &governed.arms);
        // Publish one governed setpoint per arm on its pairing slot; the slot
        // scopes the stream to its paired arm, so the message names no side.
        for side in Side::BOTH {
            let governed_q = governed.arms[side];
            // Desired velocity is the per-tick position delta; low-pass it per joint so a
            // noisy stream does not drive the arm's Kd term into buzz. The published
            // position (`governed_q`) is untouched, so tracking is unaffected.
            let dq = filtered_velocity(&mut dq_filters[side], &governed_q, &prev.arms[side], dt);
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
                sent_efforts[side] = target.max_effort;
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

/// The loop's state that the first half of a tick reads and writes
/// ([`admit_and_serve`]), borrowed for that half: each arm follower's
/// delivery watchdog and cadence, both arms' channels and planners, the
/// openings the last tick commanded, the requests queued for the loop, the
/// gripper moves in flight, and the judgments of joint paths.
struct TickState<'a> {
    arm_liveness: &'a mut ArmPair<Liveness>,
    arm_cadence: &'a mut ArmPair<Cadence>,
    channels: &'a mut ArmPair<ArmChannels>,
    planners: &'a mut ArmPair<Planner>,
    commanded_grippers: &'a ArmPair<CommandedOpening>,
    requests: &'a mut mpsc::Receiver<CoordinatorRequest>,
    gripper_moves: &'a mut ArmPair<Option<GripperMove>>,
    path_checks: &'a mut PathChecks,
}

/// The first half of a tick at `now`, in its order, ahead of the arms'
/// advance, on the loop's `state`:
/// - admit each arm's follower against `stale_limit`, which re-anchors an
///   arm whose follower recovered on its measured pose ([`admit_arms`]);
/// - answer the queued requests ([`serve_queued`]) ahead of the advance, so a
///   stop ends every move before the planners step it once more, and a plan
///   check or a path that starts plans from where a recovered arm stands, as
///   a move_arm goal its planner takes this tick does;
/// - judge each path that starts against the guard `guard_at` gives for the
///   limbs where the last tick governed them, a recovered arm re-anchored
///   ([`PathChecks::judge`]);
/// - resume the shares of each path whose judgment came, a path judged again
///   from where its arms stand against the guard `guard_at` gives for the
///   limbs where the last tick governed them ([`PathChecks::resume_judged`]).
///
/// The arms' admissions this tick.
async fn admit_and_serve(
    state: TickState<'_>,
    gripper_fraction: &impl Fn(&watch::Receiver<Option<GripperState>>) -> f64,
    guard_at: impl Fn(GovState) -> GuardAt,
    now: Instant,
    stale_limit: Duration,
) -> ArmPair<Admission> {
    let TickState {
        arm_liveness,
        arm_cadence,
        channels,
        planners,
        commanded_grippers,
        requests,
        gripper_moves,
        path_checks,
    } = state;
    let arm_admission = admit_arms(
        arm_liveness,
        arm_cadence,
        channels,
        planners,
        now,
        stale_limit,
    );
    let held = last_governed(planners, commanded_grippers);
    let judgments = serve_queued(
        requests,
        channels,
        planners,
        arm_admission,
        gripper_moves,
        gripper_fraction,
        now,
    )
    .await;
    for judgment in judgments {
        path_checks.judge(judgment, guard_at(held));
    }
    let measured =
        ArmPair::from_fn(|side| seeded_measurement(&channels[side], arm_admission[side]));
    path_checks.resume_judged(planners, &measured, guard_at(held), now);
    arm_admission
}

/// Answer every request queued on `requests` ([`serve_request`]), giving
/// the judgments of the paths that start and still wait for them, for the
/// loop to judge: a stop ends the paths started before it among these
/// requests before they are judged. A stop also ends each
/// execute_joint_path goal admitted before it whose path has not started,
/// when the loop takes that goal ([`stop_before_path`],
/// [`stop_unstarted_path`]), as it ends the goals admitted for a limb.
async fn serve_queued(
    requests: &mut mpsc::Receiver<CoordinatorRequest>,
    channels: &mut ArmPair<ArmChannels>,
    planners: &mut ArmPair<Planner>,
    arm_admission: ArmPair<Admission>,
    gripper_moves: &mut ArmPair<Option<GripperMove>>,
    gripper_fraction: &impl Fn(&watch::Receiver<Option<GripperState>>) -> f64,
    now: Instant,
) -> Vec<PathJudgment> {
    let mut judgments = Vec::new();
    while let Ok(request) = requests.try_recv() {
        let stop = stop_before_path(&request, channels);
        let request = match (request, stop) {
            (CoordinatorRequest::StartPath { goal, started }, Some(reason)) => {
                stop_unstarted_path(goal, started, &reason, channels, planners, arm_admission);
                continue;
            }
            (request, _) => request,
        };
        let judgment = serve_request(
            request,
            channels,
            planners,
            arm_admission,
            gripper_moves,
            gripper_fraction,
            now,
        )
        .await;
        judgments.extend(judgment);
    }
    judgments.retain(|judgment| judgment.is_awaited(planners));
    judgments
}

/// Answer one request:
/// - A stop ends the moves in flight on both sides and the goals admitted
///   but not started. The planner ends an arm's move; the gripper's
///   terminal ends a gripper's move. A goal that holds a limb and that the
///   loop has not taken yet is marked on its move slot
///   ([`MoveSlot::stop_unstarted`](crate::move_slot::MoveSlot::stop_unstarted)),
///   and ends when the loop takes it.
/// - Each goal the stop ends is cancelled, for a reason that names the
///   stop. The answer names the limbs whose move was in flight. The
///   setpoints hold where they were last governed.
/// - A check asks the named arm's planner and moves nothing.
/// - A grasp request answers the grasp pose of each arm.
/// - The path of an execute_joint_path goal starts on every arm it moves, or
///   fails before any arm moves ([`start_path`]): the path's judgment is what
///   this gives, for the loop to judge. A path whose action no longer waits
///   for it starts nowhere, and its arms are freed.
///
/// The results, the grasp poses and the measurement a path starts from come
/// from each arm's [`seeded_measurement`] under `arm_admission`.
async fn serve_request(
    request: CoordinatorRequest,
    channels: &mut ArmPair<ArmChannels>,
    planners: &mut ArmPair<Planner>,
    arm_admission: ArmPair<Admission>,
    gripper_moves: &mut ArmPair<Option<GripperMove>>,
    gripper_fraction: &impl Fn(&watch::Receiver<Option<GripperState>>) -> f64,
    now: Instant,
) -> Option<PathJudgment> {
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
                    let _release = arm.move_slot.release_on_drop();
                    goal.stop(&reason, measured, planner).await;
                }
                let measured_frac = gripper_fraction(&arm.gripper);
                if let Some(m) = gripper_moves[side].take() {
                    end_gripper_move(
                        m,
                        GripperTerminal::Cancelled(reason.clone()),
                        measured_frac,
                        now,
                    )
                    .await;
                    stopped.push(side.gripper_name().to_string());
                }
                while let Ok(goal) = arm.gripper_goals.try_recv() {
                    let _release = arm.gripper_move_slot.release_on_drop();
                    goal.stop(&reason, measured_frac).await;
                }
                // Every goal the loop held is ended and its slot freed, so a
                // slot still claimed holds a goal the loop has not taken.
                arm.move_slot.stop_unstarted(&reason);
                arm.gripper_move_slot.stop_unstarted(&reason);
            }
            let _ = reply.send(stopped);
            None
        }
        CoordinatorRequest::CheckArmMove { request, reply } => {
            let _ = reply.send(planners[request.side].check_cartesian(&request));
            None
        }
        CoordinatorRequest::MeasureGrasps { reply } => {
            let _ = reply.send(measured_grasps(channels, planners, arm_admission));
            None
        }
        CoordinatorRequest::StartPath { goal, started } => {
            if started.is_closed() {
                return None;
            }
            let sides = goal.path.arms.sides().to_vec();
            let measured =
                ArmPair::from_fn(|side| seeded_measurement(&channels[side], arm_admission[side]));
            match start_path(planners, goal, &measured) {
                Ok(judgment) => {
                    if started.send(Ok(())).is_ok() {
                        return Some(judgment);
                    }
                    for side in sides {
                        planners[side].drop_share();
                    }
                    None
                }
                Err(failure) => {
                    let _ = started.send(Err(failure));
                    None
                }
            }
        }
    }
}

/// The reason of the stop that came after the admission of the
/// execute_joint_path goal of `request` and before the loop took it: the
/// mark the stop left on the move slot of an arm of the path. `None` for a
/// path that no stop came before, and for a request that starts no path.
fn stop_before_path(
    request: &CoordinatorRequest,
    channels: &ArmPair<ArmChannels>,
) -> Option<String> {
    let CoordinatorRequest::StartPath { goal, .. } = request else {
        return None;
    };
    goal.path
        .arms
        .sides()
        .iter()
        .find_map(|&side| channels[side].move_slot.take_stop())
}

/// End the path of an execute_joint_path goal that a stop came before, for
/// `reason`, which names the stop: each of its arms reports the joints it
/// measures, its arms are freed, and its action learns that the loop took
/// it. A goal whose action no longer waits is dropped.
fn stop_unstarted_path(
    goal: PathGoal,
    started: oneshot::Sender<Result<(), Failure>>,
    reason: &str,
    channels: &ArmPair<ArmChannels>,
    planners: &ArmPair<Planner>,
    arm_admission: ArmPair<Admission>,
) {
    if started.is_closed() {
        return;
    }
    let measured =
        ArmPair::from_fn(|side| seeded_measurement(&channels[side], arm_admission[side]));
    goal.stop(reason, &measured, planners);
    let _ = started.send(Ok(()));
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
        if ch.move_slot.is_claimed() {
            ch.command.send_replace(None);
        }
        if ch.gripper_move_slot.is_claimed() {
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
    // it, and the planners step streamed poses with it.
    governor.set_stream_ee_cap(cfg.max_stream_ee_velocity_m_s);
    governor.set_gripper_rate(cfg.max_gripper_rate_frac_s);
    for side in Side::BOTH {
        planners[side].set_max_stream_ee_velocity(cfg.max_stream_ee_velocity_m_s);
    }
}

/// Judge each arm's follower before commanding it.
///
/// A follower that has stopped delivering cannot be vouched for: its limb
/// freezes at the held setpoint and its wire goes silent, so the follower holds
/// its own last setpoint instead of tracking one that keeps advancing on the
/// operator's stream while the real arm drifts. The first delivery back
/// re-anchors the held setpoint on the measured pose here, as the tick
/// admits the follower and before it serves its requests, so the limb never
/// steps by the drift it accumulated unseen, and a path that starts on that
/// tick is checked and runs from where the arm stands. The per-joint
/// velocity limit in the chase then walks the setpoint back to the
/// operator's command, a tick's step at a time.
///
/// Each delivery counts once, on the tick that judges it: the watch is
/// marked seen here, so a state that arrives later in this tick counts on
/// the next one.
fn admit_arms(
    liveness: &mut ArmPair<Liveness>,
    cadence: &mut ArmPair<Cadence>,
    channels: &mut ArmPair<ArmChannels>,
    planners: &mut ArmPair<Planner>,
    now: Instant,
    stale_limit: Duration,
) -> ArmPair<Admission> {
    ArmPair::from_fn(|side| {
        let delivered = take_arm_delivery(&mut channels[side].measured);
        report_cadence(side, "arm", cadence[side].observe(delivered, now));
        let admission = liveness[side].admit(delivered, now, stale_limit);
        if admission == Admission::Reanchor {
            warn!("arm follower stream recovered; re-anchoring on the measured pose");
            let measured = channels[side]
                .measured
                .borrow()
                .expect("seed gated on the first state");
            planners[side].seed_from_measured(measured.positions);
        }
        admission
    })
}

/// The configuration the last tick governed: each arm's setpoint and each
/// gripper's commanded opening.
fn last_governed(planners: &ArmPair<Planner>, grippers: &ArmPair<CommandedOpening>) -> GovState {
    GovState::new(
        ArmPair::from_fn(|side| planners[side].setpoint()),
        ArmPair::from_fn(|side| grippers[side].frac),
    )
}

/// Log the delivery cadence of the follower of `side`'s `limb` crossing the
/// threshold under its declared rate: the signal that the launcher's
/// follower_state_rate_hz is a promise the follower is not keeping, ahead of
/// the stale limit freezing the limb.
fn report_cadence(side: Side, limb: &str, change: Option<CadenceChange>) {
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

/// Whether the arm follower of `measured` delivered a state since the tick
/// that last judged it, and the mark that this tick has judged it.
fn take_arm_delivery(measured: &mut watch::Receiver<Option<ArmState>>) -> bool {
    let delivered = measured.has_changed().unwrap_or(false);
    let _ = measured.borrow_and_update();
    delivered
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

/// The paths the arms step along this tick, from what each arm's tick
/// stepped along: one path of both arms when both step along their shares of
/// it, else each arm's own.
fn paths_of(on_path: ArmPair<Option<PathArms>>) -> &'static [PathArms] {
    match (on_path.left, on_path.right) {
        (Some(PathArms::Both), Some(PathArms::Both)) => &[PathArms::Both],
        (Some(_), Some(_)) => &[PathArms::One(Side::Left), PathArms::One(Side::Right)],
        (Some(_), None) => &[PathArms::One(Side::Left)],
        (None, Some(_)) => &[PathArms::One(Side::Right)],
        (None, None) => &[],
    }
}

/// Advance both planners to this tick's candidate setpoints and hand bases.
/// The share of a path of an arm whose follower is stale fails before either
/// arm advances, so it ends the other arm's share of a path of both arms on
/// the same tick; any other move of a stale arm fails as its arm ticks
/// ([`tick_arm`]).
async fn advance_arms(
    channels: &mut ArmPair<ArmChannels>,
    planners: &mut ArmPair<Planner>,
    admission: ArmPair<Admission>,
    now: Instant,
) -> ArmPair<planner::Tick> {
    for side in Side::BOTH {
        if admission[side] == Admission::Stale {
            planners[side].fail_path_share(stale_failure());
        }
    }
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
        let arm = &mut channels[side];
        service_gripper_move(
            &mut moves[side],
            &mut arm.gripper_goals,
            &arm.gripper_move_slot,
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
) -> ArmPair<GraspMeasurement> {
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

/// Wait for both arms' first measured states and both grippers' first
/// gripper fractions, then seed each planner's held setpoint from its measured pose
/// (clamped into the joint limits). One wait over all four goal queues: a
/// goal accepted for EITHER side before its follower reports fails with
/// [`seed_failure`], its move slot released, so a silent side cannot strand
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
                let _release = channels.left.move_slot.release_on_drop();
                goal.fail(seed_failure(), &planners.left).await;
            }
            Some(goal) = channels.right.goals.recv() => {
                let _release = channels.right.move_slot.release_on_drop();
                goal.fail(seed_failure(), &planners.right).await;
            }
            Some(goal) = channels.left.gripper_goals.recv() => {
                let _release = channels.left.gripper_move_slot.release_on_drop();
                goal.fail(seed_failure()).await;
            }
            Some(goal) = channels.right.gripper_goals.recv() => {
                let _release = channels.right.gripper_move_slot.release_on_drop();
                goal.fail(seed_failure()).await;
            }
            // Nothing moves before the seed, so a stop stops nothing, and a
            // check or a joint path has no held pose to start from. The seed
            // judges no liveness, so no measurement is known to be fresh: a
            // grasp request gets no pose for either arm.
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
                CoordinatorRequest::StartPath { started, .. } => {
                    let _ = started.send(Err(seed_failure()));
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
                error!("{} {what} channel closed before its first value", side);
                return Err(Shutdown);
            }
            Err(_) => warn!(
                "{} {what} not reported yet; backbone waiting to stream",
                side
            ),
        }
    }
}

/// Advance one arm's planner to its candidate setpoint for this tick: feed it
/// the measured pose and the latest leader command, and admit any pending move
/// goal. An arm whose follower recovered this tick was re-anchored as the tick
/// admitted it ([`admit_arms`]). Runs only after the seed, which gated on the
/// arm's first measurement.
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
    // a posture share that kept its claim would wedge its posture action for
    // good.
    if admission == Admission::Stale {
        planner.fail_active(stale_failure()).await;
        while let Ok(goal) = channels.goals.try_recv() {
            let _release = channels.move_slot.release_on_drop();
            goal.fail(stale_failure(), planner).await;
        }
        return planner::Tick {
            candidate: planner.setpoint(),
            streamed_hand: None,
            on_path: None,
        };
    }
    // A borrow, not a borrow_and_update: only `admit_arms` marks a
    // delivery seen, so a state that arrived after it judged this arm still
    // counts on the next tick.
    let measured_q = channels
        .measured
        .borrow()
        .expect("seed gated on the first state")
        .positions;
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
            &channels.move_slot,
            now,
        )
        .await
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
/// on an unpaired side). `sent_effort` is the effort cap of the last setpoint
/// the gripper got, which a move that keeps the effort of its gripper relays.
fn gripper_target(
    mv: &Option<GripperMove>,
    channels: &ArmChannels,
    sent_effort: Option<f64>,
) -> Option<GripperTarget> {
    if let Some(m) = mv {
        return Some(GripperTarget {
            frac: m.target_frac(),
            max_effort: m.effort().relayed(sent_effort),
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
    use crate::test_support::{
        ARMS_WOULD_MEET, BOTH_HELD, OpenGuard, Refusing, claimed_move_slot, distance_to_segment,
        measured_at_setpoints, path_goal_through, planner_holding, share_reply, test_guard,
        test_settle,
    };

    use std::sync::Mutex;

    use control_core::path_timing::RoundedPath;

    use crate::planner::{PathGuard, ShareReport};
    use crate::types::dual_joints;
    use openarm_description::ArmOrder;
    use srs_model::nalgebra::Vector3;

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
        // The gripper twin of the arm handoff: while a move_gripper goal holds
        // the side's move slot, `consume_streams_of_busy_sides` clears the
        // side's command watch (`send_replace(None)`) every tick, so the
        // gripper follows nothing new and holds the move's endpoint until a
        // command that arrives after the clear. Locks that contract.
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

    /// One arm's channels, with its move slot claimed as at a goal's
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
            move_slot: claimed_move_slot(),
            gripper_goals,
            gripper_move_slot: MoveSlot::default(),
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
        assert_eq!(
            outcome,
            Outcome::Failed(stale_failure().labelled(Side::Left))
        );
        assert!(
            !channels.move_slot.is_claimed(),
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

    /// Each delivery counts once, on the tick that judges it: a state that
    /// arrives after the tick judged its arm, and before the tick commands
    /// that arm, counts on the next tick.
    #[tokio::test]
    async fn a_state_that_arrives_after_its_arm_was_judged_counts_on_the_next_tick() {
        let held = [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];
        let mut planner = planner_holding(Side::Left, held);
        let (mut channels, measured_tx, _goal_tx) = claimed_arm_channels();
        measured_tx.send_replace(standing_at(held));
        assert!(take_arm_delivery(&mut channels.measured), "the first state");
        measured_tx.send_replace(standing_at(held));
        tick_arm(&mut channels, &mut planner, Admission::Live, Instant::now()).await;
        assert!(
            take_arm_delivery(&mut channels.measured),
            "the next tick counts the state that came after the judge"
        );
        assert!(!take_arm_delivery(&mut channels.measured), "once only");
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
        assert!(channels.move_slot.is_claimed());

        tick_arm(
            &mut channels,
            &mut planner,
            Admission::Stale,
            start + Duration::from_millis(100),
        )
        .await;
        let outcome = done_rx.try_recv().expect("the stale tick ends the share");
        assert_eq!(
            outcome,
            Outcome::Failed(stale_failure().labelled(Side::Left))
        );
        assert!(
            !channels.move_slot.is_claimed(),
            "the failure releases the claim"
        );
    }

    /// A grasp request measures each arm when the loop serves it, not when
    /// the arm's posture share ended:
    /// - the left share settles at its posture, then the left follower goes
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
        left_measured.send_replace(standing_at([0.3; ARM_DOF]));
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
            Outcome::Failed(stale_failure().labelled(Side::Right))
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
        assert!(!channels.left.move_slot.is_claimed());
        assert!(!channels.right.move_slot.is_claimed());
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
    // --- execute_joint_path ------------------------------------------------

    /// Each arm measuring [`BOTH_HELD`].
    fn measuring_held() -> ArmPair<Measurement> {
        ArmPair::from_fn(|side| Ok(BOTH_HELD[side]))
    }

    const LIVE: ArmPair<Admission> = ArmPair {
        left: Admission::Live,
        right: Admission::Live,
    };

    /// Both arms' channels, each arm measuring [`BOTH_HELD`] with its move slot
    /// free, the sender of each follower's measured state, and both planners
    /// holding [`BOTH_HELD`].
    fn followed_arms() -> (ArmPair<ArmChannels>, ArmPair<Follower>, ArmPair<Planner>) {
        let (channels, followers) = ArmPair::from_fn(|side| {
            let (channels, measured_tx, _goals) = claimed_arm_channels();
            channels.move_slot.release();
            measured_tx.send_replace(standing_at(BOTH_HELD[side]));
            (channels, measured_tx)
        })
        .unzip();
        let planners = ArmPair::from_fn(|side| planner_holding(side, BOTH_HELD[side]));
        (channels, followers, planners)
    }

    /// Where a follower's measured state comes from.
    type Follower = watch::Sender<Option<ArmState>>;

    /// Both arms' channels and planners as [`followed_arms`] gives them, the
    /// followers' senders dropped.
    fn both_arms() -> (ArmPair<ArmChannels>, ArmPair<Planner>) {
        let (channels, _followers, planners) = followed_arms();
        (channels, planners)
    }

    /// Both arms as [`both_arms`] gives them, the left planner holding 0.2
    /// rad off [`BOTH_HELD`], where the left arm measures, at joint 1 and 0.3 rad
    /// at joint 3.
    fn left_lagging() -> (ArmPair<ArmChannels>, ArmPair<Planner>) {
        let (channels, mut planners) = both_arms();
        let mut lagging = BOTH_HELD.left;
        lagging[0] += 0.2;
        lagging[2] += 0.3;
        planners.left = planner_holding(Side::Left, lagging);
        (channels, planners)
    }

    /// The goal of a path of the arms `arm_names` names, 0.3 rad out at their
    /// first joint over one second, claiming their slots as the action's
    /// admission does, its shares reporting on the receiver this gives.
    fn path_goal(
        channels: &ArmPair<ArmChannels>,
        arm_names: &[&str],
    ) -> (PathGoal, mpsc::Receiver<ShareReport>) {
        let arms = ArmOrder::parse(arm_names).expect("the arms of this robot");
        let to = BOTH_HELD.map(|_, mut q| {
            q[0] += 0.3;
            q
        });
        for &side in arms.sides() {
            assert!(channels[side].move_slot.claim(), "the arm is free");
        }
        let (reply, done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let goal = path_goal_through(
            &arms,
            &[to],
            Duration::from_secs(1),
            test_settle(&arms),
            reply,
            |side| &channels[side].move_slot,
        );
        (goal, done_rx)
    }

    /// Serve every request queued on `requests` at `now`, the arms admitted
    /// as `admission`: the judgments of the paths that start.
    async fn serve_all(
        requests: &mut mpsc::Receiver<CoordinatorRequest>,
        channels: &mut ArmPair<ArmChannels>,
        planners: &mut ArmPair<Planner>,
        admission: ArmPair<Admission>,
        now: Instant,
    ) -> Vec<PathJudgment> {
        serve_queued(
            requests,
            channels,
            planners,
            admission,
            &mut ArmPair::new(None, None),
            &|_: &watch::Receiver<Option<GripperState>>| 0.0,
            now,
        )
        .await
    }

    /// Ask the loop to start the path of `goal` at `now`: the loop's answer,
    /// and the judgment of a path that starts.
    async fn start_path_of(
        goal: PathGoal,
        channels: &mut ArmPair<ArmChannels>,
        planners: &mut ArmPair<Planner>,
        admission: ArmPair<Admission>,
        now: Instant,
    ) -> (Result<(), Failure>, Option<PathJudgment>) {
        let (requests_tx, mut requests) = mpsc::channel(1);
        let (started, answer) = oneshot::channel();
        requests_tx
            .try_send(CoordinatorRequest::StartPath { goal, started })
            .expect("queue the start");
        let mut judgments = serve_all(&mut requests, channels, planners, admission, now).await;
        (answer.await.expect("the loop answers"), judgments.pop())
    }

    /// Ask the loop to stop for "the operator asked" at `now`, alone on its
    /// tick: the limbs whose move the stop ended.
    async fn stop_alone(
        channels: &mut ArmPair<ArmChannels>,
        planners: &mut ArmPair<Planner>,
        now: Instant,
    ) -> Vec<String> {
        let (requests_tx, mut requests) = mpsc::channel(1);
        let (reply, stopped) = oneshot::channel();
        requests_tx
            .try_send(CoordinatorRequest::Stop {
                reason: "the operator asked".to_string(),
                reply,
            })
            .expect("queue the stop");
        serve_all(&mut requests, channels, planners, LIVE, now).await;
        stopped.await.expect("the loop answers the stop")
    }

    /// A stop marks each goal that holds a limb and that the loop has not
    /// taken, on the move slot of the limb, and no free limb.
    #[tokio::test]
    async fn a_stop_marks_the_goals_that_the_loop_has_not_taken() {
        let (mut channels, mut planners) = both_arms();
        assert!(channels.left.move_slot.claim(), "the arm is free");
        assert!(
            channels.right.gripper_move_slot.claim(),
            "the gripper is free"
        );
        let stopped = stop_alone(&mut channels, &mut planners, Instant::now()).await;
        assert!(stopped.is_empty(), "no move was in flight");

        let reason = Some("stopped: the operator asked".to_string());
        assert_eq!(channels.left.move_slot.take_stop(), reason);
        assert_eq!(channels.right.gripper_move_slot.take_stop(), reason);
        assert_eq!(channels.right.move_slot.take_stop(), None);
        assert!(
            !channels.right.move_slot.is_claimed(),
            "a stop claims no limb"
        );
        assert!(!channels.left.gripper_move_slot.is_claimed());
    }

    /// A stop that came on an earlier tick than the goal of an arm, after
    /// the admission of the goal, ends the goal when the arm's tick takes
    /// it: cancelled for the stop's reason, the arm held and freed.
    #[tokio::test]
    async fn a_stop_before_the_tick_that_takes_an_arm_goal_ends_the_goal() {
        let (mut channels, mut planners) = both_arms();
        assert!(channels.left.move_slot.claim(), "the arm is free");
        let now = Instant::now();
        stop_alone(&mut channels, &mut planners, now).await;

        let (goal_tx, goals) = mpsc::channel(1);
        channels.left.goals = goals;
        let (done_tx, mut done_rx) = mpsc::channel(1);
        goal_tx
            .try_send(Goal::posture_share([0.5; ARM_DOF], done_tx))
            .expect("queue the goal");
        let tick = tick_arm(
            &mut channels.left,
            &mut planners.left,
            Admission::Live,
            now + Duration::from_millis(10),
        )
        .await;
        assert_eq!(
            done_rx.try_recv().expect("the goal ends"),
            Outcome::Cancelled("stopped: the operator asked".to_string())
        );
        assert_eq!(tick.candidate, BOTH_HELD.left, "the arm holds");
        assert!(!channels.left.move_slot.is_claimed(), "the arm is freed");
    }

    /// A stop that came on an earlier tick than the start of a path, after
    /// the admission of its goal, ends the path before it starts, each arm
    /// cancelled for the stop's reason, and frees its arms.
    #[tokio::test]
    async fn a_stop_before_the_tick_that_takes_a_path_ends_the_path() {
        let (mut channels, mut planners) = both_arms();
        let (goal, mut done_rx) = path_goal(&channels, &["left_arm", "right_arm"]);
        let now = Instant::now();
        let stopped = stop_alone(&mut channels, &mut planners, now).await;
        assert!(stopped.is_empty(), "no move was in flight");

        let later = now + Duration::from_millis(10);
        let (started, judgment) =
            start_path_of(goal, &mut channels, &mut planners, LIVE, later).await;
        assert_eq!(started, Ok(()));
        assert!(judgment.is_none(), "no path starts");
        for side in Side::BOTH {
            let report = done_rx.try_recv().expect("each arm reports");
            assert_eq!(report.side, side);
            assert_eq!(
                report.outcome,
                Outcome::Cancelled("stopped: the operator asked".to_string())
            );
        }
        assert!(channels.iter().all(|arm| !arm.move_slot.is_claimed()));
    }

    /// A path on an arm whose follower is stale fails before any arm moves,
    /// labelled with that arm, and frees the arms it claimed; with both arms
    /// live, its shares hold their arms until it is judged.
    #[tokio::test]
    async fn a_path_on_a_stale_arm_fails_before_any_arm_moves() {
        let (mut channels, mut planners) = both_arms();
        let both = ["left_arm", "right_arm"];
        let right_stale = ArmPair::new(Admission::Live, Admission::Stale);
        let now = Instant::now();
        let (goal, _done_rx) = path_goal(&channels, &both);
        let (failed, judgment) =
            start_path_of(goal, &mut channels, &mut planners, right_stale, now).await;
        assert_eq!(
            failed,
            Err(stale_failure().labelled(Side::Right.arm_name()))
        );
        assert!(judgment.is_none());
        assert!(channels.iter().all(|arm| !arm.move_slot.is_claimed()));

        let (goal, mut done_rx) = path_goal(&channels, &both);
        let (started, judgment) =
            start_path_of(goal, &mut channels, &mut planners, LIVE, now).await;
        assert_eq!(started, Ok(()));
        assert!(judgment.is_some(), "the path waits for its judgment");
        assert!(done_rx.try_recv().is_err(), "both shares are in flight");
        assert!(channels.iter().all(|arm| arm.move_slot.is_claimed()));
    }

    /// A path whose action no longer waits for it starts on no arm, and its
    /// arms are freed.
    #[tokio::test]
    async fn a_path_whose_action_is_gone_starts_nowhere() {
        let (mut channels, mut planners) = both_arms();
        let (goal, _done_rx) = path_goal(&channels, &["left_arm"]);
        let (requests_tx, mut requests) = mpsc::channel(1);
        let (started, answer) = oneshot::channel();
        drop(answer);
        requests_tx
            .try_send(CoordinatorRequest::StartPath { goal, started })
            .expect("queue the start");
        let judgments = serve_all(
            &mut requests,
            &mut channels,
            &mut planners,
            LIVE,
            Instant::now(),
        )
        .await;
        assert!(judgments.is_empty());
        assert!(!channels.left.move_slot.is_claimed(), "the arm is freed");
        let (goal, _done_rx) = path_goal(&channels, &["left_arm"]);
        let (started, _) =
            start_path_of(goal, &mut channels, &mut planners, LIVE, Instant::now()).await;
        assert_eq!(started, Ok(()), "the arm takes the next path");
    }

    /// A stop ends the path of every execute_joint_path goal queued behind
    /// it before the path starts, each arm cancelled for the stop's reason
    /// with the joints it measures, and frees its arms; a path queued after
    /// the stop's tick starts.
    #[tokio::test]
    async fn a_stop_ends_a_path_queued_behind_it_before_the_path_starts() {
        let (mut channels, mut planners) = both_arms();
        let (requests_tx, mut requests) = mpsc::channel(2);
        let (reply, stopped) = oneshot::channel();
        requests_tx
            .try_send(CoordinatorRequest::Stop {
                reason: "the operator asked".to_string(),
                reply,
            })
            .expect("queue the stop");
        let (goal, mut done_rx) = path_goal(&channels, &["right_arm", "left_arm"]);
        let (started, answer) = oneshot::channel();
        requests_tx
            .try_send(CoordinatorRequest::StartPath { goal, started })
            .expect("queue the start");
        let now = Instant::now();
        let judgments = serve_all(&mut requests, &mut channels, &mut planners, LIVE, now).await;
        assert!(judgments.is_empty(), "no path starts");
        assert!(stopped.await.expect("the loop answers the stop").is_empty());
        assert_eq!(answer.await.expect("the loop takes the path"), Ok(()));
        for side in [Side::Right, Side::Left] {
            let report = done_rx.try_recv().expect("each arm reports");
            assert_eq!(report.side, side);
            assert_eq!(
                report.outcome,
                Outcome::Cancelled("stopped: the operator asked".to_string())
            );
            assert_eq!(report.joints, BOTH_HELD[side]);
            assert_eq!(report.elapsed_s, 0.0);
        }
        assert!(channels.iter().all(|arm| !arm.move_slot.is_claimed()));

        let (goal, _done_rx) = path_goal(&channels, &["left_arm"]);
        let (started, judgment) =
            start_path_of(goal, &mut channels, &mut planners, LIVE, now).await;
        assert_eq!(started, Ok(()));
        assert!(judgment.is_some(), "a later path starts");
    }

    /// When the follower of one arm of a path of both arms goes stale, the
    /// shares of both arms end on that tick, the stale arm's first with the
    /// stale failure, and the live arm holds where it stands, whichever arm
    /// goes stale.
    #[tokio::test]
    async fn a_stale_arm_ends_both_shares_of_a_path_on_the_same_tick() {
        for stale in Side::BOTH {
            let (mut channels, mut planners) = both_arms();
            let (goal, mut done_rx) = path_goal(&channels, &["left_arm", "right_arm"]);
            let start = Instant::now();
            let (started, judgment) =
                start_path_of(goal, &mut channels, &mut planners, LIVE, start).await;
            assert_eq!(started, Ok(()));
            let judgment = judgment.expect("the path waits for its judgment");
            let mut checks = PathChecks::new(Arc::new(Mutex::new(OpenGuard)));
            checks.judge(
                judgment,
                GuardAt::disabled(GovState::new(BOTH_HELD, ArmPair::new(1.0, 1.0))),
            );
            checks.resume_once_judged(&mut planners, start).await;
            let ticks = advance_arms(&mut channels, &mut planners, LIVE, start).await;
            commit_both(&mut planners, &ticks.map(|_, tick| tick.candidate));
            assert!(done_rx.try_recv().is_err(), "both shares are in flight");

            let live = match stale {
                Side::Left => Side::Right,
                Side::Right => Side::Left,
            };
            let admission = ArmPair::from_fn(|side| {
                if side == stale {
                    Admission::Stale
                } else {
                    Admission::Live
                }
            });
            let held = planners[live].setpoint();
            let ticks = advance_arms(
                &mut channels,
                &mut planners,
                admission,
                start + Duration::from_millis(20),
            )
            .await;
            assert_eq!(ticks[live].candidate, held, "the live arm holds");
            assert!(ticks[live].on_path.is_none());
            let first = done_rx.try_recv().expect("both shares end on this tick");
            assert_eq!(first.side, stale, "the stale arm's share reports first");
            assert_eq!(
                first.outcome,
                Outcome::Failed(stale_failure().labelled(stale.arm_name()))
            );
            let second = done_rx.try_recv().expect("both shares end on this tick");
            assert_eq!(second.side, live);
            assert!(matches!(second.outcome, Outcome::Failed(_)), "{second:?}");
        }
    }

    /// What the test hands a check of [`Gated`] at each of its probes.
    enum Gate {
        Shut,
        Open,
    }

    /// What a check of [`Gated`] did at a probe the test handed it.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Probed {
        /// The gate was shut: the check goes on to its next probe.
        Shut,
        /// The gate was open: the check lets the path through.
        Opened,
        /// The judgment's stop was raised: the check ended.
        Stopped,
    }

    /// The words of a check of [`Gated`] that ended with its test.
    const TEST_ENDED: &str = "the test ended";

    /// A checker the test steps probe by probe, as a long path keeps the
    /// checker busy until its check ends. At each probe it waits for the gate
    /// the test hands it, ends the check once its stop is raised, as the path
    /// checker does at its next measured probe, else lets the path through when
    /// the gate is open, and tells the test what it did.
    struct Gated {
        gates: std::sync::mpsc::Receiver<Gate>,
        probes: mpsc::UnboundedSender<Probed>,
    }

    impl PathGuard for Gated {
        fn check(
            &mut self,
            _arms: PathArms,
            _guard: &GuardAt,
            _path: &RoundedPath,
            stop: &AtomicBool,
        ) -> Result<(), String> {
            loop {
                let Ok(gate) = self.gates.recv() else {
                    return Err(TEST_ENDED.to_string());
                };
                let probed = match (stop.load(Ordering::Acquire), gate) {
                    (true, _) => Probed::Stopped,
                    (false, Gate::Open) => Probed::Opened,
                    (false, Gate::Shut) => Probed::Shut,
                };
                if self.probes.send(probed).is_err() {
                    return Err(TEST_ENDED.to_string());
                }
                match probed {
                    Probed::Shut => {}
                    Probed::Opened => return Ok(()),
                    Probed::Stopped => return Err("the check stopped".to_string()),
                }
            }
        }
    }

    /// The judgments of a [`Gated`] checker: where the test hands it a gate
    /// at each probe, and what it did there.
    fn gated_checks() -> (
        std::sync::mpsc::Sender<Gate>,
        mpsc::UnboundedReceiver<Probed>,
        PathChecks,
    ) {
        let (gates_tx, gates) = std::sync::mpsc::channel();
        let (probes_tx, probes) = mpsc::unbounded_channel();
        let checker = Gated {
            gates,
            probes: probes_tx,
        };
        (
            gates_tx,
            probes,
            PathChecks::new(Arc::new(Mutex::new(checker))),
        )
    }

    /// The first probe of the check of the first path being judged: a
    /// judgment that ends before its check probes fails the test. The check
    /// reports each probe before it ends, so a probe that came wins.
    async fn first_probe(
        probes: &mut mpsc::UnboundedReceiver<Probed>,
        path_checks: &mut PathChecks,
    ) -> Probed {
        tokio::select! {
            biased;
            probed = probes.recv() => probed.expect("the checker holds its probes"),
            verdict = path_checks.first_verdict() => {
                panic!("the judgment ended before its check probed: {verdict:?}")
            }
        }
    }

    /// The loop ticks on while the path of an execute_joint_path goal is
    /// judged. Serving the goal returns while the checker has not judged the
    /// path, and each tick until the judgment comes holds the path's arm
    /// where it stands while the other arm follows its leader. The first
    /// tick after the judgment starts the path. No tick waits on the checker:
    /// here the checker judges nothing until the test opens its gate, after
    /// a hundred ticks.
    #[tokio::test]
    async fn the_loop_ticks_on_while_a_path_is_judged() {
        let (mut channels, mut planners) = both_arms();
        let (gates, mut probes, mut path_checks) = gated_checks();
        let start = Instant::now();
        let (goal, mut done_rx) = path_goal(&channels, &["left_arm"]);
        let (started, judgment) =
            start_path_of(goal, &mut channels, &mut planners, LIVE, start).await;
        assert_eq!(started, Ok(()));
        let judgment = judgment.expect("the path waits for its judgment");
        let held = GovState::new(BOTH_HELD, ArmPair::new(1.0, 1.0));
        path_checks.judge(judgment, GuardAt::disabled(held));

        let mut reach = BOTH_HELD.right;
        reach[0] += 0.3;
        channels
            .right
            .command
            .send_replace(Some(Upstream::Joints(reach)));
        let period = Duration::from_millis(10);
        let mut now = start;
        for _ in 0..100 {
            channels.right.leader.note();
            let held = measured_at_setpoints(&planners);
            path_checks.resume_judged_by_test_guard(&mut planners, &held, now);
            let ticks = advance_arms(&mut channels, &mut planners, LIVE, now).await;
            assert_eq!(ticks.left.candidate, BOTH_HELD.left, "the path's arm holds");
            commit_both(&mut planners, &ticks.map(|_, tick| tick.candidate));
            now += period;
        }
        assert_eq!(
            planners.right.setpoint(),
            reach,
            "the other arm followed its leader all the way"
        );
        assert!(path_checks.is_judging(), "the path waits for its judgment");
        assert!(done_rx.try_recv().is_err(), "the share waits");
        assert!(channels.left.move_slot.is_claimed(), "and holds its arm");

        gates
            .send(Gate::Open)
            .expect("the checker waits at the gate");
        path_checks.resume_once_judged(&mut planners, now).await;
        assert_eq!(probes.try_recv(), Ok(Probed::Opened));
        let ticks = advance_arms(&mut channels, &mut planners, LIVE, now + period).await;
        assert_eq!(ticks.left.on_path, Some(PathArms::One(Side::Left)));
        assert_ne!(ticks.left.candidate, BOTH_HELD.left, "the path's arm moves");
    }

    /// The stop ends a path while its judgment runs, and the stop's own tick
    /// stops the judgment at its next measured probe, which frees the checker:
    /// the judgment of the next path runs, and that path starts. The test steps
    /// the checker probe by probe, its gate shut until the next path's check,
    /// so the first check ends on its stop alone.
    #[tokio::test]
    async fn a_stop_while_a_path_is_judged_stops_its_judgment() {
        let (mut channels, mut planners) = both_arms();
        let (gates, mut probes, mut path_checks) = gated_checks();
        let guard = GuardAt::disabled(GovState::new(BOTH_HELD, ArmPair::new(1.0, 1.0)));
        let now = Instant::now();
        let (goal, mut done_rx) = path_goal(&channels, &["left_arm"]);
        let (started, judgment) =
            start_path_of(goal, &mut channels, &mut planners, LIVE, now).await;
        assert_eq!(started, Ok(()));
        path_checks.judge(judgment.expect("the path waits for its judgment"), guard);
        gates
            .send(Gate::Shut)
            .expect("the checker waits at the gate");
        assert_eq!(
            first_probe(&mut probes, &mut path_checks).await,
            Probed::Shut,
            "the check runs on"
        );

        let (requests_tx, mut requests) = mpsc::channel(1);
        let (reply, stopped) = oneshot::channel();
        requests_tx
            .try_send(CoordinatorRequest::Stop {
                reason: "the operator asked".to_string(),
                reply,
            })
            .expect("queue the stop");
        serve_all(&mut requests, &mut channels, &mut planners, LIVE, now).await;
        stopped.await.expect("the loop answers the stop");
        let report = done_rx.try_recv().expect("the stop ends the path's share");
        assert_eq!(
            report.outcome,
            Outcome::Cancelled("stopped: the operator asked".to_string())
        );

        let held = measured_at_setpoints(&planners);
        path_checks.resume_judged_by_test_guard(&mut planners, &held, now);
        assert!(!path_checks.is_judging(), "no path waits for a judgment");
        gates
            .send(Gate::Shut)
            .expect("the checker waits at the gate");
        assert_eq!(
            probes.recv().await,
            Some(Probed::Stopped),
            "the next probe stops"
        );

        let (goal, _done_rx) = path_goal(&channels, &["left_arm"]);
        let (started, judgment) =
            start_path_of(goal, &mut channels, &mut planners, LIVE, now).await;
        assert_eq!(started, Ok(()));
        path_checks.judge(
            judgment.expect("the next path waits for its judgment"),
            guard,
        );
        gates
            .send(Gate::Open)
            .expect("the checker waits at the gate");
        path_checks.resume_once_judged(&mut planners, now).await;
        assert_eq!(probes.try_recv(), Ok(Probed::Opened));
        let ticks = advance_arms(
            &mut channels,
            &mut planners,
            LIVE,
            now + Duration::from_millis(10),
        )
        .await;
        assert_eq!(ticks.left.on_path, Some(PathArms::One(Side::Left)));
    }

    /// A path whose goal a stop served on the same tick ends is never judged:
    /// the stop ends the path's share, and the requests served give no
    /// judgment.
    #[tokio::test]
    async fn a_path_the_stop_ends_on_the_tick_it_starts_is_never_judged() {
        let (mut channels, mut planners) = both_arms();
        let (requests_tx, mut requests) = mpsc::channel(2);
        let (goal, mut done_rx) = path_goal(&channels, &["left_arm"]);
        let (started, answer) = oneshot::channel();
        requests_tx
            .try_send(CoordinatorRequest::StartPath { goal, started })
            .expect("queue the start");
        let (reply, stopped) = oneshot::channel();
        requests_tx
            .try_send(CoordinatorRequest::Stop {
                reason: "the operator asked".to_string(),
                reply,
            })
            .expect("queue the stop");
        let now = Instant::now();
        let judgments = serve_all(&mut requests, &mut channels, &mut planners, LIVE, now).await;
        assert!(judgments.is_empty(), "no path waits for a judgment");
        assert_eq!(answer.await.expect("the loop takes the path"), Ok(()));
        stopped.await.expect("the loop answers the stop");
        let report = done_rx.try_recv().expect("the stop ends the path's share");
        assert_eq!(
            report.outcome,
            Outcome::Cancelled("stopped: the operator asked".to_string())
        );
        assert!(!channels.left.move_slot.is_claimed(), "the arm is freed");
    }

    /// Dropping the loop's judgments stops each judgment still running: its
    /// check, here past its first probe, ends at its next one.
    #[tokio::test]
    async fn dropping_the_judgments_stops_each_running_judgment() {
        let (mut channels, mut planners) = both_arms();
        let (gates, mut probes, mut path_checks) = gated_checks();
        let (goal, _done_rx) = path_goal(&channels, &["left_arm"]);
        let (started, judgment) =
            start_path_of(goal, &mut channels, &mut planners, LIVE, Instant::now()).await;
        assert_eq!(started, Ok(()));
        let guard = GuardAt::disabled(GovState::new(BOTH_HELD, ArmPair::new(1.0, 1.0)));
        path_checks.judge(judgment.expect("the path waits for its judgment"), guard);
        gates
            .send(Gate::Shut)
            .expect("the checker waits at the gate");
        assert_eq!(
            first_probe(&mut probes, &mut path_checks).await,
            Probed::Shut,
            "the check runs"
        );
        drop(path_checks);
        gates
            .send(Gate::Shut)
            .expect("the checker waits at the gate");
        assert_eq!(probes.recv().await, Some(Probed::Stopped));
    }

    /// A path of an arm whose command lags where the arm measures holds the
    /// command where it stands while the path is judged, and starts on the
    /// judgment with the command moved onto the path's run start, where the
    /// arm measures: the path's first step runs from there.
    #[tokio::test]
    async fn a_path_starts_with_its_arms_command_moved_onto_where_the_arm_measures() {
        let (mut channels, mut planners) = left_lagging();
        let lagging = planners.left.setpoint();
        let (gates, mut probes, mut path_checks) = gated_checks();
        let start = Instant::now();
        let (goal, mut done_rx) = path_goal(&channels, &["left_arm"]);
        let (started, judgment) =
            start_path_of(goal, &mut channels, &mut planners, LIVE, start).await;
        assert_eq!(started, Ok(()));
        path_checks.judge(
            judgment.expect("the path waits for its judgment"),
            GuardAt::disabled(GovState::new(BOTH_HELD, ArmPair::new(1.0, 1.0))),
        );
        let period = Duration::from_millis(10);
        let measured = measuring_held();
        let mut now = start;
        for _ in 0..10 {
            path_checks.resume_judged_by_test_guard(&mut planners, &measured, now);
            let ticks = advance_arms(&mut channels, &mut planners, LIVE, now).await;
            assert_eq!(
                ticks.left.candidate, lagging,
                "the command holds while the path is judged"
            );
            commit_both(&mut planners, &ticks.map(|_, tick| tick.candidate));
            now += period;
        }

        gates
            .send(Gate::Open)
            .expect("the checker waits at the gate");
        path_checks
            .first_verdict()
            .await
            .expect("the check lets the path through");
        assert_eq!(probes.try_recv(), Ok(Probed::Opened));
        path_checks.resume_judged_by_test_guard(&mut planners, &measured, now);
        assert_eq!(
            planners.left.setpoint(),
            BOTH_HELD.left,
            "the command moves onto the path's run start"
        );
        let ticks = advance_arms(&mut channels, &mut planners, LIVE, now + period).await;
        assert_eq!(ticks.left.on_path, Some(PathArms::One(Side::Left)));
        let mut to = BOTH_HELD.left;
        to[0] += 0.3;
        let off = distance_to_segment(&ticks.left.candidate, &BOTH_HELD.left, &to);
        assert!(off < 1e-12, "the first step runs {off} rad off the path");
        assert_ne!(ticks.left.candidate, BOTH_HELD.left, "the path's arm moves");
        assert!(done_rx.try_recv().is_err(), "and nothing failed");
    }

    /// A path its judgment refuses fails with the judgment's failure, running
    /// none of the path, frees its arm and leaves the arm's command where it
    /// held, off where the arm measures.
    #[tokio::test]
    async fn a_refused_path_leaves_its_arms_command_where_it_held() {
        let (mut channels, mut planners) = left_lagging();
        let lagging = planners.left.setpoint();
        let start = Instant::now();
        let (goal, mut done_rx) = path_goal(&channels, &["left_arm"]);
        let (started, judgment) =
            start_path_of(goal, &mut channels, &mut planners, LIVE, start).await;
        assert_eq!(started, Ok(()));
        let mut path_checks = PathChecks::new(Arc::new(Mutex::new(Refusing)));
        path_checks.judge(
            judgment.expect("the path waits for its judgment"),
            GuardAt::disabled(GovState::new(BOTH_HELD, ArmPair::new(1.0, 1.0))),
        );
        let refused = Failure::new(FailureKind::Unreachable, ARMS_WOULD_MEET);
        assert_eq!(path_checks.first_verdict().await, Err(refused.clone()));
        path_checks.resume_judged_by_test_guard(&mut planners, &measuring_held(), start);
        let report = done_rx.try_recv().expect("the path ends on this tick");
        assert_eq!(report.outcome, Outcome::Failed(refused));
        assert!(!channels.left.move_slot.is_claimed(), "the arm is freed");
        assert!(!path_checks.is_judging(), "no path waits for a judgment");
        let ticks = advance_arms(&mut channels, &mut planners, LIVE, start).await;
        assert_eq!(
            ticks.left.candidate, lagging,
            "the command holds where it held"
        );
    }

    /// A checker that panics.
    struct Panicking;

    impl PathGuard for Panicking {
        fn check(
            &mut self,
            _arms: PathArms,
            _guard: &GuardAt,
            _path: &RoundedPath,
            _stop: &AtomicBool,
        ) -> Result<(), String> {
            panic!("the test's checker panics");
        }
    }

    /// A path whose checker panics fails with `attempt_failed` once the loop
    /// finds the verdict its checker's thread dropped, and frees its arm.
    #[test]
    fn a_path_whose_checker_panics_fails_with_attempt_failed() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a runtime");
        let (mut channels, mut planners) = both_arms();
        let (goal, mut done_rx) = path_goal(&channels, &["left_arm"]);
        let mut path_checks = PathChecks::new(Arc::new(Mutex::new(Panicking)));
        let now = Instant::now();
        runtime.block_on(async {
            let (started, judgment) =
                start_path_of(goal, &mut channels, &mut planners, LIVE, now).await;
            assert_eq!(started, Ok(()));
            path_checks.judge(
                judgment.expect("the path waits for its judgment"),
                GuardAt::disabled(GovState::new(BOTH_HELD, ArmPair::new(1.0, 1.0))),
            );
        });
        // Dropping the runtime waits for its blocking threads: the checker's
        // thread has panicked and dropped the verdict.
        drop(runtime);
        path_checks.resume_judged_by_test_guard(&mut planners, &measuring_held(), now);
        let report = done_rx.try_recv().expect("the path ends");
        assert_eq!(
            report.outcome,
            Outcome::Failed(Failure::new(
                FailureKind::AttemptFailed,
                "the check of the path against the robot's body failed"
            ))
        );
        assert!(!channels.left.move_slot.is_claimed(), "the arm is freed");
        assert!(!path_checks.is_judging(), "no path waits for a judgment");
    }

    /// The silence after which a test follower's arm freezes.
    const STALE_LIMIT: Duration = Duration::from_millis(100);

    /// Run the first half of the tick ([`admit_and_serve`]) on which the
    /// right arm's follower delivers again after a silence that froze its
    /// arm, the left one delivering throughout, each measuring its arm of
    /// `measuring`, with `requests` queued and each path that starts judged on
    /// `path_checks` against a disabled guard: that tick's admissions and its
    /// instant.
    async fn serve_as_right_recovers(
        channels: &mut ArmPair<ArmChannels>,
        followers: &ArmPair<Follower>,
        planners: &mut ArmPair<Planner>,
        measuring: ArmPair<JointVec>,
        requests: Vec<CoordinatorRequest>,
        path_checks: &mut PathChecks,
    ) -> (ArmPair<Admission>, Instant) {
        let start = Instant::now();
        let mut liveness = ArmPair::from_fn(|_| Liveness::seeded(start));
        let mut cadence = ArmPair::from_fn(|_| Cadence::new(100, start));
        for side in Side::BOTH {
            let _ = channels[side].measured.borrow_and_update();
        }
        let silent = start + 2 * STALE_LIMIT;
        followers.left.send_replace(standing_at(measuring.left));
        let frozen = admit_arms(
            &mut liveness,
            &mut cadence,
            channels,
            planners,
            silent,
            STALE_LIMIT,
        );
        assert_eq!(frozen, ArmPair::new(Admission::Live, Admission::Stale));
        let _ = channels.left.measured.borrow_and_update();
        for side in Side::BOTH {
            followers[side].send_replace(standing_at(measuring[side]));
        }
        let (requests_tx, mut queued) = mpsc::channel(requests.len().max(1));
        for request in requests {
            requests_tx.try_send(request).expect("queue the request");
        }
        let back = silent + Duration::from_millis(10);
        let state = TickState {
            arm_liveness: &mut liveness,
            arm_cadence: &mut cadence,
            channels,
            planners,
            commanded_grippers: &ArmPair::from_fn(|_| CommandedOpening::idle(1.0)),
            requests: &mut queued,
            gripper_moves: &mut ArmPair::new(None, None),
            path_checks,
        };
        let admission = admit_and_serve(
            state,
            &|_: &watch::Receiver<Option<GripperState>>| 0.0,
            GuardAt::disabled,
            back,
            STALE_LIMIT,
        )
        .await;
        assert_eq!(
            admission,
            ArmPair::new(Admission::Live, Admission::Reanchor)
        );
        (admission, back)
    }

    /// A plan check served on the tick the arm's follower recovers plans from
    /// the arm's measured pose, where the tick re-anchors it, as a move_arm
    /// goal the arm's planner takes on that tick does.
    #[tokio::test]
    async fn a_plan_check_as_the_follower_recovers_plans_from_where_the_arm_stands() {
        let (mut channels, followers, mut planners) = followed_arms();
        let mut recovered = BOTH_HELD.right;
        recovered[0] += 0.2;
        let pose = planner_holding(Side::Right, recovered).world_ee_pose(&recovered);
        let target = Isometry3::from_parts(
            (pose.translation.vector + Vector3::new(0.0, 0.0, 0.02)).into(),
            pose.rotation,
        );
        let request = ArmMoveRequest {
            side: Side::Right,
            position: target.translation,
            orientation: Some(target.rotation),
            duration: Duration::ZERO,
        };
        let check_from =
            |held: JointVec| planner_holding(Side::Right, held).check_cartesian(&request);
        let (from_recovered, from_frozen) = (check_from(recovered), check_from(BOTH_HELD.right));
        assert_ne!(from_recovered, from_frozen, "the two starts plan apart");
        let (reply, answer) = oneshot::channel();
        serve_as_right_recovers(
            &mut channels,
            &followers,
            &mut planners,
            ArmPair::new(BOTH_HELD.left, recovered),
            vec![CoordinatorRequest::CheckArmMove { request, reply }],
            &mut PathChecks::new(Arc::new(Mutex::new(OpenGuard))),
        )
        .await;
        assert_eq!(answer.await.expect("the loop answers"), from_recovered);
    }

    /// A checker that lets every path through, recording the guard each
    /// path is checked against and where each path starts.
    #[derive(Default)]
    struct Recording(Vec<(GuardAt, Vec<f64>)>);

    impl PathGuard for Recording {
        fn check(
            &mut self,
            _arms: PathArms,
            guard: &GuardAt,
            path: &RoundedPath,
            _stop: &AtomicBool,
        ) -> Result<(), String> {
            self.0.push((*guard, path.at(0.0)));
            Ok(())
        }
    }

    /// A path whose arm measures elsewhere once its judgment comes, within
    /// its start tolerance, is checked again from where the arm stands,
    /// against the guard as the tick holds it, the command holding, and
    /// starts once that judgment comes, its command moved onto where the arm
    /// measures: the left shoulder moves 0.005 rad, and the right arm's
    /// command 0.2 rad between the two checks.
    #[tokio::test]
    async fn a_path_whose_arm_moved_while_judged_is_checked_again_from_where_it_stands() {
        let (mut channels, mut planners) = both_arms();
        let recording = Arc::new(Mutex::new(Recording::default()));
        let mut path_checks = PathChecks::new(recording.clone());
        let start = Instant::now();
        let (goal, _done_rx) = path_goal(&channels, &["left_arm"]);
        let (started, judgment) =
            start_path_of(goal, &mut channels, &mut planners, LIVE, start).await;
        assert_eq!(started, Ok(()));
        let judgment = judgment.expect("the path waits for its judgment");
        path_checks.judge(judgment, test_guard(&planners));
        path_checks
            .first_verdict()
            .await
            .expect("the check lets the path through");
        let mut moved = BOTH_HELD.left;
        moved[0] += 0.005;
        let measured = ArmPair::new(Ok(moved), Ok(BOTH_HELD.right));
        let mut right = BOTH_HELD.right;
        right[0] += 0.2;
        planners.right.seed_from_measured(right);
        path_checks.resume_judged_by_test_guard(&mut planners, &measured, start);
        assert!(
            path_checks.is_judging(),
            "the path waits for its next judgment"
        );
        assert_eq!(
            planners.left.setpoint(),
            BOTH_HELD.left,
            "the command holds"
        );
        path_checks
            .first_verdict()
            .await
            .expect("the check lets the path through again");
        path_checks.resume_judged_by_test_guard(&mut planners, &measured, start);
        assert!(!path_checks.is_judging(), "the path starts");
        assert_eq!(
            planners.left.setpoint(),
            moved,
            "the command moves onto where the arm measures"
        );
        let recorded = recording.lock().expect("the checks ended");
        let starts: Vec<&[f64]> = recorded
            .0
            .iter()
            .map(|(_, path_start)| &path_start[..ARM_DOF])
            .collect();
        assert_eq!(starts, [&BOTH_HELD.left[..], &moved[..]]);
        let guards: Vec<GuardAt> = recorded.0.iter().map(|(guard, _)| *guard).collect();
        let right_at = |right| {
            GuardAt::disabled(GovState::new(
                ArmPair::new(BOTH_HELD.left, right),
                ArmPair::new(1.0, 1.0),
            ))
        };
        assert_eq!(guards, [right_at(BOTH_HELD.right), right_at(right)]);
    }

    /// A path of the left arm that starts on the tick the right arm's
    /// follower recovers is checked with the right arm where it stands: the
    /// tick re-anchors the right arm on its measured pose as it admits the
    /// followers, before it serves the path's goal.
    #[tokio::test]
    async fn a_path_that_starts_as_the_other_follower_recovers_is_checked_where_that_arm_stands() {
        let (mut channels, followers, mut planners) = followed_arms();
        let mut recovered = BOTH_HELD.right;
        recovered[0] += 0.2;
        let recording = Arc::new(Mutex::new(Recording::default()));
        let mut path_checks = PathChecks::new(recording.clone());
        let (goal, _done_rx) = path_goal(&channels, &["left_arm"]);
        let (started, answer) = oneshot::channel();
        serve_as_right_recovers(
            &mut channels,
            &followers,
            &mut planners,
            ArmPair::new(BOTH_HELD.left, recovered),
            vec![CoordinatorRequest::StartPath { goal, started }],
            &mut path_checks,
        )
        .await;
        assert_eq!(answer.await.expect("the loop takes the path"), Ok(()));
        path_checks
            .first_verdict()
            .await
            .expect("the check lets the path through");
        let recorded = recording.lock().expect("the check ended");
        let [(guard, path_start)] = &recorded.0[..] else {
            panic!("the path is checked once: {:?}", recorded.0);
        };
        assert_eq!(
            *guard,
            GuardAt::disabled(GovState::new(
                ArmPair::new(BOTH_HELD.left, recovered),
                ArmPair::new(1.0, 1.0)
            ))
        );
        assert_eq!(
            path_start[ARM_DOF..],
            recovered,
            "the path holds the right arm there"
        );
    }

    /// A path of both arms that starts on the tick the right arm's follower
    /// recovers runs from where both arms measure: the recovery re-anchors
    /// the right arm's command on its measured pose, the left arm's command
    /// holds off its measured pose while the path is judged, and the path
    /// starts with both commands moved onto its run start. Every command
    /// from then on lies on the path from there.
    #[tokio::test]
    async fn a_path_of_both_arms_that_starts_as_a_follower_recovers_runs_from_where_both_measure() {
        let (mut channels, followers, mut planners) = followed_arms();
        let mut recovered = BOTH_HELD.right;
        recovered[0] += 0.2;
        let mut left = BOTH_HELD.left;
        left[0] += 0.3;
        let start = ArmPair::new(left, recovered);
        let (gates, mut probes, mut path_checks) = gated_checks();
        let (goal, _done_rx) = path_goal(&channels, &["left_arm", "right_arm"]);
        let (started, answer) = oneshot::channel();
        let (admission, now) = serve_as_right_recovers(
            &mut channels,
            &followers,
            &mut planners,
            start,
            vec![CoordinatorRequest::StartPath { goal, started }],
            &mut path_checks,
        )
        .await;
        assert_eq!(answer.await.expect("the loop takes the path"), Ok(()));
        assert_eq!(
            ArmPair::from_fn(|side| planners[side].setpoint()),
            ArmPair::new(BOTH_HELD.left, recovered),
            "the recovery re-anchors the right arm's command and the left one holds"
        );
        gates
            .send(Gate::Open)
            .expect("the checker waits at the gate");
        path_checks
            .first_verdict()
            .await
            .expect("the check lets the path through");
        assert_eq!(probes.try_recv(), Ok(Probed::Opened));

        let measured = start.map(|_, q| Ok(q));
        path_checks.resume_judged_by_test_guard(&mut planners, &measured, now);
        assert_eq!(
            ArmPair::from_fn(|side| planners[side].setpoint()),
            start,
            "both commands move onto the path's run start"
        );
        let to = BOTH_HELD.map(|_, mut q| {
            q[0] += 0.3;
            q
        });
        let (from, to) = (dual_joints(&start), dual_joints(&to));
        let period = Duration::from_millis(10);
        let mut tick_admission = admission;
        for k in 0..10 {
            let at = now + period * k;
            let ticks = advance_arms(&mut channels, &mut planners, tick_admission, at).await;
            tick_admission = LIVE;
            let candidates = chase_together(&planners, ticks.map(|_, tick| tick.candidate));
            commit_both(&mut planners, &candidates);
            let commanded = dual_joints(&ArmPair::from_fn(|side| planners[side].setpoint()));
            let off = distance_to_segment(&commanded, &from, &to);
            assert!(off < 1e-12, "tick {k} commands {off} rad off the path");
        }
        assert_ne!(
            ArmPair::from_fn(|side| planners[side].setpoint()),
            start,
            "the path moves the arms"
        );
    }

    /// Both arms on their shares of one path of both arms step along that
    /// path; otherwise each arm on a path steps along its own.
    #[test]
    fn the_arms_step_along_one_path_only_when_both_follow_its_shares() {
        let both = Some(PathArms::Both);
        assert_eq!(paths_of(ArmPair::new(both, both)), [PathArms::Both]);
        assert_eq!(
            paths_of(ArmPair::new(both, None)),
            [PathArms::One(Side::Left)]
        );
        assert_eq!(
            paths_of(ArmPair::new(Some(PathArms::One(Side::Left)), both)),
            [PathArms::One(Side::Left), PathArms::One(Side::Right)]
        );
        assert!(paths_of(ArmPair::new(None, None)).is_empty());
    }
}
