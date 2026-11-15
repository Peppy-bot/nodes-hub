//! Gripper move execution: an accepted gripper goal ([`GripperGoal`]), a
//! `move_gripper` goal or the close of a gripper for a `move_to_home` goal,
//! run as a [`GripperMove`] through the same per-tick governing as every
//! other DOF. Each tick the coordinator services each arm here: a queued
//! goal is admitted into a free arm, and the move in flight ends once the
//! jaws stand still from the planned end of the commanded travel on, at its
//! settle timeout, at its travel timeout when it has one, on a cancel, or on
//! a follower gone stale. The coordinator hands in the opening the governor
//! published last tick ([`CommandedOpening`]) and the opening the follower
//! measured ([`MeasuredOpening`]). The move holds its arm's move slot until
//! it ends.

use std::time::{Duration, Instant};

use peppygen::exposed_actions::limb_motion::move_gripper;
use robot_description::ArmId;
use tokio::sync::mpsc;
use tracing::{error, warn};

use crate::failure::{Failure, FailureKind, stale_failure};
use crate::liveness::Admission;
use crate::move_slot::{MoveSlot, SlotRelease};
use crate::planner::{Outcome, POSTURE_SETTLE, PostureShare};
use crate::robot::LIMBS;

/// The opening of a closed gripper.
const CLOSED_OPENING: f64 = 0.0;

/// The longest the close of a gripper for a `move_to_home` goal waits, after
/// the planned end of the commanded travel, for the jaws to stand still: the
/// settle timeout of an arm's share of a posture goal.
const CLOSE_SETTLE_TIMEOUT: Duration = POSTURE_SETTLE.timeout;

/// The longest the commanded travel of the close of a gripper for a
/// `move_to_home` goal may take. The operator sets the opening rate while the
/// robot runs, down to a rate at which the jaws take 100 s to close. With
/// [`CLOSE_SETTLE_TIMEOUT`] this bounds the close at 10 s, the time the
/// postures contract gives it, so a caller that waits for the longest arm
/// move and the close waits long enough.
const CLOSE_TRAVEL_TIMEOUT: Duration = Duration::from_secs(8);

/// The effort cap a gripper goal relays to the follower.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EffortCap {
    /// The cap the goal names, validated non-negative, or `None` when the
    /// goal carried no preference, which leaves the follower's configured
    /// ceiling in charge.
    Named(Option<f64>),
    /// The cap of the last setpoint the gripper got, from a move or from a
    /// leader: the goal changes the opening and keeps the effort, so a grasp
    /// keeps the effort its caller gave it.
    Kept,
}

impl EffortCap {
    /// The cap to relay, `kept` being the cap of the last setpoint the
    /// gripper got.
    pub fn relayed(self, kept: Option<f64>) -> Option<f64> {
        match self {
            Self::Named(cap) => cap,
            Self::Kept => kept,
        }
    }
}

/// An accepted gripper goal handed to the coordinator, which executes it
/// through the same per-tick governing as everything else (the gripper analog of
/// [`Goal`](crate::planner::Goal) for the arms). The opening is the validated
/// goal fraction.
pub struct GripperGoal {
    pub opening: f64,
    pub effort: EffortCap,
    /// The longest the move waits, after the planned end of the commanded
    /// travel, for the jaws to stand still.
    pub settle_timeout: Duration,
    /// The longest the commanded travel may take from the admission of the
    /// move, or `None` for a move that travels as long as the opening rate
    /// makes it.
    pub travel_timeout: Option<Duration>,
    pub reply: GripperReply,
}

impl GripperGoal {
    /// The close of the gripper of `arm` for a `move_to_home` goal, which
    /// `share` reports to: to [`CLOSED_OPENING`] under the effort cap of the
    /// last setpoint the gripper got, its travel ending within
    /// [`CLOSE_TRAVEL_TIMEOUT`] and its jaws settling within
    /// [`CLOSE_SETTLE_TIMEOUT`].
    pub fn close(arm: ArmId, share: PostureShare) -> Self {
        Self {
            opening: CLOSED_OPENING,
            effort: EffortCap::Kept,
            settle_timeout: CLOSE_SETTLE_TIMEOUT,
            travel_timeout: Some(CLOSE_TRAVEL_TIMEOUT),
            reply: GripperReply::Close { arm, share },
        }
    }

    /// Fail unstarted with `failure`, with no result. The move slot is the
    /// caller's concern.
    pub async fn fail(self, failure: Failure) {
        self.reply.fail(failure).await;
    }

    /// End unstarted on a follower that delivers no fresh state. A
    /// `move_gripper` goal fails with the failure of a stale limb. The close
    /// of a `move_to_home` goal completes with no command sent: the goal
    /// leaves a gripper it cannot see as it stands, so a robot with a silent
    /// gripper still parks its arms. The move slot is the caller's concern.
    pub async fn end_on_stale_follower(self) {
        match self.reply {
            GripperReply::MoveGripper(ctx) => fail_gripper_goal(&ctx, stale_failure()).await,
            GripperReply::Close { arm, share } => {
                let gripper = LIMBS.gripper_name(arm);
                warn!("move_to_home: {gripper} reports no state and gets no close");
                share.report(gripper, Outcome::Complete).await;
            }
        }
    }

    /// End unstarted as cancelled by the stop service, for `reason`, which
    /// names the stop, reporting `reported_frac`. The move slot is the
    /// caller's concern.
    pub async fn stop(self, reason: &str, reported_frac: f64) {
        let stopped = GripperTerminal::Cancelled(reason.to_string());
        self.reply.finish(stopped, reported_frac, 0.0).await;
    }
}

/// Where a gripper move reports its terminal: the `move_gripper` action goal
/// that carried it (boxed: the goal context dwarfs the other variant), or
/// the share of the gripper of `arm` in the close of a `move_to_home` goal.
pub enum GripperReply {
    MoveGripper(Box<move_gripper::GoalContext>),
    Close { arm: ArmId, share: PostureShare },
}

impl GripperReply {
    /// The reason of the cancel the move received, or `None` before one: the
    /// cause of the `move_gripper` goal's cancel signal, or the cancel of the
    /// `move_to_home` goal, which its action flags.
    fn cancel_reason(&self) -> Option<String> {
        match self {
            Self::MoveGripper(ctx) => ctx.cancel_cause().map(|cause| cause.reason().to_string()),
            Self::Close { share, .. } => share.cancel_reason(),
        }
    }

    /// End the move on `terminal`, with `measured_frac`, the opening the
    /// gripper measured then, and `elapsed_s`, the time since its admission.
    /// A `move_gripper` goal that completes or ends cancelled reports them
    /// as `final_opening` and `action_time`, and `settled` true only when
    /// the jaws stood still; a follower gone stale, or a travel that ran out
    /// of its timeout, fails it with no result. A close reports its
    /// [`Outcome`] to its `move_to_home` goal, and no opening.
    async fn finish(self, terminal: GripperTerminal, measured_frac: f64, elapsed_s: f64) {
        let ctx = match self {
            Self::MoveGripper(ctx) => ctx,
            Self::Close { arm, share } => {
                return share
                    .report(LIMBS.gripper_name(arm), Outcome::from(terminal))
                    .await;
            }
        };
        let ended = match terminal {
            GripperTerminal::Settled => ctx.complete(true, measured_frac, elapsed_s).await,
            GripperTerminal::SettleTimedOut => ctx.complete(false, measured_frac, elapsed_s).await,
            GripperTerminal::FollowerStale => {
                return fail_gripper_goal(&ctx, stale_failure()).await;
            }
            GripperTerminal::TravelTimedOut(timeout) => {
                return fail_gripper_goal(&ctx, travel_timeout_failure(timeout)).await;
            }
            GripperTerminal::Cancelled(reason) => {
                ctx.complete_cancelled(reason, false, measured_frac, elapsed_s)
                    .await
            }
        };
        if let Err(e) = ended {
            error!("move_gripper end: {e}");
        }
    }

    /// Fail the move with `failure`, with no result.
    async fn fail(self, failure: Failure) {
        match self {
            Self::MoveGripper(ctx) => fail_gripper_goal(&ctx, failure).await,
            Self::Close { arm, share } => {
                share
                    .report(LIMBS.gripper_name(arm), Outcome::Failed(failure))
                    .await;
            }
        }
    }
}

/// How the close of a gripper ends for its `move_to_home` goal. The close
/// completes as a gripper move of limb_motion completes: once the jaws stand
/// still, closed or on an item that they hold, and at the settle timeout,
/// where they stand. It fails when its follower goes stale, and when its
/// commanded travel runs out of its timeout.
impl From<GripperTerminal> for Outcome {
    fn from(terminal: GripperTerminal) -> Self {
        match terminal {
            GripperTerminal::Settled | GripperTerminal::SettleTimedOut => Self::Complete,
            GripperTerminal::FollowerStale => Self::Failed(stale_failure()),
            GripperTerminal::TravelTimedOut(timeout) => {
                Self::Failed(travel_timeout_failure(timeout))
            }
            GripperTerminal::Cancelled(reason) => Self::Cancelled(reason),
        }
    }
}

/// The failure of a gripper move whose commanded travel did not end within
/// `timeout`, its travel timeout.
fn travel_timeout_failure(timeout: Duration) -> Failure {
    Failure::new(
        FailureKind::Timeout,
        format!(
            "the commanded travel of the jaws did not end within {} s: the opening rate \
             (max_gripper_rate_frac_s) is too low for the move, or the collision guard slows \
             the jaws",
            timeout.as_secs_f64()
        ),
    )
}

/// Fail a `move_gripper` goal with `failure`, with no result.
async fn fail_gripper_goal(ctx: &move_gripper::GoalContext, failure: Failure) {
    if let Err(e) = ctx.fail(failure.kind.into(), failure.message).await {
        error!("move_gripper end: {e}");
    }
}

/// A backbone-executed gripper move in flight: the motion its terminal is
/// decided from, the effort cap it relays, and where it reports its
/// terminal. The move holds the arm's move slot until it ends, so a second
/// gripper goal on the arm is refused until then; the slot release frees
/// the slot on any exit.
pub struct GripperMove {
    motion: GripperMotion,
    effort: EffortCap,
    reply: GripperReply,
    _move_slot: SlotRelease,
}

impl GripperMove {
    /// The opening the move commands.
    pub fn target_frac(&self) -> f64 {
        self.motion.target_frac
    }

    /// The effort cap the move relays.
    pub fn effort(&self) -> EffortCap {
        self.effort
    }
}

/// What a gripper move's terminal is decided from (see
/// [`gripper_move_terminal`]): the target, when it was admitted, the planned
/// end of the commanded travel, the goal's settle and travel timeouts, and
/// the run of openings the jaws are judged still on. The commanded opening
/// ramps to the target through the governor at the opening rate the operator
/// sets, re-read every tick, and holds it while the jaws follow.
#[derive(Clone, Copy, Debug)]
struct GripperMotion {
    target_frac: f64,
    admitted: Instant,
    /// The first tick of the commanded travel's current rest, on the target
    /// or held by the collision guard. `None` while it travels.
    planned_end: Option<Instant>,
    settle_timeout: Duration,
    /// The longest the commanded travel may take from the admission, or
    /// `None` for no bound.
    travel_timeout: Option<Duration>,
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
    /// A move to `target_frac` admitted at `now`, whose commanded travel no
    /// timeout bounds.
    fn new(target_frac: f64, settle_timeout: Duration, now: Instant) -> Self {
        Self {
            target_frac,
            admitted: now,
            planned_end: None,
            settle_timeout,
            travel_timeout: None,
            still: None,
        }
    }

    /// Whether the commanded travel, which still runs at `now`, ran out of
    /// the move's travel timeout, and that timeout.
    fn travel_timed_out(&self, now: Instant) -> Option<Duration> {
        self.travel_timeout
            .filter(|timeout| now.duration_since(self.admitted) >= *timeout)
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
pub struct MeasuredOpening {
    pub frac: f64,
    pub fresh: bool,
}

/// A gripper's commanded opening as one tick sees it: the opening the
/// governor published last tick, and whether its collision guard held the
/// opening where it stood then.
#[derive(Clone, Copy, Debug)]
pub struct CommandedOpening {
    pub frac: f64,
    held: bool,
}

/// The largest part of the chase's step the governor may let through for the
/// guard to count as holding the opening: a hold leaves float dust, orders of
/// magnitude under any throttle, which still moves the opening and is travel.
const GUARD_HOLD_STEP_FRAC: f64 = 0.01;

impl CommandedOpening {
    /// An idle arm's opening, re-anchored on the measured gripper.
    pub fn idle(frac: f64) -> Self {
        Self { frac, held: false }
    }

    /// The opening the governor published, `governed`, from the chase's step
    /// from `prev` to `chased`: held when the governor let through under
    /// [`GUARD_HOLD_STEP_FRAC`] of that step.
    pub fn published(governed: f64, prev: f64, chased: f64) -> Self {
        Self {
            frac: governed,
            held: (governed - prev).abs() < GUARD_HOLD_STEP_FRAC * (chased - prev).abs(),
        }
    }
}

/// How a gripper move ended.
#[derive(Debug, PartialEq)]
pub enum GripperTerminal {
    /// The measured opening stood still, on the target or short of it where
    /// an object or the effort cap stopped the jaws.
    Settled,
    /// The settle timeout passed before the jaws stood still; the move
    /// completes where they stand, with `settled` false.
    SettleTimedOut,
    /// The follower stopped reporting; the move fails with
    /// `resource_unavailable`.
    FollowerStale,
    /// The commanded travel did not end within the move's travel timeout,
    /// this long; the move fails with `timeout`.
    TravelTimedOut(Duration),
    /// The goal was cancelled, for this reason: by its caller, or by the
    /// stop service, whose reason names the stop.
    Cancelled(String),
}

/// Decide whether a gripper move ends this tick, and how. No I/O and no
/// clock: everything the decision reads is passed in, and the only state it
/// writes is the move's own planned end and still run.
///
/// A cancel, `cancel` giving its reason, ends the move ahead of any other
/// verdict of the same tick, and a follower gone stale fails it: the move
/// can neither progress nor be verified. While the commanded travel runs
/// ([`GripperMotion::travel_end`], fed `commanded`, the opening the governor
/// published last tick), only the travel timeout of the move is judged, and
/// a travel that ran out of it fails the move
/// ([`GripperMotion::travel_timed_out`]). From the planned end on the move completes as soon
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
    let Some(planned_end) = motion.travel_end(commanded, now) else {
        return motion
            .travel_timed_out(now)
            .map(GripperTerminal::TravelTimedOut);
    };
    if measured.fresh && motion.stands_still(measured.frac, now) {
        return Some(GripperTerminal::Settled);
    }
    let since_end = now.duration_since(planned_end);
    (since_end >= motion.settle_timeout).then_some(GripperTerminal::SettleTimedOut)
}

/// Admit a queued gripper goal into a free arm, and end an in-flight move
/// on the terminal [`gripper_move_terminal`] decides. A queued goal does not
/// start, and frees the move slot its admission claimed:
/// - after a stop that came since its admission ([`MoveSlot::take_stop`]):
///   it ends cancelled for the stop's reason;
/// - while the follower is stale ([`GripperGoal::end_on_stale_follower`]).
///
/// A goal that finds another gripper move in flight fails. `commanded` is
/// the opening the governor published last tick and whether the guard held
/// it. A terminal with a result reports the opening measured that tick as
/// `final_opening` and the time since admission as `action_time`. The move
/// releases the move slot on every end.
pub async fn service_gripper_move(
    mv: &mut Option<GripperMove>,
    goals: &mut mpsc::Receiver<GripperGoal>,
    move_slot: &MoveSlot,
    commanded: CommandedOpening,
    measured: MeasuredOpening,
    admission: Admission,
    now: Instant,
) {
    // Drain fully: every queued goal is answered, never left parked.
    while let Ok(goal) = goals.try_recv() {
        if let Some(reason) = move_slot.take_stop() {
            let _release = move_slot.release_on_drop();
            goal.stop(&reason, measured.frac).await;
        } else if admission == Admission::Stale {
            let _release = move_slot.release_on_drop();
            goal.end_on_stale_follower().await;
        } else if mv.is_none() {
            *mv = Some(GripperMove {
                motion: GripperMotion {
                    travel_timeout: goal.travel_timeout,
                    ..GripperMotion::new(goal.opening, goal.settle_timeout, now)
                },
                effort: goal.effort,
                reply: goal.reply,
                _move_slot: move_slot.release_on_drop(),
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
    let cancel = m.reply.cancel_reason();
    let Some(terminal) =
        gripper_move_terminal(&mut m.motion, commanded, measured, admission, cancel, now)
    else {
        return;
    };
    let m = mv.take().expect("in-flight move checked above");
    end_gripper_move(m, terminal, measured.frac, now).await;
}

/// End a gripper move on `terminal` at `now` ([`GripperReply::finish`]), with
/// `measured_frac`, the opening the gripper measured then, and the time
/// since the move's admission. The move slot releases with the move.
pub async fn end_gripper_move(
    m: GripperMove,
    terminal: GripperTerminal,
    measured_frac: f64,
    now: Instant,
) {
    let GripperMove {
        motion,
        reply,
        _move_slot,
        ..
    } = m;
    let elapsed_s = now.duration_since(motion.admitted).as_secs_f64();
    reply.finish(terminal, measured_frac, elapsed_s).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::LEFT;

    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use peppygen::fixtures::exposed_actions::limb_motion::move_gripper as fixture;
    use peppygen::fixtures::harness::{Config, Harness};

    use crate::test_support::{claimed_move_slot, params};

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

    // The close of a gripper for a move_to_home goal: a goal that reports to
    // the share of its posture goal.

    /// A close of the left gripper, where its share reports, and the cancel
    /// flag of its posture goal.
    fn left_close() -> (GripperGoal, mpsc::Receiver<Outcome>, Arc<AtomicBool>) {
        let (done_tx, ends) = mpsc::channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let share = PostureShare {
            done_tx,
            cancelled: cancelled.clone(),
        };
        (GripperGoal::close(LEFT, share), ends, cancelled)
    }

    /// One tick of an arm that `goal` waits on, at `now`: the goal is
    /// admitted, or the move in flight judged.
    async fn tick_close(
        mv: &mut Option<GripperMove>,
        goals: &mut mpsc::Receiver<GripperGoal>,
        move_slot: &MoveSlot,
        commanded: CommandedOpening,
        measured: MeasuredOpening,
        now: Instant,
    ) {
        service_gripper_move(
            mv,
            goals,
            move_slot,
            commanded,
            measured,
            Admission::Live,
            now,
        )
        .await;
    }

    /// A close keeps the effort cap of the last setpoint its gripper got,
    /// and the whole of it takes at most 10 s: 8 s of commanded travel and
    /// 2 s for the jaws to stand still.
    #[test]
    fn a_close_goes_to_a_closed_gripper_and_keeps_its_effort_cap() {
        let (goal, _ends, _cancelled) = left_close();
        assert_eq!((goal.opening, goal.effort), (0.0, EffortCap::Kept));
        assert_eq!(goal.settle_timeout, Duration::from_secs(2));
        assert_eq!(goal.travel_timeout, Some(Duration::from_secs(8)));
    }

    #[test]
    fn a_kept_effort_cap_relays_the_cap_of_the_last_setpoint() {
        for kept in [None, Some(0.4)] {
            assert_eq!(EffortCap::Kept.relayed(kept), kept);
            assert_eq!(EffortCap::Named(Some(1.5)).relayed(kept), Some(1.5));
            assert_eq!(EffortCap::Named(None).relayed(kept), None);
        }
    }

    #[test]
    fn a_travel_that_runs_out_of_its_timeout_fails_the_move() {
        // The operator set an opening rate at which the jaws need 50 s: the
        // commanded opening still travels when the timeout runs out.
        let start = Instant::now();
        let timeout = Duration::from_secs(8);
        let mut motion = GripperMotion {
            travel_timeout: Some(timeout),
            ..GripperMotion::new(0.0, MOVE_TIMEOUT, start)
        };
        let ends = |m: &mut GripperMotion, commanded, now| {
            gripper_move_terminal(m, commanded, fresh(0.5), Admission::Live, None, now)
        };
        assert_eq!(ends(&mut motion, free(0.49), start + ms(1000)), None);
        assert_eq!(
            ends(&mut motion, free(0.43), start + timeout - ms(1)),
            None,
            "the travel has its whole timeout"
        );
        assert_eq!(
            ends(&mut motion, free(0.42), start + timeout),
            Some(GripperTerminal::TravelTimedOut(timeout))
        );
    }

    #[test]
    fn a_travel_that_ends_within_its_timeout_is_judged_on_its_settle_alone() {
        // The commanded opening lands just inside the travel timeout, and the
        // jaws then have the whole settle timeout to stand still.
        let start = Instant::now();
        let timeout = Duration::from_secs(8);
        let mut motion = GripperMotion {
            travel_timeout: Some(timeout),
            ..GripperMotion::new(0.0, MOVE_TIMEOUT, start)
        };
        let ends = |m: &mut GripperMotion, measured: f64, now| {
            gripper_move_terminal(m, free(0.0), fresh(measured), Admission::Live, None, now)
        };
        let landed = start + timeout - ms(10);
        assert_eq!(ends(&mut motion, 0.3, landed), None);
        assert_eq!(
            ends(&mut motion, 0.2, start + timeout + ms(500)),
            None,
            "past the travel timeout, the jaws still settle"
        );
        assert_eq!(
            ends(&mut motion, 0.1, landed + MOVE_TIMEOUT),
            Some(GripperTerminal::SettleTimedOut)
        );
    }

    #[test]
    fn a_move_with_no_travel_timeout_travels_as_long_as_its_rate_makes_it() {
        let start = Instant::now();
        let mut motion = GripperMotion::new(MOVE_TARGET, MOVE_TIMEOUT, start);
        assert_eq!(
            gripper_move_terminal(
                &mut motion,
                free(0.9),
                fresh(0.9),
                Admission::Live,
                None,
                start + Duration::from_secs(3600)
            ),
            None
        );
    }

    #[test]
    fn a_close_completes_on_still_jaws_and_at_its_timeout_alike() {
        assert_eq!(Outcome::from(GripperTerminal::Settled), Outcome::Complete);
        assert_eq!(
            Outcome::from(GripperTerminal::SettleTimedOut),
            Outcome::Complete
        );
        assert_eq!(
            Outcome::from(GripperTerminal::FollowerStale),
            Outcome::Failed(stale_failure())
        );
        assert_eq!(
            Outcome::from(GripperTerminal::TravelTimedOut(Duration::from_secs(8))),
            Outcome::Failed(Failure::new(
                FailureKind::Timeout,
                "the commanded travel of the jaws did not end within 8 s: the opening rate \
                 (max_gripper_rate_frac_s) is too low for the move, or the collision guard \
                 slows the jaws"
            ))
        );
        assert_eq!(
            Outcome::from(GripperTerminal::Cancelled("stopped".to_string())),
            Outcome::Cancelled("stopped".to_string())
        );
    }

    /// A close that an item stops completes once the jaws stand still on
    /// it, and frees its gripper.
    #[tokio::test]
    async fn a_close_whose_jaws_stand_still_on_an_item_completes() {
        let move_slot = claimed_move_slot();
        let (queue, mut goals) = mpsc::channel(1);
        let (goal, mut ends, _cancelled) = left_close();
        queue.try_send(goal).expect("queue the close");
        let mut mv = None;
        let start = Instant::now();

        let idle = CommandedOpening::idle(1.0);
        tick_close(&mut mv, &mut goals, &move_slot, idle, fresh(1.0), start).await;
        let closing = mv.as_ref().expect("the close starts");
        assert_eq!(
            (closing.target_frac(), closing.effort()),
            (0.0, EffortCap::Kept)
        );

        let landed = start + ms(10);
        tick_close(
            &mut mv,
            &mut goals,
            &move_slot,
            free(0.0),
            fresh(0.4),
            landed,
        )
        .await;
        assert!(
            mv.is_some() && ends.try_recv().is_err(),
            "the jaws are judged"
        );
        let still = landed + GRIPPER_STILL_WINDOW;
        tick_close(
            &mut mv,
            &mut goals,
            &move_slot,
            free(0.0),
            fresh(0.4),
            still,
        )
        .await;
        assert_eq!(ends.try_recv(), Ok(Outcome::Complete));
        assert!(mv.is_none() && !move_slot.is_claimed());
    }

    /// The cancel of its posture goal ends a close cancelled, for the reason
    /// every share of a cancelled posture goal ends with.
    #[tokio::test]
    async fn the_cancel_of_its_posture_goal_ends_a_close_cancelled() {
        let move_slot = claimed_move_slot();
        let (queue, mut goals) = mpsc::channel(1);
        let (goal, mut ends, cancelled) = left_close();
        queue.try_send(goal).expect("queue the close");
        let mut mv = None;
        let start = Instant::now();

        let idle = CommandedOpening::idle(1.0);
        tick_close(&mut mv, &mut goals, &move_slot, idle, fresh(1.0), start).await;
        assert!(mv.is_some(), "the close starts");
        cancelled.store(true, Ordering::Release);
        let travelling = free(0.94);
        tick_close(
            &mut mv,
            &mut goals,
            &move_slot,
            travelling,
            fresh(1.0),
            start + ms(10),
        )
        .await;
        assert_eq!(
            ends.try_recv(),
            Ok(Outcome::Cancelled(
                "its posture goal was cancelled".to_string()
            ))
        );
        assert!(mv.is_none() && !move_slot.is_claimed());
    }

    /// A close that reaches a stale follower starts no move and completes:
    /// its goal leaves the gripper as it stands and goes on to the arms. It
    /// frees its gripper.
    #[tokio::test]
    async fn a_close_that_reaches_a_stale_follower_completes_with_no_move() {
        let move_slot = claimed_move_slot();
        let (queue, mut goals) = mpsc::channel(1);
        let (goal, mut ends, _cancelled) = left_close();
        queue.try_send(goal).expect("queue the close");
        let mut mv = None;

        service_gripper_move(
            &mut mv,
            &mut goals,
            &move_slot,
            CommandedOpening::idle(1.0),
            fresh(1.0),
            Admission::Stale,
            Instant::now(),
        )
        .await;
        assert_eq!(ends.try_recv(), Ok(Outcome::Complete));
        assert!(mv.is_none() && !move_slot.is_claimed());
    }

    /// A follower that goes stale during a close fails it with the failure
    /// of a stale limb, led by the name of its gripper, and frees its
    /// gripper.
    #[tokio::test]
    async fn a_follower_gone_stale_during_a_close_fails_it_naming_its_gripper() {
        let move_slot = claimed_move_slot();
        let (queue, mut goals) = mpsc::channel(1);
        let (goal, mut ends, _cancelled) = left_close();
        queue.try_send(goal).expect("queue the close");
        let mut mv = None;
        let start = Instant::now();

        let idle = CommandedOpening::idle(1.0);
        tick_close(&mut mv, &mut goals, &move_slot, idle, fresh(1.0), start).await;
        assert!(mv.is_some(), "the close starts");
        service_gripper_move(
            &mut mv,
            &mut goals,
            &move_slot,
            free(0.94),
            stale_cached(1.0),
            Admission::Stale,
            start + ms(10),
        )
        .await;
        assert_eq!(
            ends.try_recv(),
            Ok(Outcome::Failed(stale_failure().labelled("left_gripper")))
        );
        assert!(mv.is_none() && !move_slot.is_claimed());
    }

    /// A close whose commanded travel runs out of its timeout fails with
    /// `timeout`, led by the name of its gripper, and frees its gripper.
    #[tokio::test]
    async fn a_close_whose_travel_runs_out_of_its_timeout_fails_naming_its_gripper() {
        let move_slot = claimed_move_slot();
        let (queue, mut goals) = mpsc::channel(1);
        let (goal, mut ends, _cancelled) = left_close();
        queue.try_send(goal).expect("queue the close");
        let mut mv = None;
        let start = Instant::now();

        let idle = CommandedOpening::idle(1.0);
        tick_close(&mut mv, &mut goals, &move_slot, idle, fresh(1.0), start).await;
        let travelling = free(0.92);
        let timed_out = start + CLOSE_TRAVEL_TIMEOUT;
        tick_close(
            &mut mv,
            &mut goals,
            &move_slot,
            travelling,
            fresh(0.93),
            timed_out,
        )
        .await;
        let failure = travel_timeout_failure(CLOSE_TRAVEL_TIMEOUT).labelled("left_gripper");
        assert_eq!(failure.kind, FailureKind::Timeout);
        assert_eq!(ends.try_recv(), Ok(Outcome::Failed(failure)));
        assert!(mv.is_none() && !move_slot.is_claimed());
    }

    /// A stop that came after the admission of a close, and before the
    /// coordinator took it, ends the close cancelled for the stop's reason:
    /// no move starts, and the close frees its gripper.
    #[tokio::test]
    async fn a_stop_since_its_admission_ends_a_close_before_it_starts() {
        let move_slot = claimed_move_slot();
        move_slot.stop_unstarted("stopped: the operator asked");
        let (queue, mut goals) = mpsc::channel(1);
        let (goal, mut ends, _cancelled) = left_close();
        queue.try_send(goal).expect("queue the close");
        let mut mv = None;

        let idle = CommandedOpening::idle(1.0);
        let now = Instant::now();
        tick_close(&mut mv, &mut goals, &move_slot, idle, fresh(1.0), now).await;
        assert_eq!(
            ends.try_recv(),
            Ok(Outcome::Cancelled(
                "stopped: the operator asked".to_string()
            ))
        );
        assert!(mv.is_none() && !move_slot.is_claimed());
    }

    /// The stop service ends a close that waits on its queue cancelled, for
    /// the stop's reason.
    #[tokio::test]
    async fn a_stop_ends_a_queued_close_cancelled_for_its_reason() {
        let (goal, mut ends, _cancelled) = left_close();
        goal.stop("stopped: the operator asked", 1.0).await;
        assert_eq!(
            ends.try_recv(),
            Ok(Outcome::Cancelled(
                "stopped: the operator asked".to_string()
            ))
        );
    }

    // A queued goal the coordinator ends unstarted. The goals are real: a
    // harness node exposes `move_gripper` and accepts every goal.

    /// How long the test waits on the harness for a goal or its result.
    const GOAL_TIMEOUT: Duration = Duration::from_secs(30);

    /// A harness whose node accepts every `move_gripper` goal, and the
    /// contexts of the goals it accepted.
    async fn accepting_gripper_goals() -> (Harness, mpsc::Receiver<move_gripper::GoalContext>) {
        let (contexts, accepted) = mpsc::channel(1);
        let (harness, _mocks) = Harness::start_with(
            Config {
                parameters: Some(params()),
                collision_ctrl_vacant: true,
                perception_geometry_vacant: true,
                ..Default::default()
            },
            move |_parameters, runner| async move {
                let mut action = move_gripper::ActionHandle::expose(&runner).await?;
                while let Some(ctx) = action
                    .handle_goal_next_request(|_| Ok(move_gripper::GoalDecision::accept()))
                    .await?
                {
                    if contexts.send(ctx).await.is_err() {
                        break;
                    }
                }
                Ok(())
            },
        )
        .await
        .expect("start the harness");
        (harness, accepted)
    }

    /// Send a goal for the left gripper to [`MOVE_TARGET`], and give its
    /// handle and the goal the node accepted.
    async fn send_left_gripper_goal(
        harness: &Harness,
        accepted: &mut mpsc::Receiver<move_gripper::GoalContext>,
    ) -> (fixture::GoalHandle, GripperGoal) {
        let request = move_gripper::GoalRequestData {
            gripper_name: "left_gripper".to_string(),
            opening: MOVE_TARGET,
            max_effort: 0.0,
            settle_timeout_s: MOVE_TIMEOUT.as_secs_f64(),
        };
        let handle = fixture::send_goal(
            harness,
            &request,
            peppygen::DEFAULT_FEEDBACK_BOUND,
            GOAL_TIMEOUT,
        )
        .await
        .expect("send the goal");
        let ctx = accepted.recv().await.expect("the node accepts the goal");
        let goal = GripperGoal {
            opening: MOVE_TARGET,
            effort: EffortCap::Named(None),
            settle_timeout: MOVE_TIMEOUT,
            travel_timeout: None,
            reply: GripperReply::MoveGripper(Box::new(ctx)),
        };
        (handle, goal)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_queued_goal_failed_on_a_stale_follower_frees_the_arm_for_the_next_goal() {
        let (harness, mut accepted) = accepting_gripper_goals().await;
        let move_slot = claimed_move_slot();
        let (queue, mut goals) = mpsc::channel(1);
        let mut mv = None;
        let start = Instant::now();

        let (stale_goal, goal) = send_left_gripper_goal(&harness, &mut accepted).await;
        queue.try_send(goal).expect("queue the goal");
        service_gripper_move(
            &mut mv,
            &mut goals,
            &move_slot,
            CommandedOpening::idle(1.0),
            fresh(1.0),
            Admission::Stale,
            start,
        )
        .await;
        let stale = stale_failure();
        match stale_goal
            .get_result(GOAL_TIMEOUT)
            .await
            .expect("the goal ends")
            .outcome
        {
            fixture::ResultOutcome::Failed(failure) => assert_eq!(
                failure,
                fixture::Failure::new(stale.kind.into(), stale.message)
            ),
            other => panic!("a goal on a stale follower fails: {other:?}"),
        }
        assert!(mv.is_none(), "no move starts");
        assert!(
            !move_slot.is_claimed(),
            "the failure releases the claim of its admission"
        );

        assert!(move_slot.claim(), "the handler admits the next goal");
        let (_next_goal, goal) = send_left_gripper_goal(&harness, &mut accepted).await;
        queue.try_send(goal).expect("queue the goal");
        service_gripper_move(
            &mut mv,
            &mut goals,
            &move_slot,
            CommandedOpening::idle(1.0),
            fresh(1.0),
            Admission::Live,
            start + ms(10),
        )
        .await;
        assert!(mv.is_some(), "the next goal's move starts");
        drop(mv);
        assert!(!move_slot.is_claimed(), "the move releases the claim");
        harness.shutdown().await.expect("shut the harness down");
    }
}
