//! Gripper move execution: an accepted `move_gripper` goal ([`GripperGoal`])
//! run as a [`GripperMove`] through the same per-tick governing as every
//! other DOF. Each tick the coordinator services each side here: a queued
//! goal is admitted into a free side, and the move in flight ends once the
//! jaws stand still from the planned end of the commanded travel on, at its
//! settle timeout, on a cancel, or on a follower gone stale. The coordinator
//! hands in the opening the governor published last tick
//! ([`CommandedOpening`]) and the opening the follower measured
//! ([`MeasuredOpening`]). The move holds its side's move slot until it ends.

use std::time::{Duration, Instant};

use peppygen::exposed_actions::limb_motion::move_gripper;
use tokio::sync::mpsc;
use tracing::error;

use crate::failure::{Failure, FailureKind, stale_failure};
use crate::liveness::Admission;
use crate::move_slot::{MoveSlot, SlotRelease};

/// An accepted `move_gripper` goal handed to the coordinator, which executes it
/// through the same per-tick governing as everything else (the gripper analog of
/// [`Goal`](crate::planner::Goal) for the arms). The opening is the validated
/// goal fraction; the effort cap is validated non-negative, `None` when the
/// goal carried no preference.
pub struct GripperGoal {
    pub opening: f64,
    pub max_effort: Option<f64>,
    /// The longest the move waits, after the planned end of the commanded
    /// travel, for the jaws to stand still.
    pub settle_timeout: Duration,
    pub ctx: move_gripper::GoalContext,
}

impl GripperGoal {
    /// Fail unstarted with `failure`, with no result. The move slot is the
    /// caller's concern.
    pub async fn fail(self, failure: Failure) {
        fail_gripper_goal(&self.ctx, failure).await;
    }

    /// End unstarted as cancelled by the stop service, for `reason`, which
    /// names the stop, reporting `reported_frac`. The move slot is the
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
/// holds the side's move slot until it ends, so a second gripper goal on the
/// side is refused until then; the slot release frees the slot on any exit.
pub struct GripperMove {
    motion: GripperMotion,
    max_effort: Option<f64>,
    ctx: move_gripper::GoalContext,
    _move_slot: SlotRelease,
}

impl GripperMove {
    /// The opening the move commands.
    pub fn target_frac(&self) -> f64 {
        self.motion.target_frac
    }

    /// The effort cap the move relays, `None` for no preference.
    pub fn max_effort(&self) -> Option<f64> {
        self.max_effort
    }
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
    /// An idle side's opening, re-anchored on the measured gripper.
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
/// `action_time`. The move slot releases with the move on every path.
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
        if admission == Admission::Stale {
            goal.fail(stale_failure()).await;
        } else if mv.is_none() {
            *mv = Some(GripperMove {
                motion: GripperMotion::new(goal.opening, goal.settle_timeout, now),
                max_effort: goal.max_effort,
                ctx: goal.ctx,
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
    let cancel = m.ctx.cancel_cause().map(|cause| cause.reason().to_string());
    let Some(terminal) =
        gripper_move_terminal(&mut m.motion, commanded, measured, admission, cancel, now)
    else {
        return;
    };
    let m = mv.take().expect("in-flight move checked above");
    end_gripper_move(m, terminal, measured.frac, now).await;
}

/// End a gripper move's goal on `terminal` at `now`. A goal that completes or
/// ends cancelled reports `measured_frac` as `final_opening` and the time
/// since admission as `action_time`, and `settled` true only when the jaws
/// stood still; a follower gone stale fails the goal with no result. The
/// move slot releases with the move.
pub async fn end_gripper_move(
    m: GripperMove,
    terminal: GripperTerminal,
    measured_frac: f64,
    now: Instant,
) {
    let elapsed_s = now.duration_since(m.motion.admitted).as_secs_f64();
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
