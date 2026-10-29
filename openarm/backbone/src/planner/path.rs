//! The shares of an execute_joint_path goal's path. The path moves the arms
//! its goal names ([`ArmOrder`]) along one trajectory through both arms'
//! joints, the arms it does not name held at their setpoints: [`start_path`]
//! starts it on every named arm at once, each arm's planner following its
//! arm's share of it. The shares hold their arms' commands where they stand
//! while the coordinator judges the path beside the loop ([`judge_path`]),
//! from where each named arm measured when the coordinator took the goal.
//! Once a judgment lets the path run, each named arm's run start is where it
//! measures on that tick. While a run start stands farther than
//! `RUN_START_SLACK_RAD` from the start the path was judged from, the path is
//! judged again from the run starts, the shares still holding, so the path
//! that plays is the path a judgment passed. Each named arm's command then
//! moves onto the path's start, within that slack of where the arm stands, as
//! a recovered follower's moves onto its measured pose, and follows the path
//! ([`resume_judged_path`]). Each tick a share
//! proposes the next point of the path at full pace, the governor scales the
//! step of the path's arms as one, and the path's clock plays on by the share
//! of the step the governor allowed ([`commit_both`] plays the clocks of a
//! path's two shares at one pace), so the arms stay on the path while the
//! guard slows them and pass each waypoint together. A share is judged on
//! each tick from the first at or after its planned end: it completes once
//! its clock has played to the end and every joint stands within its
//! tolerance, and fails with `timeout` `settle_timeout_s` after the planned
//! end.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use control_core::path_timing::{JointTrajectory, MIN_JOINT_CHANGE_RAD, RoundedPath};
use control_core::timing::{MotionClock, Pace};
use openarm_description::{ARM_DOF, ArmPair, Side};
use srs_model::JointVec;
use srs_model::chain_kinematics::clamp_into;
use tokio::sync::mpsc;
use tracing::{error, info};

use super::{
    ANOTHER_MOVE_IN_FLIGHT, Advance, Measurement, Mode, Origin, Outcome, Planner, Unmeasured,
    planned_end,
};
use crate::actions::MAX_MOVE_DURATION_S;
use crate::failure::{Failure, FailureKind, stale_failure};
use crate::governor::{GuardAt, PathArms, PathBlocked};
use crate::move_slot::SlotRelease;
use crate::settle::{Settle, farthest_outside, joint_shortfall, verdict};
use crate::types::{DUAL_DOF, DualJointVec, arm_joints, dual_joints};

/// The outcome of a share whose goal was cancelled: the action ends its goal
/// with the reason of its own cancel.
const SHARE_CANCELLED: &str = "its goal was cancelled";

/// The failure of a share of a path whose other share failed: the goal fails
/// with the other share's failure, which ended first.
pub(crate) const OTHER_SHARE_FAILED: &str = "the other arm's share of the move failed";

/// Where a share stands that has not played its path to the end, in the
/// words of its `timeout` failure: the collision guard held it back.
const PATH_UNFINISHED: &str = "short of the end of its path";

/// The check of a joint path against the robot's own body, before any arm
/// moves along it, against the guard as the governor held it when the path
/// started, the limbs the path does not move held where they stood then.
pub trait PathGuard: Send {
    /// Whether `guard` lets `arms` along `path`, whose points carry both
    /// arms' joints; the words of the failure when it would hold an arm
    /// short. Once `stop` is raised, the check ends at its next measured
    /// probe with the words of a stopped check.
    fn check(
        &mut self,
        arms: PathArms,
        guard: &GuardAt,
        path: &RoundedPath,
        stop: &AtomicBool,
    ) -> Result<(), String>;
}

/// The arms a joint path moves, in the order its joint arrays hold their
/// joints: [`ARM_DOF`] for each arm, each arm's in its fixed order.
#[derive(Clone, Debug)]
pub struct ArmOrder(Vec<Side>);

impl ArmOrder {
    /// Parse the arm names of a goal's `arms`, in their order: one or more
    /// of this robot's arm names, none named twice.
    pub fn parse(arm_names: &[&str]) -> Result<Self, String> {
        if arm_names.is_empty() {
            return Err(format!(
                "arms holds no arm: name one or more of {}",
                Self::robot_arms()
            ));
        }
        let sides = arm_names
            .iter()
            .map(|name| {
                Side::from_arm_name(name).map_err(|_| {
                    format!(
                        "arms names \"{name}\", no arm of this robot: its arms are {}",
                        Self::robot_arms()
                    )
                })
            })
            .collect::<Result<Vec<Side>, String>>()?;
        if let Some(twice) = sides
            .iter()
            .enumerate()
            .find_map(|(k, side)| sides[..k].contains(side).then_some(side))
        {
            return Err(format!(
                "arms names \"{}\" twice: name each arm once",
                twice.arm_name()
            ));
        }
        Ok(Self(sides))
    }

    /// This robot's arm names, quoted, as a refusal lists them.
    fn robot_arms() -> String {
        Side::BOTH
            .map(|side| format!("\"{}\"", side.arm_name()))
            .join(" and ")
    }

    pub fn sides(&self) -> &[Side] {
        &self.0
    }

    /// `failure` of the arm on `side` in a goal of these arms, as
    /// [`share_failure`] gives it.
    pub fn failure_of(&self, side: Side, failure: Failure) -> Failure {
        share_failure(failure, side, self.path_arms())
    }

    /// How many joint positions an array in this order holds.
    fn joint_count(&self) -> usize {
        self.0.len() * ARM_DOF
    }

    /// The arms the governor steps along the path as one.
    fn path_arms(&self) -> PathArms {
        match self.0[..] {
            [side] => PathArms::One(side),
            _ => PathArms::Both,
        }
    }

    /// Each arm of this order with its joints in `array`, an array in this
    /// order.
    fn split<'a>(&'a self, array: &'a [f64]) -> impl Iterator<Item = (Side, JointVec)> + 'a {
        assert_eq!(array.len(), self.joint_count(), "an array in this order");
        self.0
            .iter()
            .zip(array.chunks_exact(ARM_DOF))
            .map(|(&side, joints)| {
                (
                    side,
                    joints.try_into().expect("an arm's share of the array"),
                )
            })
    }

    /// The joints of `side`'s arm in `array`, an array in this order: none
    /// for an arm this order does not name.
    fn joints_of(&self, array: &[f64], side: Side) -> Option<JointVec> {
        self.split(array)
            .find_map(|(named, joints)| (named == side).then_some(joints))
    }

    /// `held` with each arm of this order at its joints in `array`.
    fn placed(&self, array: &[f64], held: &ArmPair<JointVec>) -> ArmPair<JointVec> {
        ArmPair::from_fn(|side| self.joints_of(array, side).unwrap_or(held[side]))
    }
}

/// A joint path of the arms `arms` names: from where they stand through
/// `waypoints` in turn, the last of which is where it ends, over `duration`
/// at least. `duration` is a requested duration off the wire, at most
/// [`MAX_MOVE_DURATION_S`].
pub struct JointPath {
    pub arms: ArmOrder,
    /// One or more, each an array in the order of `arms`.
    pub waypoints: Vec<Vec<f64>>,
    pub duration: Duration,
}

/// An execute_joint_path goal as [`start_path`] starts it: the path; the
/// expected start of each arm, none for an arm whose start is not checked;
/// the settle each arm's share ends by; where the shares report; and the
/// release of the move slot of each arm, which the goal's admission claimed:
/// dropping the goal frees them. The expected starts, the settle's tolerance
/// and the releases are in the order of the path's arms.
pub struct PathGoal {
    pub path: JointPath,
    pub expected_starts: Vec<Option<JointVec>>,
    pub settle: Settle<Vec<f64>>,
    pub reply: ShareReply,
    pub move_slots: Vec<SlotRelease>,
}

impl PathGoal {
    /// End the goal before its path starts, cancelled by the stop service for
    /// `reason`: each arm reports the joints `measured` gives it, as an arm
    /// move stopped before it starts does, and its move slot is freed.
    pub fn stop(self, reason: &str, measured: &ArmPair<Measurement>, planners: &ArmPair<Planner>) {
        for &side in self.path.arms.sides() {
            let report = ShareReport::ended(
                side,
                Outcome::Cancelled(reason.to_string()),
                &planners[side],
                &measured[side],
                0.0,
            );
            self.reply.send(report);
        }
    }
}

/// The timing of a joint path: from where the path starts through each
/// waypoint, as both arms' joints, under both arms' limits, over the
/// requested duration at least. A trajectory built from it at any time runs
/// the same path.
#[derive(Clone)]
pub struct PathPlan {
    waypoints: Vec<DualJointVec>,
    max_velocity: DualJointVec,
    max_acceleration: DualJointVec,
    duration: Duration,
}

impl PathPlan {
    /// The plan of the path from the first of `waypoints` through the
    /// others, under the limits, over `duration` at least: a path the timing
    /// reports judge.
    #[cfg(test)]
    pub fn through(
        waypoints: Vec<DualJointVec>,
        max_velocity: DualJointVec,
        max_acceleration: DualJointVec,
        duration: Duration,
    ) -> Self {
        Self {
            waypoints,
            max_velocity,
            max_acceleration,
            duration,
        }
    }

    /// How long the path's trajectory takes.
    #[cfg(test)]
    pub fn duration(&self) -> Duration {
        self.trajectory(Instant::now()).clock().duration()
    }

    /// The path's trajectory, starting at `now`.
    fn trajectory(&self, now: Instant) -> JointTrajectory<DUAL_DOF> {
        JointTrajectory::through(
            &self.waypoints,
            &self.max_velocity,
            &self.max_acceleration,
            self.duration,
            now,
        )
    }

    /// Where the path starts: its first waypoint.
    fn start(&self) -> DualJointVec {
        self.waypoints[0]
    }

    /// The plan of the same path from `start`: through the same waypoints,
    /// under the same limits, over the requested duration at least.
    fn with_start(&self, start: DualJointVec) -> Self {
        let mut waypoints = self.waypoints.clone();
        waypoints[0] = start;
        Self { waypoints, ..*self }
    }
}

/// The judgment of one joint path that the shares of its arms wait for.
#[derive(Clone, Copy, PartialEq)]
struct JudgmentId(u64);

impl JudgmentId {
    /// An id no other judgment of this process has.
    fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// A joint path whose shares hold their arms until it is judged: the
/// judgment they wait for, the path's timing from the start it is judged
/// from, the arms it moves, and how many times the path has been judged, this
/// judgment included.
pub struct PathJudgment {
    id: JudgmentId,
    pub plan: PathPlan,
    pub arms: PathArms,
    judgments: u32,
}

impl PathJudgment {
    /// The judgment of the same path from `start`, which the shares of its
    /// arms wait for in its turn: the path judged once more.
    fn again_from(&self, start: DualJointVec) -> Self {
        Self {
            id: self.id,
            plan: self.plan.with_start(start),
            arms: self.arms,
            judgments: self.judgments + 1,
        }
    }

    /// Whether a share of the path still waits for this judgment: none does
    /// once its goal's cancel, the stop or a stale follower ended it.
    pub fn is_awaited(&self, planners: &ArmPair<Planner>) -> bool {
        Side::BOTH
            .into_iter()
            .any(|side| planners[side].awaits(self.id))
    }
}

/// How the share of the arm on `side` ended: its outcome, and what its
/// arm's result gives when the goal completes or ends cancelled: the joints
/// [`Planner::reported_joints`] reports for `measured`, the arm's
/// measurement when the share ended, and the seconds since its path started.
#[derive(Clone, Debug, PartialEq)]
pub struct ShareReport {
    pub side: Side,
    pub outcome: Outcome,
    pub joints: JointVec,
    pub measured: Measurement,
    pub elapsed_s: f64,
}

impl ShareReport {
    /// The report of the share of `planner`'s arm on `side` that ended on
    /// `outcome`, the arm measuring `measured`, `elapsed_s` after its path
    /// started.
    fn ended(
        side: Side,
        outcome: Outcome,
        planner: &Planner,
        measured: &Measurement,
        elapsed_s: f64,
    ) -> Self {
        Self {
            side,
            outcome,
            joints: planner.reported_joints(measured),
            measured: *measured,
            elapsed_s,
        }
    }

    /// The report of the share of `planner`'s arm that failed with
    /// `failure`, before its path started or with no measurement to vouch
    /// for: the joints commanded last.
    fn failed(planner: &Planner, failure: Failure) -> Self {
        Self::ended(
            planner.side,
            Outcome::Failed(failure),
            planner,
            &Err(Unmeasured::NotYet),
            0.0,
        )
    }
}

/// Where the shares of one joint path report: each share's terminal on
/// `done_tx`, which holds a report for each arm of the path, the flag the
/// path's action raises on the goal's cancel, and the flag a share raises
/// when it fails, which ends the other shares.
#[derive(Clone)]
pub struct ShareReply {
    pub done_tx: mpsc::Sender<ShareReport>,
    pub cancelled: Arc<AtomicBool>,
    pub failed: Arc<AtomicBool>,
}

impl ShareReply {
    /// How the share ends before its path does, or `None` while nothing ends
    /// it: cancelled once the goal is, failed once another share failed.
    fn interruption(&self) -> Option<Outcome> {
        if self.cancelled.load(Ordering::Acquire) {
            return Some(Outcome::Cancelled(SHARE_CANCELLED.to_string()));
        }
        self.failed
            .load(Ordering::Acquire)
            .then(|| Outcome::Failed(Failure::new(FailureKind::AttemptFailed, OTHER_SHARE_FAILED)))
    }

    /// Report how the share of a path of `arms` ended, its failure as
    /// [`share_failure`] gives it; a failure raises the flag that ends the
    /// other shares.
    fn report(&self, report: ShareReport, arms: PathArms) {
        let outcome = match report.outcome {
            Outcome::Failed(failure) => {
                self.failed.store(true, Ordering::Release);
                Outcome::Failed(share_failure(failure, report.side, arms))
            }
            other => other,
        };
        self.send(ShareReport { outcome, ..report });
    }

    /// Send `report` to the path's action. Its channel holds a report for
    /// each arm of the path, and each share reports once, so the send finds
    /// room while the action waits.
    fn send(&self, report: ShareReport) {
        let side = report.side;
        if self.done_tx.try_send(report).is_err() {
            error!("{side}: the action of the joint path stopped waiting for its shares");
        }
    }
}

/// `failure` of the share of the arm on `side` in a path of `arms`: labelled
/// with the arm's name on a path of both arms, and as it is on a path of
/// one, whose goal names that arm.
fn share_failure(failure: Failure, side: Side, arms: PathArms) -> Failure {
    match arms {
        PathArms::Both => failure.labelled(side.arm_name()),
        PathArms::One(_) => failure,
    }
}

/// One arm's share of a joint path: where it stands on the path, the arms
/// the path moves, where the share reports, and the settle it ends by.
pub(super) struct PathShare {
    stage: ShareStage,
    arms: PathArms,
    reply: ShareReply,
    settle: Settle<JointVec>,
    _move_slot: SlotRelease,
}

/// Where a share of a joint path stands on its path.
enum ShareStage {
    /// Holding the arm's command where it stands until a judgment of the
    /// path, `judgment` its id, lets the path run, `start` what the arm's
    /// run start is checked against.
    Awaiting {
        judgment: JudgmentId,
        start: ShareStart,
    },
    /// Following the trajectory through both arms' joints, of which the arm
    /// follows its own joints, on the path's clock: the clock plays on by the
    /// share of each step the governor allows, and the path is played to its
    /// end once the clock is.
    Following(JointTrajectory<DUAL_DOF>),
}

/// What a share's arm is checked against on the tick its path starts: its
/// joints where it measured when the coordinator took the goal, clamped into
/// its joint limits, and the goal's expected start of the arm, none when the
/// goal gives none.
#[derive(Clone, Copy)]
struct ShareStart {
    taken: JointVec,
    expected: Option<JointVec>,
}

impl ShareStart {
    /// Fail the path where the arm on `side` measures `measured` on the tick
    /// the path starts, `run` clamped into its joint limits: farther than
    /// `tolerance` in a joint from the goal's expected start, as the
    /// coordinator checks it when it takes the goal, or `run` farther than
    /// that from where it measured then ([`check_start`]).
    fn check_run_start(
        &self,
        side: Side,
        measured: &JointVec,
        run: &JointVec,
        tolerance: f64,
    ) -> Result<(), Failure> {
        if let Some(expected) = &self.expected {
            check_start(side, StartCheck::Expected, expected, measured, tolerance)?;
        }
        check_start(side, StartCheck::Taken, &self.taken, run, tolerance)
    }
}

impl PathShare {
    /// Fail the share of `planner`'s arm with `failure`, with no result.
    pub(super) fn fail(self, planner: &Planner, failure: Failure) {
        self.reply
            .report(ShareReport::failed(planner, failure), self.arms);
    }

    /// End the share of `planner`'s arm on `outcome` at `now`, the arm
    /// measuring `measured`: the seconds since its path started, none for a
    /// share whose path has not started.
    pub(super) fn end(
        self,
        planner: &Planner,
        outcome: Outcome,
        measured: Measurement,
        now: Instant,
    ) {
        let elapsed_s = match &self.stage {
            ShareStage::Awaiting { .. } => 0.0,
            ShareStage::Following(traj) => traj.clock().elapsed_secs(now),
        };
        let report = ShareReport::ended(planner.side, outcome, planner, &measured, elapsed_s);
        self.reply.report(report, self.arms);
    }
}

/// The step a share of a path proposed this tick: from the held setpoint
/// toward the next point of its path, or to the held setpoint itself while
/// the path is judged, at `at`, the arm measuring `measured`.
/// The share of it the governor allows is the pace the path's clock plays on
/// at.
#[derive(Clone, Copy)]
pub(super) struct ProposedStep {
    pub(super) from: JointVec,
    pub(super) to: JointVec,
    pub(super) at: Instant,
    pub(super) measured: JointVec,
}

impl ProposedStep {
    /// The point `share` of the way from `from` to `to`: `to` itself for
    /// the whole step, so a whole step lands on the path's point.
    fn along(&self, share: Pace) -> JointVec {
        if share == Pace::FULL {
            return self.to;
        }
        std::array::from_fn(|i| self.from[i] + share.get() * (self.to[i] - self.from[i]))
    }

    /// The share of the step `governed` covers along it, from 0 at `from` to
    /// 1 at `to`; a step that moves nothing is all covered.
    fn pace(&self, governed: &JointVec) -> Pace {
        let step: JointVec = std::array::from_fn(|i| self.to[i] - self.from[i]);
        let length_sq: f64 = step.iter().map(|d| d * d).sum();
        if length_sq == 0.0 {
            return Pace::FULL;
        }
        let covered: f64 = (0..ARM_DOF)
            .map(|i| (governed[i] - self.from[i]) * step[i])
            .sum();
        Pace::clamped(covered / length_sq)
    }
}

impl Planner {
    /// Whether the arm runs a move.
    fn moving(&self) -> bool {
        !matches!(self.mode, Mode::Follow)
    }

    /// The arms of the joint path whose share the arm holds, if it holds one,
    /// whether the share waits for the path's judgment or follows the path.
    fn path_arms(&self) -> Option<PathArms> {
        match &self.mode {
            Mode::PathShare(PathShare { arms, .. }) => Some(*arms),
            _ => None,
        }
    }

    /// The share of the step a joint path proposed this tick that `governed`
    /// covers, when one did.
    pub(super) fn pace_of(&self, governed: &JointVec) -> Option<Pace> {
        self.proposed.map(|step| step.pace(governed))
    }

    /// Adopt `governed`, play a joint path's clock on at `pace`, and end the
    /// arm's share as its settle judges the arm the step was proposed at:
    /// settled once the arm stands within the tolerance of the path's end
    /// with the path played to it, timed out once the settle timeout has
    /// passed since the planned end, with the arm short of the end or with the
    /// path unfinished, which the guard held back. The arm then holds the
    /// governed setpoint.
    pub(super) fn commit_paced(&mut self, governed: JointVec, pace: Option<Pace>) {
        self.setpoint = governed;
        let (Some(step), Some(pace)) = (self.proposed.take(), pace) else {
            return;
        };
        let Mode::PathShare(PathShare {
            stage: ShareStage::Following(traj),
            settle,
            ..
        }) = &mut self.mode
        else {
            return;
        };
        traj.advance(step.at, pace);
        let end = arm_joints(&traj.end(), self.side);
        let Some(outcome) = settle_outcome(&step.measured, &end, settle, traj.clock(), step.at)
        else {
            return;
        };
        let elapsed_s = traj.clock().elapsed_secs(step.at);
        if let Mode::PathShare(share) = std::mem::replace(&mut self.mode, Mode::Follow) {
            let report =
                ShareReport::ended(self.side, outcome, self, &Ok(step.measured), elapsed_s);
            share.reply.report(report, share.arms);
        }
    }

    /// One tick of the arm's share of a joint path: end the share as its
    /// reply's interruption says, holding where the arm already is; while the
    /// path waits for its judgment, hold the arm's command where it stands;
    /// once it runs, propose the arm's half of the path's next point. The
    /// share settles in its commit ([`Planner::commit_paced`]). A terminal
    /// drops `m` (and with it the slot release), releasing the move slot.
    pub(super) fn advance_path_share(
        &mut self,
        m: PathShare,
        measured_q: JointVec,
        now: Instant,
    ) -> Advance {
        if let Some(outcome) = m.reply.interruption() {
            m.end(self, outcome, Ok(measured_q), now);
            return Advance::ends_move(self.setpoint);
        }
        let origin = Origin::OnPath(m.arms);
        let traj = match &m.stage {
            ShareStage::Awaiting { .. } => {
                return Advance {
                    target: self.setpoint,
                    next_mode: Mode::PathShare(m),
                    origin,
                };
            }
            ShareStage::Following(traj) => traj,
        };
        let target = arm_joints(&traj.sample(now), self.side);
        Advance {
            target,
            next_mode: Mode::PathShare(m),
            origin,
        }
    }

    /// Start the arm's share that waits for `judgment`, if it holds one, on
    /// `traj`, the judged path, the arm's command moved onto `start`, the
    /// arm's joints at the path's start.
    fn start_awaited(
        &mut self,
        judgment: JudgmentId,
        traj: &JointTrajectory<DUAL_DOF>,
        start: JointVec,
    ) {
        if !self.awaits(judgment) {
            return;
        }
        if let Mode::PathShare(m) = &mut self.mode {
            m.stage = ShareStage::Following(traj.clone());
        }
        self.setpoint = start;
        self.stream_smoothing.reset();
    }

    /// End the arm's share that waits for `judgment`, if it holds one, with
    /// `failure`, handing the arm back to Follow with its command where it
    /// held.
    fn fail_awaited(&mut self, judgment: JudgmentId, failure: &Failure) {
        if !self.awaits(judgment) {
            return;
        }
        if let Mode::PathShare(m) = std::mem::replace(&mut self.mode, Mode::Follow) {
            m.reply.send(ShareReport::failed(self, failure.clone()));
        }
    }

    /// Whether the arm holds a share that waits for `judgment`.
    fn awaits(&self, judgment: JudgmentId) -> bool {
        self.awaited_start(judgment).is_some()
    }

    /// What the arm's run start is checked against on the path whose
    /// `judgment` the arm's share waits for, if it holds one.
    fn awaited_start(&self, judgment: JudgmentId) -> Option<ShareStart> {
        match &self.mode {
            Mode::PathShare(PathShare {
                stage:
                    ShareStage::Awaiting {
                        judgment: id,
                        start,
                    },
                ..
            }) if *id == judgment => Some(*start),
            _ => None,
        }
    }

    /// Fail the arm's share of a path with `failure`, if it holds one,
    /// handing the arm back to Follow; any other move runs on.
    pub fn fail_path_share(&mut self, failure: Failure) {
        match std::mem::replace(&mut self.mode, Mode::Follow) {
            Mode::PathShare(share) => share.fail(self, failure),
            other => self.mode = other,
        }
    }

    /// Drop the arm's share of a path whose action is gone, handing the arm
    /// back to Follow where it holds its setpoint, and freeing its move slot.
    pub fn drop_share(&mut self) {
        if matches!(self.mode, Mode::PathShare(_)) {
            self.mode = Mode::Follow;
        }
    }
}

/// How a share with `settle` ends with the arm measuring `measured` against
/// `end`, the path's end, its clock at `clock`, at `now`, as [`verdict`]
/// judges it from the planned end on: settled once the arm stands within the
/// tolerance with the clock played to the end, timed out once the timeout has
/// passed since the planned end, with how far the arm stands from the end or
/// with the path unfinished.
fn settle_outcome(
    measured: &JointVec,
    end: &JointVec,
    settle: &Settle<JointVec>,
    clock: &MotionClock,
    now: Instant,
) -> Option<Outcome> {
    let shortfall = match joint_shortfall(measured, end, &settle.tolerance) {
        None if !clock.is_complete() => Some(PATH_UNFINISHED.to_string()),
        shortfall => shortfall,
    };
    verdict(shortfall, planned_end(clock), now, settle.timeout).map(Outcome::from)
}

/// Start `goal` on every arm its path names at once, or fail it before any
/// arm moves, freeing its arms:
/// - with `precondition_failed` when an arm runs another move, and with the
///   stale failure when an arm has no fresh measurement in `measured`, each
///   as [`ArmOrder::failure_of`] gives it;
/// - with `precondition_failed` when an arm measures outside the start
///   tolerance of its expected start ([`check_start`]).
///
/// The path starts at the joints each named arm measures in `measured`,
/// clamped into the arm's joint limits, the arms it does not name held at
/// their setpoints, and runs from there through each waypoint, timed under
/// both arms' limits. Each named arm's planner holds its share and its arm's
/// slot release, and holds its arm's command where it stands until the path
/// is judged. The path's judgment is what this gives: [`judge_path`] judges
/// it, and [`resume_judged_path`] starts the shares on the judgment, judges
/// the path again from where the arms stand, or fails them.
pub fn start_path(
    planners: &mut ArmPair<Planner>,
    goal: PathGoal,
    measured: &ArmPair<Measurement>,
) -> Result<PathJudgment, Failure> {
    let PathGoal {
        path,
        expected_starts,
        settle,
        reply,
        move_slots,
    } = goal;
    let arms = &path.arms;
    let path_arms = arms.path_arms();
    let starts = arms
        .sides()
        .iter()
        .zip(&expected_starts)
        .map(|(&side, expected)| {
            let planner = &planners[side];
            if planner.moving() {
                let busy = Failure::new(FailureKind::PreconditionFailed, ANOTHER_MOVE_IN_FLIGHT);
                return Err(arms.failure_of(side, busy));
            }
            let measured = measured[side].map_err(|_| arms.failure_of(side, stale_failure()))?;
            if let Some(expected) = expected {
                let tolerance = planner.cfg.start_tolerance_rad;
                check_start(side, StartCheck::Expected, expected, &measured, tolerance)?;
            }
            Ok(clamp_into(&planner.cfg.limits, &measured))
        })
        .collect::<Result<Vec<JointVec>, Failure>>()?;
    let joined = |of: &dyn Fn(&Planner) -> JointVec| {
        dual_joints(&ArmPair::from_fn(|side| of(&planners[side])))
    };
    let held = ArmPair::from_fn(|side| planners[side].setpoint);
    let start = arms.placed(&starts.concat(), &held);
    let plan = PathPlan {
        waypoints: std::iter::once(dual_joints(&start))
            .chain(
                path.waypoints
                    .iter()
                    .map(|waypoint| dual_joints(&arms.placed(waypoint, &start))),
            )
            .collect(),
        max_velocity: joined(&|p: &Planner| p.cfg.max_joint_velocity_rad_s),
        max_acceleration: joined(&|p: &Planner| p.cfg.max_joint_acceleration_rad_s2),
        duration: path.duration,
    };
    let id = JudgmentId::next();
    for ((&side, move_slot), expected) in arms.sides().iter().zip(move_slots).zip(expected_starts) {
        planners[side].mode = Mode::PathShare(PathShare {
            stage: ShareStage::Awaiting {
                judgment: id,
                start: ShareStart {
                    taken: start[side],
                    expected,
                },
            },
            arms: path_arms,
            reply: reply.clone(),
            settle: Settle {
                tolerance: arms
                    .joints_of(&settle.tolerance, side)
                    .expect("a tolerance for each arm of the path"),
                timeout: settle.timeout,
            },
            _move_slot: move_slot,
        });
    }
    Ok(PathJudgment {
        id,
        plan,
        arms: path_arms,
        judgments: 1,
    })
}

/// Judge the path `plan` times for `arms` against `guard`, before any arm moves
/// along it: `rejected` when the joint velocity and acceleration caps cannot
/// run it within [`MAX_MOVE_DURATION_S`] ([`check_path_time`]), and
/// `unreachable` when `checker` finds that the guard would hold an arm short of
/// it, both from the path's start. The coordinator runs it beside the
/// loop, so it takes as long as the path is long and the loop ticks on, and
/// raises `stop` once the path's goal ends: a judgment that finds `stop`
/// raised when it begins times nothing, and the check ends at its next
/// measured probe.
pub fn judge_path(
    plan: &PathPlan,
    arms: PathArms,
    guard: &GuardAt,
    checker: &mut dyn PathGuard,
    stop: &AtomicBool,
) -> Result<(), Failure> {
    if stop.load(Ordering::Acquire) {
        let stopped = PathBlocked::Stopped.to_string();
        return Err(Failure::new(FailureKind::Unreachable, stopped));
    }
    let traj = plan.trajectory(Instant::now());
    check_path_time(&traj)?;
    checker
        .check(arms, guard, traj.path(), stop)
        .map_err(|blocked| Failure::new(FailureKind::Unreachable, blocked))
}

/// The most an arm of a joint path may stand from the start the path was
/// judged from, in every joint, on the tick the path starts, for the path to
/// play as judged: the change in every joint under which a joint path takes a
/// waypoint as a repeat of the one before it, 0.001 rad.
const RUN_START_SLACK_RAD: f64 = MIN_JOINT_CHANGE_RAD;

/// How many times a joint path may be judged before it plays, the first
/// judgment included: an arm still farther than [`RUN_START_SLACK_RAD`] from
/// the start of the last of them fails the path. Each judgment is read on a
/// tick after the one before it, so the cap lets an arm settle for 20 control
/// ticks at least, 0.2 s at 100 Hz when each check ends within its tick; in
/// simulation, an OpenArm sent a path the moment its move completed settled
/// within 9.
const MAX_PATH_JUDGMENTS: u32 = 20;

/// Whether the shares of a judged path still wait for it.
pub enum Resumed {
    /// Each share that waited for the path started on it or ended.
    Done,
    /// An arm of the path has no measurement yet: the path waits for a later
    /// tick.
    AwaitsMeasurement,
    /// An arm of the path stands farther than [`RUN_START_SLACK_RAD`] from
    /// the start the path was judged from: the shares wait for this judgment
    /// of the path from where its arms stand now.
    JudgeAgain(Box<PathJudgment>),
}

/// Start the shares that wait for `judgment` when `verdict` lets them, at
/// `now`, on the path the judgment judged, each named arm's command moved
/// onto its joints at the path's start; or end each share with the verdict's
/// failure, the path's own and no arm's, running none of the path, each
/// command where it held. The shares start once every share's arm has a
/// measurement: while one does not, the path waits for a later tick. Each
/// named arm's run start is its joints in `measured`, clamped into its joint
/// limits as its start is when the coordinator takes the goal, and:
/// - an arm that measures farther than its start tolerance from the goal's
///   expected start in a joint, or whose run start stands that far from where
///   it measured when the coordinator took the goal, ends each share with
///   `precondition_failed`, naming the arm and the joint
///   ([`ShareStart::check_run_start`]);
/// - a run start farther than [`RUN_START_SLACK_RAD`] from the path's start
///   in a joint has the path judged again from the run starts, the arm the
///   path does not name where the path holds it ([`Resumed::JudgeAgain`]),
///   and after [`MAX_PATH_JUDGMENTS`] judgments ends each share with
///   `precondition_failed`, naming the arm that still moves.
///
/// A share its goal's cancel, the stop or a stale follower already ended
/// waits for nothing, so the verdict leaves it.
pub fn resume_judged_path(
    planners: &mut ArmPair<Planner>,
    judgment: &PathJudgment,
    verdict: &Result<(), Failure>,
    measured: &ArmPair<Measurement>,
    now: Instant,
) -> Resumed {
    let awaited: Vec<(Side, ShareStart)> = Side::BOTH
        .into_iter()
        .filter_map(|side| Some((side, planners[side].awaited_start(judgment.id)?)))
        .collect();
    if awaited.is_empty() {
        return Resumed::Done;
    }
    if let Err(failure) = verdict {
        fail_awaiting_shares(planners, judgment.id, failure);
        return Resumed::Done;
    }
    let measured_at_start = awaited
        .iter()
        .map(|&(side, start)| Some((side, start, measured[side].ok()?)))
        .collect::<Option<Vec<(Side, ShareStart, JointVec)>>>();
    let Some(measured_at_start) = measured_at_start else {
        return Resumed::AwaitsMeasurement;
    };
    let run_starts: Vec<(Side, JointVec)> = measured_at_start
        .iter()
        .map(|&(side, _, at)| (side, clamp_into(&planners[side].cfg.limits, &at)))
        .collect();
    let refused =
        measured_at_start
            .iter()
            .zip(&run_starts)
            .find_map(|(&(side, start, at), (_, run))| {
                let tolerance = planners[side].cfg.start_tolerance_rad;
                start.check_run_start(side, &at, run, tolerance).err()
            });
    if let Some(failure) = refused {
        fail_awaiting_shares(planners, judgment.id, &failure);
        return Resumed::Done;
    }
    let path_start = judgment.plan.start();
    let still_moving = run_starts
        .iter()
        .filter_map(|&(side, run)| {
            let judged = arm_joints(&path_start, side);
            let (joint, off) = farthest_outside(&run, &judged, &[RUN_START_SLACK_RAD; ARM_DOF])?;
            Some((side, joint, off))
        })
        .max_by(|a, b| a.2.total_cmp(&b.2));
    if let Some((side, joint, off)) = still_moving {
        if judgment.judgments >= MAX_PATH_JUDGMENTS {
            let failure = unsettled(side, joint, off, judgment.judgments);
            fail_awaiting_shares(planners, judgment.id, &failure);
            return Resumed::Done;
        }
        let run_start = ArmPair::from_fn(|side| {
            run_starts
                .iter()
                .find_map(|&(named, run)| (named == side).then_some(run))
                .unwrap_or_else(|| arm_joints(&path_start, side))
        });
        return Resumed::JudgeAgain(Box::new(judgment.again_from(dual_joints(&run_start))));
    }
    let traj = judgment.plan.trajectory(now);
    info!(
        "joint path start for {:?} through {} waypoint(s), duration={:.3}s, judged {} time(s)",
        judgment.arms,
        judgment.plan.waypoints.len() - 1,
        traj.clock().duration().as_secs_f64(),
        judgment.judgments
    );
    for &(side, _) in &run_starts {
        planners[side].start_awaited(judgment.id, &traj, arm_joints(&path_start, side));
    }
    Resumed::Done
}

/// The failure of a path whose arm on `side` still stands `off` rad from
/// the start of its last of `judgments` judgments at `joint`, counted from 0.
fn unsettled(side: Side, joint: usize, off: f64, judgments: u32) -> Failure {
    Failure::new(
        FailureKind::PreconditionFailed,
        format!(
            "{} still moved {off:.3} rad at joint {} while its path was judged {judgments} \
             times; let the arm settle and send the path again",
            side.arm_name(),
            joint + 1
        ),
    )
}

/// End each arm's share that waits for `judgment` with `failure`
/// ([`Planner::fail_awaited`]).
fn fail_awaiting_shares(planners: &mut ArmPair<Planner>, judgment: JudgmentId, failure: &Failure) {
    for side in Side::BOTH {
        planners[side].fail_awaited(judgment, failure);
    }
}

/// What a path's start check compares an arm's joints with.
enum StartCheck {
    /// The goal's expected start, against the joints the arm measures when
    /// the coordinator takes the goal, and on the tick the path starts.
    Expected,
    /// Where the arm measured when the coordinator took the goal, clamped
    /// into its joint limits, against its run start, the joints it measures
    /// once the path is judged, clamped the same way.
    Taken,
}

/// Fail the path where the arm on `side` stands at `joints`, farther than
/// `tolerance` from `reference`, what `check` compares with, in a joint,
/// naming the arm and the joint farthest past it, counted from 1 in the
/// arm's joint order, and what to do: plan the path from where the arm
/// stands, or send it again once the arm moved while it was planned.
fn check_start(
    side: Side,
    check: StartCheck,
    reference: &JointVec,
    joints: &JointVec,
    tolerance: f64,
) -> Result<(), Failure> {
    let Some((joint, off)) = farthest_outside(joints, reference, &[tolerance; ARM_DOF]) else {
        return Ok(());
    };
    let (arm, joint) = (side.arm_name(), joint + 1);
    let message = match check {
        StartCheck::Expected => format!(
            "{arm} stands {off:.3} rad from expected_start_joint_positions at joint {joint}, past \
             the {tolerance:.3} rad start tolerance; plan the path from where the arm stands"
        ),
        StartCheck::Taken => format!(
            "{arm} stands {off:.3} rad at joint {joint} from where it stood when the path was \
             taken, past the {tolerance:.3} rad start tolerance; send the path again"
        ),
    };
    Err(Failure::new(FailureKind::PreconditionFailed, message))
}

/// Fail a path whose trajectory `traj` takes longer than
/// [`MAX_MOVE_DURATION_S`], with `rejected` and its time. The duration a goal
/// requests lies within that ceiling, so such a trajectory takes the time
/// the joint velocity and acceleration caps run its path in.
fn check_path_time(traj: &JointTrajectory<DUAL_DOF>) -> Result<(), Failure> {
    let time_s = traj.clock().duration().as_secs_f64();
    if time_s <= MAX_MOVE_DURATION_S {
        return Ok(());
    }
    Err(Failure::new(
        FailureKind::Rejected,
        format!(
            "the joint velocity and acceleration caps take {time_s:.1} s to run this path, past \
             the {MAX_MOVE_DURATION_S:.0} s a move may take; split it into paths of fewer \
             waypoints"
        ),
    ))
}

/// Whether both planners hold their shares of one path of both arms.
fn on_one_path(planners: &ArmPair<Planner>) -> bool {
    planners
        .iter()
        .all(|planner| planner.path_arms() == Some(PathArms::Both))
}

/// `candidates`, the shares of a path of both arms stepped one share along
/// their proposed steps along the path: the smaller of the shares each arm's
/// chase took, so the step through both arms' joints keeps its line whatever
/// caps one arm's chase. Any other candidates as they are.
pub fn chase_together(
    planners: &ArmPair<Planner>,
    candidates: ArmPair<JointVec>,
) -> ArmPair<JointVec> {
    if !on_one_path(planners) {
        return candidates;
    }
    let (Some(left), Some(right)) = (planners.left.proposed, planners.right.proposed) else {
        return candidates;
    };
    let share = slower(left.pace(&candidates.left), right.pace(&candidates.right));
    ArmPair::new(left, right).map(|_, step| step.along(share))
}

/// The slower of two paces.
fn slower(a: Pace, b: Pace) -> Pace {
    if a <= b { a } else { b }
}

/// Commit each arm's governed setpoint as [`Planner::commit`] does, except
/// that the shares of a path of both arms, each holding a clone of the path's
/// trajectory, play their clocks on at one pace, the smaller of the two arms'
/// shares of their steps, so the two clocks stay together. The governor
/// scales the step of a path of both arms as one, so the two shares agree.
pub fn commit_both(planners: &mut ArmPair<Planner>, governed: &ArmPair<JointVec>) {
    let shared = on_one_path(planners)
        .then(|| {
            let paces = ArmPair::from_fn(|side| planners[side].pace_of(&governed[side]));
            paces.left.zip(paces.right)
        })
        .flatten()
        .map(|(left, right)| slower(left, right));
    for side in Side::BOTH {
        match shared {
            Some(pace) => planners[side].commit_paced(governed[side], Some(pace)),
            None => planners[side].commit(governed[side]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::AtomicBool;

    use control_core::positive_finite::PositiveFinite;

    use crate::move_slot::MoveSlot;
    use crate::planner::Goal;
    use crate::planner::Tick;
    use crate::test_support::{
        ARMS_WOULD_MEET, BOTH_HELD, NO_STOP, OpenGuard, Refusing, array_of, claimed,
        claimed_move_slot, distance_to_segment, measured_at_setpoints, path_goal_through,
        plan_config, planner_holding, share_reply, test_guard, test_settle,
    };

    /// Where the path tests take the arms from [`BOTH_HELD`], inside v1's
    /// limits: the left shoulder 0.4 rad out, the right elbow 0.1 rad in.
    const BOTH_TO: ArmPair<JointVec> = ArmPair {
        left: [0.0, -1.2, 0.0, 1.2, 0.0, 0.0, 0.0],
        right: [0.0, 0.8, 0.0, 1.3, 0.0, 0.0, 0.0],
    };

    /// The order of the arms `arm_names` names.
    fn order(arm_names: &[&str]) -> ArmOrder {
        ArmOrder::parse(arm_names).expect("the arms of this robot, each once")
    }

    /// Both arms' planners, each holding its arm at `held`.
    fn planners_holding(held: ArmPair<JointVec>) -> ArmPair<Planner> {
        ArmPair::from_fn(|side| planner_holding(side, held[side]))
    }

    /// The outcome of the report `received` holds.
    fn outcome(received: Result<ShareReport, mpsc::error::TryRecvError>) -> Option<Outcome> {
        received.ok().map(|report| report.outcome)
    }

    /// A goal of a path of `arms` to `to` over `duration` at least, with
    /// no expected start and [`test_settle`], reporting on `reply`, holding
    /// the slots of its arms in `move_slots`.
    fn path_goal(
        arms: &ArmOrder,
        to: &ArmPair<JointVec>,
        duration: Duration,
        reply: ShareReply,
        move_slots: &ArmPair<MoveSlot>,
    ) -> PathGoal {
        path_goal_through(arms, &[*to], duration, test_settle(arms), reply, |side| {
            &move_slots[side]
        })
    }

    /// Start `goal` on `planners` at `now`, each arm measured where it is
    /// held, and judge its path at once with `checker`: the failure of the
    /// start, or of the judgment.
    fn start(
        planners: &mut ArmPair<Planner>,
        goal: PathGoal,
        checker: &mut dyn PathGuard,
        now: Instant,
    ) -> Result<(), Failure> {
        let measured = measured_at_setpoints(planners);
        let guard = test_guard(planners);
        let judgment = start_path(planners, goal, &measured)?;
        let verdict = judge_path(&judgment.plan, judgment.arms, &guard, checker, &NO_STOP);
        let resumed = resume_judged_path(planners, &judgment, &verdict, &measured, now);
        assert!(matches!(resumed, Resumed::Done), "the path starts or ends");
        verdict
    }

    /// One tick of both planners at `now`, each arm measuring where it was
    /// governed last, with no goal queued.
    async fn tick_both(planners: &mut ArmPair<Planner>, now: Instant) -> ArmPair<Tick> {
        let (_goal_tx, mut goals) = mpsc::channel(1);
        let move_slot = MoveSlot::default();
        let mut ticks = Vec::new();
        for side in Side::BOTH {
            let planner = &mut planners[side];
            let measured = planner.setpoint();
            ticks.push(
                planner
                    .tick(measured, None, &mut goals, &move_slot, now)
                    .await,
            );
        }
        let [left, right]: [Tick; 2] = ticks.try_into().ok().expect("a tick per arm");
        ArmPair::new(left, right)
    }

    /// The trajectory through both arms' joints whose share `planner`
    /// follows.
    fn path_of(planner: &Planner) -> &JointTrajectory<DUAL_DOF> {
        let Mode::PathShare(PathShare {
            stage: ShareStage::Following(traj),
            ..
        }) = &planner.mode
        else {
            panic!("the arm follows its share of a joint path");
        };
        traj
    }

    // --- Arm orders ---------------------------------------------------------

    /// A goal's arms are one or more of this robot's arms, each once; any
    /// other list is refused naming what is wrong in the words of the field.
    #[test]
    fn a_goal_takes_each_arm_of_the_robot_at_most_once() {
        let refused = |names: &[&str]| ArmOrder::parse(names).expect_err("refused");
        assert_eq!(
            refused(&[]),
            "arms holds no arm: name one or more of \"left_arm\" and \"right_arm\""
        );
        assert_eq!(
            refused(&["left_arm", "head"]),
            "arms names \"head\", no arm of this robot: its arms are \"left_arm\" and \
             \"right_arm\""
        );
        assert_eq!(
            refused(&["right_arm", "left_arm", "right_arm"]),
            "arms names \"right_arm\" twice: name each arm once"
        );
        assert_eq!(order(&["right_arm"]).sides(), [Side::Right]);
        assert_eq!(
            order(&["right_arm", "left_arm"]).sides(),
            [Side::Right, Side::Left]
        );
    }

    /// An array in an order holds the named arms' joints in that order: named
    /// right first, its first seven joints are the right arm's, and the arms
    /// it does not name keep where they are held.
    #[test]
    fn an_array_holds_each_named_arm_in_the_order_of_the_goals_arms() {
        let right_first = order(&["right_arm", "left_arm"]);
        let array = array_of(&right_first, &BOTH_TO);
        assert_eq!(array[..ARM_DOF], BOTH_TO.right);
        assert_eq!(array[ARM_DOF..], BOTH_TO.left);
        assert_eq!(right_first.joint_count(), 2 * ARM_DOF);
        assert_eq!(right_first.placed(&array, &BOTH_HELD), BOTH_TO);
        assert_eq!(right_first.path_arms(), PathArms::Both);

        let left = order(&["left_arm"]);
        let array = array_of(&left, &BOTH_TO);
        assert_eq!(array, BOTH_TO.left);
        assert_eq!(
            left.placed(&array, &BOTH_HELD),
            ArmPair::new(BOTH_TO.left, BOTH_HELD.right)
        );
        assert_eq!(left.path_arms(), PathArms::One(Side::Left));
        assert_eq!(left.joints_of(&array, Side::Right), None);
    }

    // --- Starting a path ----------------------------------------------------

    /// Where the tests' left arm measures when its setpoint lags it: 0.3 rad
    /// past [`BOTH_HELD`] at its first and third joints, inside v1's limits.
    fn lagging_left() -> JointVec {
        let mut lagging = BOTH_HELD.left;
        lagging[0] += 0.3;
        lagging[2] += 0.3;
        lagging
    }

    /// A left-arm path through `waypoints` with `settle`, started on planners
    /// holding [`BOTH_HELD`] while the left arm measures `left`, and judged
    /// at once: the planners, the judgment, its verdict, and where the share
    /// reports.
    fn left_path_measured_at(
        left: JointVec,
        waypoints: &[ArmPair<JointVec>],
        settle: Duration,
    ) -> (
        ArmPair<Planner>,
        PathJudgment,
        Result<(), Failure>,
        mpsc::Receiver<ShareReport>,
    ) {
        let mut planners = planners_holding(BOTH_HELD);
        let arms = order(&["left_arm"]);
        let move_slots = claimed(&arms);
        let (reply, done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let settle = Settle {
            timeout: settle,
            ..test_settle(&arms)
        };
        let goal = path_goal_through(&arms, waypoints, Duration::ZERO, settle, reply, |side| {
            &move_slots[side]
        });
        let measured = ArmPair::new(Ok(left), Ok(BOTH_HELD.right));
        let judgment = start_path(&mut planners, goal, &measured).expect("the path starts");
        let verdict = judge_path(
            &judgment.plan,
            judgment.arms,
            &test_guard(&planners),
            &mut OpenGuard,
            &NO_STOP,
        );
        (planners, judgment, verdict, done_rx)
    }

    /// The left arm measuring `left`, the right arm where the tests hold it.
    fn left_measuring(left: JointVec) -> ArmPair<Measurement> {
        ArmPair::new(Ok(left), Ok(BOTH_HELD.right))
    }

    /// A path starts at the joints each arm it names measures, the arm it
    /// does not name at its setpoint, whether or not the goal gives an
    /// expected start, here one within the start tolerance of the measured
    /// joints: the trajectory's first sample stands there. While the path is
    /// judged the arm's command holds at its lagging setpoint; on the tick
    /// the judgment lets the path run, the arm measuring where it did, the
    /// command moves onto the path's run start and the path's clock starts.
    #[tokio::test]
    async fn a_path_starts_at_the_joints_its_arms_measure() {
        let lagging = lagging_left();
        let mut near = lagging;
        near[1] += 0.005;
        for expected in [None, Some(near)] {
            let mut planners = planners_holding(BOTH_HELD);
            let arms = order(&["left_arm"]);
            let move_slots = claimed(&arms);
            let (reply, _done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
            let mut goal = path_goal(&arms, &BOTH_TO, Duration::ZERO, reply, &move_slots);
            goal.expected_starts = vec![expected];
            let measured = left_measuring(lagging);
            let judgment = start_path(&mut planners, goal, &measured).expect("the path starts");
            let t0 = Instant::now();
            let start = dual_joints(&ArmPair::new(lagging, BOTH_HELD.right));
            assert_eq!(judgment.plan.trajectory(t0).sample(t0), start);

            let ticks = tick_both(&mut planners, t0).await;
            let candidates = ticks.map(|_, tick| tick.candidate);
            assert_eq!(
                candidates, BOTH_HELD,
                "the commands hold while the path is judged"
            );
            commit_both(&mut planners, &candidates);
            let verdict = judge_path(
                &judgment.plan,
                judgment.arms,
                &test_guard(&planners),
                &mut OpenGuard,
                &NO_STOP,
            );
            let judged_at = t0 + plan_config().cycle_period;
            let resumed =
                resume_judged_path(&mut planners, &judgment, &verdict, &measured, judged_at);
            assert!(matches!(resumed, Resumed::Done), "the path starts");
            assert_eq!(
                planners.left.setpoint(),
                lagging,
                "the command moves onto the path's run start"
            );
            assert_eq!(planners.right.setpoint(), BOTH_HELD.right);
            assert_eq!(path_of(&planners.left).clock().motion_start(), judged_at);
            assert_eq!(path_of(&planners.left).sample(judged_at), start);
        }
    }

    /// From a setpoint that lags the measured joints, every command from the
    /// tick the path starts to its end lies on the rounded path from the
    /// measured start, where the command moves, and the path ends at its last
    /// waypoint.
    #[tokio::test]
    async fn every_command_of_a_started_path_lies_on_the_path_from_the_measured_start() {
        let lagging = lagging_left();
        let mut turn = BOTH_TO.left;
        turn[3] -= 0.2;
        let waypoints = [BOTH_TO, ArmPair::new(turn, BOTH_TO.right)];
        let (mut planners, judgment, verdict, _done_rx) =
            left_path_measured_at(lagging, &waypoints, Duration::from_secs(1));
        let started = Instant::now();
        let measured = left_measuring(lagging);
        let resumed = resume_judged_path(&mut planners, &judgment, &verdict, &measured, started);
        assert!(matches!(resumed, Resumed::Done), "the path starts");

        let tick = plan_config().cycle_period;
        let mut commanded = Vec::new();
        for k in 0..2000 {
            if !planners.left.moving() {
                break;
            }
            let ticks = tick_both(&mut planners, started + tick * k).await;
            commit_both(&mut planners, &ticks.map(|_, tick| tick.candidate));
            commanded.push(planners.left.setpoint());
        }
        assert!(!planners.left.moving(), "the path ends");
        assert_eq!(planners.left.setpoint(), turn, "at its last waypoint");

        let corners: Vec<Vec<f64>> = std::iter::once(lagging)
            .chain(waypoints.iter().map(|w| w.left))
            .map(|left| dual_joints(&ArmPair::new(left, BOTH_HELD.right)).to_vec())
            .collect();
        let spacing = PositiveFinite::parse(1e-4).expect("a positive spacing");
        let path = RoundedPath::joint_path(&corners);
        let drawn: Vec<Vec<f64>> = path
            .arcs_every(spacing)
            .iter()
            .map(|&s| path.at(s))
            .collect();
        for (k, q) in commanded.iter().enumerate() {
            let at = dual_joints(&ArmPair::new(*q, BOTH_HELD.right));
            let off = drawn
                .windows(2)
                .map(|w| distance_to_segment(&at, &w[0], &w[1]))
                .fold(f64::INFINITY, f64::min);
            assert!(off < 1e-6, "tick {k} commands {off} rad off the path");
        }
    }

    /// A path whose arm's setpoint lags its measured joints, settling with
    /// no time to spare, completes on the tick of its planned end: its clock
    /// starts on the tick the arm's command moves onto the path's run start.
    #[tokio::test]
    async fn a_path_with_no_settle_time_from_a_lagging_setpoint_completes() {
        let lagging = lagging_left();
        let (mut planners, judgment, verdict, mut done_rx) =
            left_path_measured_at(lagging, &[BOTH_TO], Duration::ZERO);
        let started = Instant::now();
        let measured = left_measuring(lagging);
        let resumed = resume_judged_path(&mut planners, &judgment, &verdict, &measured, started);
        assert!(matches!(resumed, Resumed::Done), "the path starts");
        let tick = plan_config().cycle_period;
        let end = started + path_of(&planners.left).clock().duration();
        let mut k = 0;
        while planners.left.moving() {
            let now = started + tick * k;
            assert!(
                now < end + tick,
                "the share ends on the first tick from its planned end"
            );
            let ticks = tick_both(&mut planners, now).await;
            commit_both(&mut planners, &ticks.map(|_, tick| tick.candidate));
            k += 1;
        }
        let report = done_rx.try_recv().expect("the share reports");
        assert_eq!(report.outcome, Outcome::Complete);
    }

    /// A path of both arms started where they measure, `measured`, the
    /// planners holding [`BOTH_HELD`], and judged at once: the planners, the
    /// judgment and its verdict.
    fn both_path_measured_at(
        measured: ArmPair<JointVec>,
    ) -> (ArmPair<Planner>, PathJudgment, Result<(), Failure>) {
        let mut planners = planners_holding(BOTH_HELD);
        let arms = order(&["left_arm", "right_arm"]);
        let move_slots = claimed(&arms);
        let (reply, _done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let goal = path_goal(&arms, &BOTH_TO, Duration::ZERO, reply, &move_slots);
        let judgment =
            start_path(&mut planners, goal, &measured.map(|_, q| Ok(q))).expect("the path starts");
        let verdict = judge_path(
            &judgment.plan,
            judgment.arms,
            &test_guard(&planners),
            &mut OpenGuard,
            &NO_STOP,
        );
        (planners, judgment, verdict)
    }

    /// The arms of a path of both, their commands off where they measure,
    /// hold their commands while the path is judged, and start the path on
    /// one tick, the judgment's, both commands moved onto the path's run
    /// start, where the arms measure, and both shares' clocks started then:
    /// the left arm's command stands 0.3 rad off its start at joint 1 and
    /// 0.1 rad at joint 3, the right arm's 0.1 rad at joint 4.
    #[tokio::test]
    async fn the_arms_of_a_path_of_both_start_it_together_on_their_measured_start() {
        let mut start = BOTH_HELD;
        start.left[0] += 0.3;
        start.left[2] += 0.1;
        start.right[3] += 0.1;
        let (mut planners, judgment, verdict) = both_path_measured_at(start);
        let t0 = Instant::now();
        let tick = plan_config().cycle_period;
        for k in 0..3 {
            let ticks = tick_both(&mut planners, t0 + tick * k).await;
            let candidates = chase_together(&planners, ticks.map(|_, tick| tick.candidate));
            assert_eq!(
                candidates, BOTH_HELD,
                "the commands hold while the path is judged"
            );
            commit_both(&mut planners, &candidates);
        }
        let judged_at = t0 + tick * 3;
        let measured = start.map(|_, q| Ok(q));
        let resumed = resume_judged_path(&mut planners, &judgment, &verdict, &measured, judged_at);
        assert!(matches!(resumed, Resumed::Done), "the path starts");
        assert_eq!(
            ArmPair::from_fn(|side| planners[side].setpoint()),
            start,
            "both commands move onto the path's run start"
        );
        for side in Side::BOTH {
            assert_eq!(path_of(&planners[side]).clock().motion_start(), judged_at);
        }
        assert_eq!(
            path_of(&planners.left).sample(judged_at),
            dual_joints(&start)
        );
    }

    /// An arm that measures farther than its start tolerance from the path's
    /// start once the path is judged ends the path with
    /// `precondition_failed`, naming the arm and the joint, before the path
    /// plays.
    #[tokio::test]
    async fn an_arm_moved_off_the_path_start_while_judged_fails_the_path() {
        let (mut planners, judgment, verdict, mut done_rx) =
            left_path_measured_at(BOTH_HELD.left, &[BOTH_TO], Duration::from_secs(1));
        let mut moved = BOTH_HELD.left;
        moved[1] += 0.02;
        let resumed = resume_judged_path(
            &mut planners,
            &judgment,
            &verdict,
            &left_measuring(moved),
            Instant::now(),
        );
        assert!(matches!(resumed, Resumed::Done), "the path ends");
        assert!(!planners.left.moving(), "before it plays");
        assert_eq!(
            outcome(done_rx.try_recv()),
            Some(Outcome::Failed(Failure::new(
                FailureKind::PreconditionFailed,
                "left_arm stands 0.020 rad at joint 2 from where it stood when the path was \
                 taken, past the 0.010 rad start tolerance; send the path again"
            )))
        );
    }

    /// An arm that stands within its start tolerance of the goal's expected
    /// start when the coordinator takes the goal, and moves while the path is
    /// judged to a run start within the tolerance of where it was taken but
    /// farther than it from the expected start, ends the path with
    /// `precondition_failed` in the words of an expected start, before the
    /// path plays: the expected start lies 0.006 rad past where the arm was
    /// taken at joint 2, and the run start 0.006 rad short of it.
    #[tokio::test]
    async fn a_run_start_off_the_expected_start_fails_the_path_in_its_words() {
        let mut planners = planners_holding(BOTH_HELD);
        let arms = order(&["left_arm"]);
        let move_slots = claimed(&arms);
        let (reply, mut done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let mut expected = BOTH_HELD.left;
        expected[1] += 0.006;
        let goal = PathGoal {
            expected_starts: vec![Some(expected)],
            ..path_goal(&arms, &BOTH_TO, Duration::ZERO, reply, &move_slots)
        };
        let measured = measured_at_setpoints(&planners);
        let judgment = start_path(&mut planners, goal, &measured).expect("the path starts");
        let verdict = judge_path(
            &judgment.plan,
            judgment.arms,
            &test_guard(&planners),
            &mut OpenGuard,
            &NO_STOP,
        );
        let mut moved = BOTH_HELD.left;
        moved[1] -= 0.006;
        let resumed = resume_judged_path(
            &mut planners,
            &judgment,
            &verdict,
            &left_measuring(moved),
            Instant::now(),
        );
        assert!(matches!(resumed, Resumed::Done), "the path ends");
        assert!(!planners.left.moving(), "before it plays");
        assert_eq!(
            planners.left.setpoint(),
            BOTH_HELD.left,
            "the command holds"
        );
        assert_eq!(
            outcome(done_rx.try_recv()),
            Some(Outcome::Failed(Failure::new(
                FailureKind::PreconditionFailed,
                "left_arm stands 0.012 rad from expected_start_joint_positions at joint 2, past \
                 the 0.010 rad start tolerance; plan the path from where the arm stands"
            )))
        );
    }

    /// Judge `judgment`'s path with `checker`, and resume its shares at
    /// `now`, the arms measuring `measured`, judging the path again from where
    /// they stand as often as the resume asks: how many times the path was
    /// judged, and how the shares came out of the last resume.
    fn judge_until_resumed(
        planners: &mut ArmPair<Planner>,
        mut judgment: PathJudgment,
        measured: &ArmPair<Measurement>,
        checker: &mut dyn PathGuard,
        now: Instant,
    ) -> (u32, Resumed) {
        let mut judged = 0;
        loop {
            let guard = test_guard(planners);
            let verdict = judge_path(&judgment.plan, judgment.arms, &guard, checker, &NO_STOP);
            judged += 1;
            match resume_judged_path(planners, &judgment, &verdict, measured, now) {
                Resumed::JudgeAgain(again) => judgment = *again,
                resumed => return (judged, resumed),
            }
        }
    }

    /// An arm still moving while its path is judged, within its start
    /// tolerance of where it measured when the coordinator took the goal,
    /// has the path judged again from where it measures, and then runs the
    /// path from there: its command moves there, the path's first point
    /// stands there, and no command of the path steps back toward where the
    /// arm measured when the goal was taken. The commands hold while the path
    /// is judged again.
    #[tokio::test]
    async fn an_arm_moving_while_its_path_is_judged_runs_the_path_from_where_it_then_measures() {
        let taken = BOTH_HELD.left;
        let mut moved = taken;
        moved[1] -= 0.008;
        let (mut planners, judgment, verdict, _done_rx) =
            left_path_measured_at(taken, &[BOTH_TO], Duration::from_secs(1));
        let started = Instant::now();
        let measured = left_measuring(moved);
        let Resumed::JudgeAgain(again) =
            resume_judged_path(&mut planners, &judgment, &verdict, &measured, started)
        else {
            panic!("the path is judged again from where the arm stands");
        };
        assert_eq!(
            planners.left.setpoint(),
            BOTH_HELD.left,
            "the command holds"
        );
        assert_eq!(arm_joints(&again.plan.start(), Side::Left), moved);
        let (judged, resumed) =
            judge_until_resumed(&mut planners, *again, &measured, &mut OpenGuard, started);
        assert!(matches!(resumed, Resumed::Done), "the path starts");
        assert_eq!(judged, 1, "judged once more");
        assert_eq!(
            planners.left.setpoint(),
            moved,
            "the command moves onto where the arm measures"
        );
        assert_eq!(
            arm_joints(&path_of(&planners.left).sample(started), Side::Left),
            moved,
            "the path runs from there"
        );
        let tick = plan_config().cycle_period;
        let mut k = 0;
        while planners.left.moving() {
            let ticks = tick_both(&mut planners, started + tick * k).await;
            commit_both(&mut planners, &ticks.map(|_, tick| tick.candidate));
            assert!(
                planners.left.setpoint()[1] <= moved[1],
                "tick {k} steps the command back toward where the goal was taken"
            );
            k += 1;
        }
    }

    /// The arms of a path of both, one of them moving while the path is
    /// judged, have the path judged again from where both measure, and start
    /// it together from there on one trajectory: the right elbow moves 0.005
    /// rad, and the left arm stands where it was taken.
    #[tokio::test]
    async fn a_path_of_both_arms_one_of_which_moved_while_judged_runs_from_where_both_measure() {
        let mut moved = BOTH_HELD;
        moved.right[3] += 0.005;
        let (mut planners, judgment, verdict) = both_path_measured_at(BOTH_HELD);
        let now = Instant::now();
        let measured = moved.map(|_, q| Ok(q));
        let Resumed::JudgeAgain(again) =
            resume_judged_path(&mut planners, &judgment, &verdict, &measured, now)
        else {
            panic!("the path is judged again from where both arms stand");
        };
        assert_eq!(again.plan.start(), dual_joints(&moved));
        let (_, resumed) =
            judge_until_resumed(&mut planners, *again, &measured, &mut OpenGuard, now);
        assert!(matches!(resumed, Resumed::Done), "the path starts");
        assert_eq!(ArmPair::from_fn(|side| planners[side].setpoint()), moved);
        assert_eq!(path_of(&planners.left).sample(now), dual_joints(&moved));
        assert_eq!(path_of(&planners.right).sample(now), dual_joints(&moved));
    }

    /// An arm that never comes to rest within 0.001 rad of where it stood
    /// when its path was last judged, though within its start tolerance of
    /// where the goal was taken, fails the path with `precondition_failed`
    /// once the path has been judged 20 times, naming the arm and the joint,
    /// before the path plays: the left shoulder swings 0.004 rad either side
    /// of where it was taken.
    #[tokio::test]
    async fn an_arm_that_will_not_settle_fails_its_path_after_twenty_judgments() {
        let (mut planners, mut judgment, mut verdict, mut done_rx) =
            left_path_measured_at(BOTH_HELD.left, &[BOTH_TO], Duration::from_secs(1));
        let now = Instant::now();
        let swung = |k: u32| {
            let mut q = BOTH_HELD.left;
            q[0] += if k.is_multiple_of(2) { 0.004 } else { -0.004 };
            left_measuring(q)
        };
        for k in 1..MAX_PATH_JUDGMENTS {
            let Resumed::JudgeAgain(again) =
                resume_judged_path(&mut planners, &judgment, &verdict, &swung(k), now)
            else {
                panic!("judgment {k}: the path is judged again");
            };
            judgment = *again;
            verdict = judge_path(
                &judgment.plan,
                judgment.arms,
                &test_guard(&planners),
                &mut OpenGuard,
                &NO_STOP,
            );
        }
        let resumed = resume_judged_path(
            &mut planners,
            &judgment,
            &verdict,
            &swung(MAX_PATH_JUDGMENTS),
            now,
        );
        assert!(matches!(resumed, Resumed::Done), "the path ends");
        assert!(!planners.left.moving(), "before it plays");
        assert_eq!(
            planners.left.setpoint(),
            BOTH_HELD.left,
            "the command holds"
        );
        assert_eq!(
            outcome(done_rx.try_recv()),
            Some(Outcome::Failed(Failure::new(
                FailureKind::PreconditionFailed,
                "left_arm still moved 0.008 rad at joint 1 while its path was judged 20 times; \
                 let the arm settle and send the path again"
            )))
        );
    }

    /// A path the joint caps run within 60 s from where the arm stood when
    /// the goal was taken, and past the 60 s a move may take from where it
    /// stands when the path would start, fails with `rejected` when it is
    /// judged again from there, before it plays: the left shoulder swings
    /// between 1 and -3 rad at 0.5 rad/s, and the arm measures 0.9 rad
    /// behind where it was taken when the path would start, within a start
    /// tolerance of 1 rad.
    #[tokio::test]
    async fn a_path_the_caps_run_past_60_s_from_its_run_start_fails_rejected() {
        let mut planners = planners_holding(BOTH_HELD);
        for side in Side::BOTH {
            planners[side].cfg.max_joint_velocity_rad_s = [0.5; ARM_DOF];
            planners[side].cfg.start_tolerance_rad = 1.0;
        }
        let shoulder = planners.left.cfg.limits[0];
        assert!(shoulder.lo < -3.0 && shoulder.hi > 1.0, "{shoulder:?}");
        let swung = |q0: f64| {
            let mut q = BOTH_HELD;
            q.left[0] = q0;
            q
        };
        let waypoints: Vec<ArmPair<JointVec>> = (0..8)
            .map(|k| swung(if k % 2 == 0 { 1.0 } else { -3.0 }))
            .collect();
        let arms = order(&["left_arm"]);
        let move_slots = claimed(&arms);
        let (reply, mut done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let goal = path_goal_through(
            &arms,
            &waypoints,
            Duration::ZERO,
            test_settle(&arms),
            reply,
            |side| &move_slots[side],
        );
        let measured = measured_at_setpoints(&planners);
        let judgment = start_path(&mut planners, goal, &measured).expect("the path starts");
        let judged_s = judgment.plan.duration().as_secs_f64();
        assert!(
            (58.0..MAX_MOVE_DURATION_S).contains(&judged_s),
            "{judged_s} s from where the goal was taken"
        );
        let verdict = judge_path(
            &judgment.plan,
            judgment.arms,
            &test_guard(&planners),
            &mut OpenGuard,
            &NO_STOP,
        );
        assert_eq!(verdict, Ok(()), "the judgment lets the path run");
        let behind = left_measuring(swung(-0.9).left);
        let Resumed::JudgeAgain(again) =
            resume_judged_path(&mut planners, &judgment, &verdict, &behind, Instant::now())
        else {
            panic!("the path is judged again from where the arm stands");
        };
        let (_, resumed) = judge_until_resumed(
            &mut planners,
            *again,
            &behind,
            &mut OpenGuard,
            Instant::now(),
        );
        assert!(matches!(resumed, Resumed::Done), "the path ends");
        assert!(!planners.left.moving(), "before it plays");
        assert_eq!(
            planners.left.setpoint(),
            BOTH_HELD.left,
            "the command holds"
        );
        let Some(Outcome::Failed(failure)) = outcome(done_rx.try_recv()) else {
            panic!("the share fails");
        };
        assert_eq!(failure.kind, FailureKind::Rejected, "{}", failure.message);
        assert!(
            failure.message.ends_with(
                " s to run this path, past the 60 s a move may take; split it into paths of \
                 fewer waypoints"
            ),
            "{}",
            failure.message
        );
    }

    /// A path starts at the measured joints clamped into the joint limits:
    /// an elbow measuring just under its floor starts the path at the floor,
    /// where its command moves, and the path's first leg moves the arm. So
    /// does an elbow 0.03 rad under its floor, past the 0.01 rad start
    /// tolerance: once the path is planned, the arm's measured joints,
    /// clamped as the start was, stand at the start.
    #[tokio::test]
    async fn a_path_starts_at_the_measured_joints_clamped_into_the_limits() {
        let floor = planners_holding(BOTH_HELD).left.cfg.limits[3].lo;
        for below in [0.004, 0.03] {
            let mut under = BOTH_HELD.left;
            under[3] = floor - below;
            let mut clamped = under;
            clamped[3] = floor;
            let (mut planners, judgment, verdict, mut done_rx) =
                left_path_measured_at(under, &[BOTH_TO], Duration::from_secs(1));
            let started = Instant::now();
            let start = dual_joints(&ArmPair::new(clamped, BOTH_HELD.right));
            assert_eq!(judgment.plan.trajectory(started).sample(started), start);
            let measured = left_measuring(under);
            let resumed =
                resume_judged_path(&mut planners, &judgment, &verdict, &measured, started);
            assert!(
                matches!(resumed, Resumed::Done),
                "{below} rad under: the path starts"
            );
            assert_eq!(
                planners.left.setpoint(),
                clamped,
                "{below} rad under: the command moves onto the clamped start"
            );
            assert!(
                done_rx.try_recv().is_err(),
                "{below} rad under: the path runs"
            );
            let tick = plan_config().cycle_period;
            for k in 1..=10 {
                let ticks = tick_both(&mut planners, started + tick * k).await;
                commit_both(&mut planners, &ticks.map(|_, tick| tick.candidate));
            }
            assert_ne!(
                planners.left.setpoint(),
                clamped,
                "{below} rad under: the first leg moves the arm"
            );
        }
    }

    /// A path of one arm moves that arm alone: the other arm's planner keeps
    /// following its stream, and the guard checks the path with the other
    /// arm where it is held, the path's arms named as one arm.
    #[tokio::test]
    async fn a_path_of_one_arm_holds_the_other_where_it_stands() {
        struct Recording(Vec<(PathArms, Vec<f64>, Vec<f64>)>);
        impl PathGuard for Recording {
            fn check(
                &mut self,
                arms: PathArms,
                _guard: &GuardAt,
                path: &RoundedPath,
                _stop: &AtomicBool,
            ) -> Result<(), String> {
                self.0.push((arms, path.at(0.0), path.end().to_vec()));
                Ok(())
            }
        }
        let mut planners = planners_holding(BOTH_HELD);
        let arms = order(&["left_arm"]);
        let move_slots = claimed(&arms);
        let (reply, _done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let goal = path_goal(&arms, &BOTH_TO, Duration::ZERO, reply, &move_slots);
        let mut guard = Recording(Vec::new());
        start(&mut planners, goal, &mut guard, Instant::now()).expect("the path starts");
        let [(checked, from, to)] = &guard.0[..] else {
            panic!("the guard checks the path once: {:?}", guard.0);
        };
        assert_eq!(*checked, PathArms::One(Side::Left));
        assert_eq!(from[..], dual_joints(&BOTH_HELD));
        assert_eq!(
            to[..],
            dual_joints(&ArmPair::new(BOTH_TO.left, BOTH_HELD.right))
        );
        assert_eq!(planners.left.path_arms(), Some(PathArms::One(Side::Left)));
        assert!(!planners.right.moving(), "the right arm is not on the path");
        assert!(move_slots.left.is_claimed() && !move_slots.right.is_claimed());
    }

    /// A share whose arm stands within its settle tolerance of the path's
    /// end, but whose clock the guard held back, fails at the settle timeout
    /// naming the path it did not finish.
    #[tokio::test]
    async fn a_share_the_guard_holds_short_of_its_path_fails_its_settle_naming_the_path() {
        let mut planners = planners_holding(BOTH_HELD);
        let arms = order(&["left_arm"]);
        let move_slots = claimed(&arms);
        let (reply, mut done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let mut to = BOTH_HELD;
        to.left[0] += 0.005;
        let second = Duration::from_secs(1);
        let goal = path_goal(&arms, &to, second, reply, &move_slots);
        let start_at = Instant::now();
        start(&mut planners, goal, &mut OpenGuard, start_at).expect("the path starts");
        // The guard holds every step: each tick commits where the arm is held.
        for ms in [0, 500, 1000, 1500, 1990] {
            tick_both(&mut planners, start_at + Duration::from_millis(ms)).await;
            commit_both(&mut planners, &BOTH_HELD);
        }
        assert!(done_rx.try_recv().is_err(), "the share settles on");
        tick_both(&mut planners, start_at + 2 * second).await;
        commit_both(&mut planners, &BOTH_HELD);
        assert_eq!(
            outcome(done_rx.try_recv()),
            Some(Outcome::Failed(Failure::new(
                FailureKind::Timeout,
                "still short of the end of its path, 1.0 s after the planned end"
            )))
        );
        assert!(!move_slots.left.is_claimed(), "the failure frees the arm");
    }

    /// A share is judged from the first tick at or after its planned end:
    /// with a settle timeout of 0, or one under a tick, an arm that tracks
    /// its setpoints completes on the tick of the planned end. Its
    /// report gives the joints the arm measured then, one tick behind the
    /// path's end and within the tolerance of it, and the time since the path
    /// started.
    #[tokio::test]
    async fn a_share_with_no_settle_time_completes_on_the_tick_that_ends_its_path() {
        let tick = Duration::from_millis(10);
        let second = Duration::from_secs(1);
        for timeout in [Duration::ZERO, Duration::from_millis(5)] {
            let mut planners = planners_holding(BOTH_HELD);
            let arms = order(&["left_arm"]);
            let move_slots = claimed(&arms);
            let (reply, mut done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
            let settle = Settle {
                tolerance: vec![0.01; ARM_DOF],
                timeout,
            };
            let goal = path_goal_through(&arms, &[BOTH_TO], second, settle, reply, |side| {
                &move_slots[side]
            });
            let t0 = Instant::now();
            start(&mut planners, goal, &mut OpenGuard, t0).expect("the path starts");
            let mut ended = None;
            for k in 0..150 {
                let measured = planners.left.setpoint();
                let ticks = tick_both(&mut planners, t0 + tick * k).await;
                commit_both(&mut planners, &ticks.map(|_, tick| tick.candidate));
                if let Ok(report) = done_rx.try_recv() {
                    ended = Some((tick * k, measured, report));
                    break;
                }
            }
            let (at, measured, report) = ended.expect("the share ends");
            assert_eq!(at, second, "timeout {timeout:?}");
            assert_eq!(report.outcome, Outcome::Complete, "timeout {timeout:?}");
            assert_eq!(report.joints, measured);
            assert_eq!(report.elapsed_s, 1.0);
        }
    }

    /// An execute_joint_path goal whose expected start of an arm lies past
    /// the start tolerance of where that arm measures fails before any arm
    /// moves, naming the arm and the joint; one within the tolerance starts,
    /// and an arm with no expected start starts wherever it measures.
    #[tokio::test]
    async fn an_expected_start_off_the_measured_arm_fails_the_path_naming_the_arm_and_joint() {
        let arms = order(&["right_arm", "left_arm"]);
        let expecting = |right: Option<JointVec>, left: Option<JointVec>| vec![right, left];
        let measured_off = |by: f64| {
            let mut right = BOTH_HELD.right;
            right[2] += by;
            ArmPair::new(Ok(BOTH_HELD.left), Ok(right))
        };
        let try_start = |expected_starts: Vec<Option<JointVec>>, measured: ArmPair<Measurement>| {
            let mut planners = planners_holding(BOTH_HELD);
            let move_slots = claimed(&arms);
            let (reply, mut done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
            let goal = PathGoal {
                expected_starts,
                ..path_goal(&arms, &BOTH_TO, Duration::ZERO, reply, &move_slots)
            };
            let started = start_path(&mut planners, goal, &measured).map(|_| ());
            let moving = ArmPair::from_fn(|side| planners[side].moving());
            let held = move_slots.map(|_, slot| slot.is_claimed());
            assert!(done_rx.try_recv().is_err(), "no share ends at its start");
            (started, moving, held)
        };
        let both_expected = expecting(Some(BOTH_HELD.right), Some(BOTH_HELD.left));

        let (failed, moving, held) = try_start(both_expected.clone(), measured_off(0.1));
        assert_eq!(
            failed,
            Err(Failure::new(
                FailureKind::PreconditionFailed,
                "right_arm stands 0.100 rad from expected_start_joint_positions at joint 3, past \
                 the 0.010 rad start tolerance; plan the path from where the arm stands"
            ))
        );
        assert_eq!(moving, ArmPair::new(false, false), "no arm moves");
        assert_eq!(
            held,
            ArmPair::new(false, false),
            "the failure frees both arms"
        );

        let (started, moving, _) = try_start(both_expected, measured_off(0.005));
        assert_eq!(started, Ok(()), "an arm within the tolerance starts");
        assert_eq!(moving, ArmPair::new(true, true));

        let (started, _, _) = try_start(expecting(None, Some(BOTH_HELD.left)), measured_off(0.5));
        assert_eq!(
            started,
            Ok(()),
            "an arm with no expected start is not checked"
        );
    }

    /// A path on an arm that runs another move, or whose follower is stale,
    /// fails before any arm moves, labelled with that arm on a path of both
    /// arms; the move in flight goes on.
    #[tokio::test]
    async fn a_path_on_a_busy_or_stale_arm_fails_before_any_arm_moves() {
        let mut planners = planners_holding(BOTH_HELD);
        let left = order(&["left_arm"]);
        let running_slots = claimed(&left);
        let (reply, mut running) = share_reply(&Arc::new(AtomicBool::new(false)));
        let goal = path_goal(&left, &BOTH_TO, Duration::ZERO, reply, &running_slots);
        start(&mut planners, goal, &mut OpenGuard, Instant::now()).expect("the path starts");

        let both = order(&["left_arm", "right_arm"]);
        let (reply, _done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let second_slots = ArmPair::from_fn(|_| MoveSlot::default());
        let goal = path_goal(&both, &BOTH_TO, Duration::ZERO, reply, &second_slots);
        let failed = start(&mut planners, goal, &mut OpenGuard, Instant::now());
        assert_eq!(
            failed,
            Err(
                Failure::new(FailureKind::PreconditionFailed, ANOTHER_MOVE_IN_FLIGHT)
                    .labelled(Side::Left.arm_name())
            )
        );
        assert!(!planners.right.moving(), "the right arm does not start");
        assert!(running.try_recv().is_err(), "the move in flight goes on");
        assert!(running_slots.left.is_claimed(), "and holds its slot");

        let mut planners = planners_holding(BOTH_HELD);
        let move_slots = claimed(&both);
        let (reply, _done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let goal = path_goal(&both, &BOTH_TO, Duration::ZERO, reply, &move_slots);
        let measured = ArmPair::new(Ok(BOTH_HELD.left), Err(Unmeasured::Stale));
        let failed = start_path(&mut planners, goal, &measured).map(|_| ());
        assert_eq!(
            failed,
            Err(stale_failure().labelled(Side::Right.arm_name()))
        );
        assert!(!planners.left.moving() && !planners.right.moving());
        assert!(move_slots.iter().all(|slot| !slot.is_claimed()));

        let mut planners = planners_holding(BOTH_HELD);
        let move_slots = claimed(&left);
        let (reply, _done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let goal = path_goal(&left, &BOTH_TO, Duration::ZERO, reply, &move_slots);
        let measured = ArmPair::new(Err(Unmeasured::Stale), Ok(BOTH_HELD.right));
        let failed = start_path(&mut planners, goal, &measured).map(|_| ());
        assert_eq!(
            failed,
            Err(stale_failure()),
            "a goal of one arm names its arm"
        );
        assert!(!move_slots.left.is_claimed());
    }

    /// A path the guard would hold short fails with `unreachable`, running none
    /// of the path: each of its shares ends with the failure of the path, which
    /// names no arm, and frees its arm.
    #[tokio::test]
    async fn a_path_the_guard_would_hold_short_fails_each_share_and_frees_its_arms() {
        let arms = order(&["left_arm", "right_arm"]);
        let mut planners = planners_holding(BOTH_HELD);
        let move_slots = claimed(&arms);
        let (reply, mut done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let goal = path_goal(&arms, &BOTH_TO, Duration::ZERO, reply, &move_slots);
        let unreachable = Failure::new(FailureKind::Unreachable, ARMS_WOULD_MEET);
        let started = start(&mut planners, goal, &mut Refusing, Instant::now());
        assert_eq!(started, Err(unreachable.clone()));
        assert!(!planners.left.moving() && !planners.right.moving());
        assert!(move_slots.iter().all(|slot| !slot.is_claimed()));
        for _ in Side::BOTH {
            assert_eq!(
                outcome(done_rx.try_recv()),
                Some(Outcome::Failed(unreachable.clone()))
            );
        }
    }

    /// A path the joint velocity and acceleration caps cannot run within 60 s
    /// is judged `rejected`, the message giving the time they take, before
    /// the guard is asked.
    #[test]
    fn a_path_the_caps_cannot_run_within_60_s_is_judged_rejected() {
        let planners = planners_holding(BOTH_HELD);
        let plan = PathPlan {
            waypoints: (0..=100)
                .map(|k| {
                    let mut arms = BOTH_HELD;
                    arms.left[1] -= if k % 2 == 1 { 0.8 } else { 0.0 };
                    dual_joints(&arms)
                })
                .collect(),
            max_velocity: [0.5; DUAL_DOF],
            max_acceleration: [0.5; DUAL_DOF],
            duration: Duration::ZERO,
        };
        let judged = judge_path(
            &plan,
            PathArms::One(Side::Left),
            &test_guard(&planners),
            &mut Refusing,
            &NO_STOP,
        )
        .expect_err("past 60 s");
        assert_eq!(judged.kind, FailureKind::Rejected);
        assert!(
            judged
                .message
                .starts_with("the joint velocity and acceleration caps take ")
                && judged.message.ends_with(
                    " s to run this path, past the 60 s a move may take; split it into paths \
                     of fewer waypoints"
                ),
            "{}",
            judged.message
        );
    }

    /// A judgment that finds its stop raised when it begins ends at once with
    /// the failure of a stopped check, never asking the guard, which here
    /// refuses every path in words of its own.
    #[test]
    fn a_judgment_stopped_before_it_begins_asks_no_guard() {
        let planners = planners_holding(BOTH_HELD);
        let plan = PathPlan::through(
            vec![dual_joints(&BOTH_HELD), dual_joints(&BOTH_TO)],
            [1.0; DUAL_DOF],
            [1.0; DUAL_DOF],
            Duration::ZERO,
        );
        let judged = judge_path(
            &plan,
            PathArms::Both,
            &test_guard(&planners),
            &mut Refusing,
            &AtomicBool::new(true),
        );
        assert_eq!(
            judged,
            Err(Failure::new(
                FailureKind::Unreachable,
                PathBlocked::Stopped.to_string()
            ))
        );
    }

    /// Failing an arm's share of a path leaves any other move of the arm
    /// running: a joint move runs on and reports nothing.
    #[tokio::test]
    async fn failing_a_path_share_leaves_a_joint_move_running() {
        let mut planner = planner_holding(Side::Left, BOTH_HELD.left);
        let (goal_tx, mut goals) = mpsc::channel(1);
        let move_slot = claimed_move_slot();
        let (done_tx, mut done_rx) = mpsc::channel(1);
        goal_tx
            .send(Goal::posture_share(BOTH_TO.left, done_tx))
            .await
            .expect("queue the goal");
        planner
            .tick(BOTH_HELD.left, None, &mut goals, &move_slot, Instant::now())
            .await;
        assert!(planner.moving(), "the joint move runs");
        planner.fail_path_share(stale_failure());
        assert!(planner.moving(), "the joint move runs on");
        assert!(done_rx.try_recv().is_err(), "and reports nothing");
    }

    /// The shares of an execute_joint_path goal hold their arms until the
    /// path is judged, whatever time passes, and start on the judgment, the
    /// path's clock from the judgment on; a judgment for shares that ended
    /// before it moves nothing.
    #[tokio::test]
    async fn the_shares_of_a_judged_path_hold_until_the_judgment_and_start_on_it() {
        let mut planners = planners_holding(BOTH_HELD);
        let arms = order(&["left_arm", "right_arm"]);
        let move_slots = claimed(&arms);
        let (reply, mut done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let second = Duration::from_secs(1);
        let goal = path_goal(&arms, &BOTH_TO, second, reply, &move_slots);
        let t0 = Instant::now();
        let measured = measured_at_setpoints(&planners);
        let judgment = start_path(&mut planners, goal, &measured).expect("the path starts");
        for ms in [0, 500, 5000] {
            let ticks = tick_both(&mut planners, t0 + Duration::from_millis(ms)).await;
            assert!(
                ticks
                    .iter()
                    .all(|tick| tick.on_path == Some(PathArms::Both))
            );
            let candidates = ticks.map(|_, tick| tick.candidate);
            assert_eq!(candidates, BOTH_HELD, "the arms hold until the judgment");
            commit_both(&mut planners, &candidates);
        }
        assert!(done_rx.try_recv().is_err(), "the shares wait");

        let judged_at = t0 + Duration::from_secs(5);
        let guard = test_guard(&planners);
        let verdict = judge_path(
            &judgment.plan,
            judgment.arms,
            &guard,
            &mut OpenGuard,
            &NO_STOP,
        );
        let held = measured_at_setpoints(&planners);
        let resumed = resume_judged_path(&mut planners, &judgment, &verdict, &held, judged_at);
        assert!(matches!(resumed, Resumed::Done), "the path starts");
        assert!(on_one_path(&planners), "both shares follow the path");
        let clock = path_of(&planners.left).clock();
        assert_eq!(clock.motion_start(), judged_at);
        assert_eq!(
            path_of(&planners.left).sample(judged_at),
            dual_joints(&BOTH_HELD),
            "the path starts where the arms held"
        );

        let mut planners = planners_holding(BOTH_HELD);
        let move_slots = claimed(&arms);
        let (reply, mut done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let goal = path_goal(&arms, &BOTH_TO, second, reply, &move_slots);
        let measured = measured_at_setpoints(&planners);
        let judgment = start_path(&mut planners, goal, &measured).expect("the path starts");
        for side in Side::BOTH {
            assert!(
                planners[side]
                    .stop_active("stopped: operator", Ok(BOTH_HELD[side]), t0)
                    .await,
                "a share waiting for its judgment is a move in flight"
            );
            let report = done_rx.try_recv().expect("the stopped share reports");
            assert_eq!(
                report.outcome,
                Outcome::Cancelled("stopped: operator".to_string())
            );
            assert_eq!(report.joints, BOTH_HELD[side], "with the joints measured");
            assert_eq!(report.elapsed_s, 0.0, "its path never started");
        }
        assert!(move_slots.iter().all(|slot| !slot.is_claimed()));
        let guard = test_guard(&planners);
        let verdict = judge_path(
            &judgment.plan,
            judgment.arms,
            &guard,
            &mut OpenGuard,
            &NO_STOP,
        );
        let held = measured_at_setpoints(&planners);
        let resumed = resume_judged_path(&mut planners, &judgment, &verdict, &held, t0);
        assert!(
            matches!(resumed, Resumed::Done),
            "no share waits for the judgment"
        );
        assert!(
            !planners.left.moving() && !planners.right.moving(),
            "the judgment of a stopped path moves nothing"
        );
    }

    /// A path of both arms stops where both arms repeat their waypoint
    /// before, exactly or within 0.001 rad in every joint, and runs on
    /// through a waypoint only one arm repeats: there the other arm keeps
    /// moving.
    #[tokio::test]
    async fn a_path_of_both_arms_stops_only_where_both_arms_repeat_a_waypoint() {
        let arms = order(&["left_arm", "right_arm"]);
        let first = BOTH_TO;
        let mut moved_on = first;
        moved_on.left[0] += 0.3;
        moved_on.right[3] += 0.2;
        let mut right_only = first;
        right_only.right[3] += 0.1;
        let last = moved_on;
        // The speed (rad/s) of each arm and of both together, at the tick of
        // the path's trajectory nearest `second`.
        let speeds_at_second = async |second: ArmPair<JointVec>| {
            let mut planners = planners_holding(BOTH_HELD);
            let move_slots = claimed(&arms);
            let (reply, _done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
            let goal = path_goal_through(
                &arms,
                &[first, second, last],
                Duration::ZERO,
                test_settle(&arms),
                reply,
                |side| &move_slots[side],
            );
            let t0 = Instant::now();
            start(&mut planners, goal, &mut OpenGuard, t0).expect("the path starts");
            let traj = path_of(&planners.left);
            let tick = Duration::from_millis(1);
            let ticks = traj.clock().duration().div_duration_f64(tick).ceil() as u32;
            let at = |k: u32| traj.sample(t0 + tick * k);
            let to_second = |k: u32| {
                let q = at(k);
                let target = dual_joints(&second);
                (0..DUAL_DOF)
                    .map(|i| (q[i] - target[i]).powi(2))
                    .sum::<f64>()
            };
            let k = (1..ticks)
                .min_by(|&a, &b| to_second(a).total_cmp(&to_second(b)))
                .expect("a tick inside the path");
            let (before, after) = (at(k - 1), at(k + 1));
            let speed = |joints: std::ops::Range<usize>| {
                joints
                    .map(|i| (after[i] - before[i]).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    / (2.0 * tick.as_secs_f64())
            };
            (
                speed(0..ARM_DOF),
                speed(ARM_DOF..DUAL_DOF),
                speed(0..DUAL_DOF),
            )
        };

        let (_, right, both) = speeds_at_second(right_only).await;
        assert!(both > 0.05, "the path runs on through it at {both} rad/s");
        assert!(right > 0.05, "the right arm keeps moving at {right} rad/s");

        let (left, right, both) = speeds_at_second(first).await;
        assert!(
            both < 0.01,
            "the arms come to rest at the waypoint both repeat: {left} and {right} rad/s"
        );

        let mut near = first;
        near.left[1] += 0.0008;
        near.right[2] -= 0.0008;
        let (left, right, both) = speeds_at_second(near).await;
        assert!(
            both < 0.01,
            "the arms come to rest at the waypoint within 0.001 rad of the one before: \
             {left} and {right} rad/s"
        );
    }

    // --- Following a path ---------------------------------------------------

    /// The governor's share of each proposed step paces the path's clock: a
    /// held step plays nothing, so the next tick proposes the same point
    /// again, and a half step plays half the tick, so the tick after that
    /// proposes the point of the path the clock has really reached.
    #[tokio::test]
    async fn the_governed_share_of_each_step_paces_the_path_clock() {
        let mut planners = planners_holding(BOTH_HELD);
        let arms = order(&["left_arm"]);
        let move_slots = claimed(&arms);
        let (reply, _done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let second = Duration::from_secs(1);
        let goal = path_goal(&arms, &BOTH_TO, second, reply, &move_slots);
        let t0 = Instant::now();
        start(&mut planners, goal, &mut OpenGuard, t0).expect("the path starts");
        let held = BOTH_HELD.left;
        let first = propose(&mut planners.left, t0).await;
        planners.left.commit(first);
        let tenth = Duration::from_millis(100);
        let proposed = propose(&mut planners.left, t0 + tenth).await;
        planners.left.commit(held);
        let again = propose(&mut planners.left, t0 + 2 * tenth).await;
        assert_eq!(again, proposed, "a held step plays nothing");
        let half: JointVec = std::array::from_fn(|i| held[i] + 0.5 * (again[i] - held[i]));
        planners.left.commit(half);
        let next = propose(&mut planners.left, t0 + 3 * tenth).await;
        let fresh = JointTrajectory::through(
            &[
                dual_joints(&BOTH_HELD),
                dual_joints(&ArmPair::new(BOTH_TO.left, BOTH_HELD.right)),
            ],
            &[plan_config().max_joint_velocity_rad_s[0]; DUAL_DOF],
            &[plan_config().max_joint_acceleration_rad_s2[0]; DUAL_DOF],
            second,
            t0,
        );
        let want = arm_joints(&fresh.sample(t0 + Duration::from_millis(150)), Side::Left);
        for i in 0..ARM_DOF {
            assert!(
                (next[i] - want[i]).abs() <= 1e-9,
                "joint {i} proposed at {}, the path's point at 0.15 s is {}",
                next[i],
                want[i]
            );
        }
    }

    /// The candidate `planner` proposes at `at`, the arm measured where it
    /// was governed last.
    async fn propose(planner: &mut Planner, at: Instant) -> JointVec {
        let (_goal_tx, mut goals) = mpsc::channel(1);
        let held = planner.setpoint();
        planner
            .tick(held, None, &mut goals, &MoveSlot::default(), at)
            .await
            .candidate
    }

    /// How far both arms' setpoints, together, stand off the straight path
    /// through both arms' joints from [`BOTH_HELD`] to [`BOTH_TO`], the path
    /// of one waypoint the tests start from there.
    fn off_the_path_of_both(planners: &ArmPair<Planner>) -> f64 {
        let (from, to) = (dual_joints(&BOTH_HELD), dual_joints(&BOTH_TO));
        let at = dual_joints(&ArmPair::from_fn(|side| planners[side].setpoint()));
        let run: DualJointVec = std::array::from_fn(|i| to[i] - from[i]);
        let share = (0..DUAL_DOF)
            .map(|i| (at[i] - from[i]) * run[i])
            .sum::<f64>()
            / run.iter().map(|r| r * r).sum::<f64>();
        (0..DUAL_DOF)
            .map(|i| (at[i] - from[i] - share * run[i]).abs())
            .fold(0.0, f64::max)
    }

    /// Both arms start together on one trajectory, step along it as one path
    /// and play their clocks at one pace: a tick the governor holds both arms
    /// to half their step plays both clocks on by that half, and after every
    /// tick both arms' setpoints, together, lie on the path through both
    /// arms' joints.
    /// Both arms end at the last waypoint, each share completing once its arm
    /// settles.
    #[tokio::test]
    async fn both_shares_play_their_clocks_at_one_pace_and_end_together() {
        let mut planners = planners_holding(BOTH_HELD);
        let arms = order(&["right_arm", "left_arm"]);
        let move_slots = claimed(&arms);
        let (reply, mut done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let goal = path_goal(&arms, &BOTH_TO, Duration::ZERO, reply, &move_slots);
        let start_at = Instant::now();
        start(&mut planners, goal, &mut OpenGuard, start_at).expect("the move starts");
        assert!(on_one_path(&planners));
        let on_the_path = |planners: &ArmPair<Planner>| {
            let off = off_the_path_of_both(planners);
            assert!(off <= 1e-12, "the arms stand {off} rad off their path");
        };

        let at = start_at + Duration::from_millis(100);
        let ticks = tick_both(&mut planners, at).await;
        assert!(
            Side::BOTH
                .into_iter()
                .all(|side| ticks[side].on_path == Some(PathArms::Both))
        );
        let halves = ArmPair::from_fn(|side| -> JointVec {
            let held = planners[side].setpoint();
            std::array::from_fn(|i| (held[i] + ticks[side].candidate[i]) / 2.0)
        });
        commit_both(&mut planners, &halves);
        on_the_path(&planners);
        let later = at + Duration::from_millis(50);
        assert_eq!(
            path_of(&planners.left).sample(later),
            path_of(&planners.right).sample(later),
            "both shares play at one pace"
        );
        assert!(
            path_of(&planners.left).sample(later) != path_of(&planners.left).sample(at),
            "the clock moved on"
        );

        let mut now = later;
        let mut outcomes = Vec::new();
        while outcomes.len() < 2 && now < start_at + Duration::from_secs(30) {
            let ticks = tick_both(&mut planners, now).await;
            let candidates = chase_together(&planners, ticks.map(|_, tick| tick.candidate));
            commit_both(&mut planners, &candidates);
            on_the_path(&planners);
            if on_one_path(&planners) {
                assert_eq!(
                    path_of(&planners.left).sample(now),
                    path_of(&planners.right).sample(now),
                    "both shares play at one pace"
                );
            }
            while let Ok(report) = done_rx.try_recv() {
                outcomes.push(report.outcome);
            }
            now += Duration::from_millis(10);
        }
        assert_eq!(outcomes, [Outcome::Complete, Outcome::Complete]);
        assert_eq!(ArmPair::from_fn(|side| planners[side].setpoint()), BOTH_TO);
        assert!(move_slots.iter().all(|slot| !slot.is_claimed()));
    }

    /// A late tick that caps one share's chase at its velocity limit steps
    /// the other share by the same share of its step, so both stay on one
    /// line through both arms' joints.
    #[tokio::test]
    async fn a_capped_share_holds_the_other_to_its_share() {
        let mut planners = planners_holding(BOTH_HELD);
        let arms = order(&["left_arm", "right_arm"]);
        let mut to = BOTH_TO;
        to.left[1] = BOTH_HELD.left[1] - 1.0;
        let move_slots = claimed(&arms);
        let (reply, _done_rx) = share_reply(&Arc::new(AtomicBool::new(false)));
        let goal = path_goal(&arms, &to, Duration::ZERO, reply, &move_slots);
        let start_at = Instant::now();
        start(&mut planners, goal, &mut OpenGuard, start_at).expect("the move starts");
        let ticks = tick_both(&mut planners, start_at + Duration::from_millis(400)).await;
        let shares = |candidates: &ArmPair<JointVec>| {
            ArmPair::from_fn(|side| {
                planners[side]
                    .proposed
                    .expect("a share proposes its step")
                    .pace(&candidates[side])
            })
        };
        let chased = ticks.map(|_, tick| tick.candidate);
        let alone = shares(&chased);
        assert!(alone.left < alone.right, "the left share's chase is capped");
        let together = shares(&chase_together(&planners, chased));
        assert!((together.left.get() - alone.left.get()).abs() < 1e-12);
        assert!((together.right.get() - together.left.get()).abs() < 1e-12);
    }

    /// A share that fails ends the other on its next tick, which holds its
    /// arm where it stands; a cancel ends both, each holding its arm.
    #[tokio::test]
    async fn a_failed_share_ends_the_other_and_a_cancel_ends_both() {
        let arms = order(&["left_arm", "right_arm"]);
        let start_at = Instant::now();
        let started = async |cancelled: &Arc<AtomicBool>| {
            let mut planners = planners_holding(BOTH_HELD);
            let move_slots = claimed(&arms);
            let (reply, done_rx) = share_reply(cancelled);
            let goal = path_goal(&arms, &BOTH_TO, Duration::ZERO, reply, &move_slots);
            start(&mut planners, goal, &mut OpenGuard, start_at).expect("the move starts");
            (planners, done_rx, move_slots)
        };

        let (mut planners, mut done_rx, move_slots) =
            started(&Arc::new(AtomicBool::new(false))).await;
        let ticks = tick_both(&mut planners, start_at + Duration::from_millis(200)).await;
        commit_both(&mut planners, &ticks.map(|_, tick| tick.candidate));
        planners.right.fail_active(stale_failure()).await;
        let Some(Outcome::Failed(stale)) = outcome(done_rx.try_recv()) else {
            panic!("the stale share fails");
        };
        assert_eq!(stale, stale_failure().labelled(Side::Right.arm_name()));
        let held_left = planners.left.setpoint();
        let tick = tick_both(&mut planners, start_at + Duration::from_millis(300)).await;
        let Some(Outcome::Failed(ended)) = outcome(done_rx.try_recv()) else {
            panic!("the other share fails");
        };
        assert_eq!(
            ended,
            Failure::new(FailureKind::AttemptFailed, OTHER_SHARE_FAILED)
                .labelled(Side::Left.arm_name())
        );
        assert_eq!(tick.left.candidate, held_left, "the other arm holds");
        assert!(move_slots.iter().all(|slot| !slot.is_claimed()));

        let cancelled = Arc::new(AtomicBool::new(false));
        let (mut planners, mut done_rx, _move_slots) = started(&cancelled).await;
        tick_both(&mut planners, start_at + Duration::from_millis(100)).await;
        let held = ArmPair::from_fn(|side| planners[side].setpoint());
        cancelled.store(true, Ordering::Release);
        let ticks = tick_both(&mut planners, start_at + Duration::from_millis(200)).await;
        for side in Side::BOTH {
            assert_eq!(
                outcome(done_rx.try_recv()),
                Some(Outcome::Cancelled(SHARE_CANCELLED.to_string()))
            );
            assert_eq!(ticks[side].candidate, held[side], "{side} holds");
        }
    }
}
