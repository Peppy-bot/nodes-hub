//! Per-arm motion planner: the mode state machine that turns one arm's inputs
//! (the leading node's joint or pose stream, and accepted joint / Cartesian
//! move goals) into a candidate joint setpoint each tick. It does NOT command
//! anything and does not know about the other arm: it produces a candidate,
//! the coordinator governs both arms' candidates against the collision model,
//! and feeds the governed result back via [`Planner::commit`] so the next
//! tick chases from where the arm was actually allowed to go.
//!
//! Every mode reduces to "chase a target": the setpoint advances toward the
//! target at the per-joint velocity limits, so streaming and moves stay smooth
//! under throttling - when the governor holds the setpoint, the chase simply
//! catches up at the velocity limit once clear, with no jump. Follow's target
//! is the leading node's command (a streamed hand's end-effector speed is
//! capped by the governor's EE-speed limiter, fed by the Jacobians this
//! planner hands out); a joint move's target is its trajectory's sample, and a
//! Cartesian move's is the sample of the trajectory through its line's walk.
//!
//! An execute_joint_path goal moves the arms it names along one path through
//! both arms' joints, each arm's planner following its arm's share of it
//! ([`path`]): the one mode an arm's planner shares with the other arm's.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use peppygen::exposed_actions::limb_motion::{move_arm, move_arm_joints};

use control_core::chase::{chase_along, chase_step};
use control_core::path_timing::{JointTrajectory, distinct_joint_waypoints};
use control_core::positive_finite::PositiveFinite;
use control_core::time::{duration_from_secs, ticks_within};
use control_core::timing::MotionClock;
use robot_description::ArmId;
use srs_model::ARM_DOF;
use srs_model::chain_kinematics::{EeCaps, StepLimits, clamp_into, rate_step_toward};
use srs_model::nalgebra::{Isometry3, Translation3, UnitQuaternion};
use srs_model::{Arm, JointVec, Limit, plan_line};
use tokio::sync::mpsc;
use tracing::{error, info};

use crate::failure::{Failure, FailureKind};
use crate::governor::{HandBasis, PathArms};
use crate::move_slot::{ARM_BUSY, MoveSlot, SlotRelease};
use crate::robot::LIMBS;
use crate::settle::{
    ContactGap, PoseTolerance, Settle, Verdict, joint_shortfall, pose_shortfall, verdict,
};
use crate::types::world_pose_arrays;
use crate::upstream::Upstream;

mod path;
mod stream_smoothing;

use stream_smoothing::StreamSmoothing;

pub use path::{
    JointPath, PathGoal, PathGuard, PathJudgment, PathOrder, Resumed, ShareReply, ShareReport,
    chase_together, commit_both, judge_path, resume_judged_path, start_path,
};
#[cfg(test)]
pub(crate) use path::{OTHER_SHARE_FAILED, PathPlan};

/// Why a Cartesian move has no plan: the message of a `move_arm` goal's
/// `unreachable` failure, and the reason `check_arm_move` answers.
const NOT_PLANNED: &str = "goal pose not planned: no straight line to it tracks inside the joint \
     limits; plan a path with plan_joint_path and run it with execute_joint_path";

/// What a caller does about a move the joint limits cannot run within the longest
/// a move may take.
const SPLIT_INTO_SHORTER_MOVES: &str = "split it into shorter moves";
/// The same of a posture move, which a caller cannot split.
const RAISE_THE_JOINT_LIMITS: &str =
    "raise max_joint_velocity_rad_s_N or max_joint_acceleration_rad_s2_N";

/// The failure of a goal that reaches the planner while another move is in
/// flight on its arm.
const ANOTHER_MOVE_IN_FLIGHT: &str = "another move is in flight";

/// The reason a posture share ends with when its posture goal is cancelled.
/// The posture action ends its goal with the reason of its own cancel.
const POSTURE_CANCELLED: &str = "its posture goal was cancelled";

/// Per-arm static configuration for the planner (the motion limits relocated
/// from the arm). Cloned per arm.
#[derive(Clone)]
pub struct PlanConfig {
    pub cycle_period: Duration,
    pub max_joint_velocity_rad_s: JointVec,
    pub max_joint_acceleration_rad_s2: JointVec,
    /// The operator's cap on a streamed hand's speed (m/s), which a streamed
    /// pose steps at.
    pub stream_ee_velocity_m_s: f64,
    pub limits: [Limit; ARM_DOF],
    /// How far (rad, each joint) the measured arm may stand from a joint
    /// path's start: from its expected start when the coordinator takes the
    /// goal and at its run start, and at its run start from where it stood
    /// when the goal was taken.
    pub start_tolerance_rad: f64,
}

/// Why an arm has no fresh measurement of its joints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unmeasured {
    /// The seed wait is not over: an arm or a gripper has not reported its
    /// first state yet.
    NotYet,
    /// The follower stopped reporting, so its last measurement is stale.
    Stale,
}

/// The joints an arm measured, or why it has no fresh measurement of them.
pub type Measurement = Result<JointVec, Unmeasured>;

/// The note a cancelled move_arm or move_arm_joints goal adds to its reason
/// when its result reports the joints commanded last, because the arm has
/// no fresh measurement.
const COMMANDED_JOINTS_NOTE: &str = "the result comes from the joints commanded last";

/// How a move ended. A posture share reports it to the posture action that
/// sent it, the failure labelled with the name of its arm.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// The move ran to its end, and an arm move with a settle settled.
    Complete,
    /// The move ended on contact before it settled, and its goal completes:
    /// `this_arm` for the arm whose grasp point stood past the goal's contact
    /// gap, and unset for an arm of the same joint path that this contact
    /// ended.
    Contact { this_arm: bool },
    /// The move was cancelled, for this reason: by its caller, or by the
    /// stop service, whose reason names the stop.
    Cancelled(String),
    /// The move failed, and its goal gives no result.
    Failed(Failure),
}

/// What `check_arm_move` answers for an arm that can take the move.
#[derive(Debug, PartialEq)]
pub enum PlanCheck {
    /// A plan reaches the pose; the move would take this long.
    Reachable(Duration),
    /// No plan reaches the pose, in the words of the `unreachable` failure of
    /// a move_arm goal.
    Unreachable(String),
}

/// How the share of an arm in a posture goal (move_to_ready or move_to_home)
/// settles: each measured joint within 0.1 rad of the posture, the tolerance
/// the robots' endpoint calls usual for a joint move, by 2 s after the
/// planned end. A share whose arm an object or the governor holds short of
/// the posture fails with `timeout`, naming the joint and how far it stands.
pub const POSTURE_SETTLE: Settle<JointVec> = Settle {
    tolerance: [0.1; ARM_DOF],
    timeout: Duration::from_secs(2),
};

/// One limb's share of a posture goal (move_to_ready or move_to_home): where
/// its terminal reports, and the cancel flag the posture action raises on
/// the goal's cancel. The share of an arm is its move to the posture; the
/// share of a gripper is its close ahead of the move to home.
pub struct PostureShare {
    pub done_tx: mpsc::Sender<Outcome>,
    pub cancelled: Arc<AtomicBool>,
}

/// Where a joint move reports its terminal: the move_arm_joints action goal
/// that carried it (boxed: the goal context dwarfs the other variant) with
/// the settle and the contact gap the goal asked for, or one arm's share of
/// a posture goal (move_to_ready or move_to_home), which settles by
/// [`POSTURE_SETTLE`] and carries no contact gap.
pub enum JointReply {
    MoveArmJoints {
        ctx: Box<move_arm_joints::GoalContext>,
        settle: Settle<JointVec>,
        contact: ContactGap,
    },
    Posture(PostureShare),
}

impl JointReply {
    /// The reason of the cancel the move received, or `None` before one: the
    /// cause of the move_arm_joints goal's cancel signal, or the cancel of
    /// the posture goal, which its action flags.
    fn cancel_reason(&self) -> Option<String> {
        match self {
            Self::MoveArmJoints { ctx, .. } => {
                ctx.cancel_cause().map(|cause| cause.reason().to_string())
            }
            Self::Posture(share) => share.cancel_reason(),
        }
    }

    /// The contact gap the move ends on contact by: the one its
    /// move_arm_joints goal asked for, and none for a posture share.
    fn contact_gap(&self) -> ContactGap {
        match self {
            Self::MoveArmJoints { contact, .. } => *contact,
            Self::Posture(_) => ContactGap::NONE,
        }
    }

    /// How a joint move ends at `now`, with the arm measuring `measured`
    /// against `target` and `contact` saying whether it stands on contact
    /// ([`verdict`]): from its planned end on it completes once every joint
    /// stands within its settle tolerance, on any tick it completes on
    /// contact, it fails with `timeout` once its settle timeout has passed,
    /// and it stays in flight otherwise; a posture share by
    /// [`POSTURE_SETTLE`], a move_arm_joints goal by the settle it asked for.
    fn settle_outcome(
        &self,
        measured: &JointVec,
        target: &JointVec,
        contact: bool,
        clock: &MotionClock,
        now: Instant,
    ) -> Option<Outcome> {
        let settle = match self {
            Self::Posture(_) => &POSTURE_SETTLE,
            Self::MoveArmJoints { settle, .. } => settle,
        };
        verdict(
            || joint_shortfall(measured, target, &settle.tolerance),
            contact,
            planned_end(clock),
            now,
            settle.timeout,
        )
        .map(Outcome::from)
    }

    /// End the move of `planner`'s arm on `outcome`, with `measured`, the
    /// joints the arm measured when the move ended. A move_arm_joints goal
    /// completes, saying whether on contact, or ends cancelled with what
    /// [`Planner::reported_joints`] gives, and fails with no result. A
    /// posture share reports its outcome
    /// and no joints: the posture action measures both arms when its goal
    /// ends.
    async fn finish(
        self,
        planner: &Planner,
        outcome: Outcome,
        measured: Measurement,
        elapsed_s: f64,
    ) {
        let arm = planner.arm;
        let ctx = match self {
            Self::MoveArmJoints { ctx, .. } => ctx,
            Self::Posture(share) => return share.report(LIMBS.arm_name(arm), outcome).await,
        };
        let joints = planner.reported_joints(&measured);
        let ended = match outcome {
            Outcome::Complete | Outcome::Contact { .. } => {
                ctx.complete(joints, outcome.ended_on_contact(), elapsed_s)
                    .await
            }
            Outcome::Cancelled(reason) => {
                let reason = cancelled_reason(reason, &measured);
                ctx.complete_cancelled(reason, joints, false, elapsed_s)
                    .await
            }
            Outcome::Failed(failure) => ctx.fail(failure.kind.into(), failure.message).await,
        };
        if let Err(e) = ended {
            error!("{}: move_arm_joints end: {e}", LIMBS.arm_name(arm));
        }
    }

    /// Fail the move of the arm `arm` with `failure`, which reports no
    /// joints.
    async fn fail(self, arm: ArmId, failure: Failure) {
        match self {
            Self::MoveArmJoints { ctx, .. } => {
                if let Err(e) = ctx.fail(failure.kind.into(), failure.message).await {
                    error!("{}: move_arm_joints end: {e}", LIMBS.arm_name(arm));
                }
            }
            Self::Posture(share) => {
                share
                    .report(LIMBS.arm_name(arm), Outcome::Failed(failure))
                    .await
            }
        }
    }
}

impl PostureShare {
    /// The reason of the cancel of the posture goal, which its action flags,
    /// or `None` before one.
    pub fn cancel_reason(&self) -> Option<String> {
        self.cancelled
            .load(Ordering::Acquire)
            .then(|| POSTURE_CANCELLED.to_string())
    }

    /// Report how the share of `limb` ended, its failure labelled with the
    /// limb: the name of an arm or of a gripper.
    pub async fn report(self, limb: impl std::fmt::Display, outcome: Outcome) {
        let outcome = match outcome {
            Outcome::Failed(failure) => Outcome::Failed(failure.labelled(&limb)),
            other => other,
        };
        if self.done_tx.send(outcome).await.is_err() {
            error!("{limb}: posture outcome aggregation closed");
        }
    }
}

/// A Cartesian move as `move_arm` plans it and `check_arm_move` checks it,
/// parsed once off the wire: both carry the same planning fields with the
/// same meaning, and both refuse the same values with the same words.
#[derive(Clone, Copy, Debug)]
pub struct ArmMoveRequest {
    pub arm: ArmId,
    pub position: Translation3<f64>,
    /// `None` asks for the position alone, at the turn the hand holds.
    pub orientation: Option<UnitQuaternion<f64>>,
    pub duration: Duration,
}

/// A goal accepted by an action handler and handed to the planner.
pub enum Goal {
    Joint {
        target: JointVec,
        duration: Duration,
        reply: JointReply,
    },
    Cartesian {
        request: ArmMoveRequest,
        settle: Settle<PoseTolerance>,
        contact: ContactGap,
        ctx: Box<move_arm::GoalContext>,
    },
}

impl Goal {
    /// A posture share to `target` over one second, reporting on `done_tx`:
    /// the goal of the unit tests of the planner and the coordinator.
    #[cfg(test)]
    pub fn posture_share(target: JointVec, done_tx: mpsc::Sender<Outcome>) -> Self {
        Goal::Joint {
            target,
            duration: Duration::from_secs(1),
            reply: JointReply::Posture(PostureShare {
                done_tx,
                cancelled: Arc::new(AtomicBool::new(false)),
            }),
        }
    }

    /// Fail unstarted with `failure`, with no result. The move slot is the
    /// caller's concern.
    pub async fn fail(self, failure: Failure, planner: &Planner) {
        match self {
            Goal::Joint { reply, .. } => reply.fail(planner.arm, failure).await,
            Goal::Cartesian { ctx, .. } => planner.fail_cartesian(&ctx, failure).await,
        }
    }

    /// End unstarted as cancelled by the stop service, for `reason`, which
    /// names the stop, reporting `measured`. The move slot is the caller's
    /// concern.
    pub async fn stop(self, reason: &str, measured: Measurement, planner: &Planner) {
        let outcome = Outcome::Cancelled(reason.to_string());
        match self {
            Goal::Joint { reply, .. } => reply.finish(planner, outcome, measured, 0.0).await,
            Goal::Cartesian { ctx, .. } => {
                planner.finish_cartesian(&ctx, outcome, measured, 0.0).await
            }
        }
    }
}

enum Mode {
    /// Ambient: chase the leading node's stream, or hold when none is streaming.
    Follow,
    /// Tracking a joint trajectory for an accepted move_arm_joints goal.
    JointMove(JointMove),
    /// Tracking the trajectory through a straight line's walk for an accepted
    /// move_arm goal.
    CartesianMove(CartesianMove),
    /// The arm's share of an execute_joint_path goal's path.
    PathShare(path::PathShare),
}

struct JointMove {
    traj: JointTrajectory<ARM_DOF>,
    reply: JointReply,
    _move_slot: SlotRelease,
}

struct CartesianMove {
    traj: JointTrajectory<ARM_DOF>,
    ctx: Box<move_arm::GoalContext>,
    /// The commanded pose the settle judges the measured grasp point against.
    target: Isometry3<f64>,
    settle: Settle<PoseTolerance>,
    contact: ContactGap,
    _move_slot: SlotRelease,
}

/// Where this tick's joint target came from. Only a streamed target carries a
/// hand basis out for the governor's EE-speed limiter: the cap covers streams,
/// and a planned move tracks the schedule the joint limits timed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Origin {
    Streamed,
    PlannedMove,
    /// A step of the command toward where the arm stood when its move ended
    /// short of its target ([`Planner::hold_where_it_stands`]).
    Hold,
    /// The next point of the arm's share of the path of these arms, or
    /// its setpoint while the path is judged.
    OnPath(PathArms),
}

/// One planner tick's product: the velocity-limited candidate setpoint, and
/// the end-effector Jacobians of the step when the target came from the operator's
/// stream (the governor's EE-speed limiter caps only those).
/// A tick of the arm's share of a path, which holds the arm while the path is
/// judged and steps it along the path once it runs, names the path's arms,
/// which the governor scales along the step and never turns off it.
pub struct Tick {
    pub candidate: JointVec,
    /// Whether the candidate is a step of a hold: the command on its way to
    /// where its arm stands at rest ([`Planner::hold_where_it_stands`]). The
    /// arm is asked for no velocity with such a step.
    pub holds: bool,
    pub streamed_hand: Option<HandBasis>,
    pub on_path: Option<PathArms>,
}

/// One mode's advance result: the joint target this tick, the next mode (the
/// state transition), and where the target came from.
struct Advance {
    target: JointVec,
    next_mode: Mode,
    origin: Origin,
}

impl Advance {
    /// A move mode's terminal: hold `target` and hand the arm back to Follow.
    fn ends_move(target: JointVec) -> Self {
        Self {
            target,
            next_mode: Mode::Follow,
            origin: Origin::PlannedMove,
        }
    }

    /// A step of the command toward `stood`, where the arm stood when its
    /// move ended short of its target, with the arm in Follow.
    fn holds(stood: JointVec) -> Self {
        Self {
            target: stood,
            next_mode: Mode::Follow,
            origin: Origin::Hold,
        }
    }
}

/// The arm's command on its way to where the arm stood when its last move
/// ended short of its target ([`Planner::hold_where_it_stands`]).
#[derive(Clone, Copy)]
struct Hold {
    /// The joints the arm measured then, inside the joint limits.
    stood: JointVec,
    /// The steps the command has left to get there.
    steps_left: u32,
}

impl Hold {
    /// The hold of a command at `from` on its way to `stood`: the steps the
    /// way takes at `max_velocity` (rad/s per joint) with a step each
    /// `cycle_period`, and one more, which the last step takes when the
    /// governor's own speed cap trims a step before it by a rounding.
    fn toward(
        stood: JointVec,
        from: &JointVec,
        max_velocity: &JointVec,
        cycle_period: Duration,
    ) -> Self {
        let travel_s = (0..ARM_DOF)
            .map(|j| (stood[j] - from[j]).abs() / max_velocity[j])
            .fold(0.0, f64::max);
        let steps =
            duration_from_secs(travel_s).map_or(0, |travel| ticks_within(travel, cycle_period));
        Self {
            stood,
            steps_left: steps + 1,
        }
    }

    /// The hold after one more step of it: none once its steps ran out.
    fn after_a_step(self) -> Option<Self> {
        let steps_left = self.steps_left.checked_sub(1).filter(|left| *left > 0)?;
        Some(Self { steps_left, ..self })
    }
}

pub struct Planner {
    arm: ArmId,
    model: Arm,
    cfg: PlanConfig,
    mode: Mode,
    /// Last governed setpoint: the chase base and the value held when idle.
    setpoint: JointVec,
    /// The step a share of a path proposed this tick, which the commit of the
    /// governed setpoint plays the path's clock on by.
    proposed: Option<path::ProposedStep>,
    /// The command's way to where the arm stood when its last move ended
    /// short of its target, while it is under way.
    hold: Option<Hold>,
    /// The smoother a streamed pose's steps pass through; reset whenever the
    /// stream is not a pose, so a resumed stream seeds on the setpoint.
    stream_smoothing: StreamSmoothing,
}

impl Planner {
    /// Start holding zero with no producer locked. The coordinator seeds the real
    /// held pose from the first measured state (via [`Planner::commit`]) before any
    /// setpoint is published, so this initial value is never streamed.
    pub fn new(arm: ArmId, model: Arm, cfg: PlanConfig) -> Self {
        Self {
            arm,
            model,
            stream_smoothing: StreamSmoothing::new(cfg.cycle_period),
            cfg,
            mode: Mode::Follow,
            setpoint: [0.0; ARM_DOF],
            proposed: None,
            hold: None,
        }
    }

    /// Adopt the governed setpoint the coordinator actually published, so the next
    /// tick chases from there (not from the ungoverned candidate). A share of
    /// a path plays the path's clock on by the share of its proposed step the
    /// governor allowed ([`Planner::commit_paced`]).
    pub fn commit(&mut self, governed: JointVec) {
        let pace = self.pace_of(&governed);
        let contact = path::PathContact::of_one(self.stands_on_path_contact());
        self.commit_paced(governed, pace, contact);
    }

    /// Seed the held setpoint from a measured pose of the arm, its first one
    /// or the one its follower's stream recovers on, clamped into the
    /// joint limits, the limits the arm clamps every command into: a power-up pose
    /// parked past a limit (e.g. the elbow a hair below its lower limit, the straight
    /// arm) seeds the setpoint at that limit, so the backbone's held setpoint and the
    /// arm's pose agree from the first tick. The seed ends a hold under way.
    pub fn seed_from_measured(&mut self, measured: JointVec) {
        self.setpoint = clamp_into(&self.cfg.limits, &measured);
        self.hold = None;
    }

    /// The last published setpoint, the coordinator's `prev` for the governor.
    pub fn setpoint(&self) -> JointVec {
        self.setpoint
    }

    /// The advance that ends a move on `outcome`. A move that settled holds
    /// `end`, the end of its plan. A move that ended short of it leaves the
    /// arm where it stands: `stood` is where the arm measures on that tick,
    /// and the command sets out for it ([`Planner::hold_where_it_stands`]).
    /// A share of a path that had not started to play has no `stood`, and
    /// its command stays where it is.
    fn end_move(&mut self, outcome: &Outcome, end: JointVec, stood: Option<JointVec>) -> Advance {
        if !outcome.ended_short() {
            return Advance::ends_move(end);
        }
        let Some(stood) = stood else {
            return Advance::ends_move(self.setpoint);
        };
        self.hold_where_it_stands(stood);
        self.hold_step()
            .unwrap_or_else(|| Advance::ends_move(self.setpoint))
    }

    /// Leave the arm where it stands at `stood`, the joints it measures. The
    /// command moves to them, clamped into the joint limits, a step of the
    /// joint velocity limits each tick, and the governor judges each step as
    /// it judges every other. The command stops on `stood`, or where its
    /// steps run out when the collision guard slows or holds them
    /// ([`Hold::toward`]). A move or a leader's stream that takes the
    /// command, and a seed from a measured pose, end the hold; a share of a
    /// path that waits for its judgment leaves the hold standing, and the
    /// hold goes on if the path never runs. An arm that an object holds
    /// short of its command stops pushing on it this way, and the guard
    /// keeps its say over a command it holds at the stop distance.
    fn hold_where_it_stands(&mut self, stood: JointVec) {
        let stood = clamp_into(&self.cfg.limits, &stood);
        self.hold = Some(Hold::toward(
            stood,
            &self.setpoint,
            &self.cfg.max_joint_velocity_rad_s,
            self.cfg.cycle_period,
        ));
    }

    /// This tick's step of the hold under way, if one is: toward where the
    /// arm stood, until the command stands there or the hold's steps ran out.
    fn hold_step(&mut self) -> Option<Advance> {
        let hold = self
            .hold
            .take()
            .filter(|hold| hold.stood != self.setpoint)?;
        self.hold = hold.after_a_step();
        Some(Advance::holds(hold.stood))
    }

    /// Whether the arm stands on contact for a goal whose contact gap is
    /// `gap`: its measured grasp point, at the joints `measured`, farther
    /// from the grasp point at `commanded`, the joints it was commanded to
    /// last, than the gap allows. Every arm move asks this on every tick,
    /// and a goal with no contact check gets its answer with no kinematics.
    fn stands_on_contact(
        &self,
        gap: ContactGap,
        commanded: &JointVec,
        measured: &JointVec,
    ) -> bool {
        if gap == ContactGap::NONE {
            return false;
        }
        let grasp_point = |q: &JointVec| self.world_ee_pose(q).translation.vector;
        gap.is_exceeded_by((grasp_point(commanded) - grasp_point(measured)).norm())
    }

    /// The joints a move_arm or move_arm_joints result reports: the
    /// measured joints, or without a fresh measurement the held setpoint,
    /// which is the joints commanded last ([`cancelled_reason`] says so).
    fn reported_joints(&self, measured: &Measurement) -> JointVec {
        match measured {
            Ok(q) => *q,
            Err(_) => self.setpoint,
        }
    }

    /// The end effector's pose at `q` in the world frame, from the planner's
    /// arm model.
    pub fn world_ee_pose(&self, q: &JointVec) -> Isometry3<f64> {
        self.model.world_ee_pose(q)
    }

    /// Retune the stream's end-effector speed cap at runtime (the commander's
    /// control). This copy steps streamed poses; a streamed joint command is
    /// capped only by the governor's EE-speed limiter, retuned from the same
    /// control message.
    pub fn set_max_stream_ee_velocity(&mut self, v: f64) {
        if let Ok(cap) = PositiveFinite::parse(v) {
            self.cfg.stream_ee_velocity_m_s = cap.get();
        }
    }

    /// Produce this tick's candidate setpoint: admit a pending goal, advance the
    /// active mode to a joint target, then chase it under the velocity limits.
    /// A pending goal that a stop came before, after its admission
    /// ([`MoveSlot::take_stop`]), does not start: it ends cancelled for the
    /// stop's reason, reporting `measured_q`, and frees its move slot.
    ///
    /// A streamed tick also carries the end-effector Jacobians of its step: the
    /// governor's EE-speed limiter caps a streamed hand, and this is the one place
    /// the arm's kinematic model lives. A planned move
    /// carries `None`: it runs on the timing it was admitted with.
    pub async fn tick(
        &mut self,
        measured_q: JointVec,
        command: Option<Upstream>,
        goals: &mut mpsc::Receiver<Goal>,
        move_slot: &MoveSlot,
        now: Instant,
    ) -> Tick {
        let mut mode = std::mem::replace(&mut self.mode, Mode::Follow);
        // Drain fully: every queued goal is answered, never left parked.
        while let Ok(goal) = goals.try_recv() {
            if let Some(reason) = move_slot.take_stop() {
                let _release = move_slot.release_on_drop();
                goal.stop(&reason, Ok(measured_q), self).await;
            } else if matches!(mode, Mode::Follow) {
                mode = self
                    .start_goal(goal, move_slot.release_on_drop(), now)
                    .await;
            } else {
                let failure = Failure::new(FailureKind::PreconditionFailed, ANOTHER_MOVE_IN_FLIGHT);
                goal.fail(failure, self).await;
            }
        }

        let Advance {
            target,
            next_mode,
            origin,
        } = self.advance(mode, measured_q, &command, now).await;
        self.mode = next_mode;

        let dt = self.cfg.cycle_period.as_secs_f64();
        let vmax = &self.cfg.max_joint_velocity_rad_s;
        let stepped = match origin {
            Origin::OnPath(_) => {
                self.proposed = Some(path::ProposedStep {
                    from: self.setpoint,
                    to: target,
                    at: now,
                    measured: measured_q,
                });
                chase_along(&self.setpoint, &target, vmax, dt)
            }
            Origin::Hold => step_onto(&self.setpoint, &target, vmax, dt),
            Origin::Streamed | Origin::PlannedMove => chase_step(&self.setpoint, &target, vmax, dt),
        };
        let candidate = clamp_into(&self.cfg.limits, &stepped);
        Tick {
            candidate,
            holds: origin == Origin::Hold,
            streamed_hand: match origin {
                Origin::Streamed => {
                    let midpoint: JointVec =
                        std::array::from_fn(|i| 0.5 * (self.setpoint[i] + candidate[i]));
                    Some(HandBasis {
                        start: self.model.at(&self.setpoint).jacobian(),
                        midpoint: self.model.at(&midpoint).jacobian(),
                    })
                }
                Origin::PlannedMove | Origin::Hold | Origin::OnPath(_) => None,
            },
            on_path: match origin {
                Origin::OnPath(arms) => Some(arms),
                Origin::Streamed | Origin::PlannedMove | Origin::Hold => None,
            },
        }
    }

    /// Fail the active move with `failure` and answer its caller, with no
    /// result: for a follower gone stale, where the move can neither
    /// progress nor be verified. Ambient Follow mode is untouched; the move
    /// slot releases with the mode.
    pub async fn fail_active(&mut self, failure: Failure) {
        match std::mem::replace(&mut self.mode, Mode::Follow) {
            Mode::Follow => {}
            Mode::JointMove(JointMove { reply, .. }) => reply.fail(self.arm, failure).await,
            Mode::CartesianMove(m) => self.fail_cartesian(&m.ctx, failure).await,
            Mode::PathShare(m) => m.fail(self, failure),
        }
    }

    /// End the active move for the stop service: its goal ends cancelled
    /// for `reason`, which names the stop, reporting `measured`, the joints
    /// measured when the stop came. The arm holds where it stands, as after
    /// a cancel ([`Planner::hold_where_it_stands`]); the command of an arm
    /// with no fresh measurement, and of a share whose path had not started
    /// to play, stays where it was last governed. The move slot releases
    /// with the mode. Whether a move was in flight.
    pub async fn stop_active(&mut self, reason: &str, measured: Measurement, now: Instant) -> bool {
        let outcome = Outcome::Cancelled(reason.to_string());
        let played = match std::mem::replace(&mut self.mode, Mode::Follow) {
            Mode::Follow => return false,
            Mode::JointMove(JointMove {
                traj,
                reply,
                _move_slot,
            }) => {
                let elapsed = traj.clock().elapsed_secs(now);
                reply.finish(self, outcome, measured, elapsed).await;
                true
            }
            Mode::CartesianMove(m) => {
                let elapsed = m.traj.clock().elapsed_secs(now);
                self.finish_cartesian(&m.ctx, outcome, measured, elapsed)
                    .await;
                true
            }
            Mode::PathShare(m) => {
                let played = m.runs();
                m.end(self, outcome, measured, now);
                played
            }
        };
        if let (true, Ok(stood)) = (played, measured) {
            self.hold_where_it_stands(stood);
        }
        true
    }

    /// Whether a `move_arm` goal of `request` has a plan from the held setpoint,
    /// without moving: the time the move would take, or the words of its failure.
    /// Refused while a move executes, as a second goal is, with
    /// `precondition_failed`; the check claims no slot and changes no state.
    pub fn check_cartesian(&mut self, request: &ArmMoveRequest) -> Result<PlanCheck, Failure> {
        if !matches!(self.mode, Mode::Follow) {
            return Err(Failure::new(FailureKind::PreconditionFailed, ARM_BUSY));
        }
        let Some(walk) = plan_line(&self.model, &self.line_goal(request), self.setpoint) else {
            return Ok(PlanCheck::Unreachable(NOT_PLANNED.to_string()));
        };
        let traj = self.line_trajectory(&walk, request.duration, Instant::now());
        Ok(
            match path::check_move_time(&traj, "move", SPLIT_INTO_SHORTER_MOVES) {
                Ok(()) => PlanCheck::Reachable(traj.clock().duration()),
                Err(failure) => PlanCheck::Unreachable(failure.message),
            },
        )
    }

    /// Resolve the Follow target, holding the last governed setpoint when no
    /// command has arrived. The coordinator hands in `None` for a leader that
    /// has stopped streaming, so a stopped leader's limb holds here.
    ///
    /// A joint command is chased directly. A pose gets one damped
    /// resolved-rate step from the *governed* setpoint (measured lags by
    /// tracking error, and seeding from it would fold that error into every
    /// step); bounded rather than solved, so an unreachable pose saturates
    /// into a steady pull the governor can shape.
    fn follow_target(&mut self, command: &Option<Upstream>) -> JointVec {
        let target = match command {
            None => {
                self.stream_smoothing.reset();
                return self.setpoint;
            }
            Some(Upstream::Joints(positions)) => {
                self.stream_smoothing.reset();
                *positions
            }
            Some(Upstream::Pose(pose)) => {
                let q = self.setpoint;
                let ee = self.world_ee_pose(&q);
                let limits = StepLimits {
                    max_joint_velocity: self.cfg.max_joint_velocity_rad_s,
                    ee: self.stream_step_rates(),
                    dt_s: self.cfg.cycle_period.as_secs_f64(),
                };
                let step = rate_step_toward(self.model.chain(), &q, &ee, pose, &limits);
                // Smoothed, then held to the velocity limit again: a smoother
                // can overshoot its input.
                let smoothed = self.stream_smoothing.smooth(&step);
                chase_step(&q, &smoothed, &limits.max_joint_velocity, limits.dt_s)
            }
        };
        clamp_into(&self.cfg.limits, &target)
    }

    /// Advance one mode and yield an [`Advance`] (target, next mode, origin).
    /// Owns `mode` (moved in), so `self.model` is free for FK/IK here.
    async fn advance(
        &mut self,
        mode: Mode,
        measured_q: JointVec,
        command: &Option<Upstream>,
        now: Instant,
    ) -> Advance {
        let takes_the_command = match &mode {
            Mode::Follow => false,
            Mode::JointMove(_) | Mode::CartesianMove(_) => true,
            Mode::PathShare(share) => share.runs(),
        };
        if takes_the_command {
            self.hold = None;
        }
        match mode {
            Mode::Follow => self.advance_follow(command),
            Mode::JointMove(m) => self.advance_joint(m, measured_q, now).await,
            Mode::CartesianMove(m) => self.advance_cartesian(m, measured_q, now).await,
            Mode::PathShare(m) => self.advance_path_share(m, measured_q, now),
        }
    }

    /// One Follow tick: with no leader streaming, a step of the hold under
    /// way ([`Planner::hold_step`]); else the leader's stream, which ends the
    /// hold, or the held setpoint.
    fn advance_follow(&mut self, command: &Option<Upstream>) -> Advance {
        if command.is_none()
            && let Some(step) = self.hold_step()
        {
            self.stream_smoothing.reset();
            return step;
        }
        self.hold = None;
        Advance {
            target: self.follow_target(command),
            next_mode: Mode::Follow,
            origin: Origin::Streamed,
        }
    }

    /// One joint-trajectory tick: sample the trajectory, and end the goal on
    /// cancel, or as the reply judges the measured arm ([`verdict`]): against
    /// the trajectory's end from the planned end on, and against the goal's
    /// contact gap on every tick. Any terminal drops `m` (and with it the
    /// slot release), releasing the move slot.
    async fn advance_joint(&mut self, m: JointMove, measured_q: JointVec, now: Instant) -> Advance {
        let JointMove {
            traj,
            reply,
            _move_slot,
        } = m;
        let q_des = traj.sample(now);
        let outcome = match reply.cancel_reason() {
            Some(reason) => Some(Outcome::Cancelled(reason)),
            None => {
                let contact =
                    self.stands_on_contact(reply.contact_gap(), &self.setpoint, &measured_q);
                reply.settle_outcome(&measured_q, &q_des, contact, traj.clock(), now)
            }
        };
        let Some(outcome) = outcome else {
            return Advance {
                target: q_des,
                next_mode: Mode::JointMove(JointMove {
                    traj,
                    reply,
                    _move_slot,
                }),
                origin: Origin::PlannedMove,
            };
        };
        let elapsed = traj.clock().elapsed_secs(now);
        let end = self.end_move(&outcome, q_des, Some(measured_q));
        reply.finish(self, outcome, Ok(measured_q), elapsed).await;
        end
    }

    /// One Cartesian tick: sample the trajectory, and end the goal on cancel, or
    /// as [`verdict`] judges the measured grasp point: against the commanded
    /// pose from the planned end on, and against the goal's contact gap on
    /// every tick. Any terminal drops `m` (and with it the slot release),
    /// releasing the move slot.
    async fn advance_cartesian(
        &mut self,
        m: CartesianMove,
        measured_q: JointVec,
        now: Instant,
    ) -> Advance {
        let q_des = m.traj.sample(now);
        let outcome = match m.ctx.cancel_cause() {
            Some(cause) => Some(Outcome::Cancelled(cause.reason().to_string())),
            None => {
                let shortfall = || {
                    let measured = self.model.world_ee_pose(&measured_q);
                    pose_shortfall(&measured, &m.target, &m.settle.tolerance)
                };
                verdict(
                    shortfall,
                    self.stands_on_contact(m.contact, &self.setpoint, &measured_q),
                    planned_end(m.traj.clock()),
                    now,
                    m.settle.timeout,
                )
                .map(Outcome::from)
            }
        };
        let Some(outcome) = outcome else {
            return Advance {
                target: q_des,
                next_mode: Mode::CartesianMove(m),
                origin: Origin::PlannedMove,
            };
        };
        let elapsed = m.traj.clock().elapsed_secs(now);
        let end = self.end_move(&outcome, q_des, Some(measured_q));
        self.finish_cartesian(&m.ctx, outcome, Ok(measured_q), elapsed)
            .await;
        end
    }

    /// The rates a streamed pose steps at: the stream's cap, and the turn that
    /// carries the wrist around the grasp point at that cap.
    fn stream_step_rates(&self) -> EeCaps {
        let linear_m_s = self.cfg.stream_ee_velocity_m_s;
        EeCaps {
            linear_m_s,
            angular_rad_s: linear_m_s / self.model.tool().translation.vector.norm(),
        }
    }

    /// The pose a line of `request` walks to: its position, at its orientation or
    /// the turn the hand holds now.
    fn line_goal(&self, request: &ArmMoveRequest) -> Isometry3<f64> {
        Isometry3::from_parts(
            request.position,
            request
                .orientation
                .unwrap_or_else(|| self.world_ee_pose(&self.setpoint).rotation),
        )
    }

    /// The trajectory through `walk`, resting at no sample before the end, over the
    /// longer of `requested` and the time the joint limits allow.
    fn line_trajectory(
        &self,
        walk: &[JointVec],
        requested: Duration,
        now: Instant,
    ) -> JointTrajectory<ARM_DOF> {
        JointTrajectory::through(
            &distinct_joint_waypoints(walk),
            &self.cfg.max_joint_velocity_rad_s,
            &self.cfg.max_joint_acceleration_rad_s2,
            requested,
            now,
        )
    }

    /// Start an accepted goal in the mode that executes it.
    async fn start_goal(&mut self, goal: Goal, move_slot: SlotRelease, now: Instant) -> Mode {
        self.stream_smoothing.reset();
        match goal {
            Goal::Joint {
                target,
                duration,
                reply,
            } => {
                let traj = JointTrajectory::through(
                    &[self.setpoint, target],
                    &self.cfg.max_joint_velocity_rad_s,
                    &self.cfg.max_joint_acceleration_rad_s2,
                    duration,
                    now,
                );
                let (what, remedy) = match reply {
                    JointReply::MoveArmJoints { .. } => ("move", SPLIT_INTO_SHORTER_MOVES),
                    JointReply::Posture(_) => ("posture move", RAISE_THE_JOINT_LIMITS),
                };
                if let Err(failure) = path::check_move_time(&traj, what, remedy) {
                    reply.fail(self.arm, failure).await;
                    return Mode::Follow;
                }
                info!("{}: joint move start", LIMBS.arm_name(self.arm));
                Mode::JointMove(JointMove {
                    traj,
                    reply,
                    _move_slot: move_slot,
                })
            }
            Goal::Cartesian {
                request,
                settle,
                contact,
                ctx,
            } => {
                self.start_cartesian(request, settle, contact, ctx, move_slot, now)
                    .await
            }
        }
    }

    /// Plan an accepted move_arm goal and start executing it, or fail it here:
    /// with `unreachable` when no line reaches the pose, and with `rejected` when
    /// the joint limits cannot run the line within the longest a move may take.
    async fn start_cartesian(
        &mut self,
        request: ArmMoveRequest,
        settle: Settle<PoseTolerance>,
        contact: ContactGap,
        ctx: Box<move_arm::GoalContext>,
        move_slot: SlotRelease,
        now: Instant,
    ) -> Mode {
        let planning = Instant::now();
        let target = self.line_goal(&request);
        let Some(walk) = plan_line(&self.model, &target, self.setpoint) else {
            let failure = Failure::new(FailureKind::Unreachable, NOT_PLANNED);
            self.fail_cartesian(&ctx, failure).await;
            // `move_slot` drops here: the slot is released even on this early
            // exit.
            return Mode::Follow;
        };
        // The clock starts once the plan is ready: the time planning took is no part
        // of the motion.
        let traj = self.line_trajectory(&walk, request.duration, now + planning.elapsed());
        if let Err(failure) = path::check_move_time(&traj, "move", SPLIT_INTO_SHORTER_MOVES) {
            self.fail_cartesian(&ctx, failure).await;
            return Mode::Follow;
        }
        info!(
            "{}: move_arm start, duration={:.3}s",
            LIMBS.arm_name(self.arm),
            traj.clock().duration().as_secs_f64(),
        );
        Mode::CartesianMove(CartesianMove {
            traj,
            ctx,
            target,
            settle,
            contact,
            _move_slot: move_slot,
        })
    }

    /// End a Cartesian goal on `outcome`: it completes, saying whether on
    /// contact, or ends cancelled with the world pose of the joints
    /// [`Planner::reported_joints`] gives for `measured`, and fails with no
    /// result.
    async fn finish_cartesian(
        &self,
        ctx: &move_arm::GoalContext,
        outcome: Outcome,
        measured: Measurement,
        elapsed: f64,
    ) {
        let reported_pose =
            || world_pose_arrays(&self.world_ee_pose(&self.reported_joints(&measured)));
        let ended = match outcome {
            Outcome::Complete | Outcome::Contact { .. } => {
                let (pos, quat) = reported_pose();
                ctx.complete(pos, quat, outcome.ended_on_contact(), elapsed)
                    .await
            }
            Outcome::Cancelled(reason) => {
                let (pos, quat) = reported_pose();
                let reason = cancelled_reason(reason, &measured);
                ctx.complete_cancelled(reason, pos, quat, false, elapsed)
                    .await
            }
            Outcome::Failed(failure) => return self.fail_cartesian(ctx, failure).await,
        };
        if let Err(e) = ended {
            error!("{}: move_arm end: {e}", LIMBS.arm_name(self.arm));
        }
    }

    /// Fail a Cartesian goal with `failure`, with no result.
    async fn fail_cartesian(&self, ctx: &move_arm::GoalContext, failure: Failure) {
        if let Err(e) = ctx.fail(failure.kind.into(), failure.message).await {
            error!("{}: move_arm end: {e}", LIMBS.arm_name(self.arm));
        }
    }
}

/// When the motion `clock` plays ends.
fn planned_end(clock: &MotionClock) -> Instant {
    clock.motion_start() + clock.duration()
}

/// One step of a command at `from` onto `onto`: each joint steps at most its
/// own `max_velocity * dt`, and a joint that near stands on `onto` itself, so
/// the last step lands on it to the bit.
fn step_onto(from: &JointVec, onto: &JointVec, max_velocity: &JointVec, dt: f64) -> JointVec {
    let stepped = chase_step(from, onto, max_velocity, dt);
    std::array::from_fn(|j| {
        let near = (onto[j] - from[j]).abs() <= max_velocity[j] * dt;
        if near { onto[j] } else { stepped[j] }
    })
}

impl Outcome {
    /// What the result of an arm move says of `ended_on_contact`: true for
    /// the arm whose own contact ended the move.
    pub fn ended_on_contact(&self) -> bool {
        matches!(self, Self::Contact { this_arm: true })
    }

    /// Whether the move ended short of the end of its plan: on contact,
    /// cancelled or failed. Such an arm move leaves its arm where it stands
    /// ([`Planner::hold_where_it_stands`]).
    fn ended_short(&self) -> bool {
        !matches!(self, Self::Complete)
    }
}

impl From<Verdict> for Outcome {
    fn from(verdict: Verdict) -> Self {
        match verdict {
            Verdict::Settled => Self::Complete,
            Verdict::Contact => Self::Contact { this_arm: true },
            Verdict::TimedOut(message) => Self::Failed(Failure::new(FailureKind::Timeout, message)),
        }
    }
}

/// The reason a cancelled move_arm or move_arm_joints goal ends with:
/// `reason`, and [`COMMANDED_JOINTS_NOTE`] after it when the arm has no fresh
/// measurement, so its result reports the joints commanded last.
pub(crate) fn cancelled_reason(reason: String, measured: &Measurement) -> String {
    match measured {
        Ok(_) => reason,
        Err(_) => format!("{reason}; {COMMANDED_JOINTS_NOTE}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use robot_description::{PerArm, states};
    use srs_model::ELBOW;
    use srs_model::chain_kinematics::{PRECISION_M, PRECISION_RAD, pose_miss};
    use srs_model::nalgebra::{UnitQuaternion, Vector3};

    use crate::robot::ARMS;
    use crate::test_support::{
        LEFT, POSE_TEST_Q, RIGHT, V1, V2, bundled_arm, claimed_move_slot, plan_config,
        planner_holding, postures,
    };

    /// The slack the stream test reads its arrival against.
    const ARRIVED: PoseTolerance = PoseTolerance {
        position_m: 1e-3,
        orientation_rad: 1e-2,
    };

    /// A `move_arm` request to `target`, as fast as the limits allow; a planner reads
    /// no arm off it.
    fn request_to(target: Isometry3<f64>) -> ArmMoveRequest {
        ArmMoveRequest {
            arm: LEFT,
            position: target.translation,
            orientation: Some(target.rotation),
            duration: Duration::ZERO,
        }
    }

    fn joint_cmd(positions: JointVec) -> Upstream {
        Upstream::Joints(positions)
    }

    /// A planner over the bundled V1 left arm, with permissive test limits.
    fn test_planner(held: JointVec) -> Planner {
        let model = bundled_arm(V1, LEFT);
        let mut planner = Planner::new(LEFT, model, plan_config());
        planner.commit(held);
        planner
    }

    /// A planner with the arm's own limits, seeded inside them: rate_step
    /// clamps into model limits, so an off-limit seed would make the tests
    /// measure the seed correction instead of the motion.
    fn pose_planner() -> Planner {
        planner_holding(LEFT, POSE_TEST_Q)
    }

    /// The failure of a follower gone stale fails the move with its kind,
    /// labelled with the arm, and frees the slot.
    #[tokio::test]
    async fn failing_the_active_move_fails_it_and_frees_the_move_slot() {
        let mut planner = test_planner([0.0; ARM_DOF]);
        let (done_tx, mut done_rx) = mpsc::channel(1);
        let (goal_tx, mut goals) = mpsc::channel(1);
        let move_slot = claimed_move_slot();
        goal_tx
            .send(Goal::posture_share([0.5; ARM_DOF], done_tx))
            .await
            .expect("queue the goal");
        let now = Instant::now();
        planner
            .tick([0.0; ARM_DOF], None, &mut goals, &move_slot, now)
            .await;
        assert!(move_slot.is_claimed(), "move in flight holds the slot");

        let stale = Failure::new(
            FailureKind::ResourceUnavailable,
            "the follower stopped reporting",
        );
        planner.fail_active(stale.clone()).await;
        let outcome = done_rx.recv().await.expect("the failed move must report");
        assert_eq!(
            outcome,
            Outcome::Failed(stale.clone().labelled(LIMBS.arm_name(LEFT)))
        );
        assert!(!move_slot.is_claimed(), "the failure releases the slot");
        // With nothing active the failure is a no-op.
        planner.fail_active(stale).await;
    }

    /// A goal that a stop came before, after its admission, does not start:
    /// it ends cancelled for the stop's reason, the arm holds its setpoint,
    /// and the goal frees the slot.
    #[tokio::test]
    async fn a_stop_since_its_admission_ends_a_goal_before_it_starts() {
        let held = [0.0; ARM_DOF];
        let mut planner = test_planner(held);
        let (done_tx, mut done_rx) = mpsc::channel(1);
        let (goal_tx, mut goals) = mpsc::channel(1);
        let move_slot = claimed_move_slot();
        move_slot.stop_unstarted("stopped: operator");
        goal_tx
            .send(Goal::posture_share([0.5; ARM_DOF], done_tx))
            .await
            .expect("queue the goal");
        let now = Instant::now();
        let tick = planner.tick(held, None, &mut goals, &move_slot, now).await;
        let outcome = done_rx.recv().await.expect("the stopped goal reports");
        assert_eq!(outcome, Outcome::Cancelled("stopped: operator".to_string()));
        assert_eq!(tick.candidate, held, "no move starts");
        assert!(!move_slot.is_claimed(), "the stopped goal frees the slot");
    }

    /// A stop ends the move cancelled for its reason. The arm holds where it
    /// stands: its command moves to the joints measured when the stop came.
    #[tokio::test]
    async fn stopping_the_active_move_ends_it_cancelled_for_the_reason_and_holds() {
        let held = [0.0; ARM_DOF];
        let mut planner = test_planner(held);
        let (done_tx, mut done_rx) = mpsc::channel(1);
        let (goal_tx, mut goals) = mpsc::channel(1);
        let move_slot = claimed_move_slot();
        goal_tx
            .send(Goal::posture_share([0.5; ARM_DOF], done_tx))
            .await
            .expect("queue the goal");
        let start = Instant::now();
        planner
            .tick(held, None, &mut goals, &move_slot, start)
            .await;
        assert!(move_slot.is_claimed(), "move in flight holds the slot");

        let measured_at_stop = [0.1; ARM_DOF];
        let now = start + Duration::from_millis(300);
        let stopped = planner
            .stop_active("stopped: operator", Ok(measured_at_stop), now)
            .await;
        assert!(stopped, "a move was in flight");
        let outcome = done_rx.recv().await.expect("the stopped move reports");
        assert_eq!(outcome, Outcome::Cancelled("stopped: operator".to_string()));
        assert!(!move_slot.is_claimed(), "the stop releases the slot");
        command_comes_to(&mut planner, measured_at_stop, &move_slot, now).await;
        // With nothing active a stop ends nothing and moves no command.
        assert!(
            !planner
                .stop_active("stopped: again", Ok([0.3; ARM_DOF]), now)
                .await
        );
        let later = now + Duration::from_secs(1);
        let idle = tick_measuring(&mut planner, [0.3; ARM_DOF], &move_slot, later).await;
        assert_eq!(idle, measured_at_stop);

        // The stop does not latch: a goal that arrives after it runs.
        let (done_tx, mut done_rx) = mpsc::channel(1);
        assert!(
            move_slot.claim(),
            "the next goal's admission claims the slot"
        );
        goal_tx
            .send(Goal::posture_share([0.2; ARM_DOF], done_tx))
            .await
            .expect("queue the goal");
        planner.tick(held, None, &mut goals, &move_slot, now).await;
        assert!(move_slot.is_claimed(), "the later goal runs");
        assert!(
            done_rx.try_recv().is_err(),
            "the later goal is in flight, not ended"
        );
    }

    /// A joint move its joint limits cannot run within the move ceiling fails
    /// with `rejected` before the arm moves, and frees its slot.
    #[tokio::test]
    async fn a_posture_move_past_the_ceiling_fails_before_the_arm_moves_and_asks_for_higher_limits()
    {
        let held = [0.0; ARM_DOF];
        let model = bundled_arm(V1, LEFT);
        let mut planner = Planner::new(
            LEFT,
            model,
            PlanConfig {
                // 0.5 rad at 0.001 rad/s takes 500 s.
                max_joint_velocity_rad_s: [0.001; ARM_DOF],
                ..plan_config()
            },
        );
        planner.commit(held);
        let (goal_tx, mut goals) = mpsc::channel(1);
        let move_slot = claimed_move_slot();
        let (done_tx, mut done_rx) = mpsc::channel(1);
        goal_tx
            .send(Goal::posture_share([0.5; ARM_DOF], done_tx))
            .await
            .expect("queue the goal");
        planner
            .tick(held, None, &mut goals, &move_slot, Instant::now())
            .await;
        let Outcome::Failed(failure) = done_rx.try_recv().expect("the move ends at once") else {
            panic!("the move fails");
        };
        assert_eq!(failure.kind, FailureKind::Rejected);
        assert!(
            failure.message.contains("this posture move, past the 60 s")
                && failure.message.contains(RAISE_THE_JOINT_LIMITS),
            "{}",
            failure.message
        );
        assert_eq!(planner.setpoint(), held, "the arm never left its pose");
        assert!(!move_slot.is_claimed(), "the failure releases the slot");
    }

    /// A posture share whose arm never follows: the governor holds every
    /// step, or the arm stands against an obstacle the backbone cannot see.
    /// It stays in flight until [`POSTURE_SETTLE`] times out past its planned
    /// end, then fails with `timeout`, naming the joint and how far it
    /// stands.
    #[tokio::test]
    async fn a_posture_share_whose_arm_never_follows_fails_with_timeout() {
        let held = [0.0; ARM_DOF];
        let target = [0.5; ARM_DOF];
        let mut planner = test_planner(held);
        let (goal_tx, mut goals) = mpsc::channel(1);
        let move_slot = claimed_move_slot();

        // The governor holds every step: no candidate is committed.
        let (done_tx, mut done_rx) = mpsc::channel(1);
        goal_tx
            .send(Goal::posture_share(target, done_tx))
            .await
            .expect("queue the goal");
        let now = Instant::now();
        planner.tick(held, None, &mut goals, &move_slot, now).await;
        assert!(move_slot.is_claimed(), "the move is in flight");
        let Mode::JointMove(JointMove { traj, .. }) = &planner.mode else {
            panic!("the goal starts a joint move");
        };
        let planned_end = planned_end(traj.clock());
        let settling = planned_end + POSTURE_SETTLE.timeout / 2;
        planner
            .tick(held, None, &mut goals, &move_slot, settling)
            .await;
        assert!(
            done_rx.try_recv().is_err(),
            "the arm may still settle within the posture's timeout"
        );
        planner
            .tick(
                held,
                None,
                &mut goals,
                &move_slot,
                planned_end + POSTURE_SETTLE.timeout,
            )
            .await;
        let failure = match done_rx.try_recv().expect("the settle timed out") {
            Outcome::Failed(failure) => failure,
            other => panic!("an arm that never moved cannot settle: {other:?}"),
        };
        assert_eq!(failure.kind, FailureKind::Timeout);
        assert_eq!(
            failure.message,
            "left_arm: still 0.500 rad from the target at joint 1, 2.0 s after the planned end"
        );
        assert_eq!(planner.setpoint(), held, "the arm never left its pose");
        assert!(!move_slot.is_claimed(), "the end releases the slot");

        // The governor lets every step through, but the arm stands still.
        let (done_tx, mut done_rx) = mpsc::channel(1);
        assert!(
            move_slot.claim(),
            "the next goal's admission claims the slot"
        );
        goal_tx
            .send(Goal::posture_share(target, done_tx))
            .await
            .expect("queue the goal");
        let now = Instant::now();
        let tick = planner.tick(held, None, &mut goals, &move_slot, now).await;
        planner.commit(tick.candidate);
        let Mode::JointMove(JointMove { traj, .. }) = &planner.mode else {
            panic!("the goal starts a joint move");
        };
        let motion_start = traj.clock().motion_start();
        // Half of the one-second move, on the move's own clock.
        for step in 1..=5 {
            let tick = planner
                .tick(
                    held,
                    None,
                    &mut goals,
                    &move_slot,
                    motion_start + Duration::from_millis(100 * step),
                )
                .await;
            planner.commit(tick.candidate);
        }
        assert!(done_rx.try_recv().is_err(), "the move is in flight");
        assert_ne!(planner.setpoint(), held, "the commanded arm moves");
        planner
            .tick(
                held,
                None,
                &mut goals,
                &move_slot,
                now + Duration::from_secs(3600),
            )
            .await;
        let outcome = done_rx.try_recv().expect("the settle timed out");
        assert!(
            matches!(&outcome, Outcome::Failed(failure) if failure.kind == FailureKind::Timeout),
            "an arm that stands still cannot settle: {outcome:?}"
        );
        assert!(!move_slot.is_claimed(), "the end releases the slot");
    }

    /// A posture share to `target` whose goal `cancelled` cancels, queued on
    /// `planner` and started at `now`: where its terminal reports, and the
    /// start and the planned end of its motion.
    async fn started_share(
        planner: &mut Planner,
        target: JointVec,
        cancelled: &Arc<AtomicBool>,
        move_slot: &MoveSlot,
        now: Instant,
    ) -> (mpsc::Receiver<Outcome>, Instant, Instant) {
        let (done_tx, done_rx) = mpsc::channel(1);
        let (goal_tx, mut goals) = mpsc::channel(1);
        let share = PostureShare {
            done_tx,
            cancelled: cancelled.clone(),
        };
        goal_tx
            .send(Goal::Joint {
                target,
                duration: Duration::from_secs(1),
                reply: JointReply::Posture(share),
            })
            .await
            .expect("queue the goal");
        let held = planner.setpoint();
        let tick = planner.tick(held, None, &mut goals, move_slot, now).await;
        planner.commit(tick.candidate);
        let Mode::JointMove(JointMove { traj, .. }) = &planner.mode else {
            panic!("the goal starts a joint move");
        };
        (
            done_rx,
            traj.clock().motion_start(),
            planned_end(traj.clock()),
        )
    }

    /// One tick of `planner` at `now` with no goal queued, its arm measuring
    /// `measured`, the governor letting the step through: the candidate it
    /// commits.
    async fn tick_measuring(
        planner: &mut Planner,
        measured: JointVec,
        move_slot: &MoveSlot,
        now: Instant,
    ) -> JointVec {
        let (_goal_tx, mut goals) = mpsc::channel(1);
        tick_measuring_with(planner, measured, &mut goals, move_slot, now).await
    }

    /// One tick of `planner` at `now`, taking the goals queued on `goals`,
    /// its arm measuring `measured`, the governor letting the step through:
    /// the candidate it commits.
    async fn tick_measuring_with(
        planner: &mut Planner,
        measured: JointVec,
        goals: &mut mpsc::Receiver<Goal>,
        move_slot: &MoveSlot,
        now: Instant,
    ) -> JointVec {
        let tick = planner.tick(measured, None, goals, move_slot, now).await;
        planner.commit(tick.candidate);
        tick.candidate
    }

    /// Tick `planner` from `at`, a cycle period apart, its arm measuring
    /// `stood` and the governor letting each step through, until the command
    /// stands on `stood`: every command on the way. Each step keeps every
    /// joint within its velocity limit, and the way takes no more ticks than
    /// the farthest joint needs at that limit, and one.
    async fn command_comes_to(
        planner: &mut Planner,
        stood: JointVec,
        move_slot: &MoveSlot,
        at: Instant,
    ) -> Vec<JointVec> {
        let cfg = plan_config();
        let dt = cfg.cycle_period.as_secs_f64();
        let from = planner.setpoint();
        let most = (0..ARM_DOF)
            .map(|j| ((stood[j] - from[j]).abs() / (cfg.max_joint_velocity_rad_s[j] * dt)).ceil())
            .fold(0.0, f64::max) as u32
            + 1;
        let mut way = Vec::new();
        for k in 1..=most {
            if planner.setpoint() == stood {
                break;
            }
            let before = planner.setpoint();
            let commanded =
                tick_measuring(planner, stood, move_slot, at + cfg.cycle_period * k).await;
            for j in 0..ARM_DOF {
                let step = (commanded[j] - before[j]).abs();
                assert!(
                    step <= cfg.max_joint_velocity_rad_s[j] * dt * (1.0 + 1e-9),
                    "joint {} stepped {step} rad",
                    j + 1
                );
            }
            way.push(commanded);
        }
        assert_eq!(
            planner.setpoint(),
            stood,
            "the command stands where the arm stands"
        );
        way
    }

    /// A move that fails its settle holds the arm where it stands. Here the
    /// command plays the whole move while the arm stands short of it, as an
    /// arm that an object blocks does. From the tick of the `timeout` on the
    /// command moves to the joints the arm measures, within the joint
    /// velocity limits, and holds them.
    #[tokio::test]
    async fn a_move_that_times_out_holds_the_arm_where_it_stands() {
        let target = [0.5; ARM_DOF];
        let stood = [0.1; ARM_DOF];
        let mut planner = test_planner([0.0; ARM_DOF]);
        let move_slot = claimed_move_slot();
        let never = Arc::new(AtomicBool::new(false));
        let (mut done_rx, motion_start, planned_end) =
            started_share(&mut planner, target, &never, &move_slot, Instant::now()).await;
        let cycle = plan_config().cycle_period;

        let mut at = motion_start;
        while at < planned_end {
            at += cycle;
            tick_measuring(&mut planner, stood, &move_slot, at).await;
        }
        assert!(done_rx.try_recv().is_err(), "the move settles on");
        assert_eq!(
            planner.setpoint(),
            target,
            "the command stands on the target"
        );

        let timed_out = planned_end + POSTURE_SETTLE.timeout;
        let first = tick_measuring(&mut planner, stood, &move_slot, timed_out).await;
        let outcome = done_rx.try_recv().expect("the settle timed out");
        assert!(
            matches!(&outcome, Outcome::Failed(failure) if failure.kind == FailureKind::Timeout),
            "an arm that stands short cannot settle: {outcome:?}"
        );
        assert!(!move_slot.is_claimed(), "the end releases the slot");
        assert!(
            first[0] < target[0] && first[0] > stood[0],
            "the command sets out for where the arm stands: {first:?}"
        );
        let way = command_comes_to(&mut planner, stood, &move_slot, timed_out).await;
        assert!(
            !way.is_empty(),
            "the way takes more than the tick of the end"
        );
        let later = timed_out + Duration::from_secs(1);
        let held = tick_measuring(&mut planner, [0.3; ARM_DOF], &move_slot, later).await;
        assert_eq!(held, stood, "the arm holds where it stood");
    }

    /// The collision guard keeps its say over the hold: each step of the
    /// command toward where the arm stands goes to the governor, and a
    /// command whose steps the governor holds stays where it is. Once the
    /// hold's steps ran out the planner proposes no step toward the stale
    /// pose, though the governor would let one through then.
    #[tokio::test]
    async fn a_hold_the_governor_holds_leaves_the_command_and_ends() {
        let target = [0.5; ARM_DOF];
        let stood = [0.1; ARM_DOF];
        let mut planner = test_planner([0.0; ARM_DOF]);
        let move_slot = claimed_move_slot();
        let cancelled = Arc::new(AtomicBool::new(false));
        let (_done_rx, motion_start, planned_end) =
            started_share(&mut planner, target, &cancelled, &move_slot, Instant::now()).await;
        let cycle = plan_config().cycle_period;
        let mut at = motion_start;
        while at < planned_end {
            at += cycle;
            tick_measuring(&mut planner, stood, &move_slot, at).await;
        }
        cancelled.store(true, Ordering::Release);
        let held_at = planner.setpoint();

        // The governor holds every step: the planner's proposals go nowhere.
        let (_goal_tx, mut goals) = mpsc::channel(1);
        let mut proposals = 0;
        for k in 1..=50 {
            let tick = planner
                .tick(stood, None, &mut goals, &move_slot, at + cycle * k)
                .await;
            planner.commit(held_at);
            if tick.candidate == held_at {
                break;
            }
            proposals += 1;
        }
        // 0.4 rad at 0.1 rad a tick: four steps, and one.
        assert_eq!(proposals, 5, "the hold proposes its steps and ends");
        let free = tick_measuring(&mut planner, stood, &move_slot, at + cycle * 60).await;
        assert_eq!(free, held_at, "the ended hold moves no command");
    }

    /// A goal that the arm takes while its command is on the way to where
    /// it stood ends that hold: the move starts from the command as it is.
    /// A leader's stream ends it too.
    #[tokio::test]
    async fn a_move_or_a_stream_that_takes_the_arm_ends_its_hold() {
        let stood = [0.03; ARM_DOF];
        let cycle = plan_config().cycle_period;
        let ended_short = async || {
            let mut planner = test_planner([0.0; ARM_DOF]);
            let move_slot = claimed_move_slot();
            let cancelled = Arc::new(AtomicBool::new(false));
            let (_done_rx, motion_start, planned_end) = started_share(
                &mut planner,
                [0.5; ARM_DOF],
                &cancelled,
                &move_slot,
                Instant::now(),
            )
            .await;
            // The command plays the whole move while the arm stands short.
            let mut at = motion_start;
            while at < planned_end {
                at += cycle;
                tick_measuring(&mut planner, stood, &move_slot, at).await;
            }
            cancelled.store(true, Ordering::Release);
            let first = tick_measuring(&mut planner, stood, &move_slot, at + cycle).await;
            assert_ne!(first, stood, "the command is on its way");
            (planner, move_slot, at + cycle)
        };

        let (mut planner, move_slot, at) = ended_short().await;
        let on_the_way = planner.setpoint();
        assert!(
            move_slot.claim(),
            "the next goal's admission claims the slot"
        );
        let never = Arc::new(AtomicBool::new(false));
        let target = [0.5; ARM_DOF];
        let (mut done_rx, _, planned_end) =
            started_share(&mut planner, target, &never, &move_slot, at + cycle).await;
        assert_eq!(
            planner.setpoint(),
            on_the_way,
            "the move starts from the command as it stood"
        );
        assert!(on_the_way[0] > stood[0], "the hold had not come to its end");
        // The arm follows this move to its target, and the move settles. The
        // command then stays on the target: no step goes back to where the
        // arm stood before the move.
        let mut now = at + cycle;
        while now < planned_end {
            now += cycle;
            let follows = planner.setpoint();
            tick_measuring(&mut planner, follows, &move_slot, now).await;
        }
        tick_measuring(&mut planner, target, &move_slot, now + cycle).await;
        assert_eq!(done_rx.try_recv(), Ok(Outcome::Complete));
        let after = tick_measuring(&mut planner, target, &move_slot, now + cycle * 2).await;
        assert_eq!(after, target, "the settled move holds its target");

        let (mut planner, move_slot, at) = ended_short().await;
        let on_the_way = planner.setpoint();
        let (_goal_tx, mut goals) = mpsc::channel(1);
        let streamed = planner
            .tick(
                stood,
                Some(joint_cmd(on_the_way)),
                &mut goals,
                &move_slot,
                at + cycle,
            )
            .await;
        planner.commit(streamed.candidate);
        assert_eq!(streamed.candidate, on_the_way, "the stream has the arm");
        let after = tick_measuring(&mut planner, stood, &move_slot, at + cycle * 2).await;
        assert_eq!(after, on_the_way, "the hold ended with the stream");
    }

    /// A move that its caller cancels holds the arm where it stands: from
    /// the tick of the cancel on the command moves to the joints the arm
    /// measures then, here an arm that stands short of its command. A move
    /// cancelled on the tick it starts holds as any other does.
    #[tokio::test]
    async fn a_cancelled_move_holds_the_arm_where_it_stands() {
        let stood = [0.03; ARM_DOF];
        let mut planner = test_planner([0.0; ARM_DOF]);
        let move_slot = claimed_move_slot();
        let cancelled = Arc::new(AtomicBool::new(false));
        let (mut done_rx, motion_start, planned_end) = started_share(
            &mut planner,
            [0.5; ARM_DOF],
            &cancelled,
            &move_slot,
            Instant::now(),
        )
        .await;
        let halfway = motion_start + (planned_end - motion_start) / 2;
        tick_measuring(&mut planner, stood, &move_slot, halfway).await;
        assert_ne!(
            planner.setpoint(),
            stood,
            "the command runs ahead of the arm"
        );

        cancelled.store(true, Ordering::Release);
        let cycle = plan_config().cycle_period;
        tick_measuring(&mut planner, stood, &move_slot, halfway + cycle).await;
        assert!(
            matches!(done_rx.try_recv(), Ok(Outcome::Cancelled(_))),
            "the move ends cancelled"
        );
        command_comes_to(&mut planner, stood, &move_slot, halfway + cycle).await;

        // Cancelled before its first tick: the move starts and ends on one
        // tick, and the command goes to the measured joints, a step away.
        let held = [0.0; ARM_DOF];
        let mut planner = test_planner(held);
        let move_slot = claimed_move_slot();
        let (done_tx, mut done_rx) = mpsc::channel(1);
        let (goal_tx, mut goals) = mpsc::channel(1);
        let share = PostureShare {
            done_tx,
            cancelled: Arc::new(AtomicBool::new(true)),
        };
        goal_tx
            .send(Goal::Joint {
                target: [0.5; ARM_DOF],
                duration: Duration::from_secs(1),
                reply: JointReply::Posture(share),
            })
            .await
            .expect("queue the goal");
        let start = Instant::now();
        let first = tick_measuring_with(&mut planner, stood, &mut goals, &move_slot, start).await;
        assert!(
            matches!(done_rx.try_recv(), Ok(Outcome::Cancelled(_))),
            "the move ends cancelled on the tick it starts"
        );
        assert_eq!(first, stood, "the arm holds where it stands");
        let after = tick_measuring(&mut planner, stood, &move_slot, start + cycle).await;
        assert_eq!(after, stood);
    }

    /// A goal that takes the arm while its command is on the way to where
    /// it stood, and ends on the tick it starts, leaves no command short of
    /// the arm: the command sets out again, and comes to where the arm
    /// stands.
    #[tokio::test]
    async fn a_goal_that_ends_on_the_tick_it_starts_keeps_the_arm_held_where_it_stands() {
        let stood = [0.03; ARM_DOF];
        let cycle = plan_config().cycle_period;
        let mut planner = test_planner([0.0; ARM_DOF]);
        let move_slot = claimed_move_slot();
        let cancelled = Arc::new(AtomicBool::new(false));
        let (_done_rx, motion_start, planned_end) = started_share(
            &mut planner,
            [0.5; ARM_DOF],
            &cancelled,
            &move_slot,
            Instant::now(),
        )
        .await;
        let mut at = motion_start;
        while at < planned_end {
            at += cycle;
            tick_measuring(&mut planner, stood, &move_slot, at).await;
        }
        cancelled.store(true, Ordering::Release);
        at += cycle;
        tick_measuring(&mut planner, stood, &move_slot, at).await;
        assert_ne!(planner.setpoint(), stood, "the command is on its way");

        assert!(
            move_slot.claim(),
            "the next goal's admission claims the slot"
        );
        let (done_tx, mut done_rx) = mpsc::channel(1);
        let (goal_tx, mut goals) = mpsc::channel(1);
        let share = PostureShare {
            done_tx,
            cancelled: Arc::new(AtomicBool::new(true)),
        };
        goal_tx
            .send(Goal::Joint {
                target: [0.5; ARM_DOF],
                duration: Duration::from_secs(1),
                reply: JointReply::Posture(share),
            })
            .await
            .expect("queue the goal");
        at += cycle;
        tick_measuring_with(&mut planner, stood, &mut goals, &move_slot, at).await;
        assert!(
            matches!(done_rx.try_recv(), Ok(Outcome::Cancelled(_))),
            "the goal ends on the tick it starts"
        );
        command_comes_to(&mut planner, stood, &move_slot, at).await;
    }

    /// A stop that finds no fresh measurement of the arm leaves its command
    /// where it was last governed: the arm cannot be seen.
    #[tokio::test]
    async fn a_stop_without_a_fresh_measurement_holds_the_last_command() {
        let mut planner = test_planner([0.0; ARM_DOF]);
        let move_slot = claimed_move_slot();
        let never = Arc::new(AtomicBool::new(false));
        let now = Instant::now();
        let (_done_rx, motion_start, planned_end) =
            started_share(&mut planner, [0.5; ARM_DOF], &never, &move_slot, now).await;
        let halfway = motion_start + (planned_end - motion_start) / 2;
        let commanded = tick_measuring(&mut planner, [0.1; ARM_DOF], &move_slot, halfway).await;

        let stopped = planner
            .stop_active("stopped: operator", Err(Unmeasured::Stale), halfway)
            .await;
        assert!(stopped, "a move was in flight");
        assert_eq!(planner.setpoint(), commanded);
    }

    /// The arm stands on contact once its measured grasp point stands
    /// farther from the commanded one than the goal's gap allows. Here the
    /// elbow stands 0.05 rad behind its command.
    #[test]
    fn an_arm_stands_on_contact_once_its_grasp_point_stands_past_the_gap() {
        let planner = pose_planner();
        let commanded = POSE_TEST_Q;
        let mut measured = commanded;
        measured[ELBOW] -= 0.05;
        let apart = (planner.world_ee_pose(&commanded).translation.vector
            - planner.world_ee_pose(&measured).translation.vector)
            .norm();
        assert!(apart > 1e-3, "the elbow moves the grasp point: {apart} m");
        let gap = |m: f64| ContactGap::of(PositiveFinite::parse(m).expect("a positive gap"));

        assert!(planner.stands_on_contact(gap(0.9 * apart), &commanded, &measured));
        assert!(!planner.stands_on_contact(gap(1.1 * apart), &commanded, &measured));
        assert!(!planner.stands_on_contact(gap(1e-9), &commanded, &commanded));
        assert!(!planner.stands_on_contact(ContactGap::NONE, &commanded, &measured));
    }

    /// The rest posture puts each grasp point lower and closer to the
    /// column than the working posture does. This is true on both
    /// generations and both arms. On openarm_v2, the grasp points are at
    /// the numbers that the description's joints give.
    #[test]
    fn the_home_grasp_points_sit_lower_and_closer_in_than_ready() {
        let horizontal = |pose: &Isometry3<f64>| pose.translation.x.hypot(pose.translation.y);
        for model_id in [V1, V2] {
            let (ready, home) = (
                postures(model_id, states::READY),
                postures(model_id, states::HOME),
            );
            for arm in PerArm::<(), ARMS>::ids() {
                let model = bundled_arm(model_id, arm);
                let planner = Planner::new(arm, model, plan_config());
                let r = planner.world_ee_pose(&ready[arm]);
                let h = planner.world_ee_pose(&home[arm]);
                let label = format!("{model_id} {}", LIMBS.arm_name(arm));
                assert!(
                    h.translation.z < r.translation.z,
                    "{label}: home z {} is not below ready z {}",
                    h.translation.z,
                    r.translation.z
                );
                assert!(
                    horizontal(&h) < horizontal(&r),
                    "{label}: home is {} m out, ready {} m",
                    horizontal(&h),
                    horizontal(&r)
                );
            }
        }

        // openarm_v2: the right arm's grasp point of each posture, as
        // [x, y, z] in metres and [x, y, z, w].
        let right_ready = (
            [0.3996, -0.2830, 0.5662],
            [-0.2328, 0.6283, -0.2398, 0.7025],
        );
        let right_home = (
            [0.0174, -0.2070, 0.1048],
            [-0.0011, 0.9987, -0.0450, 0.0250],
        );
        // The left arm's grasp point is the mirror of the right arm's across
        // the x-z plane. y changes sign, and so do x and z of the quaternion.
        let mirror =
            |([x, y, z], [qx, qy, qz, qw]): ([f64; 3], [f64; 4])| ([x, -y, z], [-qx, qy, -qz, qw]);
        let at = |pose: &Isometry3<f64>, (position, orientation): ([f64; 3], [f64; 4])| {
            let [x, y, z, w] = orientation;
            let expected =
                UnitQuaternion::from_quaternion(srs_model::nalgebra::Quaternion::new(w, x, y, z));
            let off_m = (pose.translation.vector - Vector3::from(position)).norm();
            let off_rad = pose.rotation.angle_to(&expected);
            off_m < 1e-3 && off_rad < 5e-3
        };
        let (ready, home) = (postures(V2, states::READY), postures(V2, states::HOME));
        for arm in PerArm::<(), ARMS>::ids() {
            let (ready_grasp, home_grasp) = if arm == RIGHT {
                (right_ready, right_home)
            } else {
                (mirror(right_ready), mirror(right_home))
            };
            let model = bundled_arm(V2, arm);
            let planner = Planner::new(arm, model, plan_config());
            let r = planner.world_ee_pose(&ready[arm]);
            let h = planner.world_ee_pose(&home[arm]);
            assert!(
                at(&r, ready_grasp),
                "v2 {} ready grasp point at {:?}",
                LIMBS.arm_name(arm),
                world_pose_arrays(&r)
            );
            assert!(
                at(&h, home_grasp),
                "v2 {} home grasp point at {:?}",
                LIMBS.arm_name(arm),
                world_pose_arrays(&h)
            );
        }
    }

    #[tokio::test]
    async fn a_queued_goal_is_stopped_unstarted_for_the_reason() {
        let planner = test_planner([0.0; ARM_DOF]);
        let (done_tx, mut done_rx) = mpsc::channel(1);
        Goal::posture_share([0.5; ARM_DOF], done_tx)
            .stop("stopped: operator", Ok([0.1; ARM_DOF]), &planner)
            .await;
        let outcome = done_rx.recv().await.expect("the stopped goal reports");
        assert_eq!(outcome, Outcome::Cancelled("stopped: operator".to_string()));
    }

    /// A goal that reaches the planner while a move is in flight on its arm
    /// fails with `precondition_failed`, and the move in flight goes on.
    #[tokio::test]
    async fn a_goal_behind_a_move_in_flight_fails_and_the_move_goes_on() {
        let mut planner = test_planner([0.0; ARM_DOF]);
        let (goal_tx, mut goals) = mpsc::channel(2);
        let (running_tx, mut running_rx) = mpsc::channel(1);
        let (second_tx, mut second_rx) = mpsc::channel(1);
        let move_slot = claimed_move_slot();
        for (target, done_tx) in [([0.5; ARM_DOF], running_tx), ([0.2; ARM_DOF], second_tx)] {
            goal_tx
                .send(Goal::posture_share(target, done_tx))
                .await
                .expect("queue the goal");
        }
        planner
            .tick([0.0; ARM_DOF], None, &mut goals, &move_slot, Instant::now())
            .await;
        let failure = Failure::new(FailureKind::PreconditionFailed, ANOTHER_MOVE_IN_FLIGHT);
        assert_eq!(
            second_rx.try_recv().expect("the second goal ends at once"),
            Outcome::Failed(failure.labelled(LIMBS.arm_name(LEFT)))
        );
        assert!(running_rx.try_recv().is_err(), "the first move runs on");
        assert!(move_slot.is_claimed(), "the first move holds the slot");
    }

    /// An arm move whose settle times out fails with `timeout`, the verdict's
    /// words as its message; a settled one completes, and one that ends on
    /// contact completes on the contact of its own arm.
    #[test]
    fn a_verdict_ends_the_move_complete_on_contact_or_timed_out() {
        assert_eq!(Outcome::from(Verdict::Settled), Outcome::Complete);
        assert_eq!(
            Outcome::from(Verdict::Contact),
            Outcome::Contact { this_arm: true }
        );
        let words = "still 0.200 rad from the target at joint 3, 0.5 s after the planned end";
        assert_eq!(
            Outcome::from(Verdict::TimedOut(words.to_string())),
            Outcome::Failed(Failure::new(FailureKind::Timeout, words))
        );
    }

    /// A move_arm or move_arm_joints result reports the measured joints as
    /// they are. Without a fresh measurement it reports the held setpoint,
    /// which is the joints commanded last, and the reason of a cancelled
    /// goal says so.
    #[test]
    fn an_arm_result_without_a_fresh_measurement_reports_the_commanded_joints() {
        let held = [0.2; ARM_DOF];
        let planner = test_planner(held);
        let measured = [0.1; ARM_DOF];
        assert_eq!(planner.reported_joints(&Ok(measured)), measured);
        assert_eq!(
            cancelled_reason("stopped: operator".to_string(), &Ok(measured)),
            "stopped: operator"
        );
        for unmeasured in [Unmeasured::Stale, Unmeasured::NotYet] {
            assert_eq!(planner.reported_joints(&Err(unmeasured)), held);
            assert_eq!(
                cancelled_reason("stopped: operator".to_string(), &Err(unmeasured)),
                "stopped: operator; the result comes from the joints commanded last"
            );
        }
    }

    #[tokio::test]
    async fn a_plan_check_answers_the_duration_and_moves_nothing() {
        let mut planner = pose_planner();
        let start = planner.world_ee_pose(&POSE_TEST_Q);
        let near = Isometry3::from_parts(
            (start.translation.vector + Vector3::new(0.05, 0.0, 0.0)).into(),
            start.rotation,
        );
        let checked = planner.check_cartesian(&request_to(near));
        let Ok(PlanCheck::Reachable(duration)) = checked else {
            panic!("a pose 5 cm ahead has a plan: {checked:?}");
        };
        assert!(duration > Duration::ZERO, "{duration:?}");
        assert_eq!(planner.setpoint(), POSE_TEST_Q, "the check moves nothing");

        let far = Isometry3::from_parts(
            (start.translation.vector + Vector3::new(2.0, 0.0, 0.0)).into(),
            start.rotation,
        );
        let checked = planner.check_cartesian(&request_to(far));
        let Ok(PlanCheck::Unreachable(reason)) = checked else {
            panic!("a pose 2 m away has no plan: {checked:?}");
        };
        assert!(reason.starts_with("goal pose not planned"), "{reason}");
        for way_on in ["plan_joint_path", "execute_joint_path"] {
            assert!(reason.contains(way_on), "{reason}");
        }
        assert_eq!(planner.setpoint(), POSE_TEST_Q);

        // While a move executes, the check is refused as a goal is.
        let (done_tx, _done_rx) = mpsc::channel(1);
        let (goal_tx, mut goals) = mpsc::channel(1);
        let move_slot = claimed_move_slot();
        goal_tx
            .send(Goal::posture_share(POSE_TEST_Q, done_tx))
            .await
            .expect("queue the goal");
        planner
            .tick(POSE_TEST_Q, None, &mut goals, &move_slot, Instant::now())
            .await;
        assert_eq!(
            planner.check_cartesian(&request_to(near)),
            Err(Failure::new(FailureKind::PreconditionFailed, ARM_BUSY))
        );
    }

    #[test]
    fn seed_from_measured_clamps_a_below_limit_pose_to_the_joint_limits() {
        // Build the real arm: the elbow (j4, index 3) has its lower limit at the
        // straight arm. A power-up pose with the elbow below it must seed at the limit,
        // not off it.
        let model = bundled_arm(V1, LEFT);
        let limits = model.limits();
        let cfg = PlanConfig {
            limits,
            ..plan_config()
        };
        let mut planner = Planner::new(LEFT, model, cfg);

        let mut measured = [0.0; ARM_DOF];
        measured[3] = -0.2; // elbow below its lower limit
        planner.seed_from_measured(measured);

        let seeded = planner.setpoint();
        assert_eq!(seeded[3], limits[3].lo, "elbow seeds at its lower limit");
        assert_eq!(seeded[0], 0.0, "an in-range joint is untouched");
    }

    #[test]
    fn follow_tracks_the_command() {
        let mut planner = test_planner([0.9; ARM_DOF]);
        let target = planner.follow_target(&Some(joint_cmd([0.2; ARM_DOF])));
        assert_eq!(target, [0.2; ARM_DOF]);
    }

    #[test]
    fn follow_holds_when_no_command() {
        let held = [0.3; ARM_DOF];
        let mut planner = test_planner(held);
        assert_eq!(planner.follow_target(&None), held);
    }

    #[test]
    fn a_pose_command_moves_the_end_effector_toward_it() {
        // The Cartesian Follow path: the target is a pose, and the planner is
        // the only thing that knows the chain. One tick must close some of the
        // gap without leaping to it.
        let mut planner = pose_planner();
        let start = planner.world_ee_pose(&POSE_TEST_Q);
        let goal = Isometry3::from_parts(
            (start.translation.vector + Vector3::new(0.05, 0.0, 0.0)).into(),
            start.rotation,
        );

        let stepped = planner.follow_target(&Some(Upstream::Pose(goal)));
        let moved = planner.world_ee_pose(&stepped);
        let before = (goal.translation.vector - start.translation.vector).norm();
        let after = (goal.translation.vector - moved.translation.vector).norm();
        assert!(
            after < before,
            "the step must close the gap: {before} -> {after}"
        );
        let cfg = plan_config();
        let budget = cfg.stream_ee_velocity_m_s * cfg.cycle_period.as_secs_f64();
        assert!(
            after >= before - budget * 1.01,
            "one tick may close at most its {budget} m budget: {before} -> {after}"
        );
    }

    #[test]
    fn a_pose_step_respects_the_end_effector_speed_cap() {
        // The per-tick budget is the linear EE cap * cycle_period. A pose a
        // long way off must produce the same bounded step as a near one, so an
        // operator's flick cannot become a lunge.
        let mut planner = pose_planner();
        let start = planner.world_ee_pose(&POSE_TEST_Q);
        let far = Isometry3::from_parts(
            (start.translation.vector + Vector3::new(5.0, 0.0, 0.0)).into(),
            start.rotation,
        );

        let stepped = planner.follow_target(&Some(Upstream::Pose(far)));
        let moved = planner.world_ee_pose(&stepped);
        let travelled = (moved.translation.vector - start.translation.vector).norm();
        let cfg = plan_config();
        let budget = cfg.stream_ee_velocity_m_s * cfg.cycle_period.as_secs_f64();
        assert!(
            travelled <= budget * 1.01,
            "one tick travelled {travelled} m against a {budget} m budget"
        );
    }

    #[test]
    fn a_pose_step_respects_the_angular_speed_cap() {
        let mut planner = pose_planner();
        let start = planner.world_ee_pose(&POSE_TEST_Q);
        let twisted = Isometry3::from_parts(
            start.translation,
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 1.0) * start.rotation,
        );

        let stepped = planner.follow_target(&Some(Upstream::Pose(twisted)));
        let moved = planner.world_ee_pose(&stepped);
        let cfg = plan_config();
        let budget = planner.stream_step_rates().angular_rad_s * cfg.cycle_period.as_secs_f64();
        let turned = moved.rotation.angle_to(&start.rotation);
        assert!(
            turned <= budget * 1.01,
            "one tick turned {turned} rad against a {budget} rad budget"
        );
    }

    #[test]
    fn a_pose_command_holds_when_the_stream_stops() {
        // Same deadman as the joint path: no message means hold, not drift.
        // Step toward a pose off the held one and commit the governed result,
        // as the coordinator does, so the hold is asserted mid-motion.
        let mut planner = pose_planner();
        let start = planner.world_ee_pose(&POSE_TEST_Q);
        let goal = Isometry3::from_parts(
            (start.translation.vector + Vector3::new(0.05, 0.0, 0.0)).into(),
            start.rotation,
        );
        let stepped = planner.follow_target(&Some(Upstream::Pose(goal)));
        assert_ne!(stepped, POSE_TEST_Q, "the step toward the pose must move");
        planner.commit(stepped);
        assert_eq!(
            planner.follow_target(&None),
            stepped,
            "no message holds the setpoint where the last step left it"
        );
    }

    #[test]
    fn a_consumed_command_holds_the_move_endpoint_until_a_newer_one() {
        // The move -> Follow handoff at the command-watch seam. While a move
        // holds the arm's move slot, the coordinator's
        // `consume_streams_of_busy_arms` clears the arm's command watch
        // (`send_replace(None)`) every tick, so Follow holds at the move's
        // endpoint instead of chasing the pre-move streamed target, until a
        // command that arrives after the clear. This locks the contract Follow
        // relies on.
        let (tx, rx) = tokio::sync::watch::channel(None);
        let endpoint = [0.2; ARM_DOF];
        let mut planner = test_planner(endpoint);

        // Streamed before the move: Follow would chase it.
        tx.send_replace(Some(joint_cmd([0.9; ARM_DOF])));
        assert_eq!(
            planner.follow_target(&rx.borrow()),
            [0.9; ARM_DOF],
            "a live streamed command is chased"
        );

        // The accepted move consumes it: Follow now holds at the move endpoint.
        tx.send_replace(None);
        assert_eq!(
            planner.follow_target(&rx.borrow()),
            endpoint,
            "a consumed command leaves Follow on the move endpoint, not the stale stream"
        );

        // A command that arrives after the move resumes following.
        tx.send_replace(Some(joint_cmd([0.4; ARM_DOF])));
        assert_eq!(
            planner.follow_target(&rx.borrow()),
            [0.4; ARM_DOF],
            "a command after the move resumes following"
        );
    }

    // The joint limits slowed until a line's walk takes past the longest a move may
    // take: the check answers why, in the words the goal fails with.
    #[test]
    fn a_line_the_limits_cannot_run_within_the_longest_move_has_no_plan() {
        let mut planner = pose_planner();
        let start = planner.world_ee_pose(&POSE_TEST_Q);
        let target = Isometry3::from_parts(
            (start.translation.vector + Vector3::new(0.05, 0.0, 0.0)).into(),
            start.rotation,
        );
        planner.cfg.max_joint_velocity_rad_s = [1e-4; ARM_DOF];
        let checked = planner.check_cartesian(&request_to(target));
        let Ok(PlanCheck::Unreachable(reason)) = checked else {
            panic!("the limits cannot run this line within a minute: {checked:?}");
        };
        assert!(reason.contains(SPLIT_INTO_SHORTER_MOVES), "{reason}");
    }

    // A goal that leaves its orientation out is a position alone: a turn no line reaches
    // has a plan once the goal leaves it out, ending at the turn the hand starts with.
    #[test]
    fn a_position_alone_keeps_the_turn_the_hand_starts_with() {
        let held = postures(V2, states::READY)[RIGHT];
        let mut planner = launcher_planner(RIGHT, held);
        let start = planner.world_ee_pose(&held);
        let target = Isometry3::from_parts(
            (start.translation.vector + Vector3::new(0.05, 0.0, 0.0)).into(),
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 3.0) * start.rotation,
        );
        let named = planner.check_cartesian(&request_to(target));
        assert!(matches!(named, Ok(PlanCheck::Unreachable(_))), "{named:?}");
        let alone = ArmMoveRequest {
            orientation: None,
            ..request_to(target)
        };
        let checked = planner.check_cartesian(&alone);
        assert!(
            matches!(checked, Ok(PlanCheck::Reachable(_))),
            "{checked:?}"
        );
        assert_eq!(planner.line_goal(&alone).rotation, start.rotation);
    }

    // A 1 cm lift takes under a second at the launcher's limits.
    #[test]
    fn a_one_centimeter_line_takes_under_a_second() {
        let held = postures(V2, states::READY)[RIGHT];
        let mut planner = launcher_planner(RIGHT, held);
        let mut lift = planner.world_ee_pose(&held);
        lift.translation.vector.z += 0.01;
        let checked = planner.check_cartesian(&request_to(lift));
        let Ok(PlanCheck::Reachable(duration)) = checked else {
            panic!("a 1 cm lift from ready has a plan: {checked:?}");
        };
        assert!(duration < Duration::from_secs(1), "{duration:?}");
    }

    // A line's walk rests at none of its states within a joint path's repeat of the one
    // before it: it takes the time of the states it keeps.
    #[test]
    fn a_line_rests_at_no_near_repeat_of_its_walk() {
        let held = postures(V2, states::READY)[RIGHT];
        let planner = launcher_planner(RIGHT, held);
        let moved = |by: f64| {
            let mut q = held;
            q[0] += by;
            q[1] += by;
            q
        };
        let (middle, near_repeat, end) = (moved(0.2), moved(0.2 + 5e-4), moved(0.4));
        let time_of = |walk: &[JointVec]| {
            planner
                .line_trajectory(walk, Duration::ZERO, Instant::now())
                .clock()
                .duration()
        };
        assert_eq!(
            time_of(&[held, middle, near_repeat, end]),
            time_of(&[held, middle, end])
        );
    }

    // A streamed step's hand speed reads the step from where it starts, the held
    // setpoint, however far the measured arm lags behind it.
    #[tokio::test]
    async fn a_streamed_steps_hand_basis_starts_at_the_setpoint() {
        let mut planner = pose_planner();
        let (_goal_tx, mut goals) = mpsc::channel(1);
        let lagging = POSE_TEST_Q.map(|q| q + 0.3);
        let tick = planner
            .tick(
                lagging,
                None,
                &mut goals,
                &claimed_move_slot(),
                Instant::now(),
            )
            .await;
        let basis = tick
            .streamed_hand
            .expect("a streamed step has a hand basis");
        assert_eq!(basis.start, planner.model.at(&POSE_TEST_Q).jacobian());
    }

    // A line crosses the wrist's singular surface: its walk's wrist bend changes sign
    // and the goal has a plan, each step continuing the one before it.
    #[test]
    fn a_line_crosses_the_wrist_singularity() {
        const WRIST_BEND: usize = 5;
        let start_q: JointVec = [1.139, 1.678, 0.169, 1.339, 0.789, 0.142, 0.120];
        let goal_q: JointVec = [0.827, 1.185, -0.038, 0.856, 1.044, -0.204, 0.530];
        let mut planner = launcher_planner(RIGHT, start_q);
        let target = planner.world_ee_pose(&goal_q);
        let walk = plan_line(&planner.model, &target, start_q).expect("the line tracks");
        assert!(walk[0][WRIST_BEND] * walk[walk.len() - 1][WRIST_BEND] < 0.0);
        let checked = planner.check_cartesian(&request_to(target));
        assert!(
            matches!(checked, Ok(PlanCheck::Reachable(_))),
            "{checked:?}"
        );
    }

    // A goal across the wrist singularity with a configuration and no line: the
    // plan check refuses it, and a stream to the same pose arrives at the launcher's
    // limits, the wrist's bend changing sign on the way and every tick within the
    // velocity limit and the joint limits.
    #[tokio::test]
    async fn a_pose_stream_crosses_the_wrist_where_no_line_tracks() {
        let arm = RIGHT;
        let model = bundled_arm(V2, arm);
        let max_joint_velocity_rad_s = [
            16.754666, 16.754666, 5.445426, 5.445426, 20.943946, 20.943946, 20.943946,
        ];
        let limits = model.limits();
        let cfg = PlanConfig {
            max_joint_velocity_rad_s,
            max_joint_acceleration_rad_s2: [0.5; ARM_DOF],
            stream_ee_velocity_m_s: 0.25,
            limits,
            ..plan_config()
        };
        const WRIST_BEND: usize = 5;
        let start_q: JointVec = [-0.223, 1.600, 0.543, 0.543, 0.768, 0.205, -0.354];
        let goal_q: JointVec = [-0.541, 1.614, 0.364, 0.160, 1.046, -0.227, -0.532];
        let mut planner = Planner::new(arm, model, cfg);
        planner.commit(start_q);
        let target = planner.world_ee_pose(&goal_q);
        let checked = planner.check_cartesian(&request_to(target));
        assert!(
            matches!(checked, Ok(PlanCheck::Unreachable(_))),
            "the goal has no line: {checked:?}"
        );
        let dt = planner.cfg.cycle_period.as_secs_f64();
        let mut q = start_q;
        for _ in 0..1000 {
            let next = planner.follow_target(&Some(Upstream::Pose(target)));
            for i in 0..ARM_DOF {
                assert!(
                    (next[i] - q[i]).abs() <= max_joint_velocity_rad_s[i] * dt * (1.0 + 1e-9),
                    "joint {i} steps from {} to {}",
                    q[i],
                    next[i]
                );
                assert!(limits[i].contains(next[i]), "joint {i} at {}", next[i]);
            }
            planner.commit(next);
            q = next;
        }
        assert_eq!(
            pose_shortfall(&planner.world_ee_pose(&q), &target, &ARRIVED),
            None,
            "the stream ends at {q:?}"
        );
        assert!(
            start_q[WRIST_BEND] * q[WRIST_BEND] < 0.0,
            "the wrist never crossed: it ends at {q:?}"
        );
    }

    // A pose stream that arrives every second tick, a 6 cm/s lift of the hand
    // from ready, is followed as one motion: once the sweep is under way, each
    // joint's velocity changes from one tick to the next by under a tenth of
    // its size over the sweep.
    #[test]
    fn a_streamed_pose_arriving_every_second_tick_is_followed_smoothly() {
        let arm = RIGHT;
        let model = bundled_arm(V2, arm);
        let cfg = PlanConfig {
            max_joint_velocity_rad_s: [
                16.754666, 16.754666, 5.445426, 5.445426, 20.943946, 20.943946, 20.943946,
            ],
            stream_ee_velocity_m_s: 0.5,
            limits: model.limits(),
            ..plan_config()
        };
        let start_q = postures(V2, states::READY)[arm];
        let mut planner = Planner::new(arm, model, cfg);
        planner.commit(start_q);
        let start = planner.world_ee_pose(&start_q);
        let dt = planner.cfg.cycle_period.as_secs_f64();
        let mut target = start;
        let mut qs = vec![start_q];
        for tick in 1..=300usize {
            if tick % 2 == 1 {
                target.translation.vector.z =
                    start.translation.vector.z + 0.06 * (tick - 1) as f64 * dt;
            }
            let q = planner.follow_target(&Some(Upstream::Pose(target)));
            planner.commit(q);
            qs.push(q);
        }
        let velocities: Vec<JointVec> = qs
            .windows(2)
            .map(|w| std::array::from_fn(|i| (w[1][i] - w[0][i]) / dt))
            .collect();
        // The first second is the smoother settling and the law catching the
        // target up; the sweep is steady after it.
        let steady = &velocities[100..];
        for i in 0..ARM_DOF {
            let size = steady.iter().map(|v| v[i].abs()).fold(0.0, f64::max);
            let alternation = steady
                .windows(2)
                .map(|w| (w[1][i] - w[0][i]).abs())
                .fold(0.0, f64::max);
            assert!(
                alternation <= 0.1 * size,
                "joint {i} changes velocity by {alternation:.3} rad/s between ticks on a {size:.3} rad/s sweep"
            );
        }
    }

    /// The launcher's limits on the v2 arm, at the test period.
    const LAUNCHER_VELOCITY_RAD_S: JointVec = [
        16.754666, 16.754666, 5.445426, 5.445426, 20.943946, 20.943946, 20.943946,
    ];
    const LAUNCHER_ACCELERATION_RAD_S2: JointVec = [0.5; ARM_DOF];

    /// A planner of the v2 arm `arm` at the launcher's limits, holding `held`.
    fn launcher_planner(arm: ArmId, held: JointVec) -> Planner {
        let model = bundled_arm(V2, arm);
        let cfg = PlanConfig {
            max_joint_velocity_rad_s: LAUNCHER_VELOCITY_RAD_S,
            max_joint_acceleration_rad_s2: LAUNCHER_ACCELERATION_RAD_S2,
            limits: model.limits(),
            ..plan_config()
        };
        let mut planner = Planner::new(arm, model, cfg);
        planner.commit(held);
        planner
    }

    /// Asserts that `ticks`, one period apart with the arm at rest before and after,
    /// keep every joint within the launcher's limits, to the rounding the differences
    /// magnify.
    fn assert_within_launcher_limits(ticks: &[JointVec], period_s: f64) {
        for (k, w) in ticks.windows(2).enumerate() {
            for i in 0..ARM_DOF {
                let speed = (w[1][i] - w[0][i]).abs() / period_s;
                assert!(
                    speed <= LAUNCHER_VELOCITY_RAD_S[i] * (1.0 + 1e-6),
                    "joint {i} at {speed} rad/s at tick {k}"
                );
            }
        }
        let at_rest: Vec<JointVec> = std::iter::once(ticks[0])
            .chain(ticks.iter().copied())
            .chain(std::iter::once(ticks[ticks.len() - 1]))
            .collect();
        for (k, w) in at_rest.windows(3).enumerate() {
            for i in 0..ARM_DOF {
                let acceleration =
                    (w[2][i] - 2.0 * w[1][i] + w[0][i]).abs() / (period_s * period_s);
                assert!(
                    acceleration <= LAUNCHER_ACCELERATION_RAD_S2[i] * (1.0 + 1e-6),
                    "joint {i} at {acceleration} rad/s^2 at tick {k}"
                );
            }
        }
    }

    /// The trajectory `planner` plays a line's `walk` on, sampled each control period
    /// from `t0` to a period past its end.
    fn line_ticks(planner: &Planner, walk: &[JointVec], t0: Instant) -> Vec<JointVec> {
        let period = planner.cfg.cycle_period;
        let trajectory = planner.line_trajectory(walk, Duration::ZERO, t0);
        let count = trajectory.clock().duration().as_nanos() / period.as_nanos() + 2;
        (0..count as u32)
            .map(|k| trajectory.sample(t0 + period * k))
            .collect()
    }

    // The straight line from Ready out to where the right arm stands straight, its
    // elbow at its lower limit, and the line back in to Ready: chain_kinematics'
    // search walks both, the plan check finds a plan for the line out, the walk
    // lands the hand on the straight arm's pose within the precision a line holds,
    // and each tick of the trajectory the planner plays the walk on moves each joint
    // by no more than its velocity limit takes in a control period. At the straight
    // arm the hand's reach changes with the square of the elbow's bend, so the
    // search lands there with the elbow 0.011 rad off straight (measured, v2).
    #[test]
    fn a_line_through_the_straight_arm_steps_within_the_velocity_limits() {
        let arm = RIGHT;
        let ready = postures(V2, states::READY)[arm];
        let mut planner = launcher_planner(arm, ready);
        let mut straight = ready;
        straight[ELBOW] = planner.cfg.limits[ELBOW].lo;
        let out_to = planner.world_ee_pose(&straight);
        let checked = planner.check_cartesian(&request_to(out_to));
        assert!(
            matches!(checked, Ok(PlanCheck::Reachable(_))),
            "{checked:?}"
        );
        let out =
            plan_line(&planner.model, &out_to, ready).expect("a line out to the straight arm");
        let reached = *out.last().expect("a walk holds its seed");
        let landed = pose_miss(&planner.world_ee_pose(&reached), &out_to);
        assert!(
            landed.position_m <= PRECISION_M && landed.orientation_rad <= PRECISION_RAD,
            "the line lands {landed:?} from the straight arm's pose"
        );
        let back = plan_line(&planner.model, &planner.world_ee_pose(&ready), reached)
            .expect("a line back in to Ready");
        let walk: Vec<JointVec> = out.iter().chain(&back[1..]).copied().collect();
        let ticks = line_ticks(&planner, &walk, Instant::now());
        let period_s = planner.cfg.cycle_period.as_secs_f64();
        let limits = planner.cfg.max_joint_velocity_rad_s;
        for (k, w) in ticks.windows(2).enumerate() {
            let steps = w[1].iter().zip(&w[0]).map(|(to, from)| (to - from).abs());
            for (i, (step, limit)) in steps.zip(limits).enumerate() {
                let bound = limit * period_s;
                assert!(
                    step <= bound * (1.0 + 1e-6),
                    "joint {i} steps {step} rad at tick {k}, past its {bound} rad"
                );
            }
        }
    }

    /// A drawn posture of `planner`'s arm, away from the ends of its joints' ranges.
    fn drawn_posture(tc: &hegel::TestCase, limits: &[Limit; ARM_DOF]) -> JointVec {
        std::array::from_fn(|i| {
            let (lo, hi) = (limits[i].lo, limits[i].hi);
            tc.draw(
                hegel::generators::floats()
                    .min_value(lo + 0.1 * (hi - lo))
                    .max_value(hi - 0.1 * (hi - lo)),
            )
        })
    }

    // Every action on a drawn move, at the launcher's limits and a duration of 0, sampled
    // each control period: a joint move or posture through the planner's ticks, and a pose
    // move and a joint path through the trajectory their ticks sample.
    #[hegel::test(test_cases = 30, seed = Some(14))]
    fn every_action_stays_within_the_joint_limits(tc: hegel::TestCase) {
        let arm = if tc.draw(hegel::generators::booleans()) {
            RIGHT
        } else {
            LEFT
        };
        let limits = bundled_arm(V2, arm).limits();
        let start = drawn_posture(&tc, &limits);
        let target = drawn_posture(&tc, &limits);
        let period = plan_config().cycle_period;
        let period_s = period.as_secs_f64();

        // A joint move's and a posture's ticks.
        let ticks = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime")
            .block_on(async {
                let mut planner = launcher_planner(arm, start);
                let (done_tx, _done_rx) = mpsc::channel(1);
                let (goal_tx, mut goals) = mpsc::channel(1);
                let move_slot = claimed_move_slot();
                let goal = Goal::Joint {
                    target,
                    duration: Duration::ZERO,
                    reply: JointReply::Posture(PostureShare {
                        done_tx,
                        cancelled: Arc::new(AtomicBool::new(false)),
                    }),
                };
                goal_tx.send(goal).await.expect("queue the goal");
                let t0 = Instant::now();
                let mut ticks = vec![start];
                for k in 0..6000u32 {
                    let measured = planner.setpoint();
                    let tick = planner
                        .tick(measured, None, &mut goals, &move_slot, t0 + period * k)
                        .await;
                    planner.commit(tick.candidate);
                    ticks.push(tick.candidate);
                    if k > 0 && matches!(planner.mode, Mode::Follow) {
                        break;
                    }
                }
                ticks
            });
        assert!(
            ticks[ticks.len() - 1]
                .iter()
                .zip(&target)
                .all(|(a, b)| (a - b).abs() < 1e-6),
            "the move ends at its target"
        );
        assert_within_launcher_limits(&ticks, period_s);

        // A joint path through both postures and back.
        let t0 = Instant::now();
        let path = JointTrajectory::through(
            &[start, target, start],
            &LAUNCHER_VELOCITY_RAD_S,
            &LAUNCHER_ACCELERATION_RAD_S2,
            Duration::ZERO,
            t0,
        );
        let ticks_of = |traj: &JointTrajectory<ARM_DOF>| -> Vec<JointVec> {
            let count = traj.clock().duration().as_nanos() / period.as_nanos() + 2;
            (0..count as u32)
                .map(|k| traj.sample(t0 + period * k))
                .collect()
        };
        assert_within_launcher_limits(&ticks_of(&path), period_s);

        // A pose move to where a nearby posture puts the grasp point.
        let planner = launcher_planner(arm, start);
        let nearby: JointVec = std::array::from_fn(|i| start[i] + 0.1 * (target[i] - start[i]));
        let goal = planner.world_ee_pose(&nearby);
        let Some(walk) = plan_line(&planner.model, &goal, start) else {
            tc.event("no line");
            return;
        };
        tc.event("line");
        assert_within_launcher_limits(
            &ticks_of(&planner.line_trajectory(&walk, Duration::ZERO, t0)),
            period_s,
        );
    }
}
