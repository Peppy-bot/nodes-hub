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
//! capped by the governor's EE-speed limiter, fed by the Jacobian this
//! planner hands out); a joint move's target is its trajectory's sample; a
//! Cartesian move's is the IK of the line's pose sample, or the servo path's
//! own sample.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use peppygen::exposed_actions::limb_motion::{move_arm, move_arm_joints};

use control_core::chase::{chase_step, exceeds_rate};
use control_core::path_timing::JointTrajectory;
use control_core::timing::MotionClock;
use srs_model::chain_kinematics::{
    EeCaps, JointSmoothing, ServoPlan, ServoTrajectory, Smoother, clamp_into, rate_step_toward,
};
use srs_model::nalgebra::Isometry3;
use srs_model::{
    Arm, CartesianPlan, Jacobian, Limit, LineCursor, LinePlan, LineTrajectory, PlanLimits,
    PlanRejection, plan_cartesian,
};
use tokio::sync::mpsc;
use tracing::{error, info};

use crate::settle::{Settle, Verdict, joint_shortfall, pose_shortfall, verdict};
use crate::types::ArmMoveRequest;
use crate::types::{ARM_DOF, JointVec, PoseTolerance, Side, world_pose_arrays};
use crate::upstream::Upstream;

/// Slack the runtime per-tick velocity guard of a line allows over the planned
/// limit before aborting the move: the plan sized the line at its own
/// resolution, and the ticks that run it land where the clock puts them.
const VELOCITY_GUARD_MARGIN: f64 = 1.5;

/// The refusal of a goal, or a plan check, that reaches an arm executing a
/// move: the one message for both, whichever gate answers it.
pub const ARM_BUSY: &str = "arm is already executing a motion";

/// Per-arm static configuration for the planner (the motion limits relocated
/// from the arm). Cloned per side.
#[derive(Clone)]
pub struct PlanConfig {
    pub cycle_period: Duration,
    /// The servo's command smoother, built for `cycle_period` at setup.
    pub smoothing: control_core::filters::ButterworthFilter,
    pub max_joint_velocity_rad_s: JointVec,
    pub max_joint_acceleration_rad_s2: JointVec,
    pub ee: EeCaps,
    pub limits: [Limit; ARM_DOF],
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

/// The note a move_arm or move_arm_joints result adds to its message when it
/// reports the joints commanded last, because the follower stopped reporting.
const COMMANDED_JOINTS_NOTE: &str = "the result comes from the joints commanded last";

/// How one arm's share of a posture move ended, reported to the posture
/// action that sent it.
pub struct ReadyOutcome {
    pub success: bool,
    pub message: String,
    /// The move was ended by the stop service, whoever started it.
    pub stopped: bool,
}

/// One arm's share of a whole-robot ready move: where its terminal reports,
/// and the cancel flag the admission task flips on the action's cancel.
pub struct ReadyReply {
    pub done_tx: mpsc::Sender<ReadyOutcome>,
    pub cancelled: Arc<AtomicBool>,
}

/// Where a joint move reports its terminal: the move_arm_joints action goal
/// that carried it (boxed: the goal context dwarfs the other variant) with
/// the settle the goal asked for, or one arm's share of a posture goal
/// (move_to_ready or move_to_home), which ends at the planned end of its
/// trajectory.
pub enum JointReply {
    MoveArmJoints {
        ctx: Box<move_arm_joints::GoalContext>,
        settle: Settle<JointVec>,
    },
    Ready(ReadyReply),
}

impl JointReply {
    fn is_cancelled(&self) -> bool {
        match self {
            Self::MoveArmJoints { ctx, .. } => ctx.is_cancelled(),
            Self::Ready(r) => r.cancelled.load(Ordering::Acquire),
        }
    }

    /// How a joint move whose trajectory has run to its planned end ends at
    /// `now`, with the arm measuring `measured` against `target`: a posture
    /// share completes there; a move_arm_joints goal completes once every
    /// joint stands within its settle tolerance, fails once its settle timeout
    /// has passed, and stays in flight in between.
    fn settle_outcome(
        &self,
        measured: &JointVec,
        target: &JointVec,
        clock: &MotionClock,
        now: Instant,
    ) -> Option<Outcome> {
        match self {
            Self::Ready(_) => Some(Outcome::Complete),
            Self::MoveArmJoints { settle, .. } => verdict(
                joint_shortfall(measured, target, &settle.tolerance),
                planned_end(clock),
                now,
                settle.timeout,
            )
            .map(Outcome::from),
        }
    }

    /// Report the terminal of `planner`'s arm with `measured`, the joints
    /// the arm measured when the move ended. The joint action keeps its
    /// distinct cancelled completion, which a stop ends it with too, and
    /// reports what [`Planner::reported_joints`] gives. The ready share
    /// folds `Cancelled` into a failed outcome and carries a stop as one.
    /// It reports no joints: the posture action measures both arms when its
    /// goal completes.
    async fn finish(
        self,
        planner: &mut Planner,
        outcome: Outcome,
        measured: Measurement,
        elapsed_s: f64,
    ) {
        let side = planner.side;
        let cancelled = outcome.is_cancelled();
        let stopped = matches!(outcome, Outcome::Stopped(_));
        let (success, message) = outcome.report(|| "trajectory complete".to_string());
        match self {
            Self::MoveArmJoints { ctx, .. } => {
                let (reported_q, message) = planner.reported_joints(measured, message);
                let result = if cancelled {
                    ctx.complete_cancelled(success, message, reported_q, elapsed_s)
                        .await
                } else {
                    ctx.complete(success, message, reported_q, elapsed_s).await
                };
                if let Err(e) = result {
                    error!("{}: move_arm_joints complete: {e}", side.label());
                }
            }
            Self::Ready(r) => {
                let outcome = ReadyOutcome {
                    success,
                    message: format!("{}: {message}", side.label()),
                    stopped,
                };
                if r.done_tx.send(outcome).await.is_err() {
                    error!("{}: ready outcome aggregation closed", side.label());
                }
            }
        }
    }
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
        ctx: Box<move_arm::GoalContext>,
    },
}

impl Goal {
    /// A posture share to `target` over one second, reporting on `done_tx`:
    /// the goal of the unit tests of the planner and the coordinator.
    #[cfg(test)]
    pub fn posture_share(target: JointVec, done_tx: mpsc::Sender<ReadyOutcome>) -> Self {
        Goal::Joint {
            target,
            duration: Duration::from_secs(1),
            reply: JointReply::Ready(ReadyReply {
                done_tx,
                cancelled: Arc::new(AtomicBool::new(false)),
            }),
        }
    }

    /// Complete unstarted, `success: false`, reporting `measured` (the
    /// measured joints, per the result contract). The busy flag is the
    /// caller's concern.
    pub async fn refuse(self, reason: &str, measured: Measurement, planner: &mut Planner) {
        self.end_unstarted(Outcome::Failed(reason.to_string()), measured, planner)
            .await;
    }

    /// Complete unstarted as cancelled by the stop service, with the stop's
    /// `message`, reporting `measured`. The busy flag is the caller's
    /// concern.
    pub async fn stop(self, message: &str, measured: Measurement, planner: &mut Planner) {
        self.end_unstarted(Outcome::Stopped(message.to_string()), measured, planner)
            .await;
    }

    async fn end_unstarted(self, outcome: Outcome, measured: Measurement, planner: &mut Planner) {
        match self {
            Goal::Joint { reply, .. } => {
                reply.finish(planner, outcome, measured, 0.0).await;
            }
            Goal::Cartesian { ctx, .. } => {
                let cancelled = outcome.is_cancelled();
                let (success, message) =
                    outcome.report(|| unreachable!("an unstarted goal never completes"));
                let (reported_q, message) = planner.reported_joints(measured, message);
                planner
                    .finish_cartesian(&ctx, reported_q, success, &message, 0.0, cancelled)
                    .await;
            }
        }
    }
}

/// Releases a single-flight busy flag on drop. Held for the lifetime of a move
/// (an arm's mode here, a gripper move in the coordinator), so a move can never
/// end (success, failure, cancel, or an unreachable plan) without freeing the
/// slot the action handler claimed: no terminal path can leak it.
pub(crate) struct BusyGuard(pub(crate) Arc<AtomicBool>);

impl Drop for BusyGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

enum Mode {
    /// Ambient: chase the leading node's stream, or hold when none is streaming.
    Follow,
    /// Tracking a joint trajectory for an accepted move_arm_joints goal.
    JointMove(JointMove),
    /// Tracking a Cartesian plan for an accepted move_arm goal: a line solved
    /// by IK each tick, or the servo's path played.
    CartesianMove(CartesianMove),
}

struct JointMove {
    traj: JointTrajectory<ARM_DOF>,
    reply: JointReply,
    _busy: BusyGuard,
}

struct CartesianMove {
    path: MovePath,
    ctx: Box<move_arm::GoalContext>,
    /// The commanded pose the settle judges the measured grasp point against.
    target: Isometry3<f64>,
    settle: Settle<PoseTolerance>,
    // Last commanded joint target: held on cancel/failure so the arm never snaps.
    prev_q_des: JointVec,
    _busy: BusyGuard,
}

/// A line-tracking move's cursor: the trajectory, the walk along it the plan
/// validated, and when the last tick sampled it.
struct LineTrack {
    traj: LineTrajectory,
    cursor: LineCursor,
    prev_sample_at: Instant,
}

/// How an admitted move_arm goal executes, per its [`CartesianPlan`]: track the
/// straight line (solving IK each tick), or play the guarded servo's path when
/// no continuous joint path tracks the line. A line's track, the trajectory
/// with the IK cursor and the guard's clock, is boxed for its size.
enum MovePath {
    Line(Box<LineTrack>),
    Servo(ServoTrajectory<ARM_DOF>),
}

impl MovePath {
    fn clock(&self) -> &MotionClock {
        match self {
            Self::Line(line) => line.traj.clock(),
            Self::Servo(servo) => servo.clock(),
        }
    }

    /// How the completion reads in the goal result.
    fn completion_message(&self) -> &'static str {
        match self {
            Self::Line(_) => "cartesian move complete",
            Self::Servo(_) => "cartesian move complete (servo-guided)",
        }
    }
}

/// How a move ended. `Cancelled` is the terminal the caller asked for;
/// `Stopped` the one the stop service ended it with, carrying the message
/// that names the stop and its reason. Both are reported through the
/// cancelled completion.
enum Outcome {
    Complete,
    Cancelled,
    Stopped(String),
    Failed(String),
}

impl Outcome {
    /// Whether the goal ends through its cancelled completion.
    fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled | Self::Stopped(_))
    }

    /// What the goal reports: whether it succeeded, and the words, which
    /// `completed` gives for a move that ran to its end.
    fn report(self, completed: impl FnOnce() -> String) -> (bool, String) {
        match self {
            Self::Complete => (true, completed()),
            Self::Cancelled => (false, "goal cancelled".to_string()),
            Self::Stopped(message) | Self::Failed(message) => (false, message),
        }
    }
}

/// One tick of a Cartesian path.
enum PathStep {
    /// The joint target for this tick, and whether the path has finished.
    To { q_des: JointVec, complete: bool },
    /// The path cannot continue; the move completes failed with this reason.
    Failed(String),
}

/// Where this tick's joint target came from. Only a streamed target carries a
/// hand basis out for the governor's EE-speed limiter: a planned move was
/// rolled out against that cap up front and tracks its own schedule, so
/// capping it again would stretch it past the duration its budget was sized
/// from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Origin {
    Streamed,
    PlannedMove,
}

/// One planner tick's product: the velocity-limited candidate setpoint, and
/// the end-effector Jacobian at the measured pose when the target came from
/// the operator's stream (the governor's EE-speed limiter caps only those).
pub struct Tick {
    pub candidate: JointVec,
    pub streamed_hand: Option<Jacobian>,
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
}

pub struct Planner {
    side: Side,
    model: Arm,
    cfg: PlanConfig,
    mode: Mode,
    /// Last governed setpoint: the chase base and the value held when idle.
    setpoint: JointVec,
    /// The smoother a streamed pose's steps pass through, one Butterworth per
    /// joint as a planned servo move's; reset whenever the stream is not a
    /// pose, so a resumed stream seeds on the setpoint.
    stream_smoothing: JointSmoothing<ARM_DOF>,
}

impl Planner {
    /// Start holding zero with no producer locked. The coordinator seeds the real
    /// held pose from the first measured state (via [`Planner::commit`]) before any
    /// setpoint is published, so this initial value is never streamed.
    pub fn new(side: Side, model: Arm, cfg: PlanConfig) -> Self {
        Self {
            side,
            model,
            stream_smoothing: JointSmoothing::new(cfg.smoothing),
            cfg,
            mode: Mode::Follow,
            setpoint: [0.0; ARM_DOF],
        }
    }

    /// Adopt the governed setpoint the coordinator actually published, so the next
    /// tick chases from there (not from the ungoverned candidate).
    pub fn commit(&mut self, governed: JointVec) {
        self.setpoint = governed;
    }

    /// Seed the held setpoint from the arm's first measured pose, clamped into the
    /// joint limits. A power-up pose parked past a soft limit (e.g. the elbow below
    /// its one-sided lower bound, hard against the boundary singularity) would
    /// otherwise anchor the backbone off-limit while the arm clamps every command back to
    /// the limit, leaving the backbone's held setpoint disagreeing with the arm's actual
    /// pose. Clamping the seed keeps the two consistent from the first tick.
    pub fn seed_from_measured(&mut self, measured: JointVec) {
        self.setpoint = clamp_into(&self.cfg.limits, &measured);
    }

    /// The last published setpoint, the coordinator's `prev` for the governor.
    pub fn setpoint(&self) -> JointVec {
        self.setpoint
    }

    /// The joints a move_arm or move_arm_joints result reports, and the
    /// result's `message`:
    /// - the measured joints, with `message` as it is;
    /// - when the follower stopped reporting, the held setpoint, which is
    ///   the joints commanded last, with [`COMMANDED_JOINTS_NOTE`] added;
    /// - before the arm's first measurement, the held setpoint, with
    ///   `message` as it is: only a refusal ends then, and it says why.
    fn reported_joints(&self, measured: Measurement, message: String) -> (JointVec, String) {
        match measured {
            Ok(q) => (q, message),
            Err(Unmeasured::Stale) => {
                (self.setpoint, format!("{message}; {COMMANDED_JOINTS_NOTE}"))
            }
            Err(Unmeasured::NotYet) => (self.setpoint, message),
        }
    }

    /// The end effector's pose at `q` in the world frame, from the planner's
    /// arm model.
    pub fn world_ee_pose(&self, q: &JointVec) -> Isometry3<f64> {
        self.model.world_ee_pose(q)
    }

    /// Retune the end-effector speed cap at runtime (the commander's control).
    /// This copy sizes planned moves at admission, the servo's rollout among
    /// them, and steps streamed poses; a streamed joint command is capped
    /// only by the governor's EE-speed limiter, retuned from the same control
    /// message. The live control rescales the linear cap only: the linear cap
    /// is enforced twice, by this servo and by the governor's EE-speed limiter,
    /// so it binds on any streamed hand, while the angular cap is enforced only
    /// here and a joint stream never reaches this arm of `follow_target`. A
    /// live angular control would read as doing nothing on a joint stream, so
    /// it stays a launch parameter until the governor limits turn rate too.
    pub fn set_max_ee_velocity(&mut self, v: f64) {
        if v.is_finite() && v > 0.0 {
            self.cfg.ee.linear_m_s = v;
        }
    }

    /// Produce this tick's candidate setpoint: admit a pending goal, advance the
    /// active mode to a joint target, then chase it under the velocity limits.
    ///
    /// A streamed tick also carries the arm's end-effector Jacobian at the
    /// measured pose: the governor's EE-speed limiter caps a streamed hand, and
    /// this is the one place the arm's kinematic model lives. A planned move
    /// carries `None`, because it was budgeted against the cap at admission.
    pub async fn tick(
        &mut self,
        measured_q: JointVec,
        command: Option<Upstream>,
        goals: &mut mpsc::Receiver<Goal>,
        busy: &Arc<AtomicBool>,
        now: Instant,
    ) -> Tick {
        let mut mode = std::mem::replace(&mut self.mode, Mode::Follow);
        // Drain fully: every queued goal is answered, never left parked.
        while let Ok(goal) = goals.try_recv() {
            if matches!(mode, Mode::Follow) {
                mode = self.start_goal(goal, busy.clone(), now).await;
            } else {
                goal.refuse("another move is in flight", Ok(measured_q), self)
                    .await;
            }
        }

        let Advance {
            target,
            next_mode,
            origin,
        } = self.advance(mode, measured_q, &command, now).await;
        self.mode = next_mode;

        let dt = self.cfg.cycle_period.as_secs_f64();
        let stepped = chase_step(
            &self.setpoint,
            &target,
            &self.cfg.max_joint_velocity_rad_s,
            dt,
        );
        Tick {
            candidate: clamp_into(&self.cfg.limits, &stepped),
            streamed_hand: match origin {
                Origin::Streamed => Some(self.model.at(&measured_q).jacobian()),
                Origin::PlannedMove => None,
            },
        }
    }

    /// Fail the active move and answer its caller: for a follower gone
    /// stale, where the move can neither progress nor be verified. The result
    /// reports `measured`, which for a stale follower is
    /// [`Unmeasured::Stale`]. Ambient Follow mode is untouched; the busy slot
    /// releases with the mode.
    pub async fn abort_active(&mut self, reason: &str, measured: Measurement, now: Instant) {
        self.end_active(Outcome::Failed(reason.to_string()), measured, now)
            .await;
    }

    /// End the active move for the stop service: its goal completes as
    /// cancelled with `message`, reporting `measured`, the joints measured
    /// when the stop came. The arm holds the setpoint it was last governed
    /// to, as after a cancel, and the busy slot releases with the mode.
    /// Whether a move was in flight.
    pub async fn stop_active(
        &mut self,
        message: &str,
        measured: Measurement,
        now: Instant,
    ) -> bool {
        self.end_active(Outcome::Stopped(message.to_string()), measured, now)
            .await
    }

    /// End the active move with `outcome`, reporting `measured`; ambient
    /// Follow mode is untouched. Whether a move was in flight.
    async fn end_active(&mut self, outcome: Outcome, measured: Measurement, now: Instant) -> bool {
        match std::mem::replace(&mut self.mode, Mode::Follow) {
            Mode::Follow => false,
            Mode::JointMove(JointMove { traj, reply, _busy }) => {
                let elapsed = traj.clock().elapsed_secs(now);
                reply.finish(self, outcome, measured, elapsed).await;
                true
            }
            Mode::CartesianMove(m) => {
                let elapsed = m.path.clock().elapsed_secs(now);
                let cancelled = outcome.is_cancelled();
                let (success, message) = outcome.report(|| m.path.completion_message().to_string());
                let (reported_q, message) = self.reported_joints(measured, message);
                self.finish_cartesian(&m.ctx, reported_q, success, &message, elapsed, cancelled)
                    .await;
                true
            }
        }
    }

    /// Whether a `move_arm` goal to `target` has a plan from the held
    /// setpoint, without moving: the time the move would take, or the words
    /// its refusal gives. Refused while a move executes, with the message a
    /// second goal gets; the check claims no slot and changes no state.
    pub fn check_cartesian(
        &mut self,
        target: &Isometry3<f64>,
        tolerance: PoseTolerance,
        requested: Duration,
    ) -> Result<Duration, String> {
        if !matches!(self.mode, Mode::Follow) {
            return Err(ARM_BUSY.to_string());
        }
        let plan = plan_cartesian(
            &self.model,
            target,
            self.setpoint,
            &self.plan_limits(),
            requested,
            tolerance.servo(),
        )
        .map_err(|rejection| not_planned(tolerance, &rejection))?;
        Ok(plan.duration())
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
                let limits = self.plan_limits().servo_at(
                    STREAMED_POSE_TOLERANCE
                        .servo()
                        .expect("the servo's own slack is reachable"),
                );
                let step = rate_step_toward(self.model.chain(), &q, &ee, pose, &limits);
                // Smoothed as a planned servo step is, then held to the
                // velocity limit again: a smoother can overshoot its input.
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
        match mode {
            Mode::Follow => Advance {
                target: self.follow_target(command),
                next_mode: Mode::Follow,
                origin: Origin::Streamed,
            },
            Mode::JointMove(m) => self.advance_joint(m, measured_q, now).await,
            Mode::CartesianMove(m) => self.advance_cartesian(m, measured_q, now).await,
        }
    }

    /// One joint-trajectory tick: sample the trajectory, and complete the goal
    /// on cancel, or from the planned end on as the reply's settle judges the
    /// measured arm against the trajectory's end. Any terminal drops `m` (and
    /// with it the busy guard), releasing the slot.
    async fn advance_joint(&mut self, m: JointMove, measured_q: JointVec, now: Instant) -> Advance {
        let JointMove { traj, reply, _busy } = m;
        let q_des = traj.sample(now);
        let cancelled = reply.is_cancelled();
        let outcome = if cancelled {
            Some(Outcome::Cancelled)
        } else if traj.clock().is_complete(now) {
            reply.settle_outcome(&measured_q, &q_des, traj.clock(), now)
        } else {
            None
        };
        let Some(outcome) = outcome else {
            return Advance {
                target: q_des,
                next_mode: Mode::JointMove(JointMove { traj, reply, _busy }),
                origin: Origin::PlannedMove,
            };
        };
        let elapsed = traj.clock().elapsed_secs(now);
        reply.finish(self, outcome, Ok(measured_q), elapsed).await;
        // A cancel holds where the arm already is; a settle, met or failed,
        // holds the trajectory's end.
        Advance::ends_move(if cancelled { self.setpoint } else { q_des })
    }

    /// One Cartesian tick: advance the move's path (line-tracking IK, or the
    /// guarded servo) and complete on cancel, on the path giving up, or from
    /// the planned end on as the settle judges the measured grasp point
    /// against the commanded pose; a path at its end keeps giving its last
    /// target. Any terminal drops `m` (and with it the busy guard), releasing
    /// the slot.
    async fn advance_cartesian(
        &mut self,
        mut m: CartesianMove,
        measured_q: JointVec,
        now: Instant,
    ) -> Advance {
        let elapsed = m.path.clock().elapsed_secs(now);
        if m.ctx.is_cancelled() {
            return self
                .end_cartesian(&m, measured_q, Outcome::Cancelled, elapsed)
                .await;
        }
        let stepped = match &mut m.path {
            MovePath::Line(line) => self.step_line(line, &m.prev_q_des, now),
            MovePath::Servo(servo) => Self::step_servo(servo, now),
        };
        let (q_des, complete) = match stepped {
            PathStep::To { q_des, complete } => (q_des, complete),
            PathStep::Failed(reason) => {
                return self
                    .end_cartesian(&m, measured_q, Outcome::Failed(reason), elapsed)
                    .await;
            }
        };
        m.prev_q_des = q_des;
        let outcome = complete
            .then(|| {
                let measured = self.model.world_ee_pose(&measured_q);
                verdict(
                    pose_shortfall(&measured, &m.target, &m.settle.tolerance),
                    planned_end(m.path.clock()),
                    now,
                    m.settle.timeout,
                )
                .map(Outcome::from)
            })
            .flatten();
        let Some(outcome) = outcome else {
            return Advance {
                target: q_des,
                next_mode: Mode::CartesianMove(m),
                origin: Origin::PlannedMove,
            };
        };
        self.end_cartesian(&m, measured_q, outcome, elapsed).await
    }

    /// Track the straight line: walk the line to the blend this tick reached,
    /// the way the plan walked it (a short move can outpace the plan grid, so
    /// the walk subdivides it); the last solution is the tick's setpoint.
    fn step_line(&mut self, line: &mut LineTrack, prev_q_des: &JointVec, now: Instant) -> PathStep {
        let blend = line.traj.blend(now);
        let traj = &line.traj;
        let Some(q_next) = line
            .cursor
            .advance(&self.model, |s| traj.pose_at_blend(s), blend)
        else {
            return PathStep::Failed("IK failed mid-trajectory (unreachable / singular)".into());
        };
        let dt = now
            .duration_since(line.prev_sample_at)
            .as_secs_f64()
            .max(self.cfg.cycle_period.as_secs_f64() * 0.5);
        let guarded = self
            .cfg
            .max_joint_velocity_rad_s
            .map(|v| v * VELOCITY_GUARD_MARGIN);
        if exceeds_rate(prev_q_des, &q_next, &guarded, dt) {
            return PathStep::Failed("joint velocity limit exceeded near singularity".into());
        }
        line.prev_sample_at = now;
        PathStep::To {
            q_des: q_next,
            complete: line.traj.clock().is_complete(now),
        }
    }

    /// The motion budgets a plan is sized against and a streamed pose's step
    /// runs under,
    /// read off one config so acceptance and execution cannot disagree.
    fn plan_limits(&self) -> PlanLimits {
        PlanLimits {
            max_joint_velocity_rad_s: self.cfg.max_joint_velocity_rad_s,
            max_joint_acceleration_rad_s2: self.cfg.max_joint_acceleration_rad_s2,
            ee: self.cfg.ee,
            control_period: self.cfg.cycle_period,
            smoothing: self.cfg.smoothing,
        }
    }

    /// Play the guarded servo's path: the configuration its trajectory reaches
    /// this tick. The plan sized the trajectory within every velocity and
    /// acceleration limit.
    fn step_servo(servo: &ServoTrajectory<ARM_DOF>, now: Instant) -> PathStep {
        PathStep::To {
            q_des: servo.sample(now),
            complete: servo.clock().is_complete(now),
        }
    }

    /// End a Cartesian move: report the terminal and hold the last commanded
    /// target, so the arm never snaps when a move stops early.
    async fn end_cartesian(
        &mut self,
        m: &CartesianMove,
        measured_q: JointVec,
        outcome: Outcome,
        elapsed: f64,
    ) -> Advance {
        let cancelled = outcome.is_cancelled();
        let (success, message) = outcome.report(|| m.path.completion_message().to_string());
        self.finish_cartesian(&m.ctx, measured_q, success, &message, elapsed, cancelled)
            .await;
        Advance::ends_move(m.prev_q_des)
    }

    /// Start an accepted goal in the mode that executes it.
    async fn start_goal(&mut self, goal: Goal, busy: Arc<AtomicBool>, now: Instant) -> Mode {
        self.stream_smoothing.reset();
        let busy = BusyGuard(busy);
        match goal {
            Goal::Joint {
                target,
                duration,
                reply,
            } => {
                info!("{}: joint move start", self.side.label());
                Mode::JointMove(JointMove {
                    traj: JointTrajectory::new(
                        self.setpoint,
                        target,
                        &self.cfg.max_joint_velocity_rad_s,
                        &self.cfg.max_joint_acceleration_rad_s2,
                        duration,
                        now,
                    ),
                    reply,
                    _busy: busy,
                })
            }
            Goal::Cartesian {
                request,
                settle,
                ctx,
            } => self.start_cartesian(request, settle, ctx, busy, now).await,
        }
    }

    /// Plan an accepted move_arm goal and start executing it, or complete it
    /// failed here when no path reaches the pose.
    ///
    /// The start pose is the FK of the held setpoint (the chase base), not the
    /// measured pose, so the first-tick velocity guard compares the IK of the
    /// same configuration the chase continues from and cannot false-trip when
    /// the governor held the arm off its measured pose just before admission.
    async fn start_cartesian(
        &mut self,
        request: ArmMoveRequest,
        settle: Settle<PoseTolerance>,
        ctx: Box<move_arm::GoalContext>,
        busy: BusyGuard,
        now: Instant,
    ) -> Mode {
        let ArmMoveRequest {
            target,
            tolerance,
            duration: requested,
            ..
        } = request;
        let start_world = self.model.world_ee_pose(&self.setpoint);
        let plan = plan_cartesian(
            &self.model,
            &target,
            self.setpoint,
            &self.plan_limits(),
            requested,
            tolerance.servo(),
        );
        let plan = match plan {
            Ok(plan) => plan,
            Err(rejection) => {
                let (pos, quat) = world_pose_arrays(&start_world);
                if let Err(e) = ctx
                    .complete(false, not_planned(tolerance, &rejection), pos, quat, 0.0)
                    .await
                {
                    error!("{}: move_arm complete: {e}", self.side.label());
                }
                // `busy` drops here: the slot is released even on this early exit.
                return Mode::Follow;
            }
        };
        let path = match plan {
            CartesianPlan::Line(LinePlan {
                duration,
                elbow,
                swivel,
                start_q,
                path,
            }) => {
                info!(
                    "{}: move_arm start ({elbow:?} elbow, {swivel:?}), duration={:.3}s",
                    self.side.label(),
                    duration.as_secs_f64(),
                );
                MovePath::Line(Box::new(LineTrack {
                    // Seed from the plan's normalized start, not self.setpoint:
                    // the plan validated the walk from here, so execution must
                    // begin on the same IK branch (they can differ at a
                    // redundancy/limit boundary). The chase still moves the arm
                    // from its actual setpoint, velocity-limited.
                    traj: LineTrajectory::new(start_world, target, path, duration, now),
                    cursor: LineCursor::new(start_q, swivel),
                    prev_sample_at: now,
                }))
            }
            // No continuous joint path tracks the line: play the path the
            // guarded servo's rollout just validated, the same damped law the
            // operator's streaming jog crosses these walls with.
            CartesianPlan::Servo(ServoPlan { duration, path }) => {
                info!(
                    "{}: move_arm start (servo-guided), duration={:.3}s",
                    self.side.label(),
                    duration.as_secs_f64(),
                );
                MovePath::Servo(ServoTrajectory::new(path, duration, now))
            }
        };
        // The line's velocity guard compares each sample against the last
        // commanded target, so the first comparison must sit on the plan's own
        // IK branch (start_q), which can differ from the held setpoint at a
        // redundancy or limit boundary; seeding from the setpoint there would
        // trip the guard on a valid line's first tick. The servo's path starts
        // at the held setpoint, the seed its rollout stepped from.
        let prev_q_des = match &path {
            MovePath::Line(line) => line.cursor.q(),
            MovePath::Servo(_) => self.setpoint,
        };
        Mode::CartesianMove(CartesianMove {
            path,
            ctx,
            target,
            settle,
            prev_q_des,
            _busy: busy,
        })
    }

    /// Complete a Cartesian goal, reporting the world pose of `reported_q`:
    /// the joints measured at exit, or those [`Planner::reported_joints`]
    /// gives without a fresh measurement.
    async fn finish_cartesian(
        &mut self,
        ctx: &move_arm::GoalContext,
        reported_q: JointVec,
        success: bool,
        message: &str,
        elapsed: f64,
        cancelled: bool,
    ) {
        let (pos, quat) = world_pose_arrays(&self.world_ee_pose(&reported_q));
        let result = if cancelled {
            ctx.complete_cancelled(false, message.into(), pos, quat, elapsed)
                .await
        } else {
            ctx.complete(success, message.into(), pos, quat, elapsed)
                .await
        };
        if let Err(e) = result {
            error!("{}: move_arm complete: {e}", self.side.label());
        }
    }
}

/// The slack a streamed pose's servo step runs against: the guarded servo's
/// own arrival bar.
const STREAMED_POSE_TOLERANCE: PoseTolerance = PoseTolerance {
    position_m: control_core::servo::POSITION_TOLERANCE_M,
    orientation_rad: control_core::servo::ORIENTATION_TOLERANCE_RAD,
};

/// When the motion `clock` plays ends.
fn planned_end(clock: &MotionClock) -> Instant {
    clock.motion_start() + clock.duration()
}

impl From<Verdict> for Outcome {
    fn from(verdict: Verdict) -> Self {
        match verdict {
            Verdict::Settled => Self::Complete,
            Verdict::Failed(message) => Self::Failed(message),
        }
    }
}

/// The refusal of a Cartesian move no plan reaches within `tolerance`: what a
/// `move_arm` goal completes with, and what `check_arm_move` answers.
fn not_planned(tolerance: PoseTolerance, rejection: &PlanRejection) -> String {
    format!(
        "goal pose not planned within {:.1} mm / {:.1} deg: {rejection}",
        tolerance.position_m * 1000.0,
        tolerance.orientation_rad.to_degrees()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use srs_model::nalgebra::{UnitQuaternion, Vector3};

    fn test_cfg() -> PlanConfig {
        PlanConfig {
            cycle_period: Duration::from_millis(10),
            smoothing: srs_model::chain_kinematics::command_smoothing(Duration::from_millis(10))
                .unwrap(),
            max_joint_velocity_rad_s: [10.0; ARM_DOF],
            max_joint_acceleration_rad_s2: crate::test_support::A_MAX,
            ee: EeCaps {
                linear_m_s: 1.0,
                angular_rad_s: 0.8,
            },
            limits: [Limit {
                lo: -10.0,
                hi: 10.0,
            }; ARM_DOF],
        }
    }

    fn joint_cmd(positions: JointVec) -> Upstream {
        Upstream::Joints(positions)
    }

    fn left_arm_model() -> Arm {
        let version = openarm_description::HardwareVersion::V1;
        crate::arm_model(version, openarm_description::Side::Left)
            .expect("build left arm from bundled URDF")
    }

    /// A planner over the bundled V1 left arm, with permissive test limits.
    fn test_planner(held: JointVec) -> Planner {
        let mut planner = Planner::new(Side::Left, left_arm_model(), test_cfg());
        planner.commit(held);
        planner
    }

    /// An in-limit pose: elbow off its 0.05 floor, j2 inside [-3.32, 0.17].
    const POSE_TEST_Q: JointVec = [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];

    /// A planner with the arm's own limits, seeded inside them: rate_step
    /// clamps into model limits, so an off-limit seed would make the tests
    /// measure the seed correction instead of the motion.
    fn pose_planner() -> Planner {
        let model = left_arm_model();
        let cfg = PlanConfig {
            limits: model.limits(),
            ..test_cfg()
        };
        let mut planner = Planner::new(Side::Left, model, cfg);
        planner.commit(POSE_TEST_Q);
        planner
    }

    /// The abort of a follower gone stale fails the move and frees the slot.
    #[tokio::test]
    async fn aborting_the_active_move_fails_it_and_frees_the_busy_slot() {
        let mut planner = test_planner([0.0; ARM_DOF]);
        let (done_tx, mut done_rx) = mpsc::channel(1);
        let (goal_tx, mut goals) = mpsc::channel(1);
        let busy = Arc::new(AtomicBool::new(true)); // claimed at accept
        goal_tx
            .send(Goal::posture_share([0.5; ARM_DOF], done_tx))
            .await
            .expect("queue the goal");
        let now = Instant::now();
        planner
            .tick([0.0; ARM_DOF], None, &mut goals, &busy, now)
            .await;
        assert!(
            busy.load(Ordering::Acquire),
            "move in flight holds the slot"
        );

        planner
            .abort_active(
                "the follower stopped reporting",
                Err(Unmeasured::Stale),
                now,
            )
            .await;
        let outcome = done_rx.recv().await.expect("aborted move must report");
        assert!(!outcome.success);
        assert_eq!(outcome.message, "left: the follower stopped reporting");
        assert!(!busy.load(Ordering::Acquire), "abort releases the slot");
        // With nothing active the abort is a no-op.
        planner
            .abort_active("again", Err(Unmeasured::Stale), now)
            .await;
    }

    /// A stop ends the move as cancelled with its message. The arm holds
    /// the setpoint it was last governed to, not the pose measured when the
    /// stop came.
    #[tokio::test]
    async fn stopping_the_active_move_ends_it_cancelled_with_the_message_and_holds() {
        let held = [0.0; ARM_DOF];
        let mut planner = test_planner(held);
        let (done_tx, mut done_rx) = mpsc::channel(1);
        let (goal_tx, mut goals) = mpsc::channel(1);
        let busy = Arc::new(AtomicBool::new(true));
        goal_tx
            .send(Goal::posture_share([0.5; ARM_DOF], done_tx))
            .await
            .expect("queue the goal");
        let now = Instant::now();
        planner.tick(held, None, &mut goals, &busy, now).await;
        assert!(
            busy.load(Ordering::Acquire),
            "move in flight holds the slot"
        );

        let measured_at_stop = [0.1; ARM_DOF];
        let stopped = planner
            .stop_active("stopped: operator", Ok(measured_at_stop), now)
            .await;
        assert!(stopped, "a move was in flight");
        let outcome = done_rx.recv().await.expect("the stopped move reports");
        assert!(!outcome.success);
        assert!(outcome.stopped);
        assert_eq!(outcome.message, "left: stopped: operator");
        assert!(!busy.load(Ordering::Acquire), "the stop releases the slot");
        assert_eq!(
            planner.setpoint(),
            held,
            "the arm holds where it was last governed, not the measured pose"
        );
        // With nothing active a stop ends nothing.
        assert!(
            !planner
                .stop_active("stopped: again", Ok([0.1; ARM_DOF]), now)
                .await
        );

        // The stop does not latch: a goal that arrives after it runs.
        let (done_tx, mut done_rx) = mpsc::channel(1);
        busy.store(true, Ordering::Release);
        goal_tx
            .send(Goal::posture_share([0.2; ARM_DOF], done_tx))
            .await
            .expect("queue the goal");
        planner.tick(held, None, &mut goals, &busy, now).await;
        assert!(busy.load(Ordering::Acquire), "the later goal runs");
        assert!(
            done_rx.try_recv().is_err(),
            "the later goal is in flight, not ended"
        );
    }

    /// A posture share whose arm never follows: the governor holds every
    /// step, or the arm stands against an obstacle the backbone cannot see.
    /// The move ends on its clock with success.
    #[tokio::test]
    async fn a_joint_move_the_governor_holds_still_completes_with_success() {
        let held = [0.0; ARM_DOF];
        let target = [0.5; ARM_DOF];
        let mut planner = test_planner(held);
        let (goal_tx, mut goals) = mpsc::channel(1);
        let busy = Arc::new(AtomicBool::new(true));

        // The governor holds every step: no candidate is committed.
        let (done_tx, mut done_rx) = mpsc::channel(1);
        goal_tx
            .send(Goal::posture_share(target, done_tx))
            .await
            .expect("queue the goal");
        let now = Instant::now();
        planner.tick(held, None, &mut goals, &busy, now).await;
        assert!(busy.load(Ordering::Acquire), "the move is in flight");
        planner
            .tick(
                held,
                None,
                &mut goals,
                &busy,
                now + Duration::from_secs(3600),
            )
            .await;
        let outcome = done_rx.try_recv().expect("the move ends on its clock");
        assert!(outcome.success, "{}", outcome.message);
        assert_eq!(outcome.message, "left: trajectory complete");
        assert!(!outcome.stopped);
        assert_eq!(planner.setpoint(), held, "the arm never left its pose");
        assert!(!busy.load(Ordering::Acquire), "the end releases the slot");

        // The governor lets every step through, but the arm stands still.
        let (done_tx, mut done_rx) = mpsc::channel(1);
        busy.store(true, Ordering::Release);
        goal_tx
            .send(Goal::posture_share(target, done_tx))
            .await
            .expect("queue the goal");
        let now = Instant::now();
        let tick = planner.tick(held, None, &mut goals, &busy, now).await;
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
                    &busy,
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
                &busy,
                now + Duration::from_secs(3600),
            )
            .await;
        let outcome = done_rx.try_recv().expect("the move ends on its clock");
        assert!(outcome.success, "{}", outcome.message);
        assert_eq!(outcome.message, "left: trajectory complete");
        assert!(!busy.load(Ordering::Acquire), "the end releases the slot");
    }

    /// The rest posture puts each grasp point lower and closer to the
    /// column than the working posture does. This is true on both
    /// generations and both sides. On openarm_v2, the grasp points are at
    /// the numbers that the description's joints give.
    #[test]
    fn the_home_grasp_points_sit_lower_and_closer_in_than_ready() {
        use openarm_description::{HardwareVersion, home, ready};
        let horizontal = |pose: &Isometry3<f64>| pose.translation.x.hypot(pose.translation.y);
        for version in [HardwareVersion::V1, HardwareVersion::V2] {
            for side in [Side::Left, Side::Right] {
                let model = crate::arm_model(version, side.model()).expect("build the arm");
                let planner = Planner::new(side, model, test_cfg());
                let r = planner.world_ee_pose(&ready(side.model()));
                let h = planner.world_ee_pose(&home(side.model()));
                let label = format!("{version:?} {}", side.label());
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
            [0.3363, -0.1951, 0.2836],
            [-0.2897, 0.8363, -0.0870, 0.4573],
        );
        let right_home = ([0.0174, -0.1535, 0.1024], [0.0, 0.9997, 0.0, 0.0250]);
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
        for side in [Side::Left, Side::Right] {
            let (ready_grasp, home_grasp) = match side {
                Side::Right => (right_ready, right_home),
                Side::Left => (mirror(right_ready), mirror(right_home)),
            };
            let model = crate::arm_model(HardwareVersion::V2, side.model()).expect("build the arm");
            let planner = Planner::new(side, model, test_cfg());
            let r = planner.world_ee_pose(&ready(side.model()));
            let h = planner.world_ee_pose(&home(side.model()));
            assert!(
                at(&r, ready_grasp),
                "v2 {} ready grasp point at {:?}",
                side.label(),
                world_pose_arrays(&r)
            );
            assert!(
                at(&h, home_grasp),
                "v2 {} home grasp point at {:?}",
                side.label(),
                world_pose_arrays(&h)
            );
        }
    }

    #[tokio::test]
    async fn a_queued_goal_is_stopped_unstarted_with_the_message() {
        let mut planner = test_planner([0.0; ARM_DOF]);
        let (done_tx, mut done_rx) = mpsc::channel(1);
        Goal::posture_share([0.5; ARM_DOF], done_tx)
            .stop("stopped: operator", Ok([0.1; ARM_DOF]), &mut planner)
            .await;
        let outcome = done_rx.recv().await.expect("the stopped goal reports");
        assert!(!outcome.success && outcome.stopped);
        assert_eq!(outcome.message, "left: stopped: operator");
    }

    /// A move_arm or move_arm_joints result reports the measured joints as
    /// they are. When the follower stopped reporting, it reports the held
    /// setpoint, which is the joints commanded last, and its message says
    /// so. Before the first measurement it reports the held setpoint, and
    /// the refusal's message stays as it is.
    #[test]
    fn an_arm_result_without_a_fresh_measurement_reports_the_commanded_joints() {
        let held = [0.2; ARM_DOF];
        let planner = test_planner(held);
        let measured = [0.1; ARM_DOF];
        assert_eq!(
            planner.reported_joints(Ok(measured), "trajectory complete".to_string()),
            (measured, "trajectory complete".to_string())
        );
        assert_eq!(
            planner.reported_joints(
                Err(Unmeasured::Stale),
                "the follower stopped reporting".to_string()
            ),
            (
                held,
                "the follower stopped reporting; the result comes from the joints commanded last"
                    .to_string()
            )
        );
        let refusal = "the follower has not reported its first state yet";
        assert_eq!(
            planner.reported_joints(Err(Unmeasured::NotYet), refusal.to_string()),
            (held, refusal.to_string())
        );
    }

    #[tokio::test]
    async fn a_plan_check_answers_the_duration_and_moves_nothing() {
        let mut planner = pose_planner();
        let start = planner.world_ee_pose(&POSE_TEST_Q);
        let near = Isometry3::from_parts(
            (start.translation.vector + Vector3::new(0.05, 0.0, 0.0)).into(),
            start.rotation,
        );
        let duration = planner
            .check_cartesian(&near, STREAMED_POSE_TOLERANCE, Duration::ZERO)
            .expect("a pose 5 cm ahead has a plan");
        assert!(duration > Duration::ZERO, "{duration:?}");
        assert_eq!(planner.setpoint(), POSE_TEST_Q, "the check moves nothing");

        let far = Isometry3::from_parts(
            (start.translation.vector + Vector3::new(2.0, 0.0, 0.0)).into(),
            start.rotation,
        );
        let refusal = planner
            .check_cartesian(&far, STREAMED_POSE_TOLERANCE, Duration::ZERO)
            .expect_err("a pose 2 m away has no plan");
        assert!(
            refusal.starts_with("goal pose not planned within"),
            "{refusal}"
        );
        assert_eq!(planner.setpoint(), POSE_TEST_Q);

        // While a move executes, the check is refused as a second goal is.
        let (done_tx, _done_rx) = mpsc::channel(1);
        let (goal_tx, mut goals) = mpsc::channel(1);
        let busy = Arc::new(AtomicBool::new(true));
        goal_tx
            .send(Goal::posture_share(POSE_TEST_Q, done_tx))
            .await
            .expect("queue the goal");
        planner
            .tick(POSE_TEST_Q, None, &mut goals, &busy, Instant::now())
            .await;
        assert_eq!(
            planner.check_cartesian(&near, STREAMED_POSE_TOLERANCE, Duration::ZERO),
            Err(ARM_BUSY.to_string())
        );
    }

    #[test]
    fn seed_from_measured_clamps_a_below_limit_pose_to_the_joint_limits() {
        // Build the real arm: the elbow (j4, index 3) carries a one-sided lower bound
        // of ~0.05 (the singularity floor applied by crate::arm_model), hard against the
        // boundary singularity. A power-up pose with the elbow below it must seed at the
        // limit, not off it.
        let version = openarm_description::HardwareVersion::V1;
        let model = crate::arm_model(version, openarm_description::Side::Left)
            .expect("build left arm from bundled URDF");
        let limits = model.limits();
        let cfg = PlanConfig {
            limits,
            ..test_cfg()
        };
        let mut planner = Planner::new(Side::Left, model, cfg);

        let mut measured = [0.0; ARM_DOF];
        measured[3] = -0.2; // elbow below its lower limit
        planner.seed_from_measured(measured);

        let seeded = planner.setpoint();
        assert_eq!(
            seeded[3], limits[3].lo,
            "elbow seeds at its lower limit, off the singularity"
        );
        assert!(
            seeded[3] >= 0.04,
            "vendored URDF elbow lower limit is ~0.05"
        );
        assert_eq!(seeded[0], 0.0, "an in-range joint is untouched");
    }

    #[test]
    fn busy_guard_releases_slot_on_drop() {
        let busy = Arc::new(AtomicBool::new(true));
        {
            let _g = BusyGuard(busy.clone());
            assert!(busy.load(Ordering::Acquire));
        }
        assert!(
            !busy.load(Ordering::Acquire),
            "guard must free the slot on drop, so no move terminal can leak it"
        );
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
        let cfg = test_cfg();
        let budget = cfg.ee.linear_m_s * cfg.cycle_period.as_secs_f64();
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
        let cfg = test_cfg();
        let budget = cfg.ee.linear_m_s * cfg.cycle_period.as_secs_f64();
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
        let cfg = test_cfg();
        let budget = cfg.ee.angular_rad_s * cfg.cycle_period.as_secs_f64();
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
        // The move -> Follow handoff at the command-watch seam. An accepted move
        // clears the side's command watch (what the action handler does with
        // `send_replace(None)`), so Follow holds at the move's endpoint instead
        // of chasing the pre-move streamed target, until a command that arrives
        // after the clear. This locks the contract Follow relies on; that the
        // handler performs the clear is covered by the live regression.
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

    // A pose stream that arrives every second tick, a 6 cm/s lift of the hand
    // from ready, is followed as one motion: once the sweep is under way, each
    // joint's velocity changes from one tick to the next by under a tenth of
    // its size over the sweep.
    #[test]
    fn a_streamed_pose_arriving_every_second_tick_is_followed_smoothly() {
        use openarm_description::ready;
        let side = Side::Right;
        let model = crate::arm_model(openarm_description::HardwareVersion::V2, side.model())
            .expect("build the v2 right arm");
        let cfg = PlanConfig {
            max_joint_velocity_rad_s: [
                16.754666, 16.754666, 5.445426, 5.445426, 20.943946, 20.943946, 20.943946,
            ],
            ee: EeCaps {
                linear_m_s: 0.5,
                angular_rad_s: 0.8,
            },
            limits: model.limits(),
            ..test_cfg()
        };
        let start_q = ready(side.model());
        let mut planner = Planner::new(side, model, cfg);
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
}
