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
//! planner hands out); a joint move's target is the quintic sample; a
//! Cartesian move's target is the IK of the pose sample.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use peppygen::exposed_actions::limb_motion::{move_arm, move_arm_joints};
use srs_model::nalgebra::Isometry3;
use srs_model::{Arm, ArmAnglePolicy, Jacobian, Limit};
use tokio::sync::mpsc;
use tracing::{error, info};

use crate::chase::{chase_step, clamp_to_limits};
use crate::motion::{MOTION_TIMEOUT_FACTOR, MoveBudget};
use crate::servo::{EeCaps, ServoState, ServoStep, rate_step_toward};
use crate::trajectory::{
    ARM_ANGLE_STEP_PER_BLEND_RAD, CartesianPlan, CartesianTrajectory, JointTrajectory, PlanLimits,
    plan_cartesian, subdivided_blends,
};
use crate::types::{ARM_DOF, JointVec, Side, world_pose_arrays};
use crate::upstream::Upstream;

/// Slack the runtime per-tick Cartesian velocity check allows over the planned
/// limit before aborting (mirrors the arm's backstop over the up-front plan,
/// as [`MOTION_TIMEOUT_FACTOR`] does over the rollout's duration).
const VELOCITY_GUARD_MARGIN: f64 = 1.5;

/// Per-arm static configuration for the planner (the motion limits relocated
/// from the arm). Cloned per side.
#[derive(Clone)]
pub struct PlanConfig {
    pub cycle_period: Duration,
    /// The servo's command smoother, built for `cycle_period` at setup.
    pub smoothing: control_core::filters::ButterworthFilter,
    pub max_joint_velocity_rad_s: JointVec,
    pub ee: EeCaps,
    pub limits: [Limit; ARM_DOF],
}

/// How a joint move ended, reported to a ready-move aggregation.
pub struct ReadyOutcome {
    pub success: bool,
    pub message: String,
}

/// One arm's share of a whole-robot ready move: where its terminal reports,
/// and the cancel flag the admission task flips on the action's cancel.
pub struct ReadyReply {
    pub done_tx: mpsc::Sender<ReadyOutcome>,
    pub cancelled: Arc<AtomicBool>,
}

/// Where a joint move reports its terminal: the move_arm_joints action goal
/// that carried it (boxed: the goal context dwarfs the other variant), or one
/// arm's share of a move_to_ready goal.
pub enum JointReply {
    MoveArmJoints(Box<move_arm_joints::GoalContext>),
    Ready(ReadyReply),
}

impl JointReply {
    fn is_cancelled(&self) -> bool {
        match self {
            Self::MoveArmJoints(ctx) => ctx.is_cancelled(),
            Self::Ready(r) => r.cancelled.load(Ordering::Acquire),
        }
    }

    /// Report the terminal. The ready share folds `Cancelled` into a failed
    /// outcome; the joint action keeps its distinct cancelled completion.
    async fn finish(
        self,
        side: &'static str,
        outcome: Outcome,
        measured_q: JointVec,
        elapsed_s: f64,
    ) {
        let cancelled = matches!(outcome, Outcome::Cancelled);
        let (success, message) = match outcome {
            Outcome::Complete => (true, "trajectory complete".to_string()),
            Outcome::Cancelled => (false, "goal cancelled".to_string()),
            Outcome::Failed(reason) => (false, reason),
        };
        match self {
            Self::MoveArmJoints(ctx) => {
                let result = if cancelled {
                    ctx.complete_cancelled(success, message, measured_q, elapsed_s)
                        .await
                } else {
                    ctx.complete(success, message, measured_q, elapsed_s).await
                };
                if let Err(e) = result {
                    error!("{side}: move_arm_joints complete: {e}");
                }
            }
            Self::Ready(r) => {
                let outcome = ReadyOutcome {
                    success,
                    message: format!("{side}: {message}"),
                };
                if r.done_tx.send(outcome).await.is_err() {
                    error!("{side}: ready outcome aggregation closed");
                }
            }
        }
    }
}

/// A goal accepted by an action handler and handed to the planner.
pub enum Goal {
    Joint {
        target: JointVec,
        duration_s: f64,
        reply: JointReply,
    },
    Cartesian {
        target: Isometry3<f64>,
        duration_s: f64,
        ctx: Box<move_arm::GoalContext>,
    },
}

impl Goal {
    /// Complete unstarted, `success: false`, reporting `reported_q` (the
    /// measured joints, per the result contract). The busy flag is the
    /// caller's concern.
    pub async fn refuse(self, reason: &str, reported_q: JointVec, planner: &mut Planner) {
        match self {
            Goal::Joint { reply, .. } => {
                reply
                    .finish(
                        planner.side.label(),
                        Outcome::Failed(reason.to_string()),
                        reported_q,
                        0.0,
                    )
                    .await;
            }
            Goal::Cartesian { ctx, .. } => {
                planner
                    .finish_cartesian(&ctx, reported_q, false, reason, 0.0, false)
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
    /// Tracking a quintic joint trajectory for an accepted move_arm_joints goal.
    JointMove(JointMove),
    /// Tracking a Cartesian pose trajectory for an accepted move_arm goal, solving
    /// IK each tick for the joint target.
    CartesianMove(CartesianMove),
}

struct JointMove {
    traj: JointTrajectory,
    reply: JointReply,
    _busy: BusyGuard,
}

struct CartesianMove {
    path: MovePath,
    ctx: Box<move_arm::GoalContext>,
    // Last commanded joint target: held on cancel/failure so the arm never snaps.
    prev_q_des: JointVec,
    _busy: BusyGuard,
}

/// A line-tracking move's cursor: the trajectory, and where the last tick left
/// the IK seed, the blend parameter and the sample clock.
struct LineTrack {
    traj: CartesianTrajectory,
    seed: JointVec,
    prev_sample_at: Instant,
    /// Blend parameter at the previous tick: the walk resumes from here, and a
    /// steered line's elbow budget scales with the blend progressed since, so
    /// the executed elbow travel matches the plan's.
    prev_blend: f64,
    /// Resolve the elbow the way the plan validated: steered (manipulability
    /// budget) or held at the seed angle (the quiet default).
    steer_elbow: bool,
}

/// A servo-guided move's cursor: the feedback state, the sample clock, and the
/// rollout the plan validated.
struct ServoTrack {
    /// Boxed: [`ServoState`] carries a per-joint filter bank, so inlining it
    /// would bloat every [`Mode`] variant. One heap alloc per servo move.
    servo: Box<ServoState>,
    started: Instant,
    prev_sample_at: Instant,
    /// The plan-time rollout duration. The runtime aborts once the move runs
    /// past `MOTION_TIMEOUT_FACTOR` times this, tying the timeout to the
    /// validated motion length rather than a flat ceiling. Re-budgeted by
    /// [`MoveBudget::after_rate_change`] when the operator slows the EE speed
    /// cap mid-move, since the rollout that produced it assumed the cap in
    /// force at admission.
    budget: MoveBudget,
}

/// How an admitted move_arm goal executes, per its [`CartesianPlan`]: track the
/// straight line (solving IK each tick), or run the guarded servo when no
/// continuous joint path tracks the line.
enum MovePath {
    Line(LineTrack),
    Servo(ServoTrack),
}

impl MovePath {
    fn motion_start(&self) -> Instant {
        match self {
            Self::Line(line) => line.traj.motion_start,
            Self::Servo(servo) => servo.started,
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

/// How a move ended. `Cancelled` is the only terminal the caller asked for,
/// and the only one reported through the cancelled completion.
enum Outcome {
    Complete,
    Cancelled,
    Failed(String),
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
}

impl Planner {
    /// Start holding zero with no producer locked. The coordinator seeds the real
    /// held pose from the first measured state (via [`Planner::commit`]) before any
    /// setpoint is published, so this initial value is never streamed.
    pub fn new(side: Side, model: Arm, cfg: PlanConfig) -> Self {
        Self {
            side,
            model,
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
        self.setpoint = clamp_to_limits(&measured, &self.cfg.limits);
    }

    /// The last published setpoint, the coordinator's `prev` for the governor.
    pub fn setpoint(&self) -> JointVec {
        self.setpoint
    }

    /// World-frame end-effector pose at `q`; the planner owns this arm's chain.
    pub fn ee_pose_world(&mut self, q: &JointVec) -> Isometry3<f64> {
        let ee_base = self.model.at(q).ee_pose();
        self.model.world_pose(&ee_base)
    }

    /// Retune the end-effector speed cap at runtime (the commander's control).
    /// This copy budgets planned moves at admission, paces the servo
    /// reference, and steps streamed poses; a streamed joint command is capped
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
                goal.refuse("another move is in flight", measured_q, self)
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
            candidate: clamp_to_limits(&stepped, &self.cfg.limits),
            streamed_hand: match origin {
                Origin::Streamed => Some(self.model.at(&measured_q).jacobian()),
                Origin::PlannedMove => None,
            },
        }
    }

    /// Fail the active move and answer its caller: for a follower gone
    /// stale, where the move can neither progress nor be verified. The result
    /// reports `measured_q`, the follower's last (frozen) measurement. Ambient
    /// Follow mode is untouched; the busy slot releases with the mode.
    pub async fn abort_active(&mut self, reason: &str, measured_q: JointVec, now: Instant) {
        match std::mem::replace(&mut self.mode, Mode::Follow) {
            Mode::Follow => {}
            Mode::JointMove(JointMove { traj, reply, _busy }) => {
                let elapsed = now.duration_since(traj.motion_start).as_secs_f64();
                reply
                    .finish(
                        self.side.label(),
                        Outcome::Failed(reason.to_string()),
                        measured_q,
                        elapsed,
                    )
                    .await;
            }
            Mode::CartesianMove(m) => {
                let elapsed = now.duration_since(m.path.motion_start()).as_secs_f64();
                self.finish_cartesian(&m.ctx, measured_q, false, reason, elapsed, false)
                    .await;
            }
        }
    }

    /// Resolve the Follow target, holding the last governed setpoint when no
    /// command has arrived. If the producer stops, the latest command persists
    /// (held for joints, converged to for a pose); no freshness deadman.
    ///
    /// A joint command is chased directly. A pose gets one damped
    /// resolved-rate step from the *governed* setpoint (measured lags by
    /// tracking error, and seeding from it would fold that error into every
    /// step); bounded rather than solved, so an unreachable pose saturates
    /// into a steady pull the governor can shape.
    fn follow_target(&mut self, command: &Option<Upstream>) -> JointVec {
        let target = match command {
            None => return self.setpoint,
            Some(Upstream::Joints(positions)) => *positions,
            Some(Upstream::Pose(pose)) => {
                let q = self.setpoint;
                let ee = self.ee_pose_world(&q);
                rate_step_toward(
                    &mut self.model,
                    &q,
                    &ee,
                    pose,
                    &self.cfg.max_joint_velocity_rad_s,
                    self.cfg.ee,
                    self.cfg.cycle_period.as_secs_f64(),
                )
            }
        };
        clamp_to_limits(&target, &self.cfg.limits)
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

    /// One joint-trajectory tick: sample the quintic, and complete the goal on
    /// cancel or on running to the end. Either terminal drops `m` (and with it
    /// the busy guard), releasing the slot.
    async fn advance_joint(&mut self, m: JointMove, measured_q: JointVec, now: Instant) -> Advance {
        let JointMove { traj, reply, _busy } = m;
        let q_des = traj.sample(now);
        let cancelled = reply.is_cancelled();
        if !cancelled && !traj.is_complete(now) {
            return Advance {
                target: q_des,
                next_mode: Mode::JointMove(JointMove { traj, reply, _busy }),
                origin: Origin::PlannedMove,
            };
        }
        // Success means the trajectory ran to completion (not cancelled). The
        // result carries the measured pose, so the caller judges how close it
        // landed; the governor may have held it short.
        let elapsed = now.duration_since(traj.motion_start).as_secs_f64();
        let outcome = if cancelled {
            Outcome::Cancelled
        } else {
            Outcome::Complete
        };
        reply
            .finish(self.side.label(), outcome, measured_q, elapsed)
            .await;
        // A cancel holds where the arm already is; a completion holds the last
        // sample.
        Advance::ends_move(if cancelled { self.setpoint } else { q_des })
    }

    /// One Cartesian tick: advance the move's path (line-tracking IK, or the
    /// guarded servo) and complete on cancel, on the path giving up, or on
    /// normal completion. Any terminal drops `m` (and with it the busy guard),
    /// releasing the slot.
    async fn advance_cartesian(
        &mut self,
        mut m: CartesianMove,
        measured_q: JointVec,
        now: Instant,
    ) -> Advance {
        let elapsed = now.duration_since(m.path.motion_start()).as_secs_f64();
        if m.ctx.is_cancelled() {
            return self
                .end_cartesian(&m, measured_q, Outcome::Cancelled, elapsed)
                .await;
        }
        let stepped = match &mut m.path {
            MovePath::Line(line) => self.step_line(line, &m.prev_q_des, now),
            MovePath::Servo(servo) => self.step_servo(servo, elapsed, now),
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
        if !complete {
            return Advance {
                target: q_des,
                next_mode: Mode::CartesianMove(m),
                origin: Origin::PlannedMove,
            };
        }
        self.end_cartesian(&m, measured_q, Outcome::Complete, elapsed)
            .await
    }

    /// Track the straight line: walk the blend progressed this tick at no
    /// coarser than the plan's validated resolution (a short move's quintic can
    /// outpace the plan grid), seed-chaining each sample; the last solution is
    /// the tick's setpoint. A steered line budgets the elbow per sub-step
    /// exactly like the plan's per-sample cap; a held line pins it to the seed.
    fn step_line(&mut self, line: &mut LineTrack, prev_q_des: &JointVec, now: Instant) -> PathStep {
        let blend = line.traj.blend(now);
        let mut q_next = line.seed;
        let mut s_prev = line.prev_blend;
        for s_k in subdivided_blends(line.prev_blend, blend) {
            let policy = if line.steer_elbow {
                ArmAnglePolicy::MaxManipulability {
                    max_step_rad: ARM_ANGLE_STEP_PER_BLEND_RAD * (s_k - s_prev),
                }
            } else {
                ArmAnglePolicy::FromSeed
            };
            let base_target = self.model.base_pose(&line.traj.sample_at_blend(s_k));
            let Some(sol) = self.model.solve_ik(&base_target, policy, &q_next) else {
                return PathStep::Failed(
                    "IK failed mid-trajectory (unreachable / singular)".into(),
                );
            };
            q_next = sol.q;
            s_prev = s_k;
        }
        let dt = now
            .duration_since(line.prev_sample_at)
            .as_secs_f64()
            .max(self.cfg.cycle_period.as_secs_f64() * 0.5);
        if exceeds_velocity_limits(&q_next, prev_q_des, &self.cfg.max_joint_velocity_rad_s, dt) {
            return PathStep::Failed("joint velocity limit exceeded near singularity".into());
        }
        line.seed = q_next;
        line.prev_sample_at = now;
        line.prev_blend = blend;
        PathStep::To {
            q_des: q_next,
            complete: line.traj.is_complete(now),
        }
    }

    /// Run the guarded servo: one damped resolved-rate step toward the leashed
    /// line reference, the law the plan's rollout validated. Its steps are
    /// velocity-clamped by construction; the budget ceiling terminates a move
    /// the live geometry stops cooperating with (the plan proved the nominal
    /// path, not every disturbance).
    fn step_servo(&mut self, track: &mut ServoTrack, elapsed: f64, now: Instant) -> PathStep {
        // Measured dt keeps the feedback law honest under tick jitter (each step
        // is velocity-scaled by the same dt), clamped so a scheduling stall
        // cannot turn one tick into a giant step.
        let dt = now
            .duration_since(track.prev_sample_at)
            .clamp(self.cfg.cycle_period / 2, self.cfg.cycle_period * 4);
        track.prev_sample_at = now;
        // Feed the servo the governed setpoint, not the pre-governor target: if
        // the governor holds the arm, the loop must advance from where the arm
        // actually is, or it would run ahead and report convergence while the
        // arm sits short of the goal.
        let governed_q = self.setpoint;
        // The rollout that produced this budget assumed the cap in force at
        // admission; the step below uses whatever the operator has set since.
        track.budget = track
            .budget
            .after_rate_change(elapsed, self.cfg.ee.linear_m_s);
        let step = track.servo.step(
            &mut self.model,
            &governed_q,
            &self.cfg.max_joint_velocity_rad_s,
            self.cfg.ee,
            dt,
        );
        match step {
            ServoStep::Converged(q) => PathStep::To {
                q_des: q,
                complete: true,
            },
            ServoStep::Stepped(q) if !track.budget.timed_out(elapsed) => PathStep::To {
                q_des: q,
                complete: false,
            },
            ServoStep::Stepped(_) => {
                let short_m = track.servo.position_err_m(&mut self.model, &governed_q);
                PathStep::Failed(format!(
                    "servo overran {MOTION_TIMEOUT_FACTOR:.0}x its {:.1}s rollout, {:.0} mm short of the goal",
                    track.budget.seconds(),
                    short_m * 1000.0
                ))
            }
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
        let (success, message) = match &outcome {
            Outcome::Complete => (true, m.path.completion_message()),
            Outcome::Cancelled => (false, "goal cancelled"),
            Outcome::Failed(reason) => (false, reason.as_str()),
        };
        self.finish_cartesian(
            &m.ctx,
            measured_q,
            success,
            message,
            elapsed,
            matches!(outcome, Outcome::Cancelled),
        )
        .await;
        Advance::ends_move(m.prev_q_des)
    }

    /// Start an accepted goal in the mode that executes it.
    async fn start_goal(&mut self, goal: Goal, busy: Arc<AtomicBool>, now: Instant) -> Mode {
        let busy = BusyGuard(busy);
        match goal {
            Goal::Joint {
                target,
                duration_s,
                reply,
            } => {
                info!("{}: joint move start", self.side.label());
                Mode::JointMove(JointMove {
                    traj: JointTrajectory::new(
                        self.setpoint,
                        target,
                        self.cfg.max_joint_velocity_rad_s,
                        duration_s,
                    ),
                    reply,
                    _busy: busy,
                })
            }
            Goal::Cartesian {
                target,
                duration_s,
                ctx,
            } => {
                self.start_cartesian(target, duration_s, ctx, busy, now)
                    .await
            }
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
        target: Isometry3<f64>,
        duration_s: f64,
        ctx: Box<move_arm::GoalContext>,
        busy: BusyGuard,
        now: Instant,
    ) -> Mode {
        let ee_base = self.model.at(&self.setpoint).ee_pose();
        let start_world = self.model.world_pose(&ee_base);
        let plan = plan_cartesian(
            &mut self.model,
            &start_world,
            &target,
            self.setpoint,
            &PlanLimits {
                max_joint_velocity_rad_s: &self.cfg.max_joint_velocity_rad_s,
                ee: self.cfg.ee,
                control_period: self.cfg.cycle_period,
                smoothing: self.cfg.smoothing,
            },
            duration_s,
        );
        let Some(plan) = plan else {
            let (pos, quat) = world_pose_arrays(&start_world);
            if let Err(e) = ctx
                .complete(
                    false,
                    "goal pose unreachable (no line tracks and the servo rollout stalls)".into(),
                    pos,
                    quat,
                    0.0,
                )
                .await
            {
                error!("{}: move_arm complete: {e}", self.side.label());
            }
            // `busy` drops here: the slot is released even on this early exit.
            return Mode::Follow;
        };
        let path = match plan {
            CartesianPlan::Line {
                duration_s,
                steer_elbow,
                start_q,
            } => {
                info!(
                    "{}: move_arm start{}, duration={duration_s:.3}s",
                    self.side.label(),
                    if steer_elbow {
                        " (steered elbow)"
                    } else {
                        " (held elbow)"
                    }
                );
                MovePath::Line(LineTrack {
                    // Seed from the plan's normalized start, not self.setpoint:
                    // the plan validated the walk from here, so execution must
                    // begin on the same IK branch (they can differ at a
                    // redundancy/limit boundary). The chase still moves the arm
                    // from its actual setpoint, velocity-limited.
                    traj: CartesianTrajectory::new(start_world, target, duration_s),
                    seed: start_q,
                    prev_sample_at: now,
                    prev_blend: 0.0,
                    steer_elbow,
                })
            }
            // No continuous joint path tracks the line: run the guarded servo
            // the rollout just validated, the same damped law the operator's
            // streaming jog crosses these walls with.
            CartesianPlan::Servo { duration_s } => {
                info!(
                    "{}: move_arm start (servo-guided), rollout={duration_s:.3}s",
                    self.side.label()
                );
                MovePath::Servo(ServoTrack {
                    servo: Box::new(ServoState::new(start_world, target, self.cfg.smoothing)),
                    started: now,
                    prev_sample_at: now,
                    budget: MoveBudget::new(duration_s, self.cfg.ee.linear_m_s),
                })
            }
        };
        // The line's velocity guard compares each sample against the last
        // commanded target, so the first comparison must sit on the plan's own
        // IK branch (start_q), which can differ from the held setpoint at a
        // redundancy or limit boundary; seeding from the setpoint there would
        // trip the guard on a valid line's first tick. The servo has no branch
        // to disagree with and steps from the governed setpoint.
        let prev_q_des = match &path {
            MovePath::Line(line) => line.seed,
            MovePath::Servo(_) => self.setpoint,
        };
        Mode::CartesianMove(CartesianMove {
            path,
            ctx,
            prev_q_des,
            _busy: busy,
        })
    }

    /// Complete a Cartesian goal, reporting the measured world pose at exit.
    async fn finish_cartesian(
        &mut self,
        ctx: &move_arm::GoalContext,
        measured_q: JointVec,
        success: bool,
        message: &str,
        elapsed: f64,
        cancelled: bool,
    ) {
        let (pos, quat) = world_pose_arrays(&self.ee_pose_world(&measured_q));
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

/// Whether stepping `q_prev -> q_new` over `dt` implies any joint velocity beyond
/// the guard margin times its limit.
fn exceeds_velocity_limits(
    q_new: &JointVec,
    q_prev: &JointVec,
    max_vel: &JointVec,
    dt: f64,
) -> bool {
    q_new
        .iter()
        .zip(q_prev)
        .zip(max_vel)
        .any(|((&n, &p), &v)| (n - p).abs() > v * dt * VELOCITY_GUARD_MARGIN)
}

#[cfg(test)]
mod tests {
    use super::*;

    use srs_model::nalgebra::{UnitQuaternion, Vector3};

    fn test_cfg() -> PlanConfig {
        PlanConfig {
            cycle_period: Duration::from_millis(10),
            smoothing: crate::servo::smoothing_for(Duration::from_millis(10)).unwrap(),
            max_joint_velocity_rad_s: [10.0; ARM_DOF],
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

    #[test]
    fn slowing_the_ee_cap_mid_move_extends_the_servo_deadline() {
        // The rollout budgets a move against the cap in force at admission,
        // then every step runs at whatever the operator has since streamed. A
        // move budgeted at 1.0 m/s and then driven at half that must be given
        // the extra time, or it aborts short of a goal it was still reaching.
        let mut planner = test_planner(POSE_TEST_Q);
        let start = planner.ee_pose_world(&POSE_TEST_Q);
        let now = Instant::now();
        let mut track = ServoTrack {
            servo: Box::new(ServoState::new(start, start, planner.cfg.smoothing)),
            started: now,
            prev_sample_at: now,
            budget: MoveBudget::new(1.0, planner.cfg.ee.linear_m_s),
        };

        planner.set_max_ee_velocity(planner.cfg.ee.linear_m_s / 2.0);
        planner.step_servo(&mut track, 0.25, now + planner.cfg.cycle_period);

        // A quarter of the budget was spent at the old cap; the remaining
        // three quarters now need twice as long.
        assert!(
            (track.budget.seconds() - (0.25 + 0.75 * 2.0)).abs() < 1e-9,
            "expected the remaining deadline to double, got {}",
            track.budget.seconds()
        );
    }

    #[tokio::test]
    async fn aborting_the_active_move_fails_it_and_frees_the_busy_slot() {
        let mut planner = test_planner([0.0; ARM_DOF]);
        let (done_tx, mut done_rx) = mpsc::channel(1);
        let (goal_tx, mut goals) = mpsc::channel(1);
        let busy = Arc::new(AtomicBool::new(true)); // claimed at accept
        goal_tx
            .send(Goal::Joint {
                target: [0.5; ARM_DOF],
                duration_s: 1.0,
                reply: JointReply::Ready(ReadyReply {
                    done_tx,
                    cancelled: Arc::new(AtomicBool::new(false)),
                }),
            })
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
            .abort_active("the follower stopped reporting", [0.1; ARM_DOF], now)
            .await;
        let outcome = done_rx.recv().await.expect("aborted move must report");
        assert!(!outcome.success);
        assert!(outcome.message.contains("stopped reporting"));
        assert!(!busy.load(Ordering::Acquire), "abort releases the slot");
        // With nothing active the abort is a no-op.
        planner.abort_active("again", [0.1; ARM_DOF], now).await;
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
        let start = planner.ee_pose_world(&POSE_TEST_Q);
        let goal = Isometry3::from_parts(
            (start.translation.vector + Vector3::new(0.05, 0.0, 0.0)).into(),
            start.rotation,
        );

        let stepped = planner.follow_target(&Some(Upstream::Pose(goal)));
        let moved = planner.ee_pose_world(&stepped);
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
        let start = planner.ee_pose_world(&POSE_TEST_Q);
        let far = Isometry3::from_parts(
            (start.translation.vector + Vector3::new(5.0, 0.0, 0.0)).into(),
            start.rotation,
        );

        let stepped = planner.follow_target(&Some(Upstream::Pose(far)));
        let moved = planner.ee_pose_world(&stepped);
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
        let start = planner.ee_pose_world(&POSE_TEST_Q);
        let twisted = Isometry3::from_parts(
            start.translation,
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 1.0) * start.rotation,
        );

        let stepped = planner.follow_target(&Some(Upstream::Pose(twisted)));
        let moved = planner.ee_pose_world(&stepped);
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
        let start = planner.ee_pose_world(&POSE_TEST_Q);
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

    #[test]
    fn exceeds_velocity_limits_at_the_guard_boundary() {
        let prev = [0.0; ARM_DOF];
        let vmax = [1.0; ARM_DOF];
        let dt = 0.1;
        // limit * dt * margin = 1.0 * 0.1 * 1.5 = 0.15 rad allowed this step.
        let mut under = [0.0; ARM_DOF];
        under[0] = 0.149;
        let mut over = [0.0; ARM_DOF];
        over[0] = 0.151;
        assert!(!exceeds_velocity_limits(&under, &prev, &vmax, dt));
        assert!(exceeds_velocity_limits(&over, &prev, &vmax, dt));
    }
}
