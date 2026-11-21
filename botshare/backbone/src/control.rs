//! The control loop: it owns the velocity limiter, the follower of
//! drive_link's leader, the odometry, the wheel stream, the goal in flight
//! and every publisher of the wheels, the base's state and its transforms,
//! and it runs every event the node receives and every step on the node's
//! clock.
//!
//! Events run in arrival order: a wheel measurement, the leader's setpoint, an
//! accepted goal, a robot stop. After each event, and at each wall-time poll,
//! the loop runs every step that is due on the node's clock. The rules each
//! step, measurement and goal follow are the README's ("The step", "The
//! follower", "The robot stop", "The odometry and the reports", "Goals"):
//! each wheel setpoint is control_core's [`WheelSetpoint::next`] within
//! [`BaseConfig::anchor_bound_rad`], an odometry epoch ends on
//! [`BaseConfig::wheel_reach_rad`], the wheels fall silent after
//! [`SILENT_AFTER_NS`], and the newest measurement is reported again before
//! [`REPORT_PERIOD_NS`] pass without a report.

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use control_core::limiter::Limiter;
use control_core::move_slot::MoveSlot;
use control_core::odometry::{Odometry, WheelSetpoint};
use control_core::planar_pose::{PlanarPose, normalize_angle};
use control_core::robot_tree::RobotTree;
use control_core::stop_reason::stop_reason;
use control_core::time::{nanoseconds, seconds};
use control_core::twist::PlanarTwist;
use frame_tree::arithmetic::quaternion_of_yaw;
use frame_tree::{MovingMessage, Publisher as TreePublisher};
use peppygen::emitted_topics::base_state::base_states;
use peppygen::exposed_actions::base_motion::{drive_base, move_base};
use peppygen::paired_topics::base::joint_setpoints;
use peppygen::paired_topics::drive::drive_states;
use peppygen::{NodeRunner, Result};
use peppylib::{Payload, TopicPublisher};
use tokio::sync::{mpsc, oneshot};
use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};

use crate::clock::{self, wire_from_ns};
use crate::config::BaseConfig;
use crate::failure::{Failure, FailureKind};
use crate::follower::{Follower, LEADER_STREAMING_WINDOW_NS};
use crate::goal::{DriveRequest, MoveRequest};
use crate::move_control::MoveController;
use crate::rest::RestWindow;
use crate::status::{Admitted, BaseStatus};
use crate::transforms::{self, TransformsError};

/// The base's part name: the link id of its wheels' pairing slot, the name
/// describe_base answers and the name a robot stop answers when it stopped
/// the base.
pub const BASE_NAME: &str = joint_setpoints::LINK_ID;

/// How old the wheels' newest measurement may be before they are silent
/// (ns): base_state's 0.5 s.
pub const SILENT_AFTER_NS: i64 = 500_000_000;

/// The longest the base goes without a base_states while the wheels report
/// (ns): base_state's 0.1 s.
pub const REPORT_PERIOD_NS: i64 = 100_000_000;

/// A measurement of both wheels, [left, right]: their angles (rad), when the
/// wheels measured them on the node's clock (ns), and the instance of the
/// wheel follower that sent it.
#[derive(Clone, Debug)]
pub struct WheelState {
    pub positions: [f64; 2],
    pub timestamp_ns: i64,
    pub follower: String,
}

/// A goal the action handlers accepted, with the claim on the base it holds.
pub enum AcceptedGoal {
    Drive {
        request: DriveRequest,
        ctx: Box<drive_base::GoalContext>,
        claim: Admitted,
    },
    Move {
        request: MoveRequest,
        ctx: Box<move_base::GoalContext>,
        claim: Admitted,
    },
}

/// What the loop receives, in arrival order.
pub enum Event {
    Wheels(WheelState),
    /// A setpoint of the leader, finite, received at `received_ns` on the
    /// node's clock.
    Setpoint {
        velocity: PlanarTwist,
        received_ns: i64,
    },
    Goal(AcceptedGoal),
    /// The robot stop, answering the base's name when it ended a goal or the
    /// leader streamed.
    Stop {
        reason: String,
        reply: oneshot::Sender<Vec<String>>,
    },
}

/// A receipt time that puts a setpoint past drive_link's 0.5 s at `now_ns`:
/// the limiter takes such a setpoint as one that stopped, brings the base to
/// rest at its deceleration limits and then gives nothing.
fn expired_at(now_ns: i64) -> i64 {
    now_ns - LEADER_STREAMING_WINDOW_NS - 1
}

/// Why a goal fails when the base's clock reads earlier than a time the
/// base already stepped or admitted at.
const CLOCK_WENT_BACK: &str = "the base's clock went back";

/// Why a goal fails when the base's odometry starts over.
const ODOMETRY_STARTED_OVER: &str = "the base's odometry started over";

/// How an admitted goal ends as the loop takes it, when it does: cancelled
/// for a stop that came on its way, `stop`; failed when the base's clock went
/// back past its admission, or when the odometry it was placed by started
/// over; started otherwise.
fn refusal_at_start(
    stop: Option<String>,
    clock_went_back: bool,
    started_over: bool,
) -> Option<Outcome> {
    let failed = |message| Outcome::Failed(Failure::new(FailureKind::ResourceUnavailable, message));
    match (stop, clock_went_back, started_over) {
        (Some(reason), _, _) => Some(Outcome::Ending(Ending::Cancelled(reason))),
        (None, true, _) => Some(failed(CLOCK_WENT_BACK)),
        (None, false, true) => Some(failed(ODOMETRY_STARTED_OVER)),
        (None, false, false) => None,
    }
}

/// How a goal that comes to rest ends once the base rests.
#[derive(Clone, Debug)]
enum Ending {
    Completed,
    Cancelled(String),
}

/// How a goal ends: on its ending, or failed.
#[derive(Debug)]
enum Outcome {
    Ending(Ending),
    Failed(Failure),
}

/// How a goal that was to end on `ending` ends after a cancel for `reason`,
/// and whether the cancel changed it: a goal that runs, or that completed
/// and still comes to rest, ends cancelled; a goal already cancelled keeps
/// its reason. A stop and a caller's cancel each cancel by this rule.
fn cancelled(ending: Option<Ending>, reason: String) -> (Ending, bool) {
    match ending {
        None | Some(Ending::Completed) => (Ending::Cancelled(reason), true),
        Some(kept @ Ending::Cancelled(_)) => (kept, false),
    }
}

/// The motion a goal runs.
enum GoalMotion {
    /// The velocity, held for `hold_duration_ns` from `reached_ns`, the step
    /// at which the command reached it; then the command comes to 0 at
    /// `stopped_ns`, after which the base has `settle_timeout_ns` to rest.
    Drive {
        ctx: Box<drive_base::GoalContext>,
        velocity: PlanarTwist,
        hold_duration_ns: i64,
        settle_timeout_ns: i64,
        reached_ns: Option<i64>,
        stopped_ns: Option<i64>,
    },
    /// The steering of the move, its planned end, its settle timeout and
    /// the end of its settle (ns on the node's clock).
    Move {
        ctx: Box<move_base::GoalContext>,
        controller: MoveController,
        planned_end_ns: i64,
        settle_timeout_ns: i64,
        settle_end_ns: i64,
    },
}

impl GoalMotion {
    /// The goal's settle timeout (ns).
    fn settle_timeout_ns(&self) -> i64 {
        match self {
            Self::Drive {
                settle_timeout_ns, ..
            }
            | Self::Move {
                settle_timeout_ns, ..
            } => *settle_timeout_ns,
        }
    }
}

/// `pose`, a pose of the robot frame in odom, in the goal's frame: the robot
/// frame at `start` in odom when the goal started.
fn seen_from_start(pose: PlanarPose, start: PlanarPose) -> PlanarPose {
    pose.seen_from(start)
        .expect("two poses the odometry reaches lie a finite distance apart")
}

/// The motion an accepted goal runs from the time it was admitted at, a move
/// steering in the goal's frame, the robot frame at `start` in odom, on from
/// `command`, the limiter's velocity as it starts, and its claim on the base.
fn goal_motion(
    accepted: AcceptedGoal,
    start: PlanarPose,
    command: PlanarTwist,
) -> (GoalMotion, Admitted) {
    match accepted {
        AcceptedGoal::Drive {
            request,
            ctx,
            claim,
        } => (
            GoalMotion::Drive {
                ctx,
                velocity: request.velocity,
                hold_duration_ns: request.hold_duration_ns,
                settle_timeout_ns: request.settle_timeout_ns,
                reached_ns: None,
                stopped_ns: None,
            },
            claim,
        ),
        AcceptedGoal::Move {
            request,
            ctx,
            claim,
        } => {
            let plan = request.plan;
            let planned_end_ns = claim.admitted_ns() + nanoseconds(plan.duration_s);
            (
                GoalMotion::Move {
                    ctx,
                    controller: MoveController::new(
                        seen_from_start(request.target, start),
                        plan,
                        request.tolerances,
                        request.rotate_to_heading_min_angle,
                        command,
                    ),
                    planned_end_ns,
                    settle_timeout_ns: request.settle_timeout_ns,
                    settle_end_ns: planned_end_ns + request.settle_timeout_ns,
                },
                claim,
            )
        }
    }
}

/// The goal in flight: where it started, how it ends once the base rests,
/// the end of the settle of a cancelled goal, and the claim on the base it
/// releases when it ends.
struct RunningGoal {
    start: PlanarPose,
    started: Instant,
    ending: Option<Ending>,
    /// The goal's settle timeout after the first step that found it
    /// cancelled (ns on the node's clock), once a step has.
    cancel_settle_end_ns: Option<i64>,
    motion: GoalMotion,
    claim: Admitted,
}

/// Whether `velocity` has reached `target` on every axis.
fn reached(velocity: PlanarTwist, target: PlanarTwist) -> bool {
    /// The relative difference below which a ramp has arrived: the last step
    /// of a ramp lands on its target up to a rounding of the sum.
    const ARRIVED: f64 = 1e-9;
    velocity
        .axes()
        .iter()
        .zip(target.axes())
        .all(|(value, target)| (value - target).abs() <= ARRIVED * target.abs().max(1.0))
}

/// Where the odometry places the robot frame in odom, and the time of its
/// newest measurement: the final pose of a goal.
#[derive(Clone, Copy, Debug)]
struct Located {
    pose: PlanarPose,
    timestamp: SystemTime,
}

/// End `$ctx`, the goal context of either action, on `$outcome`, with the
/// final pose `$located` given as `$action`'s `$final_pose` type, after
/// `$action_time` (s). Written once here because the two generated contexts
/// carry distinct types with an identical surface.
macro_rules! end_goal {
    ($ctx:expr, $outcome:expr, $located:expr, $action_time:expr, $action:ident::$final_pose:ident) => {{
        let Located { pose, timestamp } = $located;
        let final_pose = $action::$final_pose {
            frame: transforms::ODOMETRY_FRAME.to_owned(),
            timestamp,
            position: [pose.x(), pose.y(), 0.0],
            orientation: quaternion_of_yaw(pose.yaw()),
        };
        match $outcome {
            Outcome::Ending(Ending::Completed) => $ctx.complete(final_pose, $action_time).await,
            Outcome::Ending(Ending::Cancelled(reason)) => {
                $ctx.complete_cancelled(reason, final_pose, $action_time)
                    .await
            }
            Outcome::Failed(failure) => $ctx.fail(failure.kind.into(), failure.message).await,
        }
    }};
}

impl RunningGoal {
    /// The accepted goal, started with the robot frame at `start` and the
    /// limiter at `command`.
    fn new(accepted: AcceptedGoal, start: PlanarPose, command: PlanarTwist) -> Self {
        let (motion, claim) = goal_motion(accepted, start, command);
        Self {
            start,
            started: Instant::now(),
            ending: None,
            cancel_settle_end_ns: None,
            motion,
            claim,
        }
    }

    /// After the step at `now_ns`: the ending of a goal a cancel or a stop
    /// ended, once its settle timeout has passed from the first step that
    /// found it cancelled, the base rested or not.
    fn unsettled_cancel(&mut self, now_ns: i64) -> Option<Ending> {
        let Some(ending @ Ending::Cancelled(_)) = &self.ending else {
            return None;
        };
        let settle_end_ns = *self
            .cancel_settle_end_ns
            .get_or_insert(now_ns + self.motion.settle_timeout_ns());
        (now_ns >= settle_end_ns).then(|| ending.clone())
    }

    /// The reason of the cancel the goal's caller sent, if it did.
    fn cancel_reason(&self) -> Option<String> {
        let cause = match &self.motion {
            GoalMotion::Drive { ctx, .. } => ctx.cancel_cause(),
            GoalMotion::Move { ctx, .. } => ctx.cancel_cause(),
        };
        cause.map(|cause| cause.reason().to_string())
    }

    /// Cancel the goal for `reason` by [`cancelled`]'s rule; answers whether
    /// the cancel changed how it ends.
    fn cancel(&mut self, reason: String) -> bool {
        let (ending, changed) = cancelled(self.ending.take(), reason);
        self.ending = Some(ending);
        changed
    }

    /// End the goal on `outcome`, with the robot frame `located` in odom. The
    /// claim on the base drops first, so a goal sent on the outcome finds the
    /// base free. A failed goal gives no result.
    async fn finish(self, outcome: Outcome, located: Located) {
        let Self {
            started,
            motion,
            claim,
            ..
        } = self;
        drop(claim);
        let action_time = started.elapsed().as_secs_f64();
        let ended = match motion {
            GoalMotion::Drive { ctx, .. } => end_goal!(
                ctx,
                outcome,
                located,
                action_time,
                drive_base::DriveBaseResultFinalPose
            ),
            GoalMotion::Move { ctx, .. } => end_goal!(
                ctx,
                outcome,
                located,
                action_time,
                move_base::MoveBaseResultFinalPose
            ),
        };
        if let Err(e) = ended {
            error!("base goal end: {e}");
        }
    }

    /// The setpoint of the step at `now_ns`, with the robot frame at `pose`
    /// in the goal's frame: zero once the goal has an ending.
    fn setpoint(&mut self, pose: PlanarPose, now_ns: i64, dt: f64) -> PlanarTwist {
        if self.ending.is_some() {
            return PlanarTwist::ZERO;
        }
        match &mut self.motion {
            GoalMotion::Drive {
                velocity,
                hold_duration_ns,
                reached_ns,
                ..
            } => {
                if reached_ns.is_some_and(|reached| now_ns >= reached + *hold_duration_ns) {
                    self.ending = Some(Ending::Completed);
                    return PlanarTwist::ZERO;
                }
                *velocity
            }
            GoalMotion::Move { controller, .. } => controller.command(pose, dt),
        }
    }

    /// After the step at `now_ns` left the limiter's command at `command`: a
    /// drive notes the step at which the command reached its velocity, and
    /// the step at which it came to 0 after the hold.
    fn note_command(&mut self, command: PlanarTwist, now_ns: i64) {
        let GoalMotion::Drive {
            velocity,
            reached_ns,
            stopped_ns,
            ..
        } = &mut self.motion
        else {
            return;
        };
        match self.ending {
            None if reached_ns.is_none() && reached(command, *velocity) => {
                *reached_ns = Some(now_ns);
            }
            Some(Ending::Completed) if stopped_ns.is_none() && command == PlanarTwist::ZERO => {
                *stopped_ns = Some(now_ns);
            }
            _ => {}
        }
    }

    /// After a step at `now_ns` that found the base moving, with the robot
    /// frame at `pose` in the goal's frame and the base measured at
    /// `measured`: the `timeout` of a goal past its settle. A move's settle
    /// ends `settle_timeout_s` after its planned end while it runs; a
    /// completed drive's, that long after its command came to 0.
    fn settle_failure(
        &self,
        pose: PlanarPose,
        measured: PlanarTwist,
        now_ns: i64,
    ) -> Option<Failure> {
        let message = match &self.motion {
            GoalMotion::Move {
                controller,
                settle_end_ns,
                ..
            } if self.ending.is_none() && now_ns >= *settle_end_ns => {
                let (distance, yaw) = controller.remaining(pose);
                format!(
                    "the base stands {distance:.3} m and {yaw:.3} rad from the target \
                     at the end of its settle timeout"
                )
            }
            GoalMotion::Drive {
                settle_timeout_ns,
                stopped_ns: Some(stopped_ns),
                ..
            } if matches!(self.ending, Some(Ending::Completed))
                && now_ns >= stopped_ns + settle_timeout_ns =>
            {
                format!(
                    "the base still moves at x {:.3} m/s and yaw {:.3} rad/s \
                     {:.3} s after the command reached 0",
                    measured.linear_x(),
                    measured.angular_z(),
                    seconds(*settle_timeout_ns)
                )
            }
            _ => return None,
        };
        Some(Failure::new(FailureKind::Timeout, message))
    }

    /// How the goal ends after the step at `now_ns`, with the robot frame at
    /// `pose` in the goal's frame, or `None` while it runs on: its ending
    /// once the base rests, and a move's completion once, at or after its
    /// planned end, the base rests within both tolerances.
    fn ready_ending(&self, pose: PlanarPose, now_ns: i64, at_rest: bool) -> Option<Ending> {
        if !at_rest {
            return None;
        }
        if let Some(ending) = &self.ending {
            return Some(ending.clone());
        }
        match &self.motion {
            GoalMotion::Move {
                controller,
                planned_end_ns,
                ..
            } if now_ns >= *planned_end_ns && controller.within_tolerances(pose) => {
                Some(Ending::Completed)
            }
            _ => None,
        }
    }

    async fn publish_feedback(&self, pose: PlanarPose) {
        let GoalMotion::Move {
            ctx, controller, ..
        } = &self.motion
        else {
            return;
        };
        let (distance, yaw) = controller.remaining(pose);
        if let Err(e) = ctx.publish_feedback(distance, yaw).await {
            warn!("move_base feedback: {e}");
        }
    }
}

/// Every publisher the loop owns.
pub struct Publishers {
    wheels: TopicPublisher,
    base_states: TopicPublisher,
    drive_states: TopicPublisher,
    transforms: TopicPublisher,
    /// The robot's tree, which takes each message of the base's transforms as
    /// it is published, under `measured_by`.
    tree: Arc<RobotTree>,
    /// This base as a publisher of the robot's tree: its core node and its
    /// instance id.
    measured_by: TreePublisher,
}

impl Publishers {
    pub async fn declare(runner: &NodeRunner, tree: Arc<RobotTree>) -> Result<Self> {
        let processor = runner.processor();
        Ok(Self {
            wheels: joint_setpoints::declare_publisher(runner).await?,
            base_states: base_states::declare_publisher(runner).await?,
            drive_states: drive_states::declare_publisher(runner).await?,
            transforms: transforms::declare_publisher(runner).await?,
            tree,
            measured_by: TreePublisher {
                core_node: processor.bound_core_node().to_owned(),
                instance_id: processor.bound_instance_id().to_owned(),
            },
        })
    }
}

/// Publish `message` on `publisher`, the log naming `topic` and why when the
/// message is not built or not sent.
async fn publish_or_warn(publisher: &TopicPublisher, topic: &str, message: Result<Payload>) {
    let sent = match message {
        Ok(message) => publisher.publish(message).await,
        Err(e) => Err(e),
    };
    if let Err(e) = sent {
        warn!("{topic}: {e}");
    }
}

/// The first whole multiple of `period` at or after `now`.
fn grid_point_at_or_after(now: i64, period: i64) -> i64 {
    now.div_euclid(period) * period
        + if now.rem_euclid(period) == 0 {
            0
        } else {
            period
        }
}

/// One odometry epoch: the wheel follower it measures, its
/// odometry_start_timestamp on the wall clock, the last measurement the
/// odometry took, the wheel setpoint last sent, and whether the base's clock
/// went back to before its last measurement, which closes it: a closed
/// epoch takes no measurement and its wheels report nothing, while the wheel
/// setpoints run on from the one it sent last until the next measurement
/// starts a new epoch.
struct Epoch {
    follower: String,
    odometry_start: SystemTime,
    last: WheelState,
    sent: Option<WheelSetpoint>,
    closed: bool,
}

impl Epoch {
    /// Whether `state` continues the epoch: an open one, from its wheel
    /// follower, stamped at or after its last measurement, each wheel within
    /// `reach_rad` of the time between them (s) of its last angle.
    fn continues_with(&self, state: &WheelState, reach_rad: impl Fn(f64) -> f64) -> bool {
        let reach = reach_rad(seconds(state.timestamp_ns - self.last.timestamp_ns));
        !self.closed
            && state.follower == self.follower
            && state.timestamp_ns >= self.last.timestamp_ns
            && state
                .positions
                .iter()
                .zip(self.last.positions)
                .all(|(now, last)| (now - last).abs() <= reach)
    }
}

/// One measurement of the base, as base_states and drive_states carry it:
/// when the wheels measured it (ns), its epoch's odometry_start_timestamp,
/// where the odometry places the robot frame, and the velocity it measures
/// (m/s, rad/s).
#[derive(Clone, Copy)]
struct StateReport {
    timestamp_ns: i64,
    odometry_start: SystemTime,
    pose: PlanarPose,
    linear: f64,
    angular: f64,
}

/// The report last published, and when on the node's clock (ns).
#[derive(Clone, Copy)]
struct Published {
    report: StateReport,
    at_ns: i64,
}

/// The control loop's state.
pub struct Base {
    config: Arc<BaseConfig>,
    status: BaseStatus,
    limits: Limiter,
    follower: Follower,
    odometry: Odometry,
    /// The odometry epoch: `None` before the first measurement, and once the
    /// base's clock went back to before the epoch's last measurement.
    epoch: Option<Epoch>,
    /// Whether the last step found the wheels silent, with no measurement
    /// taken since.
    silent: bool,
    goal: Option<RunningGoal>,
    /// The base's move slot, which an admitted goal holds on its way to the
    /// loop and while it runs.
    slot: MoveSlot,
    /// The last step the loop ran, on the node's clock (ns).
    last_step_ns: Option<i64>,
    /// The newest measurement's report, while the node's clock reads earlier
    /// than its stamp.
    due_report: Option<StateReport>,
    published: Option<Published>,
    rest: RestWindow,
    publishers: Publishers,
    /// The odometry_start_timestamp of the newest epoch.
    last_odometry_start: Option<SystemTime>,
    /// When the node started, on the wall clock: the start of the wheel
    /// joints' transforms.
    started: SystemTime,
}

impl Base {
    pub fn new(
        config: Arc<BaseConfig>,
        status: BaseStatus,
        slot: MoveSlot,
        publishers: Publishers,
        started: SystemTime,
    ) -> Self {
        Self {
            limits: config.limiter.start(),
            follower: Follower::default(),
            odometry: config.odometry.clone(),
            config,
            status,
            epoch: None,
            silent: false,
            goal: None,
            slot,
            last_step_ns: None,
            due_report: None,
            published: None,
            rest: RestWindow::default(),
            publishers,
            last_odometry_start: None,
            started,
        }
    }

    /// The robot frame in the odometry frame, its heading wrapped.
    fn pose(&self) -> PlanarPose {
        PlanarPose::new(
            self.odometry.x(),
            self.odometry.y(),
            normalize_angle(self.odometry.heading()),
        )
        .expect("the odometry takes finite updates alone")
    }

    /// Where the odometry places the robot frame in odom, at the time of the
    /// newest wheel measurement.
    fn located(&self) -> Located {
        let measured_ns = self
            .epoch
            .as_ref()
            .map_or(0, |epoch| epoch.last.timestamp_ns);
        Located {
            pose: self.pose(),
            timestamp: wire_from_ns(measured_ns),
        }
    }

    /// The velocity the odometry measures.
    fn measured_velocity(&self) -> PlanarTwist {
        PlanarTwist::new(self.odometry.linear(), 0.0, self.odometry.angular())
            .expect("the odometry takes finite updates alone")
    }

    /// Whether the base rests at `now_ns`: the wheels report, the limiter
    /// commands nothing and base_state's rest holds.
    fn at_rest(&self, now_ns: i64) -> bool {
        self.wheels_report_at(now_ns)
            && self.limits.last_velocity() == PlanarTwist::ZERO
            && self.rest.rests(self.config.axes.rest_bounds())
    }

    /// Whether the wheels report, as of `now_ns`: base_state's motion sensing
    /// is not silent. The follower holds the base while it is.
    fn set_reporting(&mut self, reporting: bool, now_ns: i64) {
        self.status.set_wheels_report(reporting);
        match reporting {
            true => self.follower.silence_ends(now_ns),
            false => self.follower.silence_starts(now_ns),
        }
    }

    /// The odometry_start_timestamp of a new epoch: the wall clock now, or
    /// 1 ns past the last epoch's when the wall clock reads no later, so it
    /// is a value base_states has not carried before.
    fn new_odometry_start(&mut self) -> SystemTime {
        let now = SystemTime::now();
        let start = match self.last_odometry_start {
            Some(last) if now <= last => last + Duration::from_nanos(1),
            _ => now,
        };
        self.last_odometry_start = Some(start);
        start
    }

    /// Whether the wheels' newest measurement, by its stamp, is at most
    /// [`SILENT_AFTER_NS`] old at `now_ns`; one stamped after `now_ns`
    /// reports.
    fn wheels_report_at(&self, now_ns: i64) -> bool {
        self.epoch.as_ref().is_some_and(|epoch| {
            !epoch.closed && now_ns - epoch.last.timestamp_ns <= SILENT_AFTER_NS
        })
    }

    /// End the goal in flight on `outcome` at once, wherever the base
    /// stands. A goal that ends while the limiter's command is not 0 leaves
    /// the base coming to rest from it, which refuses a goal until a step
    /// finds the command at 0.
    async fn end_goal_now(&mut self, outcome: Outcome) {
        let Some(goal) = self.goal.take() else { return };
        self.status
            .set_resting(self.limits.last_velocity() != PlanarTwist::ZERO);
        goal.finish(outcome, self.located()).await;
    }

    /// Fail the goal in flight with `failure` at once, and bring the base to
    /// rest from it: the limiter takes a zero setpoint past drive_link's
    /// 0.5 s at `at_ns`.
    async fn fail_goal_now(&mut self, failure: Failure, at_ns: i64) {
        if self.goal.is_none() {
            return;
        }
        self.end_goal_now(Outcome::Failed(failure)).await;
        self.limits
            .take_setpoint(PlanarTwist::ZERO, expired_at(at_ns));
    }

    pub async fn handle(&mut self, event: Event) {
        match event {
            Event::Wheels(state) => self.take_wheels(state).await,
            Event::Setpoint {
                velocity,
                received_ns,
            } => self.take_setpoint(velocity, received_ns),
            Event::Goal(goal) => self.start_goal(goal).await,
            Event::Stop { reason, reply } => {
                let _ = reply.send(self.stop(&reason));
            }
        }
    }

    /// Take a wheel measurement: into the odometry over the time since the
    /// last one when it continues the epoch, else as the start of a new
    /// epoch. A measurement the odometry refuses, stamped within a
    /// microsecond after the last one, is dropped. The wheel joints'
    /// transforms at the measurement go out, the wheels then report when the
    /// measurement is at most [`SILENT_AFTER_NS`] old at the last step's
    /// time, and its report falls due.
    async fn take_wheels(&mut self, state: WheelState) {
        let continues = self.epoch.as_ref().is_some_and(|epoch| {
            epoch.continues_with(&state, |dt_s| self.config.wheel_reach_rad(dt_s))
        });
        if !continues {
            self.start_epoch(state.clone()).await;
        } else if !self.continue_epoch(&state) {
            return;
        }
        let wheels = self.config.joint_transforms.message_at(
            wire_from_ns(state.timestamp_ns),
            state.positions,
            self.started,
        );
        self.publish_transforms(wheels).await;
        let judged_ns = self.last_step_ns.unwrap_or(state.timestamp_ns);
        if self.wheels_report_at(judged_ns) {
            self.set_reporting(true, judged_ns);
        }
        let odometry_start = self
            .epoch
            .as_ref()
            .map(|epoch| epoch.odometry_start)
            .expect("a measurement the odometry took belongs to an epoch");
        self.due_report = Some(StateReport {
            timestamp_ns: state.timestamp_ns,
            odometry_start,
            pose: self.pose(),
            linear: self.odometry.linear(),
            angular: self.odometry.angular(),
        });
    }

    /// Start a new odometry epoch on `state`: the odometry starts at the
    /// origin, anchored on its angles, the wheel setpoints anchor on them,
    /// the wheels hold where they stand, a setpoint of rate 0 at the measured
    /// angles sent at once, and base_states' rest and reports start over.
    /// When an open epoch ran, the wheels' positions started over, and a goal
    /// in flight fails with `resource_unavailable`.
    async fn start_epoch(&mut self, state: WheelState) {
        if self.epoch.as_ref().is_some_and(|epoch| !epoch.closed) {
            warn!(
                "the wheels of {} started over at {} ns: the odometry starts over",
                state.follower, state.timestamp_ns
            );
            let failure = Failure::new(FailureKind::ResourceUnavailable, ODOMETRY_STARTED_OVER);
            self.fail_goal_now(failure, state.timestamp_ns).await;
        }
        let [left, right] = state.positions;
        self.odometry = self.config.odometry.clone();
        self.odometry
            .anchor_wheels(left, right)
            .expect("the wheel stream drops every measurement whose angles are not finite");
        let odometry_start = self.new_odometry_start();
        self.status.set_odometry_start(odometry_start);
        let held_ns = clock::now_ns().unwrap_or(state.timestamp_ns);
        self.epoch = Some(Epoch {
            follower: state.follower.clone(),
            odometry_start,
            last: state,
            sent: None,
            closed: false,
        });
        self.silent = false;
        self.due_report = None;
        self.published = None;
        self.rest.clear();
        // A follower can still run the last setpoint of a node that ended
        // before this one started: the hold stops it before any goal.
        self.send_wheel_setpoint(PlanarTwist::ZERO, true, held_ns)
            .await;
    }

    /// Integrate `state` into the odometry over the time since the epoch's
    /// last measurement, a silence included; after a silence, the wheel
    /// setpoints anchor on its angles. False, taking nothing, when the
    /// odometry refuses it.
    fn continue_epoch(&mut self, state: &WheelState) -> bool {
        let Some(epoch) = self.epoch.as_mut() else {
            return false;
        };
        let [left, right] = state.positions;
        let dt = seconds(state.timestamp_ns - epoch.last.timestamp_ns);
        if !self.odometry.update_from_positions(left, right, dt) {
            return false;
        }
        epoch.last = state.clone();
        if self.silent {
            epoch.sent = None;
            self.silent = false;
        }
        true
    }

    /// A setpoint of the leader, received at `received_ns`, counts for the
    /// follower's rules and the goals' admission, and becomes the limiter's
    /// setpoint when the base follows it. A setpoint received after the time
    /// the clock reads now, as one in flight while the clock went back, takes
    /// that time, as [`Follower::clock_went_back`] gives the newest one.
    fn take_setpoint(&mut self, velocity: PlanarTwist, received_ns: i64) {
        let received_ns = clock::now_ns().map_or(received_ns, |now_ns| received_ns.min(now_ns));
        self.status.note_setpoint(received_ns);
        if self.follower.setpoint(received_ns, self.slot.is_claimed()) {
            self.limits.take_setpoint(velocity, received_ns);
        }
    }

    /// Start an accepted goal in the robot frame where the base stands, or
    /// end it at once: cancelled for a stop that came while it was on its
    /// way to the loop, and failed when the base's clock reads earlier than
    /// its admission or the odometry it was placed by has started over.
    async fn start_goal(&mut self, accepted: AcceptedGoal) {
        let goal = RunningGoal::new(accepted, self.pose(), self.limits.last_velocity());
        let admitted_ns = goal.claim.admitted_ns();
        let clock_went_back = clock::now_ns().is_some_and(|now_ns| now_ns < admitted_ns);
        let odometry_start = self.epoch.as_ref().map(|epoch| epoch.odometry_start);
        let refusal = refusal_at_start(
            self.slot.take_stop(),
            clock_went_back,
            goal.claim.odometry_start() != odometry_start,
        );
        match refusal {
            None => self.goal = Some(goal),
            Some(outcome) => goal.finish(outcome, self.located()).await,
        }
    }

    /// The robot stop: a goal that runs, or that completed and still comes
    /// to rest, comes to rest and ends cancelled for the stop; a goal
    /// admitted and on its way to the loop ends cancelled for it as it
    /// arrives. The follower holds the base against the leader, and the
    /// limiter brings the base to rest. Answers the base's name when the
    /// stop changed how a goal ends or the leader streamed.
    fn stop(&mut self, reason: &str) -> Vec<String> {
        let reason = stop_reason(reason);
        info!("robot stop: {reason}");
        let ended_goal = match &mut self.goal {
            Some(goal) => {
                // A cancel its caller sent before the stop keeps its reason.
                if let Some(cancel) = goal.cancel_reason() {
                    goal.cancel(cancel);
                }
                goal.cancel(reason)
            }
            None => self.slot.stop_unstarted(&reason),
        };
        let stream_live = match clock::now_ns() {
            Some(now_ns) => {
                self.limits
                    .take_setpoint(PlanarTwist::ZERO, expired_at(now_ns));
                self.follower.stop(now_ns)
            }
            None => false,
        };
        if !(ended_goal || stream_live) {
            return Vec::new();
        }
        vec![BASE_NAME.to_string()]
    }

    /// Run every step due on the node's clock, then publish the due report
    /// once the clock reads its stamp. A clock that reads earlier than the
    /// last step starts the base's timing over first.
    pub async fn advance(&mut self) {
        let Some(now_ns) = clock::now_ns() else {
            return;
        };
        if self.last_step_ns.is_some_and(|last_ns| now_ns < last_ns) {
            self.start_timing_over(now_ns).await;
        }
        let step_ns = self.config.step_ns;
        let mut next_ns = self.last_step_ns.map_or_else(
            || grid_point_at_or_after(now_ns, step_ns),
            |last_ns| last_ns + step_ns,
        );
        while next_ns <= now_ns {
            self.step(next_ns).await;
            next_ns += step_ns;
        }
        self.publish_reports(now_ns).await;
    }

    /// The base's clock reads `now_ns`, earlier than the last step: the
    /// base's timing starts over on that time. A goal in flight fails with
    /// `resource_unavailable` and the base comes to rest through the
    /// limiter, the steps fall on the grid of the new time, the leader's
    /// stream and the robot stop's hold last on from `now_ns`, and an epoch
    /// whose last measurement lies after `now_ns` closes and a report stamped
    /// after it ends, so the wheels start a new epoch on their next
    /// measurement.
    async fn start_timing_over(&mut self, now_ns: i64) {
        warn!("the base's clock went back to {now_ns} ns: its timing starts over");
        let failure = Failure::new(FailureKind::ResourceUnavailable, CLOCK_WENT_BACK);
        self.fail_goal_now(failure, now_ns).await;
        self.limits
            .take_setpoint(PlanarTwist::ZERO, expired_at(now_ns));
        self.last_step_ns = None;
        self.follower.clock_went_back(now_ns);
        self.status.clock_went_back(now_ns);
        if let Some(epoch) = self
            .epoch
            .as_mut()
            .filter(|epoch| epoch.last.timestamp_ns > now_ns)
        {
            epoch.closed = true;
            self.published = None;
            self.rest.clear();
            self.set_reporting(false, now_ns);
        }
        if self
            .due_report
            .is_some_and(|report| report.timestamp_ns > now_ns)
        {
            self.due_report = None;
        }
    }

    /// One step of the limiter at `now_ns`, the wheels judged at that time.
    async fn step(&mut self, now_ns: i64) {
        self.last_step_ns = Some(now_ns);
        let reporting = self.wheels_report_at(now_ns);
        self.set_reporting(reporting, now_ns);
        if !reporting {
            self.silent = true;
            let failure = Failure::new(
                FailureKind::ResourceUnavailable,
                "the wheels stopped reporting",
            );
            self.fail_goal_now(failure, now_ns).await;
            // A base on silent wheels takes no setpoint, the leader's
            // included.
            self.limits
                .take_setpoint(PlanarTwist::ZERO, expired_at(now_ns));
        }
        let dt = seconds(self.config.step_ns);
        let pose = self.pose();
        if let Some(goal) = &mut self.goal {
            if let Some(reason) = goal.cancel_reason() {
                goal.cancel(reason);
            }
            let setpoint = goal.setpoint(seen_from_start(pose, goal.start), now_ns, dt);
            self.limits.take_setpoint(setpoint, now_ns);
        }
        let output = self.limits.tick(now_ns);
        let command = self.limits.last_velocity();
        if command == PlanarTwist::ZERO {
            self.status.set_resting(false);
        }
        if let Some(velocity) = output {
            self.send_wheel_setpoint(velocity, reporting, now_ns).await;
        }
        self.publish_reports(now_ns).await;
        let at_rest = self.at_rest(now_ns);
        let measured = self.measured_velocity();
        let Some(goal) = &mut self.goal else { return };
        goal.note_command(command, now_ns);
        let pose = seen_from_start(pose, goal.start);
        // Completion is decided first: a base that rests within its
        // tolerances when its settle ends completes.
        if let Some(ending) = goal.ready_ending(pose, now_ns, at_rest) {
            return self.end_goal_now(Outcome::Ending(ending)).await;
        }
        if let Some(ending) = goal.unsettled_cancel(now_ns) {
            warn!(
                "the base still moves at x {:.3} m/s and yaw {:.3} rad/s \
                 {:.3} s after its goal was cancelled: the goal ends",
                measured.linear_x(),
                measured.angular_z(),
                seconds(goal.motion.settle_timeout_ns())
            );
            return self.end_goal_now(Outcome::Ending(ending)).await;
        }
        if let Some(failure) = goal.settle_failure(pose, measured, now_ns) {
            return self.fail_goal_now(failure, now_ns).await;
        }
        goal.publish_feedback(pose).await;
    }

    /// Stream the wheel setpoint of `velocity` over one step
    /// ([`WheelSetpoint::next`]) on the measured angles while the wheels are
    /// `reporting`; while they are silent, on the angles the setpoint last
    /// sent in the epoch put them at, so each wheel runs on from it.
    async fn send_wheel_setpoint(&mut self, velocity: PlanarTwist, reporting: bool, now_ns: i64) {
        let rates = self.config.wheels.wheel_rates(velocity);
        let dt = seconds(self.config.step_ns);
        let bound = self.config.anchor_bound_rad();
        let Some(epoch) = self.epoch.as_mut() else {
            return;
        };
        let measured = match epoch.sent {
            Some(sent) if !reporting => sent.positions,
            _ => epoch.last.positions,
        };
        let setpoint = WheelSetpoint::next(epoch.sent, measured, rates, bound, dt);
        epoch.sent = Some(setpoint);
        let message = joint_setpoints::build_message(
            wire_from_ns(now_ns),
            setpoint.positions,
            setpoint.rates,
            Vec::new(),
        );
        publish_or_warn(&self.publishers.wheels, "wheel joint_setpoints", message).await;
    }

    /// Publish `message` on transforms, the log naming why a set of
    /// transforms makes no message.
    async fn publish_transforms(
        &self,
        message: std::result::Result<MovingMessage, TransformsError>,
    ) {
        let message = match message {
            Ok(message) => message,
            Err(e) => return warn!("transforms: {e}"),
        };
        let publishers = &self.publishers;
        if let Err(e) = publishers
            .tree
            .take_moving(&publishers.measured_by, &message)
        {
            warn!("the robot's tree refused the transforms: {e}");
        }
        let payload = transforms::payload(message);
        publish_or_warn(&publishers.transforms, "transforms", payload).await;
    }

    /// At `now_ns`, while the wheels report: publish the due report once the
    /// clock reads its stamp; with no report due, publish the last one again
    /// when waiting a step more would leave [`REPORT_PERIOD_NS`] without one.
    async fn publish_reports(&mut self, now_ns: i64) {
        if !self.wheels_report_at(now_ns) {
            return;
        }
        if let Some(report) = self
            .due_report
            .take_if(|report| report.timestamp_ns <= now_ns)
        {
            self.rest
                .note(report.timestamp_ns, [report.linear, 0.0, report.angular]);
            return self.publish(report, now_ns).await;
        }
        let Some(Published { report, at_ns }) =
            self.published.filter(|_| self.due_report.is_none())
        else {
            return;
        };
        if now_ns - at_ns + self.config.step_ns > REPORT_PERIOD_NS {
            self.publish(report, now_ns).await;
        }
    }

    /// Publish `report` at `now_ns` on base_states, on drive_states to the
    /// leader, with its follow_state, and on transforms as the transform odom
    /// to base_footprint. The mode is standing while the base rests by
    /// base_state's rule as of the report and moving otherwise: the node
    /// senses no state of the base's drives. The base carries no limb, so its
    /// parts stand within its footprint.
    async fn publish(&mut self, report: StateReport, now_ns: i64) {
        self.published = Some(Published {
            report,
            at_ns: now_ns,
        });
        let StateReport {
            timestamp_ns,
            odometry_start,
            pose,
            linear,
            angular,
        } = report;
        let timestamp = wire_from_ns(timestamp_ns);
        let (position, orientation) = ([pose.x(), pose.y(), 0.0], quaternion_of_yaw(pose.yaw()));
        let (linear_velocity, angular_velocity) = ([linear, 0.0, 0.0], [0.0, 0.0, angular]);
        let mode = match self.rest.rests(self.config.axes.rest_bounds()) {
            true => "standing",
            false => "moving",
        };
        let base = base_states::build_message(
            timestamp,
            odometry_start,
            transforms::ODOMETRY_FRAME.to_owned(),
            transforms::ROBOT_FRAME.to_owned(),
            base_states::MessagePose {
                position,
                orientation,
            },
            base_states::MessageTwist {
                linear_velocity,
                angular_velocity,
            },
            mode.to_string(),
            true,
        );
        publish_or_warn(&self.publishers.base_states, "base_states", base).await;
        let follow_state = self.follower.state(now_ns, self.slot.is_claimed());
        let drive = drive_states::build_message(
            timestamp,
            odometry_start,
            transforms::ODOMETRY_FRAME.to_owned(),
            transforms::ROBOT_FRAME.to_owned(),
            drive_states::MessagePose {
                position,
                orientation,
            },
            drive_states::MessageTwist {
                linear_velocity,
                angular_velocity,
            },
            mode.to_string(),
            true,
            follow_state.as_str().to_string(),
        );
        publish_or_warn(&self.publishers.drive_states, "drive_states", drive).await;
        let odometry = transforms::odometry_message(timestamp, pose, odometry_start);
        self.publish_transforms(odometry).await;
    }
}

/// Run the loop until `cancelled` resolves or every event sender is gone.
/// The poll wakes the loop once per step of wall time, so it steps while no
/// event arrives; every time the loop reads is the node's clock.
pub async fn run(
    mut base: Base,
    mut events: mpsc::Receiver<Event>,
    cancelled: impl Future<Output = ()>,
) {
    let mut cancelled = std::pin::pin!(cancelled);
    let step = Duration::from_nanos(base.config.step_ns.unsigned_abs());
    let mut poll = tokio::time::interval(step);
    poll.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            biased;
            () = &mut cancelled => return,
            event = events.recv() => match event {
                Some(event) => base.handle(event).await,
                None => return,
            },
            _ = poll.tick() => {}
        }
        base.advance().await;
    }
}

#[cfg(test)]
mod tests;
