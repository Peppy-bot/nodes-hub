//! The control loop: it owns the goal in flight and steps it on the node's
//! clock, the simulation clock in a simulation, at whole multiples of the
//! controller's period; a step the clock passed while the loop waited is
//! skipped. Events, in arrival order: an accepted goal replaces the goal in
//! flight, which ends cancelled by its caller's cancel when one came, and
//! otherwise cancelled with its result where the robot frame stood at the
//! timestamp of the newest drive_states when the new goal was accepted, and
//! centers the grid on the base; a clear_grid empties the grid.
//!
//! Each step gathers every ending of the goal in flight, from every
//! drive_states that arrived since the goal's previous step, so a report that
//! a later drive_states replaced before the step ends the goal too, and ends
//! the goal by the earliest in [`Cause`]'s order ([`first_ending`]),
//! base_navigation's: the caller's cancel, cancelled with the result where
//! the newest drive_states placed the robot frame; a follow_state other than
//! following, robot_stop and follower_goal cancelling it with the result
//! where drive_states first reported it, base_mode `attempt_failed`,
//! base_silence `resource_unavailable`; the odometry starting over,
//! `resource_unavailable`; more than 0.5 s with no drive_states,
//! `resource_unavailable`; a silent sensor, `resource_unavailable`;
//! parts_within_footprint false, `attempt_failed`; the passing of the goal's
//! timeout_s, `timeout`; the goal's completion at its end, its result where
//! the newest drive_states placed the robot frame; every other failure, the
//! grid's and the goal's own step's. A goal with no ending runs on: its
//! feedback goes out, its command passes the stop and streams as the step's
//! setpoint, and the plans it asks for start on the planner's task.
//!
//! Each ending ends its goal at once. When a goal ends, completed, cancelled
//! or failed, while at most 0.5 s have passed since the navigator's newest
//! setpoint and the newest drive_states reports following, the loop streams
//! its release burst ([`SetpointStream`]): zero velocity for 0.25 s and at
//! least 3 setpoints, then nothing, as Nav2's controller and behaviors stop
//! the robot at each end (nav2_behaviors plugins/spin.cpp:119-121). The burst
//! streams only while those two hold, so a drive_states that reports a hold
//! ends it; the first setpoint of the next goal ends it too; and a goal that
//! ends during it starts it again from that ending. The grid rolls with the
//! base while a goal runs or the burst streams, and stays where it stands once
//! neither does.

use std::sync::Arc;
use std::time::{Instant, SystemTime};

use control_core::drive_link::{FollowState, LEADER_STREAM};
use control_core::node_clock::wire_from_ns;
use control_core::planar_pose::PlanarPose;
use control_core::robot_tree;
use control_core::sync::lock_ignoring_poison;
use control_core::time::seconds;
use control_core::twist::PlanarTwist;
use frame_tree::arithmetic::quaternion_of_yaw;
use navigation_core::grid::CostGrid;
use peppygen::exposed_actions::base_navigation::{
    back_up, drive_on_heading, navigate_to_pose, spin,
};
use peppygen::paired_topics::base::drive_setpoints;
use peppylib::TopicPublisher;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, warn};

use crate::admission::{Admitted, Shared};
use crate::base::Base;
use crate::clock;
use crate::drive::{Navigation, NavigationInputs, NavigationRequest, NavigationStep};
use crate::failure::{Ended, Failure, FailureKind, NavigationKind, of_navigation};
use crate::follow::ODOMETRY_FRAME;
use crate::grid::Rolling;
use crate::link::{DriveState, Received, Reports};
use crate::moves::{Move, MoveFeedback, MoveInputs, MoveStep};
use crate::planning::Planning;
use crate::recovery::{Direction, SpinRequest, StraightRequest};

/// How long (ns) the release burst streams zero velocity at least:
/// base_navigation's 0.25 s.
const BURST_NS: i64 = 250_000_000;

/// How many zero setpoints the release burst streams at least, so that one
/// lost setpoint cannot lose the stop.
const BURST_SETPOINTS: u32 = 3;

/// A goal the action handlers admitted, with its context.
pub enum AcceptedGoal {
    Navigate {
        request: NavigationRequest,
        ctx: Box<navigate_to_pose::GoalContext>,
        admitted: Admitted,
    },
    Spin {
        request: SpinRequest,
        ctx: Box<spin::GoalContext>,
        admitted: Admitted,
    },
    DriveOnHeading {
        request: StraightRequest,
        ctx: Box<drive_on_heading::GoalContext>,
        admitted: Admitted,
    },
    BackUp {
        request: StraightRequest,
        ctx: Box<back_up::GoalContext>,
        admitted: Admitted,
    },
}

/// What the loop receives, in arrival order.
pub enum Event {
    Goal(Box<AcceptedGoal>),
    /// Empty the grid, as clear_grid empties it.
    ClearGrid {
        reply: oneshot::Sender<()>,
    },
}

/// Where the robot frame stood in odom, and when, for a result.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Located {
    pub pose: PlanarPose,
    pub timestamp: SystemTime,
}

impl Located {
    /// Where `state` places the robot frame, at its timestamp.
    fn of(state: &DriveState) -> Self {
        Self {
            pose: state.pose,
            timestamp: state.timestamp,
        }
    }

    /// The position and the orientation of the record of this pose.
    fn arrays(self) -> ([f64; 3], [f64; 4]) {
        (
            [self.pose.x(), self.pose.y(), 0.0],
            quaternion_of_yaw(self.pose.yaw()),
        )
    }
}

/// How a goal ends, with a failure of kinds `K`.
#[derive(Debug, PartialEq)]
enum Finish<K> {
    Completed(Located),
    Cancelled(String, Located),
    Failed(robot_tree::Failure<K>),
}

impl Finish<FailureKind> {
    /// The same ending of a navigation.
    fn of_navigation(self) -> Finish<NavigationKind> {
        match self {
            Self::Completed(located) => Finish::Completed(located),
            Self::Cancelled(reason, located) => Finish::Cancelled(reason, located),
            Self::Failed(failure) => Finish::Failed(of_navigation(failure)),
        }
    }
}

fn failed(kind: FailureKind, message: impl Into<String>) -> Finish<FailureKind> {
    Finish::Failed(Failure::new(kind, message))
}

/// The reason of a goal that a new goal replaced.
const REPLACED: &str = "a new goal replaced it";

/// How a new goal's acceptance ends the goal it replaces, `at` being the
/// newest drive_states when the new goal was accepted: cancelled, its result
/// where the robot frame stood at that drive_states' timestamp.
fn replaced(at: &Received) -> Finish<FailureKind> {
    Finish::Cancelled(REPLACED.to_string(), Located::of(&at.state))
}

/// What ends a goal at a step, declared in the order a goal takes them when
/// one step gives it more than one ending: the first variant first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Cause {
    /// The goal's caller cancelled it.
    CallerCancel,
    /// drive_states reports a follow_state other than following.
    Hold,
    /// drive_states reports a new odometry_start_timestamp.
    OdometryStartedOver,
    /// More than 0.5 s passed with no drive_states.
    DriveStatesLost,
    /// No reading of a sensor the grid reads went into the grid within the
    /// source timeout.
    SilentSensor,
    /// drive_states reports parts_within_footprint false.
    LimbOutsideFootprint,
    /// The goal's timeout_s passed before its end.
    Timeout,
    /// The goal reached its end: a navigation its goal, a move its whole
    /// turn or distance.
    Reached,
    /// Every other failure: the grid's, and the goal's own step's.
    OtherFailure,
}

/// One ending of a goal at a step: its cause, and how it ends the goal, with
/// a failure of kinds `K`.
#[derive(Debug, PartialEq)]
struct Ending<K> {
    cause: Cause,
    finish: Finish<K>,
}

impl<K: From<FailureKind>> Ending<K> {
    /// The ending of a goal's own step, the robot frame `located`: completed
    /// at the goal's end, failed with `timeout` once its timeout_s passed, or
    /// one of the other failures.
    fn of_own(ended: Ended<K>, located: Located) -> Self {
        match ended {
            Ended::Completed => Self {
                cause: Cause::Reached,
                finish: Finish::Completed(located),
            },
            Ended::TimedOut(why) => Self {
                cause: Cause::Timeout,
                finish: Finish::Failed(robot_tree::Failure::new(FailureKind::Timeout.into(), why)),
            },
            Ended::Failed(failure) => Self {
                cause: Cause::OtherFailure,
                finish: Finish::Failed(failure),
            },
        }
    }
}

impl From<Ending<FailureKind>> for Ending<NavigationKind> {
    fn from(ending: Ending<FailureKind>) -> Self {
        Self {
            cause: ending.cause,
            finish: ending.finish.of_navigation(),
        }
    }
}

/// The ending a step takes of `endings`, every ending it gives the goal: the
/// earliest in [`Cause`]'s order.
fn first_ending<K>(endings: impl IntoIterator<Item = Ending<K>>) -> Option<Ending<K>> {
    endings.into_iter().min_by_key(|ending| ending.cause)
}

/// Every ending that what every goal shares gives the goal in flight at
/// `now_ns`, the newest drive_states being `at` and `reports` the reports of
/// those the goal had not read: the goal ran on the odometry started at
/// `odometry_start`, its caller's cancel gave `cancel`, and `silent` says
/// what the goal waits for of the sensors that gave no reading within the
/// source timeout, none while every sensor gave one.
fn shared_endings(
    at: &Received,
    reports: &Reports,
    now_ns: i64,
    odometry_start: SystemTime,
    cancel: Option<String>,
    silent: Option<String>,
) -> Vec<Ending<FailureKind>> {
    let state = at.state;
    let ending = |cause, finish| Ending { cause, finish };
    let limb = reports.limb_outside.unwrap_or(*at).state;
    [
        hold_ending(&reports.hold.unwrap_or(*at)),
        (state.odometry_start != odometry_start).then(|| {
            ending(
                Cause::OdometryStartedOver,
                failed(
                    FailureKind::ResourceUnavailable,
                    "the base's odometry started over: send the goal again",
                ),
            )
        }),
        at.silent_at(now_ns).then(|| {
            ending(
                Cause::DriveStatesLost,
                failed(
                    FailureKind::ResourceUnavailable,
                    format!(
                        "the base sent no drive_states for {} s",
                        seconds(now_ns - at.arrived_ns)
                    ),
                ),
            )
        }),
        silent.map(|waits| {
            ending(
                Cause::SilentSensor,
                failed(
                    FailureKind::ResourceUnavailable,
                    format!(
                        "a sensor the grid reads gave no reading within the source timeout: \
                         the navigator waits for {waits}"
                    ),
                ),
            )
        }),
        (!limb.parts_within_footprint).then(|| {
            ending(
                Cause::LimbOutsideFootprint,
                failed(
                    FailureKind::AttemptFailed,
                    "drive_states reports parts_within_footprint false: a limb left the footprint",
                ),
            )
        }),
        cancel.map(|reason| {
            ending(
                Cause::CallerCancel,
                Finish::Cancelled(reason, Located::of(&state)),
            )
        }),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// The ending that a follow_state other than following in `at` gives the
/// goal in flight: robot_stop and follower_goal
/// cancel it, its result where drive_states first reported the hold;
/// base_mode fails it with `attempt_failed` and base_silence with
/// `resource_unavailable`.
fn hold_ending(at: &Received) -> Option<Ending<FailureKind>> {
    let state = at.state;
    let first_report = Located::of(&at.left_following.unwrap_or(state));
    let reported = format!(
        "drive_states reports follow_state {}",
        state.follow.as_str()
    );
    let finish = match state.follow {
        FollowState::Following => return None,
        FollowState::RobotStop => Finish::Cancelled(
            format!("{reported}: a robot stop holds the base"),
            first_report,
        ),
        FollowState::FollowerGoal => Finish::Cancelled(
            format!("{reported}: a goal of the base's node holds the base"),
            first_report,
        ),
        FollowState::BaseMode => failed(
            FailureKind::AttemptFailed,
            format!(
                "{reported}: the base's mode {} holds it",
                state.mode.as_str()
            ),
        ),
        FollowState::BaseSilence => failed(
            FailureKind::ResourceUnavailable,
            format!("{reported}: a silence of the base's motion sensing holds the base"),
        ),
    };
    Some(Ending {
        cause: Cause::Hold,
        finish,
    })
}

/// What a step decides for the goal in flight: it ends, or it runs on by its
/// own step `S`.
#[derive(Debug)]
enum Decided<S, K> {
    Ends(Ending<K>),
    RunsOn(S),
}

/// The decision of a step on the goal in flight, the robot frame `located`:
/// the earliest of the `shared` endings and of the ending of the goal's own
/// step `own`, which `ended` reads of it, or of the grid's failure when the
/// goal had no grid to step on; with none, the goal runs on by its step.
fn decide<S, K>(
    shared: Vec<Ending<FailureKind>>,
    own: Result<S, Failure>,
    ended: impl FnOnce(&S) -> Option<Ended<K>>,
    located: Located,
) -> Decided<S, K>
where
    K: From<FailureKind>,
    Ending<FailureKind>: Into<Ending<K>>,
{
    let shared = shared.into_iter().map(Into::into);
    let step = match own {
        Ok(step) => step,
        Err(failure) => {
            let grid_failure = Ending {
                cause: Cause::OtherFailure,
                finish: Finish::Failed(failure),
            };
            let earlier = first_ending(shared).filter(|ending| ending.cause < Cause::OtherFailure);
            return Decided::Ends(earlier.unwrap_or_else(|| grid_failure.into()));
        }
    };
    let own_ending = ended(&step).map(|ended| Ending::of_own(ended, located));
    match first_ending(shared.chain(own_ending)) {
        Some(ending) => Decided::Ends(ending),
        None => Decided::RunsOn(step),
    }
}

/// The release burst after a goal's end: zero velocity at each step
/// until it has streamed [`BURST_SETPOINTS`] setpoints and [`BURST_NS`] have
/// passed since that ending, the count checked after each send.
#[derive(Clone, Copy)]
struct ReleaseBurst {
    since_ns: i64,
    sent: u32,
}

impl ReleaseBurst {
    fn new(since_ns: i64) -> Self {
        Self { since_ns, sent: 0 }
    }

    /// The burst after it streams one zero setpoint at `now_ns`: none once
    /// that setpoint completes it.
    fn after_send(self, now_ns: i64) -> Option<Self> {
        let sent = self.sent + 1;
        let done = sent >= BURST_SETPOINTS && now_ns - self.since_ns >= BURST_NS;
        (!done).then_some(Self { sent, ..self })
    }
}

/// The navigator's stream on drive_link: when its newest setpoint went out
/// (ns on the node's clock), and the release burst while one runs.
#[derive(Default)]
struct SetpointStream {
    newest_setpoint_ns: Option<i64>,
    burst: Option<ReleaseBurst>,
}

impl SetpointStream {
    /// Whether a release burst may stream at `now_ns`, the newest
    /// drive_states reporting `follow`: while at most 0.5 s have passed since
    /// the newest setpoint and the base follows.
    fn bursts_at(&self, now_ns: i64, follow: Option<FollowState>) -> bool {
        LEADER_STREAM.streams(self.newest_setpoint_ns, now_ns)
            && follow == Some(FollowState::Following)
    }

    /// A goal ended at `now_ns`, the newest drive_states reporting `follow`:
    /// the burst starts from this ending, a burst that runs included, while a
    /// burst may stream.
    fn goal_ended(&mut self, now_ns: i64, follow: Option<FollowState>) {
        self.burst = self
            .bursts_at(now_ns, follow)
            .then(|| ReleaseBurst::new(now_ns));
    }

    /// Whether a release burst runs.
    fn bursts(&self) -> bool {
        self.burst.is_some()
    }

    /// The setpoint the step at `now_ns` streams, the newest drive_states
    /// reporting `follow`: the goal's `command`, which ends a burst, or else
    /// the burst's zero while it may stream; none otherwise. A burst that may
    /// not stream ends.
    fn setpoint(
        &mut self,
        command: Option<PlanarTwist>,
        now_ns: i64,
        follow: Option<FollowState>,
    ) -> Option<PlanarTwist> {
        let setpoint = match command {
            Some(command) => {
                self.burst = None;
                Some(command)
            }
            None => self.burst_setpoint(now_ns, follow),
        };
        if setpoint.is_some() {
            self.newest_setpoint_ns = Some(now_ns);
        }
        setpoint
    }

    /// The burst's zero at `now_ns` while the burst may stream.
    fn burst_setpoint(&mut self, now_ns: i64, follow: Option<FollowState>) -> Option<PlanarTwist> {
        let burst = self
            .burst
            .take()
            .filter(|_| self.bursts_at(now_ns, follow))?;
        self.burst = burst.after_send(now_ns);
        Some(PlanarTwist::ZERO)
    }
}

/// The context of a move's action.
enum MoveContext {
    Spin(Box<spin::GoalContext>),
    DriveOnHeading(Box<drive_on_heading::GoalContext>),
    BackUp(Box<back_up::GoalContext>),
}

/// What a goal in flight runs, with its action's context.
enum Motion {
    Navigation {
        navigation: Box<Navigation>,
        ctx: Box<navigate_to_pose::GoalContext>,
    },
    Move {
        motion: Move,
        ctx: MoveContext,
    },
}

/// The goal in flight: what it runs, the base it drives, the odometry it
/// runs on, and when it was accepted on the wall clock.
struct RunningGoal {
    motion: Motion,
    base: Arc<Base>,
    odometry_start: SystemTime,
    /// The number of the newest drive_states the goal read: the one it was
    /// accepted at, then the newest at each of its steps.
    read_through: u64,
    accepted: Instant,
}

/// The record of a pose of an action's result or feedback, `$action`'s
/// `$record` type, from `$located`, in odom.
macro_rules! pose_record {
    ($action:ident::$record:ident, $located:expr) => {{
        let located: Located = $located;
        let (position, orientation) = located.arrays();
        $action::$record {
            frame: ODOMETRY_FRAME.to_owned(),
            timestamp: located.timestamp,
            position,
            orientation,
        }
    }};
}

/// End `$ctx`, the context of a move's action, on `$finish`, with the final
/// pose as `$action`'s `$final_pose` type, after `$action_time` (s). Written
/// once here because the three generated contexts carry distinct types with
/// an identical surface.
macro_rules! end_move {
    ($ctx:expr, $finish:expr, $action_time:expr, $action:ident::$final_pose:ident) => {
        match $finish {
            Finish::Completed(located) => {
                $ctx.complete(pose_record!($action::$final_pose, located), $action_time)
                    .await
            }
            Finish::Cancelled(reason, located) => {
                $ctx.complete_cancelled(
                    reason,
                    pose_record!($action::$final_pose, located),
                    $action_time,
                )
                .await
            }
            Finish::Failed(failure) => $ctx.fail(failure.kind.into(), failure.message).await,
        }
    };
}

impl MoveContext {
    fn cancel_reason(&self) -> Option<String> {
        let cause = match self {
            Self::Spin(ctx) => ctx.cancel_cause(),
            Self::DriveOnHeading(ctx) => ctx.cancel_cause(),
            Self::BackUp(ctx) => ctx.cancel_cause(),
        };
        cause.map(|cause| cause.reason().to_string())
    }

    async fn publish_feedback(&self, feedback: MoveFeedback) -> peppygen::Result<()> {
        match (self, feedback) {
            (Self::Spin(ctx), MoveFeedback::AngleTurned(angle)) => {
                ctx.publish_feedback(angle).await
            }
            (Self::DriveOnHeading(ctx), MoveFeedback::DistanceTraveled(distance)) => {
                ctx.publish_feedback(distance).await
            }
            (Self::BackUp(ctx), MoveFeedback::DistanceTraveled(distance)) => {
                ctx.publish_feedback(distance).await
            }
            (_, feedback) => unreachable!("a move gives the feedback of its action: {feedback:?}"),
        }
    }

    async fn end(self, finish: Finish<FailureKind>, action_time: f64) -> peppygen::Result<()> {
        match self {
            Self::Spin(ctx) => end_move!(ctx, finish, action_time, spin::SpinResultFinalPose),
            Self::DriveOnHeading(ctx) => end_move!(
                ctx,
                finish,
                action_time,
                drive_on_heading::DriveOnHeadingResultFinalPose
            ),
            Self::BackUp(ctx) => {
                end_move!(ctx, finish, action_time, back_up::BackUpResultFinalPose)
            }
        }
    }
}

/// End `ctx`, a navigation's context, on `finish`, the base having driven
/// `driven_m`, after `action_time` (s).
async fn end_navigation(
    ctx: Box<navigate_to_pose::GoalContext>,
    finish: Finish<NavigationKind>,
    driven_m: f64,
    action_time: f64,
) -> peppygen::Result<()> {
    let record = |located| pose_record!(navigate_to_pose::NavigateToPoseResultFinalPose, located);
    match finish {
        Finish::Completed(located) => ctx.complete(record(located), driven_m, action_time).await,
        Finish::Cancelled(reason, located) => {
            ctx.complete_cancelled(reason, record(located), driven_m, action_time)
                .await
        }
        Finish::Failed(failure) => ctx.fail(failure.kind.into(), failure.message).await,
    }
}

impl RunningGoal {
    fn cancel_reason(&self) -> Option<String> {
        match &self.motion {
            Motion::Navigation { ctx, .. } => {
                ctx.cancel_cause().map(|cause| cause.reason().to_string())
            }
            Motion::Move { ctx, .. } => ctx.cancel_reason(),
        }
    }

    /// End the goal on `finish`, an ending every goal shares.
    async fn finish(self, finish: Finish<FailureKind>) {
        let action_time = self.accepted.elapsed().as_secs_f64();
        match self.motion {
            Motion::Navigation { navigation, ctx } => {
                let driven_m = navigation.driven_m();
                log_end(end_navigation(
                    ctx,
                    finish.of_navigation(),
                    driven_m,
                    action_time,
                ))
                .await;
            }
            Motion::Move { ctx, .. } => log_end(ctx.end(finish, action_time)).await,
        }
    }
}

/// Await `end`, a goal's end on its action, logging its failure.
async fn log_end(end: impl Future<Output = peppygen::Result<()>>) {
    if let Err(e) = end.await {
        error!("base_navigation goal end: {e}");
    }
}

/// The step of a goal: whether it runs on, with its command after the stop,
/// or ended.
enum GoalStep {
    Runs(Box<RunningGoal>, Option<PlanarTwist>),
    Ended,
}

/// The grid a goal steps on: its occupancy, and whether every sensor's
/// reading went in since its newest clearing.
struct StepGrid {
    occupancy: Arc<CostGrid>,
    read_since_clearing: bool,
}

/// The control loop's state.
pub struct Navigator {
    shared: Shared,
    planning: Planning,
    /// Whether the grid rolls, which each goal's acceptance starts.
    rolling: Rolling,
    goal: Option<RunningGoal>,
    stream: SetpointStream,
    last_step_ns: Option<i64>,
}

impl Navigator {
    pub fn new(shared: Shared, planning: Planning) -> Self {
        let rolling = lock_ignoring_poison(&shared.grid).rolling();
        Self {
            shared,
            planning,
            rolling,
            goal: None,
            stream: SetpointStream::default(),
            last_step_ns: None,
        }
    }

    pub async fn handle(&mut self, event: Event) {
        match event {
            Event::Goal(goal) => self.start(*goal).await,
            Event::ClearGrid { reply } => {
                lock_ignoring_poison(&self.shared.grid).clear();
                info!("clear_grid: the grid is empty");
                let _ = reply.send(());
            }
        }
    }

    /// Start `goal` in place of the goal in flight, which ends cancelled: by
    /// its caller's cancel when one came, its result where the newest
    /// drive_states places the robot frame, and otherwise with its result
    /// where the robot frame stood at the timestamp of the newest
    /// drive_states when `goal` was accepted.
    async fn start(&mut self, goal: AcceptedGoal) {
        let admitted = match &goal {
            AcceptedGoal::Navigate { admitted, .. }
            | AcceptedGoal::Spin { admitted, .. }
            | AcceptedGoal::DriveOnHeading { admitted, .. }
            | AcceptedGoal::BackUp { admitted, .. } => admitted.clone(),
        };
        let now_ns = clock::now_ns(&self.shared.clock);
        let newest = lock_ignoring_poison(&self.shared.link)
            .newest()
            .expect("a goal is admitted once drive_states arrived");
        let follow = Some(newest.state.follow);
        if let Some(running) = self.goal.take() {
            let finish = match running.cancel_reason() {
                Some(reason) => Finish::Cancelled(reason, Located::of(&newest.state)),
                None => replaced(&admitted.at),
            };
            running.finish(finish).await;
            self.goal_ended(now_ns, follow);
        }
        let state = admitted.at.state;
        let start = state.pose;
        let motion = match goal {
            AcceptedGoal::Navigate { request, ctx, .. } => Motion::Navigation {
                navigation: Box::new(Navigation::new(
                    request,
                    admitted.base.clone(),
                    self.shared.config.clone(),
                    start,
                )),
                ctx,
            },
            AcceptedGoal::Spin { request, ctx, .. } => Motion::Move {
                motion: Move::spin(request, &admitted.base, &self.shared.config, start, now_ns),
                ctx: MoveContext::Spin(ctx),
            },
            AcceptedGoal::DriveOnHeading { request, ctx, .. } => Motion::Move {
                motion: Move::straight(
                    Direction::Forward,
                    request,
                    &admitted.base,
                    &self.shared.config,
                    start,
                    now_ns,
                ),
                ctx: MoveContext::DriveOnHeading(ctx),
            },
            AcceptedGoal::BackUp { request, ctx, .. } => Motion::Move {
                motion: Move::straight(
                    Direction::Backward,
                    request,
                    &admitted.base,
                    &self.shared.config,
                    start,
                    now_ns,
                ),
                ctx: MoveContext::BackUp(ctx),
            },
        };
        let running = RunningGoal {
            motion,
            base: admitted.base,
            odometry_start: state.odometry_start,
            read_through: admitted.at.seq,
            accepted: Instant::now(),
        };
        let centered = lock_ignoring_poison(&self.shared.grid).roll_from([start.x(), start.y()]);
        if let Err(e) = centered {
            let why = format!("the grid cannot be centered on the base: {e}");
            running
                .finish(failed(FailureKind::AttemptFailed, why))
                .await;
            return self.goal_ended(now_ns, follow);
        }
        self.goal = Some(running);
    }

    /// A goal ended at `now_ns`, the newest drive_states reporting `follow`:
    /// its plan is forgotten, the release burst follows as
    /// [`SetpointStream::goal_ended`] says, and the grid rolls on while the
    /// burst streams.
    fn goal_ended(&mut self, now_ns: i64, follow: Option<FollowState>) {
        self.planning.forget();
        self.stream.goal_ended(now_ns, follow);
        self.stop_rolling_once_idle();
    }

    /// The grid rolls with the base while a goal runs or the release burst
    /// streams, as base_navigation says: once neither does, it stays where it
    /// stands.
    fn stop_rolling_once_idle(&self) {
        if self.goal.is_none() && !self.stream.bursts() {
            self.rolling.stop();
        }
    }

    /// When the next step is due on the node's clock reading `now_ns`.
    fn next_step_ns(&self, now_ns: i64) -> i64 {
        match self.last_step_ns {
            Some(last_ns) if last_ns <= now_ns => last_ns + self.shared.config.step_ns,
            _ => now_ns,
        }
    }

    /// Run the latest step due on the node's clock, if it has not run: the
    /// setpoint it streams, if any, and the step's time.
    pub async fn advance(&mut self) -> Option<(PlanarTwist, i64)> {
        let now_ns = clock::now_ns(&self.shared.clock);
        let period = self.shared.config.step_ns;
        let due = now_ns.div_euclid(period) * period;
        if let Some(last_ns) = self.last_step_ns {
            if due == last_ns {
                return None;
            }
            if due < last_ns {
                warn!("the node's clock went back to {now_ns} ns: the steps start over from it");
            }
        }
        self.last_step_ns = Some(due);
        self.step(due).await.map(|setpoint| (setpoint, due))
    }

    /// One step at `now_ns`: the goal in flight steps or ends, and the step
    /// gives its setpoint, the goal's or the release burst's, if any.
    async fn step(&mut self, now_ns: i64) -> Option<PlanarTwist> {
        let newest = lock_ignoring_poison(&self.shared.link).newest();
        let follow = newest.map(|newest| newest.state.follow);
        let command = match self.goal.take() {
            Some(goal) => {
                let newest = newest.expect("a goal is admitted once drive_states arrived");
                match self.goal_step(goal, &newest, now_ns).await {
                    GoalStep::Runs(goal, command) => {
                        self.goal = Some(*goal);
                        command
                    }
                    GoalStep::Ended => {
                        self.goal_ended(now_ns, follow);
                        None
                    }
                }
            }
            None => None,
        };
        let setpoint = self.stream.setpoint(command, now_ns, follow);
        self.stop_rolling_once_idle();
        setpoint
    }

    /// What a goal waits for of the sensors silent at `now_ns`, each by
    /// name in the sensors' order; none while every sensor reads.
    fn silent_sensors(&self, now_ns: i64) -> Option<String> {
        let sensors = &self.shared.sensors;
        let silent =
            lock_ignoring_poison(&self.shared.grid).silent_sensors(now_ns, sensors.count().get());
        let source_timeout_s = self.shared.config.stop.source_timeout_s.get();
        (!silent.is_empty()).then(|| sensors.waits(&silent, source_timeout_s))
    }

    /// The grid the goal in flight steps on, or the failure of a goal with
    /// none.
    fn step_grid(&self) -> Result<StepGrid, Failure> {
        let mut slot = lock_ignoring_poison(&self.shared.grid);
        let Some(grid) = slot.grid_mut() else {
            return Err(Failure::new(
                FailureKind::ResourceUnavailable,
                "the grid of the floor waits for describe_base",
            ));
        };
        let occupancy = grid.occupancy().map_err(|e| {
            Failure::new(
                FailureKind::AttemptFailed,
                format!("the grid of the floor: {e}"),
            )
        })?;
        Ok(StepGrid {
            occupancy,
            read_since_clearing: grid.read_since_clearing(),
        })
    }

    /// `command` as the stop passes it at `now_ns`, the base at `pose` in
    /// odom: zero velocity when the stop cannot read the grid.
    fn stopped(&self, command: PlanarTwist, pose: PlanarPose, now_ns: i64) -> PlanarTwist {
        let slowed = lock_ignoring_poison(&self.shared.grid)
            .grid_mut()
            .map(|grid| grid.slowed(command, pose, now_ns));
        match slowed {
            Some(Ok(decision)) => decision.velocity(),
            Some(Err(e)) => {
                warn!("the stop: {e}: the base gets zero velocity");
                PlanarTwist::ZERO
            }
            None => PlanarTwist::ZERO,
        }
    }

    /// The step of `goal` at `now_ns`, the newest drive_states being `at`:
    /// the goal's own step runs on the grid, and the step decides on every
    /// ending the goal met ([`decide`]).
    async fn goal_step(&mut self, goal: RunningGoal, at: &Received, now_ns: i64) -> GoalStep {
        let state = at.state;
        let located = Located::of(&state);
        let reports = lock_ignoring_poison(&self.shared.link).reports_after(goal.read_through);
        let shared = shared_endings(
            at,
            &reports,
            now_ns,
            goal.odometry_start,
            goal.cancel_reason(),
            self.silent_sensors(now_ns),
        );
        let grid = self.step_grid();
        let RunningGoal {
            motion,
            base,
            odometry_start,
            accepted,
            ..
        } = goal;
        let footprint = base.padded_footprint.clone();
        let action_time = accepted.elapsed().as_secs_f64();
        let (motion, command) = match motion {
            Motion::Navigation {
                mut navigation,
                ctx,
            } => {
                let own = grid.map(|grid| {
                    let step = navigation.step(NavigationInputs {
                        now_ns,
                        pose: state.pose,
                        velocity: state.velocity,
                        grid: &grid.occupancy,
                        footprint: &footprint,
                        read_since_clearing: grid.read_since_clearing,
                        plan: self.planning.finished(),
                    });
                    (step, grid.occupancy)
                });
                let ended = |(step, _): &(NavigationStep, Arc<CostGrid>)| step.ended.clone();
                let (step, occupancy) = match decide(shared, own, ended, located) {
                    Decided::RunsOn(own) => own,
                    Decided::Ends(ending) => {
                        let driven_m = navigation.driven_m();
                        log_end(end_navigation(ctx, ending.finish, driven_m, action_time)).await;
                        return GoalStep::Ended;
                    }
                };
                if step.clear_grid {
                    lock_ignoring_poison(&self.shared.grid).clear();
                }
                if let Some(request) = step.plan {
                    self.planning.start(request, occupancy, footprint.clone());
                }
                let feedback = step.feedback;
                let published = ctx
                    .publish_feedback(
                        pose_record!(navigate_to_pose::NavigateToPoseFeedbackCurrentPose, located),
                        feedback.distance_remaining_m,
                        feedback.angle_remaining_rad,
                        feedback.estimated_time_remaining_s,
                        feedback.recoveries,
                    )
                    .await;
                if let Err(e) = published {
                    warn!("navigate_to_pose feedback: {e}");
                }
                (Motion::Navigation { navigation, ctx }, step.command)
            }
            Motion::Move { mut motion, ctx } => {
                let own = grid.map(|grid| {
                    motion.step(&MoveInputs {
                        now_ns,
                        pose: state.pose,
                        grid: &grid.occupancy,
                        footprint: &footprint,
                    })
                });
                let ended = |step: &MoveStep| step.ended.clone();
                let step = match decide(shared, own, ended, located) {
                    Decided::RunsOn(step) => step,
                    Decided::Ends(ending) => {
                        log_end(ctx.end(ending.finish, action_time)).await;
                        return GoalStep::Ended;
                    }
                };
                if let Err(e) = ctx.publish_feedback(step.feedback).await {
                    warn!("base_navigation move feedback: {e}");
                }
                (Motion::Move { motion, ctx }, step.command)
            }
        };
        let goal = RunningGoal {
            motion,
            base,
            odometry_start,
            read_through: at.seq,
            accepted,
        };
        let setpoint = command.map(|command| self.stopped(command, state.pose, now_ns));
        GoalStep::Runs(Box::new(goal), setpoint)
    }
}

/// Publish `velocity` on `setpoints`, drive_link's, in the robot frame,
/// stamped with `now_ns`, the time of its step on the node's clock, which is
/// the clock of transforms: a velocity in the robot frame stands in it at any
/// time, as drive_link states.
async fn publish(setpoints: &TopicPublisher, velocity: PlanarTwist, now_ns: i64) {
    let message = drive_setpoints::build_message(
        None,
        Some(wire_from_ns(now_ns)),
        [velocity.linear_x(), velocity.linear_y(), 0.0],
        [0.0, 0.0, velocity.angular_z()],
    );
    let sent = match message {
        Ok(message) => setpoints.publish(message).await,
        Err(e) => Err(e),
    };
    if let Err(e) = sent {
        warn!("drive_setpoints: {e}");
    }
}

/// Run the loop until `cancelled` resolves or every event sender is gone,
/// waking at each event and at each step due on the node's clock, and
/// publishing each step's setpoint on `setpoints`.
pub async fn run(
    mut navigator: Navigator,
    setpoints: TopicPublisher,
    mut events: mpsc::Receiver<Event>,
    cancelled: impl Future<Output = ()>,
) {
    let mut cancelled = std::pin::pin!(cancelled);
    loop {
        let now_ns = clock::now_ns(&navigator.shared.clock);
        let due_ns = u64::try_from(navigator.next_step_ns(now_ns)).unwrap_or(0);
        let clock = navigator.shared.clock.clone();
        tokio::select! {
            biased;
            () = &mut cancelled => return,
            event = events.recv() => match event {
                Some(event) => navigator.handle(event).await,
                None => return,
            },
            () = clock.until(due_ns) => {}
        }
        if let Some((velocity, at_ns)) = navigator.advance().await {
            publish(&setpoints, velocity, at_ns).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Mutex;
    use std::time::Duration;

    use control_core::node_clock::NodeClock;
    use hegel::TestCase;
    use hegel::generators as gs;

    use crate::cameras::Sensors;
    use crate::cameras::tests::camera_slot;
    use crate::config::tests::config;
    use crate::grid::GridSlot;
    use crate::grid::tests::{covers, grid, read_at};
    use crate::link::tests::state;
    use control_core::drive_link::WINDOW_NS;

    use crate::link::Link;
    use crate::planning::Planner;

    const NOW_NS: i64 = 10_000_000_000;

    /// One step of the controller at tests/parameters.json's 20 Hz.
    const STEP_NS: i64 = 50_000_000;

    /// The time `stamp_s` s after the epoch, a drive_states' stamp in
    /// [`state`].
    fn stamp(stamp_s: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(stamp_s)
    }

    /// The newest drive_states, arrived at NOW_NS: after one at 8 s that
    /// reported following with the robot frame at x 1.0, one at 9 s at x 2.0
    /// and one at 10 s at x 3.0 report `follow`.
    fn received(follow: FollowState) -> Received {
        let mut link = Link::default();
        link.take(state(5, 8, 1.0, FollowState::Following), NOW_NS - 200);
        link.take(state(5, 9, 2.0, follow), NOW_NS - 100);
        link.take(state(5, 10, 3.0, follow), NOW_NS);
        link.newest().expect("a state")
    }

    /// The odometry start of [`received`]'s drive_states.
    fn start() -> SystemTime {
        state(5, 0, 0.0, FollowState::Following).odometry_start
    }

    fn kind<K: Copy>(finish: &Finish<K>) -> Option<K> {
        match finish {
            Finish::Failed(failure) => Some(failure.kind),
            _ => None,
        }
    }

    // drive_states that report base_silence fail the goal with
    // resource_unavailable, base_mode with attempt_failed, and robot_stop and
    // follower_goal cancel it with the result where they were first reported;
    // a cancel's result is where the newest drive_states placed the robot
    // frame.
    #[test]
    fn each_follow_state_ends_the_goal_by_its_own_rule() {
        let ending = |follow, cancel| {
            first_ending(shared_endings(
                &received(follow),
                &Reports::default(),
                NOW_NS,
                start(),
                cancel,
                None,
            ))
        };
        let failure_of = |follow| ending(follow, None).and_then(|ending| kind(&ending.finish));
        assert_eq!(
            failure_of(FollowState::BaseSilence),
            Some(FailureKind::ResourceUnavailable)
        );
        assert_eq!(
            failure_of(FollowState::BaseMode),
            Some(FailureKind::AttemptFailed)
        );
        for follow in [FollowState::RobotStop, FollowState::FollowerGoal] {
            let Some(Ending {
                cause: Cause::Hold,
                finish: Finish::Cancelled(reason, located),
            }) = ending(follow, None)
            else {
                panic!("{follow:?} cancels");
            };
            assert!(reason.contains(follow.as_str()), "{reason}");
            assert_eq!(
                (located.pose.x(), located.timestamp),
                (2.0, stamp(9)),
                "where drive_states first reported it"
            );
        }
        let Some(Ending {
            cause: Cause::CallerCancel,
            finish: Finish::Cancelled(reason, located),
        }) = ending(FollowState::Following, Some("the caller".to_string()))
        else {
            panic!("a cancel ends the goal");
        };
        assert_eq!(
            (reason.as_str(), located.pose.x(), located.timestamp),
            ("the caller", 3.0, stamp(10))
        );
        assert_eq!(ending(FollowState::Following, None), None);
    }

    // A robot_stop and a parts_within_footprint false that later
    // drive_states replaced before the step each end the goal: the newest
    // alone, which reports following, gives no ending.
    #[test]
    fn a_report_a_later_drive_states_replaced_ends_the_goal() {
        let mut link = Link::default();
        link.take(state(5, 8, 1.0, FollowState::Following), NOW_NS);
        let admitted = link.newest().expect("a state").seq;
        link.take(state(5, 9, 2.0, FollowState::RobotStop), NOW_NS);
        let limb_outside = DriveState {
            parts_within_footprint: false,
            ..state(5, 10, 3.0, FollowState::Following)
        };
        link.take(limb_outside, NOW_NS);
        link.take(state(5, 11, 4.0, FollowState::Following), NOW_NS);
        let newest = link.newest().expect("a state");
        let endings = |reports: &Reports| {
            shared_endings(&newest, reports, NOW_NS, start(), None, None)
                .into_iter()
                .map(|ending| ending.cause)
                .collect::<Vec<_>>()
        };
        assert_eq!(endings(&Reports::default()), Vec::new());
        let reports = link.reports_after(admitted);
        assert_eq!(
            endings(&reports),
            vec![Cause::Hold, Cause::LimbOutsideFootprint]
        );
        let Some(Ending {
            finish: Finish::Cancelled(_, located),
            ..
        }) = hold_ending(&reports.hold.expect("the robot_stop"))
        else {
            panic!("a robot_stop cancels");
        };
        assert_eq!(located.pose.x(), 2.0, "where the robot_stop was reported");
    }

    /// What a goal's own step gives in a test.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Own {
        RunsOn,
        Fails,
        TimesOut,
        Reaches,
    }

    /// The endings a test gives one step: each true, or `follow` other than
    /// following, or `own` other than running on, meets one.
    #[derive(Clone, Copy, Debug)]
    struct Met {
        follow: FollowState,
        odometry_over: bool,
        drive_states_lost: bool,
        sensor_silent: bool,
        limb_outside: bool,
        cancelled: bool,
        own: Own,
    }

    /// The cause and the failure kind, none for a cancel or a completion, of
    /// the first ending `met` gives in base_navigation's order.
    fn earliest(met: Met) -> Option<(Cause, Option<FailureKind>)> {
        use FailureKind::{AttemptFailed, Blocked, ResourceUnavailable, Timeout};
        let hold_kind = match met.follow {
            FollowState::BaseMode => Some(AttemptFailed),
            FollowState::BaseSilence => Some(ResourceUnavailable),
            _ => None,
        };
        [
            (met.cancelled, (Cause::CallerCancel, None)),
            (
                met.follow != FollowState::Following,
                (Cause::Hold, hold_kind),
            ),
            (
                met.odometry_over,
                (Cause::OdometryStartedOver, Some(ResourceUnavailable)),
            ),
            (
                met.drive_states_lost,
                (Cause::DriveStatesLost, Some(ResourceUnavailable)),
            ),
            (
                met.sensor_silent,
                (Cause::SilentSensor, Some(ResourceUnavailable)),
            ),
            (
                met.limb_outside,
                (Cause::LimbOutsideFootprint, Some(AttemptFailed)),
            ),
            (met.own == Own::TimesOut, (Cause::Timeout, Some(Timeout))),
            (met.own == Own::Reaches, (Cause::Reached, None)),
            (met.own == Own::Fails, (Cause::OtherFailure, Some(Blocked))),
        ]
        .into_iter()
        .find_map(|(meets, ending)| meets.then_some(ending))
    }

    // One step that gives a goal several endings ends it by the first of
    // the caller's cancel, a follow_state other than following, a new
    // odometry_start_timestamp, more than 0.5 s with no drive_states, a
    // silent sensor, parts_within_footprint false, the passing of its
    // timeout_s, its completion and another failure; a goal with no ending
    // runs on.
    #[hegel::test]
    fn a_step_with_several_endings_takes_the_earliest(tc: TestCase) {
        let draw = || tc.draw(gs::booleans());
        let follows = [
            FollowState::Following,
            FollowState::BaseMode,
            FollowState::BaseSilence,
            FollowState::RobotStop,
            FollowState::FollowerGoal,
        ];
        let follow: usize = tc.draw(gs::integers().min_value(0).max_value(4));
        let own: usize = tc.draw(gs::integers().min_value(0).max_value(3));
        let met = Met {
            follow: follows[follow],
            odometry_over: draw(),
            drive_states_lost: draw(),
            sensor_silent: draw(),
            limb_outside: draw(),
            cancelled: draw(),
            own: [Own::RunsOn, Own::Fails, Own::TimesOut, Own::Reaches][own],
        };
        let mut at = received(met.follow);
        at.state.parts_within_footprint = !met.limb_outside;
        let odometry_start = if met.odometry_over {
            state(4, 0, 0.0, FollowState::Following).odometry_start
        } else {
            start()
        };
        let now_ns = if met.drive_states_lost {
            NOW_NS + WINDOW_NS + 1
        } else {
            NOW_NS
        };
        let silent = met
            .sensor_silent
            .then(|| "a reading of the camera front".to_string());
        let cancel = met.cancelled.then(|| "the caller".to_string());
        let ended = match met.own {
            Own::RunsOn => None,
            Own::Fails => Some(Ended::Failed(Failure::new(FailureKind::Blocked, "blocked"))),
            Own::TimesOut => Some(Ended::TimedOut("timeout_s passed".to_string())),
            Own::Reaches => Some(Ended::Completed),
        };
        let shared = shared_endings(
            &at,
            &Reports::default(),
            now_ns,
            odometry_start,
            cancel,
            silent,
        );
        let decided = decide(shared, Ok(()), |_| ended, Located::of(&at.state));
        let taken = match decided {
            Decided::Ends(ending) => Some((ending.cause, kind(&ending.finish))),
            Decided::RunsOn(()) => None,
        };
        assert_eq!(taken, earliest(met), "{met:?}");
    }

    // A goal with no grid to step on fails by the grid's failure, which the
    // endings every goal shares come before, a navigation taking them with
    // its own kinds; a goal with no ending runs on by its own step.
    #[test]
    fn a_goal_with_no_grid_fails_after_the_shared_endings() {
        let at = received(FollowState::Following);
        let located = Located::of(&at.state);
        let no_grid = Failure::new(FailureKind::ResourceUnavailable, "no grid");
        let alone = decide::<(), FailureKind>(Vec::new(), Err(no_grid.clone()), |_| None, located);
        assert!(
            matches!(
                &alone,
                Decided::Ends(Ending { cause: Cause::OtherFailure, finish })
                    if kind(finish) == Some(FailureKind::ResourceUnavailable)
            ),
            "{alone:?}"
        );
        let silent = shared_endings(
            &at,
            &Reports::default(),
            NOW_NS,
            start(),
            None,
            Some("a reading of the camera front".to_string()),
        );
        let navigation = decide::<(), NavigationKind>(silent, Err(no_grid), |_| None, located);
        assert!(
            matches!(
                &navigation,
                Decided::Ends(Ending { cause: Cause::SilentSensor, finish })
                    if kind(finish) == Some(NavigationKind::Goal(FailureKind::ResourceUnavailable))
            ),
            "{navigation:?}"
        );
        let runs_on = decide::<u8, FailureKind>(Vec::new(), Ok(7), |_| None, located);
        assert!(matches!(runs_on, Decided::RunsOn(7)), "{runs_on:?}");
    }

    // A replaced goal ends cancelled, its result where the robot frame
    // stood at the timestamp of the newest drive_states when the new goal
    // was accepted.
    #[test]
    fn a_replaced_goal_ends_at_the_newest_drive_states_of_the_acceptance() {
        let Finish::Cancelled(reason, located) = replaced(&received(FollowState::Following)) else {
            panic!("a replaced goal ends cancelled");
        };
        assert_eq!(
            (reason.as_str(), located.pose.x(), located.timestamp),
            (REPLACED, 3.0, stamp(10))
        );
    }

    /// A stream whose goal streamed its newest setpoint at 0.
    fn streaming() -> SetpointStream {
        let mut stream = SetpointStream::default();
        let ahead = PlanarTwist::new(0.3, 0.0, 0.0).expect("a finite twist");
        let streamed = stream.setpoint(Some(ahead), 0, Some(FollowState::Following));
        assert_eq!(streamed, Some(ahead));
        stream
    }

    /// The steps, one each STEP_NS from `from_ns`, at which `stream` streams
    /// the burst's zero with no goal streaming, the newest drive_states
    /// reporting `follow`, until it streams nothing.
    fn burst(stream: &mut SetpointStream, from_ns: i64, follow: FollowState) -> Vec<i64> {
        (0..100)
            .map(|k| from_ns + k * STEP_NS)
            .map_while(|now_ns| {
                let zero = stream.setpoint(None, now_ns, Some(follow))?;
                assert_eq!(zero, PlanarTwist::ZERO);
                Some(now_ns)
            })
            .collect()
    }

    /// The steps of a burst that streams no zero.
    const NO_STEPS: [i64; 0] = [];

    /// The steps, one each STEP_NS, from `from_ns` to `to_ns`.
    fn steps(from_ns: i64, to_ns: i64) -> Vec<i64> {
        (from_ns..=to_ns)
            .step_by(usize::try_from(STEP_NS).expect("a positive step"))
            .collect()
    }

    // A goal's end while the goal streams brings zero velocity for 0.25 s
    // and at least 3 setpoints, then nothing; three setpoints go out also
    // past 0.25 s.
    #[test]
    fn a_goals_end_brings_the_release_burst() {
        let mut ended = streaming();
        ended.goal_ended(STEP_NS, Some(FollowState::Following));
        assert_eq!(
            burst(&mut ended, STEP_NS, FollowState::Following),
            steps(STEP_NS, STEP_NS + BURST_NS)
        );
        let mut slow = streaming();
        slow.goal_ended(0, Some(FollowState::Following));
        let slow_steps: Vec<bool> = [0, 300_000_000, 600_000_000, 900_000_000]
            .into_iter()
            .map(|now_ns| {
                slow.setpoint(None, now_ns, Some(FollowState::Following))
                    .is_some()
            })
            .collect();
        assert_eq!(slow_steps, [true, true, true, false], "three past 0.25 s");
    }

    // The burst streams only while at most 0.5 s have passed since the
    // newest setpoint and the newest drive_states reports following: an
    // ending past 0.5 s or during a hold brings none, and a hold report or a
    // step past 0.5 s ends a burst.
    #[test]
    fn the_burst_streams_only_while_the_stream_lives_and_the_base_follows() {
        let mut late = streaming();
        late.goal_ended(WINDOW_NS + 1, Some(FollowState::Following));
        assert_eq!(
            burst(&mut late, WINDOW_NS + 1, FollowState::Following),
            NO_STEPS
        );
        let mut held = streaming();
        held.goal_ended(STEP_NS, Some(FollowState::RobotStop));
        assert_eq!(burst(&mut held, STEP_NS, FollowState::Following), NO_STEPS);
        let mut interrupted = streaming();
        interrupted.goal_ended(STEP_NS, Some(FollowState::Following));
        assert_eq!(
            interrupted.setpoint(None, STEP_NS, Some(FollowState::Following)),
            Some(PlanarTwist::ZERO)
        );
        assert_eq!(
            interrupted.setpoint(None, 2 * STEP_NS, Some(FollowState::RobotStop)),
            None
        );
        assert_eq!(
            burst(&mut interrupted, 3 * STEP_NS, FollowState::Following),
            NO_STEPS,
            "a hold report ended the burst"
        );
        let mut skipped = streaming();
        skipped.goal_ended(STEP_NS, Some(FollowState::Following));
        assert_eq!(
            skipped.setpoint(None, STEP_NS, Some(FollowState::Following)),
            Some(PlanarTwist::ZERO)
        );
        let past_ns = STEP_NS + WINDOW_NS + 1;
        assert_eq!(
            burst(&mut skipped, past_ns, FollowState::Following),
            NO_STEPS,
            "a step past 0.5 s ended the burst"
        );
    }

    // A goal replaced by a new goal brings the burst, which runs while
    // the new goal plans and ends at the new goal's first setpoint.
    #[test]
    fn a_replaced_goals_burst_runs_until_the_new_goals_first_setpoint() {
        let mut stream = streaming();
        stream.goal_ended(STEP_NS / 5, Some(FollowState::Following));
        let planning: Vec<Option<PlanarTwist>> = [STEP_NS, 2 * STEP_NS]
            .into_iter()
            .map(|now_ns| stream.setpoint(None, now_ns, Some(FollowState::Following)))
            .collect();
        assert_eq!(planning, [Some(PlanarTwist::ZERO); 2]);
        let first = PlanarTwist::new(0.2, 0.0, 0.1).expect("a finite twist");
        assert_eq!(
            stream.setpoint(Some(first), 3 * STEP_NS, Some(FollowState::Following)),
            Some(first)
        );
        assert_eq!(
            burst(&mut stream, 4 * STEP_NS, FollowState::Following),
            NO_STEPS
        );
    }

    // A goal that ends during a burst starts the burst again from that
    // ending: 0.25 s and at least 3 setpoints.
    #[test]
    fn a_goal_that_ends_during_a_burst_starts_it_again() {
        let mut stream = streaming();
        stream.goal_ended(STEP_NS, Some(FollowState::Following));
        let before: Vec<Option<PlanarTwist>> = steps(STEP_NS, 3 * STEP_NS)
            .into_iter()
            .map(|now_ns| stream.setpoint(None, now_ns, Some(FollowState::Following)))
            .collect();
        assert_eq!(before, [Some(PlanarTwist::ZERO); 3]);
        let later_ns = 4 * STEP_NS;
        stream.goal_ended(later_ns, Some(FollowState::Following));
        assert_eq!(
            burst(&mut stream, later_ns, FollowState::Following),
            steps(later_ns, later_ns + BURST_NS)
        );
    }

    /// A navigator with no goal in flight after a goal, accepted with the
    /// base at odom's origin, streamed a setpoint at 0: the grid around odom's
    /// origin rolls, as that goal's acceptance left it, and the newest
    /// drive_states reports following.
    fn after_a_goal_streamed() -> Navigator {
        let config = Arc::new(config());
        let mut link = Link::default();
        link.take(state(5, 10, 0.0, FollowState::Following), 0);
        let mut slot = GridSlot::default();
        slot.set(grid());
        slot.roll_from([0.0, 0.0])
            .expect("the grid centers on the base");
        let shared = Shared {
            config: config.clone(),
            clock: NodeClock::simulation().0,
            link: Arc::new(Mutex::new(link)),
            grid: Arc::new(Mutex::new(slot)),
            sensors: Arc::new(Sensors::new(vec![camera_slot("front")])),
        };
        let planning = Planning::new(Planner::default(), config.planner);
        Navigator {
            stream: streaming(),
            ..Navigator::new(shared, planning)
        }
    }

    /// Whether a reading with the base at x 2 m centers the grid of
    /// `navigator`, around odom's origin before it, on the base: the grid
    /// then holds x 4.4 m.
    fn a_reading_centers_the_grid(navigator: &Navigator) -> bool {
        let mut slot = lock_ignoring_poison(&navigator.shared.grid);
        read_at(&mut slot, 2.0, &[]);
        covers(slot.grid_mut().expect("a grid"), [4.4, 0.0])
    }

    // base_navigation's grid: once a goal ends, the grid centers on the base
    // with each reading while the release burst streams.
    #[tokio::test]
    async fn a_reading_during_the_release_burst_centers_the_grid_on_the_base() {
        let mut navigator = after_a_goal_streamed();
        navigator.goal_ended(STEP_NS, Some(FollowState::Following));
        assert_eq!(
            navigator.step(STEP_NS).await,
            Some(PlanarTwist::ZERO),
            "the burst streams"
        );
        assert!(a_reading_centers_the_grid(&navigator));
    }

    // base_navigation's grid: once no goal runs and no release burst
    // streams, a reading leaves the grid where it stands: after a burst that
    // ran to its end, after one that a hold report ended, and after an ending
    // during a hold, which brings no burst.
    #[tokio::test]
    async fn a_reading_once_the_burst_ends_leaves_the_grid_where_it_stands() {
        let mut ran_out = after_a_goal_streamed();
        ran_out.goal_ended(STEP_NS, Some(FollowState::Following));
        for now_ns in steps(STEP_NS, STEP_NS + BURST_NS) {
            assert_eq!(
                ran_out.step(now_ns).await,
                Some(PlanarTwist::ZERO),
                "at {now_ns} ns"
            );
        }
        assert!(!ran_out.stream.bursts(), "the burst ran to its end");
        assert!(!a_reading_centers_the_grid(&ran_out), "after the burst");

        let mut held = after_a_goal_streamed();
        held.goal_ended(STEP_NS, Some(FollowState::Following));
        assert_eq!(held.step(STEP_NS).await, Some(PlanarTwist::ZERO));
        lock_ignoring_poison(&held.shared.link)
            .take(state(5, 11, 0.0, FollowState::RobotStop), 2 * STEP_NS);
        assert_eq!(
            held.step(2 * STEP_NS).await,
            None,
            "the hold ends the burst"
        );
        assert!(!a_reading_centers_the_grid(&held), "after the hold");

        let mut no_burst = after_a_goal_streamed();
        no_burst.goal_ended(STEP_NS, Some(FollowState::RobotStop));
        assert!(
            !a_reading_centers_the_grid(&no_burst),
            "after an ending during a hold"
        );
    }
}
