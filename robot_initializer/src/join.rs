//! Joining a simulation: this robot attaches as the copy it runs as, naming
//! the model to stand and where, and holds that goal for as long as it is in
//! the scene.
//!
//! Its limbs pair to the engine through the backbone, and the copy the engine
//! reads on each pair is the one this goal named, so a setpoint reaches the
//! robot it was meant for and its measured state comes back on the same pair.
//!
//! The node's setup ends once the engine says that the robot stands, so the
//! node serves readiness, and the start of its copy ends, only for a robot
//! that stands in the scene. The wait for the engine's acceptance and the
//! wait for `standing` share one stand budget: the node's setup budget less
//! what its leave report needs, so a robot that does not stand fails with
//! its own reason before the daemon's limit. A robot that the engine refuses,
//! or whose goal the engine ends before the robot stands, fails the setup
//! with the engine's reason, and the join of its copy is undone. A goal that
//! the engine ends after the robot stood stops the node: `peppy stack list`
//! reports the instance finished, and `peppy stack join` puts the copy back.
//!
//! A task that outlives the setup sends the goal and holds it for the robot's
//! whole stay, and it cancels every goal that the engine accepted before it
//! lets go of it. A node that stops, or whose stand budget runs out, before
//! the engine answers waits for the answer and cancels the goal the engine
//! accepts. A node that stops takes its robot out and waits for the engine's
//! result, which arrives once the robot is out of the scene and its name is
//! free: removing a copy returns only then, so a copy joined straight back
//! under the same name finds its name free.

use std::sync::Arc;
use std::time::Duration;

use peppygen::consumed_actions::simulation::attach;
use peppygen::parameters::placement::Placement;
use peppygen::{NodeRunner, ProducerRef, QoSProfile, Result};
use peppylib::runtime::CancellationToken;
use tokio::sync::oneshot;
use tokio::time::{Instant, Interval, MissedTickBehavior};
use tracing::{info, warn};

use crate::identity::Identity;
use crate::refused;

/// How long the account of a stay the engine ended may take: the goal is
/// terminal, so the engine answers at once.
const RESULT_TIMEOUT: Duration = Duration::from_secs(2);
/// How long leaving may take in all: the engine acknowledging the cancel,
/// then its result saying the robot is out of the scene, which takes the
/// engine a scene rebuild.
const LEAVE_TIMEOUT: Duration = Duration::from_secs(4);
/// How long the shutdown hook waits for the robot to be out, inside peppy's
/// shutdown grace (5 s by default, `lifecycle.shutdown_grace_secs`).
const LEAVE_REPORT_TIMEOUT: Duration = Duration::from_millis(4500);
/// What the stand budget keeps back from the setup budget: the wait for the
/// leave report, and one second more, so a robot that does not stand has
/// left and the node has failed with its own reason before the daemon's
/// limit.
const LEAVE_RESERVE: Duration = Duration::from_millis(5500);
/// How often the wait for standing says that the robot does not stand yet.
/// Each line reaches the caller of the join as progress.
const PROGRESS_PERIOD: Duration = Duration::from_secs(10);
// The wait outlasts the leave it waits for, so a robot that does leave is
// not reported as one that did not, and the reserve outlasts the wait.
const _: () = assert!(LEAVE_REPORT_TIMEOUT.as_millis() > LEAVE_TIMEOUT.as_millis());
const _: () = assert!(LEAVE_RESERVE.as_millis() > LEAVE_REPORT_TIMEOUT.as_millis());

/// The clock that the wait for standing runs on: when to say that the robot
/// does not stand yet, and when the stand budget is spent. The node runs on
/// [`HostClock`]; a test drives the wait with a clock of its own.
pub trait StandClock: Send + 'static {
    /// The time since the robot's goal was sent.
    fn elapsed(&self) -> Duration;

    /// Resolves at the next moment of the wait. A future that is dropped
    /// before it resolves loses no moment.
    fn next(&mut self) -> impl Future<Output = Moment> + Send;
}

/// A moment of the wait for standing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Moment {
    /// The robot does not stand yet, this long after its goal was sent.
    Progress(Duration),
    /// The stand budget is spent.
    Spent,
}

/// The host's clock: a progress moment every [`PROGRESS_PERIOD`] after the
/// goal is sent, until the stand budget is spent.
pub struct HostClock {
    start: Instant,
    end: Instant,
    progress: Interval,
}

impl HostClock {
    /// A clock on the stand budget `budget`, from now.
    pub fn starting_now(budget: Duration) -> Self {
        Self::every(PROGRESS_PERIOD, budget)
    }

    fn every(period: Duration, budget: Duration) -> Self {
        let start = Instant::now();
        let mut progress = tokio::time::interval_at(start + period, period);
        progress.set_missed_tick_behavior(MissedTickBehavior::Skip);
        Self {
            start,
            end: start + budget,
            progress,
        }
    }
}

impl StandClock for HostClock {
    fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    async fn next(&mut self) -> Moment {
        // The end of the budget comes first, so a budget that is a whole
        // number of periods ends on its last moment rather than reporting it.
        tokio::select! {
            biased;
            () = tokio::time::sleep_until(self.end) => Moment::Spent,
            tick = self.progress.tick() => Moment::Progress(tick - self.start),
        }
    }
}

/// What the join of this robot came to, once the setup can go on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Joined {
    /// The robot drives its own hardware and joins no simulation.
    OwnHardware,
    /// The simulation stands the robot.
    Standing,
    /// The node was stopped before the robot stood.
    Stopped,
}

/// A duration as this node's logs give it: seconds, to the tenth, as `10 s`
/// or `174.5 s`.
fn seconds(duration: Duration) -> String {
    let tenths = (duration.as_millis() + 50) / 100;
    match tenths % 10 {
        0 => format!("{} s", tenths / 10),
        tenth => format!("{}.{tenth} s", tenths / 10),
    }
}

/// The time this robot has to stand, the acceptance included: the node's
/// setup budget less [`LEAVE_RESERVE`].
fn stand_budget(setup: Duration) -> std::result::Result<Duration, String> {
    setup
        .checked_sub(LEAVE_RESERVE)
        .filter(|budget| !budget.is_zero())
        .ok_or_else(|| {
            format!(
                "this node's setup budget of {} leaves no time to stand the robot: the node keeps {} of it to leave the simulation, so execution.setup_timeout_secs in its manifest must be more than that",
                seconds(setup),
                seconds(LEAVE_RESERVE)
            )
        })
}

/// Where the launcher asks this robot to stand, as the engine takes it.
/// `auto` leaves the spot to the engine. A spot that is not a number is
/// refused here, before it reaches the wire.
fn spot_of(
    placement: &Placement,
) -> std::result::Result<Option<attach::SimulationRobotAttachActionGoalPlacement>, String> {
    if placement.auto {
        return Ok(None);
    }
    let parts = [placement.x, placement.y, placement.z, placement.yaw];
    if !parts.iter().all(|part| part.is_finite()) {
        let [x, y, z, yaw] = parts;
        return Err(format!(
            "placement x, y, z and yaw must each be a finite number, and this robot's are {x}, {y}, {z} and {yaw}"
        ));
    }
    Ok(Some(attach::SimulationRobotAttachActionGoalPlacement {
        position: [placement.x, placement.y, placement.z],
        yaw: placement.yaw,
    }))
}

/// Joins the simulation that the launcher bound this robot to, standing the
/// model `identity` names under the name it gives the robot, and resolves
/// once the robot stands, with the wait on the clock that `clock` starts on
/// the stand budget. A robot with no simulation bound drives its own
/// hardware and returns at once.
pub async fn scene<C: StandClock>(
    identity: &Identity,
    placement: &Placement,
    runner: &Arc<NodeRunner>,
    clock: impl FnOnce(Duration) -> C + Send,
) -> Result<Joined> {
    let Some(simulation) = attach::bound_producer(runner).cloned() else {
        info!("no simulation stands this robot, so it joins none");
        return Ok(Joined::OwnHardware);
    };
    let placement = spot_of(placement).map_err(refused)?;
    let setup_budget = runner.setup_timeout();
    let budget = stand_budget(setup_budget).map_err(refused)?;

    let token = runner.cancellation_token().clone();
    let (stood, has_stood) = oneshot::channel();
    let (left, has_left) = oneshot::channel();
    // Registered before the goal is sent, so a stop at any moment of the
    // join waits for the robot to leave.
    runner.on_shutdown(wait_to_leave(token.clone(), has_left));
    let join = Join {
        runner: Arc::clone(runner),
        simulation,
        request: attach::GoalRequest {
            robot: identity.robot.clone(),
            model: identity.model.clone(),
            placement,
        },
        wire: setup_budget,
        budget,
        token: token.clone(),
    };
    tokio::spawn(stay(join, clock(budget), stood, left));

    tokio::select! {
        biased;
        () = token.cancelled() => Ok(Joined::Stopped),
        stand = has_stood => match stand {
            Ok(Ok(())) => Ok(Joined::Standing),
            Ok(Err(failure)) => Err(failure),
            Err(_) => Err(refused("the stay of this robot ended before it said whether the robot stands")),
        },
    }
}

/// A node that stops tells the robot's stay to end, and waits, inside
/// peppy's shutdown grace, for the robot to leave the scene.
async fn wait_to_leave(token: CancellationToken, has_left: oneshot::Receiver<()>) {
    token.cancel();
    if tokio::time::timeout(LEAVE_REPORT_TIMEOUT, has_left)
        .await
        .is_err()
    {
        warn!(
            "this robot did not leave the scene within {}",
            seconds(LEAVE_REPORT_TIMEOUT)
        );
    }
}

/// One robot's join: the goal it sends, and the budgets it waits on.
struct Join {
    runner: Arc<NodeRunner>,
    simulation: ProducerRef,
    request: attach::GoalRequest,
    /// How long the wire waits for the engine's answer to the goal: the
    /// setup budget, so the stand budget, which is shorter, ends that wait
    /// first.
    wire: Duration,
    /// The time the robot has to stand, the acceptance included.
    budget: Duration,
    token: CancellationToken,
}

impl Join {
    fn robot(&self) -> &str {
        &self.request.robot
    }

    fn not_standing_yet(&self, elapsed: Duration) {
        info!(
            "'{}' does not stand yet: {} of {}",
            self.robot(),
            seconds(elapsed),
            seconds(self.budget)
        );
    }

    fn out_of_budget(&self) -> peppygen::Error {
        refused(format!(
            "the simulation did not stand this robot within {}",
            seconds(self.budget)
        ))
    }
}

fn did_not_stand(why: impl std::fmt::Display) -> peppygen::Error {
    refused(format!("the simulation did not stand this robot: {why}"))
}

/// One robot's stay, from the goal it sends to the goal's end: it tells the
/// setup whether the robot stands, holds the goal for as long as the robot
/// is in the scene, and says when the robot has left.
async fn stay<C: StandClock>(
    join: Join,
    mut clock: C,
    stood: oneshot::Sender<Result<()>>,
    left: oneshot::Sender<()>,
) {
    let mut verdict = Verdict(Some(stood));
    if let Some(goal) = wait_to_stand(&join, &mut clock, &mut verdict).await {
        hold(goal, &join).await;
    }
    let _ = left.send(());
}

/// The setup's end of the wait for standing, told once whether the robot
/// stands. A node that stops is told nothing: its setup ends on the stop.
struct Verdict(Option<oneshot::Sender<Result<()>>>);

impl Verdict {
    fn give(&mut self, verdict: Result<()>) {
        if let Some(setup) = self.0.take() {
            let _ = setup.send(verdict);
        }
    }

    fn given(&self) -> bool {
        self.0.is_none()
    }
}

/// What the wait for standing hears while the goal runs.
enum Event {
    Stop,
    Clock(Moment),
    Feedback(Heard),
}

/// Sends the goal and waits for the robot to stand, within the stand budget,
/// and answers the goal of a robot that stands. Every other way out gives
/// the setup its verdict, except a stop, and cancels a goal that the engine
/// accepted and that still runs.
async fn wait_to_stand<C: StandClock>(
    join: &Join,
    clock: &mut C,
    verdict: &mut Verdict,
) -> Option<attach::ActionHandle> {
    let mut goal = admission(join, clock, verdict).await?;
    let robot = join.robot();
    // A node that stopped, or whose budget ran out, before the engine
    // answered takes back the goal the engine accepted.
    if join.token.is_cancelled() || verdict.given() {
        leave(&goal, robot, false).await;
        return None;
    }
    let Some(limbs) = goal.data.clone() else {
        leave(&goal, robot, false).await;
        verdict.give(Err(did_not_stand(
            "it admitted the robot without naming its limbs",
        )));
        return None;
    };
    info!(
        "'{robot}' was admitted to the simulation as {}; it starts once the simulation stands it, within {}",
        join.request.model,
        seconds(join.budget.saturating_sub(clock.elapsed()))
    );

    loop {
        let event = tokio::select! {
            biased;
            () = join.token.cancelled() => Event::Stop,
            feedback = goal.on_next_feedback_message() => Event::Feedback(heard(feedback)),
            moment = clock.next() => Event::Clock(moment),
        };
        match event {
            Event::Feedback(Heard::Standing) => {
                info!(
                    "'{robot}' stands in the simulation, with arms [{}] and grippers [{}]",
                    limbs.arm_names.join(", "),
                    limbs.gripper_names.join(", ")
                );
                verdict.give(Ok(()));
                return Some(goal);
            }
            Event::Feedback(Heard::NotStanding) => {}
            Event::Clock(Moment::Progress(elapsed)) => join.not_standing_yet(elapsed),
            Event::Feedback(Heard::End(StreamEnd::Ended)) => {
                verdict.give(Err(did_not_stand(reason_of(account(&goal).await))));
                return None;
            }
            Event::Feedback(Heard::End(StreamEnd::Gone)) => {
                verdict.give(Err(did_not_stand("the simulation is gone")));
                return None;
            }
            Event::Feedback(Heard::End(StreamEnd::Broken(e))) => {
                leave(&goal, robot, false).await;
                verdict.give(Err(did_not_stand(format!(
                    "its stand cannot be followed: {e}"
                ))));
                return None;
            }
            Event::Clock(Moment::Spent) => {
                leave(&goal, robot, false).await;
                verdict.give(Err(join.out_of_budget()));
                return None;
            }
            Event::Stop => {
                leave(&goal, robot, false).await;
                return None;
            }
        }
    }
}

/// Sends the goal and waits for the engine's answer, and answers the goal
/// that the engine accepted. A refusal or a wire error gives the setup its
/// verdict. Neither a stop nor the end of the stand budget ends this wait:
/// the budget gives the setup its verdict at once, and the wait goes on, so
/// a goal that the engine accepts after either can be cancelled.
async fn admission<C: StandClock>(
    join: &Join,
    clock: &mut C,
    verdict: &mut Verdict,
) -> Option<attach::ActionHandle> {
    let send = attach::ActionHandle::fire_goal(
        &join.runner,
        &join.simulation,
        join.wire,
        join.request.clone(),
        QoSProfile::Reliable,
    );
    let mut send = std::pin::pin!(send);
    let mut spent = false;
    let answer = loop {
        tokio::select! {
            biased;
            answer = &mut send => break answer,
            moment = clock.next(), if !spent => match moment {
                Moment::Progress(elapsed) => join.not_standing_yet(elapsed),
                Moment::Spent => {
                    spent = true;
                    verdict.give(Err(join.out_of_budget()));
                }
            },
        }
    };
    let goal = match answer {
        Ok(goal) => goal,
        Err(e) => {
            verdict.give(Err(e));
            return None;
        }
    };
    if !goal.accepted {
        verdict.give(Err(refused(format!(
            "the simulation refused this robot: {}",
            goal.reason.unwrap_or_else(|| "no reason given".into())
        ))));
        return None;
    }
    Some(goal)
}

/// Holds the place of a robot that stands, for as long as the node runs: the
/// goal ends when the engine takes the robot out, which stops the node,
/// because a robot that is not in the scene has no readiness to serve, and
/// a node that stops takes the robot out.
async fn hold(mut goal: attach::ActionHandle, join: &Join) {
    let robot = join.robot();
    let end = tokio::select! {
        biased;
        () = join.token.cancelled() => None,
        end = stay_ends(&mut goal) => Some(end),
    };
    match end {
        None => leave(&goal, robot, true).await,
        Some(StreamEnd::Ended) => report(robot, true, account(&goal).await),
        Some(StreamEnd::Gone) => {
            warn!("the simulation standing '{robot}' is gone");
            report(robot, true, account(&goal).await);
        }
        Some(StreamEnd::Broken(e)) => {
            warn!("the stay of '{robot}' cannot be followed: {e}");
            leave(&goal, robot, true).await;
        }
    }
    // However the stay ended, the node stops with it.
    join.token.cancel();
}

/// Resolves once the engine's side of the stay is over: the stream closes
/// when the goal ends or the engine is gone.
async fn stay_ends(goal: &mut attach::ActionHandle) -> StreamEnd {
    loop {
        if let Heard::End(end) = heard(goal.on_next_feedback_message().await) {
            return end;
        }
    }
}

/// What one read of the goal's feedback says.
#[derive(Debug, PartialEq, Eq)]
enum Heard {
    /// The engine says that the robot stands.
    Standing,
    /// A message that does not say so.
    NotStanding,
    /// The stream is over.
    End(StreamEnd),
}

/// How the goal's feedback stream ended.
#[derive(Debug, PartialEq, Eq)]
enum StreamEnd {
    /// The engine ended the goal, and the stream closed with it.
    Ended,
    /// The engine is gone.
    Gone,
    /// The stream broke, so the stand cannot be followed.
    Broken(String),
}

fn heard(feedback: Result<attach::FeedbackMessage>) -> Heard {
    match feedback {
        Ok(feedback) if feedback.standing => Heard::Standing,
        Ok(_) => Heard::NotStanding,
        Err(peppygen::Error::ActionFeedbackChannelClosed) => Heard::End(StreamEnd::Ended),
        Err(peppygen::Error::ActionFeedbackProducerGone { .. }) => Heard::End(StreamEnd::Gone),
        Err(e) => Heard::End(StreamEnd::Broken(e.to_string())),
    }
}

/// The engine's account of a stay that has ended.
async fn account(
    goal: &attach::ActionHandle,
) -> std::result::Result<attach::ResultOutcome, String> {
    goal.get_result(RESULT_TIMEOUT)
        .await
        .map(|result| result.outcome)
        .map_err(|e| e.to_string())
}

/// Takes the robot out of the scene: cancelling the goal asks the engine to,
/// and the engine's result says the robot is out and its name is free. Both
/// share [`LEAVE_TIMEOUT`].
async fn leave(goal: &attach::ActionHandle, robot: &str, stood: bool) {
    let deadline = Instant::now() + LEAVE_TIMEOUT;
    let remaining = || deadline.saturating_duration_since(Instant::now());
    if let Err(e) = goal.cancel_goal(remaining()).await {
        warn!("'{robot}' could not tell the simulation it is leaving: {e}");
        return;
    }
    match goal.get_result(remaining()).await {
        Ok(result) => report(robot, stood, Ok(result.outcome)),
        Err(e) => warn!("the simulation did not say '{robot}' is out of the scene: {e}"),
    }
}

/// Whether a goal ended as asked, and what the engine said of it.
fn said(outcome: attach::ResultOutcome) -> (bool, String) {
    match outcome {
        attach::ResultOutcome::Completed(data) | attach::ResultOutcome::Cancelled(data) => {
            (data.success, data.message)
        }
        attach::ResultOutcome::Abandoned => (false, "the simulation abandoned it".to_owned()),
        attach::ResultOutcome::Expired => (false, "it left before the reason was read".to_owned()),
    }
}

/// Why the engine ended a goal before the robot stood.
fn reason_of(outcome: std::result::Result<attach::ResultOutcome, String>) -> String {
    match outcome {
        Ok(outcome) => said(outcome).1,
        Err(e) => format!("it ended the goal and did not say why: {e}"),
    }
}

/// How a robot's stay ended, with what the engine said of it.
#[derive(Debug, PartialEq, Eq)]
enum Ending {
    /// The robot left before the engine stood it.
    LeftBeforeStanding(String),
    /// The engine took the robot out of the scene, as it was asked to.
    TakenOut(String),
    /// The engine took the robot out for a reason of its own.
    Dropped(String),
    /// The engine did not say what became of the robot.
    Unsaid(String),
}

/// What the engine's account says became of a robot that `stood` or never
/// did.
fn ending(stood: bool, outcome: std::result::Result<attach::ResultOutcome, String>) -> Ending {
    let (asked_for, message) = match outcome {
        Err(e) => return Ending::Unsaid(e),
        Ok(outcome) => said(outcome),
    };
    match (stood, asked_for) {
        (false, _) => Ending::LeftBeforeStanding(message),
        (true, true) => Ending::TakenOut(message),
        (true, false) => Ending::Dropped(message),
    }
}

/// Reports how this robot's stay ended.
fn report(robot: &str, stood: bool, outcome: std::result::Result<attach::ResultOutcome, String>) {
    match ending(stood, outcome) {
        Ending::LeftBeforeStanding(why) => {
            info!("'{robot}' left the simulation before it stood: {why}");
        }
        Ending::TakenOut(why) => {
            info!("the simulation took '{robot}' out of the scene: {why}");
        }
        Ending::Dropped(why) => warn!("the simulation took '{robot}' out of the scene: {why}"),
        Ending::Unsaid(why) => warn!("the simulation did not say why '{robot}' left: {why}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_said(success: bool, message: &str) -> attach::ResultResponseData {
        attach::ResultResponseData {
            success,
            message: message.to_owned(),
        }
    }

    #[test]
    fn a_stay_is_reported_by_whether_the_robot_ever_stood() {
        // A robot that leaves before the engine stands it left, whatever the
        // engine's own verdict on the goal.
        assert_eq!(
            ending(
                false,
                Ok(attach::ResultOutcome::Cancelled(engine_said(
                    true,
                    "left before it stood"
                )))
            ),
            Ending::LeftBeforeStanding("left before it stood".to_owned())
        );
        assert_eq!(
            ending(false, Ok(attach::ResultOutcome::Abandoned)),
            Ending::LeftBeforeStanding("the simulation abandoned it".to_owned())
        );

        // A robot that stood: the engine took it out as asked, or for a
        // reason of its own.
        assert_eq!(
            ending(
                true,
                Ok(attach::ResultOutcome::Cancelled(engine_said(true, "left")))
            ),
            Ending::TakenOut("left".to_owned())
        );
        assert_eq!(
            ending(
                true,
                Ok(attach::ResultOutcome::Completed(engine_said(
                    false, "lapsed"
                )))
            ),
            Ending::Dropped("lapsed".to_owned())
        );
        assert_eq!(
            ending(true, Ok(attach::ResultOutcome::Expired)),
            Ending::Dropped("it left before the reason was read".to_owned())
        );
        assert_eq!(
            ending(true, Err("timed out".to_owned())),
            Ending::Unsaid("timed out".to_owned())
        );
    }

    #[test]
    fn a_goal_the_engine_ended_before_the_robot_stood_gives_the_engines_reason() {
        assert_eq!(
            reason_of(Ok(attach::ResultOutcome::Completed(engine_said(
                false,
                "its files could not be fetched"
            )))),
            "its files could not be fetched"
        );
        assert_eq!(
            reason_of(Ok(attach::ResultOutcome::Abandoned)),
            "the simulation abandoned it"
        );
        assert_eq!(
            reason_of(Err("timed out".to_owned())),
            "it ended the goal and did not say why: timed out"
        );
    }

    #[test]
    fn the_feedback_says_whether_the_robot_stands_and_how_its_stream_ended() {
        let standing = |standing| Ok(attach::FeedbackMessage { standing });
        assert_eq!(heard(standing(true)), Heard::Standing);
        assert_eq!(heard(standing(false)), Heard::NotStanding);
        assert_eq!(
            heard(Err(peppygen::Error::ActionFeedbackChannelClosed)),
            Heard::End(StreamEnd::Ended)
        );
        assert_eq!(
            heard(Err(peppygen::Error::ActionFeedbackProducerGone {
                instance_id: Some("simulation_inst".to_owned()),
                action_name: "attach".to_owned(),
            })),
            Heard::End(StreamEnd::Gone)
        );
        // Anything else breaks the stream: the stand cannot be followed, and
        // the wait for standing leaves the simulation and fails on it, as it
        // does when the stand budget is spent.
        assert_eq!(
            heard(Err(refused("a feedback message that does not decode"))),
            Heard::End(StreamEnd::Broken(
                "a feedback message that does not decode".to_owned()
            ))
        );
    }

    #[test]
    fn durations_are_given_in_seconds_to_the_tenth() {
        assert_eq!(seconds(Duration::from_secs(10)), "10 s");
        assert_eq!(seconds(Duration::from_millis(174_500)), "174.5 s");
        assert_eq!(seconds(Duration::from_millis(164_460)), "164.5 s");
        assert_eq!(seconds(Duration::from_millis(4_500)), "4.5 s");
        assert_eq!(seconds(Duration::from_millis(40)), "0 s");
    }

    #[test]
    fn the_stand_budget_is_the_setup_budget_less_the_leave_reserve() {
        assert_eq!(
            stand_budget(Duration::from_secs(180)),
            Ok(Duration::from_millis(174_500))
        );
        assert_eq!(
            stand_budget(Duration::from_secs(6)),
            Ok(Duration::from_millis(500))
        );
        for setup in [Duration::from_secs(5), LEAVE_RESERVE] {
            let refusal = stand_budget(setup).unwrap_err();
            assert!(
                refusal.contains("leaves no time to stand the robot")
                    && refusal.contains("execution.setup_timeout_secs"),
                "{refusal}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn the_host_clock_says_progress_every_period_until_the_budget_is_spent() {
        let mut clock = HostClock::every(Duration::from_secs(10), Duration::from_millis(25_500));
        assert_eq!(
            clock.next().await,
            Moment::Progress(Duration::from_secs(10))
        );
        assert_eq!(clock.elapsed(), Duration::from_secs(10));
        assert_eq!(
            clock.next().await,
            Moment::Progress(Duration::from_secs(20))
        );
        assert_eq!(clock.next().await, Moment::Spent);
        assert_eq!(clock.elapsed(), Duration::from_millis(25_500));
        // A spent budget stays spent.
        assert_eq!(clock.next().await, Moment::Spent);
    }

    #[tokio::test(start_paused = true)]
    async fn a_budget_of_whole_periods_ends_on_its_last_moment() {
        let mut clock = HostClock::every(Duration::from_secs(10), Duration::from_secs(20));
        assert_eq!(
            clock.next().await,
            Moment::Progress(Duration::from_secs(10))
        );
        assert_eq!(clock.next().await, Moment::Spent);
    }

    #[tokio::test(start_paused = true)]
    async fn the_node_says_progress_every_ten_seconds() {
        let mut clock = HostClock::starting_now(Duration::from_millis(174_500));
        assert_eq!(
            clock.next().await,
            Moment::Progress(Duration::from_secs(10))
        );
        assert_eq!(
            clock.next().await,
            Moment::Progress(Duration::from_secs(20))
        );
    }
}
