//! The members of `workflow_director:v1`. The services answer from the kept
//! state of the director: its fleet snapshot, its runs and its journals. The
//! actions follow a run: `run` from its start, `resume` from a resume and
//! `abort` from an abort, each to the end of the run, and the first two to
//! its suspension as well.

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value as Json;

use peppygen::emitted_topics::director::run_events;
use peppygen::exposed_actions::director::{abort, locations, resume, run};
use peppygen::exposed_services::director::{
    check, confirm, journal, replay, robots, runs, status, suspend,
};
use serde_json::json;
use tokio::sync::{broadcast, mpsc, oneshot, watch};
use workflow_documents::result::{Ending, RUN_SUSPENDED, RunRefusal, RunResult};
use workflow_documents::{EventKind, KnownLocation, Name, RobotName, Workflow, run_lease_name};

use crate::Snapshot;
use crate::checker;
use crate::director::{Caller, Director, RunRequest, RunView, TeachHold, parses};
use crate::engine::failure::{PRECONDITION_FAILED, REJECTED};
use crate::engine::host::Event;
use crate::engine::{Phase, ResumeRefusal, RunCommand, status_of};
use crate::fleet::{self, Robot};

/// How often a goal that follows a run gives its progress, in wall time.
const PROGRESS_EVERY: Duration = Duration::from_secs(5);
/// The pause after an error of the transport before a member serves
/// again, so a broken transport cannot spin a loop or flood the log.
const RETRY_BACKOFF: Duration = Duration::from_secs(1);

/// Starts every member of the director but `run_events`, which `publish_run_events` gives from the start.
pub async fn serve(director: Arc<Director>) -> peppygen::Result<()> {
    serve_services(&director);
    serve_run(Arc::clone(&director)).await?;
    serve_resume(Arc::clone(&director)).await?;
    serve_locations(Arc::clone(&director)).await?;
    serve_abort(director).await?;
    Ok(())
}

/// Gives each event of each run on `run_events`, from before the director restores the runs of its journals, so
/// the `run_ends` of each run that a restart interrupted reaches its subscribers.
pub async fn publish_run_events(director: &Arc<Director>) -> peppygen::Result<()> {
    let publisher = run_events::declare_publisher(&director.runner).await?;
    let mut events = director.events.subscribe();
    tokio::spawn(async move {
        loop {
            let (run, event) = match events.recv().await {
                Ok(item) => item,
                Err(broadcast::error::RecvError::Lagged(missed)) => {
                    tracing::warn!("run_events missed {missed} events");
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => return,
            };
            let message = run_events::build_message(
                run,
                event.role,
                event.kind.name().to_owned(),
                event.step,
                event.failure_kind,
                event.message,
                event.details.to_string(),
            );
            match message {
                Ok(message) => {
                    if let Err(error) = publisher.publish(message).await {
                        tracing::warn!("run_events: {error}");
                    }
                }
                Err(error) => tracing::warn!("run_events: {error}"),
            }
        }
    });
    Ok(())
}

/// Answers each request of a service with `handler`, for the life of the
/// node.
macro_rules! serve_service {
    ($director:expr, $service:ident, $handler:expr) => {{
        let director = Arc::clone($director);
        tokio::spawn(async move {
            let runner = Arc::clone(&director.runner);
            loop {
                let director = Arc::clone(&director);
                if let Err(error) = $service::handle_next_request(&runner, move |request| {
                    $handler(&director, request)
                })
                .await
                {
                    tracing::warn!(concat!(stringify!($service), ": {}"), error);
                    tokio::time::sleep(RETRY_BACKOFF).await;
                }
            }
        });
    }};
}

fn serve_services(director: &Arc<Director>) {
    serve_service!(director, robots, answer_robots);
    serve_service!(director, check, answer_check);
    serve_service!(director, confirm, answer_confirm);
    serve_service!(director, suspend, answer_suspend);
    serve_service!(director, runs, answer_runs);
    serve_service!(director, status, answer_status);
    serve_service!(director, journal, answer_journal);
    serve_service!(director, replay, answer_replay);
}

/// The fleet snapshot of the director once it read the first listing, and
/// before that the message of the `resource_unavailable` refusal.
fn listed(director: &Director) -> Result<Arc<Snapshot>, &'static str> {
    let snapshot = director.snapshots.borrow().clone();
    match snapshot.ready {
        true => Ok(snapshot),
        false => Err("the director has not read the fleet listing yet"),
    }
}

fn answer_robots(
    director: &Director,
    _request: robots::Request,
) -> Result<robots::Response, robots::Failure> {
    let snapshot = listed(director).map_err(|message| {
        robots::Failure::new(robots::FailureKind::ResourceUnavailable, message)
    })?;
    let allowed = &director.config.allowed;
    let fleet = &snapshot.fleet;
    let robots = fleet
        .robots
        .values()
        .map(|robot| {
            let lease = fleet.lease_of(&robot.name);
            let contracts = robot
                .contracts
                .iter()
                .filter(|c| allowed.allows_name_tag(&c.name, &c.tag))
                .map(|c| robots::ResponseRobotsItemContractsItem {
                    name: c.name.clone(),
                    tag: c.tag.clone(),
                    sha256: c.sha256.clone(),
                    instances: c.instances.clone(),
                })
                .collect();
            robots::ResponseRobotsItem {
                name: robot.name.clone(),
                kind: robot.kind.clone(),
                model: snapshot
                    .models
                    .get(&robot.name)
                    .cloned()
                    .unwrap_or_default(),
                robot_commander: robot.robot_commander.clone(),
                software: robot.software.clone(),
                lease_holder: lease.map(|l| l.holder.clone()).unwrap_or_default(),
                lease_name: lease.map(|l| l.name.clone()).unwrap_or_default(),
                lease_open: lease.is_some_and(|l| l.open),
                contracts,
            }
        })
        .collect();
    let documents = snapshot
        .contracts
        .values()
        .filter(|shape| allowed.allows_name_tag(&shape.name, &shape.tag))
        .map(|shape| robots::ResponseDocumentsItem {
            name: shape.name.clone(),
            tag: shape.tag.clone(),
            sha256: shape.sha256.clone(),
            document: shape.document.clone(),
        })
        .collect();
    Ok(robots::Response::new(fleet.halted, robots, documents))
}

fn answer_check(
    director: &Director,
    request: check::Request,
) -> Result<check::Response, check::Failure> {
    let located = |at: String, message: String| check::ResponseErrorsItem { at, message };
    let workflow = match Workflow::parse(&request.data.workflow) {
        Ok(workflow) => workflow,
        Err(refusals) => {
            let errors = refusals
                .iter()
                .map(|r| r.located())
                .map(|l| located(l.at, l.message))
                .collect();
            return Ok(check::Response::new(false, errors));
        }
    };
    let binding = match request.data.roles.as_deref().map(roles_of).transpose() {
        Ok(binding) => binding,
        Err(message) => return Err(check::Failure::new(check::FailureKind::Rejected, message)),
    };
    let snapshot = listed(director)
        .map_err(|message| check::Failure::new(check::FailureKind::ResourceUnavailable, message))?;
    let errors: Vec<_> = checker::check(
        &workflow,
        &snapshot,
        &director.config.allowed,
        binding.as_ref(),
    )
    .into_iter()
    .map(|l| located(l.at, l.message))
    .collect();
    Ok(check::Response::new(errors.is_empty(), errors))
}

/// The roles a `check` binds: a JSON5 map of role to robot name, with no
/// `"auto"`.
fn roles_of(text: &str) -> Result<BTreeMap<Name, RobotName>, String> {
    let value = parses(text).map_err(|reason| format!("roles: {reason}"))?;
    let map = value
        .as_object()
        .ok_or("roles is a map of each role to its robot")?;
    map.iter()
        .map(|(role, robot)| {
            let role = Name::parse(role).map_err(|e| format!("roles: {e}"))?;
            let robot = robot
                .as_str()
                .ok_or(format!("roles.{role} names a robot"))?;
            if robot == "auto" {
                return Err(format!(
                    "roles.{role}: check binds named robots; `auto` binds at a run"
                ));
            }
            let robot = RobotName::parse(robot).map_err(|e| format!("roles.{role}: {e}"))?;
            Ok((role, robot))
        })
        .collect()
}

fn answer_confirm(director: &Director, request: confirm::Request) -> Result<(), confirm::Failure> {
    let failure = |kind, message: String| confirm::Failure::new(kind, message);
    let run = &request.data.run;
    let RunView {
        phase,
        commands,
        workflow,
        ..
    } = director
        .view_of(run)
        .ok_or_else(|| failure(confirm::FailureKind::Rejected, format!("no run `{run}`")))?;
    let assumption = Name::parse(&request.data.assumption)
        .map_err(|e| failure(confirm::FailureKind::Rejected, format!("assumption: {e}")))?;
    if let Some(workflow) = &workflow
        && !workflow.assumptions.contains_key(&assumption)
    {
        return Err(failure(
            confirm::FailureKind::Rejected,
            format!("the workflow has no assumption `{assumption}`"),
        ));
    }
    let Phase::Confirming(to_confirm) = phase else {
        return Err(failure(
            confirm::FailureKind::PreconditionFailed,
            format!("the run `{run}` waits for no confirmation"),
        ));
    };
    if !to_confirm.iter().any(|item| item.assumption == assumption) {
        return Err(failure(
            confirm::FailureKind::PreconditionFailed,
            format!("`{assumption}` waits for no confirmation: it has a check, or it is confirmed"),
        ));
    }
    send(commands, RunCommand::Confirm { assumption })
        .map_err(|message| failure(confirm::FailureKind::PreconditionFailed, message))?;
    Ok(())
}

fn answer_suspend(director: &Director, request: suspend::Request) -> Result<(), suspend::Failure> {
    let run = &request.data.run;
    let RunView {
        phase, commands, ..
    } = director.view_of(run).ok_or_else(|| {
        suspend::Failure::new(suspend::FailureKind::Rejected, format!("no run `{run}`"))
    })?;
    if phase != Phase::Running {
        return Err(suspend::Failure::new(
            suspend::FailureKind::PreconditionFailed,
            format!(
                "the run `{run}` is {}, and only a running run suspends",
                state_name(&phase)
            ),
        ));
    }
    send(commands, RunCommand::Suspend).map_err(|message| {
        suspend::Failure::new(suspend::FailureKind::PreconditionFailed, message)
    })?;
    Ok(())
}

fn send(
    commands: Option<mpsc::UnboundedSender<RunCommand>>,
    command: RunCommand,
) -> Result<(), String> {
    commands
        .ok_or("the run has ended".to_owned())?
        .send(command)
        .map_err(|_| "the run has ended".to_owned())
}

fn state_name(phase: &Phase) -> String {
    serde_json::to_value(phase.state())
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

fn answer_runs(
    director: &Director,
    _request: runs::Request,
) -> Result<runs::Response, runs::Failure> {
    let snapshot = director.snapshots.borrow().clone();
    let runs = director
        .runs
        .lock()
        .expect("no run panics holding the runs")
        .iter()
        .map(|(run, entry)| {
            let phase = entry.phase.borrow().clone();
            let robots = entry
                .robots
                .iter()
                .map(|(role, robot)| {
                    let lease = snapshot
                        .fleet
                        .lease_of(robot.as_str())
                        .filter(|l| l.name == run_lease_name(run));
                    runs::ResponseRunsItemRobotsItem {
                        role: role.to_string(),
                        robot: robot.to_string(),
                        leased: lease.is_some(),
                        open: lease.is_some_and(|l| l.open),
                    }
                })
                .collect();
            runs::ResponseRunsItem {
                run: run.clone(),
                workflow: entry
                    .workflow
                    .as_ref()
                    .map(|w| w.manifest.name.clone())
                    .unwrap_or_default(),
                revision: entry
                    .workflow
                    .as_ref()
                    .map_or(0, |w| w.manifest.revision.get()),
                state: state_name(&phase),
                started_by: entry.started_by.clone(),
                robots,
            }
        })
        .collect::<Vec<_>>();
    Ok(runs::Response::new(newest_first(runs)))
}

/// The runs by their number, newest first.
fn newest_first(mut runs: Vec<runs::ResponseRunsItem>) -> Vec<runs::ResponseRunsItem> {
    runs.sort_by_key(|item| std::cmp::Reverse(crate::journal::run_number(&item.run)));
    runs
}

fn answer_status(
    director: &Director,
    request: status::Request,
) -> Result<status::Response, status::Failure> {
    let run = &request.data.run;
    let RunView { phase, .. } = director.view_of(run).ok_or_else(|| {
        status::Failure::new(
            status::FailureKind::Rejected,
            format!("no run `{run}`: it never ran, or its journal was pruned"),
        )
    })?;
    let details = serde_json::to_string(&status_of(&phase)).expect("a run status serializes");
    Ok(status::Response::new(state_name(&phase), details))
}

fn answer_journal(
    director: &Director,
    request: journal::Request,
) -> Result<journal::Response, journal::Failure> {
    let run = &request.data.run;
    director
        .journals
        .text(run)
        .map(journal::Response::new)
        .map_err(|_| {
            journal::Failure::new(
                journal::FailureKind::Rejected,
                format!("no journal of `{run}`"),
            )
        })
}

fn answer_replay(
    director: &Director,
    request: replay::Request,
) -> Result<replay::Response, replay::Failure> {
    let rejected = |message: String| replay::Failure::new(replay::FailureKind::Rejected, message);
    let text = match (request.data.run, request.data.journal) {
        (Some(run), None) => director
            .journals
            .text(&run)
            .map_err(|_| rejected(format!("no journal of `{run}`")))?,
        (None, Some(journal)) => journal,
        _ => return Err(rejected("give a run or a journal".to_owned())),
    };
    let outcome = crate::replay::replay(&text).map_err(rejected)?;
    Ok(replay::Response::new(
        outcome.same,
        outcome.decisions,
        outcome.difference,
    ))
}

/// A goal that follows a run: the generated goal of `run`, `resume` or
/// `abort`, which carry the same feedback and the same result.
trait RunGoal: Send + Sync {
    fn feedback(&self, run: &str, event: &Event) -> impl Future<Output = ()> + Send;
    fn completed(&self, run: &str, result: &RunResult) -> impl Future<Output = ()> + Send;
    fn aborted(&self, run: &str, result: &RunResult) -> impl Future<Output = ()> + Send;
    fn failed(&self, kind: &str, message: String) -> impl Future<Output = ()> + Send;
    fn cancelled(&self) -> impl Future<Output = ()> + Send;
    fn sender(&self) -> Caller;

    /// Ends the goal as the run ended: completed, cancelled for an abort,
    /// and failed with the kind of every other ending.
    fn ended(&self, run: &str, result: &RunResult) -> impl Future<Output = ()> + Send {
        async move {
            match result.ending {
                Ending::Completed => self.completed(run, result).await,
                Ending::Aborted => self.aborted(run, result).await,
                ending => {
                    let message = result
                        .refusal
                        .as_ref()
                        .map(refusal_message)
                        .or_else(|| {
                            result
                                .failed
                                .as_ref()
                                .map(|f| format!("{}: {}", f.step, f.message))
                        })
                        .unwrap_or_else(|| {
                            format!("the run `{run}` ended {}", ending_name(ending))
                        });
                    self.failed(ending.failure_kind().unwrap_or_default(), message)
                        .await;
                }
            }
        }
    }
}

macro_rules! run_goal {
    ($member:ident) => {
        impl RunGoal for $member::GoalContext {
            async fn feedback(&self, run: &str, event: &Event) {
                let sent = self
                    .publish_feedback(
                        run.to_owned(),
                        event.role.clone(),
                        event.kind.name().to_owned(),
                        event.step.clone(),
                        event.failure_kind.clone(),
                        event.message.clone(),
                        event.details.to_string(),
                    )
                    .await;
                if let Err(error) = sent {
                    tracing::warn!("feedback of {run}: {error}");
                }
            }

            async fn completed(&self, run: &str, result: &RunResult) {
                let _ = self.complete(run.to_owned(), result_text(result)).await;
            }

            async fn aborted(&self, run: &str, result: &RunResult) {
                let _ = self
                    .complete_cancelled("aborted", run.to_owned(), result_text(result))
                    .await;
            }

            async fn failed(&self, kind: &str, message: String) {
                let kind = $member::FailureKind::parse(kind)
                    .unwrap_or($member::FailureKind::AttemptFailed);
                let _ = self.fail(kind, message).await;
            }

            async fn cancelled(&self) {
                self.cancel_signal().await
            }

            fn sender(&self) -> Caller {
                let sender = $member::GoalContext::sender(self);
                Caller {
                    instance: sender.instance_id.to_string(),
                    core_node: sender.core_node.to_string(),
                }
            }
        }
    };
}

run_goal!(run);
run_goal!(resume);
run_goal!(abort);

fn result_text(result: &RunResult) -> String {
    serde_json::to_string(result).expect("a run result serializes")
}

/// How a goal follows its run.
enum Follow {
    /// To the end or a suspension of the run; a cancel of the goal aborts
    /// the run through its `commands`, while the run takes commands.
    ToSuspension {
        commands: Option<mpsc::UnboundedSender<RunCommand>>,
    },
    /// To the end of the run, through its suspensions.
    ToEnd,
}

/// How long a goal waits for an event that reaches the events of the director a moment after what the goal follows:
/// the suspension that names the refusal of a resume, which the engine gives before its answer, and the `run_ends`
/// of a run, which follows its phase `Ended`.
const EVENT_IN_FLIGHT: Duration = Duration::from_secs(1);

/// Forwards the events of `run` to `goal` up to its event of `kind`.
async fn forward_until<G: RunGoal>(
    goal: &G,
    run: &str,
    events: &mut broadcast::Receiver<(String, Event)>,
    kind: EventKind,
) {
    loop {
        match events.recv().await {
            Ok((of, event)) if of == run => {
                goal.feedback(run, &event).await;
                if event.kind == kind {
                    return;
                }
            }
            Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
            Err(broadcast::error::RecvError::Closed) => return,
        }
    }
}

/// Follows `run` for `goal`: each of its events as feedback, its progress
/// every [`PROGRESS_EVERY`], and the end of the goal as the run ends.
async fn follow<G: RunGoal>(
    goal: G,
    run: String,
    mut events: broadcast::Receiver<(String, Event)>,
    mut phase: watch::Receiver<Phase>,
    how: Follow,
) {
    let aborts = match &how {
        Follow::ToSuspension { commands } => commands.clone(),
        Follow::ToEnd => None,
    };
    // The first progress comes a period after the goal: the run's first
    // event is its own.
    let mut progress =
        tokio::time::interval_at(tokio::time::Instant::now() + PROGRESS_EVERY, PROGRESS_EVERY);
    let mut cancel_sent = false;
    loop {
        tokio::select! {
            item = events.recv() => match item {
                Ok((of, event)) if of == run => {
                    goal.feedback(&run, &event).await;
                    if end_goal(&goal, &run, &event, &how).await {
                        return;
                    }
                }
                Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => return,
            },
            changed = phase.changed() => {
                if changed.is_err() {
                    return;
                }
                let ended = match &*phase.borrow() {
                    Phase::Ended(result) => Some((**result).clone()),
                    _ => None,
                };
                if let Some(result) = ended {
                    let end = forward_until(&goal, &run, &mut events, EventKind::RunEnds);
                    if tokio::time::timeout(EVENT_IN_FLIGHT, end).await.is_err() {
                        tracing::warn!("the run_ends event of {run} did not come");
                    }
                    goal.ended(&run, &result).await;
                    return;
                }
            },
            _ = progress.tick() => {
                let status = status_of(&phase.borrow().clone());
                let event = Event { role: String::new(), kind: EventKind::Progress, step: String::new(), failure_kind: String::new(), message: String::new(), details: json!(status) };
                goal.feedback(&run, &event).await;
            },
            _ = goal.cancelled(), if aborts.is_some() && !cancel_sent => {
                cancel_sent = true;
                if let Some(commands) = &aborts {
                    let _ = commands.send(RunCommand::Abort);
                }
            },
        }
    }
}

/// Ends the goal at the event that ends what it follows; true when it did.
async fn end_goal<G: RunGoal>(goal: &G, run: &str, event: &Event, how: &Follow) -> bool {
    match event.kind {
        EventKind::RunEnds => {
            let Ok(result) = serde_json::from_value::<RunResult>(event.details.clone()) else {
                goal.failed(
                    "attempt_failed",
                    "the run ended with a result that did not read".to_owned(),
                )
                .await;
                return true;
            };
            goal.ended(run, &result).await;
            true
        }
        EventKind::RunSuspends if matches!(how, Follow::ToSuspension { .. }) => {
            goal.failed(
                RUN_SUSPENDED,
                format!("the run `{run}` is suspended: resume or abort it"),
            )
            .await;
            true
        }
        _ => false,
    }
}

/// The message of a refused run: what refused it, then each error with
/// where it stands, as the run result gives them.
fn refusal_message(refusal: &RunRefusal) -> String {
    let errors: Vec<String> = refusal.errors.iter().map(ToString::to_string).collect();
    match errors.is_empty() {
        true => refusal.message.clone(),
        false => format!("{}: {}", refusal.message, errors.join("; ")),
    }
}

fn ending_name(ending: Ending) -> String {
    serde_json::to_value(ending)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

async fn serve_run(director: Arc<Director>) -> peppygen::Result<()> {
    let mut handle = run::ActionHandle::expose(&director.runner).await?;
    tokio::spawn(async move {
        loop {
            let goal = match handle.handle_goal_next_request(decide_run).await {
                Ok(Some(goal)) => goal,
                Ok(None) => return,
                Err(error) => {
                    tracing::warn!("run: {error}");
                    tokio::time::sleep(RETRY_BACKOFF).await;
                    continue;
                }
            };
            let data = goal.request().data.clone();
            let caller = RunGoal::sender(&goal);
            let request = RunRequest {
                workflow: data.workflow,
                bundle: data.bundle.map(|files| {
                    files
                        .into_iter()
                        .map(|f| workflow_documents::BundleFile {
                            path: f.path,
                            content: f.content,
                        })
                        .collect()
                }),
                run_document: data.run_document,
                roles: data.roles,
                started_by: data
                    .started_by
                    .unwrap_or_else(|| format!("{}@{}", caller.instance, caller.core_node)),
                caller_data: data.caller_data,
            };
            let director = Arc::clone(&director);
            tokio::spawn(async move {
                // The goal subscribes before the run starts, so it sees the
                // first event of the run.
                let events = director.events.subscribe();
                match director.start(request, caller) {
                    Ok((run, phase)) => {
                        let commands = director.view_of(&run).and_then(|view| view.commands);
                        let how = Follow::ToSuspension { commands };
                        follow(goal, run, events, phase, how).await
                    }
                    Err(message) => goal.failed("attempt_failed", message).await,
                }
            });
        }
    });
    Ok(())
}

/// Refuses a malformed `run` request with `rejected`: the shape of its
/// fields, and text that does not parse as JSON5.
fn decide_run(request: &run::GoalRequest) -> peppygen::Result<run::GoalDecision> {
    let data = &request.data;
    let shape = RunRequest {
        workflow: data.workflow.clone(),
        bundle: data.bundle.as_ref().map(|_| Vec::new()),
        run_document: data.run_document.clone(),
        roles: data.roles.clone(),
        started_by: String::new(),
        caller_data: None,
    }
    .shape();
    let texts = [
        ("workflow", &data.workflow),
        ("run_document", &data.run_document),
        ("roles", &data.roles),
        ("caller_data", &data.caller_data),
    ];
    let unparsed = texts.iter().find_map(|(field, text)| {
        text.as_deref()
            .and_then(|text| parses(text).err())
            .map(|reason| format!("{field}: {reason}"))
    });
    Ok(match shape.err().or(unparsed) {
        Some(message) => run::GoalDecision::refuse(run::FailureKind::Rejected, message),
        None => run::GoalDecision::accept(),
    })
}

async fn serve_resume(director: Arc<Director>) -> peppygen::Result<()> {
    let mut handle = resume::ActionHandle::expose(&director.runner).await?;
    tokio::spawn(async move {
        loop {
            let decided = {
                let director = Arc::clone(&director);
                handle
                    .handle_goal_next_request(move |request| {
                        let run = &request.data.run;
                        Ok(match director.view_of(run).map(|view| view.phase) {
                            None => resume::GoalDecision::refuse(
                                resume::FailureKind::Rejected,
                                format!("no run `{run}`"),
                            ),
                            Some(Phase::Suspended(_)) => resume::GoalDecision::accept(),
                            Some(phase) => resume::GoalDecision::refuse(
                                resume::FailureKind::PreconditionFailed,
                                format!(
                                    "the run `{run}` is {}, and only a suspended run resumes",
                                    state_name(&phase)
                                ),
                            ),
                        })
                    })
                    .await
            };
            let goal = match decided {
                Ok(Some(goal)) => goal,
                Ok(None) => return,
                Err(error) => {
                    tracing::warn!("resume: {error}");
                    tokio::time::sleep(RETRY_BACKOFF).await;
                    continue;
                }
            };
            let run = goal.request().data.run.clone();
            let director = Arc::clone(&director);
            tokio::spawn(async move {
                let Some(view) = director.view_of(&run) else {
                    goal.failed(REJECTED, pruned(&run)).await;
                    return;
                };
                let Some(commands) = view.commands else {
                    goal.failed(PRECONDITION_FAILED, format!("the run `{run}` has ended"))
                        .await;
                    return;
                };
                let events = director.events.subscribe();
                let (reply, resumed) = oneshot::channel();
                if commands.send(RunCommand::Resume { reply }).is_err() {
                    goal.failed(PRECONDITION_FAILED, format!("the run `{run}` has ended"))
                        .await;
                    return;
                }
                let how = Follow::ToSuspension {
                    commands: Some(commands),
                };
                let refusal = match resumed.await {
                    Ok(Ok(())) | Err(_) => {
                        return follow(goal, run, events, view.phases, how).await;
                    }
                    Ok(Err(refusal)) => refusal,
                };
                if let ResumeRefusal::Check(_) = refusal {
                    // The engine gave the suspension with the refusal before
                    // its answer, and the event reaches the events of the
                    // director a moment later: the caller reads it last.
                    let mut events = events;
                    let suspends = forward_until(&goal, &run, &mut events, EventKind::RunSuspends);
                    if tokio::time::timeout(EVENT_IN_FLIGHT, suspends)
                        .await
                        .is_err()
                    {
                        tracing::warn!(
                            "the suspension of the refused resume of {run} did not come"
                        );
                    }
                }
                let (kind, message) = refused_resume(&run, &refusal);
                goal.failed(kind, message).await;
            });
        }
    });
    Ok(())
}

/// The message of a goal whose run the registry of the director does not
/// hold.
fn pruned(run: &str) -> String {
    format!("no run `{run}`: its journal was pruned")
}

/// The kind and the message that end a resume goal whose resume `refusal`
/// refused.
fn refused_resume(run: &str, refusal: &ResumeRefusal) -> (&'static str, String) {
    match refusal {
        ResumeRefusal::NotSuspended => (
            PRECONDITION_FAILED,
            format!("the run `{run}` is not suspended, and only a suspended run resumes"),
        ),
        ResumeRefusal::Check(refusal) => (
            RUN_SUSPENDED,
            format!(
                "the resume of `{run}` was refused at {}: {}",
                refusal.at, refusal.message
            ),
        ),
    }
}

/// The name of the lease that a teach of locations takes.
const TEACH_LEASE: &str = "locations";

/// Admits a `locations` goal: the robot is listed, it has an instance of
/// locations:v1, and no lease holds it when the goal teaches.
fn decide_locations(
    director: &Director,
    data: &locations::GoalRequestData,
) -> locations::GoalDecision {
    use locations::{FailureKind, GoalDecision};
    let snapshot = match listed(director) {
        Ok(snapshot) => snapshot,
        Err(message) => return GoalDecision::refuse(FailureKind::ResourceUnavailable, message),
    };
    let Some(robot) = snapshot.fleet.robots.get(&data.robot) else {
        return GoalDecision::refuse(
            FailureKind::Rejected,
            format!("the fleet lists no robot `{}`", data.robot),
        );
    };
    let (name, tag) = fleet::LOCATIONS;
    if robot.instances_of(name, tag).is_empty() {
        return GoalDecision::refuse(
            FailureKind::ResourceUnavailable,
            format!("`{}` has no instance of {name}:{tag}", data.robot),
        );
    }
    if data.teach.is_some()
        && let Some(lease) = snapshot.fleet.lease_of(&data.robot)
    {
        return GoalDecision::refuse(
            FailureKind::PreconditionFailed,
            format!(
                "`{}` is leased by `{}` under `{}`: a teach waits for the lease to end",
                data.robot, lease.holder, lease.name
            ),
        );
    }
    GoalDecision::accept()
}

/// Teaches each locations:v1 instance of the robot the locations given, if
/// any, and lists what each knows. A teach holds the robot while it runs: no
/// run of the director binds it, and the lease it takes under
/// [`TEACH_LEASE`] keeps every other client off it.
async fn work_locations(director: Arc<Director>, goal: locations::GoalContext) {
    let data = goal.request().data.clone();
    let snapshot = director.snapshots.borrow().clone();
    let Some(robot) = snapshot.fleet.robots.get(&data.robot) else {
        let _ = goal
            .fail(
                locations::FailureKind::Rejected,
                format!("the fleet lists no robot `{}`", data.robot),
            )
            .await;
        return;
    };
    let teach = data.teach.as_ref().map(|items| {
        let locations: Vec<Json> = items
            .iter()
            .map(|item| {
                json!({ "name": item.name, "kind": item.kind, "position": item.position, "orientation": item.orientation, "size": item.size })
            })
            .collect();
        json!({ "locations": locations })
    });
    let held = match &teach {
        Some(_) => match hold_for_teach(&director, robot).await {
            Ok(held) => Some(held),
            Err((kind, message)) => {
                let _ = goal.fail(kind, message).await;
                return;
            }
        },
        None => None,
    };
    let skills = taught_or_listed(&director, &snapshot, robot, teach.as_ref()).await;
    if held.is_some() {
        director.leases.release(&robot.name).await;
    }
    drop(held);
    let ended = match skills {
        Ok(skills) => goal.complete(skills).await,
        Err((kind, message)) => goal.fail(kind, message).await,
    };
    if let Err(error) = ended {
        tracing::warn!("locations: {error}");
    }
}

/// Why a goal of `locations` failed: its kind and its message.
type LocationsFailure = (locations::FailureKind, String);

/// Holds `robot` for a teach: in the registry of the director, then by its
/// lease from the daemon.
async fn hold_for_teach(
    director: &Arc<Director>,
    robot: &Robot,
) -> Result<TeachHold, LocationsFailure> {
    let name = RobotName::parse(&robot.name)
        .map_err(|error| (locations::FailureKind::Rejected, error.to_string()))?;
    let held = director
        .hold_for_teach(&name)
        .map_err(|message| (locations::FailureKind::PreconditionFailed, message))?;
    director
        .leases
        .take(&robot.name, TEACH_LEASE)
        .await
        .map_err(|refused| {
            let kind = locations::FailureKind::parse(&refused.kind)
                .unwrap_or(locations::FailureKind::PreconditionFailed);
            let message = format!(
                "the lease of `{}` for the teach was refused ({}): {}",
                robot.name, refused.kind, refused.message
            );
            (kind, message)
        })?;
    Ok(held)
}

/// What each locations:v1 instance of `robot` knows, after the teach of
/// `teach` when it gives one.
async fn taught_or_listed(
    director: &Director,
    snapshot: &Snapshot,
    robot: &Robot,
    teach: Option<&Json>,
) -> Result<Vec<locations::LocationsResultSkillsItem>, LocationsFailure> {
    let (member, request) = match teach {
        Some(request) => ("teach", request.clone()),
        None => ("list", json!({})),
    };
    let (name, tag) = fleet::LOCATIONS;
    let mut skills = Vec::new();
    for instance in robot.instances_of(name, tag) {
        let known = fleet::known_locations(
            &director.runner,
            snapshot,
            robot,
            instance,
            member,
            &request,
        )
        .await
        .map_err(|error| {
            let kind = error
                .kind
                .as_deref()
                .and_then(locations::FailureKind::parse)
                .unwrap_or(locations::FailureKind::AttemptFailed);
            (kind, format!("`{}`: {}", instance.instance, error.message))
        })?;
        skills.push(locations::LocationsResultSkillsItem {
            instance: instance.instance.clone(),
            locations: known.iter().map(result_location).collect(),
        });
    }
    Ok(skills)
}

/// A known location, as the result of `locations` carries it.
fn result_location(location: &KnownLocation) -> locations::LocationsResultSkillsItemLocationsItem {
    locations::LocationsResultSkillsItemLocationsItem {
        name: location.name.to_string(),
        kind: location.kind.name().to_owned(),
        position: location.position,
        orientation: location.orientation,
        size: location.size,
    }
}

async fn serve_locations(director: Arc<Director>) -> peppygen::Result<()> {
    let mut handle = locations::ActionHandle::expose(&director.runner).await?;
    tokio::spawn(async move {
        loop {
            let decided = {
                let director = Arc::clone(&director);
                handle
                    .handle_goal_next_request(move |request| {
                        Ok(decide_locations(&director, &request.data))
                    })
                    .await
            };
            let goal = match decided {
                Ok(Some(goal)) => goal,
                Ok(None) => return,
                Err(error) => {
                    tracing::warn!("locations: {error}");
                    tokio::time::sleep(RETRY_BACKOFF).await;
                    continue;
                }
            };
            tokio::spawn(work_locations(Arc::clone(&director), goal));
        }
    });
    Ok(())
}

async fn serve_abort(director: Arc<Director>) -> peppygen::Result<()> {
    let mut handle = abort::ActionHandle::expose(&director.runner).await?;
    tokio::spawn(async move {
        loop {
            let decided = {
                let director = Arc::clone(&director);
                handle
                    .handle_goal_next_request(move |request| {
                        let run = &request.data.run;
                        Ok(match director.view_of(run) {
                            None => abort::GoalDecision::refuse(
                                abort::FailureKind::Rejected,
                                format!("no run `{run}`"),
                            ),
                            Some(_) => abort::GoalDecision::accept(),
                        })
                    })
                    .await
            };
            let goal = match decided {
                Ok(Some(goal)) => goal,
                Ok(None) => return,
                Err(error) => {
                    tracing::warn!("abort: {error}");
                    tokio::time::sleep(RETRY_BACKOFF).await;
                    continue;
                }
            };
            let run = goal.request().data.run.clone();
            let director = Arc::clone(&director);
            tokio::spawn(async move {
                let Some(RunView {
                    phase: phase_now,
                    commands,
                    phases,
                    ..
                }) = director.view_of(&run)
                else {
                    goal.failed(REJECTED, pruned(&run)).await;
                    return;
                };
                if let Phase::Ended(result) = phase_now {
                    goal.completed(&run, &result).await;
                    return;
                }
                let events = director.events.subscribe();
                if let Some(commands) = commands {
                    let _ = commands.send(RunCommand::Abort);
                }
                follow(AbortGoal(goal), run, events, phases, Follow::ToEnd).await;
            });
        }
    });
    Ok(())
}

/// The goal of `abort`, which completes at the end of the run whatever its
/// ending: the abort is done once the run has ended.
struct AbortGoal(abort::GoalContext);

impl RunGoal for AbortGoal {
    async fn feedback(&self, run: &str, event: &Event) {
        self.0.feedback(run, event).await
    }
    async fn completed(&self, run: &str, result: &RunResult) {
        self.0.completed(run, result).await
    }
    async fn aborted(&self, run: &str, result: &RunResult) {
        self.0.completed(run, result).await
    }
    async fn failed(&self, kind: &str, message: String) {
        self.0.failed(kind, message).await
    }
    async fn cancelled(&self) {
        self.0.cancelled().await
    }
    fn sender(&self) -> Caller {
        RunGoal::sender(&self.0)
    }
    async fn ended(&self, run: &str, result: &RunResult) {
        self.0.completed(run, result).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use workflow_documents::Located;

    use super::*;

    /// A goal of the tests: the events it was given, and how it ended.
    #[derive(Default)]
    struct Recorded {
        events: Mutex<Vec<EventKind>>,
        end: Mutex<Option<String>>,
    }

    impl Recorded {
        fn ends(&self, how: &str) {
            *self.end.lock().unwrap() = Some(how.to_owned());
        }
    }

    impl RunGoal for Arc<Recorded> {
        async fn feedback(&self, _run: &str, event: &Event) {
            self.events.lock().unwrap().push(event.kind);
        }
        async fn completed(&self, _run: &str, _result: &RunResult) {
            self.ends("completed");
        }
        async fn aborted(&self, _run: &str, _result: &RunResult) {
            self.ends("aborted");
        }
        async fn failed(&self, kind: &str, _message: String) {
            self.ends(kind);
        }
        async fn cancelled(&self) {
            std::future::pending().await
        }
        fn sender(&self) -> Caller {
            Caller {
                instance: "tests".to_owned(),
                core_node: "tests".to_owned(),
            }
        }
    }

    /// The phase of a run turns `Ended` a moment before its `run_ends` reaches the events of the director: a goal
    /// that sees the phase first gives the `run_ends` before it ends.
    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_goal_that_sees_the_end_of_its_run_first_gives_its_run_ends() {
        let (events_tx, events) = broadcast::channel(16);
        let (phase_tx, phase) = watch::channel(Phase::Running);
        let goal = Arc::new(Recorded::default());
        let following = tokio::spawn(follow(
            Arc::clone(&goal),
            "r1".to_owned(),
            events,
            phase,
            Follow::ToEnd,
        ));
        tokio::task::yield_now().await;
        let result = RunResult {
            ending: Ending::Completed,
            ..crate::director::empty_result()
        };
        phase_tx.send_replace(Phase::Ended(Box::new(result.clone())));
        tokio::task::yield_now().await;
        let ends = Event {
            role: String::new(),
            kind: EventKind::RunEnds,
            step: String::new(),
            failure_kind: String::new(),
            message: String::new(),
            details: json!(result),
        };
        events_tx.send(("r1".to_owned(), ends)).unwrap();
        following.await.unwrap();
        assert_eq!(*goal.events.lock().unwrap(), [EventKind::RunEnds]);
        assert_eq!(goal.end.lock().unwrap().as_deref(), Some("completed"));
    }

    /// A resume whose run is not suspended when its engine reads it fails `precondition_failed`, as the decider
    /// refuses it; a resume whose check refused it fails `run_suspended`, naming where.
    #[test]
    fn a_resume_of_a_run_that_is_not_suspended_fails_precondition_failed() {
        let (kind, message) = refused_resume("r1", &ResumeRefusal::NotSuspended);
        assert_eq!(kind, "precondition_failed");
        assert_eq!(
            message,
            "the run `r1` is not suspended, and only a suspended run resumes"
        );
        let check = ResumeRefusal::Check(Located {
            at: "assumptions.in_reach".to_owned(),
            message: "the reach check is false".to_owned(),
        });
        let (kind, message) = refused_resume("r1", &check);
        assert_eq!(kind, "run_suspended");
        assert_eq!(
            message,
            "the resume of `r1` was refused at assumptions.in_reach: the reach check is false"
        );
    }

    #[test]
    fn the_runs_come_newest_first_by_their_number() {
        let item = |run: &str| runs::ResponseRunsItem {
            run: run.to_owned(),
            workflow: String::new(),
            revision: 0,
            state: String::new(),
            started_by: String::new(),
            robots: Vec::new(),
        };
        let order: Vec<String> = newest_first(vec![item("r2"), item("r10"), item("r1")])
            .into_iter()
            .map(|item| item.run)
            .collect();
        assert_eq!(order, ["r10", "r2", "r1"]);
    }

    /// A refused goal names what refused the run, then each error at its place, so the caller reads which robot and
    /// what holds it.
    #[test]
    fn the_message_of_a_refused_run_gives_each_error_at_its_place() {
        let held = "`alpha` takes part in run r1 of this director: wait for its end, or abort it";
        let refusal = |errors: Vec<Located>| RunRefusal {
            message: "the roles do not bind".to_owned(),
            errors,
            assumption: None,
            says: None,
        };
        let located = Located {
            at: "roles.arm".to_owned(),
            message: held.to_owned(),
        };
        assert_eq!(
            refusal_message(&refusal(vec![located])),
            format!("the roles do not bind: roles.arm: {held}")
        );
        assert_eq!(
            refusal_message(&refusal(Vec::new())),
            "the roles do not bind"
        );
    }
}
