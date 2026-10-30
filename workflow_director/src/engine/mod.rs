//! The engine of one run: the checks of its assumptions, its confirmations,
//! the initialization of each role and the lanes, on the run clock of its
//! host. It decides each step from the workflow and from the results and
//! observations of its calls, with no language model; the host carries out
//! every call and read, so a run replays against a journal as it ran.

pub mod failure;
mod hold;
pub mod host;
mod lane;
pub mod plan;
mod state;

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use serde_json::{Value as Json, json};
use tokio::sync::{mpsc, oneshot, watch};
use tokio_util::sync::CancellationToken;
use workflow_documents::expression::{Expression, Type, Value};
use workflow_documents::result::{
    AssumptionKind, AssumptionOutcome, CheckToJudge, Duration, Ending, FailedStep, IncidentRecord,
    LaneOutcome, LaneState, RUN_SUSPENDED, RunRefusal, RunResult, RunState, RunStatus,
    SafeStateResult, SuspendedLane, Suspension, ToConfirm,
};
use workflow_documents::work_cell::Uncovered;
use workflow_documents::workflow::StepKind;
use workflow_documents::{EventKind, Located, Name, uncovered_watch};

use failure::{
    Cause, DirectorKind, Failure, LEADER_STREAMING, PRECONDITION_FAILED, RESOURCE_UNAVAILABLE,
    TIMEOUT,
};
use hold::Hold;
use host::{CallEnd, Event, Host, Join, Joining};
pub(crate) use lane::Stop;
use lane::{Lane, ParkCommand, Scope, StepEnd};
use plan::{ObservationRead, Plan, request};
use state::{LaneClock, Shared};

/// The bound of each check outside a step, on the run clock: the check of
/// an assumption, each check of a resume, `success` and `minimum`.
const CHECK_BOUND_NS: u64 = 30_000_000_000;
/// The interval at which a call or a read that a leader's stream refuses
/// goes again.
const LEADER_RETRY_NS: u64 = 1_000_000_000;

/// What a caller asks of a running run.
pub enum RunCommand {
    /// Confirms an assumption without a check while the run waits for its
    /// confirmations. The `confirm` service admits it from the phase of the
    /// run, and a confirmation the run no longer waits for changes nothing.
    Confirm {
        assumption: Name,
    },
    /// Suspends each running lane. The `suspend` service admits it from the
    /// phase of the run, and a run whose lanes do not run keeps on.
    Suspend,
    /// Resumes a suspended run, after its checks.
    Resume {
        reply: oneshot::Sender<Result<(), ResumeRefusal>>,
    },
    Abort,
    /// A halt or the end of the run's leases: every goal is cancelled and no
    /// command is sent, not even a safe state.
    Stop {
        halted: bool,
        reason: String,
    },
    /// The incident alerts that stand on the robot of a lane, as one
    /// producer of its alert:v1 gave them last: a resume of the lane is
    /// refused while one stands.
    Alerts {
        role: Name,
        source: String,
        standing: Vec<String>,
    },
    /// A fault of one lane: an alert of its robot, or a stopped clock.
    Fault {
        role: Name,
        failure: Failure,
    },
    /// The judgement of the operator on the checks of a run of a trial,
    /// while the run waits for it. The `judge_checks` service admits it from
    /// the phase of the run, and a judgement the run no longer waits for
    /// changes nothing.
    JudgeChecks {
        judged: ChecksJudged,
    },
}

/// The judgement of the operator on the checks of a run of a trial, which
/// the operator compares with the work cell.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChecksJudged {
    /// Each check is true and none is wrong: the run starts.
    Starts,
    /// A check is false and the operator agrees with each value: the work
    /// cell is not ready, and the run does not count.
    NotReady,
    /// The operator names each check whose value differs from the work
    /// cell: the run counts as not completed.
    FalseCheck { wrong: Vec<Name> },
}

impl ChecksJudged {
    /// The judgement of `checks` when the operator names `wrong`.
    pub fn of(checks: &[CheckToJudge], wrong: Vec<Name>) -> ChecksJudged {
        if !wrong.is_empty() {
            return ChecksJudged::FalseCheck { wrong };
        }
        match checks.iter().all(|check| check.value) {
            true => ChecksJudged::Starts,
            false => ChecksJudged::NotReady,
        }
    }

    /// The word of `judge_checks` for this judgement.
    pub fn outcome(&self) -> &'static str {
        match self {
            ChecksJudged::Starts => "starts",
            ChecksJudged::NotReady => "not_ready",
            ChecksJudged::FalseCheck { .. } => "false_check",
        }
    }
}

/// Why a resume did not resume its run.
#[derive(Clone, Debug, PartialEq)]
pub enum ResumeRefusal {
    /// The run is not suspended.
    NotSuspended,
    /// A check of the resume refused it, at its place; the run stays
    /// suspended.
    Check(Located),
}

/// Where a run stands, for the goals that follow it.
#[derive(Clone, Debug, PartialEq)]
pub enum Phase {
    Preparing,
    Confirming(Vec<ToConfirm>),
    /// A run of a trial waits for the judgement of the operator on each of
    /// its checks.
    Judging(Vec<CheckToJudge>),
    Running,
    Suspended(Suspension),
    /// The engine of the run ends: from the end of its lanes, when the run
    /// checks `success` and `minimum`, or from the end of its engine for a
    /// run that ends before its lanes (refused at its assumptions, stopped
    /// while it confirms), to the release of its leases. A run that ends
    /// before its engine starts goes from `Preparing` to `Ended`.
    Ending,
    Ended(Box<RunResult>),
}

impl Phase {
    pub fn state(&self) -> RunState {
        match self {
            Phase::Preparing => RunState::Preparing,
            Phase::Confirming(_) | Phase::Judging(_) => RunState::Confirming,
            Phase::Running => RunState::Running,
            Phase::Suspended(_) => RunState::Suspended,
            Phase::Ending => RunState::Ending,
            Phase::Ended(result) => result.ending.into(),
        }
    }
}

/// The fields of a run result that the run's engine does not decide: its
/// workflow, its versions and what its caller gave.
#[derive(Clone, Debug)]
pub struct ResultBase {
    pub result: RunResult,
}

/// What ends the wait of a read for the first message of a topic, the one
/// read that can last up to its `max_age`: for a step of a lane, the stop of
/// the run, a suspension of the lane, a fault of the lane and the cancel of
/// the step; for a check outside a step, nothing but the check's own bound.
pub(crate) trait Interrupt: Sync {
    type End;

    /// Resolves at the first interruption.
    fn interrupted(&self) -> impl Future<Output = Self::End> + Send;
}

/// The wait of a check outside a step, which its bound ends.
pub(crate) struct NoInterrupt;

impl Interrupt for NoInterrupt {
    type End = std::convert::Infallible;

    fn interrupted(&self) -> impl Future<Output = Self::End> + Send {
        std::future::pending()
    }
}

/// `work`, raced with `wait` while it waits when `long` says the read can
/// wait long (the first message of a topic); a read that answers at once
/// meets no interruption.
async fn waiting<T, W: Interrupt>(
    long: bool,
    wait: &W,
    work: impl Future<Output = T>,
) -> Result<T, W::End> {
    if !long {
        return Ok(work.await);
    }
    tokio::select! {
        biased;
        done = work => Ok(done),
        end = wait.interrupted() => Err(end),
    }
}

/// What every lane of a run shares.
pub(crate) struct RunCtx<'a, H: Host> {
    pub plan: &'a Plan,
    pub host: &'a H,
    /// The run time of the plan of the run, by which every topic of the run
    /// was subscribed.
    pub started: u64,
    pub shared: Shared,
    pub stop: watch::Sender<Option<Stop>>,
    /// The stop that cut the run off, so it sends nothing more, a safe state
    /// included: a halt, the end of its leases or the loss of its caller.
    pub cut_off: watch::Sender<Option<Stop>>,
    /// The fault of each role (an alert, a stopped clock) that no branch
    /// of its lane met yet.
    pub faults: BTreeMap<Name, watch::Sender<Option<Failure>>>,
}

impl<H: Host> RunCtx<'_, H> {
    pub fn event(
        &self,
        role: &Name,
        kind: EventKind,
        step: String,
        failure_kind: &str,
        message: String,
        details: Json,
    ) {
        self.host.event(Event {
            role: role.to_string(),
            kind,
            step,
            failure_kind: failure_kind.to_owned(),
            message,
            details,
        });
    }

    fn run_event(&self, kind: EventKind, failure_kind: &str, message: String, details: Json) {
        self.host.event(Event {
            role: String::new(),
            kind,
            step: String::new(),
            failure_kind: failure_kind.to_owned(),
            message,
            details,
        });
    }

    /// Keeps a fault of `role` until a branch of its lane meets it.
    fn fault(&self, role: &Name, failure: Failure) {
        if let Some(fault) = self.faults.get(role) {
            fault.send_replace(Some(failure));
        }
    }

    /// Keeps the incident alerts that stand on the robot of `role`, as its
    /// producer `source` gave them last.
    fn alerts_stand(&self, role: &Name, source: String, standing: Vec<String>) {
        self.shared.change(|s| {
            let of_role = s.alerts.entry(role.clone()).or_default();
            match standing.is_empty() {
                true => of_role.remove(&source),
                false => of_role.insert(source, standing),
            };
        });
    }

    /// An incident alert that stands on the robot of `role`.
    fn standing_alert(&self, role: &Name) -> Option<String> {
        self.shared.read(|s| {
            s.alerts
                .get(role)
                .and_then(|by_source| by_source.values().flatten().next().cloned())
        })
    }

    /// Takes the fault of `role` that no branch of its lane met yet.
    pub fn take_fault(&self, role: &Name) -> Option<Failure> {
        let fault = self.faults.get(role)?;
        let mut taken = None;
        fault.send_if_modified(|current| {
            taken = current.take();
            taken.is_some()
        });
        taken
    }

    /// Records the incident of `role` at `step`, with the safe state that its
    /// lane ran for it.
    pub fn record_incident(
        &self,
        role: &Name,
        step: String,
        failure: &Failure,
        safe_state: &[SafeStateResult],
    ) {
        let observations = self.shared.read(|s| {
            s.last_values
                .iter()
                .map(|(name, value)| (name.to_string(), value.clone()))
                .collect()
        });
        let record = IncidentRecord {
            step: step.clone(),
            lane: role.clone(),
            failure_kind: failure.kind.clone(),
            message: failure.message.clone(),
            observations,
            anomaly: match failure.cause.as_deref() {
                Some(Cause::Anomaly(anomaly)) => Some(anomaly.clone()),
                _ => None,
            },
            alert: match failure.cause.as_deref() {
                Some(Cause::Alert(alert)) => Some(alert.clone()),
                _ => None,
            },
            affected_lanes: vec![role.clone()],
            safe_state: safe_state.to_vec(),
        };
        self.shared.change(|s| s.incidents.push(record.clone()));
        self.event(
            role,
            EventKind::Incident,
            step,
            &failure.kind,
            failure.message.clone(),
            serde_json::to_value(&record).unwrap_or(Json::Null),
        );
    }

    /// Whether a stop cut the run off: it sends nothing more.
    pub fn is_cut_off(&self) -> bool {
        self.cut_off.borrow().is_some()
    }

    /// The stop that cuts the run off, once it comes.
    pub async fn until_cut_off(&self) -> Stop {
        let mut cut_off = self.cut_off.subscribe();
        cut_off
            .wait_for(Option::is_some)
            .await
            .ok()
            .and_then(|stop| *stop)
            .unwrap_or(Stop::Interrupted)
    }

    /// Stops the run with `stop`, for `reason`, unless it stopped already; a
    /// halt replaces an interruption, since a halt ends the leases first. A
    /// halt or an interruption cuts the run off, whatever stopped it first.
    pub fn stop_run(&self, stop: Stop, reason: &str) {
        tracing::info!("the run stops ({stop:?}): {reason}");
        if stop != Stop::Aborted {
            self.cut_off.send_if_modified(|current| match current {
                Some(_) => false,
                None => {
                    *current = Some(stop);
                    true
                }
            });
        }
        self.stop
            .send_if_modified(|current| match (current.as_ref(), &stop) {
                (None, _) | (Some(Stop::Interrupted), Stop::Halted) => {
                    *current = Some(stop);
                    true
                }
                _ => false,
            });
    }

    /// Reads an observation, once at each instant of the run clock: a
    /// reader at the instant of a read in flight takes its answer, and reads
    /// itself when that read was dropped. The host gives how each read goes,
    /// so a replay joins the reads that its run joined.
    pub async fn observe<W: Interrupt>(
        &self,
        name: &Name,
        wait: &W,
    ) -> Result<Result<Json, Failure>, W::End> {
        let long = matches!(
            self.plan.observations[name].read,
            ObservationRead::Topic { .. }
        );
        let nth = self.shared.change(|s| {
            let count = s.observed.entry(name.clone()).or_default();
            *count += 1;
            *count - 1
        });
        match waiting(long, wait, self.host.joining(name, nth)).await? {
            Joining::Took(answer) => return Ok(answer),
            Joining::Own => return self.read_now(name, wait).await,
            Joining::AsFound => {}
        }
        let now = self.host.now();
        let reading = self.shared.change(|s| {
            if s.read_at != Some(now) {
                s.read_at = Some(now);
                s.reads.clear();
            }
            match s.reads.get(name) {
                Some(read) => Reading::Theirs(read.clone()),
                None => {
                    let (answer, read) = watch::channel(None);
                    s.reads.insert(name.clone(), read);
                    Reading::Mine(answer)
                }
            }
        });
        let answer = match reading {
            Reading::Mine(answer) => answer,
            Reading::Theirs(mut read) => {
                self.host.joined(name, nth, &Join::Waits);
                let taken = waiting(long, wait, async {
                    read.wait_for(Option::is_some)
                        .await
                        .ok()
                        .and_then(|answer| answer.clone())
                })
                .await?;
                if let Some(answer) = taken {
                    self.host.joined(name, nth, &Join::Took(answer.clone()));
                    return Ok(answer);
                }
                self.host.joined(name, nth, &Join::ReadsItself);
                return self.read_now(name, wait).await;
            }
        };
        let read = self.read_now(name, wait).await?;
        answer.send_replace(Some(read.clone()));
        Ok(read)
    }

    /// Reads an observation, and keeps its value as the last one. The topics
    /// of a run are subscribed before its engine starts, so a read before
    /// the first message of a topic waits for it, up to `max_age` past that
    /// start, and is stale after; an interruption of `wait` ends that wait.
    async fn read_now<W: Interrupt>(
        &self,
        name: &Name,
        wait: &W,
    ) -> Result<Result<Json, Failure>, W::End> {
        let observation = &self.plan.observations[name];
        let read = match &observation.read {
            ObservationRead::Service { args } => {
                let request = match request(args, &BTreeMap::new()) {
                    Ok(request) => request,
                    Err(missing) => {
                        return Ok(Err(Failure::new(PRECONDITION_FAILED, missing.to_string())));
                    }
                };
                match self
                    .host
                    .call(&observation.target, request, CancellationToken::new())
                    .await
                {
                    CallEnd::Completed(json) => Ok(json),
                    CallEnd::Refused(f) | CallEnd::Failed(f) => Err(f),
                    _ => Err(Failure::new(
                        RESOURCE_UNAVAILABLE,
                        format!("the observation `{name}` gave no answer"),
                    )),
                }
            }
            ObservationRead::Topic { max_age } => loop {
                let now = self.host.now();
                let first_due = self.started + max_age.nanos();
                match self.host.latest(&observation.target) {
                    Some((json, at)) if now.saturating_sub(at) <= max_age.nanos() => {
                        break Ok(json);
                    }
                    Some(_) => {
                        break Err(Failure::of(
                            failure::DirectorKind::StaleObservation,
                            format!(
                                "the last message of `{name}` is older than {} s",
                                max_age.seconds()
                            ),
                        ));
                    }
                    None if now >= first_due => {
                        break Err(Failure::of(
                            failure::DirectorKind::StaleObservation,
                            format!(
                                "`{name}` has given no message in the {} s since the run started",
                                max_age.seconds()
                            ),
                        ));
                    }
                    None => tokio::select! {
                        biased;
                        () = self.host.message_arrives(&observation.target) => {}
                        () = self.host.until(first_due) => {}
                        end = wait.interrupted() => return Err(end),
                    },
                }
            },
        };
        if let Ok(json) = &read {
            self.shared
                .change(|s| s.last_values.insert(name.clone(), json.clone()));
        }
        Ok(read)
    }

    /// The value of an expression over `scope`, typed by `locals`, the
    /// tables and the observations it reads, each observation read once at
    /// this instant, a wait for the first message of a topic ended by `wait`.
    pub async fn value<W: Interrupt>(
        &self,
        expression: &Expression,
        scope: &Scope,
        locals: Option<&BTreeMap<Name, Type>>,
        wait: &W,
    ) -> Result<Result<Value, Failure>, W::End> {
        let mut observed = BTreeMap::new();
        for name in expression.roots() {
            let observes = !scope.contains_key(&name)
                && !self.plan.tables.contains_key(&name)
                && self.plan.observations.contains_key(&name);
            if observes {
                match self.observe(&name, wait).await? {
                    Ok(json) => observed.insert(name, json),
                    Err(failure) => return Ok(Err(failure)),
                };
            }
        }
        Ok(self.evaluate(expression, scope, locals, &observed))
    }

    /// The value of an expression over the scope, the tables and the
    /// observations `observed`.
    fn evaluate(
        &self,
        expression: &Expression,
        scope: &Scope,
        locals: Option<&BTreeMap<Name, Type>>,
        observed: &BTreeMap<Name, Json>,
    ) -> Result<Value, Failure> {
        let mut values = BTreeMap::new();
        for name in expression.roots() {
            let (json, ty): (&Json, &Type) = if let Some(json) = scope.get(&name) {
                let ty = locals.and_then(|types| types.get(&name)).ok_or_else(|| {
                    Failure::new(PRECONDITION_FAILED, format!("`{name}` has no type"))
                })?;
                (json, ty)
            } else if let Some((json, ty)) = self.plan.tables.get(&name) {
                (json, ty)
            } else if let Some(json) = observed.get(&name) {
                (json, &self.plan.observations[&name].ty)
            } else {
                continue;
            };
            let value = Value::read(json, ty).map_err(|reason| {
                Failure::new(PRECONDITION_FAILED, format!("`{name}`: {reason}"))
            })?;
            values.insert(name, value);
        }
        expression
            .expr()
            .evaluate(&values)
            .map_err(|error| Failure::new(PRECONDITION_FAILED, error.to_string()))
    }

    /// The truth of an expression, as `value` reads it.
    pub async fn truth_in<W: Interrupt>(
        &self,
        expression: &Expression,
        scope: &Scope,
        locals: Option<&BTreeMap<Name, Type>>,
        wait: &W,
    ) -> Result<Result<bool, Failure>, W::End> {
        Ok(match self.value(expression, scope, locals, wait).await? {
            Ok(Value::Bool(b)) => Ok(b),
            Ok(_) => Err(Failure::new(
                PRECONDITION_FAILED,
                format!("`{}` gives no bool", expression.source()),
            )),
            Err(failure) => Err(failure),
        })
    }

    /// The truth of an expression outside a lane: a check of the run, whose
    /// own bound ends its waits.
    async fn truth(&self, expression: &Expression) -> Result<bool, Failure> {
        let Ok(truth) = self
            .truth_in(expression, &Scope::new(), None, &NoInterrupt)
            .await;
        truth
    }

    /// The number that an expression outside a lane gives, as `truth` reads
    /// it: the value of a watch parameter.
    async fn number(&self, expression: &Expression) -> Result<f64, Failure> {
        let Ok(value) = self
            .value(expression, &Scope::new(), None, &NoInterrupt)
            .await;
        match value? {
            Value::Int(int) => Ok(int as f64),
            Value::Double(double) => Ok(double),
            _ => Err(Failure::new(
                PRECONDITION_FAILED,
                format!("`{}` gives no number", expression.source()),
            )),
        }
    }

    /// The answer of `check`, a check outside a step, within its bound
    /// ([`CHECK_BOUND_NS`]); a read that a leader's stream refuses goes
    /// again every second. `what` names the check in a timeout.
    pub async fn bounded_check<T, F, Fut>(&self, what: &str, check: F) -> Result<T, Failure>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, Failure>>,
    {
        let deadline = self.host.now() + CHECK_BOUND_NS;
        let answer = async {
            loop {
                match check().await {
                    Err(failure) if failure.kind == LEADER_STREAMING => {
                        self.host.until(self.host.now() + LEADER_RETRY_NS).await;
                    }
                    answer => return answer,
                }
            }
        };
        tokio::select! {
            answer = answer => answer,
            () = self.host.until(deadline) => Err(Failure::new(
                TIMEOUT,
                format!(
                    "{what} gave no answer within its bound of {} s",
                    CHECK_BOUND_NS / 1_000_000_000
                ),
            )),
        }
    }
}

/// A read of an observation at one instant: the reader's own, whose answer it
/// gives to the others, or one in flight that it waits for.
enum Reading {
    Mine(watch::Sender<Option<Result<Json, Failure>>>),
    Theirs(watch::Receiver<Option<Result<Json, Failure>>>),
}

/// The emitters of each signal: the lanes that hold a `signal` step for it.
fn emitters(plan: &Plan) -> BTreeMap<Name, BTreeSet<Name>> {
    let mut emitters: BTreeMap<Name, BTreeSet<Name>> = BTreeMap::new();
    for (role, lane) in &plan.workflow.lanes {
        lane.tree.visit(&mut |step| {
            if let StepKind::Signal(signal) = &step.kind {
                emitters
                    .entry(signal.clone())
                    .or_default()
                    .insert(role.clone());
            }
        });
    }
    emitters
}

/// Drives one run to its end, or to a suspension that its caller follows on
/// `phase`; it ends at the end of the run, whatever goal follows it, with
/// the run `Ending`. Its caller marks the run ended on `phase` once it gave
/// the leases back and sent `run_ends`, so a goal that follows the run ends
/// after both. `started` is the run time of the plan of the run, which its
/// journal keeps, by which every topic of the run was subscribed.
pub async fn drive<H: Host>(
    plan: &Plan,
    host: &H,
    base: ResultBase,
    commands: mpsc::UnboundedReceiver<RunCommand>,
    phase: watch::Sender<Phase>,
    started: u64,
) -> RunResult {
    let result = run_to_its_end(plan, host, base, commands, &phase, started).await;
    mark_ending(&phase);
    result
}

/// Marks the run `Ending`, once.
fn mark_ending(phase: &watch::Sender<Phase>) {
    phase.send_if_modified(|current| match current {
        Phase::Ending => false,
        _ => {
            *current = Phase::Ending;
            true
        }
    });
}

async fn run_to_its_end<H: Host>(
    plan: &Plan,
    host: &H,
    base: ResultBase,
    mut commands: mpsc::UnboundedReceiver<RunCommand>,
    phase: &watch::Sender<Phase>,
    started: u64,
) -> RunResult {
    let ctx = RunCtx {
        plan,
        host,
        started,
        shared: Shared::new(emitters(plan)),
        stop: watch::Sender::new(None),
        cut_off: watch::Sender::new(None),
        faults: plan
            .roles
            .keys()
            .map(|role| (role.clone(), watch::Sender::new(None)))
            .collect(),
    };
    let started = (host.now(), host.wall_now());
    let mut outcome = Outcome::default();
    // The commands that came during the preparation come first.
    let refused = tokio::select! {
        biased;
        stop = stopped_while_checking(&ctx, &mut commands) => {
            return finish(&ctx, base, ending_of(stop), None, outcome, started).await;
        }
        refused = async {
            match watch(&ctx, &base.result.uncovered, &mut outcome).await {
                Some(refusal) => Some(refusal),
                None => assumptions(&ctx, &mut outcome).await,
            }
        } => refused,
    };
    if let Some(refusal) = refused {
        return finish(&ctx, base, Ending::Refused, Some(refusal), outcome, started).await;
    }
    if plan.of_a_trial {
        match judgement(&ctx, &outcome, &mut commands, phase).await {
            Judgement::Starts => {}
            Judgement::Refused(refusal) => {
                return finish(&ctx, base, Ending::Refused, Some(refusal), outcome, started).await;
            }
            Judgement::Stopped(stop) => {
                return finish(&ctx, base, ending_of(stop), None, outcome, started).await;
            }
        }
    }
    if let Some(stop) = confirmations(&ctx, &mut outcome, &mut commands, phase).await {
        return finish(&ctx, base, ending_of(stop), None, outcome, started).await;
    }
    phase.send_replace(Phase::Running);
    ctx.run_event(
        EventKind::InitializationStarts,
        "",
        String::new(),
        Json::Null,
    );
    let init_roles: Vec<Name> = plan.workflow.initialization.keys().cloned().collect();
    let init = lanes(&ctx, &init_roles, true, &mut commands, phase).await;
    let init_failed = init.values().any(|end| *end != StepEnd::Done);
    let stopped = *ctx.stop.borrow();
    if let Some(stop) = stopped {
        return finish(&ctx, base, ending_of(stop), None, outcome, started).await;
    }
    if init_failed {
        outcome.failed = first_failure(&init);
        return finish(&ctx, base, Ending::Failed, None, outcome, started).await;
    }
    ctx.run_event(EventKind::ActionsStart, "", String::new(), Json::Null);
    let roles: Vec<Name> = plan.workflow.lanes.keys().cloned().collect();
    let ends = lanes(&ctx, &roles, false, &mut commands, phase).await;
    mark_ending(phase);
    let stopped = *ctx.stop.borrow();
    if let Some(stop) = stopped {
        return finish(&ctx, base, ending_of(stop), None, outcome, started).await;
    }
    outcome.failed = first_failure(&ends);
    // `success` is read once every lane completed, `minimum` once
    // `success` does not hold; a command that stops the run goes first.
    if ends.values().all(|end| *end == StepEnd::Done) {
        let success = &plan.workflow.success;
        match checked(&ctx, &mut commands, "`success`", success).await {
            Err(stop) => return finish(&ctx, base, ending_of(stop), None, outcome, started).await,
            Ok(Ok(true)) => {
                return finish(&ctx, base, Ending::Completed, None, outcome, started).await;
            }
            Ok(read) => outcome.failed = Some(failed_success(read)),
        }
    }
    let Some(minimum) = &plan.workflow.minimum else {
        return finish(&ctx, base, Ending::Failed, None, outcome, started).await;
    };
    let ending = match checked(&ctx, &mut commands, "`minimum`", minimum).await {
        Err(stop) => ending_of(stop),
        Ok(Ok(true)) => Ending::Degraded,
        Ok(_) => Ending::Failed,
    };
    finish(&ctx, base, ending, None, outcome, started).await
}

/// The truth of `expression`, the check `name` of a run whose lanes ended,
/// within the bound of a check, unless a command stops the run first.
async fn checked<H: Host>(
    ctx: &RunCtx<'_, H>,
    commands: &mut mpsc::UnboundedReceiver<RunCommand>,
    name: &str,
    expression: &Expression,
) -> Result<Result<bool, Failure>, Stop> {
    tokio::select! {
        biased;
        stop = stopped_while_checking(ctx, commands) => Err(stop),
        truth = ctx.bounded_check(name, || ctx.truth(expression)) => Ok(truth),
    }
}

/// The failure of a run whose lanes completed and whose `success` `read`
/// false, or gave no answer.
fn failed_success(read: Result<bool, Failure>) -> FailedStep {
    let failure = read
        .err()
        .unwrap_or_else(|| Failure::of(DirectorKind::ConditionFailed, "`success` does not hold"));
    FailedStep {
        step: "success".to_owned(),
        failure_kind: failure.kind,
        message: failure.message,
    }
}

/// What the engine records for the run result as the run goes.
#[derive(Default)]
struct Outcome {
    assumptions: Vec<AssumptionOutcome>,
    failed: Option<FailedStep>,
    /// The value of each watch parameter.
    watch: BTreeMap<Name, f64>,
    /// Each watch value that no test covered.
    uncovered: Vec<Uncovered>,
}

fn ending_of(stop: Stop) -> Ending {
    match stop {
        Stop::Aborted => Ending::Aborted,
        Stop::Halted => Ending::Halted,
        Stop::Interrupted => Ending::Interrupted,
    }
}

/// The step that failed the first failed lane of `ends`.
fn first_failure(ends: &BTreeMap<Name, StepEnd>) -> Option<FailedStep> {
    ends.values().find_map(|end| match end {
        StepEnd::LaneFailed { step, failure } => Some(FailedStep {
            step: step.clone(),
            failure_kind: failure.kind.clone(),
            message: failure.message.clone(),
        }),
        _ => None,
    })
}

/// Reads each watch parameter through the robots, each within the bound of
/// a check, and gives the event `uncovered` with each value of the work cell
/// (`work_cell`, which the director named) and each watch value that no test
/// covered. A watch value that does not read refuses the run, naming it.
async fn watch<H: Host>(
    ctx: &RunCtx<'_, H>,
    work_cell: &[Uncovered],
    outcome: &mut Outcome,
) -> Option<RunRefusal> {
    let watch = &ctx.plan.watch;
    for (name, observation) in &watch.observations {
        let what = format!("the watch parameter `{name}`");
        match ctx.bounded_check(&what, || ctx.number(observation)).await {
            Ok(value) => {
                outcome.watch.insert(name.clone(), value);
            }
            Err(failure) => {
                return Some(RunRefusal {
                    message: format!("{what} did not read: {failure}"),
                    errors: Vec::new(),
                    assumption: None,
                    says: None,
                });
            }
        }
    }
    outcome.uncovered = uncovered_watch(watch.validation.as_ref(), &outcome.watch);
    let uncovered: Vec<&Uncovered> = work_cell.iter().chain(&outcome.uncovered).collect();
    if !uncovered.is_empty() {
        let message = listed(&uncovered);
        ctx.run_event(EventKind::Uncovered, "", message, json!(uncovered));
    }
    if !ctx.plan.is_unattended_on_a_physical_robot() || outcome.uncovered.is_empty() {
        return None;
    }
    let watched: Vec<&Uncovered> = outcome.uncovered.iter().collect();
    Some(RunRefusal {
        message: outside_the_envelope(&watched),
        errors: Vec::new(),
        assumption: None,
        says: None,
    })
}

/// Each value of `uncovered`, as its text says it, separated by `; `.
pub fn listed(uncovered: &[&Uncovered]) -> String {
    let texts: Vec<String> = uncovered.iter().map(ToString::to_string).collect();
    texts.join("; ")
}

/// The refusal of an unattended run on a physical robot with the values
/// `uncovered`, which no test covered.
pub fn outside_the_envelope(uncovered: &[&Uncovered]) -> String {
    format!(
        "an unattended run on a physical robot runs inside the envelope of its bundle, and no test covered {}",
        listed(uncovered)
    )
}

/// Evaluates the check of each assumption, within its bound; a false check
/// refuses the run and names the assumption, but in a run of a trial, whose
/// checks the operator judges. A check that gives no value refuses the run.
async fn assumptions<H: Host>(ctx: &RunCtx<'_, H>, outcome: &mut Outcome) -> Option<RunRefusal> {
    for (name, assumption) in &ctx.plan.workflow.assumptions {
        let Some(check) = &assumption.check else {
            continue;
        };
        let value = ctx
            .bounded_check(&format!("the check of `{name}`"), || ctx.truth(check))
            .await;
        let kind = if assumption.reach {
            AssumptionKind::Reach
        } else {
            AssumptionKind::Check
        };
        outcome.assumptions.push(AssumptionOutcome {
            name: name.clone(),
            says: assumption.says.clone(),
            kind,
            value: value.as_ref().ok().copied(),
            confirmed: false,
        });
        match value {
            Ok(true) => {}
            Ok(false) if ctx.plan.of_a_trial => {}
            Ok(false) => {
                return Some(RunRefusal {
                    message: format!("the assumption `{name}` is false: {}", assumption.says),
                    errors: Vec::new(),
                    assumption: Some(name.clone()),
                    says: Some(assumption.says.clone()),
                });
            }
            Err(failure) => {
                return Some(RunRefusal {
                    message: format!("the check of `{name}` failed: {failure}"),
                    errors: Vec::new(),
                    assumption: Some(name.clone()),
                    says: Some(assumption.says.clone()),
                });
            }
        }
    }
    None
}

/// How the judgement of the checks of a run of a trial ends its wait.
enum Judgement {
    Starts,
    Refused(RunRefusal),
    Stopped(Stop),
}

/// Gives each check of a run of a trial with its value (`checks_to_judge`)
/// and waits for the judgement of the operator; a stop ends the wait. A work
/// cell that is not ready, or a wrong check, refuses the run, naming the
/// checks.
async fn judgement<H: Host>(
    ctx: &RunCtx<'_, H>,
    outcome: &Outcome,
    commands: &mut mpsc::UnboundedReceiver<RunCommand>,
    phase: &watch::Sender<Phase>,
) -> Judgement {
    let assumptions = &ctx.plan.workflow.assumptions;
    let checks: Vec<CheckToJudge> = outcome
        .assumptions
        .iter()
        .filter_map(|checked| {
            let check = assumptions.get(&checked.name)?.check.as_ref()?;
            Some(CheckToJudge {
                assumption: checked.name.clone(),
                says: checked.says.clone(),
                check: check.source().to_owned(),
                value: checked.value?,
            })
        })
        .collect();
    phase.send_replace(Phase::Judging(checks.clone()));
    ctx.run_event(EventKind::ChecksToJudge, "", String::new(), json!(checks));
    loop {
        let Some(command) = commands.recv().await else {
            return Judgement::Stopped(Stop::Interrupted);
        };
        let judged = match command {
            RunCommand::JudgeChecks { judged } => judged,
            other => match before_the_lanes(ctx, other) {
                Some(stop) => return Judgement::Stopped(stop),
                None => continue,
            },
        };
        let named = |names: Vec<&Name>| {
            let quoted: Vec<String> = names.iter().map(|name| format!("`{name}`")).collect();
            quoted.join(", ")
        };
        let message = match judged {
            ChecksJudged::Starts => return Judgement::Starts,
            ChecksJudged::NotReady => {
                let false_checks = checks
                    .iter()
                    .filter(|check| !check.value)
                    .map(|check| &check.assumption)
                    .collect();
                format!(
                    "the work cell is not ready: the check of {} is false, and the operator agrees with each value",
                    named(false_checks)
                )
            }
            ChecksJudged::FalseCheck { wrong } => format!(
                "the operator names the check of {} wrong: its value differs from the work cell",
                named(wrong.iter().collect())
            ),
        };
        return Judgement::Refused(RunRefusal {
            message,
            errors: Vec::new(),
            assumption: None,
            says: None,
        });
    }
}

/// Waits for a confirmation of each assumption without a check that the
/// run document does not confirm; a stop ends the wait.
async fn confirmations<H: Host>(
    ctx: &RunCtx<'_, H>,
    outcome: &mut Outcome,
    commands: &mut mpsc::UnboundedReceiver<RunCommand>,
    phase: &watch::Sender<Phase>,
) -> Option<Stop> {
    let mut waiting: BTreeMap<Name, String> = ctx
        .plan
        .workflow
        .assumptions
        .iter()
        .filter(|(name, assumption)| {
            assumption.check.is_none() && !ctx.plan.confirmed.contains(*name)
        })
        .map(|(name, assumption)| (name.clone(), assumption.says.clone()))
        .collect();
    for (name, assumption) in &ctx.plan.workflow.assumptions {
        if assumption.check.is_none() && ctx.plan.confirmed.contains(name) {
            outcome.assumptions.push(AssumptionOutcome {
                name: name.clone(),
                says: assumption.says.clone(),
                kind: AssumptionKind::Confirmation,
                value: None,
                confirmed: true,
            });
        }
    }
    while !waiting.is_empty() {
        let to_confirm: Vec<ToConfirm> = waiting
            .iter()
            .map(|(assumption, says)| ToConfirm {
                assumption: assumption.clone(),
                says: says.clone(),
            })
            .collect();
        phase.send_replace(Phase::Confirming(to_confirm.clone()));
        ctx.run_event(EventKind::ToConfirm, "", String::new(), json!(to_confirm));
        let Some(command) = commands.recv().await else {
            return Some(Stop::Interrupted);
        };
        match command {
            RunCommand::Confirm { assumption } => {
                if let Some(says) = waiting.remove(&assumption) {
                    outcome.assumptions.push(AssumptionOutcome {
                        name: assumption,
                        says,
                        kind: AssumptionKind::Confirmation,
                        value: None,
                        confirmed: true,
                    });
                }
            }
            other => {
                if let Some(stop) = before_the_lanes(ctx, other) {
                    return Some(stop);
                }
            }
        }
    }
    None
}

/// Serves the commands of the run while it checks, before or after its
/// lanes, until one stops it.
async fn stopped_while_checking<H: Host>(
    ctx: &RunCtx<'_, H>,
    commands: &mut mpsc::UnboundedReceiver<RunCommand>,
) -> Stop {
    loop {
        let Some(command) = commands.recv().await else {
            return Stop::Interrupted;
        };
        if let Some(stop) = before_the_lanes(ctx, command) {
            return stop;
        }
    }
}

/// A command of a run whose lanes do not run: a confirmation and a
/// suspension change nothing, a resume is refused, a fault waits for the
/// first branch of its lane, and an abort or a stop gives the stop of the
/// run.
fn before_the_lanes<H: Host>(ctx: &RunCtx<'_, H>, command: RunCommand) -> Option<Stop> {
    match command {
        RunCommand::Confirm { .. } | RunCommand::Suspend | RunCommand::JudgeChecks { .. } => None,
        RunCommand::Resume { reply } => {
            let _ = reply.send(Err(ResumeRefusal::NotSuspended));
            None
        }
        RunCommand::Abort => Some(Stop::Aborted),
        RunCommand::Stop { halted, .. } => Some(Stop::of_halt(halted)),
        RunCommand::Fault { role, failure } => {
            ctx.fault(&role, failure);
            None
        }
        RunCommand::Alerts {
            role,
            source,
            standing,
        } => {
            ctx.alerts_stand(&role, source, standing);
            None
        }
    }
}

/// The lanes of `roles` at once (the initialization trees, or the lanes of
/// the actions), with the commands of the run between them: a suspension
/// of the run once no lane runs and one is suspended, and a resume after
/// its checks.
async fn lanes<H: Host>(
    ctx: &RunCtx<'_, H>,
    roles: &[Name],
    initialization: bool,
    commands: &mut mpsc::UnboundedReceiver<RunCommand>,
    phase: &watch::Sender<Phase>,
) -> BTreeMap<Name, StepEnd> {
    let mut parks = BTreeMap::new();
    let lanes: Vec<(Lane<'_, H>, mpsc::UnboundedReceiver<ParkCommand>)> = roles
        .iter()
        .map(|role| {
            let (park_tx, park_rx) = mpsc::unbounded_channel();
            parks.insert(role.clone(), park_tx);
            let lane = Lane {
                ctx,
                role: role.clone(),
                robot: ctx.plan.roles[role].robot.clone(),
                clock: LaneClock::new(),
                hold: Hold::new(),
            };
            (lane, park_rx)
        })
        .collect();
    for role in roles {
        ctx.shared.set_lane(role, LaneState::Waiting, None, None);
    }
    // Made before any lane runs, since a lane can reach its suspension in
    // its first poll.
    let mut changes = ctx.shared.changes();
    let runs = lanes.into_iter().map(|(lane, park)| async move {
        let mut scope = Scope::new();
        let tree = match initialization {
            true => &ctx.plan.workflow.initialization[&lane.role],
            false => &ctx.plan.workflow.lanes[&lane.role].tree,
        };
        let end = tokio::select! {
            end = lane.run_tree(tree, &mut scope) => end,
            never = lane.serve_suspensions(park) => match never {},
        };
        if !initialization {
            lane.ended(&end);
        }
        (lane.role.clone(), end)
    });
    // The commands run beside the lanes and beside the checks of a resume.
    let control = async {
        let mut suspended = false;
        // A closed channel answers at once and for good: the run stops once.
        let mut caller_gone = false;
        let mut resuming: Option<Resuming<'_>> = None;
        loop {
            tokio::select! {
                command = commands.recv(), if !caller_gone => match command {
                    None => {
                        caller_gone = true;
                        ctx.stop_run(Stop::Interrupted, "the caller of the run is gone");
                    }
                    Some(command) => {
                        command_while_running(ctx, command, &parks, suspended, phase, &mut resuming)
                            .await
                    }
                },
                answer = async { resuming.as_mut().expect("a resume runs").checks.as_mut().await },
                    if resuming.is_some() =>
                {
                    for reply in resuming.take().map(|r| r.replies).unwrap_or_default() {
                        let _ = reply.send(answer.clone());
                    }
                }
                _ = changes.changed() => {
                    let now_suspended = run_is_suspended(ctx, roles);
                    if now_suspended != suspended {
                        suspended = now_suspended;
                        match suspended {
                            true => {
                                let suspension = suspension(ctx, roles, None);
                                ctx.run_event(EventKind::RunSuspends, RUN_SUSPENDED, String::new(), json!(suspension));
                                phase.send_replace(Phase::Suspended(suspension));
                            }
                            false => {
                                phase.send_replace(Phase::Running);
                            }
                        }
                    }
                }
            }
        }
    };
    // A command goes before the next step of a lane.
    tokio::select! {
        biased;
        _ = control => unreachable!("the commands of a run never end"),
        ends = futures::future::join_all(runs) => ends.into_iter().collect(),
    }
}

/// A run is suspended when one of its lanes is suspended and none runs.
fn run_is_suspended<H: Host>(ctx: &RunCtx<'_, H>, roles: &[Name]) -> bool {
    ctx.shared.read(|s| {
        let states: Vec<LaneState> = roles
            .iter()
            .filter_map(|role| s.lanes.get(role).map(|l| l.state))
            .collect();
        states.contains(&LaneState::Suspended)
            && !states
                .iter()
                .any(|state| matches!(state, LaneState::Running | LaneState::Waiting))
    })
}

fn suspension<H: Host>(
    ctx: &RunCtx<'_, H>,
    roles: &[Name],
    refusal: Option<Located>,
) -> Suspension {
    ctx.shared.read(|s| Suspension {
        lanes: roles
            .iter()
            .filter(|role| s.is_suspended(role))
            .map(|role| SuspendedLane {
                role: role.clone(),
                step: s.lanes[role].step.clone().unwrap_or_default(),
            })
            .collect(),
        incidents: s.incidents.clone(),
        resume_refusal: refusal,
    })
}

/// A resume whose checks run, and the callers that wait for its answer.
struct Resuming<'a> {
    checks: Pin<Box<dyn Future<Output = Result<(), ResumeRefusal>> + Send + 'a>>,
    replies: Vec<oneshot::Sender<Result<(), ResumeRefusal>>>,
}

/// A command of a run whose lanes run. A resume starts the checks of the
/// parked lanes, and a resume that comes while they run takes the answer of
/// the first; an abort or a stop ends the checks, and their callers read
/// the end of the run.
async fn command_while_running<'a, H: Host>(
    ctx: &'a RunCtx<'_, H>,
    command: RunCommand,
    parks: &'a BTreeMap<Name, mpsc::UnboundedSender<ParkCommand>>,
    suspended: bool,
    phase: &'a watch::Sender<Phase>,
    resuming: &mut Option<Resuming<'a>>,
) {
    match command {
        RunCommand::Confirm { .. } | RunCommand::JudgeChecks { .. } => {}
        RunCommand::Suspend => {
            for park in parks.values() {
                let _ = park.send(ParkCommand::Suspend);
            }
        }
        RunCommand::Resume { reply } => match resuming {
            Some(running) => running.replies.push(reply),
            None => {
                *resuming = Some(Resuming {
                    checks: Box::pin(resume(ctx, parks, suspended, phase)),
                    replies: vec![reply],
                });
            }
        },
        RunCommand::Abort => {
            *resuming = None;
            // Each suspended lane closes its own open lease as it leaves its
            // park on the stop, so the lanes of the run end after the closes.
            ctx.stop_run(Stop::Aborted, "aborted");
        }
        RunCommand::Stop { halted, reason } => {
            *resuming = None;
            ctx.stop_run(Stop::of_halt(halted), &reason);
        }
        RunCommand::Fault { role, failure } => ctx.fault(&role, failure),
        RunCommand::Alerts {
            role,
            source,
            standing,
        } => ctx.alerts_stand(&role, source, standing),
    }
}

/// The roles of the suspended lanes, whose leases stand open.
fn parked_roles<H: Host>(
    ctx: &RunCtx<'_, H>,
    parks: &BTreeMap<Name, mpsc::UnboundedSender<ParkCommand>>,
) -> Vec<Name> {
    ctx.shared.read(|s| {
        parks
            .keys()
            .filter(|role| s.is_suspended(role))
            .cloned()
            .collect()
    })
}

/// Resumes the suspended lanes: it closes the open lease of each of their
/// robots, checks each lane and the reach checks of their roles, and lets
/// them go on; a refused resume opens the leases again.
async fn resume<H: Host>(
    ctx: &RunCtx<'_, H>,
    parks: &BTreeMap<Name, mpsc::UnboundedSender<ParkCommand>>,
    suspended: bool,
    phase: &watch::Sender<Phase>,
) -> Result<(), ResumeRefusal> {
    if !suspended {
        return Err(ResumeRefusal::NotSuspended);
    }
    let parked = parked_roles(ctx, parks);
    let mut closed = Vec::new();
    let refused = 'checks: {
        for role in &parked {
            let robot = &ctx.plan.roles[role].robot;
            match ctx.host.set_lease_open(robot, false).await {
                Ok(()) => closed.push(robot),
                Err(error) => {
                    break 'checks Some(Located {
                        at: role.to_string(),
                        message: format!("the lease of `{robot}` did not close: {error}"),
                    });
                }
            }
        }
        for role in &parked {
            if let Some(alert) = ctx.standing_alert(role) {
                let robot = &ctx.plan.roles[role].robot;
                break 'checks Some(Located {
                    at: role.to_string(),
                    message: format!(
                        "an alert of severity 2 or 3 stands on `{robot}`: {alert}; resume once it clears"
                    ),
                });
            }
        }
        for role in &parked {
            let (reply, answer) = oneshot::channel();
            let _ = parks[role].send(ParkCommand::Check(reply));
            match answer.await {
                Ok(Ok(())) => {}
                Ok(Err(refusal)) => break 'checks Some(refusal),
                Err(_) => {
                    break 'checks Some(Located {
                        at: role.to_string(),
                        message: "the lane is gone".to_owned(),
                    });
                }
            }
        }
        let reach_checks_of_parked_roles =
            ctx.plan
                .workflow
                .assumptions
                .iter()
                .filter_map(|(name, assumption)| {
                    let check = assumption.check.as_ref().filter(|_| assumption.reach)?;
                    let of_a_parked_role = check.roots().iter().any(|root| {
                        ctx.plan
                            .observations
                            .get(root)
                            .is_some_and(|o| parked.contains(&o.target.role))
                    });
                    of_a_parked_role.then_some((name, check))
                });
        for (name, check) in reach_checks_of_parked_roles {
            let message = match ctx
                .bounded_check(&format!("the reach check of `{name}`"), || ctx.truth(check))
                .await
            {
                Ok(true) => continue,
                Ok(false) => format!("the reach check `{}` is false", check.source()),
                Err(failure) => failure.to_string(),
            };
            break 'checks Some(Located {
                at: format!("assumptions.{name}"),
                message,
            });
        }
        None
    };
    if let Some(refusal) = refused {
        for robot in closed {
            let _ = ctx.host.set_lease_open(robot, true).await;
        }
        let roles: Vec<Name> = parks.keys().cloned().collect();
        let suspension = suspension(ctx, &roles, Some(refusal.clone()));
        ctx.run_event(
            EventKind::RunSuspends,
            RUN_SUSPENDED,
            refusal.message.clone(),
            json!(suspension),
        );
        phase.send_replace(Phase::Suspended(suspension));
        return Err(ResumeRefusal::Check(refusal));
    }
    for role in &parked {
        let _ = parks[role].send(ParkCommand::Go);
    }
    Ok(())
}

async fn finish<H: Host>(
    ctx: &RunCtx<'_, H>,
    base: ResultBase,
    ending: Ending,
    refusal: Option<RunRefusal>,
    outcome: Outcome,
    started: (u64, u64),
) -> RunResult {
    // A fault that came once its lane ended is an incident of that lane.
    for role in ctx.plan.roles.keys() {
        if let Some(failure) = ctx.take_fault(role) {
            let step = ctx.shared.read(|s| {
                s.lanes
                    .get(role)
                    .and_then(|lane| lane.step.clone())
                    .unwrap_or_default()
            });
            ctx.record_incident(role, step, &failure, &[]);
        }
    }
    let (lanes, loops, incidents, observations) = ctx.shared.read(|s| {
        let lanes = ctx
            .plan
            .roles
            .iter()
            .map(|(role, bound)| {
                let status = s.lanes.get(role);
                LaneOutcome {
                    role: role.clone(),
                    robot: bound.robot.clone(),
                    state: status.map(|l| l.state).unwrap_or(LaneState::Waiting),
                    step: status.and_then(|l| l.step.clone()),
                    failure_kind: status.and_then(|l| l.failure_kind.clone()),
                }
            })
            .collect();
        let observations = s
            .last_values
            .iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect();
        (
            lanes,
            s.loops.values().cloned().collect(),
            s.incidents.clone(),
            observations,
        )
    });
    let ns = |from: u64, to: u64| to.saturating_sub(from) as f64 / 1e9;
    let ResultBase { mut result } = base;
    let mut uncovered = std::mem::take(&mut result.uncovered);
    uncovered.extend(outcome.uncovered);
    RunResult {
        ending,
        lanes,
        refusal,
        failed: outcome.failed,
        assumptions: outcome.assumptions,
        loops,
        observations,
        duration: Duration {
            run_s: ns(started.0, ctx.host.now()),
            wall_s: ns(started.1, ctx.host.wall_now()),
        },
        incidents,
        watch: outcome.watch,
        uncovered,
        ..result
    }
}

/// The status of a run in its phase, as `status` and `progress` give it.
pub fn status_of(phase: &Phase) -> RunStatus {
    let (to_confirm, suspension, result) = match phase {
        Phase::Confirming(to_confirm) => (to_confirm.clone(), None, None),
        Phase::Suspended(suspension) => (Vec::new(), Some(suspension.clone()), None),
        Phase::Ended(result) => (Vec::new(), None, Some((**result).clone())),
        Phase::Preparing | Phase::Judging(_) | Phase::Running | Phase::Ending => {
            (Vec::new(), None, None)
        }
    };
    RunStatus {
        state: phase.state(),
        to_confirm,
        suspension,
        result,
    }
}

#[cfg(test)]
mod tests;
