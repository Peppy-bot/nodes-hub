//! The interpreter of one lane: each step on the run clock, its bound, its
//! invariant, its retries and its attempts; the incidents of the lane, which
//! suspend it at the step where they happen in an attended run and fail it
//! in an unattended run; and the suspension of the lane, which parks each of
//! its branches at its step until a resume or the stop of the run.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Value as Json, json};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use workflow_documents::expression::{Expression, RefPath, Type, Value};
use workflow_documents::result::{
    Anomaly, LaneState, LoopCount, SafeStateOutcome, SafeStateResult,
};
use workflow_documents::run::Attendance;
use workflow_documents::workflow::{Call, Policy, SafeCall, Step, StepKind};
use workflow_documents::{EventKind, Located, Name, RobotName};

use super::failure::{
    ATTEMPT_FAILED, Cause, Class, DirectorKind, Failure, LEADER_STREAMING, PRECONDITION_FAILED,
    RESOURCE_UNAVAILABLE, TIMEOUT,
};
use super::hold::{BranchCommand, Hold, Met};
use super::host::{CallEnd, Host};
use super::plan::{BoundCall, CallSite, MemberKind, read_path, request};
use super::state::LaneClock;
use super::{Interrupt, LEADER_RETRY_NS, NoInterrupt, RunCtx};

/// How a run stops every lane at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stop {
    Aborted,
    Halted,
    Interrupted,
}

impl Stop {
    /// The stop of a run that the daemon refused or ended: a halt of the
    /// stack (`halted`), else an interruption.
    pub(crate) fn of_halt(halted: bool) -> Stop {
        match halted {
            true => Stop::Halted,
            false => Stop::Interrupted,
        }
    }
}

/// How a step ended.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum StepEnd {
    Done,
    /// A pass failure that the nearest enclosing loop counts.
    Failed(Failure),
    /// An incident at `step` that failed the lane: it ends every enclosing
    /// step.
    LaneFailed {
        step: String,
        failure: Failure,
    },
    /// A wait or a call that the run, or the closure of its signals, asks to
    /// suspend.
    Suspend,
    /// An enclosing step ended the step: a parallel step that ended, a bound
    /// that passed, an invariant that broke.
    Cancelled,
    Stopped(Stop),
}

/// What the run tells a lane about its suspension.
pub(crate) enum ParkCommand {
    /// Suspend the lane.
    Suspend,
    /// Check that each parked branch of the lane can go on: the invariant
    /// of each enclosing step.
    Check(oneshot::Sender<Result<(), Located>>),
    Go,
}

/// The local values of a lane: the results named so far and the loop
/// elements.
pub(crate) type Scope = BTreeMap<Name, Json>;

/// A read of a step: its answer or the failure of the read, unless an
/// interruption ended the read first (the outer error).
type Read<T> = Result<Result<T, Failure>, StepEnd>;

/// What interrupts the work of a lane: the stop of the run and the cancel of
/// the step always, and with `lane`, a suspension of the lane and its faults.
#[derive(Clone, Copy)]
struct Interrupts {
    lane: bool,
}

impl Interrupts {
    /// Every interruption: of a wait or a call of the lane, and of a wait
    /// for the first message of a topic.
    const ALL: Interrupts = Interrupts { lane: true };
    /// The stop and the cancel alone: of a read that answers once its
    /// producer does.
    const RUN: Interrupts = Interrupts { lane: false };
}

/// The wait of a read of a step, whose token is `cancel`, for the first
/// message of a topic: every interruption of the lane ends it.
struct StepWait<'w, 'a, H: Host> {
    lane: &'w Lane<'a, H>,
    cancel: &'w CancellationToken,
}

impl<H: Host> Interrupt for StepWait<'_, '_, H> {
    type End = StepEnd;

    fn interrupted(&self) -> impl Future<Output = StepEnd> + Send {
        self.lane.next_interruption(self.cancel, Interrupts::ALL)
    }
}

type Boxed<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Where a step runs: the pass of the loop it runs in, if any; what the
/// items of the enclosing `for_each` steps add to its name, `#<index>` for
/// each; the invariants of the steps that enclose it, the earliest of their
/// deadlines in lane time, and the token that cancels it.
#[derive(Clone)]
struct Within<'s> {
    pass: Option<Pass>,
    items: String,
    guards: Vec<&'s Expression>,
    deadline: Option<u64>,
    cancel: CancellationToken,
}

/// A pass of a loop: whether a `match` in it took `otherwise`.
#[derive(Clone, Default)]
struct Pass {
    fell_back: Arc<AtomicBool>,
}

impl Pass {
    fn fall_back(&self) {
        self.fell_back.store(true, Ordering::Relaxed);
    }

    fn fell_back(&self) -> u32 {
        u32::from(self.fell_back.load(Ordering::Relaxed))
    }
}

/// One lane of a run.
pub(crate) struct Lane<'a, H: Host> {
    pub ctx: &'a RunCtx<'a, H>,
    pub role: Name,
    pub robot: RobotName,
    pub clock: LaneClock,
    pub hold: Hold,
}

impl<'a, H: Host> Lane<'a, H> {
    fn named(&self, id: &impl Display) -> String {
        format!("{}.{id}", self.role)
    }

    /// The name of `step` where it runs: `<role>.<id>`, with `#<index>` for
    /// the item of each enclosing `for_each`.
    fn name_in(&self, step: &Step, within: &Within<'_>) -> String {
        format!("{}.{}{}", self.role, step.id, within.items)
    }

    fn lane_now(&self) -> u64 {
        self.clock.lane_time(self.ctx.host.now())
    }

    /// Resolves when lane time reaches `lane_at`.
    async fn lane_until(&self, lane_at: u64) {
        let mut changes = self.clock.changes();
        loop {
            if self.lane_now() >= lane_at {
                return;
            }
            match self.clock.run_time_of(lane_at) {
                Some(run_at) => tokio::select! {
                    _ = self.ctx.host.until(run_at) => {}
                    _ = changes.changed() => {}
                },
                None => {
                    let _ = changes.changed().await;
                }
            }
        }
    }

    fn event(&self, kind: EventKind, step: String, failure_kind: &str, details: Json) {
        self.ctx
            .event(&self.role, kind, step, failure_kind, String::new(), details);
    }

    fn at_step(&self, name: &str) {
        self.ctx.shared.change(|s| {
            if let Some(lane) = s.lanes.get_mut(&self.role) {
                lane.step = Some(name.to_owned());
            }
        });
    }

    /// Whether the lane stands suspended, its lease open for a take-over.
    fn is_suspended(&self) -> bool {
        self.ctx.shared.read(|s| s.is_suspended(&self.role))
    }

    /// The step the lane is at, or was last at.
    fn last_step(&self) -> String {
        self.ctx
            .shared
            .read(|s| s.lanes.get(&self.role).and_then(|lane| lane.step.clone()))
            .unwrap_or_default()
    }

    /// Runs a tree of the lane, then the safe state of the lane when the
    /// tree fails or the run aborts it while it runs, and records the
    /// incidents its branches met; gives the end of the tree.
    pub async fn run_tree(&self, tree: &Step, scope: &mut Scope) -> StepEnd {
        self.ctx.shared.set_lane(
            &self.role,
            LaneState::Running,
            Some(self.named(&tree.id)),
            None,
        );
        let within = Within {
            pass: None,
            items: String::new(),
            guards: Vec::new(),
            deadline: None,
            cancel: CancellationToken::new(),
        };
        let end = match self.run_step(tree, scope, within).await {
            StepEnd::Done => self.fault_at_the_end().await,
            end => end,
        };
        // An abort closes the open lease of a suspended lane as the lane
        // leaves its park, before the lane ends.
        if end == StepEnd::Stopped(Stop::Aborted) && self.is_suspended() {
            let _ = self.ctx.host.set_lease_open(&self.robot, false).await;
        }
        let safe_state = match &end {
            StepEnd::LaneFailed { .. } => self.safe_state().await,
            StepEnd::Stopped(Stop::Aborted) if !self.hold.parked() => self.safe_state().await,
            _ => Vec::new(),
        };
        self.record_incidents(self.hold.take_met(), &safe_state);
        end
    }

    /// The end of a tree that completed while a fault of its lane waits for a
    /// branch: an incident at the last step of the lane, which fails the lane
    /// in an unattended run, and suspends it in an attended run until a
    /// resume ends it.
    async fn fault_at_the_end(&self) -> StepEnd {
        let Some(failure) = self.ctx.take_fault(&self.role) else {
            return StepEnd::Done;
        };
        let step = self.last_step();
        self.hold.meet(Met {
            step: step.clone(),
            failure: failure.clone(),
        });
        if self.ctx.plan.attendance == Attendance::Unattended {
            return StepEnd::LaneFailed { step, failure };
        }
        match self.park_branch(step, || async { Ok(true) }).await {
            Parked::Stopped(stop) => StepEnd::Stopped(stop),
            Parked::Resumed | Parked::Completed => StepEnd::Done,
        }
    }

    /// Marks the lane ended, with the event of its end.
    pub fn ended(&self, end: &StepEnd) {
        let state = match end {
            StepEnd::Done => LaneState::Completed,
            StepEnd::Failed(_) | StepEnd::LaneFailed { .. } => LaneState::Failed,
            StepEnd::Stopped(Stop::Aborted) => LaneState::Aborted,
            _ => LaneState::Interrupted,
        };
        let failure_kind = match end {
            StepEnd::Failed(f) | StepEnd::LaneFailed { failure: f, .. } => Some(f.kind.clone()),
            _ => None,
        };
        let last = self.last_step();
        self.ctx
            .shared
            .set_lane(&self.role, state, Some(last.clone()), failure_kind.clone());
        self.event(
            EventKind::LaneEnds,
            last,
            failure_kind.as_deref().unwrap_or_default(),
            json!(state),
        );
    }

    /// Runs a step, and meets an incident at the step where it happens: an
    /// attended run suspends the lane, parks the branch here and runs the
    /// step again from its first attempt at the resume; an unattended run
    /// fails the lane. A wait goes on at the resume with the time it had
    /// left, since the lane time counts nothing while the lane is suspended.
    fn run_step<'s>(
        &'s self,
        step: &'s Step,
        scope: &'s mut Scope,
        within: Within<'s>,
    ) -> Boxed<'s, StepEnd> {
        Box::pin(async move {
            let name = self.name_in(step, &within);
            let first_start = self.lane_now();
            loop {
                let started = match &step.kind {
                    StepKind::WaitFor { .. } | StepKind::WaitUntil { .. } => first_start,
                    _ => self.lane_now(),
                };
                let end = self.attempts(step, scope, within.clone(), started).await;
                let failure = match end {
                    StepEnd::Failed(f) if f.class != Class::Lane && within.pass.is_some() => {
                        return StepEnd::Failed(f);
                    }
                    StepEnd::Failed(f) => Some(f),
                    StepEnd::Suspend => None,
                    other => return other,
                };
                if let Some(failure) = failure {
                    self.hold.meet(Met {
                        step: name.clone(),
                        failure: failure.clone(),
                    });
                    if self.ctx.plan.attendance == Attendance::Unattended {
                        return StepEnd::LaneFailed {
                            step: name,
                            failure,
                        };
                    }
                }
                let parked: &Scope = scope;
                let check = || self.resume_check(&name, step, parked, &within.guards);
                match self.park_branch(name.clone(), check).await {
                    Parked::Resumed => continue,
                    Parked::Completed => return StepEnd::Done,
                    Parked::Stopped(stop) => return StepEnd::Stopped(stop),
                }
            }
        })
    }

    /// The attempts of a step within its bound from `started`, a lane time,
    /// and under its invariant, which it checks at the start and at each
    /// poll tick.
    async fn attempts<'s>(
        &'s self,
        step: &'s Step,
        scope: &mut Scope,
        within: Within<'s>,
        started: u64,
    ) -> StepEnd {
        let name = self.name_in(step, &within);
        let own = step.timeout.map(|bound| started + bound.nanos());
        // A bound of an enclosing step that passes no later than this
        // step's own takes the failure: the outermost step whose bound
        // passed owns it.
        let deadline = own.filter(|own| within.deadline.is_none_or(|outer| *own < outer));
        if let Some(invariant) = &step.invariant
            && let Err(end) = self
                .holds_by(invariant, scope, &within.cancel, deadline, step)
                .await
        {
            return end;
        }
        let mut guards = within.guards.clone();
        guards.extend(step.invariant.as_ref());
        self.at_step(&name);
        self.event(EventKind::StepStarts, name.clone(), "", Json::Null);
        let cancel = within.cancel.child_token();
        let inner = Within {
            pass: within.pass.clone(),
            items: within.items.clone(),
            guards,
            deadline: [within.deadline, own].into_iter().flatten().min(),
            cancel: cancel.clone(),
        };
        let poll = self.poll_of(step);
        let observed = scope.clone();
        let end = {
            let work = self.with_retries(step, scope, inner);
            tokio::pin!(work);
            let mut tick = started + poll;
            loop {
                let wake = match (deadline, &step.invariant) {
                    (Some(d), Some(_)) => d.min(tick),
                    (Some(d), None) => d,
                    (None, Some(_)) => tick,
                    (None, None) => u64::MAX,
                };
                tokio::select! {
                    end = &mut work => break end,
                    _ = self.lane_until(wake), if wake != u64::MAX => {}
                }
                if deadline.is_some_and(|d| self.lane_now() >= d) {
                    cancel.cancel();
                    break after_settle(timed_out(&step.id), (&mut work).await);
                }
                if let Some(invariant) = &step.invariant {
                    tick = next_tick(started, poll, self.lane_now());
                    if let Err(end) = self
                        .holds_by(invariant, &observed, &cancel, deadline, step)
                        .await
                    {
                        cancel.cancel();
                        break after_settle(end, (&mut work).await);
                    }
                }
            }
        };
        if !matches!(
            end,
            StepEnd::Suspend | StepEnd::Stopped(_) | StepEnd::Cancelled
        ) {
            let failure_kind = match &end {
                StepEnd::Failed(f) | StepEnd::LaneFailed { failure: f, .. } => f.kind.clone(),
                _ => String::new(),
            };
            self.event(
                EventKind::StepEnds,
                name,
                &failure_kind,
                step_result(step, scope),
            );
        }
        end
    }

    fn poll_of(&self, step: &Step) -> u64 {
        match &step.kind {
            StepKind::WaitUntil {
                poll: Some(poll), ..
            } => poll.nanos(),
            _ => self.ctx.plan.poll.nanos(),
        }
    }

    /// Runs the kind of a step again after each retryable failure while its
    /// retries last. A call checks its `done_when` again first, so an action
    /// that completed despite an error never runs twice.
    async fn with_retries<'s>(
        &'s self,
        step: &'s Step,
        scope: &mut Scope,
        within: Within<'s>,
    ) -> StepEnd {
        let mut left = step.retry;
        loop {
            let end = self.kind(step, scope, within.clone()).await;
            let StepEnd::Failed(failure) = &end else {
                return end;
            };
            if !failure.is_retryable() || left == 0 {
                return end;
            }
            left -= 1;
            if let StepKind::Call(Call {
                done_when: Some(done_when),
                ..
            }) = &step.kind
            {
                // An interruption ends the step; a read that failed is a call
                // that is not done.
                match self
                    .read(
                        &within.cancel,
                        self.step_truth(done_when, scope, &within.cancel),
                    )
                    .await
                {
                    Ok(Ok(true)) => return StepEnd::Done,
                    Ok(Ok(false) | Err(_)) => {}
                    Err(end) => return end,
                }
            }
        }
    }

    fn kind<'s>(
        &'s self,
        step: &'s Step,
        scope: &'s mut Scope,
        within: Within<'s>,
    ) -> Boxed<'s, StepEnd> {
        Box::pin(async move {
            match &step.kind {
                StepKind::Call(call) => self.call_step(step, call, scope, &within).await,
                StepKind::Sequence(steps) => {
                    for child in steps {
                        let end = self.run_step(child, scope, within.clone()).await;
                        if end != StepEnd::Done {
                            return end;
                        }
                    }
                    StepEnd::Done
                }
                StepKind::Parallel { policy, steps } => {
                    self.parallel(*policy, steps, scope, &within).await
                }
                StepKind::If {
                    cond,
                    then,
                    otherwise,
                } => match self
                    .read_or_fail(&within.cancel, self.step_truth(cond, scope, &within.cancel))
                    .await
                {
                    Err(end) => end,
                    Ok(true) => self.run_step(then, scope, within).await,
                    Ok(false) => match otherwise {
                        Some(otherwise) => self.run_step(otherwise, scope, within).await,
                        None => StepEnd::Done,
                    },
                },
                StepKind::Match {
                    on,
                    cases,
                    otherwise,
                } => {
                    let value = match self
                        .read_or_fail(&within.cancel, self.step_value(on, scope, &within.cancel))
                        .await
                    {
                        Ok(Value::String(value)) => value,
                        Ok(_) => {
                            return StepEnd::Failed(Failure::new(
                                PRECONDITION_FAILED,
                                format!("`{}` gives no enum value", on.source()),
                            ));
                        }
                        Err(end) => return end,
                    };
                    match (cases.get(&value), otherwise) {
                        (Some(case), _) => self.run_step(case, scope, within).await,
                        (None, Some(otherwise)) => {
                            if let Some(pass) = &within.pass {
                                pass.fall_back();
                            }
                            self.run_step(otherwise, scope, within).await
                        }
                        (None, None) => StepEnd::Failed(Failure::new(
                            PRECONDITION_FAILED,
                            format!("no case of `{}` is '{value}'", step.id),
                        )),
                    }
                }
                StepKind::ForEach {
                    list,
                    element,
                    max_items,
                    body,
                } => {
                    let items = match self
                        .read_or_fail(&within.cancel, self.step_list(list, scope, &within.cancel))
                        .await
                    {
                        Ok(items) => items,
                        Err(end) => return end,
                    };
                    if items.len() > max_items.get() as usize {
                        return StepEnd::Failed(Failure::new(
                            PRECONDITION_FAILED,
                            format!(
                                "`{}` gives {} items, more than its `max_items` of {max_items}",
                                list.source(),
                                items.len()
                            ),
                        ));
                    }
                    for (index, item) in items.into_iter().enumerate() {
                        let mut inside = scope.clone();
                        inside.insert(element.clone(), item);
                        let pass = Pass::default();
                        let item_within = Within {
                            pass: Some(pass.clone()),
                            items: format!("{}#{index}", within.items),
                            ..within.clone()
                        };
                        let end = self.run_step(body, &mut inside, item_within).await;
                        if let Some(end) = self.counted(step, &within, &pass, end) {
                            return end;
                        }
                    }
                    StepEnd::Done
                }
                StepKind::RepeatUntil {
                    until,
                    fail_when,
                    max_iterations,
                    breaker,
                    body,
                } => {
                    let mut failed = 0u32;
                    for _ in 0..max_iterations.get() {
                        match self
                            .test(until, fail_when.as_ref(), step, scope, &within.cancel)
                            .await
                        {
                            Ok(true) => return StepEnd::Done,
                            Ok(false) => {}
                            Err(end) => return end,
                        }
                        let mut inside = scope.clone();
                        let pass = Pass::default();
                        let pass_within = Within {
                            pass: Some(pass.clone()),
                            ..within.clone()
                        };
                        let end = self.run_step(body, &mut inside, pass_within).await;
                        let pass_failed = matches!(end, StepEnd::Failed(_));
                        if let Some(end) = self.counted(step, &within, &pass, end) {
                            return end;
                        }
                        failed += u32::from(pass_failed);
                        if pass_failed && breaker.is_some_and(|max| failed >= max.get()) {
                            return StepEnd::Failed(Failure::of(
                                DirectorKind::BreakerOpen,
                                format!("{failed} passes of `{}` failed", step.id),
                            ));
                        }
                    }
                    match self
                        .test(until, fail_when.as_ref(), step, scope, &within.cancel)
                        .await
                    {
                        Ok(true) => StepEnd::Done,
                        Ok(false) => StepEnd::Failed(Failure::of(
                            DirectorKind::BoundReached,
                            format!("`{}` reached its {max_iterations} iterations", step.id),
                        )),
                        Err(end) => end,
                    }
                }
                StepKind::WaitUntil {
                    until, fail_when, ..
                } => {
                    let poll = self.poll_of(step);
                    let waiting = async {
                        let start = self.lane_now();
                        loop {
                            match self
                                .test(until, fail_when.as_ref(), step, scope, &within.cancel)
                                .await
                            {
                                Ok(true) => return Ok(()),
                                Ok(false) => {}
                                Err(end) => return Err(end),
                            }
                            self.lane_until(next_tick(start, poll, self.lane_now()))
                                .await;
                        }
                    };
                    match self.leaf(&within.cancel, waiting, false).await {
                        Ok(()) => StepEnd::Done,
                        Err(end) => end,
                    }
                }
                StepKind::Signal(signal) => {
                    self.ctx
                        .shared
                        .change(|s| *s.pending.entry(signal.clone()).or_default() += 1);
                    StepEnd::Done
                }
                StepKind::WaitFor { signals, result } => {
                    match self
                        .leaf(&within.cancel, self.wait_for(signals), false)
                        .await
                    {
                        Ok(signal) => {
                            if let Some(result) = result {
                                scope.insert(
                                    result.clone(),
                                    json!({ "signal": signal.to_string() }),
                                );
                            }
                            StepEnd::Done
                        }
                        Err(end) => end,
                    }
                }
            }
        })
    }

    /// The test of a loop or a wait: `fail_when` first, then its condition.
    async fn test(
        &self,
        until: &Expression,
        fail_when: Option<&Expression>,
        step: &Step,
        scope: &Scope,
        cancel: &CancellationToken,
    ) -> Result<bool, StepEnd> {
        if let Some(fail_when) = fail_when
            && self
                .read_or_fail(cancel, self.step_truth(fail_when, scope, cancel))
                .await?
        {
            return Err(StepEnd::Failed(Failure::of(
                DirectorKind::ConditionFailed,
                format!("`fail_when` of `{}` is true", step.id),
            )));
        }
        self.read_or_fail(cancel, self.step_truth(until, scope, cancel))
            .await
    }

    /// Consumes one pending event of the first signal of `signals` that has
    /// one; fails with signal_unreachable once every lane that can emit them
    /// has ended, and asks to suspend while every lane that can still emit
    /// them is suspended.
    async fn wait_for(&self, signals: &[Name]) -> Result<Name, StepEnd> {
        let mut changes = self.ctx.shared.changes();
        loop {
            // A pending event is taken under one lock with its check, so two
            // waiters never take the same one; the rest only reads, so the
            // wait wakes at the changes of others, never at its own.
            if let Some(signal) = self.ctx.shared.take_signal(signals) {
                return Ok(signal);
            }
            let found = self.ctx.shared.read(|s| {
                let emitters: BTreeSet<&Name> = signals
                    .iter()
                    .filter_map(|signal| s.emitters.get(signal))
                    .flatten()
                    .filter(|lane| **lane != self.role)
                    .collect();
                let live: Vec<&Name> = emitters
                    .into_iter()
                    .filter(|lane| !s.lanes.get(*lane).is_some_and(|status| status.is_ended()))
                    .collect();
                if live.is_empty() {
                    return Err(StepEnd::Failed(Failure::of(
                        DirectorKind::SignalUnreachable,
                        "every lane that can emit the signals it waits for has ended",
                    )));
                }
                let all_suspended = live.iter().all(|lane| s.is_suspended(lane));
                if all_suspended {
                    return Err(StepEnd::Suspend);
                }
                Ok(())
            });
            found?;
            if changes.changed().await.is_err() {
                return Err(StepEnd::Stopped(Stop::Interrupted));
            }
        }
    }

    /// The interruption of `interrupts` that has come already: the stop of
    /// the run, then a suspension asked of the lane. They are read at once,
    /// since a watch can wait out the task's budget.
    fn interruption_now(&self, interrupts: Interrupts) -> Option<StepEnd> {
        if let Some(stop) = *self.ctx.stop.borrow() {
            return Some(StepEnd::Stopped(stop));
        }
        (interrupts.lane && self.hold.is_suspending()).then_some(StepEnd::Suspend)
    }

    /// The first interruption of `interrupts` to come, in this order: the
    /// stop of the run, the suspension of the lane, a fault of the lane (an
    /// alert, a stopped clock), which it takes, and the cancel of `cancel`.
    async fn next_interruption(
        &self,
        cancel: &CancellationToken,
        interrupts: Interrupts,
    ) -> StepEnd {
        tokio::select! {
            biased;
            stop = self.stop_of_the_run() => StepEnd::Stopped(stop),
            () = self.hold.suspending(), if interrupts.lane => StepEnd::Suspend,
            fault = self.next_fault(), if interrupts.lane => StepEnd::Failed(fault),
            () = cancel.cancelled() => StepEnd::Cancelled,
        }
    }

    /// Races a wait or a call with every interruption of the lane
    /// ([`Interrupts::ALL`]). A run that stopped or a lane that suspends
    /// starts no work; each interruption cancels the work, and a call
    /// (`settle`) then ends as its goal ends, and a wait is dropped.
    async fn leaf<T, F: Future<Output = Result<T, StepEnd>>>(
        &self,
        cancel: &CancellationToken,
        work: F,
        settle: bool,
    ) -> Result<T, StepEnd> {
        if let Some(end) = self.interruption_now(Interrupts::ALL) {
            return Err(end);
        }
        tokio::pin!(work);
        let interrupted = tokio::select! {
            biased;
            end = self.next_interruption(cancel, Interrupts::ALL) => end,
            end = &mut work => return end,
        };
        cancel.cancel();
        if !settle {
            return Err(interrupted);
        }
        let settled = (&mut work).await.err().unwrap_or(StepEnd::Done);
        Err(after_settle(interrupted, settled))
    }

    /// The answer of `reading`, a read of the step whose token is `cancel`,
    /// or the failure of the read, unless the run stops or the token is
    /// cancelled first ([`Interrupts::RUN`]): the step then ends as each
    /// says, and a service call of the read in flight is dropped. A read
    /// that waits for the first message of a topic meets every interruption
    /// of the lane inside that wait (`StepWait`); every other read answers
    /// once its producer does, and the lane meets a suspension or a fault at
    /// its next wait or call.
    async fn read<T>(
        &self,
        cancel: &CancellationToken,
        reading: impl Future<Output = Read<T>>,
    ) -> Read<T> {
        if let Some(end) = self.interruption_now(Interrupts::RUN) {
            return Err(end);
        }
        tokio::select! {
            biased;
            end = self.next_interruption(cancel, Interrupts::RUN) => Err(end),
            read = reading => read,
        }
    }

    /// `read`, whose failure fails the step.
    async fn read_or_fail<T>(
        &self,
        cancel: &CancellationToken,
        reading: impl Future<Output = Read<T>>,
    ) -> Result<T, StepEnd> {
        self.read(cancel, reading).await?.map_err(StepEnd::Failed)
    }

    /// The stop of the run, once it comes.
    async fn stop_of_the_run(&self) -> Stop {
        let mut stop = self.ctx.stop.subscribe();
        stop.wait_for(Option::is_some)
            .await
            .ok()
            .and_then(|stop| *stop)
            .unwrap_or(Stop::Interrupted)
    }

    /// The next fault of the lane, which it takes, so one branch of the lane
    /// meets each fault once.
    async fn next_fault(&self) -> Failure {
        let mut faults = self.ctx.faults[&self.role].subscribe();
        loop {
            if faults.wait_for(Option::is_some).await.is_err() {
                return std::future::pending().await;
            }
            if let Some(failure) = self.ctx.take_fault(&self.role) {
                return failure;
            }
        }
    }

    /// A call step: the call with its arguments, the resends while a leader
    /// streams the limb, the result, the `done_when` and the monitor.
    async fn call_step(
        &self,
        step: &Step,
        call: &Call,
        scope: &mut Scope,
        within: &Within<'_>,
    ) -> StepEnd {
        let site = CallSite {
            role: self.role.clone(),
            step: step.id.clone(),
        };
        let bound = self
            .ctx
            .plan
            .calls
            .get(&site)
            .expect("the checker binds each call step");
        let settle = bound.target.kind == MemberKind::Action;
        let name = self.name_in(step, within);
        let result = match self
            .leaf(
                &within.cancel,
                self.send(bound, scope, &within.cancel),
                settle,
            )
            .await
        {
            Ok(result) => result,
            Err(end) => return end,
        };
        if let Some(name) = &call.result {
            scope.insert(name.clone(), result);
        }
        let Some(done_when) = &call.done_when else {
            return StepEnd::Done;
        };
        match self
            .read_or_fail(
                &within.cancel,
                self.step_truth(done_when, scope, &within.cancel),
            )
            .await
        {
            Ok(true) => self.monitored(name, &within.cancel).await,
            Ok(false) => StepEnd::Failed(Failure::new(
                ATTEMPT_FAILED,
                format!("`done_when` of `{}` is false", step.id),
            )),
            Err(end) => end,
        }
    }

    /// Sends a bound call until it ends, again every second of run time
    /// while a leader's stream refuses it, so a call of the safe state of a
    /// park, whose lane time stands, goes again too.
    async fn send(
        &self,
        bound: &BoundCall,
        scope: &Scope,
        cancel: &CancellationToken,
    ) -> Result<Json, StepEnd> {
        let request = request(&bound.args, scope).map_err(|missing| {
            StepEnd::Failed(Failure::new(PRECONDITION_FAILED, missing.to_string()))
        })?;
        loop {
            // A step cancelled before a try sends nothing, as when the leaf
            // settles work that it never started.
            if cancel.is_cancelled() {
                return Err(StepEnd::Cancelled);
            }
            let end = match bound.target.kind {
                MemberKind::Service => tokio::select! {
                    end = self.ctx.host.call(&bound.target, request.clone(), cancel.clone()) => end,
                    _ = cancel.cancelled() => return Err(StepEnd::Cancelled),
                },
                MemberKind::Action => {
                    self.ctx
                        .host
                        .call(&bound.target, request.clone(), cancel.clone())
                        .await
                }
            };
            match end {
                CallEnd::Completed(result) => return Ok(result),
                // A leader that streams runs no goal of this call, so a cancel
                // before the next try ends the step cancelled.
                CallEnd::Refused(f) | CallEnd::Failed(f) if f.kind == LEADER_STREAMING => {
                    let again = self.ctx.host.now() + LEADER_RETRY_NS;
                    tokio::select! {
                        () = self.ctx.host.until(again) => {}
                        _ = cancel.cancelled() => return Err(StepEnd::Cancelled),
                    }
                }
                CallEnd::Refused(f) | CallEnd::Failed(f) => return Err(StepEnd::Failed(f)),
                CallEnd::Cancelled { asked: true, .. } => return Err(StepEnd::Cancelled),
                CallEnd::Cancelled {
                    asked: false,
                    reason,
                } => {
                    return Err(StepEnd::Failed(Failure::of(
                        DirectorKind::Stopped,
                        format!("another client stopped the goal: {reason}"),
                    )));
                }
                CallEnd::Unknown => {
                    return Err(StepEnd::Failed(Failure::new(
                        RESOURCE_UNAVAILABLE,
                        "the goal did not end within the cancel timeout; the robot is in an unknown state",
                    )));
                }
                CallEnd::MoveRefused { halted, message } => {
                    let stop = Stop::of_halt(halted);
                    self.ctx.stop_run(stop, &message);
                    return Err(StepEnd::Stopped(stop));
                }
            }
        }
    }

    /// The anomaly monitor after a `done_when` that holds: an anomaly fails
    /// the step, unless the run is unattended and its severity lets the run
    /// go on.
    async fn monitored(&self, step: String, cancel: &CancellationToken) -> StepEnd {
        let Some(monitor) = &self.ctx.plan.monitor else {
            return StepEnd::Done;
        };
        let request = json!({ "robot": self.robot.to_string(), "step": step });
        let asked = async {
            match self
                .ctx
                .host
                .call(&monitor.target, request, cancel.clone())
                .await
            {
                CallEnd::Completed(answer) => Ok(answer),
                CallEnd::Refused(f) | CallEnd::Failed(f) => Err(f),
                _ => Err(Failure::new(
                    RESOURCE_UNAVAILABLE,
                    "the anomaly monitor gave no answer",
                )),
            }
        };
        let answer = match self.read_or_fail(cancel, async { Ok(asked.await) }).await {
            Ok(answer) => answer,
            Err(end) => return end,
        };
        if answer.get("anomaly") != Some(&Json::Bool(true)) {
            return StepEnd::Done;
        }
        let anomaly: Anomaly = match serde_json::from_value(answer.clone()) {
            Ok(anomaly) => anomaly,
            Err(error) => {
                return StepEnd::Failed(Failure::new(
                    RESOURCE_UNAVAILABLE,
                    format!("the anomaly monitor gave an anomaly that does not read: {error}"),
                ));
            }
        };
        let lets_go_on = self.ctx.plan.attendance == Attendance::Unattended
            && monitor.continue_on_severity.contains(&anomaly.severity);
        if lets_go_on {
            return StepEnd::Done;
        }
        let message = format!(
            "{} anomaly: {}",
            answer["severity"].as_str().unwrap_or_default(),
            anomaly.description
        );
        StepEnd::Failed(
            Failure::of(DirectorKind::Anomaly, message).because(Cause::Anomaly(anomaly)),
        )
    }

    /// The children of a parallel step at once. The first failure of a child
    /// fails the step; `any` ends at the first success; when the step ends it
    /// cancels the children that still run.
    async fn parallel(
        &self,
        policy: Policy,
        steps: &[Step],
        scope: &mut Scope,
        within: &Within<'_>,
    ) -> StepEnd {
        let siblings = within.cancel.child_token();
        let mut scopes: Vec<Scope> = steps.iter().map(|_| scope.clone()).collect();
        self.hold.split(steps.len());
        let children: Vec<_> = steps
            .iter()
            .zip(scopes.iter_mut())
            .map(|(child, child_scope)| {
                let siblings = siblings.clone();
                let child_within = Within {
                    cancel: siblings.clone(),
                    ..within.clone()
                };
                async move {
                    let end = self.run_step(child, child_scope, child_within).await;
                    self.hold.branch_ends();
                    let ends_the_step = match end {
                        StepEnd::Done => policy == Policy::Any,
                        StepEnd::Cancelled => false,
                        _ => true,
                    };
                    if ends_the_step {
                        siblings.cancel();
                    }
                    end
                }
            })
            .collect();
        let ends = futures::future::join_all(children).await;
        self.hold.join();
        let first_end = ends
            .iter()
            .find(|end| !matches!(end, StepEnd::Done | StepEnd::Cancelled))
            .cloned();
        if let Some(end) = first_end {
            return end;
        }
        if within.cancel.is_cancelled() {
            return StepEnd::Cancelled;
        }
        for (end, child_scope) in ends.iter().zip(scopes) {
            if *end == StepEnd::Done {
                scope.extend(child_scope);
            }
        }
        StepEnd::Done
    }

    /// Whether an invariant holds, read within `deadline`, the bound of the
    /// step that reads it, when it has one: a read of a topic that has given
    /// no message yet waits for one, and a read past the bound fails the
    /// step with `timeout`.
    async fn holds_by(
        &self,
        invariant: &Expression,
        scope: &Scope,
        cancel: &CancellationToken,
        deadline: Option<u64>,
        step: &Step,
    ) -> Result<(), StepEnd> {
        let Some(deadline) = deadline else {
            return self.holds(invariant, scope, cancel).await;
        };
        tokio::select! {
            held = self.holds(invariant, scope, cancel) => held,
            () = self.lane_until(deadline) => Err(timed_out(&step.id)),
        }
    }

    /// Whether an invariant holds; a false one fails the lane with
    /// invariant_violated, never retried.
    async fn holds(
        &self,
        invariant: &Expression,
        scope: &Scope,
        cancel: &CancellationToken,
    ) -> Result<(), StepEnd> {
        match self
            .read_or_fail(cancel, self.step_truth(invariant, scope, cancel))
            .await
        {
            Ok(true) => Ok(()),
            Ok(false) => Err(StepEnd::Failed(Failure::of(
                DirectorKind::InvariantViolated,
                format!("`{}` is false", invariant.source()),
            ))),
            Err(end) => Err(end),
        }
    }

    /// The types of the locals of the lane's role.
    fn locals(&self) -> Option<&BTreeMap<Name, Type>> {
        self.ctx.plan.locals.get(&self.role)
    }

    /// The truth of `expression` in a step whose token is `cancel`: a read
    /// of it that waits for the first message of a topic meets every
    /// interruption of the lane.
    async fn step_truth(
        &self,
        expression: &Expression,
        scope: &Scope,
        cancel: &CancellationToken,
    ) -> Read<bool> {
        self.ctx
            .truth_in(
                expression,
                scope,
                self.locals(),
                &StepWait { lane: self, cancel },
            )
            .await
    }

    /// The value of `expression` in a step whose token is `cancel`, as
    /// `step_truth` reads it.
    async fn step_value(
        &self,
        expression: &Expression,
        scope: &Scope,
        cancel: &CancellationToken,
    ) -> Read<Value> {
        self.ctx
            .value(
                expression,
                scope,
                self.locals(),
                &StepWait { lane: self, cancel },
            )
            .await
    }

    /// The items of the list of a `for_each` in a step whose token is
    /// `cancel`, as `step_truth` reads it.
    async fn step_list(
        &self,
        list: &Expression,
        scope: &Scope,
        cancel: &CancellationToken,
    ) -> Read<Vec<Json>> {
        self.list(list, scope, &StepWait { lane: self, cancel })
            .await
    }

    /// The truth of `expression` outside a step: a check of a resume, which
    /// its own bound ends.
    async fn truth_at_rest(&self, expression: &Expression, scope: &Scope) -> Result<bool, Failure> {
        let Ok(truth) = self
            .ctx
            .truth_in(expression, scope, self.locals(), &NoInterrupt)
            .await;
        truth
    }

    /// The items of the list of a `for_each`, evaluated once at its start.
    async fn list<W: Interrupt>(
        &self,
        list: &Expression,
        scope: &Scope,
        wait: &W,
    ) -> Result<Result<Vec<Json>, Failure>, W::End> {
        let no_list = || {
            Failure::new(
                PRECONDITION_FAILED,
                format!("`{}` gives no list", list.source()),
            )
        };
        match self.ctx.value(list, scope, self.locals(), wait).await? {
            Ok(Value::List(_)) => {}
            Ok(_) => return Ok(Err(no_list())),
            Err(failure) => return Ok(Err(failure)),
        }
        let Ok(path) = RefPath::parse(list.source()) else {
            return Ok(Err(no_list()));
        };
        let mut all = scope.clone();
        if self.ctx.plan.observations.contains_key(&path.root) && !all.contains_key(&path.root) {
            match self.ctx.observe(&path.root, wait).await? {
                Ok(json) => all.insert(path.root.clone(), json),
                Err(failure) => return Ok(Err(failure)),
            };
        }
        Ok(match read_path(&all, &path) {
            Ok(Json::Array(items)) => Ok(items),
            _ => Err(no_list()),
        })
    }

    /// Counts a pass of the loop `step` that ended `end`, and gives the end
    /// that ends the loop: a pass that neither completed nor failed.
    fn counted(
        &self,
        step: &Step,
        within: &Within<'_>,
        pass: &Pass,
        end: StepEnd,
    ) -> Option<StepEnd> {
        let failed = match end {
            StepEnd::Done => 0,
            StepEnd::Failed(_) => 1,
            other => return Some(other),
        };
        let fallback = pass.fell_back();
        self.count(step, within, |count| {
            count.passes += 1;
            count.failed += failed;
            count.fallback += fallback;
        });
        None
    }

    fn count(&self, step: &Step, within: &Within<'_>, change: impl FnOnce(&mut LoopCount)) {
        let key = self.name_in(step, within);
        self.ctx.shared.change(|s| {
            change(s.loops.entry(key.clone()).or_insert(LoopCount {
                step: key,
                passes: 0,
                failed: 0,
                fallback: 0,
            }))
        });
    }

    /// Records the incidents that the branches of the lane met, each with
    /// the result of the safe state that the lane ran for them.
    fn record_incidents(&self, met: Vec<Met>, safe_state: &[SafeStateResult]) {
        for Met { step, failure } in met {
            self.ctx
                .record_incident(&self.role, step, &failure, safe_state);
        }
    }

    /// Runs the calls of the lane's safe state in order, each within its own
    /// bound on the run clock, up to the first that fails. A halt, a lease
    /// end or the loss of the caller of the run cancels the call it meets
    /// and sends no more. Its arguments are literals, so it reads no scope.
    pub(crate) async fn safe_state(&self) -> Vec<SafeStateResult> {
        let calls: &[SafeCall] = self
            .ctx
            .plan
            .workflow
            .lanes
            .get(&self.role)
            .map(|lane| lane.safe_state.as_slice())
            .unwrap_or_default();
        let literals = Scope::new();
        let mut results = Vec::new();
        for call in calls {
            if self.ctx.is_cut_off() {
                break;
            }
            let site = CallSite {
                role: self.role.clone(),
                step: call.id.clone(),
            };
            let bound = self
                .ctx
                .plan
                .calls
                .get(&site)
                .expect("the checker binds each safe-state call");
            let cancel = CancellationToken::new();
            let deadline = self.ctx.host.now() + call.timeout.nanos();
            let sent = self.send(bound, &literals, &cancel);
            tokio::pin!(sent);
            let interrupted = tokio::select! {
                end = &mut sent => Err(end),
                () = self.ctx.host.until(deadline) => Ok(timed_out(&call.id)),
                stop = self.ctx.until_cut_off() => Ok(StepEnd::Stopped(stop)),
            };
            let end = match interrupted {
                Err(end) => end,
                Ok(interrupted) => {
                    cancel.cancel();
                    let settled = (&mut sent).await.err().unwrap_or(StepEnd::Done);
                    Err(after_settle(interrupted, settled))
                }
            };
            let outcome = match &end {
                Ok(_) => SafeStateOutcome::Completed,
                Err(StepEnd::Failed(f)) => SafeStateOutcome::Failed {
                    failure_kind: f.kind.clone(),
                },
                Err(_) => SafeStateOutcome::Stopped,
            };
            results.push(SafeStateResult {
                step: self.named(&call.id),
                outcome,
            });
            if end.is_err() {
                break;
            }
        }
        results
    }

    /// Parks the branch at `step` in the suspension of the lane until the
    /// lane resumes it or the run stops. `check` gives whether the branch can
    /// go on, at each check of a resume and once it resumes: `Ok(true)` ends
    /// its step, and `Ok(false)` runs it again.
    async fn park_branch<F, Fut>(&self, step: String, check: F) -> Parked
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<bool, Located>>,
    {
        let mut commands = match self.hold.park(step, &self.ctx.stop) {
            Ok(commands) => commands,
            Err(stop) => return Parked::Stopped(stop),
        };
        let leave = loop {
            match commands.recv().await {
                Some(BranchCommand::Check(reply)) => {
                    let _ = reply.send(check().await);
                }
                Some(BranchCommand::Go) => break None,
                Some(BranchCommand::Stop) => {
                    break Some(self.ctx.stop.borrow().unwrap_or(Stop::Interrupted));
                }
                None => break Some(Stop::Interrupted),
            }
        };
        self.hold.unpark();
        if let Some(stop) = leave {
            return Parked::Stopped(stop);
        }
        match check().await {
            Ok(true) => Parked::Completed,
            _ => Parked::Resumed,
        }
    }

    /// Serves the suspensions of the lane while its tree runs: it parks
    /// the lane once every branch of a suspension is parked, passes the
    /// commands of the run to the parked branches, and tells them when the
    /// run stops.
    pub async fn serve_suspensions(
        &self,
        mut commands: mpsc::UnboundedReceiver<ParkCommand>,
    ) -> std::convert::Infallible {
        let mut stop = self.ctx.stop.subscribe();
        let stopped = async move {
            let _ = stop.wait_for(Option::is_some).await;
        };
        tokio::pin!(stopped);
        loop {
            tokio::select! {
                biased;
                () = &mut stopped => break,
                command = commands.recv() => match command {
                    Some(command) => self.on_park_command(command).await,
                    None => break,
                },
                () = self.hold.settled() => self.park().await,
            }
        }
        self.hold.stop_branches();
        std::future::pending().await
    }

    /// Parks the lane: its lane time stands from here, then its safe state,
    /// the incidents its branches met, the open lease of its robot for a
    /// take-over, then the suspended lane.
    async fn park(&self) {
        if self.ctx.stop.borrow().is_some() {
            return;
        }
        let step = self.hold.mark_parked();
        self.clock.pause(self.ctx.host.now());
        let safe_state = self.safe_state().await;
        let met = self.hold.take_met();
        let failure_kind = met.first().map(|met| met.failure.kind.clone());
        self.record_incidents(met, &safe_state);
        if self.ctx.stop.borrow().is_some() {
            return;
        }
        let lease = match self.ctx.host.set_lease_open(&self.robot, true).await {
            Ok(()) => String::new(),
            Err(error) => format!(
                "the lease of `{}` did not open for a take-over: {error}",
                self.robot
            ),
        };
        // A stop that came while the lease opened found no parked lane to
        // close: the lane closes the lease itself, and does not suspend.
        if self.ctx.stop.borrow().is_some() {
            let _ = self.ctx.host.set_lease_open(&self.robot, false).await;
            return;
        }
        self.ctx.shared.set_lane(
            &self.role,
            LaneState::Suspended,
            Some(step.clone()),
            failure_kind.clone(),
        );
        self.ctx.event(
            &self.role,
            EventKind::LaneSuspends,
            step,
            failure_kind.as_deref().unwrap_or_default(),
            lease,
            json!(LaneState::Suspended),
        );
    }

    async fn on_park_command(&self, command: ParkCommand) {
        match command {
            ParkCommand::Suspend => self.hold.request(),
            ParkCommand::Check(reply) => {
                let mut answer = Ok(());
                for branch in self.hold.branches() {
                    let (check, checked) = oneshot::channel();
                    let _ = branch.send(BranchCommand::Check(check));
                    let refusal = match checked.await {
                        Ok(Ok(_)) => continue,
                        Ok(Err(refusal)) => refusal,
                        Err(_) => Located {
                            at: self.role.to_string(),
                            message: "the lane is gone".to_owned(),
                        },
                    };
                    answer = Err(refusal);
                    break;
                }
                let _ = reply.send(answer);
            }
            ParkCommand::Go => {
                let (step, branches) = self.hold.resume();
                self.clock.resume(self.ctx.host.now());
                self.ctx
                    .shared
                    .set_lane(&self.role, LaneState::Running, Some(step.clone()), None);
                self.event(EventKind::LaneResumes, step, "", json!(LaneState::Running));
                for branch in branches {
                    let _ = branch.send(BranchCommand::Go);
                }
            }
        }
    }

    /// The checks of a resume in the context of the lane, each within the
    /// bound of a check outside a step: the invariant of each enclosing step,
    /// and whether the `done_when` of its call holds (`Ok(true)`, which
    /// completes the step).
    async fn resume_check(
        &self,
        name: &str,
        step: &Step,
        scope: &Scope,
        enclosing: &[&Expression],
    ) -> Result<bool, Located> {
        for invariant in enclosing {
            let what = format!("the invariant `{}`", invariant.source());
            let message = match self
                .ctx
                .bounded_check(&what, || self.truth_at_rest(invariant, scope))
                .await
            {
                Ok(true) => continue,
                Ok(false) => format!("{what} is false"),
                Err(failure) => failure.to_string(),
            };
            return Err(Located {
                at: name.to_owned(),
                message,
            });
        }
        match &step.kind {
            StepKind::Call(Call {
                done_when: Some(done_when),
                ..
            }) => Ok(self
                .ctx
                .bounded_check("`done_when`", || self.truth_at_rest(done_when, scope))
                .await
                == Ok(true)),
            _ => Ok(false),
        }
    }
}

/// The failure of the step or the safe-state call `id`, whose bound passed.
fn timed_out(id: &impl Display) -> StepEnd {
    StepEnd::Failed(Failure::new(TIMEOUT, format!("the bound of `{id}` passed")))
}

/// The first poll tick after `now` of the ticks at `start + k * poll`, so the
/// ticks that a slow read passed merge into it.
fn next_tick(start: u64, poll: u64, now: u64) -> u64 {
    start + poll * (now.saturating_sub(start) / poll + 1)
}

/// What a step that an interruption ended reports once its work settled:
/// a failure of the lane that the settle gives, such as a goal that did not
/// end within the cancel timeout and left the robot in an unknown state,
/// comes before the interruption, and a stop of the run before both.
fn after_settle(interrupted: StepEnd, settled: StepEnd) -> StepEnd {
    match (&interrupted, settled) {
        (StepEnd::Stopped(_), _) => interrupted,
        (_, StepEnd::Failed(failure)) if failure.class == Class::Lane => StepEnd::Failed(failure),
        _ => interrupted,
    }
}

/// How a suspension ended.
enum Parked {
    Resumed,
    Completed,
    Stopped(Stop),
}

/// The named result of a step, for the `step_ends` event.
fn step_result(step: &Step, scope: &Scope) -> Json {
    match &step.kind {
        StepKind::Call(Call {
            result: Some(name), ..
        }) => scope.get(name).cloned().unwrap_or(Json::Null),
        _ => Json::Null,
    }
}
