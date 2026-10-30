//! The record of a run's inputs and its replay. While a run goes, the
//! [`Recorded`] host writes each call with how it ended, each topic read,
//! each answer to an open or a close of a lease and the plan's inputs to the
//! journal; the director writes each command. A
//! replay rebuilds the plan from the journal and drives the engine again on
//! a clock of its own, with each call answered as the journal says, and
//! compares the decisions of each lane: the calls it sends and the events it
//! gives.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value as Json, json};
use tokio::sync::{mpsc, oneshot, watch};
use tokio_util::sync::CancellationToken;
use workflow_documents::result::Ending;
use workflow_documents::run::RunDocument;
use workflow_documents::{EventKind, Name, RobotName, Validation, Workflow};

use crate::checker::{self, RunChoices};
use crate::config::AllowedContracts;
use crate::contract::ContractShape;
use crate::engine::failure::Failure;
use crate::engine::host::{CallEnd, Event, Host, Join, Joining};
use crate::engine::plan::{Target, TopicKey, Watch};
use crate::engine::{self, Phase, ResultBase, RunCommand};
use crate::fleet::{Fleet, Snapshot};
use crate::journal::{self, Journals, Record};

/// The inputs of the plan of a run, as its journal keeps them.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Planned {
    pub fleet: Fleet,
    /// The pinned documents of the contracts of the fleet, by sha256.
    pub documents: BTreeMap<String, String>,
    pub models: BTreeMap<String, String>,
    pub allowed: String,
    pub binding: BTreeMap<Name, RobotName>,
    pub run_document: RunDocument,
    pub poll_s: f64,
    pub cancel_timeout_ns: u64,
    /// The run clock and wall time when the engine started.
    pub run_ns: u64,
    pub wall_ns: u64,
}

/// A call of a run with how it ended, and its place among the calls of its
/// role in the order they were sent. A call with no end is one that the
/// engine dropped at the run time of its record, as a read past its bound.
#[derive(Clone, Serialize, Deserialize)]
struct CallRecord {
    index: u64,
    role: String,
    contract: String,
    member: String,
    instance: String,
    request: Json,
    end: Option<CallEnd>,
}

/// The record of a call that runs, which the journal gets when the engine
/// drops the call before it ends.
struct Unanswered<'a, 'h, H: Host> {
    recorder: &'a Recorded<'h, H>,
    record: Option<CallRecord>,
}

impl<H: Host> Drop for Unanswered<'_, '_, H> {
    fn drop(&mut self) {
        if let Some(record) = self.record.take() {
            self.recorder.write(journal::CALL, json!(record));
        }
    }
}

/// A topic read of a run: its latest message with the run time it arrived,
/// and the run time of the read.
#[derive(Clone, Serialize, Deserialize)]
struct ReadRecord {
    instance: String,
    contract: String,
    member: String,
    latest: Option<(Json, u64)>,
    read_ns: u64,
}

/// What the daemon answered to an open or a close of a lease of a run; the
/// run time of its record is when the answer came.
#[derive(Serialize, Deserialize)]
struct LeaseRecord {
    robot: String,
    open: bool,
    refusal: Option<String>,
}

/// A step of the read `nth` of `name` in the run, which joined another
/// reader's read in flight; the run time of its record is when it came. A
/// later record of one read replaces an earlier one.
#[derive(Serialize, Deserialize)]
struct SharedRecord {
    name: Name,
    nth: u64,
    join: Join,
}

/// A command to a run as its journal keeps it: `queued` when it came before
/// the engine started, which takes it before its first step.
#[derive(Clone, Serialize, Deserialize)]
struct CommandEntry {
    queued: bool,
    #[serde(flatten)]
    command: CommandRecord,
}

/// Writes `command` of `run` to its journal at `run_ns`.
pub fn journal_command(
    journals: &Journals,
    run: &str,
    run_ns: u64,
    queued: bool,
    command: &RunCommand,
) {
    let entry = CommandEntry {
        queued,
        command: CommandRecord::of(command),
    };
    if let Err(error) = journals.append(run, run_ns, journal::COMMAND, json!(entry)) {
        tracing::error!("the journal of {run} lost a record: {error}");
    }
}

/// A command to a run, as its journal keeps it.
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum CommandRecord {
    Confirm {
        assumption: Name,
    },
    Suspend,
    Resume,
    Abort,
    Stop {
        halted: bool,
        reason: String,
    },
    Fault {
        role: Name,
        failure: Failure,
    },
    Alerts {
        role: Name,
        source: String,
        standing: Vec<String>,
    },
}

impl CommandRecord {
    fn of(command: &RunCommand) -> CommandRecord {
        match command {
            RunCommand::Confirm { assumption, .. } => CommandRecord::Confirm {
                assumption: assumption.clone(),
            },
            RunCommand::Suspend => CommandRecord::Suspend,
            RunCommand::Resume { .. } => CommandRecord::Resume,
            RunCommand::Abort => CommandRecord::Abort,
            RunCommand::Stop { halted, reason } => CommandRecord::Stop {
                halted: *halted,
                reason: reason.clone(),
            },
            RunCommand::Alerts {
                role,
                source,
                standing,
            } => CommandRecord::Alerts {
                role: role.clone(),
                source: source.clone(),
                standing: standing.clone(),
            },
            RunCommand::Fault { role, failure } => CommandRecord::Fault {
                role: role.clone(),
                failure: failure.clone(),
            },
        }
    }

    fn command(self) -> RunCommand {
        match self {
            CommandRecord::Confirm { assumption } => RunCommand::Confirm { assumption },
            CommandRecord::Suspend => RunCommand::Suspend,
            CommandRecord::Resume => RunCommand::Resume {
                reply: oneshot::channel().0,
            },
            CommandRecord::Abort => RunCommand::Abort,
            CommandRecord::Stop { halted, reason } => RunCommand::Stop { halted, reason },
            CommandRecord::Fault { role, failure } => RunCommand::Fault { role, failure },
            CommandRecord::Alerts {
                role,
                source,
                standing,
            } => RunCommand::Alerts {
                role,
                source,
                standing,
            },
        }
    }
}

/// A host that writes each call and each topic read of `inner` to the
/// journal of `run`.
pub struct Recorded<'a, H: Host> {
    pub inner: H,
    journals: &'a Journals,
    run: String,
    /// The calls each role has sent.
    sent: Mutex<BTreeMap<String, u64>>,
}

impl<'a, H: Host> Recorded<'a, H> {
    pub fn new(inner: H, journals: &'a Journals, run: &str) -> Recorded<'a, H> {
        Recorded {
            inner,
            journals,
            run: run.to_owned(),
            sent: Mutex::new(BTreeMap::new()),
        }
    }

    fn write(&self, kind: &str, body: Json) {
        if let Err(error) = self
            .journals
            .append(&self.run, self.inner.now(), kind, body)
        {
            tracing::error!("the journal of {} lost a record: {error}", self.run);
        }
    }
}

impl<H: Host> Host for Recorded<'_, H> {
    fn now(&self) -> u64 {
        self.inner.now()
    }

    fn wall_now(&self) -> u64 {
        self.inner.wall_now()
    }

    async fn until(&self, at: u64) {
        self.inner.until(at).await
    }

    async fn call(&self, target: &Target, request: Json, cancel: CancellationToken) -> CallEnd {
        let index = {
            let mut sent = self.sent.lock().expect("no recorder panics");
            let count = sent.entry(target.role.to_string()).or_insert(0);
            *count += 1;
            *count - 1
        };
        let mut unanswered = Unanswered {
            recorder: self,
            record: Some(CallRecord {
                index,
                role: target.role.to_string(),
                contract: target.contract.clone(),
                member: target.member.to_string(),
                instance: target.producer.instance.clone(),
                request: request.clone(),
                end: None,
            }),
        };
        let end = self.inner.call(target, request, cancel).await;
        if let Some(record) = unanswered.record.take() {
            let answered = CallRecord {
                end: Some(end.clone()),
                ..record
            };
            self.write(journal::CALL, json!(answered));
        }
        end
    }

    fn latest(&self, target: &Target) -> Option<(Json, u64)> {
        let latest = self.inner.latest(target);
        let record = ReadRecord {
            instance: target.producer.instance.clone(),
            contract: target.contract.clone(),
            member: target.member.to_string(),
            latest: latest.clone(),
            read_ns: self.inner.now(),
        };
        self.write(journal::READ, json!(record));
        latest
    }

    async fn message_arrives(&self, target: &Target) {
        self.inner.message_arrives(target).await
    }

    async fn joining(&self, name: &Name, nth: u64) -> Joining {
        self.inner.joining(name, nth).await
    }

    fn joined(&self, name: &Name, nth: u64, join: &Join) {
        let record = SharedRecord {
            name: name.clone(),
            nth,
            join: join.clone(),
        };
        self.write(journal::SHARED, json!(record));
        self.inner.joined(name, nth, join);
    }

    async fn set_lease_open(&self, robot: &RobotName, open: bool) -> Result<(), String> {
        let answer = self.inner.set_lease_open(robot, open).await;
        let record = LeaseRecord {
            robot: robot.to_string(),
            open,
            refusal: answer.clone().err(),
        };
        self.write(journal::LEASE, json!(record));
        answer
    }

    fn event(&self, event: Event) {
        self.inner.event(event)
    }
}

/// What a replay found.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// Whether every decision of the replay is the decision of the run.
    pub same: bool,
    /// The decisions compared.
    pub decisions: u32,
    /// The first decision that differs, empty when none does.
    pub difference: String,
}

/// The decisions of a run, in order: the calls of each role, and the
/// events of each role and of the run (the empty role).
type Decisions = BTreeMap<(String, Stream), Vec<Decision>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Stream {
    Calls,
    Events,
}

/// A decision of a lane: a call it sent, or an event it gave.
#[derive(Clone, Debug, PartialEq)]
enum Decision {
    Call {
        contract: String,
        member: String,
        instance: String,
        request: Json,
    },
    Event {
        kind: String,
        step: String,
        failure_kind: String,
    },
}

impl Decision {
    fn of_call(record: &CallRecord) -> Decision {
        Decision::Call {
            contract: record.contract.clone(),
            member: record.member.clone(),
            instance: record.instance.clone(),
            request: record.request.clone(),
        }
    }
}

impl std::fmt::Display for Decision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Decision::Call {
                contract,
                member,
                instance,
                request,
            } => write!(f, "call {contract}.{member} on {instance} with {request}"),
            Decision::Event {
                kind,
                step,
                failure_kind,
            } => write!(f, "event {kind} {step} {failure_kind}"),
        }
    }
}

/// The run's own events: the director gives them, not its engine.
fn of_the_engine(kind: EventKind) -> bool {
    !matches!(
        kind,
        EventKind::RunStarts | EventKind::Differences | EventKind::RunEnds | EventKind::Progress
    )
}

/// The recorded calls of each role, with the run time each ended.
type Calls = BTreeMap<String, VecDeque<(u64, CallRecord)>>;

/// The recorded reads of each topic.
type Reads = BTreeMap<TopicKey, VecDeque<ReadRecord>>;

/// The recorded answers to the opens and closes of the lease of each robot,
/// with the run time each came.
type Leases = BTreeMap<String, VecDeque<(u64, Result<(), String>)>>;

/// The recorded reads that joined another reader's read, by the observation
/// and the place of the read among its reads: how each went on last, and
/// when.
type Shares = BTreeMap<(Name, u64), (u64, Join)>;

/// The journal of a run, read for its replay.
struct Recording {
    workflow: Workflow,
    /// The validation of the bundle of the run, when it has one.
    validation: Option<Validation>,
    planned: Planned,
    calls: Calls,
    reads: Reads,
    leases: Leases,
    shares: Shares,
    commands: Vec<(u64, CommandEntry)>,
    decisions: Decisions,
    ending: Option<Ending>,
}

fn body<T: for<'de> Deserialize<'de>>(record: &Record) -> Result<T, String> {
    serde_json::from_value(record.body.clone())
        .map_err(|e| format!("record {} ({}): {e}", record.seq, record.kind))
}

impl Recording {
    fn read(text: &str) -> Result<Recording, String> {
        let records: Vec<Record> = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                serde_json::from_str(line).map_err(|e| format!("a line of the journal: {e}"))
            })
            .collect::<Result<_, _>>()?;
        let opened = records
            .iter()
            .find(|r| r.kind == journal::OPENED)
            .ok_or("the journal has no opened record")?;
        let workflow = crate::director::workflow_of(&opened.body)?;
        let validation =
            crate::director::bundle_of(&opened.body)?.and_then(|bundle| bundle.document.validation);
        let planned = records
            .iter()
            .find(|r| r.kind == journal::PLANNED)
            .ok_or("the run ended before its plan: there is nothing to replay")?;
        let planned: Planned = body(planned)?;
        let mut recording = Recording {
            workflow,
            validation,
            planned,
            calls: BTreeMap::new(),
            reads: BTreeMap::new(),
            leases: BTreeMap::new(),
            shares: BTreeMap::new(),
            commands: Vec::new(),
            decisions: BTreeMap::new(),
            ending: None,
        };
        for record in &records {
            match record.kind.as_str() {
                journal::CALL => {
                    let call: CallRecord = body(record)?;
                    recording
                        .calls
                        .entry(call.role.clone())
                        .or_default()
                        .push_back((record.run_ns, call));
                }
                journal::READ => {
                    let read: ReadRecord = body(record)?;
                    recording
                        .reads
                        .entry((
                            read.instance.clone(),
                            read.contract.clone(),
                            read.member.clone(),
                        ))
                        .or_default()
                        .push_back(read);
                }
                journal::LEASE => {
                    let lease: LeaseRecord = body(record)?;
                    let answer = lease.refusal.map_or(Ok(()), Err);
                    recording
                        .leases
                        .entry(lease.robot)
                        .or_default()
                        .push_back((record.run_ns, answer));
                }
                journal::SHARED => {
                    let shared: SharedRecord = body(record)?;
                    recording
                        .shares
                        .insert((shared.name, shared.nth), (record.run_ns, shared.join));
                }
                journal::COMMAND => recording.commands.push((record.run_ns, body(record)?)),
                journal::EVENT => {
                    let name = record.body["event"].as_str().unwrap_or_default();
                    let kind = EventKind::parse(name).ok_or_else(|| {
                        format!(
                            "record {}: the event `{name}` is no event of a run",
                            record.seq
                        )
                    })?;
                    if of_the_engine(kind) {
                        let role = record.body["role"].as_str().unwrap_or_default().to_owned();
                        recording
                            .decisions
                            .entry((role, Stream::Events))
                            .or_default()
                            .push(Decision::Event {
                                kind: kind.name().to_owned(),
                                step: record.body["step"].as_str().unwrap_or_default().to_owned(),
                                failure_kind: record.body["failure_kind"]
                                    .as_str()
                                    .unwrap_or_default()
                                    .to_owned(),
                            });
                    }
                }
                journal::END => {
                    recording.ending = serde_json::from_value(record.body["ending"].clone()).ok()
                }
                _ => {}
            }
        }
        for (role, calls) in recording.calls.iter_mut() {
            calls.make_contiguous().sort_by_key(|(_, call)| call.index);
            let decisions = calls
                .iter()
                .map(|(_, call)| Decision::of_call(call))
                .collect();
            recording
                .decisions
                .insert((role.clone(), Stream::Calls), decisions);
        }
        Ok(recording)
    }
}

/// The host of a replay: a clock that starts where the run's started and
/// moves only while every task waits, and each call answered as the
/// journal says, at the run time it ended.
struct ReplayHost {
    base_ns: u64,
    wall_base_ns: u64,
    start: tokio::time::Instant,
    calls: Mutex<Calls>,
    reads: Mutex<Reads>,
    leases: Mutex<Leases>,
    shares: Mutex<Shares>,
    decisions: Mutex<Decisions>,
}

impl ReplayHost {
    fn elapsed(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    fn instant_of(&self, at: u64) -> tokio::time::Instant {
        self.start + Duration::from_nanos(at.saturating_sub(self.base_ns))
    }

    fn decide(&self, role: &str, stream: Stream, decision: Decision) {
        self.decisions
            .lock()
            .expect("no replay panics")
            .entry((role.to_owned(), stream))
            .or_default()
            .push(decision);
    }
}

impl Host for ReplayHost {
    fn now(&self) -> u64 {
        self.base_ns + self.elapsed()
    }

    fn wall_now(&self) -> u64 {
        self.wall_base_ns + self.elapsed()
    }

    async fn until(&self, at: u64) {
        tokio::time::sleep_until(self.instant_of(at)).await
    }

    async fn call(&self, target: &Target, request: Json, _cancel: CancellationToken) -> CallEnd {
        let role = target.role.to_string();
        self.decide(
            &role,
            Stream::Calls,
            Decision::Call {
                contract: target.contract.clone(),
                member: target.member.to_string(),
                instance: target.producer.instance.clone(),
                request,
            },
        );
        let next = self
            .calls
            .lock()
            .expect("no replay panics")
            .get_mut(&role)
            .and_then(VecDeque::pop_front);
        let Some((ended_at, record)) = next else {
            return CallEnd::Unknown;
        };
        // The engine dropped this call when it ran, and drops it again at the
        // same run time.
        let Some(end) = record.end else {
            return std::future::pending().await;
        };
        tokio::time::sleep_until(self.instant_of(ended_at)).await;
        end
    }

    /// The message of the next recorded read, as old at the replay's own
    /// run time as it was at the run time of the read.
    fn latest(&self, target: &Target) -> Option<(Json, u64)> {
        let mut reads = self.reads.lock().expect("no replay panics");
        let queue = reads.get_mut(&target.topic_key())?;
        let read = match queue.len() {
            0 => None,
            1 => queue.front().cloned(),
            _ => queue.pop_front(),
        }?;
        let (message, at) = read.latest?;
        let age = read.read_ns.saturating_sub(at);
        Some((message, self.now().saturating_sub(age)))
    }

    /// Resolves at the run time the first message of the topic came: that
    /// of the first recorded read of it still to come that holds a message.
    async fn message_arrives(&self, target: &Target) {
        let arrived = self
            .reads
            .lock()
            .expect("no replay panics")
            .get(&target.topic_key())
            .and_then(|reads| reads.iter().find_map(|read| read.latest.as_ref()))
            .map(|(_, at)| *at);
        match arrived {
            Some(at) => tokio::time::sleep_until(self.instant_of(at)).await,
            None => std::future::pending().await,
        }
    }

    /// A read that joined another reader's read in the run goes on as it
    /// did: it takes the answer it took, at the run time it took it; it reads
    /// itself from the run time the read it joined was dropped; and it waits
    /// with no end when the run dropped it before an answer. Every other read
    /// goes on its own.
    async fn joining(&self, name: &Name, nth: u64) -> Joining {
        let shared = self
            .shares
            .lock()
            .expect("no replay panics")
            .remove(&(name.clone(), nth));
        let Some((at, join)) = shared else {
            return Joining::Own;
        };
        match join {
            Join::Waits => std::future::pending().await,
            Join::Took(answer) => {
                tokio::time::sleep_until(self.instant_of(at)).await;
                Joining::Took(answer)
            }
            Join::ReadsItself => {
                tokio::time::sleep_until(self.instant_of(at)).await;
                Joining::Own
            }
        }
    }

    fn joined(&self, _name: &Name, _nth: u64, _join: &Join) {}

    /// The next recorded answer to an open or a close of the lease of
    /// `robot`, at the run time it came.
    async fn set_lease_open(&self, robot: &RobotName, _open: bool) -> Result<(), String> {
        let answer = self
            .leases
            .lock()
            .expect("no replay panics")
            .get_mut(robot.as_str())
            .and_then(VecDeque::pop_front);
        let Some((at, answer)) = answer else {
            return Ok(());
        };
        tokio::time::sleep_until(self.instant_of(at)).await;
        answer
    }

    fn event(&self, event: Event) {
        if of_the_engine(event.kind) {
            self.decide(
                &event.role,
                Stream::Events,
                Decision::Event {
                    kind: event.kind.name().to_owned(),
                    step: event.step,
                    failure_kind: event.failure_kind,
                },
            );
        }
    }
}

/// Replays the journal `text`.
pub fn replay(text: &str) -> Result<Outcome, String> {
    let recording = Recording::read(text)?;
    let planned = &recording.planned;
    let allowed = AllowedContracts::parse(&planned.allowed)?;
    let contracts = planned
        .documents
        .iter()
        .map(|(sha256, document)| {
            ContractShape::parse(document, sha256).map(|shape| (sha256.clone(), Arc::new(shape)))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let snapshot = Snapshot {
        fleet: planned.fleet.clone(),
        contracts,
        unreadable: BTreeMap::new(),
        models: planned.models.clone(),
        ready: true,
    };
    let document = &planned.run_document;
    let choices = RunChoices {
        attendance: document.attendance,
        confirmed: document.confirmed.clone(),
        incidents: &document.incidents,
        poll: workflow_documents::Bound::from_seconds(planned.poll_s)
            .map_err(|e| format!("poll: {e}"))?,
        cancel_timeout: Duration::from_nanos(planned.cancel_timeout_ns),
        watch: Watch::of(document, recording.validation.as_ref()),
    };
    let plan = checker::plan(
        Arc::new(recording.workflow.clone()),
        &snapshot,
        &allowed,
        &planned.binding,
        choices,
    )
    .map_err(|errors| {
        format!(
            "the plan of the journal does not hold: {}",
            errors
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ")
        )
    })?;
    let host = ReplayHost {
        base_ns: planned.run_ns,
        wall_base_ns: planned.wall_ns,
        start: tokio::time::Instant::now(),
        calls: Mutex::new(recording.calls.clone()),
        reads: Mutex::new(recording.reads.clone()),
        leases: Mutex::new(recording.leases.clone()),
        shares: Mutex::new(recording.shares.clone()),
        decisions: Mutex::new(BTreeMap::new()),
    };
    // The commands that came before the engine started wait for it in its
    // channel, as they did in the run; the others come at their run time.
    let (queued, later): (Vec<_>, Vec<_>) = recording
        .commands
        .iter()
        .cloned()
        .partition(|(_, entry)| entry.queued);
    // The replay runs on a runtime of its own whose clock moves only while
    // every task waits, on a thread of its own, since the service that asks
    // for it answers on a thread of the node's runtime.
    let replayed = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_time()
                    .start_paused(true)
                    .build()
                    .map_err(|e| format!("the replay did not start: {e}"))?;
                runtime.block_on(async {
                    let host = ReplayHost {
                        start: tokio::time::Instant::now(),
                        ..host
                    };
                    let (commands_tx, commands_rx) = mpsc::unbounded_channel();
                    for (_, entry) in queued {
                        let _ = commands_tx.send(entry.command.command());
                    }
                    let feeder = {
                        let base = tokio::time::Instant::now();
                        let run_ns = planned.run_ns;
                        tokio::spawn(async move {
                            for (at, entry) in later {
                                tokio::time::sleep_until(
                                    base + Duration::from_nanos(at.saturating_sub(run_ns)),
                                )
                                .await;
                                if commands_tx.send(entry.command.command()).is_err() {
                                    return;
                                }
                            }
                            // The paused clock reaches the horizon once only a
                            // command can move the run; the commands then
                            // close, which ends the run interrupted, as its
                            // director ended it.
                            tokio::time::sleep(REPLAY_HORIZON).await;
                        })
                    };
                    let (phase, _) = watch::channel(Phase::Preparing);
                    let base = ResultBase {
                        result: crate::director::empty_result(),
                    };
                    let result =
                        engine::drive(&plan, &host, base, commands_rx, phase, planned.run_ns).await;
                    feeder.abort();
                    let decisions = host.decisions.into_inner().expect("no replay panics");
                    Ok::<_, String>((decisions, result.ending))
                })
            })
            .join()
            .map_err(|_| "the replay panicked".to_owned())?
    })?;
    Ok(compare(
        &recording.decisions,
        &replayed.0,
        recording.ending,
        replayed.1,
    ))
}

/// How long a replay waits after the last command of its journal, on its
/// paused clock: a year of run time, longer than a run of the director
/// lasts, so a replay that still waits then waits for a command that its
/// journal does not hold.
const REPLAY_HORIZON: Duration = Duration::from_secs(366 * 24 * 3600);

/// Compares the decisions of each lane of a run and of its replay. The
/// journal of an interrupted run ends where its director stopped, so its
/// replay is compared up to there.
fn compare(
    recorded: &Decisions,
    replayed: &Decisions,
    recorded_ending: Option<Ending>,
    replayed_ending: Ending,
) -> Outcome {
    let interrupted = recorded_ending == Some(Ending::Interrupted);
    let mut decisions = 0u32;
    let streams: std::collections::BTreeSet<&(String, Stream)> =
        recorded.keys().chain(replayed.keys()).collect();
    for key in streams {
        let empty = Vec::new();
        let run = recorded.get(key).unwrap_or(&empty);
        let again = replayed.get(key).unwrap_or(&empty);
        let (role, stream) = key;
        let lane = match (role.is_empty(), stream) {
            (true, _) => "the run".to_owned(),
            (false, Stream::Calls) => format!("the calls of {role}"),
            (false, Stream::Events) => format!("the events of {role}"),
        };
        let compared = match interrupted {
            true => run.len(),
            false => run.len().max(again.len()),
        };
        for index in 0..compared {
            decisions += 1;
            match (run.get(index), again.get(index)) {
                (Some(a), Some(b)) if a == b => {}
                (a, b) => {
                    let said =
                        |d: Option<&Decision>| d.map_or("nothing".to_owned(), Decision::to_string);
                    return Outcome {
                        same: false,
                        decisions,
                        difference: json!({
                            "at": lane,
                            "decision": index + 1,
                            "journal": said(a),
                            "replay": said(b),
                        })
                        .to_string(),
                    };
                }
            }
        }
    }
    if let Some(ending) = recorded_ending
        && ending != replayed_ending
    {
        return Outcome {
            same: false,
            decisions,
            difference: json!({ "at": "the ending", "journal": ending, "replay": replayed_ending })
                .to_string(),
        };
    }
    Outcome {
        same: true,
        decisions,
        difference: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tokio::sync::{mpsc, watch};
    use workflow_documents::WorkCell;
    use workflow_documents::run::{Attendance, RunDocument};

    use super::*;
    use crate::engine::{Phase, ResultBase, RunCommand, drive};
    use crate::testing::{
        Answer, ScriptedHost, found, home, name, one_arm, rest, robot, robot_name, snapshot,
        with_beta,
    };

    /// The journal of one run of a workflow that looks and grabs, its grab
    /// missed once and taken again.
    async fn journal_of_a_run() -> String {
        let grabs = vec![
            Answer::after(1, json!({ "success": false, "item_id": "b1" })),
            Answer::after(2, json!({ "success": true, "item_id": "b1" })),
        ];
        journal_of_a_run_with(grabs, Attendance::Attended).await
    }

    /// The longest time that a grab of `journal_of_a_run_with` takes, so two
    /// of them end within the 30 s bound of their step.
    const GRAB_ANSWERS_WITHIN_S: u64 = 14;

    /// The journal of one run of the workflow of `journal_of_a_run`, whose
    /// grabs answer `grabs` in order, with `attendance`.
    async fn journal_of_a_run_with(grabs: Vec<Answer>, attendance: Attendance) -> String {
        journal_of_a_run_until(Answer::now(found()), grabs, attendance, |_| false).await
    }

    /// The journal of the run of `journal_of_a_run_with`, whose `look`
    /// answers `look`, and whose director stops once the run reaches a phase
    /// that `stops` holds; the next director ends its journal interrupted.
    async fn journal_of_a_run_until(
        look: Answer,
        grabs: Vec<Answer>,
        attendance: Attendance,
        stops: impl Fn(&Phase) -> bool,
    ) -> String {
        let scripted = ScriptedHost::new()
            .answer("find", vec![Answer::now(found()), look])
            .answer("grab", grabs)
            .answer("home", vec![Answer::now(json!({ "at_home": true }))]);
        journal_of(scripted, attendance, Commands::default(), stops).await
    }

    /// The commands of a run of a test: those that wait before the engine
    /// starts, and those that come at their second of the run.
    #[derive(Default)]
    struct Commands {
        queued: Vec<RunCommand>,
        later: Vec<(u64, RunCommand)>,
    }

    /// The journal of the run of `journal_of_a_run_until` on `scripted`, with
    /// `commands`, each journaled as the director journals it.
    async fn journal_of(
        scripted: ScriptedHost,
        attendance: Attendance,
        commands: Commands,
        stops: impl Fn(&Phase) -> bool,
    ) -> String {
        journal_of_a_run_of(&picking(), scripted, attendance, commands, stops).await
    }

    /// The workflow of `journal_of_a_run`: a look, then a grab with one retry.
    fn picking() -> Json {
        one_arm(
            "alpha",
            json!({ "id": "pick", "sequence": [
            { "id": "look", "timeout_s": 5, "call": { "member": "arm.find", "args": { "description": "block" }, "as": "seen" } },
            { "id": "grab", "timeout_s": 30, "retry": 1, "call": {
                "member": "arm.grab",
                "args": { "item_id": { "$ref": "seen.item_id" }, "position": { "$ref": "seen.position" }, "speed": 1 },
                "as": "grabbed", "done_when": "grabbed.success",
            } },
        ] }),
            home(),
        )
    }

    /// The journal of a run of `value`, each role on the robot of its name,
    /// on `scripted`, with `commands`, each journaled as the director
    /// journals it, up to its end or a phase that `stops` holds.
    async fn journal_of_a_run_of(
        value: &Json,
        scripted: ScriptedHost,
        attendance: Attendance,
        commands: Commands,
        stops: impl Fn(&Phase) -> bool,
    ) -> String {
        let workflow = Workflow::parse(&value.to_string()).unwrap();
        let document = RunDocument {
            attendance,
            ..RunDocument::of_roles(&workflow, binding_of(&workflow), WorkCell::new()).unwrap()
        };
        journal_of_a_document(value, scripted, document, commands, stops).await
    }

    /// Each role of `workflow` bound to the robot of its name.
    fn binding_of(workflow: &Workflow) -> BTreeMap<Name, RobotName> {
        workflow
            .roles
            .keys()
            .map(|role| (role.clone(), robot_name(role.as_str())))
            .collect()
    }

    /// The journal of a run of `value` by `document`, as
    /// `journal_of_a_run_of` gives it.
    async fn journal_of_a_document(
        value: &Json,
        scripted: ScriptedHost,
        document: RunDocument,
        commands: Commands,
        stops: impl Fn(&Phase) -> bool,
    ) -> String {
        let text = value.to_string();
        let workflow = Workflow::parse(&text).unwrap();
        let binding = binding_of(&workflow);
        let fleet = snapshot(
            binding
                .values()
                .map(|robot| self::robot(robot.as_str()))
                .collect(),
        );
        let dir = tempfile::tempdir().unwrap();
        let journals = Journals::open(dir.path(), 10, u64::MAX).unwrap();
        let run = journals.next_run().unwrap();
        journals
            .append(&run, 0, journal::OPENED, json!({ "workflow": text }))
            .unwrap();
        let mut planned = Planned {
            fleet: fleet.fleet.clone(),
            documents: fleet
                .contracts
                .iter()
                .map(|(pin, shape)| (pin.clone(), shape.document.clone()))
                .collect(),
            models: BTreeMap::new(),
            allowed: crate::testing::allowed().to_string(),
            binding,
            run_document: document,
            poll_s: 0.5,
            cancel_timeout_ns: 10_000_000_000,
            run_ns: 0,
            wall_ns: 0,
        };
        let mut plan = crate::testing::plan(value, planned.run_document.attendance);
        plan.watch = Watch::of(&planned.run_document, None);
        let host = Recorded::new(scripted, &journals, &run);
        planned.run_ns = host.now();
        planned.wall_ns = host.wall_now();
        journals
            .append(&run, planned.run_ns, journal::PLANNED, json!(planned))
            .unwrap();
        let (commands_tx, commands_rx) = mpsc::unbounded_channel();
        for command in commands.queued {
            journal_command(&journals, &run, planned.run_ns, true, &command);
            commands_tx.send(command).unwrap();
        }
        let sender = async {
            let start = tokio::time::Instant::now();
            for (at, command) in commands.later {
                tokio::time::sleep_until(start + Duration::from_secs(at)).await;
                journal_command(&journals, &run, host.now(), false, &command);
                let _ = commands_tx.send(command);
            }
            std::future::pending::<()>().await
        };
        let (phase, mut phases) = watch::channel(Phase::Preparing);
        let driven = drive(
            &plan,
            &host,
            ResultBase {
                result: crate::director::empty_result(),
            },
            commands_rx,
            phase,
            planned.run_ns,
        );
        let ending = tokio::select! {
            result = driven => result.ending,
            _ = phases.wait_for(|phase| stops(phase)) => Ending::Interrupted,
            () = sender => unreachable!("the commands of a test never end"),
        };
        for event in host.inner.events.lock().unwrap().iter() {
            let body = json!({ "role": event.role, "event": event.kind.name(), "step": event.step, "failure_kind": event.failure_kind });
            journals.append(&run, 0, journal::EVENT, body).unwrap();
        }
        journals
            .append(&run, 0, journal::END, json!({ "ending": ending }))
            .unwrap();
        journals.text(&run).unwrap()
    }

    /// An anomaly of a severity that an unattended run lets pass leaves no
    /// incident, and the journal keeps the monitor's answer, anomaly and all.
    #[tokio::test(start_paused = true)]
    async fn the_journal_keeps_the_anomaly_that_an_unattended_run_lets_pass() {
        let tree = json!({ "id": "pick", "sequence": [
            { "id": "look", "timeout_s": 5, "call": { "member": "arm.find", "args": { "description": "block" }, "as": "seen" } },
            { "id": "grab", "timeout_s": 30, "call": {
                "member": "arm.grab",
                "args": { "item_id": { "$ref": "seen.item_id" }, "position": { "$ref": "seen.position" }, "speed": 1 },
                "as": "grabbed",
                "done_when": "grabbed.success",
            } },
        ] });
        let mut plan =
            crate::testing::plan(&one_arm("alpha", tree, json!([])), Attendance::Unattended);
        let arm = plan.calls.values().next().unwrap().target.clone();
        plan.monitor = Some(crate::engine::plan::MonitorPlan {
            target: Target {
                member: name("check"),
                ..arm
            },
            continue_on_severity: [workflow_documents::run::Severity::Low]
                .into_iter()
                .collect(),
        });
        let scripted = ScriptedHost::new()
            .answer("find", vec![Answer::now(found())])
            .answer("grab", vec![Answer::now(json!({ "success": true, "item_id": "b1" }))])
            .answer(
                "check",
                vec![Answer::now(json!({ "anomaly": true, "kind": "slip", "severity": "low", "description": "the block slipped" }))],
            );
        let dir = tempfile::tempdir().unwrap();
        let journals = Journals::open(dir.path(), 10, u64::MAX).unwrap();
        let run = journals.next_run().unwrap();
        let host = Recorded::new(scripted, &journals, &run);
        let (_commands, commands_rx) = mpsc::unbounded_channel();
        let (phase, _phases) = watch::channel(Phase::Preparing);
        let result = drive(
            &plan,
            &host,
            ResultBase {
                result: crate::director::empty_result(),
            },
            commands_rx,
            phase,
            host.now(),
        )
        .await;
        assert_eq!(result.ending, Ending::Completed, "{result:#?}");
        assert!(result.incidents.is_empty(), "{:?}", result.incidents);
        let journal = journals.text(&run).unwrap();
        let check = journal
            .lines()
            .find(|line| line.contains(r#""member":"check""#))
            .expect("the monitor's call is in the journal");
        assert!(
            check.contains("the block slipped") && check.contains(r#""severity":"low""#),
            "{check}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_replay_of_a_journal_makes_the_decisions_of_its_run() {
        let journal = journal_of_a_run().await;
        let outcome = replay(&journal).unwrap();
        assert!(outcome.same, "{outcome:#?}");
        assert!(outcome.decisions > 10, "{outcome:#?}");
    }

    /// Whatever the grabs answer, and whenever within the bound of their
    /// step they answer, the journal of the run replays to the decisions of
    /// the run. The run is unattended, so a step that fails ends it, where an
    /// attended run would wait for a person.
    #[hegel::test]
    fn the_journal_of_any_run_replays_to_its_decisions(tc: hegel::TestCase) {
        use hegel::generators as gs;
        let grabs: Vec<Answer> = (0..2)
            .map(|_| {
                let success = tc.draw(gs::booleans());
                let after = tc.draw(gs::integers::<u64>().max_value(GRAB_ANSWERS_WITHIN_S));
                Answer::after(after, json!({ "success": success, "item_id": "b1" }))
            })
            .collect();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .unwrap();
        let journal = runtime.block_on(journal_of_a_run_with(grabs, Attendance::Unattended));
        let outcome = replay(&journal).unwrap();
        assert!(outcome.same, "{outcome:#?}");
    }

    /// The first message of a topic comes within this many seconds of the
    /// run: past the 20 s bound of the wait that reads it, the wait ends first.
    const FIRST_STATE_WITHIN_S: u64 = 25;

    #[hegel::test]
    fn the_journal_of_two_lanes_that_wait_for_a_topic_replays_to_its_decisions(
        tc: hegel::TestCase,
    ) {
        use hegel::generators as gs;
        let at = tc.draw(gs::integers::<u64>().max_value(FIRST_STATE_WITHIN_S));
        let busy = tc.draw(gs::booleans());
        let scripted = ScriptedHost::new()
            .answer("find", vec![Answer::now(found())])
            .publishes_at(at, "state", json!({ "busy": busy, "holding": false }))
            .moving_at_each_reading();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .unwrap();
        let journal = runtime.block_on(journal_unattended(
            &waiting_for_the_state(true),
            scripted,
            Commands::default(),
        ));
        let reads = journal
            .lines()
            .filter(|line| line.contains(r#""type":"read""#))
            .count();
        assert!(
            reads >= 2,
            "each lane reads the topic on its own: {journal}"
        );
        let outcome = replay(&journal).unwrap();
        assert!(outcome.same, "{outcome:#?}");
    }

    #[test]
    fn the_replay_of_an_interrupted_run_ends_where_its_journal_ends() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .unwrap();
        // The director stops while the lane waits, suspended, for a person.
        let journal = runtime.block_on(journal_of_a_run_until(
            Answer::now(found()),
            vec![Answer::failing("unreachable")],
            Attendance::Attended,
            |phase| matches!(phase, Phase::Suspended(_)),
        ));
        let outcome = replay(&journal).unwrap();
        assert!(outcome.same, "{outcome:#?}");
    }

    #[tokio::test(start_paused = true)]
    async fn a_journal_whose_answer_changed_replays_to_other_decisions() {
        let journal = journal_of_a_run().await;
        let changed = journal.replacen(r#""success":false"#, r#""success":true"#, 1);
        assert_ne!(changed, journal);
        let outcome = replay(&changed).unwrap();
        assert!(!outcome.same, "{outcome:#?}");
        assert!(outcome.difference.contains("alpha"), "{outcome:#?}");
    }

    /// A call that the engine dropped, here a `look` that answers past the
    /// 5 s bound of its step, is in the journal, and the replay drops it at
    /// the same run time: the safe state after it replays as it ran.
    #[tokio::test(start_paused = true)]
    async fn a_call_dropped_past_its_bound_replays_as_it_ran() {
        let journal = journal_of_a_run_until(
            Answer::after(6, found()),
            Vec::new(),
            Attendance::Unattended,
            |_| false,
        )
        .await;
        let dropped = journal
            .lines()
            .filter(|line| line.contains(r#""type":"call""#) && line.contains(r#""end":null"#))
            .count();
        assert_eq!(dropped, 1, "{journal}");
        let outcome = replay(&journal).unwrap();
        assert!(outcome.same, "{outcome:#?}");
    }

    /// An abort that came before the engine started ends the run before its
    /// first check, in the run and in its replay.
    #[tokio::test(start_paused = true)]
    async fn a_command_that_waited_for_the_engine_replays_before_its_first_step() {
        let scripted = ScriptedHost::new()
            .answer("find", vec![Answer::now(found())])
            .answer("home", vec![Answer::now(json!({ "at_home": true }))]);
        let commands = Commands {
            queued: vec![RunCommand::Abort],
            later: Vec::new(),
        };
        let journal = journal_of(scripted, Attendance::Attended, commands, |_| false).await;
        assert!(!journal.contains(r#""type":"call""#), "{journal}");
        let outcome = replay(&journal).unwrap();
        assert!(outcome.same, "{outcome:#?}");
    }

    /// A resume that the daemon refuses as the lease does not close replays
    /// refused: the journal keeps each answer of the daemon to an open and a
    /// close.
    #[tokio::test(start_paused = true)]
    async fn a_resume_refused_by_a_lease_that_did_not_close_replays_refused() {
        let scripted = ScriptedHost::new()
            .answer("find", vec![Answer::now(found())])
            .answer("grab", vec![Answer::failing("unreachable")])
            .answer("home", vec![Answer::now(json!({ "at_home": true }))]);
        *scripted.close_refusal.lock().unwrap() = Some("the daemon is gone".to_owned());
        let (reply, _resumed) = oneshot::channel();
        let commands = Commands {
            queued: Vec::new(),
            later: vec![(60, RunCommand::Resume { reply }), (70, RunCommand::Abort)],
        };
        let journal = journal_of(scripted, Attendance::Attended, commands, |_| false).await;
        assert!(journal.contains(r#""type":"lease""#), "{journal}");
        let outcome = replay(&journal).unwrap();
        assert!(outcome.same, "{outcome:#?}");
    }

    /// A topic read replays as old as it was when the run read it, whatever
    /// the run time of the replay at that read.
    #[tokio::test(start_paused = true)]
    async fn a_topic_read_replays_as_old_as_it_was_when_read() {
        let mut value = one_arm(
            "alpha",
            json!({ "id": "idle", "wait_until": "!state.busy", "timeout_s": 5 }),
            json!([]),
        );
        value["observations"]["state"] =
            json!({ "topic": "arm.state", "role": "alpha", "max_age_s": 1 });
        let plan = crate::testing::plan(&value, Attendance::Unattended);
        let target = plan.observations[&name("state")].target.clone();
        let message = json!({ "busy": false, "holding": false });
        let read = ReadRecord {
            instance: target.producer.instance.clone(),
            contract: target.contract.clone(),
            member: target.member.to_string(),
            latest: Some((message.clone(), 9_500_000_000)),
            read_ns: 10_000_000_000,
        };
        let host = ReplayHost {
            base_ns: 9_900_000_000,
            wall_base_ns: 0,
            start: tokio::time::Instant::now(),
            calls: Mutex::new(BTreeMap::new()),
            reads: Mutex::new(BTreeMap::from([(
                target.topic_key(),
                VecDeque::from([read]),
            )])),
            leases: Mutex::new(BTreeMap::new()),
            shares: Mutex::new(BTreeMap::new()),
            decisions: Mutex::new(BTreeMap::new()),
        };
        assert_eq!(host.latest(&target), Some((message, 9_400_000_000)));
    }

    /// Two lanes, `alpha` and `beta`, that each wait until `until`, with the
    /// observation `again` of `find` beside the observations of `one_arm`.
    fn two_lanes_waiting_until(until: &str) -> Json {
        let wait = json!({ "id": "idle", "wait_until": until, "poll_s": 0.5, "timeout_s": 20 });
        let mut value = with_beta(one_arm("alpha", wait.clone(), json!([])), wait);
        value["observations"]["again"] =
            json!({ "member": "arm.find", "role": "alpha", "args": { "description": "block" } });
        value
    }

    /// A scripted arm whose `find` answers the check of the assumption at
    /// once, and each later read after 1 s.
    fn finding_after_a_second() -> ScriptedHost {
        ScriptedHost::new().answer(
            "find",
            vec![Answer::now(found()), Answer::after(1, found())],
        )
    }

    /// The journal of an unattended run of `value` on `scripted`, with
    /// `commands`.
    async fn journal_unattended(
        value: &Json,
        scripted: ScriptedHost,
        commands: Commands,
    ) -> String {
        journal_of_a_run_of(value, scripted, Attendance::Unattended, commands, |_| false).await
    }

    /// Two lanes that wait on one service observation read it at their own
    /// instants on a clock that moves at each reading, as wall time does: the
    /// run and its replay read it on their own, as the journal says, though
    /// the replay reads both at one instant.
    #[tokio::test(start_paused = true)]
    async fn reads_that_the_run_made_on_their_own_replay_on_their_own() {
        let value = two_lanes_waiting_until("again.found");
        let scripted = finding_after_a_second().moving_at_each_reading();
        let journal = journal_unattended(&value, scripted, Commands::default()).await;
        let calls = journal
            .lines()
            .filter(|line| line.contains(r#""type":"call""#))
            .count();
        assert_eq!(calls, 3, "the check, then one read per lane: {journal}");
        let outcome = replay(&journal).unwrap();
        assert!(outcome.same, "{outcome:#?}");
    }

    /// Two lanes that read one service observation at one instant share its
    /// read: the journal keeps the read that took the other's answer, and the
    /// replay takes it again.
    #[tokio::test(start_paused = true)]
    async fn a_read_that_took_another_readers_answer_replays_taking_it() {
        let value = two_lanes_waiting_until("again.found");
        let journal =
            journal_unattended(&value, finding_after_a_second(), Commands::default()).await;
        assert!(journal.contains(r#""type":"shared""#), "{journal}");
        let outcome = replay(&journal).unwrap();
        assert!(outcome.same, "{outcome:#?}");
    }

    /// A read that joined another reader's read and that the run dropped
    /// before the answer, here at a suspension, replays dropped: it sends no
    /// call of its own.
    #[tokio::test(start_paused = true)]
    async fn a_joined_read_that_the_run_dropped_replays_dropped() {
        let value = two_lanes_waiting_until("again.found");
        let scripted = ScriptedHost::new().answer(
            "find",
            vec![Answer::now(found()), Answer::after(2, found())],
        );
        let commands = Commands {
            queued: Vec::new(),
            later: vec![(1, RunCommand::Suspend), (30, RunCommand::Abort)],
        };
        let journal =
            journal_of_a_run_of(&value, scripted, Attendance::Attended, commands, |_| false).await;
        assert!(journal.contains(r#""join":"waits""#), "{journal}");
        let outcome = replay(&journal).unwrap();
        assert!(outcome.same, "{outcome:#?}");
    }

    /// `alpha` waiting until `state` is not busy, with the topic `state` of a
    /// max_age of 600 s, and beside it `beta` when `two` says.
    fn waiting_for_the_state(two: bool) -> Json {
        let wait = json!({ "id": "idle", "wait_until": "!state.busy", "timeout_s": 20 });
        let mut value = one_arm("alpha", wait.clone(), json!([]));
        if two {
            value = with_beta(value, wait);
        }
        value["observations"]["state"] =
            json!({ "topic": "arm.state", "role": "alpha", "max_age_s": 600 });
        value
    }

    /// A topic that gives its first message 2 s after the run starts: a read
    /// before it waits for it, in the run and in its replay.
    fn first_state_at_two_seconds() -> ScriptedHost {
        ScriptedHost::new()
            .answer("find", vec![Answer::now(found())])
            .publishes_at(2, "state", json!({ "busy": false, "holding": false }))
    }

    /// A read before the first message of a topic replays as it ran: it waits
    /// for that message, which comes at its run time.
    #[tokio::test(start_paused = true)]
    async fn a_read_before_the_first_message_of_a_topic_replays_waiting_for_it() {
        let journal = journal_unattended(
            &waiting_for_the_state(false),
            first_state_at_two_seconds(),
            Commands::default(),
        )
        .await;
        let outcome = replay(&journal).unwrap();
        assert!(outcome.same, "{outcome:#?}");
    }

    /// Two lanes that read a topic before its first message, each at its own
    /// instant on a clock that moves at each reading, both wait for it, in
    /// the run and in its replay, though the read of the other lane comes
    /// first in the journal.
    #[tokio::test(start_paused = true)]
    async fn two_lanes_that_wait_for_the_first_message_of_a_topic_replay_waiting() {
        let journal = journal_unattended(
            &waiting_for_the_state(true),
            first_state_at_two_seconds().moving_at_each_reading(),
            Commands::default(),
        )
        .await;
        let reads = journal
            .lines()
            .filter(|line| line.contains(r#""type":"read""#))
            .count();
        assert_eq!(
            reads, 4,
            "each lane reads on its own, before and at the message: {journal}"
        );
        let outcome = replay(&journal).unwrap();
        assert!(outcome.same, "{outcome:#?}");
    }

    /// An abort that comes while one lease opens and another closes replays as
    /// it ran: the journal keeps the run time of each answer of the daemon.
    #[tokio::test(start_paused = true)]
    async fn the_answers_of_the_daemon_to_leases_replay_at_their_run_time() {
        let value = with_beta(
            one_arm("alpha", rest("rest"), json!([])),
            json!({ "id": "look", "timeout_s": 30, "call": { "member": "arm.find", "args": { "description": "block" } } }),
        );
        let late_failure = Answer {
            after: Duration::from_secs(3),
            ..Answer::failing("unreachable")
        };
        let scripted = ScriptedHost::new()
            .answer("find", vec![Answer::now(found()), late_failure])
            .answer("home", vec![Answer::failing("unreachable")])
            .opens_leases_after(Duration::from_secs(5))
            .closes_leases_after(Duration::from_secs(5));
        let commands = Commands {
            queued: Vec::new(),
            later: vec![(6, RunCommand::Abort)],
        };
        let journal =
            journal_of_a_run_of(&value, scripted, Attendance::Attended, commands, |_| false).await;
        let outcome = replay(&journal).unwrap();
        assert!(outcome.same, "{outcome:#?}");
    }

    /// The journal of a run that reads a watch parameter before its
    /// assumptions replays to its decisions, the event `uncovered` included.
    #[tokio::test(start_paused = true)]
    async fn the_journal_of_a_run_that_watches_replays_to_its_decisions() {
        let value = picking();
        let workflow = Workflow::parse(&value.to_string()).unwrap();
        let mut document =
            RunDocument::of_roles(&workflow, binding_of(&workflow), WorkCell::new()).unwrap();
        document.watch = serde_json::from_value(
            json!({ "loose": { "unit": "items", "observation": "target.count" } }),
        )
        .unwrap();
        let scripted = ScriptedHost::new()
            .answer("find", vec![Answer::now(found())])
            .answer(
                "grab",
                vec![Answer::now(json!({ "success": true, "item_id": "b1" }))],
            )
            .answer("home", vec![Answer::now(json!({ "at_home": true }))]);
        let text =
            journal_of_a_document(&value, scripted, document, Commands::default(), |_| false).await;
        assert!(text.contains("\"uncovered\""), "{text}");
        let outcome = replay(&text).unwrap();
        assert!(outcome.same, "{}", outcome.difference);
    }
}
