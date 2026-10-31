//! The director: its runs, from the request of a caller to the end of each
//! run. A run reads a fresh fleet listing when it arrives, binds its roles,
//! takes the leases of the robots it binds, compares the lock of a bundle,
//! and hands the plan to the engine; at its end it gives each lease back
//! before the event of its end. The runs of an earlier process of the
//! director come back from their journals, and one that did not end ends
//! interrupted.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use peppygen::NodeRunner;
use serde_json::{Value as Json, json};
use tokio::sync::{broadcast, mpsc, watch};
use workflow_documents::bundle::{BUNDLE_PATH, Bundle, BundleDocument, BundleFile, WORKFLOW_PATH};
use workflow_documents::lock::{Lock, LockedNode, UNKNOWN_PEPPY};
use workflow_documents::result::{
    Difference, DifferenceKind, Duration as RunDuration, Ending, RefusalCause, RunRefusal,
    RunResult, RunWorkflow,
};
use workflow_documents::run::{Attendance, RoleBinding, RunDocument, VersionPolicy};
use workflow_documents::work_cell::Uncovered;
use workflow_documents::{
    EventKind, KnownLocation, Located, Name, NameTag, RobotName, Run, Sha256, Validation, WorkCell,
    Workflow, run_lease_name, uncovered_work_cell,
};

use crate::alerts;
use crate::checker::{self, RunChoices};
use crate::clock::{self, ClockSource};
use crate::config::Config;
use crate::engine::host::{Event, Host};
use crate::engine::plan::{Plan, Watch};
use crate::engine::{self, Phase, ResultBase, RunCommand, outside_the_envelope};
use crate::fleet::{self, Fleet, Snapshot};
use crate::journal::{self, Journals, Record, wall_ns};
use crate::leases::{LeaseEventKind, Leases};
use crate::live::{LiveHost, RunClock};
use crate::replay::{Planned, Recorded, journal_command};
use crate::sign_off::{SignOff, SignOffRecord, SignOffs, refusal_of_no_record};
use crate::trial::{TrialEntry, Trials, robots_of_trials};

/// The caller of a run: the instance and the machine that sent its goal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Caller {
    pub instance: String,
    pub core_node: String,
}

/// What a caller asks a run of.
#[derive(Clone, Debug)]
pub struct RunRequest {
    pub workflow: Option<String>,
    pub bundle: Option<Vec<BundleFile>>,
    pub run_document: Option<String>,
    pub roles: Option<String>,
    pub work_cell: Option<String>,
    pub started_by: String,
    pub caller_data: Option<String>,
    /// The trial of a run of a trial, `t<N>`.
    pub trial: Option<String>,
}

/// Where the workflow of a run comes from: its text, or a bundle.
pub enum Source<'r> {
    Workflow(&'r str),
    Bundle(&'r [BundleFile]),
}

/// How a run binds its roles: by a run document, or by roles and a work
/// cell block, each optional.
pub enum Binding<'r> {
    Document(&'r str),
    Roles {
        roles: Option<&'r str>,
        work_cell: Option<&'r str>,
    },
}

impl RunRequest {
    /// The source and the binding of the run. Refuses a request that gives
    /// a run document with roles or with a work cell block, or neither or
    /// both of a workflow and a bundle.
    pub fn shape(&self) -> Result<(Source<'_>, Binding<'_>), String> {
        let source = match (&self.workflow, &self.bundle) {
            (Some(text), None) => Source::Workflow(text),
            (None, Some(files)) => Source::Bundle(files),
            (Some(_), Some(_)) => return Err("give a workflow or a bundle, not both".to_owned()),
            (None, None) => return Err("give a workflow or a bundle".to_owned()),
        };
        let binding = match (&self.run_document, &self.roles, &self.work_cell) {
            (Some(text), None, None) => Binding::Document(text),
            (None, roles, work_cell) => Binding::Roles {
                roles: roles.as_deref(),
                work_cell: work_cell.as_deref(),
            },
            (Some(_), Some(_), _) => {
                return Err("give a run_document or roles, not both".to_owned());
            }
            (Some(_), None, Some(_)) => {
                return Err(
                    "a run_document gives its own work_cell: give work_cell with roles".to_owned(),
                );
            }
        };
        Ok((source, binding))
    }
}

/// A run the director keeps: its workflow once it parsed, who started it,
/// the trial of a run of a trial, its robots once bound, where it stands and
/// how to reach its engine.
pub struct RunEntry {
    pub workflow: Option<Arc<Workflow>>,
    pub started_by: String,
    pub trial: Option<String>,
    pub robots: Vec<(Name, RobotName)>,
    pub phase: watch::Receiver<Phase>,
    pub commands: Option<mpsc::UnboundedSender<RunCommand>>,
}

/// What a member of the director reads of a run: where it stands, and the
/// receiver that follows it, how to reach its engine while it goes, its
/// workflow once it parsed, and its trial.
pub struct RunView {
    pub phase: Phase,
    pub phases: watch::Receiver<Phase>,
    pub commands: Option<mpsc::UnboundedSender<RunCommand>>,
    pub workflow: Option<Arc<Workflow>>,
    pub trial: Option<String>,
}

/// The director.
pub struct Director {
    pub runner: Arc<NodeRunner>,
    pub config: Config,
    pub leases: Arc<Leases>,
    pub snapshots: watch::Receiver<Arc<Snapshot>>,
    pub journals: Journals,
    pub sign_offs: SignOffs,
    pub trial_records: Trials,
    pub trials: Mutex<BTreeMap<String, TrialEntry>>,
    pub runs: Mutex<BTreeMap<String, RunEntry>>,
    /// The robots that a teach of their locations holds. Read under the
    /// lock of `runs`, so a run and a teach never hold one robot.
    pub teaching: Mutex<BTreeSet<RobotName>>,
    pub events: broadcast::Sender<(String, Event)>,
    pub clock: ClockSource,
}

/// What of this director holds a robot: another of its runs, a teach of its
/// locations, or a running trial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Holder {
    Run(String),
    Teach,
    Trial(String),
}

impl Holder {
    /// Why a run or a trial does not bind `robot`, which this holds, and
    /// what to do.
    pub fn why(&self, robot: &str) -> String {
        match self {
            Holder::Run(run) => format!(
                "`{robot}` takes part in run {run} of this director: wait for its end, or abort it"
            ),
            Holder::Teach => format!(
                "`{robot}` is being taught its locations by this director: run again once the teach ends"
            ),
            Holder::Trial(trial) => format!(
                "`{robot}` takes part in trial {trial} of this director: wait for its end, or cancel it"
            ),
        }
    }
}

/// The robots that the runs of `runs` other than `except` bind while they
/// have not ended, with the run of each.
fn robots_of_runs(runs: &BTreeMap<String, RunEntry>, except: &str) -> BTreeMap<RobotName, String> {
    runs.iter()
        .filter(|(id, entry)| {
            id.as_str() != except && !matches!(*entry.phase.borrow(), Phase::Ended(_))
        })
        .flat_map(|(id, entry)| {
            entry
                .robots
                .iter()
                .map(move |(_, robot)| (robot.clone(), id.clone()))
        })
        .collect()
}

/// A robot that a teach of its locations holds, until this is dropped.
pub struct TeachHold {
    director: Arc<Director>,
    robot: RobotName,
}

impl Drop for TeachHold {
    fn drop(&mut self) {
        self.director
            .teaching
            .lock()
            .expect("no teach panics holding the robots it teaches")
            .remove(&self.robot);
    }
}

/// What a run needs from its preparation: its plan, the base of its result,
/// the leases it took, the robots whose leases it uses (those it took, or
/// those of its trial), the inputs of its plan and its fleet.
struct Prepared {
    plan: Plan,
    base: ResultBase,
    taken: Vec<RobotName>,
    leased: Vec<RobotName>,
    planned: Planned,
    snapshot: Snapshot,
}

/// What a run request gives once it reads: its workflow, its bundle, the
/// base of its run result and its run document.
struct ParsedRun {
    workflow: Arc<Workflow>,
    bundle: Option<Bundle>,
    base: RunResult,
    run_document: RunDocument,
}

/// The run result of a run that `message` refuses, with `errors`.
fn refusal_of(
    base: RunResult,
    message: &str,
    errors: Vec<Located>,
    started: u64,
) -> Box<RunResult> {
    let mut result = refused_result(base, message, started);
    if let Some(refusal) = result.refusal.as_mut() {
        refusal.errors = errors;
    }
    Box::new(result)
}

fn located(refusals: workflow_documents::Refusals) -> Vec<Located> {
    refusals.iter().map(|r| r.located()).collect()
}

/// How many events wait for a slow follower of the runs.
pub const EVENTS_KEPT: usize = 4096;

/// What a run refused or ended by a halt of the stack tells its caller to do.
pub const HALTED: &str = "the stack is halted; run `peppy fleet clear-halt`, then run again";

impl Director {
    fn journal(&self, run: &str, run_ns: u64, kind: &str, body: Json) {
        if let Err(error) = self.journals.append(run, run_ns, kind, body) {
            tracing::error!("the journal of {run} lost a record: {error}");
        }
    }

    pub fn view_of(&self, run: &str) -> Option<RunView> {
        let runs = self.runs.lock().expect("no run panics holding the runs");
        let entry = runs.get(run)?;
        Some(RunView {
            phase: entry.phase.borrow().clone(),
            phases: entry.phase.clone(),
            commands: entry.commands.clone(),
            workflow: entry.workflow.clone(),
            trial: entry.trial.clone(),
        })
    }

    fn update(&self, run: &str, change: impl FnOnce(&mut RunEntry)) {
        if let Some(entry) = self
            .runs
            .lock()
            .expect("no run panics holding the runs")
            .get_mut(run)
        {
            change(entry);
        }
    }

    /// The instance and the machine of the director, as the daemon names a
    /// client.
    pub fn me(&self) -> Caller {
        let processor = self.runner.processor();
        Caller {
            instance: processor.bound_instance_id().to_owned(),
            core_node: processor.bound_core_node().to_owned(),
        }
    }

    /// Holds `robot` for a teach of its locations, so no run of this
    /// director binds it while the teach runs. Refused while a run of this
    /// director binds it, or another teach holds it.
    pub fn hold_for_teach(self: &Arc<Self>, robot: &RobotName) -> Result<TeachHold, String> {
        let runs = self.runs.lock().expect("no run panics holding the runs");
        if let Some(run) = robots_of_runs(&runs, "").get(robot) {
            return Err(format!(
                "`{robot}` takes part in run {run} of this director: teach it once the run ends"
            ));
        }
        let mut teaching = self
            .teaching
            .lock()
            .expect("no teach panics holding the robots it teaches");
        if !teaching.insert(robot.clone()) {
            return Err(format!(
                "a teach of the locations of `{robot}` runs: teach it once that teach ends"
            ));
        }
        Ok(TeachHold {
            director: Arc::clone(self),
            robot: robot.clone(),
        })
    }

    /// Gives an event of `run`: to its goals, `run_events` and its journal.
    pub fn event(&self, run: &str, event: Event) {
        self.journal(run, 0, journal::EVENT, json!(event));
        let _ = self.events.send((run.to_owned(), event));
    }

    fn run_event(&self, run: &str, kind: EventKind, details: Json) {
        let failure_kind = match (
            kind,
            details
                .get("ending")
                .and_then(|e| serde_json::from_value::<Ending>(e.clone()).ok()),
        ) {
            (EventKind::RunEnds, Some(ending)) => ending.failure_kind().unwrap_or_default(),
            _ => "",
        };
        let event = Event {
            role: String::new(),
            kind,
            step: String::new(),
            failure_kind: failure_kind.to_owned(),
            message: String::new(),
            details,
        };
        self.event(run, event);
    }

    /// Brings back the runs of the journals: each ended run as it ended, and
    /// each run that did not end ends interrupted, with its robots in an
    /// unknown state.
    pub async fn restore(&self) -> std::io::Result<()> {
        for run in self.journals.runs() {
            let records = self.journals.read(&run)?;
            let opened = records
                .iter()
                .find(|r| r.kind == journal::OPENED)
                .map(|r| r.body.clone())
                .unwrap_or(Json::Null);
            let workflow = workflow_of(&opened).ok().map(Arc::new);
            let robots = records
                .iter()
                .find(|r| r.kind == journal::PLANNED)
                .and_then(|r| serde_json::from_value::<Planned>(r.body.clone()).ok())
                .map(|planned| planned.binding.into_iter().collect())
                .unwrap_or_default();
            let ended = records
                .iter()
                .find(|r| r.kind == journal::END)
                .and_then(|r| serde_json::from_value::<RunResult>(r.body["result"].clone()).ok());
            let result = match ended {
                Some(result) => result,
                None => {
                    let result = interrupted_result(workflow.as_deref(), &records);
                    self.record_end(&run, &result, &watch::Sender::new(Phase::Preparing));
                    result
                }
            };
            let (_, phase) = watch::channel(Phase::Ended(Box::new(result)));
            let started_by = opened["started_by"].as_str().unwrap_or_default().to_owned();
            let trial = opened["trial"].as_str().map(str::to_owned);
            let entry = RunEntry {
                workflow,
                started_by,
                trial,
                robots,
                phase,
                commands: None,
            };
            self.runs
                .lock()
                .expect("no run panics holding the runs")
                .insert(run, entry);
        }
        // Pruned once every journal read, so no journal goes before its
        // run is restored.
        self.prune_journals();
        Ok(())
    }

    /// Starts a run and gives its id and where it stands; the run goes on
    /// in its own task.
    pub fn start(
        self: &Arc<Self>,
        request: RunRequest,
        caller: Caller,
    ) -> Result<(String, watch::Receiver<Phase>), String> {
        let run = self
            .journals
            .next_run()
            .map_err(|error| format!("the journal of the run did not open: {error}"))?;
        let (phase_tx, phase_rx) = watch::channel(Phase::Preparing);
        let (commands_tx, commands_rx) = mpsc::unbounded_channel();
        let entry = RunEntry {
            workflow: None,
            started_by: request.started_by.clone(),
            trial: request.trial.clone(),
            robots: Vec::new(),
            phase: phase_rx.clone(),
            commands: Some(commands_tx.clone()),
        };
        self.runs
            .lock()
            .expect("no run panics holding the runs")
            .insert(run.clone(), entry);
        let bundle = request.bundle.as_deref().map(journaled_bundle);
        let opened = json!({
            "workflow": request.workflow, "bundle": bundle, "run_document": request.run_document,
            "roles": request.roles, "work_cell": request.work_cell, "started_by": request.started_by,
            "caller_data": request.caller_data, "trial": request.trial,
            "caller": { "instance": caller.instance, "core_node": caller.core_node },
        });
        self.journal(&run, 0, journal::OPENED, opened);
        let director = Arc::clone(self);
        let id = run.clone();
        tokio::spawn(async move {
            director
                .run(&id, request, caller, &phase_tx, commands_tx, commands_rx)
                .await;
        });
        Ok((run, phase_rx))
    }

    async fn run(
        self: &Arc<Self>,
        run: &str,
        request: RunRequest,
        caller: Caller,
        phase: &watch::Sender<Phase>,
        commands_tx: mpsc::UnboundedSender<RunCommand>,
        mut commands: mpsc::UnboundedReceiver<RunCommand>,
    ) -> RunResult {
        let started = wall_ns();
        let prepared = match self.prepare(run, &request, &caller).await {
            Ok(prepared) => prepared,
            Err(refused) => {
                self.end(run, &refused, phase).await;
                return *refused;
            }
        };
        let Prepared {
            plan,
            base,
            taken,
            leased,
            mut planned,
            snapshot,
        } = prepared;
        let refuse = |message: String| refused_result(base.result.clone(), &message, started);
        if let Err(message) = self.skills_know_the_locations(run, &plan, &snapshot).await {
            return self.end_released(run, refuse(message), &taken, phase).await;
        }
        let run_clock = match &self.clock {
            ClockSource::Simulation { ticks, domain }
                if reads_the_clock(domain, &plan, &snapshot.fleet) =>
            {
                if let Err(message) = clock::first_tick(ticks).await {
                    return self.end_released(run, refuse(message), &taken, phase).await;
                }
                RunClock::Simulation(ticks.clone())
            }
            _ => RunClock::Wall,
        };
        let note = {
            let director = Arc::clone(self);
            let run = run.to_owned();
            Box::new(move |body: Json| director.journal(&run, 0, journal::ALERT, body))
                as alerts::Note
        };
        let alerts = match alerts::watch(&self.runner, &snapshot, &plan, commands_tx.clone(), note)
            .await
        {
            Ok(alerts) => alerts,
            Err(message) => return self.end_released(run, refuse(message), &taken, phase).await,
        };
        let stops = match &run_clock {
            RunClock::Simulation(ticks) => {
                let roles = plan.roles.keys().cloned().collect();
                Some(tokio::spawn(clock::watch_for_stops(
                    ticks.clone(),
                    phase.subscribe(),
                    roles,
                    commands_tx.clone(),
                )))
            }
            RunClock::Wall => None,
        };
        let (events_tx, mut events_rx) = mpsc::unbounded_channel::<Event>();
        let live = LiveHost::new(
            Arc::clone(&self.runner),
            &plan,
            &snapshot.contracts,
            Arc::clone(&self.leases),
            run_clock,
            events_tx,
        )
        .await;
        let live = match live {
            Ok(live) => live,
            Err(message) => return self.end_released(run, refuse(message), &taken, phase).await,
        };
        let host = Recorded::new(live, &self.journals, run);
        planned.run_ns = host.now();
        planned.wall_ns = host.wall_now();
        self.journal(run, planned.run_ns, journal::PLANNED, json!(planned));
        let stopper = self.follow_leases(run, &leased, commands_tx).await;
        let (engine_tx, engine_rx) = mpsc::unbounded_channel();
        let pass_on = |command: RunCommand, queued: bool| {
            let at = if queued { planned.run_ns } else { host.now() };
            journal_command(&self.journals, run, at, queued, &command);
            engine_tx.send(command).is_ok()
        };
        // The commands that came during the preparation reach the engine
        // before its first step.
        while let Ok(command) = commands.try_recv() {
            pass_on(command, true);
        }
        let forward = async {
            while let Some(command) = commands.recv().await {
                if !pass_on(command, false) {
                    break;
                }
            }
            std::future::pending::<()>().await
        };
        let events = async {
            while let Some(event) = events_rx.recv().await {
                self.event(run, event);
            }
            std::future::pending::<()>().await
        };
        let result = tokio::select! {
            result = engine::drive(&plan, &host, base, engine_rx, phase.clone(), planned.run_ns) => result,
            _ = forward => unreachable!("the forward of commands never ends"),
            _ = events => unreachable!("the forward of events never ends"),
        };
        stopper.abort();
        stops.iter().for_each(tokio::task::JoinHandle::abort);
        drop(alerts);
        drop(host);
        while let Ok(event) = events_rx.try_recv() {
            self.event(run, event);
        }
        self.end_released(run, result, &taken, phase).await
    }

    /// Ends a run that holds leases: each lease goes back first.
    async fn end_released(
        &self,
        run: &str,
        result: RunResult,
        taken: &[RobotName],
        phase: &watch::Sender<Phase>,
    ) -> RunResult {
        for robot in taken {
            self.leases.release(robot.as_str()).await;
        }
        self.end(run, &result, phase).await;
        result
    }

    /// Follows the lease events and the halt for the robots of `run`, whose
    /// leases are `leased`: the end of one of its leases, or a halt, stops
    /// the run; the end of a lease in a halt stops it halted. A fresh listing
    /// first gives each lease that ended between its take and this follow,
    /// and its stop reaches the commands before the engine starts.
    async fn follow_leases(
        &self,
        run: &str,
        leased: &[RobotName],
        commands: mpsc::UnboundedSender<RunCommand>,
    ) -> tokio::task::JoinHandle<()> {
        let mut follower = LeaseFollower {
            events: self.leases.events(),
            snapshots: self.snapshots.clone(),
            runner: Arc::clone(&self.runner),
            robots: leased.iter().map(|r| r.to_string()).collect(),
        };
        match fleet::read_fleet(&self.runner).await {
            Ok(fleet) => {
                if let Some(stop) = lost_leases(&fleet, &follower.robots, &self.me()) {
                    let _ = commands.send(stop);
                }
            }
            Err(error) => {
                tracing::warn!("the fleet listing at the start of {run} did not come: {error}")
            }
        }
        tokio::spawn(async move {
            while let Some(LeaseStop { halted, reason }) = follower.next_stop().await {
                let _ = commands.send(RunCommand::Stop { halted, reason });
            }
        })
    }

    /// The binding of the roles of `run` (`bind`), kept as its robots in the registry under the same lock that reads
    /// the robots of the other runs and of the teaches, so two runs, or a run and a teach, never hold one robot.
    fn bind_and_reserve(
        &self,
        run: &str,
        workflow: &Workflow,
        document: &RunDocument,
        snapshot: &Snapshot,
        parties: &Parties<'_>,
        records: &[SignOffRecord],
    ) -> Result<BTreeMap<Name, RobotName>, Vec<Located>> {
        let mut runs = self.runs.lock().expect("no run panics holding the runs");
        let held = self.holders(&runs, run, parties.trial);
        let binding = bind(workflow, document, snapshot, parties, &held, records)?;
        if let Some(entry) = runs.get_mut(run) {
            entry.robots = binding
                .iter()
                .map(|(role, robot)| (role.clone(), robot.clone()))
                .collect();
        }
        Ok(binding)
    }

    /// The robots that the director holds, with what holds each: the runs
    /// of `runs` other than `run` while they have not ended, the teaches,
    /// and the trials other than `trial` that hold their robots
    /// ([`robots_of_trials`]). The caller holds the lock of the runs, then
    /// this takes those of the teaches and of the trials.
    pub fn holders(
        &self,
        runs: &BTreeMap<String, RunEntry>,
        run: &str,
        trial: Option<&str>,
    ) -> BTreeMap<RobotName, Holder> {
        let teaching = self
            .teaching
            .lock()
            .expect("no teach panics holding the robots it teaches")
            .clone();
        let of_trials: Vec<(RobotName, Holder)> = robots_of_trials(
            &self
                .trials
                .lock()
                .expect("no trial panics holding the trials"),
            trial,
        )
        .into_iter()
        .map(|(robot, trial)| (robot, Holder::Trial(trial)))
        .collect();
        robots_of_runs(runs, run)
            .into_iter()
            .map(|(robot, run)| (robot, Holder::Run(run)))
            .chain(teaching.into_iter().map(|robot| (robot, Holder::Teach)))
            .chain(of_trials)
            .collect()
    }

    /// Lists the locations each skill of the run knows, into the journal,
    /// and refuses the run when a skill lacks a location its role gives it.
    async fn skills_know_the_locations(
        &self,
        run: &str,
        plan: &Plan,
        snapshot: &Snapshot,
    ) -> Result<(), String> {
        for skill in &plan.locations {
            let robot = plan.roles[&skill.role].robot.as_str();
            let robot = snapshot
                .fleet
                .robots
                .get(robot)
                .ok_or_else(|| format!("the fleet lists no robot `{robot}`"))?;
            let instance = robot
                .instances
                .iter()
                .find(|instance| instance.instance == skill.producer.instance)
                .ok_or_else(|| {
                    format!(
                        "`{}` has no instance `{}`",
                        robot.name, skill.producer.instance
                    )
                })?;
            let known =
                fleet::known_locations(&self.runner, snapshot, robot, instance, "list", &json!({}))
                    .await
                    .map_err(|error| {
                        format!("the locations of `{}`: {error}", instance.instance)
                    })?;
            self.journal(
                run,
                0,
                journal::LOCATIONS,
                json!({ "role": skill.role, "instance": instance.instance, "locations": known }),
            );
            let names: BTreeSet<&str> = known
                .iter()
                .map(|location| location.name.as_str())
                .collect();
            if let Some(missing) = skill
                .names
                .iter()
                .find(|name| !names.contains(name.as_str()))
            {
                let lists = self.lists_of_robot(snapshot, robot, instance, &known).await;
                let hint = TeachHint {
                    skill: &instance.instance,
                    role: &skill.role,
                    director: &self.me().instance,
                    robot: &robot.name,
                };
                return Err(hint.refusal(&lists, missing));
            }
        }
        Ok(())
    }

    /// What each locations:v1 skill of `robot` knows, by its instance:
    /// `known` for `asked`, which listed it already, a fresh list for each
    /// other, and `None` for a skill that gave no list.
    async fn lists_of_robot(
        &self,
        snapshot: &Snapshot,
        robot: &fleet::Robot,
        asked: &fleet::Instance,
        known: &[KnownLocation],
    ) -> Vec<(String, Option<Vec<KnownLocation>>)> {
        let (name, tag) = fleet::LOCATIONS;
        let mut lists = Vec::new();
        for instance in robot.instances_of(name, tag) {
            let list = match instance.instance == asked.instance {
                true => Some(known.to_vec()),
                false => fleet::known_locations(
                    &self.runner,
                    snapshot,
                    robot,
                    instance,
                    "list",
                    &json!({}),
                )
                .await
                .ok(),
            };
            lists.push((instance.instance.clone(), list));
        }
        lists
    }

    /// The end of a run: its record, its phase `Ended`, its event, then the
    /// pruning of the journals.
    async fn end(&self, run: &str, result: &RunResult, phase: &watch::Sender<Phase>) {
        self.record_end(run, result, phase);
        self.prune_journals();
    }

    /// The record of the end of a run, its phase `Ended` on `phase` with no
    /// more commands, then the event of its end. A follower that ends its
    /// goal at `run_ends` finds the run ended in the registry.
    fn record_end(&self, run: &str, result: &RunResult, phase: &watch::Sender<Phase>) {
        self.journal(
            run,
            0,
            journal::END,
            json!({ "ending": result.ending, "result": result }),
        );
        self.update(run, |entry| entry.commands = None);
        phase.send_replace(Phase::Ended(Box::new(result.clone())));
        self.run_event(run, EventKind::RunEnds, json!(result));
    }

    /// Removes the journals of ended runs past the bounds, and their runs
    /// from the registry.
    fn prune_journals(&self) {
        match self.journals.prune() {
            Ok(removed) => {
                let mut runs = self.runs.lock().expect("no run panics holding the runs");
                removed.iter().for_each(|run| {
                    runs.remove(run);
                });
            }
            Err(error) => tracing::warn!("the journals were not pruned: {error}"),
        }
    }

    /// The snapshot of the fresh fleet listing `fleet`, with the documents
    /// and the models that the director knows already.
    pub async fn snapshot_of_listing(&self, fleet: Fleet) -> Snapshot {
        let (known, known_models) = {
            let latest = self.snapshots.borrow();
            (latest.contracts.clone(), latest.models.clone())
        };
        let mut snapshot = fleet::snapshot_of(&self.runner, fleet, &known).await;
        snapshot.models = fleet::models_of(&self.runner, &snapshot, &known_models).await;
        snapshot
    }

    /// The workflow, the bundle and the run document of `request`, and
    /// the base of its run result.
    fn parse_run(
        &self,
        run: &str,
        request: &RunRequest,
        started: u64,
    ) -> Result<Box<ParsedRun>, Box<RunResult>> {
        let refuse = |base: RunResult, message: String, errors: Vec<Located>| {
            refusal_of(base, &message, errors, started)
        };
        let (source, binding) = request
            .shape()
            .map_err(|message| refuse(empty_result(), message, Vec::new()))?;
        let (workflow, bundle) = match source {
            Source::Workflow(text) => match Workflow::parse(text) {
                Ok(workflow) => (workflow, None),
                Err(refusals) => {
                    return Err(refuse(
                        empty_result(),
                        "the workflow does not parse".to_owned(),
                        located(refusals),
                    ));
                }
            },
            Source::Bundle(files) => match Bundle::open(files.to_vec()) {
                Ok(bundle) => (bundle.workflow.clone(), Some(bundle)),
                Err(refusals) => {
                    return Err(refuse(
                        empty_result(),
                        "the bundle does not open".to_owned(),
                        located(refusals),
                    ));
                }
            },
        };
        let workflow = Arc::new(workflow);
        self.update(run, |entry| entry.workflow = Some(Arc::clone(&workflow)));
        let mut base = empty_result();
        base.workflow = Some(RunWorkflow::of(&workflow));
        base.caller_data = request
            .caller_data
            .as_deref()
            .and_then(|text| parses(text).ok());
        let document = match binding {
            Binding::Document(text) => RunDocument::parse(text).map_err(|refusals| {
                refuse(
                    base.clone(),
                    "the run document does not parse".to_owned(),
                    located(refusals),
                )
            })?,
            Binding::Roles { roles, work_cell } => {
                let named = match roles.map(named_roles).transpose() {
                    Ok(named) => named.unwrap_or_default(),
                    Err(reason) => {
                        return Err(refuse(base, format!("roles: {reason}"), Vec::new()));
                    }
                };
                let work_cell = match work_cell.map(work_cell_of).transpose() {
                    Ok(work_cell) => work_cell.unwrap_or_default(),
                    Err(reason) => {
                        return Err(refuse(base, format!("work_cell: {reason}"), Vec::new()));
                    }
                };
                RunDocument::of_roles(&workflow, named, work_cell).map_err(|refusals| {
                    refuse(
                        base.clone(),
                        "the roles do not fit the workflow".to_owned(),
                        located(refusals),
                    )
                })?
            }
        };
        let run_document = Run::new(&workflow, document)
            .map_err(|refusals| {
                refuse(
                    base.clone(),
                    "the run does not fit the workflow".to_owned(),
                    located(refusals),
                )
            })?
            .document;
        base.work_cell = run_document.work_cell.clone();
        let misfits = bundle_misfits(validation_of(bundle.as_ref()), &run_document);
        if !misfits.is_empty() {
            return Err(refuse(
                base,
                "the run does not fit its bundle".to_owned(),
                misfits,
            ));
        }
        Ok(Box::new(ParsedRun {
            workflow,
            bundle,
            base,
            run_document,
        }))
    }

    /// The checks before the initialization, in their order, up to the
    /// plan that the engine runs: the documents, a fresh fleet listing, the
    /// halt, the binding, the rules, the lock, then the takes.
    async fn prepare(
        &self,
        run: &str,
        request: &RunRequest,
        caller: &Caller,
    ) -> Result<Prepared, Box<RunResult>> {
        let started = wall_ns();
        let refuse = |base: RunResult, message: String, errors: Vec<Located>| {
            refusal_of(base, &message, errors, started)
        };
        // Every accepted run starts: its run_starts comes before any check
        // of the request refuses it, with its run document once that reads.
        let parsed = self.parse_run(run, request, started);
        let details = parsed
            .as_ref()
            .map_or(Json::Null, |parsed| json!(parsed.run_document));
        self.run_event(run, EventKind::RunStarts, details);
        let ParsedRun {
            workflow,
            bundle,
            mut base,
            run_document,
        } = *parsed?;
        if self.config.allowed.is_empty() {
            return Err(refuse(
                base,
                "the director allows no contract: its launcher gives allowed_contracts".to_owned(),
                Vec::new(),
            ));
        }
        let fleet = fleet::read_fleet(&self.runner).await.map_err(|error| {
            refuse(
                base.clone(),
                format!("the fleet listing did not come: {error}"),
                Vec::new(),
            )
        })?;
        if fleet.halted {
            return Err(refuse(base, HALTED.to_owned(), Vec::new()));
        }
        let snapshot = self.snapshot_of_listing(fleet).await;
        base.core_node = snapshot.fleet.core_node.clone();
        let me = self.me();
        let digest = bundle.as_ref().map(digest_of);
        let records = digest
            .as_ref()
            .map(|digest| self.sign_offs.of_bundle(digest))
            .unwrap_or_default();
        let binding = self
            .bind_and_reserve(
                run,
                &workflow,
                &run_document,
                &snapshot,
                &Parties {
                    caller,
                    me: &me,
                    trial: request.trial.as_deref(),
                },
                &records,
            )
            .map_err(|errors| refuse(base.clone(), "the roles do not bind".to_owned(), errors))?;
        let validation = validation_of(bundle.as_ref());
        let choices = RunChoices {
            attendance: run_document.attendance,
            confirmed: run_document.confirmed.clone(),
            incidents: &run_document.incidents,
            poll: self.config.poll,
            cancel_timeout: self.config.cancel_timeout,
            watch: Watch::of(&run_document, validation),
            of_a_trial: request.trial.is_some(),
        };
        let plan = checker::plan(
            Arc::clone(&workflow),
            &snapshot,
            &self.config.allowed,
            &binding,
            choices,
        )
        .map_err(|errors| {
            refuse(
                base.clone(),
                "the workflow does not hold for these robots".to_owned(),
                errors,
            )
        })?;
        let nodes = run_nodes(&snapshot, &plan, &me);
        base.versions = versions(&snapshot, &plan, &nodes)
            .map_err(|message| refuse(base.clone(), message, Vec::new()))?;
        let gated = plan.is_unattended_on_a_physical_robot();
        if gated && run_document.versions == VersionPolicy::Warn {
            return Err(refuse(
                base,
                WARN_ON_A_PHYSICAL_ROBOT.to_owned(),
                Vec::new(),
            ));
        }
        if let Some(bundle) = &bundle {
            let (differences, unlocked) =
                differences(&bundle.document.lock, &base.versions, &nodes);
            base.unlocked_nodes = unlocked;
            if !differences.is_empty() {
                self.run_event(run, EventKind::Differences, json!(differences));
            }
            match under_the_policy(run_document.versions, differences) {
                Ok(differences) => base.differences = differences,
                Err((message, differences)) => {
                    let mut refused = refuse(base, message, Vec::new());
                    refused.differences = differences;
                    return Err(refused);
                }
            }
        }
        base.uncovered = uncovered_work_cell(validation, &run_document.work_cell);
        if gated && let Some(message) = work_cell_unfit(bundle.as_ref(), &base.uncovered) {
            if !base.uncovered.is_empty() {
                self.run_event(run, EventKind::Uncovered, json!(base.uncovered));
            }
            return Err(refuse(base, message, Vec::new()));
        }
        // A run of a trial needs no sign-off record: the trial makes it.
        if let (true, None, Some(digest)) = (gated, &request.trial, digest) {
            let sign_off = SignOff::of_run(
                digest,
                &snapshot,
                &binding,
                &base.versions,
                &run_document.incidents,
            );
            if let Some(message) = refusal_of_no_record(&records, &sign_off) {
                return Err(refuse(base, message, Vec::new()));
            }
        }
        let leased: Vec<RobotName> = binding.values().cloned().collect();
        let mut taken: Vec<RobotName> = Vec::new();
        // A run of a trial uses the leases of its trial, which hold the
        // robots from one run to the next.
        let to_take = match request.trial {
            Some(_) => Vec::new(),
            None => leased.clone(),
        };
        for robot in &to_take {
            if let Err(refused) = self.leases.take(robot.as_str(), &run_lease_name(run)).await {
                for robot in &taken {
                    self.leases.release(robot.as_str()).await;
                }
                let message = match refused.halted {
                    true => HALTED.to_owned(),
                    false => format!(
                        "the lease of `{robot}` was refused ({}): {}",
                        refused.kind, refused.message
                    ),
                };
                return Err(refuse(base, message, Vec::new()));
            }
            taken.push(robot.clone());
        }
        let planned = Planned {
            fleet: snapshot.fleet.clone(),
            documents: snapshot
                .contracts
                .iter()
                .map(|(sha256, shape)| (sha256.clone(), shape.document.clone()))
                .collect(),
            models: snapshot.models.clone(),
            allowed: self.config.allowed.to_string(),
            binding,
            run_document,
            poll_s: self.config.poll.seconds(),
            cancel_timeout_ns: self.config.cancel_timeout.as_nanos() as u64,
            run_ns: 0,
            wall_ns: 0,
        };
        Ok(Prepared {
            plan,
            base: ResultBase { result: base },
            taken,
            leased,
            planned,
            snapshot,
        })
    }
}

/// The refusal of a run whose skill `skill`, of the robot `robot` of `role`,
/// knows no location the run gives it, with how to teach it through the
/// director `director`.
struct TeachHint<'h> {
    skill: &'h str,
    role: &'h Name,
    director: &'h str,
    robot: &'h str,
}

impl TeachHint<'_> {
    /// The refusal for `missing`. A teach gives every locations:v1 skill of
    /// the robot the one list it names, in place of what each knows, so the
    /// command teaches every location that a skill of `lists` keeps, each
    /// once, and `missing`; when the locations of a skill are unknown, as it
    /// gave no list, the refusal names that skill and gives no command.
    fn refusal(&self, lists: &[(String, Option<Vec<KnownLocation>>)], missing: &Name) -> String {
        let TeachHint {
            skill,
            role,
            director,
            robot,
        } = self;
        let refused =
            format!("the skill `{skill}` of the robot of `{role}` knows no location `{missing}`");
        let unknown: Vec<&str> = lists
            .iter()
            .filter(|(_, list)| list.is_none())
            .map(|(instance, _)| instance.as_str())
            .collect();
        if !unknown.is_empty() {
            return format!(
                "{refused}. A teach gives every locations:v1 skill of the robot one list in place of what each knows, and the locations of `{}` are unknown, as it gave no list: list them with `peppy node call {director} locations --set robot={robot}` once it answers, then teach them with `{missing}`; or apply a world that names it",
                unknown.join("`, `")
            );
        }
        let mut kept: Vec<&KnownLocation> = Vec::new();
        for location in lists.iter().filter_map(|(_, list)| list.as_ref()).flatten() {
            if !kept.iter().any(|kept| kept.name == location.name) {
                kept.push(location);
            }
        }
        let entry = format!(
            "{{ name: \"{missing}\", kind: \"<pose, landmark, surface or region>\", position: [<x>, <y>, <z>], orientation: [<x>, <y>, <z>, <w>], size: [<x>, <y>, <z>] }}"
        );
        let entries: Vec<String> = kept
            .iter()
            .map(|location| json!(location).to_string())
            .chain([entry])
            .collect();
        format!(
            "{refused}. A teach gives every locations:v1 skill of the robot the list it names, in place of what each knows, so teach the locations its skills keep and `{missing}`, with the pose of `{missing}` in its entry: `peppy node call {director} locations --set robot={robot} --set 'teach=[{}]'`, where an orientation is a unit quaternion [x, y, z, w] and a pose has the size [0, 0, 0]; or apply a world that names it",
            entries.join(", ")
        )
    }
}

/// The end of a lease of a run or a trial, or a halt.
pub struct LeaseStop {
    pub halted: bool,
    pub reason: String,
}

/// Follows the lease events and the halt for the robots `robots`, whose
/// leases the director holds.
pub struct LeaseFollower {
    pub events: broadcast::Receiver<crate::leases::LeaseEvent>,
    pub snapshots: watch::Receiver<Arc<Snapshot>>,
    pub runner: Arc<NodeRunner>,
    pub robots: BTreeSet<String>,
}

impl LeaseFollower {
    /// The next end of a lease of the robots, or halt: the end of a lease in
    /// a halt is the halt. None once the director stops.
    pub async fn next_stop(&mut self) -> Option<LeaseStop> {
        loop {
            tokio::select! {
                event = self.events.recv() => match event {
                    Ok(event) if event.kind == LeaseEventKind::LeaseEnd && self.robots.contains(&event.robot) => {
                        // The daemon ends every lease when it halts the stack, before the listing of the halt
                        // reaches the follower of the fleet: a fresh listing tells a halt from a lost lease.
                        let halted = fleet::read_fleet(&self.runner).await.is_ok_and(|fleet| fleet.halted);
                        let reason = match halted {
                            true => HALTED.to_owned(),
                            false => format!("the lease of {} ended: {}", event.robot, event.cause),
                        };
                        return Some(LeaseStop { halted, reason });
                    }
                    Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => return None,
                },
                changed = self.snapshots.changed() => {
                    changed.ok()?;
                    if self.snapshots.borrow().fleet.halted {
                        return Some(LeaseStop { halted: true, reason: HALTED.to_owned() });
                    }
                }
            }
        }
    }
}

/// The stop of a run whose robots `robots` the fleet listing gives halted,
/// or not leased by `me`, the director.
fn lost_leases(fleet: &Fleet, robots: &BTreeSet<String>, me: &Caller) -> Option<RunCommand> {
    if fleet.halted {
        return Some(RunCommand::Stop {
            halted: true,
            reason: HALTED.to_owned(),
        });
    }
    let lost = robots.iter().find(|robot| {
        !fleet.lease_of(robot).is_some_and(|lease| {
            lease.holder == me.instance && lease.holder_core_node == me.core_node
        })
    })?;
    Some(RunCommand::Stop {
        halted: false,
        reason: format!("the lease of {lost} ended before the run started"),
    })
}

/// The workflow a journal opened with: its text, or the workflow of its
/// bundle.
pub fn workflow_of(opened: &Json) -> Result<Workflow, String> {
    let text = match opened["workflow"].as_str() {
        Some(text) => text.to_owned(),
        None => journaled_text(opened, WORKFLOW_PATH)?
            .ok_or_else(|| "the journal opened with no workflow and no bundle".to_owned())?,
    };
    Workflow::parse(&text)
        .map_err(|refusals| format!("the workflow of the journal does not parse: {refusals}"))
}

/// The bundle document a journal opened with, when it gave a bundle.
pub fn bundle_document_of(opened: &Json) -> Result<Option<BundleDocument>, String> {
    let Some(text) = journaled_text(opened, BUNDLE_PATH)? else {
        return Ok(None);
    };
    BundleDocument::parse(&text).map(Some).map_err(|refusals| {
        format!("the bundle document of the journal does not parse: {refusals}")
    })
}

/// The documents of `bundle` that the journal of a run keeps: the bundle
/// document and the workflow, which the director parses. The bundle
/// document names each other file by its digest, and a file of `files/`,
/// such as the report of a sweep with the journal of each of its runs, can
/// be large.
fn journaled_bundle(bundle: &[BundleFile]) -> Vec<&BundleFile> {
    bundle
        .iter()
        .filter(|file| file.path == BUNDLE_PATH || file.path == WORKFLOW_PATH)
        .collect()
}

/// The text of the document at `path` of the bundle a journal opened with;
/// none for a journal that opened with no bundle.
fn journaled_text(opened: &Json, path: &str) -> Result<Option<String>, String> {
    if opened["bundle"].is_null() {
        return Ok(None);
    }
    let files: Vec<BundleFile> = serde_json::from_value(opened["bundle"].clone())
        .map_err(|e| format!("the bundle of the journal: {e}"))?;
    files
        .into_iter()
        .find(|file| file.path == path)
        .map(|file| Some(file.content))
        .ok_or_else(|| format!("the bundle of the journal has no `{path}`"))
}

/// Whether a run reads the simulation clock `domain` of the director:
/// every robot it binds reads that clock.
fn reads_the_clock(domain: &str, plan: &Plan, fleet: &Fleet) -> bool {
    plan.roles.values().all(|role| {
        fleet
            .robots
            .get(role.robot.as_str())
            .is_some_and(|robot| robot.clock == domain)
    })
}

/// Parses JSON5 text within the limits of a document.
pub fn parses(text: &str) -> Result<Json, String> {
    workflow_documents::parse(text).map_err(|e| e.to_string())
}

/// The work cell block of a run made from roles, from its JSON5 text.
fn work_cell_of(text: &str) -> Result<WorkCell, String> {
    serde_json::from_value(parses(text)?).map_err(|e| e.to_string())
}

/// The validation of `bundle`, when it has one.
fn validation_of(bundle: Option<&Bundle>) -> Option<&Validation> {
    bundle.and_then(|bundle| bundle.document.validation.as_ref())
}

/// What of a run does not fit the validation of its bundle: each value of
/// its work cell block of another shape than its parameter, or of a watch
/// parameter, and each of its own watch parameters that the bundle names,
/// whose value the director reads by the parameter of the bundle.
fn bundle_misfits(validation: Option<&Validation>, document: &RunDocument) -> Vec<Located> {
    let Some(validation) = validation else {
        return Vec::new();
    };
    let work_cell = validation
        .work_cell_refusals("work_cell", &document.work_cell)
        .into_iter()
        .map(|refusal| refusal.located());
    let watch = document
        .watch
        .keys()
        .filter(|name| validation.work_cell_parameters.contains_key(*name))
        .map(|name| Located {
            at: format!("watch.{name}"),
            message: "is a parameter of the bundle, which the director reads by the bundle; name the watch parameter of the run another way".to_owned(),
        });
    work_cell.chain(watch).collect()
}

/// The roles a run names: a JSON5 map of each role to a robot name or
/// `"auto"`; a role it leaves out binds `"auto"` too.
fn named_roles(text: &str) -> Result<BTreeMap<Name, RobotName>, String> {
    let named: BTreeMap<Name, RoleBinding> =
        serde_json::from_value(parses(text)?).map_err(|e| e.to_string())?;
    Ok(named
        .into_iter()
        .filter_map(|(role, binding)| match binding {
            RoleBinding::Robot(robot) => Some((role, robot)),
            RoleBinding::Auto => None,
        })
        .collect())
}

/// What a run of an unattended run on a physical robot whose run document
/// gives `versions: "warn"` is told.
const WARN_ON_A_PHYSICAL_ROBOT: &str = "an unattended run on a physical robot runs only on a stack that matches the lock of its bundle: give its run document `versions: \"refuse\"`, or no `versions`";

/// The digest of `bundle`, without its trials.
pub fn digest_of(bundle: &Bundle) -> Sha256 {
    Sha256::parse(&bundle.document.digest().to_string())
        .expect("a digest is 64 lowercase hex digits")
}

/// Why the work cell of an unattended run on a physical robot does not fit
/// its bundle, given the values `uncovered` that no test covered: the run
/// has no bundle, its bundle has no envelope, or a value is missing or
/// outside the envelope. None when it fits.
fn work_cell_unfit(bundle: Option<&Bundle>, uncovered: &[Uncovered]) -> Option<String> {
    let Some(bundle) = bundle else {
        return Some("an unattended run on a physical robot runs the bundle of a signed-off sweep, inside its envelope: give the bundle in place of the workflow".to_owned());
    };
    if bundle.document.validation.is_none() {
        return Some("the bundle has no envelope: an unattended run on a physical robot runs only the bundle of a signed-off sweep".to_owned());
    }
    if uncovered.is_empty() {
        return None;
    }
    let uncovered: Vec<&Uncovered> = uncovered.iter().collect();
    Some(outside_the_envelope(&uncovered))
}

/// Why a run of `caller` does not bind `robot`, which `lease` holds: a
/// lease of the caller itself goes to the director with an offer.
fn leased_by(robot: &str, lease: &fleet::Lease, caller: &Caller) -> String {
    match lease.holder == caller.instance && lease.holder_core_node == caller.core_node {
        true => format!(
            "`{robot}` is leased by the caller of the run under `{}`: offer the lease to the director (fleet.offer) before the run",
            lease.name
        ),
        false => format!(
            "`{robot}` is leased by {} under `{}`: wait for its release, or name another robot",
            lease.holder, lease.name
        ),
    }
}

/// Who a binding is for: the caller of the run, the director, to which the
/// caller offers its leases, and the trial of a run of a trial, whose leases
/// the director holds for its runs.
#[derive(Clone, Copy)]
struct Parties<'p> {
    caller: &'p Caller,
    me: &'p Caller,
    trial: Option<&'p str>,
}

/// The name of the leases that a trial `trial` takes.
pub fn trial_lease_name(trial: &str) -> String {
    format!("trial {trial}")
}

/// The roles of a run bound to robots: the named roles first, then the
/// `"auto"` roles. In an unattended run, the `"auto"` roles take the robots
/// of the first record of `records` (the sign-off records of the bundle of
/// the run, in their order) whose binding agrees with the named roles and
/// whose robots are all present and free by the rule of `"auto"`; with no
/// such record, and in an attended run, each `"auto"` role in name order
/// takes the first fitting robot in name order, simulated in an unattended
/// run. `held` gives each robot that another run or a teach of the director
/// holds.
fn bind(
    workflow: &Workflow,
    document: &RunDocument,
    snapshot: &Snapshot,
    parties: &Parties<'_>,
    held: &BTreeMap<RobotName, Holder>,
    records: &[SignOffRecord],
) -> Result<BTreeMap<Name, RobotName>, Vec<Located>> {
    let Parties { caller, me, trial } = *parties;
    let fleet = &snapshot.fleet;
    let of_the_trial = trial.map(trial_lease_name);
    let offered: BTreeSet<&str> = fleet
        .leases
        .iter()
        .filter(|lease| {
            let offered_by_the_caller = lease.receiver == me.instance
                && lease.receiver_core_node == me.core_node
                && lease.holder == caller.instance
                && lease.holder_core_node == caller.core_node;
            let held_for_the_trial = lease.holder == me.instance
                && lease.holder_core_node == me.core_node
                && Some(&lease.name) == of_the_trial.as_ref();
            offered_by_the_caller || held_for_the_trial
        })
        .map(|lease| lease.robot.as_str())
        .collect();
    let mut errors = Vec::new();
    let mut binding = BTreeMap::new();
    for (role, bound) in &document.roles {
        let RoleBinding::Robot(robot) = bound else {
            continue;
        };
        let at = format!("roles.{role}");
        if let Some(holder) = held.get(robot) {
            errors.push(Located {
                at,
                message: holder.why(robot.as_str()),
            });
            continue;
        }
        if let Some(lease) = fleet.lease_of(robot.as_str())
            && !offered.contains(robot.as_str())
        {
            errors.push(Located {
                at,
                message: leased_by(robot.as_str(), lease, caller),
            });
            continue;
        }
        binding.insert(role.clone(), robot.clone());
    }
    let autos: Vec<&Name> = document
        .roles
        .iter()
        .filter(|(_, bound)| **bound == RoleBinding::Auto)
        .map(|(role, _)| role)
        .collect();
    // The first reason `robot` cannot take `role` by the rule of `"auto"`,
    // none when it can, with the robots `taken` by other roles.
    let free = |role: &Name, robot: &fleet::Robot, taken: &BTreeSet<&str>| -> Option<String> {
        let name = &robot.name;
        if taken.contains(name.as_str()) {
            return Some(format!("`{name}` takes another role of the run"));
        }
        if let Some(holder) = RobotName::parse(name)
            .ok()
            .and_then(|robot| held.get(&robot))
        {
            return Some(holder.why(name));
        }
        match offered.is_empty() {
            true => {
                if let Some(lease) = fleet.lease_of(name) {
                    return Some(leased_by(name, lease, caller));
                }
            }
            false => {
                if !offered.contains(name.as_str()) {
                    return Some(format!("the caller offered `{name}` no lease"));
                }
            }
        }
        checker::misfits(workflow, snapshot, role, robot, &document.confirmed)
            .into_iter()
            .next()
    };
    let unattended = document.attendance == Attendance::Unattended;
    let by_record = (unattended && !autos.is_empty()).then(|| {
        let taken: BTreeSet<&str> = binding.values().map(RobotName::as_str).collect();
        let fits = |record: &&SignOffRecord| {
            let binds = &record.sign_off.binding;
            let agrees = document.roles.iter().all(|(role, bound)| match bound {
                RoleBinding::Robot(robot) => binds.get(role).is_some_and(|b| b.name == *robot),
                RoleBinding::Auto => binds.contains_key(role),
            });
            agrees
                && autos.iter().all(|role| {
                    let robot = binds[*role].name.as_str();
                    fleet
                        .robots
                        .get(robot)
                        .is_some_and(|robot| free(role, robot, &taken).is_none())
                })
        };
        records.iter().find(fits).map(|record| {
            autos
                .iter()
                .map(|role| ((*role).clone(), record.sign_off.binding[*role].name.clone()))
                .collect::<Vec<_>>()
        })
    });
    // The roles that the robots of a record do not bind, each to its own robot.
    let one_by_one = match by_record.flatten() {
        Some(bound) => {
            binding.extend(bound);
            Vec::new()
        }
        None => autos,
    };
    for role in one_by_one {
        let taken: BTreeSet<&str> = binding.values().map(RobotName::as_str).collect();
        // The first reason `robot` cannot take the role, none when it can.
        let why_not = |robot: &fleet::Robot| -> Option<String> {
            if unattended && robot.is_physical() {
                return Some(format!(
                    "`{}` is physical: in an unattended run, \"auto\" binds a physical robot only as a sign-off record of the bundle binds it",
                    robot.name
                ));
            }
            free(role, robot, &taken)
        };
        match fleet.robots.values().find(|robot| why_not(robot).is_none()) {
            Some(robot) => {
                binding.insert(
                    role.clone(),
                    RobotName::parse(&robot.name).expect("a robot name of the listing"),
                );
            }
            None => {
                let reasons: Vec<String> = fleet.robots.values().filter_map(why_not).collect();
                let message = match reasons.is_empty() {
                    true => format!("no robot fits the role `{role}`: the fleet lists no robot"),
                    false => format!("no robot fits the role `{role}`: {}", reasons.join("; ")),
                };
                errors.push(Located {
                    at: format!("roles.{role}"),
                    message,
                });
            }
        }
    }
    match errors.is_empty() {
        true => Ok(binding),
        false => Err(errors),
    }
}

/// The contract a fleet entry with limbs implements.
const LIMB_STATE: (&str, &str) = ("limb_state", "v1");

/// A node that a run uses: its entry of a lock, whether a `call` step of the
/// run calls it, and whether each call it serves is of a role bound to a
/// fleet entry with no limbs.
#[derive(Clone, Debug, PartialEq)]
pub struct RunNode {
    name: NameTag,
    locked: LockedNode,
    called: bool,
    serves_devices_only: bool,
}

/// The nodes a run uses: the director, the anomaly monitor and each node of
/// the copy of each robot it binds.
pub fn run_nodes(snapshot: &Snapshot, plan: &Plan, me: &Caller) -> Vec<RunNode> {
    let fleet = &snapshot.fleet;
    let robots: BTreeSet<&str> = plan
        .roles
        .values()
        .map(|role| role.robot.as_str())
        .collect();
    let monitor = plan
        .monitor
        .as_ref()
        .map(|m| m.target.producer.instance.as_str());
    let has_limbs = |role: &Name| {
        let (name, tag) = LIMB_STATE;
        fleet
            .robots
            .get(plan.roles[role].robot.as_str())
            .is_some_and(|robot| robot.contract(name, tag).is_some())
    };
    let called: BTreeMap<&str, bool> =
        plan.calls
            .values()
            .fold(BTreeMap::new(), |mut called, call| {
                let device = !has_limbs(&call.target.role);
                let all_devices = called
                    .entry(call.target.producer.instance.as_str())
                    .or_insert(true);
                *all_devices = *all_devices && device;
                called
            });
    fleet
        .nodes
        .iter()
        .filter(|node| {
            node.instances.iter().any(|(instance, core_node, robot)| {
                robots.contains(robot.as_str())
                    || (instance == &me.instance && core_node == &me.core_node)
                    || Some(instance.as_str()) == monitor
            })
        })
        .filter_map(|node| {
            let name = NameTag::parse(&format!("{}:{}", node.name, node.tag)).ok()?;
            let sha256 = Sha256::parse(&node.machines.first()?.manifest_sha256).ok()?;
            let commits: BTreeSet<&str> = node.machines.iter().map(|m| m.commit.as_str()).collect();
            let commit = match commits.into_iter().collect::<Vec<_>>().as_slice() {
                [one] if !one.is_empty() => Some((*one).to_owned()),
                _ => None,
            };
            let serving: Vec<bool> = node
                .instances
                .iter()
                .filter_map(|(instance, _, _)| called.get(instance.as_str()).copied())
                .collect();
            Some(RunNode {
                name,
                locked: LockedNode { sha256, commit },
                called: !serving.is_empty(),
                serves_devices_only: !serving.is_empty() && serving.iter().all(|device| *device),
            })
        })
        .collect()
}

/// The versions of a run: the peppy version, the pin of each contract it
/// reads, and each node it uses. The contracts are those of its calls, its
/// observations and its monitor, at the pin of the instance each targets;
/// locations:v1 of each robot whose skill it gives a location; alert:v1 of
/// each physical robot it binds; and robot_identity:v1 of each robot whose
/// role names a model. The lock holds one pin per contract, so a robot that
/// lists one of these last three at two pins refuses the run, naming the
/// robot; so does a contract that the run reads at two shas, naming the
/// robot when both reads are of one robot, else the robots of the run; and
/// a pin of the listing that does not read.
pub fn versions(snapshot: &Snapshot, plan: &Plan, nodes: &[RunNode]) -> Result<Lock, String> {
    let targets = plan
        .calls
        .values()
        .map(|call| &call.target)
        .chain(plan.observations.values().map(|o| &o.target))
        .chain(plan.monitor.iter().map(|m| &m.target))
        .map(|target| {
            let robot = plan.roles[&target.role].robot.as_str().to_owned();
            (target.contract.clone(), target.sha256.clone(), robot)
        });
    // The contracts the director reads itself, at every pin that the robot
    // lists for them: a robot may list one at two pins, one per instance.
    let mut read_by_the_director = Vec::new();
    for (role, bound) in &plan.roles {
        let Some(robot) = snapshot.fleet.robots.get(bound.robot.as_str()) else {
            continue;
        };
        let gives_locations = plan.locations.iter().any(|skill| skill.role == *role);
        let names_a_model = plan
            .workflow
            .roles
            .get(role)
            .is_some_and(|role| role.model.is_some());
        let read = [
            (fleet::LOCATIONS, gives_locations),
            (alerts::ALERT, bound.physical),
            (fleet::ROBOT_IDENTITY, names_a_model),
        ];
        for ((name, tag), _) in read.into_iter().filter(|(_, read)| *read) {
            let pins: BTreeSet<&str> = robot
                .contracts
                .iter()
                .filter(|pin| pin.name == name && pin.tag == tag)
                .map(|pin| pin.sha256.as_str())
                .collect();
            if let [first, second, ..] = pins.into_iter().collect::<Vec<_>>()[..] {
                return Err(format!(
                    "`{}` pins `{name}:{tag}` at two shas, `{first}` and `{second}`: run one build of it on that robot",
                    robot.name
                ));
            }
            if let Some(pin) = robot.contract(name, tag) {
                read_by_the_director.push((
                    format!("{name}:{tag}"),
                    pin.sha256.clone(),
                    robot.name.clone(),
                ));
            }
        }
    }
    let mut contracts: BTreeMap<NameTag, (Sha256, String)> = BTreeMap::new();
    for (contract, sha256, robot) in targets.chain(read_by_the_director) {
        let name = NameTag::parse(&contract).map_err(|error| {
            format!("the contract `{contract}` of the fleet listing does not read: {error}")
        })?;
        let pin = Sha256::parse(&sha256).map_err(|error| {
            format!(
                "the pin `{sha256}` of `{contract}` in the fleet listing does not read: {error}"
            )
        })?;
        if let Some((other, other_robot)) = contracts.get(&name)
            && *other != pin
        {
            return Err(if *other_robot == robot {
                format!(
                    "`{robot}` pins `{name}` at two shas, `{other}` and `{pin}`: run one build of it on that robot"
                )
            } else {
                format!(
                    "the robots of the run pin `{name}` at two shas, `{other}` and `{pin}`: bind robots whose software pins it alike"
                )
            });
        }
        contracts.insert(name, (pin, robot));
    }
    let nodes = nodes
        .iter()
        .map(|node| (node.name.clone(), node.locked.clone()))
        .collect();
    Ok(Lock {
        contracts: contracts
            .into_iter()
            .map(|(name, (pin, _))| (name, pin))
            .collect(),
        peppy: snapshot.fleet.version.clone(),
        nodes,
    })
}

/// The differences of a run from the lock of its bundle as its versions
/// policy takes them: with `warn`, or with no difference, the run goes on and
/// its result names each; with `refuse`, a difference refuses the run, and
/// the refusal says how many and what to give the run document to run on
/// this stack as it is.
fn under_the_policy(
    policy: VersionPolicy,
    differences: Vec<Difference>,
) -> Result<Vec<Difference>, (String, Vec<Difference>)> {
    if differences.is_empty() || policy == VersionPolicy::Warn {
        return Ok(differences);
    }
    let message = format!(
        "the stack differs from the lock of the bundle in {} ways; to run on this stack as it is, give the run document `versions: \"warn\"`",
        differences.len()
    );
    Err((message, differences))
}

/// The differences of a run from the lock of its bundle, and the nodes it
/// uses that the lock does not name and that no `call` step calls.
pub fn differences(
    lock: &Lock,
    stack: &Lock,
    nodes: &[RunNode],
) -> (Vec<Difference>, Vec<NameTag>) {
    let difference =
        |kind: DifferenceKind, subject: String, locked: Option<String>, stack: Option<String>| {
            Difference {
                kind,
                subject,
                locked,
                stack,
            }
        };
    let mut found = Vec::new();
    if lock.peppy != stack.peppy || lock.peppy == UNKNOWN_PEPPY || stack.peppy == UNKNOWN_PEPPY {
        found.push(difference(
            DifferenceKind::Peppy,
            String::new(),
            Some(lock.peppy.clone()),
            Some(stack.peppy.clone()),
        ));
    }
    for (contract, sha256) in &stack.contracts {
        if let Some(locked) = lock.contracts.get(contract)
            && locked != sha256
        {
            found.push(difference(
                DifferenceKind::Contract,
                contract.to_string(),
                Some(locked.to_string()),
                Some(sha256.to_string()),
            ));
        }
    }
    let mut unlocked = Vec::new();
    for node in nodes {
        let running = &node.locked;
        match lock
            .nodes
            .iter()
            .find(|(locked, _)| locked.name == node.name.name)
        {
            Some((locked_name, locked))
                if locked_name != &node.name || locked.sha256 != running.sha256 =>
            {
                let text = |name: &NameTag, at: &LockedNode| format!("{name} {}", at.sha256);
                found.push(difference(
                    DifferenceKind::Node,
                    node.name.to_string(),
                    Some(text(locked_name, locked)),
                    Some(text(&node.name, running)),
                ));
            }
            Some((_, locked))
                if locked.commit.is_none()
                    || running.commit.is_none()
                    || locked.commit != running.commit =>
            {
                found.push(difference(
                    DifferenceKind::NodeCommit,
                    node.name.to_string(),
                    locked.commit.clone(),
                    running.commit.clone(),
                ));
            }
            Some(_) => {}
            None if node.called && !node.serves_devices_only => {
                found.push(difference(
                    DifferenceKind::UnlockedNode,
                    node.name.to_string(),
                    None,
                    Some(running.sha256.to_string()),
                ));
            }
            None if !node.called => unlocked.push(node.name.clone()),
            None => {}
        }
    }
    (found, unlocked)
}

/// The result of a run before its preparation reads its workflow.
pub fn empty_result() -> RunResult {
    RunResult {
        ending: Ending::Refused,
        workflow: None,
        lanes: Vec::new(),
        refusal: None,
        failed: None,
        assumptions: Vec::new(),
        loops: Vec::new(),
        observations: BTreeMap::new(),
        duration: RunDuration {
            run_s: 0.0,
            wall_s: 0.0,
        },
        caller_data: None,
        core_node: String::new(),
        versions: Lock::default(),
        differences: Vec::new(),
        unlocked_nodes: Vec::new(),
        incidents: Vec::new(),
        work_cell: WorkCell::new(),
        watch: BTreeMap::new(),
        uncovered: Vec::new(),
    }
}

fn refused_result(base: RunResult, message: &str, started: u64) -> RunResult {
    RunResult {
        ending: Ending::Refused,
        refusal: Some(RunRefusal {
            message: message.to_owned(),
            cause: RefusalCause::Checks,
            errors: Vec::new(),
            assumption: None,
            says: None,
        }),
        duration: RunDuration {
            run_s: 0.0,
            wall_s: wall_ns().saturating_sub(started) as f64 / 1e9,
        },
        ..base
    }
}

/// The result of a run that an earlier process of the director did not
/// end: interrupted, its robots in an unknown state.
fn interrupted_result(workflow: Option<&Workflow>, records: &[Record]) -> RunResult {
    let mut result = empty_result();
    result.ending = Ending::Interrupted;
    result.workflow = workflow.map(RunWorkflow::of);
    let wall: Vec<u64> = records.iter().map(|r| r.wall_ns).collect();
    if let (Some(first), Some(last)) = (wall.iter().min(), wall.iter().max()) {
        result.duration.wall_s = (last - first) as f64 / 1e9;
    }
    result
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use workflow_documents::run::Attendance;

    use super::*;
    use crate::fleet::{Lease, Machine, Node};
    use crate::testing::{name, one_arm, plan, plan_on, robot, robot_name, snapshot, workflow};

    const DIRECTOR: &str = "director_inst";
    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn me() -> Caller {
        Caller {
            instance: DIRECTOR.to_owned(),
            core_node: "core".to_owned(),
        }
    }

    fn chat() -> Caller {
        Caller {
            instance: "chat".to_owned(),
            core_node: "core".to_owned(),
        }
    }

    fn rest() -> serde_json::Value {
        one_arm(
            "arm",
            json!({ "id": "rest", "timeout_s": 5, "call": { "member": "arm.home", "args": { "duration_s": 1 } } }),
            json!([]),
        )
    }

    fn lease(robot: &str, holder: &Caller, receiver: Option<&Caller>) -> Lease {
        Lease {
            robot: robot.to_owned(),
            holder: holder.instance.clone(),
            holder_core_node: holder.core_node.clone(),
            name: "held".to_owned(),
            open: false,
            receiver: receiver.map(|r| r.instance.clone()).unwrap_or_default(),
            receiver_core_node: receiver.map(|r| r.core_node.clone()).unwrap_or_default(),
        }
    }

    fn bound(
        fleet: &Snapshot,
        roles: serde_json::Value,
        attendance: Attendance,
    ) -> Result<BTreeMap<Name, RobotName>, Vec<String>> {
        let workflow = workflow(&rest());
        let named: BTreeMap<Name, RobotName> = serde_json::from_value(roles).unwrap();
        let mut document = RunDocument::of_roles(&workflow, named, WorkCell::new()).unwrap();
        document.attendance = attendance;
        bind(
            &workflow,
            &document,
            fleet,
            &Parties {
                caller: &chat(),
                me: &me(),
                trial: None,
            },
            &BTreeMap::new(),
            &[],
        )
        .map_err(|errors| errors.into_iter().map(|e| e.message).collect())
    }

    /// A request of a run with no workflow source, a run document with roles or with a work cell block, has no
    /// shape, each named.
    #[test]
    fn a_run_document_goes_with_neither_roles_nor_a_work_cell() {
        let request = RunRequest {
            workflow: Some("{}".to_owned()),
            bundle: None,
            run_document: Some("{}".to_owned()),
            roles: None,
            work_cell: None,
            started_by: String::new(),
            caller_data: None,
            trial: None,
        };
        assert!(request.shape().is_ok());
        let refused = |request: RunRequest| request.shape().err().unwrap_or_default();
        assert_eq!(
            refused(RunRequest {
                roles: Some("{}".to_owned()),
                ..request.clone()
            }),
            "give a run_document or roles, not both"
        );
        assert_eq!(
            refused(RunRequest {
                work_cell: Some("{}".to_owned()),
                ..request.clone()
            }),
            "a run_document gives its own work_cell: give work_cell with roles"
        );
        assert_eq!(
            refused(RunRequest {
                workflow: None,
                ..request
            }),
            "give a workflow or a bundle"
        );
    }

    /// A value of the work cell of another shape than its parameter in the bundle, and a watch parameter of the run
    /// that the bundle names, do not fit the bundle; with no bundle, nothing misfits.
    #[test]
    fn a_work_cell_of_another_shape_or_a_watch_the_bundle_names_does_not_fit_the_bundle() {
        let validation = crate::testing::validation(
            json!({
                "belt_speed_mps": { "unit": "m/s", "kind": "control" },
                "loose": { "unit": "items", "kind": "watch", "observation": "target.count" },
            }),
            json!({ "belt_speed_mps": [0.05, 0.25], "loose": [0, 3] }),
        );
        let workflow = workflow(&rest());
        let mut document = RunDocument::of_roles(
            &workflow,
            BTreeMap::new(),
            serde_json::from_value(json!({ "belt_speed_mps": { "arm": 0.2 } })).unwrap(),
        )
        .unwrap();
        document.watch = serde_json::from_value(
            json!({ "loose": { "unit": "items", "observation": "target.count" } }),
        )
        .unwrap();
        let found: Vec<String> = bundle_misfits(Some(&validation), &document)
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(
            found,
            [
                "work_cell.belt_speed_mps: has one value in the bundle; give one value, not a value per role",
                "watch.loose: is a parameter of the bundle, which the director reads by the bundle; name the watch parameter of the run another way",
            ]
        );
        assert!(bundle_misfits(None, &document).is_empty());
    }

    #[test]
    fn auto_binds_the_first_free_robot_that_fits_in_name_order() {
        let mut physical = robot("able");
        physical.kind = "physical".to_owned();
        let mut commanded = robot("baker");
        commanded.robot_commander = "headset".to_owned();
        let mut fleet = snapshot(vec![
            physical,
            commanded,
            robot("charlie"),
            robot("delta"),
            robot("echo"),
        ]);
        fleet.fleet.leases.push(lease("charlie", &chat(), None));
        let binding = bound(&fleet, json!({}), Attendance::Unattended).unwrap();
        assert_eq!(
            binding[&name("arm")],
            robot_name("delta"),
            "able is physical, baker commanded, charlie leased"
        );
    }

    #[test]
    fn a_robot_the_caller_leases_and_does_not_offer_asks_for_the_offer() {
        let mut fleet = snapshot(vec![robot("alpha")]);
        fleet.fleet.leases.push(lease("alpha", &chat(), None));
        let errors = bound(&fleet, json!({ "arm": "alpha" }), Attendance::Attended).unwrap_err();
        assert!(
            errors[0].contains("offer the lease to the director"),
            "{errors:?}"
        );
    }

    /// `"auto"` passes over a robot whose instance that serves a call giving a location implements no
    /// locations:v1, and binds the next that fits.
    #[test]
    fn auto_binds_no_robot_whose_skill_knows_no_locations() {
        let mut value = one_arm(
            "arm",
            json!({ "id": "go", "timeout_s": 5, "call": { "member": "arm.go_to", "args": { "place": { "$location": "bin" } } } }),
            json!([]),
        );
        value["locations"] = json!(["bin"]);
        let workflow = workflow(&value);
        let mut unknowing = robot("able");
        unknowing
            .contracts
            .retain(|contract| contract.name != "locations");
        let fleet = snapshot(vec![unknowing, robot("baker")]);
        let document = RunDocument::of_roles(&workflow, BTreeMap::new(), WorkCell::new()).unwrap();
        let binding = bind(
            &workflow,
            &document,
            &fleet,
            &Parties {
                caller: &chat(),
                me: &me(),
                trial: None,
            },
            &BTreeMap::new(),
            &[],
        )
        .unwrap();
        assert_eq!(binding[&name("arm")], robot_name("baker"));
    }

    #[test]
    fn a_role_that_no_robot_fits_names_why_each_does_not() {
        let mut physical = robot("able");
        physical.kind = "physical".to_owned();
        let mut fleet = snapshot(vec![physical, robot("baker")]);
        fleet.fleet.leases.push(lease("baker", &chat(), None));
        let errors = bound(&fleet, json!({}), Attendance::Unattended).unwrap_err();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(
            errors[0].starts_with("no robot fits the role `arm`: "),
            "{errors:?}"
        );
        assert!(errors[0].contains("`able` is physical"), "{errors:?}");
        assert!(
            errors[0].contains("`baker` is leased by the caller"),
            "{errors:?}"
        );
    }

    #[test]
    fn auto_binds_no_physical_robot_for_a_caller_that_confirms_assumptions() {
        let mut physical = robot("able");
        physical.kind = "physical".to_owned();
        let fleet = snapshot(vec![physical, robot("baker")]);
        let workflow = workflow(&rest());
        let mut document =
            RunDocument::of_roles(&workflow, BTreeMap::new(), WorkCell::new()).unwrap();
        document.attendance = Attendance::Attended;
        document.confirmed = [name("clear")].into();
        let binding = bind(
            &workflow,
            &document,
            &fleet,
            &Parties {
                caller: &chat(),
                me: &me(),
                trial: None,
            },
            &BTreeMap::new(),
            &[],
        )
        .unwrap();
        assert_eq!(binding[&name("arm")], robot_name("baker"));
    }

    #[test]
    fn auto_binds_only_the_robots_the_caller_offered_when_it_offered_any() {
        let mut fleet = snapshot(vec![robot("alpha"), robot("bravo")]);
        fleet
            .fleet
            .leases
            .push(lease("bravo", &chat(), Some(&me())));
        assert_eq!(
            bound(&fleet, json!({}), Attendance::Attended).unwrap()[&name("arm")],
            robot_name("bravo")
        );
    }

    #[test]
    fn a_named_robot_leased_by_another_client_is_refused_and_an_offered_one_binds() {
        let mut fleet = snapshot(vec![robot("alpha")]);
        let headset = Caller {
            instance: "headset".to_owned(),
            core_node: "core".to_owned(),
        };
        fleet.fleet.leases.push(lease("alpha", &headset, None));
        let refused = bound(&fleet, json!({ "arm": "alpha" }), Attendance::Attended).unwrap_err();
        assert_eq!(
            refused[0],
            "`alpha` is leased by headset under `held`: wait for its release, or name another robot"
        );
        fleet.fleet.leases = vec![lease("alpha", &chat(), Some(&me()))];
        assert_eq!(
            bound(&fleet, json!({ "arm": "alpha" }), Attendance::Attended).unwrap()[&name("arm")],
            robot_name("alpha")
        );
    }

    #[test]
    fn a_robot_of_another_run_of_the_director_is_neither_named_nor_auto() {
        let fleet = snapshot(vec![robot("alpha")]);
        let workflow = workflow(&rest());
        let elsewhere = BTreeMap::from([(robot_name("alpha"), Holder::Run("r1".to_owned()))]);
        let document = |roles: serde_json::Value| {
            RunDocument::of_roles(
                &workflow,
                serde_json::from_value(roles).unwrap(),
                WorkCell::new(),
            )
            .unwrap()
        };
        let named = bind(
            &workflow,
            &document(json!({ "arm": "alpha" })),
            &fleet,
            &Parties {
                caller: &chat(),
                me: &me(),
                trial: None,
            },
            &elsewhere,
            &[],
        );
        let held = "`alpha` takes part in run r1 of this director: wait for its end, or abort it";
        assert_eq!(
            named.unwrap_err(),
            [Located {
                at: "roles.arm".to_owned(),
                message: held.to_owned()
            }]
        );
        assert!(
            bind(
                &workflow,
                &document(json!({})),
                &fleet,
                &Parties {
                    caller: &chat(),
                    me: &me(),
                    trial: None,
                },
                &elsewhere,
                &[],
            )
            .is_err()
        );
        let taught = BTreeMap::from([(robot_name("alpha"), Holder::Teach)]);
        let teaching = bind(
            &workflow,
            &document(json!({ "arm": "alpha" })),
            &fleet,
            &Parties {
                caller: &chat(),
                me: &me(),
                trial: None,
            },
            &taught,
            &[],
        );
        assert_eq!(
            teaching.unwrap_err()[0].message,
            "`alpha` is being taught its locations by this director: run again once the teach ends"
        );
    }

    /// A physical robot named `name`, with the contracts of the tests.
    fn physical(name: &str) -> crate::fleet::Robot {
        let mut robot = robot(name);
        robot.kind = "physical".to_owned();
        robot
    }

    /// The binding of the roles of an unattended run of `value` that names
    /// the robots `named`, on `fleet`, with the sign-off records `records`.
    fn bound_by_records(
        value: &serde_json::Value,
        named: serde_json::Value,
        fleet: &Snapshot,
        records: &[SignOffRecord],
    ) -> Result<BTreeMap<Name, RobotName>, Vec<Located>> {
        let workflow = workflow(value);
        let mut document = RunDocument::of_roles(
            &workflow,
            serde_json::from_value(named).unwrap(),
            WorkCell::new(),
        )
        .unwrap();
        document.attendance = Attendance::Unattended;
        bind(
            &workflow,
            &document,
            fleet,
            &Parties {
                caller: &chat(),
                me: &me(),
                trial: None,
            },
            &BTreeMap::new(),
            records,
        )
    }

    /// In an unattended run, `"auto"` binds a physical robot as the first
    /// sign-off record that fits binds it, in the order of their robots; a
    /// lease of a robot of the first gives the binding of the second, and
    /// with no record that fits, `"auto"` binds only a simulated robot.
    #[test]
    fn auto_binds_physical_robots_only_as_the_first_record_that_fits_binds_them() {
        use crate::sign_off::tests::record;
        let mut fleet = snapshot(vec![physical("arm_a"), physical("arm_b"), robot("sim")]);
        let records = [
            record("t2", &[("arm", "arm_a")]),
            record("t1", &[("arm", "arm_b")]),
        ];
        let arm = |fleet: &Snapshot, records: &[SignOffRecord]| {
            bound_by_records(&rest(), json!({}), fleet, records).unwrap()[&name("arm")].clone()
        };
        assert_eq!(arm(&fleet, &records), robot_name("arm_a"));
        assert_eq!(
            arm(&fleet, &records),
            arm(&fleet, &records),
            "the same listing and records give the same binding"
        );
        fleet.fleet.leases.push(lease("arm_a", &chat(), None));
        assert_eq!(arm(&fleet, &records), robot_name("arm_b"));
        fleet.fleet.leases.push(lease("arm_b", &chat(), None));
        assert_eq!(arm(&fleet, &records), robot_name("sim"));
        assert_eq!(
            arm(&snapshot(vec![physical("arm_a"), robot("sim")]), &[]),
            robot_name("sim")
        );
    }

    /// A record binds the `"auto"` roles only when its binding agrees with
    /// the named roles; with none that agrees, an `"auto"` role takes a
    /// simulated robot, and with none either, the refusal names why each
    /// physical robot does not fit.
    #[test]
    fn a_record_binds_the_auto_roles_only_when_it_agrees_with_the_named_roles() {
        use crate::sign_off::tests::record;
        let mut value = rest();
        value["roles"] = json!({ "arm": {}, "helper": {} });
        value["actions"]["helper"] = value["actions"]["arm"].clone();
        let fleet = snapshot(vec![
            physical("arm_a"),
            physical("arm_b"),
            physical("arm_c"),
        ]);
        let records = [
            record("t1", &[("arm", "arm_a"), ("helper", "arm_b")]),
            record("t2", &[("arm", "arm_c"), ("helper", "arm_a")]),
        ];
        let bound = bound_by_records(&value, json!({ "arm": "arm_c" }), &fleet, &records).unwrap();
        assert_eq!(bound[&name("helper")], robot_name("arm_a"));
        let refused =
            bound_by_records(&value, json!({ "arm": "arm_b" }), &fleet, &records).unwrap_err();
        assert_eq!(refused.len(), 1, "{refused:?}");
        assert!(
            refused[0].message.contains(
                "`arm_a` is physical: in an unattended run, \"auto\" binds a physical robot only as a sign-off record of the bundle binds it"
            ),
            "{refused:?}"
        );
    }

    /// The journal of a run of a bundle keeps the bundle document and the
    /// workflow, which the director parses, and no file of `files/`, which
    /// the bundle document names by its digest: the workflow and the
    /// envelope read back from it.
    #[test]
    fn the_journal_of_a_run_of_a_bundle_keeps_only_the_documents_the_director_parses() {
        let workflow = workflow(&rest());
        let validation = crate::testing::validation(
            json!({ "belt_speed_mps": { "unit": "m/s", "kind": "control" } }),
            json!({ "belt_speed_mps": [0.05, 0.25] }),
        );
        let report = BundleFile {
            path: "files/sweep.report.json5".to_owned(),
            content: "{ runs: [\"the journal of each run of the sweep\"] }".to_owned(),
        };
        let files =
            Bundle::write_signed_off(&workflow, vec![report], Lock::default(), validation.clone())
                .unwrap();
        let opened = json!({ "workflow": null, "bundle": journaled_bundle(&files) });
        let paths: Vec<&str> = opened["bundle"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|file| file["path"].as_str())
            .collect();
        assert_eq!(paths.len(), 2, "{paths:?}");
        assert!(paths.contains(&BUNDLE_PATH) && paths.contains(&WORKFLOW_PATH));
        assert_eq!(workflow_of(&opened).unwrap(), workflow);
        assert_eq!(
            bundle_document_of(&opened).unwrap().unwrap().validation,
            Some(validation)
        );
        let without = json!({ "workflow": null, "bundle": null });
        assert_eq!(bundle_document_of(&without).unwrap(), None);
        assert!(workflow_of(&without).is_err());
    }

    /// The work cell of an unattended run on a physical robot fits only a
    /// bundle with an envelope, each value inside it, and each refusal names
    /// its cause.
    #[test]
    fn the_work_cell_of_an_unattended_run_on_a_physical_robot_fits_only_inside_an_envelope() {
        let workflow = workflow(&rest());
        let validation = crate::testing::validation(
            json!({ "belt_speed_mps": { "unit": "m/s", "kind": "control" } }),
            json!({ "belt_speed_mps": [0.05, 0.25] }),
        );
        let lock = Lock::default();
        let open = |validation: Option<workflow_documents::Validation>| {
            let files = match validation {
                Some(validation) => {
                    Bundle::write_signed_off(&workflow, Vec::new(), lock.clone(), validation)
                }
                None => Bundle::write(&workflow, Vec::new(), lock.clone()),
            };
            Bundle::open(files.unwrap()).unwrap()
        };
        assert!(
            work_cell_unfit(None, &[])
                .unwrap()
                .contains("give the bundle in place of the workflow")
        );
        assert!(
            work_cell_unfit(Some(&open(None)), &[])
                .unwrap()
                .starts_with("the bundle has no envelope")
        );
        let with_envelope = open(Some(validation.clone()));
        assert_eq!(work_cell_unfit(Some(&with_envelope), &[]), None);
        let outside: WorkCell = serde_json::from_value(json!({ "belt_speed_mps": 0.3 })).unwrap();
        let uncovered = uncovered_work_cell(Some(&validation), &outside);
        assert_eq!(
            work_cell_unfit(Some(&with_envelope), &uncovered).unwrap(),
            "an unattended run on a physical robot runs inside the envelope of its bundle, and no test covered `belt_speed_mps` is 0.3, outside the tested range [0.05, 0.25]"
        );
    }

    fn node(name: &str, instance: &str, robot: &str, sha256: &str, commit: &str) -> Node {
        Node {
            name: name.to_owned(),
            tag: "v1".to_owned(),
            machines: vec![Machine {
                core_node: "core".to_owned(),
                manifest_sha256: sha256.to_owned(),
                commit: commit.to_owned(),
            }],
            instances: vec![(instance.to_owned(), "core".to_owned(), robot.to_owned())],
        }
    }

    /// The nodes of a run of `rest` on `arm`: the director, the brain it
    /// calls and a camera of the copy that it does not call, and a node of
    /// another robot. With `limbs`, `arm` implements limb_state:v1; without,
    /// it is a fleet entry with no limbs.
    fn stack(brain_commit: &str, limbs: bool) -> (Snapshot, Plan) {
        let plan = plan(&rest(), Attendance::Attended);
        let mut arm = robot("arm");
        if limbs {
            let (name, tag) = LIMB_STATE;
            arm.contracts.push(crate::fleet::RobotContract {
                name: name.to_owned(),
                tag: tag.to_owned(),
                sha256: SHA_B.to_owned(),
                instances: vec!["brain".to_owned()],
            });
        }
        let mut fleet = snapshot(vec![arm, robot("other")]);
        fleet.fleet.nodes = vec![
            node("workflow_director", DIRECTOR, "", SHA_A, "c0"),
            node("test_brain", "arm_brain", "arm", SHA_A, brain_commit),
            node("camera", "arm_camera", "arm", SHA_A, "c2"),
            node("test_brain", "other_brain", "other", SHA_B, "c9"),
        ];
        (fleet, plan)
    }

    fn compared(lock: &Lock, brain_commit: &str) -> (Vec<String>, Vec<NameTag>) {
        let (found, unlocked) = differences_of(lock, brain_commit);
        (
            found
                .into_iter()
                .map(|d| format!("{} {}", d.kind, d.subject))
                .collect(),
            unlocked,
        )
    }

    /// The differences from `lock` of the stack whose brain runs the commit `brain_commit`, and the nodes the lock
    /// does not name.
    fn differences_of(lock: &Lock, brain_commit: &str) -> (Vec<Difference>, Vec<NameTag>) {
        let (fleet, plan) = stack(brain_commit, true);
        let nodes = run_nodes(&fleet, &plan, &me());
        differences(lock, &versions(&fleet, &plan, &nodes).unwrap(), &nodes)
    }

    fn lock_of_the_stack() -> Lock {
        let (fleet, plan) = stack("c1", true);
        versions(&fleet, &plan, &run_nodes(&fleet, &plan, &me())).unwrap()
    }

    #[test]
    fn the_lock_holds_the_director_and_the_nodes_of_the_bound_copy() {
        let lock = lock_of_the_stack();
        let names: Vec<String> = lock.nodes.keys().map(NameTag::to_string).collect();
        assert_eq!(
            names,
            ["camera:v1", "test_brain:v1", "workflow_director:v1"]
        );
        assert_eq!(
            lock.nodes[&NameTag::parse("test_brain:v1").unwrap()]
                .commit
                .as_deref(),
            Some("c1")
        );
        assert_eq!(compared(&lock, "c1"), (Vec::new(), Vec::new()));
    }

    /// With `refuse`, the differences of a stack from the lock refuse the run, which names how many and what to give
    /// instead, and keeps them; with `warn` the run goes on with them; no difference refuses nothing.
    #[test]
    fn a_difference_refuses_a_run_with_versions_refuse_and_goes_into_the_result_with_warn() {
        let found = differences_of(&lock_of_the_stack(), "c7").0;
        assert!(!found.is_empty(), "a changed commit is a difference");
        let Err((message, kept)) = under_the_policy(VersionPolicy::Refuse, found.clone()) else {
            panic!("`refuse` refuses a run whose stack differs");
        };
        assert_eq!(
            message,
            format!(
                "the stack differs from the lock of the bundle in {} ways; to run on this stack as it is, give the run document `versions: \"warn\"`",
                found.len()
            )
        );
        assert_eq!(kept, found);
        assert_eq!(
            under_the_policy(VersionPolicy::Warn, found.clone()),
            Ok(found)
        );
        for policy in [VersionPolicy::Refuse, VersionPolicy::Warn] {
            assert_eq!(under_the_policy(policy, Vec::new()), Ok(Vec::new()));
        }
    }

    #[test]
    fn each_change_of_a_locked_node_or_the_peppy_version_is_a_difference() {
        let brain = NameTag::parse("test_brain:v1").unwrap();
        assert_eq!(
            compared(&lock_of_the_stack(), "c7").0,
            ["node_commit test_brain:v1"]
        );
        assert_eq!(
            compared(&lock_of_the_stack(), "").0,
            ["node_commit test_brain:v1"],
            "a missing commit on the stack"
        );
        let mut lock = lock_of_the_stack();
        lock.nodes.get_mut(&brain).unwrap().sha256 = Sha256::parse(SHA_B).unwrap();
        assert_eq!(compared(&lock, "c1").0, ["node test_brain:v1"]);
        let mut lock = lock_of_the_stack();
        lock.peppy = UNKNOWN_PEPPY.to_owned();
        assert_eq!(compared(&lock, "c1").0, ["peppy "]);
    }

    /// One change of the lock of a stack, to its peppy version, to the sha256 of one of its nodes or to the commit of
    /// one, gives one difference, of that kind and of that node.
    #[hegel::test]
    fn one_change_of_a_lock_gives_one_difference_of_its_kind(tc: hegel::TestCase) {
        use hegel::generators as gs;
        let mut lock = lock_of_the_stack();
        let nodes: Vec<NameTag> = lock.nodes.keys().cloned().collect();
        let node = nodes[tc.draw(gs::integers::<usize>().max_value(nodes.len() - 1))].clone();
        let expected = match tc.draw(gs::integers::<u8>().max_value(2)) {
            0 => {
                lock.peppy = tc
                    .draw(gs::sampled_from(&["0.35.1", UNKNOWN_PEPPY]))
                    .to_owned();
                "peppy ".to_owned()
            }
            1 => {
                lock.nodes.get_mut(&node).unwrap().sha256 = Sha256::parse(SHA_B).unwrap();
                format!("node {node}")
            }
            _ => {
                lock.nodes.get_mut(&node).unwrap().commit = Some("c8".to_owned());
                format!("node_commit {node}")
            }
        };
        assert_eq!(compared(&lock, "c1").0, [expected]);
    }

    #[test]
    fn an_unlocked_node_differs_when_a_call_reaches_it_and_is_named_when_none_does() {
        let mut lock = lock_of_the_stack();
        lock.nodes
            .retain(|name, _| name.name.as_str() == "workflow_director");
        let (found, unlocked) = compared(&lock, "c1");
        assert_eq!(found, ["unlocked_node test_brain:v1"]);
        assert_eq!(unlocked, [NameTag::parse("camera:v1").unwrap()]);
    }

    #[test]
    fn a_node_that_serves_only_a_fleet_entry_with_no_limbs_is_no_difference() {
        let mut lock = lock_of_the_stack();
        lock.nodes
            .retain(|name, _| name.name.as_str() != "test_brain");
        let (fleet, plan) = stack("c1", false);
        let nodes = run_nodes(&fleet, &plan, &me());
        assert!(nodes.iter().any(|node| node.serves_devices_only));
        let (found, _) = differences(&lock, &versions(&fleet, &plan, &nodes).unwrap(), &nodes);
        assert!(found.is_empty(), "{found:?}");
    }

    /// The lock of a run names locations:v1 of a robot whose skill it gives a location, alert:v1 of a physical
    /// robot and robot_identity:v1 of a robot whose role names a model, each at the pin of the listing.
    #[test]
    fn the_lock_names_each_contract_the_run_reads() {
        let mut value = one_arm(
            "arm",
            json!({ "id": "go", "timeout_s": 5, "call": { "member": "arm.go_to", "args": { "place": { "$location": "bin" } } } }),
            json!([]),
        );
        value["locations"] = json!(["bin"]);
        value["roles"]["arm"]["model"] = json!("test_model");
        let mut arm = robot("arm");
        for (name, sha256) in [("alert", SHA_A), ("robot_identity", SHA_B)] {
            arm.contracts.push(crate::fleet::RobotContract {
                name: name.to_owned(),
                tag: "v1".to_owned(),
                sha256: sha256.to_owned(),
                instances: vec!["brain".to_owned()],
            });
        }
        let mut fleet = snapshot(vec![arm]);
        fleet
            .models
            .insert("arm".to_owned(), "test_model".to_owned());
        let mut plan = plan_on(&value, Attendance::Attended, &fleet);
        plan.roles.get_mut(&name("arm")).unwrap().physical = true;
        let lock = versions(&fleet, &plan, &[]).unwrap();
        let pins: BTreeMap<String, String> = lock
            .contracts
            .iter()
            .map(|(contract, pin)| (contract.to_string(), pin.to_string()))
            .collect();
        assert_eq!(pins["locations:v1"], crate::testing::LOCATIONS_PIN);
        assert_eq!(pins["alert:v1"], SHA_A);
        assert_eq!(pins["robot_identity:v1"], SHA_B);
        assert!(pins.contains_key("test_arm:v1"), "{pins:?}");
    }

    /// A contract that the robots of a run pin at two shas refuses the run, naming the contract and both pins.
    #[test]
    fn a_contract_pinned_at_two_shas_refuses_the_run_naming_both() {
        let mut value = rest();
        value["roles"] = json!({ "left": {}, "right": {} });
        let tree = value["actions"]["arm"]["tree"].clone();
        value["actions"] = json!({ "left": { "tree": tree.clone() }, "right": { "tree": tree } });
        value["observations"]["target"]["role"] = json!("left");
        let robots = [("left", SHA_A), ("right", SHA_B)]
            .into_iter()
            .map(|(name, sha256)| {
                let mut arm = robot(name);
                arm.contracts.push(crate::fleet::RobotContract {
                    name: "alert".to_owned(),
                    tag: "v1".to_owned(),
                    sha256: sha256.to_owned(),
                    instances: vec!["brain".to_owned()],
                });
                arm
            })
            .collect();
        let fleet = snapshot(robots);
        let mut plan = plan_on(&value, Attendance::Attended, &fleet);
        plan.roles
            .values_mut()
            .for_each(|bound| bound.physical = true);
        assert_eq!(
            versions(&fleet, &plan, &[]).unwrap_err(),
            format!(
                "the robots of the run pin `alert:v1` at two shas, `{SHA_A}` and `{SHA_B}`: bind robots whose software pins it alike"
            )
        );
    }

    /// The plan of a run of `arm` that gives the location `bin` to `go_to`, on `arm`, and the snapshot of its
    /// fleet.
    fn going_to_the_bin(arm: crate::fleet::Robot) -> (Snapshot, Plan) {
        let mut value = one_arm(
            "arm",
            json!({ "id": "go", "timeout_s": 5, "call": { "member": "arm.go_to", "args": { "place": { "$location": "bin" } } } }),
            json!([]),
        );
        value["locations"] = json!(["bin"]);
        let fleet = snapshot(vec![arm]);
        let plan = plan_on(&value, Attendance::Attended, &fleet);
        (fleet, plan)
    }

    /// A robot that lists one contract the run reads at two pins, one per skill, refuses the run, naming the robot,
    /// the contract and both pins; one entry that two skills serve does not.
    #[test]
    fn a_robot_that_lists_a_contract_at_two_pins_refuses_the_run() {
        let mut arm = robot("arm");
        arm.instances.push(crate::fleet::Instance {
            name: "left_skill".to_owned(),
            instance: "arm_left_skill".to_owned(),
            core_node: "core".to_owned(),
        });
        let mut two_pins = arm.clone();
        two_pins.contracts.push(crate::fleet::RobotContract {
            name: "locations".to_owned(),
            tag: "v1".to_owned(),
            sha256: SHA_B.to_owned(),
            instances: vec!["left_skill".to_owned()],
        });
        let (fleet, plan) = going_to_the_bin(two_pins);
        assert_eq!(
            versions(&fleet, &plan, &[]).unwrap_err(),
            format!(
                "`arm` pins `locations:v1` at two shas, `{}` and `{SHA_B}`: run one build of it on that robot",
                crate::testing::LOCATIONS_PIN
            )
        );
        let mut one_pin = arm;
        one_pin
            .contracts
            .iter_mut()
            .filter(|contract| contract.name == "locations")
            .for_each(|contract| contract.instances.push("left_skill".to_owned()));
        let (fleet, plan) = going_to_the_bin(one_pin);
        assert!(versions(&fleet, &plan, &[]).is_ok());
    }

    /// The fleet of `arm`, whose instance `spare` runs another build of test_arm:v1, at `SHA_B`.
    fn arm_with_a_spare_build() -> Snapshot {
        let mut arm = robot("arm");
        arm.instances.push(crate::fleet::Instance {
            name: "spare".to_owned(),
            instance: "arm_spare".to_owned(),
            core_node: "core".to_owned(),
        });
        arm.contracts.push(crate::fleet::RobotContract {
            name: "test_arm".to_owned(),
            tag: "v1".to_owned(),
            sha256: SHA_B.to_owned(),
            instances: vec!["spare".to_owned()],
        });
        let mut fleet = snapshot(vec![arm]);
        let other_build =
            crate::contract::ContractShape::parse(crate::contract::tests::TEST_ARM, SHA_B).unwrap();
        fleet
            .contracts
            .insert(SHA_B.to_owned(), Arc::new(other_build));
        fleet
    }

    /// The plan of a run of `tree` on `fleet`, its observation `target` on `brain`.
    fn planned_on_brain(fleet: &Snapshot, tree: serde_json::Value) -> Plan {
        let mut value = one_arm("arm", tree, json!([]));
        value["observations"]["target"]["instance"] = json!("brain");
        plan_on(&value, Attendance::Attended, fleet)
    }

    /// A call of `arm.home` within 5 s, the step `id`, on the instance `instance`.
    fn home_on(id: &str, instance: &str) -> serde_json::Value {
        json!({ "id": id, "timeout_s": 5, "call": { "member": "arm.home", "args": { "duration_s": 1 }, "instance": instance } })
    }

    /// A robot that runs another build of a contract on an instance the run does not use passes, and a call to
    /// that instance binds the pin of its own build.
    #[test]
    fn a_second_build_on_an_instance_the_run_does_not_use_passes() {
        let fleet = arm_with_a_spare_build();
        let on = |instance: &str| planned_on_brain(&fleet, home_on("rest", instance));
        let used = on("brain");
        assert!(versions(&fleet, &used, &[]).is_ok());
        let spare = on("spare");
        let target = &spare.calls.values().next().unwrap().target;
        assert_eq!(
            (target.producer.instance.as_str(), target.sha256.as_str()),
            ("arm_spare", SHA_B)
        );
    }

    /// A run that calls one contract on two instances of one robot, at two pins, is refused naming that robot.
    #[test]
    fn a_run_that_reads_two_builds_of_one_robot_names_the_robot() {
        let fleet = arm_with_a_spare_build();
        let tree = json!({ "id": "both", "timeout_s": 10, "sequence": [home_on("one", "spare"), home_on("two", "brain")] });
        let plan = planned_on_brain(&fleet, tree);
        assert_eq!(
            versions(&fleet, &plan, &[]).unwrap_err(),
            format!(
                "`arm` pins `test_arm:v1` at two shas, `{SHA_B}` and `{}`: run one build of it on that robot",
                crate::testing::ARM_PIN
            )
        );
    }

    /// The location `name` of a skill.
    fn location(name: &str) -> KnownLocation {
        KnownLocation {
            name: Name::parse(name).unwrap(),
            kind: workflow_documents::LocationKind::Surface,
            position: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],
            size: [0.5, 0.5, 0.0],
        }
    }

    /// The hint of a refusal teaches the locations of every skill of the robot, each once, and the missing one; a
    /// skill whose locations are unknown gives no command and is named.
    #[test]
    fn the_teach_hint_keeps_the_locations_of_every_skill() {
        let hint = TeachHint {
            skill: "arm_brain",
            role: &name("arm"),
            director: DIRECTOR,
            robot: "arm",
        };
        let both = hint.refusal(
            &[
                ("arm_brain".to_owned(), Some(vec![location("shelf")])),
                (
                    "arm_left".to_owned(),
                    Some(vec![location("bin"), location("shelf")]),
                ),
            ],
            &name("crate"),
        );
        let teach = both
            .split("--set 'teach=")
            .nth(1)
            .and_then(|rest| rest.split("'`").next())
            .unwrap_or_else(|| panic!("{both}"));
        for kept in [r#""name":"shelf""#, r#""name":"bin""#, r#"name: "crate""#] {
            assert_eq!(teach.matches(kept).count(), 1, "{kept} in {teach}");
        }
        let unknown = hint.refusal(
            &[
                ("arm_brain".to_owned(), Some(vec![location("shelf")])),
                ("arm_left".to_owned(), None),
            ],
            &name("crate"),
        );
        assert!(!unknown.contains("--set 'teach="), "{unknown}");
        assert!(
            unknown.contains("the locations of `arm_left` are unknown"),
            "{unknown}"
        );
    }

    /// A pin of the listing that does not read refuses the run, naming the pin and its contract.
    #[test]
    fn a_pin_that_does_not_read_refuses_the_run_naming_it() {
        let mut plan = plan(&rest(), Attendance::Attended);
        let fleet = snapshot(vec![robot("arm")]);
        plan.calls
            .values_mut()
            .for_each(|call| call.target.sha256 = "not-a-pin".to_owned());
        let refused = versions(&fleet, &plan, &[]).unwrap_err();
        assert!(
            refused.starts_with(
                "the pin `not-a-pin` of `test_arm:v1` in the fleet listing does not read"
            ),
            "{refused}"
        );
    }
}
