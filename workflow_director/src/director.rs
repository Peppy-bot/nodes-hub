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
use workflow_documents::bundle::{Bundle, BundleFile};
use workflow_documents::lock::{Lock, LockedNode, UNKNOWN_PEPPY};
use workflow_documents::result::{
    Difference, DifferenceKind, Duration as RunDuration, Ending, RunRefusal, RunResult, RunWorkflow,
};
use workflow_documents::run::{RoleBinding, RunDocument, VersionPolicy};
use workflow_documents::{
    EventKind, KnownLocation, Located, Name, NameTag, RobotName, Run, Sha256, WorkCell, Workflow,
    run_lease_name,
};

use crate::alerts;
use crate::checker::{self, RunChoices};
use crate::clock::{self, ClockSource};
use crate::config::Config;
use crate::engine::host::{Event, Host};
use crate::engine::plan::Plan;
use crate::engine::{self, Phase, ResultBase, RunCommand};
use crate::fleet::{self, Fleet, Snapshot};
use crate::journal::{self, Journals, Record, wall_ns};
use crate::leases::{LeaseEventKind, Leases};
use crate::live::{LiveHost, RunClock};
use crate::replay::{Planned, Recorded, journal_command};

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
    pub started_by: String,
    pub caller_data: Option<String>,
}

impl RunRequest {
    /// Refuses a request that gives both a run document and roles, or
    /// neither or both of a workflow and a bundle.
    pub fn shape(&self) -> Result<(), String> {
        if self.run_document.is_some() && self.roles.is_some() {
            return Err("give a run_document or roles, not both".to_owned());
        }
        match (&self.workflow, &self.bundle) {
            (Some(_), Some(_)) => Err("give a workflow or a bundle, not both".to_owned()),
            (None, None) => Err("give a workflow or a bundle".to_owned()),
            _ => Ok(()),
        }
    }
}

/// A run the director keeps: its workflow once it parsed, who started it,
/// its robots once bound, where it stands and how to reach its engine.
pub struct RunEntry {
    pub workflow: Option<Arc<Workflow>>,
    pub started_by: String,
    pub robots: Vec<(Name, RobotName)>,
    pub phase: watch::Receiver<Phase>,
    pub commands: Option<mpsc::UnboundedSender<RunCommand>>,
}

/// What a member of the director reads of a run: where it stands, and the
/// receiver that follows it, how to reach its engine while it goes, and its
/// workflow once it parsed.
pub struct RunView {
    pub phase: Phase,
    pub phases: watch::Receiver<Phase>,
    pub commands: Option<mpsc::UnboundedSender<RunCommand>>,
    pub workflow: Option<Arc<Workflow>>,
}

/// The director.
pub struct Director {
    pub runner: Arc<NodeRunner>,
    pub config: Config,
    pub leases: Arc<Leases>,
    pub snapshots: watch::Receiver<Arc<Snapshot>>,
    pub journals: Journals,
    pub runs: Mutex<BTreeMap<String, RunEntry>>,
    /// The robots that a teach of their locations holds. Read under the
    /// lock of `runs`, so a run and a teach never hold one robot.
    pub teaching: Mutex<BTreeSet<RobotName>>,
    pub events: broadcast::Sender<(String, Event)>,
    pub clock: ClockSource,
}

/// What of this director holds a robot: another of its runs, or a teach of
/// its locations.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Holder {
    Run(String),
    Teach,
}

impl Holder {
    /// Why a run does not bind `robot`, which this holds, and what to do.
    fn why(&self, robot: &str) -> String {
        match self {
            Holder::Run(run) => format!(
                "`{robot}` takes part in run {run} of this director: wait for its end, or abort it"
            ),
            Holder::Teach => format!(
                "`{robot}` is being taught its locations by this director: run again once the teach ends"
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

/// What a run needs from its preparation.
struct Prepared {
    plan: Plan,
    base: ResultBase,
    taken: Vec<RobotName>,
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
const HALTED: &str = "the stack is halted; run `peppy fleet clear-halt`, then run again";

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
    fn me(&self) -> Caller {
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
            let entry = RunEntry {
                workflow,
                started_by,
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
            robots: Vec::new(),
            phase: phase_rx.clone(),
            commands: Some(commands_tx.clone()),
        };
        self.runs
            .lock()
            .expect("no run panics holding the runs")
            .insert(run.clone(), entry);
        let opened = json!({
            "workflow": request.workflow, "bundle": request.bundle, "run_document": request.run_document,
            "roles": request.roles, "started_by": request.started_by, "caller_data": request.caller_data,
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
        let stopper = self.follow_leases(run, &taken, commands_tx).await;
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

    /// Follows the lease events and the halt for the robots of `run`: the
    /// end of one of its leases, or a halt, stops the run; the end of a lease
    /// in a halt stops it halted. A fresh listing first gives each lease that
    /// ended between its take and this follow, and its stop reaches the
    /// commands before the engine starts.
    async fn follow_leases(
        &self,
        run: &str,
        taken: &[RobotName],
        commands: mpsc::UnboundedSender<RunCommand>,
    ) -> tokio::task::JoinHandle<()> {
        let mut events = self.leases.events();
        let mut snapshots = self.snapshots.clone();
        let runner = Arc::clone(&self.runner);
        let robots: BTreeSet<String> = taken.iter().map(|r| r.to_string()).collect();
        match fleet::read_fleet(&self.runner).await {
            Ok(fleet) => {
                if let Some(stop) = lost_leases(&fleet, &robots, &self.me()) {
                    let _ = commands.send(stop);
                }
            }
            Err(error) => {
                tracing::warn!("the fleet listing at the start of {run} did not come: {error}")
            }
        }
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    event = events.recv() => match event {
                        Ok(event) if event.kind == LeaseEventKind::LeaseEnd && robots.contains(&event.robot) => {
                            // The daemon ends every lease when it halts the stack, before the listing of the halt
                            // reaches the follower of the fleet: a fresh listing tells a halt from a lost lease.
                            let halted = fleet::read_fleet(&runner).await.is_ok_and(|fleet| fleet.halted);
                            let reason = match halted {
                                true => HALTED.to_owned(),
                                false => format!("the lease of {} ended: {}", event.robot, event.cause),
                            };
                            let _ = commands.send(RunCommand::Stop { halted, reason });
                        }
                        Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                        Err(broadcast::error::RecvError::Closed) => return,
                    },
                    changed = snapshots.changed() => {
                        if changed.is_err() {
                            return;
                        }
                        if snapshots.borrow().fleet.halted {
                            let _ = commands.send(RunCommand::Stop { halted: true, reason: HALTED.to_owned() });
                        }
                    }
                }
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
        caller: &Caller,
        me: &Caller,
    ) -> Result<BTreeMap<Name, RobotName>, Vec<Located>> {
        let mut runs = self.runs.lock().expect("no run panics holding the runs");
        let teaching = self
            .teaching
            .lock()
            .expect("no teach panics holding the robots it teaches")
            .clone();
        let held: BTreeMap<RobotName, Holder> = robots_of_runs(&runs, run)
            .into_iter()
            .map(|(robot, run)| (robot, Holder::Run(run)))
            .chain(teaching.into_iter().map(|robot| (robot, Holder::Teach)))
            .collect();
        let binding = bind(workflow, document, snapshot, caller, me, &held)?;
        if let Some(entry) = runs.get_mut(run) {
            entry.robots = binding
                .iter()
                .map(|(role, robot)| (role.clone(), robot.clone()))
                .collect();
        }
        Ok(binding)
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
        let (workflow, bundle) = match (&request.workflow, &request.bundle) {
            (Some(text), None) => match Workflow::parse(text) {
                Ok(workflow) => (workflow, None),
                Err(refusals) => {
                    return Err(refuse(
                        empty_result(),
                        "the workflow does not parse".to_owned(),
                        located(refusals),
                    ));
                }
            },
            (None, Some(files)) => match Bundle::open(files.clone()) {
                Ok(bundle) => (bundle.workflow.clone(), Some(bundle)),
                Err(refusals) => {
                    return Err(refuse(
                        empty_result(),
                        "the bundle does not open".to_owned(),
                        located(refusals),
                    ));
                }
            },
            _ => {
                return Err(refuse(
                    empty_result(),
                    "give a workflow or a bundle".to_owned(),
                    Vec::new(),
                ));
            }
        };
        let workflow = Arc::new(workflow);
        self.update(run, |entry| entry.workflow = Some(Arc::clone(&workflow)));
        let mut base = empty_result();
        base.workflow = Some(RunWorkflow::of(&workflow));
        base.caller_data = request
            .caller_data
            .as_deref()
            .and_then(|text| parses(text).ok());
        let document = match (&request.run_document, &request.roles) {
            (Some(text), None) => RunDocument::parse(text).map_err(|refusals| {
                refuse(
                    base.clone(),
                    "the run document does not parse".to_owned(),
                    located(refusals),
                )
            })?,
            (None, roles) => {
                let named = match roles.as_deref().map(named_roles).transpose() {
                    Ok(named) => named.unwrap_or_default(),
                    Err(reason) => {
                        return Err(refuse(base, format!("roles: {reason}"), Vec::new()));
                    }
                };
                RunDocument::of_roles(&workflow, named, WorkCell::new()).map_err(|refusals| {
                    refuse(
                        base.clone(),
                        "the roles do not fit the workflow".to_owned(),
                        located(refusals),
                    )
                })?
            }
            (Some(_), Some(_)) => {
                return Err(refuse(
                    base,
                    "give a run_document or roles, not both".to_owned(),
                    Vec::new(),
                ));
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
        let (known, known_models) = {
            let latest = self.snapshots.borrow();
            (latest.contracts.clone(), latest.models.clone())
        };
        let mut snapshot = fleet::snapshot_of(&self.runner, fleet, &known).await;
        snapshot.models = fleet::models_of(&self.runner, &snapshot, &known_models).await;
        base.core_node = snapshot.fleet.core_node.clone();
        let me = self.me();
        let binding = self
            .bind_and_reserve(run, &workflow, &run_document, &snapshot, caller, &me)
            .map_err(|errors| refuse(base.clone(), "the roles do not bind".to_owned(), errors))?;
        let choices = RunChoices {
            attendance: run_document.attendance,
            confirmed: run_document.confirmed.clone(),
            incidents: &run_document.incidents,
            poll: self.config.poll,
            cancel_timeout: self.config.cancel_timeout,
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
        let mut taken: Vec<RobotName> = Vec::new();
        for robot in binding.values() {
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
    if let Some(text) = opened["workflow"].as_str() {
        return Workflow::parse(text)
            .map_err(|refusals| format!("the workflow of the journal does not parse: {refusals}"));
    }
    let files: Vec<BundleFile> = serde_json::from_value(opened["bundle"].clone())
        .map_err(|e| format!("the bundle of the journal: {e}"))?;
    Bundle::open(files)
        .map(|bundle| bundle.workflow)
        .map_err(|refusals| format!("the bundle of the journal does not open: {refusals}"))
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

/// The roles of a run bound to robots: the named roles first, then each
/// `"auto"` role in name order to the first fitting robot in name order.
/// `held` gives each robot that another run or a teach of the director
/// holds.
fn bind(
    workflow: &Workflow,
    document: &RunDocument,
    snapshot: &Snapshot,
    caller: &Caller,
    me: &Caller,
    held: &BTreeMap<RobotName, Holder>,
) -> Result<BTreeMap<Name, RobotName>, Vec<Located>> {
    let fleet = &snapshot.fleet;
    let offered: BTreeSet<&str> = fleet
        .leases
        .iter()
        .filter(|lease| lease.receiver == me.instance && lease.receiver_core_node == me.core_node)
        .filter(|lease| {
            lease.holder == caller.instance && lease.holder_core_node == caller.core_node
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
    for (role, bound) in &document.roles {
        if *bound != RoleBinding::Auto {
            continue;
        }
        let taken: BTreeSet<&str> = binding.values().map(RobotName::as_str).collect();
        // The first reason `robot` cannot take the role, none when it can.
        let why_not = |robot: &fleet::Robot| -> Option<String> {
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
            checker::misfits(
                workflow,
                snapshot,
                role,
                robot,
                document.attendance,
                &document.confirmed,
            )
            .into_iter()
            .next()
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
struct RunNode {
    name: NameTag,
    locked: LockedNode,
    called: bool,
    serves_devices_only: bool,
}

/// The nodes a run uses: the director, the anomaly monitor and each node of
/// the copy of each robot it binds.
fn run_nodes(snapshot: &Snapshot, plan: &Plan, me: &Caller) -> Vec<RunNode> {
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
fn versions(snapshot: &Snapshot, plan: &Plan, nodes: &[RunNode]) -> Result<Lock, String> {
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
fn differences(lock: &Lock, stack: &Lock, nodes: &[RunNode]) -> (Vec<Difference>, Vec<NameTag>) {
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
            &chat(),
            &me(),
            &BTreeMap::new(),
        )
        .map_err(|errors| errors.into_iter().map(|e| e.message).collect())
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
            &chat(),
            &me(),
            &BTreeMap::new(),
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
            &chat(),
            &me(),
            &BTreeMap::new(),
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
            &chat(),
            &me(),
            &elsewhere,
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
                &chat(),
                &me(),
                &elsewhere
            )
            .is_err()
        );
        let taught = BTreeMap::from([(robot_name("alpha"), Holder::Teach)]);
        let teaching = bind(
            &workflow,
            &document(json!({ "arm": "alpha" })),
            &fleet,
            &chat(),
            &me(),
            &taught,
        );
        assert_eq!(
            teaching.unwrap_err()[0].message,
            "`alpha` is being taught its locations by this director: run again once the teach ends"
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
