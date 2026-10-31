//! The trials of the director: a trial (`trial/v1`) runs a bundle on the
//! physical robots of one stack, a person preparing and judging each run,
//! and signs off the bundle when the runs that count meet its thresholds.
//!
//! A trial takes the lease of each of its robots for its whole life, under
//! the name `trial <id>`, and opens them while the operator prepares the work
//! cell. Each run has four steps: the operator prepares the work cell and
//! calls `next_run`, which closes the leases and starts the run; the operator
//! judges each check of the run (`judge_checks`); the run runs unattended;
//! the operator judges its ending (`judge_ending`). The record of each trial
//! stays on the disk of the director, and a trial that the director did not
//! end when it stopped ends interrupted when it starts again.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use peppygen::exposed_actions::director::trial as trial_goal;
use serde::{Deserialize, Serialize};
use serde_json::{Value as Json, json};
use tokio::sync::{mpsc, oneshot, watch};
use workflow_documents::bundle::{Bundle, BundleFile};
use workflow_documents::result::{Difference, Ending, RunResult};
use workflow_documents::run::Attendance;
use workflow_documents::sign_off::Unmet;
use workflow_documents::{
    Measures, Name, RobotName, Sha256, Thresholds, Trial, TrialDocument, Workflow,
};

use crate::checker::{self, RunChoices};
use crate::director::{
    Caller, Director, HALTED, LeaseFollower, LeaseStop, RunRequest, differences, digest_of,
    run_nodes, trial_lease_name, versions,
};
use crate::engine::plan::Watch;
use crate::engine::{ChecksJudged, Phase, RunCommand};
use crate::journal::next_number;
use crate::sign_off::{SignOff, SignOffRecord};

/// How often a trial gives its progress, in wall time.
const PROGRESS_EVERY: Duration = Duration::from_secs(5);

/// The state of a trial.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialState {
    Running,
    SignedOff,
    NotSignedOff,
    Refused,
    Aborted,
    Halted,
    Interrupted,
}

impl TrialState {
    /// The word of the state, as `runs` and `status` give it.
    pub fn name(self) -> &'static str {
        match self {
            TrialState::Running => "running",
            TrialState::SignedOff => "signed_off",
            TrialState::NotSignedOff => "not_signed_off",
            TrialState::Refused => "refused",
            TrialState::Aborted => "aborted",
            TrialState::Halted => "halted",
            TrialState::Interrupted => "interrupted",
        }
    }

    /// The failure kind of a trial that ended in this state, with neither a
    /// sign-off nor an abort.
    fn failure_kind(self) -> Option<&'static str> {
        match self {
            TrialState::NotSignedOff => Some("trial_not_signed_off"),
            TrialState::Refused => Some("trial_refused"),
            TrialState::Halted => Some("trial_halted"),
            TrialState::Interrupted => Some("trial_interrupted"),
            TrialState::Running | TrialState::SignedOff | TrialState::Aborted => None,
        }
    }
}

/// Where a trial stands.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "step")]
pub enum TrialStep {
    /// The checks of the trial and the takes of its leases.
    Starting,
    /// The operator prepares the work cell for the run `number` that counts,
    /// and calls `next_run`.
    Prepares {
        number: u32,
    },
    /// The run `run` goes: it waits for the judgement of its checks, or it
    /// runs.
    Runs {
        run: String,
    },
    /// The run `run` ended, and waits for the judgement of its ending.
    JudgesEnding {
        run: String,
    },
    Ended,
}

/// The judgement of the operator on the ending of a run of a trial.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EndingJudged {
    pub task_done: bool,
    pub emergency_stop: bool,
    pub unsafe_events: Vec<String>,
}

impl EndingJudged {
    /// The unsafe events of the run: each press of the emergency stop and
    /// each other unsafe event.
    fn unsafe_events(&self) -> u32 {
        u32::from(self.emergency_stop) + self.unsafe_events.len() as u32
    }
}

/// How a run of a trial counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Count {
    /// The work cell was not ready: the run does not count.
    NotCounted,
    Completed,
    NotCompleted,
}

/// A run of a trial: its id, how it ended, the judgements of the operator
/// on its checks and on its ending, and how it counts once the trial
/// counted it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrialRun {
    pub run: String,
    pub ending: Option<Ending>,
    pub checks: Option<ChecksJudged>,
    pub judged_ending: Option<EndingJudged>,
    pub count: Option<Count>,
}

/// What a trial does once it counted a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Next {
    /// The operator prepares the next run.
    GoesOn,
    /// The runs that count meet the thresholds of the bundle.
    SignsOff,
    /// The runs that count do not meet the thresholds, or cannot meet them
    /// any more; the message says which.
    NotSignedOff(String),
}

/// What a trial does once its run ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AfterRun {
    /// The run started: the trial waits for the judgement of its ending.
    JudgesEnding,
    /// The trial counted the run as `count`, and does `next`.
    Counted { count: Count, next: Next },
    /// The run ended halted or interrupted, and so does the trial.
    Ends { state: TrialState, message: String },
}

/// The record of a trial, which stays on the disk of the director: its
/// document, its bundle, who started it, where it stands, why it ended,
/// the core node of its stack, its runs, the measures over the runs that
/// count, and its sign-off record when it signed off.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrialRecord {
    pub trial: String,
    pub document: TrialDocument,
    /// The digest of the bundle, once it opened.
    pub bundle: Option<Sha256>,
    pub started_by: String,
    pub state: TrialState,
    pub step: TrialStep,
    /// Why the trial ended, for each end but a sign-off.
    pub message: String,
    pub core_node: String,
    pub runs: Vec<TrialRun>,
    pub measures: Measures,
    pub sign_off: Option<SignOffRecord>,
}

impl TrialRecord {
    pub fn new(trial: &str, document: TrialDocument, started_by: String) -> TrialRecord {
        TrialRecord {
            trial: trial.to_owned(),
            document,
            bundle: None,
            started_by,
            state: TrialState::Running,
            step: TrialStep::Starting,
            message: String::new(),
            core_node: String::new(),
            runs: Vec::new(),
            measures: Measures::default(),
            sign_off: None,
        }
    }

    /// The number of the run that counts next.
    fn number(&self) -> u32 {
        self.measures.runs + 1
    }

    /// The run of the trial that goes, or that waits for the judgement of
    /// its ending.
    pub fn current_run(&self) -> Option<&str> {
        match &self.step {
            TrialStep::Runs { run } | TrialStep::JudgesEnding { run } => Some(run),
            TrialStep::Starting | TrialStep::Prepares { .. } | TrialStep::Ended => None,
        }
    }

    /// Forgets the judgement of the checks of the run `run`, which did not
    /// reach the run.
    pub fn unjudge_checks(&mut self, run: &str) {
        if let Some(last) = self.runs.last_mut().filter(|last| last.run == run) {
            last.checks = None;
        }
    }

    /// Keeps the judgement of the checks of the run `run`, which waits for
    /// it; refused once its checks are judged.
    pub fn judge_checks(&mut self, run: &str, judged: &ChecksJudged) -> Result<(), String> {
        let waits = self.current_run() == Some(run);
        let Some(last) = self.runs.last_mut().filter(|last| waits && last.run == run) else {
            return Err(format!(
                "no run of the trial `{}` waits for the judgement of its checks",
                self.trial
            ));
        };
        if last.checks.is_some() {
            return Err(format!("the checks of the run `{run}` are judged already"));
        }
        last.checks = Some(judged.clone());
        Ok(())
    }

    /// Takes the result of the run that goes: the trial waits for the
    /// judgement of the ending of a run that started, counts every other
    /// run (not at all when its work cell was not ready, by the judgement of
    /// the operator or by a watch value outside the envelope), and ends with
    /// a run that ended halted or interrupted.
    pub fn run_ended(&mut self, result: &RunResult, thresholds: &Thresholds) -> AfterRun {
        let Some(last) = self.runs.last_mut() else {
            return AfterRun::Ends {
                state: TrialState::Interrupted,
                message: "the trial has no run that goes".to_owned(),
            };
        };
        let ending = result.ending;
        last.ending = Some(ending);
        let run = last.run.clone();
        match ending {
            Ending::Halted => {
                return AfterRun::Ends {
                    state: TrialState::Halted,
                    message: HALTED.to_owned(),
                };
            }
            Ending::Interrupted => {
                let why = match why_it_ended(result) {
                    Some(message) => format!(": {message}"),
                    None => String::new(),
                };
                return AfterRun::Ends {
                    state: TrialState::Interrupted,
                    message: format!("the run `{run}` of the trial ended interrupted{why}"),
                };
            }
            _ => {}
        }
        let (count, measures) = match &last.checks {
            Some(ChecksJudged::Starts) => {
                self.step = TrialStep::JudgesEnding { run };
                return AfterRun::JudgesEnding;
            }
            Some(ChecksJudged::NotReady) => (Count::NotCounted, self.measures),
            Some(ChecksJudged::FalseCheck { wrong }) => (
                Count::NotCompleted,
                Measures {
                    runs: self.measures.runs + 1,
                    false_assumption_checks: self.measures.false_assumption_checks
                        + wrong.len() as u32,
                    ..self.measures
                },
            ),
            None if refused_on_the_watch(result) => (Count::NotCounted, self.measures),
            None => (
                Count::NotCompleted,
                Measures {
                    runs: self.measures.runs + 1,
                    ..self.measures
                },
            ),
        };
        last.count = Some(count);
        self.measures = measures;
        let next = self.next(thresholds);
        AfterRun::Counted { count, next }
    }

    /// Counts the run that waits for the judgement of its ending: completed
    /// when it completed, a false success when it completed and its task is
    /// not done, and each emergency stop and each other unsafe event as an
    /// unsafe event.
    pub fn judge_ending(
        &mut self,
        judged: EndingJudged,
        thresholds: &Thresholds,
    ) -> Result<(Count, Next), String> {
        let TrialStep::JudgesEnding { run } = &self.step else {
            return Err(format!(
                "no ending of a run of the trial `{}` waits for its judgement",
                self.trial
            ));
        };
        let Some(last) = self.runs.last_mut().filter(|last| last.run == *run) else {
            return Err(format!("the trial `{}` has no run `{run}`", self.trial));
        };
        let completed = last.ending == Some(Ending::Completed);
        let count = match completed {
            true => Count::Completed,
            false => Count::NotCompleted,
        };
        self.measures = Measures {
            runs: self.measures.runs + 1,
            completed: self.measures.completed + u32::from(completed),
            false_successes: self.measures.false_successes
                + u32::from(completed && !judged.task_done),
            unsafe_events: self.measures.unsafe_events + judged.unsafe_events(),
            false_assumption_checks: self.measures.false_assumption_checks,
        };
        last.judged_ending = Some(judged);
        last.count = Some(count);
        Ok((count, self.next(thresholds)))
    }

    /// What the trial does after the runs that count so far: it ends after
    /// its last run that counts, signing off only when the runs meet the
    /// thresholds, or once the runs left cannot meet them; else the
    /// operator prepares the next run. A trial that ends waits for no other
    /// judgement.
    fn next(&mut self, thresholds: &Thresholds) -> Next {
        let measures = &self.measures;
        let remaining = self.document.runs.saturating_sub(measures.runs);
        let unmet = thresholds.unmet_by(measures);
        let next = if remaining == 0 {
            match unmet.is_empty() {
                true => Next::SignsOff,
                false => Next::NotSignedOff(format!(
                    "the {} runs that count do not meet the thresholds of the bundle: {}",
                    measures.runs,
                    unmet_words(&unmet)
                )),
            }
        } else if !thresholds.can_still_be_met(measures, remaining) {
            Next::NotSignedOff(format!(
                "the {remaining} runs left cannot meet the thresholds of the bundle any more: {}",
                unmet_words(&unmet)
            ))
        } else {
            Next::GoesOn
        };
        self.step = match next {
            Next::GoesOn => TrialStep::Prepares {
                number: self.number(),
            },
            Next::SignsOff | Next::NotSignedOff(_) => TrialStep::Ended,
        };
        next
    }
}

/// Why the run of `result` ended, as its refusal or its failed step says.
fn why_it_ended(result: &RunResult) -> Option<String> {
    result
        .refusal
        .as_ref()
        .map(|refusal| refusal.message.clone())
        .or_else(|| result.failed.as_ref().map(|failed| failed.message.clone()))
        .filter(|message| !message.is_empty())
}

/// Whether the director refused the run of `result` on a watch value
/// outside the envelope: the work cell is not ready. A run of a trial is
/// unattended on physical robots, so the director refuses it on each watch
/// value that no test covered, before it evaluates any check.
fn refused_on_the_watch(result: &RunResult) -> bool {
    result.ending == Ending::Refused
        && result
            .uncovered
            .iter()
            .any(|uncovered| result.watch.contains_key(&uncovered.parameter))
}

/// The names of the thresholds `unmet`, as the bundle writes them.
fn unmet_words(unmet: &[Unmet]) -> String {
    let words: Vec<String> = unmet
        .iter()
        .map(|unmet| {
            serde_json::to_value(unmet)
                .ok()
                .and_then(|word| word.as_str().map(|word| format!("`{word}`")))
                .unwrap_or_default()
        })
        .collect();
    words.join(", ")
}

/// A difference of the stack from the lock of a bundle, in words.
fn difference_text(difference: &Difference) -> String {
    let side = |value: &Option<String>| value.clone().unwrap_or_else(|| "none".to_owned());
    let what = match difference.subject.is_empty() {
        true => difference.kind.to_string(),
        false => format!("{} {}", difference.kind, difference.subject),
    };
    format!(
        "{what}: locked {}, on the stack {}",
        side(&difference.locked),
        side(&difference.stack)
    )
}

/// A run that a trial started: its id, and where it stands.
pub type StartedRun = (String, watch::Receiver<Phase>);

/// A command to the task of a running trial.
pub enum TrialCommand {
    /// Starts the next run, for the caller of `next_run`: the reply gives
    /// the run and where it stands, or the kind and the message of the
    /// failure of the goal.
    NextRun {
        caller: Caller,
        reply: oneshot::Sender<Result<StartedRun, (&'static str, String)>>,
    },
    /// The operator judged the ending of the run that waited for it, which
    /// counts as `count`; the trial does `next`.
    EndingJudged { count: Count, next: Next },
}

/// What a running trial gives the members that judge its runs: its
/// workflow, the thresholds of its bundle, and its task.
#[derive(Clone)]
pub struct LiveTrial {
    pub workflow: Arc<Workflow>,
    pub thresholds: Thresholds,
    pub commands: mpsc::UnboundedSender<TrialCommand>,
}

/// A trial the director keeps: its record, what judges it while it runs,
/// and whether it holds its robots: from the check of its admission that
/// no other run, teach or trial holds them, to its end.
pub struct TrialEntry {
    pub record: Arc<watch::Sender<TrialRecord>>,
    pub live: Option<LiveTrial>,
    pub holds_robots: bool,
}

/// The robots that the trials of `trials` other than `trial` hold, each with
/// the id of its trial: those of each running trial past the check of its
/// admission. A trial in its other checks holds no robot, so it refuses
/// neither another trial nor a run.
pub fn robots_of_trials(
    trials: &BTreeMap<String, TrialEntry>,
    trial: Option<&str>,
) -> Vec<(RobotName, String)> {
    trials
        .iter()
        .filter(|(id, entry)| {
            Some(id.as_str()) != trial
                && entry.holds_robots
                && entry.record.borrow().state == TrialState::Running
        })
        .flat_map(|(id, entry)| {
            let robots: Vec<RobotName> = entry
                .record
                .borrow()
                .document
                .roles
                .values()
                .cloned()
                .collect();
            robots.into_iter().map(move |robot| (robot, id.clone()))
        })
        .collect()
}

/// The records of the trials on the disk of the director, each in its file
/// `<journal_root>/workflow_director/trials/<trial>.json`.
pub struct Trials {
    dir: PathBuf,
}

/// The file that keeps the next trial number, so ids grow.
const NEXT_TRIAL_FILE: &str = "next_trial";

/// The number of the trial id `t<N>`.
pub fn trial_number(trial: &str) -> Option<u64> {
    trial.strip_prefix('t')?.parse().ok()
}

impl Trials {
    /// Opens the records under `root` and reads each: a trial that its
    /// record gives as running, which the director did not end when it
    /// stopped, ends interrupted. A record that does not read refuses the
    /// start, naming its file.
    pub fn open(root: &Path) -> std::io::Result<(Trials, Vec<TrialRecord>)> {
        let dir = root.join("workflow_director").join("trials");
        fs::create_dir_all(&dir)?;
        let trials = Trials { dir };
        let mut records = Vec::new();
        for entry in fs::read_dir(&trials.dir)? {
            let path = entry?.path();
            if path.extension().is_none_or(|extension| extension != "json") {
                continue;
            }
            let text = fs::read_to_string(&path)?;
            let mut record: TrialRecord = serde_json::from_str(&text).map_err(|error| {
                std::io::Error::other(format!(
                    "{}: {error}; move this trial record out of its directory, and the director starts without it",
                    path.display()
                ))
            })?;
            if record.state == TrialState::Running {
                record.state = TrialState::Interrupted;
                record.step = TrialStep::Ended;
                record.message = "the director stopped during the trial".to_owned();
                trials.keep(&record)?;
            }
            records.push(record);
        }
        Ok((trials, records))
    }

    /// The next trial id, `t<N>`, one more than every id given before.
    pub fn next_trial(&self) -> std::io::Result<String> {
        let mut on_disk = 0;
        for entry in fs::read_dir(&self.dir)? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            if let Some(number) = name.strip_suffix(".json").and_then(trial_number) {
                on_disk = on_disk.max(number);
            }
        }
        let next = next_number(&self.dir.join(NEXT_TRIAL_FILE), on_disk)?;
        Ok(format!("t{next}"))
    }

    /// Keeps `record` in its file.
    pub fn keep(&self, record: &TrialRecord) -> std::io::Result<()> {
        let text = serde_json::to_string_pretty(record).map_err(std::io::Error::other)?;
        let path = self.dir.join(format!("{}.json", record.trial));
        let written = path.with_extension("json.part");
        fs::write(&written, text)?;
        fs::rename(&written, &path)
    }
}

/// A trial admitted to its runs: its bundle, its workflow, the sign-off it
/// makes when it signs off, and the robots whose leases it holds.
struct Admitted {
    files: Vec<BundleFile>,
    trial: Trial,
    workflow: Arc<Workflow>,
    thresholds: Thresholds,
    sign_off: SignOff,
    robots: Vec<RobotName>,
}

/// How a trial ended: its state, why, and its sign-off record when it
/// signed off.
struct TrialEnd {
    state: TrialState,
    message: String,
    sign_off: Option<SignOffRecord>,
}

impl TrialEnd {
    fn new(state: TrialState, message: impl Into<String>) -> TrialEnd {
        TrialEnd {
            state,
            message: message.into(),
            sign_off: None,
        }
    }
}

impl Director {
    /// Keeps the record of a trial, logging a record that did not reach
    /// the disk.
    fn keep_trial(&self, record: &TrialRecord) {
        if let Err(error) = self.trial_records.keep(record) {
            tracing::error!("the record of {} was not kept: {error}", record.trial);
        }
    }

    /// Brings back the trials of the records, as they ended.
    pub fn restore_trials(&self, records: Vec<TrialRecord>) {
        let mut trials = self
            .trials
            .lock()
            .expect("no trial panics holding the trials");
        for record in records {
            let entry = TrialEntry {
                record: Arc::new(watch::Sender::new(record.clone())),
                live: None,
                holds_robots: false,
            };
            trials.insert(record.trial, entry);
        }
    }

    /// The report of a trial: its record, the trial document, whether it
    /// signs off, and the journal of each of its runs.
    pub fn trial_report(&self, record: &TrialRecord) -> Json {
        let mut report = json!(record);
        report["document"] = record.document.document().tree().clone();
        report["signs_off"] = json!(record.state == TrialState::SignedOff);
        for (index, run) in record.runs.iter().enumerate() {
            let journal = self.journals.read(&run.run).unwrap_or_default();
            report["runs"][index]["journal"] = json!(journal);
        }
        report
    }

    /// The record and what judges the trial `trial`, while the director
    /// keeps it.
    pub fn trial_entry(
        &self,
        trial: &str,
    ) -> Option<(Arc<watch::Sender<TrialRecord>>, Option<LiveTrial>)> {
        let trials = self
            .trials
            .lock()
            .expect("no trial panics holding the trials");
        let entry = trials.get(trial)?;
        Some((Arc::clone(&entry.record), entry.live.clone()))
    }

    fn set_live(&self, trial: &str, live: Option<LiveTrial>) {
        let mut trials = self
            .trials
            .lock()
            .expect("no trial panics holding the trials");
        if let Some(entry) = trials.get_mut(trial) {
            entry.live = live;
        }
    }

    /// Runs the trial of the goal `goal`, which its decider admitted, to its
    /// end.
    pub async fn run_trial(self: Arc<Self>, goal: trial_goal::GoalContext) {
        let data = goal.request().data.clone();
        let document = match TrialDocument::parse(&data.trial) {
            Ok(document) => document,
            Err(refusals) => {
                let _ = goal
                    .fail(
                        trial_goal::FailureKind::Rejected,
                        format!("trial: {refusals}"),
                    )
                    .await;
                return;
            }
        };
        let trial = match self.trial_records.next_trial() {
            Ok(trial) => trial,
            Err(error) => {
                let _ = goal
                    .fail(
                        trial_goal::FailureKind::AttemptFailed,
                        format!("the record of the trial did not open: {error}"),
                    )
                    .await;
                return;
            }
        };
        let sender = goal.sender();
        let started_by = data
            .started_by
            .clone()
            .unwrap_or_else(|| format!("{}@{}", sender.instance_id, sender.core_node));
        let record = Arc::new(watch::Sender::new(TrialRecord::new(
            &trial,
            document.clone(),
            started_by,
        )));
        self.trials
            .lock()
            .expect("no trial panics holding the trials")
            .insert(
                trial.clone(),
                TrialEntry {
                    record: Arc::clone(&record),
                    live: None,
                    holds_robots: false,
                },
            );
        self.keep_trial(&record.borrow());
        let tree = document.document().tree().clone();
        feedback(&goal, &trial, "trial_starts", "", String::new(), tree).await;
        let files: Vec<BundleFile> = data
            .bundle
            .into_iter()
            .map(|file| BundleFile {
                path: file.path,
                content: file.content,
            })
            .collect();
        // The follower subscribes before the takes, so it sees the end of
        // each lease the trial takes, and runs on its own, so no end is lost
        // while the trial does something else.
        let mut follower = LeaseFollower {
            events: self.leases.events(),
            snapshots: self.snapshots.clone(),
            runner: Arc::clone(&self.runner),
            robots: document
                .roles
                .values()
                .map(|robot| robot.to_string())
                .collect(),
        };
        let (stops_tx, mut stops) = mpsc::unbounded_channel();
        let following = tokio::spawn(async move {
            while let Some(stop) = follower.next_stop().await {
                if stops_tx.send(stop).is_err() {
                    return;
                }
            }
        });
        let end = match self.admit(&trial, document, files, &record).await {
            Err(message) => TrialEnd::new(TrialState::Refused, message),
            Ok(admitted) => {
                let (commands_tx, commands) = mpsc::unbounded_channel();
                let live = LiveTrial {
                    workflow: Arc::clone(&admitted.workflow),
                    thresholds: admitted.thresholds.clone(),
                    commands: commands_tx,
                };
                self.set_live(&trial, Some(live));
                let end = self
                    .trial_runs(&trial, &goal, &record, &admitted, commands, &mut stops)
                    .await;
                following.abort();
                for robot in &admitted.robots {
                    self.leases.release(robot.as_str()).await;
                }
                match end.state {
                    TrialState::SignedOff => self.sign_off(&trial, admitted.sign_off),
                    _ => end,
                }
            }
        };
        following.abort();
        self.set_live(&trial, None);
        record.send_modify(|record| {
            record.state = end.state;
            record.step = TrialStep::Ended;
            record.message = end.message.clone();
            record.sign_off = end.sign_off.clone();
        });
        let ended = record.borrow().clone();
        self.keep_trial(&ended);
        let report = self.trial_report(&ended);
        let text = report.to_string();
        feedback(&goal, &trial, "trial_ends", "", end.message.clone(), report).await;
        let finished = match (end.state, end.state.failure_kind()) {
            (TrialState::SignedOff, _) => goal.complete(trial.clone(), text).await,
            (TrialState::Aborted, _) => {
                goal.complete_cancelled("aborted", trial.clone(), text)
                    .await
            }
            (_, kind) => {
                let kind = kind
                    .and_then(trial_goal::FailureKind::parse)
                    .unwrap_or(trial_goal::FailureKind::AttemptFailed);
                goal.fail(kind, end.message).await
            }
        };
        if let Err(error) = finished {
            tracing::warn!("the goal of {trial}: {error}");
        }
    }

    /// Keeps the sign-off record of the trial `trial`; a record that does
    /// not reach the disk signs off nothing.
    fn sign_off(&self, trial: &str, sign_off: SignOff) -> TrialEnd {
        let record = SignOffRecord {
            trial: trial.to_owned(),
            sign_off,
        };
        match self.sign_offs.keep(record.clone()) {
            Ok(()) => TrialEnd {
                sign_off: Some(record),
                ..TrialEnd::new(TrialState::SignedOff, "")
            },
            Err(error) => TrialEnd::new(
                TrialState::NotSignedOff,
                format!(
                    "the runs met the thresholds, and the sign-off record was not kept: {error}"
                ),
            ),
        }
    }

    /// The checks of a trial before its first run, then the takes of its
    /// leases: the stack is not halted, the bundle opens and has an
    /// envelope, the work cell is inside it, each assumption has a check,
    /// each bound robot is physical, the workflow holds for the robots, and
    /// the stack matches the lock of the bundle with no version policy.
    /// Each refusal names its causes. Once the leases are taken, the trial
    /// removes each sign-off record of its bundle and binding.
    async fn admit(
        &self,
        trial: &str,
        document: TrialDocument,
        files: Vec<BundleFile>,
        record: &watch::Sender<TrialRecord>,
    ) -> Result<Admitted, String> {
        let fleet = crate::fleet::read_fleet(&self.runner)
            .await
            .map_err(|error| format!("the fleet listing did not come: {error}"))?;
        if fleet.halted {
            return Err(HALTED.to_owned());
        }
        let bundle = Bundle::open(files.clone())
            .map_err(|refusals| format!("the bundle does not open: {refusals}"))?;
        let digest = digest_of(&bundle);
        record.send_modify(|record| {
            record.bundle = Some(digest.clone());
            record.core_node = fleet.core_node.clone();
        });
        let (fits, mut causes) = match Trial::new(&bundle, document.clone()) {
            Ok(fits) => (Some(fits), Vec::new()),
            Err(refusals) => (None, refusals.iter().map(ToString::to_string).collect()),
        };
        for (role, robot) in &document.roles {
            match fleet.robots.get(robot.as_str()) {
                None => causes.push(format!("roles.{role}: the fleet lists no robot `{robot}`")),
                Some(listed) if !listed.is_physical() => causes.push(format!(
                    "roles.{role}: `{robot}` is simulated; a trial binds only physical robots"
                )),
                Some(_) => {}
            }
        }
        if !causes.is_empty() {
            return Err(format!("the trial is refused: {}", causes.join("; ")));
        }
        // `Trial::new` refuses a bundle with no envelope, so a trial that it
        // admits has both.
        let (Some(trial_of_bundle), Some(validation)) = (fits, bundle.document.validation.clone())
        else {
            return Err("the trial is refused: the bundle has no envelope".to_owned());
        };
        let snapshot = self.snapshot_of_listing(fleet).await;
        let workflow = Arc::new(bundle.workflow.clone());
        let run_document = trial_of_bundle.run_document();
        let choices = RunChoices {
            attendance: Attendance::Unattended,
            confirmed: BTreeSet::new(),
            incidents: &run_document.incidents,
            poll: self.config.poll,
            cancel_timeout: self.config.cancel_timeout,
            watch: Watch::of(&run_document, Some(&validation)),
            of_a_trial: true,
        };
        let binding: BTreeMap<Name, RobotName> = document.roles.clone();
        let plan = checker::plan(
            Arc::clone(&workflow),
            &snapshot,
            &self.config.allowed,
            &binding,
            choices,
        )
        .map_err(|errors| {
            let errors: Vec<String> = errors.iter().map(ToString::to_string).collect();
            format!(
                "the workflow does not hold for these robots: {}",
                errors.join("; ")
            )
        })?;
        let nodes = run_nodes(&snapshot, &plan, &self.me());
        let stack = versions(&snapshot, &plan, &nodes)?;
        let (found, _) = differences(&bundle.document.lock, &stack, &nodes);
        if !found.is_empty() {
            let found: Vec<String> = found.iter().map(difference_text).collect();
            return Err(format!(
                "the stack differs from the lock of the bundle in {} ways, and a trial takes no version policy: {}",
                found.len(),
                found.join("; ")
            ));
        }
        let sign_off = SignOff::of_run(
            digest.clone(),
            &snapshot,
            &binding,
            &stack,
            &document.incidents,
        );
        self.hold_robots(trial, &binding)?;
        let mut taken: Vec<RobotName> = Vec::new();
        for robot in binding.values() {
            if let Err(refused) = self
                .leases
                .take(robot.as_str(), &trial_lease_name(trial))
                .await
            {
                for robot in &taken {
                    self.leases.release(robot.as_str()).await;
                }
                return Err(match refused.halted {
                    true => HALTED.to_owned(),
                    false => format!(
                        "the lease of `{robot}` was refused ({}): {}",
                        refused.kind, refused.message
                    ),
                });
            }
            taken.push(robot.clone());
        }
        if let Err(error) = self.sign_offs.forget_binding(&digest, &binding) {
            tracing::error!(
                "the sign-off records of the bundle of {trial} were not removed: {error}"
            );
        }
        Ok(Admitted {
            files,
            trial: trial_of_bundle,
            workflow,
            thresholds: validation.thresholds,
            sign_off,
            robots: taken,
        })
    }

    /// Refuses the trial `trial` when another run, teach or trial holds a
    /// robot of `binding`, naming each; else the trial holds its robots
    /// from now on. The check and the hold are under the lock of the runs,
    /// so two trials, or a trial and a run, never hold one robot.
    fn hold_robots(&self, trial: &str, binding: &BTreeMap<Name, RobotName>) -> Result<(), String> {
        let runs = self.runs.lock().expect("no run panics holding the runs");
        let held = self.holders(&runs, "", Some(trial));
        let holders: Vec<String> = binding
            .values()
            .filter_map(|robot| held.get(robot).map(|holder| holder.why(robot.as_str())))
            .collect();
        if !holders.is_empty() {
            return Err(format!("the trial is refused: {}", holders.join("; ")));
        }
        let mut trials = self
            .trials
            .lock()
            .expect("no trial panics holding the trials");
        if let Some(entry) = trials.get_mut(trial) {
            entry.holds_robots = true;
        }
        Ok(())
    }

    /// Opens the leases of the trial for a take-over while the operator
    /// prepares the work cell, and gives `prepares`.
    async fn prepares(
        &self,
        trial: &str,
        goal: &trial_goal::GoalContext,
        record: &watch::Sender<TrialRecord>,
        robots: &[RobotName],
    ) {
        for robot in robots {
            if let Err(error) = self.leases.set_open(robot.as_str(), true).await {
                tracing::warn!("the lease of {robot} did not open for {trial}: {error}");
            }
        }
        let (number, measures) = {
            let record = record.borrow();
            (record.number(), record.measures)
        };
        self.keep_trial(&record.borrow());
        let details = json!({ "number": number, "measures": measures });
        let message = format!("prepare the work cell for run {number}, then call next_run");
        feedback(goal, trial, "prepares", "", message, details).await;
    }

    /// Closes the leases of the trial and starts its next run for `caller`;
    /// a lease that does not close, or a run that does not start, opens the
    /// leases again, so the operator can prepare the work cell, and starts
    /// no run.
    async fn next_run(
        self: &Arc<Self>,
        trial: &str,
        record: &watch::Sender<TrialRecord>,
        admitted: &Admitted,
        caller: Caller,
    ) -> Result<StartedRun, (&'static str, String)> {
        let precondition = "precondition_failed";
        if !matches!(record.borrow().step, TrialStep::Prepares { .. }) {
            return Err((
                precondition,
                format!("the trial `{trial}` does not wait for its next run"),
            ));
        }
        let mut closed: Vec<&RobotName> = Vec::new();
        for robot in &admitted.robots {
            if let Err(error) = self.leases.set_open(robot.as_str(), false).await {
                self.open_again(trial, &closed).await;
                return Err((
                    precondition,
                    format!(
                        "the lease of `{robot}` did not close: {error}; prepare the work cell, then call next_run again"
                    ),
                ));
            }
            closed.push(robot);
        }
        let request = RunRequest {
            workflow: None,
            bundle: Some(admitted.files.clone()),
            run_document: Some(admitted.trial.run_document().document().to_json5()),
            roles: None,
            work_cell: None,
            started_by: trial.to_owned(),
            caller_data: None,
            trial: Some(trial.to_owned()),
        };
        let (run, phase) = match self.start(request, caller) {
            Ok(started) => started,
            Err(message) => {
                self.open_again(trial, &closed).await;
                return Err((
                    "attempt_failed",
                    format!("{message}; prepare the work cell, then call next_run again"),
                ));
            }
        };
        record.send_modify(|record| {
            record.runs.push(TrialRun {
                run: run.clone(),
                ending: None,
                checks: None,
                judged_ending: None,
                count: None,
            });
            record.step = TrialStep::Runs { run: run.clone() };
        });
        self.keep_trial(&record.borrow());
        Ok((run, phase))
    }

    /// Opens the leases of `robots` again for a take-over, after a next run
    /// of the trial `trial` that did not start.
    async fn open_again(&self, trial: &str, robots: &[&RobotName]) {
        for robot in robots {
            if let Err(error) = self.leases.set_open(robot.as_str(), true).await {
                tracing::warn!("the lease of {robot} did not open again for {trial}: {error}");
            }
        }
    }

    /// The runs of an admitted trial, to the end of the trial: the operator
    /// prepares each and calls `next_run`, judges its checks and its ending,
    /// and the trial counts it, until it signs off or ends.
    async fn trial_runs(
        self: &Arc<Self>,
        trial: &str,
        goal: &trial_goal::GoalContext,
        record: &watch::Sender<TrialRecord>,
        admitted: &Admitted,
        mut commands: mpsc::UnboundedReceiver<TrialCommand>,
        stops: &mut mpsc::UnboundedReceiver<LeaseStop>,
    ) -> TrialEnd {
        record.send_modify(|record| {
            record.step = TrialStep::Prepares {
                number: record.number(),
            }
        });
        self.prepares(trial, goal, record, &admitted.robots).await;
        let mut progress =
            tokio::time::interval_at(tokio::time::Instant::now() + PROGRESS_EVERY, PROGRESS_EVERY);
        // The run that goes, and where it stands.
        let mut current: Option<watch::Receiver<Phase>> = None;
        let mut aborted = false;
        loop {
            tokio::select! {
                command = commands.recv() => match command {
                    Some(TrialCommand::NextRun { caller, reply }) => {
                        let started = self.next_run(trial, record, admitted, caller).await;
                        if let Ok((_, phase)) = &started {
                            current = Some(phase.clone());
                        }
                        let _ = reply.send(started);
                    }
                    Some(TrialCommand::EndingJudged { count, next }) => {
                        if let Some(end) = self.counted(trial, goal, record, admitted, count, next).await {
                            return end;
                        }
                    }
                    None => return TrialEnd::new(TrialState::Interrupted, "the director stopped during the trial"),
                },
                result = ended(&mut current) => {
                    current = None;
                    if aborted {
                        return TrialEnd::new(TrialState::Aborted, "the caller of the trial aborted it");
                    }
                    let after = {
                        let mut after = None;
                        record.send_modify(|record| {
                            after = Some(record.run_ended(&result, &admitted.thresholds));
                        });
                        after.expect("the record took the end of the run")
                    };
                    self.keep_trial(&record.borrow());
                    match after {
                        AfterRun::JudgesEnding => {
                            let run = record.borrow().current_run().unwrap_or_default().to_owned();
                            let message = format!("judge the ending of the run `{run}` with judge_ending");
                            let details = json!({ "measures": record.borrow().measures });
                            feedback(goal, trial, "progress", &run, message, details).await;
                        }
                        AfterRun::Counted { count, next } => {
                            if let Some(end) = self.counted(trial, goal, record, admitted, count, next).await {
                                return end;
                            }
                        }
                        AfterRun::Ends { state, message } => return TrialEnd::new(state, message),
                    }
                },
                _ = goal.cancel_signal(), if !aborted => {
                    aborted = true;
                    let run = record.borrow().current_run().map(str::to_owned);
                    let commands = run.and_then(|run| self.view_of(&run)).and_then(|view| view.commands);
                    match (&current, commands) {
                        (Some(_), Some(commands)) => {
                            let _ = commands.send(RunCommand::Abort);
                        }
                        _ => return TrialEnd::new(TrialState::Aborted, "the caller of the trial aborted it"),
                    }
                },
                stop = stops.recv(), if current.is_none() => {
                    return match stop {
                        Some(LeaseStop { halted: true, reason }) => TrialEnd::new(TrialState::Halted, reason),
                        Some(LeaseStop { halted: false, reason }) => TrialEnd::new(TrialState::Interrupted, reason),
                        None => TrialEnd::new(TrialState::Interrupted, "the director stopped during the trial"),
                    };
                },
                _ = progress.tick() => {
                    let details = {
                        let record = record.borrow();
                        json!({ "state": record.state, "step": record.step, "measures": record.measures })
                    };
                    feedback(goal, trial, "progress", "", String::new(), details).await;
                },
            }
        }
    }

    /// Gives the count of the last run, and goes on to the next run or to
    /// the end of the trial; the end, when the trial ends.
    async fn counted(
        &self,
        trial: &str,
        goal: &trial_goal::GoalContext,
        record: &watch::Sender<TrialRecord>,
        admitted: &Admitted,
        count: Count,
        next: Next,
    ) -> Option<TrialEnd> {
        let (run, measures) = {
            let record = record.borrow();
            let run = record
                .runs
                .last()
                .map(|run| run.run.clone())
                .unwrap_or_default();
            (run, record.measures)
        };
        self.keep_trial(&record.borrow());
        let event = match count {
            Count::NotCounted => "run_not_counted",
            Count::Completed | Count::NotCompleted => "run_counted",
        };
        let details = json!({ "count": count, "measures": measures });
        feedback(goal, trial, event, &run, String::new(), details).await;
        match next {
            Next::GoesOn => {
                self.prepares(trial, goal, record, &admitted.robots).await;
                None
            }
            Next::SignsOff => Some(TrialEnd::new(TrialState::SignedOff, "")),
            Next::NotSignedOff(message) => Some(TrialEnd::new(TrialState::NotSignedOff, message)),
        }
    }
}

/// The result of the run that `current` follows, once it ended; pending
/// with no run.
async fn ended(current: &mut Option<watch::Receiver<Phase>>) -> RunResult {
    let Some(phase) = current.as_mut() else {
        return std::future::pending().await;
    };
    loop {
        if let Phase::Ended(result) = &*phase.borrow_and_update() {
            return (**result).clone();
        }
        if phase.changed().await.is_err() {
            return std::future::pending().await;
        }
    }
}

/// Gives a feedback of the trial `trial` on its goal.
async fn feedback(
    goal: &trial_goal::GoalContext,
    trial: &str,
    event: &str,
    run: &str,
    message: String,
    details: Json,
) {
    let sent = goal
        .publish_feedback(
            trial.to_owned(),
            event.to_owned(),
            run.to_owned(),
            message,
            details.to_string(),
        )
        .await;
    if let Err(error) = sent {
        tracing::warn!("feedback of {trial}: {error}");
    }
}

#[cfg(test)]
mod tests {
    use workflow_documents::Rate;

    use super::*;
    use crate::testing::name;

    /// A trial of `runs` runs of `tape_packing_cycle` on `arm_left`.
    fn document(runs: u32) -> TrialDocument {
        let text = json!({
            "workflow_schema": "trial/v1",
            "manifest": { "name": "line_coordinator", "revision": 1 },
            "workflow": { "name": "tape_packing_cycle", "revision": 3 },
            "roles": { "so101_a": "arm_left" },
            "runs": runs,
        });
        TrialDocument::parse(&text.to_string()).unwrap()
    }

    /// Thresholds that a run of each that counts meets: every run
    /// completes, with no unsafe event, false success or false check.
    fn strict() -> Thresholds {
        Thresholds {
            completed_at_least: Rate::new(1.0).unwrap(),
            unsafe_events_at_most: 0,
            false_successes_at_most: Rate::new(0.0).unwrap(),
            false_assumption_checks_at_most: 0,
        }
    }

    /// Thresholds that any run meets.
    fn lenient() -> Thresholds {
        Thresholds {
            completed_at_least: Rate::new(0.0).unwrap(),
            unsafe_events_at_most: 10,
            false_successes_at_most: Rate::new(1.0).unwrap(),
            false_assumption_checks_at_most: 10,
        }
    }

    /// A record of a trial of `runs` runs that waits for its first run.
    fn trial(runs: u32) -> TrialRecord {
        let mut record = TrialRecord::new("t1", document(runs), "tests".to_owned());
        record.step = TrialStep::Prepares { number: 1 };
        record
    }

    /// Starts the run `run` of `record`, whose checks the operator judges
    /// `judged`.
    fn judged(record: &mut TrialRecord, run: &str, judged: Option<ChecksJudged>) {
        record.runs.push(TrialRun {
            run: run.to_owned(),
            ending: None,
            checks: None,
            judged_ending: None,
            count: None,
        });
        record.step = TrialStep::Runs {
            run: run.to_owned(),
        };
        if let Some(judged) = judged {
            record.judge_checks(run, &judged).unwrap();
        }
    }

    /// The result of a run that ended `ending`.
    fn ended(ending: Ending) -> RunResult {
        RunResult {
            ending,
            ..crate::director::empty_result()
        }
    }

    fn done() -> EndingJudged {
        EndingJudged {
            task_done: true,
            emergency_stop: false,
            unsafe_events: Vec::new(),
        }
    }

    /// A run that starts waits for the judgement of its ending, and counts
    /// as completed when it completed; the operator then prepares the next
    /// run.
    #[test]
    fn a_run_that_starts_counts_by_the_judgement_of_its_ending() {
        let mut record = trial(3);
        judged(&mut record, "r1", Some(ChecksJudged::Starts));
        assert_eq!(
            record.run_ended(&ended(Ending::Completed), &strict()),
            AfterRun::JudgesEnding
        );
        assert_eq!(
            record.step,
            TrialStep::JudgesEnding {
                run: "r1".to_owned()
            }
        );
        let (count, next) = record.judge_ending(done(), &strict()).unwrap();
        assert_eq!((count, next), (Count::Completed, Next::GoesOn));
        assert_eq!((record.measures.runs, record.measures.completed), (1, 1));
        assert_eq!(record.step, TrialStep::Prepares { number: 2 });
        assert!(
            record.judge_ending(done(), &strict()).is_err(),
            "no ending waits any more"
        );
    }

    /// A run whose work cell is not ready does not count, and the operator
    /// prepares the same run again.
    #[test]
    fn a_run_whose_work_cell_is_not_ready_does_not_count() {
        let mut record = trial(3);
        judged(&mut record, "r1", Some(ChecksJudged::NotReady));
        let after = record.run_ended(&ended(Ending::Refused), &strict());
        assert_eq!(
            after,
            AfterRun::Counted {
                count: Count::NotCounted,
                next: Next::GoesOn
            }
        );
        assert_eq!(record.measures, Measures::default());
        assert_eq!(record.step, TrialStep::Prepares { number: 1 });
    }

    /// A run that the director refused on a watch value outside the
    /// envelope does not count: the work cell is not ready, and the operator
    /// prepares the same run again. A watch value that does not read is no
    /// work cell that is not ready: the run counts as not completed.
    #[test]
    fn a_run_refused_on_a_watch_value_outside_the_envelope_does_not_count() {
        use workflow_documents::work_cell::{NotCovered, TestedRange};
        let mut record = trial(3);
        judged(&mut record, "r1", None);
        let mut outside = ended(Ending::Refused);
        outside.watch.insert(name("loose_tapes"), 9.0);
        outside.uncovered.push(workflow_documents::Uncovered {
            parameter: name("loose_tapes"),
            role: None,
            value: Some(9.0),
            why: NotCovered::Outside {
                tested: TestedRange { min: 0.0, max: 2.0 },
            },
        });
        assert_eq!(
            record.run_ended(&outside, &strict()),
            AfterRun::Counted {
                count: Count::NotCounted,
                next: Next::GoesOn
            }
        );
        assert_eq!(record.measures, Measures::default());
        assert_eq!(record.step, TrialStep::Prepares { number: 1 });

        judged(&mut record, "r2", None);
        let unread = ended(Ending::Refused);
        assert!(matches!(
            record.run_ended(&unread, &lenient()),
            AfterRun::Counted {
                count: Count::NotCompleted,
                ..
            }
        ));
        assert_eq!(record.measures.runs, 1);
    }

    /// A wrong check counts the run as not completed and each wrong check
    /// as a false check; a run that ended before the judgement of its
    /// checks, as a refused or an aborted run, counts as not completed.
    #[test]
    fn a_wrong_check_or_a_run_that_did_not_start_counts_as_not_completed() {
        let mut record = trial(5);
        let wrong = vec![name("belt_runs"), name("in_reach")];
        judged(&mut record, "r1", Some(ChecksJudged::FalseCheck { wrong }));
        let after = record.run_ended(&ended(Ending::Refused), &lenient());
        assert_eq!(
            after,
            AfterRun::Counted {
                count: Count::NotCompleted,
                next: Next::GoesOn
            }
        );
        assert_eq!(
            (
                record.measures.runs,
                record.measures.false_assumption_checks
            ),
            (1, 2)
        );
        for ending in [Ending::Refused, Ending::Aborted] {
            judged(&mut record, "r2", None);
            assert!(matches!(
                record.run_ended(&ended(ending), &lenient()),
                AfterRun::Counted {
                    count: Count::NotCompleted,
                    ..
                }
            ));
        }
        assert_eq!((record.measures.runs, record.measures.completed), (3, 0));
    }

    /// An aborted run that started waits for the judgement of its ending,
    /// and counts as not completed whatever the task; a completed run whose
    /// task is not done counts as completed and as a false success; each
    /// emergency stop and other unsafe event counts.
    #[test]
    fn the_judgement_of_an_ending_counts_false_successes_and_unsafe_events() {
        let mut record = trial(5);
        judged(&mut record, "r1", Some(ChecksJudged::Starts));
        assert_eq!(
            record.run_ended(&ended(Ending::Aborted), &lenient()),
            AfterRun::JudgesEnding
        );
        let (count, _) = record.judge_ending(done(), &lenient()).unwrap();
        assert_eq!(count, Count::NotCompleted);
        judged(&mut record, "r2", Some(ChecksJudged::Starts));
        record.run_ended(&ended(Ending::Completed), &lenient());
        let unsafe_and_undone = EndingJudged {
            task_done: false,
            emergency_stop: true,
            unsafe_events: vec!["the box fell off the belt".to_owned()],
        };
        let (count, _) = record.judge_ending(unsafe_and_undone, &lenient()).unwrap();
        assert_eq!(count, Count::Completed);
        assert_eq!(
            record.measures,
            Measures {
                runs: 2,
                completed: 1,
                false_successes: 1,
                unsafe_events: 2,
                false_assumption_checks: 0,
            }
        );
    }

    /// The trial ends after its last run that counts, and signs off only
    /// when the runs meet the thresholds: a trial of a false success does
    /// not meet `false_successes_at_most`.
    #[test]
    fn the_trial_signs_off_after_its_last_run_only_when_its_runs_meet_the_thresholds() {
        let mut record = trial(2);
        for run in ["r1", "r2"] {
            judged(&mut record, run, Some(ChecksJudged::Starts));
            record.run_ended(&ended(Ending::Completed), &strict());
            let (_, next) = record.judge_ending(done(), &strict()).unwrap();
            let expected = match run {
                "r1" => Next::GoesOn,
                _ => Next::SignsOff,
            };
            assert_eq!(next, expected);
        }
        assert_eq!(record.step, TrialStep::Ended);
        assert!(
            record.judge_ending(done(), &strict()).is_err(),
            "a trial that ends counts no judgement again"
        );
        assert_eq!(record.measures.runs, 2);
        let mut record = trial(1);
        judged(&mut record, "r1", Some(ChecksJudged::Starts));
        record.run_ended(&ended(Ending::Completed), &strict());
        let undone = EndingJudged {
            task_done: false,
            ..done()
        };
        let (_, next) = record.judge_ending(undone, &strict()).unwrap();
        assert_eq!(
            next,
            Next::NotSignedOff(
                "the 1 runs that count do not meet the thresholds of the bundle: `false_successes_at_most`".to_owned()
            )
        );
        assert!(record.judge_ending(done(), &strict()).is_err());
        assert_eq!(
            (record.measures.completed, record.measures.false_successes),
            (1, 1)
        );
    }

    /// A trial ends, signing off nothing, once the runs left cannot meet
    /// the thresholds.
    #[test]
    fn the_trial_ends_early_once_its_thresholds_cannot_be_met() {
        let mut record = trial(20);
        judged(&mut record, "r1", None);
        let after = record.run_ended(&ended(Ending::Refused), &strict());
        assert_eq!(
            after,
            AfterRun::Counted {
                count: Count::NotCompleted,
                next: Next::NotSignedOff(
                    "the 19 runs left cannot meet the thresholds of the bundle any more: `completed_at_least`".to_owned()
                )
            }
        );
    }

    /// A run that ended halted or interrupted ends the trial in the same
    /// way, and counts nothing.
    #[test]
    fn a_halted_or_interrupted_run_ends_the_trial() {
        for (ending, state) in [
            (Ending::Halted, TrialState::Halted),
            (Ending::Interrupted, TrialState::Interrupted),
        ] {
            let mut record = trial(3);
            judged(&mut record, "r1", Some(ChecksJudged::Starts));
            let after = record.run_ended(&ended(ending), &strict());
            assert!(
                matches!(&after, AfterRun::Ends { state: ended, .. } if *ended == state),
                "{after:?}"
            );
            assert_eq!(record.measures, Measures::default());
        }
    }

    /// A running trial holds its robots only once the check of its admission
    /// passed, and no more once it ended; the trial that asks holds none of
    /// its own.
    #[test]
    fn a_trial_holds_its_robots_from_its_admission_to_its_end() {
        let entry = |state: TrialState, holds_robots: bool| {
            let mut record = trial(3);
            record.state = state;
            TrialEntry {
                record: Arc::new(watch::Sender::new(record)),
                live: None,
                holds_robots,
            }
        };
        let trials = BTreeMap::from([
            ("t1".to_owned(), entry(TrialState::Running, false)),
            ("t2".to_owned(), entry(TrialState::Running, true)),
            ("t3".to_owned(), entry(TrialState::Refused, true)),
        ]);
        let arm = RobotName::parse("arm_left").unwrap();
        assert_eq!(
            robots_of_trials(&trials, None),
            [(arm.clone(), "t2".to_owned())]
        );
        assert!(robots_of_trials(&trials, Some("t2")).is_empty());
        assert_eq!(
            robots_of_trials(&trials, Some("t1")),
            [(arm, "t2".to_owned())]
        );
    }

    /// The checks of a run are judged once, and only for the run that goes.
    #[test]
    fn the_checks_of_a_run_are_judged_once_and_only_for_the_run_that_goes() {
        let mut record = trial(3);
        judged(&mut record, "r1", Some(ChecksJudged::Starts));
        assert_eq!(
            record.judge_checks("r1", &ChecksJudged::NotReady),
            Err("the checks of the run `r1` are judged already".to_owned())
        );
        assert!(record.judge_checks("r2", &ChecksJudged::Starts).is_err());
    }

    /// The trial ids grow across the records on disk, and a trial that its
    /// record gives as running ends interrupted when the store opens again.
    #[test]
    fn a_trial_running_when_the_director_stopped_ends_interrupted() {
        let dir = tempfile::tempdir().unwrap();
        let (store, kept) = Trials::open(dir.path()).unwrap();
        assert!(kept.is_empty());
        let id = store.next_trial().unwrap();
        assert_eq!(id, "t1");
        let mut record = TrialRecord::new(&id, document(3), "tests".to_owned());
        store.keep(&record).unwrap();
        record.trial = "t5".to_owned();
        record.state = TrialState::SignedOff;
        record.step = TrialStep::Ended;
        store.keep(&record).unwrap();
        let (store, kept) = Trials::open(dir.path()).unwrap();
        let states: BTreeMap<String, (TrialState, String)> = kept
            .into_iter()
            .map(|record| (record.trial, (record.state, record.message)))
            .collect();
        assert_eq!(
            states["t1"],
            (
                TrialState::Interrupted,
                "the director stopped during the trial".to_owned()
            )
        );
        assert_eq!(states["t5"].0, TrialState::SignedOff);
        assert_eq!(store.next_trial().unwrap(), "t6");
    }
}
