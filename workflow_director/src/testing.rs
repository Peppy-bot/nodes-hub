//! The fixtures of the tests of the director: a fleet of simulated robots
//! that implement synthetic contracts, the plan of a workflow on it, and a
//! host whose calls a test scripts, on a paused clock.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value as Json, json};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use workflow_documents::run::{Attendance, Incidents};
use workflow_documents::{EventKind, Name, RobotName, Workflow};

use crate::checker::{self, RunChoices};
use crate::config::AllowedContracts;
use crate::contract::ContractShape;
use crate::contract::tests::TEST_ARM;
use crate::engine::failure::Failure;
use crate::engine::host::{CallEnd, Event, Host, Join, Joining};
use crate::engine::plan::{Plan, Target};
use crate::fleet::{Fleet, Instance, Robot, RobotContract, Snapshot};

/// A perception contract for the tests: a scan that gives item ids and a
/// locate that gives the position of one.
const TEST_PERCEPTION: &str = r#"
{
  peppy_schema: "contract/v1",
  manifest: { name: "test_perception", tag: "v1" },
  interfaces: {
    services: [
      {
        name: "scan",
        request_message_format: { description: "string" },
        response_message_format: { item_ids: { $type: "array", $items: { $type: "string", $marks: ["item_id"] } } },
      },
      {
        name: "locate",
        request_message_format: { item_id: { $type: "string", $marks: ["item_id"] } },
        response_message_format: { found: "bool", position: { $type: "array", $items: "f64", $length: 3, $marks: ["robot_frame"] } },
      },
    ],
  },
}
"#;

pub const ARM_PIN: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const PERCEPTION_PIN: &str = "2222222222222222222222222222222222222222222222222222222222222222";
/// The contract of the locations a skill knows, as the hub gives it, and its pin.
const TEST_LOCATIONS: &str = include_str!("../tests/fixtures/locations.json5");
pub const LOCATIONS_PIN: &str = "3333333333333333333333333333333333333333333333333333333333333333";
const CORE: &str = "core";

pub fn name(text: &str) -> Name {
    Name::parse(text).unwrap()
}

pub fn robot_name(text: &str) -> RobotName {
    RobotName::parse(text).unwrap()
}

/// The instance of `robot` that serves both test contracts.
fn instance_of(robot: &str) -> String {
    format!("{robot}_brain")
}

/// A simulated, running robot that implements both test contracts on one
/// instance.
pub fn robot(name: &str) -> Robot {
    let contract = |contract: &str, sha256: &str| RobotContract {
        name: contract.to_owned(),
        tag: "v1".to_owned(),
        sha256: sha256.to_owned(),
        instances: vec!["brain".to_owned()],
    };
    Robot {
        name: name.to_owned(),
        core_node: CORE.to_owned(),
        kind: "simulated".to_owned(),
        clock: "wall".to_owned(),
        software: "running".to_owned(),
        robot_commander: String::new(),
        instances: vec![Instance {
            name: "brain".to_owned(),
            instance: instance_of(name),
            core_node: CORE.to_owned(),
        }],
        contracts: vec![
            contract("test_arm", ARM_PIN),
            contract("test_perception", PERCEPTION_PIN),
            contract("locations", LOCATIONS_PIN),
        ],
    }
}

/// A snapshot of a fleet of `robots`.
pub fn snapshot(robots: Vec<Robot>) -> Snapshot {
    let contracts = [
        (ARM_PIN, TEST_ARM),
        (PERCEPTION_PIN, TEST_PERCEPTION),
        (LOCATIONS_PIN, TEST_LOCATIONS),
    ]
    .into_iter()
    .map(|(pin, document)| {
        (
            pin.to_owned(),
            Arc::new(ContractShape::parse(document, pin).unwrap()),
        )
    })
    .collect();
    let fleet = Fleet {
        version: "0.0.0-test".to_owned(),
        core_node: CORE.to_owned(),
        robots: robots
            .into_iter()
            .map(|robot| (robot.name.clone(), robot))
            .collect(),
        ..Fleet::default()
    };
    Snapshot {
        fleet,
        contracts,
        unreadable: BTreeMap::new(),
        models: BTreeMap::new(),
        ready: true,
    }
}

pub fn allowed() -> AllowedContracts {
    AllowedContracts::parse("test_arm:v1, test_perception:v1").unwrap()
}

/// A workflow from its JSON value.
pub fn workflow(value: &Json) -> Workflow {
    Workflow::parse(&value.to_string())
        .unwrap_or_else(|refusals| panic!("the workflow does not parse: {refusals}"))
}

/// The plan of `workflow` with each role bound to the robot of its name,
/// in a run of `attendance`.
pub fn plan(workflow: &Json, attendance: Attendance) -> Plan {
    let roles: Vec<Robot> = self::workflow(workflow)
        .roles
        .keys()
        .map(|role| robot(role.as_str()))
        .collect();
    plan_on(workflow, attendance, &snapshot(roles))
}

/// The plan of `workflow` on the fleet of `snapshot`, with each role bound
/// to the robot of its name, in a run of `attendance`.
pub fn plan_on(workflow: &Json, attendance: Attendance, snapshot: &Snapshot) -> Plan {
    let workflow = self::workflow(workflow);
    let binding: BTreeMap<Name, RobotName> = workflow
        .roles
        .keys()
        .map(|role| (role.clone(), robot_name(role.as_str())))
        .collect();
    let choices = RunChoices {
        attendance,
        confirmed: Default::default(),
        incidents: &Incidents::None,
        poll: workflow_documents::Bound::from_seconds(0.5).unwrap(),
        cancel_timeout: Duration::from_secs(10),
    };
    checker::plan(Arc::new(workflow), snapshot, &allowed(), &binding, choices)
        .unwrap_or_else(|errors| panic!("the workflow does not hold: {errors:#?}"))
}

/// A workflow of one role named after its robot, with the reach
/// assumption a lane that grabs needs.
pub fn one_arm(role: &str, tree: Json, safe_state: Json) -> Json {
    json!({
        "workflow_schema": "workflow/v1",
        "manifest": { "name": "test", "revision": 1 },
        "task": { "objective": "Move a block" },
        "contracts": { "arm": "test_arm:v1", "eyes": "test_perception:v1" },
        "roles": { role: {} },
        "observations": {
            "target": { "member": "arm.find", "role": role, "args": { "description": "block" } },
        },
        "assumptions": {
            "in_reach": { "kind": "reach", "says": "The block is in reach", "check": "target.found" },
        },
        "success": "true",
        "actions": { role: { "tree": tree, "safe_state": safe_state } },
    })
}

/// The step `id`: a rest of the arm at home for 1 s, within 10 s.
pub fn rest(id: &str) -> Json {
    json!({ "id": id, "timeout_s": 10, "call": { "member": "arm.home", "args": { "duration_s": 1 } } })
}

/// A safe state that sends the arm home, within 10 s.
pub fn home() -> Json {
    json!([{ "id": "home", "timeout_s": 10, "call": { "member": "arm.home", "args": { "duration_s": 2 } } }])
}

/// `value` with a second role `beta` whose lane is `beta_tree`, on the robot
/// `beta`.
pub fn with_beta(mut value: Json, beta_tree: Json) -> Json {
    value["roles"]["beta"] = json!({});
    value["actions"]["beta"] = json!({ "tree": beta_tree, "safe_state": [] });
    value
}

/// The answer of a scripted call: after how long, how it ends, and how
/// it ends when the director cancels it first (`None`: cancelled).
#[derive(Clone, Debug)]
pub struct Answer {
    pub after: Duration,
    pub end: CallEnd,
    pub on_cancel: Option<CallEnd>,
}

impl Answer {
    pub fn now(result: Json) -> Answer {
        Answer {
            after: Duration::ZERO,
            end: CallEnd::Completed(result),
            on_cancel: None,
        }
    }

    pub fn after(seconds: u64, result: Json) -> Answer {
        Answer {
            after: Duration::from_secs(seconds),
            end: CallEnd::Completed(result),
            on_cancel: None,
        }
    }

    pub fn failing(kind: &str) -> Answer {
        let failure = Failure::new(kind, format!("scripted {kind}"));
        Answer {
            after: Duration::ZERO,
            end: CallEnd::Failed(failure),
            on_cancel: None,
        }
    }

    /// A call that never ends by itself.
    pub fn hangs() -> Answer {
        Answer {
            after: Duration::from_secs(1_000_000),
            end: CallEnd::Completed(Json::Null),
            on_cancel: None,
        }
    }

    /// A goal that never ends by itself, nor within the cancel timeout
    /// once cancelled.
    pub fn hangs_past_its_cancel() -> Answer {
        Answer {
            on_cancel: Some(CallEnd::Unknown),
            ..Answer::hangs()
        }
    }
}

/// A call the host received.
#[derive(Clone, Debug, PartialEq)]
pub struct Sent {
    pub role: String,
    pub member: String,
    pub request: Json,
    pub at_s: f64,
    pub cancelled: bool,
}

/// A host whose calls answer from a script per member, in order, with the
/// last answer kept for the calls past it. Its clock is tokio's, paused in
/// each test, so a run takes no wall time.
pub struct ScriptedHost {
    start: tokio::time::Instant,
    answers: Mutex<BTreeMap<String, VecDeque<Answer>>>,
    topics: Mutex<BTreeMap<String, (Json, u64)>>,
    /// Counts the messages published, so a read that waits for one wakes.
    published: watch::Sender<u64>,
    /// The messages to publish at their run time: `(at, member, message)`.
    scheduled: Mutex<Vec<(u64, String, Json)>>,
    /// The nanoseconds the clock moved at its readings, when it moves at
    /// each reading as wall time does.
    readings: Option<AtomicU64>,
    pub sent: Mutex<Vec<Sent>>,
    pub events: Mutex<Vec<Event>>,
    pub leases: Mutex<Vec<(String, bool)>>,
    /// How many events the host had given at each lease change.
    pub events_before_lease: Mutex<Vec<usize>>,
    /// Why the daemon refuses each close of a lease, when it does.
    pub close_refusal: Mutex<Option<String>>,
    /// How long the daemon takes to open a lease.
    opening: Duration,
    /// How long the daemon takes to close a lease.
    closing: Duration,
}

/// The run clock starts at 1 s, so 0 never reads as a time.
const START_NS: u64 = 1_000_000_000;

impl ScriptedHost {
    pub fn new() -> ScriptedHost {
        ScriptedHost {
            start: tokio::time::Instant::now(),
            answers: Mutex::new(BTreeMap::new()),
            topics: Mutex::new(BTreeMap::new()),
            published: watch::Sender::new(0),
            scheduled: Mutex::new(Vec::new()),
            readings: None,
            sent: Mutex::new(Vec::new()),
            events: Mutex::new(Vec::new()),
            leases: Mutex::new(Vec::new()),
            events_before_lease: Mutex::new(Vec::new()),
            close_refusal: Mutex::new(None),
            opening: Duration::ZERO,
            closing: Duration::ZERO,
        }
    }

    /// A clock that moves by a nanosecond at each reading, as wall time
    /// moves between any two readings.
    pub fn moving_at_each_reading(self) -> ScriptedHost {
        ScriptedHost {
            readings: Some(AtomicU64::new(0)),
            ..self
        }
    }

    /// Closes each lease `after` it was asked.
    pub fn closes_leases_after(self, after: Duration) -> ScriptedHost {
        ScriptedHost {
            closing: after,
            ..self
        }
    }

    /// Opens each lease `after` it was asked.
    pub fn opens_leases_after(self, after: Duration) -> ScriptedHost {
        ScriptedHost {
            opening: after,
            ..self
        }
    }

    /// Answers the calls of `member` with `answers`, in order.
    pub fn answer(self, member: &str, answers: Vec<Answer>) -> ScriptedHost {
        self.answers
            .lock()
            .unwrap()
            .insert(member.to_owned(), answers.into());
        self
    }

    /// Publishes `message` on the topic `member` once the run reaches
    /// `seconds`.
    pub fn publishes_at(self, seconds: u64, member: &str, message: Json) -> ScriptedHost {
        let at = START_NS + Duration::from_secs(seconds).as_nanos() as u64;
        self.scheduled
            .lock()
            .unwrap()
            .push((at, member.to_owned(), message));
        self
    }

    /// Delivers each scheduled message whose run time came.
    fn deliver_due(&self) {
        let now = self.now();
        let mut scheduled = self.scheduled.lock().unwrap();
        let (due, later): (Vec<_>, Vec<_>) = std::mem::take(&mut *scheduled)
            .into_iter()
            .partition(|(at, _, _)| *at <= now);
        *scheduled = later;
        let mut topics = self.topics.lock().unwrap();
        for (at, member, message) in due {
            topics.insert(member, (message, at));
        }
    }

    /// Sets the latest message of the topic `member`, arrived now.
    pub fn publish(&self, member: &str, message: Json) {
        let now = self.now();
        self.topics
            .lock()
            .unwrap()
            .insert(member.to_owned(), (message, now));
        self.published.send_modify(|count| *count += 1);
    }

    /// The members called, in order.
    pub fn members(&self) -> Vec<String> {
        self.sent
            .lock()
            .unwrap()
            .iter()
            .map(|sent| sent.member.clone())
            .collect()
    }

    pub fn calls_of(&self, member: &str) -> Vec<Sent> {
        self.sent
            .lock()
            .unwrap()
            .iter()
            .filter(|sent| sent.member == member)
            .cloned()
            .collect()
    }

    /// The kinds of the events given, with their step, in order.
    pub fn event_kinds(&self) -> Vec<(EventKind, String)> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .map(|event| (event.kind, event.step.clone()))
            .collect()
    }

    fn next_answer(&self, member: &str) -> Answer {
        let mut answers = self.answers.lock().unwrap();
        let queue = answers
            .get_mut(member)
            .unwrap_or_else(|| panic!("no answer of `{member}` is scripted"));
        match queue.len() {
            0 => panic!("no answer of `{member}` is left"),
            1 => queue.front().cloned().unwrap(),
            _ => queue.pop_front().unwrap(),
        }
    }
}

impl Host for ScriptedHost {
    fn now(&self) -> u64 {
        let moved = self
            .readings
            .as_ref()
            .map_or(0, |readings| readings.fetch_add(1, Ordering::Relaxed));
        START_NS + self.start.elapsed().as_nanos() as u64 + moved
    }

    fn wall_now(&self) -> u64 {
        self.now()
    }

    async fn until(&self, at: u64) {
        tokio::time::sleep_until(self.start + Duration::from_nanos(at.saturating_sub(START_NS)))
            .await
    }

    async fn call(&self, target: &Target, request: Json, cancel: CancellationToken) -> CallEnd {
        let index = {
            let mut sent = self.sent.lock().unwrap();
            sent.push(Sent {
                role: target.role.to_string(),
                member: target.member.to_string(),
                request,
                at_s: (self.now() - START_NS) as f64 / 1e9,
                cancelled: false,
            });
            sent.len() - 1
        };
        let answer = self.next_answer(target.member.as_str());
        tokio::select! {
            _ = tokio::time::sleep(answer.after) => answer.end,
            _ = cancel.cancelled() => {
                self.sent.lock().unwrap()[index].cancelled = true;
                answer.on_cancel.unwrap_or(CallEnd::Cancelled { asked: true, reason: "cancelled".to_owned() })
            }
        }
    }

    fn latest(&self, target: &Target) -> Option<(Json, u64)> {
        self.deliver_due();
        self.topics
            .lock()
            .unwrap()
            .get(target.member.as_str())
            .cloned()
    }

    async fn message_arrives(&self, target: &Target) {
        let due = self
            .scheduled
            .lock()
            .unwrap()
            .iter()
            .find(|(_, member, _)| member == target.member.as_str())
            .map(|(at, _, _)| *at);
        if let Some(at) = due {
            let after = Duration::from_nanos(at.saturating_sub(START_NS));
            tokio::time::sleep_until(self.start + after).await;
            return;
        }
        let mut published = self.published.subscribe();
        let _ = published
            .wait_for(|_| {
                self.topics
                    .lock()
                    .unwrap()
                    .contains_key(target.member.as_str())
            })
            .await;
    }

    async fn joining(&self, _name: &Name, _nth: u64) -> Joining {
        Joining::AsFound
    }

    fn joined(&self, _name: &Name, _nth: u64, _join: &Join) {}

    async fn set_lease_open(&self, robot: &RobotName, open: bool) -> Result<(), String> {
        if let Some(refusal) = self.close_refusal.lock().unwrap().clone()
            && !open
        {
            return Err(refusal);
        }
        tokio::time::sleep(match open {
            true => self.opening,
            false => self.closing,
        })
        .await;
        let given = self.events.lock().unwrap().len();
        self.events_before_lease.lock().unwrap().push(given);
        self.leases.lock().unwrap().push((robot.to_string(), open));
        Ok(())
    }

    fn event(&self, event: Event) {
        self.events.lock().unwrap().push(event);
    }
}

/// The answer of `find` for a block in reach.
pub fn found() -> Json {
    json!({ "found": true, "item_id": "b1", "position": [0.3, 0.0, 0.1], "kind": "block", "count": "1" })
}
