//! What the tests of the director run it against: the generated harness, a fleet that a test scripts on the fleet
//! mock of the harness, and one simulated robot `alpha` whose instance `alpha_arm` implements `test_arm:v1` and
//! answers its `home` action. Each test binary compiles its own copy, so a helper that only one of them calls is
//! unused in the others.
#![allow(dead_code)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use peppy_codec::MessageCodec;
use peppy_codec::contract::{InterfaceKind, MessageSide, PeppyContractParser};
use peppygen::Parameters;
use peppygen::consumed_actions::fleet::changes::{
    FeedbackMessage as ChangesFeedback, RobotLeasesChangesActionFeedbackMessageListing,
    RobotLeasesChangesActionFeedbackMessageListingDocumentsItem,
    RobotLeasesChangesActionFeedbackMessageListingLeasesItem,
    RobotLeasesChangesActionFeedbackMessageListingNodesItem,
    RobotLeasesChangesActionFeedbackMessageListingNodesItemInstancesItem,
    RobotLeasesChangesActionFeedbackMessageListingNodesItemMachinesItem,
    RobotLeasesChangesActionFeedbackMessageListingRobotsItem,
    RobotLeasesChangesActionFeedbackMessageListingRobotsItemContractsItem,
    RobotLeasesChangesActionFeedbackMessageListingRobotsItemInstancesItem,
};
use peppygen::consumed_actions::fleet::lease::FeedbackMessage as LeaseFeedback;
use peppygen::consumed_services::fleet::check_move::ResponseData as CheckMove;
use peppygen::consumed_services::fleet::contract::ResponseData as ContractDocument;
use peppygen::consumed_services::fleet::list::{
    ResponseData as ListResponse, ResponseDataDocumentsItem, ResponseDataLeasesItem,
    ResponseDataNodesItem, ResponseDataNodesItemInstancesItem, ResponseDataNodesItemMachinesItem,
    ResponseDataRobotsItem, ResponseDataRobotsItemContractsItem,
    ResponseDataRobotsItemInstancesItem,
};
use peppygen::fixtures::exposed_actions::director::run;
use peppygen::fixtures::exposed_services::director::robots;
use peppygen::fixtures::harness::{Config, Harness, Mocks};
use peppygen::mock::deps::fleet;
use peppylib::Payload;
use peppylib::messaging::{Failure, FailureKindName};
use peppylib::messaging::{SenderTarget, TopicMessenger};
use peppylib::testing::{MockActionServerCore, MockServiceCore};
use serde_json::{Value, json};
use tokio::sync::{mpsc, watch};
use workflow_director::Fault;

/// How long the wire may take for any one exchange.
pub const WIRE: Duration = Duration::from_secs(10);
/// How often a test reads the director again while it waits for the director to see the fleet.
pub const SETTLE: Duration = Duration::from_millis(50);
/// How long a mock waits for the next request of the director.
const IDLE: Duration = Duration::from_secs(3600);
/// The arm of the tests, as the fleet serves it.
pub const TEST_ARM: &str = include_str!("../fixtures/test_arm.json5");
/// The pin of the arm in the listing.
pub const ARM_PIN: &str = "1111111111111111111111111111111111111111111111111111111111111111";
/// The contract of the locations the arm knows, as the hub gives it, and its pin in the listing.
pub const LOCATIONS: &str = include_str!("../fixtures/locations.json5");
pub const LOCATIONS_PIN: &str = "3333333333333333333333333333333333333333333333333333333333333333";
/// The robot of the tests.
pub const ROBOT: &str = "alpha";
/// The instance of the robot that serves the arm.
pub const ARM_INSTANCE: &str = "alpha_arm";
/// The name of that instance in the copy of the robot.
const ARM_NAME: &str = "arm";
/// The node of that instance.
const ARM_NODE: &str = "test_arm_node";

/// How many journals of ended runs a director of the tests keeps, unless a test names its own bound.
const JOURNAL_MAX_RUNS: u32 = 100;

/// How long a cancelled goal has to end, for a director of the tests.
pub const CANCEL_TIMEOUT: Duration = Duration::from_secs(1);

/// The parameters of a director whose journals live in `journal_root`, keeping those of `journal_max_runs` ended
/// runs, which allows the arm of the tests.
pub fn parameters(journal_root: &std::path::Path, journal_max_runs: u32) -> Parameters {
    Parameters {
        allowed_contracts: "test_arm:v1".to_owned(),
        cancel_timeout_s: CANCEL_TIMEOUT.as_secs_f64(),
        journal_max_mib: 64,
        journal_max_runs,
        journal_root: journal_root.display().to_string(),
        poll_s: 0.1,
    }
}

/// A lease of the listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lease {
    pub robot: String,
    pub holder: String,
    pub holder_core_node: String,
    pub name: String,
    pub open: bool,
}

/// How the scripted fleet stands: its halt, its leases, whether it refuses each close of a lease, whether each
/// lease ends as it is taken, with no event, as a removal of the robot right after the take ends it, and whether
/// `alpha` is a physical robot.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FleetState {
    pub halted: bool,
    pub leases: Vec<Lease>,
    pub close_refused: bool,
    pub lease_ends_at_take: bool,
    pub physical: bool,
}

/// The fleet of the tests, on the fleet mock of the harness: it lists `alpha` with the arm of the tests, answers
/// each member that the director calls from its state, keeps each call as `<member> <robot> <name>` in order, and
/// sends each change of its state as a feedback of the `changes` goal of the director.
pub struct ScriptedFleet {
    pub state: watch::Sender<FleetState>,
    pub calls: Arc<Mutex<Vec<String>>>,
    lease_goal: mpsc::UnboundedSender<OnLeaseGoal>,
    core_node: String,
}

/// What the scripted daemon does next on the lease goal of the director.
enum OnLeaseGoal {
    Send(LeaseFeedback),
    End,
}

impl ScriptedFleet {
    /// Serve `mock` for a stack on `core_node`, whose daemon answers the director as `director`.
    pub fn serve(mock: fleet::Mock, core_node: &str, director: &str) -> ScriptedFleet {
        let (state, _) = watch::channel(FleetState::default());
        let calls = Arc::new(Mutex::new(Vec::new()));
        let (lease_goal, mut lease_goal_rx) = mpsc::unbounded_channel::<OnLeaseGoal>();
        let fleet::Mock {
            list,
            contract,
            take,
            release,
            open,
            close,
            check_move,
            mut changes,
            mut lease,
            ..
        } = mock;
        let core = core_node.to_owned();
        let director = director.to_owned();
        {
            let state = state.clone();
            let core = core.clone();
            tokio::spawn(async move {
                while let Ok(responder) = list.next_request(IDLE).await {
                    let listing = list_response(&core, &state.borrow());
                    let _ = responder.respond(listing).await;
                }
            });
        }
        tokio::spawn(async move {
            while let Ok((_, responder)) = contract.next_request(IDLE).await {
                let _ = responder
                    .respond(ContractDocument {
                        document: TEST_ARM.to_owned(),
                    })
                    .await;
            }
        });
        {
            let (state, calls, core, director) =
                (state.clone(), calls.clone(), core.clone(), director.clone());
            tokio::spawn(async move {
                while let Ok((request, responder)) = take.next_request(IDLE).await {
                    calls
                        .lock()
                        .unwrap()
                        .push(format!("take {} {}", request.robot, request.name));
                    state.send_modify(|state| {
                        state.leases.retain(|lease| lease.robot != request.robot);
                        if state.lease_ends_at_take {
                            return;
                        }
                        state.leases.push(Lease {
                            robot: request.robot.clone(),
                            holder: director.clone(),
                            holder_core_node: core.clone(),
                            name: request.name.clone(),
                            open: false,
                        });
                    });
                    let _ = responder.respond().await;
                }
            });
        }
        {
            let (state, calls) = (state.clone(), calls.clone());
            tokio::spawn(async move {
                while let Ok((request, responder)) = release.next_request(IDLE).await {
                    calls
                        .lock()
                        .unwrap()
                        .push(format!("release {}", request.robot));
                    state.send_modify(|state| {
                        state.leases.retain(|lease| lease.robot != request.robot)
                    });
                    let _ = responder.respond().await;
                }
            });
        }
        for (member, service, opens) in [
            ("open", OpenOrClose::Open(open), true),
            ("close", OpenOrClose::Close(close), false),
        ] {
            let (state, calls) = (state.clone(), calls.clone());
            tokio::spawn(async move {
                loop {
                    let (robot, responder) = match service.next(IDLE).await {
                        Some(next) => next,
                        None => return,
                    };
                    calls.lock().unwrap().push(format!("{member} {robot}"));
                    if !opens && state.borrow().close_refused {
                        responder
                            .refuse_close("the daemon did not close the lease")
                            .await;
                        continue;
                    }
                    state.send_modify(|state| {
                        state
                            .leases
                            .iter_mut()
                            .filter(|lease| lease.robot == robot)
                            .for_each(|lease| lease.open = opens)
                    });
                    responder.respond().await;
                }
            });
        }
        {
            let (state, calls) = (state.clone(), calls.clone());
            tokio::spawn(async move {
                while let Ok((request, responder)) = check_move.next_request(IDLE).await {
                    let target = request
                        .instance
                        .clone()
                        .or(request.robot.clone())
                        .unwrap_or_default();
                    calls.lock().unwrap().push(format!("check_move {target}"));
                    let refusal = state.borrow().halted.then(|| {
                        peppygen::consumed_services::fleet::check_move::ResponseDataRefusal {
                            failure_kind: "halted".to_owned(),
                            message: "the stack is halted".to_owned(),
                        }
                    });
                    let _ = responder
                        .respond(CheckMove {
                            refusal,
                            revision: 1,
                        })
                        .await;
                }
            });
        }
        {
            let mut watched = state.subscribe();
            let core = core.clone();
            tokio::spawn(async move {
                let Ok(goal) = changes.next_goal(IDLE).await else {
                    return;
                };
                let Ok(goal) = goal.accept().await else {
                    return;
                };
                let mut revision = 1;
                loop {
                    let listing = changes_listing(&core, &watched.borrow_and_update());
                    let feedback = ChangesFeedback {
                        change: "listing".to_owned(),
                        robot: String::new(),
                        revision,
                        listing,
                    };
                    if goal.publish_feedback(&feedback).await.is_err()
                        || watched.changed().await.is_err()
                    {
                        return;
                    }
                    revision += 1;
                }
            });
        }
        tokio::spawn(async move {
            let Ok(goal) = lease.next_goal(IDLE).await else {
                return;
            };
            let Ok(goal) = goal.accept().await else {
                return;
            };
            while let Some(next) = lease_goal_rx.recv().await {
                let OnLeaseGoal::Send(event) = next else {
                    let _ = goal
                        .complete_cancelled("the daemon ended the lease goal (by tests)")
                        .await;
                    return;
                };
                if goal.publish_feedback(&event).await.is_err() {
                    return;
                }
            }
        });
        ScriptedFleet {
            state,
            calls,
            lease_goal,
            core_node: core,
        }
    }

    /// Halt the fleet, as `fleet.halt` does: the stack is halted, every lease ends with a `lease_end` on the lease
    /// goal of its holder, and the listing of the halt follows.
    pub fn halt(&self) {
        let ended: Vec<String> = self
            .state
            .borrow()
            .leases
            .iter()
            .map(|lease| lease.robot.clone())
            .collect();
        self.state.send_modify(|state| {
            state.halted = true;
            state.leases.clear();
        });
        for robot in ended {
            self.lease_event("lease_end", &robot, "the stack is halted (by tests)");
        }
    }

    /// The calls of the director so far, in order.
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }

    /// Send `event` of `robot` on the lease goal of the director, as the daemon does.
    pub fn lease_event(&self, event: &str, robot: &str, cause: &str) {
        let _ = self.lease_goal.send(OnLeaseGoal::Send(LeaseFeedback {
            event: event.to_owned(),
            robot: robot.to_owned(),
            giver: String::new(),
            giver_core_node: String::new(),
            cause: cause.to_owned(),
        }));
    }

    /// End the lease goal of the director, as a daemon that starts again ends the goals of its clients.
    pub fn end_lease_goal(&self) {
        let _ = self.lease_goal.send(OnLeaseGoal::End);
    }
}

/// `open` or `close` of the fleet mock, which answer the same way.
enum OpenOrClose {
    Open(fleet::open::Service),
    Close(fleet::close::Service),
}

impl OpenOrClose {
    /// The robot of the next request, with its answer; none once the mock ends.
    async fn next(&self, timeout: Duration) -> Option<(String, LeaseAnswer)> {
        match self {
            OpenOrClose::Open(service) => {
                let (request, responder) = service.next_request(timeout).await.ok()?;
                Some((request.robot, LeaseAnswer::Open(responder)))
            }
            OpenOrClose::Close(service) => {
                let (request, responder) = service.next_request(timeout).await.ok()?;
                Some((request.robot, LeaseAnswer::Close(responder)))
            }
        }
    }
}

/// The answer to a request of `open` or of `close`.
enum LeaseAnswer {
    Open(fleet::open::Responder),
    Close(fleet::close::Responder),
}

impl LeaseAnswer {
    async fn respond(self) {
        let _ = match self {
            LeaseAnswer::Open(responder) => responder.respond().await,
            LeaseAnswer::Close(responder) => responder.respond().await,
        };
    }

    /// Refuses a close with `message`; an open is answered as it is.
    async fn refuse_close(self, message: &str) {
        match self {
            LeaseAnswer::Close(responder) => {
                let _ = responder
                    .respond_failure(fleet::close::FailureKind::AttemptFailed, message)
                    .await;
            }
            open => open.respond().await,
        }
    }
}

/// The listing of `state` on `core_node`, as the generated type of the member that `$root` names: `alpha`, physical
/// or simulated as `state` says, on wall time, whose instance `alpha_arm` serves the arm of the tests, the document of that arm, the node of that
/// instance, and the leases of `state`.
macro_rules! listing {
    ($core_node:expr, $state:expr, $listing:ident, $robot:ident, $instance:ident, $contract:ident, $lease:ident,
     $document:ident, $node:ident, $machine:ident, $node_instance:ident) => {{
        let core_node: &str = $core_node;
        let state: &FleetState = $state;
        $listing {
            version: "0.36.0".to_owned(),
            core_node: core_node.to_owned(),
            launcher: "tests".to_owned(),
            halted: state.halted,
            halted_by: String::new(),
            robots: vec![$robot {
                name: ROBOT.to_owned(),
                option: "test_sim".to_owned(),
                core_node: core_node.to_owned(),
                kind: match state.physical {
                    true => "physical".to_owned(),
                    false => "simulated".to_owned(),
                },
                clock: "wall".to_owned(),
                software: "running".to_owned(),
                robot_commander: String::new(),
                instances: vec![$instance {
                    name: ARM_NAME.to_owned(),
                    instance: ARM_INSTANCE.to_owned(),
                    core_node: core_node.to_owned(),
                    node: ARM_NODE.to_owned(),
                    node_tag: "v1".to_owned(),
                }],
                contracts: vec![
                    $contract {
                        name: "test_arm".to_owned(),
                        tag: "v1".to_owned(),
                        sha256: ARM_PIN.to_owned(),
                        instances: vec![ARM_NAME.to_owned()],
                    },
                    $contract {
                        name: "locations".to_owned(),
                        tag: "v1".to_owned(),
                        sha256: LOCATIONS_PIN.to_owned(),
                        instances: vec![ARM_NAME.to_owned()],
                    },
                ],
            }],
            leases: state
                .leases
                .iter()
                .map(|lease| $lease {
                    robot: lease.robot.clone(),
                    holder: lease.holder.clone(),
                    holder_core_node: lease.holder_core_node.clone(),
                    name: lease.name.clone(),
                    open: lease.open,
                    giver: String::new(),
                    giver_core_node: String::new(),
                    receiver: String::new(),
                    receiver_core_node: String::new(),
                })
                .collect(),
            options: Vec::new(),
            documents: vec![
                $document {
                    name: "test_arm".to_owned(),
                    tag: "v1".to_owned(),
                    sha256: ARM_PIN.to_owned(),
                    document: TEST_ARM.to_owned(),
                },
                $document {
                    name: "locations".to_owned(),
                    tag: "v1".to_owned(),
                    sha256: LOCATIONS_PIN.to_owned(),
                    document: LOCATIONS.to_owned(),
                },
            ],
            nodes: vec![$node {
                name: ARM_NODE.to_owned(),
                tag: "v1".to_owned(),
                machines: vec![$machine {
                    core_node: core_node.to_owned(),
                    manifest_sha256: "a".repeat(64),
                    commit: "0123456".to_owned(),
                }],
                instances: vec![$node_instance {
                    instance: ARM_INSTANCE.to_owned(),
                    core_node: core_node.to_owned(),
                    robot: ROBOT.to_owned(),
                }],
            }],
        }
    }};
}

/// The answer of `fleet.list` over `state`.
fn list_response(core_node: &str, state: &FleetState) -> ListResponse {
    listing!(
        core_node,
        state,
        ListResponse,
        ResponseDataRobotsItem,
        ResponseDataRobotsItemInstancesItem,
        ResponseDataRobotsItemContractsItem,
        ResponseDataLeasesItem,
        ResponseDataDocumentsItem,
        ResponseDataNodesItem,
        ResponseDataNodesItemMachinesItem,
        ResponseDataNodesItemInstancesItem
    )
}

/// The listing of a feedback of `fleet.changes` over `state`.
fn changes_listing(
    core_node: &str,
    state: &FleetState,
) -> RobotLeasesChangesActionFeedbackMessageListing {
    listing!(
        core_node,
        state,
        RobotLeasesChangesActionFeedbackMessageListing,
        RobotLeasesChangesActionFeedbackMessageListingRobotsItem,
        RobotLeasesChangesActionFeedbackMessageListingRobotsItemInstancesItem,
        RobotLeasesChangesActionFeedbackMessageListingRobotsItemContractsItem,
        RobotLeasesChangesActionFeedbackMessageListingLeasesItem,
        RobotLeasesChangesActionFeedbackMessageListingDocumentsItem,
        RobotLeasesChangesActionFeedbackMessageListingNodesItem,
        RobotLeasesChangesActionFeedbackMessageListingNodesItemMachinesItem,
        RobotLeasesChangesActionFeedbackMessageListingNodesItemInstancesItem
    )
}

/// How the `home` goals of the arm end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Home {
    /// At home at once.
    AtOnce,
    /// Never: the goal stays until the director cancels it.
    Never,
    /// The first goal stays until the director cancels it; every later goal is at home at once.
    NeverFirst,
    /// The first goal ends cancelled by another client, as a stop of the robot ends it; every later goal is at home
    /// at once.
    StoppedFirst,
    /// The first goal stays until the director cancels it, and the producer of the arm then stops, as a process that
    /// dies during the cancel; no goal of `home` reaches it after.
    DiesOnCancel,
}

/// The arm of `alpha` on the session of `harness`: it answers each `home` goal as `home` says, counts the goals, and
/// tells `arrived` of each; it goes to each place a `go_to` goal names, and keeps the names in `visited`; it serves
/// `locations:v1`, knowing the places of `known`, which a teach replaces; and it answers each `find` once the test
/// gives `finds` a permit, with a block found.
pub struct ScriptedArm {
    pub goals: Arc<Mutex<u32>>,
    pub arrived: Arc<tokio::sync::Notify>,
    pub visited: Arc<Mutex<Vec<String>>>,
    pub known: Arc<Mutex<Vec<String>>>,
    pub finds: Arc<tokio::sync::Semaphore>,
    /// The error reply, a failure kind and its message, that each answer of `find` gives while it is set.
    pub find_fails: Arc<Mutex<Option<(String, String)>>>,
    /// Told when a `DiesOnCancel` arm gets the cancel of its goal.
    pub cancelled: Arc<tokio::sync::Notify>,
}

/// The locations `known`, as `list` and `teach` of locations:v1 answer them.
fn known_locations(known: &[String]) -> Value {
    let locations: Vec<Value> = known
        .iter()
        .map(|name| {
            json!({ "name": name, "kind": "surface", "position": [0.0, 0.0, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0], "size": [0.0, 0.0, 0.0] })
        })
        .collect();
    json!({ "locations": locations })
}

impl ScriptedArm {
    pub async fn serve(
        harness: &Harness,
        core_node: &str,
        home: Home,
        known: &[&str],
    ) -> peppygen::Result<ScriptedArm> {
        let known = Arc::new(Mutex::new(
            known
                .iter()
                .map(|place| (*place).to_owned())
                .collect::<Vec<_>>(),
        ));
        let visited = Arc::new(Mutex::new(Vec::new()));
        let places =
            PeppyContractParser::from_content(LOCATIONS).expect("the locations contract parses");
        let listing = places
            .message_format(InterfaceKind::Service, "list", MessageSide::Response)
            .expect("list gives a response")
            .clone();
        let teaching = places
            .message_format(InterfaceKind::Service, "teach", MessageSide::Request)
            .expect("teach takes a request")
            .clone();
        let identity = SenderTarget::contract("locations", "v1")?;
        let list = MockServiceCore::listen(
            harness.session(),
            core_node,
            ARM_INSTANCE,
            identity.clone(),
            "list",
        )
        .await?;
        let teach = MockServiceCore::listen(
            harness.session(),
            core_node,
            ARM_INSTANCE,
            identity,
            "teach",
        )
        .await?;
        {
            let (known, listing) = (Arc::clone(&known), listing.clone());
            tokio::spawn(async move {
                let codec = MessageCodec::new("list response", &listing).expect("a codec of list");
                while let Ok((_, responder)) = list.next_request(IDLE).await {
                    let answer = known_locations(&known.lock().unwrap());
                    let payload = codec.encode(&answer).expect("a list response encodes");
                    let _ = responder.respond(Payload::from(payload)).await;
                }
            });
        }
        {
            let known = Arc::clone(&known);
            tokio::spawn(async move {
                let request =
                    MessageCodec::new("teach request", &teaching).expect("a codec of teach");
                let response =
                    MessageCodec::new("teach response", &listing).expect("a codec of teach");
                while let Ok((context, responder)) = teach.next_request(IDLE).await {
                    let taught = request
                        .decode(context.message().payload_bytes().as_ref())
                        .expect("a teach request decodes");
                    let names: Vec<String> = taught["locations"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|location| location["name"].as_str().map(str::to_owned))
                        .collect();
                    *known.lock().unwrap() = names;
                    let answer = known_locations(&known.lock().unwrap());
                    let payload = response.encode(&answer).expect("a teach response encodes");
                    let _ = responder.respond(Payload::from(payload)).await;
                }
            });
        }
        let target = SenderTarget::contract("test_arm", "v1")?;
        let finds = Arc::new(tokio::sync::Semaphore::new(0));
        let find_fails: Arc<Mutex<Option<(String, String)>>> = Arc::new(Mutex::new(None));
        let finding = MockServiceCore::listen(
            harness.session(),
            core_node,
            ARM_INSTANCE,
            target.clone(),
            "find",
        )
        .await?;
        {
            let finds = Arc::clone(&finds);
            let find_fails = Arc::clone(&find_fails);
            let arm =
                PeppyContractParser::from_content(TEST_ARM).expect("the arm of the tests parses");
            let response = arm
                .message_format(InterfaceKind::Service, "find", MessageSide::Response)
                .expect("find gives a response")
                .clone();
            tokio::spawn(async move {
                let codec = MessageCodec::new("find response", &response).expect("a codec of find");
                while let Ok((_, responder)) = finding.next_request(IDLE).await {
                    let Ok(permit) = finds.acquire().await else {
                        return;
                    };
                    permit.forget();
                    let fails = find_fails.lock().unwrap().clone();
                    if let Some((kind, message)) = fails {
                        let kind =
                            FailureKindName::parse(&kind).expect("a failure kind of the tests");
                        let _ = responder.respond_failure(Failure::new(kind, message)).await;
                        continue;
                    }
                    let found = json!({ "found": true, "item_id": "b1", "position": [0.3, 0.0, 0.1], "kind": "block", "count": "1" });
                    let payload = codec.encode(&found).expect("a find response encodes");
                    let _ = responder.respond(Payload::from(payload)).await;
                }
            });
        }
        let mut going = MockActionServerCore::expose(
            harness.session(),
            core_node,
            ARM_INSTANCE,
            target.clone(),
            "go_to",
            false,
        )
        .await?;
        {
            let arm =
                PeppyContractParser::from_content(TEST_ARM).expect("the arm of the tests parses");
            let goal = arm
                .message_format(InterfaceKind::Action, "go_to", MessageSide::Goal)
                .expect("go_to takes a goal")
                .clone();
            let result = arm
                .message_format(InterfaceKind::Action, "go_to", MessageSide::Result)
                .expect("go_to gives a result")
                .clone();
            let visited = Arc::clone(&visited);
            tokio::spawn(async move {
                let goal = MessageCodec::new("go_to goal", &goal).expect("a codec of the goal");
                let result =
                    MessageCodec::new("go_to result", &result).expect("a codec of the result");
                while let Ok(pending) = going.next_goal(IDLE).await {
                    let asked = goal
                        .decode(pending.request_bytes())
                        .expect("a go_to goal decodes");
                    visited
                        .lock()
                        .unwrap()
                        .push(asked["place"].as_str().unwrap_or_default().to_owned());
                    let Ok(goal) = pending.accept(Payload::new()).await else {
                        continue;
                    };
                    let arrived = result
                        .encode(&json!({ "arrived": true }))
                        .expect("a result of go_to");
                    let _ = goal.complete(Payload::from(arrived)).await;
                }
            });
        }
        let mut action = MockActionServerCore::expose(
            harness.session(),
            core_node,
            ARM_INSTANCE,
            target,
            "home",
            false,
        )
        .await?;
        let contract =
            PeppyContractParser::from_content(TEST_ARM).expect("the arm of the tests parses");
        let format = contract
            .message_format(InterfaceKind::Action, "home", MessageSide::Result)
            .expect("home gives a result")
            .clone();
        let codec =
            MessageCodec::new("home result", &format).expect("a codec of the result of home");
        let goals = Arc::new(Mutex::new(0));
        let arrived = Arc::new(tokio::sync::Notify::new());
        let cancelled = Arc::new(tokio::sync::Notify::new());
        let (counted, told, told_cancel) = (goals.clone(), arrived.clone(), cancelled.clone());
        tokio::spawn(async move {
            while let Ok(pending) = action.next_goal(IDLE).await {
                let number = {
                    let mut counted = counted.lock().unwrap();
                    *counted += 1;
                    *counted
                };
                let Ok(goal) = pending.accept(Payload::new()).await else {
                    continue;
                };
                told.notify_one();
                match (home, number) {
                    (Home::Never, _) | (Home::NeverFirst, 1) => {
                        tokio::spawn(async move {
                            goal.cancel_signal().await;
                            let _ = goal.complete_cancelled("cancelled", Payload::new()).await;
                        });
                    }
                    (Home::StoppedFirst, 1) => {
                        let _ = goal
                            .complete_cancelled("stopped: another client", Payload::new())
                            .await;
                    }
                    (Home::DiesOnCancel, _) => {
                        goal.cancel_signal().await;
                        told_cancel.notify_one();
                        action.stop();
                        drop(goal);
                        return;
                    }
                    (Home::AtOnce | Home::NeverFirst | Home::StoppedFirst, _) => {
                        let result = codec
                            .encode(&json!({ "at_home": true }))
                            .expect("a result of home");
                        let _ = goal.complete(Payload::from(result)).await;
                    }
                }
            }
        });
        Ok(ScriptedArm {
            goals,
            arrived,
            visited,
            known,
            finds,
            find_fails,
            cancelled,
        })
    }
}

/// A director over the scripted fleet of `alpha` and its arm, with its journals in a directory of its own.
pub struct Rig {
    pub harness: Harness,
    pub fleet: ScriptedFleet,
    pub arm: ScriptedArm,
    pub journals: tempfile::TempDir,
    pub fault: Fault,
}

impl Rig {
    pub async fn start(home: Home) -> peppygen::Result<Rig> {
        Rig::knowing(home, &[]).await
    }

    /// A director over an arm that knows the places `known`.
    pub async fn knowing(home: Home, known: &[&str]) -> peppygen::Result<Rig> {
        Rig::start_on(
            home,
            known,
            tempfile::tempdir().expect("a temporary directory"),
        )
        .await
    }

    /// A director whose journals are in `journals`, as a director that starts again over the journals of its runs.
    pub async fn start_on(
        home: Home,
        known: &[&str],
        journals: tempfile::TempDir,
    ) -> peppygen::Result<Rig> {
        Rig::start_keeping(home, known, journals, JOURNAL_MAX_RUNS).await
    }

    /// A director over the journals in `journals` that keeps those of `journal_max_runs` ended runs.
    pub async fn start_keeping(
        home: Home,
        known: &[&str],
        journals: tempfile::TempDir,
        journal_max_runs: u32,
    ) -> peppygen::Result<Rig> {
        // The codec writes the Cap'n Proto compiler the process carries into
        // this directory, once per process.
        let compilers = std::env::temp_dir().join("workflow_director_tests");
        std::fs::create_dir_all(&compilers).expect("a directory for the compiler");
        peppy_codec::compiler::extract_into(compilers);
        let config = Config {
            parameters: Some(parameters(journals.path(), journal_max_runs)),
            ..Config::default()
        };
        let fault = Fault::default();
        let raised = fault.clone();
        let (harness, Mocks { deps, .. }) = Harness::start_with(config, move |params, runner| {
            workflow_director::setup(params, runner, raised)
        })
        .await?;
        let core_node = harness
            .node_runner()
            .processor()
            .bound_core_node()
            .to_owned();
        let fleet = ScriptedFleet::serve(deps.fleet, &core_node, harness.instance_id());
        let arm = ScriptedArm::serve(&harness, &core_node, home, known).await?;
        Ok(Rig {
            harness,
            fleet,
            arm,
            journals,
            fault,
        })
    }
}

/// Publishes the `state` of the arm, idle and holding nothing, every `period` from now on, as its instance
/// `alpha_arm` does.
pub async fn publish_state_every(
    harness: &Harness,
    period: Duration,
) -> peppygen::Result<tokio::task::JoinHandle<()>> {
    let core_node = harness
        .node_runner()
        .processor()
        .bound_core_node()
        .to_owned();
    let arm = PeppyContractParser::from_content(TEST_ARM).expect("the arm of the tests parses");
    let topic = arm
        .topic("state")
        .expect("the arm of the tests has the topic state")
        .clone();
    let format = arm
        .message_format(InterfaceKind::Topic, "state", MessageSide::Message)
        .expect("state gives a message")
        .clone();
    let codec = MessageCodec::new("state message", &format).expect("a codec of state");
    let message = codec
        .encode(&json!({ "busy": false, "holding": false }))
        .expect("a state message encodes");
    let publisher = TopicMessenger::declare_publisher(
        harness.session(),
        &core_node,
        ARM_INSTANCE,
        SenderTarget::contract("test_arm", "v1")?,
        None,
        "state",
        topic.qos_profile,
        topic.retention,
    )
    .await?;
    Ok(tokio::spawn(async move {
        loop {
            let _ = publisher.publish(Payload::from(message.clone())).await;
            tokio::time::sleep(period).await;
        }
    }))
}

/// A workflow of the arm `arm`: one `home` step, and `home` again as the safe state.
pub fn home_workflow() -> String {
    json!({
        "workflow_schema": "workflow/v1",
        "manifest": { "name": "go_home", "revision": 1 },
        "task": { "objective": "Go home" },
        "contracts": { "arm": "test_arm:v1" },
        "roles": { "arm": {} },
        "success": "true",
        "actions": {
            "arm": {
                "safe_state": [{ "id": "safe", "timeout_s": 5, "call": { "member": "arm.home", "args": { "duration_s": 0 } } }],
                "tree": { "id": "home", "timeout_s": 5, "call": { "member": "arm.home", "args": { "duration_s": 0 } } },
            },
        },
    })
    .to_string()
}

/// The roles of a run of `home_workflow` on `alpha`.
pub fn on_alpha() -> String {
    json!({ "arm": ROBOT }).to_string()
}

/// The robots of the director, once it lists `alpha`: before the first feedback of the fleet, `robots` answers
/// `resource_unavailable`.
pub async fn listed_robots(rig: &Rig) -> peppygen::Result<robots::Response> {
    loop {
        match robots::poll(&rig.harness, WIRE).await {
            Ok(listed) if !listed.robots.is_empty() => return Ok(listed),
            Ok(_) => {}
            Err(error)
                if robots::Failure::of(&error)
                    .is_some_and(|failure| failure.kind.to_string() == "resource_unavailable") => {}
            Err(error) => return Err(error),
        }
        tokio::time::sleep(SETTLE).await;
    }
}

/// The events of the feedback of the run `goal` up to the end of its stream.
pub async fn feedback_of(goal: &mut run::GoalHandle) -> Vec<String> {
    let mut events = Vec::new();
    while let Ok(feedback) = goal.on_next_feedback().await {
        events.push(feedback.event);
    }
    events
}
