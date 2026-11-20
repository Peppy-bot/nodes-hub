//! The fleet as the director knows it: the fleet listing, with the pinned
//! document of each contract of each robot and the model and the links of
//! each robot, kept current by a `fleet.changes` goal. The services of the
//! director read the latest snapshot; a run reads a fresh listing when it
//! arrives.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use peppy_codec::consumer::{ConsumerError, ConsumerIdentity, MemberBinding, ServiceClient};
use peppygen::NodeRunner;
use peppygen::consumed_actions::fleet::changes;
use peppygen::consumed_services::fleet::{contract, list};
use peppylib::messaging::ProducerRef;
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use workflow_documents::KnownLocation;

use crate::contract::ContractShape;

/// The contract that gives the model of a robot and its description.
pub const ROBOT_IDENTITY: (&str, &str) = ("robot_identity", "v1");
/// The contract of the locations a skill knows, which the director lists
/// before a run and teaches through `locations`.
pub const LOCATIONS: (&str, &str) = ("locations", "v1");

/// How long a skill has to teach or list its locations.
const SKILL_TIMEOUT: Duration = Duration::from_secs(10);

/// How long a robot has to give its model.
const IDENTITY_TIMEOUT: Duration = Duration::from_secs(2);

/// How long a robot has to give its description, which carries its
/// collision meshes.
const DESCRIPTION_TIMEOUT: Duration = Duration::from_secs(10);

/// How long a call of the daemon may take.
pub const DAEMON_TIMEOUT: Duration = Duration::from_secs(10);

/// An instance of a robot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Instance {
    /// Its name in the robot.
    pub name: String,
    /// The id the stack runs it under.
    pub instance: String,
    pub core_node: String,
}

/// A contract a robot implements, with the name in the robot of each
/// instance that implements it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RobotContract {
    pub name: String,
    pub tag: String,
    pub sha256: String,
    pub instances: Vec<String>,
}

/// A robot of the fleet listing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Robot {
    pub name: String,
    pub core_node: String,
    /// `physical` or `simulated`.
    pub kind: String,
    pub clock: String,
    /// The state of its software, as the listing gives it.
    pub software: String,
    pub robot_commander: String,
    pub instances: Vec<Instance>,
    pub contracts: Vec<RobotContract>,
}

impl Robot {
    pub fn is_physical(&self) -> bool {
        self.kind == "physical"
    }

    pub fn is_running(&self) -> bool {
        self.software == "running"
    }

    /// The instances that implement `name:tag`.
    pub fn instances_of(&self, name: &str, tag: &str) -> Vec<&Instance> {
        self.contracts
            .iter()
            .filter(|c| c.name == name && c.tag == tag)
            .flat_map(|c| c.instances.iter())
            .filter_map(|instance| self.instances.iter().find(|i| &i.name == instance))
            .collect()
    }

    /// The entry of `name:tag` that lists the instance named `instance` in
    /// the robot: the pin that instance serves.
    pub fn pin_of(&self, name: &str, tag: &str, instance: &str) -> Option<&RobotContract> {
        self.contracts.iter().find(|c| {
            c.name == name && c.tag == tag && c.instances.iter().any(|served| served == instance)
        })
    }

    pub fn contract(&self, name: &str, tag: &str) -> Option<&RobotContract> {
        self.contracts
            .iter()
            .find(|c| c.name == name && c.tag == tag)
    }
}

/// A lease of the fleet listing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lease {
    pub robot: String,
    pub holder: String,
    pub holder_core_node: String,
    pub name: String,
    pub open: bool,
    pub receiver: String,
    pub receiver_core_node: String,
}

/// A machine that runs a node, with the sha256 of the manifest it runs and
/// the commit its daemon built.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Machine {
    pub core_node: String,
    pub manifest_sha256: String,
    pub commit: String,
}

/// A node of the stack.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    pub name: String,
    pub tag: String,
    pub machines: Vec<Machine>,
    /// Each instance: its id, its machine and its robot (empty outside any
    /// copy).
    pub instances: Vec<(String, String, String)>,
}

/// A pin of an option of a scope of the launch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptionPin {
    pub name: String,
    pub tag: String,
    pub sha256: String,
}

/// The fleet listing.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fleet {
    pub version: String,
    pub core_node: String,
    pub halted: bool,
    pub robots: BTreeMap<String, Robot>,
    pub leases: Vec<Lease>,
    pub nodes: Vec<Node>,
    pub option_pins: Vec<OptionPin>,
    /// The pinned documents the listing carries: `(name, tag, sha256)` to
    /// the text.
    #[serde(skip)]
    pub documents: BTreeMap<(String, String, String), String>,
}

impl Fleet {
    pub fn lease_of(&self, robot: &str) -> Option<&Lease> {
        self.leases.iter().find(|lease| lease.robot == robot)
    }

    /// Each distinct pin of `name:tag` that a robot or an option of the
    /// listing carries.
    pub fn pins_of(&self, name: &str, tag: &str) -> Vec<String> {
        let mut pins: Vec<String> = self
            .robots
            .values()
            .flat_map(|robot| robot.contracts.iter())
            .filter(|c| c.name == name && c.tag == tag)
            .map(|c| c.sha256.clone())
            .chain(
                self.option_pins
                    .iter()
                    .filter(|p| p.name == name && p.tag == tag)
                    .map(|p| p.sha256.clone()),
            )
            .collect();
        pins.sort();
        pins.dedup();
        pins
    }

    /// Every pin the listing carries, `(name, tag, sha256)`.
    fn all_pins(&self) -> Vec<(String, String, String)> {
        let mut pins: Vec<(String, String, String)> = self
            .robots
            .values()
            .flat_map(|robot| {
                robot
                    .contracts
                    .iter()
                    .map(|c| (c.name.clone(), c.tag.clone(), c.sha256.clone()))
            })
            .chain(
                self.option_pins
                    .iter()
                    .map(|p| (p.name.clone(), p.tag.clone(), p.sha256.clone())),
            )
            .collect();
        pins.sort();
        pins.dedup();
        pins
    }
}

/// The listing of the generated `list` response and of the generated
/// `changes` feedback, which carry the same fields under two generated
/// types.
macro_rules! fleet_of {
    ($listing:expr) => {{
        let listing = $listing;
        Fleet {
            version: listing.version.clone(),
            core_node: listing.core_node.clone(),
            halted: listing.halted,
            robots: listing
                .robots
                .iter()
                .map(|r| {
                    let robot = Robot {
                        name: r.name.clone(),
                        core_node: r.core_node.clone(),
                        kind: r.kind.clone(),
                        clock: r.clock.clone(),
                        software: r.software.clone(),
                        robot_commander: r.robot_commander.clone(),
                        instances: r
                            .instances
                            .iter()
                            .map(|i| Instance {
                                name: i.name.clone(),
                                instance: i.instance.clone(),
                                core_node: i.core_node.clone(),
                            })
                            .collect(),
                        contracts: r
                            .contracts
                            .iter()
                            .map(|c| RobotContract {
                                name: c.name.clone(),
                                tag: c.tag.clone(),
                                sha256: c.sha256.clone(),
                                instances: c.instances.clone(),
                            })
                            .collect(),
                    };
                    (robot.name.clone(), robot)
                })
                .collect(),
            leases: listing
                .leases
                .iter()
                .map(|l| Lease {
                    robot: l.robot.clone(),
                    holder: l.holder.clone(),
                    holder_core_node: l.holder_core_node.clone(),
                    name: l.name.clone(),
                    open: l.open,
                    receiver: l.receiver.clone(),
                    receiver_core_node: l.receiver_core_node.clone(),
                })
                .collect(),
            nodes: listing
                .nodes
                .iter()
                .map(|n| Node {
                    name: n.name.clone(),
                    tag: n.tag.clone(),
                    machines: n
                        .machines
                        .iter()
                        .map(|m| Machine {
                            core_node: m.core_node.clone(),
                            manifest_sha256: m.manifest_sha256.clone(),
                            commit: m.commit.clone(),
                        })
                        .collect(),
                    instances: n
                        .instances
                        .iter()
                        .map(|i| (i.instance.clone(), i.core_node.clone(), i.robot.clone()))
                        .collect(),
                })
                .collect(),
            option_pins: listing
                .options
                .iter()
                .flat_map(|o| {
                    o.contracts
                        .iter()
                        .map(|c| OptionPin {
                            name: c.name.clone(),
                            tag: c.tag.clone(),
                            sha256: c.sha256.clone(),
                        })
                        .chain(o.features.iter().flat_map(|f| {
                            f.contracts.iter().map(|c| OptionPin {
                                name: c.name.clone(),
                                tag: c.tag.clone(),
                                sha256: c.sha256.clone(),
                            })
                        }))
                        .collect::<Vec<_>>()
                })
                .collect(),
            documents: listing
                .documents
                .iter()
                .map(|d| {
                    (
                        (d.name.clone(), d.tag.clone(), d.sha256.clone()),
                        d.document.clone(),
                    )
                })
                .collect(),
        }
    }};
}

/// A snapshot of the fleet with the pinned documents of its contracts.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub fleet: Fleet,
    /// Each pinned document of the listing, by sha256.
    pub contracts: BTreeMap<String, Arc<ContractShape>>,
    /// The documents that did not parse, by sha256, with why.
    pub unreadable: BTreeMap<String, String>,
    /// The model of each robot that gave it, by robot name.
    pub models: BTreeMap<String, String>,
    /// The links of each robot whose description gave them, as its URDF
    /// names them, by robot name.
    pub links: BTreeMap<String, RobotLinks>,
    /// The robots whose description gave no links, by name: the director
    /// asks such a robot again when the listing gives its entry otherwise.
    pub unread_links: BTreeSet<String>,
    /// Whether the director has read the listing yet.
    pub ready: bool,
}

/// Reads the fleet listing now.
pub async fn read_fleet(runner: &NodeRunner) -> peppygen::Result<Fleet> {
    let daemon = list::bound_producer(runner);
    let response = list::poll(runner, daemon, DAEMON_TIMEOUT).await?;
    Ok(fleet_of!(&response.data))
}

/// The snapshot of `fleet`, each pin of which `known` lacks fetched with
/// `fleet.contract` (or taken from the documents of the listing) and parsed.
async fn snapshot_of(
    runner: &NodeRunner,
    fleet: Fleet,
    known: &BTreeMap<String, Arc<ContractShape>>,
) -> Snapshot {
    let mut contracts = BTreeMap::new();
    let mut unreadable = BTreeMap::new();
    for (name, tag, sha256) in fleet.all_pins() {
        if let Some(shape) = known.get(&sha256) {
            contracts.insert(sha256, Arc::clone(shape));
            continue;
        }
        let document = match fleet
            .documents
            .get(&(name.clone(), tag.clone(), sha256.clone()))
        {
            Some(document) => Ok(document.clone()),
            None => fetch_document(runner, &name, &tag, &sha256).await,
        };
        match document.and_then(|document| ContractShape::parse(&document, &sha256)) {
            Ok(shape) => {
                contracts.insert(sha256, Arc::new(shape));
            }
            Err(reason) => {
                unreadable.insert(sha256, format!("{name}:{tag}: {reason}"));
            }
        }
    }
    Snapshot {
        fleet,
        contracts,
        unreadable,
        models: BTreeMap::new(),
        links: BTreeMap::new(),
        unread_links: BTreeSet::new(),
        ready: true,
    }
}

/// The links of a robot as its URDF names them: its root link, the robot
/// frame, and the others.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RobotLinks {
    pub root: String,
    pub others: BTreeSet<String>,
}

impl RobotLinks {
    /// The links that `urdf` names, its root link the one link that no joint
    /// names as its child.
    fn of_urdf(urdf: &str) -> Result<RobotLinks, String> {
        let robot = urdf_rs::read_from_string(urdf)
            .map_err(|error| format!("its URDF does not read: {error}"))?;
        let children: BTreeSet<&str> = robot
            .joints
            .iter()
            .map(|joint| joint.child.link.as_str())
            .collect();
        let (roots, others): (Vec<String>, Vec<String>) = robot
            .links
            .into_iter()
            .map(|link| link.name)
            .partition(|name| !children.contains(name.as_str()));
        let [root] = <[String; 1]>::try_from(roots).map_err(|roots| {
            format!(
                "its URDF has {} root links, where a robot has one",
                roots.len()
            )
        })?;
        Ok(RobotLinks {
            root,
            others: others.into_iter().collect(),
        })
    }
}

/// The snapshot of `fleet` with the documents, the models and the links of
/// its robots, each taken from `latest` when it holds it.
pub async fn snapshot_described(runner: &NodeRunner, fleet: Fleet, latest: &Snapshot) -> Snapshot {
    let mut snapshot = snapshot_of(runner, fleet, &latest.contracts).await;
    snapshot.models = models_of(runner, &snapshot, &latest.models).await;
    (snapshot.links, snapshot.unread_links) = links_of(runner, &snapshot, latest).await;
    snapshot
}

/// The model of each robot of `snapshot`: from `known` for a robot of the
/// same name, else from `robot_identity.get_identity` of the robot.
async fn models_of(
    runner: &NodeRunner,
    snapshot: &Snapshot,
    known: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut models = BTreeMap::new();
    for robot in snapshot.fleet.robots.values() {
        if let Some(model) = known.get(&robot.name) {
            models.insert(robot.name.clone(), model.clone());
            continue;
        }
        match model_of(runner, snapshot, robot).await {
            Ok(model) => {
                models.insert(robot.name.clone(), model);
            }
            Err(reason) => tracing::info!("the model of {}: {reason}", robot.name),
        }
    }
    models
}

async fn model_of(
    runner: &NodeRunner,
    snapshot: &Snapshot,
    robot: &Robot,
) -> Result<String, String> {
    let response =
        identity_answer(runner, snapshot, robot, "get_identity", IDENTITY_TIMEOUT).await?;
    response["model"]
        .as_str()
        .map(str::to_owned)
        .ok_or("get_identity gave no model".to_owned())
}

/// The links of each robot of `snapshot`, and the robots whose description
/// gave none: as `latest` knows them where [`known_links`] says, else from
/// the URDF that `robot_identity.get_description` of the robot gives.
async fn links_of(
    runner: &NodeRunner,
    snapshot: &Snapshot,
    latest: &Snapshot,
) -> (BTreeMap<String, RobotLinks>, BTreeSet<String>) {
    let mut links = BTreeMap::new();
    let mut unread = BTreeSet::new();
    for robot in snapshot.fleet.robots.values() {
        match known_links(latest, robot) {
            KnownLinks::Read(known) => {
                links.insert(robot.name.clone(), known.clone());
            }
            KnownLinks::Unread => {
                unread.insert(robot.name.clone());
            }
            KnownLinks::Unasked => match links_of_robot(runner, snapshot, robot).await {
                Ok(of_robot) => {
                    links.insert(robot.name.clone(), of_robot);
                }
                Err(reason) => {
                    tracing::info!("the links of {}: {reason}", robot.name);
                    unread.insert(robot.name.clone());
                }
            },
        }
    }
    (links, unread)
}

/// What a snapshot knows of the links of a robot.
#[derive(Debug, PartialEq)]
enum KnownLinks<'s> {
    /// The links its description gave.
    Read(&'s RobotLinks),
    /// Its description gave none.
    Unread,
    /// The director asks the robot.
    Unasked,
}

/// What `latest` knows of the links of `robot`, as the listing gives it
/// now: for a robot whose entry is unchanged, the links its description gave
/// or that it gave none; for a new robot, or one whose entry changed, as at a
/// repin or a new model, nothing.
fn known_links<'s>(latest: &'s Snapshot, robot: &Robot) -> KnownLinks<'s> {
    if latest.fleet.robots.get(&robot.name) != Some(robot) {
        return KnownLinks::Unasked;
    }
    match latest.links.get(&robot.name) {
        Some(links) => KnownLinks::Read(links),
        None if latest.unread_links.contains(&robot.name) => KnownLinks::Unread,
        None => KnownLinks::Unasked,
    }
}

async fn links_of_robot(
    runner: &NodeRunner,
    snapshot: &Snapshot,
    robot: &Robot,
) -> Result<RobotLinks, String> {
    let response = identity_answer(
        runner,
        snapshot,
        robot,
        "get_description",
        DESCRIPTION_TIMEOUT,
    )
    .await?;
    let urdf = response["urdf"]
        .as_str()
        .ok_or("get_description gave no URDF")?;
    RobotLinks::of_urdf(urdf)
}

/// The answer of `member` of the one robot_identity:v1 instance of `robot`,
/// within `bound`.
async fn identity_answer(
    runner: &NodeRunner,
    snapshot: &Snapshot,
    robot: &Robot,
    member: &str,
    bound: Duration,
) -> Result<serde_json::Value, String> {
    let (name, tag) = ROBOT_IDENTITY;
    let [instance] = robot.instances_of(name, tag)[..] else {
        return Err("it has no single instance of robot_identity:v1".to_owned());
    };
    call_skill(
        runner,
        snapshot,
        robot,
        ROBOT_IDENTITY,
        instance,
        member,
        &serde_json::json!({}),
        bound,
    )
    .await
    .map_err(|error| error.message)
}

/// The answer of `teach` and of `list` of locations:v1.
#[derive(Deserialize)]
struct KnownLocations {
    locations: Vec<KnownLocation>,
}

/// Calls `member` of the locations:v1 `instance` of `robot` with `request`
/// (`list`, or `teach` with the locations it teaches), within
/// [`SKILL_TIMEOUT`], and gives the locations the skill knows after.
pub async fn known_locations(
    runner: &NodeRunner,
    snapshot: &Snapshot,
    robot: &Robot,
    instance: &Instance,
    member: &str,
    request: &serde_json::Value,
) -> Result<Vec<KnownLocation>, SkillCallError> {
    let answer = call_skill(
        runner,
        snapshot,
        robot,
        LOCATIONS,
        instance,
        member,
        request,
        SKILL_TIMEOUT,
    )
    .await?;
    serde_json::from_value::<KnownLocations>(answer)
        .map(|known| known.locations)
        .map_err(|error| {
            SkillCallError::of(format!("{member} gave locations that do not read: {error}"))
        })
}

/// A call of a skill that did not answer: the failure kind of a refusal,
/// and the message.
#[derive(Clone, Debug)]
pub struct SkillCallError {
    pub kind: Option<String>,
    pub message: String,
}

impl SkillCallError {
    fn of(message: String) -> SkillCallError {
        SkillCallError {
            kind: None,
            message,
        }
    }
}

impl std::fmt::Display for SkillCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// Calls the service `member` of the contract `(name, tag)` that `robot`
/// implements on `instance`, through the codec API, with `request`, within
/// `bound`.
#[allow(clippy::too_many_arguments)]
async fn call_skill(
    runner: &NodeRunner,
    snapshot: &Snapshot,
    robot: &Robot,
    (name, tag): (&str, &str),
    instance: &Instance,
    member: &str,
    request: &serde_json::Value,
    bound: Duration,
) -> Result<serde_json::Value, SkillCallError> {
    let pin = robot.pin_of(name, tag, &instance.name).ok_or_else(|| {
        SkillCallError::of(format!(
            "the instance `{}` of `{}` does not implement {name}:{tag}",
            instance.name, robot.name
        ))
    })?;
    let shape = snapshot
        .contracts
        .get(&pin.sha256)
        .ok_or_else(|| SkillCallError::of(format!("the document of {name}:{tag} did not read")))?;
    let client = ServiceClient::of_contract(&shape.parsed, member)
        .map_err(|e| SkillCallError::of(e.to_string()))?;
    let binding = MemberBinding::of_contract(&shape.parsed, member)
        .map_err(|e| SkillCallError::of(e.to_string()))?;
    let processor = runner.processor();
    let identity = ConsumerIdentity {
        core_node: processor.bound_core_node().to_owned(),
        instance_id: processor.bound_instance_id().to_owned(),
    };
    let producer = ProducerRef::new(instance.core_node.clone(), instance.instance.clone());
    let call = client.call(
        runner.messenger(),
        &identity,
        &binding,
        &producer,
        request,
        None,
    );
    match tokio::time::timeout(bound, call).await {
        Err(_) => Err(SkillCallError::of(format!(
            "{member} gave no answer within {} s",
            bound.as_secs()
        ))),
        Ok(Ok(response)) => Ok(response),
        Ok(Err(error)) => Err(SkillCallError {
            kind: match &error {
                ConsumerError::Messaging(error) => error.failure().map(|f| f.kind().to_string()),
                _ => None,
            },
            message: error.to_string(),
        }),
    }
}

async fn fetch_document(
    runner: &NodeRunner,
    name: &str,
    tag: &str,
    sha256: &str,
) -> Result<String, String> {
    let daemon = contract::bound_producer(runner);
    let request = contract::Request::new(name.to_owned(), tag.to_owned(), sha256.to_owned());
    contract::poll(runner, daemon, DAEMON_TIMEOUT, request)
        .await
        .map(|response| response.data.document)
        .map_err(|error| error.to_string())
}

/// Follows the fleet with a `fleet.changes` goal for the life of the node,
/// and publishes a snapshot at each change; a lost goal starts again.
pub async fn follow(runner: Arc<NodeRunner>, snapshots: watch::Sender<Arc<Snapshot>>) {
    loop {
        let daemon = changes::bound_producer(&runner);
        let goal = changes::ActionHandle::fire_goal(
            &runner,
            daemon,
            DAEMON_TIMEOUT,
            peppygen::DEFAULT_FEEDBACK_BOUND,
        )
        .await;
        let mut goal = match goal {
            Ok(goal) if goal.accepted => goal,
            Ok(goal) => {
                tracing::warn!("the daemon refused fleet.changes: {:?}", goal.refusal);
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
            Err(error) => {
                tracing::warn!("fleet.changes did not start: {error}");
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
        };
        loop {
            let feedback = match goal.on_next_feedback_message().await {
                Ok(feedback) => feedback,
                Err(error) => {
                    tracing::warn!("fleet.changes ended: {error}");
                    break;
                }
            };
            let latest = Arc::clone(&snapshots.borrow());
            let snapshot = snapshot_described(&runner, fleet_of!(&feedback.listing), &latest).await;
            snapshots.send_replace(Arc::new(snapshot));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{LOCATIONS_PIN, robot};

    /// The links of a robot are the links its URDF names: its root link, the
    /// one no joint names as its child, and the others. A URDF with two
    /// roots, or one that does not read, gives none.
    #[test]
    fn the_links_of_a_robot_are_the_root_and_the_other_links_of_its_urdf() {
        let urdf = r#"<robot name="arm">
            <link name="grasp_link"/>
            <link name="base_link"/>
            <link name="arm_link"/>
            <joint name="shoulder" type="fixed">
              <parent link="base_link"/>
              <child link="arm_link"/>
            </joint>
            <joint name="wrist" type="fixed">
              <parent link="arm_link"/>
              <child link="grasp_link"/>
            </joint>
          </robot>"#;
        assert_eq!(
            RobotLinks::of_urdf(urdf),
            Ok(RobotLinks {
                root: "base_link".to_owned(),
                others: ["arm_link", "grasp_link"].map(str::to_owned).into(),
            })
        );
        let two_roots = urdf.replace(
            r#"<joint name="wrist" type="fixed">
              <parent link="arm_link"/>
              <child link="grasp_link"/>
            </joint>"#,
            "",
        );
        assert_eq!(
            RobotLinks::of_urdf(&two_roots),
            Err("its URDF has 2 root links, where a robot has one".to_owned())
        );
        assert!(
            RobotLinks::of_urdf("<robot")
                .unwrap_err()
                .starts_with("its URDF does not read")
        );
    }

    /// A robot is not asked for its links again while the listing gives its
    /// entry as it did: one whose links were read keeps them, and one whose
    /// description gave none stays without. A robot whose entry changes, as at
    /// a stop or a repin, or a new one, is asked.
    #[test]
    fn a_robot_is_asked_for_its_links_again_when_its_entry_changes() {
        let mut latest = crate::testing::snapshot(vec![robot("arm"), robot("eyes")]);
        latest.links.remove("arm");
        latest.unread_links.insert("arm".to_owned());
        assert_eq!(known_links(&latest, &robot("arm")), KnownLinks::Unread);
        let stopped = Robot {
            software: "stopped".to_owned(),
            ..robot("arm")
        };
        assert_eq!(known_links(&latest, &stopped), KnownLinks::Unasked);
        let mut repinned = robot("arm");
        repinned.contracts[0].sha256 = "b".repeat(64);
        assert_eq!(known_links(&latest, &repinned), KnownLinks::Unasked);
        assert_eq!(known_links(&latest, &robot("hand")), KnownLinks::Unasked);
        assert_eq!(
            known_links(&latest, &robot("eyes")),
            KnownLinks::Read(&latest.links["eyes"])
        );
        let mut repinned_eyes = robot("eyes");
        repinned_eyes.contracts[0].sha256 = "b".repeat(64);
        assert_eq!(known_links(&latest, &repinned_eyes), KnownLinks::Unasked);
    }

    /// Each instance of a contract that a robot lists at two pins serves the
    /// pin of its own entry, and an instance that serves none has none.
    #[test]
    fn each_instance_serves_the_pin_of_its_own_entry() {
        let other = "b".repeat(64);
        let mut arm = robot("arm");
        arm.contracts.push(RobotContract {
            name: "locations".to_owned(),
            tag: "v1".to_owned(),
            sha256: other.clone(),
            instances: vec!["left_skill".to_owned()],
        });
        let pin = |instance: &str| {
            arm.pin_of("locations", "v1", instance)
                .map(|pin| pin.sha256.clone())
        };
        assert_eq!(pin("brain").as_deref(), Some(LOCATIONS_PIN));
        assert_eq!(pin("left_skill"), Some(other));
        assert_eq!(pin("eyes"), None);
    }
}
