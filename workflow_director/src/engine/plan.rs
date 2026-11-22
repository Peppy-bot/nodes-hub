//! What a run runs: a workflow bound to the robots of the stack, each call
//! and observation resolved to the producer that serves it, each argument
//! in the canonical form of its field, and the type of each value an
//! expression reads. The checker makes it; the engine runs it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value as Json;
use workflow_documents::expression::{Expression, PathStep, RefPath, Type};

use super::failure::{PRECONDITION_FAILED, REJECTED};
use crate::frames::{BETWEEN_ROBOTS_IN_WORLD, FrameOfValues, Placed};
use workflow_documents::run::{Attendance, RunDocument, Severity};
use workflow_documents::work_cell::ParameterKind;
use workflow_documents::{Bound, Name, RobotName, StepId, Validation, Workflow};

/// The node instance that serves a call or an observation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Producer {
    pub core_node: String,
    pub instance: String,
}

/// Whether a member is a service or an action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemberKind {
    Service,
    Action,
}

/// A member of a contract on one producer.
#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    pub role: Name,
    /// The contract, `name:tag`, and its pin.
    pub contract: String,
    pub sha256: String,
    pub member: Name,
    pub kind: MemberKind,
    pub producer: Producer,
    /// Whether the member has the mark moves_robot, so the director asks
    /// `fleet.check_move` before each call.
    pub moves_robot: bool,
    /// The field of the feedback of an action that holds the sentence of
    /// what the robot does now (the mark `activity`), so the call of a step
    /// gives an event `step_reports` for each new sentence; `None` for a
    /// service, an observation, the monitor, and an action with no such
    /// field.
    pub activity: Option<String>,
}

/// A topic of one producer: its instance, its contract and its name.
pub type TopicKey = (String, String, String);

impl Target {
    pub fn topic_key(&self) -> TopicKey {
        (
            self.producer.instance.clone(),
            self.contract.clone(),
            self.member.to_string(),
        )
    }
}

/// An argument of a call or an observation, its literals in the canonical
/// form of their fields.
#[derive(Clone, Debug, PartialEq)]
pub enum BoundArg {
    Literal(Json),
    Object(BTreeMap<String, BoundArg>),
    List(Vec<BoundArg>),
    Ref(RefPath),
    /// The timestamp at the path, of the record whose values a request
    /// takes: a record that leaves it out, as a report does for values that
    /// hold at every time, leaves it out of the object that holds the
    /// argument too, which a request reads as the latest time.
    Timestamp(RefPath),
    /// The frame of the record whose values a request in one frame takes.
    Frame(FrameOfValues),
    /// The records that name their frame which a `$ref` gives a field.
    Placed(Placed),
    /// The row that the key selects, each row in the canonical form of the
    /// field it goes to.
    Table {
        rows: BTreeMap<String, Json>,
        key: Box<BoundArg>,
    },
}

impl BoundArg {
    /// The root of each path that the argument reads.
    pub fn roots(&self) -> BTreeSet<Name> {
        match self {
            BoundArg::Literal(_) => BTreeSet::new(),
            BoundArg::Object(fields) => fields.values().flat_map(BoundArg::roots).collect(),
            BoundArg::List(items) => items.iter().flat_map(BoundArg::roots).collect(),
            BoundArg::Ref(path) | BoundArg::Timestamp(path) => BTreeSet::from([path.root.clone()]),
            BoundArg::Frame(frame) => BTreeSet::from([frame.frame.root.clone()]),
            BoundArg::Placed(placed) => placed.roots(),
            BoundArg::Table { key, .. } => key.roots(),
        }
    }
}

/// A call step or a call of a `safe_state`, bound.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundCall {
    pub target: Target,
    pub args: BTreeMap<String, BoundArg>,
    /// The type of the response or result, for the expressions that read it.
    pub result_type: Type,
}

/// What an observation reads.
#[derive(Clone, Debug, PartialEq)]
pub enum ObservationRead {
    Service { args: BTreeMap<String, BoundArg> },
    Topic { max_age: Bound },
}

/// An observation, bound.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundObservation {
    pub target: Target,
    pub read: ObservationRead,
    pub ty: Type,
}

/// A role and the robot that the run binds to it.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundRole {
    pub robot: RobotName,
    pub physical: bool,
}

/// The anomaly monitor of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct MonitorPlan {
    pub target: Target,
    pub continue_on_severity: BTreeSet<Severity>,
}

/// The locations that the calls of a role give one skill instance, which
/// the instance knows before the run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillLocations {
    pub role: Name,
    pub producer: Producer,
    pub names: BTreeSet<Name>,
}

/// The watch parameters of a run, which the engine reads through the robots
/// before the assumptions: the observation that gives the value of each, and
/// the validation of the bundle of the run, whose envelope says which values
/// a test covered.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Watch {
    pub observations: BTreeMap<Name, Expression>,
    pub validation: Option<Validation>,
}

impl Watch {
    /// The watch parameters of `document` and those of the validation of
    /// its bundle.
    pub fn of(document: &RunDocument, validation: Option<&Validation>) -> Watch {
        let of_the_run = document
            .watch
            .iter()
            .map(|(name, watch)| (name.clone(), watch.observation.clone()));
        let of_the_bundle = validation
            .into_iter()
            .flat_map(|validation| &validation.work_cell_parameters)
            .filter_map(|(name, parameter)| match &parameter.kind {
                ParameterKind::Watch { observation } => Some((name.clone(), observation.clone())),
                ParameterKind::Control | ParameterKind::Distance { .. } => None,
            });
        Watch {
            observations: of_the_run.chain(of_the_bundle).collect(),
            validation: validation.cloned(),
        }
    }
}

/// A call step of a role.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CallSite {
    pub role: Name,
    pub step: StepId,
}

/// A workflow bound to the robots of the stack.
#[derive(Clone, Debug)]
pub struct Plan {
    pub workflow: Arc<Workflow>,
    pub attendance: Attendance,
    pub confirmed: BTreeSet<Name>,
    pub roles: BTreeMap<Name, BoundRole>,
    pub calls: BTreeMap<CallSite, BoundCall>,
    pub observations: BTreeMap<Name, BoundObservation>,
    /// The type of each name of the trees of each role: results, loop
    /// elements and signal results.
    pub locals: BTreeMap<Name, BTreeMap<Name, Type>>,
    /// The value and the type of each table.
    pub tables: BTreeMap<Name, (Json, Type)>,
    pub monitor: Option<MonitorPlan>,
    /// The locations each skill instance of the run must know.
    pub locations: Vec<SkillLocations>,
    /// The poll interval of a step without a `poll_s` of its own.
    pub poll: Bound,
    /// How long, on wall time, a cancelled goal has to end.
    pub cancel_timeout: Duration,
    pub watch: Watch,
    /// Whether the run is a run of a trial: the engine evaluates each check
    /// of its assumptions, a false one included, and waits for the
    /// judgement of the operator before the initialization.
    pub of_a_trial: bool,
    /// The roles whose literal records the checker held against the links
    /// of their robots, which the URDF of robot_identity:v1 gives.
    pub reads_links: BTreeSet<Name>,
}

impl Plan {
    /// Whether the run is unattended and binds a physical robot: it runs
    /// only inside the envelope of its bundle, as a sign-off record admits
    /// it.
    pub fn is_unattended_on_a_physical_robot(&self) -> bool {
        self.attendance == Attendance::Unattended && self.roles.values().any(|role| role.physical)
    }
}

/// Why a value at a path cannot be read.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("`{path}` has no value")]
pub struct NoValue {
    pub path: String,
}

/// Why the request of a call is not sent.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum Unsendable {
    #[error(transparent)]
    NoValue(#[from] NoValue),
    #[error(
        "`{path}` is in `{frame}`, a frame of the robot of `{from}`, and goes to the robot of `{to}`: {why}; give the call a point, a pose or a twist of a report of the robot of `{to}` by `$ref`",
        why = BETWEEN_ROBOTS_IN_WORLD
    )]
    OfAnotherRobot {
        path: String,
        frame: String,
        from: Name,
        to: Name,
    },
}

impl Unsendable {
    pub fn no_value(path: &RefPath) -> Unsendable {
        Unsendable::NoValue(NoValue {
            path: path.to_string(),
        })
    }

    /// The failure kind of a call whose request is not sent: a value the run
    /// does not hold yet fails its precondition, and a value of another
    /// robot's frame is a request no member takes.
    pub fn kind(&self) -> &'static str {
        match self {
            Unsendable::NoValue(_) => PRECONDITION_FAILED,
            Unsendable::OfAnotherRobot { .. } => REJECTED,
        }
    }
}

/// The value at `path` in `scope`.
pub fn read_path(scope: &BTreeMap<Name, Json>, path: &RefPath) -> Result<Json, NoValue> {
    let missing = || NoValue {
        path: path.to_string(),
    };
    let mut value = scope.get(&path.root).ok_or_else(missing)?;
    for step in &path.steps {
        value = match step {
            PathStep::Field(field) => value.get(field.as_str()),
            PathStep::Index(index) => value.get(*index as usize),
        }
        .ok_or_else(missing)?;
    }
    Ok(value.clone())
}

/// The field at `path` in `scope`, of the record that its last step leaves:
/// `None` when the record leaves the field out. A path whose last step names
/// no field is read whole.
pub fn read_field(scope: &BTreeMap<Name, Json>, path: &RefPath) -> Result<Option<Json>, NoValue> {
    let Some((PathStep::Field(field), record_steps)) = path.steps.split_last() else {
        return read_path(scope, path).map(Some);
    };
    let record = RefPath {
        root: path.root.clone(),
        steps: record_steps.to_vec(),
    };
    Ok(read_path(scope, &record)?.get(field.as_str()).cloned())
}

/// The JSON of `arg` in `scope`.
fn resolve(arg: &BoundArg, scope: &BTreeMap<Name, Json>) -> Result<Json, Unsendable> {
    match arg {
        BoundArg::Literal(value) => Ok(value.clone()),
        BoundArg::Object(fields) => {
            let resolved = fields
                .iter()
                .map(|(key, arg)| Ok(resolve_field(arg, scope)?.map(|value| (key.clone(), value))))
                .collect::<Result<Vec<_>, Unsendable>>()?;
            Ok(Json::Object(resolved.into_iter().flatten().collect()))
        }
        BoundArg::List(items) => items
            .iter()
            .map(|arg| resolve(arg, scope))
            .collect::<Result<Vec<_>, _>>()
            .map(Json::Array),
        BoundArg::Ref(path) | BoundArg::Timestamp(path) => Ok(read_path(scope, path)?),
        BoundArg::Frame(frame) => frame.resolve(scope),
        BoundArg::Placed(placed) => placed.resolve(scope),
        BoundArg::Table { rows, key } => {
            let key = resolve(key, scope)?;
            key.as_str()
                .and_then(|key| rows.get(key))
                .cloned()
                .ok_or_else(|| {
                    Unsendable::NoValue(NoValue {
                        path: format!("the row {key} of a table"),
                    })
                })
        }
    }
}

/// The JSON of `arg`, a field of an object, in `scope`: `None` for a
/// timestamp that its record leaves out.
fn resolve_field(arg: &BoundArg, scope: &BTreeMap<Name, Json>) -> Result<Option<Json>, Unsendable> {
    match arg {
        BoundArg::Timestamp(path) => Ok(read_field(scope, path)?),
        arg => resolve(arg, scope).map(Some),
    }
}

/// The JSON object of `args` in `scope`.
pub fn request(
    args: &BTreeMap<String, BoundArg>,
    scope: &BTreeMap<Name, Json>,
) -> Result<Json, Unsendable> {
    resolve(&BoundArg::Object(args.clone()), scope)
}
