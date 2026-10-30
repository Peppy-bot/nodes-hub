//! What a run runs: a workflow bound to the robots of the stack, each call
//! and observation resolved to the producer that serves it, each argument
//! in the canonical form of its field, and the type of each value an
//! expression reads. The checker makes it; the engine runs it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value as Json;
use workflow_documents::expression::{PathStep, RefPath, Type};
use workflow_documents::run::{Attendance, Severity};
use workflow_documents::{Bound, Name, RobotName, StepId, Workflow};

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
    /// The row that the key selects, each row in the canonical form of the
    /// field it goes to.
    Table {
        rows: BTreeMap<String, Json>,
        key: Box<BoundArg>,
    },
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
}

/// Why a value at a path cannot be read.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("`{path}` has no value")]
pub struct NoValue {
    pub path: String,
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

/// The JSON of `arg` in `scope`.
fn resolve(arg: &BoundArg, scope: &BTreeMap<Name, Json>) -> Result<Json, NoValue> {
    match arg {
        BoundArg::Literal(value) => Ok(value.clone()),
        BoundArg::Object(fields) => fields
            .iter()
            .map(|(key, arg)| resolve(arg, scope).map(|value| (key.clone(), value)))
            .collect::<Result<serde_json::Map<_, _>, _>>()
            .map(Json::Object),
        BoundArg::List(items) => items
            .iter()
            .map(|arg| resolve(arg, scope))
            .collect::<Result<Vec<_>, _>>()
            .map(Json::Array),
        BoundArg::Ref(path) => read_path(scope, path),
        BoundArg::Table { rows, key } => {
            let key = resolve(key, scope)?;
            key.as_str()
                .and_then(|key| rows.get(key))
                .cloned()
                .ok_or_else(|| NoValue {
                    path: format!("the row {key} of a table"),
                })
        }
    }
}

/// The JSON object of `args` in `scope`.
pub fn request(
    args: &BTreeMap<String, BoundArg>,
    scope: &BTreeMap<Name, Json>,
) -> Result<Json, NoValue> {
    resolve(&BoundArg::Object(args.clone()), scope)
}
