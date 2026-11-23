//! The checker: the rules of a workflow that read its contracts, and the
//! binding of its roles to robots of the fleet listing. Checked with roles,
//! a workflow becomes the [`Plan`] that the engine runs.
//!
//! Without roles, each alias is checked against every pin of its contract
//! in the fleet listing. With roles, each role reads the pins of its robot,
//! each call and observation resolves to the instance that serves it, and
//! the item ids that a workflow passes are proven to come from the instance
//! that takes them. Each point, pose and twist that a workflow passes names
//! its frame and its time as [`crate::frames`] says.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use peppy_codec::contract::MOVES_ROBOT;
use serde_json::Value as Json;
use workflow_documents::expression::{
    Expr, Expression, FieldType, PathStep, RefPath, Type, TypeScope,
};
use workflow_documents::marks::{CHECKS_REACH, ITEM_ID, LOCATION, REACHES_FOR_ITEM};
use workflow_documents::run::{Attendance, Incidents};
use workflow_documents::workflow::{Arg, Call, ObservationSource, Step, StepKind};
use workflow_documents::{Located, Name, NameTag, RobotName, StepId, Workflow, robots_named_twice};

use crate::config::AllowedContracts;
use crate::contract::{ContractShape, Field, FieldKind, Format, Member};
use crate::engine::plan::{
    BoundArg, BoundCall, BoundObservation, BoundRole, CallSite, MemberKind, MonitorPlan,
    ObservationRead, Plan, Producer, SkillLocations, Target, Watch,
};
use crate::fleet::{LOCATIONS, Robot, Snapshot};
use crate::frames::{self, Crossing, FrameOfValues, Holder, Holding, Placed};

/// The run-wide choices that a plan carries beside its workflow.
pub struct RunChoices<'a> {
    pub attendance: Attendance,
    pub confirmed: BTreeSet<Name>,
    pub incidents: &'a Incidents,
    pub poll: workflow_documents::Bound,
    pub cancel_timeout: std::time::Duration,
    pub watch: Watch,
    pub of_a_trial: bool,
}

/// Where each name of a role comes from: the role whose call or
/// observation gives it, the producer of the result or the observation, the
/// format of its value, its source: the name of the result or the
/// observation, which a loop element takes from the list of its loop, and
/// what holds the value.
#[derive(Clone, Debug)]
struct Origin {
    role: Name,
    producer: Option<Producer>,
    field: Field,
    source: Name,
    holding: Holding,
}

/// What a refusal of a literal record says to write.
const PASS_FROM_A_REPORT: &str = "pass the record from a report of a skill of the robot by `$ref`, which brings its frame and its timestamp";

/// The frames that a literal record takes, as its refusal names them.
const LITERAL_FRAMES: &str = "a literal names the robot frame, by its name or by leaving `frame` out, or a frame that the robot's tree holds above its robot frame, odom, map, or world where a node publishes it, and no node publishes one into a robot's tree";

/// The call that an argument goes to: its role, the alias of its contract,
/// and the producer that serves it once the role is bound.
struct Callee<'c> {
    role: &'c Name,
    alias: &'c Name,
    producer: Option<&'c Producer>,
}

struct Checker<'a> {
    workflow: &'a Workflow,
    snapshot: &'a Snapshot,
    allowed: &'a AllowedContracts,
    binding: Option<&'a BTreeMap<Name, RobotName>>,
    errors: Vec<Located>,
    calls: BTreeMap<CallSite, BoundCall>,
    observations: BTreeMap<Name, BoundObservation>,
    locals: BTreeMap<Name, BTreeMap<Name, Type>>,
    origins: BTreeMap<Name, BTreeMap<Name, Origin>>,
    /// Where each observation comes from, which every role reads.
    observation_origins: BTreeMap<Name, Origin>,
    tables: BTreeMap<Name, (Json, Type)>,
    /// The tables whose every row is a `{ $location }`, with the location of
    /// each row by its key, which a field with the mark location takes and
    /// no expression reads.
    location_tables: BTreeMap<Name, BTreeMap<String, Name>>,
    /// The locations each skill instance is given, by role.
    locations: Vec<SkillLocations>,
    /// The roles whose literal records were held against the links of their
    /// robots.
    reads_links: BTreeSet<Name>,
}

impl<'a> Checker<'a> {
    fn error(&mut self, at: impl Into<String>, message: impl Into<String>) {
        self.errors.push(Located {
            at: at.into(),
            message: message.into(),
        });
    }

    fn robot_of(&self, role: &Name) -> Option<&'a Robot> {
        let binding = self.binding?;
        self.snapshot.fleet.robots.get(binding.get(role)?.as_str())
    }

    /// The contract that `alias` names, as `name:tag`.
    fn contract_ref(&self, alias: &Name) -> Option<&'a NameTag> {
        self.workflow.contracts.get(alias)
    }

    /// The pins that the role reads for `alias` on `instance`: with roles,
    /// the pin of the instance of its robot that serves it (`instance`, or
    /// the one instance), or each pin of the robot when that names none;
    /// without roles, every pin of the listing.
    fn shapes(
        &self,
        role: &Name,
        alias: &Name,
        instance: Option<&workflow_documents::InstanceName>,
    ) -> Vec<Arc<ContractShape>> {
        let Some(contract) = self.contract_ref(alias) else {
            return Vec::new();
        };
        let (name, tag) = (contract.name.as_str(), contract.tag());
        let pins = match self.robot_of(role) {
            Some(robot) => {
                let served = robot.instances_of(name, tag);
                let serving = match (instance, served.as_slice()) {
                    (Some(named), _) => served.iter().find(|i| i.name == named.as_str()),
                    (None, [only]) => Some(only),
                    (None, _) => None,
                };
                match serving {
                    Some(serving) => robot
                        .pin_of(name, tag, &serving.name)
                        .map(|pin| vec![pin.sha256.clone()])
                        .unwrap_or_default(),
                    None => robot
                        .contracts
                        .iter()
                        .filter(|pin| pin.name == name && pin.tag == tag)
                        .map(|pin| pin.sha256.clone())
                        .collect(),
                }
            }
            None => self.snapshot.fleet.pins_of(name, tag),
        };
        pins.iter()
            .filter_map(|pin| self.snapshot.contracts.get(pin).cloned())
            .collect()
    }

    /// The producer that serves `alias` for `role`, with `instance` when the
    /// robot has several.
    fn producer(
        &mut self,
        at: &str,
        role: &Name,
        alias: &Name,
        instance: Option<&workflow_documents::InstanceName>,
    ) -> Option<Producer> {
        let robot = self.robot_of(role)?;
        let contract = self.contract_ref(alias)?;
        let instances = robot.instances_of(contract.name.as_str(), contract.tag());
        let chosen = match (instance, instances.as_slice()) {
            (Some(name), _) => instances.iter().find(|i| i.name == name.as_str()).copied(),
            (None, [only]) => Some(*only),
            (None, []) => None,
            (None, several) => {
                let names: Vec<&str> = several.iter().map(|i| i.name.as_str()).collect();
                self.error(at, format!(
                    "the robot `{}` has several instances of `{contract}` ({}); name one with `instance`",
                    robot.name,
                    names.join(", ")
                ));
                return None;
            }
        };
        match chosen {
            Some(instance) => Some(Producer {
                core_node: instance.core_node.clone(),
                instance: instance.instance.clone(),
            }),
            None => {
                self.error(
                    at,
                    format!(
                        "the robot `{}` has no instance `{}` of `{contract}`",
                        robot.name,
                        instance.map(|i| i.to_string()).unwrap_or_default()
                    ),
                );
                None
            }
        }
    }

    fn check_contracts(&mut self) {
        for (alias, contract) in &self.workflow.contracts {
            if !self.allowed.allows(contract) {
                self.error(
                    format!("contracts.{alias}"),
                    format!("`{contract}` is not an allowed contract of this director; the launcher names the contracts runs may command"),
                );
            }
            let pins = self
                .snapshot
                .fleet
                .pins_of(contract.name.as_str(), contract.tag());
            if self.binding.is_none() && pins.is_empty() {
                self.error(
                    format!("contracts.{alias}"),
                    format!("no robot or option of the fleet listing pins `{contract}`"),
                );
            }
            for reason in pins
                .iter()
                .filter_map(|pin| self.snapshot.unreadable.get(pin))
            {
                self.error(
                    format!("contracts.{alias}"),
                    format!("a pinned document of `{contract}` does not read: {reason}"),
                );
            }
        }
    }

    /// The binding rules: each role bound, to a present robot of its own,
    /// with the contracts and the model of the role and no robot commander.
    fn check_binding(&mut self, confirmed: &BTreeSet<Name>) {
        let Some(binding) = self.binding else {
            return;
        };
        let named = self
            .workflow
            .roles
            .keys()
            .filter_map(|role| binding.get_key_value(role));
        for refusal in robots_named_twice(named) {
            let Located { at, message } = refusal.located();
            self.error(at, message);
        }
        for role in self.workflow.roles.keys() {
            let Some(robot_name) = binding.get(role) else {
                self.error(
                    format!("roles.{role}"),
                    format!("binds no robot to the role `{role}`"),
                );
                continue;
            };
            let Some(robot) = self.snapshot.fleet.robots.get(robot_name.as_str()) else {
                self.error(
                    format!("roles.{role}"),
                    format!("`{robot_name}` is no robot of the fleet listing"),
                );
                continue;
            };
            let reasons = misfits(self.workflow, self.snapshot, role, robot, confirmed);
            for reason in reasons {
                self.error(format!("roles.{role}"), reason);
            }
        }
        for role in binding.keys() {
            if !self.workflow.roles.contains_key(role) {
                self.error(
                    format!("roles.{role}"),
                    format!("`{role}` is no role of the workflow"),
                );
            }
        }
    }

    /// The member `member` of each pin of `alias` for `role`, refusing one
    /// the contract does not have.
    fn members(
        &mut self,
        at: &str,
        role: &Name,
        member: &workflow_documents::MemberRef,
        instance: Option<&workflow_documents::InstanceName>,
    ) -> Vec<(Arc<ContractShape>, Member)> {
        let shapes = self.shapes(role, &member.alias, instance);
        let mut found = Vec::new();
        for shape in shapes {
            match shape.members.get(member.member.as_str()) {
                Some(m) => found.push((Arc::clone(&shape), m.clone())),
                None => self.error(
                    at,
                    format!("`{}` has no member `{}`", shape.name_tag(), member.member),
                ),
            }
        }
        found
    }

    fn check_observations(&mut self) {
        for (name, observation) in &self.workflow.observations {
            let at = format!("observations.{name}");
            let role = &observation.role;
            let named = observation.named();
            let shapes = self.shapes(role, &named.alias, observation.instance.as_ref());
            for shape in shapes {
                let read = match &observation.source {
                    ObservationSource::Member { member, args } => {
                        let Some(found) = shape.members.get(member.member.as_str()) else {
                            self.error(
                                &at,
                                format!(
                                    "`{}` has no service `{}`",
                                    shape.name_tag(),
                                    member.member
                                ),
                            );
                            continue;
                        };
                        if found.kind == MemberKind::Action {
                            self.error(&at, format!("`{member}` is an action; an observation reads a service or a topic"));
                            continue;
                        }
                        if found.marks.contains(MOVES_ROBOT) {
                            self.error(
                                &at,
                                format!("`{member}` moves the robot; an observation only reads"),
                            );
                            continue;
                        }
                        let callee = Callee {
                            role,
                            alias: &member.alias,
                            producer: None,
                        };
                        let owner = format!("`{member}`");
                        let bound = self.args(&at, &callee, &owner, args, &found.input);
                        let Some(producer) =
                            self.producer(&at, role, &member.alias, observation.instance.as_ref())
                        else {
                            self.record_observation(name, role, &found.output, None);
                            continue;
                        };
                        self.record_observation(name, role, &found.output, Some(producer.clone()));
                        bound.map(|args| {
                            (
                                ObservationRead::Service { args },
                                found.output.clone(),
                                MemberKind::Service,
                                producer,
                                member.member.clone(),
                            )
                        })
                    }
                    ObservationSource::Topic { topic, max_age } => {
                        let Some(format) = shape.topics.get(topic.member.as_str()).cloned() else {
                            self.error(
                                &at,
                                format!("`{}` has no topic `{}`", shape.name_tag(), topic.member),
                            );
                            continue;
                        };
                        let Some(producer) =
                            self.producer(&at, role, &topic.alias, observation.instance.as_ref())
                        else {
                            self.record_observation(name, role, &format, None);
                            continue;
                        };
                        self.record_observation(name, role, &format, Some(producer.clone()));
                        Some((
                            ObservationRead::Topic { max_age: *max_age },
                            format,
                            MemberKind::Service,
                            producer,
                            topic.member.clone(),
                        ))
                    }
                };
                if let Some((read, format, kind, producer, member)) = read {
                    self.observations.insert(
                        name.clone(),
                        BoundObservation {
                            target: Target {
                                role: role.clone(),
                                contract: shape.name_tag(),
                                sha256: shape.sha256.clone(),
                                member,
                                kind,
                                producer,
                                moves_robot: false,
                                activity: None,
                            },
                            read,
                            ty: format.ty(),
                        },
                    );
                }
            }
        }
    }

    fn record_observation(
        &mut self,
        name: &Name,
        role: &Name,
        format: &Format,
        producer: Option<Producer>,
    ) {
        self.observation_origins.insert(
            name.clone(),
            Origin {
                role: role.clone(),
                producer,
                field: Field {
                    kind: FieldKind::Object(format.clone()),
                    optional: false,
                    marks: BTreeSet::new(),
                },
                source: name.clone(),
                holding: Holding::Root,
            },
        );
    }

    fn observation_types(&self) -> BTreeMap<Name, Type> {
        self.observation_origins
            .iter()
            .map(|(name, origin)| (name.clone(), origin.field.ty()))
            .collect()
    }

    /// The types and the values of the tables: each row a literal, all rows
    /// of one type; or, for a table whose every row is a `{ $location }`,
    /// the location of each row, which no expression reads.
    fn check_tables(&mut self) {
        for (name, rows) in &self.workflow.tables {
            let locations: Option<BTreeMap<String, Name>> = rows
                .iter()
                .map(|(key, row)| Some((key.clone(), location_of_row(row)?)))
                .collect();
            if let Some(locations) = locations.filter(|locations| !locations.is_empty()) {
                self.location_tables.insert(name.clone(), locations);
                continue;
            }
            let mut row_type = None;
            for (key, row) in rows {
                let ty = literal_type(row);
                match (&row_type, ty) {
                    (_, None) => self.error(
                        format!("tables.{name}.{key}"),
                        "a row is a bool, a number, a string or a list of them, or `{ $location: \"<name>\" }` in a table whose every row is one",
                    ),
                    (None, Some(ty)) => row_type = Some(ty),
                    (Some(first), Some(ty))
                        if *first != ty && !(is_number(first) && is_number(&ty)) =>
                    {
                        self.error(
                            format!("tables.{name}.{key}"),
                            "every row of a table has one type",
                        )
                    }
                    _ => {}
                }
            }
            if let Some(row) = row_type {
                let keys: BTreeSet<String> = rows.keys().cloned().collect();
                let value =
                    Json::Object(rows.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
                self.tables.insert(
                    name.clone(),
                    (
                        value,
                        Type::Table {
                            keys,
                            row: Box::new(row),
                        },
                    ),
                );
            }
        }
    }

    /// The scope of the expressions of a role: its locals, the observations
    /// and the tables.
    fn scope_of(&self, role: &Name) -> TypeScope {
        let mut names = self.observation_types();
        for (table, (_, ty)) in &self.tables {
            names.insert(table.clone(), ty.clone());
        }
        if let Some(locals) = self.locals.get(role) {
            names.extend(locals.iter().map(|(n, t)| (n.clone(), t.clone())));
        }
        TypeScope::new(names)
    }

    fn typed(&mut self, at: &str, expression: &Expression, scope: &TypeScope, want: Want) {
        match expression.expr().infer(scope) {
            Ok(ty) => {
                let fits = match want {
                    Want::Bool => ty == Type::Bool,
                    Want::Number => is_number(&ty),
                    Want::Enum => matches!(ty, Type::Enum(_)),
                    Want::List => matches!(ty, Type::List(_)),
                };
                if !fits {
                    self.error(
                        at,
                        format!("`{}` is {ty}; {}", expression.source(), want.says()),
                    );
                }
            }
            Err(reason) => self.error(at, reason),
        }
    }

    /// Walks the trees of each role: binds each call, records the type and
    /// the origin of each local name, and types each expression.
    fn check_trees(&mut self) {
        let roles: Vec<Name> = self.workflow.roles.keys().cloned().collect();
        for role in roles {
            if let Some(init) = self.workflow.initialization.get(&role) {
                self.walk(&role, init);
            }
            if let Some(lane) = self.workflow.lanes.get(&role) {
                self.walk(&role, &lane.tree);
                for call in &lane.safe_state {
                    let at = format!("{role}.{}", call.id);
                    self.bind_call(
                        &at,
                        &role,
                        &call.id,
                        &call.member,
                        call.instance.as_ref(),
                        &call.args,
                    );
                }
            }
        }
        for role in self.workflow.roles.keys().cloned().collect::<Vec<_>>() {
            let scope = self.scope_of(&role);
            let workflow = self.workflow;
            for tree in workflow.trees_of(&role) {
                self.type_tree(&role, tree, &scope);
            }
        }
    }

    fn walk(&mut self, role: &Name, step: &'a Step) {
        let at = format!("{role}.{}", step.id);
        match &step.kind {
            StepKind::Call(call) => {
                if let Some(result_format) = self.bind_call(
                    &at,
                    role,
                    &step.id,
                    &call.member,
                    call.instance.as_ref(),
                    &call.args,
                ) && let Some(result) = &call.result
                {
                    self.locals
                        .entry(role.clone())
                        .or_default()
                        .insert(result.clone(), result_format.0.ty());
                    self.origins.entry(role.clone()).or_default().insert(
                        result.clone(),
                        Origin {
                            role: role.clone(),
                            producer: result_format.1,
                            field: Field {
                                kind: FieldKind::Object(result_format.0),
                                optional: false,
                                marks: BTreeSet::new(),
                            },
                            source: result.clone(),
                            holding: Holding::Root,
                        },
                    );
                }
            }
            StepKind::Sequence(steps) | StepKind::Parallel { steps, .. } => {
                steps.iter().for_each(|s| self.walk(role, s))
            }
            StepKind::If {
                then, otherwise, ..
            } => {
                self.walk(role, then);
                if let Some(o) = otherwise {
                    self.walk(role, o);
                }
            }
            StepKind::Match {
                cases, otherwise, ..
            } => {
                cases.values().for_each(|s| self.walk(role, s));
                if let Some(o) = otherwise {
                    self.walk(role, o);
                }
            }
            StepKind::ForEach {
                list,
                element,
                body,
                ..
            } => {
                let item = self.element_origin(role, list);
                match item {
                    Some(origin) => {
                        self.locals
                            .entry(role.clone())
                            .or_default()
                            .insert(element.clone(), origin.field.ty());
                        self.origins
                            .entry(role.clone())
                            .or_default()
                            .insert(element.clone(), origin);
                    }
                    None => self.error(
                        &at,
                        format!(
                            "`{}` is no path to a list of a result or an observation",
                            list.source()
                        ),
                    ),
                }
                self.walk(role, body);
            }
            StepKind::RepeatUntil { body, .. } => self.walk(role, body),
            StepKind::WaitFor {
                signals,
                result: Some(result),
            } => {
                let values: BTreeSet<String> = signals.iter().map(Name::to_string).collect();
                let ty = Type::Object(BTreeMap::from([(
                    "signal".to_owned(),
                    FieldType {
                        ty: Type::Enum(values),
                        optional: false,
                    },
                )]));
                self.locals
                    .entry(role.clone())
                    .or_default()
                    .insert(result.clone(), ty);
            }
            StepKind::WaitUntil { .. } | StepKind::Signal(_) | StepKind::WaitFor { .. } => {}
        }
    }

    /// The origin of each element of the list a `for_each` walks: the item of
    /// the list field at that path, held by the record that holds the list
    /// when that record is of a result. A loop reads the list of an
    /// observation from one message, which a call of its body does not read
    /// again, so its elements are held by no record a call reads.
    fn element_origin(&self, role: &Name, list: &Expression) -> Option<Origin> {
        let path = RefPath::parse(list.source()).ok()?;
        let origin = self.origin_of(role, &path)?;
        let of_a_result = self
            .origins
            .get(role)
            .is_some_and(|origins| origins.contains_key(&path.root));
        match origin.field.kind {
            FieldKind::List(item) => {
                let mut item = *item;
                item.marks.extend(origin.field.marks.iter().cloned());
                let holding = match of_a_result {
                    true => origin.holding,
                    false => Holding::ObservationElement { list: path },
                };
                Some(Origin {
                    field: item,
                    holding,
                    ..origin
                })
            }
            _ => None,
        }
    }

    /// The field that `path` reads, with the producer of its root and the
    /// record that holds it.
    fn origin_of(&self, role: &Name, path: &RefPath) -> Option<Origin> {
        let root = self
            .origins
            .get(role)
            .and_then(|o| o.get(&path.root))
            .or_else(|| self.observation_origins.get(&path.root))?
            .clone();
        let mut field = root.field;
        let mut holding = root.holding;
        let mut at = RefPath {
            root: path.root.clone(),
            steps: Vec::new(),
        };
        for step in &path.steps {
            field = match (step, field.kind) {
                (PathStep::Field(name), FieldKind::Object(format)) => {
                    let next = format.0.get(name)?.clone();
                    holding = Holding::Record(Holder {
                        path: at.clone(),
                        record: format,
                    });
                    next
                }
                (PathStep::Index(_), FieldKind::List(item)) => {
                    let mut item = *item;
                    item.marks.extend(field.marks.iter().cloned());
                    item
                }
                _ => return None,
            };
            at.steps.push(step.clone());
        }
        Some(Origin {
            field,
            holding,
            ..root
        })
    }

    /// The robots that a call of `callee` may reach: the robot of its role,
    /// or before a binding each robot of the listing that implements the
    /// contract of its alias.
    fn robots_of(&self, callee: &Callee) -> Vec<&'a Robot> {
        if self.binding.is_some() {
            return self.robot_of(callee.role).into_iter().collect();
        }
        let Some(contract) = self.contract_ref(callee.alias) else {
            return Vec::new();
        };
        self.snapshot
            .fleet
            .robots
            .values()
            .filter(|robot| {
                robot
                    .contract(contract.name.as_str(), contract.tag())
                    .is_some()
            })
            .collect()
    }

    /// Whether a literal record of a call of `callee` takes `frame`, the
    /// frame it names: the robot frame of each robot the call may reach, its
    /// root link, or a frame that is no link of it, as the URDF of its
    /// robot_identity:v1 names its links. An empty frame names no frame. For
    /// a robot whose links the director does not know, a named frame is
    /// refused, as the director cannot tell its robot frame from its other
    /// links; a literal with no frame names the robot frame without a name and
    /// comes here not at all.
    fn takes_literal_frame(&mut self, at: &str, callee: &Callee, frame: &str) -> bool {
        if frame.is_empty() {
            self.error(
                at,
                format!(
                    "a literal in an empty frame, which names no frame: {LITERAL_FRAMES}: {PASS_FROM_A_REPORT}"
                ),
            );
            return false;
        }
        self.reads_links.insert(callee.role.clone());
        for robot in self.robots_of(callee) {
            match self.snapshot.links.get(&robot.name) {
                None => {
                    self.error(at, format!(
                        "`{}` gives no URDF through robot_identity:v1, so the director cannot tell whether `{frame}` is its robot frame, another link of it or a frame its tree holds above its robot frame: leave `frame` out for the robot frame, or {PASS_FROM_A_REPORT}",
                        robot.name
                    ));
                    return false;
                }
                Some(links) if links.others.contains(frame) => {
                    self.error(at, format!(
                        "a literal in `{frame}`, a link of `{}` other than its robot frame `{}`: {LITERAL_FRAMES}: {PASS_FROM_A_REPORT}",
                        robot.name, links.root
                    ));
                    return false;
                }
                Some(_) => {}
            }
        }
        true
    }

    /// A record that names its frame, given field by field: its frame, its
    /// timestamp and its values in that frame are literals, and the record
    /// takes the frame it names as `takes_literal_frame` says.
    fn record_in_parts(
        &mut self,
        at: &str,
        callee: &Callee,
        args: &BTreeMap<String, Arg>,
        format: &Format,
    ) -> bool {
        let mut fine = true;
        for (name, arg) in args {
            let Some(field) = format.0.get(name) else {
                continue;
            };
            let part = name == frames::FRAME
                || name == frames::TIMESTAMP
                || frames::in_the_records_frame(name, field);
            if part && !is_literal(arg) {
                self.error(at, format!(
                    "`{name}` is a part of a record that names its frame: give the frame, the timestamp and the values of the record as literals, the record whole by one `$ref` to a point, a pose or a twist, or its records of values by `$ref` to one record that names one `frame`, which brings them"
                ));
                fine = false;
            }
        }
        if let Some(Arg::Literal(Json::String(frame))) = args.get(frames::FRAME) {
            fine &= self.takes_literal_frame(at, callee, frame);
        }
        fine
    }

    /// A record that names its frame at the level of `format`, whose values
    /// in that frame a call of `callee` gives by `$ref` (`values_by_ref`):
    /// each value of one record that gives one frame and one timestamp, or
    /// one value that names its own, whose frame and timestamp the record
    /// takes with them, each value with the fields its field takes, and no
    /// timestamp where their record leaves it out. The values of another
    /// role go to the call in world alone, which the run checks. A literal
    /// value beside them, values of two records, a list of values that each
    /// name their own, and a frame or a timestamp given beside them that is
    /// not the `$ref` of theirs are refused.
    fn record_by_refs(
        &mut self,
        at: &str,
        callee: &Callee,
        args: &BTreeMap<String, Arg>,
        format: &Format,
    ) -> Option<BTreeMap<String, BoundArg>> {
        let mut bound = BTreeMap::new();
        let mut stamps: Vec<(&RefPath, frames::Stamp, Option<Crossing>)> = Vec::new();
        for (name, arg) in args {
            let Some(field) = format.0.get(name) else {
                continue;
            };
            if name == frames::FRAME
                || name == frames::TIMESTAMP
                || !frames::in_the_records_frame(name, field)
            {
                continue;
            }
            let literal_beside = || {
                format!(
                    "`{name}` is a literal beside values given by `$ref`: give every value of a record that names its frame by `$ref` to one record that names one `frame`, or the frame, the timestamp and every value as literals"
                )
            };
            let value = match (arg, &field.kind) {
                (Arg::Ref(path), _) => {
                    let (placed, stamp, crossing) =
                        self.value_by_ref(at, callee, name, path, field)?;
                    stamps.push((path, stamp, crossing));
                    BoundArg::Placed(placed)
                }
                (Arg::List(items), FieldKind::List(item)) => {
                    let Some(paths) = items.iter().map(ref_path).collect::<Option<Vec<_>>>() else {
                        self.error(at, literal_beside());
                        return None;
                    };
                    let mut placed_items = Vec::new();
                    for path in paths {
                        let (placed, stamp, crossing) =
                            self.value_by_ref(at, callee, name, path, item)?;
                        stamps.push((path, stamp, crossing));
                        placed_items.push(BoundArg::Placed(placed));
                    }
                    BoundArg::List(placed_items)
                }
                _ => {
                    self.error(at, literal_beside());
                    return None;
                }
            };
            bound.insert(name.clone(), value);
        }
        let Some((first_path, stamp, crossing)) = stamps.first().cloned() else {
            return Some(bound);
        };
        if let Some((other, _, _)) = stamps.iter().find(|(_, other, _)| *other != stamp) {
            self.error(at, format!(
                "`{first_path}` and `{other}` come from two records: give values of one record that names one `frame`, or one value that names its frame"
            ));
            return None;
        }
        for (name, given) in [
            (frames::FRAME, Some(&stamp.frame)),
            (frames::TIMESTAMP, stamp.timestamp.as_ref()),
        ] {
            match (args.get(name), given) {
                (None, _) => {}
                (Some(Arg::Ref(path)), Some(given)) if path == given => {}
                (Some(_), Some(given)) => {
                    self.error(at, format!(
                        "remove `{name}`: the values of the record come by `$ref` from a record that gives `{given}`, which the record takes with them"
                    ));
                    return None;
                }
                (Some(_), None) => {
                    self.error(at, format!(
                        "remove `{name}`: the values of the record come by `$ref` from a record whose values hold at every time, which gives none"
                    ));
                    return None;
                }
            }
        }
        bound.insert(
            frames::FRAME.to_owned(),
            BoundArg::Frame(FrameOfValues {
                frame: stamp.frame,
                values: first_path.clone(),
                crossing,
            }),
        );
        if let Some(timestamp) = stamp.timestamp {
            bound.insert(frames::TIMESTAMP.to_owned(), BoundArg::Timestamp(timestamp));
        }
        Some(bound)
    }

    /// The value at `path` that the field `name` of a record that names its
    /// frame at its level takes, which `field` gives the format of: the
    /// record or the list of records at `path`, each with the fields that
    /// the field takes, and where its frame and its timestamp are read: in
    /// the record that holds it, or in the record itself when it names its
    /// own; and the roles the value passes between when the role that gives
    /// it is not the role of the call.
    fn value_by_ref(
        &mut self,
        at: &str,
        callee: &Callee,
        name: &str,
        path: &RefPath,
        field: &Field,
    ) -> Option<(Placed, frames::Stamp, Option<Crossing>)> {
        let role = callee.role;
        let Some(origin) = self.origin_of(role, path) else {
            self.error(
                at,
                format!("`$ref: {path}` reads no field of a result or an observation"),
            );
            return None;
        };
        let (Some((target, each)), Some((record, listed))) =
            (frames::records_of(field), frames::records_of(&origin.field))
        else {
            self.error(at, format!(
                "`$ref: {path}` is {}, and `{name}` takes {}: give a record or a list of records of a point, a pose or a twist",
                origin.field.ty(),
                field.ty()
            ));
            return None;
        };
        if !compatible(&origin.field.kind, &field.kind) {
            self.error(
                at,
                format!(
                    "`$ref: {path}` is {}, and `{name}` takes {}: give `{name}` a `$ref` to {}",
                    origin.field.ty(),
                    field.ty(),
                    named_records(target, each)
                ),
            );
            return None;
        }
        let stamp = match frames::stamp_of(path, record, &origin.holding) {
            Ok(Some(stamp)) => stamp,
            Ok(None) if !listed => frames::Stamp::of_record(path, record),
            Ok(None) => {
                self.error(at, format!(
                    "`{path}` holds values that each name a frame of their own, and `{name}` takes values in the one frame of the request: give values of one record that names one `frame`, or one value that names its frame"
                ));
                return None;
            }
            Err(reason) => {
                self.error(at, reason);
                return None;
            }
        };
        let placed = Placed {
            value: path.clone(),
            each,
            stamp: None,
            fields: target.0.keys().cloned().collect(),
            crossing: None,
        };
        Some((placed, stamp, Crossing::between(&origin.role, role)))
    }

    /// The records that name their frame which a `$ref` to `path` gives the
    /// field `name`, which takes `target` records, or a list of them
    /// (`each`). A record that names its frame goes as it is,
    /// and a plain value with the frame and the timestamp of the record that
    /// holds it; each gives the fields of the target, and a field that the
    /// target may leave out, as the orientation of a target that takes one
    /// from a point, goes absent where the value gives none.
    fn placed_ref(
        &mut self,
        at: &str,
        role: &Name,
        name: &str,
        path: &RefPath,
        origin: &Origin,
        field: &Field,
    ) -> Option<BoundArg> {
        let (target, each) = frames::framed_records(field)?;
        let record = frames::records_of(&origin.field)
            .filter(|(_, listed)| *listed == each)
            .map(|(record, _)| record);
        let Some(record) = record else {
            self.error(
                at,
                format!(
                    "`$ref: {path}` is {}, and `{name}` takes {}: give `{name}` a `$ref` to {}",
                    origin.field.ty(),
                    field.ty(),
                    named_records(target, each)
                ),
            );
            return None;
        };
        let stamp = match frames::stamp_of(path, record, &origin.holding) {
            Ok(stamp) => stamp,
            Err(reason) => {
                self.error(at, reason);
                return None;
            }
        };
        let values: Vec<&String> = target
            .0
            .keys()
            .filter(|field| *field != frames::FRAME && *field != frames::TIMESTAMP)
            .collect();
        if !values.iter().any(|field| record.0.contains_key(*field)) {
            self.error(
                at,
                format!(
                    "`{path}` gives none of the values that `{name}` takes: give `{name}` a `$ref` to {}",
                    named_records(target, each)
                ),
            );
            return None;
        }
        for field in values {
            let wanted = &target.0[field];
            match record.0.get(field) {
                Some(have) if compatible(&have.kind, &wanted.kind) => {}
                None if wanted.optional => {}
                _ => {
                    let giving = match each {
                        false => "that gives",
                        true => "that each give",
                    };
                    self.error(
                        at,
                        format!(
                            "`{path}` gives no `{field}` of the kind `{name}` takes: give `{name}` a `$ref` to {} {giving} `{field}`",
                            named_records(target, each)
                        ),
                    );
                    return None;
                }
            }
        }
        Some(BoundArg::Placed(Placed {
            value: path.clone(),
            each,
            stamp,
            fields: target.0.keys().cloned().collect(),
            crossing: Crossing::between(&origin.role, role),
        }))
    }

    /// The value of the field `name` with the mark location that a call of
    /// `callee` gives `location`: the name of the location, in the canonical
    /// form of the field, which the skill knows.
    fn location_value(
        &mut self,
        at: &str,
        callee: &Callee,
        name: &str,
        field: &Field,
        location: &Name,
    ) -> Option<Json> {
        let value = match field.canonical(&Json::String(location.to_string())) {
            Ok(value) => value,
            Err(reason) => {
                self.error(
                    at,
                    format!("`{name}` has the mark location and takes no location name: {reason}"),
                );
                return None;
            }
        };
        self.uses_location(callee.role, location, callee.producer);
        Some(value)
    }

    /// Records that a call of `role` on `producer` gives `location`, so the
    /// run checks that the skill knows it. Whether the instance implements
    /// locations:v1 is a rule of the binding (`misfits`).
    fn uses_location(&mut self, role: &Name, location: &Name, producer: Option<&Producer>) {
        let Some(producer) = producer else {
            return;
        };
        match self
            .locations
            .iter_mut()
            .find(|skill| skill.role == *role && skill.producer == *producer)
        {
            Some(skill) => {
                skill.names.insert(location.clone());
            }
            None => self.locations.push(SkillLocations {
                role: role.clone(),
                producer: producer.clone(),
                names: BTreeSet::from([location.clone()]),
            }),
        }
    }

    /// Binds a call: its member on each pin, its arguments against the goal
    /// or the request, and its target; gives the format of its result and
    /// its producer.
    #[allow(clippy::too_many_arguments)]
    fn bind_call(
        &mut self,
        at: &str,
        role: &Name,
        id: &StepId,
        member: &workflow_documents::MemberRef,
        instance: Option<&workflow_documents::InstanceName>,
        args: &BTreeMap<String, Arg>,
    ) -> Option<(Format, Option<Producer>)> {
        let found = self.members(at, role, member, instance);
        let mut result = None;
        for (shape, found_member) in found {
            let producer = self.producer(at, role, &member.alias, instance);
            let callee = Callee {
                role,
                alias: &member.alias,
                producer: producer.as_ref(),
            };
            let owner = format!("`{member}`");
            let bound = self.args(at, &callee, &owner, args, &found_member.input);
            if let (Some(producer), Some(args)) = (&producer, bound) {
                self.calls.insert(
                    CallSite {
                        role: role.clone(),
                        step: id.clone(),
                    },
                    BoundCall {
                        target: Target {
                            role: role.clone(),
                            contract: shape.name_tag(),
                            sha256: shape.sha256.clone(),
                            member: member.member.clone(),
                            kind: found_member.kind,
                            producer: producer.clone(),
                            moves_robot: found_member.marks.contains(MOVES_ROBOT),
                            activity: found_member.activity.clone(),
                        },
                        args,
                        result_type: found_member.output.ty(),
                    },
                );
            }
            result = Some((found_member.output.clone(), producer));
        }
        result
    }

    /// The arguments bound against `format`, the request of `owner` or the
    /// record of one of its fields: each field of the format, the required
    /// ones present, each literal in its canonical form, each field with a
    /// mark fed only as its mark says, and a record that names its frame
    /// given in parts as `record_in_parts` says.
    fn args(
        &mut self,
        at: &str,
        callee: &Callee,
        owner: &str,
        args: &BTreeMap<String, Arg>,
        format: &Format,
    ) -> Option<BTreeMap<String, BoundArg>> {
        let mut bound = BTreeMap::new();
        let mut fine = true;
        if frames::names_its_frame(format) {
            if values_by_ref(args, format) {
                bound = self.record_by_refs(at, callee, args, format)?;
            } else if !self.record_in_parts(at, callee, args, format) {
                return None;
            }
        }
        for (name, field) in &format.0 {
            if !field.optional && !args.contains_key(name) && !bound.contains_key(name) {
                self.error(at, format!("`{name}` is a required argument"));
                fine = false;
            }
        }
        for (name, arg) in args {
            if bound.contains_key(name) {
                continue;
            }
            let Some(field) = format.0.get(name) else {
                let arguments: Vec<String> = format
                    .0
                    .iter()
                    .map(|(name, field)| argument_as_listed(name, field))
                    .collect();
                self.error(
                    at,
                    format!(
                        "{owner} has no argument `{name}`: its arguments are {}",
                        arguments.join(", ")
                    ),
                );
                fine = false;
                continue;
            };
            match self.arg(at, callee, name, arg, field) {
                Some(arg) => {
                    bound.insert(name.clone(), arg);
                }
                None => fine = false,
            }
        }
        fine.then_some(bound)
    }

    fn arg(
        &mut self,
        at: &str,
        callee: &Callee,
        name: &str,
        arg: &Arg,
        field: &Field,
    ) -> Option<BoundArg> {
        let role = callee.role;
        let marked = field.marks.contains(ITEM_ID);
        match arg {
            Arg::Literal(_) | Arg::Object(_) | Arg::List(_) | Arg::Ref(_)
                if field.marks.contains(LOCATION) =>
            {
                self.error(
                    at,
                    format!("`{name}` takes a location; give it a `{{ $location }}` of a declared location"),
                );
                None
            }
            Arg::Location(location) if field.marks.contains(LOCATION) => self
                .location_value(at, callee, name, field, location)
                .map(BoundArg::Literal),
            Arg::Location(_) => {
                self.error(
                    at,
                    format!("`{name}` takes no location: a `$location` goes only to a field with the mark location"),
                );
                None
            }
            Arg::Literal(_) | Arg::Object(_) | Arg::List(_) if marked => {
                self.error(
                    at,
                    format!(
                        "`{name}` holds an item id; give it a `$ref` to a result or an observation"
                    ),
                );
                None
            }
            Arg::Literal(literal) => {
                let value = match field.canonical(literal) {
                    Ok(value) => value,
                    Err(reason) => {
                        self.error(at, format!("`{name}`: {reason}"));
                        return None;
                    }
                };
                let named = frames::literal_frames(field, &value);
                let taken = named
                    .into_iter()
                    .all(|frame| self.takes_literal_frame(at, callee, frame));
                taken.then_some(BoundArg::Literal(value))
            }
            Arg::Object(fields) => match &field.kind {
                FieldKind::Object(format) => {
                    let owner = format!("`{name}`");
                    self.args(at, callee, &owner, fields, format)
                        .map(BoundArg::Object)
                }
                _ => {
                    self.error(at, format!("`{name}` is no object"));
                    None
                }
            },
            Arg::List(items) => match &field.kind {
                FieldKind::List(item) => items
                    .iter()
                    .map(|a| self.arg(at, callee, name, a, item))
                    .collect::<Option<Vec<_>>>()
                    .map(BoundArg::List),
                _ => {
                    self.error(at, format!("`{name}` is no list"));
                    None
                }
            },
            Arg::Ref(path) => {
                let Some(origin) = self.origin_of(role, path) else {
                    self.error(
                        at,
                        format!("`$ref: {path}` reads no field of a result or an observation"),
                    );
                    return None;
                };
                if frames::framed_records(field).is_some() {
                    return self.placed_ref(at, role, name, path, &origin, field);
                }
                if origin.role != *role && frames::holds_framed_records(field) {
                    self.error(at, format!(
                        "`{name}` holds points, poses or twists, and `{path}` comes from the role `{}`: {}; give `{name}` field by field, with a `$ref` of its own to each point, pose or twist, which the run sends in world alone",
                        origin.role,
                        frames::BETWEEN_ROBOTS_IN_WORLD
                    ));
                    return None;
                }
                if !compatible(&origin.field.kind, &field.kind) {
                    self.error(
                        at,
                        format!(
                            "`$ref: {path}` is {}, and `{name}` takes {}",
                            origin.field.ty(),
                            field.ty()
                        ),
                    );
                    return None;
                }
                if marked {
                    if !origin.field.marks.contains(ITEM_ID) {
                        self.error(
                            at,
                            format!("`{name}` takes an item id, and `{path}` holds none"),
                        );
                        return None;
                    }
                    if let (Some(from), Some(to)) = (&origin.producer, callee.producer)
                        && from != to
                    {
                        self.error(at, format!("the item id of `{path}` comes from `{}`, and `{name}` goes to `{}`: an item id is valid only on the instance that gives it", from.instance, to.instance));
                        return None;
                    }
                }
                Some(BoundArg::Ref(path.clone()))
            }
            Arg::Table { table, key } => {
                if marked {
                    self.error(at, format!("`{name}` holds an item id; give it a `$ref`"));
                    return None;
                }
                let canonical = match (
                    field.marks.contains(LOCATION),
                    self.location_tables.get(table).cloned(),
                ) {
                    (true, Some(locations)) => {
                        let mut canonical = BTreeMap::new();
                        for (row_key, location) in locations {
                            let value = self.location_value(at, callee, name, field, &location)?;
                            canonical.insert(row_key, value);
                        }
                        canonical
                    }
                    (true, None) => {
                        self.error(at, format!("`{name}` takes a location, and `$table` names `{table}`, whose rows are not `{{ $location }}`s: give it a table whose every row is `{{ $location: \"<name>\" }}`"));
                        return None;
                    }
                    (false, Some(_)) => {
                        self.error(at, format!("`$table` names `{table}`, a table of locations, and `{name}` takes no location: a `$location` goes only to a field with the mark location"));
                        return None;
                    }
                    (false, None) => {
                        let Some((Json::Object(rows), Type::Table { .. })) =
                            self.tables.get(table).cloned()
                        else {
                            self.error(
                                at,
                                format!("`$table` names `{table}`, which has no rows of one type"),
                            );
                            return None;
                        };
                        let mut canonical = BTreeMap::new();
                        for (row_key, row) in rows {
                            let value = match field.canonical(&row) {
                                Ok(value) => value,
                                Err(reason) => {
                                    self.error(
                                        at,
                                        format!(
                                            "the row `{row_key}` of `{table}` for `{name}`: {reason}"
                                        ),
                                    );
                                    return None;
                                }
                            };
                            canonical.insert(row_key, value);
                        }
                        canonical
                    }
                };
                let keys: BTreeSet<String> = canonical.keys().cloned().collect();
                let key = match key.as_ref() {
                    Arg::Ref(path) => {
                        let enum_values =
                            self.origin_of(role, path).and_then(|o| match o.field.kind {
                                FieldKind::Enum(values) => Some(values),
                                _ => None,
                            });
                        match enum_values {
                            Some(values) if values == keys => BoundArg::Ref(path.clone()),
                            Some(values) => {
                                let want: Vec<&String> = values.iter().collect();
                                self.error(at, format!("the table `{table}` has one row per value of the enum of `{path}`: {want:?}"));
                                return None;
                            }
                            None => {
                                self.error(
                                    at,
                                    format!("the key `{path}` of the table `{table}` is no enum"),
                                );
                                return None;
                            }
                        }
                    }
                    Arg::Literal(Json::String(row)) if keys.contains(row) => {
                        BoundArg::Literal(Json::String(row.clone()))
                    }
                    _ => {
                        self.error(at, format!("the key of the table `{table}` is a `$ref` to an enum or one of its rows"));
                        return None;
                    }
                };
                Some(BoundArg::Table {
                    rows: canonical,
                    key: Box::new(key),
                })
            }
        }
    }

    /// Types each expression of a tree of a role.
    fn type_tree(&mut self, role: &Name, step: &Step, scope: &TypeScope) {
        let at = format!("{role}.{}", step.id);
        if let Some(invariant) = &step.invariant {
            self.typed(&at, invariant, scope, Want::Bool);
        }
        match &step.kind {
            StepKind::Call(call) => {
                if let Some(done_when) = &call.done_when {
                    self.typed(&at, done_when, scope, Want::Bool);
                }
            }
            StepKind::Sequence(steps) | StepKind::Parallel { steps, .. } => {
                steps.iter().for_each(|s| self.type_tree(role, s, scope))
            }
            StepKind::If {
                cond,
                then,
                otherwise,
            } => {
                self.typed(&at, cond, scope, Want::Bool);
                self.type_tree(role, then, scope);
                if let Some(o) = otherwise {
                    self.type_tree(role, o, scope);
                }
            }
            StepKind::Match {
                on,
                cases,
                otherwise,
            } => {
                self.typed(&at, on, scope, Want::Enum);
                if let Ok(Type::Enum(values)) = on.expr().infer(scope) {
                    for case in cases.keys() {
                        if !values.contains(case) {
                            self.error(
                                &at,
                                format!(
                                    "the case '{case}' is not a value of the enum of `{}`",
                                    on.source()
                                ),
                            );
                        }
                    }
                    let missing: Vec<&String> =
                        values.iter().filter(|v| !cases.contains_key(*v)).collect();
                    if otherwise.is_none() && !missing.is_empty() {
                        self.error(&at, format!("`match` covers its enum with its cases or an `otherwise`; it misses {missing:?}"));
                    }
                }
                cases.values().for_each(|s| self.type_tree(role, s, scope));
                if let Some(o) = otherwise {
                    self.type_tree(role, o, scope);
                }
            }
            StepKind::ForEach { list, body, .. } => {
                self.typed(&at, list, scope, Want::List);
                self.type_tree(role, body, scope);
            }
            StepKind::RepeatUntil {
                until,
                fail_when,
                body,
                ..
            } => {
                self.typed(&at, until, scope, Want::Bool);
                if let Some(f) = fail_when {
                    self.typed(&at, f, scope, Want::Bool);
                }
                self.type_tree(role, body, scope);
            }
            StepKind::WaitUntil {
                until, fail_when, ..
            } => {
                self.typed(&at, until, scope, Want::Bool);
                if let Some(f) = fail_when {
                    self.typed(&at, f, scope, Want::Bool);
                }
            }
            StepKind::Signal(_) | StepKind::WaitFor { .. } => {}
        }
    }

    /// The scope of the expressions outside the trees: the observations and
    /// the tables.
    fn whole_run_scope(&self) -> TypeScope {
        let mut names = self.observation_types();
        for (table, (_, ty)) in &self.tables {
            names.insert(table.clone(), ty.clone());
        }
        TypeScope::new(names)
    }

    fn check_whole_run_expressions(&mut self) {
        let scope = self.whole_run_scope();
        let workflow = self.workflow;
        self.typed("success", &workflow.success, &scope, Want::Bool);
        if let Some(minimum) = &workflow.minimum {
            self.typed("minimum", minimum, &scope, Want::Bool);
        }
        for (name, assumption) in &workflow.assumptions {
            if let Some(check) = &assumption.check {
                self.typed(
                    &format!("assumptions.{name}.check"),
                    check,
                    &scope,
                    Want::Bool,
                );
            }
        }
    }

    /// Each watch parameter reads a number.
    fn check_watch(&mut self, watch: &Watch) {
        let scope = self.whole_run_scope();
        for (name, observation) in &watch.observations {
            self.typed(&format!("watch.{name}"), observation, &scope, Want::Number);
        }
    }

    /// A lane that calls a member that reaches for an item first checks that
    /// the robot reaches: with a reach assumption whose check reads an
    /// observation of its role, or with reach checks (`reach_check_of`)
    /// named before the reaching call on every path that read a position of
    /// each source of the items it reaches (`reached_sources`).
    fn check_reach(&mut self) {
        let workflow = self.workflow;
        for role in workflow.lanes.keys() {
            if self.reach_assumed(role) {
                continue;
            }
            let calls = calls_of(workflow.trees_of(role));
            let reaching: Vec<(&Step, &Call)> = calls
                .iter()
                .copied()
                .filter(|(_, call)| self.may_have_mark(role, call, REACHES_FOR_ITEM))
                .collect();
            if reaching.is_empty() {
                continue;
            }
            let checks: BTreeMap<&Name, BTreeSet<Name>> = calls
                .iter()
                .filter_map(|(_, call)| self.reach_check_of(role, call))
                .collect();
            let names_before = workflow.names_before_calls(role);
            for (step, call) in reaching {
                let checked: BTreeSet<&Name> = names_before
                    .get(&step.id)
                    .into_iter()
                    .flatten()
                    .filter_map(|name| checks.get(name))
                    .flatten()
                    .collect();
                let reached = self.reached_sources(role, call);
                let unchecked: Vec<String> = reached
                    .iter()
                    .filter(|source| !checked.contains(source))
                    .map(|source| format!("`{source}`"))
                    .collect();
                let at = format!("{role}.{}", step.id);
                if checked.is_empty() || reached.is_empty() {
                    self.error(at, format!("`{}` reaches for an item; give the workflow a `kind: \"reach\"` assumption whose check reads an observation of `{role}`, or put a reach check before it on every path: a call of a member whose result has a field with the mark `checks_reach`, with `as`, a `done_when` that is that field of its result, and the position of the item read from the same result as the item", call.member));
                } else if !unchecked.is_empty() {
                    self.error(at, format!("`{}` reaches for an item of {}, and no reach check before it on every path reads a position of it; give a reach check the position of each item that the call reaches, read from the same result or observation", call.member, unchecked.join(", ")));
                }
            }
        }
    }

    /// Whether a reach assumption of the workflow has a check that reads an
    /// observation of `role`.
    fn reach_assumed(&self, role: &Name) -> bool {
        self.workflow.assumptions.values().any(|assumption| {
            assumption.reach
                && assumption.check.as_ref().is_some_and(|check| {
                    check.roots().iter().any(|root| {
                        self.workflow
                            .observations
                            .get(root)
                            .is_some_and(|o| &o.role == role)
                    })
                })
        })
    }

    /// The result of `call` of `role` and the sources of the positions that
    /// it checks (`checked_sources`), when the call is a reach check: the
    /// result of its member has a field that tells whether the robot reaches
    /// (`reach_verdict`), and its `done_when` is that field of its result,
    /// alone or as a term of `&&`, so the step fails when the robot does not
    /// reach.
    fn reach_check_of<'c>(
        &self,
        role: &Name,
        call: &'c Call,
    ) -> Option<(&'c Name, BTreeSet<Name>)> {
        let result = call.result.as_ref()?;
        let done_when = call.done_when.as_ref()?;
        let members = self.members_on_every_pin(role, call)?;
        let verdict = reach_verdict(&members)?;
        gates_on(done_when.expr(), result, &verdict)
            .then(|| (result, self.checked_sources(role, call, &members)))
    }

    /// The member of `call` on each pin that the role reads, when every pin
    /// has it and the role reads at least one pin.
    fn members_on_every_pin(&self, role: &Name, call: &Call) -> Option<Vec<Member>> {
        let members: Option<Vec<Member>> = self
            .shapes(role, &call.member.alias, call.instance.as_ref())
            .iter()
            .map(|shape| shape.members.get(call.member.member.as_str()).cloned())
            .collect();
        members.filter(|members| !members.is_empty())
    }

    /// The sources of the points that `call` of `role` checks: of each
    /// `$ref` of an argument that takes records that name their frame, or
    /// values in the frame of a request that names its frame at its level,
    /// in the request of `members`, the member of the call on each pin.
    fn checked_sources(&self, role: &Name, call: &Call, members: &[Member]) -> BTreeSet<Name> {
        call.args
            .iter()
            .filter(|(name, _)| {
                members.iter().all(|member| {
                    member.input.0.get(name.as_str()).is_some_and(|field| {
                        frames::framed_records(field).is_some()
                            || (frames::names_its_frame(&member.input)
                                && frames::in_the_records_frame(name, field))
                    })
                })
            })
            .flat_map(|(_, arg)| refs_of(arg))
            .filter_map(|path| self.origin_of(role, path))
            .map(|origin| origin.source)
            .collect()
    }

    /// The sources of the items that `call` of `role` reaches: of each
    /// `$ref` of its arguments that reads an item id, a point or a pose.
    fn reached_sources(&self, role: &Name, call: &Call) -> BTreeSet<Name> {
        call.args
            .values()
            .flat_map(refs_of)
            .filter_map(|path| self.origin_of(role, path))
            .filter(|origin| {
                origin.field.marks.contains(ITEM_ID) || frames::is_geometry(&origin.field)
            })
            .map(|origin| origin.source)
            .collect()
    }

    /// Whether a pin that the role reads gives the member of `call` the mark
    /// `mark`: a call that may reach for an item is checked as one.
    fn may_have_mark(&self, role: &Name, call: &Call, mark: &str) -> bool {
        self.shapes(role, &call.member.alias, call.instance.as_ref())
            .iter()
            .any(|shape| {
                shape
                    .members
                    .get(call.member.member.as_str())
                    .is_some_and(|member| member.marks.contains(mark))
            })
    }
}

/// Each call of `trees`, with its step, in tree order.
fn calls_of<'w>(trees: impl Iterator<Item = &'w Step>) -> Vec<(&'w Step, &'w Call)> {
    let mut calls = Vec::new();
    for tree in trees {
        tree.visit(&mut |step| {
            if let StepKind::Call(call) = &step.kind {
                calls.push((step, call));
            }
        });
    }
    calls
}

/// What a field that takes `target` records, or a list of them (`each`),
/// takes, as a refusal names it: "a pose", "a list of points".
fn named_records(target: &Format, each: bool) -> String {
    let (one, many) = frames::kind_of(target);
    match each {
        false => one.to_owned(),
        true => format!("a list of {many}"),
    }
}

/// The argument `name` of `field` as the refusal of an unknown argument
/// lists it: with what it takes when it takes records that name their frame.
fn argument_as_listed(name: &str, field: &Field) -> String {
    let Some((record, each)) = frames::framed_records(field) else {
        return format!("`{name}`");
    };
    let with = match each {
        false => "with",
        true => "each with",
    };
    format!(
        "`{name}` ({} {with} frame and timestamp)",
        named_records(record, each)
    )
}

/// Whether `arg` holds literals alone.
fn is_literal(arg: &Arg) -> bool {
    match arg {
        Arg::Literal(_) => true,
        Arg::List(items) => items.iter().all(is_literal),
        Arg::Object(fields) => fields.values().all(is_literal),
        Arg::Ref(_) | Arg::Table { .. } | Arg::Location(_) => false,
    }
}

/// Whether `args` give a value of the record of `format`, which names its
/// frame at its level, by `$ref`: a record of values in that frame, or a
/// list of them, that is a `$ref` or a list of `$ref`s.
fn values_by_ref(args: &BTreeMap<String, Arg>, format: &Format) -> bool {
    args.iter().any(|(name, arg)| {
        let in_frame = name != frames::FRAME
            && name != frames::TIMESTAMP
            && format.0.get(name).is_some_and(|field| {
                frames::in_the_records_frame(name, field) && frames::records_of(field).is_some()
            });
        let by_ref = match arg {
            Arg::Ref(_) => true,
            Arg::List(items) => items.iter().any(|item| ref_path(item).is_some()),
            _ => false,
        };
        in_frame && by_ref
    })
}

/// The path of `arg` when it is a `$ref`.
fn ref_path(arg: &Arg) -> Option<&RefPath> {
    match arg {
        Arg::Ref(path) => Some(path),
        _ => None,
    }
}

/// Each `$ref` that `arg` holds, in an object, a list or the key of a table.
fn refs_of(arg: &Arg) -> Vec<&RefPath> {
    match arg {
        Arg::Ref(path) => vec![path],
        Arg::Object(fields) => fields.values().flat_map(refs_of).collect(),
        Arg::List(items) => items.iter().flat_map(refs_of).collect(),
        Arg::Table { key, .. } => refs_of(key),
        Arg::Literal(_) | Arg::Location(_) => Vec::new(),
    }
}

/// The name of the field of the result that tells whether the robot
/// reaches, when `members`, the member of a call on each pin, agree on it:
/// on every pin, the one required boolean field of the result with the mark
/// `checks_reach`, of the same name. A reach check counts only when every
/// instance that may serve it checks reach.
fn reach_verdict(members: &[Member]) -> Option<String> {
    let mut verdicts = members.iter().map(|member| {
        let mut marked = member.output.0.iter().filter(|(_, field)| {
            field.marks.contains(CHECKS_REACH) && field.kind == FieldKind::Bool && !field.optional
        });
        match (marked.next(), marked.next()) {
            (Some((name, _)), None) => Some(name.clone()),
            _ => None,
        }
    });
    let first = verdicts.next()??;
    verdicts
        .all(|verdict| verdict.as_ref() == Some(&first))
        .then_some(first)
}

/// Whether `expr` is false whenever `<result>.<field>` is false: that path
/// alone, or a conjunction (`&&`) with it as one of its terms.
fn gates_on(expr: &Expr, result: &Name, field: &str) -> bool {
    match expr {
        Expr::Select(operand, selected) => {
            selected == field && matches!(operand.as_ref(), Expr::Ident(name) if name == result)
        }
        Expr::And(left, right) => gates_on(left, result, field) || gates_on(right, result, field),
        _ => false,
    }
}

enum Want {
    Bool,
    Number,
    Enum,
    List,
}

impl Want {
    fn says(&self) -> &'static str {
        match self {
            Want::Bool => "the field takes a bool",
            Want::Number => "a watch parameter reads a number",
            Want::Enum => "a `match` takes an enum",
            Want::List => "a `for_each` takes a list",
        }
    }
}

fn is_number(ty: &Type) -> bool {
    matches!(ty, Type::Int | Type::Double)
}

fn literal_type(value: &Json) -> Option<Type> {
    match value {
        Json::Bool(_) => Some(Type::Bool),
        Json::Number(n) if n.is_i64() || n.is_u64() => Some(Type::Int),
        Json::Number(_) => Some(Type::Double),
        Json::String(_) => Some(Type::String),
        Json::Array(items) => {
            let first = items.first().and_then(literal_type)?;
            Some(Type::List(Box::new(first)))
        }
        _ => None,
    }
}

/// Whether a value of `from` fits a field of `to`.
fn compatible(from: &FieldKind, to: &FieldKind) -> bool {
    match (from, to) {
        (FieldKind::Int { .. }, FieldKind::Int { .. }) => true,
        (FieldKind::Int { .. } | FieldKind::Double, FieldKind::Double) => true,
        (FieldKind::Enum(_) | FieldKind::String, FieldKind::String) => true,
        (FieldKind::Enum(a), FieldKind::Enum(b)) => a.is_subset(b),
        (FieldKind::List(a), FieldKind::List(b)) => compatible(&a.kind, &b.kind),
        (FieldKind::Object(a), FieldKind::Object(b)) => b.0.iter().all(|(name, field)| {
            a.0.get(name)
                .map_or(field.optional, |have| compatible(&have.kind, &field.kind))
        }),
        (a, b) => a == b,
    }
}

/// Checks a workflow against the contracts of the snapshot, with its roles
/// bound to robots when `binding` gives them, and gives its errors.
pub fn check(
    workflow: &Workflow,
    snapshot: &Snapshot,
    allowed: &AllowedContracts,
    binding: Option<&BTreeMap<Name, RobotName>>,
) -> Vec<Located> {
    run_checker(workflow, snapshot, allowed, binding, &BTreeSet::new()).errors
}

fn run_checker<'a>(
    workflow: &'a Workflow,
    snapshot: &'a Snapshot,
    allowed: &'a AllowedContracts,
    binding: Option<&'a BTreeMap<Name, RobotName>>,
    confirmed: &BTreeSet<Name>,
) -> Checker<'a> {
    let mut checker = Checker {
        workflow,
        snapshot,
        allowed,
        binding,
        errors: Vec::new(),
        calls: BTreeMap::new(),
        observations: BTreeMap::new(),
        locals: BTreeMap::new(),
        origins: BTreeMap::new(),
        observation_origins: BTreeMap::new(),
        tables: BTreeMap::new(),
        location_tables: BTreeMap::new(),
        locations: Vec::new(),
        reads_links: BTreeSet::new(),
    };
    checker.check_contracts();
    checker.check_binding(confirmed);
    checker.check_tables();
    checker.check_observations();
    checker.check_trees();
    checker.check_whole_run_expressions();
    checker.check_reach();
    checker
        .errors
        .sort_by(|a, b| (&a.at, &a.message).cmp(&(&b.at, &b.message)));
    checker.errors.dedup();
    checker
}

/// The plan of a run of `workflow` with its roles bound, or the errors that
/// refuse it.
pub fn plan(
    workflow: Arc<Workflow>,
    snapshot: &Snapshot,
    allowed: &AllowedContracts,
    binding: &BTreeMap<Name, RobotName>,
    choices: RunChoices<'_>,
) -> Result<Plan, Vec<Located>> {
    let mut checker = run_checker(
        &workflow,
        snapshot,
        allowed,
        Some(binding),
        &choices.confirmed,
    );
    checker.check_watch(&choices.watch);
    let mut errors = checker.errors.clone();
    let monitor = match choices.incidents {
        Incidents::None => None,
        Incidents::Monitored {
            monitor,
            continue_on_severity,
        } => {
            if choices.attendance == Attendance::Attended && !continue_on_severity.is_empty() {
                errors.push(Located {
                    at: "incidents".to_owned(),
                    message: "`continue_on_severity` is for an unattended run".to_owned(),
                });
            }
            match monitor_target(snapshot, allowed, monitor) {
                Ok(target) => Some(MonitorPlan {
                    target,
                    continue_on_severity: continue_on_severity.clone(),
                }),
                Err(message) => {
                    errors.push(Located {
                        at: "incidents.monitor".to_owned(),
                        message,
                    });
                    None
                }
            }
        }
    };
    if !errors.is_empty() {
        return Err(errors);
    }
    let roles = binding
        .iter()
        .map(|(role, robot)| {
            let physical = snapshot
                .fleet
                .robots
                .get(robot.as_str())
                .is_some_and(|r| r.is_physical());
            (
                role.clone(),
                BoundRole {
                    robot: robot.clone(),
                    physical,
                },
            )
        })
        .collect();
    Ok(Plan {
        attendance: choices.attendance,
        confirmed: choices.confirmed,
        roles,
        calls: checker.calls,
        observations: checker.observations,
        locals: checker.locals,
        tables: checker.tables,
        monitor,
        locations: checker.locations,
        poll: choices.poll,
        cancel_timeout: choices.cancel_timeout,
        watch: choices.watch,
        of_a_trial: choices.of_a_trial,
        reads_links: checker.reads_links,
        workflow,
    })
}

/// The target of the anomaly monitor: an allowed contract that a robot or an
/// option of the listing pins, a service of the form of a monitor, on an
/// instance of the listing.
fn monitor_target(
    snapshot: &Snapshot,
    allowed: &AllowedContracts,
    monitor: &workflow_documents::run::Monitor,
) -> Result<Target, String> {
    let contract = &monitor.contract;
    if !allowed.allows(contract) {
        return Err(format!(
            "`{contract}` is not an allowed contract of this director"
        ));
    }
    let pins = snapshot
        .fleet
        .pins_of(contract.name.as_str(), contract.tag());
    let [pin] = pins.as_slice() else {
        return Err(format!(
            "no robot or option of the fleet listing pins `{contract}` once; the monitor reads its document from the listing"
        ));
    };
    let shape = snapshot
        .contracts
        .get(pin)
        .ok_or_else(|| format!("the document of `{contract}` did not read"))?;
    let member = shape
        .members
        .get(monitor.member.as_str())
        .ok_or_else(|| format!("`{contract}` has no member `{}`", monitor.member))?;
    let request_fits = member.kind == MemberKind::Service
        && member.input.0.keys().collect::<BTreeSet<_>>()
            == BTreeSet::from([&"robot".to_owned(), &"step".to_owned()]);
    let response_fits = ["anomaly", "kind", "severity", "description", "data"]
        .iter()
        .all(|field| member.output.0.contains_key(*field));
    if !request_fits || !response_fits {
        return Err(format!(
            "`{contract}.{}` is no anomaly monitor: a service of {{ robot, step }} that answers {{ anomaly, kind, severity, description, data }}",
            monitor.member
        ));
    }
    let instance = snapshot
        .fleet
        .nodes
        .iter()
        .flat_map(|node| node.instances.iter())
        .find(|(instance, _, _)| instance == &monitor.stack_instance)
        .ok_or_else(|| format!("`{}` is no instance of the stack", monitor.stack_instance))?;
    Ok(Target {
        role: Name::parse("monitor").expect("a name"),
        contract: shape.name_tag(),
        sha256: shape.sha256.clone(),
        member: monitor.member.clone(),
        kind: MemberKind::Service,
        producer: Producer {
            core_node: instance.1.clone(),
            instance: instance.0.clone(),
        },
        moves_robot: false,
        activity: None,
    })
}

#[cfg(test)]
mod tests;

/// The aliases that `role` calls: in its trees, its safe state and its
/// observations.
fn aliases_of_role(workflow: &Workflow, role: &Name) -> BTreeSet<Name> {
    let mut aliases = BTreeSet::new();
    let trees = workflow
        .initialization
        .get(role)
        .into_iter()
        .chain(workflow.lanes.get(role).map(|lane| &lane.tree));
    for tree in trees {
        tree.visit(&mut |step| {
            if let StepKind::Call(call) = &step.kind {
                aliases.insert(call.member.alias.clone());
            }
        });
    }
    if let Some(lane) = workflow.lanes.get(role) {
        for call in &lane.safe_state {
            aliases.insert(call.member.alias.clone());
        }
    }
    for observation in workflow.observations.values().filter(|o| &o.role == role) {
        aliases.insert(observation.named().alias.clone());
    }
    aliases
}

/// Each call of `role` that gives a location: its alias and the instance it
/// names, in its trees, its safe state and its observations.
fn location_calls_of_role<'w>(
    workflow: &'w Workflow,
    role: &Name,
) -> Vec<(&'w Name, Option<&'w workflow_documents::InstanceName>)> {
    let gives =
        |args: &BTreeMap<String, Arg>| args.values().any(|arg| arg_gives_a_location(workflow, arg));
    let mut calls = Vec::new();
    let trees = workflow
        .initialization
        .get(role)
        .into_iter()
        .chain(workflow.lanes.get(role).map(|lane| &lane.tree));
    for tree in trees {
        tree.visit(&mut |step| {
            if let StepKind::Call(call) = &step.kind
                && gives(&call.args)
            {
                calls.push((&call.member.alias, call.instance.as_ref()));
            }
        });
    }
    let safe_state = workflow
        .lanes
        .get(role)
        .into_iter()
        .flat_map(|lane| &lane.safe_state);
    calls.extend(
        safe_state
            .filter(|call| gives(&call.args))
            .map(|call| (&call.member.alias, call.instance.as_ref())),
    );
    for observation in workflow.observations.values().filter(|o| &o.role == role) {
        if let ObservationSource::Member { member, args } = &observation.source
            && gives(args)
        {
            calls.push((&member.alias, observation.instance.as_ref()));
        }
    }
    calls
}

/// The location that a row of a table names: `{ $location: <name> }`, an
/// object that holds `$location` alone.
fn location_of_row(row: &Json) -> Option<Name> {
    let row = row.as_object().filter(|row| row.len() == 1)?;
    row.get("$location")
        .and_then(Json::as_str)
        .and_then(|location| Name::parse(location).ok())
}

/// Whether `arg` gives a location: a `{ $location }`, or a table whose rows
/// do.
fn arg_gives_a_location(workflow: &Workflow, arg: &Arg) -> bool {
    match arg {
        Arg::Location(_) => true,
        Arg::Object(fields) => fields
            .values()
            .any(|arg| arg_gives_a_location(workflow, arg)),
        Arg::List(items) => items.iter().any(|arg| arg_gives_a_location(workflow, arg)),
        Arg::Table { table, .. } => workflow
            .tables
            .get(table)
            .is_some_and(|rows| rows.values().any(|row| location_of_row(row).is_some())),
        Arg::Literal(_) | Arg::Ref(_) => false,
    }
}

/// Why `robot` cannot serve the locations that `role` gives: each instance
/// of `robot` that serves a call of the role giving a location, and
/// implements no locations:v1.
fn skills_that_know_no_locations(workflow: &Workflow, role: &Name, robot: &Robot) -> Vec<String> {
    let (contract, tag) = LOCATIONS;
    let knowing = robot.instances_of(contract, tag);
    location_calls_of_role(workflow, role)
        .into_iter()
        .filter_map(|(alias, instance)| {
            let called = workflow.contracts.get(alias)?;
            let serving: Vec<_> = robot
                .instances_of(called.name.as_str(), called.tag())
                .into_iter()
                .filter(|serving| instance.is_none_or(|named| serving.name == named.as_str()))
                .collect();
            let [serving] = serving.as_slice() else {
                return None;
            };
            let knows = knowing.iter().any(|k| k.instance == serving.instance);
            (!knows).then(|| {
                format!(
                    "the instance `{}` of `{}` serves `{alias}`, to which the role `{role}` gives a location, and implements no {contract}:{tag}",
                    serving.name, robot.name
                )
            })
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Why `robot` cannot take `role`, none when it can: its software runs, it
/// has no robot commander of its own, it is simulated when the caller of the
/// run confirms assumptions, it implements each
/// contract, each of its instances that serves a call of the role that
/// gives a location implements locations:v1, and it reports the model of
/// the role. A named role and an
/// `"auto"` role both follow these rules.
pub fn misfits(
    workflow: &Workflow,
    snapshot: &Snapshot,
    role: &Name,
    robot: &Robot,
    confirmed: &BTreeSet<Name>,
) -> Vec<String> {
    let name = &robot.name;
    let mut reasons = Vec::new();
    if !robot.is_running() {
        reasons.push(format!("the software of `{name}` does not run"));
    }
    if !robot.robot_commander.is_empty() {
        reasons.push(format!("`{name}` has the robot commander `{}`, which moves it outside the leases; bind a robot without one", robot.robot_commander));
    }
    if robot.is_physical() && !confirmed.is_empty() {
        reasons.push(format!(
            "`{name}` is physical: a run whose caller confirms assumptions binds only simulated robots"
        ));
    }
    for alias in aliases_of_role(workflow, role) {
        let Some(contract) = workflow.contracts.get(&alias) else {
            continue;
        };
        if robot
            .contract(contract.name.as_str(), contract.tag())
            .is_none()
        {
            reasons.push(format!(
                "`{name}` does not implement `{contract}`, which the role `{role}` calls"
            ));
        }
    }
    reasons.extend(skills_that_know_no_locations(workflow, role, robot));
    if let Some(model) = &workflow.roles[role].model {
        let reported = snapshot.models.get(name);
        if reported != Some(model) {
            reasons.push(format!(
                "the role `{role}` needs the model `{model}`, and `{name}` reports {}",
                reported
                    .map(|m| format!("`{m}`"))
                    .unwrap_or_else(|| "no model".to_owned())
            ));
        }
    }
    reasons
}
