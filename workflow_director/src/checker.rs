//! The checker: the rules of a workflow that read its contracts, and the
//! binding of its roles to robots of the fleet listing. Checked with roles,
//! a workflow becomes the [`Plan`] that the engine runs.
//!
//! Without roles, each alias is checked against every pin of its contract
//! in the fleet listing. With roles, each role reads the pins of its robot,
//! each call and observation resolves to the instance that serves it, and
//! the item ids that a workflow passes are proven to come from the instance
//! that takes them.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use peppy_codec::contract::MOVES_ROBOT;
use serde_json::Value as Json;
use workflow_documents::expression::{Expression, FieldType, PathStep, RefPath, Type, TypeScope};
use workflow_documents::marks::{ITEM_ID, LOCATION, REACHES_FOR_ITEM, ROBOT_FRAME};
use workflow_documents::run::{Attendance, Incidents};
use workflow_documents::workflow::{Arg, ObservationSource, Step, StepKind};
use workflow_documents::{Located, Name, NameTag, RobotName, StepId, Workflow, robots_named_twice};

use crate::config::AllowedContracts;
use crate::contract::{ContractShape, Field, FieldKind, Format, Member};
use crate::engine::plan::{
    BoundArg, BoundCall, BoundObservation, BoundRole, CallSite, MemberKind, MonitorPlan,
    ObservationRead, Plan, Producer, SkillLocations, Target, Watch,
};
use crate::fleet::{LOCATIONS, Robot, Snapshot};

/// The run-wide choices that a plan carries beside its workflow.
pub struct RunChoices<'a> {
    pub attendance: Attendance,
    pub confirmed: BTreeSet<Name>,
    pub incidents: &'a Incidents,
    pub poll: workflow_documents::Bound,
    pub cancel_timeout: std::time::Duration,
    pub watch: Watch,
}

/// Where each name of a role comes from: the role whose call or
/// observation gives it, the producer of the result or the observation, and
/// the format of its value.
#[derive(Clone, Debug)]
struct Origin {
    role: Name,
    producer: Option<Producer>,
    field: Field,
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
    /// The locations each skill instance is given, by role.
    locations: Vec<SkillLocations>,
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

    fn trees(&self) -> Vec<(&'a Name, &'a Step)> {
        self.workflow
            .initialization
            .iter()
            .chain(
                self.workflow
                    .lanes
                    .iter()
                    .map(|(role, lane)| (role, &lane.tree)),
            )
            .collect()
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
                        let bound = self.args(&at, role, args, &found.input, None);
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
    /// of one type.
    fn check_tables(&mut self) {
        for (name, rows) in &self.workflow.tables {
            let mut row_type = None;
            for (key, row) in rows {
                let ty = literal_type(row);
                match (&row_type, ty) {
                    (_, None) => self.error(
                        format!("tables.{name}.{key}"),
                        "a row is a bool, a number, a string or a list of them",
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
            let trees: Vec<&Step> = self
                .workflow
                .initialization
                .get(&role)
                .into_iter()
                .chain(self.workflow.lanes.get(&role).map(|lane| &lane.tree))
                .collect();
            for tree in trees {
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
    /// the list field at that path.
    fn element_origin(&self, role: &Name, list: &Expression) -> Option<Origin> {
        let path = RefPath::parse(list.source()).ok()?;
        let origin = self.origin_of(role, &path)?;
        match origin.field.kind {
            FieldKind::List(item) => {
                let mut item = *item;
                item.marks.extend(origin.field.marks.iter().cloned());
                Some(Origin {
                    role: origin.role,
                    producer: origin.producer,
                    field: item,
                })
            }
            _ => None,
        }
    }

    /// The field that `path` reads, with the producer of its root.
    fn origin_of(&self, role: &Name, path: &RefPath) -> Option<Origin> {
        let root = self
            .origins
            .get(role)
            .and_then(|o| o.get(&path.root))
            .or_else(|| self.observation_origins.get(&path.root))?
            .clone();
        let mut field = root.field;
        for step in &path.steps {
            field = match (step, field.kind) {
                (PathStep::Field(name), FieldKind::Object(format)) => format.0.get(name)?.clone(),
                (PathStep::Index(_), FieldKind::List(item)) => {
                    let mut item = *item;
                    item.marks.extend(field.marks.iter().cloned());
                    item
                }
                _ => return None,
            };
        }
        Some(Origin {
            role: root.role,
            producer: root.producer,
            field,
        })
    }

    /// A row of a table that goes to a field with the mark location: a
    /// `{ $location }` of a declared location, which the skill takes by its
    /// name.
    #[allow(clippy::too_many_arguments)]
    fn location_row(
        &mut self,
        at: &str,
        role: &Name,
        name: &str,
        field: &Field,
        table: &Name,
        row_key: &str,
        row: &Json,
        producer: Option<&Producer>,
    ) -> Option<Json> {
        let Some(location) = location_of_row(row) else {
            self.error(
                at,
                format!("the row `{row_key}` of `{table}` for `{name}`: a location takes `{{ $location }}` rows"),
            );
            return None;
        };
        self.location_value(at, role, name, field, &location, producer)
    }

    /// The value of the field `name` with the mark location that a call of
    /// `role` on `producer` gives `location`: the name of the location, in
    /// the canonical form of the field, which the skill knows.
    fn location_value(
        &mut self,
        at: &str,
        role: &Name,
        name: &str,
        field: &Field,
        location: &Name,
        producer: Option<&Producer>,
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
        self.uses_location(role, location, producer);
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
            let bound = self.args(at, role, args, &found_member.input, producer.as_ref());
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

    /// The arguments bound against `format`: each field of the format, the
    /// required ones present, each literal in its canonical form, each field
    /// with a mark fed only as its mark says.
    fn args(
        &mut self,
        at: &str,
        role: &Name,
        args: &BTreeMap<String, Arg>,
        format: &Format,
        producer: Option<&Producer>,
    ) -> Option<BTreeMap<String, BoundArg>> {
        let mut bound = BTreeMap::new();
        let mut fine = true;
        for (name, field) in &format.0 {
            if !field.optional && !args.contains_key(name) {
                self.error(at, format!("`{name}` is a required argument"));
                fine = false;
            }
        }
        for (name, arg) in args {
            let Some(field) = format.0.get(name) else {
                self.error(at, format!("the member takes no argument `{name}`"));
                fine = false;
                continue;
            };
            match self.arg(at, role, name, arg, field, producer) {
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
        role: &Name,
        name: &str,
        arg: &Arg,
        field: &Field,
        producer: Option<&Producer>,
    ) -> Option<BoundArg> {
        let marked = field.marks.contains(ROBOT_FRAME) || field.marks.contains(ITEM_ID);
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
                .location_value(at, role, name, field, location, producer)
                .map(BoundArg::Literal),
            Arg::Location(_) => {
                self.error(
                    at,
                    format!("`{name}` takes no location: a `$location` goes only to a field with the mark location"),
                );
                None
            }
            Arg::Literal(_) | Arg::Object(_) | Arg::List(_) if marked => {
                self.error(at, format!("`{name}` holds a position, a pose or an item id; give it a `$ref` to a result or an observation"));
                None
            }
            Arg::Literal(literal) => match field.canonical(literal) {
                Ok(value) => Some(BoundArg::Literal(value)),
                Err(reason) => {
                    self.error(at, format!("`{name}`: {reason}"));
                    None
                }
            },
            Arg::Object(fields) => match &field.kind {
                FieldKind::Object(format) => self
                    .args(at, role, fields, format, producer)
                    .map(BoundArg::Object),
                _ => {
                    self.error(at, format!("`{name}` is no object"));
                    None
                }
            },
            Arg::List(items) => match &field.kind {
                FieldKind::List(item) => items
                    .iter()
                    .map(|a| self.arg(at, role, name, a, item, producer))
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
                if field.marks.contains(ROBOT_FRAME) {
                    if !origin.field.marks.contains(ROBOT_FRAME) {
                        self.error(
                            at,
                            format!("`{name}` takes a position or a pose in the frame of the robot, and `{path}` holds none"),
                        );
                        return None;
                    }
                    if origin.role != *role {
                        self.error(at, format!("`{path}` is in the frame of the robot of `{}`, and `{name}` goes to the robot of `{role}`: a position is valid only for the robot whose frame it is in", origin.role));
                        return None;
                    }
                }
                if field.marks.contains(ITEM_ID) {
                    if !origin.field.marks.contains(ITEM_ID) {
                        self.error(
                            at,
                            format!("`{name}` takes an item id, and `{path}` holds none"),
                        );
                        return None;
                    }
                    if let (Some(from), Some(to)) = (&origin.producer, producer)
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
                    self.error(
                        at,
                        format!(
                            "`{name}` holds a position, a pose or an item id; give it a `$ref`"
                        ),
                    );
                    return None;
                }
                let Some((value, Type::Table { keys, .. })) = self.tables.get(table).cloned()
                else {
                    self.error(
                        at,
                        format!("`$table` names `{table}`, which has no rows of one type"),
                    );
                    return None;
                };
                let rows: BTreeMap<String, Json> = value
                    .as_object()
                    .map(|rows| rows.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                let mut canonical = BTreeMap::new();
                for (row_key, row) in rows {
                    let value = match field.marks.contains(LOCATION) {
                        true => self
                            .location_row(at, role, name, field, table, &row_key, &row, producer)?,
                        false => match field.canonical(&row) {
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
                        },
                    };
                    canonical.insert(row_key, value);
                }
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

    /// A lane that calls a member that reaches for an item has a reach
    /// assumption whose check reads an observation of its role.
    fn check_reach(&mut self) {
        for (role, tree) in self.trees() {
            let mut reaches = Vec::new();
            tree.visit(&mut |step| {
                if let StepKind::Call(call) = &step.kind {
                    reaches.push((step.id.clone(), call.member.clone(), call.instance.clone()));
                }
            });
            for (id, member, instance) in reaches {
                let marked = self
                    .shapes(role, &member.alias, instance.as_ref())
                    .iter()
                    .any(|shape| {
                        shape
                            .members
                            .get(member.member.as_str())
                            .is_some_and(|m| m.marks.contains(REACHES_FOR_ITEM))
                    });
                if !marked {
                    continue;
                }
                let covered = self.workflow.assumptions.values().any(|assumption| {
                    assumption.reach
                        && assumption.check.as_ref().is_some_and(|check| {
                            check.roots().iter().any(|root| {
                                self.workflow
                                    .observations
                                    .get(root)
                                    .is_some_and(|o| &o.role == role)
                            })
                        })
                });
                if !covered {
                    self.error(
                        format!("{role}.{id}"),
                        format!("`{member}` reaches for an item; give the workflow a `kind: \"reach\"` assumption whose check reads an observation of `{role}`"),
                    );
                }
            }
        }
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
        locations: Vec::new(),
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
    let response_fits = ["anomaly", "kind", "severity", "description", "frame"]
        .iter()
        .all(|field| member.output.0.contains_key(*field));
    if !request_fits || !response_fits {
        return Err(format!(
            "`{contract}.{}` is no anomaly monitor: a service of {{ robot, step }} that answers {{ anomaly, kind, severity, description, frame }}",
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

/// The location that a row of a table names: `{ $location: <name> }`.
fn location_of_row(row: &Json) -> Option<Name> {
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
