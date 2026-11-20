//! The points, poses and twists of the requests of a workflow: the geometry
//! types of contracts-hub (its README, "Geometry fields") as the formats of a
//! contract hold them, the record that gives a plain value its frame and its
//! timestamp, and the records that a `$ref` gives a call.
//!
//! A record that names its frame holds `frame`: a stamped type, a constraint
//! or a collision object. Its `timestamp` is optional where a command leaves
//! it out for the latest time, or a report for values that hold at every
//! time; a record whose values always hold at every time, as a plan result,
//! holds no `timestamp`. A command's `frame` is optional too, absent for its
//! robot frame. A plain value (a point, a quaternion, a pose or a
//! twist) holds its fields and no frame, in a record that gives one `frame`
//! and its `timestamp`, itself or in its `header`; a twist beside a
//! `child_frame`, as an odometry record holds it, is in that frame. A plain
//! value in a record that holds a position or an orientation of its own is
//! posed relative to that record's pose, as a shape of a collision object is
//! posed relative to its object and a vertex of a mesh to its shape.

use std::collections::{BTreeMap, BTreeSet};

use control_core::geometry::WORLD;
use serde_json::Value as Json;
use workflow_documents::Name;
use workflow_documents::expression::{PathStep, RefPath};

use crate::contract::{Field, FieldKind, Format};
use crate::engine::plan::{Unsendable, read_field, read_path};

pub const FRAME: &str = "frame";
pub const TIMESTAMP: &str = "timestamp";
const CHILD_FRAME: &str = "child_frame";
const HEADER: &str = "header";
const ORIENTATION: &str = "orientation";
const POSITION: &str = "position";
const LINEAR_VELOCITY: &str = "linear_velocity";
const ANGULAR_VELOCITY: &str = "angular_velocity";
/// The fields of the plain types: the position of a point, the orientation
/// of a quaternion, both of a pose, and the velocities of a twist.
const PLAIN_FIELDS: [&str; 4] = [POSITION, ORIENTATION, LINEAR_VELOCITY, ANGULAR_VELOCITY];

/// Why a point, a pose or a twist of one robot goes to a call of another in
/// world alone.
pub const BETWEEN_ROBOTS_IN_WORLD: &str = "a point, a pose or a twist passes between robots in world alone, a frame their trees hold above their robot frames where a node publishes it, and no node publishes it into a robot's tree";

/// Whether a record names its frame: it holds `frame`.
pub fn names_its_frame(record: &Format) -> bool {
    record.0.contains_key(FRAME)
}

/// Whether the field `name` of a record is a plain field: an array of
/// numbers that one of [`PLAIN_FIELDS`] names.
fn is_plain_field(name: &str, field: &Field) -> bool {
    let numbers = matches!(&field.kind, FieldKind::List(item) if item.kind == FieldKind::Double);
    numbers && PLAIN_FIELDS.contains(&name)
}

/// Whether `record` holds the plain field `name`.
fn holds_plain(record: &Format, name: &str) -> bool {
    record
        .0
        .get(name)
        .is_some_and(|field| is_plain_field(name, field))
}

fn holds_plain_fields(record: &Format) -> bool {
    PLAIN_FIELDS.iter().any(|name| holds_plain(record, name))
}

/// The record that a value of `field` is, or that each item of its list is
/// (`each`).
pub fn records_of(field: &Field) -> Option<(&Format, bool)> {
    let (kind, each) = match &field.kind {
        FieldKind::List(item) => (&item.kind, true),
        kind => (kind, false),
    };
    match kind {
        FieldKind::Object(record) => Some((record, each)),
        _ => None,
    }
}

/// The record that names its frame which `field` takes: its own, or that of
/// each item of its list (`each`).
pub fn framed_records(field: &Field) -> Option<(&Format, bool)> {
    records_of(field).filter(|(record, _)| names_its_frame(record))
}

/// What a record that names its frame holds, as a workflow names it: one of
/// them, and many.
pub fn kind_of(record: &Format) -> (&'static str, &'static str) {
    let holds = |name| holds_plain(record, name);
    match (
        holds(POSITION),
        holds(ORIENTATION),
        holds(LINEAR_VELOCITY) || holds(ANGULAR_VELOCITY),
    ) {
        (true, true, _) => ("a pose", "poses"),
        (true, false, _) => ("a point", "points"),
        (false, _, true) => ("a twist", "twists"),
        (false, true, false) => ("an orientation", "orientations"),
        (false, false, false) => ("a record", "records"),
    }
}

/// Whether the field `name` of a record holds a value in that record's
/// frame: a plain field, or an object or a list that holds one and names no
/// frame of its own.
pub fn in_the_records_frame(name: &str, field: &Field) -> bool {
    if is_plain_field(name, field) {
        return true;
    }
    match &field.kind {
        FieldKind::Object(record) => {
            !names_its_frame(record)
                && record
                    .0
                    .iter()
                    .any(|(name, field)| in_the_records_frame(name, field))
        }
        FieldKind::List(item) => in_the_records_frame("", item),
        _ => false,
    }
}

/// Whether `field` holds a record that names its frame, at any depth.
pub fn holds_framed_records(field: &Field) -> bool {
    match &field.kind {
        FieldKind::Object(record) => {
            names_its_frame(record) || record.0.values().any(holds_framed_records)
        }
        FieldKind::List(item) => holds_framed_records(item),
        _ => false,
    }
}

/// Whether a value of `field` is a point, a pose or a twist, or a list of
/// them: a record that names its frame or holds a plain field.
pub fn is_geometry(field: &Field) -> bool {
    match &field.kind {
        FieldKind::Object(record) => names_its_frame(record) || holds_plain_fields(record),
        FieldKind::List(item) => is_geometry(item),
        _ => false,
    }
}

/// The frame of each record that names its frame in `value`, a literal of
/// `field`. A record that leaves its frame out names none: it lies in the
/// frame of the robot of its call, its robot frame or world in a simulation,
/// and goes with no frame.
pub fn literal_frames<'v>(field: &Field, value: &'v Json) -> Vec<&'v str> {
    match (&field.kind, value) {
        (FieldKind::Object(record), Json::Object(object)) => {
            let own = names_its_frame(record)
                .then(|| object.get(FRAME).and_then(Json::as_str))
                .flatten();
            own.into_iter()
                .chain(record.0.iter().flat_map(|(name, field)| {
                    object
                        .get(name)
                        .map(|value| literal_frames(field, value))
                        .unwrap_or_default()
                }))
                .collect()
        }
        (FieldKind::List(item), Json::Array(items)) => items
            .iter()
            .flat_map(|value| literal_frames(item, value))
            .collect(),
        _ => Vec::new(),
    }
}

/// The record that holds a value that a `$ref` reads, with its path in the
/// scope of a call.
#[derive(Clone, Debug)]
pub struct Holder {
    pub path: RefPath,
    pub record: Format,
}

/// What holds a value that a `$ref` reads, which gives a plain value its
/// frame and its timestamp.
#[derive(Clone, Debug)]
pub enum Holding {
    /// The value is a result or an observation whole.
    Root,
    /// The value is a field of the record, or an item of a list field of it.
    Record(Holder),
    /// The value is an element of a loop over `list`, a list of an
    /// observation, which the loop reads from one message.
    ObservationElement { list: RefPath },
}

/// Where a call reads the frame and the timestamp of a plain value: no
/// timestamp from a record whose values hold at every time, which holds
/// none.
#[derive(Clone, Debug, PartialEq)]
pub struct Stamp {
    pub frame: RefPath,
    pub timestamp: Option<RefPath>,
}

impl Stamp {
    /// The frame and the timestamp of `record` at `path`, which names its
    /// own.
    pub fn of_record(path: &RefPath, record: &Format) -> Stamp {
        Stamp {
            frame: field_path(path, &[FRAME]),
            timestamp: timestamp_in(path, record),
        }
    }

    /// The stamp that `record` at `path` gives in its field `frame`, when it
    /// holds one.
    fn named_in(path: &RefPath, record: &Format, frame: &str) -> Option<Stamp> {
        record.0.contains_key(frame).then(|| Stamp {
            frame: field_path(path, &[frame]),
            timestamp: timestamp_in(path, record),
        })
    }
}

/// The path of the timestamp of `record` at `path`: none for a record that
/// holds no timestamp, as its values hold at every time.
fn timestamp_in(path: &RefPath, record: &Format) -> Option<RefPath> {
    record
        .0
        .contains_key(TIMESTAMP)
        .then(|| field_path(path, &[TIMESTAMP]))
}

/// The path of the field that `names` reach, one by one, from `path`.
fn field_path(path: &RefPath, names: &[&str]) -> RefPath {
    RefPath {
        root: path.root.clone(),
        steps: (path.steps.iter().cloned())
            .chain(names.iter().map(|name| PathStep::Field((*name).to_owned())))
            .collect(),
    }
}

impl Holder {
    /// Whether the record holds a position or an orientation of its own, to
    /// which the plain values it holds are posed relative.
    fn holds_a_pose(&self) -> bool {
        holds_plain(&self.record, POSITION) || holds_plain(&self.record, ORIENTATION)
    }

    /// The frame and the timestamp that this record gives the plain value
    /// `value`: its own or its header's, and its `child_frame` for a twist
    /// beside one.
    fn stamp_of(&self, value: &Format) -> Option<Stamp> {
        let twist = [LINEAR_VELOCITY, ANGULAR_VELOCITY]
            .iter()
            .any(|name| holds_plain(value, name));
        let frame = match twist && self.record.0.contains_key(CHILD_FRAME) {
            true => CHILD_FRAME,
            false => FRAME,
        };
        if let Some(stamp) = Stamp::named_in(&self.path, &self.record, frame) {
            return Some(stamp);
        }
        let Some(FieldKind::Object(header)) = self.record.0.get(HEADER).map(|field| &field.kind)
        else {
            return None;
        };
        Stamp::named_in(&field_path(&self.path, &[HEADER]), header, FRAME)
    }
}

/// Where the records that a `$ref` to `path` reads take their frame and
/// their timestamp: none for a `record` that names its frame, the frame and
/// the timestamp of the record that holds it (`holding`) for a plain value;
/// or why they name no one frame.
pub fn stamp_of(
    path: &RefPath,
    record: &Format,
    holding: &Holding,
) -> Result<Option<Stamp>, String> {
    match (
        record.0.contains_key(FRAME),
        record.0.contains_key(TIMESTAMP),
    ) {
        (true, _) => return Ok(None),
        (false, true) => {
            return Err(format!(
                "`{path}` names its time and no frame; {GIVE_A_STAMPED_REF}"
            ));
        }
        (false, false) => {}
    }
    if !holds_plain_fields(record) {
        return Err(format!(
            "`{path}` holds no point, pose or twist; {GIVE_A_STAMPED_REF}"
        ));
    }
    let holder = match holding {
        Holding::Record(holder) => holder,
        Holding::Root => {
            return Err(format!(
                "`{path}` names no frame and no time, and no record holds it; {GIVE_A_STAMPED_REF}"
            ));
        }
        Holding::ObservationElement { list } => {
            return Err(format!(
                "`{path}` is an element of `{list}`, a list of an observation, whose message its loop read once, so a call in the loop reads no frame and no time for it; give a `$ref` to an item of the list by its index, such as `{list}[0]`, which brings the frame and the timestamp of the message, or loop over a list of a result"
            ));
        }
    };
    if holder.holds_a_pose() {
        return Err(format!(
            "`{path}` is posed relative to the position and the orientation of `{}`, as a shape of a collision object is posed relative to its object; {GIVE_A_STAMPED_REF}",
            holder.path
        ));
    }
    holder.stamp_of(record).map(Some).ok_or_else(|| {
        format!(
            "`{path}` names no frame and no time, and `{}`, the record that holds it, gives no `frame`; {GIVE_A_STAMPED_REF}",
            holder.path
        )
    })
}

/// What a refusal of a `$ref` to a value with no one frame says to give.
const GIVE_A_STAMPED_REF: &str = "give a `$ref` to a point, a pose or a twist that names its frame, or to one in a record that names one";

/// The two roles of a value that passes between them: the role whose call
/// or observation gives it and the role of the call that takes it.
#[derive(Clone, Debug, PartialEq)]
pub struct Crossing {
    pub from: Name,
    pub to: Name,
}

impl Crossing {
    /// The roles that a value of the role `from` passes between to a call of
    /// the role `to`: none when they are one role.
    pub fn between(from: &Name, to: &Name) -> Option<Crossing> {
        (from != to).then(|| Crossing {
            from: from.clone(),
            to: to.clone(),
        })
    }

    /// Admits the value at `path`, in `frame`, to the call in world alone.
    fn admit(&self, path: &RefPath, frame: &str) -> Result<(), Unsendable> {
        if frame == WORLD {
            return Ok(());
        }
        Err(Unsendable::OfAnotherRobot {
            path: path.to_string(),
            frame: frame.to_owned(),
            from: self.from.clone(),
            to: self.to.clone(),
        })
    }
}

/// The frame that a request in one frame takes with the values that a
/// `$ref` gives it: the frame of their record, at `frame`. `values` is the
/// path of the first value, and `crossing` the roles the values pass between
/// when the role that gives them is not the role of the call: the run then
/// sends the request in world alone.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameOfValues {
    pub frame: RefPath,
    pub values: RefPath,
    pub crossing: Option<Crossing>,
}

impl FrameOfValues {
    /// The frame in `scope`.
    pub fn resolve(&self, scope: &BTreeMap<Name, Json>) -> Result<Json, Unsendable> {
        let frame = read_path(scope, &self.frame)?;
        if let Some(crossing) = &self.crossing {
            crossing.admit(&self.values, frame.as_str().unwrap_or_default())?;
        }
        Ok(frame)
    }
}

/// The records that name their frame which a `$ref` gives a field: the
/// record at `value`, or each record of the list there (`each`), with the
/// frame and the timestamp of `stamp` for a plain value, and the fields of
/// the target alone. A record that leaves its timestamp out, or holds none,
/// for values that hold at every time, goes with none, which a request reads
/// as the latest time.
#[derive(Clone, Debug, PartialEq)]
pub struct Placed {
    pub value: RefPath,
    pub each: bool,
    pub stamp: Option<Stamp>,
    /// The fields of the record that the target takes.
    pub fields: BTreeSet<String>,
    /// The roles the records pass between, when the role that gives them is
    /// not the role of the call: each then names world.
    pub crossing: Option<Crossing>,
}

impl Placed {
    /// The root of each path that the records read.
    pub fn roots(&self) -> BTreeSet<Name> {
        let stamp = self
            .stamp
            .iter()
            .flat_map(|stamp| std::iter::once(&stamp.frame).chain(&stamp.timestamp));
        std::iter::once(&self.value)
            .chain(stamp)
            .map(|path| path.root.clone())
            .collect()
    }

    /// The records in `scope`.
    pub fn resolve(&self, scope: &BTreeMap<Name, Json>) -> Result<Json, Unsendable> {
        let value = read_path(scope, &self.value)?;
        let stamp = match &self.stamp {
            Some(stamp) => {
                let timestamp = match &stamp.timestamp {
                    Some(timestamp) => read_field(scope, timestamp)?,
                    None => None,
                };
                Some((read_path(scope, &stamp.frame)?, timestamp))
            }
            None => None,
        };
        let placed = |record: &Json| self.record(record, stamp.as_ref());
        if !self.each {
            return placed(&value);
        }
        value
            .as_array()
            .ok_or_else(|| Unsendable::no_value(&self.value))?
            .iter()
            .map(placed)
            .collect::<Result<Vec<_>, _>>()
            .map(Json::Array)
    }

    fn record(
        &self,
        record: &Json,
        stamp: Option<&(Json, Option<Json>)>,
    ) -> Result<Json, Unsendable> {
        let mut placed: serde_json::Map<String, Json> = self
            .fields
            .iter()
            .filter_map(|name| Some((name.clone(), record.get(name)?.clone())))
            .collect();
        if let Some((frame, timestamp)) = stamp {
            placed.insert(FRAME.to_owned(), frame.clone());
            if let Some(timestamp) = timestamp {
                placed.insert(TIMESTAMP.to_owned(), timestamp.clone());
            }
        }
        if let Some(crossing) = &self.crossing {
            let frame = placed.get(FRAME).and_then(Json::as_str).unwrap_or_default();
            crossing.admit(&self.value, frame)?;
        }
        Ok(Json::Object(placed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::ContractShape;
    use crate::contract::tests::TEST_ARM;
    use serde_json::json;

    fn path(text: &str) -> RefPath {
        RefPath::parse(text).unwrap()
    }

    fn name(text: &str) -> Name {
        Name::parse(text).unwrap()
    }

    /// The format of a member's output in the arm of the tests.
    fn output(member: &str) -> Format {
        let shape = ContractShape::parse(TEST_ARM, &"a".repeat(64)).unwrap();
        shape.members[member].output.clone()
    }

    fn object(format: &Format, field: &str) -> Format {
        match &format.0[field].kind {
            FieldKind::Object(record) => record.clone(),
            FieldKind::List(item) => match &item.kind {
                FieldKind::Object(record) => record.clone(),
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
    }

    /// A stamped value names its own frame and time, or its frame alone where
    /// its values hold at every time; a plain value takes those of the record
    /// that holds it, its header's, or the child frame for a twist beside
    /// one, and the frame alone of a record that holds no timestamp; a value
    /// in a record without a frame, or one that names its time alone, is
    /// refused.
    #[test]
    fn a_plain_value_takes_the_frame_and_the_timestamp_of_its_record() {
        let found = output("find");
        assert_eq!(
            stamp_of(
                &path("seen.item_point"),
                &object(&found, "item_point"),
                &Holding::Root
            ),
            Ok(None)
        );
        let scene = output("get_planning_scene");
        let holder = Holding::Record(Holder {
            path: path("scene"),
            record: scene.clone(),
        });
        let objects = object(&scene, "objects");
        assert_eq!(
            stamp_of(&path("scene.objects[0]"), &objects, &holder),
            Ok(Some(Stamp {
                frame: path("scene.frame"),
                timestamp: Some(path("scene.timestamp")),
            }))
        );
        let odometry = output("odometry");
        let held = Holding::Record(Holder {
            path: path("odo"),
            record: odometry.clone(),
        });
        assert_eq!(
            stamp_of(&path("odo.twist"), &object(&odometry, "twist"), &held),
            Ok(Some(Stamp {
                frame: path("odo.child_frame"),
                timestamp: Some(path("odo.timestamp")),
            }))
        );
        assert_eq!(
            stamp_of(&path("odo.pose"), &object(&odometry, "pose"), &held),
            Ok(Some(Stamp {
                frame: path("odo.frame"),
                timestamp: Some(path("odo.timestamp")),
            }))
        );
        let plan = output("plan_path");
        let planned = Holding::Record(Holder {
            path: path("plan"),
            record: plan.clone(),
        });
        assert_eq!(
            stamp_of(
                &path("plan.waypoints[0]"),
                &object(&plan, "waypoints"),
                &planned
            ),
            Ok(Some(Stamp {
                frame: path("plan.frame"),
                timestamp: None,
            }))
        );
        let cloud = output("cloud");
        let sensed = Holding::Record(Holder {
            path: path("cloud"),
            record: cloud.clone(),
        });
        assert_eq!(
            stamp_of(&path("cloud.points[1]"), &object(&cloud, "points"), &sensed),
            Ok(Some(Stamp {
                frame: path("cloud.header.frame"),
                timestamp: Some(path("cloud.header.timestamp")),
            }))
        );
        let attached = Format(BTreeMap::from([
            (FRAME.to_owned(), object_field(&found, "item_point", FRAME)),
            (
                "position".to_owned(),
                object_field(&found, "item_point", "position"),
            ),
        ]));
        assert_eq!(
            stamp_of(&path("scene.attached[0]"), &attached, &holder),
            Ok(None)
        );
        let timed = Format(BTreeMap::from([
            (
                TIMESTAMP.to_owned(),
                object_field(&found, "item_point", TIMESTAMP),
            ),
            (
                "position".to_owned(),
                object_field(&found, "item_point", "position"),
            ),
        ]));
        assert_eq!(
            stamp_of(&path("scene.timed[0]"), &timed, &holder),
            Err(format!(
                "`scene.timed[0]` names its time and no frame; {GIVE_A_STAMPED_REF}"
            ))
        );
        let unframed = Holding::Record(Holder {
            path: path("bare"),
            record: Format(BTreeMap::from([(
                "tips".to_owned(),
                cloud.0["points"].clone(),
            )])),
        });
        assert_eq!(
            stamp_of(&path("bare.tips[0]"), &object(&cloud, "points"), &unframed),
            Err(format!(
                "`bare.tips[0]` names no frame and no time, and `bare`, the record that holds it, gives no `frame`; {GIVE_A_STAMPED_REF}"
            ))
        );
        assert_eq!(
            stamp_of(&path("tip"), &object(&cloud, "points"), &Holding::Root),
            Err(format!(
                "`tip` names no frame and no time, and no record holds it; {GIVE_A_STAMPED_REF}"
            ))
        );
        assert_eq!(
            stamp_of(&path("seen"), &found, &Holding::Root),
            Err(format!(
                "`seen` holds no point, pose or twist; {GIVE_A_STAMPED_REF}"
            ))
        );
    }

    /// A plain value in a record that holds a pose of its own is posed relative to that pose: a shape of a
    /// collision object, a vertex of the mesh of a shape, and a shape of an object that names its frame and its
    /// time are refused.
    #[test]
    fn a_value_posed_relative_to_its_record_is_refused() {
        let scene = output("get_planning_scene");
        let objects = object(&scene, "objects");
        let shapes = object(&objects, "shapes");
        let relative = |value: &str, record: &str, format: &Format, holder: Format| {
            assert_eq!(
                stamp_of(
                    &path(value),
                    format,
                    &Holding::Record(Holder {
                        path: path(record),
                        record: holder,
                    })
                ),
                Err(format!(
                    "`{value}` is posed relative to the position and the orientation of `{record}`, as a shape of a collision object is posed relative to its object; {GIVE_A_STAMPED_REF}"
                ))
            );
        };
        relative(
            "scene.objects[0].shapes[0]",
            "scene.objects[0]",
            &shapes,
            objects.clone(),
        );
        relative(
            "scene.objects[0].shapes[0].mesh_vertices[2]",
            "scene.objects[0].shapes[0]",
            &object(&shapes, "mesh_vertices"),
            shapes.clone(),
        );
        let mut stamped_object = objects.clone();
        stamped_object
            .0
            .insert(FRAME.to_owned(), scene.0[FRAME].clone());
        stamped_object
            .0
            .insert(TIMESTAMP.to_owned(), scene.0[TIMESTAMP].clone());
        relative(
            "added.objects[0].shapes[0]",
            "added.objects[0]",
            &shapes,
            stamped_object,
        );
    }

    /// An element of a loop over the list of an observation, which the loop reads once, is refused naming the list
    /// and what to give instead.
    #[test]
    fn an_element_of_a_list_of_an_observation_is_refused_saying_what_to_give() {
        let states = Format(BTreeMap::from([(
            "position".to_owned(),
            object_field(&output("find"), "item_point", "position"),
        )]));
        assert_eq!(
            stamp_of(
                &path("point"),
                &states,
                &Holding::ObservationElement {
                    list: path("arms.grasp_points")
                }
            ),
            Err("`point` is an element of `arms.grasp_points`, a list of an observation, whose message its loop read once, so a call in the loop reads no frame and no time for it; give a `$ref` to an item of the list by its index, such as `arms.grasp_points[0]`, which brings the frame and the timestamp of the message, or loop over a list of a result".to_owned())
        );
    }

    fn object_field(format: &Format, record: &str, field: &str) -> Field {
        object(format, record).0[field].clone()
    }

    /// The fields in a record's frame are its plain fields, its plain
    /// objects and the records that hold them; a record that names its own
    /// frame and a value of another kind are not.
    #[test]
    fn the_fields_in_a_records_frame_are_its_plain_values() {
        let scene = output("get_planning_scene");
        let objects = &scene.0["objects"];
        assert!(in_the_records_frame("objects", objects));
        assert!(!in_the_records_frame(FRAME, &scene.0[FRAME]));
        assert!(!in_the_records_frame(TIMESTAMP, &scene.0[TIMESTAMP]));
        let found = output("find");
        assert!(!in_the_records_frame("item_point", &found.0["item_point"]));
        assert!(!in_the_records_frame("item_id", &found.0["item_id"]));
        assert!(holds_framed_records(&found.0["item_point"]));
        assert!(is_geometry(objects) && is_geometry(&found.0["item_point"]));
        assert!(!is_geometry(&found.0["item_id"]));
    }

    /// A field is a plain field by its shape and its name: a `position` that
    /// is no array of numbers, and a `pose` that names its own frame, hold no
    /// value in the frame of their record.
    #[test]
    fn a_plain_field_is_an_array_of_numbers_that_a_plain_type_names() {
        let field = |kind: FieldKind| Field {
            kind,
            optional: false,
            marks: BTreeSet::new(),
        };
        let found = output("find");
        let position = object_field(&found, "item_point", POSITION);
        assert!(in_the_records_frame(POSITION, &position));
        assert!(!in_the_records_frame("item_id", &position));
        let text = field(FieldKind::String);
        assert!(!in_the_records_frame(POSITION, &text));
        let named = Format(BTreeMap::from([(POSITION.to_owned(), text)]));
        assert!(!is_geometry(&field(FieldKind::Object(named))));
        let framed_pose = found.0["item_point"].clone();
        assert!(!in_the_records_frame("pose", &framed_pose));
        let odometry = output("odometry");
        assert!(in_the_records_frame("pose", &odometry.0["pose"]));
    }

    /// A literal names the frame of each record that names its frame, at
    /// any depth, and none for a record that leaves its frame out.
    #[test]
    fn a_literal_names_the_frame_of_each_of_its_records() {
        let shape = ContractShape::parse(TEST_ARM, &"a".repeat(64)).unwrap();
        let points = &shape.members["check_reach"].input.0["points"];
        let FieldKind::List(point) = &points.kind else {
            panic!("{points:?}");
        };
        let literal = json!({ "frame": "map", "position": [1.0, 0.0, 0.0] });
        assert_eq!(literal_frames(point, &literal), ["map"]);
        let both = json!([literal, { "frame": "odom", "position": [0.0, 0.0, 0.0] }]);
        assert_eq!(literal_frames(points, &both), ["map", "odom"]);
        let unframed = json!([{ "position": [0.3, 0.0, 0.1] }]);
        assert!(literal_frames(points, &unframed).is_empty());
    }

    fn placed(stamp: Option<Stamp>, fields: &[&str], crossing: Option<Crossing>) -> Placed {
        Placed {
            value: path("state.grasp_points[1]"),
            each: false,
            stamp,
            fields: fields.iter().map(|field| (*field).to_owned()).collect(),
            crossing,
        }
    }

    /// A plain value reaches its target with the frame and the timestamp of
    /// its record, and with no timestamp from a record that leaves it out;
    /// each value reaches it with the fields of the target alone, a point to
    /// a pose with no orientation.
    #[test]
    fn a_plain_value_reaches_its_target_as_its_stamped_type() {
        let scope = BTreeMap::from([(
            name("state"),
            json!({
                "frame": "base_link",
                "timestamp": "2026-11-18T12:00:00.250000000Z",
                "arm_names": ["left_arm", "right_arm"],
                "grasp_points": [
                    { "position": [0.1, 0.2, 0.3], "orientation": [0.0, 0.0, 0.0, 1.0] },
                    { "position": [0.4, 0.5, 0.6], "orientation": [0.0, 0.0, 1.0, 0.0] },
                ],
            }),
        )]);
        let stamp = Some(Stamp {
            frame: path("state.frame"),
            timestamp: Some(path("state.timestamp")),
        });
        let pose = [FRAME, TIMESTAMP, "position", ORIENTATION];
        assert_eq!(
            placed(stamp.clone(), &pose, None).resolve(&scope),
            Ok(json!({
                "frame": "base_link",
                "timestamp": "2026-11-18T12:00:00.250000000Z",
                "position": [0.4, 0.5, 0.6],
                "orientation": [0.0, 0.0, 1.0, 0.0],
            }))
        );
        let point = [FRAME, TIMESTAMP, "position"];
        assert_eq!(
            placed(stamp.clone(), &point, None).resolve(&scope),
            Ok(json!({
                "frame": "base_link",
                "timestamp": "2026-11-18T12:00:00.250000000Z",
                "position": [0.4, 0.5, 0.6],
            }))
        );
        let mut of_a_point = placed(None, &pose, None);
        of_a_point.value = path("seen.item_point");
        let seen = BTreeMap::from([(
            name("seen"),
            json!({ "item_point": { "frame": "base_link", "timestamp": "2026-11-18T12:00:01Z", "position": [0.3, 0.0, 0.1] } }),
        )]);
        assert_eq!(
            of_a_point.resolve(&seen),
            Ok(json!({
                "frame": "base_link",
                "timestamp": "2026-11-18T12:00:01Z",
                "position": [0.3, 0.0, 0.1],
            }))
        );
        let each = Placed {
            value: path("state.grasp_points"),
            each: true,
            ..placed(stamp.clone(), &point, None)
        };
        let both = each.resolve(&scope).unwrap();
        assert_eq!(both.as_array().map(Vec::len), Some(2));
        assert_eq!(both[0]["frame"], json!("base_link"));
        let mut at_every_time = scope.clone();
        let state = at_every_time.get_mut(&name("state")).unwrap();
        state.as_object_mut().unwrap().remove(TIMESTAMP);
        assert_eq!(
            placed(stamp.clone(), &pose, None).resolve(&at_every_time),
            Ok(json!({
                "frame": "base_link",
                "position": [0.4, 0.5, 0.6],
                "orientation": [0.0, 0.0, 1.0, 0.0],
            }))
        );
        assert_eq!(
            placed(stamp, &pose, None).resolve(&BTreeMap::new()),
            Err(Unsendable::no_value(&path("state.grasp_points[1]")))
        );
    }

    /// A value of another role reaches the call in world alone.
    #[test]
    fn a_value_of_another_role_goes_only_in_world() {
        let crossing = Some(Crossing {
            from: name("alpha"),
            to: name("beta"),
        });
        let mut of_alpha = placed(None, &[FRAME, TIMESTAMP, "position"], crossing);
        of_alpha.value = path("seen.item_point");
        let in_frame = |frame: &str| {
            BTreeMap::from([(
                name("seen"),
                json!({ "item_point": { "frame": frame, "timestamp": "2026-11-18T12:00:01Z", "position": [0.3, 0.0, 0.1] } }),
            )])
        };
        assert!(of_alpha.resolve(&in_frame(WORLD)).is_ok());
        let refused = of_alpha.resolve(&in_frame("base_link")).unwrap_err();
        assert_eq!(
            refused,
            Unsendable::OfAnotherRobot {
                path: "seen.item_point".to_owned(),
                frame: "base_link".to_owned(),
                from: name("alpha"),
                to: name("beta"),
            }
        );
        assert_eq!(
            refused.to_string(),
            "`seen.item_point` is in `base_link`, a frame of the robot of `alpha`, and goes to the robot of `beta`: a point, a pose or a twist passes between robots in world alone, a frame their trees hold above their robot frames where a node publishes it, and no node publishes it into a robot's tree; give the call a point, a pose or a twist of a report of the robot of `beta` by `$ref`"
        );
    }
}
