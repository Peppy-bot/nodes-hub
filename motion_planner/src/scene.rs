//! The planning scene's members on the wire: the requests of apply_collision_objects and
//! apply_attached_collision_object in the terms of motion_planning_core, whose
//! `PlanningScene` decides every rule of them, each object placed in the root of its frame's
//! chain, the robot frame's for an object that names no frame, at its timestamp and in the
//! robot frame at the latest time, and the scene as get_planning_scene answers it.

use std::time::SystemTime;

use chain_kinematics::nalgebra::Isometry3;
use control_core::geometry::{Point, Pose, Stamped, UnitQuaternion};
use control_core::robot_tree::PlaceError;
use frame_tree::FrameName;
use motion_planning_core::{AttachRequest, ObjectRequest, SceneSnapshot, ShapeFields};
use peppygen::exposed_services::motion_planning::get_planning_scene::{
    Response, ResponseAllowedPairsItem, ResponseAttachedObjectsItem,
    ResponseAttachedObjectsItemShapesItem, ResponseAttachedObjectsItemShapesItemMeshVerticesItem,
    ResponseObjectsItem, ResponseObjectsItemShapesItem,
    ResponseObjectsItemShapesItemMeshVerticesItem,
};
use peppygen::exposed_services::motion_planning::{
    apply_attached_collision_object, apply_collision_objects,
};

use crate::request::{RequestError, frame_or_robot, listed};

/// An entry of apply_collision_objects as its operation reads it: an add its pose and its
/// shapes, a move its pose, and a remove its id alone; each pose in the frame and at the
/// timestamp the entry gives.
pub enum ObjectRead {
    Add {
        id: String,
        pose: Stamped<Pose>,
        shapes: Vec<ShapeFields>,
    },
    Move {
        id: String,
        pose: Stamped<Pose>,
    },
    Remove {
        id: String,
    },
}

/// An entry of apply_collision_objects as the scene applies it, its object in the robot
/// frame, and, for an add or a move, where it places the object in the root of its frame's
/// chain, the frame that holds still across a move of the robot.
pub struct ObjectEntry {
    pub request: ObjectRequest,
    pub in_root: Option<Stamped<Pose>>,
}

/// What places an object's pose of apply_collision_objects, in the frame it names at its
/// timestamp: in the root of that frame's chain, stamped there at the latest time, and in the
/// robot frame; the planner's tree.
pub type ObjectPlacer<'p> =
    dyn Fn(&Stamped<Pose>) -> Result<(Stamped<Pose>, Pose), PlaceError> + 'p;

/// Why the robot's tree placed none of the objects that one placement places, of the planning
/// scene or of a request: their ids, and the failure of the placement.
#[derive(Debug)]
pub struct UnplacedObjects {
    pub ids: Vec<String>,
    pub error: Box<PlaceError>,
}

impl std::fmt::Display for UnplacedObjects {
    /// The objects by their ids, then the cause, as transform_lookup's message gives it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ids: Vec<String> = self.ids.iter().map(|id| format!("{id:?}")).collect();
        let objects = match ids.as_slice() {
            [_] => "the object",
            _ => "the objects",
        };
        write!(f, "{objects} {}: {}", listed(&ids, "and"), self.error)
    }
}

/// The entries of an apply_collision_objects request, in order, each as its operation reads
/// it, each pose in the frame it names, or in the robot frame `robot` for none; refused with
/// rejected, naming the entry and saying what to send, for an entry `read` refuses.
pub fn read_objects(
    raw: &apply_collision_objects::RequestData,
    robot: &FrameName,
) -> Result<Vec<ObjectRead>, RequestError> {
    (raw.objects.iter().enumerate())
        .map(|(k, object)| {
            read(object, robot).map_err(|e| format!("objects[{k}] ({:?}): {e}", object.id))
        })
        .collect::<Result<Vec<_>, String>>()
        .map_err(RequestError::Rejected)
}

/// The entries `read`, in order, each pose of an add or a move placed by `place` in the root of
/// its frame's chain at its timestamp and in the robot frame; or the failure of the first
/// placement `place` cannot make, naming its object.
pub fn place_objects(
    read: Vec<ObjectRead>,
    place: &ObjectPlacer,
) -> Result<Vec<ObjectEntry>, UnplacedObjects> {
    let placed = |id: &str, pose: &Stamped<Pose>| {
        place(pose)
            .map(|(in_root, in_robot)| (in_root, Isometry3::from(in_robot)))
            .map_err(|error| UnplacedObjects {
                ids: vec![id.to_string()],
                error: Box::new(error),
            })
    };
    read.into_iter()
        .map(|read| {
            Ok(match read {
                ObjectRead::Add { id, pose, shapes } => {
                    let (in_root, pose) = placed(&id, &pose)?;
                    ObjectEntry {
                        request: ObjectRequest::Add { id, pose, shapes },
                        in_root: Some(in_root),
                    }
                }
                ObjectRead::Move { id, pose } => {
                    let (in_root, pose) = placed(&id, &pose)?;
                    ObjectEntry {
                        request: ObjectRequest::Move { id, pose },
                        in_root: Some(in_root),
                    }
                }
                ObjectRead::Remove { id } => ObjectEntry {
                    request: ObjectRequest::Remove { id },
                    in_root: None,
                },
            })
        })
        .collect()
}

/// What `object` reads, as its operation names it, its pose in the robot frame `robot` when it
/// names no frame; refused, saying what to send, for an operation the contract does not name,
/// an add or a move with no position or orientation, an add with no shapes, and a field that
/// does not parse.
fn read(
    object: &apply_collision_objects::ApplyCollisionObjectsObjectsItem,
    robot: &FrameName,
) -> Result<ObjectRead, String> {
    let id = object.id.clone();
    match object.operation.as_str() {
        "add" => Ok(ObjectRead::Add {
            id,
            pose: pose_of(object, robot)?,
            shapes: shapes_of(object)?,
        }),
        "move" => Ok(ObjectRead::Move {
            id,
            pose: pose_of(object, robot)?,
        }),
        "remove" => Ok(ObjectRead::Remove { id }),
        given => Err(format!(
            "operation is {given:?}; it is \"add\", \"move\" or \"remove\""
        )),
    }
}

/// The pose an add or a move reads: in its frame, or in the robot frame `robot` for none, at
/// its timestamp, or at the latest time for none, its position and its orientation.
fn pose_of(
    object: &apply_collision_objects::ApplyCollisionObjectsObjectsItem,
    robot: &FrameName,
) -> Result<Stamped<Pose>, String> {
    let (Some(position), Some(orientation)) = (object.position, object.orientation) else {
        let missing: Vec<&str> = [
            ("position", object.position.is_none()),
            ("orientation", object.orientation.is_none()),
        ]
        .into_iter()
        .filter_map(|(name, absent)| absent.then_some(name))
        .collect();
        return Err(format!(
            "\"{}\" reads position and orientation: send {}",
            object.operation,
            listed(&missing, "and")
        ));
    };
    let pose = Pose::parse(position, orientation).map_err(|e| e.to_string())?;
    let frame = frame_or_robot(object.frame.as_deref(), robot);
    Stamped::parse(frame, object.timestamp, pose).map_err(|e| e.to_string())
}

/// The shapes an add reads, each posed relative to the object's pose.
fn shapes_of(
    object: &apply_collision_objects::ApplyCollisionObjectsObjectsItem,
) -> Result<Vec<ShapeFields>, String> {
    let shapes = (object.shapes.as_ref())
        .ok_or_else(|| "\"add\" reads shapes: send one or more".to_string())?;
    (shapes.iter().enumerate())
        .map(|(k, shape)| {
            let pose = Pose::parse(shape.position, shape.orientation)
                .map_err(|e| format!("shape {}: {e}", k + 1))?;
            Ok(ShapeFields {
                kind: shape.kind.clone(),
                dimensions: shape.dimensions.clone(),
                mesh_vertices: shape
                    .mesh_vertices
                    .iter()
                    .flat_map(|v| v.position)
                    .collect(),
                mesh_triangles: shape.mesh_triangles.clone(),
                position: pose.position().position(),
                orientation: pose.orientation().quaternion(),
            })
        })
        .collect()
}

/// The origin of a frame, as a pose.
pub fn origin() -> Pose {
    Pose::new(
        Point::parse([0.0; 3]).expect("the origin is finite"),
        UnitQuaternion::IDENTITY,
    )
}

/// An apply_attached_collision_object request as its operation reads it: an attach its object,
/// its link and its touch links, and a detach its object alone; refused with rejected, saying
/// what to send, for an operation the contract does not name.
pub fn attach_request(
    raw: &apply_attached_collision_object::RequestData,
) -> Result<AttachRequest, RequestError> {
    let object_id = raw.object_id.clone();
    match raw.operation.as_str() {
        "attach" => Ok(AttachRequest::Attach {
            object_id,
            link_name: raw.link_name.clone(),
            touch_links: raw.touch_links.clone(),
        }),
        "detach" => Ok(AttachRequest::Detach { object_id }),
        given => Err(RequestError::Rejected(format!(
            "operation is {given:?}; it is \"attach\" or \"detach\""
        ))),
    }
}

/// The shapes of an object as get_planning_scene answers them, each an item of the
/// generated type `$item`, its mesh's vertices items of `$vertex`.
macro_rules! shape_items {
    ($item:ident, $vertex:ident, $shapes:expr) => {
        $shapes
            .into_iter()
            .map(|shape: ShapeFields| $item {
                kind: shape.kind,
                dimensions: shape.dimensions,
                mesh_vertices: (shape.mesh_vertices.chunks_exact(3))
                    .map(|v| $vertex {
                        position: [v[0], v[1], v[2]],
                    })
                    .collect(),
                mesh_triangles: shape.mesh_triangles,
                position: shape.position,
                orientation: shape.orientation,
            })
            .collect()
    };
}

/// get_planning_scene's answer of the scene's `snapshot`: each object no link carries in
/// `frame`, the robot frame, at `timestamp`, and each attached object in the link that
/// carries it.
pub fn scene_response(snapshot: SceneSnapshot, frame: String, timestamp: SystemTime) -> Response {
    let objects = snapshot
        .objects
        .into_iter()
        .map(|object| ResponseObjectsItem {
            id: object.id,
            position: object.position,
            orientation: object.orientation,
            shapes: shape_items!(
                ResponseObjectsItemShapesItem,
                ResponseObjectsItemShapesItemMeshVerticesItem,
                object.shapes
            ),
        })
        .collect();
    let attached_objects = snapshot
        .attached_objects
        .into_iter()
        .map(|attached| ResponseAttachedObjectsItem {
            id: attached.object.id,
            frame: attached.link_name,
            touch_links: attached.touch_links,
            position: attached.object.position,
            orientation: attached.object.orientation,
            shapes: shape_items!(
                ResponseAttachedObjectsItemShapesItem,
                ResponseAttachedObjectsItemShapesItemMeshVerticesItem,
                attached.object.shapes
            ),
        })
        .collect();
    let allowed_pairs = snapshot
        .allowed_pairs
        .into_iter()
        .map(|[name_a, name_b]| ResponseAllowedPairsItem { name_a, name_b })
        .collect();
    Response::new(
        frame,
        timestamp,
        objects,
        attached_objects,
        allowed_pairs,
        snapshot.clearance_m,
        snapshot.voxel_size_m,
    )
}
