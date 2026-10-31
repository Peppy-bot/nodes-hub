//! The planning scene's members on the wire: the requests of apply_collision_objects and
//! apply_attached_collision_object in the terms of motion_planning_core, whose
//! `PlanningScene` decides every rule of them, and the scene as get_planning_scene
//! answers it.

use motion_planning_core::{
    AttachRequest, ObjectFields, ObjectRequest, SceneSnapshot, ShapeFields,
};
use peppygen::exposed_services::motion_planning::get_planning_scene::{
    Response, ResponseAllowedPairsItem, ResponseAttachedObjectsItem,
    ResponseAttachedObjectsItemShapesItem, ResponseObjectsItem, ResponseObjectsItemShapesItem,
};
use peppygen::exposed_services::motion_planning::{
    apply_attached_collision_object, apply_collision_objects,
};

/// The entries of an apply_collision_objects request, in order.
pub fn object_requests(raw: &apply_collision_objects::RequestData) -> Vec<ObjectRequest> {
    raw.objects
        .iter()
        .map(|object| ObjectRequest {
            operation: object.operation.clone(),
            object: ObjectFields {
                id: object.id.clone(),
                position: object.position,
                orientation: object.orientation,
                shapes: object
                    .shapes
                    .iter()
                    .map(|shape| ShapeFields {
                        kind: shape.kind.clone(),
                        dimensions: shape.dimensions.clone(),
                        mesh_vertices: shape.mesh_vertices.clone(),
                        mesh_triangles: shape.mesh_triangles.clone(),
                        position: shape.position,
                        orientation: shape.orientation,
                    })
                    .collect(),
            },
        })
        .collect()
}

/// An apply_attached_collision_object request.
pub fn attach_request(raw: &apply_attached_collision_object::RequestData) -> AttachRequest {
    AttachRequest {
        object_id: raw.object_id.clone(),
        link_name: raw.link_name.clone(),
        touch_links: raw.touch_links.clone(),
        operation: raw.operation.clone(),
    }
}

/// The shapes of an object as get_planning_scene answers them, each an item of the
/// generated type `$item`.
macro_rules! shape_items {
    ($item:ident, $shapes:expr) => {
        $shapes
            .into_iter()
            .map(|shape: ShapeFields| $item {
                kind: shape.kind,
                dimensions: shape.dimensions,
                mesh_vertices: shape.mesh_vertices,
                mesh_triangles: shape.mesh_triangles,
                position: shape.position,
                orientation: shape.orientation,
            })
            .collect()
    };
}

/// get_planning_scene's answer of the scene's `snapshot`.
pub fn scene_response(snapshot: SceneSnapshot) -> Response {
    let objects = snapshot
        .objects
        .into_iter()
        .map(|object| ResponseObjectsItem {
            id: object.id,
            position: object.position,
            orientation: object.orientation,
            shapes: shape_items!(ResponseObjectsItemShapesItem, object.shapes),
        })
        .collect();
    let attached_objects = snapshot
        .attached_objects
        .into_iter()
        .map(|attached| ResponseAttachedObjectsItem {
            id: attached.object.id,
            link_name: attached.link_name,
            touch_links: attached.touch_links,
            position: attached.object.position,
            orientation: attached.object.orientation,
            shapes: shape_items!(
                ResponseAttachedObjectsItemShapesItem,
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
        objects,
        attached_objects,
        allowed_pairs,
        snapshot.clearance_m,
        snapshot.voxel_size_m,
    )
}
