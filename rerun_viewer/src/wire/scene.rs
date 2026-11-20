//! The planning scene: the arms as the motion planner describes them, asked once, then
//! the scene read at the poll rate and drawn each time it changes.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use control_core::asking;
use control_core::latched_log::LatchedLog;
use peppygen::NodeRunner;
use peppygen::consumed_services::motion_planning::{describe_arms, get_planning_scene};
use tokio::time::MissedTickBehavior;

use crate::canvas::Canvas;
use crate::draw::scene::draw_scene;
use crate::model::arms::{ArmDescription, ArmsDescription};
use crate::model::changes::scene_changes;
use crate::model::placement::Placement;
use crate::model::scene::{AttachedObject, PlanningScene, SceneObject, Shape, ShapeKind};
use crate::wire::{SERVICE_TIMEOUT, Shared, node_now, sendable, stopped};

/// A planning scene the viewer cannot read.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SceneError {
    #[error("object {id}: {reason}")]
    Object { id: String, reason: String },
    #[error("clearance_m is {0}; a clearance is 0 m or more")]
    Clearance(f64),
}

/// The arms as `description` names them.
pub fn parse_arms_description(description: &describe_arms::ResponseData) -> ArmsDescription {
    ArmsDescription {
        arms: description
            .arms
            .iter()
            .map(|arm| ArmDescription {
                name: arm.name.clone(),
                joint_names: arm.joint_names.clone(),
                grasp_link: arm.grasp_link.clone(),
            })
            .collect(),
    }
}

/// The planning scene `data` holds.
pub fn parse_scene(data: &get_planning_scene::ResponseData) -> Result<PlanningScene, SceneError> {
    if !(data.clearance_m.is_finite() && data.clearance_m >= 0.0) {
        return Err(SceneError::Clearance(data.clearance_m));
    }
    let objects = data
        .objects
        .iter()
        .map(|object| {
            let shapes = object.shapes.iter().map(|shape| ShapeFields {
                kind: &shape.kind,
                dimensions: &shape.dimensions,
                vertices: shape
                    .mesh_vertices
                    .iter()
                    .map(|vertex| vertex.position)
                    .collect(),
                triangles: &shape.mesh_triangles,
                position: shape.position,
                orientation: shape.orientation,
            });
            let (placement, shapes) =
                parse_placed(&object.id, object.position, object.orientation, shapes)?;
            Ok(SceneObject {
                id: object.id.clone(),
                placement,
                shapes,
            })
        })
        .collect::<Result<Vec<_>, SceneError>>()?;
    let attached = data
        .attached_objects
        .iter()
        .map(|attached| {
            let shapes = attached.shapes.iter().map(|shape| ShapeFields {
                kind: &shape.kind,
                dimensions: &shape.dimensions,
                vertices: shape
                    .mesh_vertices
                    .iter()
                    .map(|vertex| vertex.position)
                    .collect(),
                triangles: &shape.mesh_triangles,
                position: shape.position,
                orientation: shape.orientation,
            });
            let (placement, shapes) = parse_placed(
                &attached.id,
                attached.position,
                attached.orientation,
                shapes,
            )?;
            Ok(AttachedObject {
                id: attached.id.clone(),
                link: attached.frame.clone(),
                placement,
                shapes,
            })
        })
        .collect::<Result<Vec<_>, SceneError>>()?;
    Ok(PlanningScene {
        objects,
        attached,
        allowed_pairs: data
            .allowed_pairs
            .iter()
            .map(|pair| (pair.name_a.clone(), pair.name_b.clone()))
            .collect(),
        clearance_m: data.clearance_m,
    })
}

/// One shape as motion_planning carries it, free or attached: its kind, its dimensions,
/// its mesh, and its pose in its object's frame.
struct ShapeFields<'a> {
    kind: &'a str,
    dimensions: &'a [f64],
    vertices: Vec<[f64; 3]>,
    triangles: &'a [u32],
    position: [f64; 3],
    orientation: [f64; 4],
}

/// The placement and the shapes of the object `id`.
fn parse_placed<'a>(
    id: &str,
    position: [f64; 3],
    orientation: [f64; 4],
    shapes: impl Iterator<Item = ShapeFields<'a>>,
) -> Result<(Placement, Vec<Shape>), SceneError> {
    let failed = |reason: String| SceneError::Object {
        id: id.to_string(),
        reason,
    };
    let placement = Placement::parse(position, orientation).map_err(|e| failed(e.to_string()))?;
    let shapes = shapes
        .map(|shape| parse_shape(&shape))
        .collect::<Result<Vec<_>, _>>()
        .map_err(failed)?;
    Ok((placement, shapes))
}

fn parse_shape(shape: &ShapeFields) -> Result<Shape, String> {
    let placement =
        Placement::parse(shape.position, shape.orientation).map_err(|e| e.to_string())?;
    let kind = match (shape.kind, shape.dimensions) {
        ("box", &[x, y, z]) if lengths_are_valid(&[x, y, z]) => ShapeKind::Box { size: [x, y, z] },
        ("sphere", &[radius]) if lengths_are_valid(&[radius]) => ShapeKind::Sphere { radius },
        ("cylinder", &[height, radius]) if lengths_are_valid(&[height, radius]) => {
            ShapeKind::Cylinder { height, radius }
        }
        ("mesh", _) => parse_mesh(&shape.vertices, shape.triangles)?,
        ("box", _) => return Err(lengths_error("box", 3, shape.dimensions)),
        ("sphere", _) => return Err(lengths_error("sphere", 1, shape.dimensions)),
        ("cylinder", _) => return Err(lengths_error("cylinder", 2, shape.dimensions)),
        (kind, _) => {
            return Err(format!(
                "kind {kind:?} is none of box, sphere, cylinder and mesh"
            ));
        }
    };
    Ok(Shape { kind, placement })
}

fn lengths_error(kind: &str, count: usize, dimensions: &[f64]) -> String {
    format!("a {kind} takes {count} lengths above 0, got {dimensions:?}")
}

fn lengths_are_valid(lengths: &[f64]) -> bool {
    lengths
        .iter()
        .all(|length| length.is_finite() && *length > 0.0)
}

fn parse_mesh(vertices: &[[f64; 3]], triangles: &[u32]) -> Result<ShapeKind, String> {
    if vertices.iter().flatten().any(|c| !c.is_finite()) {
        return Err("a mesh takes three finite coordinates a vertex".to_string());
    }
    let (triangles, rest) = triangles.as_chunks::<3>();
    if !rest.is_empty()
        || triangles
            .iter()
            .flatten()
            .any(|&i| i as usize >= vertices.len())
    {
        return Err(format!(
            "a mesh takes three indices below its {} vertices a triangle",
            vertices.len()
        ));
    }
    Ok(ShapeKind::Mesh {
        vertices: vertices.to_vec(),
        triangles: triangles.to_vec(),
    })
}

/// Asks the planner for its arms until it answers, then reads the scene every
/// `poll_interval` and draws it at each read, until the node stops: a page that opens
/// late shows the scene as it is within one read. A read in another frame than the robot
/// frame is dropped, the first of a run logged. The changes from the scene before are
/// logged only when there are some.
pub async fn follow_scene(
    runner: Arc<NodeRunner>,
    canvas: Arc<dyn Canvas>,
    poll_interval: Duration,
    shared: Arc<Shared>,
) {
    let Some(planner) = get_planning_scene::bound_producer(&runner) else {
        return;
    };
    let token = runner.cancellation_token().clone();
    let asker = runner.clone();
    let asked = asking::ask("describe_arms", stopped(&runner), async move || {
        ask_arms(&asker).await
    })
    .await;
    let Some(described) = asked else {
        return;
    };
    shared.arms.send_replace(Some(described));
    let mut ticks = tokio::time::interval(poll_interval);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut last: Option<PlanningScene> = None;
    let mut drawn = BTreeSet::new();
    let failures = LatchedLog::warn("reading the planning scene");
    loop {
        tokio::select! {
            biased;
            () = token.cancelled() => return,
            _ = ticks.tick() => {}
        }
        let read = tokio::select! {
            biased;
            () = token.cancelled() => return,
            read = sendable(read_scene(&runner, planner, &shared)) => read,
        };
        let scene = match read {
            Ok(scene) => scene,
            Err(reason) => {
                failures.failure(reason);
                continue;
            }
        };
        // A scene is drawn at its time on the node's clock; until that clock is ready,
        // the scene waits for the next read.
        let at = match node_now() {
            Ok(at) => at,
            Err(reason) => {
                failures.failure(reason);
                continue;
            }
        };
        failures.success();
        let before = last
            .take()
            .unwrap_or_else(|| PlanningScene::empty_like(&scene));
        // An unchanged scene gives no changes.
        let changes = scene_changes(&before, &scene);
        let description = shared.arms.borrow().clone().unwrap_or_default();
        drawn = draw_scene(canvas.as_ref(), at, &scene, &changes, &description, &drawn);
        last = Some(scene);
    }
}

/// One read of the planning scene, held to the robot frame.
async fn read_scene(
    runner: &Arc<NodeRunner>,
    planner: &peppygen::ProducerRef,
    shared: &Arc<Shared>,
) -> Result<PlanningScene, String> {
    let response = sendable(get_planning_scene::poll(runner, planner, SERVICE_TIMEOUT))
        .await
        .map_err(|e| format!("get_planning_scene: {e}"))?;
    shared.hold_to_robot_frame(runner, "get_planning_scene", &response.data.frame)?;
    parse_scene(&response.data).map_err(|e| e.to_string())
}

/// One ask of the arms the planner describes.
async fn ask_arms(runner: &NodeRunner) -> Result<ArmsDescription, String> {
    let planner =
        describe_arms::bound_producer(runner).ok_or("the motion_planning link is vacant")?;
    let described = sendable(describe_arms::poll(runner, planner, SERVICE_TIMEOUT))
        .await
        .map_err(|e| format!("describe_arms: {e}"))?;
    Ok(parse_arms_description(&described.data))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use get_planning_scene::{
        ResponseData, ResponseDataAllowedPairsItem, ResponseDataAttachedObjectsItem,
        ResponseDataAttachedObjectsItemShapesItem, ResponseDataObjectsItem,
        ResponseDataObjectsItemShapesItem,
    };
    use std::time::SystemTime;

    const IDENTITY: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

    fn fields<'a>(
        kind: &'a str,
        dimensions: &'a [f64],
        vertices: &[[f64; 3]],
        triangles: &'a [u32],
    ) -> ShapeFields<'a> {
        ShapeFields {
            kind,
            dimensions,
            vertices: vertices.to_vec(),
            triangles,
            position: [0.0; 3],
            orientation: IDENTITY,
        }
    }

    pub(crate) fn sphere_shape(radius: f64) -> ResponseDataObjectsItemShapesItem {
        ResponseDataObjectsItemShapesItem {
            kind: "sphere".to_string(),
            dimensions: vec![radius],
            mesh_vertices: vec![],
            mesh_triangles: vec![],
            position: [0.0; 3],
            orientation: IDENTITY,
        }
    }

    pub(crate) fn lime(position: [f64; 3]) -> ResponseDataObjectsItem {
        ResponseDataObjectsItem {
            id: "brain/lime_1".to_string(),
            position,
            orientation: IDENTITY,
            shapes: vec![sphere_shape(0.03)],
        }
    }

    pub(crate) fn scene(objects: Vec<ResponseDataObjectsItem>) -> ResponseData {
        ResponseData::new(
            "base_link".to_string(),
            SystemTime::UNIX_EPOCH,
            objects,
            vec![],
            vec![],
            0.04,
            0.02,
        )
    }

    #[test]
    fn a_scene_of_each_part_parses() {
        let mut data = scene(vec![lime([0.4, -0.2, 0.1])]);
        data.attached_objects = vec![ResponseDataAttachedObjectsItem {
            id: "brain/apple_1".to_string(),
            frame: "openarm_right_tcp".to_string(),
            touch_links: vec![],
            position: [0.0, 0.0, 0.05],
            orientation: IDENTITY,
            shapes: vec![ResponseDataAttachedObjectsItemShapesItem {
                kind: "box".to_string(),
                dimensions: vec![0.1, 0.2, 0.3],
                mesh_vertices: vec![],
                mesh_triangles: vec![],
                position: [0.0; 3],
                orientation: IDENTITY,
            }],
        }];
        data.allowed_pairs = vec![ResponseDataAllowedPairsItem {
            name_a: "brain/apple_1".to_string(),
            name_b: "openarm_right_finger".to_string(),
        }];
        let parsed = parse_scene(&data).expect("a valid scene");
        assert_eq!(parsed.objects[0].id, "brain/lime_1");
        assert_eq!(
            parsed.objects[0].shapes[0].kind,
            ShapeKind::Sphere { radius: 0.03 }
        );
        assert_eq!(parsed.attached[0].link, "openarm_right_tcp");
        assert_eq!(
            parsed.attached[0].shapes[0].kind,
            ShapeKind::Box {
                size: [0.1, 0.2, 0.3]
            }
        );
        assert_eq!(
            parsed.allowed_pairs,
            [(
                "brain/apple_1".to_string(),
                "openarm_right_finger".to_string()
            )]
        );
        assert_eq!(parsed.clearance_m, 0.04);
    }

    #[test]
    fn each_kind_takes_its_own_dimensions() {
        let parse = |kind: &str, dimensions: &[f64]| {
            parse_shape(&fields(kind, dimensions, &[], &[])).map(|shape| shape.kind)
        };
        assert_eq!(
            parse("cylinder", &[0.2, 0.05]),
            Ok(ShapeKind::Cylinder {
                height: 0.2,
                radius: 0.05
            })
        );
        assert!(parse("box", &[0.1, 0.2]).is_err());
        assert!(parse("box", &[0.1, 0.0, 0.3]).is_err());
        assert!(parse("sphere", &[]).is_err());
        assert!(parse("sphere", &[f64::NAN]).is_err());
        assert!(parse("cylinder", &[0.2]).is_err());
        assert!(parse("cone", &[0.2]).is_err());
    }

    #[test]
    fn a_mesh_takes_whole_vertices_and_triangles_within_them() {
        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let parsed =
            parse_shape(&fields("mesh", &[], &vertices, &[0, 1, 2])).expect("a valid mesh");
        assert_eq!(
            parsed.kind,
            ShapeKind::Mesh {
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                triangles: vec![[0, 1, 2]],
            }
        );
        let unplaced = [[0.0, 0.0, 0.0], [1.0, f64::NAN, 0.0], [0.0, 1.0, 0.0]];
        assert!(parse_shape(&fields("mesh", &[], &unplaced, &[0, 1, 2])).is_err());
        assert!(parse_shape(&fields("mesh", &[], &vertices, &[0, 1, 3])).is_err());
        assert!(parse_shape(&fields("mesh", &[], &vertices, &[0, 1])).is_err());
    }

    #[test]
    fn a_scene_with_an_unreadable_object_or_clearance_is_refused() {
        let mut data = scene(vec![lime([0.4, -0.2, 0.1])]);
        data.objects[0].orientation = [0.0, 0.0, 0.0, 2.0];
        assert!(matches!(
            parse_scene(&data),
            Err(SceneError::Object { id, .. }) if id == "brain/lime_1"
        ));
        let mut data = scene(vec![]);
        data.clearance_m = -0.1;
        assert_eq!(parse_scene(&data), Err(SceneError::Clearance(-0.1)));
    }

    #[test]
    fn the_description_keeps_each_arms_joints_and_grasp_link() {
        let data = describe_arms::ResponseData::new(
            vec![describe_arms::ResponseDataArmsItem {
                name: "right_arm".to_string(),
                gripper_name: "right_gripper".to_string(),
                joint_names: vec!["j1".to_string(), "j2".to_string()],
                joint_lower_limits: vec![-1.0, -1.0],
                joint_upper_limits: vec![1.0, 1.0],
                link_names: vec![],
                grasp_link: "openarm_right_tcp".to_string(),
                approach_axis: "+z".to_string(),
                closing_axis: "+y".to_string(),
                gripper_links: vec![],
            }],
            vec![],
        );
        assert_eq!(
            parse_arms_description(&data),
            ArmsDescription {
                arms: vec![ArmDescription {
                    name: "right_arm".to_string(),
                    joint_names: vec!["j1".to_string(), "j2".to_string()],
                    grasp_link: "openarm_right_tcp".to_string(),
                }]
            }
        );
    }
}
