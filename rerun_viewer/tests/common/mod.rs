//! What the node's tests share: the node over the generated harness, drawing on a fake
//! canvas, and the messages and answers the tests send it.
//!
//! Each integration test is its own crate, so items unused by one suite are expected.
#![allow(dead_code)]

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use peppygen::Parameters;
use peppygen::consumed_services::motion_planning::{describe_arms, get_planning_scene};
use peppygen::consumed_topics::limb_state::limb_states;
use peppygen::fixtures::harness::{Config, Harness, Mocks};
use rerun_viewer::config::default_parameters;
use rerun_viewer::testing::FakeCanvas;

/// Bounds every wait on the node. Generous: the tests are about what is drawn, never
/// about how fast.
pub const TIMEOUT: Duration = Duration::from_secs(60);

pub const IDENTITY: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

/// The harness's configuration with every optional link vacant.
pub fn every_optional_link_vacant() -> Config {
    Config {
        parameters: Some(default_parameters()),
        camera_vacant: true,
        geometry_vacant: true,
        camera_mounts_vacant: true,
        motion_planning_vacant: true,
        ..Config::default()
    }
}

/// The harness's configuration with the links `bound` bound, the others vacant, and the
/// node's parameters `parameters`.
pub fn bound(links: &[&str], parameters: Parameters) -> Config {
    Config {
        parameters: Some(parameters),
        camera_vacant: !links.contains(&"camera"),
        geometry_vacant: !links.contains(&"geometry"),
        camera_mounts_vacant: !links.contains(&"camera_mounts"),
        motion_planning_vacant: !links.contains(&"motion_planning"),
        ..Config::default()
    }
}

/// The node over the harness as `config` binds it, drawing on the canvas it gives.
pub async fn start(config: Config) -> Result<(Harness, Mocks, Arc<FakeCanvas>), String> {
    let canvas = Arc::new(FakeCanvas::default());
    let drawn_on = canvas.clone();
    let (harness, mocks) = Harness::start_with(config, move |params, runner| async move {
        let config = rerun_viewer::config::Config::parse(&params)
            .map_err(|e| peppygen::Error::Node(e.into()))?;
        rerun_viewer::node::setup(config, runner, drawn_on).await
    })
    .await
    .map_err(|e| e.to_string())?;
    Ok((harness, mocks, canvas))
}

/// Runs `publish` once `subscribed`, the wait for the node's subscription to the topic,
/// has seen it: a message published before the node subscribes never reaches it.
pub async fn publish_once_subscribed(
    subscribed: impl Future<Output = peppygen::Result<bool>>,
    publish: impl Future<Output = peppygen::Result<()>>,
) -> Result<(), String> {
    if !subscribed.await.map_err(|e| e.to_string())? {
        return Err(format!("the node did not subscribe within {TIMEOUT:?}"));
    }
    publish.await.map_err(|e| e.to_string())
}

/// A snapshot of the two arms of an OpenArm, 2 and 1 joints for brevity, and its right
/// gripper.
pub fn two_arms() -> limb_states::Message {
    limb_states::Message {
        timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(10),
        arm_names: vec!["left_arm".to_string(), "right_arm".to_string()],
        joints_per_arm: vec![2, 1],
        joint_positions: vec![0.1, 0.2, 0.3],
        positions: vec![0.3, 0.2, 0.4, 0.3, -0.2, 0.4],
        orientations: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        gripper_names: vec!["right_gripper".to_string()],
        gripper_openings: vec![0.5],
    }
}

/// The planner's description of the right arm: one joint and its grasp link.
pub fn right_arm() -> describe_arms::ResponseData {
    describe_arms::ResponseData::new(
        vec![describe_arms::ResponseDataArmsItem {
            name: "right_arm".to_string(),
            gripper_name: "right_gripper".to_string(),
            joint_names: vec!["openarm_right_joint1".to_string()],
            joint_lower_limits: vec![-1.0],
            joint_upper_limits: vec![1.0],
            link_names: vec!["openarm_right_tcp".to_string()],
            grasp_link: "openarm_right_tcp".to_string(),
            approach_axis: "+z".to_string(),
            closing_axis: "+y".to_string(),
            gripper_links: vec![],
        }],
        vec![],
    )
}

/// A scene of the objects `objects` and the attached objects `attached`, a clearance of
/// 0.04 m and the planner's cubes of 0.02 m.
pub fn scene(
    objects: Vec<get_planning_scene::ResponseDataObjectsItem>,
    attached: Vec<get_planning_scene::ResponseDataAttachedObjectsItem>,
) -> get_planning_scene::ResponseData {
    get_planning_scene::ResponseData::new(objects, attached, vec![], 0.04, 0.02)
}

/// The brain's lime, a sphere of 3 cm, free at `position`.
pub fn free_lime(position: [f64; 3]) -> get_planning_scene::ResponseDataObjectsItem {
    get_planning_scene::ResponseDataObjectsItem {
        id: "brain/lime_1".to_string(),
        position,
        orientation: IDENTITY,
        shapes: vec![get_planning_scene::ResponseDataObjectsItemShapesItem {
            kind: "sphere".to_string(),
            dimensions: vec![0.03],
            mesh_vertices: vec![],
            mesh_triangles: vec![],
            position: [0.0; 3],
            orientation: IDENTITY,
        }],
    }
}

/// The brain's lime, carried 5 cm along the grasp link's z.
pub fn held_lime() -> get_planning_scene::ResponseDataAttachedObjectsItem {
    get_planning_scene::ResponseDataAttachedObjectsItem {
        id: "brain/lime_1".to_string(),
        link_name: "openarm_right_tcp".to_string(),
        touch_links: vec![],
        position: [0.0, 0.0, 0.05],
        orientation: IDENTITY,
        shapes: vec![
            get_planning_scene::ResponseDataAttachedObjectsItemShapesItem {
                kind: "sphere".to_string(),
                dimensions: vec![0.03],
                mesh_vertices: vec![],
                mesh_triangles: vec![],
                position: [0.0; 3],
                orientation: IDENTITY,
            },
        ],
    }
}

/// True when the canvas drew anything at `path`.
pub fn drew_at(drawn: &[rerun_viewer::testing::Drawn], path: &str) -> bool {
    drawn.iter().any(|call| call.path() == Some(path))
}
