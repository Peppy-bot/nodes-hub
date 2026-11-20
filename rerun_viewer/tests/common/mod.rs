//! What the node's tests share: the node over the generated harness, drawing on a fake
//! canvas, and the messages and answers the tests send it.
//!
//! Each integration test is its own crate, so items unused by one suite are expected.
#![allow(dead_code)]

use std::future::Future;
use std::sync::{Arc, LazyLock, Once};
use std::time::{Duration, SystemTime};

use control_core::fault::Fault;
use peppygen::Parameters;
use peppygen::consumed_services::identity::get_description;
use peppygen::consumed_services::motion_planning::{describe_arms, get_planning_scene};
use peppygen::consumed_topics::limb_state::limb_states;
use peppygen::fixtures::harness::{Config, Harness, Mocks};
use peppygen::mock::deps::sensor_mounts::describe_sensors;
use rerun_viewer::config::default_parameters;
use rerun_viewer::testing::{FakeCanvas, MINI_SRDF, MINI_URDF, mini_meshes};
use tokio::sync::watch;

/// Bounds every wait on the node. Generous: the tests are about what is drawn, never
/// about how fast.
pub const TIMEOUT: Duration = Duration::from_secs(60);

pub const IDENTITY: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

/// The harness's configuration with every optional link vacant.
pub fn every_optional_link_vacant() -> Config {
    Config {
        parameters: Some(default_parameters()),
        identity_vacant: true,
        camera_vacant: true,
        geometry_vacant: true,
        sensor_mounts_vacant: true,
        motion_planning_vacant: true,
        ..Config::default()
    }
}

/// The harness's configuration with the links `bound` bound, the others vacant, and the
/// node's parameters `parameters`. A bound transforms link has one publisher.
pub fn bound(links: &[&str], parameters: Parameters) -> Config {
    Config {
        parameters: Some(parameters),
        identity_vacant: !links.contains(&"identity"),
        camera_vacant: !links.contains(&"camera"),
        geometry_vacant: !links.contains(&"geometry"),
        sensor_mounts_vacant: !links.contains(&"sensor_mounts"),
        transforms_instances: usize::from(links.contains(&"transforms")),
        motion_planning_vacant: !links.contains(&"motion_planning"),
        ..Config::default()
    }
}

/// The node over the harness as `config` binds it, drawing on the canvas it gives. Every
/// line the test binary logs from then on is captured for [`logged`].
pub async fn start(config: Config) -> Result<(Harness, Mocks, Arc<FakeCanvas>), String> {
    capture_logs();
    let canvas = Arc::new(FakeCanvas::default());
    let drawn_on = canvas.clone();
    let (harness, mocks) = Harness::start_with(config, move |params, runner| async move {
        let config = rerun_viewer::config::Config::parse(&params)
            .map_err(|e| peppygen::Error::Node(e.into()))?;
        rerun_viewer::node::setup(config, runner, drawn_on, Fault::default()).await
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

/// The robot frame of the tests' robots.
pub const ROBOT_FRAME: &str = "base_link";

/// The odometry frame of a robot on a mobile base: a frame other than the robot frame.
pub const ODOM_FRAME: &str = "odom";

/// The frame of the robot's one camera, the chest camera.
pub const CAMERA_FRAME: &str = "chest_color_optical_frame";

pub fn sensor_mounts(mocks: &Mocks) -> &peppygen::mock::deps::sensor_mounts::Mock {
    mocks
        .deps
        .sensor_mounts
        .as_ref()
        .expect("sensor_mounts is linked")
}

/// The robot frame, and the robot's one sensor, the chest camera.
pub fn chest_sensor() -> describe_sensors::ResponseData {
    describe_sensors::ResponseData::new(
        ROBOT_FRAME.to_string(),
        vec!["chest".to_string()],
        vec!["rgbd_camera".to_string()],
        vec![CAMERA_FRAME.to_string()],
        vec![String::new()],
    )
}

/// How long [`publish_arms_until_drawn`] waits for a draw before it publishes again.
const PUBLISH_AGAIN_AFTER: Duration = Duration::from_millis(50);

/// Answers the node's ask of the robot frame with [`chest_sensor`]'s, and returns once the
/// node holds it. The viewer drops each value that needs the robot frame until sensor_mounts
/// answers its ask, which the first such value starts: a snapshot of [`two_arms`] starts it,
/// and one published until the viewer draws it shows the answer taken.
pub async fn hold_the_robot_frame(mocks: &Mocks, canvas: &FakeCanvas) -> Result<(), String> {
    let limb_states = &mocks.deps.limb_state.limb_states;
    publish_once_subscribed(
        limb_states.wait_for_subscriber(TIMEOUT),
        limb_states.publish(&two_arms()),
    )
    .await?;
    sensor_mounts(mocks)
        .describe_sensors
        .next_request(TIMEOUT)
        .await
        .map_err(|e| e.to_string())?
        .respond(chest_sensor())
        .await
        .map_err(|e| e.to_string())?;
    publish_arms_until_drawn(mocks, canvas, &two_arms()).await
}

/// Publishes `snapshot` again until the viewer draws one more grasp point than it had drawn:
/// a snapshot the viewer drops leaves no trace to wait for.
pub async fn publish_arms_until_drawn(
    mocks: &Mocks,
    canvas: &FakeCanvas,
    snapshot: &limb_states::Message,
) -> Result<(), String> {
    let grasps = |drawn: &[rerun_viewer::drawn::Drawn]| {
        drawn
            .iter()
            .filter(|call| call.path() == Some(GRASP_PATH))
            .count()
    };
    let before = grasps(&canvas.drawn());
    let limb_states = &mocks.deps.limb_state.limb_states;
    let deadline = tokio::time::Instant::now() + TIMEOUT;
    while tokio::time::Instant::now() < deadline {
        limb_states
            .publish(snapshot)
            .await
            .map_err(|e| e.to_string())?;
        let drawn = canvas
            .wait_until(PUBLISH_AGAIN_AFTER, "a grasp point", |drawn| {
                grasps(drawn) > before
            })
            .await;
        if drawn.is_ok() {
            return Ok(());
        }
    }
    Err(format!("no snapshot was drawn within {TIMEOUT:?}"))
}

/// Where the viewer draws the grasp point of the right arm of [`two_arms`].
pub const GRASP_PATH: &str = "world/arms/right_arm/grasp";

/// A grasp point at `position`, turned as the robot frame.
pub fn grasp_point(position: [f64; 3]) -> limb_states::MessageGraspPointsItem {
    limb_states::MessageGraspPointsItem {
        position,
        orientation: IDENTITY,
    }
}

/// A snapshot of the two arms of an OpenArm, 2 and 1 joints for brevity, and its right
/// gripper.
pub fn two_arms() -> limb_states::Message {
    limb_states::Message {
        timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(10),
        frame: ROBOT_FRAME.to_string(),
        arm_names: vec!["left_arm".to_string(), "right_arm".to_string()],
        joints_per_arm: vec![2, 1],
        joint_positions: vec![0.1, 0.2, 0.3],
        grasp_points: vec![grasp_point([0.3, 0.2, 0.4]), grasp_point([0.3, -0.2, 0.4])],
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
    get_planning_scene::ResponseData::new(
        ROBOT_FRAME.to_string(),
        SystemTime::UNIX_EPOCH + Duration::from_secs(10),
        objects,
        attached,
        vec![],
        0.04,
        0.02,
    )
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
        frame: "openarm_right_tcp".to_string(),
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
pub fn drew_at(drawn: &[rerun_viewer::drawn::Drawn], path: &str) -> bool {
    drawn.iter().any(|call| call.path() == Some(path))
}

/// The description of the test robot `mini` (`rerun_viewer::testing`), as robot_identity
/// gives it.
pub fn mini_description() -> get_description::ResponseData {
    get_description::ResponseData::new(
        MINI_URDF.to_string(),
        MINI_SRDF.to_string(),
        mini_meshes()
            .into_iter()
            .map(|(name, data)| get_description::ResponseDataMeshesItem { name, data })
            .collect(),
        "mini".to_string(),
    )
}

/// A snapshot of `mini`'s arm at `lift` and `reach` and its gripper at `opening`.
pub fn mini_snapshot(lift: f64, reach: f64, opening: f64) -> limb_states::Message {
    limb_states::Message {
        timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(10),
        frame: ROBOT_FRAME.to_string(),
        arm_names: vec!["arm".to_string()],
        joints_per_arm: vec![2],
        joint_positions: vec![lift, reach],
        grasp_points: vec![grasp_point([0.0, 0.0, 0.3])],
        gripper_names: vec!["gripper".to_string()],
        gripper_openings: vec![opening],
    }
}

/// Every line logged in this test binary, as an operator reads it.
static LOGGED: LazyLock<watch::Sender<Vec<String>>> =
    LazyLock::new(|| watch::Sender::new(Vec::new()));

/// One logged line on its way to [`LOGGED`]: the formatter writes an event whole, then
/// drops the line.
#[derive(Default)]
struct Line(Vec<u8>);

impl std::io::Write for Line {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for Line {
    fn drop(&mut self) {
        let line = String::from_utf8_lossy(&self.0).trim_end().to_owned();
        LOGGED.send_modify(|lines| lines.push(line));
    }
}

/// Captures every line logged in this test binary from now on. Each test starts its node
/// through [`start`], which calls this first, so no node logs before the capture holds.
fn capture_logs() {
    static CAPTURING: Once = Once::new();
    CAPTURING.call_once(|| {
        tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(Line::default)
            .init();
    });
}

/// Waits until a line that holds every one of `texts` is logged, failing after
/// [`TIMEOUT`].
pub async fn logged(texts: &[&str]) -> Result<(), String> {
    let mut lines = LOGGED.subscribe();
    tokio::time::timeout(
        TIMEOUT,
        lines.wait_for(|lines| {
            lines
                .iter()
                .any(|line| texts.iter().all(|text| line.contains(text)))
        }),
    )
    .await
    .map_err(|_| format!("no line holding {texts:?} was logged within {TIMEOUT:?}"))?
    .map(|_| ())
    .map_err(|_| "the log stays open for as long as the test binary runs".to_string())
}
