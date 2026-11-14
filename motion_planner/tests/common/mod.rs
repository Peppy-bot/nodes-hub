//! What the node's tests share: the parameters the launchers give the planner, the planning
//! time each plan asks for and what a refused start says to do, OpenArm v2's model, a depth
//! camera and the frames it renders of the robot and of boxes, a table across the left arm's
//! motion and the node with the arm under it once the camera has seen it, the node over the
//! generated harness with that camera linked or vacant, and the plans and checks of the left
//! arm.
//!
//! Each integration test is its own crate, so items unused by one suite are expected;
//! the allow keeps clippy's -D warnings gate meaningful for the rest.
#![allow(dead_code)]

mod parameters;

use std::sync::LazyLock;
use std::time::{Duration, SystemTime};

use chain_kinematics::nalgebra::{
    Isometry3, Matrix3, Point3, Rotation3, Translation3, UnitQuaternion, Vector3,
};
use collision_geometry::chain::{BodyMask, Padding, PlacedMask, RiderBody};
use control_core::fault::Fault;
use motion_planner::history::ArmJoints;
use motion_planner::request::{Named, Planner, Refiner};
use motion_planner::robot::{RobotModel, model};
use peppygen::Parameters;
use peppygen::consumed_services::identity::get_description::ResponseDataMeshesItem;
use peppygen::consumed_topics::camera::depth_stream::{Message as DepthFrame, MessageHeader};
use peppygen::exposed_actions::motion_planning::plan_joint_path::{
    GoalArmsItem, GoalGoalJointConstraintsItem,
};
use peppygen::fixtures::exposed_actions::motion_planning::plan_joint_path::{
    Failure, GoalRequestData, ResultData, ResultOutcome, send_goal,
};
use peppygen::fixtures::exposed_services::motion_planning::check_joint_path;
use peppygen::fixtures::harness::{Config, Harness, Mocks};
use peppygen::mock::deps::identity::get_description;
use peppygen::mock::deps::{camera, camera_mounts, geometry, identity, limb_state, workspace};
use robot_description::{ArmId, PerArm};

/// Bounds every wait on the node, past the longest a plan takes. Generous: the
/// assertions are about the answers, never about how fast they come.
pub const TIMEOUT: Duration = Duration::from_secs(60);

/// The rendered image's pixels across and down, and its focal length in pixels.
pub const WIDTH: u32 = 48;
pub const HEIGHT: u32 = 36;
pub const FOCAL_PX: f64 = 30.0;
/// Meters per count of a rendered depth frame.
pub const DEPTH_UNIT: f32 = 0.001;
/// The depths a ray is sampled over, and the step between samples (m).
const NEAREST_M: f64 = 0.2;
const FARTHEST_M: f64 = 1.8;
const STEP_M: f64 = 0.004;

/// The name the robot gives its perception camera.
pub const CAMERA: &str = "chest";

pub use parameters::parameters;

/// The planning time a test's plan asks for (s), unless the test names its own.
pub const PLANNING_TIME_S: f64 = 20.0;

/// What a refused start in contact within the robot's own body says to do.
pub const OWN_BODY_WAY_OUT: &str =
    "; move the arms to the home posture with move_to_home and plan from there";
/// What a refused start in contact with, or inside the clearance of, the map or an object
/// says to do.
pub const CLEAR_OR_ALLOW_WAY_OUT: &str =
    "; move the arms clear first, or allow the pair with set_allowed_collision";

/// OpenArm v2's model, its grippers open, as the node fits it.
pub static OPENARM_V2: LazyLock<RobotModel<ARM_DOF, DUAL_DOF, ARMS>> =
    LazyLock::new(|| bundled_model("openarm_v2"));

/// The joints of each arm of OpenArm v2, the robot most suites run, and of both.
pub const ARM_DOF: usize = 7;
pub const ARMS: usize = 2;
pub const DUAL_DOF: usize = ARMS * ARM_DOF;
/// OpenArm v2's arms, in the order of its description: the left, then the right.
pub const LEFT: ArmId = ArmId::new(0);
pub const RIGHT: ArmId = ArmId::new(1);
pub const BOTH: [ArmId; ARMS] = [LEFT, RIGHT];

/// The planner's model of the description robot_description bundles as `id`, its meshes
/// written to a directory for the build.
pub fn bundled_model<const N: usize, const D: usize, const K: usize>(
    id: &str,
) -> RobotModel<N, D, K> {
    let bundled = robot_description::model(id).expect("robot_description bundles it");
    let meshes = tempfile::tempdir().expect("a temporary directory");
    bundled
        .write_meshes_to(meshes.path())
        .expect("the meshes write");
    model(
        &bundled.description(),
        meshes.path().to_str().expect("the directory is UTF-8"),
    )
    .unwrap_or_else(|error| panic!("{id}: {error}"))
}

/// The posture `name` of OpenArm v2's arm `arm`, as its description gives it.
pub fn posture(arm: ArmId, name: &str) -> [f64; ARM_DOF] {
    posture_of("openarm_v2", arm, name)
}

/// The posture `name` of the arm `arm` of the seven-joint model `id`, as its description
/// gives it.
pub fn posture_of(id: &str, arm: ArmId, name: &str) -> [f64; ARM_DOF] {
    let description = robot_description::model(id)
        .expect("robot_description bundles the model")
        .description();
    let group = &description.arms()[arm.place()].name;
    description
        .state(group, name)
        .unwrap_or_else(|| panic!("{group} has the posture {name}"))
        .try_into()
        .expect("a posture of a seven-joint arm")
}

/// A camera at `eye` whose optical axis looks at `target`, image right across the
/// optical axis and the robot's +z, so image down points down where it can.
pub fn looking_at(eye: Vector3<f64>, target: Vector3<f64>) -> Isometry3<f64> {
    let forward = (target - eye).normalize();
    let across = if forward.cross(&Vector3::z()).norm() > 1e-6 {
        forward.cross(&Vector3::z()).normalize()
    } else {
        Vector3::y()
    };
    let down = forward.cross(&across);
    let axes = Matrix3::from_columns(&[across, down, forward]);
    let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(axes));
    Isometry3::from_parts(Translation3::from(eye), rotation)
}

/// A camera 0.8 m in front of the robot at shoulder height, facing it.
pub fn facing_the_robot() -> Isometry3<f64> {
    looking_at(Vector3::new(0.8, 0.0, 0.6), Vector3::new(0.0, 0.0, 0.6))
}

/// A box of the robot frame: its lowest and its highest corner.
pub type Solid = ([f64; 3], [f64; 3]);

/// What a depth camera at `camera` sees, pixel by pixel in rows: along each pixel's ray,
/// the point in its optical frame of the first sample inside the robot, its arms at
/// `arms` with their fingers open and its body, when given, or inside one of `boxes`. The
/// renderer places the robot from its pieces itself.
pub fn render(
    camera: &Isometry3<f64>,
    arms: Option<&ArmJoints<ARM_DOF, ARMS>>,
    boxes: &[Solid],
) -> Vec<Option<[f64; 3]>> {
    let none = Padding::new(1.0, 0.0).expect("the padding that grows nothing");
    let chains: Vec<_> = OPENARM_V2
        .arms
        .iter()
        .map(|arm| (arm.id, arm.geometry.mask(none)))
        .collect();
    let body = OPENARM_V2.body.mask(none);
    let fingers = arms.map(|arms| {
        BodyMask::of(
            BOTH.into_iter().flat_map(|side| {
                let arm = &OPENARM_V2.arms[side].geometry;
                let posed = arm.chain().at(&arms[side]);
                OPENARM_V2
                    .own
                    .fingers(side, arm)
                    .expect("the arm's fingers")
                    .into_iter()
                    .map(move |finger| {
                        let hand = posed.link_pose_world(finger.segment);
                        let RiderBody::Part(part) = finger.body else {
                            panic!("a finger is a part of the robot");
                        };
                        (finger.name, part.pieces(), hand * finger.offset)
                    })
            }),
            none,
        )
    });
    let parts: Vec<PlacedMask> = match (arms, &fingers) {
        (Some(arms), Some(fingers)) => chains
            .iter()
            .map(|(side, mask)| mask.at(&arms[*side]))
            .chain([fingers.placed(), body.placed()])
            .collect(),
        _ => Vec::new(),
    };
    let inside = |point: &Point3<f64>| {
        parts.iter().any(|part| part.contains(point))
            || boxes
                .iter()
                .any(|(low, high)| (0..3).all(|i| low[i] <= point[i] && point[i] <= high[i]))
    };
    let samples = ((FARTHEST_M - NEAREST_M) / STEP_M) as usize;
    (0..HEIGHT)
        .flat_map(|v| (0..WIDTH).map(move |u| (u, v)))
        .map(|(u, v)| {
            let x = (f64::from(u) - f64::from(WIDTH) / 2.0) / FOCAL_PX;
            let y = (f64::from(v) - f64::from(HEIGHT) / 2.0) / FOCAL_PX;
            let ray = Vector3::new(x, y, 1.0).normalize();
            (0..=samples)
                .map(|k| ray * (NEAREST_M + STEP_M * k as f64))
                .find(|seen| inside(&(camera * Point3::from(*seen))))
                .map(|seen| [seen.x, seen.y, seen.z])
        })
        .collect()
}

/// The z16 depth frame of a render: each pixel's depth in counts of `DEPTH_UNIT`, 0
/// where it sees nothing.
pub fn depth_frame(render: &[Option<[f64; 3]>]) -> Vec<u8> {
    render
        .iter()
        .map(|p| p.map_or(0, |[_, _, z]| (z / f64::from(DEPTH_UNIT)).round() as u16))
        .flat_map(u16::to_le_bytes)
        .collect()
}

/// The instant `ms` milliseconds into the tests' time line.
pub fn at_ms(ms: u64) -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_000) + Duration::from_millis(ms)
}

/// Both arms where `left` and `right` put them.
pub fn arms(left: [f64; ARM_DOF], right: [f64; ARM_DOF]) -> ArmJoints<ARM_DOF, ARMS> {
    PerArm::new([left, right])
}

/// Each arm's Ready posture: elbows bent, hands forward.
pub fn left_ready() -> [f64; ARM_DOF] {
    posture(LEFT, robot_description::states::READY)
}
pub fn right_ready() -> [f64; ARM_DOF] {
    posture(RIGHT, robot_description::states::READY)
}

/// The left arm at Ready with its elbow at `angle`.
pub fn left_elbow(angle: f64) -> [f64; ARM_DOF] {
    let mut q = left_ready();
    q[3] = angle;
    q
}

/// The left arm with its hand under the table of `table_across`, the elbow open, and
/// over it, the elbow bent.
pub fn under_the_table() -> [f64; ARM_DOF] {
    left_elbow(0.6)
}
pub fn over_the_table() -> [f64; ARM_DOF] {
    left_elbow(2.0)
}

/// The configuration `t` of the way along the straight joint motion from `from` to
/// `to`.
pub fn along(from: &[f64; ARM_DOF], to: &[f64; ARM_DOF], t: f64) -> [f64; ARM_DOF] {
    std::array::from_fn(|i| from[i] + (to[i] - from[i]) * t)
}

/// A table 20 cm square and 5 cm thick across the straight joint motion of the left arm
/// from `under` to `over`, along which the grasp point rises at least 15 cm: its top
/// lies where the grasp point crosses two fifths of the way up, found by bisection. The
/// table, and the share of the motion at which the grasp point crosses its top.
pub fn table_across(under: &[f64; ARM_DOF], over: &[f64; ARM_DOF]) -> (Solid, f64) {
    let left = OPENARM_V2.arms[LEFT].geometry.chain();
    let hand = |q: &[f64; ARM_DOF]| left.world_ee_pose(q).translation.vector;
    let (low, high) = (hand(under).z, hand(over).z);
    assert!(high - low > 0.15, "the hand rises from {low} to {high}");
    let height = low + (high - low) * 0.4;
    let (mut crossing_t, mut past_t) = (0.0, 1.0);
    while past_t - crossing_t > 1e-6 {
        let t = (crossing_t + past_t) / 2.0;
        if hand(&along(under, over, t)).z < height {
            crossing_t = t;
        } else {
            past_t = t;
        }
    }
    let crossing = hand(&along(under, over, crossing_t));
    let table = (
        [crossing.x - 0.1, crossing.y - 0.1, height - 0.05],
        [crossing.x + 0.1, crossing.y + 0.1, height],
    );
    (table, crossing_t)
}

/// The node with the left arm under `table_across`'s table, which the camera has seen, and
/// the share of the motion from under the table to over it where the hand crosses its top.
pub async fn under_a_seen_table() -> peppygen::Result<(Harness, Mocks, f64)> {
    let (table, crossing_t) = table_across(&under_the_table(), &over_the_table());
    let ([x0, y0, _], [x1, y1, top]) = table;
    let (x, y) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let camera = looking_at(Vector3::new(x + 0.5, y, top + 0.7), Vector3::new(x, y, top));
    let (harness, mocks) = start_with_camera(&camera).await?;
    publish_arms(&mocks, at_ms(0), &arms(under_the_table(), right_ready())).await?;
    take_frame(
        &harness,
        &mocks,
        at_ms(0),
        &render(&camera, None, &[table]),
        under_the_table(),
    )
    .await?;
    Ok((harness, mocks, crossing_t))
}

/// The node over the harness with the four camera slots vacant, once the arms' joint
/// positions at `arms` have arrived: a check of the left arm standing still stops being
/// refused, and answers with no map stamp.
pub async fn start_without_camera(
    arms: &ArmJoints<ARM_DOF, ARMS>,
) -> peppygen::Result<(Harness, Mocks)> {
    start_without_camera_with(parameters(), arms).await
}

/// `start_without_camera` with the node's parameters `params`.
pub async fn start_without_camera_with(
    params: Parameters,
    arms: &ArmJoints<ARM_DOF, ARMS>,
) -> peppygen::Result<(Harness, Mocks)> {
    start_vacant(
        Config {
            parameters: Some(params),
            camera_mounts_vacant: true,
            workspace_vacant: true,
            ..vacant_camera()
        },
        arms,
    )
    .await
}

/// `start_without_camera` with only the camera and geometry slots vacant, as on a robot
/// without a camera rig whose backbone answers camera_mounts and workspace.
pub async fn start_without_the_rig(
    arms: &ArmJoints<ARM_DOF, ARMS>,
) -> peppygen::Result<(Harness, Mocks)> {
    start_vacant(vacant_camera(), arms).await
}

/// The harness's configuration with the camera and geometry slots vacant.
pub fn vacant_camera() -> Config {
    Config {
        parameters: Some(parameters()),
        camera_vacant: true,
        geometry_vacant: true,
        ..Config::default()
    }
}

/// The node over the harness as `config` binds it, its robot's identity answering that
/// it is an OpenArm v2.
pub async fn start(config: Config) -> peppygen::Result<(Harness, Mocks)> {
    let (harness, mocks, _) = start_identified_as(config, "openarm_v2").await?;
    Ok((harness, mocks))
}

/// The node over the harness as `config` binds it, its robot's identity answering the
/// node's first asks that it is `model`, described as robot_description bundles it, and
/// the fault that ends the node.
pub async fn start_identified_as(
    config: Config,
    model: &str,
) -> peppygen::Result<(Harness, Mocks, Fault)> {
    let bundled = robot_description::model(model).expect("robot_description bundles it");
    start_described(
        config,
        model,
        get_description::ResponseData::new(
            bundled.urdf().to_string(),
            bundled.srdf().to_string(),
            bundled
                .meshes()
                .iter()
                .map(|(name, data)| ResponseDataMeshesItem {
                    name: name.to_string(),
                    data: data.to_vec(),
                })
                .collect(),
            bundled.digest().to_string(),
        ),
    )
    .await
}

/// The node over the harness as `config` binds it, its robot's identity answering the
/// node's first asks that it is `model`, described by `description`, and the fault that
/// ends the node.
pub async fn start_described(
    config: Config,
    model: &str,
    description: get_description::ResponseData,
) -> peppygen::Result<(Harness, Mocks, Fault)> {
    let fault = Fault::default();
    let raised = fault.clone();
    let (harness, mocks) = Harness::start_with(config, move |params, runner| {
        motion_planner::node::setup(params, runner, raised)
    })
    .await?;
    let responder = mocks
        .deps
        .identity
        .get_identity
        .next_request(TIMEOUT)
        .await?;
    responder
        .respond(identity::get_identity::ResponseData::new(
            "alpha".to_string(),
            model.to_string(),
            "harness".to_string(),
        ))
        .await?;
    mocks
        .deps
        .identity
        .get_description
        .next_request(TIMEOUT)
        .await?
        .respond(description)
        .await?;
    Ok((harness, mocks, fault))
}

/// The node over the harness as `config` binds it, with no camera to read, once the arms'
/// joint positions at `arms` have arrived: a check of the left arm standing still stops
/// being refused, and answers with no map stamp.
async fn start_vacant(
    config: Config,
    arms: &ArmJoints<ARM_DOF, ARMS>,
) -> peppygen::Result<(Harness, Mocks)> {
    let (harness, mocks) = start(config).await?;
    publish_arms(&mocks, at_ms(0), arms).await?;
    let answer = wait_for_the_arms(&harness, arms[LEFT]).await;
    assert_eq!(answer.map_stamp, None, "{answer:?}");
    Ok((harness, mocks))
}

/// The answer of a check of the left arm standing still at `left`, once it stops being
/// refused: once the arms' joint positions have arrived, on a node that reads no camera.
pub async fn wait_for_the_arms(
    harness: &Harness,
    left: [f64; ARM_DOF],
) -> check_joint_path::Response {
    let still = left_path(&[], &[left]);
    for _ in 0..TIMEOUT.as_secs() {
        match check(harness, &still).await {
            Ok(answer) => return answer,
            Err(failure) => assert_eq!(
                failure.kind,
                check_joint_path::FailureKind::PreconditionFailed,
                "{failure}"
            ),
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    panic!("the arms' joint positions never arrived");
}

/// The alignment the camera's frames and geometry carry unless a test names another.
pub const ALIGNED: &str = "depth_to_color";

/// The node over the harness, with the perception camera at `camera` linked: the
/// camera's description, its frames `ALIGNED`, is scripted, and the node subscribes to
/// its depth frames.
pub async fn start_with_camera(camera: &Isometry3<f64>) -> peppygen::Result<(Harness, Mocks)> {
    let config = Config {
        parameters: Some(parameters()),
        ..Config::default()
    };
    let (harness, mocks) = start(config).await?;
    describe_camera(&mocks, camera, ALIGNED)?;
    let stream = mocks
        .deps
        .camera
        .as_ref()
        .expect("the camera slot is linked");
    assert!(
        stream.depth_stream.wait_for_subscriber(TIMEOUT).await?,
        "the node subscribes to the depth frames"
    );
    assert!(
        mocks
            .deps
            .limb_state
            .limb_states
            .wait_for_subscriber(TIMEOUT)
            .await?,
        "the node subscribes to limb_states"
    );
    Ok((harness, mocks))
}

/// Scripts one description of the perception camera at `camera`, its geometry
/// describing `align_mode`: the answer to each call the node makes once to describe it.
pub fn describe_camera(
    mocks: &Mocks,
    camera: &Isometry3<f64>,
    align_mode: &str,
) -> peppygen::Result<()> {
    let workspace = mocks
        .deps
        .workspace
        .as_ref()
        .expect("the workspace slot is linked");
    workspace
        .describe_workspace
        .enqueue_response(workspace::describe_workspace::ResponseData {
            perception_camera: CAMERA.to_string(),
            workable: true,
            area: 0.0,
            rectangle: None,
            reach: None,
            view: None,
        })?;
    let rotation = camera.rotation;
    let mounts = mocks
        .deps
        .camera_mounts
        .as_ref()
        .expect("the camera_mounts slot is linked");
    mounts
        .get_camera_poses
        .enqueue_response(camera_mounts::get_camera_poses::ResponseData {
            timestamp: SystemTime::UNIX_EPOCH,
            camera_names: vec![CAMERA.to_string()],
            positions: camera.translation.vector.iter().copied().collect(),
            orientations: vec![rotation.i, rotation.j, rotation.k, rotation.w],
            carried_by: vec![String::new()],
        })?;
    let lens = mocks
        .deps
        .geometry
        .as_ref()
        .expect("the geometry slot is linked");
    lens.get_depth_intrinsics
        .enqueue_response(geometry::get_depth_intrinsics::ResponseData {
            width: WIDTH,
            height: HEIGHT,
            fx: FOCAL_PX,
            fy: FOCAL_PX,
            cx: f64::from(WIDTH) / 2.0,
            cy: f64::from(HEIGHT) / 2.0,
            distortion_model: "none".to_string(),
            distortion: Vec::new(),
            depth_model: "z".to_string(),
            min_depth_m: NEAREST_M,
            max_depth_m: FARTHEST_M,
            align_mode: align_mode.to_string(),
        })?;
    lens.get_depth_to_color_extrinsics.enqueue_response(
        geometry::get_depth_to_color_extrinsics::ResponseData {
            align_mode: align_mode.to_string(),
            depth_to_color_position: [0.0; 3],
            depth_to_color_orientation: [0.0, 0.0, 0.0, 1.0],
        },
    )?;
    mocks
        .deps
        .camera
        .as_ref()
        .expect("the camera slot is linked")
        .depth_stream_info
        .enqueue_response(camera::depth_stream_info::ResponseData {
            width: WIDTH,
            height: HEIGHT,
            frames_per_second: 30,
            encoding: "z16".to_string(),
            depth_unit: DEPTH_UNIT,
        })
}

/// Publishes the arms at `arms` measured at `at`, both grippers open.
pub async fn publish_arms(
    mocks: &Mocks,
    at: SystemTime,
    arms: &ArmJoints<ARM_DOF, ARMS>,
) -> peppygen::Result<()> {
    publish_arms_with_grippers_at(mocks, at, arms, 1.0).await
}

/// Publishes the arms at `arms` measured at `at`, both grippers at `opening`.
pub async fn publish_arms_with_grippers_at(
    mocks: &Mocks,
    at: SystemTime,
    arms: &ArmJoints<ARM_DOF, ARMS>,
    opening: f64,
) -> peppygen::Result<()> {
    let rest = |arm: ArmId| arms[arm].to_vec();
    mocks
        .deps
        .limb_state
        .limb_states
        .publish(&limb_state::limb_states::Message {
            timestamp: at,
            arm_names: BOTH.map(|arm| OPENARM_V2.arms[arm].name.clone()).to_vec(),
            joints_per_arm: vec![ARM_DOF as u32; ARMS],
            joint_positions: [rest(LEFT), rest(RIGHT)].concat(),
            positions: vec![0.0; 3 * ARMS],
            orientations: [[0.0, 0.0, 0.0, 1.0]; ARMS].concat(),
            gripper_names: BOTH
                .map(|arm| OPENARM_V2.arms[arm].gripper_name.clone())
                .to_vec(),
            gripper_openings: vec![opening; ARMS],
        })
        .await
}

/// Publishes the depth frame of `render` taken at `at`, its header carrying
/// `align_mode`.
pub async fn publish_frame(
    mocks: &Mocks,
    at: SystemTime,
    render: &[Option<[f64; 3]>],
    align_mode: &str,
) -> peppygen::Result<()> {
    mocks
        .deps
        .camera
        .as_ref()
        .expect("the camera slot is linked")
        .depth_stream
        .publish(&DepthFrame {
            header: MessageHeader {
                timestamp: at,
                frame_id: 0,
                align_mode: align_mode.to_string(),
            },
            encoding: "z16".to_string(),
            width: WIDTH,
            height: HEIGHT,
            frame: depth_frame(render),
        })
        .await
}

/// A plan of the left arm from where it stands to every joint within 0.01 of `q`.
pub fn plan_to(q: &[f64; ARM_DOF]) -> GoalRequestData {
    plan_from(&[], q)
}

/// A plan of the left arm from `start` (empty: where it stands) to every joint within
/// 0.01 of `q`.
pub fn plan_from(start: &[f64], q: &[f64; ARM_DOF]) -> GoalRequestData {
    GoalRequestData {
        arms: vec![GoalArmsItem {
            arm_name: "left_arm".to_string(),
            start_joint_positions: start.to_vec(),
        }],
        goal_joint_constraints: OPENARM_V2.arms[LEFT]
            .joint_names
            .iter()
            .zip(q)
            .map(|(name, &position)| GoalGoalJointConstraintsItem {
                joint_name: name.clone(),
                position,
                tolerance_above: 0.01,
                tolerance_below: 0.01,
            })
            .collect(),
        goal_position_constraints: Vec::new(),
        path_orientation_constraints: Vec::new(),
        goal_orientation_constraints: Vec::new(),
        planner: Planner::RrtConnect.name().to_string(),
        refiner: Refiner::None.name().to_string(),
        allowed_planning_time_s: PLANNING_TIME_S,
    }
}

/// How a plan ends without a path.
#[derive(Debug)]
pub enum Ended {
    /// The node refused the goal: it took nothing.
    Refused(Failure),
    /// The node took the goal and failed it.
    Failed(Failure),
}

/// How a plan ends: its result, or the failure it was refused or failed with.
pub async fn plan(harness: &Harness, goal: &GoalRequestData) -> Result<ResultData, Ended> {
    let handle = send_goal(harness, goal, peppygen::DEFAULT_FEEDBACK_BOUND, TIMEOUT)
        .await
        .expect("the goal is sent");
    if let Some(refusal) = handle.refusal {
        return Err(Ended::Refused(refusal));
    }
    match handle
        .get_result(TIMEOUT)
        .await
        .expect("the result arrives")
        .outcome
    {
        ResultOutcome::Completed(data) => Ok(data),
        ResultOutcome::Failed(failure) => Err(Ended::Failed(failure)),
        other => panic!("the plan ends {other:?}"),
    }
}

/// The waypoints of a plan's arm, as a check takes them.
pub fn planned_waypoints(result: &ResultData) -> Vec<[f64; ARM_DOF]> {
    result.arms[0]
        .waypoints
        .iter()
        .map(|w| {
            w.joint_positions
                .as_slice()
                .try_into()
                .expect("a position for each joint")
        })
        .collect()
}

/// The check of the left arm's path from `start` (empty: where it stands) through
/// `waypoints`.
pub fn left_path(start: &[f64], waypoints: &[[f64; ARM_DOF]]) -> check_joint_path::RequestData {
    use peppygen::exposed_services::motion_planning::check_joint_path::{
        CheckJointPathArmsItem, CheckJointPathArmsItemWaypointsItem,
    };
    check_joint_path::RequestData {
        arms: vec![CheckJointPathArmsItem {
            arm_name: "left_arm".to_string(),
            start_joint_positions: start.to_vec(),
            waypoints: waypoints
                .iter()
                .map(|q| CheckJointPathArmsItemWaypointsItem {
                    joint_positions: q.to_vec(),
                })
                .collect(),
        }],
    }
}

/// Checks `request`, refused or answered.
pub async fn check(
    harness: &Harness,
    request: &check_joint_path::RequestData,
) -> Result<check_joint_path::Response, check_joint_path::Failure> {
    check_joint_path::poll(harness, request, TIMEOUT)
        .await
        .map_err(|e| check_joint_path::Failure::of(&e).unwrap_or_else(|| panic!("{e}")))
}

/// Publishes the `ALIGNED` frame of `render` taken at `at` until the map has taken it
/// in: a check of the left arm from where it stands to `left` stops being refused and
/// answers the frame's stamp.
pub async fn take_frame(
    harness: &Harness,
    mocks: &Mocks,
    at: SystemTime,
    render: &[Option<[f64; 3]>],
    left: [f64; ARM_DOF],
) -> peppygen::Result<()> {
    take_frame_aligned(harness, mocks, at, render, left, ALIGNED).await
}

/// `take_frame` of a frame whose header carries `align_mode`.
pub async fn take_frame_aligned(
    harness: &Harness,
    mocks: &Mocks,
    at: SystemTime,
    render: &[Option<[f64; 3]>],
    left: [f64; ARM_DOF],
    align_mode: &str,
) -> peppygen::Result<()> {
    let probe = left_path(&[], &[left]);
    for _ in 0..TIMEOUT.as_secs() {
        publish_frame(mocks, at, render, align_mode).await?;
        match check(harness, &probe).await {
            Ok(answer) if answer.map_stamp == Some(at) => return Ok(()),
            Ok(_) => {}
            Err(failure) => assert_eq!(
                failure.kind,
                check_joint_path::FailureKind::PreconditionFailed,
                "{failure}"
            ),
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    panic!("the map never took in the frame of {at:?}");
}
