//! What the node's tests share: the parameters the launchers give the planner and the
//! planning time each plan asks for, OpenArm v2's model, a depth camera and the frames it
//! renders of the robot and of boxes, a table across the left arm's motion, the node over
//! the generated harness with that camera linked or vacant, and the plans and checks of
//! the left arm.
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
use motion_planner::arm_set::PerArm;
use motion_planner::history::ArmJoints;
use motion_planner::node::Fault;
use motion_planner::request::OBJECTIVE;
use motion_planner::robot::{Robot, RobotModel, openarm_v2};
use openarm_description::{ARM_DOF, DUAL_DOF, Side, ready};
use peppygen::Parameters;
use peppygen::consumed_topics::camera::depth_stream::{Message as DepthFrame, MessageHeader};
use peppygen::exposed_actions::motion_planning::plan_joint_path::{
    GoalArmsItem, GoalGoalJointConstraintsItem,
};
use peppygen::fixtures::exposed_actions::motion_planning::plan_joint_path::{
    Failure, GoalRequestData, ResultData, ResultOutcome, send_goal,
};
use peppygen::fixtures::exposed_services::motion_planning::check_joint_path;
use peppygen::fixtures::harness::{Config, Harness, Mocks};
use peppygen::mock::deps::{camera, camera_mounts, geometry, identity, limb_state, workspace};

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

/// The planning time every test's plan asks for (s).
pub const PLANNING_TIME_S: f64 = 20.0;

/// OpenArm v2's model, its grippers open, as the node fits it.
pub static OPENARM_V2: LazyLock<RobotModel<ARM_DOF, DUAL_DOF, 2>> =
    LazyLock::new(|| openarm_v2().expect("OpenArm v2's model"));

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
    arms: Option<&ArmJoints<ARM_DOF>>,
    boxes: &[Solid],
) -> Vec<Option<[f64; 3]>> {
    let none = Padding::new(1.0, 0.0).expect("the padding that grows nothing");
    let chains: Vec<_> = OPENARM_V2
        .arms
        .iter()
        .map(|arm| (arm.id.side(), arm.geometry.mask(none)))
        .collect();
    let body = OPENARM_V2.body.mask(none);
    let fingers = arms.map(|arms| {
        BodyMask::of(
            Side::BOTH.into_iter().flat_map(|side| {
                let arm = &OPENARM_V2.arms[side].geometry;
                let posed = arm.chain().at(&arms[side]);
                OPENARM_V2
                    .own
                    .fingers(side.into(), arm)
                    .expect("the arm's fingers")
                    .into_iter()
                    .map(move |finger| {
                        let hand = posed.link_pose_world(finger.segment);
                        let RiderBody::Pieces(pieces) = finger.body else {
                            panic!("a finger is a part of the robot");
                        };
                        (finger.name, pieces, hand * finger.offset)
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
pub fn arms(left: [f64; ARM_DOF], right: [f64; ARM_DOF]) -> ArmJoints<ARM_DOF> {
    PerArm::both(left, right)
}

/// Each arm's Ready posture: elbows bent, hands forward.
pub const LEFT_READY: [f64; ARM_DOF] = ready(Side::Left);
pub const RIGHT_READY: [f64; ARM_DOF] = ready(Side::Right);

/// The left arm at Ready with its elbow at `angle`.
pub const fn left_elbow(angle: f64) -> [f64; ARM_DOF] {
    let mut q = LEFT_READY;
    q[3] = angle;
    q
}

/// The left arm with its hand under the table of `table_across`, the elbow open, and
/// over it, the elbow bent.
pub const UNDER_THE_TABLE: [f64; ARM_DOF] = left_elbow(0.6);
pub const OVER_THE_TABLE: [f64; ARM_DOF] = left_elbow(2.0);

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
    let left = OPENARM_V2.arms[Side::Left].geometry.chain();
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

/// The node over the harness with the four camera slots vacant, once the arms' joint
/// positions at `arms` have arrived: a check of the left arm standing still stops being
/// refused, and answers with no map stamp.
pub async fn start_without_camera(arms: &ArmJoints<ARM_DOF>) -> peppygen::Result<(Harness, Mocks)> {
    start_without_camera_with(parameters(), arms).await
}

/// `start_without_camera` with the node's parameters `params`.
pub async fn start_without_camera_with(
    params: Parameters,
    arms: &ArmJoints<ARM_DOF>,
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
    arms: &ArmJoints<ARM_DOF>,
) -> peppygen::Result<(Harness, Mocks)> {
    start_vacant(vacant_camera(), arms).await
}

/// The harness's configuration with the camera and geometry slots vacant.
fn vacant_camera() -> Config {
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
    let (harness, mocks, _) = start_identified_as(config, &Robot::OpenArmV2.to_string()).await?;
    Ok((harness, mocks))
}

/// The node over the harness as `config` binds it, its robot's identity answering the
/// node's first ask that it is `model`, and the fault that ends the node.
pub async fn start_identified_as(
    config: Config,
    model: &str,
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
    Ok((harness, mocks, fault))
}

/// The node over the harness as `config` binds it, with no camera to read, once the arms'
/// joint positions at `arms` have arrived: a check of the left arm standing still stops
/// being refused, and answers with no map stamp.
async fn start_vacant(
    config: Config,
    arms: &ArmJoints<ARM_DOF>,
) -> peppygen::Result<(Harness, Mocks)> {
    let (harness, mocks) = start(config).await?;
    publish_arms(&mocks, at_ms(0), arms).await?;
    let answer = wait_for_the_arms(&harness, arms[Side::Left]).await;
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
    arms: &ArmJoints<ARM_DOF>,
) -> peppygen::Result<()> {
    let rest = |side: Side| arms[side].to_vec();
    mocks
        .deps
        .limb_state
        .limb_states
        .publish(&limb_state::limb_states::Message {
            timestamp: at,
            arm_names: vec!["left_arm".to_string(), "right_arm".to_string()],
            joints_per_arm: vec![ARM_DOF as u32; 2],
            joint_positions: [rest(Side::Left), rest(Side::Right)].concat(),
            positions: vec![0.0; 6],
            orientations: [[0.0, 0.0, 0.0, 1.0]; 2].concat(),
            gripper_names: vec!["left_gripper".to_string(), "right_gripper".to_string()],
            gripper_openings: vec![1.0; 2],
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
        goal_joint_constraints: OPENARM_V2.arms[Side::Left]
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
        objective: OBJECTIVE.to_string(),
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
