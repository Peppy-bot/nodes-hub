//! What the node's tests share: the parameters the launchers give the planner and the
//! planning time each plan asks for, OpenArm v2's model, a depth camera and the frames it
//! renders of the robot and of boxes, and the node over the generated harness with that
//! camera linked or vacant.
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
use collision_geometry::chain::{BodyMask, Padding, PlacedMask};
use motion_planner::fingers::riders;
use motion_planner::history::Arms;
use motion_planner::robot::{Robot, RobotModel};
use openarm_description::{ARM_DOF, ArmPair, Side, ready};
use peppygen::consumed_topics::camera::depth_stream::{Message as DepthFrame, MessageHeader};
use peppygen::fixtures::exposed_services::motion_planning::check_joint_path;
use peppygen::fixtures::harness::{Config, Harness, Mocks};
use peppygen::mock::deps::{camera, camera_mounts, geometry, limb_state, workspace};

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
pub static OPENARM_V2: LazyLock<RobotModel> =
    LazyLock::new(|| Robot::OpenArmV2.model().expect("OpenArm v2's model"));

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
    arms: Option<&Arms>,
    boxes: &[Solid],
) -> Vec<Option<[f64; 3]>> {
    let none = Padding::new(1.0, 0.0).expect("the padding that grows nothing");
    let chains: Vec<_> = OPENARM_V2
        .arms
        .iter()
        .map(|arm| (arm.side, arm.geometry.mask(none)))
        .collect();
    let body = OPENARM_V2.body.mask(none);
    let fingers = arms.map(|arms| {
        BodyMask::of(
            Side::BOTH.into_iter().flat_map(|side| {
                let arm = &OPENARM_V2.arms[side].geometry;
                let posed = arm.chain().at(&arms[side]);
                riders(&OPENARM_V2.self_model, arm, side)
                    .expect("the arm's fingers")
                    .into_iter()
                    .map(move |finger| {
                        let hand = posed.link_pose_world(finger.segment);
                        (finger.link, finger.pieces, hand * finger.offset)
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
pub fn arms(left: [f64; ARM_DOF], right: [f64; ARM_DOF]) -> Arms {
    ArmPair::new(left, right)
}

/// Each arm's Ready posture: elbows bent, hands forward.
pub const LEFT_READY: [f64; ARM_DOF] = ready(Side::Left);
pub const RIGHT_READY: [f64; ARM_DOF] = ready(Side::Right);

/// The left arm at Ready with its elbow at `angle`.
pub fn left_elbow(angle: f64) -> [f64; ARM_DOF] {
    let mut q = LEFT_READY;
    q[3] = angle;
    q
}

/// The node over the harness with the four camera slots vacant, once the arms' joint
/// positions at `arms` have arrived: a check of the left arm standing still stops being
/// refused, and answers with no map stamp.
pub async fn start_without_camera(arms: &Arms) -> peppygen::Result<(Harness, Mocks)> {
    start_vacant(
        Config {
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
pub async fn start_without_the_rig(arms: &Arms) -> peppygen::Result<(Harness, Mocks)> {
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

/// The node over the harness as `config` binds it, with no camera to read, once the arms'
/// joint positions at `arms` have arrived: a check of the left arm standing still stops
/// being refused, and answers with no map stamp.
async fn start_vacant(config: Config, arms: &Arms) -> peppygen::Result<(Harness, Mocks)> {
    let (harness, mocks) = Harness::start_with(config, motion_planner::node::setup).await?;
    publish_arms(&mocks, at_ms(0), arms).await?;
    let still = left_path(&[], &[arms[Side::Left]]);
    for _ in 0..TIMEOUT.as_secs() {
        match check(&harness, &still).await {
            Ok(answer) => {
                assert_eq!(answer.map_stamp, None, "{answer:?}");
                return Ok((harness, mocks));
            }
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
    let (harness, mocks) = Harness::start_with(config, motion_planner::node::setup).await?;
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
pub async fn publish_arms(mocks: &Mocks, at: SystemTime, arms: &Arms) -> peppygen::Result<()> {
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
