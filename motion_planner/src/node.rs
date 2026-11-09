//! The node: its state, the followers of the arms' joint positions and of the depth
//! frames, and the action and the services of motion_planning, the planning scene's
//! among them.

use std::num::{NonZeroU32, NonZeroUsize};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant, SystemTime};

use camera_core::encoding::DEPTH_ENCODING;
use camera_core::pinhole::{DepthModel, Intrinsics, Pinhole};
use camera_core::pose::Pose;
use chain_kinematics::nalgebra::Isometry3;
use collision_geometry::chain::BodyGeometry;
use control_core::latched_log::LatchedLog;
use control_core::{receiving, serving};
use motion_planning_core::{
    Arm, ArmGoal, Arms, CartesianFailure, CartesianRequest, CheckFailure, CheckRequest, Collisions,
    DepthCamera, Goal, IkGoalSampler, PathCheck, PathOrientation, PlanFailure,
    PlanRequest, PlanningScene, RobotLinks, RobotPart, Scene, SceneError, SelfModel, StartFault,
    cartesian_path, check_path, insert_cut_out, plan_path, pose_parts, start_inside_limits,
};
use occupancy_map::{DistanceField, Frame, Scan, VoxelMap};
use ompl_bridge::{PLANNING_THREADS, Planner};
use peppygen::consumed_services::camera::depth_stream_info;
use peppygen::consumed_services::camera_mounts::get_camera_poses;
use peppygen::consumed_services::geometry::{get_depth_intrinsics, get_depth_to_color_extrinsics};
use peppygen::consumed_services::identity::get_identity;
use peppygen::consumed_services::workspace::describe_workspace;
use peppygen::consumed_topics::camera::depth_stream;
use peppygen::consumed_topics::limb_state::limb_states;
use peppygen::exposed_actions::motion_planning::plan_cartesian_path;
use peppygen::exposed_actions::motion_planning::plan_joint_path::{
    self, FailureKind, GoalRequestData, PlanJointPathResultArmsItem,
    PlanJointPathResultArmsItemWaypointsItem,
};
use peppygen::exposed_services::motion_planning::{
    apply_attached_collision_object, apply_collision_objects, check_joint_path,
    clear_occupancy_map, describe_arms, get_planning_scene, set_allowed_collision,
};
use peppygen::{NodeRunner, Parameters, ProducerRef, Result};
use tokio::sync::watch;
use tracing::{error, info};

use crate::arm_set::{ArmId, PerArm};
use crate::config::Config;
use crate::fingers::{FingerFault, riders_of_the_arms, with_held_objects};
use crate::history::{ArmJoints, History, Measured};
use crate::intake::robot_mask;
use crate::own_body::OwnBody;
use crate::request::{
    ArmGoals, ArmStarts, CartesianQuery, CheckQuery, PlanQuery, parse_cartesian, parse_check,
    parse_plan,
};
use crate::robot::{ArmModel, ModelError, Robot, RobotModel, openarm_v2};
use crate::scene::{attach_request, object_requests, scene_response};

/// Bounds every call to another node.
const SERVICE_TIMEOUT: Duration = Duration::from_secs(5);
/// How long after a check is asked a walk of a run or of an arc of its path may start: a
/// check with a walk left then fails with timeout, under the 15 s the robot's MCP endpoint
/// waits on check_joint_path. The map's copy, its distance field and the checks of the
/// waypoints before the walks run to their end.
const CHECK_TIME: Duration = Duration::from_secs(10);
/// How long the history keeps joint positions: longer than any camera's delay.
const HISTORY_SPAN: Duration = Duration::from_secs(2);
/// The pause before asking the camera again for its description.
const RETRY: Duration = Duration::from_secs(1);
/// The seed of every plan, joined by the map's stamp: the same request from the same
/// start on the same map gives the same path when no part of the plan ends at its
/// deadline.
const PLAN_SEED: u32 = 1;

/// Why a plan or a check is refused while the camera has given the map no frame; the
/// refusal names what the map waits on after it.
pub const NO_DEPTH_FRAME: &str =
    "the map has taken in no depth frame since the planner started or the map was cleared";
/// What the map waits on while no fault holds the camera's frames back.
const AWAITING_FRAME: &str = "no depth frame has reached the map yet";
/// Why a plan, a check or an attach is refused before the arms' joint positions are
/// measured.
const NO_JOINTS: &str = "no joint positions of the arms have been measured yet; ask again \
                         once limb_state publishes every arm's joints and every gripper's opening";
/// What the node logs on a robot whose launcher linked no depth camera.
const NO_CAMERA: &str = "no depth camera is linked to the planner: plans and checks keep \
                         the clearance from the planning scene's objects and, on a robot with a \
                         self model, from its own body, alone";

/// The links of the perception camera the map reads.
struct CameraLinks {
    /// rgbd_camera: the depth frames and their stream's description.
    camera: ProducerRef,
    /// camera_geometry: the depth lens and the depth frame's offset.
    geometry: ProducerRef,
    /// camera_mounts: where the robot's design carries the camera.
    camera_mounts: ProducerRef,
    /// workspace: which camera is the perception camera.
    workspace: ProducerRef,
}

/// The perception camera's slots, as the launcher bound them.
enum CameraSlots {
    /// The four slots bound.
    Linked(CameraLinks),
    /// The camera and geometry slots vacant: the map stays empty.
    Vacant,
    /// Any other binding: the map waits on a camera it cannot read, for the reason
    /// given.
    Mixed(String),
}

/// The perception camera's slots, as the launcher bound them: camera and geometry
/// decide whether the map reads a camera, and a camera the map reads needs
/// camera_mounts and workspace bound too.
fn camera_slots(runner: &NodeRunner) -> CameraSlots {
    let slots = [
        ("camera", depth_stream_info::bound_producer(runner)),
        ("geometry", get_depth_intrinsics::bound_producer(runner)),
        ("camera_mounts", get_camera_poses::bound_producer(runner)),
        ("workspace", describe_workspace::bound_producer(runner)),
    ];
    let names = |bound: bool| -> Vec<&str> {
        slots
            .iter()
            .filter(|(_, producer)| producer.is_some() == bound)
            .map(|(name, _)| *name)
            .collect()
    };
    match slots.map(|(_, producer)| producer) {
        [
            Some(camera),
            Some(geometry),
            Some(camera_mounts),
            Some(workspace),
        ] => CameraSlots::Linked(CameraLinks {
            camera: camera.clone(),
            geometry: geometry.clone(),
            camera_mounts: camera_mounts.clone(),
            workspace: workspace.clone(),
        }),
        [None, None, _, _] => CameraSlots::Vacant,
        _ => CameraSlots::Mixed(format!(
            "the launcher binds the planner's {} and leaves its {} vacant; bind camera, \
             geometry, camera_mounts and workspace, or leave camera and geometry vacant",
            listed(&names(true)),
            listed(&names(false))
        )),
    }
}

/// `names` as a list in a sentence: "a", "a and b", "a, b and c".
fn listed(names: &[&str]) -> String {
    match names {
        [] => String::new(),
        [one] => (*one).to_string(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

/// The map, the stamp of the last depth frame it took in, what it waits on while it has
/// none, and the instant of the newest joint positions measured when it was last
/// cleared: a frame stamped at or before that instant marks nothing.
struct Map {
    voxels: VoxelMap,
    stamp: Option<SystemTime>,
    waiting: Waiting,
    cleared_at: Option<SystemTime>,
}

/// What the map waits on while it holds no frame.
#[derive(Debug, PartialEq)]
enum Waiting {
    /// A frame, which the reason given may hold back for a while.
    Passing(String),
    /// A fault that ends the map's intake for good, for the reason given.
    Lasting(String),
}

impl Waiting {
    fn reason(&self) -> &str {
        match self {
            Self::Passing(reason) | Self::Lasting(reason) => reason,
        }
    }
}

impl Map {
    /// Notes that the map waits on `waiting` while it holds no frame; a lasting fault
    /// stays.
    fn wait_on(&mut self, waiting: Waiting) {
        if !matches!(self.waiting, Waiting::Lasting(_)) {
            self.waiting = waiting;
        }
    }
}

/// What the node holds.
struct Node<const N: usize, const D: usize, const K: usize> {
    config: Config,
    /// The robot frame, the URDF's root link: the frame of the map, of each depth
    /// frame's sensor and of the robot cut out of it.
    frame: Frame,
    arms: PerArm<ArmModel<N>>,
    /// The robot's body around its arms, which the map cuts out of each frame.
    body: BodyGeometry,
    /// The robot's own body, its grippers at the openings it was built with.
    own: OwnBody<N, K>,
    /// The collision objects, the objects attached to the arms' grasp links, and the
    /// pairs allowed to touch.
    scene: Mutex<PlanningScene>,
    map: Mutex<Map>,
    /// The measured joint positions and gripper openings, which a depth frame waits on
    /// until they reach its stamp.
    history: watch::Sender<History<N>>,
    /// How many times the history has restarted on a clock that stepped back.
    restarts: AtomicU64,
    /// Whether the launcher bound a slot of the depth camera.
    camera_linked: bool,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The pose in the robot frame of the grasp link `link` with the arms at `arms`.
///
/// # Panics
///
/// When `link` is no arm's grasp link: the scene attaches objects to grasp links alone.
fn grasp_pose<const N: usize>(
    models: &PerArm<ArmModel<N>>,
    arms: &ArmJoints<N>,
    link: &str,
) -> Isometry3<f64> {
    let arm = models
        .iter()
        .find(|arm| arm.grasp_link == link)
        .expect("the scene attaches objects to grasp links alone");
    arm.geometry.chain().world_ee_pose(&arms[arm.id])
}

impl<const N: usize, const D: usize, const K: usize> Node<N, D, K> {
    /// The planning scene and the map, held together: every holder of both takes them
    /// here, the scene first.
    fn scene_and_map(&self) -> (MutexGuard<'_, PlanningScene>, MutexGuard<'_, Map>) {
        let scene = lock(&self.scene);
        (scene, lock(&self.map))
    }

    /// Notes the passing fault the map waits on while it holds no frame.
    fn wait_on(&self, reason: impl Into<String>) {
        lock(&self.map).wait_on(Waiting::Passing(reason.into()));
    }

    /// Notes the lasting fault that ends the map's intake: the map waits on it for good.
    fn wait_on_lasting(&self, reason: impl Into<String>) {
        lock(&self.map).wait_on(Waiting::Lasting(reason.into()));
    }

    /// The newest measurement.
    fn latest(&self) -> Option<Measured<N>> {
        self.history.borrow().latest()
    }

    /// The robot's own body with the grippers at the openings of `measured`, or at the
    /// openings it was built with when nothing is measured.
    fn own_at(&self, measured: Option<&Measured<N>>) -> OwnBody<N, K> {
        let mut own = self.own.clone();
        if let Some(measured) = measured {
            own.set_openings(&measured.openings);
        }
        own
    }
}

/// The fault that ends the node after its setup: a model the robot's identity names that
/// the planner holds no model of, or a model that fails to start. The first raise shuts
/// the node down, and `main` exits with the fault as its error.
#[derive(Clone, Debug, Default)]
pub struct Fault(Arc<OnceLock<String>>);

impl Fault {
    /// Records `message`, unless a fault came first, and shuts the node down.
    fn raise(&self, runner: &NodeRunner, message: &str) {
        error!("{message}");
        let _ = self.0.set(message.to_owned());
        runner.cancellation_token().cancel();
    }

    /// The end of the node: the error of its fault, if one was raised.
    pub fn end(&self) -> Result<()> {
        self.0.get().map_or(Ok(()), |message| {
            Err(peppygen::Error::Node(
                std::io::Error::other(message.clone()).into(),
            ))
        })
    }
}

/// Starts the node: its parameters parsed, then, once the robot's identity names its
/// model, its state, the followers and the servers, each on its own task. The
/// initializer that serves the identity may start after this node.
pub async fn setup(params: Parameters, runner: Arc<NodeRunner>, fault: Fault) -> Result<()> {
    let config = Config::parse(&params).map_err(|e| peppygen::Error::Node(e.into()))?;
    tokio::spawn(serve_once_identified(runner, config, fault));
    Ok(())
}

/// Asks the robot's identity until it names the model, then starts the node for it. A
/// model the planner holds no model of, or one that fails to start, raises `fault`.
async fn serve_once_identified(runner: Arc<NodeRunner>, config: Config, fault: Fault) {
    let Some(model) = identify(&runner).await else {
        return;
    };
    let robot: Robot = match model.parse() {
        Ok(robot) => robot,
        Err(e) => return fault.raise(&runner, &format!("robot_identity: {e}")),
    };
    match robot {
        Robot::OpenArmV2 => serve_robot(runner, config, fault, robot, openarm_v2()),
    }
}

/// Starts the node for `robot`, whose model is `model`, and its followers and servers. A
/// model that fails to build, or to start, raises `fault`.
fn serve_robot<const N: usize, const D: usize, const K: usize>(
    runner: Arc<NodeRunner>,
    config: Config,
    fault: Fault,
    robot: Robot,
    model: std::result::Result<RobotModel<N, D, K>, ModelError>,
) {
    let slots = camera_slots(&runner);
    let linked = !matches!(slots, CameraSlots::Vacant);
    let started = model
        .map_err(|e| e.to_string())
        .and_then(|model| start(config, robot, model, linked).map_err(|e| e.to_string()));
    let node = match started {
        Ok(node) => Arc::new(node),
        Err(e) => return fault.raise(&runner, &format!("the motion planner for {robot}: {e}")),
    };
    tokio::spawn(follow_limb_states(runner.clone(), node.clone()));
    match slots {
        CameraSlots::Linked(links) => {
            tokio::spawn(follow_depth(runner.clone(), node.clone(), links));
        }
        CameraSlots::Vacant => info!("{NO_CAMERA}"),
        CameraSlots::Mixed(reason) => {
            error!("perception camera: {reason}");
            node.wait_on_lasting(reason);
        }
    }
    tokio::spawn(serve_descriptions(runner.clone(), node.clone()));
    tokio::spawn(serve_clear(runner.clone(), node.clone()));
    tokio::spawn(serve_objects(runner.clone(), node.clone()));
    tokio::spawn(serve_attachments(runner.clone(), node.clone()));
    tokio::spawn(serve_allowed_pairs(runner.clone(), node.clone()));
    tokio::spawn(serve_scene(runner.clone(), node.clone()));
    tokio::spawn(serve_checks(runner.clone(), node.clone()));
    tokio::spawn(serve_cartesian_paths(runner.clone(), node.clone()));
    tokio::spawn(serve_plans(runner, node));
}

/// The model the robot's identity names, or none once the node is cancelled: each ask
/// waits up to `SERVICE_TIMEOUT` for the answer, and the next follows `RETRY` after.
async fn identify(runner: &NodeRunner) -> Option<String> {
    let token = runner.cancellation_token().clone();
    let passing = LatchedLog::warn("robot_identity");
    loop {
        let producer = get_identity::bound_producer(runner);
        match get_identity::poll(runner, producer, SERVICE_TIMEOUT).await {
            Ok(answer) => {
                passing.success();
                return Some(answer.data.model);
            }
            Err(e) => passing.failure(format!("get_identity: {e}; asking again")),
        }
        tokio::select! {
            _ = token.cancelled() => return None,
            _ = tokio::time::sleep(RETRY) => {}
        }
    }
}

/// The node at start: the model of `robot`, an empty scene of the robot's links, and an
/// empty map.
fn start<const N: usize, const D: usize, const K: usize>(
    config: Config,
    robot: Robot,
    model: RobotModel<N, D, K>,
    camera_linked: bool,
) -> std::result::Result<Node<N, D, K>, Box<dyn std::error::Error + Send + Sync>> {
    let RobotModel {
        frame,
        arms,
        body,
        own,
    } = model;
    let links = RobotLinks::new(
        arms.iter()
            .flat_map(|arm| arm.link_names.iter().cloned())
            .chain(body.link_names().map(str::to_string)),
        arms.iter().map(|arm| arm.grasp_link.to_string()),
    );
    let voxels = VoxelMap::new(frame.clone(), config.voxel_size, config.bounds)?;
    info!(
        "motion planner: {}, {} m voxels, clearance {} m, {} attempts",
        robot,
        config.voxel_size.get(),
        config.min_clearance_m,
        config.planning_attempts
    );
    let history = History::new(&arms, HISTORY_SPAN);
    Ok(Node {
        config,
        frame,
        arms,
        body,
        own,
        scene: Mutex::new(PlanningScene::new(links)),
        map: Mutex::new(Map {
            voxels,
            stamp: None,
            waiting: Waiting::Passing(AWAITING_FRAME.to_string()),
            cleared_at: None,
        }),
        history: watch::Sender::new(history),
        restarts: AtomicU64::new(0),
        camera_linked,
    })
}

/// Records each limb_states message; a run of messages that lack an arm's joints logs
/// once.
async fn follow_limb_states<const N: usize, const D: usize, const K: usize>(
    runner: Arc<NodeRunner>,
    node: Arc<Node<N, D, K>>,
) {
    let token = runner.cancellation_token().clone();
    let mut subscription = match limb_states::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => return error!("limb_states subscribe: {e}"),
    };
    let malformed = LatchedLog::warn("limb_states");
    receiving::receive(
        "limb_states",
        token.cancelled(),
        async move || subscription.next().await,
        async move |(_producer, message)| {
            record(&node, &message, &malformed);
            ControlFlow::Continue(())
        },
    )
    .await;
}

/// Records one limb_states message in the history, at its stamp: each arm's joint
/// positions and each gripper's opening.
fn record<const N: usize, const D: usize, const K: usize>(
    node: &Node<N, D, K>,
    message: &limb_states::Message,
    malformed: &LatchedLog,
) {
    let measured = arms_of(&node.arms, message)
        .zip(openings_of(&node.arms, message))
        .map(|(arms, openings)| Measured { arms, openings });
    match measured {
        Some(measured) => {
            push_measured(node, message.timestamp, &measured);
            malformed.success();
        }
        None => malformed.failure(format!(
            "a message holds no {N} finite joint positions for each arm, or no opening for each \
             gripper"
        )),
    }
}

/// Records `measured` in the history at `at`. A stamp before the newest restarts the
/// history on a clock that stepped back, and clears the map at that stamp; the frames that
/// wait for joint positions on the clock before it are dropped (`WaitingFrames`).
fn push_measured<const N: usize, const D: usize, const K: usize>(
    node: &Node<N, D, K>,
    at: SystemTime,
    measured: &Measured<N>,
) {
    let stepped_back = node
        .history
        .borrow()
        .latest_at()
        .is_some_and(|newest| at < newest);
    if stepped_back {
        node.restarts.fetch_add(1, Ordering::SeqCst);
    }
    node.history
        .send_modify(|history| history.push(at, measured));
    if stepped_back {
        clear(node);
        info!(
            "limb_states stepped back to {at:?}: the joint history restarts and the map is cleared"
        );
    }
}

/// Each arm of `models`' joint positions in a limb_states message, each finite.
fn arms_of<const N: usize>(
    models: &PerArm<ArmModel<N>>,
    message: &limb_states::Message,
) -> Option<ArmJoints<N>> {
    let mut offset = 0;
    let mut found: PerArm<Option<[f64; N]>> = models.each(|_, _| None);
    for (name, &count) in message.arm_names.iter().zip(&message.joints_per_arm) {
        let count = count as usize;
        let values = message.joint_positions.get(offset..offset + count)?;
        offset += count;
        if let Some(arm) = models.ids().find(|arm| models[*arm].name == name) {
            found[arm] = <[f64; N]>::try_from(values)
                .ok()
                .filter(|q| q.iter().all(|v| v.is_finite()));
        }
    }
    found.transpose()
}

/// Each gripper of `models`' opening in a limb_states message.
fn openings_of<const N: usize>(
    models: &PerArm<ArmModel<N>>,
    message: &limb_states::Message,
) -> Option<PerArm<f64>> {
    let mut found: PerArm<Option<f64>> = models.each(|_, _| None);
    for (name, &opening) in message.gripper_names.iter().zip(&message.gripper_openings) {
        if let Some(arm) = models.ids().find(|arm| models[*arm].gripper_name == name) {
            found[arm] = Some(opening);
        }
    }
    found.transpose()
}

/// The perception camera as the map reads it.
struct PerceptionCamera {
    depth: DepthCamera,
    /// The alignment the lens and the depth frame's offset describe: a depth frame read
    /// with them carries it in its header (camera_geometry's `align_mode`).
    align_mode: String,
}

/// Why the perception camera cannot be read.
#[derive(Debug, PartialEq)]
enum CameraFault {
    /// A fault that asking again may clear: a call that failed, or answers that
    /// disagree.
    Passing(String),
    /// A fault of the robot's description, which asking again never clears.
    Lasting(String),
}

/// The depth camera, asked for until the robot names one and the camera describes
/// itself; `None` once a lasting fault, or the node's shutdown, ends the asking. The
/// map waits on each fault meanwhile.
async fn discover_camera<const N: usize, const D: usize, const K: usize>(
    runner: &Arc<NodeRunner>,
    node: &Node<N, D, K>,
    links: &CameraLinks,
) -> Option<PerceptionCamera> {
    let token = runner.cancellation_token().clone();
    let passing = LatchedLog::warn("perception camera");
    loop {
        match describe_camera(runner, links).await {
            Ok(camera) => {
                passing.success();
                node.wait_on(AWAITING_FRAME);
                return Some(camera);
            }
            Err(CameraFault::Lasting(reason)) => {
                error!("perception camera: {reason}");
                node.wait_on_lasting(format!("the perception camera: {reason}"));
                return None;
            }
            Err(CameraFault::Passing(reason)) => {
                node.wait_on(format!("the perception camera: {reason}; asking again"));
                passing.failure(reason);
            }
        }
        tokio::select! {
            _ = token.cancelled() => return None,
            _ = tokio::time::sleep(RETRY) => {}
        }
    }
}

async fn describe_camera(
    runner: &Arc<NodeRunner>,
    links: &CameraLinks,
) -> std::result::Result<PerceptionCamera, CameraFault> {
    let name = describe_workspace::poll(
        runner,
        &links.workspace,
        SERVICE_TIMEOUT,
        describe_workspace::Request::new(0.0),
    )
    .await
    .map_err(|e| CameraFault::Passing(format!("describe_workspace: {e}")))?
    .data
    .perception_camera;
    if name.is_empty() {
        return Err(CameraFault::Lasting(
            "the robot names no perception camera".to_string(),
        ));
    }
    let poses = get_camera_poses::poll(runner, &links.camera_mounts, SERVICE_TIMEOUT)
        .await
        .map_err(|e| CameraFault::Passing(format!("get_camera_poses: {e}")))?
        .data;
    let color_in_robot = placed_camera(&poses, &name)?;
    let intrinsics = get_depth_intrinsics::poll(runner, &links.geometry, SERVICE_TIMEOUT)
        .await
        .map_err(|e| CameraFault::Passing(format!("get_depth_intrinsics: {e}")))?
        .data;
    let extrinsics = get_depth_to_color_extrinsics::poll(runner, &links.geometry, SERVICE_TIMEOUT)
        .await
        .map_err(|e| CameraFault::Passing(format!("get_depth_to_color_extrinsics: {e}")))?
        .data;
    if intrinsics.align_mode != extrinsics.align_mode {
        return Err(CameraFault::Passing(format!(
            "get_depth_intrinsics describes align_mode {:?} and get_depth_to_color_extrinsics {:?}",
            intrinsics.align_mode, extrinsics.align_mode
        )));
    }
    let info = depth_stream_info::poll(runner, &links.camera, SERVICE_TIMEOUT)
        .await
        .map_err(|e| CameraFault::Passing(format!("depth_stream_info: {e}")))?
        .data;
    let depth_in_color = Pose::new(
        extrinsics.depth_to_color_position,
        extrinsics.depth_to_color_orientation,
    )
    .map_err(|e| CameraFault::Passing(format!("get_depth_to_color_extrinsics: {e}")))?;
    let lens = Pinhole::parse(&Intrinsics {
        width: intrinsics.width,
        height: intrinsics.height,
        fx: intrinsics.fx,
        fy: intrinsics.fy,
        cx: intrinsics.cx,
        cy: intrinsics.cy,
        distortion_model: &intrinsics.distortion_model,
        distortion: &intrinsics.distortion,
    })
    .map_err(|e| CameraFault::Passing(format!("get_depth_intrinsics: {e}")))?;
    let depth_model = DepthModel::parse(&intrinsics.depth_model)
        .map_err(|e| CameraFault::Passing(format!("get_depth_intrinsics: {e}")))?;
    info!(
        "perception camera {name}: depth {}x{}, align_mode {:?}",
        intrinsics.width, intrinsics.height, intrinsics.align_mode
    );
    Ok(PerceptionCamera {
        depth: DepthCamera::new(
            &color_in_robot,
            &depth_in_color,
            lens,
            depth_model,
            info.depth_unit,
        ),
        align_mode: intrinsics.align_mode,
    })
}

/// Where camera_mounts' `poses` place the color frame of the camera `name` in the robot
/// frame. An answer whose lists do not give each named camera its three coordinates,
/// four quaternion components and carrier, no camera by that name, one an arm carries,
/// or a pose that is none, is a lasting fault.
fn placed_camera(
    poses: &get_camera_poses::ResponseData,
    name: &str,
) -> std::result::Result<Pose, CameraFault> {
    let (positions, partial_position) = poses.positions.as_chunks::<3>();
    let (orientations, partial_orientation) = poses.orientations.as_chunks::<4>();
    let count = poses.camera_names.len();
    if !partial_position.is_empty()
        || !partial_orientation.is_empty()
        || positions.len() != count
        || orientations.len() != count
        || poses.carried_by.len() != count
    {
        return Err(CameraFault::Lasting(format!(
            "camera_mounts' get_camera_poses names {count} cameras and gives {} position \
             values, {} orientation values and {} carried_by entries: 3, 4 and 1 for each \
             camera",
            poses.positions.len(),
            poses.orientations.len(),
            poses.carried_by.len()
        )));
    }
    let (position, orientation, carrier) = poses
        .camera_names
        .iter()
        .zip(positions.iter().zip(orientations).zip(&poses.carried_by))
        .find_map(|(named, ((position, orientation), carrier))| {
            (named == name).then_some((*position, *orientation, carrier))
        })
        .ok_or_else(|| {
            CameraFault::Lasting(format!("camera_mounts places no camera named {name:?}"))
        })?;
    if !carrier.is_empty() {
        return Err(CameraFault::Lasting(format!(
            "{name} rides on {carrier}; the planner reads a camera the body carries"
        )));
    }
    Pose::new(position, orientation)
        .map_err(|e| CameraFault::Lasting(format!("camera_mounts' pose of {name}: {e}")))
}

/// The newest depth frame that has arrived, as the intake takes it.
type Newest = Option<Arc<depth_stream::Message>>;

/// Receives the perception camera's depth frames, each in place of the one before it,
/// and takes the newest into the map whenever the intake is free.
async fn follow_depth<const N: usize, const D: usize, const K: usize>(
    runner: Arc<NodeRunner>,
    node: Arc<Node<N, D, K>>,
    links: CameraLinks,
) {
    let token = runner.cancellation_token().clone();
    let mut subscription = match depth_stream::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => return error!("depth_stream subscribe: {e}"),
    };
    let (newest, frames) = watch::channel::<Newest>(None);
    tokio::spawn(take_frames(runner.clone(), node, links, frames));
    receiving::receive(
        "depth_stream",
        token.cancelled(),
        async move || subscription.next().await,
        async move |(_producer, frame)| {
            newest.send_replace(Some(Arc::new(frame)));
            ControlFlow::Continue(())
        },
    )
    .await;
}

/// Takes the frames of `frames` into the map, one at a time, once the camera has
/// described itself: of the frames that wait for the joint positions at their stamps,
/// the newest whose joint positions are measured (`WaitingFrames`). A frame whose
/// alignment the description does not describe asks the camera again.
async fn take_frames<const N: usize, const D: usize, const K: usize>(
    runner: Arc<NodeRunner>,
    node: Arc<Node<N, D, K>>,
    links: CameraLinks,
    frames: watch::Receiver<Newest>,
) {
    let token = runner.cancellation_token().clone();
    let Some(camera) = discover_camera(&runner, &node, &links).await else {
        return;
    };
    let mut intake = Intake::new(node.clone(), camera);
    let mut waiting = WaitingFrames::new(node.clone(), frames);
    loop {
        let next = tokio::select! {
            _ = token.cancelled() => None,
            frame = waiting.next() => frame,
        };
        let Some(frame) = next else {
            return;
        };
        if frame.header.align_mode != intake.camera.align_mode {
            let reason = format!(
                "the depth frames carry align_mode {:?}, and the camera's geometry describes {:?}",
                frame.header.align_mode, intake.camera.align_mode
            );
            intake.skip(format!("{reason}; asking the camera again"));
            tokio::select! {
                _ = token.cancelled() => return,
                _ = tokio::time::sleep(RETRY) => {}
            }
            let Some(camera) = discover_camera(&runner, &node, &links).await else {
                return;
            };
            intake.camera = camera;
            continue;
        }
        tokio::task::block_in_place(|| intake.take(&frame));
    }
}

/// The depth frames that wait for the joint positions of `node` at their stamps: the
/// oldest frame that has arrived since the intake last took one, and the newest that
/// arrived after it. A frame between the two is never taken.
struct WaitingFrames<const N: usize, const D: usize, const K: usize> {
    node: Arc<Node<N, D, K>>,
    frames: watch::Receiver<Newest>,
    history: watch::Receiver<History<N>>,
    /// The history's restarts the waiting frames were stamped after.
    restarts: u64,
    oldest: Option<Arc<depth_stream::Message>>,
    newest: Option<Arc<depth_stream::Message>>,
}

impl<const N: usize, const D: usize, const K: usize> WaitingFrames<N, D, K> {
    fn new(node: Arc<Node<N, D, K>>, frames: watch::Receiver<Newest>) -> Self {
        Self {
            history: node.history.subscribe(),
            restarts: node.restarts.load(Ordering::SeqCst),
            node,
            frames,
            oldest: None,
            newest: None,
        }
    }

    /// The newest waiting frame whose stamp the newest joint positions have reached, once
    /// one has: so a frame comes while the joint positions lag the camera by more than a
    /// frame interval. Meanwhile the map waits on the oldest waiting frame. A restart of the
    /// history drops the frames that wait. `None` once the follower of the frames or the
    /// history has ended.
    async fn next(&mut self) -> Option<Arc<depth_stream::Message>> {
        loop {
            self.drop_on_restart();
            self.note_arrived()?;
            let measured_at = self.history.borrow_and_update().latest_at();
            if let Some(frame) = self.take_measured(measured_at) {
                self.note_handed_off(measured_at);
                return Some(frame);
            }
            self.note_waiting(measured_at);
            let arrived = tokio::select! {
                frame = next_frame(&mut self.frames) => Some(frame?),
                changed = self.history.changed() => {
                    changed.ok()?;
                    None
                }
            };
            if let Some(frame) = arrived {
                self.arrive(frame);
            }
        }
    }

    /// Notes the frame that has arrived since the last one noted, if one has; `None` once
    /// the follower of the frames has ended.
    fn note_arrived(&mut self) -> Option<()> {
        if self.frames.has_changed().ok()? {
            let arrived = self.frames.borrow_and_update().clone();
            if let Some(frame) = arrived {
                self.arrive(frame);
            }
        }
        Some(())
    }

    /// Drops the waiting frames when the history has restarted since they arrived: they
    /// are stamped on the clock before the restart.
    fn drop_on_restart(&mut self) {
        let restarts = self.node.restarts.load(Ordering::SeqCst);
        if restarts != self.restarts {
            self.restarts = restarts;
            self.newest = None;
            if self.oldest.take().is_some() {
                self.node.wait_on(AWAITING_FRAME);
            }
        }
    }

    /// Notes on the map what it waits on once a frame is handed to the intake: the frame
    /// that still waits, or else the next frame. The intake notes a frame it skips.
    fn note_handed_off(&self, measured_at: Option<SystemTime>) {
        match self.oldest {
            Some(_) => self.note_waiting(measured_at),
            None => self.node.wait_on(AWAITING_FRAME),
        }
    }

    /// Notes on the map that the oldest waiting frame, if one waits, waits for the joint
    /// positions at its stamp, and how far they lag it.
    fn note_waiting(&self, measured_at: Option<SystemTime>) {
        let Some(oldest) = &self.oldest else {
            return;
        };
        let lag = match measured_at {
            Some(newest) => format!(
                "{:?} after the newest limb_states message",
                oldest
                    .header
                    .timestamp
                    .duration_since(newest)
                    .unwrap_or(Duration::ZERO)
            ),
            None => "and no limb_states message has arrived".to_string(),
        };
        self.node.wait_on(format!(
            "a depth frame waits for the joint positions at its stamp, {lag}"
        ));
    }

    /// Notes `frame`'s arrival: the oldest waiting frame when none waits, else the newest.
    fn arrive(&mut self, frame: Arc<depth_stream::Message>) {
        if self.oldest.is_none() {
            self.oldest = Some(frame);
        } else {
            self.newest = Some(frame);
        }
    }

    /// The newest waiting frame stamped at or before `measured_at`: the newest frame,
    /// the oldest then dropped, or else the oldest, the newest then the oldest.
    fn take_measured(
        &mut self,
        measured_at: Option<SystemTime>,
    ) -> Option<Arc<depth_stream::Message>> {
        let measured_at = measured_at?;
        let measured = |frame: &Arc<depth_stream::Message>| frame.header.timestamp <= measured_at;
        if self.newest.as_ref().is_some_and(measured) {
            self.oldest = None;
            return self.newest.take();
        }
        if self.oldest.as_ref().is_some_and(measured) {
            return std::mem::replace(&mut self.oldest, self.newest.take());
        }
        None
    }
}

/// The newest frame of `frames` once a frame arrives after the last one taken from it;
/// a frame the next replaces before the intake takes it is never taken. `None` once the
/// follower of the frames has ended.
async fn next_frame(frames: &mut watch::Receiver<Newest>) -> Option<Arc<depth_stream::Message>> {
    loop {
        frames.changed().await.ok()?;
        if let Some(frame) = frames.borrow_and_update().clone() {
            return Some(frame);
        }
    }
}

/// The map's intake of the perception camera's depth frames.
struct Intake<const N: usize, const D: usize, const K: usize> {
    node: Arc<Node<N, D, K>>,
    camera: PerceptionCamera,
    /// The robot's own body, whose grippers the intake puts at each frame's openings.
    own: OwnBody<N, K>,
    /// The stamp of the last frame the map took in.
    taken: Option<SystemTime>,
    /// The frames the map skips.
    skipped: LatchedLog,
}

impl<const N: usize, const D: usize, const K: usize> Intake<N, D, K> {
    fn new(node: Arc<Node<N, D, K>>, camera: PerceptionCamera) -> Self {
        Self {
            own: node.own.clone(),
            node,
            camera,
            taken: None,
            skipped: LatchedLog::warn("depth frame intake"),
        }
    }

    /// Takes `frame` into the map, with the robot cut out where it stood at the frame's
    /// stamp, its grippers at their openings then and the objects attached to its grasp
    /// links held there, and the planning scene's other objects where they stand, unless
    /// it comes within a frame interval after the last one taken, or the camera took it at
    /// or before the newest joint positions measured when the map was last cleared,
    /// whenever it arrives. A frame of another encoding than `DEPTH_ENCODING` is skipped.
    fn take(&mut self, frame: &depth_stream::Message) {
        let node = &self.node;
        if frame.encoding != DEPTH_ENCODING {
            return self.skip(format!(
                "it is encoded {:?}; the map reads {DEPTH_ENCODING:?} depth frames",
                frame.encoding
            ));
        }
        let at = frame.header.timestamp;
        let recent = self.taken.is_some_and(|t| {
            at.duration_since(t)
                .is_ok_and(|d| d < node.config.frame_interval)
        });
        if recent {
            return;
        }
        let Some(measured) = node.history.borrow().at(at) else {
            return self.skip("no joint positions were measured at its stamp");
        };
        let points = match self.camera.depth.points(&frame.frame) {
            Ok(points) => points,
            Err(e) => return self.skip(e.to_string()),
        };
        self.own.set_openings(&measured.openings);
        let fingers = match riders_of_the_arms(&self.own, &node.arms) {
            Ok(fingers) => fingers,
            Err(e) => return self.skip(format!("the robot's fingers: {e}")),
        };
        let (scene, mut map) = node.scene_and_map();
        if map.cleared_at.is_some_and(|cleared| at <= cleared) {
            return;
        }
        let carried =
            fingers.map(|arm, fingers| with_held_objects(fingers, &node.arms[arm], &scene));
        let robot = robot_mask(
            &node.arms,
            &node.body,
            &measured.arms,
            &carried,
            &scene,
            node.config.padding,
        );
        insert_cut_out(
            &mut map.voxels,
            &Scan {
                frame: &node.frame,
                sensor: self.camera.depth.sensor,
                points: &points,
                max_range_m: node.config.max_range_m,
            },
            &robot,
        )
        .expect("the map and every scan stand in the robot frame");
        map.stamp = Some(at);
        self.taken = Some(at);
        self.skipped.success();
    }

    /// Skips a frame for `reason`, which the map waits on while it holds no frame.
    fn skip(&self, reason: impl Into<String>) {
        let reason = format!("a depth frame was skipped: {}", reason.into());
        self.node.wait_on(reason.clone());
        self.skipped.failure(reason);
    }
}

/// Why a request is refused, or ends with no answer: it does not parse or the planning
/// scene refuses it, the planner cannot plan or check yet, a check runs past
/// `CHECK_TIME`, or a finger of the robot's own body does not fit its arm, which the robot's
/// model refuses at start.
#[derive(Debug, PartialEq)]
enum Refusal {
    Rejected(String),
    PreconditionFailed(String),
    TimedOut(String),
    AttemptFailed(String),
}

impl From<SceneError> for Refusal {
    /// The scene refuses a request with rejected, but one whose grasp link stands where
    /// no measured joint positions place it yet, which plans and checks refuse with the
    /// same message.
    fn from(error: SceneError) -> Self {
        match error {
            SceneError::Unplaced(_) => Self::PreconditionFailed(NO_JOINTS.to_string()),
            _ => Self::Rejected(error.to_string()),
        }
    }
}

/// Gives the failure of each member `$member`, the plan's goal and each service, for a
/// refusal: the kind of the same name.
macro_rules! member_failures {
    ($($member:ident),+) => {$(
        impl From<Refusal> for $member::Failure {
            fn from(refusal: Refusal) -> Self {
                use $member::{Failure, FailureKind};
                match refusal {
                    Refusal::Rejected(message) => Failure::new(FailureKind::Rejected, message),
                    Refusal::PreconditionFailed(message) => {
                        Failure::new(FailureKind::PreconditionFailed, message)
                    }
                    Refusal::TimedOut(message) => Failure::new(FailureKind::Timeout, message),
                    Refusal::AttemptFailed(message) => {
                        Failure::new(FailureKind::AttemptFailed, message)
                    }
                }
            }
        }
    )+};
}

member_failures!(
    plan_joint_path,
    plan_cartesian_path,
    check_joint_path,
    apply_collision_objects,
    apply_attached_collision_object,
    set_allowed_collision
);

/// Refuses `map` while a linked camera has given it no frame, naming what it waits on.
fn framed<const N: usize, const D: usize, const K: usize>(
    node: &Node<N, D, K>,
    map: &Map,
) -> std::result::Result<(), Refusal> {
    if node.camera_linked && map.stamp.is_none() {
        return Err(Refusal::PreconditionFailed(format!(
            "{NO_DEPTH_FRAME}: {}",
            map.waiting.reason()
        )));
    }
    Ok(())
}

/// The arms a plan or a check moves, in the order a state of it holds their joints: one
/// arm, or every arm of the robot, in the order of its model.
#[derive(Clone, Copy)]
enum Group {
    One(ArmId),
    Every,
}

impl Group {
    /// The group of the arms of `moving`, each once: every arm of the robot when more than
    /// one moves.
    ///
    /// # Panics
    ///
    /// When `moving` holds no arm.
    fn of(moving: &[ArmId]) -> Self {
        match moving {
            [] => panic!("a plan or a check moves an arm"),
            [arm] => Self::One(*arm),
            _ => Self::Every,
        }
    }

    /// Its arms, of the arms of `all`, in the order of the robot's model.
    fn arms<T>(self, all: &PerArm<T>) -> Vec<ArmId> {
        match self {
            Self::One(arm) => vec![arm],
            Self::Every => all.ids().collect(),
        }
    }

    /// The state of its arms, of `G` joints in all, each at the joints `at` gives it.
    fn state<T, const N: usize, const G: usize>(
        self,
        all: &PerArm<T>,
        at: impl Fn(ArmId) -> [f64; N],
    ) -> [f64; G] {
        let parts: Vec<[f64; N]> = self.arms(all).into_iter().map(at).collect();
        Arms::<N, G>::join(&parts)
    }
}

/// What a plan or a check runs on, taken at once: the planning scene and the map as they
/// stand and the stamp of the map's newest frame, the arms' last measured joint
/// positions, and the robot's own body with the grippers at their measured openings.
struct Snapshot<const N: usize, const D: usize, const K: usize> {
    scene: PlanningScene,
    voxels: VoxelMap,
    stamp: Option<SystemTime>,
    measured: ArmJoints<N>,
    own: OwnBody<N, K>,
}

impl<const N: usize, const D: usize, const K: usize> Snapshot<N, D, K> {
    /// The snapshot of `node`'s planning scene and map now, with the arms where `latest`
    /// measured them and `own` at its openings; refused while a linked camera has given the
    /// map no frame, and before the arms' joint positions are measured.
    fn of(
        node: &Node<N, D, K>,
        latest: Option<Measured<N>>,
        own: OwnBody<N, K>,
    ) -> std::result::Result<Self, Refusal> {
        let (scene, voxels, stamp) = {
            let (scene, map) = node.scene_and_map();
            framed(node, &map)?;
            (scene.clone(), map.voxels.clone(), map.stamp)
        };
        let measured = measured_arms(latest)?;
        Ok(Self {
            scene,
            voxels,
            stamp,
            measured,
            own,
        })
    }

    /// Runs `run` on the arms of `group`, `G` joints in all, as a plan or a check moves
    /// them: each arm's fingers carried by its hand at the gripper's opening and the objects
    /// attached to its grasp link carried with them, the robot's own body checked by
    /// `self_model`, the scene's other objects where they stand or where an arm the plan
    /// does not move holds them, and the pieces of the robot it does not move, which a
    /// carried object keeps clear of, each arm at its joints of `standing`. A finger that
    /// does not fit its arm, which the robot's model refuses at start, runs nothing.
    fn run<const G: usize, T>(
        &self,
        node: &Node<N, D, K>,
        group: Group,
        standing: &ArmJoints<N>,
        self_model: &dyn SelfModel<G>,
        run: impl FnOnce(&Arms<N, G>, &Scene) -> T,
    ) -> std::result::Result<T, FingerFault> {
        let models: Vec<&ArmModel<N>> = group
            .arms(&node.arms)
            .into_iter()
            .map(|arm| &node.arms[arm])
            .collect();
        let field = DistanceField::of(&self.voxels);
        let carried = models
            .iter()
            .map(|model| {
                let fingers = self.own.fingers(model.id, &model.geometry)?;
                Ok(with_held_objects(fingers, model, &self.scene))
            })
            .collect::<std::result::Result<Vec<_>, FingerFault>>()?;
        let arms = models
            .iter()
            .zip(&carried)
            .map(|(model, riders)| Arm::new(model.name, &model.geometry).riding(riders))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let arms = Arms::new(arms, Some(self_model));
        let grasp_links: Vec<&str> = models.iter().map(|model| model.grasp_link).collect();
        let objects = self
            .scene
            .for_arms(&grasp_links, |link| grasp_pose(&node.arms, standing, link));
        let mut placed = self.own.clone();
        let parts = unmoved_parts(&mut placed, standing, &models);
        let stamp_ns = self
            .stamp
            .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos() as u64);
        let scene = Scene {
            map: &self.voxels,
            field: &field,
            stamp: stamp_ns,
            objects: objects.objects(&parts),
        };
        Ok(run(&arms, &scene))
    }

    /// Where each arm stands for a request whose arms start at `starts`: an arm's start,
    /// or where it was measured.
    fn standing(&self, starts: &ArmStarts<N>) -> ArmJoints<N> {
        standing_at(&self.measured, starts)
    }
}

/// The arms' joints `latest` measured; refused before they are measured.
fn measured_arms<const N: usize>(
    latest: Option<Measured<N>>,
) -> std::result::Result<ArmJoints<N>, Refusal> {
    latest
        .map(|measured| measured.arms)
        .ok_or_else(|| Refusal::PreconditionFailed(NO_JOINTS.to_string()))
}

/// Where each arm stands for a request: at its start in `starts`, or where `measured` has
/// it.
fn standing_at<const N: usize>(measured: &ArmJoints<N>, starts: &ArmStarts<N>) -> ArmJoints<N> {
    measured.each(|arm, at| starts[arm].unwrap_or(*at))
}

/// The pieces of the robot, its own body `own` with the arms at `standing`, that a plan of
/// the arms of `moving` does not move: each other arm's with its fingers, and the body's,
/// each by the name of its link.
fn unmoved_parts<'m, const N: usize, const K: usize>(
    own: &'m mut OwnBody<N, K>,
    standing: &ArmJoints<N>,
    moving: &[&ArmModel<N>],
) -> Vec<RobotPart<'m>> {
    own.standing_pieces(standing)
        .into_iter()
        .filter(|(link, _)| {
            !moving
                .iter()
                .any(|arm| arm.link_names.iter().any(|own| own == link))
        })
        .map(|(link, piece)| RobotPart { link, piece })
        .collect()
}

/// A start tolerance for each joint of a state of `G` joints.
fn start_tolerance<const G: usize>(config: &Config) -> [f64; G] {
    [config.start_tolerance.get(); G]
}

/// The check on `snapshot` of the path of the arms of `group`, `G` joints in all, each from
/// its joints of `standing` through its waypoints of `paths`, against `self_model`, ending by
/// `deadline`.
fn check_group<const N: usize, const D: usize, const K: usize, const G: usize>(
    node: &Node<N, D, K>,
    snapshot: &Snapshot<N, D, K>,
    group: Group,
    standing: &ArmJoints<N>,
    self_model: &dyn SelfModel<G>,
    paths: &ArmPaths<N>,
    deadline: Instant,
) -> std::result::Result<std::result::Result<PathCheck, CheckFailure>, FingerFault> {
    let path_of = |arm: ArmId| {
        paths[arm]
            .as_ref()
            .expect("an arm the check moves is named")
    };
    let start = group.state(&node.arms, |arm| standing[arm]);
    let count = path_of(group.arms(&node.arms)[0]).len();
    let waypoints: Vec<[f64; G]> = (0..count)
        .map(|k| group.state(&node.arms, |arm| path_of(arm)[k]))
        .collect();
    snapshot.run(node, group, standing, self_model, |arms, scene| {
        check_path(
            arms,
            scene,
            &CheckRequest {
                start,
                start_tolerance: start_tolerance(&node.config),
                waypoints: &waypoints,
                min_clearance_m: node.config.min_clearance_m,
                deadline,
            },
        )
    })
}

async fn serve_descriptions<const N: usize, const D: usize, const K: usize>(
    runner: Arc<NodeRunner>,
    node: Arc<Node<N, D, K>>,
) {
    let token = runner.cancellation_token().clone();
    serving::serve("describe_arms", token.cancelled(), || {
        describe_arms::handle_requests(&runner, |_request| Ok(description(&node)))
    })
    .await;
}

/// Each arm's joints, limits and links, and the body's links.
fn description<const N: usize, const D: usize, const K: usize>(
    node: &Node<N, D, K>,
) -> describe_arms::Response {
    let arms = node
        .arms
        .iter()
        .map(|arm| {
            let limits = arm.geometry.chain().limits();
            describe_arms::ResponseArmsItem {
                name: arm.name.to_string(),
                gripper_name: arm.gripper_name.to_string(),
                joint_names: arm.joint_names.clone(),
                joint_lower_limits: limits.iter().map(|l| l.lo).collect(),
                joint_upper_limits: limits.iter().map(|l| l.hi).collect(),
                link_names: arm.link_names.clone(),
                grasp_link: arm.grasp_link.to_string(),
                approach_axis: arm.approach_axis.to_string(),
                closing_axis: arm.closing_axis.to_string(),
                gripper_links: arm.gripper_links.clone(),
            }
        })
        .collect();
    describe_arms::Response::new(arms, node.body.link_names().map(str::to_string).collect())
}

/// Serves clear_occupancy_map.
async fn serve_clear<const N: usize, const D: usize, const K: usize>(
    runner: Arc<NodeRunner>,
    node: Arc<Node<N, D, K>>,
) {
    let token = runner.cancellation_token().clone();
    serving::serve("clear_occupancy_map", token.cancelled(), || {
        clear_occupancy_map::handle_requests(&runner, |_request| {
            clear(&node);
            info!("occupancy map cleared");
            Ok(())
        })
    })
    .await;
}

/// Empties the map: it waits on its next frame, or on the fault that holds the camera's
/// frames back while it held none, or on a lasting fault, and takes no frame the camera
/// took at or before the newest joint positions measured now.
fn clear<const N: usize, const D: usize, const K: usize>(node: &Node<N, D, K>) {
    let cleared_at = node.history.borrow().latest_at();
    let mut map = lock(&node.map);
    map.voxels.clear();
    map.cleared_at = cleared_at;
    if map.stamp.take().is_some() {
        map.wait_on(Waiting::Passing(AWAITING_FRAME.to_string()));
    }
}

/// Serves apply_collision_objects: a call blocks its handler's thread in place while it
/// waits on the scene and the map, which a frame's intake holds, and clears the map, and
/// the runtime moves its other tasks off that thread.
async fn serve_objects<const N: usize, const D: usize, const K: usize>(
    runner: Arc<NodeRunner>,
    node: Arc<Node<N, D, K>>,
) {
    let token = runner.cancellation_token().clone();
    serving::serve("apply_collision_objects", token.cancelled(), || {
        apply_collision_objects::handle_requests(&runner, |request| {
            tokio::task::block_in_place(|| apply_objects(&node, &request.data))
                .map_err(apply_collision_objects::Failure::from)
        })
    })
    .await;
}

/// Applies apply_collision_objects' `request`: the scene applies its objects in order, all
/// or none, and clears from the map every voxel an object it places intersects.
fn apply_objects<const N: usize, const D: usize, const K: usize>(
    node: &Node<N, D, K>,
    request: &apply_collision_objects::RequestData,
) -> std::result::Result<(), Refusal> {
    let (mut scene, mut map) = node.scene_and_map();
    scene
        .apply_collision_objects(&object_requests(request), &mut map.voxels)
        .map_err(Refusal::from)
}

/// Serves apply_attached_collision_object: a call blocks its handler's thread in place as
/// apply_collision_objects' does.
async fn serve_attachments<const N: usize, const D: usize, const K: usize>(
    runner: Arc<NodeRunner>,
    node: Arc<Node<N, D, K>>,
) {
    let token = runner.cancellation_token().clone();
    serving::serve("apply_attached_collision_object", token.cancelled(), || {
        apply_attached_collision_object::handle_requests(&runner, |request| {
            tokio::task::block_in_place(|| apply_attachment(&node, &request.data))
                .map_err(apply_attached_collision_object::Failure::from)
        })
    })
    .await;
}

/// Applies apply_attached_collision_object's `request`: an object attaches to a grasp
/// link, or detaches from it, where the link stands at the arms' last measured joint
/// positions, and an attached object clears from the map every voxel it intersects.
fn apply_attachment<const N: usize, const D: usize, const K: usize>(
    node: &Node<N, D, K>,
    request: &apply_attached_collision_object::RequestData,
) -> std::result::Result<(), Refusal> {
    let measured = node.latest().map(|measured| measured.arms);
    let (mut scene, mut map) = node.scene_and_map();
    scene
        .apply_attached_collision_object(
            &attach_request(request),
            |link| {
                measured
                    .as_ref()
                    .map(|arms| grasp_pose(&node.arms, arms, link))
            },
            &mut map.voxels,
        )
        .map_err(Refusal::from)
}

/// Serves set_allowed_collision: a call blocks its handler's thread in place while it waits
/// on the scene, which a frame's intake holds.
async fn serve_allowed_pairs<const N: usize, const D: usize, const K: usize>(
    runner: Arc<NodeRunner>,
    node: Arc<Node<N, D, K>>,
) {
    let token = runner.cancellation_token().clone();
    serving::serve("set_allowed_collision", token.cancelled(), || {
        set_allowed_collision::handle_requests(&runner, |request| {
            tokio::task::block_in_place(|| allow_pairs(&node, &request.data))
                .map_err(set_allowed_collision::Failure::from)
        })
    })
    .await;
}

/// Applies set_allowed_collision's `request`: the scene lets its object touch each of its
/// other names, or stops it.
fn allow_pairs<const N: usize, const D: usize, const K: usize>(
    node: &Node<N, D, K>,
    request: &set_allowed_collision::RequestData,
) -> std::result::Result<(), Refusal> {
    lock(&node.scene)
        .set_allowed_collision(&request.name, &request.other_names, request.allowed)
        .map_err(Refusal::from)
}

/// Serves get_planning_scene: the scene, the clearance each state keeps, and the side of
/// the map's voxels. A call blocks its handler's thread in place while it waits on the
/// scene, which a frame's intake holds.
async fn serve_scene<const N: usize, const D: usize, const K: usize>(
    runner: Arc<NodeRunner>,
    node: Arc<Node<N, D, K>>,
) {
    let token = runner.cancellation_token().clone();
    serving::serve("get_planning_scene", token.cancelled(), || {
        get_planning_scene::handle_requests(&runner, |_request| {
            let snapshot = tokio::task::block_in_place(|| {
                lock(&node.scene)
                    .snapshot(node.config.min_clearance_m, node.config.voxel_size.get())
            });
            Ok(scene_response(snapshot))
        })
    })
    .await;
}

/// The instant a check asked at `asked_at` ends by.
fn check_deadline(asked_at: Instant) -> Instant {
    asked_at + CHECK_TIME
}

/// Serves check_joint_path: a request is refused when it does not parse and before the
/// planner can check; a check blocks its handler's thread in place, and the runtime
/// moves its other tasks off that thread.
async fn serve_checks<const N: usize, const D: usize, const K: usize>(
    runner: Arc<NodeRunner>,
    node: Arc<Node<N, D, K>>,
) {
    let token = runner.cancellation_token().clone();
    serving::serve("check_joint_path", token.cancelled(), || {
        check_joint_path::handle_requests(&runner, |request| {
            tokio::task::block_in_place(|| {
                check(&node, &request.data, check_deadline(Instant::now()))
            })
            .map_err(check_joint_path::Failure::from)
        })
    })
    .await;
}

/// One check of a path on the map as it stands, ending by `deadline`. An arm of the
/// request whose every waypoint is its start, inside its joint limits, stands still where
/// it starts, as a plan holds an arm it does not move, and the check moves the others;
/// when every arm stands still, it moves them all.
fn check<const N: usize, const D: usize, const K: usize>(
    node: &Node<N, D, K>,
    request: &check_joint_path::RequestData,
    deadline: Instant,
) -> std::result::Result<check_joint_path::Response, Refusal> {
    let CheckQuery { starts, waypoints } =
        parse_check(request, &node.arms).map_err(Refusal::Rejected)?;
    let latest = node.latest();
    let own = node.own_at(latest.as_ref());
    let snapshot = Snapshot::of(node, latest, own)?;
    let standing = snapshot.standing(&starts);
    let paths: ArmPaths<N> = node.arms.each(|arm, _| waypoints.of(arm));
    let still = |arm: ArmId| {
        let limits = node.arms[arm].geometry.chain().limits();
        paths[arm].as_ref().is_some_and(|path| {
            path.iter().all(|q| {
                *q == standing[arm] && (0..N).all(|j| (limits[j].lo..=limits[j].hi).contains(&q[j]))
            })
        })
    };
    let named: Vec<ArmId> = paths.ids().filter(|arm| paths[*arm].is_some()).collect();
    let moved: Vec<ArmId> = named.iter().copied().filter(|arm| !still(*arm)).collect();
    let group = Group::of(if moved.is_empty() { &named } else { &moved });
    let checked = match group {
        Group::One(arm) => {
            let self_model = snapshot.own.of_one_arm(NonZeroUsize::MIN, arm, &standing);
            check_group::<N, D, K, N>(
                node,
                &snapshot,
                group,
                &standing,
                &self_model,
                &paths,
                deadline,
            )
        }
        Group::Every => {
            let self_model = snapshot.own.of_every_arm(NonZeroUsize::MIN);
            check_group::<N, D, K, D>(
                node,
                &snapshot,
                group,
                &standing,
                &self_model,
                &paths,
                deadline,
            )
        }
    }
    .map_err(|fault| Refusal::AttemptFailed(fault.to_string()))?
    .map_err(|failure| match failure {
        CheckFailure::Request(e) => Refusal::Rejected(e.to_string()),
        CheckFailure::TimeEnded => Refusal::TimedOut(format!(
            "the check of the path ran past {CHECK_TIME:?}: send a path of fewer waypoints"
        )),
    })?;
    Ok(check_joint_path::Response::new(
        checked.fault.is_none(),
        u32::try_from(checked.clear_waypoints).expect("as many waypoints as a request carries"),
        checked
            .fault
            .map_or_else(String::new, |fault| fault.to_string()),
        snapshot.stamp,
    ))
}

/// Gives `$serve`, which serves the action `$member`: `$admit` refuses a goal that does
/// not parse and one the planner cannot take yet, and each goal it takes runs by `$run` on
/// its own task, on what its admission took.
macro_rules! serve_action {
    ($serve:ident, $member:ident, $admit:ident, $run:ident) => {
        async fn $serve<const N: usize, const D: usize, const K: usize>(
            runner: Arc<NodeRunner>,
            node: Arc<Node<N, D, K>>,
        ) {
            let token = runner.cancellation_token().clone();
            let mut action = match $member::ActionHandle::expose(&runner).await {
                Ok(action) => action,
                Err(e) => return error!("{} expose: {e}", stringify!($member)),
            };
            loop {
                // What the decider takes of the goal it accepts.
                let admitted = Mutex::new(None);
                let next = tokio::select! {
                    _ = token.cancelled() => return,
                    next = action.handle_goal_next_request(|goal| {
                        Ok(match $admit(&node, &goal.data) {
                            Ok(taken) => {
                                *lock(&admitted) = Some(taken);
                                $member::GoalDecision::accept()
                            }
                            Err(refusal) => {
                                let failure = $member::Failure::from(refusal);
                                $member::GoalDecision::refuse(failure.kind, failure.message)
                            }
                        })
                    }) => next,
                };
                match next {
                    Ok(Some(context)) => {
                        let taken = lock(&admitted)
                            .take()
                            .expect("an accepted goal was admitted");
                        tokio::spawn($run(node.clone(), context, taken));
                    }
                    Ok(None) => return,
                    Err(e) => {
                        error!("{}: {e}", stringify!($member));
                        tokio::time::sleep(RETRY).await;
                    }
                }
            }
        }
    };
}

serve_action!(serve_plans, plan_joint_path, admit, run_plan);
serve_action!(
    serve_cartesian_paths,
    plan_cartesian_path,
    admit_cartesian,
    run_cartesian
);

/// A goal the planner takes: its query, the snapshot it plans on, where each arm stands
/// for it, each arm the goal names at its start moved inside its joint limits, and the
/// instant its plan ends by, its time limit after it is taken.
struct Taken<const N: usize, const D: usize, const K: usize> {
    query: PlanQuery<N>,
    snapshot: Snapshot<N, D, K>,
    standing: ArmJoints<N>,
    deadline: Instant,
}

/// The goal the planner takes: refused with rejected when it does not parse, and with
/// precondition_failed before the planner can plan and when the start of an arm it names
/// lies outside a joint limit by more than the start tolerance.
fn admit<const N: usize, const D: usize, const K: usize>(
    node: &Node<N, D, K>,
    goal: &GoalRequestData,
) -> std::result::Result<Taken<N, D, K>, Refusal> {
    let latest = node.latest();
    let own = node.own_at(latest.as_ref());
    let fingers = riders_of_the_arms(&own, &node.arms)
        .map_err(|e| Refusal::AttemptFailed(format!("the robot's fingers: {e}")))?;
    let query = parse_plan(goal, &node.arms, &fingers).map_err(Refusal::Rejected)?;
    let snapshot = Snapshot::of(node, latest, own)?;
    let standing = inside_limits(node, snapshot.standing(&query.starts), &query.order)?;
    let deadline = Instant::now() + query.time_limit;
    Ok(Taken {
        query,
        snapshot,
        standing,
        deadline,
    })
}

/// `standing` with each arm of `moving` at its start moved inside its joint limits; refused
/// with precondition_failed when one lies outside a limit by more than the start tolerance.
fn inside_limits<const N: usize, const D: usize, const K: usize>(
    node: &Node<N, D, K>,
    standing: ArmJoints<N>,
    moving: &[ArmId],
) -> std::result::Result<ArmJoints<N>, Refusal> {
    moving.iter().try_fold(standing, |mut standing, arm| {
        standing[*arm] = start_inside_limits(
            node.arms[*arm].geometry.chain(),
            &standing[*arm],
            &start_tolerance(&node.config),
        )
        .map_err(|fault| Refusal::PreconditionFailed(fault.to_string()))?;
        Ok(standing)
    })
}

/// Gives `$run`, which computes one goal the planner took of the action `$member` by `$compute`
/// and answers it, and `$end_cancelled`: a cancel stops the computation, `$what` to a message,
/// at any part of it, and ends the goal cancelled, whatever it answered after, with the reason
/// the cancel gave and `$empty` as its result. `$field`s are the result `$compute` gives.
macro_rules! run_action {
    (
        $run:ident, $end_cancelled:ident, $member:ident, $taken:ty, $compute:ident, $what:literal,
        ($($field:ident),+), ($($empty:expr),+)
    ) => {
        async fn $run<const N: usize, const D: usize, const K: usize>(
            node: Arc<Node<N, D, K>>,
            context: $member::GoalContext,
            taken: $taken,
        ) {
            let stop = Arc::new(AtomicBool::new(false));
            let computed = {
                let node = node.clone();
                let stop = stop.clone();
                tokio::task::spawn_blocking(move || $compute(&node, &taken, &stop))
            };
            let answer = answer_or_stop(computed, context.cancel_signal(), &stop).await;
            let ended = match answer {
                Err(e) => {
                    context
                        .fail(
                            $member::FailureKind::AttemptFailed,
                            format!("the {} task failed: {e}", $what),
                        )
                        .await
                }
                Ok(Err(PlanEnd::Stopped)) => $end_cancelled(&context).await,
                Ok(_) if context.is_cancelled() => $end_cancelled(&context).await,
                Ok(Err(PlanEnd::Failed(kind, message))) => context.fail(kind, message).await,
                Ok(Ok(($($field),+))) => context.complete($($field),+).await,
            };
            if let Err(e) = ended {
                error!("{} answer: {e}", stringify!($member));
            }
        }

        async fn $end_cancelled(context: &$member::GoalContext) -> Result<()> {
            let reason = context.cancel_cause().map_or_else(
                || format!("the {} was stopped", $what),
                |c| c.reason().to_string(),
            );
            context.complete_cancelled(reason, $($empty),+).await
        }
    };
}

run_action!(
    run_plan,
    end_cancelled,
    plan_joint_path,
    Taken<N, D, K>,
    plan,
    "plan",
    (arms, stamp),
    (Vec::new(), None)
);
run_action!(
    run_cartesian,
    end_cartesian_cancelled,
    plan_cartesian_path,
    TakenCartesian<N, D, K>,
    cartesian,
    "path",
    (start_joint_positions, waypoints, fraction, stamp),
    (Vec::new(), Vec::new(), 0.0, None)
);

/// The answer of `planned`. A `cancel` that comes first sets `stop`, which ends the plan
/// at any part of it, and the answer is the plan's after it.
async fn answer_or_stop<T>(
    planned: impl Future<Output = T>,
    cancel: impl Future<Output = ()>,
    stop: &AtomicBool,
) -> T {
    tokio::pin!(planned);
    tokio::select! {
        answer = &mut planned => answer,
        () = cancel => {
            stop.store(true, Ordering::SeqCst);
            planned.await
        }
    }
}

/// How a plan or a Cartesian path ends without a path: `Kind` is its member's failure kind.
#[derive(Debug)]
enum PlanEnd<Kind> {
    /// A cancel stopped it.
    Stopped,
    Failed(Kind, String),
}

/// One plan of a goal the planner took, on its snapshot and by its deadline: each arm the
/// goal names, in its order, with its waypoints from the start, an arm the plan does not
/// move at its start as many times as the others' waypoints; and the stamp of the map's
/// newest frame. The attempts that run at once ask the self model from their threads, each
/// on a copy no other holds.
fn plan<const N: usize, const D: usize, const K: usize>(
    node: &Node<N, D, K>,
    taken: &Taken<N, D, K>,
    stop: &AtomicBool,
) -> std::result::Result<(Vec<PlanJointPathResultArmsItem>, Option<SystemTime>), PlanEnd<FailureKind>>
{
    let Taken {
        query,
        snapshot,
        standing,
        ..
    } = taken;
    let copies = self_model_copies(node.config.planning_attempts);
    let planned = match &query.arm_goals {
        ArmGoals::One { arm, goal } => {
            if let Some(fault) = standing_arms_fault(node, taken, *arm) {
                return Err(PlanEnd::Failed(FailureKind::NoPath, fault));
            }
            let self_model = snapshot.own.of_one_arm(copies, *arm, standing);
            plan_group::<N, D, K, N>(node, taken, Group::One(*arm), &self_model, &|_| goal, stop)
        }
        ArmGoals::Every(goals) => {
            let self_model = snapshot.own.of_every_arm(copies);
            plan_group::<N, D, K, D>(
                node,
                taken,
                Group::Every,
                &self_model,
                &|arm| &goals[arm],
                stop,
            )
        }
    }
    .map_err(|fault| PlanEnd::Failed(FailureKind::AttemptFailed, fault.to_string()))?;
    planned
        .map(|paths| {
            let count = paths
                .iter()
                .flatten()
                .map(Vec::len)
                .next()
                .expect("the plan moves an arm");
            let arms = query
                .order
                .iter()
                .map(|arm| {
                    let waypoints = paths[*arm]
                        .clone()
                        .unwrap_or_else(|| vec![standing[*arm]; count]);
                    result_arm(&node.arms[*arm], &waypoints)
                })
                .collect();
            (arms, snapshot.stamp)
        })
        .map_err(|failure| {
            let kind = match &failure {
                PlanFailure::Stopped => return PlanEnd::Stopped,
                PlanFailure::Request(_) => FailureKind::Rejected,
                PlanFailure::Start(StartFault::OutsideLimits { .. }) => {
                    FailureKind::PreconditionFailed
                }
                PlanFailure::Start(StartFault::InCollision(_)) => FailureKind::StartInCollision,
                PlanFailure::Unreachable(_) => FailureKind::Unreachable,
                PlanFailure::NoPath(_) => FailureKind::NoPath,
                PlanFailure::Planner(_) => FailureKind::AttemptFailed,
            };
            PlanEnd::Failed(kind, failure.to_string())
        })
}

/// Why an arm the plan leaves where it starts, every arm but `moving`, breaks a path
/// orientation of its links; none where each such arm's start holds its orientations.
fn standing_arms_fault<const N: usize, const D: usize, const K: usize>(
    node: &Node<N, D, K>,
    taken: &Taken<N, D, K>,
    moving: ArmId,
) -> Option<String> {
    let paths = &taken.query.path_orientations;
    paths.ids().filter(|arm| *arm != moving).find_map(|arm| {
        let chain = node.arms[arm].geometry.chain();
        paths[arm].iter().find_map(|held| {
            let error = held.target.error_at(chain, &taken.standing[arm]);
            let tolerance = held.target.tolerance_rad.get();
            (error > tolerance).then(|| {
                format!(
                    "the start holds {} {error:.4} rad from its path orientation, past its \
                     tolerance_rad {tolerance:.4}, and its arm stays where it starts; send a start \
                     within it or a wider tolerance_rad",
                    held.link_name
                )
            })
        })
    })
}

/// The plan of the arms of `group`, `G` joints in all, which `taken` takes, each from its
/// joints of where they stand to the goal `goal_of` gives it, against `self_model`: each
/// arm's waypoints from its start, none for an arm the plan does not move.
fn plan_group<'g, const N: usize, const D: usize, const K: usize, const G: usize>(
    node: &Node<N, D, K>,
    taken: &Taken<N, D, K>,
    group: Group,
    self_model: &dyn SelfModel<G>,
    goal_of: &dyn Fn(ArmId) -> &'g Goal,
    stop: &AtomicBool,
) -> std::result::Result<std::result::Result<ArmPaths<N>, PlanFailure>, FingerFault> {
    let moving = group.arms(&node.arms);
    let samplers: Vec<IkGoalSampler<N>> = moving
        .iter()
        .map(|arm| IkGoalSampler::new(node.arms[*arm].geometry.chain().clone()))
        .collect();
    let goals = moving
        .iter()
        .zip(&samplers)
        .map(|(arm, sampler)| ArmGoal {
            goal: goal_of(*arm),
            sampler,
        })
        .collect();
    let path_orientations: Vec<PathOrientation> = moving
        .iter()
        .enumerate()
        .flat_map(|(index, arm)| {
            taken.query.path_orientations[*arm]
                .iter()
                .map(move |held| PathOrientation {
                    arm: index,
                    link_name: &held.link_name,
                    target: held.target,
                })
        })
        .collect();
    let start = group.state(&node.arms, |arm| taken.standing[arm]);
    taken
        .snapshot
        .run(node, group, &taken.standing, self_model, |arms, scene| {
            let request = plan_request(
                &node.config,
                start,
                goals,
                &path_orientations,
                taken.deadline,
                stop,
            );
            plan_path(arms, scene, &request).map(|path| split(&path, group, &node.arms))
        })
}

/// The request of a plan of the arms of `goals` from `start`, holding `path_orientations`,
/// by `deadline`, which `stop` ends.
fn plan_request<'a, const N: usize, const G: usize>(
    config: &Config,
    start: [f64; G],
    goals: Vec<ArmGoal<'a, N>>,
    path_orientations: &'a [PathOrientation<'a>],
    deadline: Instant,
    stop: &'a AtomicBool,
) -> PlanRequest<'a, N, G> {
    PlanRequest {
        start,
        start_tolerance: start_tolerance(config),
        goals,
        path_orientations,
        goal_samples: config.goal_samples,
        planner: Planner::RrtConnect,
        deadline,
        max_iterations: config.max_iterations,
        attempts: config.planning_attempts,
        seed: PLAN_SEED,
        min_clearance_m: config.min_clearance_m,
        stop: Some(stop),
    }
}

/// Each arm's waypoints from its start, none for an arm a plan does not move.
type ArmPaths<const N: usize> = PerArm<Option<Vec<[f64; N]>>>;

/// Each arm of `all`'s waypoints of `path`, a path of the arms of `group`, `G` joints in all;
/// none for an arm it does not move.
fn split<T, const N: usize, const G: usize>(
    path: &[[f64; G]],
    group: Group,
    all: &PerArm<T>,
) -> ArmPaths<N> {
    let moving = group.arms(all);
    all.each(|arm, _| {
        let k = moving.iter().position(|named| *named == arm)?;
        Some(path.iter().map(|q| Arms::<N, G>::slice(q, k)).collect())
    })
}

/// A Cartesian path the planner takes: its query, where each arm stands for it, the arm it
/// moves at its start moved inside its joint limits, and the snapshot its states keep clear
/// of, where it checks collisions.
struct TakenCartesian<const N: usize, const D: usize, const K: usize> {
    query: CartesianQuery<N>,
    standing: ArmJoints<N>,
    snapshot: Option<Snapshot<N, D, K>>,
}

/// The Cartesian path the planner takes: refused with rejected when it does not parse, and
/// with precondition_failed before the arms are measured, where it checks collisions before
/// the planner's depth frame, and when its start lies outside a joint limit by more than the
/// start tolerance.
fn admit_cartesian<const N: usize, const D: usize, const K: usize>(
    node: &Node<N, D, K>,
    goal: &plan_cartesian_path::GoalRequestData,
) -> std::result::Result<TakenCartesian<N, D, K>, Refusal> {
    let latest = node.latest();
    let own = node.own_at(latest.as_ref());
    let fingers = riders_of_the_arms(&own, &node.arms)
        .map_err(|e| Refusal::AttemptFailed(format!("the robot's fingers: {e}")))?;
    let query = parse_cartesian(goal, &node.arms, &fingers).map_err(Refusal::Rejected)?;
    let (standing, snapshot) = if query.avoid_collisions {
        let snapshot = Snapshot::of(node, latest, own)?;
        (snapshot.standing(&query.starts), Some(snapshot))
    } else {
        (standing_at(&measured_arms(latest)?, &query.starts), None)
    };
    Ok(TakenCartesian {
        standing: inside_limits(node, standing, &[query.arm])?,
        query,
        snapshot,
    })
}

/// The request of the Cartesian path `taken`, its states keeping clear of `collisions`.
fn cartesian_request<'a, const N: usize, const D: usize, const K: usize>(
    config: &Config,
    taken: &'a TakenCartesian<N, D, K>,
    collisions: Collisions<'a>,
    stop: &'a AtomicBool,
) -> CartesianRequest<'a, N> {
    CartesianRequest {
        start: taken.standing[taken.query.arm],
        start_tolerance: start_tolerance(config),
        frame: taken.query.frame,
        waypoints: &taken.query.poses,
        collisions,
        seed: u64::from(PLAN_SEED),
        stop: Some(stop),
    }
}

/// The result of a Cartesian path, in the order of plan_cartesian_path's: where it starts, its
/// waypoints, the fraction covered, and the stamp of the map's newest frame it kept clear of.
type CartesianResult = (
    Vec<f64>,
    Vec<plan_cartesian_path::PlanCartesianPathResultWaypointsItem>,
    f64,
    Option<SystemTime>,
);

/// One Cartesian path of a goal the planner took, keeping clear of its snapshot where it
/// checks collisions: each waypoint with the grasp-point pose there, as limb_state reports
/// it, and the stamp of the map's newest frame it kept clear of.
fn cartesian<const N: usize, const D: usize, const K: usize>(
    node: &Node<N, D, K>,
    taken: &TakenCartesian<N, D, K>,
    stop: &AtomicBool,
) -> std::result::Result<CartesianResult, PlanEnd<plan_cartesian_path::FailureKind>> {
    use plan_cartesian_path::FailureKind as Kind;
    let TakenCartesian {
        query,
        standing,
        snapshot,
    } = taken;
    let id = query.arm;
    let model = &node.arms[id];
    let path = match snapshot {
        Some(snapshot) => {
            let self_model = snapshot.own.of_one_arm(NonZeroUsize::MIN, id, standing);
            snapshot
                .run::<N, _>(
                    node,
                    Group::One(id),
                    standing,
                    &self_model,
                    |arms, scene| {
                        let collisions = Collisions::Checked {
                            scene,
                            min_clearance_m: node.config.min_clearance_m,
                        };
                        cartesian_path(
                            arms,
                            &cartesian_request(&node.config, taken, collisions, stop),
                        )
                    },
                )
                .map_err(|fault| PlanEnd::Failed(Kind::AttemptFailed, fault.to_string()))?
        }
        None => {
            let arms = Arms::new(vec![Arm::new(model.name, &model.geometry)], None);
            let unchecked = cartesian_request(&node.config, taken, Collisions::Unchecked, stop);
            cartesian_path(&arms, &unchecked)
        }
    };
    let path = path.map_err(|failure| {
        let kind = match &failure {
            CartesianFailure::Stopped => return PlanEnd::Stopped,
            CartesianFailure::Request(_) => Kind::Rejected,
            CartesianFailure::Start(StartFault::OutsideLimits { .. }) => Kind::PreconditionFailed,
            CartesianFailure::Start(StartFault::InCollision(_)) => Kind::StartInCollision,
        };
        PlanEnd::Failed(kind, failure.to_string())
    })?;
    let waypoints = path
        .waypoints
        .iter()
        .map(|q| {
            let (position, orientation) = pose_parts(&model.geometry.chain().world_ee_pose(q));
            plan_cartesian_path::PlanCartesianPathResultWaypointsItem {
                joint_positions: q.to_vec(),
                position,
                orientation,
            }
        })
        .collect();
    let stamp = snapshot.as_ref().and_then(|snapshot| snapshot.stamp);
    Ok((path.start.to_vec(), waypoints, path.fraction, stamp))
}

/// The copies of the self model a plan of `attempts` attempts keeps: one for each attempt
/// that runs at once, at most `PLANNING_THREADS`.
fn self_model_copies(attempts: NonZeroU32) -> NonZeroUsize {
    NonZeroUsize::try_from(attempts.min(PLANNING_THREADS)).expect("a u32 count fits a usize")
}

/// The answer of `arm`'s planned `waypoints`: where the path starts, and each waypoint
/// after it with the grasp-point pose there, as limb_state reports it. A path that
/// stays where it starts has the start as its one waypoint.
fn result_arm<const N: usize>(
    arm: &ArmModel<N>,
    waypoints: &[[f64; N]],
) -> PlanJointPathResultArmsItem {
    let (start, after) = waypoints
        .split_first()
        .expect("a planned path has its start");
    let after = if after.is_empty() {
        std::slice::from_ref(start)
    } else {
        after
    };
    PlanJointPathResultArmsItem {
        arm_name: arm.name.to_string(),
        start_joint_positions: start.to_vec(),
        waypoints: after
            .iter()
            .map(|q| {
                let pose = arm.geometry.chain().world_ee_pose(q);
                let rotation = pose.rotation;
                PlanJointPathResultArmsItemWaypointsItem {
                    joint_positions: q.to_vec(),
                    position: pose.translation.vector.into(),
                    orientation: [rotation.i, rotation.j, rotation.k, rotation.w],
                }
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::{Node as ANode, *};

    use openarm_description::{ARM_DOF, DUAL_DOF, Side};

    /// The node of OpenArm v2, which these tests run.
    type Node = ANode<ARM_DOF, DUAL_DOF, 2>;

    use std::task::{Context, Poll, Waker};

    use chain_kinematics::nalgebra::{Point3, UnitQuaternion, Vector3};
    use collision_geometry::chain::{Padding, RiderBody};
    use motion_planning_core::OCCUPANCY_MAP;
    use occupancy_map::Voxel;
    use openarm_description::ready;
    use peppygen::exposed_actions::motion_planning::plan_joint_path::{
        GoalArmsItem, GoalGoalJointConstraintsItem,
    };
    use peppygen::exposed_services::motion_planning::apply_collision_objects::{
        ApplyCollisionObjectsObjectsItem, ApplyCollisionObjectsObjectsItemShapesItem,
    };
    use peppygen::exposed_services::motion_planning::check_joint_path::{
        CheckJointPathArmsItem, CheckJointPathArmsItemWaypointsItem,
    };

    use crate::config::tests::parameters;

    /// The node for an OpenArm v2 from `params`, the camera linked or not.
    fn started(params: &Parameters, camera_linked: bool) -> Node {
        let config = Config::parse(params).expect("a config");
        let model = openarm_v2().expect("OpenArm v2's model");
        start(config, Robot::OpenArmV2, model, camera_linked).expect("the node starts")
    }

    /// Both arms at Ready, both grippers open.
    fn still() -> Measured<ARM_DOF> {
        Measured {
            arms: PerArm::both(ready(Side::Left), ready(Side::Right)),
            openings: PerArm::both(1.0, 1.0),
        }
    }

    fn snapshot(node: &Node) -> std::result::Result<(), Refusal> {
        let latest = node.latest();
        let own = node.own_at(latest.as_ref());
        Snapshot::of(node, latest, own).map(|_| ())
    }

    // A check ends 10 s after it is asked. One whose time has ended fails with timeout,
    // saying what to send; one with an hour left answers.
    #[test]
    fn a_check_ends_by_its_time() {
        let asked_at = Instant::now();
        assert_eq!(check_deadline(asked_at) - asked_at, Duration::from_secs(10));
        let node = started(&parameters(), false);
        measure_at(&node, SystemTime::UNIX_EPOCH, still());
        let mut turned = ready(Side::Left);
        turned[0] += 0.1;
        let request = path_of(Side::Left, &[turned]);
        let late = check_joint_path::Failure::from(
            check(&node, &request, Instant::now()).expect_err("a check with no time left"),
        );
        assert_eq!(late.kind, check_joint_path::FailureKind::Timeout);
        assert_eq!(
            late.message,
            "the check of the path ran past 10s: send a path of fewer waypoints"
        );
        assert!(check(&node, &request, an_hour_on()).is_ok());
    }

    /// An instant an hour after now, a deadline no test's check or plan reaches.
    fn an_hour_on() -> Instant {
        Instant::now() + Duration::from_secs(3600)
    }

    /// The check of the path of the arm on `side` from where it stands through
    /// `waypoints`.
    fn path_of(side: Side, waypoints: &[[f64; ARM_DOF]]) -> check_joint_path::RequestData {
        check_joint_path::RequestData {
            arms: vec![CheckJointPathArmsItem {
                arm_name: side.arm_name().to_string(),
                start_joint_positions: Vec::new(),
                waypoints: waypoints
                    .iter()
                    .map(|q| CheckJointPathArmsItemWaypointsItem {
                        joint_positions: q.to_vec(),
                    })
                    .collect(),
            }],
        }
    }

    /// A goal for the arm on `side` of `node`, from where it stands to every joint within
    /// 0.01 of `q`, in the longest time a plan may take.
    fn joint_goal(node: &Node, side: Side, q: &[f64; ARM_DOF]) -> GoalRequestData {
        GoalRequestData {
            arms: vec![GoalArmsItem {
                arm_name: side.arm_name().to_string(),
                start_joint_positions: Vec::new(),
            }],
            goal_joint_constraints: node.arms[side]
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
            objective: crate::request::OBJECTIVE.to_string(),
            allowed_planning_time_s: crate::request::MAX_PLANNING_TIME_S,
        }
    }

    /// The arm on `side` at Ready with its elbow at `angle`.
    fn elbow_at(side: Side, angle: f64) -> [f64; ARM_DOF] {
        let mut q = ready(side);
        q[3] = angle;
        q
    }

    // The right arm is checked and planned at its own joint positions, the left arm held
    // at its own: from both arms at Ready, the right elbow's bend passes its check and
    // plans. The hands touch with each arm at the other arm's Ready joint positions.
    #[test]
    fn the_right_arm_checks_and_plans_at_its_own_joint_positions() {
        let node = started(&parameters(), false);
        measure_at(&node, SystemTime::UNIX_EPOCH, still());
        let bent = elbow_at(Side::Right, 1.3);
        let checked =
            check(&node, &path_of(Side::Right, &[bent]), an_hour_on()).expect("an answer");
        assert!(
            checked.clear && checked.clear_waypoints == 1,
            "the right elbow's bend: {checked:?}"
        );
        let taken =
            admit(&node, &joint_goal(&node, Side::Right, &bent)).expect("the goal is taken");
        let (arms, _) =
            plan(&node, &taken, &AtomicBool::new(false)).expect("a path of the right arm");
        let [right] = arms.as_slice() else {
            panic!("the answer holds the one arm the goal names: {arms:?}");
        };
        assert_eq!(right.arm_name, "right_arm");
        assert_eq!(right.start_joint_positions, ready(Side::Right).to_vec());
        let last = &right.waypoints.last().expect("a waypoint").joint_positions;
        assert!(
            (0..ARM_DOF).all(|j| (last[j] - bent[j]).abs() <= 0.01 + 1e-9),
            "the path ends at {last:?}, not within 0.01 of {bent:?}"
        );
    }

    // The other arm stands where it was measured. The left hand's reach across to the
    // middle passes with the right arm at Ready; with the right shoulder measured swung
    // in, 0.1 rad in place of Ready's 0.4, the right forearm stands in the way, and the
    // path fails on its run from the start, through the robot's own body alone, the
    // map empty.
    #[test]
    fn a_path_keeps_clear_of_the_other_arm_where_it_was_measured() {
        let node = started(&parameters(), false);
        measure_at(&node, after(0), still());
        let across = path_of(Side::Left, &[[-0.35, -0.2, 1.38, 1.3, 0.0, 0.0, 0.0]]);
        let passing = check(&node, &across, an_hour_on()).expect("an answer");
        assert!(
            passing.clear && passing.clear_waypoints == 1,
            "the reach across with the right arm at Ready: {passing:?}"
        );
        let mut swung_in = ready(Side::Right);
        swung_in[1] = 0.1;
        let measured = Measured {
            arms: PerArm::both(ready(Side::Left), swung_in),
            ..still()
        };
        measure_at(&node, after(100), measured);
        let blocked = check(&node, &across, an_hour_on()).expect("an answer");
        assert!(
            !blocked.clear
                && blocked.clear_waypoints == 0
                && blocked.reason.starts_with("on the run from the start, ")
                && blocked
                    .reason
                    .contains("the robot's own links and what they carry come"),
            "the reach across with the right shoulder swung in: {blocked:?}"
        );
    }

    /// A check of the arms of `paths`, in their order, each through its waypoints from
    /// where it stands.
    fn paths_of(paths: &[(Side, &[[f64; ARM_DOF]])]) -> check_joint_path::RequestData {
        check_joint_path::RequestData {
            arms: paths
                .iter()
                .flat_map(|(side, waypoints)| path_of(*side, waypoints).arms)
                .collect(),
        }
    }

    /// A goal of the arms of `goals`, in their order, each to its joints, and of every arm
    /// of `held` with no constraint.
    fn goal_of(node: &Node, goals: &[(Side, [f64; ARM_DOF])], held: &[Side]) -> GoalRequestData {
        let mut goal = joint_goal(node, goals[0].0, &goals[0].1);
        for (side, q) in &goals[1..] {
            let other = joint_goal(node, *side, q);
            goal.arms.extend(other.arms);
            goal.goal_joint_constraints
                .extend(other.goal_joint_constraints);
        }
        goal.arms.extend(held.iter().map(|side| GoalArmsItem {
            arm_name: side.arm_name().to_string(),
            start_joint_positions: Vec::new(),
        }));
        goal
    }

    /// Whether `arm` of a plan's answer ends within 0.01 of `q` in every joint.
    fn ends_at(arm: &PlanJointPathResultArmsItem, q: &[f64; ARM_DOF]) -> bool {
        let last = &arm.waypoints.last().expect("a waypoint").joint_positions;
        (0..ARM_DOF).all(|j| (last[j] - q[j]).abs() <= 0.01 + 1e-9)
    }

    // A goal of both arms, the right named first, plans both together: the answer holds
    // the right arm and then the left, each from Ready to its goal, with as many
    // waypoints as the other.
    #[test]
    fn a_plan_of_both_arms_answers_them_in_the_goals_order() {
        let node = started(&parameters(), false);
        measure_at(&node, SystemTime::UNIX_EPOCH, still());
        let (left, right) = (elbow_at(Side::Left, 1.3), elbow_at(Side::Right, 1.3));
        let taken = admit(
            &node,
            &goal_of(&node, &[(Side::Right, right), (Side::Left, left)], &[]),
        )
        .expect("the goal is taken");
        let (arms, _) = plan(&node, &taken, &AtomicBool::new(false)).expect("a path of both");
        let names: Vec<&str> = arms.iter().map(|arm| arm.arm_name.as_str()).collect();
        assert_eq!(names, ["right_arm", "left_arm"]);
        assert_eq!(arms[0].waypoints.len(), arms[1].waypoints.len());
        assert_eq!(arms[0].start_joint_positions, ready(Side::Right).to_vec());
        assert_eq!(arms[1].start_joint_positions, ready(Side::Left).to_vec());
        assert!(
            ends_at(&arms[0], &right) && ends_at(&arms[1], &left),
            "{arms:?}"
        );
    }

    // An arm the goal names with no constraint stays where it starts: each of its
    // waypoints is its start, as many as the moving arm's.
    #[test]
    fn an_arm_with_no_constraint_holds_its_start_at_every_waypoint() {
        let node = started(&parameters(), false);
        measure_at(&node, SystemTime::UNIX_EPOCH, still());
        let right = elbow_at(Side::Right, 1.3);
        let taken = admit(
            &node,
            &goal_of(&node, &[(Side::Right, right)], &[Side::Left]),
        )
        .expect("the goal is taken");
        let (arms, _) = plan(&node, &taken, &AtomicBool::new(false)).expect("a path");
        let [moving, held] = arms.as_slice() else {
            panic!("the answer holds both arms: {arms:?}");
        };
        assert_eq!(held.arm_name, "left_arm");
        assert_eq!(moving.waypoints.len(), held.waypoints.len());
        assert!(
            held.waypoints
                .iter()
                .all(|w| w.joint_positions == ready(Side::Left).to_vec()),
            "{held:?}"
        );
        assert!(ends_at(moving, &right), "{moving:?}");
    }

    // A check of both arms moves the arms whose waypoints leave where they stand: the
    // right elbow's bend beside a left arm that stays at Ready passes, and so do both
    // elbows' bends at once; each arm swung to the other's Ready brings the hands together,
    // and the check fails on the run from the start, through the robot's own body.
    #[test]
    fn a_check_of_both_arms_keeps_them_apart() {
        let node = started(&parameters(), false);
        measure_at(&node, SystemTime::UNIX_EPOCH, still());
        let (left, right) = (elbow_at(Side::Left, 1.3), elbow_at(Side::Right, 1.3));
        let beside = check(
            &node,
            &paths_of(&[(Side::Left, &[ready(Side::Left)]), (Side::Right, &[right])]),
            an_hour_on(),
        )
        .expect("an answer");
        assert!(beside.clear && beside.clear_waypoints == 1, "{beside:?}");
        let both = check(
            &node,
            &paths_of(&[(Side::Right, &[right]), (Side::Left, &[left])]),
            an_hour_on(),
        )
        .expect("an answer");
        assert!(both.clear && both.clear_waypoints == 1, "{both:?}");
        let crossed = check(
            &node,
            &paths_of(&[
                (Side::Left, &[ready(Side::Right)]),
                (Side::Right, &[ready(Side::Left)]),
            ]),
            an_hour_on(),
        )
        .expect("an answer");
        assert!(
            !crossed.clear
                && crossed.clear_waypoints == 0
                && crossed
                    .reason
                    .contains("the robot's own links and what they carry come"),
            "{crossed:?}"
        );
    }

    // An arm whose every waypoint is its start stands still beside an arm that moves: its
    // pairs with the torso are not checked. With its third joint turned up from Ready until
    // its gripper stands 2 cm from the torso, half the clearance, the left arm standing
    // there beside the right elbow's bend passes; checked alone, every arm standing still,
    // it moves and falls short of the clearance at its waypoint.
    #[test]
    fn an_arm_that_stays_at_its_start_beside_a_moving_one_is_not_checked() {
        let node = started(&parameters(), false);
        measure_at(&node, SystemTime::UNIX_EPOCH, still());
        let mut model = node.own.model.clone();
        let right = ready(Side::Right);
        let raised = |turn: f64| {
            let mut q = ready(Side::Left);
            q[2] = turn;
            q
        };
        let mut gripper_to_torso = |turn: f64| -> f64 {
            model
                .min_distance_of_chain(Side::Left.index(), &[raised(turn), right])
                .expect("a finite configuration")
                .distance
        };
        let (mut apart, mut near) = (
            ready(Side::Left)[2],
            node.arms[Side::Left].geometry.chain().limits()[2].hi,
        );
        for _ in 0..60 {
            let middle = (apart + near) / 2.0;
            if gripper_to_torso(middle) > 0.02 {
                apart = middle;
            } else {
                near = middle;
            }
        }
        let left = raised(apart);
        let nearest = model
            .min_distance_of_chain(Side::Left.index(), &[left, right])
            .expect("a finite configuration");
        assert_eq!(
            (nearest.link_a, nearest.link_b),
            ("openarm_body_link0", "openarm_left_ee_link1"),
            "the gripper meets the torso"
        );
        let held_by = |side: Side, start: [f64; ARM_DOF], waypoint: [f64; ARM_DOF]| {
            let mut path = path_of(side, &[waypoint]).arms;
            path[0].start_joint_positions = start.to_vec();
            path
        };
        let beside = check(
            &node,
            &check_joint_path::RequestData {
                arms: [
                    held_by(Side::Left, left, left),
                    held_by(Side::Right, right, elbow_at(Side::Right, 1.3)),
                ]
                .concat(),
            },
            an_hour_on(),
        )
        .expect("an answer");
        assert!(beside.clear && beside.clear_waypoints == 1, "{beside:?}");
        let alone = check(
            &node,
            &check_joint_path::RequestData {
                arms: held_by(Side::Left, left, left),
            },
            an_hour_on(),
        )
        .expect("an answer");
        assert!(
            !alone.clear && alone.reason.starts_with("at waypoint 1, "),
            "{alone:?}"
        );
    }

    // A plan keeps a copy of the self model for each attempt that runs at once: one for a
    // single attempt, and at most the planning threads, four, for any more.
    #[test]
    fn a_plan_copies_the_self_model_once_for_each_attempt_at_once() {
        let copies = |attempts: u32| {
            self_model_copies(NonZeroU32::new(attempts).expect("an attempt or more")).get()
        };
        assert_eq!(PLANNING_THREADS.get(), 4);
        assert_eq!([1, 3, 4, 5, 9].map(copies), [1, 3, 4, 4, 4]);
    }

    // A stop set before the plan starts ends it stopped, whatever its time: the plan
    // answers no path.
    #[test]
    fn a_set_stop_ends_the_plan_stopped() {
        let node = started(&parameters(), false);
        measure_at(&node, SystemTime::UNIX_EPOCH, still());
        let goal = joint_goal(&node, Side::Left, &elbow_at(Side::Left, 1.3));
        let taken = admit(&node, &goal).expect("the goal is taken");
        assert!(matches!(
            plan(&node, &taken, &AtomicBool::new(true)),
            Err(PlanEnd::Stopped)
        ));
    }

    // A stop set before a Cartesian path starts ends it stopped: it answers no path.
    #[test]
    fn a_set_stop_ends_the_cartesian_path_stopped() {
        let node = started(&parameters(), false);
        measure_at(&node, SystemTime::UNIX_EPOCH, still());
        let grasp_link = node.arms[Side::Left].grasp_link;
        let chain = node.arms[Side::Left].geometry.chain();
        let (position, orientation) = pose_parts(&chain.world_ee_pose(&still().arms[Side::Left]));
        let goal = plan_cartesian_path::GoalRequestData {
            arm_name: "left_arm".to_string(),
            start_joint_positions: Vec::new(),
            link_name: grasp_link.to_string(),
            waypoints: vec![plan_cartesian_path::GoalWaypointsItem {
                position: [position[0], position[1], position[2] - 0.05],
                orientation,
            }],
            avoid_collisions: true,
        };
        let taken = admit_cartesian(&node, &goal).expect("the goal is taken");
        assert!(matches!(
            cartesian(&node, &taken, &AtomicBool::new(true)),
            Err(PlanEnd::Stopped)
        ));
    }

    // camera_mounts' answer places the camera it names at its pose. An answer whose lists
    // do not give each camera its 3 positions, 4 orientations and 1 carrier, a camera it
    // does not place, and one an arm carries, are lasting faults, each named.
    #[test]
    fn the_camera_mounts_answer_places_the_camera_or_names_a_lasting_fault() {
        let poses = get_camera_poses::ResponseData {
            timestamp: SystemTime::UNIX_EPOCH,
            camera_names: vec!["wrist".to_string(), "chest".to_string()],
            positions: vec![0.0, 0.0, 0.0, 0.1, 0.2, 0.3],
            orientations: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            carried_by: vec!["openarm_left_link7".to_string(), String::new()],
        };
        let chest = placed_camera(&poses, "chest").expect("the chest camera's pose");
        let expected = Pose::new([0.1, 0.2, 0.3], [0.0, 0.0, 1.0, 0.0]).expect("a pose");
        assert_eq!(format!("{chest:?}"), format!("{expected:?}"));
        let ragged = get_camera_poses::ResponseData {
            positions: vec![0.0; 7],
            ..poses.clone()
        };
        assert_eq!(
            placed_camera(&ragged, "chest").err(),
            Some(CameraFault::Lasting(
                "camera_mounts' get_camera_poses names 2 cameras and gives 7 position values, 8 \
                 orientation values and 2 carried_by entries: 3, 4 and 1 for each camera"
                    .to_string()
            ))
        );
        let few_carriers = get_camera_poses::ResponseData {
            carried_by: vec![String::new()],
            ..poses.clone()
        };
        assert_eq!(
            placed_camera(&few_carriers, "chest").err(),
            Some(CameraFault::Lasting(
                "camera_mounts' get_camera_poses names 2 cameras and gives 6 position values, 8 \
                 orientation values and 1 carried_by entries: 3, 4 and 1 for each camera"
                    .to_string()
            ))
        );
        assert_eq!(
            placed_camera(&poses, "head").err(),
            Some(CameraFault::Lasting(
                "camera_mounts places no camera named \"head\"".to_string()
            ))
        );
        assert_eq!(
            placed_camera(&poses, "wrist").err(),
            Some(CameraFault::Lasting(
                "wrist rides on openarm_left_link7; the planner reads a camera the body carries"
                    .to_string()
            ))
        );
    }

    // A node with a camera refuses a plan or a check until the map takes in a frame,
    // naming what the map waits on, and after the map is cleared; it refuses one before
    // the arms are measured. A node with no camera needs the arms alone.
    #[test]
    fn a_plan_or_a_check_waits_for_a_frame_and_the_joints() {
        let node = started(&parameters(), true);
        let waiting = |on: &str| {
            Err(Refusal::PreconditionFailed(format!(
                "{NO_DEPTH_FRAME}: {on}"
            )))
        };
        assert_eq!(snapshot(&node), waiting(AWAITING_FRAME));
        node.wait_on("the perception camera: the robot names no perception camera");
        assert_eq!(
            snapshot(&node),
            waiting("the perception camera: the robot names no perception camera")
        );
        lock(&node.map).stamp = Some(SystemTime::UNIX_EPOCH);
        assert_eq!(
            snapshot(&node),
            Err(Refusal::PreconditionFailed(NO_JOINTS.to_string()))
        );
        measure_at(&node, SystemTime::UNIX_EPOCH, still());
        assert_eq!(snapshot(&node), Ok(()));

        let unlinked = started(&parameters(), false);
        measure_at(&unlinked, SystemTime::UNIX_EPOCH, still());
        assert_eq!(snapshot(&unlinked), Ok(()));
    }

    /// The instant `ms` milliseconds after the epoch.
    fn after(ms: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_millis(ms)
    }

    /// A node with a camera, both arms measured every 100 ms from `from_ms` to `to_ms`.
    fn measured_node(from_ms: u64, to_ms: u64) -> Arc<Node> {
        let node = Arc::new(started(&parameters(), true));
        measure(&node, from_ms, to_ms);
        node
    }

    /// Records `measured` in the history of `node` at `at`.
    fn measure_at(node: &Node, at: SystemTime, measured: Measured<ARM_DOF>) {
        push_measured(node, at, &measured);
    }

    /// Both arms of `node` measured every 100 ms from `from_ms` to `to_ms`.
    fn measure(node: &Node, from_ms: u64, to_ms: u64) {
        for ms in (from_ms..=to_ms).step_by(100) {
            measure_at(node, after(ms), still());
        }
    }

    /// The intake of `node` from a camera of one pixel at the robot frame's origin.
    fn intake(node: Arc<Node>) -> Intake<ARM_DOF, DUAL_DOF, 2> {
        intake_at(node, &identity())
    }

    /// The pose that moves nothing.
    fn identity() -> Pose {
        Pose::new([0.0; 3], [0.0, 0.0, 0.0, 1.0]).expect("the identity is a pose")
    }

    /// The intake of `node` from a camera of one pixel at `pose`, which sees along its
    /// optical axis.
    fn intake_at(node: Arc<Node>, pose: &Pose) -> Intake<ARM_DOF, DUAL_DOF, 2> {
        let lens = Pinhole::parse(&Intrinsics {
            width: 1,
            height: 1,
            fx: 1.0,
            fy: 1.0,
            cx: 0.0,
            cy: 0.0,
            distortion_model: "none",
            distortion: &[],
        })
        .expect("a pinhole model");
        Intake::new(
            node,
            PerceptionCamera {
                depth: DepthCamera::new(pose, &identity(), lens, DepthModel::Z, 0.001),
                align_mode: "depth_to_color".to_string(),
            },
        )
    }

    /// A frame of the one-pixel camera, taken at `at`, that sees nothing.
    fn frame_at(at: SystemTime) -> depth_stream::Message {
        depth_stream::Message {
            header: depth_stream::MessageHeader {
                timestamp: at,
                frame_id: 0,
                align_mode: "depth_to_color".to_string(),
            },
            encoding: "z16".to_string(),
            width: 1,
            height: 1,
            frame: vec![0, 0],
        }
    }

    /// A frame of the one-pixel camera, taken at `at`, that sees a point `depth_mm`
    /// millimeters along its optical axis.
    fn frame_seeing(at: SystemTime, depth_mm: u16) -> depth_stream::Message {
        depth_stream::Message {
            frame: depth_mm.to_le_bytes().to_vec(),
            ..frame_at(at)
        }
    }

    /// The stamp of the newest frame the map of `node` holds.
    fn map_stamp(node: &Node) -> Option<SystemTime> {
        lock(&node.map).stamp
    }

    // The map takes in a frame, skips the one half a frame interval after it, and takes
    // the one a whole frame interval after it.
    #[test]
    fn the_map_takes_at_most_one_frame_a_frame_interval() {
        let node = measured_node(0, 2_000);
        let interval = node.config.frame_interval;
        let first = after(1_000);
        let mut intake = intake(node.clone());
        intake.take(&frame_at(first));
        assert_eq!(map_stamp(&node), Some(first));
        intake.take(&frame_at(first + interval / 2));
        assert_eq!(
            map_stamp(&node),
            Some(first),
            "the frame half an interval on"
        );
        intake.take(&frame_at(first + interval));
        assert_eq!(map_stamp(&node), Some(first + interval));
    }

    /// The waiting frames of `node`'s depth frames, which the sender gives, and of its
    /// joint positions.
    fn waiting_frames(
        node: &Arc<Node>,
    ) -> (watch::Sender<Newest>, WaitingFrames<ARM_DOF, DUAL_DOF, 2>) {
        let (newest, frames) = watch::channel::<Newest>(None);
        (newest, WaitingFrames::new(node.clone(), frames))
    }

    /// Sends the frame of the one-pixel camera taken `ms` milliseconds after the epoch.
    fn send_frame(newest: &watch::Sender<Newest>, ms: u64) {
        newest.send_replace(Some(Arc::new(frame_at(after(ms)))));
    }

    /// The stamp of the frame `waiting` gives now; `None` while every frame waits.
    fn given(waiting: &mut WaitingFrames<ARM_DOF, DUAL_DOF, 2>) -> Option<SystemTime> {
        let mut context = Context::from_waker(Waker::noop());
        match std::pin::pin!(waiting.next()).poll(&mut context) {
            Poll::Ready(frame) => Some(frame.expect("the frames' follower runs").header.timestamp),
            Poll::Pending => None,
        }
    }

    // A frame that arrives before the joint positions at its stamp waits for them. Of the
    // frames that wait, the newest whose stamp the joint positions reach comes: the newest
    // frame, the ones before it dropped, or else the oldest, the newest then waiting on its
    // own.
    #[test]
    fn the_newest_frame_whose_joints_are_measured_comes() {
        let node = measured_node(0, 1_000);
        let (newest, mut waiting) = waiting_frames(&node);
        send_frame(&newest, 1_050);
        assert_eq!(given(&mut waiting), None, "the frame 50 ms past the joints");
        send_frame(&newest, 1_150);
        send_frame(&newest, 1_250);
        measure_at(&node, after(1_300), still());
        assert_eq!(given(&mut waiting), Some(after(1_250)));
        assert_eq!(
            given(&mut waiting),
            None,
            "the frames before it are dropped"
        );

        send_frame(&newest, 1_350);
        assert_eq!(given(&mut waiting), None, "the frame 50 ms past the joints");
        send_frame(&newest, 1_450);
        measure_at(&node, after(1_400), still());
        assert_eq!(given(&mut waiting), Some(after(1_350)));
        assert_eq!(
            given(&mut waiting),
            None,
            "the newest frame waits on its own"
        );
        measure_at(&node, after(1_500), still());
        assert_eq!(given(&mut waiting), Some(after(1_450)));
    }

    // While a frame waits for the joint positions at its stamp, a refusal names it and how
    // far the newest limb_states message lags it, or that none has arrived.
    #[test]
    fn a_refusal_names_the_frame_that_waits_for_its_joints() {
        let waiting_on = |on: &str| {
            Err(Refusal::PreconditionFailed(format!(
                "{NO_DEPTH_FRAME}: a depth frame waits for the joint positions at its stamp, {on}"
            )))
        };
        let node = measured_node(0, 1_000);
        let (newest, mut waiting) = waiting_frames(&node);
        send_frame(&newest, 1_050);
        assert_eq!(given(&mut waiting), None);
        assert_eq!(
            snapshot(&node),
            waiting_on("50ms after the newest limb_states message")
        );

        let unmeasured = Arc::new(started(&parameters(), true));
        let (newest, mut waiting) = waiting_frames(&unmeasured);
        send_frame(&newest, 1_050);
        assert_eq!(given(&mut waiting), None);
        assert_eq!(
            snapshot(&unmeasured),
            waiting_on("and no limb_states message has arrived")
        );
    }

    // The map's waiting reason follows what it waits on: a frame that waits for its joint
    // positions; the next frame once that frame is handed to the intake, which drops it as
    // taken before a clear; and the next frame once a clock that steps back drops the frame
    // that waits.
    #[test]
    fn the_waiting_reason_follows_what_the_map_waits_on() {
        let waiting_on = |on: String| {
            Err(Refusal::PreconditionFailed(format!(
                "{NO_DEPTH_FRAME}: {on}"
            )))
        };
        let joints_after = |lag: &str| {
            waiting_on(format!(
                "a depth frame waits for the joint positions at its stamp, {lag} after the newest limb_states message"
            ))
        };
        let node = measured_node(0, 1_000);
        let mut intake = intake(node.clone());
        let (newest, mut waiting) = waiting_frames(&node);
        send_frame(&newest, 1_050);
        assert_eq!(given(&mut waiting), None);
        assert_eq!(snapshot(&node), joints_after("50ms"));
        measure_at(&node, after(1_100), still());
        clear(&node);
        assert_eq!(given(&mut waiting), Some(after(1_050)));
        intake.take(&frame_at(after(1_050)));
        assert_eq!(map_stamp(&node), None, "the frame taken before the clear");
        assert_eq!(snapshot(&node), waiting_on(AWAITING_FRAME.to_string()));

        send_frame(&newest, 1_250);
        assert_eq!(given(&mut waiting), None);
        assert_eq!(snapshot(&node), joints_after("150ms"));
        measure_at(&node, after(500), still());
        assert_eq!(given(&mut waiting), None, "the frame of the clock before");
        assert_eq!(snapshot(&node), waiting_on(AWAITING_FRAME.to_string()));
    }

    // A limb_states message stamped before the newest, as after a clock that stepped back,
    // restarts the history and clears the map at its stamp: the map waits on its next
    // frame, a frame stamped at or before the restart marks nothing, a later one fills the
    // map, and a frame that waited for joint positions on the clock before is dropped.
    #[test]
    fn a_clock_that_steps_back_clears_the_map() {
        let node = measured_node(0, 2_000);
        let mut intake = intake(node.clone());
        intake.take(&frame_at(after(1_000)));
        assert_eq!(map_stamp(&node), Some(after(1_000)));
        let (newest, mut waiting) = waiting_frames(&node);
        send_frame(&newest, 2_050);
        assert_eq!(given(&mut waiting), None);

        let malformed = LatchedLog::warn("limb_states");
        record(
            &node,
            &limb_states_at(
                after(500),
                ready(Side::Left),
                ready(Side::Right),
                &[1.0, 1.0],
            ),
            &malformed,
        );
        assert_eq!(map_stamp(&node), None);
        assert_eq!(
            snapshot(&node),
            Err(Refusal::PreconditionFailed(format!(
                "{NO_DEPTH_FRAME}: {AWAITING_FRAME}"
            )))
        );
        measure(&node, 600, 2_100);
        assert_eq!(given(&mut waiting), None, "the frame of the clock before");
        intake.take(&frame_at(after(500)));
        assert_eq!(map_stamp(&node), None, "the frame at the restart");
        intake.take(&frame_at(after(700)));
        assert_eq!(map_stamp(&node), Some(after(700)));
    }

    // A cancel that comes before the plan answers sets the stop: the plan, which answers
    // whether its stop is set once it is asked again, answers true. A plan that answers
    // first is taken with no stop set.
    #[tokio::test]
    async fn a_cancel_sets_the_stop_of_the_plan() {
        let stop = AtomicBool::new(false);
        let mut polled = false;
        let stopped_once_asked = std::future::poll_fn(|context| {
            if !polled {
                polled = true;
                context.waker().wake_by_ref();
                return Poll::Pending;
            }
            Poll::Ready(stop.load(Ordering::SeqCst))
        });
        assert!(answer_or_stop(stopped_once_asked, std::future::ready(()), &stop).await);

        let unstopped = AtomicBool::new(false);
        assert_eq!(
            answer_or_stop(
                std::future::ready("a path"),
                std::future::pending(),
                &unstopped
            )
            .await,
            "a path"
        );
        assert!(!unstopped.load(Ordering::SeqCst));
    }

    // While limb_states lags the camera by more than a frame interval, frames still come:
    // each newer frame arrives before the joint positions at the frame before it, and that
    // frame comes once they arrive.
    #[test]
    fn frames_come_while_the_joints_lag_behind_them() {
        let node = measured_node(0, 1_000);
        let (newest, mut waiting) = waiting_frames(&node);
        send_frame(&newest, 1_100);
        assert_eq!(
            given(&mut waiting),
            None,
            "the frame 100 ms past the joints"
        );
        let taken: Vec<_> = (1..=5)
            .filter_map(|k| {
                send_frame(&newest, 1_100 + 100 * k);
                measure_at(&node, after(1_050 + 100 * k), still());
                given(&mut waiting)
            })
            .collect();
        assert_eq!(taken, [1_100, 1_200, 1_300, 1_400, 1_500].map(after));
    }

    /// Where each finger of `model` hangs in its link's frame, on the arms of `node`.
    fn finger_offsets(node: &Node, own: &OwnBody<ARM_DOF, 2>) -> Vec<Isometry3<f64>> {
        riders_of_the_arms(own, &node.arms)
            .expect("the arms' fingers")
            .into_iter()
            .flatten()
            .map(|finger| finger.offset)
            .collect()
    }

    /// A limb_states message of both arms at `left` and `right`, measured at `at`, its
    /// grippers at `openings`.
    fn limb_states_at(
        at: SystemTime,
        left: [f64; ARM_DOF],
        right: [f64; ARM_DOF],
        openings: &[f64],
    ) -> limb_states::Message {
        limb_states::Message {
            timestamp: at,
            arm_names: vec!["left_arm".to_string(), "right_arm".to_string()],
            joints_per_arm: vec![ARM_DOF as u32; 2],
            joint_positions: [left, right].concat(),
            positions: vec![0.0; 6],
            orientations: [[0.0, 0.0, 0.0, 1.0]; 2].concat(),
            gripper_names: ["left_gripper", "right_gripper"][..openings.len()]
                .iter()
                .map(|name| name.to_string())
                .collect(),
            gripper_openings: openings.to_vec(),
        }
    }

    // A limb_states message records both arms' joints and both grippers' openings, which
    // plans and checks then read in the self model; a message that lacks a gripper's
    // opening, or one whose joint position is not finite, records nothing.
    #[test]
    fn a_message_records_the_joints_and_the_openings() {
        let node = started(&parameters(), false);
        let malformed = LatchedLog::warn("limb_states");
        let left = std::array::from_fn(|i| 0.1 * i as f64);
        let right = std::array::from_fn(|i| -0.1 * i as f64);
        record(
            &node,
            &limb_states_at(after(1_000), left, right, &[0.25, 0.75]),
            &malformed,
        );
        let measured = Measured {
            arms: PerArm::both(left, right),
            openings: PerArm::both(0.25, 0.75),
        };
        assert_eq!(node.latest().as_ref(), Some(&measured));
        let mut opened = node.own.clone();
        opened.set_openings(&PerArm::both(0.25, 0.75));
        assert_eq!(
            finger_offsets(&node, &node.own_at(node.latest().as_ref())),
            finger_offsets(&node, &opened)
        );
        record(
            &node,
            &limb_states_at(after(1_100), right, left, &[0.5]),
            &malformed,
        );
        assert_eq!(node.latest().as_ref(), Some(&measured));
        let mut broken = left;
        broken[3] = f64::NAN;
        record(
            &node,
            &limb_states_at(after(1_200), broken, right, &[0.5, 0.5]),
            &malformed,
        );
        assert_eq!(node.latest().as_ref(), Some(&measured));
    }

    // The intake cuts the fingers out of a frame where the grippers' openings, read at its
    // stamp between two measurements, put them: halfway between closed and open.
    #[test]
    fn a_frame_cuts_the_fingers_out_at_its_openings() {
        let node = Arc::new(started(&parameters(), true));
        let opened = |left: f64| Measured {
            openings: PerArm::both(left, 1.0 - left),
            ..still()
        };
        measure_at(&node, after(1_000), opened(0.0));
        measure_at(&node, after(1_100), opened(1.0));
        let mut intake = intake(node.clone());
        intake.take(&frame_at(after(1_050)));
        assert_eq!(map_stamp(&node), Some(after(1_050)));
        let mut halfway = node.own.clone();
        halfway.set_openings(&PerArm::both(0.5, 0.5));
        assert_eq!(
            finger_offsets(&node, &intake.own),
            finger_offsets(&node, &halfway)
        );
        assert_ne!(
            finger_offsets(&node, &node.own),
            finger_offsets(&node, &halfway)
        );
    }

    // A frame the camera took before the map was cleared marks nothing, however late it
    // comes to the intake: the map, cleared with the arms last measured at 2 s, takes no
    // frame of 1.5 s, and takes the frame of 2.4 s.
    #[test]
    fn a_frame_taken_before_a_clear_marks_nothing() {
        let node = measured_node(0, 2_000);
        let mut intake = intake(node.clone());
        intake.take(&frame_at(after(1_000)));
        assert_eq!(map_stamp(&node), Some(after(1_000)));
        clear(&node);
        assert_eq!(map_stamp(&node), None);
        intake.take(&frame_at(after(1_500)));
        assert_eq!(map_stamp(&node), None, "the frame of 1.5 s");
        assert_eq!(
            lock(&node.map).waiting,
            Waiting::Passing(AWAITING_FRAME.to_string())
        );
        measure(&node, 2_100, 2_500);
        intake.take(&frame_at(after(2_400)));
        assert_eq!(map_stamp(&node), Some(after(2_400)));
    }

    // A clear empties the map, which then waits on its next frame, whatever passing fault
    // it last noted; a lasting fault, which ends the intake, stays what it waits on
    // through a clear and through any passing fault after it.
    #[test]
    fn a_clear_waits_on_the_next_frame_unless_a_fault_lasts() {
        let node = measured_node(0, 2_000);
        let waiting = |on: &str| {
            Err(Refusal::PreconditionFailed(format!(
                "{NO_DEPTH_FRAME}: {on}"
            )))
        };
        let mut intake = intake(node.clone());
        node.wait_on("a depth frame was skipped: no joint positions were measured at its stamp");
        intake.take(&frame_at(after(1_000)));
        clear(&node);
        assert_eq!(snapshot(&node), waiting(AWAITING_FRAME));

        measure(&node, 2_100, 2_500);
        intake.take(&frame_at(after(2_400)));
        assert_eq!(map_stamp(&node), Some(after(2_400)));
        let lasting = "the perception camera: the robot names no perception camera";
        node.wait_on_lasting(lasting);
        node.wait_on("a depth frame was skipped: it is encoded \"rgb8\"");
        clear(&node);
        assert_eq!(snapshot(&node), waiting(lasting));
    }

    // A frame encoded other than z16 marks nothing, and the map waits on it, naming the
    // encoding.
    #[test]
    fn a_frame_of_another_encoding_is_skipped() {
        let node = measured_node(0, 2_000);
        let mut intake = intake(node.clone());
        intake.take(&depth_stream::Message {
            encoding: "rgb8".to_string(),
            ..frame_at(after(1_000))
        });
        assert_eq!(map_stamp(&node), None);
        assert_eq!(
            snapshot(&node),
            Err(Refusal::PreconditionFailed(format!(
                "{NO_DEPTH_FRAME}: a depth frame was skipped: it is encoded \"rgb8\"; the map \
                 reads \"z16\" depth frames"
            )))
        );
    }

    // With the grippers measured closed, the map cuts the fingers out where they hang
    // closed: a point at the middle of a piece of a closed finger, a point outside the
    // robot with its fingers open, marks nothing. The robot is grown by nothing, so only
    // the fingers' openings decide.
    #[test]
    fn a_frame_cuts_the_fingers_out_where_the_measured_openings_put_them() {
        let params = Parameters {
            padding_offset_m: 0.0,
            ..parameters()
        };
        let node = Arc::new(started(&params, true));
        let closed = Measured {
            openings: PerArm::both(0.0, 0.0),
            ..still()
        };
        measure_at(&node, after(1_000), closed.clone());
        measure_at(&node, after(1_100), closed.clone());
        let mut shut = node.own.clone();
        shut.set_openings(&closed.openings);
        let unpadded = Padding::new(1.0, 0.0).expect("the padding that grows nothing");
        let robot_holds = |own: &OwnBody<ARM_DOF, 2>, point: &Point3<f64>| {
            let fingers = riders_of_the_arms(own, &node.arms).expect("the arms' fingers");
            let scene = lock(&node.scene);
            robot_mask(
                &node.arms,
                &node.body,
                &closed.arms,
                &fingers,
                &scene,
                unpadded,
            )
            .placed()
            .iter()
            .any(|part| part.contains(point))
        };
        let shut_fingers = riders_of_the_arms(&shut, &node.arms).expect("the arms' fingers");
        let point = Side::BOTH
            .into_iter()
            .flat_map(|side| {
                let posed = node.arms[side].geometry.chain().at(&closed.arms[side]);
                shut_fingers[side]
                    .iter()
                    .flat_map(move |finger| {
                        let placed = posed.link_pose_world(finger.segment) * finger.offset;
                        let RiderBody::Pieces(pieces) = finger.body else {
                            panic!("a finger is a part of the robot");
                        };
                        pieces.iter().map(move |piece| placed * piece.center())
                    })
                    .collect::<Vec<_>>()
            })
            .find(|point| !robot_holds(&node.own, point))
            .expect("a closed finger's piece the open fingers leave outside the robot");
        assert!(robot_holds(&shut, &point), "{point}");

        let eye = point + Vector3::new(0.3, 0.0, 0.0);
        let facing = UnitQuaternion::rotation_between(&Vector3::z(), &-Vector3::x())
            .expect("a turn from +z to -x");
        let camera = Pose::new(eye.coords.into(), [facing.i, facing.j, facing.k, facing.w])
            .expect("the camera's pose");
        let mut intake = intake_at(node.clone(), &camera);
        intake.take(&frame_seeing(after(1_050), 300));
        assert_eq!(map_stamp(&node), Some(after(1_050)));
        assert_eq!(lock(&node.map).voxels.occupied().count(), 0);
    }

    /// A point to the right of the robot and in front of it, about 0.2 m above the floor,
    /// at the center of a voxel of the launcher's 2 cm map, so the hit of a ray that ends
    /// on it lands in that voxel whatever the rounding.
    const BESIDE: [f64; 3] = [0.51, -0.61, 0.21];

    /// A camera 0.3 m above `point`, looking down at it.
    fn above(point: [f64; 3]) -> Pose {
        let down = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI);
        Pose::new(
            [point[0], point[1], point[2] + 0.3],
            [down.i, down.j, down.k, down.w],
        )
        .expect("the camera's pose")
    }

    /// The entry of apply_collision_objects that applies `operation` to the cube `id`,
    /// 5 cm on a side, about `center`.
    fn cube(operation: &str, id: &str, center: [f64; 3]) -> ApplyCollisionObjectsObjectsItem {
        ApplyCollisionObjectsObjectsItem {
            id: id.to_string(),
            operation: operation.to_string(),
            position: center,
            orientation: [0.0, 0.0, 0.0, 1.0],
            shapes: vec![ApplyCollisionObjectsObjectsItemShapesItem {
                kind: "box".to_string(),
                dimensions: vec![0.05; 3],
                mesh_vertices: Vec::new(),
                mesh_triangles: Vec::new(),
                position: [0.0; 3],
                orientation: [0.0, 0.0, 0.0, 1.0],
            }],
        }
    }

    fn objects(
        objects: Vec<ApplyCollisionObjectsObjectsItem>,
    ) -> apply_collision_objects::RequestData {
        apply_collision_objects::RequestData { objects }
    }

    /// The attach of the object `id` to the left arm's grasp link, touching nothing else.
    fn attach_to_the_left_hand(id: &str) -> apply_attached_collision_object::RequestData {
        apply_attached_collision_object::RequestData {
            object_id: id.to_string(),
            link_name: "openarm_left_tcp".to_string(),
            touch_links: Vec::new(),
            operation: "attach".to_string(),
        }
    }

    /// Marks the voxel of `point` in the map of `node` with three hits from a camera above
    /// it: OctoMap's hit adds 0.85 to a voxel's log-odds, which a cleared voxel holds at
    /// its floor, -2.0, and a voxel at 0 or above is occupied.
    fn mark(node: &Node, point: [f64; 3]) {
        let mut map = lock(&node.map);
        let scan = Scan {
            frame: &node.frame,
            sensor: above(point),
            points: &[[0.0, 0.0, 0.3]],
            max_range_m: node.config.max_range_m,
        };
        for _ in 0..3 {
            map.voxels
                .insert(&scan, &[])
                .expect("the scan stands in the map's frame");
        }
    }

    /// Whether the voxel of `point` in the map of `node` is occupied.
    fn occupied_at(node: &Node, point: [f64; 3]) -> bool {
        let map = lock(&node.map);
        let key = map.voxels.key(point).expect("a point inside the map");
        map.voxels.is_occupied(key)
    }

    // An object clears from the map every voxel it intersects where a call leaves it: an
    // add elsewhere leaves a voxel occupied, a move onto the voxel clears it, and an attach
    // clears it again under the object where it stands. A refused batch clears nothing and
    // places nothing.
    #[test]
    fn objects_clear_the_voxels_they_intersect_where_a_call_leaves_them() {
        let node = started(&parameters(), false);
        measure_at(&node, after(1_000), still());
        let elsewhere = [BESIDE[0], BESIDE[1], BESIDE[2] + 0.4];
        mark(&node, BESIDE);
        assert!(occupied_at(&node, BESIDE), "the voxel the camera marks");
        apply_objects(&node, &objects(vec![cube("add", "box", elsewhere)])).expect("added");
        assert!(occupied_at(&node, BESIDE), "with the box elsewhere");
        let refused = apply_objects(
            &node,
            &objects(vec![
                cube("move", "box", BESIDE),
                cube("remove", "nothing", BESIDE),
            ]),
        );
        assert!(matches!(refused, Err(Refusal::Rejected(_))), "{refused:?}");
        assert!(occupied_at(&node, BESIDE), "after a refused batch");
        apply_objects(&node, &objects(vec![cube("move", "box", BESIDE)])).expect("moved");
        assert!(!occupied_at(&node, BESIDE), "with the box moved onto it");
        mark(&node, BESIDE);
        apply_attachment(&node, &attach_to_the_left_hand("box")).expect("attached");
        assert!(
            !occupied_at(&node, BESIDE),
            "with the box attached where it stands"
        );
    }

    /// Every voxel of the map of `node` a ray has reached, by its center, with what it holds.
    fn known_voxels(node: &Node) -> Vec<([f64; 3], Voxel)> {
        let map = lock(&node.map);
        map.voxels
            .known()
            .map(|key| (map.voxels.center(key), map.voxels.voxel(key)))
            .collect()
    }

    // An apply_collision_objects call of no objects, and a set_allowed_collision call of no
    // other names, allowing or stopping, change nothing: the scene and the map stay as
    // they were.
    #[test]
    fn a_call_of_no_objects_or_no_names_changes_nothing() {
        let node = started(&parameters(), false);
        measure_at(&node, after(1_000), still());
        let elsewhere = [BESIDE[0], BESIDE[1], BESIDE[2] + 0.4];
        mark(&node, BESIDE);
        apply_objects(&node, &objects(vec![cube("add", "box", elsewhere)])).expect("added");
        let pair = |other_names: Vec<String>, allowed: bool| set_allowed_collision::RequestData {
            name: "box".to_string(),
            other_names,
            allowed,
        };
        allow_pairs(&node, &pair(vec![OCCUPANCY_MAP.to_string()], true)).expect("allowed");
        assert_eq!(
            lock(&node.scene).snapshot(0.0, 0.0).allowed_pairs,
            [["box".to_string(), OCCUPANCY_MAP.to_string()]]
        );
        let (scene, voxels) = (lock(&node.scene).clone(), known_voxels(&node));
        apply_objects(&node, &objects(Vec::new())).expect("an empty list applies");
        for allowed in [true, false] {
            allow_pairs(&node, &pair(Vec::new(), allowed)).expect("no other names apply");
        }
        assert_eq!(*lock(&node.scene), scene);
        assert_eq!(known_voxels(&node), voxels);
    }

    // An attach before the arms' joint positions are measured is refused with
    // precondition_failed and the message a plan or a check gives for the same cause.
    #[test]
    fn an_attach_before_the_arms_are_measured_names_what_it_waits_on() {
        let node = started(&parameters(), false);
        apply_objects(&node, &objects(vec![cube("add", "box", BESIDE)])).expect("added");
        let unmeasured = Err(Refusal::PreconditionFailed(NO_JOINTS.to_string()));
        assert_eq!(
            apply_attachment(&node, &attach_to_the_left_hand("box")),
            unmeasured
        );
        assert_eq!(snapshot(&node), unmeasured);
    }

    // Three frames that see a point inside an object mark nothing, and nor do three that see
    // it inside an object the left arm holds, where the arm held it at their stamps, though
    // the arm was measured elsewhere since; three frames taken once the arm had moved the
    // object away mark the point's voxel.
    #[test]
    fn a_frame_marks_nothing_inside_an_object_where_it_stood_at_the_stamp() {
        let node = Arc::new(started(&parameters(), true));
        let mut intake = intake_at(node.clone(), &above(BESIDE));
        let three_frames = |intake: &mut Intake<ARM_DOF, DUAL_DOF, 2>, from_ms: u64| {
            for ms in [from_ms, from_ms + 200, from_ms + 400] {
                intake.take(&frame_seeing(after(ms), 300));
                assert_eq!(map_stamp(&node), Some(after(ms)));
            }
        };
        measure(&node, 1_000, 2_100);
        apply_objects(&node, &objects(vec![cube("add", "box", BESIDE)])).expect("added");
        three_frames(&mut intake, 1_100);
        assert!(!occupied_at(&node, BESIDE), "a point inside the box");
        apply_attachment(&node, &attach_to_the_left_hand("box")).expect("attached");
        let mut moved = still();
        moved.arms[Side::Left][3] += 0.5;
        for ms in (2_300..=2_900).step_by(100) {
            measure_at(&node, after(ms), moved.clone());
        }
        three_frames(&mut intake, 1_700);
        assert!(
            !occupied_at(&node, BESIDE),
            "a point inside the box the hand held at the frames' stamps"
        );
        three_frames(&mut intake, 2_400);
        assert!(
            occupied_at(&node, BESIDE),
            "the point once the hand moved the box away"
        );
    }

    // Of three frames that arrive while the intake is busy, it takes the newest, and then
    // waits: neither the newest nor a frame it replaced comes again. A frame that arrives
    // later is the next it takes. Once the follower ends, no frame comes.
    #[tokio::test]
    async fn the_intake_takes_the_newest_frame_and_never_one_it_replaced() {
        let (newest, mut frames) = watch::channel::<Newest>(None);
        for ms in [0, 100, 200] {
            newest.send_replace(Some(Arc::new(frame_at(after(ms)))));
        }
        let stamp =
            |frame: Option<Arc<depth_stream::Message>>| frame.map(|frame| frame.header.timestamp);
        assert_eq!(stamp(next_frame(&mut frames).await), Some(after(200)));
        assert!(
            tokio::time::timeout(Duration::ZERO, next_frame(&mut frames))
                .await
                .is_err(),
            "the intake waits for a frame it has not taken"
        );
        newest.send_replace(Some(Arc::new(frame_at(after(300)))));
        assert_eq!(stamp(next_frame(&mut frames).await), Some(after(300)));
        drop(newest);
        assert_eq!(stamp(next_frame(&mut frames).await), None);
    }
}
