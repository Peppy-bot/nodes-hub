//! Node composition: setup parses every parameter and serves the limb names, then,
//! once the robot's initializer names the model, the node builds the arm models and
//! the self-collision governor from its description, wires the channels, and spawns
//! the services answered from bringup (camera mounts, workspace) plus the
//! readiness-gated coordination loop, every action handler and every stream
//! listener. All motion and stream logic lives in the sibling modules; this is only
//! the assembly.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use control_core::fault::Fault;
use control_core::positive_finite::{NotPositiveFinite, PositiveFinite};
use control_core::serving;
use control_core::time::{RateOutOfRange, period_from_hz};
use peppygen::consumed_topics::collision_ctrl::governor_control;
use peppygen::exposed_services::limb_state::get_limb_names;
use peppygen::{NodeRunner, Parameters, Result};
use robot_description::{Driven, PerArm, states};
use srs_model::{ARM_DOF, JointVec};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;
use tracing::{error, info};

use workspace_core::surface_block::ArmBodies;

use crate::actions;
use crate::camera_mounts::{self, CameraMounts};
use crate::coordinator::{self, ArmChannels};
use crate::governor;
use crate::identity;
use crate::liveness;
use crate::move_slot::MoveSlot;
use crate::path_checks::PathChecks;
use crate::planner::{PathGuard, PlanConfig, Planner};
use crate::robot::{self, ARMS, LIMBS};
use crate::startup;
use crate::streams;
use crate::streams::LeaderActivity;
use crate::types::limb_names;
use crate::upstream::UpstreamMode;
use crate::workspace::Workspace;
use crate::workspace_service;

/// The fastest the coordination loop is driven. It paces the same CAN FD arms
/// as openarm_arm.
const MAX_RATE_HZ: u32 = 1_000;

/// Everything this node refuses to run on: a parameter, which `setup` refuses, or the
/// robot its initializer names, which ends the node with a [`Fault`] after `setup`
/// has returned.
///
/// Every variant either names what it rejected or carries the error it came from, so
/// this list is the one place to read what a launch can be rejected for and nothing
/// is flattened to a message on the way out. It exists because returning a refusal,
/// rather than panicking it, is what runs the runtime's shutdown hooks: a panic
/// unwinds straight past them.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("parameter control_rate_hz")]
    ControlRate(#[source] RateOutOfRange),

    #[error("parameter follower_state_rate_hz")]
    FollowerStateRate(#[source] liveness::StaleLimitOutOfRange),

    #[error("parameter max_stream_ee_velocity_m_s")]
    EeVelocity(#[source] NotPositiveFinite),

    #[error("parameter start_tolerance_rad")]
    StartTolerance(#[source] NotPositiveFinite),

    #[error("parameter max_joint_velocity_rad_s_{joint}")]
    JointVelocityCap {
        joint: usize,
        #[source]
        source: NotPositiveFinite,
    },

    #[error("parameter max_joint_acceleration_rad_s2_{joint}")]
    JointAccelerationCap {
        joint: usize,
        #[source]
        source: NotPositiveFinite,
    },

    #[error(
        "velocity_filter_cutoff_hz ({cutoff_hz}) must be in \
         (0, Nyquist = control_rate_hz/2 = {nyquist_hz})"
    )]
    FilterCutoff { cutoff_hz: f64, nyquist_hz: f64 },

    #[error("the model the robot's initializer serves")]
    Model(#[source] robot::ModelError),

    #[error(
        "the robot's initializer serves the description {served} of {model}, and this \
         backbone was built with {built}: build the backbone and the initializer from one \
         public-peppy-libs commit"
    )]
    OtherDescription {
        model: &'static str,
        served: String,
        built: String,
    },

    #[error(transparent)]
    UpstreamMode(#[from] crate::upstream::UnknownUpstreamMode),

    #[error("build the model of {arm}")]
    ArmModel {
        arm: &'static str,
        source: srs_model::SrsError,
    },

    #[error("create a scratch dir for the collision meshes")]
    MeshesScratchDir(#[source] std::io::Error),

    #[error("write the collision meshes to {dir:?}")]
    MeshesWrite {
        dir: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("meshes dir path is not valid UTF-8: {0:?}")]
    MeshesPathNotUtf8(std::path::PathBuf),

    #[error("the self-collision governor's settings")]
    GovernorSettings(#[source] governor::GovernorError),

    #[error("build the self-collision governor")]
    Governor(#[source] governor::GovernorError),

    #[error("place the cameras the description carries")]
    CameraMounts(#[from] camera_mounts::CameraMountError),

    #[error("find the perception camera among the cameras the description carries")]
    PerceptionCamera(#[from] workspace_core::PerceptionCameraError),

    #[error("build the arm bodies the workspace holds clear of a surface")]
    WorkspaceBodies(#[from] workspace_core::surface_block::ArmBodiesError),

    #[error("joint position limits must be finite and well-ordered (lo <= hi)")]
    JointLimits,

    /// The runtime's own failures pass through unchanged rather than being
    /// re-wrapped, so a messaging or config error keeps the variant it was
    /// raised as.
    #[error(transparent)]
    Runtime(#[from] peppygen::Error),
}

/// This node's own results, distinct from `peppygen::Result`, which the runtime
/// takes at the boundary and which is what bare `Result` means in this crate.
type NodeResult<T = ()> = std::result::Result<T, NodeError>;

impl From<NodeError> for peppygen::Error {
    /// The one place this node's refusals meet the runtime's error type.
    ///
    /// A runtime error passes back unchanged; everything else is this node's own
    /// and travels as `Error::Node`, which keeps the wrapped error reachable
    /// through `Error::source` rather than flattening it to a message.
    fn from(e: NodeError) -> Self {
        match e {
            NodeError::Runtime(e) => e,
            other => peppygen::Error::Node(Box::new(other)),
        }
    }
}

/// Spawn a task meant to run for the life of the node (an inbound listener
/// or a service loop) into the backbone's supervised task set, adapting its
/// `()` output to the set's `Result` so its exit trips the fatal-first-exit
/// like any other backbone task.
fn spawn_supervised<F>(set: &mut JoinSet<Result<()>>, task: F)
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    set.spawn(async move {
        task.await;
        Ok(())
    });
}

/// The per-joint caps of one kind, each a positive finite number, or the
/// refusal `refuse` builds naming the joint (counted from 1) and what was
/// wrong with its value.
fn joint_caps(
    caps: JointVec,
    refuse: impl Fn(usize, NotPositiveFinite) -> NodeError,
) -> std::result::Result<JointVec, NodeError> {
    for (joint, cap) in caps.iter().enumerate() {
        PositiveFinite::parse(*cap).map_err(|source| refuse(joint + 1, source))?;
    }
    Ok(caps)
}

/// What the launch parameters resolve to, each parsed in `setup`, before the robot's
/// model is known.
struct Launch {
    cycle_period: Duration,
    stale_limit: Duration,
    follower_state_rate_hz: u32,
    max_joint_velocity_rad_s: JointVec,
    max_joint_acceleration_rad_s2: JointVec,
    max_stream_ee_velocity: PositiveFinite,
    start_tolerance: PositiveFinite,
    velocity_filter_cutoff_hz: f64,
    upstream_mode: UpstreamMode,
    governor: governor::Settings,
}

impl Launch {
    /// Parses every parameter: a bad launch fails here, by name, before anything is
    /// built from the value.
    fn parse(params: &Parameters) -> NodeResult<Self> {
        let cycle_period =
            period_from_hz(params.control_rate_hz, MAX_RATE_HZ).map_err(NodeError::ControlRate)?;
        let stale_limit = liveness::stale_limit(params.follower_state_rate_hz, cycle_period)
            .map_err(NodeError::FollowerStateRate)?;
        let max_joint_velocity_rad_s = joint_caps(
            [
                params.max_joint_velocity_rad_s_1,
                params.max_joint_velocity_rad_s_2,
                params.max_joint_velocity_rad_s_3,
                params.max_joint_velocity_rad_s_4,
                params.max_joint_velocity_rad_s_5,
                params.max_joint_velocity_rad_s_6,
                params.max_joint_velocity_rad_s_7,
            ],
            |joint, source| NodeError::JointVelocityCap { joint, source },
        )?;
        let max_joint_acceleration_rad_s2 = joint_caps(
            [
                params.max_joint_acceleration_rad_s2_1,
                params.max_joint_acceleration_rad_s2_2,
                params.max_joint_acceleration_rad_s2_3,
                params.max_joint_acceleration_rad_s2_4,
                params.max_joint_acceleration_rad_s2_5,
                params.max_joint_acceleration_rad_s2_6,
                params.max_joint_acceleration_rad_s2_7,
            ],
            |joint, source| NodeError::JointAccelerationCap { joint, source },
        )?;
        let max_stream_ee_velocity = PositiveFinite::parse(params.max_stream_ee_velocity_m_s)
            .map_err(NodeError::EeVelocity)?;
        let start_tolerance =
            PositiveFinite::parse(params.start_tolerance_rad).map_err(NodeError::StartTolerance)?;
        // Enforce the documented contract: the cutoff must sit below the control loop's
        // Nyquist frequency, or the low-pass does not attenuate (and above it is
        // meaningless). A hard bound at Nyquist; the node default sits well under it.
        let nyquist_hz = params.control_rate_hz as f64 / 2.0;
        if !(params.velocity_filter_cutoff_hz.is_finite()
            && params.velocity_filter_cutoff_hz > 0.0
            && params.velocity_filter_cutoff_hz < nyquist_hz)
        {
            return Err(NodeError::FilterCutoff {
                cutoff_hz: params.velocity_filter_cutoff_hz,
                nyquist_hz,
            });
        }
        let governor = governor::Settings::parse(
            params.d_stop_m,
            params.d_safe_m,
            max_joint_velocity_rad_s
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
            max_stream_ee_velocity,
            params.max_gripper_rate_frac_s,
            params.collision_governor_enabled,
        )
        .map_err(NodeError::GovernorSettings)?;
        Ok(Self {
            cycle_period,
            stale_limit,
            follower_state_rate_hz: params.follower_state_rate_hz,
            max_joint_velocity_rad_s,
            max_joint_acceleration_rad_s2,
            max_stream_ee_velocity,
            start_tolerance,
            velocity_filter_cutoff_hz: params.velocity_filter_cutoff_hz,
            upstream_mode: params.upstream_mode.parse::<UpstreamMode>()?,
            governor,
        })
    }
}

/// The node's entry point, in the shape `NodeBuilder::run` (and the test harness) take
/// once `fault` is bound: parse every parameter, serve the limb names, and, on a task
/// of its own, ask the robot's initializer for the model and run the node for it. The
/// initializer may start after this node, so `setup` returns without its answer, and a
/// robot this node cannot run raises `fault`.
pub async fn setup(params: Parameters, node_runner: Arc<NodeRunner>, fault: Fault) -> Result<()> {
    let launch = Launch::parse(&params)?;
    // Pairing timestamps read this instance's bound clock, so setpoint
    // consumers age samples on one timeline.
    peppygen::clock::init(&node_runner).await?;
    info!(
        "a follower silent for over {:?} is frozen (followers report at {} Hz, control loop {} Hz)",
        launch.stale_limit, launch.follower_state_rate_hz, params.control_rate_hz
    );
    info!("following upstream {} commands", launch.upstream_mode);
    // Governor controls are optional and exclusive: zero producers leaves
    // the launch-time band standing, more than one is a mis-wired launch.
    // At most one producer is the slot's zero_or_one cardinality: an
    // over-bound launch is rejected at validation, never seen here.
    if governor_control::bound_producer(&node_runner).is_none() {
        info!("no governor_control producer bound; the launch-time band stands");
    }
    serve_limb_names(&node_runner);
    let token = node_runner.cancellation_token().clone();
    let run = run_once_identified(node_runner, launch, fault.clone());
    tokio::spawn(
        fault.raise_on_panic("the start for the robot's model", run, move || {
            token.cancel()
        }),
    );
    Ok(())
}

/// The limb name tables, answered for the life of the node and ahead of the robot's
/// identity and the readiness gate: this node's arms fix them, so a consumer learns
/// which limbs this robot has well before it can move one.
fn serve_limb_names(node_runner: &Arc<NodeRunner>) {
    let runner = node_runner.clone();
    let token = node_runner.cancellation_token().clone();
    tokio::spawn(async move {
        serving::serve("get_limb_names", token.cancelled(), || {
            get_limb_names::handle_requests(&runner, |_request| {
                let names = limb_names();
                Ok(get_limb_names::Response::new(
                    names.arm_names,
                    names.joints_per_arm,
                    names.gripper_names,
                ))
            })
        })
        .await;
    });
}

/// Asks the robot's initializer for the model until it answers, then runs the node for
/// it. A model this node does not drive, or a description it cannot build from,
/// raises `fault`.
async fn run_once_identified(node_runner: Arc<NodeRunner>, launch: Launch, fault: Fault) {
    let Some(identified) = identity::identify(&node_runner).await else {
        return;
    };
    let ran = match identified {
        Ok(driven) => run(&node_runner, launch, driven),
        Err(refusal) => Err(refusal),
    };
    if let Err(refusal) = ran {
        let token = node_runner.cancellation_token().clone();
        fault.raise(refusal, || token.cancel());
    }
}

/// Builds the arm models and the governor from the description of `driven`, wires the
/// channels, and spawns the services answered from bringup plus the readiness-gated
/// coordination loop, every action handler and every stream listener.
fn run(node_runner: &Arc<NodeRunner>, launch: Launch, driven: Driven<ARM_DOF, ARMS>) -> NodeResult {
    let Launch {
        cycle_period,
        stale_limit,
        follower_state_rate_hz,
        max_joint_velocity_rad_s,
        max_joint_acceleration_rad_s2,
        max_stream_ee_velocity,
        start_tolerance,
        velocity_filter_cutoff_hz,
        upstream_mode,
        governor: governor_settings,
    } = launch;
    // The robot's description, held to the arms this node drives, for the life of the
    // process: the arm models, the self-collision model, the cameras and the postures all
    // read it, and the cameras and the workspace name what they read by its names.
    let driven: &'static Driven<ARM_DOF, ARMS> = Box::leak(Box::new(driven));
    let (model, description) = (driven.model(), driven.description());
    info!("driving {}", model.id());
    let ready = robot::postures_of(driven, states::READY).map_err(NodeError::Model)?;
    let home = robot::postures_of(driven, states::HOME).map_err(NodeError::Model)?;

    // The arm models (FK/IK/Jacobian/limits), each from its arm of the description:
    // its chain from its base link with the URDF's joint limits, for IK seeding,
    // trajectory sizing and the chase clamp, and its tool at the gripper's grasp
    // point, where every pose the backbone solves, reports, or caps the speed of lies.
    // The collision model walks the same chains from the same base links.
    let build_arms = || {
        PerArm::from_fn(|arm| {
            srs_model::Arm::from_description(description, &description.arms()[arm.place()]).map_err(
                |source| NodeError::ArmModel {
                    arm: LIMBS.arm_name(arm),
                    source,
                },
            )
        })
        .transpose()
    };
    let models = build_arms()?;
    info!("arm models loaded");

    // The collision model reads the description and the meshes on disk; the file-based
    // builder reads the meshes the bundled model holds, written into a per-process
    // scratch dir. A unique tempdir (not a fixed shared path) avoids a start/restart
    // race on the files; `Governor::build` and `ArmBodies::new` read them synchronously, so
    // the handle can drop right after and self-clean.
    let meshes_tmp = tempfile::tempdir().map_err(NodeError::MeshesScratchDir)?;
    model
        .write_meshes_to(meshes_tmp.path())
        .map_err(|source| NodeError::MeshesWrite {
            dir: meshes_tmp.path().to_path_buf(),
            source,
        })?;
    let meshes_dir = meshes_tmp
        .path()
        .to_str()
        .ok_or_else(|| NodeError::MeshesPathNotUtf8(meshes_tmp.path().to_path_buf()))?;

    let launch_config = governor_settings.launch_config();
    let governor = governor::Governor::build(description, meshes_dir, governor_settings)
        .map_err(NodeError::Governor)?;
    // The joint paths are checked beside the coordination loop, on a copy of
    // the governor's collision model.
    let path_checker: Arc<Mutex<dyn PathGuard>> = Arc::new(Mutex::new(governor.path_checker()));
    info!(
        "self-collision governor ready (d_stop_m={} d_safe_m={} default {})",
        launch_config.d_stop,
        launch_config.d_safe,
        if launch_config.enabled {
            "ENABLED"
        } else {
            "DISABLED"
        },
    );

    // The cameras the robot carries, each placed in the frame that
    // carries it: a mount this backbone cannot place stops bringup here.
    let mounts = Arc::new(CameraMounts::resolve(description)?);

    // Where the robot can work, judged from its design on arm models of its
    // own (the planners own theirs), its ready posture and its perception
    // camera: a design with more than one candidate stops bringup here.
    let bodies = ArmBodies::new(description, meshes_dir)?;
    let workspace = Arc::new(Workspace::new(build_arms()?, &ready, bodies, &mounts)?);

    let limits = PerArm::from_fn(|arm| models[arm].limits());
    // The chase clamps every streamed/planned target into these limits with
    // `f64::clamp`, which is total only for finite, well-ordered bounds. Refuse
    // here so a malformed URDF stops bringup, not a mid-tick clamp.
    if !limits
        .iter()
        .flatten()
        .all(|l| l.lo.is_finite() && l.hi.is_finite() && l.lo <= l.hi)
    {
        return Err(NodeError::JointLimits);
    }
    let plan_cfg = |limits| PlanConfig {
        cycle_period,
        max_joint_velocity_rad_s,
        max_joint_acceleration_rad_s2,
        stream_ee_velocity_m_s: max_stream_ee_velocity.get(),
        limits,
        start_tolerance_rad: start_tolerance.get(),
    };
    let planners = models.map(|arm, model| Planner::new(arm, model, plan_cfg(limits[arm])));

    // Per-arm channels. Listeners fill the watch slots; action handlers send
    // accepted goals; the coordinator reads all of it and, while a move runs,
    // clears that arm's command watch. The command streams are held by the
    // coordinator as their sender (read + clear); the listener keeps a clone.
    // State streams stay reader-side: the coordinator reads a receiver
    // subscribed to the listener's sender.
    let command_txs = PerArm::from_fn(|_| watch::channel(None).0);
    let gripper_command_txs = PerArm::from_fn(|_| watch::channel(None).0);
    let measured_txs = PerArm::from_fn(|_| watch::channel(None).0);
    let gripper_state_txs = PerArm::from_fn(|_| watch::channel(None).0);
    // The limb_motion services, the posture moves and execute_joint_path send
    // their requests to the coordinator here: a stop, a plan check, a grasp
    // request or the start of a joint path. The coordinator answers each on
    // the tick after it is queued.
    let (request_tx, request_rx) = mpsc::channel(8);
    // The grasp poses of the last limb_state snapshot, for the camera
    // mounts service.
    let (grasps_tx, grasps_rx) = watch::channel(None);
    let move_slots = PerArm::from_fn(|_| MoveSlot::default());
    let gripper_move_slots = PerArm::from_fn(|_| MoveSlot::default());
    // Whether each limb's follower delivers fresh state: the coordinator
    // writes these every tick, and the goal handlers refuse on them.
    let fresh = PerArm::from_fn(|_| Arc::new(AtomicBool::new(false)));
    let gripper_fresh = PerArm::from_fn(|_| Arc::new(AtomicBool::new(false)));
    let (config_tx, config_rx) = watch::channel(launch_config);

    // When a leader last streamed each limb: the goal handlers refuse a goal
    // on a limb a leader is streaming.
    let arm_activity = PerArm::from_fn(|_| LeaderActivity::default());
    let gripper_activity = PerArm::from_fn(|_| LeaderActivity::default());
    // Each arm's channels, the goals of its arm and of its gripper received
    // there and sent by the action handlers.
    let (goal_txs, channels) = PerArm::from_fn(|arm| {
        let (goal_tx, goals) = mpsc::channel(1);
        let (gripper_goal_tx, gripper_goals) = mpsc::channel(1);
        let channels = ArmChannels {
            command: command_txs[arm].clone(),
            gripper_command: gripper_command_txs[arm].clone(),
            leader: arm_activity[arm].clone(),
            gripper_leader: gripper_activity[arm].clone(),
            measured: measured_txs[arm].subscribe(),
            gripper: gripper_state_txs[arm].subscribe(),
            goals,
            move_slot: move_slots[arm].clone(),
            gripper_goals,
            gripper_move_slot: gripper_move_slots[arm].clone(),
            fresh: fresh[arm].clone(),
            gripper_fresh: gripper_fresh[arm].clone(),
        };
        ((goal_tx, gripper_goal_tx), channels)
    })
    .unzip();
    let (goal_txs, gripper_goal_txs) = goal_txs.unzip();

    // Where the cameras stand, answered for the life of the node from the
    // grasp poses the coordinator measured last, and refused until it has.
    tokio::spawn(camera_mounts::serve(
        node_runner.clone(),
        node_runner.cancellation_token().clone(),
        mounts,
        grasps_rx,
    ));

    // Where the robot can work, answered for the life of the node from the
    // design alone, so ahead of the readiness gate like the mounts.
    workspace_service::spawn(
        node_runner.clone(),
        node_runner.cancellation_token().clone(),
        workspace,
    );

    // Gate exposing actions + streaming on the robot being ready, on a task of
    // its own.
    let runner = node_runner.clone();
    let token = node_runner.cancellation_token().clone();
    tokio::spawn(async move {
        startup::wait_until_ready(&runner, &token).await;

        // The coordination loop (owns the governor, both planners, the channels;
        // streams governed setpoints once both arms report) and the action
        // handlers are all meant to run for the life of the node.
        let mut set = JoinSet::new();
        set.spawn(coordinator::run(
            runner.clone(),
            governor,
            planners,
            channels,
            request_rx,
            PathChecks::new(path_checker),
            config_rx,
            grasps_tx,
            coordinator::RunConfig {
                cycle_period,
                stale_limit,
                follower_state_rate_hz,
                velocity_filter_cutoff_hz,
                upstream_mode,
            },
            token.clone(),
        ));
        set.spawn(actions::arm::run_move_arm_joints(
            runner.clone(),
            goal_txs.clone(),
            move_slots.clone(),
            fresh.clone(),
            arm_activity.clone(),
            limits,
        ));
        set.spawn(actions::path::run_execute_joint_path(
            runner.clone(),
            move_slots.clone(),
            fresh.clone(),
            arm_activity.clone(),
            limits,
            request_tx.clone(),
        ));
        set.spawn(actions::arm::run_move_arm(
            runner.clone(),
            goal_txs.clone(),
            move_slots.clone(),
            fresh.clone(),
            arm_activity.clone(),
        ));
        // The limbs the posture moves claim and command: move_to_ready the
        // arms alone, and move_to_home the grippers, which it closes, then
        // the arms.
        let posture_arms = actions::postures::Limbs {
            goal_txs,
            move_slots,
            activity: arm_activity.clone(),
        };
        let posture_grippers = actions::postures::Limbs {
            goal_txs: gripper_goal_txs.clone(),
            move_slots: gripper_move_slots.clone(),
            activity: gripper_activity.clone(),
        };
        set.spawn(actions::postures::run_move_to_ready(
            runner.clone(),
            posture_arms.clone(),
            ready,
            fresh.clone(),
            None,
            request_tx.clone(),
        ));
        set.spawn(actions::postures::run_move_to_home(
            runner.clone(),
            posture_arms,
            home,
            fresh.clone(),
            Some(posture_grippers),
            request_tx.clone(),
        ));
        set.spawn(actions::gripper::run_move_gripper(
            runner.clone(),
            gripper_goal_txs,
            gripper_move_slots,
            gripper_fresh.clone(),
            gripper_activity.clone(),
        ));
        spawn_supervised(
            &mut set,
            actions::stop::run_stop(runner.clone(), token.clone(), request_tx.clone()),
        );
        spawn_supervised(
            &mut set,
            actions::arm::run_check_arm_move(
                runner.clone(),
                token.clone(),
                request_tx,
                fresh.clone(),
                arm_activity.clone(),
            ),
        );

        // Inbound listeners buffer the latest message into the watch slots. They
        // run under the same fatal-first-exit supervision as the rest of the backbone,
        // so a listener that dies takes the node down instead of leaving the
        // coordinator streaming on stale measured state or governor controls while
        // the node still reports healthy.
        // Exactly one upstream listener; the other slot kind is never
        // subscribed.
        match upstream_mode {
            UpstreamMode::Joints => spawn_supervised(
                &mut set,
                streams::run_joint_command_listener(runner.clone(), command_txs, arm_activity),
            ),
            UpstreamMode::Pose => spawn_supervised(
                &mut set,
                streams::run_pose_command_listener(runner.clone(), command_txs, arm_activity),
            ),
        }
        spawn_supervised(
            &mut set,
            streams::run_gripper_command_listener(
                runner.clone(),
                gripper_command_txs,
                gripper_activity,
            ),
        );
        spawn_supervised(
            &mut set,
            streams::run_joint_state_listener(runner.clone(), measured_txs),
        );
        spawn_supervised(
            &mut set,
            streams::run_gripper_state_listener(runner.clone(), gripper_state_txs),
        );
        spawn_supervised(
            &mut set,
            streams::run_governor_config_listener(runner.clone(), config_tx),
        );

        // The first task to finish is fatal: cancel the node so the daemon
        // restarts a clean process rather than running on with a dead
        // coordination loop or a missing action handler while reporting healthy.
        if let Some(joined) = set.join_next().await {
            match joined {
                Ok(Ok(())) => error!("backbone task exited; shutting node down"),
                Ok(Err(e)) => error!(error = %e, "backbone task failed; shutting node down"),
                Err(e) if e.is_panic() => {
                    error!(error = %e, "backbone task panicked; shutting node down")
                }
                Err(e) => error!(error = %e, "backbone task join failed; shutting node down"),
            }
        }
        token.cancel();
        set.shutdown().await;
    });

    Ok(())
}
