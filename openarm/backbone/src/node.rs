// Node composition: validates every parameter up front, builds the arm models
// and the self-collision governor, wires the channels, and spawns the services
// answered from bringup (limb names, camera mounts, workspace) plus the
// readiness-gated coordination loop, every action handler and every stream
// listener. All motion and stream logic lives in the sibling modules; this is
// only the assembly.

use crate::arm_pair::ArmPair;
use crate::types::{JointVec, Side, limb_names};

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use control_core::positive_finite::{NotPositiveFinite, PositiveFinite};
use control_core::time::{RateOutOfRange, period_from_hz};
use openarm_description::HardwareVersion;
use peppygen::consumed_topics::collision_ctrl::governor_control;
use peppygen::exposed_services::limb_state::get_limb_names;
use peppygen::{NodeRunner, Parameters, Result};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinSet;
use tracing::{error, info};

use crate::actions;
use crate::camera_mounts::{self, CameraMounts};
use crate::coordinator::{self, ArmChannels};
use crate::governor;
use crate::liveness;
use crate::planner::{PlanConfig, Planner};
use crate::serving;
use crate::startup;
use crate::streams;
use crate::streams::LeaderActivity;
use crate::upstream::UpstreamMode;
use crate::workspace::Workspace;
use crate::workspace_service;
use srs_model::chain_kinematics::EeCaps;

/// The fastest the coordination loop is driven. It paces the same 1 Mbit
/// CAN FD arms as openarm_arm; the stack runs at 100 Hz.
const MAX_RATE_HZ: u32 = 1_000;

/// Everything this node refuses to run on.
///
/// Every variant either names the parameter it rejected or carries the error it
/// came from, so this list is the one place to read what a launch can be
/// rejected for and nothing is flattened to a message on the way out. It
/// exists because returning a refusal, rather than panicking it, is what runs
/// the runtime's shutdown hooks: a panic in `setup` unwinds straight past them.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("parameter control_rate_hz")]
    ControlRate(#[source] RateOutOfRange),

    #[error("parameter follower_state_rate_hz")]
    FollowerStateRate(#[source] liveness::StaleLimitOutOfRange),

    #[error(
        "control_rate_hz ({rate_hz}) is too slow for the servo's command smoothing; \
         its cutoff needs the loop's Nyquist rate above it"
    )]
    ServoSmoothing {
        rate_hz: u32,
        #[source]
        source: control_core::filters::FilterError,
    },

    #[error("parameter max_ee_velocity_m_s")]
    EeVelocity(#[source] NotPositiveFinite),

    #[error("parameter max_ee_angular_velocity_rad_s")]
    EeAngularVelocity(#[source] NotPositiveFinite),

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

    #[error(transparent)]
    HardwareVersion(#[from] openarm_description::UnknownHardwareVersion),

    #[error(transparent)]
    UpstreamMode(#[from] crate::upstream::UnknownUpstreamMode),

    #[error("build the {side} arm model")]
    ArmModel {
        side: &'static str,
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

    #[error("build the self-collision governor")]
    Governor(#[from] governor::GovernorError),

    #[error("place the cameras the description carries")]
    CameraMounts(#[from] camera_mounts::CameraMountError),

    #[error("find the perception camera among the cameras the description carries")]
    PerceptionCamera(#[from] workspace_core::PerceptionCameraError),

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

/// Spawn a never-returning inbound listener into the backbone's supervised task set,
/// adapting its `()` output to the set's `Result` so its exit trips the
/// fatal-first-exit like any other backbone task.
fn spawn_listener<F>(set: &mut JoinSet<Result<()>>, listener: F)
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    set.spawn(async move {
        listener.await;
        Ok(())
    });
}

/// Build one side's arm model from the embedded OpenArm description: the elbow
/// singularity margin applied, and the URDF's tcp frame mounted so every pose the
/// backbone solves, reports, or caps the speed of is at the gripper's grasp point.
/// This is the single site the backbone imposes either, so the model's `limits()`
/// carry the margin for IK seeding, trajectory sizing and the chase clamp.
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

pub fn arm_model(
    version: HardwareVersion,
    side: openarm_description::Side,
) -> std::result::Result<srs_model::Arm, srs_model::SrsError> {
    srs_model::Arm::from_urdf(version.urdf(), version.base_link(side))?
        .with_lower_floor(
            version.elbow_joint_index(),
            version.elbow_singularity_floor_rad(),
        )
        .with_tool_link(version.tcp_link(side))
}

/// The node's whole runtime, in the exact shape `NodeBuilder::run` (and the
/// test harness) take: validate the parameters, build the models and the
/// governor, wire the channels, and spawn the limb-name responder plus the
/// readiness-gated coordination loop, every action handler and every stream
/// listener.
pub async fn setup(params: Parameters, node_runner: Arc<NodeRunner>) -> Result<()> {
    assemble(params, node_runner).await.map_err(Into::into)
}

async fn assemble(params: Parameters, node_runner: Arc<NodeRunner>) -> NodeResult {
    // Pairing timestamps read this instance's bound clock, so setpoint
    // consumers age samples on one timeline.
    peppygen::clock::init(&node_runner).await?;

    // Parse every parameter up front: a bad launch fails here, by name, before
    // anything downstream is built from the value.
    let cycle_period =
        period_from_hz(params.control_rate_hz, MAX_RATE_HZ).map_err(NodeError::ControlRate)?;
    let stale_limit = liveness::stale_limit(params.follower_state_rate_hz, cycle_period)
        .map_err(NodeError::FollowerStateRate)?;
    info!(
        "a follower silent for over {:?} is frozen (followers report at {} Hz, control loop {} Hz)",
        stale_limit, params.follower_state_rate_hz, params.control_rate_hz
    );
    // Built here, where a rate that cannot carry the cutoff is a refusal by
    // name; every servo move copies the value.
    let servo_smoothing =
        srs_model::chain_kinematics::command_smoothing(cycle_period).map_err(|source| {
            NodeError::ServoSmoothing {
                rate_hz: params.control_rate_hz,
                source,
            }
        })?;
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
    let max_ee_velocity =
        PositiveFinite::parse(params.max_ee_velocity_m_s).map_err(NodeError::EeVelocity)?;
    let max_ee_angular = PositiveFinite::parse(params.max_ee_angular_velocity_rad_s)
        .map_err(NodeError::EeAngularVelocity)?;
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
    // Governor controls are optional and exclusive: zero producers leaves
    // the launch-time band standing, more than one is a mis-wired launch.
    // At most one producer is the slot's zero_or_one cardinality: an
    // over-bound launch is rejected at validation, never seen here.
    if governor_control::bound_producer(&node_runner).is_none() {
        info!("no governor_control producer bound; the launch-time band stands");
    }

    // Which OpenArm generation the arms are; selects the embedded description for both
    // the srs_model arms and the bimanual collision model.
    let hardware_version: HardwareVersion = params.hardware_version.parse()?;

    // Which upstream command kind this instance follows; parsed once, and
    // only that kind's listener is spawned below.
    let upstream_mode: UpstreamMode = params.upstream_mode.parse::<UpstreamMode>()?;
    info!("following upstream {upstream_mode} commands");

    // Two arm models (FK/IK/Jacobian/limits, with the elbow singularity margin)
    // and the bimanual collision model, all from the embedded OpenArm description.
    // The per-side chain base and tcp links are facts of the generation's URDF,
    // resolved from the description rather than configured, so a v2 launch cannot
    // inherit a v1 name. The collision model walks the same chains and takes the
    // base links directly.
    let left_base = hardware_version.base_link(openarm_description::Side::Left);
    let right_base = hardware_version.base_link(openarm_description::Side::Right);
    let build_arm = |side: Side| {
        arm_model(hardware_version, side.model()).map_err(|source| NodeError::ArmModel {
            side: side.label(),
            source,
        })
    };
    let build_arms = || ArmPair::from_fn(&build_arm).transpose();
    let models = build_arms()?;
    info!("arm models loaded");

    // The collision model needs the URDF string (joint limits are irrelevant to it,
    // so no margin) and the meshes on disk; the file-based builder reads the meshes
    // materialized from the embedded description into a per-process scratch dir. A
    // unique tempdir (not a fixed shared path) avoids a start/restart race on the
    // files; `Governor::build` reads them synchronously, so the handle can drop right
    // after and self-clean.
    let meshes_tmp = tempfile::tempdir().map_err(NodeError::MeshesScratchDir)?;
    hardware_version
        .write_meshes_to(meshes_tmp.path())
        .map_err(|source| NodeError::MeshesWrite {
            dir: meshes_tmp.path().to_path_buf(),
            source,
        })?;
    let meshes_dir = meshes_tmp
        .path()
        .to_str()
        .ok_or_else(|| NodeError::MeshesPathNotUtf8(meshes_tmp.path().to_path_buf()))?;

    let governor = governor::Governor::build(
        hardware_version.urdf(),
        meshes_dir,
        left_base,
        right_base,
        params.d_stop_m,
        params.d_safe_m,
        max_joint_velocity_rad_s
            .iter()
            .copied()
            .fold(0.0_f64, f64::max),
        max_ee_velocity.get(),
        params.max_gripper_rate_frac_s,
        params.collision_governor_enabled,
    )?;
    info!(
        "self-collision governor ready (d_stop_m={} d_safe_m={} default {})",
        params.d_stop_m,
        params.d_safe_m,
        if params.collision_governor_enabled {
            "ENABLED"
        } else {
            "DISABLED"
        },
    );

    // The cameras the generation carries, each placed in the frame that
    // carries it: a mount this backbone cannot place stops bringup here.
    let mounts = Arc::new(CameraMounts::resolve(hardware_version)?);

    // Where the robot can work, judged from its design on a pair of arm
    // models of its own (the planners own theirs) and its perception
    // camera: a design with more than one candidate stops bringup here.
    let workspace = Arc::new(Workspace::new(build_arms()?, &mounts)?);

    let limits = ArmPair::from_fn(|side| models[side].limits());
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
        smoothing: servo_smoothing,
        max_joint_velocity_rad_s,
        max_joint_acceleration_rad_s2,
        ee: EeCaps {
            linear_m_s: max_ee_velocity.get(),
            angular_rad_s: max_ee_angular.get(),
        },
        limits,
    };
    let planners = models.map(|side, model| Planner::new(side, model, plan_cfg(limits[side])));

    // Per-arm channels. Listeners fill the watch slots; action handlers send
    // accepted goals; the coordinator reads all of it and, while a move runs,
    // clears that side's command watch. The command streams are held by the
    // coordinator as their sender (read + clear); the listener keeps a clone.
    // State streams stay reader-side: the coordinator reads a receiver
    // subscribed to the listener's sender.
    let command_txs = ArmPair::from_fn(|_| watch::channel(None).0);
    let gripper_command_txs = ArmPair::from_fn(|_| watch::channel(None).0);
    let measured_txs = ArmPair::from_fn(|_| watch::channel(None).0);
    let gripper_state_txs = ArmPair::from_fn(|_| watch::channel(None).0);
    let (goal_txs, goal_rxs) = ArmPair::from_fn(|_| mpsc::channel(1)).unzip();
    let (gripper_goal_txs, gripper_goal_rxs) = ArmPair::from_fn(|_| mpsc::channel(1)).unzip();
    // The limb_motion services and the posture moves send their requests to
    // the coordinator here: a stop, a plan check or a grasp request. The
    // coordinator answers each on the tick after it is queued.
    let (request_tx, request_rx) = mpsc::channel(8);
    // The grasp poses of the last limb_state snapshot, for the camera
    // mounts service.
    let (grasps_tx, grasps_rx) = watch::channel(None);
    let busy = ArmPair::from_fn(|_| Arc::new(AtomicBool::new(false)));
    let gripper_busy = ArmPair::from_fn(|_| Arc::new(AtomicBool::new(false)));
    // Whether each limb's follower delivers fresh state: the coordinator
    // writes these every tick, and the goal handlers refuse on them.
    let fresh = ArmPair::from_fn(|_| Arc::new(AtomicBool::new(false)));
    let gripper_fresh = ArmPair::from_fn(|_| Arc::new(AtomicBool::new(false)));
    let (config_tx, config_rx) = watch::channel(streams::GovernorConfig {
        enabled: params.collision_governor_enabled,
        d_stop: params.d_stop_m,
        d_safe: params.d_safe_m,
        max_ee_velocity_m_s: max_ee_velocity.get(),
        max_gripper_rate_frac_s: params.max_gripper_rate_frac_s,
    });

    // When a leader last streamed each limb: the goal handlers refuse a goal
    // on a limb a leader is streaming.
    let arm_activity = ArmPair::from_fn(|_| LeaderActivity::default());
    let gripper_activity = ArmPair::from_fn(|_| LeaderActivity::default());
    let channels = goal_rxs
        .zip(gripper_goal_rxs)
        .map(|side, (goals, gripper_goals)| ArmChannels {
            command: command_txs[side].clone(),
            gripper_command: gripper_command_txs[side].clone(),
            leader: arm_activity[side].clone(),
            gripper_leader: gripper_activity[side].clone(),
            measured: measured_txs[side].subscribe(),
            gripper: gripper_state_txs[side].subscribe(),
            goals,
            busy: busy[side].clone(),
            gripper_goals,
            gripper_busy: gripper_busy[side].clone(),
            fresh: fresh[side].clone(),
            gripper_fresh: gripper_fresh[side].clone(),
        });

    // The limb name tables, answered for the life of the node and ahead of the
    // readiness gate below: they are fixed, so a consumer learns which limbs
    // this robot has well before it can move one.
    {
        let runner = node_runner.clone();
        let token = node_runner.cancellation_token().clone();
        tokio::spawn(async move {
            serving::serve("get_limb_names", &token, || {
                get_limb_names::handle_next_request(&runner, |_request| {
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

    // Gate exposing actions + streaming on the robot being ready, in a spawned
    // task so this setup closure returns promptly for the health probe.
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
            config_rx,
            grasps_tx,
            coordinator::RunConfig {
                cycle_period,
                stale_limit,
                follower_state_rate_hz: params.follower_state_rate_hz,
                velocity_filter_cutoff_hz: params.velocity_filter_cutoff_hz,
                upstream_mode,
            },
            token.clone(),
        ));
        set.spawn(actions::arm::run_move_arm_joints(
            runner.clone(),
            goal_txs.clone(),
            busy.clone(),
            fresh.clone(),
            arm_activity.clone(),
            limits,
        ));
        set.spawn(actions::arm::run_move_arm(
            runner.clone(),
            goal_txs.clone(),
            busy.clone(),
            fresh.clone(),
            arm_activity.clone(),
        ));
        set.spawn(actions::postures::run_move_to_ready(
            runner.clone(),
            goal_txs.clone(),
            busy.clone(),
            fresh.clone(),
            arm_activity.clone(),
            request_tx.clone(),
        ));
        set.spawn(actions::postures::run_move_to_home(
            runner.clone(),
            goal_txs,
            busy,
            fresh.clone(),
            arm_activity.clone(),
            request_tx.clone(),
        ));
        set.spawn(actions::gripper::run_move_gripper(
            runner.clone(),
            gripper_goal_txs,
            gripper_busy,
            gripper_fresh.clone(),
            gripper_activity.clone(),
        ));
        set.spawn(actions::stop::run_stop(runner.clone(), request_tx.clone()));
        set.spawn(actions::arm::run_check_arm_move(
            runner.clone(),
            request_tx,
            fresh.clone(),
            arm_activity.clone(),
        ));

        // Inbound listeners buffer the latest message into the watch slots. They
        // run under the same fatal-first-exit supervision as the rest of the backbone,
        // so a listener that dies takes the node down instead of leaving the
        // coordinator streaming on stale measured state or governor controls while
        // the node still reports healthy.
        // Exactly one upstream listener; the other slot kind is never
        // subscribed.
        match upstream_mode {
            UpstreamMode::Joints => spawn_listener(
                &mut set,
                streams::run_joint_command_listener(runner.clone(), command_txs, arm_activity),
            ),
            UpstreamMode::Pose => spawn_listener(
                &mut set,
                streams::run_pose_command_listener(runner.clone(), command_txs, arm_activity),
            ),
        }
        spawn_listener(
            &mut set,
            streams::run_gripper_command_listener(
                runner.clone(),
                gripper_command_txs,
                gripper_activity,
            ),
        );
        spawn_listener(
            &mut set,
            streams::run_joint_state_listener(runner.clone(), measured_txs),
        );
        spawn_listener(
            &mut set,
            streams::run_gripper_state_listener(runner.clone(), gripper_state_txs),
        );
        spawn_listener(
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
