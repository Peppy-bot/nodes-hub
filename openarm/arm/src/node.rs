//! Node composition: setup parses every parameter, takes the per-arm instance
//! lock, opens the CAN interface and serves readiness; the first `set_model` call
//! of the robot's initializer names the robot's model, and the node then builds
//! the srs_model arm of that model, brings up the motors, and wires the control
//! loop to the stream tasks. All control, health, and stream logic lives in the
//! sibling modules; this is only the assembly.

use crate::control::{self, ControlConfig};
use crate::friction;
use crate::health;
use crate::stream;
use control_core::component_model::ComponentModel;
use control_core::fault::Fault;
use control_core::requested_arms::limb_at;
use control_core::serving;
use control_core::sync::lock_ignoring_poison;
use control_core::time::{RateOutOfRange, period_from_hz};
use openarm_can::{ARM_DOF, ArmCan, CanInterface};
use peppygen::exposed_services::ready::{is_ready, set_model};
use peppygen::{NodeRunner, Parameters, Result};
use robot_description::{RobotDescription, UndrivenModel};
use srs_model::nalgebra::Isometry3;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, watch};
use tracing::{error, info, warn};

// Sleep durations chosen to match ROS2 enactic/openarm_ros2 v10_simple_hardware behaviour.
const POST_ENABLE_SLEEP: Duration = Duration::from_millis(100);
const BRINGUP_RECV_US: u32 = 500;
const ENABLE_FD: bool = true;
/// Bound on the shutdown hook that awaits the health task's final flush.
/// Hooks share one grace window and this one runs before the motor-disable
/// hook, so an unbounded wait on a stalled publish would hold the motors
/// energised until the force-kill deadline. Sized well inside even the
/// minimum 1 s grace window so the disable hook keeps most of the budget.
const HEALTH_FLUSH_TIMEOUT: Duration = Duration::from_millis(300);
/// The fastest this node drives the arm. The motors sit on a CAN FD bus that
/// cannot service a faster cycle.
const MAX_RATE_HZ: u32 = 1_000;

/// Everything this node refuses to run on, or stops for: a parameter, the
/// instance lock or the CAN interface, which `setup` refuses, and the model the
/// robot's initializer names or a step of the bring-up, which end the node with
/// a [`Fault`] after `setup` has returned.
///
/// Named rather than stringly typed so each refusal keeps its source, and so
/// this list is the one place to read what a launch can be rejected for. It
/// exists because returning a refusal, rather than panicking it, is what runs
/// the shutdown hooks: a panic unwinds past them, leaving the motors
/// energised.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("parameter control_rate_hz")]
    ControlRate(#[source] RateOutOfRange),

    #[error("parameter state_rate_hz")]
    StateRate(#[source] RateOutOfRange),

    #[error(transparent)]
    Model(UndrivenModel),

    #[error("{0}")]
    ArmId(String),

    #[error(
        "model {model}'s {arm} has {joints} joints, and this node drives {ARM_DOF}, one on each \
         motor of its CAN interface"
    )]
    ArmShape {
        model: String,
        arm: String,
        joints: usize,
    },

    #[error("build the {model} model of {arm} from base '{base}'")]
    ArmModel {
        model: String,
        arm: String,
        base: String,
        source: srs_model::SrsError,
    },

    #[error(transparent)]
    Claim(#[from] instance_lock::Error),

    #[error("open the CAN interface")]
    CanOpen(#[from] openarm_can::CanError),

    #[error("cancelled before bring-up")]
    CancelledDuringBringUp,

    #[error(transparent)]
    Ratings(#[from] health::MisScaledReadings),

    #[error("enable the motors")]
    Enable(#[from] openarm_can::EnableFailure),

    /// `JoinError` says itself whether the thread panicked or was cancelled,
    /// so no prefix asserts a cause this node cannot know.
    #[error(transparent)]
    BringUp(#[from] tokio::task::JoinError),

    #[error("the control loop never started")]
    ControlLoopNeverStarted,

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

/// What the launch parameters resolve to, parsed in `setup`, before any device
/// contact and before the robot's model is known.
#[derive(Debug)]
struct Launch {
    cycle_period: Duration,
    state_period: Duration,
}

/// What the robot's model resolves to for this arm, built on the first readiness
/// call that names a model this node drives.
#[derive(Debug)]
struct Config {
    /// The robot's model id.
    model: &'static str,
    /// The description of the robot's model.
    description: RobotDescription,
    /// The arm at `arm_id` among the description's arms, of [`ARM_DOF`] joints. Its
    /// name names its motors on the alert wire: "left_arm" extends to "left_arm j2"
    /// per joint.
    arm: robot_description::Arm,
    friction: friction::Params,
}

/// The arm's motors on the CAN interface `setup` opened, and the sender the control task
/// signals once it has disabled them.
struct Motors {
    can: Arc<Mutex<ArmCan>>,
    disabled: oneshot::Sender<()>,
}

/// The arm at `arm_id` among the arms of `model`'s description, refused naming each
/// place with its arm, or when it holds other than [`ARM_DOF`] joints.
fn arm_at(
    description: &RobotDescription,
    model: &str,
    arm_id: u8,
) -> NodeResult<robot_description::Arm> {
    let arm = limb_at("arm", "arm_id", arm_id, description.arms(), |arm| {
        arm.name.as_str()
    })
    .map_err(NodeError::ArmId)?;
    if arm.joints.len() != ARM_DOF {
        return Err(NodeError::ArmShape {
            model: model.to_owned(),
            arm: arm.name.clone(),
            joints: arm.joints.len(),
        });
    }
    Ok(arm.clone())
}

/// Parse every launch parameter, refusing the first that cannot drive an arm.
fn parse_launch(params: &Parameters) -> NodeResult<Launch> {
    Ok(Launch {
        cycle_period: period_from_hz(params.control_rate_hz, MAX_RATE_HZ)
            .map_err(NodeError::ControlRate)?,
        state_period: period_from_hz(params.state_rate_hz, MAX_RATE_HZ)
            .map_err(NodeError::StateRate)?,
    })
}

/// The configuration of the arm at `arm_id` of a robot of `model`: refused when
/// this node drives no arm of the model, naming the models it drives, or when the
/// model has no arm of [`ARM_DOF`] joints at `arm_id`.
fn configure(model: &str, arm_id: u8) -> NodeResult<Config> {
    let (model, friction) = friction::DRIVEN_MODELS
        .model(model)
        .map_err(NodeError::Model)?;
    let description = model.description();
    let arm = arm_at(&description, model.id(), arm_id)?;
    Ok(Config {
        model: model.id(),
        description,
        arm,
        friction,
    })
}

/// Readiness as the initializer sees it. A latched fault or a started
/// shutdown revokes it: peppy keeps services reachable while the shutdown
/// hooks run, so without the cancellation term a leader polling during a
/// bounce would be told a limb is ready while its motor is being disabled.
fn reports_ready(brought_up: bool, hard_faulted: bool, shutting_down: bool) -> bool {
    brought_up && !hard_faulted && !shutting_down
}

/// The runtime's entry point, in the shape `NodeBuilder::run` takes once `fault`
/// is bound: the setup runs as [`NodeError`] and is converted once, here, so every
/// step inside can use `?` on its own failures. The robot's initializer, which
/// names the model with its first `set_model` call, may start after this node, so
/// `setup` returns before the arm is brought up, and a refusal after it raises
/// `fault`.
pub async fn setup(params: Parameters, node_runner: Arc<NodeRunner>, fault: Fault) -> Result<()> {
    assemble(params, node_runner, fault)
        .await
        .map_err(Into::into)
}

async fn assemble(params: Parameters, node_runner: Arc<NodeRunner>, fault: Fault) -> NodeResult {
    // Pairing timestamps read this instance's bound clock, so state
    // consumers age samples on one timeline.
    peppygen::clock::init(&node_runner).await?;

    // Parse every parameter up front (parse, don't validate): a bad launch
    // fails here with the reason, before touching the CAN bus.
    let launch = parse_launch(&params)?;
    let arm_id = params.arm_id;
    info!("config: arm_id={arm_id} rate={}Hz", params.control_rate_hz);

    // One instance per arm: two instances sharing an `arm_id` command one arm's
    // motors, each decoding the other's replies. The claim is an exclusive lock
    // on a file the kernel releases when this process exits by any route,
    // including SIGKILL, so a crashed instance leaves its arm free for the next
    // start, and of two starts that race exactly one takes the arm.
    let arm_claim = instance_lock::claim(
        &format!("openarm_arm_{arm_id}"),
        node_runner.processor().bound_instance_id(),
    )
    .inspect_err(|e| error!("{e}"))?;

    // Shutdown: the claim outlives the motors. The control task is the sole
    // motor writer and signals `shutdown_tx` once they are disabled; this hook
    // waits for that, then drops the claim, so the next start of this arm finds
    // the motors already off. Registered here, before the motors are ever
    // enabled, it also covers a bringup that ends early or never starts:
    // dropping `shutdown_tx` completes `shutdown_rx`, so the hook runs.
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    node_runner.on_shutdown(async move {
        let _ = shutdown_rx.await;
        drop(arm_claim);
    });

    // Hardware bringup: sequence mirrors ROS2 v10_simple_hardware on_init/on_activate.
    // Arm motor lineup + CAN addressing are identical across generations; open()
    // registers them, so a missing or down interface refuses the launch here,
    // before the robot's model is known.
    let can_interface = &params.can_interface;
    info!("opening CAN interface {can_interface} (FD={ENABLE_FD})");
    let arm = Arc::new(Mutex::new(ArmCan::open(CanInterface::open(
        can_interface,
        ENABLE_FD,
    )?)?));

    // Motor-disable hook, registered before the motors are ever enabled
    // so every stop path disables them, including a cancellation part way
    // through bring-up. The control loop's own disable on a clean stop runs
    // first and this is then a no-op on already-disabled motors.
    {
        let arm = arm.clone();
        node_runner.on_shutdown(async move {
            let mut a = lock_ignoring_poison(&arm);
            if let Err(e) = a.disable_all() {
                error!("disable motors: {e}");
            }
        });
    }

    let ready = Arc::new(AtomicBool::new(false));
    let (configured_tx, configured_rx) = mpsc::unbounded_channel();
    serve_readiness(
        &node_runner,
        arm_id,
        ready.clone(),
        configured_tx,
        fault.clone(),
    );
    let token = node_runner.cancellation_token().clone();
    let motors = Motors {
        can: arm,
        disabled: shutdown_tx,
    };
    let bring_up = bring_up_once_configured(
        node_runner,
        params,
        launch,
        configured_rx,
        motors,
        ready,
        fault.clone(),
    );
    tokio::spawn(fault.raise_on_panic("the bring-up", bring_up, move || token.cancel()));
    Ok(())
}

/// Serves component_ready for as long as the node runs. `set_model` names the robot's
/// model, and the first one configures the arm for it, which `configured` hands to
/// the bring-up; `is_ready` answers not ready until the bring-up and control wiring
/// complete, then ready. A model this node does not drive, or one other than the first
/// `set_model`'s, gets an error reply naming it and raises `fault`.
fn serve_readiness(
    node_runner: &Arc<NodeRunner>,
    arm_id: u8,
    ready: Arc<AtomicBool>,
    configured: mpsc::UnboundedSender<Config>,
    fault: Fault,
) {
    let runner = node_runner.clone();
    let token = node_runner.cancellation_token().clone();
    tokio::spawn(async move {
        let model = ComponentModel::default();
        let set = serving::serve("set_model", token.cancelled(), || {
            set_model::handle_requests(&runner, |request| {
                model
                    .set(
                        &request.data.model,
                        |named| configure(named, arm_id),
                        |config| {
                            let _ = configured.send(config);
                        },
                        &fault,
                        || token.cancel(),
                    )
                    .map_err(|reply| {
                        set_model::Failure::new(set_model::FailureKind::Rejected, reply)
                    })
            })
        });
        let answer = serving::serve("is_ready", token.cancelled(), || {
            is_ready::handle_requests(&runner, |_request| {
                Ok(is_ready::Response::new(reports_ready(
                    ready.load(Ordering::SeqCst),
                    control::HARD_FAULT.load(Ordering::SeqCst),
                    token.is_cancelled(),
                )))
            })
        });
        tokio::join!(set, answer);
    });
}

/// Waits for the configuration the first `set_model` call names, then brings the arm's
/// `motors` up for it. A bring-up that fails raises `fault`.
async fn bring_up_once_configured(
    node_runner: Arc<NodeRunner>,
    params: Parameters,
    launch: Launch,
    mut configured: mpsc::UnboundedReceiver<Config>,
    motors: Motors,
    ready: Arc<AtomicBool>,
    fault: Fault,
) {
    let token = node_runner.cancellation_token().clone();
    let config = tokio::select! {
        _ = token.cancelled() => return,
        config = configured.recv() => match config {
            Some(config) => config,
            None => return,
        },
    };
    if let Err(refusal) = bring_up(
        &node_runner,
        &params,
        launch,
        config,
        motors,
        &ready,
        &fault,
    )
    .await
    {
        fault.raise(refusal, || token.cancel());
    }
}

/// Builds the arm's model, brings up its `motors`, and wires the control loop to the
/// stream tasks; marks the arm ready once the control loop runs.
async fn bring_up(
    node_runner: &Arc<NodeRunner>,
    params: &Parameters,
    launch: Launch,
    config: Config,
    motors: Motors,
    ready: &AtomicBool,
    fault: &Fault,
) -> NodeResult {
    let Launch {
        cycle_period,
        state_period,
    } = launch;
    let Config {
        model: model_id,
        description,
        arm,
        friction,
    } = config;
    let arm_id = params.arm_id;
    let base_link = &arm.base_link;

    // The srs_model arm of the description's arm: forward kinematics for the
    // in-process gravity/Coriolis feedforward, plus the URDF's joint limits off the
    // same parsed chain. Its tool at the grasp point moves the end-effector frame,
    // not the masses the feedforward weighs. A non-SRS or short chain from the
    // arm's base link errors here.
    let model = srs_model::Arm::from_description(&description, &arm).map_err(|source| {
        NodeError::ArmModel {
            model: model_id.to_owned(),
            arm: arm.name.clone(),
            base: base_link.clone(),
            source,
        }
    })?;
    info!("model loaded ({model_id} {}, base '{base_link}')", arm.name);

    // Gravity acts along world -Z, so it is only correct if the URDF carries the
    // mount tree above base_link to orient that frame. We do not force one (a
    // base-rooted URDF legitimately evaluates gravity in the base frame), so log
    // which frame is in play: identity mount means base_link is the URDF root.
    if model.base_from_world() == Isometry3::identity() {
        warn!(
            "no world->base mount tree above '{base_link}': gravity/Coriolis evaluated in \
             the base frame (correct only if base_link's frame is gravity-aligned)"
        );
    } else {
        info!("mount tree resolved: gravity/Coriolis evaluated in the world frame");
    }

    let kp = [
        params.kp1, params.kp2, params.kp3, params.kp4, params.kp5, params.kp6, params.kp7,
    ];
    let kd = [
        params.kd1, params.kd2, params.kd3, params.kd4, params.kd5, params.kd6, params.kd7,
    ];
    info!("config: arm_id={arm_id} ({})", arm.name);
    info!("config: kp={kp:?} kd={kd:?}");
    // The arm's name names its motors on the alert wire.
    let alert_source = arm.name.clone();

    let Motors {
        can: arm,
        disabled: shutdown_tx,
    } = motors;

    // Registers first: the reads are the slowest part of bring-up and ask
    // the motors nothing but their own configuration, so doing them while
    // the arm is still limp keeps it unenergised until the moment the
    // control loop is ready to hold it. The CAN calls block, so the whole
    // mutex-guarded sequence runs on a blocking thread rather than
    // starving this worker.
    let ratings = {
        let arm = arm.clone();
        let token = node_runner.cancellation_token().clone();
        tokio::task::spawn_blocking(move || -> NodeResult<_> {
            let mut a = lock_ignoring_poison(&arm);
            // spawn_blocking cannot be aborted, so a cancellation during
            // bring-up leaves this queued task to run. Cancellation
            // precedes the shutdown hooks and the disable hook, registered
            // above, takes this same lock, so checking under the lock
            // closes both orders: a post-shutdown start bails here before
            // enabling, and a shutdown during bring-up disables after the
            // enable completes.
            if token.is_cancelled() {
                return Err(NodeError::CancelledDuringBringUp);
            }
            let ratings = health::resolve_ratings(&mut a, BRINGUP_RECV_US)?;
            enable_motors(&mut a)?;
            Ok(ratings)
        })
        .await??
    };
    info!("arm ready (every motor confirmed enabled)");

    let cfg = ControlConfig {
        kp,
        kd,
        cycle_period,
        friction,
        limits: model.limits(),
        ratings,
    };

    // Stream plumbing: the listener keeps the latest governed setpoint for
    // the control loop, and the publishers emit the measured joint state
    // and the per-motor health at their own rates (the backbone consumes
    // the state; any motor_health consumer the launcher wires gets the
    // health).
    let (governed_tx, governed_rx) = watch::channel(None);
    let (measured_tx, measured_rx) = watch::channel(None);
    let (health_tx, health_rx) = watch::channel(None);
    let listener = tokio::spawn(stream::run_governed_setpoint_listener(
        node_runner.clone(),
        governed_tx,
        node_runner.cancellation_token().clone(),
    ));
    let publisher = tokio::spawn(stream::run_state_publisher(
        node_runner.clone(),
        state_period,
        measured_rx,
    ));
    // The final flush round after cancellation is only awaited if a
    // shutdown hook waits for it: the runtime awaits hooks, not plain
    // tasks racing the token. Dropping the sender (return or panic)
    // completes the receiver, so the hook cannot hang on a dead task.
    let (health_done_tx, health_done_rx) = oneshot::channel::<()>();
    let health_publisher = tokio::spawn({
        let runner = node_runner.clone();
        let token = node_runner.cancellation_token().clone();
        async move {
            let _done = health_done_tx;
            health::run_publisher(runner, alert_source, health_rx, token).await;
        }
    });
    node_runner.on_shutdown(async move {
        if tokio::time::timeout(HEALTH_FLUSH_TIMEOUT, health_done_rx)
            .await
            .is_err()
        {
            warn!("health flush missed its shutdown budget; disabling without it");
        }
    });
    // Cancel the node the moment any stream task stops: a dead listener
    // or publisher would otherwise hold the arm silently while is_ready
    // stays true (same supervision as the sim followers). A stop before
    // the token was cancelled is the fault, not a reaction to shutdown,
    // so raise it and exit as failed.
    {
        let token = node_runner.cancellation_token().clone();
        let fault = fault.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = listener => {}
                _ = publisher => {}
                _ = health_publisher => {}
            }
            if token.is_cancelled() {
                return;
            }
            control::HARD_FAULT.store(true, Ordering::SeqCst);
            fault.raise("a stream task exited unexpectedly", || token.cancel());
        });
    }
    let wiring = stream::StreamWiring {
        governed: governed_rx,
        measured: measured_tx,
        health: health_tx,
    };

    // Single control task (the only motor writer): follows the governed
    // setpoint with in-process feedforward and a final limit clamp, and on
    // shutdown disables the motors.
    let (control_started_tx, control_started_rx) = oneshot::channel::<()>();
    control::spawn(
        node_runner,
        arm.clone(),
        cfg,
        model,
        wiring,
        control::Signals {
            started: control_started_tx,
            stopped: shutdown_tx,
            fault: fault.clone(),
        },
    );
    // The ack arrives once the control loop is actually running; a task
    // that dies first drops the sender and the error return runs the
    // disable hooks. Reporting ready any earlier would let the robot
    // gate open on a spawned-but-dead controller.
    control_started_rx
        .await
        .map_err(|_| NodeError::ControlLoopNeverStarted)?;
    ready.store(true, Ordering::SeqCst);

    Ok(())
}

/// Enable the motors, verifying each acknowledges torque authority and
/// retrying stragglers; readiness is refused naming the motors that never
/// confirm. Blocking, so a stop cannot land mid-sequence with the motors
/// half enabled.
fn enable_motors(arm: &mut ArmCan) -> std::result::Result<(), openarm_can::EnableFailure> {
    arm.enable_and_confirm(
        openarm_can::ENABLE_ATTEMPTS,
        POST_ENABLE_SLEEP,
        BRINGUP_RECV_US,
    )?;
    // One whole state pass so `get_state()` is real before the control loop's
    // first tick; the confirmation's own decode may be several hundred
    // milliseconds old by now. A pass that stopped at the first gap between
    // replies would leave the later joints reading as never-heard-from.
    arm.refresh_state(BRINGUP_RECV_US)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::friction::DRIVEN_MODELS;

    /// The model the tests' robot is.
    const MODEL: &str = "openarm_v2";

    /// A launch the node accepts, one field per parameter the manifest names.
    fn params() -> Parameters {
        Parameters {
            arm_id: 0,
            can_interface: "can0".to_string(),
            control_rate_hz: 100,
            kd1: 1.0,
            kd2: 1.0,
            kd3: 1.0,
            kd4: 1.0,
            kd5: 1.0,
            kd6: 1.0,
            kd7: 1.0,
            kp1: 10.0,
            kp2: 10.0,
            kp3: 10.0,
            kp4: 10.0,
            kp5: 10.0,
            kp6: 10.0,
            kp7: 10.0,
            state_rate_hz: 100,
        }
    }

    /// A refusal as an operator reads it: the error and its source.
    fn reading(error: NodeError) -> String {
        // The source carries the reason; the variant names the parameter.
        match std::error::Error::source(&error) {
            Some(source) => format!("{error}: {source}"),
            None => error.to_string(),
        }
    }

    fn refusal(params: Parameters) -> String {
        reading(parse_launch(&params).expect_err("refused"))
    }

    /// The bundled description of `model`.
    fn description_of(model: &str) -> RobotDescription {
        robot_description::model(model)
            .expect("a bundled model")
            .description()
    }

    #[test]
    fn a_launch_with_every_parameter_set_parses() {
        let launch = parse_launch(&params()).expect("parses");
        assert_eq!(launch.cycle_period, Duration::from_millis(10));
        assert_eq!(launch.state_period, Duration::from_millis(10));
    }

    /// Every arm of every model this node drives configures: the description's arm
    /// at its place, which names its motors on the alert wire, with its model's
    /// friction.
    #[test]
    fn every_arm_of_every_driven_model_configures_under_its_name() {
        for (model, friction) in DRIVEN_MODELS {
            for (place, arm) in description_of(model).arms().iter().enumerate() {
                let arm_id = u8::try_from(place).expect("a place a u8 holds");
                let config = configure(model, arm_id)
                    .unwrap_or_else(|e| panic!("{model} {}: {e}", arm.name));
                assert_eq!(config.model, model);
                assert_eq!(&config.arm, arm, "{model}");
                assert_eq!(config.friction, friction, "{model}");
                assert_eq!(config.description.urdf(), description_of(model).urdf());
            }
        }
    }

    #[test]
    fn every_refusal_names_its_parameter() {
        let cases = [
            (
                "control_rate_hz",
                Parameters {
                    control_rate_hz: 0,
                    ..params()
                },
            ),
            (
                "control_rate_hz",
                Parameters {
                    control_rate_hz: MAX_RATE_HZ + 1,
                    ..params()
                },
            ),
            (
                "state_rate_hz",
                Parameters {
                    state_rate_hz: 0,
                    ..params()
                },
            ),
        ];
        for (parameter, params) in cases {
            let refused = refusal(params);
            assert!(
                refused.contains(parameter),
                "a refusal must name {parameter}: {refused}"
            );
        }
    }

    #[test]
    fn readiness_needs_bringup_and_is_revoked_by_a_fault_or_a_shutdown() {
        assert!(reports_ready(true, false, false));
        // Each revoking condition alone is enough, and bringup is required.
        assert!(!reports_ready(false, false, false), "never brought up");
        assert!(!reports_ready(true, true, false), "latched fault");
        assert!(!reports_ready(true, false, true), "shutdown started");
    }

    #[test]
    fn a_shutdown_revokes_readiness_before_the_disable_hook_finishes() {
        // Services stay reachable while shutdown hooks run, so a limb that is
        // still brought up and unfaulted must stop reporting ready the moment
        // cancellation fires, not when the motor is finally disabled.
        assert!(!reports_ready(true, false, true));
    }

    /// An arm_id past the model's arms is refused naming the value and each place
    /// the model has, with its arm.
    #[test]
    fn an_unknown_arm_id_names_the_arms_of_the_model() {
        let refusal =
            reading(configure(MODEL, 7).expect_err("a place with no arm must be refused"));
        let places: Vec<String> = description_of(MODEL)
            .arms()
            .iter()
            .enumerate()
            .map(|(place, arm)| format!("{place} (\"{}\")", arm.name))
            .collect();
        assert_eq!(
            refusal,
            format!(
                "arm_id names 7, no arm of this robot: its arms are {}",
                places.join(" and ")
            )
        );
    }

    /// An arm of other than this node's joints is refused naming the model, the arm
    /// and both counts.
    #[test]
    fn an_arm_of_other_joints_is_refused_naming_both_counts() {
        let so101 = description_of("so101");
        let arm = &so101.arms()[0];
        let refusal = arm_at(&so101, "so101", 0)
            .expect_err("an arm of other joints must be refused")
            .to_string();
        assert_eq!(
            refusal,
            format!(
                "model so101's {} has {} joints, and this node drives {ARM_DOF}, one on each \
                 motor of its CAN interface",
                arm.name,
                arm.joints.len()
            )
        );
    }

    /// A model this node drives no arm of is refused naming the models it drives,
    /// whether robot_description bundles it or not.
    #[test]
    fn an_undriven_model_is_refused_naming_the_driven_ones() {
        for model in ["so101", "openarm_v3"] {
            let refused = configure(model, 0)
                .expect_err("this node drives no arm of it")
                .to_string();
            assert!(
                refused.contains(&format!("model {model} is no model this node drives")),
                "{refused}"
            );
            for (driven, _) in DRIVEN_MODELS {
                assert!(refused.contains(&format!("{driven:?}")), "{refused}");
            }
        }
    }
}
