//! Node composition: setup parses every parameter, takes the per-gripper
//! instance lock, opens the CAN interface and serves readiness; the first
//! `set_model` call of the robot's initializer names the robot's model, and the
//! node then brings the motor up in the control mode of that model's mechanism
//! and wires the setpoint stream to the follow loop and the state/health
//! publishers. All
//! device and stream logic lives in the sibling modules; this is only the
//! assembly.

use crate::follow::{self, FollowConfig};
use crate::geometry::driven_finger;
use crate::hardware::{DRIVEN_MODELS, Gripper, Mechanism, PosForceLimits};
use crate::{health, stream};
use control_core::component_model::ComponentModel;
use control_core::fault::Fault;
use control_core::requested_arms::limb_at;
use control_core::serving;
use control_core::sync::lock_ignoring_poison;
use control_core::time::{RateOutOfRange, period_from_hz};
use openarm_can::{CanInterface, ReceivePass};
use peppygen::exposed_services::ready::{is_ready, set_model};
use peppygen::{NodeRunner, Parameters, Result};
use peppylib::runtime::CancellationToken;
use robot_description::{Finger, RobotDescription, UndrivenModel};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, watch};
use tracing::{error, info, warn};

// Mirrors the ROS2 reference hardware's on_activate / on_deactivate sleep durations.
const POST_ENABLE_SLEEP: Duration = Duration::from_millis(100);
const POST_DISABLE_SLEEP: Duration = Duration::from_millis(100);
const BRINGUP_RECV_US: u32 = 2000;
const ENABLE_FD: bool = true;
/// Bound on the shutdown hook that awaits the health task's final flush.
/// Hooks share one grace window and this one runs before the motor-disable
/// hook, so an unbounded wait on a stalled publish would hold the motors
/// energised until the force-kill deadline. Sized well inside even the
/// minimum 1 s grace window so the disable hook keeps most of the budget.
const HEALTH_FLUSH_TIMEOUT: Duration = Duration::from_millis(300);

/// The fastest this node drives the gripper. The motor shares the arm's
/// CAN FD bus and is paced the same way.
const MAX_RATE_HZ: u32 = 1_000;

/// Everything this node refuses to run on, or stops for: a parameter, the
/// instance lock or the CAN interface, which `setup` refuses, and the model the
/// robot's initializer names, the parameters that model's mechanism reads, or a
/// step of the bring-up, which end the node with a [`Fault`] after `setup` has
/// returned.
///
/// Named rather than stringly typed so each refusal keeps its source, and so
/// this list is the one place to read what a launch can be rejected for. It
/// exists because returning a refusal, rather than panicking it, is what runs
/// the shutdown hooks: a panic unwinds past them, leaving the motor
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
    GripperId(String),

    #[error("model {model}'s {gripper} moves no finger for the motor to drive")]
    NoFinger { model: String, gripper: String },

    #[error(transparent)]
    Limits(#[from] crate::hardware::PosForceLimitsError),

    #[error(transparent)]
    Claim(#[from] instance_lock::Error),

    #[error("cancelled before bring-up")]
    CancelledDuringBringUp,

    #[error("enable the gripper motor")]
    Enable(#[from] openarm_can::EnableFailure),

    #[error("gripper CAN")]
    Can(#[from] openarm_can::CanError),

    #[error("the follow loop never started")]
    FollowLoopNeverStarted,

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

/// What the robot's model resolves to for this gripper, built on the first
/// `set_model` call that names a model this node drives.
#[derive(Debug)]
struct Config {
    /// The robot's model id.
    model: &'static str,
    /// The mechanism of the model's gripper: the control mode the motor is
    /// opened in and its travel.
    mechanism: Mechanism,
    /// The gripper's name in the description, which names its motor on the
    /// alert wire.
    gripper_name: String,
    /// The finger the motor drives, whose opening direction signs the motor's.
    finger: Finger,
    limits: PosForceLimits,
}

/// The gripper of the arm at `gripper_id` among the arms of `description`,
/// refused naming each place with its gripper.
fn gripper_at(
    description: &RobotDescription,
    gripper_id: u8,
) -> NodeResult<&robot_description::Gripper> {
    limb_at(
        "gripper",
        "gripper_id",
        gripper_id,
        description.arms(),
        |arm| arm.gripper.name.as_str(),
    )
    .map(|arm| &arm.gripper)
    .map_err(NodeError::GripperId)
}

/// Parse every launch parameter the model leaves alone, refusing the first that
/// cannot drive a gripper.
fn parse_launch(params: &Parameters) -> NodeResult<Launch> {
    Ok(Launch {
        cycle_period: period_from_hz(params.control_rate_hz, MAX_RATE_HZ)
            .map_err(NodeError::ControlRate)?,
        state_period: period_from_hz(params.state_rate_hz, MAX_RATE_HZ)
            .map_err(NodeError::StateRate)?,
    })
}

/// The configuration of the gripper at `params.gripper_id` of a robot of
/// `model`: refused when this node drives no gripper of the model, naming the
/// models it drives, when the model has no gripper at `gripper_id` or none whose
/// motor turns a finger, or when the POS_FORCE limits do not fit the model's
/// motor.
fn configure(model: &str, params: &Parameters) -> NodeResult<Config> {
    let (model, mechanism) = DRIVEN_MODELS.model(model).map_err(NodeError::Model)?;
    let description = model.description();
    let gripper = gripper_at(&description, params.gripper_id)?;
    let finger = driven_finger(gripper)
        .ok_or_else(|| NodeError::NoFinger {
            model: model.id().to_owned(),
            gripper: gripper.name.clone(),
        })?
        .clone();
    // Checked whatever the mechanism: the ranges are properties of the
    // parameters, not of the gripper that reads them. Only the pinch
    // fingers' POS_FORCE command path puts them on the wire.
    let limits = PosForceLimits::new(
        mechanism.motor_type(),
        params.speed_rad_s,
        params.max_effort_nm,
    )?;
    Ok(Config {
        model: model.id(),
        mechanism,
        gripper_name: gripper.name.clone(),
        finger,
        limits,
    })
}

/// Cancel the node when a supervised task exits while it should still be
/// running, raising `fault`. An unsupervised dead loop leaves the node
/// reachable but inert; a dead follow loop is worse, because the health task
/// would keep judging a frozen state cache. Watching the JoinHandle means a
/// panic lands here too, not only a clean return.
fn supervise(
    task: tokio::task::JoinHandle<()>,
    what: &'static str,
    token: CancellationToken,
    fault: Fault,
) {
    tokio::spawn(async move {
        let _ = task.await;
        if token.is_cancelled() {
            return;
        }
        follow::HARD_FAULT.store(true, Ordering::SeqCst);
        fault.raise(format!("the {what} exited unexpectedly"), || token.cancel());
    });
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
/// `setup` returns before the motor is brought up, and a refusal after it raises
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
    let gripper_id = params.gripper_id;
    info!(
        "config: gripper_id={gripper_id} rate={}Hz state_rate={}Hz",
        params.control_rate_hz, params.state_rate_hz
    );

    // One instance per gripper: two instances sharing a `gripper_id` command one
    // motor id on one bus, each decoding the other's replies. The claim is an
    // exclusive lock on a file the kernel releases when this process exits by
    // any route, including SIGKILL, so a crashed instance leaves its gripper
    // free for the next start, and of two starts that race exactly one takes the
    // gripper. Same scheme as openarm_arm.
    let gripper_claim = instance_lock::claim(
        &format!("openarm_gripper_{gripper_id}"),
        node_runner.processor().bound_instance_id(),
    )
    .inspect_err(|e| error!("{e}"))?;

    // Claim-release hook, registered first so it runs last, after the
    // motor-disable hook the bring-up registers: the next start of this
    // gripper finds its motor already disabled.
    node_runner.on_shutdown(async move {
        drop(gripper_claim);
    });

    // The interface is the same whatever the robot's model, so a missing or
    // down interface refuses the launch here; the bring-up writes the control
    // mode of the model's mechanism into the motor on it.
    let can_interface = &params.can_interface;
    info!("opening CAN interface {can_interface} (FD={ENABLE_FD})");
    let interface = CanInterface::open(can_interface, ENABLE_FD)?;

    let ready = Arc::new(AtomicBool::new(false));
    let (configured_tx, configured_rx) = mpsc::unbounded_channel();
    serve_readiness(
        &node_runner,
        &params,
        ready.clone(),
        configured_tx,
        fault.clone(),
    );
    let token = node_runner.cancellation_token().clone();
    let bring_up = bring_up_once_configured(
        node_runner,
        params,
        launch,
        configured_rx,
        interface,
        ready,
        fault.clone(),
    );
    tokio::spawn(fault.raise_on_panic("the bring-up", bring_up, move || token.cancel()));
    Ok(())
}

/// Serves component_ready for as long as the node runs. `set_model` names the robot's
/// model, and the first one configures the gripper for it, which `configured` hands to
/// the bring-up; `is_ready` answers not ready until the bring-up and control wiring
/// complete, then ready. A model this node does not drive, or one other than the first
/// `set_model`'s, gets an error reply naming it and raises `fault`.
fn serve_readiness(
    node_runner: &Arc<NodeRunner>,
    params: &Parameters,
    ready: Arc<AtomicBool>,
    configured: mpsc::UnboundedSender<Config>,
    fault: Fault,
) {
    let runner = node_runner.clone();
    let params = params.clone();
    let token = node_runner.cancellation_token().clone();
    tokio::spawn(async move {
        let model = ComponentModel::default();
        let set = serving::serve("set_model", token.cancelled(), || {
            set_model::handle_requests(&runner, |request| {
                model
                    .set(
                        &request.data.model,
                        |named| configure(named, &params),
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
                    follow::HARD_FAULT.load(Ordering::SeqCst),
                    token.is_cancelled(),
                )))
            })
        });
        tokio::join!(set, answer);
    });
}

/// Waits for the configuration the first `set_model` call names, then brings the
/// gripper up for it on `interface`. A bring-up that fails raises `fault`.
async fn bring_up_once_configured(
    node_runner: Arc<NodeRunner>,
    params: Parameters,
    launch: Launch,
    mut configured: mpsc::UnboundedReceiver<Config>,
    interface: CanInterface,
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
        interface,
        &ready,
        &fault,
    )
    .await
    {
        fault.raise(refusal, || token.cancel());
    }
}

/// Brings the motor on `interface` up in the control mode of the model's mechanism
/// and wires the setpoint stream to the follow loop and the state/health
/// publishers; marks the gripper ready once the follow loop runs.
async fn bring_up(
    node_runner: &Arc<NodeRunner>,
    params: &Parameters,
    launch: Launch,
    config: Config,
    interface: CanInterface,
    ready: &AtomicBool,
    fault: &Fault,
) -> NodeResult {
    let Launch {
        cycle_period,
        state_period,
    } = launch;
    let Config {
        model,
        mechanism,
        gripper_name,
        finger,
        limits,
    } = config;

    // The resolved configuration, so a parameter that this mechanism
    // does not read is visibly absent rather than silently ignored.
    info!("config: {gripper_name} of {model} ({mechanism:?})");
    if mechanism == Mechanism::PinchFingers {
        info!(
            "config: POS_FORCE speed={} rad/s grip-force ceiling={} N*m at the shaft",
            params.speed_rad_s, params.max_effort_nm
        );
    }

    let cfg = FollowConfig { cycle_period };
    let receive = ReceivePass::new(cycle_period);

    // Hardware bringup, mirroring the ROS2 reference's on_init / on_configure
    // / on_activate: opening writes this mechanism's control mode into the
    // motor and consumes the reply, so the bus is quiet on return.
    let gripper = Gripper::open(mechanism, &finger, interface, limits)?;
    let gripper = Arc::new(Mutex::new(gripper));

    // Motor-disable hook, registered before the motor is ever enabled and
    // second overall so it runs first at shutdown (before the lock-release
    // hook above). The runtime fires it on every stop path and on a
    // bring-up error return, so the motor never stays energised.
    {
        let gripper = gripper.clone();
        node_runner.on_shutdown(async move {
            info!("shutdown: disabling motor");
            // Hold the lock across the whole disable -> settle -> drain so a
            // still-live follow loop can't interleave CAN traffic before
            // the disable ACKs are drained. Blocking sleep (not tokio) keeps the
            // guard held, which it could not be across an await.
            // Recover even if poisoned (panic in control loop)
            // so disable_all() always runs and the motor doesn't stay energised.
            let mut g = lock_ignoring_poison(&gripper);
            if let Err(e) = g.disable_all() {
                error!("disable motor: {e}");
            }
            std::thread::sleep(POST_DISABLE_SLEEP);
            if let Err(e) = g.recv_all(receive) {
                warn!("drain disable replies: {e}");
            }
        });
    }

    // The datasheet ratings resolved with the wire limits, well before the
    // motor is energised. The DM4310's configured trip derates above its
    // datasheet peak, so the registers would resolve to the datasheet anyway.
    let ratings = limits.ratings();

    // Enable and verify the motor acknowledges torque authority, then
    // return to closed (motor angle = 0.0 rad) before serving requests.
    // Errors raise the node's fault, whose stop runs the hooks above: they
    // disable the motor and release the lock. Cancellation precedes the
    // shutdown hooks and the disable hook, registered above, takes this same
    // lock, so checking under the lock closes both orders: a bring-up after
    // shutdown began bails here before enabling, and a shutdown during the
    // enable disables after it completes.
    {
        let mut g = lock_ignoring_poison(&gripper);
        if node_runner.cancellation_token().is_cancelled() {
            return Err(NodeError::CancelledDuringBringUp);
        }
        g.enable_and_confirm(
            openarm_can::ENABLE_ATTEMPTS,
            POST_ENABLE_SLEEP,
            BRINGUP_RECV_US,
        )?;
    }
    {
        let mut g = lock_ignoring_poison(&gripper);
        info!("returning to zero");
        g.command(0.0, None)?;
        g.recv_all(receive)?;
    }
    info!("gripper ready (motor confirmed enabled)");

    // Always-on gripper_states publisher: reads the motor's cached state at
    // state_rate_hz and emits the opening, effort, and effort ceiling. It
    // issues no CAN traffic of its own, so it never contends with the
    // follow loop for the bus.
    let publisher = tokio::spawn(stream::run_state_publisher(
        node_runner.clone(),
        state_period,
        gripper.clone(),
        node_runner.cancellation_token().clone(),
    ));
    supervise(
        publisher,
        "state publisher",
        node_runner.cancellation_token().clone(),
        fault.clone(),
    );

    // Motor condition telemetry and operator alerts, off the same cached
    // state. The final flush round after cancellation is only awaited if
    // a shutdown hook waits for it: the runtime awaits hooks, not plain
    // tasks racing the token. Dropping the sender (return or panic)
    // completes the receiver, so the hook cannot hang on a dead task.
    let (health_done_tx, health_done_rx) = oneshot::channel::<()>();
    let health_task = tokio::spawn({
        let runner = node_runner.clone();
        let gripper = gripper.clone();
        let cycle_period = cfg.cycle_period;
        let token = node_runner.cancellation_token().clone();
        async move {
            let _done = health_done_tx;
            health::run(runner, gripper_name, gripper, ratings, cycle_period, token).await;
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
    supervise(
        health_task,
        "health publisher",
        node_runner.cancellation_token().clone(),
        fault.clone(),
    );

    // Stream listener -> follow loop: the listener keeps the latest streamed
    // opening addressed to this gripper, the follow loop drives the motor
    // toward it.
    let (setpoint_tx, setpoint_rx) = watch::channel(None);
    let setpoints = tokio::spawn(stream::run_setpoint_listener(
        node_runner.clone(),
        setpoint_tx,
        node_runner.cancellation_token().clone(),
    ));
    supervise(
        setpoints,
        "command stream",
        node_runner.cancellation_token().clone(),
        fault.clone(),
    );
    let (follow_started_tx, follow_started_rx) = oneshot::channel::<()>();
    let follower = tokio::spawn(follow::run(
        gripper,
        setpoint_rx,
        cfg,
        node_runner.cancellation_token().clone(),
        follow_started_tx,
        fault.clone(),
    ));
    supervise(
        follower,
        "follow loop",
        node_runner.cancellation_token().clone(),
        fault.clone(),
    );

    // The ack arrives once the follow loop is actually running; a task
    // that dies first drops the sender and the error return runs the
    // disable hooks. Reporting ready any earlier would let the robot
    // gate open on a spawned-but-dead controller.
    follow_started_rx
        .await
        .map_err(|_| NodeError::FollowLoopNeverStarted)?;
    ready.store(true, Ordering::SeqCst);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model the tests' robot is.
    const MODEL: &str = "openarm_v2";

    /// A launch the node accepts, one field per parameter the manifest names.
    fn params() -> Parameters {
        Parameters {
            can_interface: "can0".to_string(),
            control_rate_hz: 100,
            gripper_id: 0,
            max_effort_nm: 3.0,
            speed_rad_s: 25.0,
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

    /// The refusal of `params` at setup, or at the first `set_model` call naming
    /// [`MODEL`] when setup takes it.
    fn refusal(params: Parameters) -> String {
        let refused = parse_launch(&params)
            .err()
            .or_else(|| configure(MODEL, &params).err())
            .expect("refused");
        reading(refused)
    }

    /// The bundled description of `model`.
    fn description_of(model: &str) -> RobotDescription {
        robot_description::model(model)
            .expect("a bundled model")
            .description()
    }

    #[test]
    fn a_launch_with_every_parameter_set_parses_and_configures() {
        let launch = parse_launch(&params()).expect("parses");
        assert_eq!(launch.cycle_period, Duration::from_millis(10));
        assert_eq!(launch.state_period, Duration::from_millis(10));
        let config = configure(MODEL, &params()).expect("configures");
        assert_eq!(config.model, MODEL);
        assert_eq!(config.mechanism, Mechanism::PinchFingers);
        assert_eq!(
            config.gripper_name,
            description_of(MODEL).arms()[0].gripper.name
        );
    }

    /// Every gripper of every model this node drives configures: its mechanism,
    /// its name on the alert wire, and the finger its motor drives, each read
    /// from the model's description.
    #[test]
    fn every_gripper_of_every_driven_model_configures_under_its_name() {
        for (model, mechanism) in DRIVEN_MODELS {
            for (place, arm) in description_of(model).arms().iter().enumerate() {
                let config = configure(
                    model,
                    &Parameters {
                        gripper_id: u8::try_from(place).expect("a place a u8 holds"),
                        ..params()
                    },
                )
                .unwrap_or_else(|e| panic!("{model} {}: {e}", arm.gripper.name));
                assert_eq!(config.mechanism, mechanism, "{model}");
                assert_eq!(config.gripper_name, arm.gripper.name, "{model}");
                assert_eq!(Some(&config.finger), driven_finger(&arm.gripper), "{model}");
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
            (
                "gripper_id",
                Parameters {
                    gripper_id: 2,
                    ..params()
                },
            ),
            (
                "speed_rad_s",
                Parameters {
                    speed_rad_s: 0.0,
                    ..params()
                },
            ),
            (
                "max_effort_nm",
                Parameters {
                    max_effort_nm: f64::NAN,
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

    /// A gripper_id past the model's arms is refused naming the value and
    /// each place the model has, with its gripper.
    #[test]
    fn an_unknown_gripper_id_names_the_grippers_of_the_model() {
        let refusal = configure(
            MODEL,
            &Parameters {
                gripper_id: 7,
                ..params()
            },
        )
        .expect_err("a place with no arm must be refused")
        .to_string();
        assert!(
            refusal.starts_with("gripper_id names 7, no gripper of this robot: "),
            "{refusal}"
        );
        for (place, arm) in description_of(MODEL).arms().iter().enumerate() {
            let named = format!("{place} (\"{}\")", arm.gripper.name);
            assert!(refusal.contains(&named), "{named} in {refusal}");
        }
    }

    /// A model this node drives no gripper of is refused naming the models it
    /// drives, whether robot_description bundles it or not.
    #[test]
    fn an_undriven_model_is_refused_naming_the_driven_ones() {
        for model in ["so101", "openarm_v3"] {
            let refused = configure(model, &params())
                .expect_err("this node drives no gripper of it")
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
}
