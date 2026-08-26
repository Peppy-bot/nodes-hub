// Node composition: parses every parameter up front, takes the per-gripper
// instance lock, brings the motor up over CAN in its generation's control
// mode, then wires the command stream to the follow loop and the state/health
// publishers. All device and stream logic lives in the sibling modules; this
// is only the assembly.

use crate::follow::{self, ControlConfig};
use crate::hardware::{Gripper, PosForceLimits, gripper_motor_type};
use crate::{command_stream, health, stream};
use control_core::time::{RateOutOfRange, period_from_hz};
use openarm_description::{HardwareVersion, Side};
use peppygen::exposed_services::ready::is_ready;
use peppygen::{NodeRunner, Parameters, Result};
use peppylib::datastore::{self, Encoding};
use peppylib::runtime::CancellationToken;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{oneshot, watch};
use tracing::{error, info, warn};

// Mirrors the ROS2 reference hardware's on_activate / on_deactivate sleep durations.
const POST_ENABLE_SLEEP: Duration = Duration::from_millis(100);
const POST_DISABLE_SLEEP: Duration = Duration::from_millis(100);
const BRINGUP_RECV_US: u32 = 2000;
const ENABLE_FD: bool = true;
const DATASTORE_TIMEOUT: Duration = Duration::from_secs(3);
/// Tighter bound for shutdown lock removal so disable + drain + removal stays
/// inside the default 5 s shutdown grace window.
const LOCK_REMOVE_TIMEOUT: Duration = Duration::from_secs(1);
/// Bound on the shutdown hook that awaits the health task's final flush.
/// Hooks share one grace window and this one runs before the motor-disable
/// hook, so an unbounded wait on a stalled publish would hold the motors
/// energised until the force-kill deadline. Sized well inside even the
/// minimum 1 s grace window so the disable hook keeps most of the budget.
const HEALTH_FLUSH_TIMEOUT: Duration = Duration::from_millis(300);

/// Ceiling on the per-cycle CAN read deadline; see its use for the rationale.
const MAX_RECV_TIMEOUT_US: u32 = 100_000;

const GRIPPER_ID_LEFT: u8 = 0;
const GRIPPER_ID_RIGHT: u8 = 1;

/// The fastest this node drives the gripper. The motor shares the arm's
/// 1 Mbit CAN FD bus and is paced the same way; the stack runs at 100 Hz.
const MAX_RATE_HZ: u32 = 1_000;

/// Everything this node refuses to run on, or stops for.
///
/// Named rather than stringly typed so each refusal keeps its source, and so
/// this list is the one place to read what a launch can be rejected for. It
/// exists because returning a refusal, rather than panicking it, is what runs
/// the shutdown hooks: a panic in `setup` unwinds past them, leaving the motor
/// energised and the instance lock held.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("parameter control_rate_hz")]
    ControlRate(#[source] RateOutOfRange),

    #[error("parameter state_rate_hz")]
    StateRate(#[source] RateOutOfRange),

    #[error("gripper_id must be {GRIPPER_ID_LEFT} (left) or {GRIPPER_ID_RIGHT} (right), got {0}")]
    GripperId(u8),

    #[error(transparent)]
    HardwareVersion(#[from] openarm_description::UnknownHardwareVersion),

    #[error("recv_timeout_us must be in 1..={MAX_RECV_TIMEOUT_US} (100 ms), got {0}")]
    RecvTimeout(u32),

    #[error(transparent)]
    Limits(#[from] crate::hardware::PosForceLimitsError),

    #[error("instance lock {key} held by {holder}")]
    LockHeld { key: String, holder: String },

    #[error("enable the gripper motor")]
    Enable(#[from] openarm_can::EnableFailure),

    #[error("gripper CAN")]
    Can(#[from] openarm_can::CanError),

    #[error("the follow loop never started")]
    FollowLoopNeverStarted,

    #[error("hard fault stopped this node; the log names the failing component")]
    HardFault,

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

/// Cancel the node when a supervised task exits while it should still be
/// running. An unsupervised dead loop leaves the node reachable but inert;
/// a dead follow loop is worse, because the health task would keep judging
/// a frozen state cache. Watching the JoinHandle means a panic lands here
/// too, not only a clean return.
fn supervise(task: tokio::task::JoinHandle<()>, what: &'static str, token: CancellationToken) {
    tokio::spawn(async move {
        let _ = task.await;
        if !token.is_cancelled() {
            error!("{what} exited unexpectedly");
            follow::HARD_FAULT.store(true, Ordering::SeqCst);
        }
        token.cancel();
    });
}

/// Readiness as the initializer sees it. A latched fault or a started
/// shutdown revokes it: peppy keeps services reachable while the shutdown
/// hooks run, so without the cancellation term a leader polling during a
/// bounce would be told a limb is ready while its motor is being disabled.
fn reports_ready(brought_up: bool, hard_faulted: bool, shutting_down: bool) -> bool {
    brought_up && !hard_faulted && !shutting_down
}

/// True once the follow loop has latched a hard CAN fault; read by `main`
/// after the runtime returns (the `follow` module stays private to this
/// crate).
pub fn hard_fault_latched() -> bool {
    follow::HARD_FAULT.load(Ordering::SeqCst)
}

/// The runtime's entry point: the whole bring-up runs as [`NodeError`] and is
/// converted once, here, so every step inside can use `?` on its own failures.
pub async fn setup(params: Parameters, node_runner: Arc<NodeRunner>) -> Result<()> {
    assemble(params, node_runner).await.map_err(Into::into)
}

async fn assemble(params: Parameters, node_runner: Arc<NodeRunner>) -> NodeResult {
    // Pairing timestamps read the daemon-resolved clock (sim time under a
    // simulated clock), so state consumers age samples on one timeline.
    peppygen::clock::init(&node_runner).await?;

    let gripper_id = params.gripper_id;
    // One parse of gripper_id: the side whose opening geometry this
    // instance drives, and the label naming it on the alert wire.
    // Exhaustive so a value outside the convention is refused rather than
    // published as the wrong side.
    let (side, alert_source) = match gripper_id {
        GRIPPER_ID_LEFT => (Side::Left, "left gripper"),
        GRIPPER_ID_RIGHT => (Side::Right, "right gripper"),
        other => {
            return Err(NodeError::GripperId(other));
        }
    };

    // Which OpenArm generation this gripper drives; selects the control
    // mode the motor is opened in and this side's opening geometry.
    let hardware_version: HardwareVersion = params.hardware_version.parse()?;

    let can_interface = params.can_interface.clone();

    let cycle_period =
        period_from_hz(params.control_rate_hz, MAX_RATE_HZ).map_err(NodeError::ControlRate)?;
    let state_period =
        period_from_hz(params.state_rate_hz, MAX_RATE_HZ).map_err(NodeError::StateRate)?;
    // Bounded above so a config typo cannot park recv_all in a long ppoll
    // while it holds the CAN mutex the shutdown hooks need: 100 ms keeps
    // the whole hook sequence inside even the minimum 1 s grace window,
    // and real configs run around 1 ms. Bounded below because the driver
    // refuses a zero wait on its other receive passes, and a receive that
    // never waits turns every quiet tick into a silence pass.
    if !(1..=MAX_RECV_TIMEOUT_US).contains(&params.recv_timeout_us) {
        return Err(NodeError::RecvTimeout(params.recv_timeout_us));
    }

    // Checked whatever the generation: the ranges are properties of the
    // parameters, not of the gripper that reads them. Only the v2
    // POS_FORCE command path puts them on the wire.
    let limits = PosForceLimits::new(
        gripper_motor_type(hardware_version),
        params.speed_rad_s,
        params.max_effort_nm,
    )?;

    // The resolved configuration, so a parameter that this generation
    // does not read is visibly absent rather than silently ignored.
    info!(
        "config: {alert_source} ({hardware_version}) rate={}Hz state_rate={}Hz \
         recv_timeout={}us",
        params.control_rate_hz, params.state_rate_hz, params.recv_timeout_us
    );
    if hardware_version == HardwareVersion::V2 {
        info!(
            "config: POS_FORCE speed={} rad/s grip-force ceiling={} N*m at the shaft",
            params.speed_rad_s, params.max_effort_nm
        );
    }

    let cfg = ControlConfig {
        cycle_period,
        recv_timeout_us: params.recv_timeout_us,
    };

    // Instance lock: refuse to start if another instance with the same
    // gripper_id is running. Held in the core-node datastore (released from the on_shutdown
    // hook below), so a lock leaked by a hard crash clears with the stack
    // instead of lingering like a /tmp file. get-then-store is not atomic; two
    // simultaneous starts can race (single-writer in practice). Same scheme as
    // openarm_arm.
    //
    // The superseded openarm_gripper_v2 node keyed its lock on its own name,
    // which this key does not exclude. Both drive the same motor id on the
    // same bus, so a survivor of that node is checked for too: without it two
    // processes would command one gripper, each decoding the other's replies.
    // Drop the legacy key once no deployment can still be running that node.
    let lock_key = format!("openarm_gripper_{gripper_id}_instance_lock");
    let superseded_lock_key = format!("openarm_gripper_v2_{gripper_id}_instance_lock");
    for key in [superseded_lock_key.as_str(), lock_key.as_str()] {
        if let Some(held) = datastore::get(&node_runner, key, DATASTORE_TIMEOUT).await? {
            return Err(NodeError::LockHeld {
                key: key.to_string(),
                holder: held.last_modified_by,
            });
        }
    }
    datastore::store(
        &node_runner,
        lock_key.as_str(),
        b"locked".to_vec(),
        Encoding::TEXT_PLAIN,
        DATASTORE_TIMEOUT,
    )
    .await?;

    // Lock-release hook, registered first so it runs last (after the
    // motor-disable hook below). The runtime fires it on every stop path with
    // the messenger still connected, so the key never outlives the process.
    {
        let runner = node_runner.clone();
        let lock_key = lock_key.clone();
        node_runner.on_shutdown(async move {
            if let Err(e) = datastore::remove(&runner, lock_key.as_str(), LOCK_REMOVE_TIMEOUT).await
            {
                warn!("failed to remove lock {lock_key}: {e}");
            }
        });
    }

    // Hardware bringup, mirroring the ROS2 reference's on_init / on_configure
    // / on_activate: opening writes this generation's control mode into the
    // motor and consumes the reply, so the bus is quiet on return.
    info!("opening CAN interface {can_interface} (FD={ENABLE_FD})");
    let gripper = Gripper::open(hardware_version, side, &can_interface, ENABLE_FD, limits)?;
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
            // unwrap_or_else: recover even if poisoned (panic in control loop)
            // so disable_all() always runs and the motor doesn't stay energised.
            let mut g = gripper.lock().unwrap_or_else(|e| e.into_inner());
            if let Err(e) = g.disable_all() {
                error!("disable motor: {e}");
            }
            std::thread::sleep(POST_DISABLE_SLEEP);
            if let Err(e) = g.recv_all(BRINGUP_RECV_US) {
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
    // Errors return through the runtime so the hooks above disable the
    // motor and release the lock.
    gripper
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .enable_and_confirm(
            openarm_can::ENABLE_ATTEMPTS,
            POST_ENABLE_SLEEP,
            BRINGUP_RECV_US,
        )?;
    {
        let mut g = gripper.lock().unwrap_or_else(|e| e.into_inner());
        info!("returning to zero");
        g.command(0.0, None)?;
        g.recv_all(BRINGUP_RECV_US)?;
    }
    info!("gripper ready (motor confirmed enabled)");

    // Always-on gripper_states publisher: reads the motor's cached state at
    // state_rate_hz and emits the opening, effort, and effort ceiling. It
    // issues no CAN traffic of its own, so it never contends with the
    // follow loop for the bus.
    let publisher = tokio::spawn(stream::run(
        node_runner.clone(),
        state_period,
        gripper.clone(),
        node_runner.cancellation_token().clone(),
    ));
    supervise(
        publisher,
        "state publisher",
        node_runner.cancellation_token().clone(),
    );

    // Motor condition telemetry and operator alerts, off the same cached
    // state. The final flush round after cancellation is only awaited if
    // a shutdown hook waits for it: the runtime awaits hooks, not plain
    // tasks racing the token. Dropping the sender (return or panic)
    // completes the receiver, so the hook cannot hang on a dead task.
    let (health_done_tx, health_done_rx) = oneshot::channel::<()>();
    let health_task = tokio::spawn({
        let runner = node_runner.clone();
        let alert_source = alert_source.to_string();
        let gripper = gripper.clone();
        let cycle_period = cfg.cycle_period;
        let token = node_runner.cancellation_token().clone();
        async move {
            let _done = health_done_tx;
            health::run(runner, alert_source, gripper, ratings, cycle_period, token).await;
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
    );

    // is_ready service: false until bringup and control wiring complete, then
    // true. The real robot_initializer polls this (component_ready) to
    // gate the whole robot.
    let ready = Arc::new(AtomicBool::new(false));
    {
        let runner = node_runner.clone();
        let ready = ready.clone();
        let token = node_runner.cancellation_token().clone();
        tokio::spawn(async move {
            loop {
                if let Err(e) = is_ready::handle_next_request(&runner, |_req| {
                    Ok(is_ready::Response::new(reports_ready(
                        ready.load(Ordering::SeqCst),
                        follow::HARD_FAULT.load(Ordering::SeqCst),
                        token.is_cancelled(),
                    )))
                })
                .await
                {
                    error!("is_ready: {e}");
                }
            }
        });
    }

    // Stream listener -> follow loop: the listener keeps the latest streamed
    // opening addressed to this gripper, the follow loop drives the motor
    // toward it.
    let (cmd_tx, cmd_rx) = watch::channel(None);
    let commands = tokio::spawn(command_stream::run(
        node_runner.clone(),
        cmd_tx,
        node_runner.cancellation_token().clone(),
    ));
    supervise(
        commands,
        "command stream",
        node_runner.cancellation_token().clone(),
    );
    let (follow_started_tx, follow_started_rx) = oneshot::channel::<()>();
    let follower = tokio::spawn(follow::run(
        gripper,
        cmd_rx,
        cfg,
        node_runner.cancellation_token().clone(),
        follow_started_tx,
    ));
    supervise(
        follower,
        "follow loop",
        node_runner.cancellation_token().clone(),
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
