// Node composition: parses every parameter up front, takes the single-instance
// lock, then wires the reader thread to the publish tasks. All device and
// stream logic lives in the sibling modules; this is only the assembly.

use std::sync::Arc;
use std::time::Duration;

use control_core::time::{DurationError, RateOutOfRange, duration_from_secs, period_from_hz};
use openarm_description::HardwareVersion;
use peppygen::{NodeRunner, Parameters, Result};
use peppylib::datastore::{self, Encoding};
use tokio::sync::watch;
use tracing::{error, info, warn};

use crate::mapping::{Calibration, CalibrationParams};
use crate::publish;
use crate::reader::{self, EngageMode, ReaderConfig};
use crate::transport::TransportConfig;

const DATASTORE_TIMEOUT: Duration = Duration::from_secs(3);
const LOCK_REMOVE_TIMEOUT: Duration = Duration::from_secs(1);
/// One node instance per KER device: a second reader would fight for the USB
/// claim (or interleave on the serial port) in confusing ways, so fail fast.
const LOCK_KEY: &str = "openarm_ker_instance_lock";
/// The fastest this node streams the leader's pose. The KER reference loop
/// runs at 1 kHz; nothing downstream consumes faster.
const MAX_RATE_HZ: u32 = 1_000;

/// Latched when a task stopped on its own rather than in reaction to shutdown.
static TASK_FAILED: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();

/// Everything this node refuses to run on, or stops for.
///
/// Named rather than stringly typed so each refusal keeps its source, and so
/// this list is the one place to read what a launch can be rejected for. It
/// exists because returning a refusal, rather than panicking it, is what runs
/// the shutdown hooks: a panic in `setup` unwinds past them, leaving the
/// instance lock standing against the next start.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("parameter command_rate_hz")]
    CommandRate(#[source] RateOutOfRange),

    #[error("parameter stale_timeout_s")]
    StaleTimeout(#[source] DurationError),

    #[error(transparent)]
    HardwareVersion(#[from] openarm_description::UnknownHardwareVersion),

    #[error(transparent)]
    Transport(#[from] crate::transport::UnknownTransport),

    #[error(transparent)]
    EngageMode(#[from] crate::reader::UnknownEngageMode),

    #[error("calibration")]
    Calibration(#[from] crate::mapping::MapError),

    #[error("instance lock {key} held by {holder}")]
    LockHeld { key: String, holder: String },

    #[error("spawn the KER reader thread")]
    ReaderThread(#[source] std::io::Error),

    #[error("the KER {0} stopped this node; the log has its reason")]
    TaskStopped(&'static str),

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

/// Which task stopped on its own, if one did; read by `main` after the
/// runtime returns, so a reader or publisher death is recorded as a failure
/// that names the task.
pub fn task_failed() -> Option<&'static str> {
    TASK_FAILED.get().copied()
}

/// The runtime's entry point: the whole bring-up runs as [`NodeError`] and is
/// converted once, here, so every step inside can use `?` on its own failures.
pub async fn setup(params: Parameters, node_runner: Arc<NodeRunner>) -> Result<()> {
    assemble(params, node_runner).await.map_err(Into::into)
}

async fn assemble(params: Parameters, node_runner: Arc<NodeRunner>) -> NodeResult {
    // Pairing timestamps read the daemon-resolved clock (sim time under a
    // simulated clock), so the backbone ages setpoints on one timeline.
    peppygen::clock::init(&node_runner).await?;
    let token = node_runner.cancellation_token().clone();

    // Parse every parameter up front (parse, don't validate): a bad launch
    // fails here with the reason, before touching the device or the stack.
    let version: HardwareVersion = params.hardware_version.parse()?;
    let command_period =
        period_from_hz(params.command_rate_hz, MAX_RATE_HZ).map_err(NodeError::CommandRate)?;
    let stale_timeout =
        duration_from_secs(params.stale_timeout_s).map_err(NodeError::StaleTimeout)?;
    let transport =
        TransportConfig::parse(&params.transport, &params.serial_port, params.serial_baud)?;
    let engage_mode: EngageMode = params.engage_mode.parse()?;
    let calibration = Calibration::parse(
        version,
        &CalibrationParams {
            left_channels: &params.left_channels,
            left_signs: &params.left_signs,
            left_offsets_deg: &params.left_offsets_deg,
            right_channels: &params.right_channels,
            right_signs: &params.right_signs,
            right_offsets_deg: &params.right_offsets_deg,
            left_trigger_channel: params.left_trigger_channel,
            left_trigger_closed_deg: params.left_trigger_closed_deg,
            left_trigger_open_deg: params.left_trigger_open_deg,
            right_trigger_channel: params.right_trigger_channel,
            right_trigger_closed_deg: params.right_trigger_closed_deg,
            right_trigger_open_deg: params.right_trigger_open_deg,
        },
    )?;

    info!(
        "config: {version} follower, transport {transport:?}, {} Hz, engage {engage_mode:?}, \
         {} channels required",
        params.command_rate_hz,
        calibration.required_channels()
    );

    // Instance lock: refuse to start if another instance is running. Held in the
    // core-node datastore (released from the on_shutdown hook below), so a
    // lock leaked by a hard crash clears with the stack instead of
    // lingering like a /tmp file. get-then-store is not atomic; two
    // simultaneous starts can race (single-writer in practice).
    if let Some(held) = datastore::get(&node_runner, LOCK_KEY, DATASTORE_TIMEOUT).await? {
        return Err(NodeError::LockHeld {
            key: LOCK_KEY.to_string(),
            holder: held.last_modified_by,
        });
    }
    datastore::store(
        &node_runner,
        LOCK_KEY,
        b"locked".to_vec(),
        Encoding::TEXT_PLAIN,
        DATASTORE_TIMEOUT,
    )
    .await?;
    {
        let runner = node_runner.clone();
        node_runner.on_shutdown(async move {
            if let Err(e) = datastore::remove(&runner, LOCK_KEY, LOCK_REMOVE_TIMEOUT).await {
                warn!("failed to remove lock {LOCK_KEY}: {e}");
            }
        });
    }

    // The reader thread owns the device and keeps the newest calibrated
    // sample on the watch channel; the publish tasks stream it. Returning
    // promptly matters: peppylib registers node_health only after this
    // closure returns, so the device connect must not be awaited here.
    let (sample_tx, sample_rx) = watch::channel(None);
    let reader_exited = reader::spawn(
        ReaderConfig {
            transport,
            calibration,
            engage_mode,
            log_raw: params.log_raw,
        },
        sample_tx,
        token.clone(),
    )
    .map_err(NodeError::ReaderThread)?;
    let publisher = tokio::spawn(publish::run(
        node_runner.clone(),
        sample_rx,
        command_period,
        stale_timeout,
        token,
    ));

    // The supervisor is the one place a task's death becomes the node's:
    // either one dead leaves the followers holding their last setpoints with
    // nothing driving them, and no other task here would notice. The tasks
    // report their outcome and never cancel the token themselves, so a fatal
    // stop cannot masquerade as a shutdown already under way.
    {
        let token = node_runner.cancellation_token().clone();
        tokio::spawn(async move {
            let fault: Option<&'static str> = tokio::select! {
                exit = reader_exited => match exit {
                    Ok(reader::ReaderExit::Cancelled) => None,
                    // Fatal reports and a reader panic (closed channel) alike.
                    Ok(reader::ReaderExit::Fatal) | Err(_) => Some("reader"),
                },
                joined = publisher => match joined {
                    Ok(Ok(())) => None,
                    Ok(Err(fault)) => {
                        error!("publisher stopped: {fault}");
                        Some("publisher")
                    }
                    Err(join) => {
                        error!("publisher panicked: {join}");
                        Some("publisher")
                    }
                },
            };
            if let Some(task) = fault {
                error!("KER {task} stopped; cancelling the node");
                let _ = TASK_FAILED.set(task);
            }
            token.cancel();
        });
    }
    Ok(())
}
