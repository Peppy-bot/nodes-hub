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

use crate::mapping::{ChannelMap, GripperOpenFraction, GripperOpenFractionOutOfRange};
use crate::publish;
use crate::reader::{self, EngageOpening, ReaderConfig};
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
/// this list names every refusal `setup` itself raises (a device the channel
/// map does not describe is refused by the reader and arrives as
/// [`NodeError::TaskStopped`], with the reason in the log). It
/// exists because returning a refusal, rather than panicking it, is what runs
/// the shutdown hooks: a panic in `setup` unwinds past them, leaving the
/// instance lock standing against the next start.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("parameter command_rate_hz")]
    CommandRate(#[source] RateOutOfRange),

    #[error("parameter stale_timeout_s")]
    StaleTimeout(#[source] DurationError),

    #[error(
        "stale_timeout_s {stale_timeout_s} is at or under the {command_rate_hz} Hz command \
         period: a limb would age out between its own ticks. Raise stale_timeout_s above \
         {command_period_s} s"
    )]
    StaleTimeoutUnderCommandPeriod {
        stale_timeout_s: f64,
        command_rate_hz: u32,
        command_period_s: f64,
    },

    #[error(transparent)]
    HardwareVersion(#[from] openarm_description::UnknownHardwareVersion),

    #[error(transparent)]
    Transport(#[from] crate::transport::TransportError),

    #[error(transparent)]
    EngageOpening(#[from] crate::reader::EngageOpeningOutOfRange),

    #[error(transparent)]
    GripperOpenFraction(#[from] GripperOpenFractionOutOfRange),

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

/// What the launch parameters resolve to, parsed once before any device or
/// stack contact.
#[derive(Debug)]
struct Config {
    version: HardwareVersion,
    command_period: Duration,
    stale_timeout: Duration,
    transport: TransportConfig,
    engage_opening: EngageOpening,
    gripper_open_fraction: GripperOpenFraction,
}

/// Parse every launch parameter, refusing the first that cannot drive a KER.
fn parse_config(params: &Parameters) -> NodeResult<Config> {
    let command_period =
        period_from_hz(params.command_rate_hz, MAX_RATE_HZ).map_err(NodeError::CommandRate)?;
    let stale_timeout =
        duration_from_secs(params.stale_timeout_s).map_err(NodeError::StaleTimeout)?;
    if stale_timeout <= command_period {
        return Err(NodeError::StaleTimeoutUnderCommandPeriod {
            stale_timeout_s: params.stale_timeout_s,
            command_rate_hz: params.command_rate_hz,
            command_period_s: command_period.as_secs_f64(),
        });
    }
    Ok(Config {
        version: params.hardware_version.parse()?,
        command_period,
        stale_timeout,
        transport: TransportConfig::parse(
            &params.transport,
            &params.device_path,
            params.serial_baud,
        )?,
        engage_opening: EngageOpening::try_from(params.engage_opening)?,
        gripper_open_fraction: GripperOpenFraction::try_from(params.gripper_open_fraction)?,
    })
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
    // Pairing timestamps read this instance's bound clock, so the backbone
    // ages setpoints on one timeline.
    peppygen::clock::init(&node_runner).await?;
    let token = node_runner.cancellation_token().clone();

    // Parse every parameter up front (parse, don't validate): a bad launch
    // fails here with the reason, before touching the device or the stack.
    let config = parse_config(&params)?;

    info!(
        "config: {} follower, transport {}, {} Hz, engage at trigger opening <= {}, \
         released gripper opening {}",
        config.version,
        config.transport,
        params.command_rate_hz,
        config.engage_opening.fraction(),
        config.gripper_open_fraction.fraction(),
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

    // The reader thread owns the device and keeps the newest mapped sample
    // on the watch channel; the publish tasks stream it. Returning
    // promptly matters: peppylib registers node_health only after this
    // closure returns, so the device connect must not be awaited here.
    let (sample_tx, sample_rx) = watch::channel(None);
    let stale_timeout = config.stale_timeout;
    let command_period = config.command_period;
    let reader_exited = reader::spawn(
        ReaderConfig {
            transport: config.transport,
            channels: ChannelMap::for_follower(config.version),
            engage_opening: config.engage_opening,
            gripper_open_fraction: config.gripper_open_fraction,
            stale_timeout,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A launch the node accepts, one field per parameter the manifest names.
    fn params() -> Parameters {
        Parameters {
            command_rate_hz: 100,
            device_path: "/dev/openarm/ker".to_string(),
            engage_opening: 0.2,
            gripper_open_fraction: 0.5,
            hardware_version: "v2".to_string(),
            log_raw: false,
            serial_baud: 2_000_000,
            stale_timeout_s: 0.25,
            transport: "usb".to_string(),
        }
    }

    fn refusal(params: Parameters) -> String {
        let error = parse_config(&params).expect_err("refused");
        // The source carries the reason; the variant names the parameter.
        match std::error::Error::source(&error) {
            Some(source) => format!("{error}: {source}"),
            None => error.to_string(),
        }
    }

    #[test]
    fn a_launch_with_every_parameter_set_parses() {
        let config = parse_config(&params()).expect("parses");
        assert_eq!(config.command_period, Duration::from_millis(10));
        assert_eq!(config.stale_timeout, Duration::from_millis(250));
        assert_eq!(config.engage_opening.fraction(), 0.2);
        assert_eq!(config.gripper_open_fraction.fraction(), 0.5);
        assert_eq!(config.transport.to_string(), "usb (303a:4002)");
    }

    #[test]
    fn every_refusal_names_its_parameter() {
        let cases = [
            (
                "command_rate_hz",
                Parameters {
                    command_rate_hz: 0,
                    ..params()
                },
            ),
            (
                "command_rate_hz",
                Parameters {
                    command_rate_hz: MAX_RATE_HZ + 1,
                    ..params()
                },
            ),
            (
                "stale_timeout_s",
                Parameters {
                    stale_timeout_s: 0.0,
                    ..params()
                },
            ),
            (
                "stale_timeout_s",
                Parameters {
                    stale_timeout_s: f64::NAN,
                    ..params()
                },
            ),
            (
                "hardware_version",
                Parameters {
                    hardware_version: "v3".into(),
                    ..params()
                },
            ),
            (
                "transport",
                Parameters {
                    transport: "spi".into(),
                    ..params()
                },
            ),
            (
                "serial_baud",
                Parameters {
                    transport: "serial".into(),
                    serial_baud: 0,
                    ..params()
                },
            ),
            (
                "engage_opening",
                Parameters {
                    engage_opening: 1.0,
                    ..params()
                },
            ),
            (
                "gripper_open_fraction",
                Parameters {
                    gripper_open_fraction: 0.0,
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
    fn a_stale_timeout_inside_the_command_period_is_refused() {
        // 100 Hz commands with a 5 ms stale window: every tick would age out.
        let refused = refusal(Parameters {
            stale_timeout_s: 0.005,
            ..params()
        });
        assert!(refused.contains("stale_timeout_s"), "{refused}");
        assert!(
            refused.contains("0.01"),
            "names the command period: {refused}"
        );

        // One period is still too short; anything past it parses.
        assert!(
            parse_config(&Parameters {
                stale_timeout_s: 0.01,
                ..params()
            })
            .is_err()
        );
        assert!(
            parse_config(&Parameters {
                stale_timeout_s: 0.011,
                ..params()
            })
            .is_ok()
        );
    }
}
