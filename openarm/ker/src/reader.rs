// The device thread: owns the transport, handshakes, decodes and maps frames,
// and keeps the newest calibrated sample on a watch channel for the publish
// tasks. Device I/O is blocking, so this runs on a dedicated OS thread rather
// than inside the tokio runtime.
//
// Failure policy: configuration-vs-device mismatches found at handshake (too
// few channels, no button in toggle mode) cancel the node so the launch fails
// loudly; everything transient (unplug, bad checksums, silence) clears the
// sample, backs off and reconnects. Engage state lives here because only this
// thread sees every frame (edge detection would be lossy on the coalescing
// watch channel), and it resets to disengaged on every reconnect so a
// returning device never resumes motion on its own.

use std::str::FromStr;
use std::time::{Duration, Instant};

use openarm_description::{ARM_DOF, Side};
use peppylib::runtime::CancellationToken;
use tokio::sync::{oneshot, watch};
use tracing::{error, info, warn};

use crate::mapping::Calibration;
use crate::protocol::{CMD_PING, CMD_STANDBY, Deframer, FrameLayout, KerFrame, PingParse, Schema};
use crate::transport::{self, TransportConfig};

/// A launcher `engage_mode` this node has no engagement rule for.
#[derive(Debug, thiserror::Error)]
#[error("engage_mode must be 'toggle' or 'always', got '{0}'")]
pub struct UnknownEngageMode(pub String);

const HANDSHAKE_DEADLINE: Duration = Duration::from_secs(3);
const PING_INTERVAL: Duration = Duration::from_millis(500);
const RECONNECT_BACKOFF: Duration = Duration::from_secs(1);
/// A connected device yielding no valid frame for this long is re-handshaken,
/// not just stale. Gated on frame age alone so it also covers a device that
/// keeps sending bytes that never frame (headerless garbage evades both the
/// read-timeout and the checksum counter).
const SILENCE_RECONNECT: Duration = Duration::from_secs(5);
/// This many corrupt frames in a row means framing is lost; reconnect.
const MAX_CONSECUTIVE_BAD_CHECKSUMS: u32 = 50;
const RAW_LOG_INTERVAL: Duration = Duration::from_secs(1);

/// One calibrated bimanual sample: what the publish tasks stream while fresh
/// and engaged.
#[derive(Debug, Clone)]
pub struct KerSample {
    pub left_joints: [f64; ARM_DOF],
    pub right_joints: [f64; ARM_DOF],
    pub left_opening: f64,
    pub right_opening: f64,
    pub engaged: bool,
    pub received_at: Instant,
}

impl KerSample {
    pub fn joints(&self, side: Side) -> [f64; ARM_DOF] {
        match side {
            Side::Left => self.left_joints,
            Side::Right => self.right_joints,
        }
    }

    pub fn opening(&self, side: Side) -> f64 {
        match side {
            Side::Left => self.left_opening,
            Side::Right => self.right_opening,
        }
    }
}

/// What arms the streams: the thumb button as a toggle deadman, or always-on
/// for a unit whose button is absent or unreliable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngageMode {
    Toggle,
    Always,
}

impl FromStr for EngageMode {
    type Err = UnknownEngageMode;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "toggle" => Ok(Self::Toggle),
            "always" => Ok(Self::Always),
            other => Err(UnknownEngageMode(other.to_string())),
        }
    }
}

pub struct ReaderConfig {
    pub transport: TransportConfig,
    pub calibration: Calibration,
    pub engage_mode: EngageMode,
    pub log_raw: bool,
}

/// Why the reader thread stopped. The supervisor, not the reader, decides
/// what a stop means for the node, so the reader only reports.
#[derive(Debug)]
pub enum ReaderExit {
    /// Shutdown was already under way; nothing to record.
    Cancelled,
    /// The device's configuration cannot drive this launch, or the reader's
    /// consumer vanished while the node was still running.
    Fatal,
}

/// Spawn the device thread. It publishes `None` whenever the device is not
/// delivering valid frames.
///
/// The returned receiver yields the thread's outcome when it stops, however it
/// stops: the sender lives in the thread's own frame, so an unwind drops it
/// and the receiver reads the closed channel as a fault.
pub fn spawn(
    cfg: ReaderConfig,
    tx: watch::Sender<Option<KerSample>>,
    token: CancellationToken,
) -> std::io::Result<oneshot::Receiver<ReaderExit>> {
    let (exited_tx, exited_rx) = oneshot::channel();
    std::thread::Builder::new()
        .name("ker-reader".into())
        .spawn(move || {
            let _ = exited_tx.send(run(cfg, tx, token));
        })?;
    Ok(exited_rx)
}

enum SessionEnd {
    /// The token was cancelled, or the sample channel closed; `run` reads the
    /// token to tell the two apart.
    Stop,
    Transient(String),
    Fatal(String),
}

fn run(
    cfg: ReaderConfig,
    tx: watch::Sender<Option<KerSample>>,
    token: CancellationToken,
) -> ReaderExit {
    while !token.is_cancelled() {
        match session(&cfg, &tx, &token) {
            SessionEnd::Stop => break,
            SessionEnd::Transient(reason) => {
                let _ = tx.send(None);
                warn!("KER connection lost ({reason}); retrying in {RECONNECT_BACKOFF:?}");
                sleep_cancellable(RECONNECT_BACKOFF, &token);
            }
            SessionEnd::Fatal(reason) => {
                let _ = tx.send(None);
                error!("KER configuration mismatch: {reason}");
                return ReaderExit::Fatal;
            }
        }
    }
    // A stop is only a shutdown if the token says so. The other way to reach
    // here is the sample channel closing under a live token, which means the
    // publisher died: without this check the two race in the supervisor's
    // select and a dead publisher could be recorded as a clean finish.
    if token.is_cancelled() {
        ReaderExit::Cancelled
    } else {
        error!("KER sample consumer vanished while the node was running");
        ReaderExit::Fatal
    }
}

/// One connection lifetime: connect, handshake, stream until it breaks.
fn session(
    cfg: &ReaderConfig,
    tx: &watch::Sender<Option<KerSample>>,
    token: &CancellationToken,
) -> SessionEnd {
    let mut transport = match transport::open(&cfg.transport) {
        Ok(t) => t,
        Err(e) => return SessionEnd::Transient(format!("open: {e}")),
    };

    let (schema, leftover) = match handshake(transport.as_mut(), token) {
        Ok(parsed) => parsed,
        Err(end) => return end,
    };
    let layout = match FrameLayout::try_new(&schema) {
        Ok(layout) => layout,
        Err(e) => return SessionEnd::Fatal(e.to_string()),
    };
    let required = cfg.calibration.required_channels();
    if layout.angle_count() < required {
        return SessionEnd::Fatal(format!(
            "calibration references CH{required:02} but the device streams only {} channels",
            layout.angle_count()
        ));
    }
    if cfg.engage_mode == EngageMode::Toggle && !layout.has_button() {
        return SessionEnd::Fatal(
            "engage_mode 'toggle' needs the encoder_button field, which this schema lacks; \
             use engage_mode 'always'"
                .into(),
        );
    }
    info!(
        "KER connected: fw {} hw {} updated {} ({} channels)",
        schema.metadata.firmware,
        schema.metadata.hardware,
        schema.metadata.updated,
        layout.angle_count()
    );

    let mut deframer = Deframer::new(layout.payload_len());
    deframer.push(&leftover);
    let mut engage = EngageLatch::new(cfg.engage_mode);
    let mut chunk = [0u8; 4096];
    let mut last_frame_at = Instant::now();
    let mut last_raw_log = Instant::now();
    let mut consecutive_bad = 0u32;
    let mut mapping_warned = false;

    while !token.is_cancelled() {
        let read = match transport.read(&mut chunk) {
            Ok(n) => n,
            Err(e) => return SessionEnd::Transient(format!("read: {e}")),
        };
        if last_frame_at.elapsed() > SILENCE_RECONNECT {
            return SessionEnd::Transient(format!(
                "no valid frames for {SILENCE_RECONNECT:?} while connected"
            ));
        }
        deframer.push(&chunk[..read]);

        while let Some(result) = deframer.next_payload() {
            let payload = match result {
                Ok(payload) => payload,
                Err(_) => {
                    consecutive_bad += 1;
                    if consecutive_bad >= MAX_CONSECUTIVE_BAD_CHECKSUMS {
                        return SessionEnd::Transient(format!(
                            "{consecutive_bad} corrupt frames in a row"
                        ));
                    }
                    continue;
                }
            };
            consecutive_bad = 0;
            let frame = layout.parse(&payload);
            last_frame_at = Instant::now();
            if cfg.log_raw && last_raw_log.elapsed() >= RAW_LOG_INTERVAL {
                last_raw_log = Instant::now();
                info!("KER raw: {}", format_raw(&frame));
            }
            let engaged = engage.update(&frame);
            match map_sample(&cfg.calibration, &frame, engaged) {
                Ok(sample) => {
                    mapping_warned = false;
                    if tx.send(Some(sample)).is_err() {
                        return SessionEnd::Stop;
                    }
                }
                // A non-finite reading is a frame to skip, not a stream to
                // kill; latch the warning so a flaky encoder cannot spam.
                Err(e) if !mapping_warned => {
                    mapping_warned = true;
                    warn!("KER frame dropped, suppressing repeats: {e}");
                }
                Err(_) => {}
            }
        }
    }
    // Best effort: leave the device quiet on the way out.
    let _ = transport.write_all(&[CMD_STANDBY]);
    SessionEnd::Stop
}

/// STANDBY, flush, then ping until the schema arrives (or the deadline).
/// Returns the schema and any stream bytes read past it.
fn handshake(
    transport: &mut dyn transport::KerTransport,
    token: &CancellationToken,
) -> Result<(Schema, Vec<u8>), SessionEnd> {
    let transient = |e| SessionEnd::Transient(format!("handshake: {e}"));
    transport.write_all(&[CMD_STANDBY]).map_err(transient)?;
    transport.flush_input().map_err(transient)?;

    let deadline = Instant::now() + HANDSHAKE_DEADLINE;
    let mut next_ping = Instant::now();
    let mut buf = Vec::new();
    let mut chunk = [0u8; 512];
    while Instant::now() < deadline {
        if token.is_cancelled() {
            return Err(SessionEnd::Stop);
        }
        if Instant::now() >= next_ping {
            transport.write_all(&[CMD_PING]).map_err(transient)?;
            next_ping = Instant::now() + PING_INTERVAL;
        }
        let read = transport.read(&mut chunk).map_err(transient)?;
        buf.extend_from_slice(&chunk[..read]);
        match Schema::parse_ping(&buf) {
            PingParse::NeedMore => continue,
            PingParse::Parsed { schema, consumed } => {
                return Ok((schema, buf.split_off(consumed)));
            }
            PingParse::Invalid(e) => return Err(SessionEnd::Fatal(e.to_string())),
        }
    }
    Err(SessionEnd::Transient(
        "handshake: no schema within the deadline".into(),
    ))
}

/// The device-wide engage deadman. In toggle mode a button rising edge flips
/// it; frame-exact because this runs on every decoded frame.
struct EngageLatch {
    mode: EngageMode,
    engaged: bool,
    button_was_down: bool,
}

impl EngageLatch {
    fn new(mode: EngageMode) -> Self {
        Self {
            mode,
            engaged: mode == EngageMode::Always,
            button_was_down: false,
        }
    }

    fn update(&mut self, frame: &KerFrame) -> bool {
        if self.mode == EngageMode::Always {
            return true;
        }
        if frame.encoder_button && !self.button_was_down {
            self.engaged = !self.engaged;
            info!(
                "KER {}",
                if self.engaged {
                    "ENGAGED, streaming"
                } else {
                    "disengaged, holding"
                }
            );
        }
        self.button_was_down = frame.encoder_button;
        self.engaged
    }
}

fn map_sample(
    calibration: &Calibration,
    frame: &KerFrame,
    engaged: bool,
) -> Result<KerSample, crate::mapping::MapError> {
    Ok(KerSample {
        left_joints: calibration.left.joint_radians(&frame.angles_deg)?,
        right_joints: calibration.right.joint_radians(&frame.angles_deg)?,
        left_opening: calibration.left_trigger.opening(&frame.angles_deg)?,
        right_opening: calibration.right_trigger.opening(&frame.angles_deg)?,
        engaged,
        received_at: Instant::now(),
    })
}

fn format_raw(frame: &KerFrame) -> String {
    let channels: Vec<String> = frame
        .angles_deg
        .iter()
        .enumerate()
        .map(|(i, a)| format!("CH{:02}={a:.2}", i + 1))
        .collect();
    format!(
        "{} enc={} btn={}",
        channels.join(" "),
        frame.encoder_value,
        frame.encoder_button
    )
}

fn sleep_cancellable(total: Duration, token: &CancellationToken) {
    let deadline = Instant::now() + total;
    while Instant::now() < deadline && !token.is_cancelled() {
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(button: bool) -> KerFrame {
        KerFrame {
            timestamp: 0,
            angles_deg: vec![],
            encoder_value: 0,
            encoder_button: button,
        }
    }

    #[test]
    fn toggle_flips_only_on_the_rising_edge() {
        let mut latch = EngageLatch::new(EngageMode::Toggle);
        assert!(!latch.update(&frame(false)), "starts disengaged");
        assert!(latch.update(&frame(true)), "press engages");
        assert!(latch.update(&frame(true)), "holding does not retoggle");
        assert!(latch.update(&frame(false)), "release keeps engaged");
        assert!(!latch.update(&frame(true)), "second press disengages");
        assert!(!latch.update(&frame(false)));
    }

    #[test]
    fn always_mode_is_always_engaged() {
        let mut latch = EngageLatch::new(EngageMode::Always);
        assert!(latch.update(&frame(false)));
        assert!(latch.update(&frame(true)));
    }
}
