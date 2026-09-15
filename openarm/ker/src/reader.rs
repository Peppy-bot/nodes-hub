// The device thread: owns the transport, handshakes, decodes and maps frames,
// and keeps the newest calibrated sample on a watch channel for the publish
// tasks. Device I/O is blocking, so this runs on a dedicated OS thread.
//
// Failure policy: configuration-vs-device mismatches found at handshake (too
// few channels) cancel the node so the launch fails loudly; everything
// transient (unplug, bad checksums, silence) clears the sample, backs off and
// reconnects. Engagement lives here because only this thread sees every
// frame: an arm engages on the first frame its trigger is squeezed to the
// engage opening, and both arms disengage when usable frames stop for the
// stale timeout or the device reconnects, so a returning device never resumes
// motion on its own.

use std::time::{Duration, Instant};

use openarm_description::{ARM_DOF, Side};
use peppylib::runtime::CancellationToken;
use tokio::sync::{oneshot, watch};
use tracing::{error, info, warn};

use crate::mapping::Calibration;
use crate::protocol::{CMD_PING, CMD_STANDBY, Deframer, FrameLayout, KerFrame, PingParse, Schema};
use crate::transport::{self, TransportConfig};

/// A launcher `engage_opening` outside [0, 1).
#[derive(Debug, thiserror::Error)]
#[error("engage_opening must be a trigger opening fraction in [0, 1), got {0}")]
pub struct EngageOpeningOutOfRange(pub f64);

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

/// The trigger opening at or below which a squeeze engages its arm; strictly
/// below 1 (fully open).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngageOpening(f64);

impl EngageOpening {
    pub fn fraction(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for EngageOpening {
    type Error = EngageOpeningOutOfRange;

    fn try_from(fraction: f64) -> Result<Self, Self::Error> {
        (0.0..1.0)
            .contains(&fraction)
            .then_some(Self(fraction))
            .ok_or(EngageOpeningOutOfRange(fraction))
    }
}

/// Which arms track the KER.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Engaged {
    pub left: bool,
    pub right: bool,
}

impl Engaged {
    pub fn side(self, side: Side) -> bool {
        match side {
            Side::Left => self.left,
            Side::Right => self.right,
        }
    }
}

/// One calibrated bimanual sample: what the publish tasks stream for each
/// engaged arm while fresh.
#[derive(Debug, Clone)]
pub struct KerSample {
    pub left_joints: [f64; ARM_DOF],
    pub right_joints: [f64; ARM_DOF],
    pub left_opening: f64,
    pub right_opening: f64,
    pub engaged: Engaged,
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

pub struct ReaderConfig {
    pub transport: TransportConfig,
    pub calibration: Calibration,
    pub engage_opening: EngageOpening,
    pub stale_timeout: Duration,
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
    info!(
        "KER connected: fw {} hw {} updated {} ({} channels)",
        schema.metadata.firmware,
        schema.metadata.hardware,
        schema.metadata.updated,
        layout.angle_count()
    );

    let mut deframer = Deframer::new(layout.payload_len());
    deframer.push(&leftover);
    let mut engage = EngageLatch::new(cfg.engage_opening, cfg.stale_timeout);
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
            match map_sample(&cfg.calibration, &frame, &mut engage) {
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

/// Per-arm engagement over the stream of usable frames. A trigger squeezed to
/// the engage opening engages its arm; a gap between usable frames of at least
/// the stale timeout disengages both. Each session starts a fresh latch, so a
/// reconnect disengages too.
struct EngageLatch {
    engage_opening: EngageOpening,
    stale_timeout: Duration,
    engaged: Engaged,
    last_frame_at: Option<Instant>,
}

impl EngageLatch {
    fn new(engage_opening: EngageOpening, stale_timeout: Duration) -> Self {
        Self {
            engage_opening,
            stale_timeout,
            engaged: Engaged::default(),
            last_frame_at: None,
        }
    }

    /// Fold in one usable frame's trigger openings, captured at `at`.
    fn update(&mut self, left_opening: f64, right_opening: f64, at: Instant) -> Engaged {
        let stalled = self
            .last_frame_at
            .is_some_and(|last| at.duration_since(last) >= self.stale_timeout);
        if stalled && self.engaged != Engaged::default() {
            info!("KER frames stalled; both arms disengaged, squeeze a trigger to re-engage");
        }
        let held = if stalled {
            Engaged::default()
        } else {
            self.engaged
        };
        let threshold = self.engage_opening.fraction();
        let next = Engaged {
            left: held.left || left_opening <= threshold,
            right: held.right || right_opening <= threshold,
        };
        for side in [Side::Left, Side::Right] {
            if next.side(side) && !held.side(side) {
                info!("KER {side:?} arm engaged, tracking the leader");
            }
        }
        self.engaged = next;
        self.last_frame_at = Some(at);
        next
    }
}

fn map_sample(
    calibration: &Calibration,
    frame: &KerFrame,
    engage: &mut EngageLatch,
) -> Result<KerSample, crate::mapping::MapError> {
    let left_joints = calibration.left.joint_radians(&frame.angles_deg)?;
    let right_joints = calibration.right.joint_radians(&frame.angles_deg)?;
    let left_opening = calibration.left_trigger.opening(&frame.angles_deg)?;
    let right_opening = calibration.right_trigger.opening(&frame.angles_deg)?;
    let received_at = Instant::now();
    Ok(KerSample {
        left_joints,
        right_joints,
        left_opening,
        right_opening,
        engaged: engage.update(left_opening, right_opening, received_at),
        received_at,
    })
}

fn format_raw(frame: &KerFrame) -> String {
    frame
        .angles_deg
        .iter()
        .enumerate()
        .map(|(i, a)| format!("CH{:02}={a:.2}", i + 1))
        .collect::<Vec<_>>()
        .join(" ")
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

    const STALE: Duration = Duration::from_millis(250);
    const FRAME: Duration = Duration::from_millis(5);
    const ENGAGE_AT: f64 = 0.1;
    const OPEN: f64 = 1.0;
    const SQUEEZED: f64 = 0.02;
    const LEFT: Engaged = Engaged {
        left: true,
        right: false,
    };
    const RIGHT: Engaged = Engaged {
        left: false,
        right: true,
    };
    const BOTH: Engaged = Engaged {
        left: true,
        right: true,
    };

    fn latch() -> EngageLatch {
        EngageLatch::new(EngageOpening::try_from(ENGAGE_AT).expect("in range"), STALE)
    }

    #[test]
    fn engage_opening_accepts_only_fractions_below_fully_open() {
        for fraction in [0.0, ENGAGE_AT, 0.999] {
            assert!(EngageOpening::try_from(fraction).is_ok(), "{fraction}");
        }
        for fraction in [-0.01, 1.0, 1.5, f64::NAN, f64::INFINITY] {
            assert!(EngageOpening::try_from(fraction).is_err(), "{fraction}");
        }
    }

    #[test]
    fn a_squeeze_engages_only_its_own_arm_and_release_keeps_it_tracking() {
        let mut latch = latch();
        let t0 = Instant::now();
        assert_eq!(latch.update(OPEN, OPEN, t0), Engaged::default());
        assert_eq!(latch.update(OPEN, SQUEEZED, t0 + FRAME), RIGHT);
        assert_eq!(latch.update(OPEN, OPEN, t0 + FRAME * 2), RIGHT);
        assert_eq!(latch.update(SQUEEZED, OPEN, t0 + FRAME * 3), BOTH);
        assert_eq!(latch.update(OPEN, OPEN, t0 + FRAME * 4), BOTH);
    }

    #[test]
    fn the_engage_opening_itself_engages() {
        let mut latch = latch();
        assert_eq!(latch.update(ENGAGE_AT, OPEN, Instant::now()), LEFT);
    }

    #[test]
    fn a_frame_gap_of_the_stale_timeout_disengages_both_arms() {
        let mut latch = latch();
        let t0 = Instant::now();
        latch.update(SQUEEZED, SQUEEZED, t0);
        let just_inside = t0 + STALE - FRAME;
        assert_eq!(latch.update(OPEN, OPEN, just_inside), BOTH);
        assert_eq!(
            latch.update(OPEN, OPEN, just_inside + STALE),
            Engaged::default()
        );
    }

    #[test]
    fn a_squeeze_on_the_frame_after_a_stall_engages_only_that_arm() {
        let mut latch = latch();
        let t0 = Instant::now();
        latch.update(SQUEEZED, SQUEEZED, t0);
        assert_eq!(latch.update(OPEN, SQUEEZED, t0 + STALE), RIGHT);
    }
}
