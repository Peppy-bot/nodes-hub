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

use crate::mapping::{Calibration, REQUIRED_CHANNELS, check_hardware};
use crate::protocol::{
    CMD_PING, CMD_STANDBY, CMD_STREAM, Deframer, FrameLayout, KerFrame, PingParse, Schema,
};
use crate::transport::{self, KerTransport, TransportConfig};

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

/// A launcher `gripper_open_fraction` outside (0, 1].
#[derive(Debug, thiserror::Error)]
#[error("gripper_open_fraction must be in (0, 1], got {0}")]
pub struct GripperOpenFractionOutOfRange(pub f64);

/// The gripper opening a released trigger commands; a full squeeze closes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GripperOpenFraction(f64);

impl GripperOpenFraction {
    pub fn fraction(self) -> f64 {
        self.0
    }

    /// The opening commanded for a trigger opening (1 released, 0 squeezed).
    fn opening(self, trigger_opening: f64) -> f64 {
        self.0 * trigger_opening
    }
}

impl TryFrom<f64> for GripperOpenFraction {
    type Error = GripperOpenFractionOutOfRange;

    fn try_from(fraction: f64) -> Result<Self, Self::Error> {
        (fraction > 0.0 && fraction <= 1.0)
            .then_some(Self(fraction))
            .ok_or(GripperOpenFractionOutOfRange(fraction))
    }
}

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
/// engaged arm while fresh. Openings are the commanded gripper openings.
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
    pub gripper_open_fraction: GripperOpenFraction,
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

#[derive(Debug)]
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
    match transport::open(&cfg.transport) {
        Ok(mut transport) => run_session(transport.as_mut(), cfg, tx, token),
        Err(e) => SessionEnd::Transient(format!("open: {e}")),
    }
}

/// Handshake, start the stream, and map frames until the link breaks.
fn run_session(
    transport: &mut dyn KerTransport,
    cfg: &ReaderConfig,
    tx: &watch::Sender<Option<KerSample>>,
    token: &CancellationToken,
) -> SessionEnd {
    let (schema, leftover) = match handshake(transport, token) {
        Ok(parsed) => parsed,
        Err(end) => return end,
    };
    let layout = match FrameLayout::try_new(&schema) {
        Ok(layout) => layout,
        Err(e) => return SessionEnd::Fatal(e.to_string()),
    };
    if let Err(e) = check_hardware(&schema.metadata.hardware) {
        return SessionEnd::Fatal(e.to_string());
    }
    if layout.angle_count() < REQUIRED_CHANNELS {
        return SessionEnd::Fatal(format!(
            "the channel map reads CH{REQUIRED_CHANNELS:02} but the device streams only {} channels",
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
    // The firmware answers PING in standby; frames flow once STREAM arrives.
    if let Err(e) = transport.write_all(&[CMD_STREAM]) {
        return SessionEnd::Transient(format!("start stream: {e}"));
    }

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
            match map_sample(
                &cfg.calibration,
                cfg.gripper_open_fraction,
                &frame,
                &mut engage,
            ) {
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
    transport: &mut dyn KerTransport,
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

/// Engagement reads the trigger's own travel; the gripper is commanded that
/// travel scaled by the open fraction.
fn map_sample(
    calibration: &Calibration,
    gripper_open_fraction: GripperOpenFraction,
    frame: &KerFrame,
    engage: &mut EngageLatch,
) -> Result<KerSample, crate::mapping::MapError> {
    let left_joints = calibration.left.joint_radians(&frame.angles_deg)?;
    let right_joints = calibration.right.joint_radians(&frame.angles_deg)?;
    let left_trigger = calibration.left_trigger.opening(&frame.angles_deg)?;
    let right_trigger = calibration.right_trigger.opening(&frame.angles_deg)?;
    let received_at = Instant::now();
    Ok(KerSample {
        left_joints,
        right_joints,
        left_opening: gripper_open_fraction.opening(left_trigger),
        right_opening: gripper_open_fraction.opening(right_trigger),
        engaged: engage.update(left_trigger, right_trigger, received_at),
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

    /// A KER that answers PING with the reference schema and sends frames
    /// only once STREAM arrives, as firmware 2.0.0 does.
    struct FakeKer {
        hardware: &'static str,
        writes: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
        pending: Vec<u8>,
        streaming: bool,
    }

    impl FakeKer {
        fn new(hardware: &'static str, writes: std::sync::Arc<std::sync::Mutex<Vec<u8>>>) -> Self {
            Self {
                hardware,
                writes,
                pending: Vec::new(),
                streaming: false,
            }
        }
    }

    fn reader_config() -> ReaderConfig {
        ReaderConfig {
            transport: TransportConfig::Usb,
            calibration: calibration(),
            engage_opening: EngageOpening::try_from(ENGAGE_AT).expect("in range"),
            gripper_open_fraction: GripperOpenFraction::try_from(0.5).expect("in range"),
            stale_timeout: STALE,
            log_raw: false,
        }
    }

    impl KerTransport for FakeKer {
        fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
            self.writes.lock().unwrap().extend_from_slice(bytes);
            match bytes {
                [CMD_PING] => self
                    .pending
                    .extend(crate::protocol::fixtures::ping_response_for(
                        self.hardware,
                        16,
                    )),
                [CMD_STREAM] => self.streaming = true,
                _ => {}
            }
            Ok(())
        }

        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.pending.is_empty() && self.streaming {
                self.pending
                    .extend(crate::protocol::fixtures::stream_packet(
                        1, &[0.0; 16], 0, false,
                    ));
            }
            if self.pending.is_empty() {
                std::thread::sleep(Duration::from_millis(1));
                return Ok(0);
            }
            let n = self.pending.len().min(buf.len());
            buf[..n].copy_from_slice(&self.pending[..n]);
            self.pending.drain(..n);
            Ok(n)
        }

        fn flush_input(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_session_starts_the_stream_after_the_handshake() {
        const SAMPLE_DEADLINE: Duration = Duration::from_secs(2);
        let writes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut ker = FakeKer::new("2.0.0", writes.clone());
        let cfg = reader_config();
        let (tx, rx) = watch::channel(None);
        let token = CancellationToken::new();
        let session = {
            let token = token.clone();
            std::thread::spawn(move || run_session(&mut ker, &cfg, &tx, &token))
        };

        let deadline = Instant::now() + SAMPLE_DEADLINE;
        while rx.borrow().is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let sampled = rx.borrow().is_some();
        token.cancel();
        assert!(matches!(session.join().unwrap(), SessionEnd::Stop));

        assert!(sampled, "no frame reached the sample channel");
        let writes = writes.lock().unwrap();
        let ping = writes.iter().position(|&b| b == CMD_PING).expect("pinged");
        let stream = writes
            .iter()
            .position(|&b| b == CMD_STREAM)
            .expect("stream started");
        assert!(ping < stream, "STREAM follows the handshake: {writes:?}");
        assert_eq!(writes.last(), Some(&CMD_STANDBY), "leaves the device quiet");
    }

    #[test]
    fn a_session_refuses_a_device_of_another_hardware_generation() {
        let writes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut ker = FakeKer::new("3.0.0", writes.clone());
        let cfg = reader_config();
        let (tx, rx) = watch::channel(None);
        let token = CancellationToken::new();

        let end = run_session(&mut ker, &cfg, &tx, &token);

        assert!(matches!(end, SessionEnd::Fatal(_)), "{end:?}");
        assert!(rx.borrow().is_none(), "a refused device streams nothing");
        assert!(
            !writes.lock().unwrap().contains(&CMD_STREAM),
            "a refused device is never started"
        );
    }

    #[test]
    fn gripper_open_fraction_accepts_only_fractions_in_zero_exclusive_to_one() {
        for fraction in [0.01, 0.5, 1.0] {
            assert!(
                GripperOpenFraction::try_from(fraction).is_ok(),
                "{fraction}"
            );
        }
        for fraction in [0.0, -0.5, 1.01, f64::NAN, f64::INFINITY] {
            assert!(
                GripperOpenFraction::try_from(fraction).is_err(),
                "{fraction}"
            );
        }
    }

    fn calibration() -> Calibration {
        Calibration::for_follower(openarm_description::HardwareVersion::V2)
    }

    /// Trigger angles in degrees: released reads 0, a full squeeze -60 on the
    /// right (CH08) and +60 on the left (CH16).
    fn frame(right_trigger_deg: f32, left_trigger_deg: f32) -> KerFrame {
        let mut angles_deg = vec![0.0; 16];
        angles_deg[7] = right_trigger_deg;
        angles_deg[15] = left_trigger_deg;
        KerFrame {
            timestamp: 0,
            angles_deg,
        }
    }

    #[test]
    fn the_gripper_is_commanded_the_scaled_trigger_but_engage_reads_the_trigger() {
        let half = GripperOpenFraction::try_from(0.5).expect("in range");
        let mut latch = latch();
        // Released: the gripper rests at the open fraction.
        let released = map_sample(&calibration(), half, &frame(0.0, 0.0), &mut latch).unwrap();
        assert_eq!((released.left_opening, released.right_opening), (0.5, 0.5));
        // Trigger travel 0.3 commands 0.15, below ENGAGE_AT, yet engages nothing.
        let partial = map_sample(&calibration(), half, &frame(-42.0, 0.0), &mut latch).unwrap();
        assert!((partial.right_opening - 0.15).abs() < 1e-9);
        assert_eq!(partial.engaged, Engaged::default());
        // A full squeeze closes the gripper and engages that arm.
        let squeezed = map_sample(&calibration(), half, &frame(-60.0, 0.0), &mut latch).unwrap();
        assert_eq!(squeezed.right_opening, 0.0);
        assert_eq!(squeezed.engaged, RIGHT);
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
