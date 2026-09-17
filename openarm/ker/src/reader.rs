// The device thread: owns the transport, handshakes, decodes and maps frames,
// and keeps the newest mapped sample on a watch channel for the publish
// tasks. Device I/O is blocking, so this runs on a dedicated OS thread.
//
// Failure policy: a device this node's channel map does not describe (wrong
// hardware generation, too few channels, an undecodable schema) cancels the
// node so the launch fails loudly; everything transient (unplug, bad
// checksums, silence) clears the sample, backs off and reconnects.
//
// Engagement lives here because one owner has to hold it: four publish tasks
// read it, and a latch per task would let a side's arm and gripper disagree.
// An arm engages when its trigger is squeezed to the engage opening, having
// been seen released first, so a device returning mid-teleop under a held
// trigger never resumes motion on its own.

use std::time::{Duration, Instant};

use openarm_description::{ARM_DOF, Side};
use peppylib::runtime::CancellationToken;
use tokio::sync::{oneshot, watch};
use tracing::{error, info, warn};

use crate::label;
use crate::mapping::{ChannelMap, GripperOpenFraction, MapError};
use crate::protocol::{
    CMD_PING, CMD_STANDBY, CMD_STREAM, Deframer, FrameLayout, KerFrame, PingParse, Schema,
};
use crate::transport::{self, KerTransport, TransportConfig};

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
/// Handshake bytes kept while scanning for a PING response. A device left
/// streaming by a previous session fills this between pings; a response spans
/// far less than this, so trimming the front never splits one.
const MAX_HANDSHAKE_BUFFER: usize = 8192;

/// A launcher `engage_opening` outside [0, 1).
#[derive(Debug, thiserror::Error)]
#[error(
    "engage_opening is the trigger opening a squeeze must reach: at least 0 and below 1, \
     since a released trigger reads 1 and would engage on connect. Try 0.2, got {0}"
)]
pub struct EngageOpeningOutOfRange(pub f64);

/// The trigger opening at or below which a squeeze engages its arm; strictly
/// below 1 (fully open).
#[derive(Debug, Clone, Copy)]
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

/// One flag per arm: which arms track the KER, and inside the latch, which
/// triggers have been seen released.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SideFlags {
    pub left: bool,
    pub right: bool,
}

impl SideFlags {
    /// Neither side set.
    pub const NONE: Self = Self {
        left: false,
        right: false,
    };

    pub fn side(self, side: Side) -> bool {
        match side {
            Side::Left => self.left,
            Side::Right => self.right,
        }
    }

    fn set(&mut self, side: Side, value: bool) {
        match side {
            Side::Left => self.left = value,
            Side::Right => self.right = value,
        }
    }
}

/// One mapped bimanual sample: what the publish tasks stream for each engaged
/// arm while fresh.
#[derive(Debug, Clone)]
pub struct KerSample {
    pub left_joints: [f64; ARM_DOF],
    pub right_joints: [f64; ARM_DOF],
    pub left_gripper_opening: f64,
    pub right_gripper_opening: f64,
    pub engaged: SideFlags,
    pub received_at: Instant,
}

impl KerSample {
    pub fn joints(&self, side: Side) -> [f64; ARM_DOF] {
        match side {
            Side::Left => self.left_joints,
            Side::Right => self.right_joints,
        }
    }

    pub fn gripper_opening(&self, side: Side) -> f64 {
        match side {
            Side::Left => self.left_gripper_opening,
            Side::Right => self.right_gripper_opening,
        }
    }
}

pub struct ReaderConfig {
    pub transport: TransportConfig,
    pub channels: ChannelMap,
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
    // Latched per reason: an unplugged KER retries every second, and each
    // attempt fails the same way until something changes.
    let mut warned: Option<String> = None;
    while !token.is_cancelled() {
        match session(&cfg, &tx, &token) {
            SessionEnd::Stop => break,
            SessionEnd::Transient(reason) => {
                let _ = tx.send(None);
                if warned.as_deref() != Some(reason.as_str()) {
                    warn!("KER connection lost ({reason}); retrying every {RECONNECT_BACKOFF:?}");
                    warned = Some(reason);
                }
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

/// Handshake, start the stream, and map frames until the link breaks. Every
/// exit past the handshake leaves the device in standby, so the next session
/// handshakes against a quiet device.
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
    if let Err(e) = ChannelMap::accepts(&schema.metadata, layout.angle_count()) {
        return SessionEnd::Fatal(e.to_string());
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

    let end = stream_frames(transport, cfg, tx, token, &layout, leftover);
    // Best effort: leave the device quiet on the way out.
    let _ = transport.write_all(&[CMD_STANDBY]);
    end
}

/// Decode and map frames until the link breaks or the node stops.
fn stream_frames(
    transport: &mut dyn KerTransport,
    cfg: &ReaderConfig,
    tx: &watch::Sender<Option<KerSample>>,
    token: &CancellationToken,
    layout: &FrameLayout,
    leftover: Vec<u8>,
) -> SessionEnd {
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
        // Pushed before the silence check, so bytes that arrive after a long
        // quiet spell count as the recovery they are.
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
            match map_frame(&cfg.channels, cfg.gripper_open_fraction, &frame) {
                Ok(mapped) => {
                    mapping_warned = false;
                    let received_at = Instant::now();
                    let engaged = engage.update(mapped.triggers, received_at);
                    if tx
                        .send(Some(mapped.into_sample(engaged, received_at)))
                        .is_err()
                    {
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

        if last_frame_at.elapsed() > SILENCE_RECONNECT {
            return SessionEnd::Transient(format!(
                "no valid frames for {SILENCE_RECONNECT:?} while connected"
            ));
        }
    }
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
        // A device still streaming from a previous session fills this buffer
        // between pings; keep only what a response could still span.
        if buf.len() > MAX_HANDSHAKE_BUFFER {
            buf.drain(..buf.len() - MAX_HANDSHAKE_BUFFER);
        }
        match Schema::parse_ping(&buf) {
            PingParse::NeedMore => continue,
            PingParse::Parsed { schema, consumed } => {
                return Ok((schema, buf.split_off(consumed)));
            }
        }
    }
    Err(SessionEnd::Transient(format!(
        "the device answered no PING in {HANDSHAKE_DEADLINE:?}; confirm it is the KER \
         and not another Espressif device (`lsusb -d 303a:`)"
    )))
}

/// Per-arm engagement over the stream of usable frames. A trigger squeezed to
/// the engage opening engages its arm, once that trigger has been seen
/// released; a gap between usable frames of at least the stale timeout
/// disengages both and requires a release again. Each session starts a fresh
/// latch, so a reconnect does too.
struct EngageLatch {
    engage_opening: EngageOpening,
    stale_timeout: Duration,
    engaged: SideFlags,
    /// Per side: the trigger has read open since the last reset, so the next
    /// squeeze is a new one and not a hand that never let go.
    released: SideFlags,
    last_frame_at: Option<Instant>,
}

impl EngageLatch {
    fn new(engage_opening: EngageOpening, stale_timeout: Duration) -> Self {
        Self {
            engage_opening,
            stale_timeout,
            engaged: SideFlags::NONE,
            released: SideFlags::NONE,
            last_frame_at: None,
        }
    }

    /// Fold in one usable frame's trigger openings, captured at `at`.
    fn update(&mut self, triggers: SideValues, at: Instant) -> SideFlags {
        let stalled = self
            .last_frame_at
            .is_some_and(|last| at.duration_since(last) >= self.stale_timeout);
        if stalled {
            if self.engaged != SideFlags::NONE {
                info!("KER frames stalled; both arms disengaged, release and squeeze to re-engage");
            }
            self.engaged = SideFlags::NONE;
            self.released = SideFlags::NONE;
        }
        let threshold = self.engage_opening.fraction();
        for side in [Side::Left, Side::Right] {
            let opening = triggers.side(side);
            if opening > threshold {
                self.released.set(side, true);
            } else if self.released.side(side) && !self.engaged.side(side) {
                self.engaged.set(side, true);
                info!("KER {} arm engaged, tracking the leader", label(side));
            }
        }
        self.last_frame_at = Some(at);
        self.engaged
    }
}

/// One value per arm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SideValues {
    pub left: f64,
    pub right: f64,
}

impl SideValues {
    fn side(self, side: Side) -> f64 {
        match side {
            Side::Left => self.left,
            Side::Right => self.right,
        }
    }
}

/// One frame's mapped values, before engagement decides what streams.
struct MappedFrame {
    left_joints: [f64; ARM_DOF],
    right_joints: [f64; ARM_DOF],
    /// Trigger openings, which engagement reads unscaled.
    triggers: SideValues,
    gripper_openings: SideValues,
}

impl MappedFrame {
    fn into_sample(self, engaged: SideFlags, received_at: Instant) -> KerSample {
        KerSample {
            left_joints: self.left_joints,
            right_joints: self.right_joints,
            left_gripper_opening: self.gripper_openings.left,
            right_gripper_opening: self.gripper_openings.right,
            engaged,
            received_at,
        }
    }
}

/// Map one frame's channels. Pure: engagement is folded in by the caller.
fn map_frame(
    channels: &ChannelMap,
    open_fraction: GripperOpenFraction,
    frame: &KerFrame,
) -> Result<MappedFrame, MapError> {
    let angles = &frame.angles_deg;
    Ok(MappedFrame {
        left_joints: channels.left.joint_radians(angles)?,
        right_joints: channels.right.joint_radians(angles)?,
        triggers: SideValues {
            left: channels.left_trigger.opening(angles)?,
            right: channels.right_trigger.opening(angles)?,
        },
        gripper_openings: SideValues {
            left: channels
                .left_trigger
                .gripper_opening(angles, open_fraction)?,
            right: channels
                .right_trigger
                .gripper_opening(angles, open_fraction)?,
        },
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
    use std::sync::{Arc, Mutex};

    use openarm_description::HardwareVersion;

    use super::*;
    use crate::mapping::tests::channel_map;
    use crate::protocol::fixtures::{ping_response_for, stream_packet};

    const STALE: Duration = Duration::from_millis(250);
    const FRAME: Duration = Duration::from_millis(5);
    const ENGAGE_AT: f64 = 0.2;
    const OPEN: f64 = 1.0;
    const SQUEEZED: f64 = 0.02;
    const CHANNELS: u8 = 16;
    /// Trigger angles (deg): a released trigger reads 0, a full squeeze -60 on
    /// the right (CH08) and +60 on the left (CH16).
    const SQUEEZE_DEG: f32 = 60.0;
    /// How long a session test waits for the device to deliver a sample.
    const SESSION_DEADLINE: Duration = Duration::from_secs(30);
    const LEFT: SideFlags = SideFlags {
        left: true,
        right: false,
    };
    const RIGHT: SideFlags = SideFlags {
        left: false,
        right: true,
    };
    const BOTH: SideFlags = SideFlags {
        left: true,
        right: true,
    };

    fn latch() -> EngageLatch {
        EngageLatch::new(EngageOpening::try_from(ENGAGE_AT).expect("in range"), STALE)
    }

    fn triggers(left: f64, right: f64) -> SideValues {
        SideValues { left, right }
    }

    /// Engage a latch the documented way: a released frame, then a squeeze.
    fn engage_both(latch: &mut EngageLatch, at: Instant) -> Instant {
        latch.update(triggers(OPEN, OPEN), at);
        latch.update(triggers(SQUEEZED, SQUEEZED), at + FRAME);
        at + FRAME
    }

    #[test]
    fn engage_opening_accepts_only_fractions_below_fully_open() {
        for fraction in [0.0, ENGAGE_AT, 0.999] {
            assert!(EngageOpening::try_from(fraction).is_ok(), "{fraction}");
        }
        for fraction in [-0.01, 1.0, 1.5, f64::NAN, f64::INFINITY] {
            let refused = EngageOpening::try_from(fraction)
                .expect_err("out of range")
                .to_string();
            assert!(refused.contains("engage_opening"), "{refused}");
        }
    }

    #[test]
    fn a_squeeze_engages_only_its_own_arm_and_release_keeps_it_tracking() {
        let mut latch = latch();
        let t0 = Instant::now();
        assert_eq!(latch.update(triggers(OPEN, OPEN), t0), SideFlags::NONE);
        assert_eq!(latch.update(triggers(OPEN, SQUEEZED), t0 + FRAME), RIGHT);
        assert_eq!(latch.update(triggers(OPEN, OPEN), t0 + FRAME * 2), RIGHT);
        assert_eq!(latch.update(triggers(SQUEEZED, OPEN), t0 + FRAME * 3), BOTH);
        assert_eq!(latch.update(triggers(OPEN, OPEN), t0 + FRAME * 4), BOTH);
    }

    #[test]
    fn the_engage_opening_itself_engages() {
        let mut latch = latch();
        let t0 = Instant::now();
        latch.update(triggers(OPEN, OPEN), t0);
        assert_eq!(latch.update(triggers(ENGAGE_AT, OPEN), t0 + FRAME), LEFT);
    }

    #[test]
    fn a_trigger_held_from_the_first_frame_engages_nothing() {
        let mut latch = latch();
        let t0 = Instant::now();
        for tick in 0..5 {
            assert_eq!(
                latch.update(triggers(SQUEEZED, SQUEEZED), t0 + FRAME * tick),
                SideFlags::NONE,
                "tick {tick}: a hand already on the trigger must not engage"
            );
        }
        // Releasing arms it; the next squeeze engages.
        latch.update(triggers(OPEN, OPEN), t0 + FRAME * 5);
        assert_eq!(
            latch.update(triggers(SQUEEZED, SQUEEZED), t0 + FRAME * 6),
            BOTH
        );
    }

    #[test]
    fn a_frame_gap_of_the_stale_timeout_disengages_both_arms() {
        let mut latch = latch();
        let engaged_at = engage_both(&mut latch, Instant::now());
        let just_inside = engaged_at + STALE - FRAME;
        assert_eq!(latch.update(triggers(OPEN, OPEN), just_inside), BOTH);
        assert_eq!(
            latch.update(triggers(OPEN, OPEN), just_inside + STALE),
            SideFlags::NONE
        );
    }

    #[test]
    fn a_held_trigger_after_a_stall_stays_disengaged_until_released() {
        let mut latch = latch();
        let engaged_at = engage_both(&mut latch, Instant::now());
        // Frames return with the hand still on both triggers.
        let returned = engaged_at + STALE;
        assert_eq!(
            latch.update(triggers(SQUEEZED, SQUEEZED), returned),
            SideFlags::NONE
        );
        assert_eq!(
            latch.update(triggers(SQUEEZED, SQUEEZED), returned + FRAME),
            SideFlags::NONE
        );
        // Only a release, then a fresh squeeze, engages again.
        latch.update(triggers(OPEN, SQUEEZED), returned + FRAME * 2);
        assert_eq!(
            latch.update(triggers(SQUEEZED, SQUEEZED), returned + FRAME * 3),
            LEFT
        );
    }

    /// A KER that answers PING with the reference schema and sends frames only
    /// once STREAM arrives, as firmware 2.0.0 does.
    struct FakeKer {
        hardware: &'static str,
        channels: u8,
        /// Trigger angle (deg) both triggers report on every frame.
        squeeze_deg: f32,
        /// Bytes delivered per read, so a caller can force partial reads.
        chunk_size: usize,
        writes: Arc<Mutex<Vec<u8>>>,
        pending: Vec<u8>,
        streaming: bool,
    }

    impl FakeKer {
        fn new(writes: Arc<Mutex<Vec<u8>>>) -> Self {
            Self {
                hardware: "2.0.0",
                channels: CHANNELS,
                squeeze_deg: 0.0,
                chunk_size: usize::MAX,
                writes,
                pending: Vec::new(),
                streaming: false,
            }
        }

        fn frame(&self) -> Vec<u8> {
            let mut angles = [0.0f32; CHANNELS as usize];
            // CH08 squeezes negative, CH16 positive.
            angles[7] = -self.squeeze_deg;
            angles[15] = self.squeeze_deg;
            stream_packet(1, &angles[..self.channels as usize], 0, false)
        }
    }

    impl KerTransport for FakeKer {
        fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
            self.writes.lock().unwrap().extend_from_slice(bytes);
            match bytes {
                [CMD_PING] => self
                    .pending
                    .extend(ping_response_for(self.hardware, self.channels)),
                [CMD_STREAM] => self.streaming = true,
                [CMD_STANDBY] => self.streaming = false,
                _ => {}
            }
            Ok(())
        }

        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.pending.is_empty() && self.streaming {
                self.pending.extend(self.frame());
            }
            if self.pending.is_empty() {
                std::thread::sleep(Duration::from_millis(1));
                return Ok(0);
            }
            let n = self.pending.len().min(buf.len()).min(self.chunk_size);
            buf[..n].copy_from_slice(&self.pending[..n]);
            self.pending.drain(..n);
            Ok(n)
        }

        fn flush_input(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn reader_config() -> ReaderConfig {
        ReaderConfig {
            transport: TransportConfig::Usb,
            channels: channel_map(),
            engage_opening: EngageOpening::try_from(ENGAGE_AT).expect("in range"),
            gripper_open_fraction: GripperOpenFraction::try_from(0.5).expect("in range"),
            stale_timeout: STALE,
            log_raw: false,
        }
    }

    /// Run one session against `ker` until it delivers a sample (or the
    /// deadline passes), then cancel it and hand back what it streamed.
    fn first_sample(mut ker: FakeKer) -> (Option<KerSample>, SessionEnd, Vec<u8>) {
        let writes = ker.writes.clone();
        let cfg = reader_config();
        let (tx, rx) = watch::channel(None);
        let token = CancellationToken::new();
        let session = {
            let token = token.clone();
            std::thread::spawn(move || run_session(&mut ker, &cfg, &tx, &token))
        };

        let deadline = Instant::now() + SESSION_DEADLINE;
        while rx.borrow().is_none() && Instant::now() < deadline && !session.is_finished() {
            std::thread::sleep(Duration::from_millis(5));
        }
        let sample = rx.borrow().clone();
        token.cancel();
        let end = session.join().expect("session thread");
        let writes = writes.lock().unwrap().clone();
        (sample, end, writes)
    }

    /// Run a session that is expected to refuse the device, with a watchdog so
    /// a regression fails the test rather than wedging it.
    fn refusal(ker: FakeKer) -> (SessionEnd, Vec<u8>) {
        let (sample, end, writes) = first_sample(ker);
        assert!(sample.is_none(), "a refused device streams nothing");
        (end, writes)
    }

    #[test]
    fn a_session_starts_the_stream_and_maps_the_first_frame() {
        let writes = Arc::new(Mutex::new(Vec::new()));
        let (sample, end, writes) = first_sample(FakeKer::new(writes));

        assert!(matches!(end, SessionEnd::Stop), "{end:?}");
        let sample = sample.expect("a frame reached the sample channel");
        // Released triggers: nothing engages, and each gripper rests at the
        // open fraction.
        assert_eq!(sample.engaged, SideFlags::NONE);
        assert_eq!(sample.left_gripper_opening, 0.5);
        assert_eq!(sample.right_gripper_opening, 0.5);
        // All-zero angles clamp into the follower's limits, so j4 sits on the
        // elbow floor rather than at 0.
        for side in [Side::Left, Side::Right] {
            let limits = HardwareVersion::V2.joint_limits(side);
            let joints = sample.joints(side);
            for (j, (joint, [lo, hi])) in joints.into_iter().zip(limits).enumerate() {
                assert_eq!(joint, 0.0f64.clamp(lo, hi), "{side:?} j{}", j + 1);
            }
        }

        let ping = writes.iter().position(|&b| b == CMD_PING).expect("pinged");
        let stream = writes
            .iter()
            .position(|&b| b == CMD_STREAM)
            .expect("stream started");
        assert!(ping < stream, "STREAM follows the handshake: {writes:?}");
        assert_eq!(writes.last(), Some(&CMD_STANDBY), "leaves the device quiet");
    }

    #[test]
    fn a_session_delivers_a_frame_split_across_reads() {
        let mut ker = FakeKer::new(Arc::new(Mutex::new(Vec::new())));
        ker.chunk_size = 7;
        let (sample, end, _) = first_sample(ker);
        assert!(matches!(end, SessionEnd::Stop), "{end:?}");
        assert!(
            sample.is_some(),
            "a packet arriving in pieces still reaches the sample channel"
        );
    }

    #[test]
    fn a_held_trigger_does_not_re_engage_on_a_new_session() {
        let mut ker = FakeKer::new(Arc::new(Mutex::new(Vec::new())));
        ker.squeeze_deg = SQUEEZE_DEG;
        let (sample, _, _) = first_sample(ker);
        assert_eq!(
            sample.expect("a frame arrived").engaged,
            SideFlags::NONE,
            "a session that opens under a held trigger must not resume motion"
        );
    }

    #[test]
    fn a_session_refuses_a_device_of_another_hardware_generation() {
        let mut ker = FakeKer::new(Arc::new(Mutex::new(Vec::new())));
        ker.hardware = "3.0.0";
        let (end, writes) = refusal(ker);
        let SessionEnd::Fatal(reason) = end else {
            panic!("expected a fatal refusal, got {end:?}");
        };
        assert!(reason.contains("3.0.0"), "{reason}");
        assert!(
            !writes.contains(&CMD_STREAM),
            "a refused device is never started"
        );
    }

    #[test]
    fn a_session_refuses_a_device_streaming_too_few_channels() {
        let mut ker = FakeKer::new(Arc::new(Mutex::new(Vec::new())));
        ker.channels = CHANNELS - 1;
        let (end, writes) = refusal(ker);
        let SessionEnd::Fatal(reason) = end else {
            panic!("expected a fatal refusal, got {end:?}");
        };
        assert!(reason.contains("15 channels"), "{reason}");
        assert!(
            !writes.contains(&CMD_STREAM),
            "a refused device is never started"
        );
    }
}
