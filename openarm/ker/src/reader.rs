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

use openarm_description::{ARM_DOF, HardwareVersion, Side};
use peppylib::runtime::CancellationToken;
use tokio::sync::{oneshot, watch};
use tracing::{error, info, warn};

use crate::engage::{EngageLatch, EngageOpening};
use crate::mapping::{ChannelMap, GripperOpenFraction, MapError};
use crate::protocol::{
    CMD_PING, CMD_STANDBY, CMD_STREAM, Deframer, FrameLayout, KerFrame, PingParse, Schema,
};
use crate::side::{SideFlags, SideValues};
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
/// streaming by a previous session fills this between pings.
const MAX_HANDSHAKE_BUFFER: usize = 8192;
/// Trimming the front of that buffer cannot split the newest response.
const _: () = assert!(MAX_HANDSHAKE_BUFFER > crate::protocol::MAX_PING_RESPONSE_LEN);

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
    pub version: HardwareVersion,
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
    // Latched per reason while a run of attempts keeps failing the same way,
    // and cleared by a connection, so every episode warns once.
    let mut warned: Option<String> = None;
    while !token.is_cancelled() {
        let mut connected = false;
        let end = match transport::open(&cfg.transport) {
            Ok(mut transport) => run_session(transport.as_mut(), &cfg, &tx, &token, &mut connected),
            Err(e) => SessionEnd::Transient(format!("open: {e}")),
        };
        if connected {
            warned = None;
        }
        match end {
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

/// One connection lifetime: handshake, start the stream, map frames until the
/// link breaks. `connected` reports whether the handshake reached a device
/// this node can read. The stream is stopped on the way out, so the next
/// session handshakes against a quiet device.
fn run_session(
    transport: &mut dyn KerTransport,
    cfg: &ReaderConfig,
    tx: &watch::Sender<Option<KerSample>>,
    token: &CancellationToken,
    connected: &mut bool,
) -> SessionEnd {
    let (schema, leftover) = match handshake(transport, token) {
        Ok(parsed) => parsed,
        Err(end) => return end,
    };
    let layout = match FrameLayout::try_new(&schema) {
        Ok(layout) => layout,
        Err(e) => return SessionEnd::Fatal(e.to_string()),
    };
    let channels = match ChannelMap::for_device(cfg.version, &schema.metadata, layout.angle_count())
    {
        Ok(channels) => channels,
        Err(e) => return SessionEnd::Fatal(e.to_string()),
    };
    info!(
        "KER connected: fw {} hw {} updated {} ({} channels)",
        schema.metadata.firmware,
        schema.metadata.hardware,
        schema.metadata.updated,
        layout.angle_count()
    );
    *connected = true;

    let end = stream_frames(transport, cfg, &channels, tx, token, &layout, leftover);
    // Best effort: stop the stream on the way out, whether or not it started.
    let _ = transport.write_all(&[CMD_STANDBY]);
    end
}

/// Decode and map frames until the link breaks or the node stops.
#[allow(clippy::too_many_arguments)]
fn stream_frames(
    transport: &mut dyn KerTransport,
    cfg: &ReaderConfig,
    channels: &ChannelMap,
    tx: &watch::Sender<Option<KerSample>>,
    token: &CancellationToken,
    layout: &FrameLayout,
    leftover: Vec<u8>,
) -> SessionEnd {
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
            match map_frame(channels, cfg.gripper_open_fraction, &frame) {
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
    let triggers = SideValues {
        left: channels.left_trigger.opening(angles)?,
        right: channels.right_trigger.opening(angles)?,
    };
    Ok(MappedFrame {
        left_joints: channels.left.joint_radians(angles)?,
        right_joints: channels.right.joint_radians(angles)?,
        gripper_openings: triggers.scaled(open_fraction.fraction()),
        triggers,
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

    use super::*;
    use crate::mapping::fixtures::{
        CHANNELS, LEFT_SQUEEZE_DEG, LEFT_TRIGGER, RIGHT_SQUEEZE_DEG, RIGHT_TRIGGER, numbered_frame,
        probe_radians,
    };
    use crate::protocol::fixtures::{ping_response_for, stream_packet};

    const STALE: Duration = Duration::from_millis(250);
    const ENGAGE_AT: f64 = 0.2;
    const OPEN_FRACTION: f64 = 0.5;
    /// How long a session test waits for a sample, and then for the session
    /// thread to notice cancellation.
    const SESSION_DEADLINE: Duration = Duration::from_secs(30);

    /// A KER that answers PING with the reference schema and sends frames only
    /// once STREAM arrives, as firmware 2.0.0 does.
    struct FakeKer {
        hardware: &'static str,
        channels: usize,
        /// Angles every frame reports, CH01 at index 0.
        angles: Vec<f32>,
        /// Bytes delivered per read, so a caller can force partial reads.
        chunk_size: usize,
        writes: Arc<Mutex<Vec<u8>>>,
        pending: Vec<u8>,
        streaming: bool,
    }

    impl FakeKer {
        fn new() -> Self {
            Self {
                hardware: "2.0.0",
                channels: CHANNELS,
                angles: numbered_frame(),
                chunk_size: usize::MAX,
                writes: Arc::new(Mutex::new(Vec::new())),
                pending: Vec::new(),
                streaming: false,
            }
        }

        /// Both triggers squeezed to their stops, the rest of the channels as
        /// they were.
        fn squeezing(mut self) -> Self {
            self.angles[RIGHT_TRIGGER] = RIGHT_SQUEEZE_DEG as f32;
            self.angles[LEFT_TRIGGER] = LEFT_SQUEEZE_DEG as f32;
            self
        }

        /// Both triggers released, which is what a resting KER reports.
        fn released(mut self) -> Self {
            self.angles[RIGHT_TRIGGER] = 0.0;
            self.angles[LEFT_TRIGGER] = 0.0;
            self
        }
    }

    impl KerTransport for FakeKer {
        fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
            self.writes.lock().expect("writes").extend_from_slice(bytes);
            match bytes {
                [CMD_PING] => self
                    .pending
                    .extend(ping_response_for(self.hardware, self.channels as u8)),
                [CMD_STREAM] => self.streaming = true,
                [CMD_STANDBY] => self.streaming = false,
                _ => {}
            }
            Ok(())
        }

        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.pending.is_empty() && self.streaming {
                self.pending
                    .extend(stream_packet(1, &self.angles[..self.channels], 0, false));
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
            version: HardwareVersion::V2,
            engage_opening: EngageOpening::try_from(ENGAGE_AT).expect("in range"),
            gripper_open_fraction: GripperOpenFraction::try_from(OPEN_FRACTION).expect("in range"),
            stale_timeout: STALE,
            log_raw: false,
        }
    }

    struct SessionRun {
        sample: Option<KerSample>,
        end: SessionEnd,
        writes: Vec<u8>,
        connected: bool,
    }

    /// Run one session against `ker` until it delivers a sample or stops, then
    /// cancel it. A session that ignores cancellation fails the test instead of
    /// wedging the suite.
    fn run_one(mut ker: FakeKer) -> SessionRun {
        let writes = ker.writes.clone();
        let cfg = reader_config();
        let (tx, rx) = watch::channel(None);
        let token = CancellationToken::new();
        let connected = Arc::new(Mutex::new(false));
        let session = {
            let token = token.clone();
            let connected = connected.clone();
            std::thread::spawn(move || {
                let mut flag = false;
                let end = run_session(&mut ker, &cfg, &tx, &token, &mut flag);
                *connected.lock().expect("connected") = flag;
                end
            })
        };

        let deadline = Instant::now() + SESSION_DEADLINE;
        while rx.borrow().is_none() && Instant::now() < deadline && !session.is_finished() {
            std::thread::sleep(Duration::from_millis(5));
        }
        let sample = rx.borrow().clone();
        token.cancel();

        let stop_by = Instant::now() + SESSION_DEADLINE;
        while !session.is_finished() && Instant::now() < stop_by {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            session.is_finished(),
            "the session ignored cancellation and would hang the suite"
        );
        SessionRun {
            sample,
            end: session.join().expect("session thread"),
            writes: writes.lock().expect("writes").clone(),
            connected: *connected.lock().expect("connected"),
        }
    }

    #[test]
    fn a_session_starts_the_stream_and_maps_the_first_frame() {
        let run = run_one(FakeKer::new().released());

        assert!(matches!(run.end, SessionEnd::Stop), "{:?}", run.end);
        assert!(run.connected, "a readable device counts as connected");
        assert_eq!(
            run.writes,
            vec![CMD_STANDBY, CMD_PING, CMD_STREAM, CMD_STANDBY],
            "standby and ping to handshake, stream to start, standby to stop"
        );

        let sample = run.sample.expect("a frame reached the sample channel");
        // Released triggers: nothing engages, and each gripper rests at the
        // open fraction.
        assert_eq!(sample.engaged, SideFlags::NONE);
        assert_eq!(sample.left_gripper_opening, OPEN_FRACTION);
        assert_eq!(sample.right_gripper_opening, OPEN_FRACTION);
        // Each arm reads its own channels, end to end through the session.
        let frame = numbered_frame();
        for (side, first_channel) in [(Side::Right, 0), (Side::Left, 8)] {
            for (j, expected) in probe_radians(&frame, first_channel).into_iter().enumerate() {
                assert!(
                    (sample.joints(side)[j] - expected).abs() < 1e-12,
                    "{side:?} j{}: {} != {expected}",
                    j + 1,
                    sample.joints(side)[j]
                );
            }
        }
    }

    #[test]
    fn a_session_delivers_a_frame_split_across_reads() {
        let mut ker = FakeKer::new().released();
        ker.chunk_size = 7;
        let run = run_one(ker);

        assert!(matches!(run.end, SessionEnd::Stop), "{:?}", run.end);
        let sample = run
            .sample
            .expect("a split packet still reaches the channel");
        assert_eq!(sample.left_gripper_opening, OPEN_FRACTION);
        let frame = numbered_frame();
        let expected = probe_radians(&frame, 0);
        assert!((sample.joints(Side::Right)[0] - expected[0]).abs() < 1e-12);
    }

    #[test]
    fn a_held_trigger_does_not_engage_on_a_new_session() {
        let run = run_one(FakeKer::new().squeezing());
        assert_eq!(
            run.sample.expect("a frame arrived").engaged,
            SideFlags::NONE,
            "a session that opens under a held trigger must not resume motion"
        );
    }

    #[test]
    fn a_session_refuses_a_device_of_another_hardware_generation() {
        let mut ker = FakeKer::new();
        ker.hardware = "3.0.0";
        let run = run_one(ker);

        let SessionEnd::Fatal(reason) = run.end else {
            panic!("expected a fatal refusal, got {:?}", run.end);
        };
        assert!(reason.contains("3.0.0"), "{reason}");
        assert!(run.sample.is_none(), "a refused device streams nothing");
        assert!(!run.connected, "a refused device is not a connection");
        assert!(
            !run.writes.contains(&CMD_STREAM),
            "a refused device is never started"
        );
    }

    #[test]
    fn a_session_refuses_a_device_streaming_too_few_channels() {
        let mut ker = FakeKer::new();
        ker.channels = CHANNELS - 1;
        let run = run_one(ker);

        let SessionEnd::Fatal(reason) = run.end else {
            panic!("expected a fatal refusal, got {:?}", run.end);
        };
        assert!(reason.contains("15 channels"), "{reason}");
        assert!(run.sample.is_none(), "a refused device streams nothing");
        assert!(
            !run.writes.contains(&CMD_STREAM),
            "a refused device is never started"
        );
    }
}
