// The device thread: owns the transport, handshakes, decodes and maps frames,
// and keeps the newest mapped sample on a watch channel for the publish
// tasks. Device I/O is blocking, so this runs on a dedicated OS thread.
//
// Failure policy: a device this node's channel map does not describe (wrong
// hardware generation, too few channels, an undecodable schema) cancels the
// node so the launch fails loudly; everything transient (unplug, bad
// checksums, silence) clears the sample, backs off and reconnects.
//
// One EngageLatch per session is constructed here: the four publish tasks all
// read the engagement it folds into each sample, so a side's arm and gripper
// always agree. The policy itself lives in engage.

use std::time::{Duration, Instant};

use openarm_description::{ARM_DOF, HardwareVersion, Side};
use peppylib::runtime::CancellationToken;
use tokio::sync::{oneshot, watch};
use tracing::{error, info, warn};

use crate::engage::{EngageLatch, EngageTriggerOpening};
use crate::mapping::{ChannelMap, GripperOpenFraction, MappedFrame};
use crate::protocol::{
    CMD_PING, CMD_STANDBY, CMD_STREAM, Deframer, FrameLayout, KerFrame, PingParse, Schema,
};
use crate::side::SideFlags;
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
    /// One mapped frame plus the engagement the latch folded in.
    fn from_mapped(mapped: MappedFrame, engaged: SideFlags, received_at: Instant) -> Self {
        Self {
            left_joints: mapped.left_joints,
            right_joints: mapped.right_joints,
            left_gripper_opening: mapped.gripper_openings.left,
            right_gripper_opening: mapped.gripper_openings.right,
            engaged,
            received_at,
        }
    }

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
    pub engage_trigger_opening: EngageTriggerOpening,
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
        let report = match transport::open(&cfg.transport) {
            Ok(mut transport) => run_session(transport.as_mut(), &cfg, &tx, &token),
            Err(e) => SessionReport::refused(SessionEnd::Transient(format!("open: {e}"))),
        };
        // A session that delivered frames is a link that worked, so the next
        // failure is a new episode and warns again.
        if report.streamed {
            warned = None;
        }
        match report.end {
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

/// What a session did: how it ended, and whether it ever delivered a frame.
struct SessionReport {
    end: SessionEnd,
    streamed: bool,
}

impl SessionReport {
    /// A session that never reached a readable device.
    fn refused(end: SessionEnd) -> Self {
        Self {
            end,
            streamed: false,
        }
    }
}

/// A handshaken device this node can read.
struct Connected {
    layout: FrameLayout,
    channels: ChannelMap,
    /// Bytes read past the PING response, which may already hold frames.
    leftover: Vec<u8>,
}

/// One connection lifetime: handshake, start the stream, map frames until the
/// link breaks. The stream is stopped on the way out, so the next session
/// handshakes against a quiet device.
fn run_session(
    transport: &mut dyn KerTransport,
    cfg: &ReaderConfig,
    tx: &watch::Sender<Option<KerSample>>,
    token: &CancellationToken,
) -> SessionReport {
    let device = match connect(transport, cfg, token) {
        Ok(device) => device,
        Err(end) => return SessionReport::refused(end),
    };
    let (end, streamed) = stream_frames(transport, cfg, &device, tx, token);
    // Best effort: stop the stream on the way out, whether or not it started.
    let _ = transport.write_all(&[CMD_STANDBY]);
    SessionReport { end, streamed }
}

/// Handshake and check the device against the channel map this node reads.
fn connect(
    transport: &mut dyn KerTransport,
    cfg: &ReaderConfig,
    token: &CancellationToken,
) -> Result<Connected, SessionEnd> {
    let (schema, leftover) = handshake(transport, token)?;
    let layout = FrameLayout::try_new(&schema).map_err(|e| SessionEnd::Fatal(e.to_string()))?;
    let channels = ChannelMap::for_device(cfg.version, &schema.metadata, layout.angle_count())
        .map_err(|e| SessionEnd::Fatal(e.to_string()))?;
    info!(
        "KER connected: fw {} hw {} updated {} ({} channels)",
        schema.metadata.firmware,
        schema.metadata.hardware,
        schema.metadata.updated,
        layout.angle_count()
    );
    Ok(Connected {
        layout,
        channels,
        leftover,
    })
}

/// Decode and map frames until the link breaks or the node stops, reporting
/// whether any frame reached the sample channel.
fn stream_frames(
    transport: &mut dyn KerTransport,
    cfg: &ReaderConfig,
    device: &Connected,
    tx: &watch::Sender<Option<KerSample>>,
    token: &CancellationToken,
) -> (SessionEnd, bool) {
    // The firmware answers PING in standby; frames flow once STREAM arrives.
    if let Err(e) = transport.write_all(&[CMD_STREAM]) {
        return (SessionEnd::Transient(format!("start stream: {e}")), false);
    }
    let mut deframer = Deframer::new(device.layout.payload_len());
    deframer.push(&device.leftover);
    let mut engage = EngageLatch::new(cfg.engage_trigger_opening, cfg.stale_timeout);
    let mut chunk = [0u8; 4096];
    let mut last_frame_at = Instant::now();
    let mut last_raw_log = Instant::now();
    let mut consecutive_bad = 0u32;
    let mut mapping_warned = false;
    let mut streamed = false;

    while !token.is_cancelled() {
        let read = match transport.read(&mut chunk) {
            Ok(n) => n,
            Err(e) => return (SessionEnd::Transient(format!("read: {e}")), streamed),
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
                        return (
                            SessionEnd::Transient(format!(
                                "{consecutive_bad} corrupt frames in a row"
                            )),
                            streamed,
                        );
                    }
                    continue;
                }
            };
            consecutive_bad = 0;
            let frame = device.layout.parse(&payload);
            last_frame_at = Instant::now();
            if cfg.log_raw && last_raw_log.elapsed() >= RAW_LOG_INTERVAL {
                last_raw_log = Instant::now();
                info!("KER raw: {}", format_raw(&frame));
            }
            match device.channels.map(&frame, cfg.gripper_open_fraction) {
                Ok(mapped) => {
                    mapping_warned = false;
                    let received_at = Instant::now();
                    let engaged = engage.update(mapped.triggers, received_at);
                    if tx
                        .send(Some(KerSample::from_mapped(mapped, engaged, received_at)))
                        .is_err()
                    {
                        return (SessionEnd::Stop, streamed);
                    }
                    streamed = true;
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
            return (
                SessionEnd::Transient(format!(
                    "no valid frames for {SILENCE_RECONNECT:?} while connected"
                )),
                streamed,
            );
        }
    }
    (SessionEnd::Stop, streamed)
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
    /// How long a session test waits for what it expects, and then for the
    /// session thread to notice cancellation.
    const SESSION_DEADLINE: Duration = Duration::from_secs(5);

    /// A frame with both triggers released, every other channel numbered.
    fn released_frame() -> Vec<f32> {
        let mut angles = numbered_frame();
        angles[RIGHT_TRIGGER] = 0.0;
        angles[LEFT_TRIGGER] = 0.0;
        angles
    }

    /// A frame with both triggers at their stops.
    fn squeezed_frame() -> Vec<f32> {
        let mut angles = released_frame();
        angles[RIGHT_TRIGGER] = RIGHT_SQUEEZE_DEG as f32;
        angles[LEFT_TRIGGER] = LEFT_SQUEEZE_DEG as f32;
        angles
    }

    /// A KER that answers PING with the reference schema and sends frames only
    /// once STREAM arrives, as firmware 2.0.0 does.
    struct FakeKer {
        hardware: &'static str,
        channels: usize,
        /// One entry per frame; the last repeats once the script runs out.
        script: Vec<Vec<f32>>,
        frames_sent: usize,
        /// Bytes delivered per read, so a caller can force partial reads.
        chunk_size: usize,
        /// Reads to fail after, as an unplugged cable does.
        fail_read_after: Option<usize>,
        reads: usize,
        writes: Arc<Mutex<Vec<u8>>>,
        pending: Vec<u8>,
        streaming: bool,
    }

    impl FakeKer {
        fn new(script: Vec<Vec<f32>>) -> Self {
            Self {
                hardware: "2.0.0",
                channels: CHANNELS,
                script,
                frames_sent: 0,
                chunk_size: usize::MAX,
                fail_read_after: None,
                reads: 0,
                writes: Arc::new(Mutex::new(Vec::new())),
                pending: Vec::new(),
                streaming: false,
            }
        }

        /// A device already streaming when this node connects, as one left
        /// running by a previous session is. It honours STANDBY only after
        /// `standby_after` more frames, so its bytes land in the handshake.
        fn already_streaming(mut self, standby_after: usize) -> Self {
            self.streaming = true;
            self.fill(standby_after);
            self
        }

        fn fill(&mut self, frames: usize) {
            for _ in 0..frames {
                let angles = self.next_angles();
                self.pending
                    .extend(stream_packet(1, &angles[..self.channels], 0, false));
            }
        }

        fn next_angles(&mut self) -> Vec<f32> {
            let index = self.frames_sent.min(self.script.len() - 1);
            self.frames_sent += 1;
            self.script[index].clone()
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
            self.reads += 1;
            if self.fail_read_after.is_some_and(|after| self.reads > after) {
                return Err(std::io::Error::other("the cable came out"));
            }
            if self.pending.is_empty() && self.streaming {
                self.fill(1);
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
            engage_trigger_opening: EngageTriggerOpening::try_from(ENGAGE_AT).expect("in range"),
            gripper_open_fraction: GripperOpenFraction::try_from(OPEN_FRACTION).expect("in range"),
            stale_timeout: STALE,
            log_raw: false,
        }
    }

    struct SessionRun {
        sample: Option<KerSample>,
        report: SessionReport,
        writes: Vec<u8>,
    }

    /// Run one session until `wanted` holds of a delivered sample, or the
    /// session stops, then cancel it. A session that ignores cancellation
    /// fails the test instead of wedging the suite.
    fn run_until(mut ker: FakeKer, wanted: impl Fn(&KerSample) -> bool) -> SessionRun {
        let writes = ker.writes.clone();
        let cfg = reader_config();
        let (tx, rx) = watch::channel(None);
        let token = CancellationToken::new();
        let session = {
            let token = token.clone();
            std::thread::spawn(move || run_session(&mut ker, &cfg, &tx, &token))
        };

        let deadline = Instant::now() + SESSION_DEADLINE;
        let mut sample = None;
        while Instant::now() < deadline {
            sample = rx.borrow().clone();
            if sample.as_ref().is_some_and(&wanted) || session.is_finished() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        token.cancel();

        let stop_by = Instant::now() + SESSION_DEADLINE;
        while !session.is_finished() && Instant::now() < stop_by {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(
            session.is_finished(),
            "the session ignored cancellation and would hang the suite"
        );
        SessionRun {
            sample,
            report: session.join().expect("session thread"),
            writes: writes.lock().expect("writes").clone(),
        }
    }

    /// Run one session until it delivers any sample.
    fn run_one(ker: FakeKer) -> SessionRun {
        run_until(ker, |_| true)
    }

    #[test]
    fn a_session_starts_the_stream_and_maps_the_first_frame() {
        let run = run_one(FakeKer::new(vec![released_frame()]));

        assert!(
            matches!(run.report.end, SessionEnd::Stop),
            "{:?}",
            run.report.end
        );
        assert!(run.report.streamed, "a mapped frame counts as streaming");
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
        let frame = released_frame();
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
    fn each_gripper_follows_its_own_trigger() {
        // Left half squeezed, right released: the two openings must differ, so
        // a side swap anywhere from the channel to the sample fails here.
        let mut angles = released_frame();
        angles[LEFT_TRIGGER] = LEFT_SQUEEZE_DEG as f32 / 2.0;
        let run = run_one(FakeKer::new(vec![angles]));

        let sample = run.sample.expect("a frame arrived");
        assert!(
            (sample.left_gripper_opening - OPEN_FRACTION * 0.5).abs() < 1e-12,
            "left: {}",
            sample.left_gripper_opening
        );
        assert_eq!(sample.right_gripper_opening, OPEN_FRACTION);
    }

    #[test]
    fn a_release_then_a_squeeze_engages_through_the_session() {
        // Enough released frames to arm, then a squeeze on both triggers.
        let mut script = vec![released_frame(); 8];
        script.push(squeezed_frame());
        let run = run_until(FakeKer::new(script), |sample| {
            sample.engaged != SideFlags::NONE
        });

        let sample = run.sample.expect("a frame arrived");
        assert_eq!(
            sample.engaged,
            SideFlags {
                left: true,
                right: true
            },
            "a squeeze after a run of open frames engages both arms"
        );
        assert_eq!(sample.left_gripper_opening, 0.0, "a full squeeze closes");
    }

    #[test]
    fn a_session_delivers_a_frame_split_across_reads() {
        let mut ker = FakeKer::new(vec![released_frame()]);
        ker.chunk_size = 7;
        let run = run_one(ker);

        assert!(
            matches!(run.report.end, SessionEnd::Stop),
            "{:?}",
            run.report.end
        );
        let sample = run
            .sample
            .expect("a split packet still reaches the channel");
        assert_eq!(sample.left_gripper_opening, OPEN_FRACTION);
    }

    #[test]
    fn a_device_still_streaming_is_handshaken_through_its_own_frames() {
        // The case a previous session left behind: frames arrive before and
        // during the handshake, so the ping response lands behind them.
        let run = run_one(FakeKer::new(vec![released_frame()]).already_streaming(40));

        assert!(
            matches!(run.report.end, SessionEnd::Stop),
            "{:?}",
            run.report.end
        );
        assert!(
            run.sample.is_some(),
            "the response must be found among the device's own bytes"
        );
    }

    #[test]
    fn a_read_error_ends_the_session_for_a_retry() {
        let mut ker = FakeKer::new(vec![released_frame()]);
        ker.fail_read_after = Some(3);
        let run = run_one(ker);

        let SessionEnd::Transient(reason) = run.report.end else {
            panic!(
                "an unplugged cable must be retried, got {:?}",
                run.report.end
            );
        };
        assert!(reason.contains("read"), "{reason}");
    }

    #[test]
    fn a_frame_this_node_cannot_map_does_not_end_the_session() {
        // One encoder reading arrives non-finite, then the device recovers.
        let mut poisoned = released_frame();
        poisoned[3] = f32::NAN;
        let script = vec![poisoned, released_frame()];
        let run = run_one(FakeKer::new(script));

        assert!(
            matches!(run.report.end, SessionEnd::Stop),
            "{:?}",
            run.report.end
        );
        assert!(
            run.sample.is_some(),
            "the frame after an unmappable one still streams"
        );
    }

    #[test]
    fn a_session_refuses_a_device_of_another_hardware_generation() {
        let mut ker = FakeKer::new(vec![released_frame()]);
        ker.hardware = "3.0.0";
        let run = run_one(ker);

        let SessionEnd::Fatal(reason) = run.report.end else {
            panic!("expected a fatal refusal, got {:?}", run.report.end);
        };
        assert!(reason.contains("3.0.0"), "{reason}");
        assert!(run.sample.is_none(), "a refused device streams nothing");
        assert!(!run.report.streamed, "a refused device never streamed");
        assert!(
            !run.writes.contains(&CMD_STREAM),
            "a refused device is never started"
        );
    }

    #[test]
    fn a_session_refuses_a_device_streaming_too_few_channels() {
        let mut ker = FakeKer::new(vec![released_frame()]);
        ker.channels = CHANNELS - 1;
        let run = run_one(ker);

        let SessionEnd::Fatal(reason) = run.report.end else {
            panic!("expected a fatal refusal, got {:?}", run.report.end);
        };
        assert!(reason.contains("15 channels"), "{reason}");
        assert!(
            !run.writes.contains(&CMD_STREAM),
            "a refused device is never started"
        );
    }
}
