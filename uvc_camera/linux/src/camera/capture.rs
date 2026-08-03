use peppygen::emitted_topics::camera::video_stream::{self, MessageHeader};
use peppylib::runtime::CancellationToken;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

use crate::camera::v4l_device::{CameraDevice, ControlHandle, StreamDescription};
use crate::pipeline;
use crate::types::{CameraConfig, Error, FrameId, Result};

/// Pause after a failed frame before trying the next one.
const FRAME_RETRY_DELAY: Duration = Duration::from_millis(10);
const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(3);
/// Granularity at which rate-limiting sleeps re-check the cancellation token
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(50);
/// How long the loop tolerates back-to-back frame failures before giving up.
/// Driver hiccups clear in milliseconds, but a deterministic failure (an
/// unplugged camera, a stream the pipeline cannot convert) never clears, and
/// retrying it forever publishes nothing while the node still answers services
/// as though it were healthy.
const FRAME_FAILURE_GRACE: Duration = Duration::from_secs(5);

/// Everything setup needs from a successfully opened camera: what the stream
/// actually is, and the handle services use to drive the hardware.
pub struct CameraReadout {
    pub description: StreamDescription,
    pub controls: ControlHandle,
}

/// Cancels the node's token when the capture thread exits by any path,
/// including a panic. Without a capture loop the node publishes nothing while
/// its service handlers keep answering, so it should not linger. Cancelling is
/// idempotent, which makes the clean shutdown path a no-op.
struct CancelOnExit(CancellationToken);

impl Drop for CancelOnExit {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// How long the loop has gone without publishing a frame.
///
/// Every step that can fail a frame (capture, conversion, serialization,
/// publication) reports here, so one deterministic failure cannot hide behind
/// another's retry path.
struct PublishWindow {
    last_publish: Instant,
    episode_logged: bool,
}

impl PublishWindow {
    fn new() -> Self {
        Self {
            last_publish: Instant::now(),
            episode_logged: false,
        }
    }

    fn published(&mut self) {
        self.last_publish = Instant::now();
        self.episode_logged = false;
    }

    /// Record a failed frame and pause before the next attempt.
    ///
    /// # Errors
    ///
    /// Returns an error once the grace window has elapsed with nothing
    /// published, which ends the loop rather than retrying forever.
    fn failed(&mut self, context: &str, error: &dyn fmt::Display) -> Result<()> {
        let stalled = self.last_publish.elapsed();
        if stalled > FRAME_FAILURE_GRACE {
            return Err(Error::Camera(format!(
                "nothing published for {stalled:?}; last failure: {context}: {error}"
            )));
        }
        // Only the first failure of an episode warns; a wedged camera would
        // otherwise emit hundreds of identical lines before the window closes.
        if self.episode_logged {
            tracing::debug!("{context}: {error}");
        } else {
            tracing::warn!("{context}: {error}");
            self.episode_logged = true;
        }
        std::thread::sleep(FRAME_RETRY_DELAY);
        Ok(())
    }
}

/// Spawn the camera capture loop on a dedicated OS thread.
///
/// The loop opens the camera, configures it, captures frames, processes them,
/// and emits them to the video stream topic. Camera controls are not routed
/// through here: service handlers apply them synchronously via the
/// [`ControlHandle`] returned from the open handshake.
///
/// A dedicated `std::thread` is used instead of `spawn_blocking` on purpose:
/// `Runtime::drop` blocks until every blocking-pool task returns, so a V4L2
/// call wedged in the driver (e.g. an untimed frame dequeue after the camera
/// is unplugged) would hang shutdown past the grace window. A plain thread
/// cannot outlive process exit. It also keeps the mmap stream on a single OS
/// thread: the stream is bound to the thread that dequeues it, and the camera
/// is built here rather than passed in so the stream never crosses a thread
/// boundary. The device fd itself is shareable, which is what the
/// [`ControlHandle`] hands to services.
///
/// Returns two receivers:
/// - the open outcome: on success, the negotiated [`StreamDescription`] plus a
///   [`ControlHandle`], so services report and act on the camera as it actually
///   is; on failure, the error that setup turns into a non-zero exit;
/// - a completion signal that resolves once the thread has exited and the
///   camera has been dropped (device closed). Await it from an `on_shutdown`
///   hook so device teardown is bounded by the shutdown grace window.
pub fn spawn_capture_loop(
    config: CameraConfig,
    node_runner: Arc<peppygen::NodeRunner>,
    cancel_token: CancellationToken,
) -> (
    oneshot::Receiver<Result<CameraReadout>>,
    oneshot::Receiver<()>,
) {
    let (open_tx, open_rx) = oneshot::channel();
    let (done_tx, done_rx) = oneshot::channel();
    let runtime = tokio::runtime::Handle::current();

    std::thread::spawn(move || {
        // The camera is dropped at the end of this closure either way, so by the
        // time `done_tx` fires the device is closed.
        if let Some(camera) = open_camera(CameraDevice::new(), &config, open_tx) {
            // Armed only once the camera is streaming. Arming it any earlier
            // would race the open report: setup selects over its own future and
            // the cancellation token, so a cancel landing alongside the failure
            // would be read as an operator stop and exit zero, which is exactly
            // what reporting the failure is meant to prevent.
            let _cancel_on_exit = CancelOnExit(cancel_token.clone());

            if let Err(e) =
                run_camera_capture_loop(camera, &config, &node_runner, &runtime, &cancel_token)
            {
                tracing::error!("camera capture loop failed: {e}");
            }
        }

        // Signal completion only after the camera has been dropped above, so
        // the shutdown hook awaiting this knows the device is closed.
        let _ = done_tx.send(());
    });

    (open_rx, done_rx)
}

/// Open the camera and hand the outcome to setup.
///
/// Returns `None` once a failure has been reported, so the caller neither logs
/// it a second time nor starts a loop with no device. Setup turns the reported
/// failure into the node's exit error, which is what distinguishes a broken
/// camera from an operator stop.
fn open_camera(
    mut camera: CameraDevice,
    config: &CameraConfig,
    open_tx: oneshot::Sender<Result<CameraReadout>>,
) -> Option<CameraDevice> {
    match camera.open(config) {
        Ok(()) => {
            let readout = camera
                .stream_description()
                .zip(camera.control_handle())
                .map(|(description, controls)| CameraReadout {
                    description,
                    controls,
                })
                .ok_or_else(|| {
                    Error::Camera("camera opened without a negotiated stream".to_string())
                });
            let failed = readout.is_err();
            let _ = open_tx.send(readout);
            (!failed).then_some(camera)
        }
        Err(e) => {
            let _ = open_tx.send(Err(e));
            None
        }
    }
}

/// Run the camera capture loop (blocking; runs on the dedicated thread).
///
/// # Errors
///
/// Returns an error if the publisher cannot be declared, or if frames keep
/// failing for longer than [`FRAME_FAILURE_GRACE`].
fn run_camera_capture_loop(
    mut camera: CameraDevice,
    config: &CameraConfig,
    node_runner: &Arc<peppygen::NodeRunner>,
    runtime: &tokio::runtime::Handle,
    cancel_token: &CancellationToken,
) -> Result<()> {
    let topic_encoding = config.topic_encoding;
    let frame_rate = config.frame_rate.as_u16();
    tracing::info!(
        "capture loop started: {}x{} @ {frame_rate} fps, topic_encoding {topic_encoding}",
        config.resolution.width(),
        config.resolution.height(),
    );

    let mut frame_id = FrameId::default();
    let mut last_log_time = Instant::now();

    // Declare the publisher once; every publish below is then lock-free.
    let publisher = runtime
        .block_on(video_stream::declare_publisher(node_runner))
        .map_err(|e| format!("Failed to declare video stream publisher: {e}"))?;

    // Both clocks start after the publisher is up, so a slow cold-start
    // declaration does not count as a stall or leave the first deadline stale.
    let mut window = PublishWindow::new();
    let target_frame_duration = Duration::from_secs(1) / u32::from(frame_rate);
    let mut next_frame_time = Instant::now() + target_frame_duration;

    while !cancel_token.is_cancelled() {
        // Capture, convert and serialize as one fallible step: every failure
        // below reaches the same grace window, so none of them can spin.
        // `build_message` is pure, so this keeps serialization off the messenger.
        let prepared = camera
            .capture_frame()
            .and_then(|raw| pipeline::process_frame(raw, frame_id, topic_encoding))
            .and_then(|frame| {
                let header = MessageHeader {
                    stamp: frame.captured_at(),
                    frame_id: frame.frame_id().as_u32(),
                };
                video_stream::build_message(
                    header,
                    frame.encoding().to_string(),
                    frame.width(),
                    frame.height(),
                    frame.data().to_vec(),
                )
                .map_err(|e| Error::Other(format!("failed to build frame message: {e}")))
            });

        let payload = match prepared {
            Ok(payload) => payload,
            Err(e) => {
                window.failed("frame failed", &e)?;
                continue;
            }
        };

        // Publish by blocking this dedicated thread on the async call. Racing
        // the publish against the token keeps shutdown from stalling on
        // messaging once cancellation has been requested.
        let publish_outcome = runtime.block_on(async {
            tokio::select! {
                _ = cancel_token.cancelled() => None,
                result = publisher.publish(payload) => Some(result),
            }
        });

        match publish_outcome {
            // Cancelled mid-publish: leave the loop rather than counting a
            // shutdown as a failure.
            None => break,
            Some(Err(e)) => {
                window.failed("failed to emit frame", &e)?;
                continue;
            }
            Some(Ok(())) => window.published(),
        }

        if last_log_time.elapsed() >= STATUS_LOG_INTERVAL {
            tracing::info!("emitted frame {}", frame_id.as_u32());
            last_log_time = Instant::now();
        }

        frame_id = frame_id.next();

        sleep_until_unless_cancelled(next_frame_time, cancel_token);
        next_frame_time =
            advance_frame_deadline(next_frame_time, target_frame_duration, Instant::now());
    }

    tracing::info!("shutdown requested, stopping camera capture loop");
    Ok(())
}

/// Sleep until `deadline` in short slices, returning early once the token is
/// cancelled, so the rate limiter (up to a full frame interval at low fps)
/// does not eat into the shutdown grace window.
fn sleep_until_unless_cancelled(deadline: Instant, cancel_token: &CancellationToken) {
    while !cancel_token.is_cancelled() {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        std::thread::sleep((deadline - now).min(CANCEL_POLL_INTERVAL));
    }
}

/// Advance the frame deadline by one period, accumulating to avoid drift.
///
/// Clamping to `now` is what keeps a stall bounded: without it a loop that
/// fell behind banks the whole deficit and repays it as a burst at the camera's
/// full rate, ignoring the configured frame rate for as long as it takes.
fn advance_frame_deadline(deadline: Instant, period: Duration, now: Instant) -> Instant {
    (deadline + period).max(now)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadline_accumulates_when_keeping_up() {
        let period = Duration::from_millis(100);
        let deadline = Instant::now();
        // Woken on time: the next deadline is exactly one period on, so
        // scheduling jitter cannot drift the stream slow.
        let now = deadline + Duration::from_millis(1);
        assert_eq!(
            advance_frame_deadline(deadline, period, now),
            deadline + period
        );
    }

    #[test]
    fn deadline_clamps_after_a_stall() {
        let period = Duration::from_millis(100);
        let deadline = Instant::now();
        // A 60s stall would otherwise leave the deadline 60s in the past and
        // license ~600 unthrottled frames; clamping caps the catch-up at one.
        let now = deadline + Duration::from_secs(60);
        assert_eq!(advance_frame_deadline(deadline, period, now), now);
    }

    #[test]
    fn cancel_on_exit_cancels_even_when_unwinding() {
        let token = CancellationToken::new();
        let guard_token = token.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = CancelOnExit(guard_token);
            panic!("capture thread blew up");
        }));
        assert!(result.is_err());
        assert!(
            token.is_cancelled(),
            "a panicking capture thread must still shut the node down"
        );
    }
}
