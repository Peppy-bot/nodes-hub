use peppygen::emitted_topics::camera::video_stream::{self, MessageHeader};
use peppylib::runtime::CancellationToken;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

use crate::camera::controls::ControlReceiver;
use crate::camera::nokhwa_impl::NokhwaCamera;
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

/// Spawn the camera capture loop on a dedicated OS thread.
///
/// The loop opens the camera, configures it, captures frames, processes them,
/// and emits them to the video stream topic. Between frames, any pending
/// camera control commands from the `control_rx` channel are drained and
/// applied immediately.
///
/// A dedicated `std::thread` is used instead of `spawn_blocking` on purpose:
/// `Runtime::drop` blocks until every blocking-pool task returns, so a V4L2
/// call wedged in the driver (e.g. an untimed frame dequeue after the camera
/// is unplugged) would hang shutdown past the grace window. A plain thread
/// cannot outlive process exit. It also keeps the camera on a single OS
/// thread: `nokhwa::Camera` is `!Send`, and it is built here rather than passed
/// in so it never crosses a thread boundary at all.
///
/// Returns two receivers:
/// - the open outcome, so setup can fail the node when the camera is
///   unreachable instead of leaving it running with nothing to publish;
/// - a completion signal that resolves once the thread has exited and the
///   camera has been dropped (device closed). Await it from an `on_shutdown`
///   hook so device teardown is bounded by the shutdown grace window.
pub fn spawn_nokhwa_capture_loop(
    config: CameraConfig,
    node_runner: Arc<peppygen::NodeRunner>,
    cancel_token: CancellationToken,
    control_rx: ControlReceiver,
) -> (oneshot::Receiver<Result<()>>, oneshot::Receiver<()>) {
    let (open_tx, open_rx) = oneshot::channel();
    let (done_tx, done_rx) = oneshot::channel();
    let runtime = tokio::runtime::Handle::current();

    std::thread::spawn(move || {
        let _cancel_on_exit = CancelOnExit(cancel_token.clone());

        // The camera is moved into and dropped inside `run_camera_capture_loop`,
        // so by the time the result is back the device is closed.
        let result = run_camera_capture_loop(
            NokhwaCamera::new(),
            &config,
            &node_runner,
            &runtime,
            &cancel_token,
            &control_rx,
            open_tx,
        );

        if let Err(e) = result {
            tracing::error!("camera capture loop failed: {e}");
        }

        // Signal completion only after the camera has been dropped above, so
        // the shutdown hook awaiting this knows the device is closed.
        let _ = done_tx.send(());
    });

    (open_rx, done_rx)
}

/// Run the camera capture loop (blocking; runs on the dedicated thread).
///
/// # Errors
///
/// Returns an error if the camera cannot be opened or configured, or if frames
/// keep failing for longer than [`FRAME_FAILURE_GRACE`].
fn run_camera_capture_loop(
    mut camera: NokhwaCamera,
    config: &CameraConfig,
    node_runner: &Arc<peppygen::NodeRunner>,
    runtime: &tokio::runtime::Handle,
    cancel_token: &CancellationToken,
    control_rx: &ControlReceiver,
    open_tx: oneshot::Sender<Result<()>>,
) -> Result<()> {
    // Report the open outcome before streaming: setup turns a failure into a
    // non-zero exit, which is what distinguishes a broken camera from an
    // operator stop.
    if let Err(e) = camera.open(config) {
        let _ = open_tx.send(Err(e.clone()));
        return Err(e);
    }
    let _ = open_tx.send(Ok(()));

    let topic_encoding = config.topic_encoding;
    let frame_rate = config.frame_rate.as_u16();
    tracing::info!(
        "capture loop started: {}x{} @ {frame_rate} fps, topic_encoding {topic_encoding}",
        config.resolution.width(),
        config.resolution.height(),
    );

    let mut frame_id = FrameId::default();
    let mut last_log_time = Instant::now();
    let mut last_success = Instant::now();

    // Calculate target frame duration using nanoseconds for high FPS support
    let target_frame_duration = Duration::from_nanos(1_000_000_000 / u64::from(frame_rate));
    let mut next_frame_time = Instant::now() + target_frame_duration;

    // Declare the publisher once; every publish below is then lock-free.
    let publisher = runtime
        .block_on(video_stream::declare_publisher(node_runner))
        .map_err(|e| format!("Failed to declare video stream publisher: {e}"))?;

    while !cancel_token.is_cancelled() {
        // Drain all pending camera control commands before capturing the next frame
        while let Ok(cmd) = control_rx.try_recv() {
            let result = camera.apply_control(&cmd.request);
            // If the receiver has gone away (service handler timed out), ignore the error
            let _ = cmd.reply.send(result);
        }

        let processed = camera
            .capture_frame()
            .and_then(|raw| pipeline::process_frame(raw, frame_id, topic_encoding));

        let frame = match processed {
            Ok(frame) => frame,
            Err(e) => {
                if last_success.elapsed() > FRAME_FAILURE_GRACE {
                    return Err(Error::Camera(format!(
                        "no frame published for {FRAME_FAILURE_GRACE:?}; last failure: {e}"
                    )));
                }
                tracing::warn!("frame failed: {e}");
                std::thread::sleep(FRAME_RETRY_DELAY);
                continue;
            }
        };

        let header = MessageHeader {
            stamp: frame.captured_at(),
            frame_id: frame.frame_id().as_u32(),
        };

        // Serialize off the messenger (build_message is pure), then publish.
        let payload = match video_stream::build_message(
            header,
            frame.encoding().to_string(),
            frame.width(),
            frame.height(),
            frame.data().to_vec(),
        ) {
            Ok(payload) => payload,
            Err(e) => {
                tracing::warn!("failed to build frame message: {e}");
                continue;
            }
        };

        // Publish by blocking this dedicated thread on the async call. Racing
        // the publish against the token keeps shutdown from stalling on
        // messaging once cancellation has been requested.
        runtime.block_on(async {
            tokio::select! {
                _ = cancel_token.cancelled() => {}
                result = publisher.publish(payload) => {
                    if let Err(e) = result {
                        tracing::warn!("failed to emit frame: {e}");
                    }
                }
            }
        });

        last_success = Instant::now();
        if last_log_time.elapsed() >= STATUS_LOG_INTERVAL {
            tracing::info!("emitted frame {}", frame.frame_id().as_u32());
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
