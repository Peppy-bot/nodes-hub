use peppygen::emitted_topics::rgb_camera::v1::video_stream::{self, MessageHeader};
use peppylib::runtime::CancellationToken;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::oneshot;

use super::device::CameraDevice;
use crate::camera::controls::ControlReceiver;
use crate::pipeline;
use crate::types::{CameraConfig, FrameId, Result};

/// Camera capture loop configuration
const FRAME_RETRY_DELAY_MS: u64 = 10;
const STATUS_PRINT_INTERVAL_SECS: u64 = 3;
/// Granularity at which rate-limiting sleeps re-check the cancellation token
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(50);

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
/// thread (V4L2 mmap streams are thread-local).
///
/// The returned receiver resolves once the thread has exited and the camera
/// has been dropped (device closed). Await it from an `on_shutdown` hook so
/// device teardown is bounded by the shutdown grace window.
///
/// If the loop fails (camera cannot be opened, configured, or captured from),
/// the error is logged and `cancel_token` is cancelled so the node shuts down
/// instead of lingering without a capture loop.
fn spawn_camera_capture_loop<C: CameraDevice + 'static>(
    camera: C,
    config: CameraConfig,
    node_runner: Arc<peppygen::NodeRunner>,
    cancel_token: CancellationToken,
    control_rx: ControlReceiver,
) -> oneshot::Receiver<()> {
    let (done_tx, done_rx) = oneshot::channel();
    let runtime = tokio::runtime::Handle::current();

    std::thread::spawn(move || {
        // `camera` is moved into and dropped inside `run_camera_capture_loop`,
        // so by the time the result is back the device is closed.
        let result = run_camera_capture_loop(
            camera,
            &config,
            &node_runner,
            &runtime,
            &cancel_token,
            &control_rx,
        );

        if let Err(e) = result {
            tracing::error!("[uvc_camera] Camera capture loop failed: {e}");
            // Request node shutdown: without a capture loop the node serves
            // no purpose and should not linger as a zombie.
            cancel_token.cancel();
        }

        // Signal completion only after the camera has been dropped above, so
        // the shutdown hook awaiting this knows the device is closed.
        let _ = done_tx.send(());
    });

    done_rx
}

/// Run the camera capture loop (blocking; runs on the dedicated thread).
///
/// # Errors
///
/// Returns an error if the camera cannot be opened or configured.
fn run_camera_capture_loop<C: CameraDevice>(
    mut camera: C,
    config: &CameraConfig,
    node_runner: &Arc<peppygen::NodeRunner>,
    runtime: &tokio::runtime::Handle,
    cancel_token: &CancellationToken,
    control_rx: &ControlReceiver,
) -> Result<()> {
    println!("[uvc_camera] Starting camera capture loop...");

    // Open and configure camera (blocking operation, done before the loop)
    println!("[uvc_camera] Opening camera {}...", config.device_path);

    let resolution = config.resolution;
    let camera_encoding = config.camera_encoding;
    let topic_encoding = config.topic_encoding;
    let frame_rate = config.frame_rate.as_u16();

    camera.open(config)?;
    println!(
        "[uvc_camera] Camera configured: {}x{} @ {} fps, camera_encoding: {}, topic_encoding: {}",
        resolution.width(),
        resolution.height(),
        frame_rate,
        camera_encoding,
        topic_encoding
    );

    let mut frame_id = FrameId::default();
    let mut last_print_time = Instant::now();

    // Calculate target frame duration using nanoseconds for high FPS support
    let frame_duration_ns = 1_000_000_000u64 / u64::from(frame_rate);
    let target_frame_duration = Duration::from_nanos(frame_duration_ns);
    let mut next_frame_time = Instant::now() + target_frame_duration;

    // Declare the publisher once; every publish below is then lock-free.
    let publisher = runtime
        .block_on(video_stream::declare_publisher(node_runner))
        .map_err(|e| format!("Failed to declare video stream publisher: {e}"))?;

    loop {
        if cancel_token.is_cancelled() {
            println!("[uvc_camera] Shutdown requested, stopping camera capture loop");
            break;
        }

        // Drain all pending camera control commands before capturing the next frame
        while let Ok(cmd) = control_rx.try_recv() {
            let result = camera.apply_control(&cmd.request);
            // If the receiver has gone away (service handler timed out), ignore the error
            let _ = cmd.reply.send(result);
        }

        // Capture frame from camera
        let raw_frame = match camera.capture_frame() {
            Ok(frame) => frame,
            Err(e) => {
                tracing::warn!("Failed to capture frame: {}", e);
                std::thread::sleep(Duration::from_millis(FRAME_RETRY_DELAY_MS));
                continue;
            }
        };

        // Process frame (convert encoding if needed)
        let frame = match pipeline::process_frame(raw_frame, frame_id, topic_encoding) {
            Ok(frame) => frame,
            Err(e) => {
                tracing::warn!("Failed to process frame: {}", e);
                std::thread::sleep(Duration::from_millis(FRAME_RETRY_DELAY_MS));
                continue;
            }
        };

        let header = MessageHeader {
            stamp: SystemTime::now(),
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
                tracing::warn!("Failed to build frame message: {}", e);
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
                        tracing::warn!("Failed to emit frame: {}", e);
                    }
                }
            }
        });

        if last_print_time.elapsed().as_secs() >= STATUS_PRINT_INTERVAL_SECS {
            println!("[uvc_camera] Emitted frame {}", frame.frame_id().as_u32());
            last_print_time = Instant::now();
        }

        frame_id = frame_id.next();

        // Rate limiting using accumulator to prevent drift
        sleep_until_unless_cancelled(next_frame_time, cancel_token);
        next_frame_time += target_frame_duration;
    }

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

/// Helper function to spawn the capture loop with a Nokhwa camera.
///
/// The returned receiver resolves once the capture thread has exited and the
/// camera has been dropped; await it from an `on_shutdown` hook so device
/// teardown is bounded by the shutdown grace window.
pub fn spawn_nokhwa_capture_loop(
    config: CameraConfig,
    node_runner: Arc<peppygen::NodeRunner>,
    cancel_token: CancellationToken,
    control_rx: ControlReceiver,
) -> oneshot::Receiver<()> {
    let camera = super::NokhwaCamera::new();
    spawn_camera_capture_loop(camera, config, node_runner, cancel_token, control_rx)
}
