use peppygen::emitted_topics::video_stream::{self, MessageHeader};
use peppylib::runtime::CancellationToken;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use super::device::CameraDevice;
use crate::camera::controls::ControlReceiver;
use crate::pipeline;
use crate::types::{CameraConfig, Error, FrameId, Result};

/// Camera capture loop configuration
const FRAME_RETRY_DELAY_MS: u64 = 10;
const STATUS_PRINT_INTERVAL_SECS: u64 = 3;

/// Run the camera capture loop with trait-based abstraction
///
/// This is the main entry point for camera capture. It opens the camera,
/// configures it, and enters a loop that captures frames, processes them,
/// and emits them to the video stream topic.
///
/// Between frames, any pending camera control commands from the `control_rx`
/// channel are drained and applied immediately.
///
/// # Errors
///
/// Returns an error if:
/// - Camera cannot be opened or configured
/// - Thread panics during execution
async fn run_camera_capture_loop<C: CameraDevice + 'static>(
    mut camera: C,
    config: CameraConfig,
    node_runner: Arc<peppygen::NodeRunner>,
    cancel_token: CancellationToken,
    control_rx: ControlReceiver,
) -> Result<()> {
    println!("[uvc_camera] Starting camera capture loop...");

    // Open and configure camera (blocking operation, done before the loop)
    println!("[uvc_camera] Opening camera {}...", config.device_path);

    let resolution = config.resolution;
    let camera_encoding = config.camera_encoding;
    let topic_encoding = config.topic_encoding;
    let frame_rate = config.frame_rate.as_u16();

    // Run the open + capture loop in a single blocking task so the camera is
    // never moved between OS threads (V4L2 mmap streams are thread-local).
    tokio::task::spawn_blocking(move || {
        camera.open(&config)?;
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

            // Emit frame using block_in_place to call async code from blocking context
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    if let Err(e) = video_stream::emit(
                        &node_runner,
                        header,
                        frame.encoding().to_string(),
                        frame.width(),
                        frame.height(),
                        frame.data().to_vec(),
                    )
                    .await {
                        tracing::warn!("Failed to emit frame: {}", e);
                    }
                });
            });

            if last_print_time.elapsed().as_secs() >= STATUS_PRINT_INTERVAL_SECS {
                println!("[uvc_camera] Emitted frame {}", frame.frame_id().as_u32());
                last_print_time = Instant::now();
            }

            frame_id = frame_id.next();

            // Rate limiting using accumulator to prevent drift
            let now = Instant::now();
            if next_frame_time > now {
                std::thread::sleep(next_frame_time - now);
            }
            next_frame_time += target_frame_duration;
        }

        Ok::<(), Error>(())
    })
    .await
    .map_err(|join_err| Error::ThreadPanic(format!("Camera capture task panicked: {}", join_err)))??;

    Ok(())
}

/// Helper function to create and run the capture loop with Nokhwa camera
pub async fn run_nokhwa_capture_loop(
    config: CameraConfig,
    node_runner: Arc<peppygen::NodeRunner>,
    cancel_token: CancellationToken,
    control_rx: ControlReceiver,
) -> Result<()> {
    let camera = super::NokhwaCamera::new();
    run_camera_capture_loop(camera, config, node_runner, cancel_token, control_rx).await
}
