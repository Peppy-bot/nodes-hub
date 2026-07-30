use peppygen::{NodeBuilder, Parameters, Result, StandaloneConfig};
use std::sync::Arc;

use uvc_camera_linux::camera::{create_control_channel, spawn_nokhwa_capture_loop};
use uvc_camera_linux::services::{
    listen_for_set_brightness_requests, listen_for_set_contrast_requests,
    listen_for_set_exposure_requests, listen_for_set_gain_requests,
    listen_for_set_white_balance_requests, listen_for_video_stream_info_requests,
};
use uvc_camera_linux::types::{CameraConfigBuilder, Encoding};

fn main() -> Result<()> {
    // INFO by default, but honour RUST_LOG so the device-open path's debug
    // lines (resolved index, validation, requested format) can be turned on
    // when a camera refuses to open.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Load parameters from mock file for standalone execution
    let mock_params: Parameters = serde_json::from_str(
        &std::fs::read_to_string("mock_parameters.json")
            .expect("Failed to read mock_parameters.json"),
    )
    .expect("Failed to parse mock_parameters.json");
    let standalone_config = StandaloneConfig::new().with_parameters(&mock_params);

    NodeBuilder::new().standalone(standalone_config).run(
        move |args: Parameters, node_runner| async move {
            let video_params = args.video.clone();
            let device_path = args.device_path.clone();

            tracing::info!(
                "device {device_path}: {}x{} @ {} fps, camera_encoding {}, topic_encoding {}",
                video_params.resolution.width,
                video_params.resolution.height,
                video_params.frame_rate,
                video_params.camera_encoding,
                video_params.topic_encoding
            );

            // Parse encodings up front: an unusable value is a misconfiguration
            // that should fail the node, not surface later as a frame error.
            let camera_encoding =
                video_params
                    .camera_encoding
                    .parse::<Encoding>()
                    .map_err(|e| {
                        std::io::Error::other(format!(
                            "Invalid camera_encoding '{}': {e}",
                            video_params.camera_encoding
                        ))
                    })?;
            let topic_encoding = video_params
                .topic_encoding
                .parse::<Encoding>()
                .map_err(|e| {
                    std::io::Error::other(format!(
                        "Invalid topic_encoding '{}': {e}",
                        video_params.topic_encoding
                    ))
                })?;

            // Create camera configuration
            let camera_config = CameraConfigBuilder::new()
                .device_path(device_path.clone())
                .resolution(
                    video_params.resolution.width,
                    video_params.resolution.height,
                )
                .frame_rate(video_params.frame_rate)
                .camera_encoding(camera_encoding)
                .topic_encoding(topic_encoding)
                .build()
                .map_err(std::io::Error::other)?;

            // Create control channel shared between service handlers and capture loop
            let (control_tx, control_rx) = create_control_channel();

            // ── capture loop (long-running, dedicated thread) ──────────────
            // Once streaming, the loop cancels the token on any exit, shutting
            // the node down instead of leaving it running without a capture loop.
            let cancel_token = node_runner.cancellation_token().clone();
            let (camera_opened, capture_done) = spawn_nokhwa_capture_loop(
                camera_config.clone(),
                Arc::clone(&node_runner),
                cancel_token,
                control_rx,
            );

            // Block readiness on the camera actually streaming. A node that
            // cannot reach its camera has nothing to offer, so it exits
            // non-zero here rather than idling while answering services.
            camera_opened
                .await
                .map_err(|_| {
                    std::io::Error::other("capture thread exited before reporting camera status")
                })?
                .map_err(std::io::Error::other)?;

            // Services come up only once the loop is draining controls. Exposing
            // them earlier lets a control enqueue against a camera that is still
            // opening: the caller times out after two seconds and is told the
            // write failed, then the loop starts and applies it anyway.
            let info_runner = Arc::clone(&node_runner);
            tokio::spawn(async move {
                listen_for_video_stream_info_requests(info_runner, camera_config).await;
            });

            let exposure_runner = Arc::clone(&node_runner);
            let exposure_tx = control_tx.clone();
            tokio::spawn(async move {
                listen_for_set_exposure_requests(exposure_runner, exposure_tx).await;
            });

            let wb_runner = Arc::clone(&node_runner);
            let wb_tx = control_tx.clone();
            tokio::spawn(async move {
                listen_for_set_white_balance_requests(wb_runner, wb_tx).await;
            });

            let gain_runner = Arc::clone(&node_runner);
            let gain_tx = control_tx.clone();
            tokio::spawn(async move {
                listen_for_set_gain_requests(gain_runner, gain_tx).await;
            });

            let brightness_runner = Arc::clone(&node_runner);
            let brightness_tx = control_tx.clone();
            tokio::spawn(async move {
                listen_for_set_brightness_requests(brightness_runner, brightness_tx).await;
            });

            let contrast_runner = Arc::clone(&node_runner);
            let contrast_tx = control_tx;
            tokio::spawn(async move {
                listen_for_set_contrast_requests(contrast_runner, contrast_tx).await;
            });

            // The camera (V4L2 stream + device fd) is closed when the capture
            // thread drops it; await that here so device teardown is bounded
            // by the shutdown grace window instead of racing process exit.
            node_runner.on_shutdown(async move {
                let _ = capture_done.await;
            });

            node_runner.on_shutdown(async move {
                tracing::info!("shutdown signal received");
            });

            Ok(())
        },
    )
}
