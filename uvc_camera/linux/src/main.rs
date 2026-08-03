use peppygen::{NodeBuilder, Parameters, Result, StandaloneConfig};
use std::sync::Arc;

use uvc_camera_linux::camera::spawn_capture_loop;
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

            // The synchronized clock stamping every frame: the OS clock in
            // wall mode, the simulator's time under sim time. Must be ready
            // before the capture loop takes its first stamp.
            peppygen::clock::init(&node_runner)
                .await
                .map_err(std::io::Error::other)?;

            // ── capture loop (long-running, dedicated thread) ──────────────
            // Once streaming, the loop cancels the token on any exit, shutting
            // the node down instead of leaving it running without a capture loop.
            let cancel_token = node_runner.cancellation_token().clone();
            let topic_encoding = camera_config.topic_encoding;
            let (camera_opened, capture_done) =
                spawn_capture_loop(camera_config, Arc::clone(&node_runner), cancel_token);

            // Block readiness on the camera actually streaming. A node that
            // cannot reach its camera has nothing to offer, so it exits
            // non-zero here rather than idling while answering services. The
            // readout carries the negotiated stream and the control handle, so
            // every service below reports and acts on the camera as it is.
            let readout = camera_opened
                .await
                .map_err(|_| {
                    std::io::Error::other("capture thread exited before reporting camera status")
                })?
                .map_err(std::io::Error::other)?;
            let controls = readout.controls;

            // Services come up only once the loop is draining controls. Exposing
            // them earlier lets a control enqueue against a camera that is still
            // opening: the caller times out after two seconds and is told the
            // write failed, then the loop starts and applies it anyway.
            let info_runner = Arc::clone(&node_runner);
            let stream = readout.description;
            tokio::spawn(async move {
                listen_for_video_stream_info_requests(info_runner, stream, topic_encoding).await;
            });

            let exposure_runner = Arc::clone(&node_runner);
            let exposure_controls = controls.clone();
            tokio::spawn(async move {
                listen_for_set_exposure_requests(exposure_runner, exposure_controls).await;
            });

            let wb_runner = Arc::clone(&node_runner);
            let wb_controls = controls.clone();
            tokio::spawn(async move {
                listen_for_set_white_balance_requests(wb_runner, wb_controls).await;
            });

            let gain_runner = Arc::clone(&node_runner);
            let gain_controls = controls.clone();
            tokio::spawn(async move {
                listen_for_set_gain_requests(gain_runner, gain_controls).await;
            });

            let brightness_runner = Arc::clone(&node_runner);
            let brightness_controls = controls.clone();
            tokio::spawn(async move {
                listen_for_set_brightness_requests(brightness_runner, brightness_controls).await;
            });

            let contrast_runner = Arc::clone(&node_runner);
            tokio::spawn(async move {
                listen_for_set_contrast_requests(contrast_runner, controls).await;
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
