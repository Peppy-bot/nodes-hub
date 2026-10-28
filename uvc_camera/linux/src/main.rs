use peppygen::{NodeBuilder, Parameters, Result, StandaloneConfig};
use std::sync::Arc;

use uvc_camera_linux::camera::spawn_capture_loop;
use uvc_camera_linux::geometry::{
    listen_for_get_color_intrinsics_requests, listen_for_get_depth_intrinsics_requests,
    listen_for_get_depth_to_color_extrinsics_requests,
};
use uvc_camera_linux::services::{
    listen_for_set_brightness_requests, listen_for_set_contrast_requests,
    listen_for_set_exposure_requests, listen_for_set_gain_requests,
    listen_for_set_white_balance_requests, listen_for_video_stream_info_requests,
};
use uvc_camera_linux::types::CameraConfig;

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
            let video = &args.video;
            tracing::info!(
                "device {}: {}x{} @ {} fps, camera_encoding {}, topic_encoding {}",
                args.device_path,
                video.resolution.width,
                video.resolution.height,
                video.frame_rate,
                video.camera_encoding,
                video.topic_encoding
            );

            // An unusable parameter is a misconfiguration that fails the node
            // here, not later as a frame error.
            let camera_config = CameraConfig::from_parameters(&args)?;

            // The clock stamping every frame: the OS clock under wall time,
            // the domain's instant under a clock domain. Must be ready before
            // the capture loop takes its first timestamp.
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

            // Services come up only once the camera is streaming, so every
            // control applies to an open device and the stream description is
            // the negotiated one.
            tokio::spawn(listen_for_video_stream_info_requests(
                Arc::clone(&node_runner),
                readout.description,
                topic_encoding,
            ));
            tokio::spawn(listen_for_set_exposure_requests(
                Arc::clone(&node_runner),
                controls.clone(),
            ));
            tokio::spawn(listen_for_set_white_balance_requests(
                Arc::clone(&node_runner),
                controls.clone(),
            ));
            tokio::spawn(listen_for_set_gain_requests(
                Arc::clone(&node_runner),
                controls.clone(),
            ));
            tokio::spawn(listen_for_set_brightness_requests(
                Arc::clone(&node_runner),
                controls.clone(),
            ));
            tokio::spawn(listen_for_set_contrast_requests(
                Arc::clone(&node_runner),
                controls,
            ));

            // camera_geometry: a UVC device carries no calibration, so each
            // member gives an error reply with `rejected`.
            tokio::spawn(listen_for_get_color_intrinsics_requests(Arc::clone(
                &node_runner,
            )));
            tokio::spawn(listen_for_get_depth_intrinsics_requests(Arc::clone(
                &node_runner,
            )));
            tokio::spawn(listen_for_get_depth_to_color_extrinsics_requests(
                Arc::clone(&node_runner),
            ));

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
