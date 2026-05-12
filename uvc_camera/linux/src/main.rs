use peppygen::{NodeBuilder, Parameters, Result, StandaloneConfig};
use std::sync::Arc;

use uvc_camera::camera::{create_control_channel, run_nokhwa_capture_loop};
use uvc_camera::services::{
    listen_for_set_brightness_requests, listen_for_set_contrast_requests,
    listen_for_set_exposure_requests, listen_for_set_gain_requests,
    listen_for_set_white_balance_requests, listen_for_video_stream_info_requests,
};
use uvc_camera::types::{CameraConfigBuilder, Encoding};

fn main() -> Result<()> {
    // Load parameters from mock file for standalone execution
    let mock_params: Parameters = serde_json::from_str(
        &std::fs::read_to_string("mock_parameters.json")
            .expect("Failed to read mock_parameters.json"),
    )
    .expect("Failed to parse mock_parameters.json");
    let standalone_config = StandaloneConfig::new().with_parameters(&mock_params);

    NodeBuilder::new()
        .standalone(standalone_config)
        .run(move |args: Parameters, node_runner| async move {
            let video_params = args.video.clone();
            let device_path = args.device_path.clone();

            println!(
                "[uvc_camera] Video params: {}x{} @ {} fps, camera_encoding: {}, topic_encoding: {}",
                video_params.resolution.width,
                video_params.resolution.height,
                video_params.frame_rate,
                video_params.camera_encoding,
                video_params.topic_encoding
            );

            println!("[uvc_camera] Device: {device_path}");

            // Parse and validate encoding formats
            let camera_encoding = video_params.camera_encoding.parse::<Encoding>()
                .unwrap_or_else(|e| {
                    panic!("Invalid camera_encoding '{}': {}", video_params.camera_encoding, e)
                });
            let topic_encoding = video_params.topic_encoding.parse::<Encoding>()
                .unwrap_or_else(|e| {
                    panic!("Invalid topic_encoding '{}': {}", video_params.topic_encoding, e)
                });

            // Create camera configuration
            let camera_config = CameraConfigBuilder::new()
                .device_path(device_path.clone())
                .resolution(video_params.resolution.width, video_params.resolution.height)
                .frame_rate(video_params.frame_rate)
                .camera_encoding(camera_encoding)
                .topic_encoding(topic_encoding)
                .build()
                .unwrap_or_else(|e| panic!("Failed to create camera config: {}", e));

            // Create control channel shared between service handlers and capture loop
            let (control_tx, control_rx) = create_control_channel();

            // ── video_stream_info ──────────────────────────────────────────
            let info_runner = Arc::clone(&node_runner);
            let info_config = camera_config.clone();
            tokio::spawn(async move {
                listen_for_video_stream_info_requests(info_runner, info_config).await;
            });

            // ── set_exposure ───────────────────────────────────────────────
            let exposure_runner = Arc::clone(&node_runner);
            let exposure_tx = control_tx.clone();
            tokio::spawn(async move {
                listen_for_set_exposure_requests(exposure_runner, exposure_tx).await;
            });

            // ── set_white_balance ──────────────────────────────────────────
            let wb_runner = Arc::clone(&node_runner);
            let wb_tx = control_tx.clone();
            tokio::spawn(async move {
                listen_for_set_white_balance_requests(wb_runner, wb_tx).await;
            });

            // ── set_gain ───────────────────────────────────────────────────
            let gain_runner = Arc::clone(&node_runner);
            let gain_tx = control_tx.clone();
            tokio::spawn(async move {
                listen_for_set_gain_requests(gain_runner, gain_tx).await;
            });

            // ── set_brightness ─────────────────────────────────────────────
            let brightness_runner = Arc::clone(&node_runner);
            let brightness_tx = control_tx.clone();
            tokio::spawn(async move {
                listen_for_set_brightness_requests(brightness_runner, brightness_tx).await;
            });

            // ── set_contrast ───────────────────────────────────────────────
            let contrast_runner = Arc::clone(&node_runner);
            let contrast_tx = control_tx.clone();
            tokio::spawn(async move {
                listen_for_set_contrast_requests(contrast_runner, contrast_tx).await;
            });

            // ── capture loop (long-running) ────────────────────────────────
            let cancel_token = node_runner.cancellation_token().clone();
            tokio::spawn(async move {
                if let Err(e) =
                    run_nokhwa_capture_loop(camera_config, node_runner, cancel_token, control_rx)
                        .await
                {
                    panic!("[uvc_camera] Camera capture loop failed: {e}");
                }
            });

            Ok(())
        })
}
