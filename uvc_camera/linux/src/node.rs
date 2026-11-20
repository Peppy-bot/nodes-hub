//! The node's start: the camera's configuration, the frame every picture names, the capture
//! loop and the services, which come up once the camera streams.

use std::sync::Arc;

use camera_core::sensor_mounts::{CAMERA, DESCRIBE_SENSORS_TIMEOUT, DescribedSensors, read_sensor};
use peppygen::consumed_services::sensor_mounts::describe_sensors;
use peppygen::{NodeRunner, Parameters, Result};

use crate::camera::spawn_capture_loop;
use crate::geometry::{
    listen_for_get_color_intrinsics_requests, listen_for_get_depth_intrinsics_requests,
    listen_for_get_depth_to_color_extrinsics_requests,
};
use crate::services::{
    listen_for_set_brightness_requests, listen_for_set_contrast_requests,
    listen_for_set_exposure_requests, listen_for_set_gain_requests,
    listen_for_set_white_balance_requests, listen_for_video_stream_info_requests,
};
use crate::types::CameraConfig;

/// The node's entry point, which the binary and the test harness boot. It reads the frame
/// every picture names before it opens the camera, and fails its start when the camera does
/// not open.
pub async fn setup(args: Parameters, node_runner: Arc<NodeRunner>) -> Result<()> {
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

    // The frame every picture names, read once before the camera opens: with
    // the sensor_mounts link bound, the one the robot's sensor_mounts gives
    // the camera's name, and a name it does not list fails the node here;
    // with the link vacant, the camera's name.
    let asker = Arc::clone(&node_runner);
    let describe = describe_sensors::bound_producer(&node_runner)
        .cloned()
        .map(|mounts| {
            async move || {
                describe_sensors::poll(&asker, &mounts, DESCRIBE_SENSORS_TIMEOUT)
                    .await
                    .map(|answer| DescribedSensors {
                        sensor_names: answer.data.sensor_names,
                        sensor_kinds: answer.data.sensor_kinds,
                        sensor_frames: answer.data.sensor_frames,
                        carried_by: answer.data.carried_by,
                    })
            }
        });
    let Some(camera) = read_sensor(
        CAMERA,
        node_runner.copy(),
        node_runner.processor().bound_instance_id(),
        describe,
        node_runner.cancellation_token().cancelled(),
    )
    .await
    .map_err(std::io::Error::other)?
    else {
        return Ok(());
    };
    let camera_frame = camera.frame.to_string();
    tracing::info!("each picture names frame {camera_frame}");

    // ── capture loop (long-running, dedicated thread) ──────────────
    // Once streaming, the loop cancels the token on any exit, shutting
    // the node down instead of leaving it running without a capture loop.
    let cancel_token = node_runner.cancellation_token().clone();
    let topic_encoding = camera_config.topic_encoding;
    let (camera_opened, capture_done) = spawn_capture_loop(
        camera_config,
        camera_frame,
        Arc::clone(&node_runner),
        cancel_token,
    );

    // Block readiness on the camera actually streaming. A node that
    // cannot reach its camera has nothing to offer, so it exits
    // non-zero here rather than idling while answering services. The
    // readout carries the negotiated stream and the control handle, so
    // every service below reports and acts on the camera as it is.
    let readout = camera_opened
        .await
        .map_err(|_| std::io::Error::other("capture thread exited before reporting camera status"))?
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
}
