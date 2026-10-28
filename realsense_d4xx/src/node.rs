//! The node's wiring: parses the parameters, opens the pipeline, hands its
//! framesets from the capture loop to the emit task, and serves the
//! rgbd_camera and camera_geometry services from the pipeline handle.

use std::num::NonZeroU8;
use std::sync::Arc;

use peppygen::emitted_topics::camera::{depth_stream, video_stream};
use peppygen::exposed_services::camera::{
    depth_stream_info, set_color_brightness, set_color_contrast, set_color_exposure,
    set_color_gain, set_color_white_balance, video_stream_info,
};
use peppygen::exposed_services::geometry::{
    get_color_intrinsics, get_depth_intrinsics, get_depth_to_color_extrinsics,
};
use peppygen::exposed_services::{set_align_mode, set_depth_gain, set_depth_laser_power_mw};
use peppygen::{NodeRunner, Parameters, Result};
use peppylib::runtime::CancellationToken;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info};

use camera_core::encoding::DEPTH_ENCODING;
use camera_core::geometry::Z_DEPTH_MODEL;

use crate::frame::FrameSet;
use crate::geometry::Pinhole;
use crate::modes::{AlignMode, parse_color_format, parse_mode};
use crate::pipeline::{
    Capture, EMIT_CHANNEL_CAPACITY, PipelineConfig, PipelineHandle, PublishedGeometry,
    StreamSettings, open,
};
use crate::stream::StreamPair;

/// Answer every request of `$service` with `$respond`, which maps the request
/// to the response or the failure of an error reply, until the node stops.
/// Each request that fails is logged with the service's name.
macro_rules! serve_requests {
    ($runner:expr, $service:ident, $respond:expr) => {{
        let cancel = $runner.cancellation_token().clone();
        loop {
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = $service::handle_next_request(&$runner, $respond) => result,
            };
            if let Err(e) = result {
                error!("{}: {e}", stringify!($service));
            }
        }
    }};
}

/// Opens the pipeline, starts the capture loop and the emit task, and serves
/// every service until the node stops.
pub async fn setup(params: Parameters, node_runner: Arc<NodeRunner>) -> Result<()> {
    let Parameters {
        serial,
        color_width,
        color_height,
        color_fps,
        color_format,
        depth_width,
        depth_height,
        depth_fps,
        align_mode,
    } = params;

    let color_format = parse_color_format(&color_format).map_err(std::io::Error::other)?;
    let align_mode = AlignMode::try_from(align_mode.as_str())
        .map_err(|e| std::io::Error::other(format!("align_mode: {e}")))?;
    let color_fps = parse_fps("color_fps", color_fps)?;
    let depth_fps = parse_fps("depth_fps", depth_fps)?;

    let cfg = PipelineConfig {
        serial: if serial.is_empty() {
            None
        } else {
            Some(serial.clone())
        },
        streams: StreamPair {
            color: StreamSettings {
                width: color_width,
                height: color_height,
                fps: color_fps,
            },
            depth: StreamSettings {
                width: depth_width,
                height: depth_height,
                fps: depth_fps,
            },
        },
        color_format,
        align_mode,
    };
    let color_encoding = color_format.name().to_string();
    let depth_encoding = DEPTH_ENCODING.to_string();
    info!(
        "realsense_d4xx opening serial={} color={}x{}@{} {} depth={}x{}@{}",
        if serial.is_empty() {
            "<first device>"
        } else {
            serial.as_str()
        },
        color_width,
        color_height,
        color_fps,
        color_format,
        depth_width,
        depth_height,
        depth_fps,
    );

    // The clock stamping every emission: the OS clock under wall time,
    // the domain's instant under a clock domain.
    peppygen::clock::init(&node_runner).await?;

    let capture =
        open(cfg).map_err(|e| std::io::Error::other(format!("open realsense pipeline: {e}")))?;
    let handle = capture.handle();

    let (frame_tx, frame_rx) = mpsc::channel::<FrameSet>(EMIT_CHANNEL_CAPACITY);
    let cancel = node_runner.cancellation_token().clone();

    // Long-running video topics
    let capture_done = spawn_capture(capture, frame_tx, cancel);
    // The capture loop owns the pipeline and stops it (`rs2_pipeline_stop`)
    // as it exits; await that from a hook so the hardware teardown
    // completes inside the bounded hook phase rather than blocking the
    // runtime teardown afterwards, unbounded.
    node_runner.on_shutdown(async move {
        let _ = capture_done.await;
    });
    spawn_emit_task(
        node_runner.clone(),
        frame_rx,
        color_encoding.clone(),
        depth_encoding.clone(),
    );

    // Services
    let configured = StreamPair {
        color: (color_width, color_height),
        depth: (depth_width, depth_height),
    };
    spawn_video_stream_info(
        node_runner.clone(),
        handle.clone(),
        configured,
        color_fps.get(),
        color_encoding,
    );
    spawn_depth_stream_info(
        node_runner.clone(),
        handle.clone(),
        configured,
        depth_fps.get(),
        depth_encoding,
        handle.depth_unit(),
    );
    spawn_get_color_intrinsics(node_runner.clone(), handle.clone());
    spawn_get_depth_intrinsics(node_runner.clone(), handle.clone());
    spawn_get_depth_to_color_extrinsics(node_runner.clone(), handle.clone());
    spawn_set_color_exposure(node_runner.clone(), handle.clone());
    spawn_set_color_white_balance(node_runner.clone(), handle.clone());
    spawn_set_color_gain(node_runner.clone(), handle.clone());
    spawn_set_color_brightness(node_runner.clone(), handle.clone());
    spawn_set_color_contrast(node_runner.clone(), handle.clone());
    spawn_set_depth_gain(node_runner.clone(), handle.clone());
    spawn_set_depth_laser_power_mw(node_runner.clone(), handle.clone());
    spawn_set_align_mode(node_runner.clone(), handle);

    // Log when the shutdown/cancel signal is received so it is visible in
    // the node's stdout (tracing's fmt subscriber writes to stdout).
    node_runner.on_shutdown(async move {
        info!("[realsense_d4xx] Shutdown signal received");
    });

    Ok(())
}

/// Narrow a peppy `u32` fps parameter to `NonZeroU8`. Topic schema's
/// `frames_per_second` is `u8`, and `0` is meaningless.
fn parse_fps(name: &str, value: u32) -> Result<NonZeroU8> {
    u8::try_from(value)
        .ok()
        .and_then(NonZeroU8::new)
        .ok_or_else(|| {
            std::io::Error::other(format!("{name} must be in 1..=255 (got {value})")).into()
        })
}

/// Start the capture loop on a blocking thread. Returns a receiver that
/// resolves once the loop has exited and the pipeline is stopped, for the
/// shutdown hook to await.
fn spawn_capture(
    capture: Capture,
    frame_tx: mpsc::Sender<FrameSet>,
    cancel: CancellationToken,
) -> oneshot::Receiver<()> {
    let cancel_on_panic = cancel.clone();
    let join = tokio::task::spawn_blocking(move || {
        capture.run(frame_tx, cancel);
    });
    let (done_tx, done_rx) = oneshot::channel();
    // A silent capture panic would leave service handlers answering with
    // stale state; propagate it.
    tokio::spawn(async move {
        if let Err(e) = join.await {
            error!("capture task failed: {e}; shutting down");
            cancel_on_panic.cancel();
        }
        let _ = done_tx.send(());
    });
    done_rx
}

// Read from capture and emit to topic
fn spawn_emit_task(
    runner: Arc<NodeRunner>,
    mut frame_rx: mpsc::Receiver<FrameSet>,
    color_encoding: String,
    depth_encoding: String,
) {
    tokio::spawn(async move {
        let color_publisher = match video_stream::declare_publisher(&runner).await {
            Ok(publisher) => publisher,
            Err(e) => {
                error!("video_stream declare_publisher: {e}");
                return;
            }
        };
        let depth_publisher = match depth_stream::declare_publisher(&runner).await {
            Ok(publisher) => publisher,
            Err(e) => {
                error!("depth_stream declare_publisher: {e}");
                return;
            }
        };
        while let Some(frameset) = frame_rx.recv().await {
            let FrameSet {
                frame_id,
                timestamp,
                align_mode,
                images: StreamPair { color, depth },
            } = frameset;
            let align_mode = align_mode.as_str();

            let color_header = video_stream::MessageHeader {
                timestamp,
                frame_id,
                align_mode: align_mode.to_string(),
            };
            match video_stream::build_message(
                color_header,
                color_encoding.clone(),
                color.width,
                color.height,
                color.bytes,
            ) {
                Ok(payload) => {
                    if let Err(e) = color_publisher.publish(payload).await {
                        error!("video_stream publish: {e}");
                    }
                }
                Err(e) => error!("video_stream build_message: {e}"),
            }

            let depth_header = depth_stream::MessageHeader {
                timestamp,
                frame_id,
                align_mode: align_mode.to_string(),
            };
            match depth_stream::build_message(
                depth_header,
                depth_encoding.clone(),
                depth.width,
                depth.height,
                depth.bytes,
            ) {
                Ok(payload) => {
                    if let Err(e) = depth_publisher.publish(payload).await {
                        error!("depth_stream publish: {e}");
                    }
                }
                Err(e) => error!("depth_stream build_message: {e}"),
            }
        }
        info!("emit task stopped");
    });
}

/// video_stream_info answers the size the colour frames have under the
/// current align mode, worked out per request because set_align_mode
/// changes it while the node runs.
fn spawn_video_stream_info(
    runner: Arc<NodeRunner>,
    handle: Arc<PipelineHandle>,
    configured: StreamPair<(u32, u32)>,
    fps: u8,
    encoding: String,
) {
    tokio::spawn(async move {
        serve_requests!(runner, video_stream_info, |_req| {
            let (width, height) = handle.align_mode().published(configured).color;
            Ok(video_stream_info::Response::new(
                width,
                height,
                fps,
                encoding.clone(),
            ))
        });
    });
}

/// depth_stream_info answers the size the depth frames have under the
/// current align mode, as video_stream_info does for colour.
fn spawn_depth_stream_info(
    runner: Arc<NodeRunner>,
    handle: Arc<PipelineHandle>,
    configured: StreamPair<(u32, u32)>,
    fps: u8,
    encoding: String,
    depth_unit: f32,
) {
    tokio::spawn(async move {
        serve_requests!(runner, depth_stream_info, |_req| {
            let (width, height) = handle.align_mode().published(configured).depth;
            Ok(depth_stream_info::Response::new(
                width,
                height,
                fps,
                encoding.clone(),
                depth_unit,
            ))
        });
    });
}

/// get_color_intrinsics: the colour stream's model under the current mode.
/// A calibration that could not be read, or that the contract cannot
/// express, gives an error reply with `rejected`.
fn color_intrinsics(
    published: &PublishedGeometry,
) -> std::result::Result<get_color_intrinsics::Response, get_color_intrinsics::Failure> {
    let refuse = |reason: &String| {
        get_color_intrinsics::Failure::new(
            get_color_intrinsics::FailureKind::Rejected,
            reason.clone(),
        )
    };
    match published.as_ref().map(|p| &p.pinholes.color) {
        Ok(Ok(Pinhole {
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            distortion_model,
            distortion,
        })) => Ok(get_color_intrinsics::Response::new(
            *width,
            *height,
            *fx,
            *fy,
            *cx,
            *cy,
            distortion_model.to_string(),
            distortion.clone(),
        )),
        Ok(Err(reason)) | Err(reason) => Err(refuse(reason)),
    }
}

/// get_depth_intrinsics: the depth stream's model under the current mode,
/// what a sample measures, and the mode the answer describes. Refused as
/// get_color_intrinsics is.
fn depth_intrinsics(
    published: &PublishedGeometry,
) -> std::result::Result<get_depth_intrinsics::Response, get_depth_intrinsics::Failure> {
    let refuse = |reason: &String| {
        get_depth_intrinsics::Failure::new(
            get_depth_intrinsics::FailureKind::Rejected,
            reason.clone(),
        )
    };
    let published = published.as_ref().map_err(refuse)?;
    match (&published.pinholes.depth, &published.depth_range_m) {
        (Ok(depth), Ok((min_depth_m, max_depth_m))) => Ok(get_depth_intrinsics::Response::new(
            depth.width,
            depth.height,
            depth.fx,
            depth.fy,
            depth.cx,
            depth.cy,
            depth.distortion_model.to_string(),
            depth.distortion.clone(),
            Z_DEPTH_MODEL.to_string(),
            *min_depth_m,
            *max_depth_m,
            published.align_mode.as_str().to_string(),
        )),
        (Err(reason), _) | (_, Err(reason)) => Err(refuse(reason)),
    }
}

/// get_depth_to_color_extrinsics: the device's transform while the streams
/// are unaligned, the identity while one is warped into the other. Refused
/// as get_color_intrinsics is.
fn depth_to_color_extrinsics(
    published: &PublishedGeometry,
) -> std::result::Result<
    get_depth_to_color_extrinsics::Response,
    get_depth_to_color_extrinsics::Failure,
> {
    let refuse = |reason: &String| {
        get_depth_to_color_extrinsics::Failure::new(
            get_depth_to_color_extrinsics::FailureKind::Rejected,
            reason.clone(),
        )
    };
    let published = published.as_ref().map_err(refuse)?;
    let pose = published.depth_to_color.as_ref().map_err(refuse)?;
    Ok(get_depth_to_color_extrinsics::Response::new(
        published.align_mode.as_str().to_string(),
        pose.position,
        pose.orientation,
    ))
}

/// One camera_geometry service. The geometry is worked out per request,
/// because set_align_mode changes what the node publishes while it runs.
macro_rules! spawn_geometry_service {
    ($fn_name:ident, $service:ident, $respond:ident) => {
        fn $fn_name(runner: Arc<NodeRunner>, handle: Arc<PipelineHandle>) {
            tokio::spawn(async move {
                serve_requests!(runner, $service, |_req| {
                    $respond(&handle.published_geometry())
                });
            });
        }
    };
}

spawn_geometry_service!(
    spawn_get_color_intrinsics,
    get_color_intrinsics,
    color_intrinsics
);
spawn_geometry_service!(
    spawn_get_depth_intrinsics,
    get_depth_intrinsics,
    depth_intrinsics
);
spawn_geometry_service!(
    spawn_get_depth_to_color_extrinsics,
    get_depth_to_color_extrinsics,
    depth_to_color_extrinsics
);

/// One service that writes a device option in an auto or a manual mode,
/// with `$set` of the pipeline handle, and answers the value of the
/// request. A mode other than "auto" and "manual" gives an error reply with
/// `rejected`.
macro_rules! spawn_mode_control_service {
    ($fn_name:ident, $service:ident, $set:ident, $value:ident, $label:literal) => {
        fn $fn_name(runner: Arc<NodeRunner>, handle: Arc<PipelineHandle>) {
            tokio::spawn(async move {
                serve_requests!(runner, $service, |req| {
                    let mode = parse_mode(&req.data.mode, $label).map_err(|reason| {
                        $service::Failure::new($service::FailureKind::Rejected, reason)
                    })?;
                    handle
                        .$set(mode, req.data.$value)
                        .map(|()| $service::Response::new(req.data.$value))
                        .map_err(|failure| camera_core::control_failure!($service, failure))
                });
            });
        }
    };
}

spawn_mode_control_service!(
    spawn_set_color_exposure,
    set_color_exposure,
    set_color_exposure,
    value,
    "exposure"
);
spawn_mode_control_service!(
    spawn_set_color_white_balance,
    set_color_white_balance,
    set_color_white_balance,
    temperature,
    "white_balance"
);

/// One service that writes a single device option, with `$set` of the
/// pipeline handle, and answers the value of the request.
macro_rules! spawn_value_control_service {
    ($fn_name:ident, $service:ident, $set:ident) => {
        fn $fn_name(runner: Arc<NodeRunner>, handle: Arc<PipelineHandle>) {
            tokio::spawn(async move {
                serve_requests!(runner, $service, |req| {
                    handle
                        .$set(req.data.value)
                        .map(|()| $service::Response::new(req.data.value))
                        .map_err(|failure| camera_core::control_failure!($service, failure))
                });
            });
        }
    };
}

spawn_value_control_service!(spawn_set_color_gain, set_color_gain, set_color_gain);
spawn_value_control_service!(
    spawn_set_color_brightness,
    set_color_brightness,
    set_color_brightness
);
spawn_value_control_service!(
    spawn_set_color_contrast,
    set_color_contrast,
    set_color_contrast
);
spawn_value_control_service!(spawn_set_depth_gain, set_depth_gain, set_depth_gain);
spawn_value_control_service!(
    spawn_set_depth_laser_power_mw,
    set_depth_laser_power_mw,
    set_depth_laser_power_mw
);

/// set_align_mode switches the alignment the capture loop applies from the
/// next frame on. An unknown mode gives an error reply with `rejected`.
fn spawn_set_align_mode(runner: Arc<NodeRunner>, handle: Arc<PipelineHandle>) {
    tokio::spawn(async move {
        serve_requests!(runner, set_align_mode, |req| {
            let mode = AlignMode::try_from(req.data.mode.as_str()).map_err(|reason| {
                set_align_mode::Failure::new(set_align_mode::FailureKind::Rejected, reason)
            })?;
            handle.set_align_mode(mode);
            Ok(set_align_mode::Response::new(mode.as_str().to_string()))
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_fps_accepts_valid_range() {
        assert_eq!(parse_fps("fps", 1).unwrap().get(), 1);
        assert_eq!(parse_fps("fps", 30).unwrap().get(), 30);
        assert_eq!(parse_fps("fps", 255).unwrap().get(), 255);
    }

    #[test]
    fn parse_fps_rejects_zero() {
        let err = parse_fps("color_fps", 0).unwrap_err();
        assert!(
            err.to_string()
                .contains("color_fps must be in 1..=255 (got 0)")
        );
    }

    #[test]
    fn parse_fps_rejects_over_u8() {
        let err = parse_fps("depth_fps", 256).unwrap_err();
        assert!(
            err.to_string()
                .contains("depth_fps must be in 1..=255 (got 256)")
        );
    }

    use crate::geometry::tests::d435;
    use crate::geometry::{Calibration, Distortion, StreamExtrinsics};

    /// The D435 of the geometry tests, its depth imager straight beside the
    /// colour one.
    fn calibration() -> Calibration {
        Calibration {
            depth_to_color: StreamExtrinsics {
                rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                translation: [0.015, 0.0, 0.0],
            },
            ..d435()
        }
    }

    #[test]
    fn both_depth_answers_name_the_mode_the_frames_carry() {
        for mode in [
            AlignMode::None,
            AlignMode::DepthToColor,
            AlignMode::ColorToDepth,
        ] {
            let published = Ok(calibration().published(mode));
            let depth = depth_intrinsics(&published).unwrap();
            let pose = depth_to_color_extrinsics(&published).unwrap();
            // The spelling the frame headers use.
            assert_eq!(depth.align_mode, mode.as_str());
            assert_eq!(pose.align_mode, mode.as_str());
        }
    }

    #[test]
    fn the_answers_follow_the_align_mode() {
        let unaligned = Ok(calibration().published(AlignMode::None));
        let depth = depth_intrinsics(&unaligned).unwrap();
        assert_eq!((depth.width, depth.height, depth.fx), (848, 480, 423.5));
        assert_eq!(depth.depth_model, "z");
        assert_eq!(depth.distortion_model, "none");
        let pose = depth_to_color_extrinsics(&unaligned).unwrap();
        assert_eq!(
            pose.depth_to_color_position,
            [f64::from(0.015_f32), 0.0, 0.0]
        );
        assert_eq!(pose.depth_to_color_orientation, [0.0, 0.0, 0.0, 1.0]);

        // Aligned to colour, the depth frames are colour-shaped and there is
        // nothing left between the two frames.
        let aligned = Ok(calibration().published(AlignMode::DepthToColor));
        let depth = depth_intrinsics(&aligned).unwrap();
        assert_eq!((depth.width, depth.height, depth.fx), (1280, 720, 915.25));
        let pose = depth_to_color_extrinsics(&aligned).unwrap();
        assert_eq!(pose.depth_to_color_position, [0.0, 0.0, 0.0]);

        // Aligned to depth, it is the colour frames that change shape.
        let color =
            color_intrinsics(&Ok(calibration().published(AlignMode::ColorToDepth))).unwrap();
        assert_eq!((color.width, color.height, color.fx), (848, 480, 423.5));
    }

    #[test]
    fn a_calibration_that_could_not_be_read_refuses_everything_with_the_reason() {
        let unread: PublishedGeometry =
            Err("read the Color intrinsics: device disconnected".to_string());
        let color = color_intrinsics(&unread).unwrap_err();
        assert_eq!(color.kind, get_color_intrinsics::FailureKind::Rejected);
        let depth = depth_intrinsics(&unread).unwrap_err();
        assert_eq!(depth.kind, get_depth_intrinsics::FailureKind::Rejected);
        let pose = depth_to_color_extrinsics(&unread).unwrap_err();
        assert_eq!(
            pose.kind,
            get_depth_to_color_extrinsics::FailureKind::Rejected
        );
        for message in [&color.message, &depth.message, &pose.message] {
            assert!(message.contains("device disconnected"), "{message}");
        }
    }

    #[test]
    fn a_colour_distortion_the_contract_cannot_express_refuses_only_what_it_touches() {
        let mut distorted = calibration();
        distorted.streams.color.distortion = Distortion::ModifiedBrownConrady;
        distorted.streams.color.coeffs = [-0.055, 0.066, -0.0007, 0.0005, -0.021];
        let unaligned = Ok(distorted.published(AlignMode::None));
        let color = color_intrinsics(&unaligned).unwrap_err();
        assert_eq!(color.kind, get_color_intrinsics::FailureKind::Rejected);
        assert!(
            color.message.contains("cannot express"),
            "{}",
            color.message
        );
        assert!(depth_intrinsics(&unaligned).is_ok());
        assert!(depth_to_color_extrinsics(&unaligned).is_ok());
        // Warped into the colour image, depth carries the colour distortion.
        let depth =
            depth_intrinsics(&Ok(distorted.published(AlignMode::DepthToColor))).unwrap_err();
        assert_eq!(depth.kind, get_depth_intrinsics::FailureKind::Rejected);
    }

    #[test]
    fn the_stream_infos_take_the_targets_size_under_alignment() {
        // What the geometry answers say of the aligned streams, the stream
        // infos say too: a consumer comparing the two never sees a depth
        // answer of 1280x720 beside a depth_stream_info of 848x480.
        let configured = StreamPair {
            color: (1280, 720),
            depth: (848, 480),
        };
        let unaligned = AlignMode::None.published(configured);
        assert_eq!(
            (unaligned.color, unaligned.depth),
            ((1280, 720), (848, 480))
        );
        let to_color = AlignMode::DepthToColor.published(configured);
        assert_eq!((to_color.color, to_color.depth), ((1280, 720), (1280, 720)));
        let to_depth = AlignMode::ColorToDepth.published(configured);
        assert_eq!((to_depth.color, to_depth.depth), ((848, 480), (848, 480)));
        for mode in [
            AlignMode::None,
            AlignMode::DepthToColor,
            AlignMode::ColorToDepth,
        ] {
            let sizes = mode.published(configured);
            let published = calibration().published(mode);
            let (color, depth) = (
                published.pinholes.color.unwrap(),
                published.pinholes.depth.unwrap(),
            );
            assert_eq!(sizes.color, (color.width, color.height), "{mode}");
            assert_eq!(sizes.depth, (depth.width, depth.height), "{mode}");
        }
    }

    #[test]
    fn the_inverse_model_a_d4xx_colour_stream_reports_passes_through_by_name() {
        let mut distorted = calibration();
        distorted.streams.color.coeffs = [-0.055, 0.066, -0.0007, 0.0005, -0.021];
        let color = color_intrinsics(&Ok(distorted.published(AlignMode::None))).unwrap();
        assert_eq!(color.distortion_model, "inverse_plumb_bob");
        assert_eq!(
            color.distortion,
            [-0.055_f32, 0.066, -0.0007, 0.0005, -0.021].map(f64::from)
        );
        let depth = depth_intrinsics(&Ok(distorted.published(AlignMode::DepthToColor))).unwrap();
        assert_eq!(depth.distortion_model, "inverse_plumb_bob");
    }
}
