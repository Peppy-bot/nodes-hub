mod frame;
mod geometry;
mod modes;
mod pipeline;

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
use peppygen::{NodeBuilder, NodeRunner, Parameters, Result};
use peppylib::runtime::CancellationToken;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info};

use crate::frame::FrameSet;
use crate::geometry::{Pinhole, Published};
use crate::modes::{AlignMode, AutoManualMode, ColorFormat};
use crate::pipeline::{
    Capture, DEPTH_TOPIC_ENCODING, EMIT_CHANNEL_CAPACITY, PipelineConfig, PipelineHandle, open,
};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    NodeBuilder::new().run(|params: Parameters, node_runner| async move {
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

        let color_format =
            ColorFormat::try_from(color_format.as_str()).map_err(std::io::Error::other)?;
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
            color_width,
            color_height,
            color_fps,
            color_format,
            depth_width,
            depth_height,
            depth_fps,
            align_mode,
        };
        let color_encoding = color_format.topic_encoding().to_string();
        let depth_encoding = DEPTH_TOPIC_ENCODING.to_string();
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

        let capture = open(cfg)
            .map_err(|e| std::io::Error::other(format!("open realsense pipeline: {e}")))?;
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
        spawn_video_stream_info(
            node_runner.clone(),
            color_width,
            color_height,
            color_fps.get(),
            color_encoding,
        );
        spawn_depth_stream_info(
            node_runner.clone(),
            depth_width,
            depth_height,
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
    })
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
                color,
                depth,
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

fn spawn_video_stream_info(
    runner: Arc<NodeRunner>,
    width: u32,
    height: u32,
    fps: u8,
    encoding: String,
) {
    tokio::spawn(async move {
        let cancel = runner.cancellation_token().clone();
        loop {
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = video_stream_info::handle_next_request(&runner, |_req| {
                    Ok(video_stream_info::Response::new(width, height, fps, encoding.clone()))
                }) => result,
            };
            if let Err(e) = result {
                error!("video_stream_info: {e}");
            }
        }
    });
}

fn spawn_depth_stream_info(
    runner: Arc<NodeRunner>,
    width: u32,
    height: u32,
    fps: u8,
    encoding: String,
    depth_unit: f32,
) {
    tokio::spawn(async move {
        let cancel = runner.cancellation_token().clone();
        loop {
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = depth_stream_info::handle_next_request(&runner, |_req| {
                    Ok(depth_stream_info::Response::new(width, height, fps, encoding.clone(), depth_unit))
                }) => result,
            };
            if let Err(e) = result {
                error!("depth_stream_info: {e}");
            }
        }
    });
}

/// What a geometry answer says beside its numbers: where they came from.
const GEOMETRY_MESSAGE: &str =
    "geometry from the device's calibration, for the streams as published";

/// RealSense depth is the distance along the optical axis.
const DEPTH_MODEL: &str = "z";

/// What `PipelineHandle::published_geometry` hands the services: the geometry
/// under the current align mode, or why the calibration could not be read.
type PublishedGeometry = std::result::Result<Published, String>;

/// get_color_intrinsics: the colour stream's model under the current mode.
fn color_intrinsics(published: &PublishedGeometry) -> get_color_intrinsics::Response {
    let refuse = |reason: &String| {
        get_color_intrinsics::Response::new(
            false,
            reason.clone(),
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            String::new(),
            Vec::new(),
        )
    };
    match published.as_ref().map(|p| &p.color) {
        Ok(Ok(Pinhole {
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            distortion_model,
            distortion,
        })) => get_color_intrinsics::Response::new(
            true,
            GEOMETRY_MESSAGE.to_string(),
            *width,
            *height,
            *fx,
            *fy,
            *cx,
            *cy,
            distortion_model.to_string(),
            distortion.clone(),
        ),
        Ok(Err(reason)) | Err(reason) => refuse(reason),
    }
}

/// get_depth_intrinsics: the depth stream's model under the current mode,
/// what a sample measures, and the mode the answer describes.
fn depth_intrinsics(published: &PublishedGeometry) -> get_depth_intrinsics::Response {
    let refuse = |reason: &String| {
        get_depth_intrinsics::Response::new(
            false,
            reason.clone(),
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            String::new(),
            Vec::new(),
            String::new(),
            0.0,
            0.0,
            String::new(),
        )
    };
    let published = match published {
        Ok(published) => published,
        Err(reason) => return refuse(reason),
    };
    match (&published.depth, &published.depth_range_m) {
        (Ok(depth), Ok((min_depth_m, max_depth_m))) => get_depth_intrinsics::Response::new(
            true,
            GEOMETRY_MESSAGE.to_string(),
            depth.width,
            depth.height,
            depth.fx,
            depth.fy,
            depth.cx,
            depth.cy,
            depth.distortion_model.to_string(),
            depth.distortion.clone(),
            DEPTH_MODEL.to_string(),
            *min_depth_m,
            *max_depth_m,
            published.align_mode.as_str().to_string(),
        ),
        (Err(reason), _) | (_, Err(reason)) => refuse(reason),
    }
}

/// get_depth_to_color_extrinsics: the device's transform while the streams
/// are unaligned, the identity while one is warped into the other.
fn depth_to_color_extrinsics(
    published: &PublishedGeometry,
) -> get_depth_to_color_extrinsics::Response {
    let refuse = |reason: &String| {
        get_depth_to_color_extrinsics::Response::new(
            false,
            reason.clone(),
            String::new(),
            [0.0; 3],
            [0.0; 4],
        )
    };
    let published = match published {
        Ok(published) => published,
        Err(reason) => return refuse(reason),
    };
    match &published.depth_to_color {
        Ok(pose) => get_depth_to_color_extrinsics::Response::new(
            true,
            GEOMETRY_MESSAGE.to_string(),
            published.align_mode.as_str().to_string(),
            pose.position,
            pose.orientation,
        ),
        Err(reason) => refuse(reason),
    }
}

/// One camera_geometry service. The geometry is worked out per request,
/// because set_align_mode changes what the node publishes while it runs.
macro_rules! spawn_geometry_service {
    ($fn_name:ident, $service:ident, $respond:ident) => {
        fn $fn_name(runner: Arc<NodeRunner>, handle: Arc<PipelineHandle>) {
            tokio::spawn(async move {
                let cancel = runner.cancellation_token().clone();
                loop {
                    let result = tokio::select! {
                        _ = cancel.cancelled() => break,
                        result = $service::handle_next_request(&runner, |_req| {
                            Ok($respond(&handle.published_geometry()))
                        }) => result,
                    };
                    if let Err(e) = result {
                        error!("{}: {e}", stringify!($service));
                    }
                }
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

fn spawn_set_color_exposure(runner: Arc<NodeRunner>, handle: Arc<PipelineHandle>) {
    tokio::spawn(async move {
        let cancel = runner.cancellation_token().clone();
        loop {
            let handle = handle.clone();
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = set_color_exposure::handle_next_request(&runner, move |req| {
                    let response = match AutoManualMode::parse(&req.data.mode, "exposure") {
                        Ok(mode) => match handle.set_color_exposure(mode, req.data.value) {
                            Ok(()) => set_color_exposure::Response::new(
                                true,
                                format!("color exposure set ({mode})"),
                                req.data.value,
                            ),
                            Err(e) => set_color_exposure::Response::new(
                                false,
                                format!("set color exposure: {e}"),
                                req.data.value,
                            ),
                        },
                        Err(msg) => set_color_exposure::Response::new(false, msg, req.data.value),
                    };
                    Ok(response)
                }) => result,
            };
            if let Err(e) = result {
                error!("set_color_exposure: {e}");
            }
        }
    });
}

fn spawn_set_color_white_balance(runner: Arc<NodeRunner>, handle: Arc<PipelineHandle>) {
    tokio::spawn(async move {
        let cancel = runner.cancellation_token().clone();
        loop {
            let handle = handle.clone();
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = set_color_white_balance::handle_next_request(&runner, move |req| {
                    let response = match AutoManualMode::parse(&req.data.mode, "white_balance") {
                        Ok(mode) => match handle.set_color_white_balance(mode, req.data.temperature) {
                            Ok(()) => set_color_white_balance::Response::new(
                                true,
                                format!("color white_balance set ({mode})"),
                                req.data.temperature,
                            ),
                            Err(e) => set_color_white_balance::Response::new(
                                false,
                                format!("set color white_balance: {e}"),
                                req.data.temperature,
                            ),
                        },
                        Err(msg) => set_color_white_balance::Response::new(
                            false,
                            msg,
                            req.data.temperature,
                        ),
                    };
                    Ok(response)
                }) => result,
            };
            if let Err(e) = result {
                error!("set_color_white_balance: {e}");
            }
        }
    });
}

fn spawn_set_color_gain(runner: Arc<NodeRunner>, handle: Arc<PipelineHandle>) {
    tokio::spawn(async move {
        let cancel = runner.cancellation_token().clone();
        loop {
            let handle = handle.clone();
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = set_color_gain::handle_next_request(&runner, move |req| {
                    let response = match handle.set_color_gain(req.data.value) {
                        Ok(()) => set_color_gain::Response::new(
                            true,
                            "color gain set".into(),
                            req.data.value,
                        ),
                        Err(e) => set_color_gain::Response::new(
                            false,
                            format!("set color gain: {e}"),
                            req.data.value,
                        ),
                    };
                    Ok(response)
                }) => result,
            };
            if let Err(e) = result {
                error!("set_color_gain: {e}");
            }
        }
    });
}

fn spawn_set_color_brightness(runner: Arc<NodeRunner>, handle: Arc<PipelineHandle>) {
    tokio::spawn(async move {
        let cancel = runner.cancellation_token().clone();
        loop {
            let handle = handle.clone();
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = set_color_brightness::handle_next_request(&runner, move |req| {
                    let response = match handle.set_color_brightness(req.data.value) {
                        Ok(()) => set_color_brightness::Response::new(
                            true,
                            "color brightness set".into(),
                            req.data.value,
                        ),
                        Err(e) => set_color_brightness::Response::new(
                            false,
                            format!("set color brightness: {e}"),
                            req.data.value,
                        ),
                    };
                    Ok(response)
                }) => result,
            };
            if let Err(e) = result {
                error!("set_color_brightness: {e}");
            }
        }
    });
}

fn spawn_set_color_contrast(runner: Arc<NodeRunner>, handle: Arc<PipelineHandle>) {
    tokio::spawn(async move {
        let cancel = runner.cancellation_token().clone();
        loop {
            let handle = handle.clone();
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = set_color_contrast::handle_next_request(&runner, move |req| {
                    let response = match handle.set_color_contrast(req.data.value) {
                        Ok(()) => set_color_contrast::Response::new(
                            true,
                            "color contrast set".into(),
                            req.data.value,
                        ),
                        Err(e) => set_color_contrast::Response::new(
                            false,
                            format!("set color contrast: {e}"),
                            req.data.value,
                        ),
                    };
                    Ok(response)
                }) => result,
            };
            if let Err(e) = result {
                error!("set_color_contrast: {e}");
            }
        }
    });
}

fn spawn_set_depth_gain(runner: Arc<NodeRunner>, handle: Arc<PipelineHandle>) {
    tokio::spawn(async move {
        let cancel = runner.cancellation_token().clone();
        loop {
            let handle = handle.clone();
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = set_depth_gain::handle_next_request(&runner, move |req| {
                    let response = match handle.set_depth_gain(req.data.value) {
                        Ok(()) => set_depth_gain::Response::new(
                            true,
                            "depth gain set".into(),
                            req.data.value,
                        ),
                        Err(e) => set_depth_gain::Response::new(
                            false,
                            format!("set depth gain: {e}"),
                            req.data.value,
                        ),
                    };
                    Ok(response)
                }) => result,
            };
            if let Err(e) = result {
                error!("set_depth_gain: {e}");
            }
        }
    });
}

fn spawn_set_depth_laser_power_mw(runner: Arc<NodeRunner>, handle: Arc<PipelineHandle>) {
    tokio::spawn(async move {
        let cancel = runner.cancellation_token().clone();
        loop {
            let handle = handle.clone();
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = set_depth_laser_power_mw::handle_next_request(&runner, move |req| {
                    let response = match handle.set_depth_laser_power_mw(req.data.value) {
                        Ok(()) => set_depth_laser_power_mw::Response::new(
                            true,
                            "depth laser_power set".into(),
                            req.data.value,
                        ),
                        Err(e) => set_depth_laser_power_mw::Response::new(
                            false,
                            format!("set depth laser_power: {e}"),
                            req.data.value,
                        ),
                    };
                    Ok(response)
                }) => result,
            };
            if let Err(e) = result {
                error!("set_depth_laser_power_mw: {e}");
            }
        }
    });
}

fn spawn_set_align_mode(runner: Arc<NodeRunner>, handle: Arc<PipelineHandle>) {
    tokio::spawn(async move {
        let cancel = runner.cancellation_token().clone();
        loop {
            let handle = handle.clone();
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = set_align_mode::handle_next_request(&runner, move |req| {
                    let response = match AlignMode::try_from(req.data.mode.as_str()) {
                        Ok(mode) => {
                            handle.set_align_mode(mode);
                            set_align_mode::Response::new(
                                true,
                                format!("align mode set to {mode}"),
                                mode.as_str().to_string(),
                            )
                        }
                        Err(msg) => set_align_mode::Response::new(
                            false,
                            msg,
                            handle.align_mode().as_str().to_string(),
                        ),
                    };
                    Ok(response)
                }) => result,
            };
            if let Err(e) = result {
                error!("set_align_mode: {e}");
            }
        }
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

    use crate::geometry::{Calibration, Distortion, StreamExtrinsics, StreamIntrinsics};

    fn calibration() -> Calibration {
        Calibration {
            color: StreamIntrinsics {
                width: 1280,
                height: 720,
                fx: 915.25,
                fy: 914.75,
                ppx: 641.5,
                ppy: 362.25,
                distortion: Distortion::InverseBrownConrady,
                coeffs: [0.0; 5],
            },
            depth: StreamIntrinsics {
                width: 848,
                height: 480,
                fx: 423.5,
                fy: 423.5,
                ppx: 421.75,
                ppy: 238.125,
                distortion: Distortion::BrownConrady,
                coeffs: [0.0; 5],
            },
            depth_to_color: StreamExtrinsics {
                rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                translation: [0.015, 0.0, 0.0],
            },
            depth_unit: 0.001,
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
            let depth = depth_intrinsics(&published);
            let pose = depth_to_color_extrinsics(&published);
            assert!(depth.success && pose.success);
            // The spelling the frame headers use.
            assert_eq!(depth.align_mode, mode.as_str());
            assert_eq!(pose.align_mode, mode.as_str());
        }
    }

    #[test]
    fn the_answers_follow_the_align_mode() {
        let unaligned = Ok(calibration().published(AlignMode::None));
        let depth = depth_intrinsics(&unaligned);
        assert_eq!((depth.width, depth.height, depth.fx), (848, 480, 423.5));
        assert_eq!(depth.depth_model, "z");
        assert_eq!(depth.distortion_model, "none");
        let pose = depth_to_color_extrinsics(&unaligned);
        assert_eq!(
            pose.depth_to_color_position,
            [f64::from(0.015_f32), 0.0, 0.0]
        );
        assert_eq!(pose.depth_to_color_orientation, [0.0, 0.0, 0.0, 1.0]);

        // Aligned to colour, the depth frames are colour-shaped and there is
        // nothing left between the two frames.
        let aligned = Ok(calibration().published(AlignMode::DepthToColor));
        let depth = depth_intrinsics(&aligned);
        assert_eq!((depth.width, depth.height, depth.fx), (1280, 720, 915.25));
        let pose = depth_to_color_extrinsics(&aligned);
        assert_eq!(pose.depth_to_color_position, [0.0, 0.0, 0.0]);

        // Aligned to depth, it is the colour frames that change shape.
        let color = color_intrinsics(&Ok(calibration().published(AlignMode::ColorToDepth)));
        assert_eq!((color.width, color.height, color.fx), (848, 480, 423.5));
    }

    #[test]
    fn a_calibration_that_could_not_be_read_refuses_everything_with_the_reason() {
        let unread: PublishedGeometry =
            Err("read the Color intrinsics: device disconnected".to_string());
        let color = color_intrinsics(&unread);
        let depth = depth_intrinsics(&unread);
        let pose = depth_to_color_extrinsics(&unread);
        for (success, message) in [
            (color.success, &color.message),
            (depth.success, &depth.message),
            (pose.success, &pose.message),
        ] {
            assert!(!success);
            assert!(message.contains("device disconnected"), "{message}");
        }
        assert_eq!((color.width, color.fx), (0, 0.0));
        assert!(depth.depth_model.is_empty() && depth.align_mode.is_empty());
        assert_eq!(pose.depth_to_color_orientation, [0.0; 4]);
    }

    #[test]
    fn a_colour_distortion_the_contract_cannot_express_refuses_only_what_it_touches() {
        let mut distorted = calibration();
        distorted.color.distortion = Distortion::ModifiedBrownConrady;
        distorted.color.coeffs = [-0.055, 0.066, -0.0007, 0.0005, -0.021];
        let unaligned = Ok(distorted.published(AlignMode::None));
        let color = color_intrinsics(&unaligned);
        assert!(!color.success);
        assert!(
            color.message.contains("cannot express"),
            "{}",
            color.message
        );
        assert!(depth_intrinsics(&unaligned).success);
        assert!(depth_to_color_extrinsics(&unaligned).success);
        // Warped into the colour image, depth carries the colour distortion.
        assert!(!depth_intrinsics(&Ok(distorted.published(AlignMode::DepthToColor))).success);
    }

    #[test]
    fn the_inverse_model_a_d4xx_colour_stream_reports_passes_through_by_name() {
        let mut distorted = calibration();
        distorted.color.coeffs = [-0.055, 0.066, -0.0007, 0.0005, -0.021];
        let color = color_intrinsics(&Ok(distorted.published(AlignMode::None)));
        assert!(color.success);
        assert_eq!(color.distortion_model, "inverse_plumb_bob");
        assert_eq!(
            color.distortion,
            [-0.055_f32, 0.066, -0.0007, 0.0005, -0.021].map(f64::from)
        );
        let depth = depth_intrinsics(&Ok(distorted.published(AlignMode::DepthToColor)));
        assert!(depth.success);
        assert_eq!(depth.distortion_model, "inverse_plumb_bob");
    }
}
