mod frame;
mod modes;
mod pipeline;

use std::num::NonZeroU8;
use std::sync::Arc;

use peppygen::emitted_topics::rgbd_camera::v1::{depth_stream, video_stream};
use peppygen::exposed_services::rgbd_camera::v1::{
    depth_stream_info, set_align_mode, set_color_brightness, set_color_contrast,
    set_color_exposure, set_color_gain, set_color_white_balance, set_depth_gain,
    set_depth_laser_power_mw, video_stream_info,
};
use peppygen::{NodeBuilder, NodeRunner, Parameters, Result};
use peppylib::runtime::CancellationToken;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info};

use crate::frame::FrameSet;
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
        } = params;

        let color_format =
            ColorFormat::try_from(color_format.as_str()).map_err(std::io::Error::other)?;
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
                stamp,
                align_mode,
                color,
                depth,
            } = frameset;
            let align_mode = align_mode.as_str();

            let color_header = video_stream::MessageHeader {
                stamp,
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
                stamp,
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
}
