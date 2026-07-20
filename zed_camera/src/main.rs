//! Stereolabs ZED as an rgbd_camera producer without the ZED SDK.
//!
//! A blocking pipeline thread owns the frame path: grab side-by-side YUYV
//! through the library's capture layer, rectify through the launcher-provided
//! factory calibration, publish the rectified left eye as video_stream and
//! stereo-matched depth (millimeters, z16) as depth_stream. Depth lives in
//! the rectified-left frame, so the pair is permanently color-aligned. The
//! pipeline loop and the control services share one capture handle behind a
//! mutex, so a control call waits at most one frame.

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use peppygen::emitted_topics::rgbd_camera::v1::{depth_stream, video_stream};
use peppygen::exposed_services::rgbd_camera::v1::{
    depth_stream_info, set_color_brightness, set_color_contrast, set_color_exposure,
    set_color_gain, set_color_white_balance, video_stream_info,
};
use peppygen::{NodeBuilder, NodeRunner, Parameters, Result};
use peppylib::runtime::CancellationToken;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, warn};
use zed_camera::capture::{
    CID_AWB_AUTO, CID_AWB_TEMPERATURE, CID_BRIGHTNESS, CID_CONTRAST, CID_GAIN, Capture, Grab,
    device_index, zed_serial,
};
use zed_camera::cv_depth::CvDepth;
use zed_camera::{DepthSettings, Resolution};

const GRAB_TIMEOUT: Duration = Duration::from_millis(500);
const EMIT_CHANNEL_CAPACITY: usize = 2;
const DEPTH_UNIT_M_PER_LSB: f32 = 0.001;
const COLOR_ENCODING: &str = "rgb8";
const DEPTH_ENCODING: &str = "z16";
/// Depth is computed in the rectified-left (= published color) frame.
const ALIGN_MODE: &str = "depth_to_color";

/// The capture device shared between the pipeline loop and the control
/// services; every access is a short lock.
type Camera = Arc<Mutex<Capture>>;

struct PipelineConfig {
    dev_id: usize,
    resolution: Resolution,
    fps: u32,
    calibration_path: std::path::PathBuf,
    depth: DepthSettings,
}

/// What the pipeline thread reports back once the camera and matcher are up.
struct Opened {
    camera: Camera,
    eye_width: u32,
    eye_height: u32,
    depth_width: u32,
    depth_height: u32,
}

/// One processed capture: rectified left RGB plus depth, sharing a frame id.
struct FrameSet {
    frame_id: u32,
    stamp: SystemTime,
    left_rgb: Vec<u8>,
    depth_z16: Vec<u8>,
}

fn stamp_now() -> std::result::Result<SystemTime, String> {
    let ns = peppygen::clock::now_ns().map_err(|e| e.to_string())?;
    Ok(UNIX_EPOCH + Duration::from_nanos(ns))
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    NodeBuilder::new().run(|params: Parameters, node_runner| async move {
        let resolution = Resolution::parse(&params.resolution)
            .map_err(|e| std::io::Error::other(format!("resolution: {e}")))?;
        resolution
            .validate_fps(params.frame_rate)
            .map_err(|e| std::io::Error::other(format!("frame_rate: {e}")))?;
        let dev_id = device_index(&params.device_path).map_err(std::io::Error::other)?;
        let fps = params.frame_rate;
        let fps_u8 = u8::try_from(fps)
            .map_err(|_| std::io::Error::other(format!("frame_rate {fps} does not fit u8")))?;
        let depth = DepthSettings::new(
            params.depth.min_depth_m,
            params.depth.block_size,
            params.depth.downscale,
        )
        .map_err(|e| std::io::Error::other(format!("depth: {e}")))?;
        let config = PipelineConfig {
            dev_id,
            resolution,
            fps,
            calibration_path: std::path::PathBuf::from(&params.calibration_path),
            depth,
        };

        // The synchronized clock stamping every emission.
        peppygen::clock::init(&node_runner).await?;

        let (frame_tx, frame_rx) = mpsc::channel::<FrameSet>(EMIT_CHANNEL_CAPACITY);
        let (ready_tx, ready_rx) = oneshot::channel();
        let cancel = node_runner.cancellation_token().clone();

        // The pipeline thread owns open, calibration, the matcher, and the
        // frame loop; it reports the negotiated geometry through ready_tx.
        let pipeline_cancel = cancel.clone();
        let pipeline = tokio::task::spawn_blocking(move || {
            run_pipeline(config, ready_tx, frame_tx, pipeline_cancel);
        });
        // A dead pipeline must take the node down rather than leaving the
        // services answering for a stream that no longer exists.
        let watchdog_cancel = cancel.clone();
        tokio::spawn(async move {
            if let Err(e) = pipeline.await {
                error!("pipeline task failed: {e}");
            }
            watchdog_cancel.cancel();
        });

        let Opened {
            camera,
            eye_width,
            eye_height,
            depth_width,
            depth_height,
        } = ready_rx
            .await
            .map_err(|_| std::io::Error::other("pipeline exited before opening the camera"))?
            .map_err(std::io::Error::other)?;
        info!(
            "zed_camera {resolution} ({eye_width}x{eye_height}) @ {fps} fps, \
             depth {depth_width}x{depth_height}"
        );

        spawn_emit_task(
            node_runner.clone(),
            frame_rx,
            (eye_width, eye_height),
            (depth_width, depth_height),
        );
        spawn_stream_infos(
            node_runner.clone(),
            eye_width,
            eye_height,
            depth_width,
            depth_height,
            fps_u8,
        );
        spawn_set_color_exposure(node_runner.clone());
        spawn_set_color_white_balance(node_runner.clone(), camera.clone());
        spawn_set_color_gain(node_runner.clone(), camera.clone());
        spawn_set_color_brightness(node_runner.clone(), camera.clone());
        spawn_set_color_contrast(node_runner, camera);

        Ok(())
    })
}

/// Opens the camera and matcher, reports through `ready`, then runs the frame
/// loop. The CvDepth matcher never leaves this thread.
fn run_pipeline(
    config: PipelineConfig,
    ready: oneshot::Sender<std::result::Result<Opened, String>>,
    frame_tx: mpsc::Sender<FrameSet>,
    cancel: CancellationToken,
) {
    let (mut matcher, opened) = match open_pipeline(&config) {
        Ok(opened) => opened,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let camera = opened.camera.clone();
    let (eye_width, eye_height) = (opened.eye_width, opened.eye_height);
    let (depth_width, depth_height) = (opened.depth_width, opened.depth_height);
    let _ = ready.send(Ok(opened));

    let mut yuyv = vec![0u8; (2 * eye_width * eye_height * 2) as usize];
    let mut left_rgb = vec![0u8; (eye_width * eye_height * 3) as usize];
    let mut depth_mm = vec![0u16; (depth_width * depth_height) as usize];
    let mut frame_id: u32 = 0;
    let mut processing_failing = false;

    while !cancel.is_cancelled() {
        let grabbed = lock_camera(&camera).grab(&mut yuyv, GRAB_TIMEOUT);
        match grabbed {
            Grab::Timeout => continue,
            Grab::Dead => {
                error!("capture stream died");
                return;
            }
            Grab::Frame { .. } => {}
        }
        let Ok(stamp) = stamp_now() else {
            continue; // clock not ready yet: skip rather than mis-stamp
        };
        if let Err(e) = matcher.process(&yuyv, &mut left_rgb, &mut depth_mm) {
            if !processing_failing {
                processing_failing = true;
                warn!("depth processing failing, suppressing repeats: {e}");
            }
            continue;
        }
        processing_failing = false;

        let frame = FrameSet {
            frame_id,
            stamp,
            left_rgb: left_rgb.clone(),
            depth_z16: depth_mm.iter().flat_map(|d| d.to_le_bytes()).collect(),
        };
        frame_id = frame_id.wrapping_add(1);
        // Latest-value semantics: drop the frame if the emitter is behind.
        let _ = frame_tx.try_send(frame);
    }
}

fn open_pipeline(config: &PipelineConfig) -> std::result::Result<(CvDepth, Opened), String> {
    // Best-effort unit identification for the logs; calibration comes from
    // the launcher-provided file.
    match zed_serial() {
        Ok(serial) => info!("zed unit serial {serial}"),
        Err(e) => info!("zed serial unavailable: {e}"),
    }
    let capture = Capture::open(config.dev_id, config.resolution, config.fps)?;
    let (full_width, height) = capture.frame_size();
    let (eye_width, eye_height) = (full_width / 2, height);

    let matcher = CvDepth::create(
        &config.calibration_path,
        eye_width,
        eye_height,
        config.depth,
    )?;
    let (depth_width, depth_height) = matcher.out_size();

    Ok((
        matcher,
        Opened {
            camera: Arc::new(Mutex::new(capture)),
            eye_width,
            eye_height,
            depth_width,
            depth_height,
        },
    ))
}

fn spawn_emit_task(
    runner: Arc<NodeRunner>,
    mut frame_rx: mpsc::Receiver<FrameSet>,
    (color_width, color_height): (u32, u32),
    (depth_width, depth_height): (u32, u32),
) {
    tokio::spawn(async move {
        let color_pub = match video_stream::declare_publisher(&runner).await {
            Ok(publisher) => publisher,
            Err(e) => return error!("video_stream declare_publisher: {e}"),
        };
        let depth_pub = match depth_stream::declare_publisher(&runner).await {
            Ok(publisher) => publisher,
            Err(e) => return error!("depth_stream declare_publisher: {e}"),
        };
        while let Some(frame) = frame_rx.recv().await {
            let FrameSet {
                frame_id,
                stamp,
                left_rgb,
                depth_z16,
            } = frame;
            let color_header = video_stream::MessageHeader {
                stamp,
                frame_id,
                align_mode: ALIGN_MODE.to_string(),
            };
            match video_stream::build_message(
                color_header,
                COLOR_ENCODING.to_string(),
                color_width,
                color_height,
                left_rgb,
            ) {
                Ok(payload) => {
                    if let Err(e) = color_pub.publish(payload).await {
                        error!("video_stream publish: {e}");
                    }
                }
                Err(e) => error!("video_stream build_message: {e}"),
            }

            let depth_header = depth_stream::MessageHeader {
                stamp,
                frame_id,
                align_mode: ALIGN_MODE.to_string(),
            };
            match depth_stream::build_message(
                depth_header,
                DEPTH_ENCODING.to_string(),
                depth_width,
                depth_height,
                depth_z16,
            ) {
                Ok(payload) => {
                    if let Err(e) = depth_pub.publish(payload).await {
                        error!("depth_stream publish: {e}");
                    }
                }
                Err(e) => error!("depth_stream build_message: {e}"),
            }
        }
        info!("emit task stopped");
    });
}

fn spawn_stream_infos(
    runner: Arc<NodeRunner>,
    eye_width: u32,
    eye_height: u32,
    depth_width: u32,
    depth_height: u32,
    fps: u8,
) {
    let video_runner = runner.clone();
    tokio::spawn(async move {
        let cancel = video_runner.cancellation_token().clone();
        loop {
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = video_stream_info::handle_next_request(&video_runner, |_req| {
                    Ok(video_stream_info::Response::new(
                        eye_width,
                        eye_height,
                        fps,
                        COLOR_ENCODING.to_string(),
                    ))
                }) => result,
            };
            if let Err(e) = result {
                error!("video_stream_info: {e}");
            }
        }
    });
    tokio::spawn(async move {
        let cancel = runner.cancellation_token().clone();
        loop {
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = depth_stream_info::handle_next_request(&runner, |_req| {
                    Ok(depth_stream_info::Response::new(
                        depth_width,
                        depth_height,
                        fps,
                        DEPTH_ENCODING.to_string(),
                        DEPTH_UNIT_M_PER_LSB,
                    ))
                }) => result,
            };
            if let Err(e) = result {
                error!("depth_stream_info: {e}");
            }
        }
    });
}

fn lock_camera(camera: &Camera) -> std::sync::MutexGuard<'_, Capture> {
    camera
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Writes one control and reads it back, folding failures into the service
/// response instead of a transport error.
fn apply_control(camera: &Camera, cid: u32, value: i32) -> (bool, String, i32) {
    let cam = lock_camera(camera);
    let applied = cam.set_control(cid, value);
    let current = cam.control(cid).unwrap_or(-1);
    match applied {
        Ok(()) => (true, String::new(), current),
        Err(e) => (false, e, current),
    }
}

/// The ZED runs auto exposure and exposes no exposure control over UVC; the
/// service reports that capability honestly instead of pretending to act.
fn spawn_set_color_exposure(runner: Arc<NodeRunner>) {
    tokio::spawn(async move {
        let cancel = runner.cancellation_token().clone();
        loop {
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = set_color_exposure::handle_next_request(&runner, |req| {
                    let accepted = req.data.mode == "auto";
                    let message = if accepted {
                        "the zed always runs auto exposure".to_string()
                    } else {
                        "manual exposure is not available on the zed over uvc".to_string()
                    };
                    Ok(set_color_exposure::Response::new(accepted, message, req.data.value))
                }) => result,
            };
            if let Err(e) = result {
                error!("set_color_exposure: {e}");
            }
        }
    });
}

fn spawn_set_color_white_balance(runner: Arc<NodeRunner>, camera: Camera) {
    tokio::spawn(async move {
        let cancel = runner.cancellation_token().clone();
        loop {
            let camera = camera.clone();
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = set_color_white_balance::handle_next_request(&runner, move |req| {
                    let response = match req.data.mode.as_str() {
                        "auto" => {
                            let (ok, message, _) = apply_control(&camera, CID_AWB_AUTO, 1);
                            let current = lock_camera(&camera)
                                .control(CID_AWB_TEMPERATURE)
                                .unwrap_or(-1);
                            set_color_white_balance::Response::new(ok, message, current)
                        }
                        "manual" => {
                            let (auto_off, message, _) =
                                apply_control(&camera, CID_AWB_AUTO, 0);
                            if auto_off {
                                let (ok, message, current) = apply_control(
                                    &camera,
                                    CID_AWB_TEMPERATURE,
                                    req.data.temperature,
                                );
                                set_color_white_balance::Response::new(ok, message, current)
                            } else {
                                set_color_white_balance::Response::new(
                                    false,
                                    message,
                                    req.data.temperature,
                                )
                            }
                        }
                        other => set_color_white_balance::Response::new(
                            false,
                            format!("mode must be auto|manual, got {other:?}"),
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

/// One service driving a single V4L2 control by CID; a non-empty ok_message
/// annotates successful writes (e.g. the device's value range).
macro_rules! spawn_cid_control_service {
    ($fn_name:ident, $service:ident, $cid:expr, $ok_message:expr) => {
        fn $fn_name(runner: Arc<NodeRunner>, camera: Camera) {
            tokio::spawn(async move {
                let cancel = runner.cancellation_token().clone();
                loop {
                    let camera = camera.clone();
                    let result = tokio::select! {
                        _ = cancel.cancelled() => break,
                        result = $service::handle_next_request(&runner, move |req| {
                            let (ok, message, current) =
                                apply_control(&camera, $cid, req.data.value);
                            let message = if ok { $ok_message.to_string() } else { message };
                            Ok($service::Response::new(ok, message, current))
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

spawn_cid_control_service!(
    spawn_set_color_gain,
    set_color_gain,
    CID_GAIN,
    "uvc gain (zed range 0..8)"
);
spawn_cid_control_service!(
    spawn_set_color_brightness,
    set_color_brightness,
    CID_BRIGHTNESS,
    ""
);
spawn_cid_control_service!(
    spawn_set_color_contrast,
    set_color_contrast,
    CID_CONTRAST,
    ""
);
