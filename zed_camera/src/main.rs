//! Stereolabs ZED as an rgbd_camera producer without the ZED SDK.
//!
//! A blocking pipeline thread owns the frame path: grab side-by-side YUYV
//! through the library's capture layer, rectify through the per-serial factory
//! calibration fetched at startup, publish the rectified left eye as video_stream and
//! stereo-matched depth (millimeters, z16) as depth_stream. Depth lives in
//! the rectified-left frame, so the pair is permanently color-aligned. The
//! pipeline loop and the control services share one capture handle behind a
//! mutex, so a control call waits at most one frame. camera_geometry answers
//! from the same rectification, computed once when the pipeline opens.

#[cfg(not(target_os = "linux"))]
compile_error!("zed_camera captures over V4L2 and builds for Linux only");

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use peppygen::emitted_topics::camera::{depth_stream, video_stream};
use peppygen::exposed_services::camera::{
    depth_stream_info, set_color_brightness, set_color_contrast, set_color_exposure,
    set_color_gain, set_color_white_balance, video_stream_info,
};
use peppygen::exposed_services::geometry::{
    get_color_intrinsics, get_depth_intrinsics, get_depth_to_color_extrinsics,
};
use peppygen::{NodeBuilder, NodeRunner, Parameters, Result};
use peppylib::runtime::CancellationToken;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, warn};
use zed_camera::calibration::fetch_conf;
use zed_camera::capture::{
    CID_AWB_AUTO, CID_AWB_TEMPERATURE, CID_BRIGHTNESS, CID_CONTRAST, CID_GAIN, Capture,
    ControlFailure, ControlFailureKind, Grab, device_index, zed_serial,
};
use zed_camera::cv_depth::CvDepth;
use zed_camera::geometry::{Geometry, Pinhole};
use zed_camera::{DepthSettings, Resolution};

const GRAB_TIMEOUT: Duration = Duration::from_millis(500);
const EMIT_CHANNEL_CAPACITY: usize = 2;
const DEPTH_UNIT_M_PER_LSB: f32 = 0.001;
const COLOR_ENCODING: &str = "rgb8";
const DEPTH_ENCODING: &str = "z16";
/// Depth is computed in the rectified-left (= published color) frame.
const ALIGN_MODE: &str = "depth_to_color";
/// Both streams are rectified images: camera_geometry's ideal pinhole.
const DISTORTION_MODEL: &str = "none";
/// SGBM disparity becomes the distance along the optical axis.
const DEPTH_MODEL: &str = "z";

/// The capture device shared between the pipeline loop and the control
/// services; every access is a short lock.
type Camera = Arc<Mutex<Capture>>;

struct PipelineConfig {
    dev_id: usize,
    resolution: Resolution,
    fps: u32,
    depth: DepthSettings,
}

/// What the pipeline thread reports back once the camera and matcher are up.
struct Opened {
    camera: Camera,
    eye_width: u32,
    eye_height: u32,
    depth_width: u32,
    depth_height: u32,
    geometry: SessionGeometry,
}

/// The geometry of the published streams, or why there is none to answer.
/// The streams do not depend on it, so a rectification that yields numbers
/// the contract cannot carry costs the geometry services their answer, not
/// the node its frames.
type SessionGeometry = std::result::Result<Geometry, String>;

/// One processed capture: rectified left RGB plus depth, sharing a frame id.
struct FrameSet {
    frame_id: u32,
    timestamp: SystemTime,
    left_rgb: Vec<u8>,
    depth_z16: Vec<u8>,
}

fn timestamp_now() -> std::result::Result<SystemTime, String> {
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
            geometry,
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
        let geometry = Arc::new(geometry);
        spawn_get_color_intrinsics(node_runner.clone(), geometry.clone());
        spawn_get_depth_intrinsics(node_runner.clone(), geometry.clone());
        spawn_get_depth_to_color_extrinsics(node_runner.clone(), geometry);
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
        let Ok(timestamp) = timestamp_now() else {
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
            timestamp,
            left_rgb: left_rgb.clone(),
            depth_z16: depth_mm.iter().flat_map(|d| d.to_le_bytes()).collect(),
        };
        frame_id = frame_id.wrapping_add(1);
        // Latest-value semantics: drop the frame if the emitter is behind.
        let _ = frame_tx.try_send(frame);
    }
}

fn open_pipeline(config: &PipelineConfig) -> std::result::Result<(CvDepth, Opened), String> {
    // The serial keys the factory calibration, fetched fresh per startup.
    let serial = zed_serial()?;
    info!("zed unit serial {serial}");
    let conf_text = fetch_conf(serial)?;

    let capture = Capture::open(config.dev_id, config.resolution, config.fps)?;
    let (full_width, height) = capture.frame_size();
    let (eye_width, eye_height) = (full_width / 2, height);

    let matcher = CvDepth::create(&conf_text, eye_width, eye_height, config.depth)?;
    let (depth_width, depth_height) = matcher.out_size();
    let geometry = Geometry::new(
        (eye_width, eye_height),
        matcher.rectified_left(),
        (depth_width, depth_height),
        matcher.min_depth_floor_mm(),
        matcher.max_depth_mm(),
    );
    match &geometry {
        Ok(g) => info!(
            "geometry: colour fx {:.3} fy {:.3} cx {:.3} cy {:.3}, depth fx {:.3} fy {:.3} cx {:.3} cy {:.3}, {:.3} m to {:.3} m",
            g.color.fx,
            g.color.fy,
            g.color.cx,
            g.color.cy,
            g.depth.fx,
            g.depth.fy,
            g.depth.cx,
            g.depth.cy,
            g.min_depth_m,
            g.max_depth_m,
        ),
        Err(e) => warn!("no geometry to answer, the geometry services refuse: {e}"),
    }

    Ok((
        matcher,
        Opened {
            camera: Arc::new(Mutex::new(capture)),
            eye_width,
            eye_height,
            depth_width,
            depth_height,
            geometry,
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
                timestamp,
                left_rgb,
                depth_z16,
            } = frame;
            let color_header = video_stream::MessageHeader {
                timestamp,
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
                timestamp,
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

/// get_color_intrinsics: the rectified left projection at the eye size. A
/// session without a geometry gives an error reply with `rejected`, as do
/// the other two geometry services.
fn color_intrinsics(
    geometry: &SessionGeometry,
) -> std::result::Result<get_color_intrinsics::Response, get_color_intrinsics::Failure> {
    let g = geometry.as_ref().map_err(|reason| {
        get_color_intrinsics::Failure::new(
            get_color_intrinsics::FailureKind::Rejected,
            reason.clone(),
        )
    })?;
    let Pinhole {
        width,
        height,
        fx,
        fy,
        cx,
        cy,
    } = g.color;
    Ok(get_color_intrinsics::Response::new(
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        DISTORTION_MODEL.to_string(),
        Vec::new(),
    ))
}

/// get_depth_intrinsics: the same camera through the depth grid, and what a
/// sample measures.
fn depth_intrinsics(
    geometry: &SessionGeometry,
) -> std::result::Result<get_depth_intrinsics::Response, get_depth_intrinsics::Failure> {
    let g = geometry.as_ref().map_err(|reason| {
        get_depth_intrinsics::Failure::new(
            get_depth_intrinsics::FailureKind::Rejected,
            reason.clone(),
        )
    })?;
    let Pinhole {
        width,
        height,
        fx,
        fy,
        cx,
        cy,
    } = g.depth;
    Ok(get_depth_intrinsics::Response::new(
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        DISTORTION_MODEL.to_string(),
        Vec::new(),
        DEPTH_MODEL.to_string(),
        g.min_depth_m,
        g.max_depth_m,
        ALIGN_MODE.to_string(),
    ))
}

/// get_depth_to_color_extrinsics: depth is matched in the colour stream's own
/// view, so the two optical frames are one and the transform is the identity.
fn depth_to_color_extrinsics(
    geometry: &SessionGeometry,
) -> std::result::Result<
    get_depth_to_color_extrinsics::Response,
    get_depth_to_color_extrinsics::Failure,
> {
    geometry.as_ref().map_err(|reason| {
        get_depth_to_color_extrinsics::Failure::new(
            get_depth_to_color_extrinsics::FailureKind::Rejected,
            reason.clone(),
        )
    })?;
    Ok(get_depth_to_color_extrinsics::Response::new(
        ALIGN_MODE.to_string(),
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ))
}

/// One camera_geometry service answering from the session's geometry.
macro_rules! spawn_geometry_service {
    ($fn_name:ident, $service:ident, $respond:ident) => {
        fn $fn_name(runner: Arc<NodeRunner>, geometry: Arc<SessionGeometry>) {
            tokio::spawn(async move {
                let cancel = runner.cancellation_token().clone();
                loop {
                    let result = tokio::select! {
                        _ = cancel.cancelled() => break,
                        result = $service::handle_next_request(&runner, |_req| {
                            $respond(&geometry)
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

fn lock_camera(camera: &Camera) -> std::sync::MutexGuard<'_, Capture> {
    camera
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Writes one control and reads it back: the value the driver kept, or why
/// the write gave none.
fn apply_control(
    camera: &Camera,
    cid: u32,
    value: i32,
) -> std::result::Result<i32, ControlFailure> {
    let cam = lock_camera(camera);
    cam.set_control(cid, value)?;
    Ok(cam.control(cid).unwrap_or(-1))
}

/// The error reply of `$service` for a control that gave no value: the same
/// kind, from the service's own kinds.
macro_rules! control_failure {
    ($service:ident, $failure:expr) => {{
        let ControlFailure { kind, message } = $failure;
        let kind = match kind {
            ControlFailureKind::Rejected => $service::FailureKind::Rejected,
            ControlFailureKind::PreconditionFailed => $service::FailureKind::PreconditionFailed,
            ControlFailureKind::ResourceUnavailable => $service::FailureKind::ResourceUnavailable,
            ControlFailureKind::AttemptFailed => $service::FailureKind::AttemptFailed,
        };
        $service::Failure::new(kind, message)
    }};
}

/// The ZED runs auto exposure and exposes no exposure control over UVC: the
/// service answers "auto" with the value of the request, and gives an error
/// reply with `rejected` for any other mode.
fn color_exposure(
    req: set_color_exposure::Request,
) -> std::result::Result<set_color_exposure::Response, set_color_exposure::Failure> {
    let refuse = |message: String| {
        set_color_exposure::Failure::new(set_color_exposure::FailureKind::Rejected, message)
    };
    match req.data.mode.as_str() {
        "auto" => Ok(set_color_exposure::Response::new(req.data.value)),
        "manual" => Err(refuse(
            "manual exposure is not available on the zed over uvc".to_string(),
        )),
        other => Err(refuse(format!("mode must be auto|manual, got {other:?}"))),
    }
}

fn spawn_set_color_exposure(runner: Arc<NodeRunner>) {
    tokio::spawn(async move {
        let cancel = runner.cancellation_token().clone();
        loop {
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = set_color_exposure::handle_next_request(&runner, color_exposure) => result,
            };
            if let Err(e) = result {
                error!("set_color_exposure: {e}");
            }
        }
    });
}

/// "auto" turns auto white balance on and answers the temperature it runs
/// at; "manual" turns it off and writes the temperature of the request.
fn color_white_balance(
    camera: &Camera,
    mode: &str,
    temperature: i32,
) -> std::result::Result<i32, ControlFailure> {
    match mode {
        "auto" => {
            apply_control(camera, CID_AWB_AUTO, 1)?;
            Ok(lock_camera(camera)
                .control(CID_AWB_TEMPERATURE)
                .unwrap_or(-1))
        }
        "manual" => {
            apply_control(camera, CID_AWB_AUTO, 0)?;
            apply_control(camera, CID_AWB_TEMPERATURE, temperature)
        }
        other => Err(ControlFailure::rejected(format!(
            "mode must be auto|manual, got {other:?}"
        ))),
    }
}

fn spawn_set_color_white_balance(runner: Arc<NodeRunner>, camera: Camera) {
    tokio::spawn(async move {
        let cancel = runner.cancellation_token().clone();
        loop {
            let camera = camera.clone();
            let result = tokio::select! {
                _ = cancel.cancelled() => break,
                result = set_color_white_balance::handle_next_request(&runner, move |req| {
                    color_white_balance(&camera, &req.data.mode, req.data.temperature)
                        .map(set_color_white_balance::Response::new)
                        .map_err(|failure| control_failure!(set_color_white_balance, failure))
                }) => result,
            };
            if let Err(e) = result {
                error!("set_color_white_balance: {e}");
            }
        }
    });
}

/// One service driving a single V4L2 control by CID.
macro_rules! spawn_cid_control_service {
    ($fn_name:ident, $service:ident, $cid:expr) => {
        fn $fn_name(runner: Arc<NodeRunner>, camera: Camera) {
            tokio::spawn(async move {
                let cancel = runner.cancellation_token().clone();
                loop {
                    let camera = camera.clone();
                    let result = tokio::select! {
                        _ = cancel.cancelled() => break,
                        result = $service::handle_next_request(&runner, move |req| {
                            apply_control(&camera, $cid, req.data.value)
                                .map($service::Response::new)
                                .map_err(|failure| control_failure!($service, failure))
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

// The ZED takes a UVC gain of 0..8.
spawn_cid_control_service!(spawn_set_color_gain, set_color_gain, CID_GAIN);
spawn_cid_control_service!(
    spawn_set_color_brightness,
    set_color_brightness,
    CID_BRIGHTNESS
);
spawn_cid_control_service!(spawn_set_color_contrast, set_color_contrast, CID_CONTRAST);

#[cfg(test)]
mod tests {
    use super::*;
    use zed_camera::geometry::RectifiedLeft;

    fn session() -> SessionGeometry {
        Geometry::new(
            (1280, 720),
            RectifiedLeft {
                fx: 700.25,
                fy: 700.75,
                cx: 652.125,
                cy: 351.5,
            },
            (640, 360),
            98.5,
            65534.0,
        )
    }

    #[test]
    fn each_answer_reads_its_own_stream_and_names_the_frames_alignment() {
        let color = color_intrinsics(&session()).unwrap();
        assert_eq!((color.width, color.height), (1280, 720));
        assert_eq!(
            (color.fx, color.fy, color.cx, color.cy),
            (700.25, 700.75, 652.125, 351.5)
        );
        assert_eq!(color.distortion_model, "none");
        assert!(color.distortion.is_empty());

        let depth = depth_intrinsics(&session()).unwrap();
        assert_eq!((depth.width, depth.height), (640, 360));
        assert_eq!((depth.fx, depth.cx), (350.125, 325.8125));
        assert_eq!(depth.depth_model, "z");
        assert_eq!((depth.min_depth_m, depth.max_depth_m), (0.0985, 65.534));
        // The value every frame header carries.
        assert_eq!(depth.align_mode, ALIGN_MODE);

        let pose = depth_to_color_extrinsics(&session()).unwrap();
        assert_eq!(pose.align_mode, depth.align_mode);
        assert_eq!(pose.depth_to_color_position, [0.0, 0.0, 0.0]);
        assert_eq!(pose.depth_to_color_orientation, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn without_a_geometry_every_answer_refuses_with_the_reason() {
        let none: SessionGeometry =
            Err("the rectified focal lengths are not positive numbers".into());
        let color = color_intrinsics(&none).unwrap_err();
        assert_eq!(color.kind, get_color_intrinsics::FailureKind::Rejected);
        assert!(color.message.contains("focal lengths"));

        let depth = depth_intrinsics(&none).unwrap_err();
        assert_eq!(depth.kind, get_depth_intrinsics::FailureKind::Rejected);
        assert!(depth.message.contains("focal lengths"));

        let pose = depth_to_color_extrinsics(&none).unwrap_err();
        assert_eq!(
            pose.kind,
            get_depth_to_color_extrinsics::FailureKind::Rejected
        );
        assert!(pose.message.contains("focal lengths"));
    }

    fn exposure_request(mode: &str, value: i32) -> set_color_exposure::Request {
        set_color_exposure::Request {
            instance_id: "caller".to_string(),
            core_node: "core".to_string(),
            data: set_color_exposure::RequestData {
                mode: mode.to_string(),
                value,
            },
        }
    }

    #[test]
    fn auto_exposure_is_answered_with_the_value_of_the_request() {
        let response = color_exposure(exposure_request("auto", 120)).unwrap();
        assert_eq!(response.current_value, 120);
    }

    #[test]
    fn manual_exposure_and_an_unknown_mode_are_rejected() {
        let manual = color_exposure(exposure_request("manual", 120)).unwrap_err();
        assert_eq!(manual.kind, set_color_exposure::FailureKind::Rejected);
        assert!(
            manual.message.contains("not available"),
            "{}",
            manual.message
        );
        let unknown = color_exposure(exposure_request("dim", 120)).unwrap_err();
        assert_eq!(unknown.kind, set_color_exposure::FailureKind::Rejected);
        assert!(unknown.message.contains("\"dim\""), "{}", unknown.message);
    }
}
