//! librealsense2 pipeline wrapper: opens a synchronized depth + color stream,
//! drives the capture loop, and exposes a thread-safe handle so peppy service
//! handlers can adjust sensor options at runtime without contending with the
//! capture thread.
//!
//! The capture loop runs on the caller's blocking thread; the
//! [`PipelineHandle`] is `Send + Sync` and holds mutexed [`Sensor`]s for the
//! color and depth halves so set_option calls from async tasks complete quickly
//!  without locking the pipeline itself.

use std::ffi::CStr;
use std::num::NonZeroU8;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use camera_core::controls::{AutoManualMode, ControlFailure, ControlFailureKind};
use camera_core::encoding::ColorEncoding;
use control_core::sync::lock_ignoring_poison;
use peppylib::runtime::CancellationToken;
use realsense_rust::{
    config::Config,
    context::Context,
    frame::{ColorFrame, DepthFrame, ImageFrame},
    kind::{OptionSetError, Rs2DistortionModel, Rs2Exception, Rs2Format, Rs2Option, Rs2StreamKind},
    pipeline::{ActivePipeline, FrameWaitError, InactivePipeline},
    processing_blocks::align::Align,
    sensor::Sensor,
    stream_profile::StreamProfile,
};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::frame::{FrameSet, Image};
use crate::geometry::{Calibration, Distortion, Published, StreamExtrinsics, StreamIntrinsics};
use crate::modes::{AlignMode, rs2_format};
use crate::stream::{Stream, StreamPair};

/// Now on this instance's bound clock: the timeline its messages are
/// stamped and aged on. Errors until the clock delivers its first tick.
fn clock_now() -> std::result::Result<SystemTime, String> {
    let ns = peppygen::clock::now_ns().map_err(|e| e.to_string())?;
    Ok(UNIX_EPOCH + Duration::from_nanos(ns))
}

/// Bounded pipeline depth between capture and emit. Deep enough to absorb a
/// transient emit hiccup, shallow enough that backpressure-dropped frames
/// match `sensor_data` QoS.
pub const EMIT_CHANNEL_CAPACITY: usize = 4;

/// D-series RealSense depth sensors only emit Z16, which consumers see on
/// the topic as [`camera_core::encoding::DEPTH_ENCODING`].
const DEPTH_RS2_FORMAT: Rs2Format = Rs2Format::Z16;

/// Bounds how long `Pipeline::wait` blocks before we re-check `cancel`.
/// A timeout just re-polls (no frame is lost), so any fps works. Kept small:
/// cancellation latency during shutdown is at most this value, and the
/// shutdown hook awaiting the capture loop needs most of the (default 3s)
/// grace window left over for `rs2_pipeline_stop` itself.
const WAIT_TIMEOUT: Duration = Duration::from_millis(500);

/// Frame queue depth for the librealsense2 Align processor.
const ALIGN_QUEUE_DEPTH: i32 = 1;

/// Bounds how long we wait for the Align processor to deliver a result.
/// Should be well under one frame interval; longer means alignment is
/// effectively broken and we drop the frame.
const ALIGN_WAIT: Duration = Duration::from_millis(200);

/// Heartbeat interval. At each tick, log either the latest captured frame
/// id (healthy) or a "no frame" warning (`Pipeline::wait` timing out is
/// otherwise silent, so a wedged camera looks identical to a healthy one
/// from the outside).
const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(10);

/// The size and frame rate one stream is opened at.
pub struct StreamSettings {
    pub width: u32,
    pub height: u32,
    pub fps: NonZeroU8,
}

/// Parsed pipeline configuration. Construct via plain struct literal; the
/// field types make invalid states unrepresentable. `NonZeroU8` rejects
/// `fps = 0` and over-u8 wrap at the type level, so there's no separate
/// `validate()`.
pub struct PipelineConfig {
    /// `None` selects the first device reported by `Context::query_devices`.
    pub serial: Option<String>,
    pub streams: StreamPair<StreamSettings>,
    pub color_format: ColorEncoding,
    /// The align mode the capture loop starts in.
    pub align_mode: AlignMode,
}

/// Owns the [`ActivePipeline`] and the Align processors. Consumed by
/// [`Capture::run`] which is intended to execute on a blocking thread.
pub struct Capture {
    pipeline: ActivePipeline,
    handle: Arc<PipelineHandle>,
}

/// The failure of a write of `option` that librealsense refused: rejected
/// when the device does not offer the option or does not take the value,
/// unavailable when the device is disconnected, and a failed attempt for any
/// other reason.
fn option_write_failure(option: Rs2Option, error: OptionSetError) -> ControlFailure {
    let kind = match &error {
        OptionSetError::OptionNotSupported
        | OptionSetError::OptionIsReadOnly
        | OptionSetError::CouldNotSetOption(
            Rs2Exception::InvalidValue | Rs2Exception::NotImplemented,
            _,
        ) => ControlFailureKind::Rejected,
        OptionSetError::CouldNotSetOption(Rs2Exception::CameraDisconnected, _) => {
            ControlFailureKind::ResourceUnavailable
        }
        OptionSetError::CouldNotSetOption(..) => ControlFailureKind::AttemptFailed,
    };
    ControlFailure {
        kind,
        message: format!("set {option:?}: {error}"),
    }
}

/// Handle for runtime control.
pub struct PipelineHandle {
    /// The color and depth sensors. On D405 both resolve to the *same*
    /// physical sensor, so a single mutex guards them together: `set_option`
    /// writes stay serialized even when the handles alias one device. Option
    /// writes are infrequent, so giving up color/depth concurrency costs
    /// nothing.
    sensors: Mutex<StreamPair<Sensor>>,
    align_mode: Mutex<AlignMode>,
    /// Meters per Z16 depth sample, read from the device at open(). Static for
    /// the session; surfaced to consumers via `depth_stream_info`.
    depth_unit: f32,
    /// The device's calibration for the two opened streams, read at open(),
    /// or why it could not be read. Static for the session; the geometry
    /// services answer from it without touching the device.
    calibration: Result<Calibration, String>,
}

impl Capture {
    pub fn handle(&self) -> Arc<PipelineHandle> {
        self.handle.clone()
    }

    /// Capture loop. Runs until `cancel.is_cancelled()` or an unrecoverable
    /// pipeline error. Frames are emitted on `tx` with backpressure: when the
    /// channel is full, the frame is dropped (matches `sensor_data` QoS).
    pub fn run(self, tx: mpsc::Sender<FrameSet>, cancel: CancellationToken) {
        let Capture {
            mut pipeline,
            handle,
        } = self;
        let mut align_to_color: Option<Align> = None;
        let mut align_to_depth: Option<Align> = None;
        let mut frame_id: u32 = 0;
        let mut last_status = Instant::now();
        let mut latest_id_this_interval: Option<u32> = None;

        info!("capture loop started");
        loop {
            if cancel.is_cancelled() {
                break;
            }

            // Heartbeat: at each interval, either acknowledge progress or
            // warn that the pipeline is silent (otherwise indistinguishable
            // from a healthy idle stream).
            if last_status.elapsed() >= STATUS_LOG_INTERVAL {
                match latest_id_this_interval {
                    Some(id) => info!("captured frame {id}"),
                    None => warn!(
                        "no frame in last {}s; capture appears stalled",
                        last_status.elapsed().as_secs(),
                    ),
                }
                last_status = Instant::now();
                latest_id_this_interval = None;
            }

            let frames = match pipeline.wait(Some(WAIT_TIMEOUT)) {
                Ok(f) => f,
                Err(FrameWaitError::DidTimeoutBeforeFrameArrival) => continue,
                Err(e) => {
                    error!("pipeline wait: {e}; stopping capture loop");
                    cancel.cancel();
                    break;
                }
            };

            let mode = *lock_ignoring_poison(&handle.align_mode);
            let processed = match mode {
                AlignMode::None => frames,
                AlignMode::DepthToColor => {
                    let aligner =
                        align_to_color.get_or_insert_with(|| Self::new_align(Rs2StreamKind::Color));
                    match Self::run_align(aligner, frames) {
                        Some(f) => f,
                        None => continue,
                    }
                }
                AlignMode::ColorToDepth => {
                    let aligner =
                        align_to_depth.get_or_insert_with(|| Self::new_align(Rs2StreamKind::Depth));
                    match Self::run_align(aligner, frames) {
                        Some(f) => f,
                        None => continue,
                    }
                }
            };

            let depth_frames: Vec<DepthFrame> = processed.frames_of_type();
            let color_frames: Vec<ColorFrame> = processed.frames_of_type();
            let (Some(depth), Some(color)) = (
                depth_frames.into_iter().next(),
                color_frames.into_iter().next(),
            ) else {
                warn!("incomplete frameset (depth or color missing)");
                continue;
            };

            let Ok(timestamp) = clock_now() else {
                continue; // clock not ready yet: skip rather than mis-stamp
            };
            let id = frame_id;
            frame_id = frame_id.wrapping_add(1);
            latest_id_this_interval = Some(id);

            let frameset = FrameSet {
                frame_id: id,
                timestamp,
                align_mode: mode,
                images: StreamPair {
                    color: Image {
                        bytes: frame_bytes(&color),
                        width: color.width() as u32,
                        height: color.height() as u32,
                    },
                    depth: Image {
                        bytes: frame_bytes(&depth),
                        width: depth.width() as u32,
                        height: depth.height() as u32,
                    },
                },
            };

            match tx.try_send(frameset) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    warn!("emit backlog full, dropping frame {id}");
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    error!("emit task closed; stopping capture loop");
                    cancel.cancel();
                    break;
                }
            }
        }
        info!("capture loop stopped");
        // Drops `pipeline` here, which calls `rs2_pipeline_stop` cleanly. The
        // shutdown hook that `node::setup` registers awaits this loop's
        // completion, so the stop runs inside the bounded hook phase.
    }

    fn new_align(target: Rs2StreamKind) -> Align {
        Align::new(target, ALIGN_QUEUE_DEPTH).expect("create Align processor")
    }

    fn run_align(
        aligner: &mut Align,
        frames: realsense_rust::frame::CompositeFrame,
    ) -> Option<realsense_rust::frame::CompositeFrame> {
        if let Err(e) = aligner.queue(frames) {
            warn!("align queue: {e}");
            return None;
        }
        match aligner.wait(ALIGN_WAIT) {
            Ok(f) => Some(f),
            Err(e) => {
                warn!("align wait: {e}");
                None
            }
        }
    }
}

/// What [`PipelineHandle::published_geometry`] hands the services: the
/// geometry under the current align mode, or why the calibration could not be
/// read.
pub type PublishedGeometry = Result<Published, String>;

impl PipelineHandle {
    pub fn set_align_mode(&self, mode: AlignMode) {
        *lock_ignoring_poison(&self.align_mode) = mode;
    }

    pub fn align_mode(&self) -> AlignMode {
        *lock_ignoring_poison(&self.align_mode)
    }

    pub fn depth_unit(&self) -> f32 {
        self.depth_unit
    }

    /// The geometry of the streams as the current align mode publishes them,
    /// or why the device's calibration could not be read. The mode is read
    /// here, once per answer, so every part of one answer describes the same
    /// mode.
    pub fn published_geometry(&self) -> PublishedGeometry {
        self.calibration
            .as_ref()
            .map(|calibration| calibration.published(self.align_mode()))
            .map_err(Clone::clone)
    }

    fn set_option(
        &self,
        stream: Stream,
        option: Rs2Option,
        value: f32,
    ) -> Result<(), ControlFailure> {
        let mut sensors = lock_ignoring_poison(&self.sensors);
        sensors
            .get_mut(stream)
            .set_option(option, value)
            .map_err(|error| option_write_failure(option, error))
    }

    /// `mode = Auto` sets `EnableAutoExposure` on; `mode = Manual` sets
    /// it off and writes `value` (microseconds) to `Exposure`.
    pub fn set_color_exposure(
        &self,
        mode: AutoManualMode,
        value: i32,
    ) -> Result<(), ControlFailure> {
        self.set_color_auto_manual(
            Rs2Option::EnableAutoExposure,
            Rs2Option::Exposure,
            mode,
            value,
        )
    }

    /// `mode = Auto` sets `EnableAutoWhiteBalance`; `mode = Manual` writes
    /// `temperature` (Kelvin) to `WhiteBalance`.
    pub fn set_color_white_balance(
        &self,
        mode: AutoManualMode,
        temperature: i32,
    ) -> Result<(), ControlFailure> {
        self.set_color_auto_manual(
            Rs2Option::EnableAutoWhiteBalance,
            Rs2Option::WhiteBalance,
            mode,
            temperature,
        )
    }

    /// `mode = Auto` sets the colour option `auto_option` on; `mode = Manual`
    /// sets it off and writes `value` to `setpoint_option`.
    fn set_color_auto_manual(
        &self,
        auto_option: Rs2Option,
        setpoint_option: Rs2Option,
        mode: AutoManualMode,
        value: i32,
    ) -> Result<(), ControlFailure> {
        if mode.is_auto() {
            return self.set_option(Stream::Color, auto_option, 1.0);
        }
        self.set_option(Stream::Color, auto_option, 0.0)?;
        self.set_option(Stream::Color, setpoint_option, value as f32)
    }

    pub fn set_color_gain(&self, value: i32) -> Result<(), ControlFailure> {
        self.set_option(Stream::Color, Rs2Option::Gain, value as f32)
    }

    pub fn set_color_brightness(&self, value: i32) -> Result<(), ControlFailure> {
        self.set_option(Stream::Color, Rs2Option::Brightness, value as f32)
    }

    pub fn set_color_contrast(&self, value: i32) -> Result<(), ControlFailure> {
        self.set_option(Stream::Color, Rs2Option::Contrast, value as f32)
    }

    pub fn set_depth_gain(&self, value: i32) -> Result<(), ControlFailure> {
        self.set_option(Stream::Depth, Rs2Option::Gain, value as f32)
    }

    pub fn set_depth_laser_power_mw(&self, value: i32) -> Result<(), ControlFailure> {
        self.set_option(Stream::Depth, Rs2Option::LaserPower, value as f32)
    }
}

/// Open the librealsense2 pipeline configured for synchronized Z16 depth +
/// `config.color_format` color streams. Resolves the device by serial (or
/// picks the first available if `serial` is `None`), locates the color and
/// depth sensors for the control plane, and returns a [`Capture`] that the
/// caller hands to a blocking thread.
pub fn open(config: PipelineConfig) -> Result<Capture, String> {
    let context = Context::new().map_err(|e| format!("realsense context: {e}"))?;
    let inactive =
        InactivePipeline::try_from(&context).map_err(|e| format!("create pipeline: {e}"))?;

    let mut rs2_config = Config::new();
    if let Some(serial) = &config.serial {
        let cstr = std::ffi::CString::new(serial.as_str())
            .map_err(|_| format!("serial contains NUL byte: {serial:?}"))?;
        rs2_config
            .enable_device_from_serial(cstr.as_c_str())
            .map_err(|e| format!("enable device by serial '{serial}': {e}"))?;
    }
    rs2_config
        .disable_all_streams()
        .map_err(|e| format!("disable_all_streams: {e}"))?
        .enable_stream(
            Rs2StreamKind::Depth,
            None,
            config.streams.depth.width as usize,
            config.streams.depth.height as usize,
            DEPTH_RS2_FORMAT,
            config.streams.depth.fps.get() as usize,
        )
        .map_err(|e| format!("enable depth stream: {e}"))?
        .enable_stream(
            Rs2StreamKind::Color,
            None,
            config.streams.color.width as usize,
            config.streams.color.height as usize,
            rs2_format(config.color_format),
            config.streams.color.fps.get() as usize,
        )
        .map_err(|e| format!("enable color stream: {e}"))?;

    let pipeline = inactive
        .start(Some(rs2_config))
        .map_err(|e| format!("start pipeline: {e}"))?;

    let device = pipeline.profile().device();
    let sensors = StreamPair::STREAMS.try_map(|stream| {
        let kind = rs2_kind(stream);
        find_sensor_for_stream(device, kind)
            .ok_or_else(|| format!("device has no sensor exposing the {kind:?} stream"))
    })?;

    // Metric depth scale (meters per Z16 LSB). Static per device/config; read
    // once here so depth_stream_info can report it without touching the sensor.
    let depth_unit = sensors
        .depth
        .get_option(Rs2Option::DepthUnits)
        .unwrap_or_else(|| {
            warn!("depth sensor did not report DepthUnits; defaulting to 0.001 m/LSB");
            0.001
        });

    let serial_log = device_serial(device).unwrap_or_else(|| "<unknown>".into());
    info!(
        "pipeline started for device serial={serial_log} color={}x{}@{}fps depth={}x{}@{}fps depth_unit={depth_unit}",
        config.streams.color.width,
        config.streams.color.height,
        config.streams.color.fps,
        config.streams.depth.width,
        config.streams.depth.height,
        config.streams.depth.fps,
    );

    let calibration = read_calibration(pipeline.profile().streams(), depth_unit);
    match &calibration {
        Ok(c) => info!(
            "calibration: colour {}x{} fx {} fy {} ppx {} ppy {} {:?}, depth {}x{} fx {} fy {} ppx {} ppy {} {:?}, depth to colour t {:?}",
            c.streams.color.width,
            c.streams.color.height,
            c.streams.color.fx,
            c.streams.color.fy,
            c.streams.color.ppx,
            c.streams.color.ppy,
            c.streams.color.distortion,
            c.streams.depth.width,
            c.streams.depth.height,
            c.streams.depth.fx,
            c.streams.depth.fy,
            c.streams.depth.ppx,
            c.streams.depth.ppy,
            c.streams.depth.distortion,
            c.depth_to_color.translation,
        ),
        Err(e) => warn!("no calibration read, the geometry services refuse: {e}"),
    }

    let handle = Arc::new(PipelineHandle {
        sensors: Mutex::new(sensors),
        align_mode: Mutex::new(config.align_mode),
        depth_unit,
        calibration,
    });

    Ok(Capture { pipeline, handle })
}

/// Copy the device's calibration for the opened colour and depth streams out
/// of librealsense's types. The streams do not depend on it, so a read that
/// fails costs the geometry services their answer, not the node its frames.
fn read_calibration(streams: &[StreamProfile], depth_unit: f32) -> Result<Calibration, String> {
    let profiles = StreamPair::STREAMS.try_map(|stream| {
        let kind = rs2_kind(stream);
        streams
            .iter()
            .find(|profile| profile.kind() == kind)
            .ok_or_else(|| format!("the started pipeline has no {kind:?} stream profile"))
    })?;
    let depth_to_color = profiles
        .depth
        .extrinsics(profiles.color)
        .map_err(|e| format!("read the depth-to-colour extrinsics: {e}"))?;
    Ok(Calibration {
        streams: profiles.try_map(read_intrinsics)?,
        depth_to_color: StreamExtrinsics {
            rotation: depth_to_color.rotation(),
            translation: depth_to_color.translation(),
        },
        depth_unit,
    })
}

fn read_intrinsics(profile: &StreamProfile) -> Result<StreamIntrinsics, String> {
    let intrinsics = profile
        .intrinsics()
        .map_err(|e| format!("read the {:?} intrinsics: {e}", profile.kind()))?;
    Ok(StreamIntrinsics {
        width: intrinsics.width() as u32,
        height: intrinsics.height() as u32,
        fx: intrinsics.fx(),
        fy: intrinsics.fy(),
        ppx: intrinsics.ppx(),
        ppy: intrinsics.ppy(),
        // Read from the raw value: `Rs2Intrinsics::distortion` unwraps the
        // model, and a value a newer librealsense adds would panic there.
        distortion: distortion_of(intrinsics.0.model as i32),
        coeffs: intrinsics.0.coeffs,
    })
}

/// librealsense's distortion model value under this node's own name for it.
fn distortion_of(model: i32) -> Distortion {
    const NONE: i32 = Rs2DistortionModel::None as i32;
    const BROWN_CONRADY: i32 = Rs2DistortionModel::BrownConrady as i32;
    const MODIFIED: i32 = Rs2DistortionModel::BrownConradyModified as i32;
    const INVERSE: i32 = Rs2DistortionModel::BrownConradyInverse as i32;
    const FTHETA: i32 = Rs2DistortionModel::FThetaFisheye as i32;
    const KANNALA_BRANDT: i32 = Rs2DistortionModel::KannalaBrandt as i32;
    match model {
        NONE => Distortion::None,
        BROWN_CONRADY => Distortion::BrownConrady,
        MODIFIED => Distortion::ModifiedBrownConrady,
        INVERSE => Distortion::InverseBrownConrady,
        FTHETA => Distortion::FTheta,
        KANNALA_BRANDT => Distortion::KannalaBrandt4,
        other => Distortion::Unknown(other),
    }
}

/// Bulk-copy a librealsense2 [`ImageFrame`]'s raw byte buffer into an owned
/// `Vec<u8>` so the emit task can outlive any single frame. Used for both
/// color and depth frames; bytes pass through unchanged and downstream
/// consumers decode according to the topic's `encoding` field. The
/// alternative `ImageFrame::iter()` path costs an enum dispatch + bounds
/// check per pixel and is measurably slower.
fn frame_bytes<K>(frame: &ImageFrame<K>) -> Vec<u8> {
    let size = frame.get_data_size();
    if size == 0 {
        return Vec::new();
    }
    // SAFETY: `ImageFrame::get_data()` returns a `&c_void` aimed at the
    // librealsense2-owned frame buffer; it is valid for `get_data_size()`
    // bytes for the lifetime of `frame`. The first cast (`as *const _`)
    // demotes the reference to a raw `*const c_void`; the second
    // (`as *const u8`) reinterprets it as a byte pointer. Rust does not
    // permit fusing these. We immediately copy into an owned `Vec<u8>`,
    // so no reference outlives the frame.
    unsafe {
        let data = frame.get_data() as *const _ as *const u8;
        std::slice::from_raw_parts(data, size).to_vec()
    }
}

/// The librealsense kind of `stream`.
fn rs2_kind(stream: Stream) -> Rs2StreamKind {
    match stream {
        Stream::Color => Rs2StreamKind::Color,
        Stream::Depth => Rs2StreamKind::Depth,
    }
}

/// Return the first sensor on `device` that advertises a stream profile of
/// `kind`. On D435/D455 this distinguishes the color and depth sensors;
/// on D405 the same sensor satisfies both lookups.
fn find_sensor_for_stream(
    device: &realsense_rust::device::Device,
    kind: Rs2StreamKind,
) -> Option<Sensor> {
    device
        .sensors()
        .into_iter()
        .find(|s| s.stream_profiles().iter().any(|p| p.kind() == kind))
}

fn device_serial(device: &realsense_rust::device::Device) -> Option<String> {
    use realsense_rust::kind::Rs2CameraInfo;
    device
        .info(Rs2CameraInfo::SerialNumber)
        .and_then(|c: &CStr| c.to_str().ok().map(str::to_owned))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The depth stream opens in librealsense's Z16 format.
    #[test]
    fn the_depth_stream_opens_in_z16() {
        assert_eq!(DEPTH_RS2_FORMAT, Rs2Format::Z16);
    }

    /// Every model librealsense names maps to this node's name for it, and a
    /// value it does not name is kept as it came instead of panicking.
    #[test]
    fn every_sdk_distortion_model_has_a_name_here() {
        assert_eq!(
            distortion_of(Rs2DistortionModel::None as i32),
            Distortion::None
        );
        assert_eq!(
            distortion_of(Rs2DistortionModel::BrownConrady as i32),
            Distortion::BrownConrady
        );
        assert_eq!(
            distortion_of(Rs2DistortionModel::BrownConradyModified as i32),
            Distortion::ModifiedBrownConrady
        );
        assert_eq!(
            distortion_of(Rs2DistortionModel::BrownConradyInverse as i32),
            Distortion::InverseBrownConrady
        );
        assert_eq!(
            distortion_of(Rs2DistortionModel::FThetaFisheye as i32),
            Distortion::FTheta
        );
        assert_eq!(
            distortion_of(Rs2DistortionModel::KannalaBrandt as i32),
            Distortion::KannalaBrandt4
        );
        assert_eq!(distortion_of(9001), Distortion::Unknown(9001));
    }

    /// A device that does not offer the option or the value refuses it, a
    /// device that is gone is unavailable, and any other refusal of
    /// librealsense is a failed attempt.
    #[test]
    fn each_refused_option_write_gives_the_kind_of_its_cause() {
        let kind_of = |error| option_write_failure(Rs2Option::Gain, error).kind;
        let sdk = |exception| OptionSetError::CouldNotSetOption(exception, "why".to_string());
        assert_eq!(
            kind_of(OptionSetError::OptionNotSupported),
            ControlFailureKind::Rejected
        );
        assert_eq!(
            kind_of(OptionSetError::OptionIsReadOnly),
            ControlFailureKind::Rejected
        );
        assert_eq!(
            kind_of(sdk(Rs2Exception::InvalidValue)),
            ControlFailureKind::Rejected
        );
        assert_eq!(
            kind_of(sdk(Rs2Exception::NotImplemented)),
            ControlFailureKind::Rejected
        );
        assert_eq!(
            kind_of(sdk(Rs2Exception::CameraDisconnected)),
            ControlFailureKind::ResourceUnavailable
        );
        for exception in [
            Rs2Exception::Unknown,
            Rs2Exception::Backend,
            Rs2Exception::WrongApiCallSequence,
            Rs2Exception::DeviceInRecoveryMode,
            Rs2Exception::IoDeviceFailure,
        ] {
            assert_eq!(
                kind_of(sdk(exception)),
                ControlFailureKind::AttemptFailed,
                "{exception}"
            );
        }
    }

    #[test]
    fn a_refused_option_write_names_the_option_and_the_reason() {
        let failure = option_write_failure(
            Rs2Option::LaserPower,
            OptionSetError::CouldNotSetOption(
                Rs2Exception::InvalidValue,
                "value out of range".to_string(),
            ),
        );
        assert!(
            failure.message.contains("LaserPower"),
            "{}",
            failure.message
        );
        assert!(
            failure.message.contains("value out of range"),
            "{}",
            failure.message
        );
    }
}
