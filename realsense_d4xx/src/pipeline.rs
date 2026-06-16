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
use std::time::{Duration, Instant, SystemTime};

use peppylib::runtime::CancellationToken;
use realsense_rust::{
    config::Config,
    context::Context,
    frame::{ColorFrame, DepthFrame, ImageFrame},
    kind::{Rs2Format, Rs2Option, Rs2StreamKind},
    pipeline::{ActivePipeline, FrameWaitError, InactivePipeline},
    processing_blocks::align::Align,
    sensor::Sensor,
};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::frame::{FrameSet, Image};
use crate::modes::{AlignMode, AutoManualMode, ColorFormat};

/// Bounded pipeline depth between capture and emit. Deep enough to absorb a
/// transient emit hiccup, shallow enough that backpressure-dropped frames
/// match `sensor_data` QoS.
pub const EMIT_CHANNEL_CAPACITY: usize = 4;

/// D-series RealSense depth sensors only emit Z16. `DEPTH_RS2_FORMAT` is
/// internal to `open()`; `DEPTH_TOPIC_ENCODING` is the matching wire
/// encoding string consumers see on the topic.
const DEPTH_RS2_FORMAT: Rs2Format = Rs2Format::Z16;
pub const DEPTH_TOPIC_ENCODING: &str = "z16";

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

/// Parsed pipeline configuration. Construct via plain struct literal; the
/// field types make invalid states unrepresentable. `NonZeroU8` rejects
/// `fps = 0` and over-u8 wrap at the type level, so there's no separate
/// `validate()`.
pub struct PipelineConfig {
    /// `None` selects the first device reported by `Context::query_devices`.
    pub serial: Option<String>,
    pub color_width: u32,
    pub color_height: u32,
    pub color_fps: NonZeroU8,
    pub color_format: ColorFormat,
    pub depth_width: u32,
    pub depth_height: u32,
    pub depth_fps: NonZeroU8,
}

/// Owns the [`ActivePipeline`] and the Align processors. Consumed by
/// [`Capture::run`] which is intended to execute on a blocking thread.
pub struct Capture {
    pipeline: ActivePipeline,
    handle: Arc<PipelineHandle>,
}

/// The color and depth sensors. On D405 both resolve to the *same* physical
/// sensor, so a single mutex guards them together: `set_option` writes stay
/// serialized even when the handles alias one device. Option writes are
/// infrequent, so giving up color/depth concurrency costs nothing.
struct Sensors {
    color: Sensor,
    depth: Sensor,
}

/// Handle for runtime control.
pub struct PipelineHandle {
    sensors: Mutex<Sensors>,
    align_mode: Mutex<AlignMode>,
    /// Meters per Z16 depth sample, read from the device at open(). Static for
    /// the session; surfaced to consumers via `depth_stream_info`.
    depth_unit: f32,
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

            let mode = *handle.align_mode.lock().unwrap_or_else(|p| p.into_inner());
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

            let id = frame_id;
            frame_id = frame_id.wrapping_add(1);
            let stamp = SystemTime::now();
            latest_id_this_interval = Some(id);

            let frameset = FrameSet {
                frame_id: id,
                stamp,
                align_mode: mode,
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
        // shutdown hook registered in main awaits this loop's completion, so
        // the stop runs inside the bounded hook phase.
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

impl PipelineHandle {
    pub fn set_align_mode(&self, mode: AlignMode) {
        *self.align_mode.lock().unwrap_or_else(|p| p.into_inner()) = mode;
    }

    pub fn align_mode(&self) -> AlignMode {
        *self.align_mode.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn depth_unit(&self) -> f32 {
        self.depth_unit
    }

    fn set_color_option(&self, option: Rs2Option, value: f32) -> Result<(), String> {
        let mut sensors = self.sensors.lock().unwrap_or_else(|p| p.into_inner());
        sensors
            .color
            .set_option(option, value)
            .map_err(|e| format!("{e}"))
    }

    fn set_depth_option(&self, option: Rs2Option, value: f32) -> Result<(), String> {
        let mut sensors = self.sensors.lock().unwrap_or_else(|p| p.into_inner());
        sensors
            .depth
            .set_option(option, value)
            .map_err(|e| format!("{e}"))
    }

    /// `mode = Auto` sets `EnableAutoExposure` on; `mode = Manual` sets
    /// it off and writes `value` (microseconds) to `Exposure`.
    pub fn set_color_exposure(&self, mode: AutoManualMode, value: i32) -> Result<(), String> {
        self.set_color_option(
            Rs2Option::EnableAutoExposure,
            if mode.is_auto() { 1.0 } else { 0.0 },
        )?;
        if !mode.is_auto() {
            self.set_color_option(Rs2Option::Exposure, value as f32)?;
        }
        Ok(())
    }

    /// `mode = Auto` sets `EnableAutoWhiteBalance`; `mode = Manual` writes
    /// `temperature` (Kelvin) to `WhiteBalance`.
    pub fn set_color_white_balance(
        &self,
        mode: AutoManualMode,
        temperature: i32,
    ) -> Result<(), String> {
        self.set_color_option(
            Rs2Option::EnableAutoWhiteBalance,
            if mode.is_auto() { 1.0 } else { 0.0 },
        )?;
        if !mode.is_auto() {
            self.set_color_option(Rs2Option::WhiteBalance, temperature as f32)?;
        }
        Ok(())
    }

    pub fn set_color_gain(&self, value: i32) -> Result<(), String> {
        self.set_color_option(Rs2Option::Gain, value as f32)
    }

    pub fn set_color_brightness(&self, value: i32) -> Result<(), String> {
        self.set_color_option(Rs2Option::Brightness, value as f32)
    }

    pub fn set_color_contrast(&self, value: i32) -> Result<(), String> {
        self.set_color_option(Rs2Option::Contrast, value as f32)
    }

    pub fn set_depth_gain(&self, value: i32) -> Result<(), String> {
        self.set_depth_option(Rs2Option::Gain, value as f32)
    }

    pub fn set_depth_laser_power_mw(&self, value: i32) -> Result<(), String> {
        self.set_depth_option(Rs2Option::LaserPower, value as f32)
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
            config.depth_width as usize,
            config.depth_height as usize,
            DEPTH_RS2_FORMAT,
            config.depth_fps.get() as usize,
        )
        .map_err(|e| format!("enable depth stream: {e}"))?
        .enable_stream(
            Rs2StreamKind::Color,
            None,
            config.color_width as usize,
            config.color_height as usize,
            config.color_format.rs2_format(),
            config.color_fps.get() as usize,
        )
        .map_err(|e| format!("enable color stream: {e}"))?;

    let pipeline = inactive
        .start(Some(rs2_config))
        .map_err(|e| format!("start pipeline: {e}"))?;

    let device = pipeline.profile().device();
    let color_sensor = find_sensor_for_stream(&device, Rs2StreamKind::Color)
        .ok_or("device has no sensor exposing the Color stream")?;
    let depth_sensor = find_sensor_for_stream(&device, Rs2StreamKind::Depth)
        .ok_or("device has no sensor exposing the Depth stream")?;

    // Metric depth scale (meters per Z16 LSB). Static per device/config; read
    // once here so depth_stream_info can report it without touching the sensor.
    let depth_unit = depth_sensor
        .get_option(Rs2Option::DepthUnits)
        .unwrap_or_else(|| {
            warn!("depth sensor did not report DepthUnits; defaulting to 0.001 m/LSB");
            0.001
        });

    let serial_log = device_serial(&device).unwrap_or_else(|| "<unknown>".into());
    info!(
        "pipeline started for device serial={serial_log} color={}x{}@{}fps depth={}x{}@{}fps depth_unit={depth_unit}",
        config.color_width,
        config.color_height,
        config.color_fps,
        config.depth_width,
        config.depth_height,
        config.depth_fps,
    );

    let handle = Arc::new(PipelineHandle {
        sensors: Mutex::new(Sensors {
            color: color_sensor,
            depth: depth_sensor,
        }),
        align_mode: Mutex::new(AlignMode::None),
        depth_unit,
    });

    Ok(Capture { pipeline, handle })
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

    /// Tripwire: if anyone edits the literal `"z16"` or swaps the `Rs2Format`,
    /// this test fails. The depth wire encoding is part of the public topic
    /// contract; consumers parse it as a string.
    #[test]
    fn pinned_depth_formats_match() {
        assert_eq!(DEPTH_RS2_FORMAT, Rs2Format::Z16);
        assert_eq!(DEPTH_TOPIC_ENCODING, "z16");
    }
}
