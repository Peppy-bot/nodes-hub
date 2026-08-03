//! V4L2 capture device.
//!
//! Talks to the kernel through the `v4l` crate directly, matching `zed_camera`,
//! the other V4L2 node in this hub. The mmap stream is bound to the thread that
//! dequeues it, so frame capture lives on the dedicated capture thread; the
//! device fd is shareable, and [`ControlHandle`] exposes it to service handlers
//! so controls apply synchronously.

use std::io::ErrorKind;
use std::os::unix::fs::MetadataExt;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use v4l::buffer::Type;
use v4l::capability::Flags;
use v4l::control::{Control, Value};
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture as _;
use v4l::{Format, FourCC};

use crate::camera::controls::{
    CameraControlRequest, ControlResult, ExposureMode, VALUE_UNAVAILABLE, WhiteBalanceMode,
};
use crate::types::{CameraConfig, Encoding, Error, Frame, Result};

/// Control ids from `<linux/v4l2-controls.h>`. Written out rather than
/// generated: they are stable kernel ABI, and seven constants do not justify a
/// libclang build dependency. Pinned by `cids_match_the_kernel_abi`.
const V4L2_CID_BASE: u32 = 0x0098_0900;
const V4L2_CID_BRIGHTNESS: u32 = V4L2_CID_BASE;
const V4L2_CID_CONTRAST: u32 = V4L2_CID_BASE + 1;
const V4L2_CID_AUTO_WHITE_BALANCE: u32 = V4L2_CID_BASE + 12;
const V4L2_CID_GAIN: u32 = V4L2_CID_BASE + 19;
const V4L2_CID_WHITE_BALANCE_TEMPERATURE: u32 = V4L2_CID_BASE + 26;
const V4L2_CID_CAMERA_CLASS_BASE: u32 = 0x009a_0900;
const V4L2_CID_EXPOSURE_AUTO: u32 = V4L2_CID_CAMERA_CLASS_BASE + 1;
const V4L2_CID_EXPOSURE_ABSOLUTE: u32 = V4L2_CID_CAMERA_CLASS_BASE + 2;

/// `V4L2_EXPOSURE_AUTO` / `V4L2_EXPOSURE_MANUAL` from `enum v4l2_exposure_auto_type`.
const V4L2_EXPOSURE_AUTO_VALUE: i64 = 0;
const V4L2_EXPOSURE_MANUAL_VALUE: i64 = 1;

/// Buffers in the mmap queue: deep enough to ride out a scheduling hiccup,
/// shallow enough that a late dequeue does not hand back a stale frame.
const BUFFER_COUNT: u32 = 4;

/// Upper bound on a single dequeue. Without it a driver that stops delivering
/// (an unplugged camera) blocks the capture thread indefinitely; with it the
/// loop sees an error and its stall window decides when to give up.
const DEQUEUE_TIMEOUT: Duration = Duration::from_secs(1);

/// The kernel's overflow gid: what a rootless container reports for a host
/// group that has no mapping inside its user namespace.
const OVERFLOW_GID: u32 = 65534;

/// Where `captured_at` stamps come from, injected so contexts without a node
/// runtime (the loopback integration tests) can use the wall clock directly.
pub type StampSource = fn() -> Result<SystemTime>;

/// Stamp from the daemon-resolved clock: the OS clock in wall mode, the
/// simulator's time when the stack runs under sim time. Requires
/// `peppygen::clock::init` to have run, which setup does before the capture
/// loop spawns.
pub fn clock_stamp() -> Result<SystemTime> {
    let ns =
        peppygen::clock::now_ns().map_err(|e| Error::Other(format!("clock not ready: {e}")))?;
    Ok(UNIX_EPOCH + Duration::from_nanos(ns))
}

/// Wall-clock stamps for contexts with no node runtime.
pub fn wall_stamp() -> Result<SystemTime> {
    Ok(SystemTime::now())
}

/// The format the driver actually accepted, which may differ from the request.
struct Negotiated {
    encoding: Encoding,
    width: u32,
    height: u32,
}

/// What the stream actually publishes, as negotiated with the driver at open.
/// This is what `video_stream_info` reports, so it must describe the wire, not
/// the request.
#[derive(Debug, Clone)]
pub struct StreamDescription {
    pub width: u32,
    pub height: u32,
    /// The effective publish rate: the configured pace, capped by what the
    /// driver actually delivers.
    pub frames_per_second: u8,
}

/// Shared handle for applying camera controls from service handlers.
///
/// Control ioctls go through the same fd the stream uses; the kernel
/// serializes them, and a blocked dequeue does not hold the serialization lock
/// while it waits, so controls apply promptly even mid-stream. Applying them
/// synchronously in the handler means the caller's response always reflects
/// what actually happened to the hardware.
#[derive(Clone)]
pub struct ControlHandle {
    device: Arc<Device>,
}

impl ControlHandle {
    /// Apply a control request, reporting the value the driver actually kept.
    pub fn apply(&self, request: &CameraControlRequest) -> ControlResult {
        let device = self.device.as_ref();
        match request {
            CameraControlRequest::SetBrightness { value } => {
                set_integer_control(device, V4L2_CID_BRIGHTNESS, "brightness", *value)
            }
            CameraControlRequest::SetContrast { value } => {
                set_integer_control(device, V4L2_CID_CONTRAST, "contrast", *value)
            }
            CameraControlRequest::SetGain { value } => {
                set_integer_control(device, V4L2_CID_GAIN, "gain", *value)
            }
            CameraControlRequest::SetExposure { mode, value } => set_exposure(device, mode, *value),
            CameraControlRequest::SetWhiteBalance { mode, temperature } => {
                set_white_balance(device, mode, *temperature)
            }
        }
    }
}

/// A V4L2 capture device, streaming once [`CameraDevice::open`] returns.
pub struct CameraDevice {
    stamp_now: StampSource,
    device: Option<Arc<Device>>,
    stream: Option<MmapStream<'static>>,
    negotiated: Option<Negotiated>,
    description: Option<StreamDescription>,
}

impl CameraDevice {
    pub fn new(stamp_now: StampSource) -> Self {
        Self {
            stamp_now,
            device: None,
            stream: None,
            negotiated: None,
            description: None,
        }
    }

    /// Open and configure the camera.
    ///
    /// The path is opened directly, so a udev-pinned symlink such as
    /// `/dev/openarm/left_wrist_cam` works without resolving it to an index.
    pub fn open(&mut self, config: &CameraConfig) -> Result<()> {
        let path = config.device_path.as_str();
        tracing::debug!("opening {path}");

        let device = Device::with_path(path).map_err(|e| open_error(path, &e))?;

        let caps = device
            .query_caps()
            .map_err(|e| Error::Camera(format!("{path}: cannot query capabilities: {e}")))?;
        if !caps.capabilities.contains(Flags::VIDEO_CAPTURE) {
            return Err(Error::Camera(format!(
                "{path} is not a video capture device (driver {}, card {})",
                caps.driver, caps.card
            )));
        }
        tracing::debug!(
            "{path} validated: driver {}, card {}",
            caps.driver,
            caps.card
        );

        let requested = Format::new(
            config.resolution.width(),
            config.resolution.height(),
            fourcc_for(config.camera_encoding),
        );
        let actual = device
            .set_format(&requested)
            .map_err(|e| Error::Camera(format!("{path}: cannot set format: {e}")))?;

        let frame_rate = u32::from(config.frame_rate.as_u16());
        let accepted_params = device
            .set_params(&v4l::video::capture::Parameters::with_fps(frame_rate))
            .map_err(|e| Error::Camera(format!("{path}: cannot set frame rate: {e}")))?;

        let interval = accepted_params.interval;
        let publish_fps = effective_publish_fps(
            config.frame_rate.as_u8(),
            interval.numerator,
            interval.denominator,
        );
        if u32::from(publish_fps) != frame_rate {
            tracing::info!(
                "{path}: driver interval {}/{}s, publishing at {publish_fps} fps \
                 (requested {frame_rate})",
                interval.numerator,
                interval.denominator,
            );
        }

        let encoding = encoding_for(actual.fourcc).ok_or_else(|| {
            Error::Camera(format!(
                "{path} negotiated {}, which this node cannot decode",
                actual.fourcc
            ))
        })?;

        tracing::info!(
            "camera {path} streaming: requested {}x{} {}, negotiated {}x{} {encoding}",
            requested.width,
            requested.height,
            config.camera_encoding,
            actual.width,
            actual.height,
        );

        // A yuyv topic encoding is passthrough-only, so a driver that negotiated
        // anything else would fail every single frame downstream. Refuse here
        // instead of streaming into a loop that can never publish.
        if config.topic_encoding == Encoding::Yuyv && encoding != Encoding::Yuyv {
            return Err(Error::Camera(format!(
                "topic_encoding yuyv requires a yuyv camera stream, but {path} negotiated \
                 {encoding}; set camera_encoding to yuyv on a camera that supports it, or pick \
                 another topic_encoding"
            )));
        }

        let mut stream = MmapStream::with_buffers(&device, Type::VideoCapture, BUFFER_COUNT)
            .map_err(|e| Error::Camera(format!("{path}: cannot start mmap stream: {e}")))?;
        stream.set_timeout(DEQUEUE_TIMEOUT);

        self.negotiated = Some(Negotiated {
            encoding,
            width: actual.width,
            height: actual.height,
        });
        self.description = Some(StreamDescription {
            width: actual.width,
            height: actual.height,
            frames_per_second: publish_fps,
        });
        self.stream = Some(stream);
        self.device = Some(Arc::new(device));
        Ok(())
    }

    /// The negotiated stream geometry and rate; `None` until opened.
    pub fn stream_description(&self) -> Option<StreamDescription> {
        self.description.clone()
    }

    /// A shareable handle for applying controls; `None` until opened.
    pub fn control_handle(&self) -> Option<ControlHandle> {
        self.device.as_ref().map(|device| ControlHandle {
            device: Arc::clone(device),
        })
    }

    /// Capture a single frame in the camera's negotiated encoding.
    pub fn capture_frame(&mut self) -> Result<Frame> {
        let (Some(stream), Some(negotiated)) = (self.stream.as_mut(), self.negotiated.as_ref())
        else {
            return Err(Error::Camera("Camera not open".to_string()));
        };

        let (buffer, meta) = stream
            .next()
            .map_err(|e| Error::Camera(format!("Failed to capture frame: {e}")))?;
        let captured_at = (self.stamp_now)()?;

        // A short transfer is a corrupt frame, not a small one: publishing it
        // would hand consumers a payload that disagrees with the dimensions.
        let used = meta.bytesused as usize;
        if used == 0 || used > buffer.len() {
            return Err(Error::Camera(format!(
                "driver reported {used} bytes for a {}-byte buffer",
                buffer.len()
            )));
        }

        Ok(Frame::from_capture(
            buffer[..used].to_vec(),
            negotiated.width,
            negotiated.height,
            captured_at,
            negotiated.encoding,
        ))
    }

    /// Check if the camera is open.
    pub fn is_open(&self) -> bool {
        self.device.is_some()
    }
}

/// The rate `video_stream_info` reports: the configured pace, capped by the
/// driver-accepted frame interval (`numerator/denominator` seconds per frame).
///
/// Never zero: the contract cannot express it and `FrameRate` rejects it at
/// setup, so a sub-1fps driver readback rounds up to 1 rather than collapsing
/// the reported rate. A readback with a zero term is unusable and falls back
/// to the configured rate.
fn effective_publish_fps(configured_fps: u8, numerator: u32, denominator: u32) -> u8 {
    if numerator == 0 || denominator == 0 {
        return configured_fps;
    }
    let driver_fps = denominator.div_ceil(numerator);
    configured_fps
        .min(u8::try_from(driver_fps).unwrap_or(u8::MAX))
        .max(1)
}

/// Turn an open failure into something actionable, diagnosing the common
/// container case where the process is not in the device's group.
fn open_error(path: &str, error: &std::io::Error) -> Error {
    match error.kind() {
        ErrorKind::NotFound => Error::Camera(format!(
            "Device {path} does not exist. Check that the camera is connected and the device \
             path is correct."
        )),
        ErrorKind::PermissionDenied => Error::Camera(diagnose_permission_error(path)),
        _ => Error::Camera(format!("Failed to open camera {path}: {error}")),
    }
}

/// Explain a permission failure in terms of the groups actually involved.
fn diagnose_permission_error(path: &str) -> String {
    let mut message = format!("Permission denied opening {path}.");

    let Ok(gid) = std::fs::metadata(path).map(|m| m.gid()) else {
        return message;
    };
    let owner = group_name(gid).unwrap_or_else(|| gid.to_string());
    message.push_str(&format!(" The device is owned by group {owner} ({gid})."));

    // Joining a group cannot help here: the host group does not exist inside
    // the container's user namespace.
    if gid == OVERFLOW_GID {
        message.push_str(
            " That is the overflow gid, so the host group has no mapping inside this container.",
        );
        return message;
    }

    match process_groups() {
        Some(groups) if groups.contains(&gid) => {
            message.push_str(" This process is in that group, so the cause is elsewhere.");
        }
        Some(groups) => {
            let names: Vec<String> = groups
                .iter()
                .map(|g| group_name(*g).unwrap_or_else(|| g.to_string()))
                .collect();
            message.push_str(&format!(
                " This process is in [{}]; add it to {owner}.",
                names.join(", ")
            ));
        }
        None => {}
    }
    message
}

/// Resolve a gid to its name via `/etc/group`.
fn group_name(gid: u32) -> Option<String> {
    let contents = std::fs::read_to_string("/etc/group").ok()?;
    contents.lines().find_map(|line| {
        let mut fields = line.splitn(4, ':');
        let name = fields.next()?;
        let _password = fields.next()?;
        (fields.next()?.parse::<u32>().ok()? == gid).then(|| name.to_string())
    })
}

/// Supplementary gids of this process, from `/proc/self/status`.
fn process_groups() -> Option<Vec<u32>> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("Groups:"))?;
    Some(
        line.trim_start_matches("Groups:")
            .split_whitespace()
            .filter_map(|g| g.parse::<u32>().ok())
            .collect(),
    )
}

/// Set an integer control and read back what the driver kept.
fn set_integer_control(device: &Device, cid: u32, name: &str, value: i32) -> ControlResult {
    if let Err(e) = write_control(device, cid, i64::from(value)) {
        return ControlResult::err(format!("Failed to set {name}: {e}"));
    }
    // The driver clamps to its own range, so report what it kept rather than
    // what was asked for.
    let current = read_control(device, cid).unwrap_or(value);
    ControlResult::ok(format!("{name} set to {current}"), current)
}

/// Set exposure mode and, in manual mode, the absolute exposure value.
fn set_exposure(device: &Device, mode: &ExposureMode, value: i32) -> ControlResult {
    let auto_value = match mode {
        ExposureMode::Auto => V4L2_EXPOSURE_AUTO_VALUE,
        ExposureMode::Manual => V4L2_EXPOSURE_MANUAL_VALUE,
    };
    if let Err(e) = write_control(device, V4L2_CID_EXPOSURE_AUTO, auto_value) {
        return ControlResult::err(format!("Failed to set exposure mode: {e}"));
    }

    match mode {
        ExposureMode::Auto => ControlResult::ok("Exposure set to auto mode", VALUE_UNAVAILABLE),
        ExposureMode::Manual => {
            // Absolute exposure is in 100us units for V4L2.
            if let Err(e) = write_control(device, V4L2_CID_EXPOSURE_ABSOLUTE, i64::from(value)) {
                return ControlResult::err(format!(
                    "Exposure mode set to manual but value failed: {e}"
                ));
            }
            let current = read_control(device, V4L2_CID_EXPOSURE_ABSOLUTE).unwrap_or(value);
            ControlResult::ok(format!("Exposure set to manual, value {current}"), current)
        }
    }
}

/// Set white balance mode and, in manual mode, the colour temperature.
fn set_white_balance(device: &Device, mode: &WhiteBalanceMode, temperature: i32) -> ControlResult {
    let auto = i64::from(matches!(mode, WhiteBalanceMode::Auto));
    if let Err(e) = write_control(device, V4L2_CID_AUTO_WHITE_BALANCE, auto) {
        return ControlResult::err(format!("Failed to set white balance mode: {e}"));
    }

    match mode {
        WhiteBalanceMode::Auto => {
            ControlResult::ok("White balance set to auto mode", VALUE_UNAVAILABLE)
        }
        WhiteBalanceMode::Manual => {
            if let Err(e) = write_control(
                device,
                V4L2_CID_WHITE_BALANCE_TEMPERATURE,
                i64::from(temperature),
            ) {
                return ControlResult::err(format!(
                    "White balance mode set to manual but temperature failed: {e}"
                ));
            }
            let current =
                read_control(device, V4L2_CID_WHITE_BALANCE_TEMPERATURE).unwrap_or(temperature);
            ControlResult::ok(
                format!("White balance set to manual, temperature {current}K"),
                current,
            )
        }
    }
}

fn write_control(device: &Device, cid: u32, value: i64) -> std::result::Result<(), String> {
    device
        .set_control(Control {
            id: cid,
            value: Value::Integer(value),
        })
        .map_err(|e| e.to_string())
}

fn read_control(device: &Device, cid: u32) -> Option<i32> {
    match device.control(cid).ok()?.value {
        Value::Integer(value) => Some(value as i32),
        Value::Boolean(value) => Some(i32::from(value)),
        _ => None,
    }
}

/// The pixel format to request for an [`Encoding`].
fn fourcc_for(encoding: Encoding) -> FourCC {
    match encoding {
        Encoding::Rgb8 => FourCC::new(b"RGB3"),
        Encoding::Bgr8 => FourCC::new(b"BGR3"),
        Encoding::Mjpeg => FourCC::new(b"MJPG"),
        Encoding::Yuyv => FourCC::new(b"YUYV"),
    }
}

/// The [`Encoding`] a negotiated pixel format corresponds to, if this node can
/// decode it.
fn encoding_for(fourcc: FourCC) -> Option<Encoding> {
    match &fourcc.repr {
        b"RGB3" => Some(Encoding::Rgb8),
        b"BGR3" => Some(Encoding::Bgr8),
        b"MJPG" | b"JPEG" => Some(Encoding::Mjpeg),
        b"YUYV" => Some(Encoding::Yuyv),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_fps_never_reports_zero() {
        // A driver interval slower than one frame per second (5s per frame
        // here) must round up to 1, not collapse to a rate the contract
        // cannot express and consumers would divide by.
        assert_eq!(effective_publish_fps(15, 5, 1), 1);
        // Unusable readback falls back to the configured rate.
        assert_eq!(effective_publish_fps(15, 0, 30), 15);
        assert_eq!(effective_publish_fps(15, 1, 0), 15);
        // Normal negotiation: capped by the slower of the two sides.
        assert_eq!(effective_publish_fps(15, 1, 30), 15);
        assert_eq!(effective_publish_fps(60, 1, 30), 30);
    }

    #[test]
    fn cids_match_the_kernel_abi() {
        // Values from <linux/v4l2-controls.h>, checked against the generated
        // v4l2-sys bindings before that dependency was dropped.
        assert_eq!(V4L2_CID_BRIGHTNESS, 0x0098_0900);
        assert_eq!(V4L2_CID_CONTRAST, 0x0098_0901);
        assert_eq!(V4L2_CID_AUTO_WHITE_BALANCE, 0x0098_090c);
        assert_eq!(V4L2_CID_GAIN, 0x0098_0913);
        assert_eq!(V4L2_CID_WHITE_BALANCE_TEMPERATURE, 0x0098_091a);
        assert_eq!(V4L2_CID_EXPOSURE_AUTO, 0x009a_0901);
        assert_eq!(V4L2_CID_EXPOSURE_ABSOLUTE, 0x009a_0902);
    }

    #[test]
    fn every_encoding_round_trips_through_its_fourcc() {
        for encoding in [
            Encoding::Rgb8,
            Encoding::Bgr8,
            Encoding::Mjpeg,
            Encoding::Yuyv,
        ] {
            assert_eq!(
                encoding_for(fourcc_for(encoding)),
                Some(encoding),
                "{encoding} did not survive the fourcc mapping"
            );
        }
    }

    #[test]
    fn unsupported_fourcc_is_rejected_rather_than_guessed() {
        // NV12 is common on capture hardware and this node cannot decode it, so
        // it must fail at open rather than stream bytes the pipeline would
        // reinterpret as something else.
        assert_eq!(encoding_for(FourCC::new(b"NV12")), None);
    }
}
