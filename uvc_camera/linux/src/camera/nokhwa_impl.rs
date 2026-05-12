use std::os::unix::fs::{FileTypeExt, OpenOptionsExt};
use std::os::unix::io::AsRawFd;

use nokhwa::Camera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    CameraFormat, CameraIndex, ControlValueSetter, FrameFormat, KnownCameraControl,
    RequestedFormat, RequestedFormatType, Resolution as NokhwaResolution,
};

use super::device::CameraDevice;
use crate::camera::controls::{
    CameraControlRequest, ControlResult, ExposureMode, WhiteBalanceMode,
};
use crate::types::{CameraConfig, Encoding, Error, Frame, Result};

use v4l2_sys_mit::{
    V4L2_CID_AUTO_WHITE_BALANCE, V4L2_CID_EXPOSURE_ABSOLUTE, V4L2_CID_EXPOSURE_AUTO,
};

/// V4L2_EXPOSURE_AUTO = 0 (camera controls exposure automatically)
const V4L2_EXPOSURE_AUTO_VALUE: i64 = 0;
/// V4L2_EXPOSURE_MANUAL = 1 (manual exposure value via V4L2_CID_EXPOSURE_ABSOLUTE)
const V4L2_EXPOSURE_MANUAL_VALUE: i64 = 1;

/// Nokhwa-based camera implementation
///
/// Note: Camera from nokhwa doesn't implement Send, but we use it in a single-threaded
/// context (spawn_blocking) where it's never actually sent between threads during execution.
/// The Send bound is required only for moving into the blocking task initially.
pub struct NokhwaCamera {
    camera: Option<SendableCamera>,
    /// The actual camera encoding negotiated after `open_stream()`. The camera
    /// driver may return a format different from what was requested, so this is
    /// read back from `camera.camera_format()` rather than taken from the config.
    actual_camera_encoding: Option<Encoding>,
}

/// Wrapper to make Camera Send-safe
///
/// SAFETY: Camera is used only within a single thread (spawn_blocking).
/// It's never accessed from multiple threads concurrently.
struct SendableCamera(Camera);
unsafe impl Send for SendableCamera {}

impl NokhwaCamera {
    pub fn new() -> Self {
        Self {
            camera: None,
            actual_camera_encoding: None,
        }
    }
}

impl Default for NokhwaCamera {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraDevice for NokhwaCamera {
    fn open(&mut self, config: &CameraConfig) -> Result<()> {
        println!(
            "[uvc_camera] Opening nokhwa camera {}...",
            config.device_path
        );
        let index = parse_camera_index(&config.device_path)?;
        println!("[uvc_camera] Parsed index {}.", index);

        // Validate the device before handing it to nokhwa, which can hang
        // indefinitely on non-capture devices (e.g. metadata nodes).
        validate_video_device(&config.device_path)?;
        println!("[uvc_camera] Device {} validated.", config.device_path);

        let frame_rate = config.frame_rate.as_u16();
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(CameraFormat::new(
                NokhwaResolution::new(config.resolution.width(), config.resolution.height()),
                encoding_to_frame_format(config.camera_encoding),
                u32::from(frame_rate),
            )));

        println!(
            "[uvc_camera] Requesting format: {}x{} @ {} fps.",
            config.resolution.width(),
            config.resolution.height(),
            frame_rate
        );

        let mut camera = Camera::new(CameraIndex::Index(index), requested).map_err(|e| {
            Error::Camera(format!(
                "Failed to open camera {}: {}",
                config.device_path, e
            ))
        })?;

        println!(
            "[uvc_camera] Camera {} opened successfully.",
            config.device_path
        );

        camera.open_stream().map_err(|e| {
            Error::Camera(format!(
                "Failed to start stream for {}: {}",
                config.device_path, e
            ))
        })?;

        // Read back the format actually negotiated by the driver — it may differ
        // from what was requested (e.g. hardware only supports MJPEG).
        let negotiated = camera.camera_format().format();
        let actual_encoding = frame_format_to_encoding(negotiated);
        println!(
            "[uvc_camera] Camera {} stream started. Requested format: {}, negotiated format: {} ({:?})",
            config.device_path, config.camera_encoding, actual_encoding, negotiated,
        );

        self.actual_camera_encoding = Some(actual_encoding);
        self.camera = Some(SendableCamera(camera));
        Ok(())
    }

    fn capture_frame(&mut self) -> Result<Frame> {
        let encoding = self
            .actual_camera_encoding
            .ok_or_else(|| Error::Camera("Camera not open".to_string()))?;
        let camera = self
            .camera
            .as_mut()
            .ok_or_else(|| Error::Camera("Camera not open".to_string()))?;

        let frame = camera
            .0
            .frame()
            .map_err(|e| Error::Camera(format!("Failed to capture frame: {}", e)))?;

        let buffer = frame.buffer_bytes().to_vec();
        let resolution = frame.resolution();
        let timestamp = std::time::Instant::now();

        Ok(Frame::from_capture(
            buffer,
            resolution.width_x,
            resolution.height_y,
            timestamp,
            encoding,
        ))
    }

    fn is_open(&self) -> bool {
        self.camera.is_some()
    }

    fn apply_control(&mut self, request: &CameraControlRequest) -> ControlResult {
        let camera = match self.camera.as_mut() {
            Some(c) => &mut c.0,
            None => return ControlResult::err("Camera not open"),
        };

        match request {
            CameraControlRequest::SetBrightness { value } => {
                set_integer_control(camera, KnownCameraControl::Brightness, *value)
            }
            CameraControlRequest::SetContrast { value } => {
                set_integer_control(camera, KnownCameraControl::Contrast, *value)
            }
            CameraControlRequest::SetGain { value } => {
                set_integer_control(camera, KnownCameraControl::Gain, *value)
            }
            CameraControlRequest::SetExposure { mode, value } => set_exposure(camera, mode, *value),
            CameraControlRequest::SetWhiteBalance { mode, temperature } => {
                set_white_balance(camera, mode, *temperature)
            }
        }
    }
}

/// Set a simple integer camera control and read back the current value
fn set_integer_control(camera: &mut Camera, kind: KnownCameraControl, value: i32) -> ControlResult {
    match camera.set_camera_control(kind, ControlValueSetter::Integer(i64::from(value))) {
        Ok(()) => {
            let current = camera
                .camera_control(kind)
                .ok()
                .and_then(|c| c.value().as_integer().copied())
                .map(|v| v as i32)
                .unwrap_or(value);
            ControlResult::ok(format!("{:?} set to {}", kind, current), current)
        }
        Err(e) => ControlResult::err(format!("Failed to set {:?}: {}", kind, e)),
    }
}

/// Set exposure mode and optionally the absolute exposure value
fn set_exposure(camera: &mut Camera, mode: &ExposureMode, value: i32) -> ControlResult {
    let auto_value = match mode {
        ExposureMode::Auto => V4L2_EXPOSURE_AUTO_VALUE,
        ExposureMode::Manual => V4L2_EXPOSURE_MANUAL_VALUE,
    };

    if let Err(e) = camera.set_camera_control(
        KnownCameraControl::Other(V4L2_CID_EXPOSURE_AUTO as u128),
        ControlValueSetter::Integer(auto_value),
    ) {
        return ControlResult::err(format!("Failed to set exposure mode: {}", e));
    }

    match mode {
        ExposureMode::Auto => ControlResult::ok("Exposure set to auto mode", -1),
        ExposureMode::Manual => {
            // Set absolute exposure value (in 100µs units for V4L2)
            if let Err(e) = camera.set_camera_control(
                KnownCameraControl::Other(V4L2_CID_EXPOSURE_ABSOLUTE as u128),
                ControlValueSetter::Integer(i64::from(value)),
            ) {
                return ControlResult::err(format!(
                    "Exposure mode set to manual but value failed: {}",
                    e
                ));
            }

            let current = camera
                .camera_control(KnownCameraControl::Other(
                    V4L2_CID_EXPOSURE_ABSOLUTE as u128,
                ))
                .ok()
                .and_then(|c| c.value().as_integer().copied())
                .map(|v| v as i32)
                .unwrap_or(value);

            ControlResult::ok(
                format!("Exposure set to manual, value {}", current),
                current,
            )
        }
    }
}

/// Set white balance mode and optionally the temperature
fn set_white_balance(
    camera: &mut Camera,
    mode: &WhiteBalanceMode,
    temperature: i32,
) -> ControlResult {
    let auto_bool = matches!(mode, WhiteBalanceMode::Auto);

    if let Err(e) = camera.set_camera_control(
        KnownCameraControl::Other(V4L2_CID_AUTO_WHITE_BALANCE as u128),
        ControlValueSetter::Boolean(auto_bool),
    ) {
        return ControlResult::err(format!("Failed to set white balance mode: {}", e));
    }

    match mode {
        WhiteBalanceMode::Auto => ControlResult::ok("White balance set to auto mode", -1),
        WhiteBalanceMode::Manual => {
            if let Err(e) = camera.set_camera_control(
                KnownCameraControl::WhiteBalance,
                ControlValueSetter::Integer(i64::from(temperature)),
            ) {
                return ControlResult::err(format!(
                    "White balance mode set to manual but temperature failed: {}",
                    e
                ));
            }

            let current = camera
                .camera_control(KnownCameraControl::WhiteBalance)
                .ok()
                .and_then(|c| c.value().as_integer().copied())
                .map(|v| v as i32)
                .unwrap_or(temperature);

            ControlResult::ok(
                format!("White balance set to manual, temperature {}K", current),
                current,
            )
        }
    }
}

/// Get the owning group ID of a device file via `stat()`.
fn get_device_gid(device_path: &str) -> Option<libc::gid_t> {
    let c_path = std::ffi::CString::new(device_path).ok()?;
    let mut stat_buf: libc::stat = unsafe { std::mem::zeroed() };
    let ret = unsafe { libc::stat(c_path.as_ptr(), &mut stat_buf) };
    if ret == 0 {
        Some(stat_buf.st_gid)
    } else {
        None
    }
}

/// Get all group IDs the current process belongs to (effective GID + supplementary).
fn get_process_groups() -> Vec<libc::gid_t> {
    let mut groups = vec![unsafe { libc::getegid() }];

    let count = unsafe { libc::getgroups(0, std::ptr::null_mut()) };
    if count > 0 {
        let mut buf = vec![0 as libc::gid_t; count as usize];
        let actual = unsafe { libc::getgroups(count, buf.as_mut_ptr()) };
        if actual >= 0 {
            buf.truncate(actual as usize);
            for gid in buf {
                if !groups.contains(&gid) {
                    groups.push(gid);
                }
            }
        }
    }

    groups
}

/// Resolve a numeric GID to its group name by parsing `/etc/group`.
fn resolve_gid_to_name(gid: libc::gid_t) -> Option<String> {
    let contents = std::fs::read_to_string("/etc/group").ok()?;
    for line in contents.lines() {
        let mut fields = line.splitn(4, ':');
        let name = fields.next()?;
        let _ = fields.next(); // password
        let gid_str = fields.next()?;
        if let Ok(parsed) = gid_str.parse::<u32>()
            && parsed == gid
        {
            return Some(name.to_string());
        }
    }
    None
}

/// Build a diagnostic error message for a device permission failure.
///
/// Stats the device to find its required GID, queries the process's actual
/// supplementary groups, and reports the specific mismatch.
fn diagnose_permission_error(device_path: &str) -> String {
    let mut msg = format!("Permission denied opening {device_path}.");

    let Some(device_gid) = get_device_gid(device_path) else {
        msg.push_str(
            " Could not stat the device to determine the required group. \
             Ensure the device exists and your user is in the 'video' group.",
        );
        return msg;
    };

    // GID 65534 is the kernel's overflow GID — it means the device's real GID
    // (e.g. 44/video on the host) is not mapped into the current user namespace.
    // This happens when Apptainer runs in unprivileged/rootless mode.
    const OVERFLOW_GID: libc::gid_t = 65534;
    if device_gid == OVERFLOW_GID {
        msg.push_str(
            " The device's group appears as 'nogroup' (gid=65534), which means \
             the real group (likely 'video') is not mapped into this user namespace. \
             The container runtime must map the host 'video' group GID. \
             On the host, run: sudo usermod -aG video $USER && newgrp video",
        );
        return msg;
    }

    let group_label = match resolve_gid_to_name(device_gid) {
        Some(name) => format!("'{name}' (gid={device_gid})"),
        None => format!("gid={device_gid}"),
    };

    let process_groups = get_process_groups();

    if process_groups.contains(&device_gid) {
        msg.push_str(&format!(
            " The device requires group {group_label} and your process has that group, \
             so this may be a mandatory access control (SELinux/AppArmor) issue. \
             Check: ls -l {device_path}",
        ));
    } else {
        let gids: Vec<String> = process_groups.iter().map(|g| g.to_string()).collect();
        msg.push_str(&format!(
            " The device requires group {group_label} but your process groups \
             are [{gids}] — gid {device_gid} is missing. \
             On the host, run: sudo usermod -aG video $USER && newgrp video",
            gids = gids.join(", "),
        ));
    }

    msg
}

/// Validate that a device path points to an accessible V4L2 video capture device.
///
/// Runs before handing the device to nokhwa, which can hang indefinitely on
/// devices that exist but are not video-capture capable (e.g. metadata devices
/// like `/dev/video1` on single-camera systems).
fn validate_video_device(device_path: &str) -> Result<()> {
    // Check the device exists
    let metadata = std::fs::metadata(device_path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => Error::Camera(format!(
            "Device {device_path} does not exist. Check that the camera is connected \
             and the device path is correct."
        )),
        std::io::ErrorKind::PermissionDenied => {
            Error::Camera(diagnose_permission_error(device_path))
        }
        _ => Error::Camera(format!("Cannot access {device_path}: {e}")),
    })?;

    // All V4L2 devices are character devices
    if !metadata.file_type().is_char_device() {
        return Err(Error::Camera(format!(
            "{device_path} is not a character device — not a valid video device"
        )));
    }

    // Open the device to verify accessibility
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(device_path)
        .map_err(|e| match e.raw_os_error() {
            Some(libc::EBUSY) => Error::Camera(format!(
                "{device_path} is busy — another process may be using the camera"
            )),
            Some(libc::EACCES) | Some(libc::EPERM) => {
                Error::Camera(diagnose_permission_error(device_path))
            }
            _ => Error::Camera(format!("Cannot open {device_path}: {e}")),
        })?;

    // Query V4L2 capabilities via VIDIOC_QUERYCAP ioctl.
    //
    // VIDIOC_QUERYCAP = _IOR('V', 0, struct v4l2_capability)
    //   direction=Read(2), size=104, type='V'(0x56), nr=0
    //   = (2 << 30) | (104 << 16) | (0x56 << 8) | 0
    #[repr(C)]
    struct V4l2Capability {
        driver: [u8; 16],
        card: [u8; 32],
        bus_info: [u8; 32],
        version: u32,
        capabilities: u32,
        device_caps: u32,
        reserved: [u32; 3],
    }

    const VIDIOC_QUERYCAP: libc::c_ulong = 0x80685600;
    const CAP_VIDEO_CAPTURE: u32 = 0x00000001;
    const CAP_DEVICE_CAPS: u32 = 0x80000000;

    let mut cap: V4l2Capability = unsafe { std::mem::zeroed() };
    let ret = unsafe { libc::ioctl(file.as_raw_fd(), VIDIOC_QUERYCAP, &mut cap) };

    if ret < 0 {
        return Err(Error::Camera(format!(
            "{device_path} is not a V4L2 video device"
        )));
    }

    // Prefer per-node device_caps when available, fall back to global capabilities
    let effective_caps = if cap.capabilities & CAP_DEVICE_CAPS != 0 {
        cap.device_caps
    } else {
        cap.capabilities
    };

    if effective_caps & CAP_VIDEO_CAPTURE == 0 {
        let card = std::str::from_utf8(&cap.card)
            .unwrap_or("unknown")
            .trim_end_matches('\0');
        return Err(Error::Camera(format!(
            "{device_path} ({card}) does not support video capture — \
             it may be a metadata device. Try another /dev/videoN index."
        )));
    }

    // Close the fd before nokhwa opens the device
    drop(file);

    Ok(())
}

/// Map an [`Encoding`] to the corresponding nokhwa [`FrameFormat`] to request
/// from the camera hardware.
fn encoding_to_frame_format(encoding: Encoding) -> FrameFormat {
    match encoding {
        Encoding::Mjpeg => FrameFormat::MJPEG,
        Encoding::Rgb8 => FrameFormat::RAWRGB,
        Encoding::Bgr8 => FrameFormat::RAWBGR,
    }
}

/// Map a negotiated nokhwa [`FrameFormat`] back to our [`Encoding`].
///
/// Used to record what the camera driver actually settled on after `open_stream`,
/// which may differ from what was requested.
fn frame_format_to_encoding(fmt: FrameFormat) -> Encoding {
    match fmt {
        FrameFormat::MJPEG => Encoding::Mjpeg,
        FrameFormat::RAWRGB => Encoding::Rgb8,
        FrameFormat::RAWBGR => Encoding::Bgr8,
        other => {
            tracing::warn!(
                "Unknown camera FrameFormat {:?}, falling back to Rgb8",
                other
            );
            Encoding::Rgb8
        }
    }
}

/// Parse camera device path into index
fn parse_camera_index(device_path: &str) -> Result<u32> {
    if let Some(stripped) = device_path.strip_prefix("/dev/video") {
        stripped
            .parse::<u32>()
            .map_err(|_| Error::InvalidDevicePath(device_path.to_string()))
    } else {
        Err(Error::InvalidDevicePath(device_path.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_to_frame_format() {
        assert_eq!(
            encoding_to_frame_format(Encoding::Rgb8),
            FrameFormat::RAWRGB
        );
        assert_eq!(
            encoding_to_frame_format(Encoding::Bgr8),
            FrameFormat::RAWBGR
        );
        assert_eq!(
            encoding_to_frame_format(Encoding::Mjpeg),
            FrameFormat::MJPEG
        );
    }

    #[test]
    fn test_frame_format_to_encoding() {
        assert_eq!(
            frame_format_to_encoding(FrameFormat::RAWRGB),
            Encoding::Rgb8
        );
        assert_eq!(
            frame_format_to_encoding(FrameFormat::RAWBGR),
            Encoding::Bgr8
        );
        assert_eq!(
            frame_format_to_encoding(FrameFormat::MJPEG),
            Encoding::Mjpeg
        );
    }

    #[test]
    fn test_encoding_frame_format_roundtrip() {
        for enc in [Encoding::Rgb8, Encoding::Bgr8, Encoding::Mjpeg] {
            let fmt = encoding_to_frame_format(enc);
            let back = frame_format_to_encoding(fmt);
            assert_eq!(back, enc, "Roundtrip failed for {enc:?}");
        }
    }

    #[test]
    fn test_parse_camera_index_valid() {
        assert_eq!(parse_camera_index("/dev/video0").unwrap(), 0);
        assert_eq!(parse_camera_index("/dev/video1").unwrap(), 1);
        assert_eq!(parse_camera_index("/dev/video42").unwrap(), 42);
        assert_eq!(parse_camera_index("/dev/video1000").unwrap(), 1000);
    }

    #[test]
    fn test_parse_camera_index_invalid() {
        // Only /dev/videoN format is accepted
        assert!(parse_camera_index("/dev/video").is_err());
        assert!(parse_camera_index("/dev/camera0").is_err());
        assert!(parse_camera_index("video0").is_err());
        assert!(parse_camera_index("0").is_err());
        assert!(parse_camera_index("").is_err());
        assert!(parse_camera_index("invalid").is_err());
    }

    #[test]
    fn test_validate_device_nonexistent_path() {
        let result = validate_video_device("/dev/video_nonexistent_99999");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("does not exist"), "Unexpected error: {err}");
    }

    #[test]
    fn test_validate_device_regular_file() {
        let tmp = std::env::temp_dir().join("uvc_camera_test_not_a_device");
        std::fs::write(&tmp, b"not a device").unwrap();
        let result = validate_video_device(tmp.to_str().unwrap());
        let _ = std::fs::remove_file(&tmp);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not a character device"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn test_validate_device_directory() {
        let result = validate_video_device("/tmp");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not a character device"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn test_validate_device_not_v4l2() {
        // /dev/null is a character device but not a V4L2 device
        let result = validate_video_device("/dev/null");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not a V4L2 video device"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn test_get_device_gid_dev_null() {
        let gid = get_device_gid("/dev/null");
        assert!(gid.is_some(), "/dev/null should be stattable");
    }

    #[test]
    fn test_get_device_gid_nonexistent() {
        let gid = get_device_gid("/dev/nonexistent_device_xyz");
        assert!(gid.is_none());
    }

    #[test]
    fn test_get_process_groups_contains_egid() {
        let groups = get_process_groups();
        let egid = unsafe { libc::getegid() };
        assert!(
            groups.contains(&egid),
            "Process groups {groups:?} should contain effective GID {egid}"
        );
    }

    #[test]
    fn test_resolve_gid_to_name_root() {
        let name = resolve_gid_to_name(0);
        assert_eq!(name.as_deref(), Some("root"));
    }

    #[test]
    fn test_resolve_gid_to_name_unknown() {
        let name = resolve_gid_to_name(99999);
        assert!(name.is_none(), "GID 99999 should not resolve to a name");
    }

    #[test]
    fn test_diagnose_permission_error_nonexistent() {
        let msg = diagnose_permission_error("/dev/nonexistent_device_xyz");
        assert!(msg.contains("Could not stat"), "Unexpected message: {msg}");
    }
}
