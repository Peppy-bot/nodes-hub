//! ZED USB capture over the `v4l` crate: side-by-side YUYV mmap streaming
//! plus standard V4L2 controls, in safe Rust. The unit serial comes from the
//! USB descriptor of the ZED's HID sibling ([`zed_serial`]); the video
//! interface carries none. The device runs auto exposure.

use std::io::ErrorKind;
use std::path::Path;
use std::time::Duration;

use v4l::buffer::Type;
use v4l::control::{Control, Value};
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture as _;
use v4l::{Format, FourCC};

use crate::resolution::Resolution;

/// Two buffers bound the queue: the newest completed frame is at most one
/// frame behind the sensor, with no drain logic.
const BUFFER_COUNT: u32 = 2;

const ZED_VENDOR_ID: &str = "2b03";
/// The ZED's HID (IMU) interface; its USB descriptor carries the unit serial.
const ZED_HID_PRODUCT_ID: &str = "f681";

// V4L2_CID_* ids (<linux/v4l2-controls.h>) the ZED exposes, for callers
// driving [`Capture::control`]/[`Capture::set_control`].
pub const CID_BRIGHTNESS: u32 = 0x0098_0900;
pub const CID_CONTRAST: u32 = 0x0098_0901;
pub const CID_SATURATION: u32 = 0x0098_0902;
pub const CID_HUE: u32 = 0x0098_0903;
pub const CID_AWB_AUTO: u32 = 0x0098_090c;
pub const CID_GAMMA: u32 = 0x0098_0910;
pub const CID_GAIN: u32 = 0x0098_0913;
pub const CID_AWB_TEMPERATURE: u32 = 0x0098_091a;
pub const CID_SHARPNESS: u32 = 0x0098_091b;

/// The result of one grab attempt.
pub enum Grab {
    Frame { hw_stamp_ns: u64 },
    Timeout,
    Dead,
}

/// Open ZED capture stream (V4L2 mmap streaming via `v4l`).
pub struct Capture {
    device: Device,
    stream: MmapStream<'static>,
    full_width: u32,
    height: u32,
}

impl Capture {
    /// Opens `/dev/video<dev_id>`, negotiates the side-by-side geometry at
    /// `fps`, resets every control to its driver default (so a restart never
    /// inherits the previous session), and starts streaming.
    pub fn open(dev_id: usize, resolution: Resolution, fps: u32) -> Result<Self, String> {
        resolution.validate_fps(fps)?;
        let (eye_width, height) = resolution.eye_size();
        let full_width = eye_width * 2;

        let device = Device::new(dev_id).map_err(|e| format!("open /dev/video{dev_id}: {e}"))?;
        let yuyv = FourCC::new(b"YUYV");
        let format = device
            .set_format(&Format::new(full_width, height, yuyv))
            .map_err(|e| format!("set format: {e}"))?;
        if (format.width, format.height, format.fourcc) != (full_width, height, yuyv) {
            return Err(format!(
                "camera negotiated {}x{} {}, expected {full_width}x{height} YUYV",
                format.width, format.height, format.fourcc
            ));
        }
        device
            .set_params(&v4l::video::capture::Parameters::with_fps(fps))
            .map_err(|e| format!("set frame rate: {e}"))?;
        reset_controls_to_defaults(&device);

        let stream = MmapStream::with_buffers(&device, Type::VideoCapture, BUFFER_COUNT)
            .map_err(|e| format!("mmap stream: {e}"))?;
        Ok(Self {
            device,
            stream,
            full_width,
            height,
        })
    }

    /// Full side-by-side geometry (width spans both eyes).
    pub fn frame_size(&self) -> (u32, u32) {
        (self.full_width, self.height)
    }

    /// Copy the next full side-by-side YUYV frame into `yuyv`, waiting up to
    /// `timeout`. A short transfer is dropped as Timeout rather than
    /// returned with a stale tail.
    pub fn grab(&mut self, yuyv: &mut [u8], timeout: Duration) -> Grab {
        self.stream.set_timeout(timeout);
        let (frame, meta) = match self.stream.next() {
            Ok(next) => next,
            Err(e) if e.kind() == ErrorKind::TimedOut => return Grab::Timeout,
            Err(_) => return Grab::Dead,
        };
        let expected = (self.full_width * self.height * 2) as usize;
        if (meta.bytesused as usize) < expected {
            return Grab::Timeout;
        }
        let n = yuyv.len().min(expected);
        yuyv[..n].copy_from_slice(&frame[..n]);
        Grab::Frame {
            hw_stamp_ns: meta.timestamp.sec as u64 * 1_000_000_000
                + meta.timestamp.usec as u64 * 1_000,
        }
    }

    /// Reads an integer control by CID.
    pub fn control(&self, cid: u32) -> Result<i32, String> {
        match self.device.control(cid).map_err(|e| e.to_string())?.value {
            Value::Integer(value) => Ok(value as i32),
            Value::Boolean(value) => Ok(value as i32),
            other => Err(format!("control {cid:#x} has non-integer value {other:?}")),
        }
    }

    /// Writes an integer control by CID; the driver clamps to its range.
    pub fn set_control(&self, cid: u32, value: i32) -> Result<(), String> {
        self.device
            .set_control(Control {
                id: cid,
                value: Value::Integer(value as i64),
            })
            .map_err(|e| e.to_string())
    }
}

/// Best-effort sweep of every exposed control back to its driver default.
fn reset_controls_to_defaults(device: &Device) {
    let Ok(descriptions) = device.query_controls() else {
        return;
    };
    for control in descriptions {
        let _ = device.set_control(Control {
            id: control.id,
            value: Value::Integer(control.default),
        });
    }
}

/// V4L2 index from a /dev/video<N> path.
pub fn device_index(device_path: &str) -> Result<usize, String> {
    device_path
        .strip_prefix("/dev/video")
        .and_then(|index| index.parse().ok())
        .ok_or_else(|| format!("device_path must be /dev/video<N>, got {device_path:?}"))
}

/// The unit serial that keys the factory calibration, read from the USB
/// descriptor of the ZED's HID interface.
pub fn zed_serial() -> Result<i32, String> {
    zed_serial_under(Path::new("/sys/bus/usb/devices"))
}

fn zed_serial_under(usb_devices: &Path) -> Result<i32, String> {
    let entries = std::fs::read_dir(usb_devices)
        .map_err(|e| format!("read {}: {e}", usb_devices.display()))?;
    let attr = |dir: &Path, name: &str| {
        std::fs::read_to_string(dir.join(name))
            .ok()
            .map(|s| s.trim().to_string())
    };
    let serials: Vec<String> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|dir| {
            attr(dir, "idVendor").as_deref() == Some(ZED_VENDOR_ID)
                && attr(dir, "idProduct").as_deref() == Some(ZED_HID_PRODUCT_ID)
        })
        .filter_map(|dir| attr(&dir, "serial"))
        .collect();
    match serials.as_slice() {
        [] => Err("no ZED HID interface on the USB bus; is the camera connected?".to_string()),
        [serial] => serial
            .parse()
            .map_err(|_| format!("ZED reported non-numeric serial {serial:?}")),
        many => Err(format!(
            "{} ZED units on the bus ({}); serial is ambiguous",
            many.len(),
            many.join(", ")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_index_parses_only_video_paths() {
        assert_eq!(device_index("/dev/video7").unwrap(), 7);
        assert!(device_index("/dev/ttyUSB0").is_err());
    }

    #[test]
    fn capture_is_send() {
        // The node shares a Capture between its pipeline thread and service
        // handlers behind a mutex; losing Send must fail here, not in-container.
        fn assert_send<T: Send>() {}
        assert_send::<Capture>();
    }

    fn fake_usb_device(root: &Path, name: &str, vendor: &str, product: &str, serial: Option<&str>) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("idVendor"), format!("{vendor}\n")).unwrap();
        std::fs::write(dir.join("idProduct"), format!("{product}\n")).unwrap();
        if let Some(serial) = serial {
            std::fs::write(dir.join("serial"), format!("{serial}\n")).unwrap();
        }
    }

    #[test]
    fn serial_comes_from_the_zed_hid_descriptor() {
        let root = tempfile::tempdir().unwrap();
        fake_usb_device(root.path(), "1-1", "1d6b", "0002", Some("0000:00:14.0"));
        fake_usb_device(root.path(), "4-2.4.1", ZED_VENDOR_ID, "f682", None);
        fake_usb_device(
            root.path(),
            "1-2.3",
            ZED_VENDOR_ID,
            ZED_HID_PRODUCT_ID,
            Some("10383163"),
        );
        assert_eq!(zed_serial_under(root.path()), Ok(10_383_163));
    }

    #[test]
    fn serial_refuses_zero_or_many_units() {
        let root = tempfile::tempdir().unwrap();
        assert!(
            zed_serial_under(root.path())
                .unwrap_err()
                .contains("no ZED")
        );

        fake_usb_device(
            root.path(),
            "1-2",
            ZED_VENDOR_ID,
            ZED_HID_PRODUCT_ID,
            Some("11111111"),
        );
        fake_usb_device(
            root.path(),
            "1-3",
            ZED_VENDOR_ID,
            ZED_HID_PRODUCT_ID,
            Some("22222222"),
        );
        assert!(
            zed_serial_under(root.path())
                .unwrap_err()
                .contains("ambiguous")
        );
    }

    #[test]
    fn serial_refuses_non_numeric() {
        let root = tempfile::tempdir().unwrap();
        fake_usb_device(
            root.path(),
            "1-2",
            ZED_VENDOR_ID,
            ZED_HID_PRODUCT_ID,
            Some("OV9782"),
        );
        assert!(
            zed_serial_under(root.path())
                .unwrap_err()
                .contains("non-numeric")
        );
    }
}
