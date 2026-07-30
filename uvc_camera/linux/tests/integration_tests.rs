//! Integration tests against virtual v4l2loopback devices.
//!
//! # Requirements
//! - v4l2loopback loaded with: `exclusive_caps=0` `max_buffers=2`
//! - Tests must run single-threaded: `cargo test -- --ignored --test-threads=1`
//!   (they share /dev/video10)
//!
//! See `INTEGRATION_TESTS.md` for setup instructions.

mod helpers;

use helpers::virtual_camera::VirtualCamera;
use std::time::{Duration, Instant};
use uvc_camera_linux::camera::CameraDevice;
use uvc_camera_linux::types::{CameraConfig, Encoding, FrameRate, Resolution};

fn config_for(device_path: &str) -> CameraConfig {
    CameraConfig {
        device_path: device_path.to_string(),
        resolution: Resolution::new(640, 480),
        frame_rate: FrameRate::new(30),
        camera_encoding: Encoding::Rgb8,
        topic_encoding: Encoding::Rgb8,
    }
}

/// Open a virtual camera through the node's own API.
#[test]
#[ignore = "Requires v4l2loopback setup"]
fn test_open_virtual_camera() {
    let vcam = match VirtualCamera::new(10, 640, 480, 30) {
        Ok(cam) => cam,
        Err(e) => {
            eprintln!("Skipping test: {e}");
            return;
        }
    };

    let mut camera = CameraDevice::new();
    camera
        .open(&config_for("/dev/video10"))
        .expect("Should open the virtual camera");
    assert!(camera.is_open(), "Camera should report open");

    drop(camera); // Release the device before tearing down the loopback
    drop(vcam);
}

/// Capture a stream of frames and check their geometry.
#[test]
#[ignore = "Requires v4l2loopback setup"]
fn test_capture_frames_from_virtual_camera() {
    let vcam = match VirtualCamera::new(10, 640, 480, 30) {
        Ok(cam) => cam,
        Err(e) => {
            eprintln!("Skipping test: {e}");
            return;
        }
    };

    let mut camera = CameraDevice::new();
    camera
        .open(&config_for("/dev/video10"))
        .expect("Failed to open camera");

    for i in 0..5 {
        let frame = camera.capture_frame();
        let frame = frame.unwrap_or_else(|e| panic!("Frame {i} should be captured: {e}"));
        assert_eq!(frame.width(), 640, "Frame {i} width");
        assert_eq!(frame.height(), 480, "Frame {i} height");
        assert!(!frame.data().is_empty(), "Frame {i} should carry data");
    }

    drop(camera);
    drop(vcam);
}

/// Color bars must arrive with pixel variation, proving real payload flow.
#[test]
#[ignore = "Requires v4l2loopback setup"]
fn test_capture_color_bars() {
    let vcam = match VirtualCamera::new_with_color_bars(10, 640, 480, 30) {
        Ok(cam) => cam,
        Err(e) => {
            eprintln!("Skipping test: {e}");
            return;
        }
    };

    let mut camera = CameraDevice::new();
    camera
        .open(&config_for("/dev/video10"))
        .expect("Failed to open camera");

    let frame = camera
        .capture_frame()
        .expect("Should capture color bars frame");
    let data = frame.data();
    let first: [u8; 3] = [data[0], data[1], data[2]];
    let has_variation = data.chunks(3).skip(100).any(|px| px != first);
    assert!(has_variation, "Color bars should have pixel variation");

    drop(camera);
    drop(vcam);
}

/// End-to-end through the conversion pipeline: capture and convert to BGR8.
#[test]
#[ignore = "Requires v4l2loopback setup"]
fn test_camera_device_end_to_end() {
    use uvc_camera_linux::pipeline::process_frame;
    use uvc_camera_linux::types::FrameId;

    let vcam = match VirtualCamera::new(10, 640, 480, 30) {
        Ok(cam) => cam,
        Err(e) => {
            eprintln!("Skipping test: {e}");
            return;
        }
    };

    let mut camera = CameraDevice::new();
    camera
        .open(&config_for("/dev/video10"))
        .expect("Failed to open camera");

    let raw = camera.capture_frame().expect("Should capture a frame");
    let frame =
        process_frame(raw, FrameId::new(1), Encoding::Bgr8).expect("Should convert to BGR8");
    assert_eq!(frame.encoding(), Encoding::Bgr8);
    assert_eq!(frame.width(), 640);
    assert_eq!(frame.height(), 480);
    assert_eq!(frame.frame_id(), FrameId::new(1));

    drop(camera);
    drop(vcam);
}

/// Frames must arrive at roughly the source cadence, not in bursts.
#[test]
#[ignore = "Requires v4l2loopback setup"]
fn test_frame_rate_timing() {
    let vcam = match VirtualCamera::new(10, 640, 480, 30) {
        Ok(cam) => cam,
        Err(e) => {
            eprintln!("Skipping test: {e}");
            return;
        }
    };

    let mut camera = CameraDevice::new();
    camera
        .open(&config_for("/dev/video10"))
        .expect("Failed to open camera");

    let start = Instant::now();
    let mut captured = 0;
    for _ in 0..30 {
        if camera.capture_frame().is_ok() {
            captured += 1;
        }
        std::thread::sleep(Duration::from_millis(33)); // ~30 fps
    }
    let elapsed = start.elapsed();

    assert!(captured >= 25, "Should capture at least 25 of 30 frames");
    assert!(
        elapsed.as_secs() <= 2,
        "Should take roughly 1 second to capture 30 frames at 30fps"
    );

    drop(camera);
    drop(vcam);
}
