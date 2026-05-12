//! Integration tests for UVC camera with virtual v4l2loopback devices.
//!
//! # Requirements
//! - ffmpeg must stream in rgb24 pixel format to match nokhwa's `RgbFormat`
//! - v4l2loopback should be loaded with: `exclusive_caps=0` `max_buffers=2`
//! - Camera object must be dropped before `VirtualCamera` to avoid device conflicts
//! - Tests must run single-threaded: `cargo test -- --ignored --test-threads=1`
//!   (to avoid multiple tests accessing the same /dev/video10 device)
//!
//! See `INTEGRATION_TESTS.md` for setup instructions.

mod helpers;

use helpers::virtual_camera::VirtualCamera;
use nokhwa::Camera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution};
use std::time::Duration;

/// Integration test: Verify we can open and capture from a virtual camera
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

    let camera_index = CameraIndex::Index(10);
    let requested_format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);

    let camera = Camera::new(camera_index, requested_format);
    assert!(camera.is_ok(), "Should successfully open virtual camera");

    drop(vcam); // Cleanup
}

/// Integration test: Capture frames from virtual camera
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

    let camera_index = CameraIndex::Index(10);
    let requested_format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);

    let mut camera = Camera::new(camera_index, requested_format).expect("Failed to open camera");

    // Set resolution
    camera
        .set_resolution(Resolution::new(640, 480))
        .expect("Failed to set resolution");

    // Open stream
    camera.open_stream().expect("Failed to open stream");

    // Capture a few frames
    for i in 0..5 {
        let frame = camera.frame();
        assert!(frame.is_ok(), "Frame {i} should be captured successfully");

        if let Ok(frame) = frame {
            let rgb_frame = frame.decode_image::<RgbFormat>();
            assert!(rgb_frame.is_ok(), "Frame {i} should decode to RGB");

            if let Ok(img) = rgb_frame {
                let data = img.into_raw();
                // 640 * 480 * 3 bytes (RGB)
                assert_eq!(
                    data.len(),
                    640 * 480 * 3,
                    "Frame {i} should have correct size"
                );
            }
        }

        std::thread::sleep(Duration::from_millis(100));
    }

    drop(camera); // Drop camera first to release device
    drop(vcam); // Cleanup
}

/// Integration test: Test with color bars pattern
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

    let camera_index = CameraIndex::Index(10);
    let requested_format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);

    let mut camera = Camera::new(camera_index, requested_format).expect("Failed to open camera");

    camera
        .set_resolution(Resolution::new(640, 480))
        .expect("Failed to set resolution");

    camera.open_stream().expect("Failed to open stream");

    // Capture frame with color bars
    let frame = camera.frame().expect("Should capture color bars frame");

    let rgb_frame = frame
        .decode_image::<RgbFormat>()
        .expect("Should decode color bars");

    let data = rgb_frame.into_raw();
    assert_eq!(data.len(), 640 * 480 * 3);

    // Color bars should have non-zero, varied pixel values
    let mut has_variation = false;
    let first_pixel = [data[0], data[1], data[2]];
    for chunk in data.chunks(3).skip(100) {
        if chunk[0] != first_pixel[0] || chunk[1] != first_pixel[1] || chunk[2] != first_pixel[2] {
            has_variation = true;
            break;
        }
    }
    assert!(has_variation, "Color bars should have pixel variation");

    drop(camera); // Drop camera first to release device
    drop(vcam);
}

/// Integration test: Capture from virtual camera using NokhwaCamera
#[test]
#[ignore = "Requires v4l2loopback setup"]
fn test_nokhwa_camera_end_to_end() {
    use uvc_camera::camera::CameraDevice;
    use uvc_camera::camera::NokhwaCamera;
    use uvc_camera::types::{CameraConfig, Encoding, FrameRate, Resolution};

    let vcam = match VirtualCamera::new(10, 640, 480, 30) {
        Ok(cam) => cam,
        Err(e) => {
            eprintln!("Skipping test: {e}");
            return;
        }
    };

    // Test end-to-end: open camera and capture frame using our API
    let mut camera = NokhwaCamera::new();
    let config = CameraConfig {
        device_path: "/dev/video10".to_string(),
        resolution: Resolution::new(640, 480),
        frame_rate: FrameRate::new(30),
        camera_encoding: Encoding::Mjpeg,
        topic_encoding: Encoding::Rgb8,
    };

    camera.open(&config).expect("Failed to open camera");

    assert!(camera.is_open(), "Camera should be open");

    let frame = camera.capture_frame();
    assert!(frame.is_ok(), "Should capture frame from virtual camera");

    let frame = frame.unwrap();
    assert_eq!(frame.width(), 640);
    assert_eq!(frame.height(), 480);
    assert!(!frame.data().is_empty(), "Frame data should not be empty");

    drop(camera);
    drop(vcam);
}

/// Integration test: Test frame rate
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

    let camera_index = CameraIndex::Index(10);
    let requested_format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);

    let mut camera = Camera::new(camera_index, requested_format).expect("Failed to open camera");

    camera
        .set_resolution(Resolution::new(640, 480))
        .expect("Failed to set resolution");

    camera.set_frame_rate(30).expect("Failed to set frame rate");

    camera.open_stream().expect("Failed to open stream");

    // Capture 30 frames and measure time
    let start = std::time::Instant::now();
    let mut captured = 0;

    for _ in 0..30 {
        if camera.frame().is_ok() {
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

    drop(camera); // Drop camera first to release device
    drop(vcam);
}
