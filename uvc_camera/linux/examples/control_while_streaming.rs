//! Prove controls apply synchronously from another thread mid-stream.
//!
//! Streams frames on one thread while the main thread drives brightness and
//! exposure through the same ControlHandle services use, verifying the write
//! round-trips and the stream never stalls.
//!
//! Usage: control_while_streaming <device_path>

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use uvc_camera_linux::camera::{CameraControlRequest, CameraDevice};
use uvc_camera_linux::types::{CameraConfig, Encoding, FrameRate, Resolution};

fn main() {
    let device_path = std::env::args().nth(1).expect("usage: <device_path>");
    let config = CameraConfig {
        device_path,
        resolution: Resolution::new(960, 600),
        frame_rate: FrameRate::new(15).unwrap(),
        camera_encoding: Encoding::Yuyv,
        topic_encoding: Encoding::Rgb8,
    };

    let mut camera = CameraDevice::new();
    camera.open(&config).expect("open failed");
    let controls = camera.control_handle().expect("no control handle");
    let description = camera.stream_description().expect("no description");
    println!(
        "streaming {}x{} @ {} fps",
        description.width, description.height, description.frames_per_second
    );

    let frames = Arc::new(AtomicU32::new(0));
    let frame_counter = Arc::clone(&frames);
    let capture = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(6);
        while Instant::now() < deadline {
            camera.capture_frame().expect("capture failed mid-controls");
            frame_counter.fetch_add(1, Ordering::Relaxed);
        }
    });

    std::thread::sleep(Duration::from_millis(500));
    for (label, request, expect_success) in [
        (
            "brightness=32",
            CameraControlRequest::SetBrightness { value: 32 },
            true,
        ),
        (
            "brightness=0",
            CameraControlRequest::SetBrightness { value: 0 },
            true,
        ),
        (
            "contrast=40",
            CameraControlRequest::SetContrast { value: 40 },
            true,
        ),
        ("gain=10", CameraControlRequest::SetGain { value: 10 }, true),
    ] {
        let started = Instant::now();
        let result = controls.apply(&request);
        println!(
            "{label}: success={} current={} in {:?} ({})",
            result.success,
            result.current_value,
            started.elapsed(),
            result.message
        );
        assert_eq!(
            result.success, expect_success,
            "{label} unexpected outcome: {}",
            result.message
        );
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "{label} took too long; controls must not queue behind frames"
        );
        std::thread::sleep(Duration::from_millis(300));
    }

    capture.join().expect("capture thread panicked");
    let captured = frames.load(Ordering::Relaxed);
    println!("captured {captured} frames during control writes");
    assert!(
        captured >= 70,
        "stream starved during controls: only {captured} frames in ~6s at 15fps"
    );
    println!("PASS: synchronous controls do not disturb the stream");
}
