use std::process::{Child, Command, Stdio};
use std::time::Duration;

/// Virtual camera device using v4l2loopback and ffmpeg
///
/// This helper manages a virtual camera device for testing.
/// Requires v4l2loopback kernel module and ffmpeg to be installed.
///
/// # Setup
/// ```bash
/// sudo apt-get install v4l2loopback-dkms v4l2loopback-utils ffmpeg
/// sudo modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=0 max_buffers=2 card_label="TestCamera"
/// ```
pub struct VirtualCamera {
    ffmpeg_process: Child,
    device_path: String,
    device_nr: u32,
}

impl VirtualCamera {
    /// Create a new virtual camera streaming a test pattern
    ///
    /// # Arguments
    /// * `device_nr` - Device number (e.g., 10 for /dev/video10)
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `fps` - Frames per second
    ///
    /// # Returns
    /// Result containing `VirtualCamera` or error if setup fails
    pub fn new(device_nr: u32, width: u32, height: u32, fps: u32) -> Result<Self, String> {
        let device_path = format!("/dev/video{device_nr}");

        // Check if device exists
        if !std::path::Path::new(&device_path).exists() {
            return Err(format!(
                "Device {device_path} not found. Is v4l2loopback loaded? Run: sudo modprobe v4l2loopback devices=1 video_nr={device_nr}"
            ));
        }

        // Check if ffmpeg is available
        if Command::new("which")
            .arg("ffmpeg")
            .output()
            .ok()
            .is_none_or(|o| !o.status.success())
        {
            return Err("ffmpeg not found. Install with: sudo apt-get install ffmpeg".to_string());
        }

        // Start ffmpeg streaming test pattern
        let ffmpeg_process = Command::new("ffmpeg")
            .args([
                "-re", // Read input at native frame rate
                "-f",
                "lavfi", // Use libavfilter virtual input
                "-i",
                &format!("testsrc=size={width}x{height}:rate={fps}"),
                "-pix_fmt",
                "rgb24", // Pixel format (must match nokhwa's RgbFormat)
                "-f",
                "v4l2", // Output to V4L2 device
                &device_path,
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start ffmpeg: {e}"))?;

        // Wait for stream to be ready - increased for v4l2loopback
        std::thread::sleep(Duration::from_secs(5));

        println!(
            "[VirtualCamera] Started streaming to {device_path} ({width}x{height} @ {fps} fps)"
        );

        Ok(Self {
            ffmpeg_process,
            device_path,
            device_nr,
        })
    }

    /// Create a virtual camera with SMPTE color bars pattern
    pub fn new_with_color_bars(
        device_nr: u32,
        width: u32,
        height: u32,
        fps: u32,
    ) -> Result<Self, String> {
        let device_path = format!("/dev/video{device_nr}");

        if !std::path::Path::new(&device_path).exists() {
            return Err(format!("Device {device_path} not found"));
        }

        let ffmpeg_process = Command::new("ffmpeg")
            .args([
                "-re",
                "-f",
                "lavfi",
                "-i",
                &format!("smptebars=size={width}x{height}:rate={fps}"),
                "-pix_fmt",
                "rgb24",
                "-f",
                "v4l2",
                &device_path,
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start ffmpeg: {e}"))?;

        std::thread::sleep(Duration::from_secs(5));

        println!("[VirtualCamera] Started color bars on {device_path}");

        Ok(Self {
            ffmpeg_process,
            device_path,
            device_nr,
        })
    }

    /// Get the device path (e.g., "/dev/video10")
    pub fn device_path(&self) -> &str {
        &self.device_path
    }

    /// Get the device number
    pub fn device_nr(&self) -> u32 {
        self.device_nr
    }

    /// Check if the ffmpeg process is still running
    pub fn is_running(&mut self) -> bool {
        self.ffmpeg_process
            .try_wait()
            .ok()
            .is_some_and(|status| status.is_none())
    }
}

impl Drop for VirtualCamera {
    fn drop(&mut self) {
        println!("[VirtualCamera] Stopping stream on {}", self.device_path);
        let _ = self.ffmpeg_process.kill();
        let _ = self.ffmpeg_process.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Requires v4l2loopback setup"]
    fn test_virtual_camera_creation() {
        let result = VirtualCamera::new(10, 640, 480, 30);

        match result {
            Ok(mut vcam) => {
                assert_eq!(vcam.device_path(), "/dev/video10");
                assert_eq!(vcam.device_nr(), 10);
                assert!(vcam.is_running(), "FFmpeg process should be running");
                // Drop will clean up
            }
            Err(e) => {
                eprintln!("Skipping test - virtual camera not available: {e}");
            }
        }
    }
}
