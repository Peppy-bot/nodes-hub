use super::Encoding;
use super::error::{Error, Result};
use std::num::NonZeroU8;

/// A validated frame rate in 1..=255.
///
/// The contract's `video_stream_info` reports frames per second as a `u8`, so
/// the range is enforced here rather than clamped at the reporting site, and
/// zero is rejected rather than silently coerced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRate(NonZeroU8);

impl FrameRate {
    pub const DEFAULT: u16 = 30;

    /// Parse a frame rate, rejecting values outside 1..=255.
    pub fn new(fps: u16) -> Result<Self> {
        u8::try_from(fps)
            .ok()
            .and_then(NonZeroU8::new)
            .map(Self)
            .ok_or_else(|| Error::Other(format!("video.frame_rate must be in 1..=255 (got {fps})")))
    }

    pub fn as_u16(&self) -> u16 {
        u16::from(self.0.get())
    }

    pub fn as_u8(&self) -> u8 {
        self.0.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_rate_accepts_valid_range() {
        assert_eq!(FrameRate::new(1).unwrap().as_u8(), 1);
        assert_eq!(FrameRate::new(30).unwrap().as_u16(), 30);
        assert_eq!(FrameRate::new(255).unwrap().as_u8(), 255);
    }

    #[test]
    fn test_frame_rate_rejects_zero() {
        // The contract reports fps as u8 and a zero rate is meaningless, so
        // both ends of the range are hard errors, not silent coercions.
        let err = FrameRate::new(0).unwrap_err();
        assert!(err.to_string().contains("must be in 1..=255 (got 0)"));
    }

    #[test]
    fn test_frame_rate_rejects_over_u8() {
        let err = FrameRate::new(300).unwrap_err();
        assert!(err.to_string().contains("must be in 1..=255 (got 300)"));
    }
}

/// Resolution - no artificial limits, hardware determines valid values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    width: u32,
    height: u32,
}

impl Resolution {
    pub const DEFAULT_WIDTH: u32 = 640;
    pub const DEFAULT_HEIGHT: u32 = 480;

    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

impl Default for Resolution {
    fn default() -> Self {
        Self::new(Self::DEFAULT_WIDTH, Self::DEFAULT_HEIGHT)
    }
}

/// Complete camera configuration
#[derive(Debug, Clone)]
pub struct CameraConfig {
    pub device_path: String,
    pub resolution: Resolution,
    pub frame_rate: FrameRate,
    /// Encoding requested when opening the camera (hardware wire format)
    pub camera_encoding: Encoding,
    /// Encoding used for published topic frames
    pub topic_encoding: Encoding,
}

impl CameraConfig {
    pub fn new(
        device_path: String,
        resolution: Resolution,
        frame_rate: FrameRate,
        camera_encoding: Encoding,
        topic_encoding: Encoding,
    ) -> Self {
        Self {
            device_path,
            resolution,
            frame_rate,
            camera_encoding,
            topic_encoding,
        }
    }
}

/// Builder for CameraConfig with validation
#[derive(Default)]
pub struct CameraConfigBuilder {
    device_path: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    frame_rate: Option<u16>,
    camera_encoding: Option<Encoding>,
    topic_encoding: Option<Encoding>,
}

impl CameraConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn device_path(mut self, path: String) -> Self {
        self.device_path = Some(path);
        self
    }

    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    pub fn frame_rate(mut self, fps: u16) -> Self {
        self.frame_rate = Some(fps);
        self
    }

    pub fn camera_encoding(mut self, encoding: Encoding) -> Self {
        self.camera_encoding = Some(encoding);
        self
    }

    pub fn topic_encoding(mut self, encoding: Encoding) -> Self {
        self.topic_encoding = Some(encoding);
        self
    }

    pub fn build(self) -> Result<CameraConfig> {
        let device_path = self
            .device_path
            .ok_or_else(|| Error::Other("Device path is required".to_string()))?;

        let resolution = Resolution::new(
            self.width.unwrap_or(Resolution::DEFAULT_WIDTH),
            self.height.unwrap_or(Resolution::DEFAULT_HEIGHT),
        );

        let frame_rate = FrameRate::new(self.frame_rate.unwrap_or(FrameRate::DEFAULT))?;

        let camera_encoding = self.camera_encoding.unwrap_or(Encoding::Mjpeg);
        let topic_encoding = self.topic_encoding.unwrap_or(Encoding::Rgb8);

        Ok(CameraConfig::new(
            device_path,
            resolution,
            frame_rate,
            camera_encoding,
            topic_encoding,
        ))
    }
}
