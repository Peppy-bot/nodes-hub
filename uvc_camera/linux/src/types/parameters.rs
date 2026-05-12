use super::Encoding;
use super::error::{Error, Result};

/// Frame rate with automatic fallback to default for invalid values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRate(u16);

impl FrameRate {
    pub const DEFAULT: u16 = 30;

    /// Create a new frame rate with automatic fallback to default for invalid values
    ///
    /// - Values of 0 fall back to DEFAULT
    pub fn new(fps: u16) -> Self {
        if fps == 0 {
            tracing::warn!("Frame rate is 0, falling back to default {}", Self::DEFAULT);
            return Self(Self::DEFAULT);
        }

        Self(fps)
    }

    pub fn as_u16(&self) -> u16 {
        self.0
    }
}

impl Default for FrameRate {
    fn default() -> Self {
        Self(Self::DEFAULT)
    }
}

impl From<u16> for FrameRate {
    fn from(fps: u16) -> Self {
        Self::new(fps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_rate_valid() {
        let fr = FrameRate::new(30);
        assert_eq!(fr.as_u16(), 30);
    }

    #[test]
    fn test_frame_rate_zero_falls_back() {
        let fr = FrameRate::new(0);
        assert_eq!(fr.as_u16(), FrameRate::DEFAULT);
    }

    #[test]
    fn test_frame_rate_high_value() {
        let fr = FrameRate::new(300);
        assert_eq!(fr.as_u16(), 300);
    }

    #[test]
    fn test_frame_rate_from_u16() {
        let fr = FrameRate::from(60);
        assert_eq!(fr.as_u16(), 60);
    }

    #[test]
    fn test_frame_rate_default() {
        let fr = FrameRate::default();
        assert_eq!(fr.as_u16(), 30);
    }

    #[test]
    fn test_frame_rate_max_u16() {
        let max_fr = FrameRate::new(u16::MAX);
        assert_eq!(max_fr.as_u16(), u16::MAX);
    }
}

/// Resolution - no artificial limits, hardware determines valid values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    width: u32,
    height: u32,
}

impl Resolution {
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
        Self::new(640, 480)
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

        let resolution = Resolution::new(self.width.unwrap_or(640), self.height.unwrap_or(480));

        let frame_rate = FrameRate::new(self.frame_rate.unwrap_or(FrameRate::DEFAULT));

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
