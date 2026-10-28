use super::encoding::parse_encoding;
use super::error::{Error, Result};
use camera_core::encoding::ColorEncoding;
use peppygen::Parameters;
use std::io;
use std::num::NonZeroU8;

/// A validated frame rate in 1..=255.
///
/// The contract's `video_stream_info` reports frames per second as a `u8`, so
/// the range is enforced here rather than clamped at the reporting site, and
/// zero is rejected rather than silently coerced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRate(NonZeroU8);

impl FrameRate {
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

/// Complete camera configuration
#[derive(Debug, Clone)]
pub struct CameraConfig {
    pub device_path: String,
    pub resolution: Resolution,
    pub frame_rate: FrameRate,
    /// Encoding requested when opening the camera (hardware wire format)
    pub camera_encoding: ColorEncoding,
    /// Encoding used for published topic frames
    pub topic_encoding: ColorEncoding,
}

impl CameraConfig {
    /// The configuration the node's parameters describe, read in the order
    /// setup checks them: the camera encoding, the topic encoding, then the
    /// frame rate. A refusal is the error setup fails with.
    pub fn from_parameters(params: &Parameters) -> io::Result<Self> {
        let video = &params.video;
        let camera_encoding = parse_encoding(&video.camera_encoding).map_err(|e| {
            io::Error::other(format!(
                "Invalid camera_encoding '{}': {e}",
                video.camera_encoding
            ))
        })?;
        let topic_encoding = parse_encoding(&video.topic_encoding).map_err(|e| {
            io::Error::other(format!(
                "Invalid topic_encoding '{}': {e}",
                video.topic_encoding
            ))
        })?;
        let frame_rate = FrameRate::new(video.frame_rate).map_err(io::Error::other)?;
        Ok(Self {
            device_path: params.device_path.clone(),
            resolution: Resolution::new(video.resolution.width, video.resolution.height),
            frame_rate,
            camera_encoding,
            topic_encoding,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use peppygen::parameters::video::{Video, VideoResolution};

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

    fn parameters(camera_encoding: &str, topic_encoding: &str, frame_rate: u16) -> Parameters {
        Parameters {
            device_path: "/dev/video10".to_string(),
            video: Video {
                camera_encoding: camera_encoding.to_string(),
                frame_rate,
                resolution: VideoResolution {
                    width: 640,
                    height: 480,
                },
                topic_encoding: topic_encoding.to_string(),
            },
        }
    }

    #[test]
    fn the_parameters_describe_the_camera() {
        let config = CameraConfig::from_parameters(&parameters("MJPEG", "rgb8", 30)).unwrap();
        assert_eq!(config.device_path, "/dev/video10");
        assert_eq!(config.resolution, Resolution::new(640, 480));
        assert_eq!(config.frame_rate.as_u8(), 30);
        assert_eq!(config.camera_encoding, ColorEncoding::Mjpeg);
        assert_eq!(config.topic_encoding, ColorEncoding::Rgb8);
    }

    #[test]
    fn each_refusal_names_its_parameter_in_the_order_setup_checks_them() {
        let refusal = |params| {
            CameraConfig::from_parameters(&params)
                .unwrap_err()
                .to_string()
        };
        assert_eq!(
            refusal(parameters("nv12", "z16", 0)),
            "Invalid camera_encoding 'nv12': Invalid encoding 'nv12'. \
             Supported encodings are: rgb8, bgr8, mjpeg, yuyv"
        );
        assert_eq!(
            refusal(parameters("mjpeg", "z16", 0)),
            "Invalid topic_encoding 'z16': Invalid encoding 'z16'. \
             Supported encodings are: rgb8, bgr8, mjpeg, yuyv"
        );
        assert_eq!(
            refusal(parameters("mjpeg", "rgb8", 0)),
            "video.frame_rate must be in 1..=255 (got 0)"
        );
    }
}
