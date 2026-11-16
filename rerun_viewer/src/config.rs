//! The node's parameters, parsed once at start: a parameter outside its range stops the
//! node with a message that names it and the range.

use std::num::{NonZeroU32, NonZeroUsize};
use std::time::Duration;

use control_core::positive_finite::PositiveFinite;
use peppygen::Parameters;

use crate::canvas::ViewerConfig;

/// The viewer's settings, from the node's parameters.
#[derive(Debug)]
pub struct Config {
    /// Where and how the Rerun viewer serves.
    pub viewer: ViewerConfig,
    /// The least time between two color images, and between two depth images, it draws.
    pub image_interval: Duration,
    /// Whether it draws the depth image.
    pub depth_image: bool,
    /// The widest color image it draws, in pixels: a wider frame is shrunk.
    pub image_max_width: NonZeroU32,
    /// The least time between two snapshots of the arms it draws.
    pub arms_interval: Duration,
    /// The least time between two depth frames it turns into points.
    pub cloud_interval: Duration,
    /// The points come from every `cloud_stride`-th pixel across and down.
    pub cloud_stride: NonZeroU32,
    /// The time between two reads of the planning scene.
    pub scene_poll_interval: Duration,
    /// How many past grasp-point positions each arm's trail holds.
    pub trail_length: NonZeroUsize,
    /// The camera of camera_mounts that the camera link is.
    pub camera_name: String,
}

/// A parameter outside its range.
#[derive(Debug, thiserror::Error, PartialEq)]
#[error("{name} is {value}; {rule}")]
pub struct ConfigError {
    pub name: &'static str,
    pub value: String,
    pub rule: String,
}

impl ConfigError {
    pub fn new(name: &'static str, value: impl ToString, rule: impl Into<String>) -> Self {
        Self {
            name,
            value: value.to_string(),
            rule: rule.into(),
        }
    }
}

impl Config {
    pub fn parse(params: &Parameters) -> Result<Self, ConfigError> {
        let viewer = ViewerConfig::parse(
            &params.bind_ip,
            params.web_port,
            params.grpc_port,
            &params.memory_limit,
            &params.recording_path,
            params.page_stall_timeout_s,
        )?;
        let cloud_stride = NonZeroU32::new(params.cloud_stride).ok_or_else(|| {
            ConfigError::new("cloud_stride", params.cloud_stride, "it is 1 or more")
        })?;
        let image_max_width = NonZeroU32::new(params.image_max_width).ok_or_else(|| {
            ConfigError::new("image_max_width", params.image_max_width, "it is 1 or more")
        })?;
        let trail_length = NonZeroUsize::new(params.trail_length as usize).ok_or_else(|| {
            ConfigError::new("trail_length", params.trail_length, "it is 1 or more")
        })?;
        if params.camera_name.is_empty() {
            return Err(ConfigError::new(
                "camera_name",
                "empty",
                "it names a camera of camera_mounts, such as chest",
            ));
        }
        Ok(Self {
            viewer,
            image_interval: interval_of("image_rate_hz", params.image_rate_hz)?,
            depth_image: params.depth_image,
            image_max_width,
            arms_interval: interval_of("arms_rate_hz", params.arms_rate_hz)?,
            cloud_interval: interval_of("cloud_rate_hz", params.cloud_rate_hz)?,
            cloud_stride,
            scene_poll_interval: interval_of("scene_poll_hz", params.scene_poll_hz)?,
            trail_length,
            camera_name: params.camera_name.clone(),
        })
    }
}

/// The time between two events of a rate of `rate_hz`: a rate above 0 whose interval
/// fits a `Duration` and is not 0 at a resolution of 1 ns.
fn interval_of(name: &'static str, rate_hz: f64) -> Result<Duration, ConfigError> {
    let rule = "it is above 0 and 1 / it seconds is at least 1 ns and fits the clock";
    let rate = PositiveFinite::parse(rate_hz).map_err(|_| ConfigError::new(name, rate_hz, rule))?;
    let interval = Duration::try_from_secs_f64(1.0 / rate.get())
        .map_err(|_| ConfigError::new(name, rate_hz, rule))?;
    if interval.is_zero() {
        return Err(ConfigError::new(name, rate_hz, rule));
    }
    Ok(interval)
}

/// The parameters with the defaults of peppy.json5, the base that tests change.
pub fn default_parameters() -> Parameters {
    Parameters {
        arms_rate_hz: 10.0,
        bind_ip: "127.0.0.1".to_string(),
        camera_name: "chest".to_string(),
        cloud_rate_hz: 2.0,
        cloud_stride: 4,
        depth_image: false,
        grpc_port: 9876,
        image_max_width: 640,
        image_rate_hz: 2.0,
        memory_limit: "8MB".to_string(),
        page_stall_timeout_s: 15.0,
        recording_path: String::new(),
        scene_poll_hz: 2.0,
        trail_length: 100,
        web_port: 9090,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rerun::MemoryLimit;

    #[test]
    fn the_defaults_parse() {
        let config = Config::parse(&default_parameters()).expect("the defaults are valid");
        assert_eq!(config.image_interval, Duration::from_millis(500));
        assert_eq!(config.cloud_interval, Duration::from_millis(500));
        assert_eq!(config.scene_poll_interval, Duration::from_millis(500));
        assert_eq!(config.cloud_stride.get(), 4);
        assert_eq!(config.trail_length.get(), 100);
        assert_eq!(config.camera_name, "chest");
        assert!(!config.depth_image);
        assert_eq!(config.image_max_width.get(), 640);
        assert_eq!(config.arms_interval, Duration::from_millis(100));
        assert_eq!(
            config.viewer.page_stall_timeout,
            Some(Duration::from_secs(15))
        );
        assert_eq!(
            config.viewer.memory_limit,
            MemoryLimit::parse("8MB").unwrap()
        );
    }

    #[test]
    fn a_rate_of_zero_or_less_or_not_a_number_is_refused() {
        for rate in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-320, 2e10] {
            let mut params = default_parameters();
            params.scene_poll_hz = rate;
            let error = Config::parse(&params).expect_err("an unusable rate");
            assert_eq!(error.name, "scene_poll_hz", "{rate}");
        }
        let mut params = default_parameters();
        params.image_rate_hz = 0.0;
        assert_eq!(Config::parse(&params).unwrap_err().name, "image_rate_hz");
        let mut params = default_parameters();
        params.cloud_rate_hz = -2.0;
        assert_eq!(Config::parse(&params).unwrap_err().name, "cloud_rate_hz");
        let mut params = default_parameters();
        params.arms_rate_hz = 0.0;
        assert_eq!(Config::parse(&params).unwrap_err().name, "arms_rate_hz");
    }

    #[test]
    fn a_stride_a_width_or_a_trail_of_zero_is_refused() {
        let mut params = default_parameters();
        params.cloud_stride = 0;
        assert_eq!(Config::parse(&params).unwrap_err().name, "cloud_stride");
        let mut params = default_parameters();
        params.image_max_width = 0;
        assert_eq!(Config::parse(&params).unwrap_err().name, "image_max_width");
        let mut params = default_parameters();
        params.trail_length = 0;
        assert_eq!(Config::parse(&params).unwrap_err().name, "trail_length");
    }

    #[test]
    fn an_empty_camera_name_is_refused() {
        let mut params = default_parameters();
        params.camera_name = String::new();
        assert_eq!(Config::parse(&params).unwrap_err().name, "camera_name");
    }
}
