//! The node's parameters, parsed once at start: a parameter outside its range stops the
//! node with a message that names it and the range.

use std::num::{NonZeroU32, NonZeroUsize};
use std::time::Duration;

use control_core::positive_finite::PositiveFinite;
use control_core::time::{DurationError, duration_from_secs, wait_from_secs};
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
    /// The camera of sensor_mounts that the camera link is.
    pub camera_name: String,
    /// How long each moving transform's past stamps are kept, counted back from its
    /// newest.
    pub cache_window: Duration,
    /// How long a depth frame waits for the tree to place the camera at its time.
    pub lookup_timeout: Duration,
}

/// A parameter outside its range.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("{name} is {value}; {rule}")]
    Parameter {
        name: &'static str,
        value: String,
        rule: String,
    },
    /// A duration parameter refused by control_core's parse of seconds.
    #[error("{name} {error}")]
    Duration {
        name: &'static str,
        error: DurationError,
    },
}

impl ConfigError {
    pub fn new(name: &'static str, value: impl ToString, rule: impl Into<String>) -> Self {
        Self::Parameter {
            name,
            value: value.to_string(),
            rule: rule.into(),
        }
    }

    /// The parameter the refusal names.
    #[cfg(test)]
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Parameter { name, .. } | Self::Duration { name, .. } => name,
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
                "it names a camera of sensor_mounts, such as chest",
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
            cache_window: seconds_of("cache_window_s", params.cache_window_s, duration_from_secs)?,
            lookup_timeout: seconds_of(
                "lookup_timeout_s",
                params.lookup_timeout_s,
                wait_from_secs,
            )?,
        })
    }
}

/// A parse of seconds into a duration, refused naming the parameter `name`.
fn seconds_of(
    name: &'static str,
    seconds: f64,
    parse: fn(f64) -> Result<Duration, DurationError>,
) -> Result<Duration, ConfigError> {
    parse(seconds).map_err(|error| ConfigError::Duration { name, error })
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

/// The parameters with the defaults of peppy.json5, and the values the tests give the
/// parameters that have none: the base that tests change.
pub fn default_parameters() -> Parameters {
    Parameters {
        arms_rate_hz: 10.0,
        bind_ip: "127.0.0.1".to_string(),
        cache_window_s: 10.0,
        camera_name: "chest".to_string(),
        cloud_rate_hz: 2.0,
        cloud_stride: 4,
        depth_image: false,
        grpc_port: 9876,
        image_max_width: 640,
        image_rate_hz: 2.0,
        lookup_timeout_s: 60.0,
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
            assert_eq!(error.name(), "scene_poll_hz", "{rate}");
        }
        let mut params = default_parameters();
        params.image_rate_hz = 0.0;
        assert_eq!(Config::parse(&params).unwrap_err().name(), "image_rate_hz");
        let mut params = default_parameters();
        params.cloud_rate_hz = -2.0;
        assert_eq!(Config::parse(&params).unwrap_err().name(), "cloud_rate_hz");
        let mut params = default_parameters();
        params.arms_rate_hz = 0.0;
        assert_eq!(Config::parse(&params).unwrap_err().name(), "arms_rate_hz");
    }

    #[test]
    fn a_stride_a_width_or_a_trail_of_zero_is_refused() {
        let mut params = default_parameters();
        params.cloud_stride = 0;
        assert_eq!(Config::parse(&params).unwrap_err().name(), "cloud_stride");
        let mut params = default_parameters();
        params.image_max_width = 0;
        assert_eq!(
            Config::parse(&params).unwrap_err().name(),
            "image_max_width"
        );
        let mut params = default_parameters();
        params.trail_length = 0;
        assert_eq!(Config::parse(&params).unwrap_err().name(), "trail_length");
    }

    // Each refusal names the parameter in control_core's words: the cache window is above 0,
    // and a depth frame's wait for the robot's tree 0 or above, 0 waiting for nothing and a
    // wait longer than the clock holds the longest one.
    #[test]
    fn the_cache_window_is_above_0_and_the_lookup_timeout_0_or_above() {
        for (window, rule) in [
            (0.0, "must be a positive finite number of seconds, got 0"),
            (-1.0, "must be a positive finite number of seconds, got -1"),
            (
                f64::NAN,
                "must be a positive finite number of seconds, got NaN",
            ),
            (
                f64::INFINITY,
                "must be a positive finite number of seconds, got inf",
            ),
            (1e-12, "1e-12 s rounds to no time at all"),
        ] {
            let mut params = default_parameters();
            params.cache_window_s = window;
            let refused = Config::parse(&params).unwrap_err().to_string();
            assert_eq!(refused, format!("cache_window_s {rule}"));
        }
        for (timeout, rule) in [
            (
                -0.5,
                "must be a finite number of seconds, 0 or more, got -0.5",
            ),
            (
                f64::NAN,
                "must be a finite number of seconds, 0 or more, got NaN",
            ),
            (
                f64::INFINITY,
                "must be a finite number of seconds, 0 or more, got inf",
            ),
        ] {
            let mut params = default_parameters();
            params.lookup_timeout_s = timeout;
            let refused = Config::parse(&params).unwrap_err().to_string();
            assert_eq!(refused, format!("lookup_timeout_s {rule}"));
        }
        let mut params = default_parameters();
        params.lookup_timeout_s = 0.0;
        params.cache_window_s = 0.25;
        let config = Config::parse(&params).expect("no wait and a short window are valid");
        assert_eq!(
            (config.lookup_timeout, config.cache_window),
            (Duration::ZERO, Duration::from_millis(250))
        );
        params.lookup_timeout_s = 1e30;
        let config = Config::parse(&params).expect("a wait past every clock is valid");
        assert_eq!(config.lookup_timeout, Duration::MAX);
    }

    #[test]
    fn an_empty_camera_name_is_refused() {
        let mut params = default_parameters();
        params.camera_name = String::new();
        assert_eq!(Config::parse(&params).unwrap_err().name(), "camera_name");
    }
}
