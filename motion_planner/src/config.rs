//! The node's parameters, parsed once at start: a parameter outside its range stops the
//! node with a message that names it and the range.

use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::time::Duration;

use collision_geometry::GeometryError;
use collision_geometry::chain::Padding;
use control_core::positive_finite::PositiveFinite;
use control_core::time::{DurationError, duration_from_secs, wait_from_secs};
use occupancy_map::Bounds;
use peppygen::Parameters;

/// The parameters of the lowest and of the highest corner of the map's box, along x, y
/// and z of the robot frame.
const MAP_MIN: [&str; 3] = ["map_min_x_m", "map_min_y_m", "map_min_z_m"];
const MAP_MAX: [&str; 3] = ["map_max_x_m", "map_max_y_m", "map_max_z_m"];

/// The planner's settings, from the node's parameters.
pub struct Config {
    pub min_clearance_m: f64,
    pub start_tolerance: PositiveFinite,
    pub padding: Padding,
    pub voxel_size: PositiveFinite,
    pub bounds: Bounds,
    pub max_range_m: PositiveFinite,
    /// The least time between two depth frames the map takes in.
    pub frame_interval: Duration,
    /// The time between two messages of occupancy_map.
    pub map_interval: Duration,
    /// How long the robot's tree keeps each moving transform's stamps, counted back from
    /// the newest stamp of its child frame.
    pub cache_window: Duration,
    /// How long a depth frame waits for the robot's tree to place the camera at its stamp.
    pub lookup_timeout: Duration,
    pub goal_samples: NonZeroUsize,
    pub max_iterations: NonZeroU64,
    pub planning_attempts: NonZeroU32,
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
    #[error("padding_scale and padding_offset_m: {0}")]
    Padding(GeometryError),
    /// A duration parameter refused by control_core's parse of seconds.
    #[error("{name} {error}")]
    Duration {
        name: &'static str,
        error: DurationError,
    },
}

fn refuse<T>(
    name: &'static str,
    value: impl ToString,
    rule: impl Into<String>,
) -> Result<T, ConfigError> {
    Err(ConfigError::Parameter {
        name,
        value: value.to_string(),
        rule: rule.into(),
    })
}

fn distance(name: &'static str, value: f64) -> Result<f64, ConfigError> {
    if value.is_finite() && value >= 0.0 {
        return Ok(value);
    }
    refuse(name, value, "it is a finite distance of 0 or more (m)")
}

fn positive(name: &'static str, value: f64) -> Result<PositiveFinite, ConfigError> {
    PositiveFinite::parse(value)
        .or_else(|_| refuse(name, value, "it is a finite number more than 0"))
}

fn at_least_one<T>(name: &'static str, value: Option<T>) -> Result<T, ConfigError> {
    value.map_or_else(|| refuse(name, 0, "it is 1 or more"), Ok)
}

/// The time between two events at `name`, a rate a second: refused unless the rate is a
/// finite number above 0 whose interval fits the clock.
fn interval(name: &'static str, rate_hz: f64) -> Result<Duration, ConfigError> {
    let rate = positive(name, rate_hz)?;
    Duration::try_from_secs_f64(1.0 / rate.get()).or_else(|_| {
        refuse(
            name,
            rate.get(),
            format!("it is more than 0 and its interval, 1 / {name} s, fits the clock"),
        )
    })
}

/// A parse of seconds into a duration, refused naming the parameter `name`.
fn seconds_of(
    name: &'static str,
    seconds: f64,
    parse: fn(f64) -> Result<Duration, DurationError>,
) -> Result<Duration, ConfigError> {
    parse(seconds).map_err(|error| ConfigError::Duration { name, error })
}

/// The map's box from its lowest and its highest corner: each coordinate finite, and
/// each of the lowest below the highest's.
fn map_bounds(min: [f64; 3], max: [f64; 3]) -> Result<Bounds, ConfigError> {
    for axis in 0..3 {
        if !min[axis].is_finite() {
            return refuse(MAP_MIN[axis], min[axis], "it is a finite number (m)");
        }
        if !max[axis].is_finite() {
            return refuse(MAP_MAX[axis], max[axis], "it is a finite number (m)");
        }
        if min[axis] >= max[axis] {
            return refuse(
                MAP_MIN[axis],
                min[axis],
                format!("it is below {}, {} (m)", MAP_MAX[axis], max[axis]),
            );
        }
    }
    Ok(Bounds { min, max })
}

impl Config {
    pub fn parse(params: &Parameters) -> Result<Self, ConfigError> {
        let padding = Padding::new(params.padding_scale, params.padding_offset_m)
            .map_err(ConfigError::Padding)?;
        let bounds = map_bounds(
            [params.map_min_x_m, params.map_min_y_m, params.map_min_z_m],
            [params.map_max_x_m, params.map_max_y_m, params.map_max_z_m],
        )?;
        let frame_interval = interval("max_update_rate_hz", params.max_update_rate_hz)?;
        let map_interval = interval("occupancy_map_rate_hz", params.occupancy_map_rate_hz)?;
        Ok(Self {
            min_clearance_m: distance("min_clearance_m", params.min_clearance_m)?,
            start_tolerance: positive("start_tolerance_rad", params.start_tolerance_rad)?,
            padding,
            voxel_size: positive("voxel_size_m", params.voxel_size_m)?,
            bounds,
            max_range_m: positive("max_range_m", params.max_range_m)?,
            frame_interval,
            map_interval,
            cache_window: seconds_of("cache_window_s", params.cache_window_s, duration_from_secs)?,
            lookup_timeout: seconds_of(
                "lookup_timeout_s",
                params.lookup_timeout_s,
                wait_from_secs,
            )?,
            goal_samples: at_least_one(
                "goal_samples",
                NonZeroUsize::new(params.goal_samples as usize),
            )?,
            max_iterations: at_least_one("max_iterations", NonZeroU64::new(params.max_iterations))?,
            planning_attempts: at_least_one(
                "planning_attempts",
                NonZeroU32::new(params.planning_attempts),
            )?,
        })
    }
}

/// The parameters the OpenArm v2 launchers give the planner, the same file the harness
/// tests reach as `common::parameters`.
#[cfg(test)]
#[path = "../tests/common/parameters.rs"]
mod launched;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) use super::launched::parameters;

    fn refused(params: Parameters) -> String {
        Config::parse(&params).err().expect("a refusal").to_string()
    }

    // The launchers' parameters parse: the map's box, a frame interval of a fifth of a
    // second, a second between two messages of the map, and the attempts.
    #[test]
    fn the_launchers_parameters_parse() {
        let config = Config::parse(&parameters()).expect("a config");
        assert_eq!(config.bounds.min, [-0.66, -0.82, 0.04]);
        assert_eq!(config.bounds.max, [0.66, 0.82, 1.36]);
        assert_eq!(config.frame_interval, Duration::from_millis(200));
        assert_eq!(config.map_interval, Duration::from_secs(1));
        assert_eq!(config.planning_attempts.get(), 1);
        assert_eq!(config.start_tolerance.get(), 0.05);
        assert_eq!(config.cache_window, Duration::from_secs(10));
        assert_eq!(config.lookup_timeout, Duration::from_millis(100));
    }

    // The cache window is above 0, and a depth frame's wait for the robot's tree 0 or
    // above, 0 waiting for nothing and a wait longer than the clock holds the longest one;
    // each refusal names the parameter in control_core's words.
    #[test]
    fn the_cache_window_is_above_0_and_the_lookup_timeout_0_or_above() {
        for window in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-12] {
            let refused_window = refused(Parameters {
                cache_window_s: window,
                ..parameters()
            });
            let error = duration_from_secs(window).expect_err("a window refused");
            assert_eq!(refused_window, format!("cache_window_s {error}"));
        }
        for timeout in [-0.1, f64::NAN, f64::INFINITY] {
            let refused_timeout = refused(Parameters {
                lookup_timeout_s: timeout,
                ..parameters()
            });
            assert_eq!(
                refused_timeout,
                format!(
                    "lookup_timeout_s must be a finite number of seconds, 0 or more, got {timeout}"
                )
            );
        }
        let wait = |lookup_timeout_s| {
            Config::parse(&Parameters {
                lookup_timeout_s,
                ..parameters()
            })
            .expect("a lookup timeout taken")
            .lookup_timeout
        };
        assert_eq!(wait(0.0), Duration::ZERO);
        assert_eq!(wait(f64::MAX), Duration::MAX);
    }

    // Each parameter outside its range stops the node with a message that names it and
    // says what it takes.
    #[test]
    fn a_parameter_outside_its_range_is_named() {
        let flat = Parameters {
            map_max_z_m: 0.0,
            ..parameters()
        };
        assert_eq!(
            refused(flat),
            "map_min_z_m is 0.04; it is below map_max_z_m, 0 (m)"
        );
        let endless = Parameters {
            map_max_x_m: f64::INFINITY,
            ..parameters()
        };
        assert_eq!(
            refused(endless),
            "map_max_x_m is inf; it is a finite number (m)"
        );
        let unscaled = Parameters {
            padding_scale: 0.0,
            ..parameters()
        };
        assert_eq!(
            refused(unscaled),
            "padding_scale and padding_offset_m: padding needs a positive finite scale and a \
             non-negative finite offset, got scale 0 and offset 0.04"
        );
        let none = Parameters {
            goal_samples: 0,
            ..parameters()
        };
        assert_eq!(refused(none), "goal_samples is 0; it is 1 or more");
        let no_attempt = Parameters {
            planning_attempts: 0,
            ..parameters()
        };
        assert_eq!(
            refused(no_attempt),
            "planning_attempts is 0; it is 1 or more"
        );
        let negative = Parameters {
            min_clearance_m: -0.01,
            ..parameters()
        };
        assert_eq!(
            refused(negative),
            "min_clearance_m is -0.01; it is a finite distance of 0 or more (m)"
        );
        let exact = Parameters {
            start_tolerance_rad: 0.0,
            ..parameters()
        };
        assert_eq!(
            refused(exact),
            "start_tolerance_rad is 0; it is a finite number more than 0"
        );
        let still = Parameters {
            max_update_rate_hz: 0.0,
            ..parameters()
        };
        assert_eq!(
            refused(still),
            "max_update_rate_hz is 0; it is a finite number more than 0"
        );
        let glacial = Parameters {
            max_update_rate_hz: 1e-20,
            ..parameters()
        };
        assert_eq!(
            refused(glacial),
            "max_update_rate_hz is 0.00000000000000000001; it is more than 0 and its \
             interval, 1 / max_update_rate_hz s, fits the clock"
        );
        let unpublished = Parameters {
            occupancy_map_rate_hz: f64::NAN,
            ..parameters()
        };
        assert_eq!(
            refused(unpublished),
            "occupancy_map_rate_hz is NaN; it is a finite number more than 0"
        );
        let rare = Parameters {
            occupancy_map_rate_hz: 1e-20,
            ..parameters()
        };
        assert_eq!(
            refused(rare),
            "occupancy_map_rate_hz is 0.00000000000000000001; it is more than 0 and its \
             interval, 1 / occupancy_map_rate_hz s, fits the clock"
        );
    }
}
