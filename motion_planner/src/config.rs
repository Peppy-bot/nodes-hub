//! The node's parameters, parsed once at start: a parameter outside its range stops the
//! node with a message that names it and the range.

use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::time::Duration;

use collision_geometry::GeometryError;
use collision_geometry::chain::Padding;
use control_core::positive_finite::PositiveFinite;
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
        let rate = positive("max_update_rate_hz", params.max_update_rate_hz)?;
        let frame_interval = Duration::try_from_secs_f64(1.0 / rate.get()).or_else(|_| {
            refuse(
                "max_update_rate_hz",
                rate.get(),
                "it is more than 0 and its frame interval, 1 / max_update_rate_hz s, fits the clock",
            )
        })?;
        Ok(Self {
            min_clearance_m: distance("min_clearance_m", params.min_clearance_m)?,
            start_tolerance: positive("start_tolerance_rad", params.start_tolerance_rad)?,
            padding,
            voxel_size: positive("voxel_size_m", params.voxel_size_m)?,
            bounds,
            max_range_m: positive("max_range_m", params.max_range_m)?,
            frame_interval,
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
    // second, and the attempts.
    #[test]
    fn the_launchers_parameters_parse() {
        let config = Config::parse(&parameters()).expect("a config");
        assert_eq!(config.bounds.min, [-0.66, -0.82, 0.04]);
        assert_eq!(config.bounds.max, [0.66, 0.82, 1.36]);
        assert_eq!(config.frame_interval, Duration::from_millis(200));
        assert_eq!(config.planning_attempts.get(), 1);
        assert_eq!(config.start_tolerance.get(), 0.05);
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
            "max_update_rate_hz is 0.00000000000000000001; it is more than 0 and its frame \
             interval, 1 / max_update_rate_hz s, fits the clock"
        );
    }
}
