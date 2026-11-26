//! The node's parameters, parsed into the settings of mapping_core and of the node. A
//! parameter the node does not take stops it at its start, the error naming the parameter.

use std::path::PathBuf;
use std::time::Duration;

use control_core::non_negative_finite::{NonNegativeFinite, NotNonNegativeFinite};
use control_core::positive_finite::{NotPositiveFinite, PositiveFinite};
use control_core::time::{DurationError, duration_from_secs};
use mapping_core::{
    AmclParameterError, AmclParameters, AmclSettings, GlobalLocalizationParameters,
    LocalizationLimits, LocalizationParameters, Mapping, MappingParameters, ParameterError,
    RobotModelType,
};
use peppygen::Parameters;
use serde::Deserialize;
use serde::de::IntoDeserializer;
use serde::de::value::{Error as ValueError, StrDeserializer};
use thiserror::Error;

/// What the node runs with.
#[derive(Debug, Clone)]
pub struct Settings {
    /// Where the maps are saved: an absolute path.
    pub maps_directory: PathBuf,
    /// The padding of describe_base's footprint within which a ray's end is left out.
    pub footprint_padding: NonNegativeFinite,
    /// room_map's source timeout.
    pub source_timeout: Duration,
    pub mapping: MappingParameters,
    /// How often the grid of a map being built is drawn again: room_map's grid period.
    pub map_update_interval: Duration,
    pub localization: LocalizationParameters,
    pub search: GlobalLocalizationParameters,
    /// The seed of the node's one drand48 stream.
    pub random_seed: u32,
}

/// Why a parameter stops the node.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("maps_directory must be an absolute path, got {0:?}")]
    MapsDirectory(String),

    #[error("{name} {error}")]
    Seconds {
        name: &'static str,
        error: DurationError,
    },

    #[error("{name} {error}")]
    NotPositive {
        name: &'static str,
        error: NotPositiveFinite,
    },

    #[error("{name} {error}")]
    Negative {
        name: &'static str,
        error: NotNonNegativeFinite,
    },

    #[error("{group}: {error}")]
    Slam {
        group: &'static str,
        error: ParameterError,
    },

    #[error("amcl: {0}")]
    Amcl(#[from] AmclParameterError),

    #[error(
        "amcl.robot_model_type must be \"nav2_amcl::DifferentialMotionModel\" or \
         \"nav2_amcl::OmniMotionModel\", Nav2's plugin names: {0}"
    )]
    RobotModel(ValueError),
}

impl Settings {
    pub fn parse(params: &Parameters) -> Result<Self, ConfigError> {
        let maps_directory = PathBuf::from(&params.maps_directory);
        if !maps_directory.is_absolute() {
            return Err(ConfigError::MapsDirectory(params.maps_directory.clone()));
        }
        let mapping = slam("mapping", mapping_parameters(&params.mapping))?;
        let localization = LocalizationParameters {
            slam_toolbox: slam(
                "localization",
                MappingParameters {
                    scan_buffer_size: params.localization.scan_buffer_size,
                    loop_match_minimum_chain_size: params
                        .localization
                        .loop_match_minimum_chain_size,
                    ..mapping
                },
            )?,
            matching_limits: limits(
                [
                    "matching_limits.position_std_m",
                    "matching_limits.orientation_std_rad",
                ],
                params.matching_limits.position_std_m,
                params.matching_limits.orientation_std_rad,
            )?,
        };
        Ok(Self {
            maps_directory,
            footprint_padding: NonNegativeFinite::parse(params.footprint_padding_m).map_err(
                |error| ConfigError::Negative {
                    name: "footprint_padding_m",
                    error,
                },
            )?,
            source_timeout: seconds("source_timeout_s", params.source_timeout_s)?,
            mapping,
            map_update_interval: seconds(
                "mapping.map_update_interval",
                params.mapping.map_update_interval,
            )?,
            localization,
            search: GlobalLocalizationParameters {
                amcl: AmclSettings::parse(amcl_parameters(&params.amcl)?)?,
                search_limits: limits(
                    [
                        "search_limits.position_std_m",
                        "search_limits.orientation_std_rad",
                    ],
                    params.search_limits.position_std_m,
                    params.search_limits.orientation_std_rad,
                )?,
                localization,
            },
            random_seed: params.amcl.random_seed,
        })
    }
}

fn seconds(name: &'static str, value: f64) -> Result<Duration, ConfigError> {
    duration_from_secs(value).map_err(|error| ConfigError::Seconds { name, error })
}

/// The limits that the parameters named `names` give, each positive.
fn limits(
    names: [&'static str; 2],
    position_std_m: f64,
    orientation_std_rad: f64,
) -> Result<LocalizationLimits, ConfigError> {
    let positive = |name: &'static str, value: f64| {
        PositiveFinite::parse(value).map_err(|error| ConfigError::NotPositive { name, error })
    };
    Ok(LocalizationLimits {
        position_std_m: positive(names[0], position_std_m)?,
        yaw_std_rad: positive(names[1], orientation_std_rad)?,
    })
}

/// `parameters`, refused as Mapping::start refuses them, named by `group`.
fn slam(
    group: &'static str,
    parameters: MappingParameters,
) -> Result<MappingParameters, ConfigError> {
    Mapping::start(parameters)
        .map(|_| parameters)
        .map_err(|error| ConfigError::Slam { group, error })
}

fn mapping_parameters(m: &peppygen::parameters::mapping::Mapping) -> MappingParameters {
    MappingParameters {
        throttle_scans: m.throttle_scans,
        minimum_time_interval: m.minimum_time_interval,
        check_min_dist_and_heading_precisely: m.check_min_dist_and_heading_precisely,
        min_laser_range: m.min_laser_range,
        max_laser_range: m.max_laser_range,
        resolution: m.resolution,
        use_scan_barycenter: m.use_scan_barycenter,
        minimum_travel_distance: m.minimum_travel_distance,
        minimum_travel_heading: m.minimum_travel_heading,
        scan_buffer_size: m.scan_buffer_size,
        scan_buffer_maximum_scan_distance: m.scan_buffer_maximum_scan_distance,
        link_match_minimum_response_fine: m.link_match_minimum_response_fine,
        link_scan_maximum_distance: m.link_scan_maximum_distance,
        loop_search_maximum_distance: m.loop_search_maximum_distance,
        do_loop_closing: m.do_loop_closing,
        loop_match_minimum_chain_size: m.loop_match_minimum_chain_size,
        loop_match_maximum_variance_coarse: m.loop_match_maximum_variance_coarse,
        loop_match_minimum_response_coarse: m.loop_match_minimum_response_coarse,
        loop_match_minimum_response_fine: m.loop_match_minimum_response_fine,
        correlation_search_space_dimension: m.correlation_search_space_dimension,
        correlation_search_space_resolution: m.correlation_search_space_resolution,
        correlation_search_space_smear_deviation: m.correlation_search_space_smear_deviation,
        loop_search_space_dimension: m.loop_search_space_dimension,
        loop_search_space_resolution: m.loop_search_space_resolution,
        loop_search_space_smear_deviation: m.loop_search_space_smear_deviation,
        distance_variance_penalty: m.distance_variance_penalty,
        angle_variance_penalty: m.angle_variance_penalty,
        fine_search_angle_offset: m.fine_search_angle_offset,
        coarse_search_angle_offset: m.coarse_search_angle_offset,
        coarse_angle_resolution: m.coarse_angle_resolution,
        minimum_angle_penalty: m.minimum_angle_penalty,
        minimum_distance_penalty: m.minimum_distance_penalty,
        use_response_expansion: m.use_response_expansion,
        min_pass_through: m.min_pass_through,
        occupancy_threshold: m.occupancy_threshold,
    }
}

fn amcl_parameters(a: &peppygen::parameters::amcl::Amcl) -> Result<AmclParameters, ConfigError> {
    let plugin: StrDeserializer<ValueError> = a.robot_model_type.as_str().into_deserializer();
    Ok(AmclParameters {
        alpha1: a.alpha1,
        alpha2: a.alpha2,
        alpha3: a.alpha3,
        alpha4: a.alpha4,
        alpha5: a.alpha5,
        laser_likelihood_max_dist: a.laser_likelihood_max_dist,
        laser_max_range: a.laser_max_range,
        laser_min_range: a.laser_min_range,
        max_beams: a.max_beams,
        max_particles: a.max_particles,
        min_particles: a.min_particles,
        pf_err: a.pf_err,
        pf_z: a.pf_z,
        resample_interval: a.resample_interval,
        robot_model_type: RobotModelType::deserialize(plugin).map_err(ConfigError::RobotModel)?,
        sigma_hit: a.sigma_hit,
        update_min_a: a.update_min_a,
        update_min_d: a.update_min_d,
        z_hit: a.z_hit,
        z_rand: a.z_rand,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parameters of `tests/parameters.json` with `change` made to them.
    fn parameters_with(change: impl FnOnce(&mut Parameters)) -> Parameters {
        let mut parameters: Parameters =
            serde_json::from_str(include_str!("../tests/parameters.json"))
                .expect("tests/parameters.json gives every parameter");
        change(&mut parameters);
        parameters
    }

    #[test]
    fn the_test_parameters_parse() {
        assert!(Settings::parse(&parameters_with(|_| {})).is_ok());
    }

    /// A parameter's name and a change that gives it a value the node does not take.
    type Case = (&'static str, fn(&mut Parameters));

    #[test]
    fn a_parameter_the_node_does_not_take_is_refused_by_its_name() {
        let cases: [Case; 8] = [
            ("maps_directory", |p| p.maps_directory = "maps".into()),
            ("footprint_padding_m", |p| p.footprint_padding_m = -0.1),
            ("source_timeout_s", |p| p.source_timeout_s = f64::NAN),
            ("mapping.map_update_interval", |p| {
                p.mapping.map_update_interval = -1.0;
            }),
            ("matching_limits.position_std_m", |p| {
                p.matching_limits.position_std_m = 0.0;
            }),
            ("search_limits.orientation_std_rad", |p| {
                p.search_limits.orientation_std_rad = -0.5;
            }),
            ("mapping", |p| p.mapping.resolution = 0.0),
            ("amcl.robot_model_type", |p| {
                p.amcl.robot_model_type = "differential".into();
            }),
        ];
        for (name, change) in cases {
            let refusal = Settings::parse(&parameters_with(change))
                .expect_err("a refusal")
                .to_string();
            assert!(refusal.contains(name), "{name}: {refusal}");
        }
    }
}
