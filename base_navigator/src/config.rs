//! The navigator's parameters, every one from the launcher, parsed into the
//! values each part takes, or the refusal that names the parameter and what
//! it takes. The setup refuses a grid whose voxel map cannot be built, and
//! warns of a controller that slows on its approach from farther than half
//! the grid's side, where the grid holds no obstacle ahead of the goal.

use std::num::NonZeroU32;
use std::time::Duration;

use control_core::non_negative_finite::NonNegativeFinite;
use control_core::positive_finite::PositiveFinite;
use control_core::time::{
    SecondsParameterError, duration_parameter, nanoseconds, seconds, wait_parameter,
};
use navigation_core::controller::{InvertedLookahead, VelocityScaledLookahead};
use navigation_core::inflation::InflationParams;
use navigation_core::obstacles::{ObstacleHeights, ObstacleHeightsError, ObstacleHeightsParams};
use navigation_core::path_handler::PathHandlerParams;
use navigation_core::planner::{
    CostTravelMultiplier, CostTravelMultiplierError, PlannerParams, SearchParams, SmootherParams,
};
use navigation_core::progress_checker::ProgressCheckerParams;
use occupancy_map::{MapError, VoxelMap};
use peppygen::Parameters;

use crate::follow::odometry_frame;
use crate::grid::bounds_around;
use crate::recovery::Stepping;

/// The controller rate (Hz) the navigator steps above: drive_link brings the
/// base to rest 0.5 s after a leader's last setpoint, so a goal streams more
/// often than twice a second, which also gives the feedback its second.
const SLOWEST_CONTROLLER_RATE_HZ: f64 = 2.0;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("{name} must be a positive number, got {value}")]
    NotPositive { name: &'static str, value: f64 },

    #[error("{name} must be a number of at least 0, got {value}")]
    Negative { name: &'static str, value: f64 },

    #[error("{name} must be at least 1, got 0")]
    Zero { name: &'static str },

    #[error(transparent)]
    Seconds(#[from] SecondsParameterError),

    #[error("the obstacle heights")]
    Heights(#[from] ObstacleHeightsError),

    #[error("the controller's lookahead")]
    Lookahead(#[from] InvertedLookahead),

    #[error("planner.cost_travel_multiplier")]
    CostTravel(#[from] CostTravelMultiplierError),

    #[error(
        "controller_rate_hz must lie above {SLOWEST_CONTROLLER_RATE_HZ} Hz, as drive_link brings \
         the base to rest 0.5 s after a leader's last setpoint, got {rate_hz} Hz"
    )]
    ControllerTooSlow { rate_hz: f64 },

    #[error(
        "the grid of the floor, grid.side_m on a side and from grid.min_z_m up to \
         max_obstacle_height_m in voxels of grid.voxel_size_m"
    )]
    Grid(#[from] MapError),
}

fn positive(name: &'static str, value: f64) -> Result<PositiveFinite, ConfigError> {
    PositiveFinite::parse(value).map_err(|_| ConfigError::NotPositive { name, value })
}

fn non_negative(name: &'static str, value: f64) -> Result<NonNegativeFinite, ConfigError> {
    NonNegativeFinite::parse(value).map_err(|_| ConfigError::Negative { name, value })
}

fn at_least_one(name: &'static str, value: u32) -> Result<NonZeroU32, ConfigError> {
    NonZeroU32::new(value).ok_or(ConfigError::Zero { name })
}

/// The nanoseconds of the parameter `name`, `secs` seconds above 0.
fn positive_ns(name: &'static str, secs: f64) -> Result<i64, ConfigError> {
    Ok(duration_ns(duration_parameter(name, secs)?))
}

/// The nanoseconds of `duration`, at most the largest an `i64` holds.
pub fn duration_ns(duration: Duration) -> i64 {
    i64::try_from(duration.as_nanos()).unwrap_or(i64::MAX)
}

/// What the grid of the floor takes, but the footprint, which describe_base
/// gives.
#[derive(Clone, Copy, Debug)]
pub struct GridSettings {
    pub side_m: PositiveFinite,
    pub voxel_m: PositiveFinite,
    pub min_z_m: f64,
    pub heights: ObstacleHeights,
    /// How far from a camera's origin in the plane a return marks.
    pub obstacle_max_range_m: PositiveFinite,
    pub footprint_padding: NonNegativeFinite,
    pub inflation: InflationParams,
}

impl GridSettings {
    /// Half the grid's side (m): how far the square reaches from its center
    /// along each axis.
    pub fn half_side_m(&self) -> f64 {
        self.side_m.get() / 2.0
    }

    /// Whether `point` lies inside the square centered on `center`, both in
    /// odom.
    pub fn square_holds(&self, center: [f64; 2], point: [f64; 2]) -> bool {
        let half = self.half_side_m();
        (0..2).all(|i| (point[i] - center[i]).abs() <= half)
    }
}

/// Regulated pure pursuit's settings but its speed, which each navigation
/// sets.
#[derive(Clone, Copy, Debug)]
pub struct ControllerSettings {
    pub lookahead: VelocityScaledLookahead,
    pub rotate_to_heading_angular_vel_rad_s: PositiveFinite,
    pub rotate_to_heading_min_angle_rad: NonNegativeFinite,
    pub max_angular_accel_rad_s2: PositiveFinite,
    pub min_approach_linear_velocity_m_s: PositiveFinite,
    pub approach_velocity_scaling_dist_m: NonNegativeFinite,
    pub regulated_linear_scaling_min_radius_m: NonNegativeFinite,
    pub regulated_linear_scaling_min_speed_m_s: NonNegativeFinite,
    pub cost_scaling_dist_m: PositiveFinite,
    pub cost_scaling_gain: NonNegativeFinite,
    /// The arc look-ahead time of base_navigation's follow (s).
    pub max_allowed_time_to_collision_up_to_carrot_s: NonNegativeFinite,
    pub controller_frequency_hz: PositiveFinite,
}

/// How a navigation follows its path: the controller, the path handler, the
/// goal checker's path length tolerance, the progress checker and the
/// failure tolerance (ns).
#[derive(Clone, Copy, Debug)]
pub struct FollowSettings {
    pub controller: ControllerSettings,
    pub path_handler: PathHandlerParams,
    pub path_length_tolerance_m: NonNegativeFinite,
    pub progress: ProgressCheckerParams,
    pub failure_tolerance_ns: i64,
}

/// The stop's settings but its time before a collision, which each
/// describe_base answer may lengthen.
#[derive(Clone, Copy, Debug)]
pub struct StopSettings {
    pub time_before_collision_s: PositiveFinite,
    pub simulation_time_step_s: PositiveFinite,
    pub source_timeout_s: PositiveFinite,
}

impl StopSettings {
    /// The source timeout in nanoseconds.
    pub fn source_timeout_ns(&self) -> i64 {
        nanoseconds(self.source_timeout_s.get())
    }
}

/// The turns and the straight drives, but the base's limits, which
/// describe_base gives: the slowest turn rate (rad/s), the slowest speed (m/s)
/// forward and backward, and how far ahead (s) each sweeps.
#[derive(Clone, Copy, Debug)]
pub struct MoveSettings {
    pub min_turn_rate_rad_s: NonNegativeFinite,
    pub min_forward_speed_m_s: NonNegativeFinite,
    pub min_backward_speed_m_s: NonNegativeFinite,
    pub simulate_ahead_time_s: NonNegativeFinite,
}

/// The recoveries of a navigation: the turn's angle (rad) and time allowance
/// (ns), the wait (ns), and the back-up's distance (m), speed (m/s) and time
/// allowance (ns).
#[derive(Clone, Copy, Debug)]
pub struct RecoverySettings {
    pub turn_angle_rad: PositiveFinite,
    pub turn_time_allowance_ns: i64,
    pub wait_ns: i64,
    pub back_up_distance_m: PositiveFinite,
    pub back_up_speed_m_s: PositiveFinite,
    pub back_up_time_allowance_ns: i64,
}

#[derive(Clone, Debug)]
pub struct NavigatorConfig {
    pub grid: GridSettings,
    pub planner: PlannerParams,
    pub replan_period_ns: i64,
    pub near_goal_distance_m: NonNegativeFinite,
    /// The period (ns) of the controller's steps.
    pub step_ns: i64,
    pub follow: FollowSettings,
    pub stop: StopSettings,
    pub moves: MoveSettings,
    pub recoveries: RecoverySettings,
    /// How long the robot's tree keeps each moving transform.
    pub cache_window: Duration,
    /// How long a placement waits for the robot's tree.
    pub lookup_timeout: Duration,
}

impl NavigatorConfig {
    pub fn parse(params: &Parameters) -> Result<Self, ConfigError> {
        let grid = parse_grid(params)?;
        VoxelMap::new(
            odometry_frame(),
            grid.voxel_m,
            bounds_around(&grid, [0.0, 0.0]),
        )?;
        let controller_rate = positive("controller_rate_hz", params.controller_rate_hz)?;
        if controller_rate.get() <= SLOWEST_CONTROLLER_RATE_HZ {
            return Err(ConfigError::ControllerTooSlow {
                rate_hz: controller_rate.get(),
            });
        }
        let replan_rate = positive("replan_rate_hz", params.replan_rate_hz)?;
        Ok(Self {
            planner: parse_planner(params)?,
            replan_period_ns: nanoseconds(1.0 / replan_rate.get()),
            near_goal_distance_m: non_negative(
                "near_goal_distance_m",
                params.near_goal_distance_m,
            )?,
            step_ns: nanoseconds(1.0 / controller_rate.get()),
            follow: FollowSettings {
                controller: parse_controller(params, controller_rate)?,
                // The path handler searches the plan for the base's nearest
                // pose within half the grid's side, as Nav2's feasible path
                // handler bounds the search by its cost map's half size.
                path_handler: PathHandlerParams {
                    max_robot_pose_search_dist_m: NonNegativeFinite::parse(grid.half_side_m())
                        .expect("half a positive side is a number of at least 0"),
                    prune_distance_m: non_negative("prune_distance_m", params.prune_distance_m)?,
                },
                path_length_tolerance_m: non_negative(
                    "path_length_tolerance_m",
                    params.path_length_tolerance_m,
                )?,
                progress: ProgressCheckerParams {
                    required_movement_m: non_negative(
                        "required_movement_m",
                        params.required_movement_m,
                    )?,
                    movement_time_allowance_s: non_negative(
                        "movement_time_allowance_s",
                        params.movement_time_allowance_s,
                    )?,
                },
                failure_tolerance_ns: duration_ns(wait_parameter(
                    "failure_tolerance_s",
                    params.failure_tolerance_s,
                )?),
            },
            stop: StopSettings {
                time_before_collision_s: positive(
                    "time_before_collision_s",
                    params.time_before_collision_s,
                )?,
                simulation_time_step_s: positive(
                    "simulation_time_step_s",
                    params.simulation_time_step_s,
                )?,
                source_timeout_s: positive("source_timeout_s", params.source_timeout_s)?,
            },
            moves: parse_moves(params)?,
            recoveries: parse_recoveries(params)?,
            cache_window: duration_parameter("cache_window_s", params.cache_window_s)?,
            lookup_timeout: wait_parameter("lookup_timeout_s", params.lookup_timeout_s)?,
            grid,
        })
    }

    /// How the turns and the straight drives step and sweep: a sweep checks
    /// the controller's rate times the simulate-ahead time in steps, cut to
    /// a whole number, as Nav2's behaviors count their `max_cycle_count`.
    pub fn stepping(&self) -> Stepping {
        let rate_hz = self.follow.controller.controller_frequency_hz.get();
        Stepping {
            step_s: seconds(self.step_ns),
            sweep_steps: (rate_hz * self.moves.simulate_ahead_time_s.get()) as usize,
        }
    }

    /// The warning of a setup whose controller slows on its approach to a
    /// goal from farther than half the grid's side, past which the grid sees
    /// nothing ahead of a goal at its edge; none for a setup that does not.
    pub fn approach_warning(&self) -> Option<String> {
        let approach_m = self
            .follow
            .controller
            .approach_velocity_scaling_dist_m
            .get();
        let half_side_m = self.grid.half_side_m();
        (approach_m > half_side_m).then(|| {
            format!(
                "controller.approach_velocity_scaling_dist_m {approach_m} m exceeds half of \
                 grid.side_m, {half_side_m} m, so the base slows on the approach to a goal from \
                 farther than the grid reaches: set it at most to half of grid.side_m"
            )
        })
    }
}

fn parse_grid(params: &Parameters) -> Result<GridSettings, ConfigError> {
    let grid = &params.grid;
    let voxel_m = positive("grid.voxel_size_m", grid.voxel_size_m)?;
    Ok(GridSettings {
        side_m: positive("grid.side_m", grid.side_m)?,
        voxel_m,
        min_z_m: grid.min_z_m,
        heights: ObstacleHeights::parse(ObstacleHeightsParams {
            min_obstacle_height_m: params.min_obstacle_height_m,
            max_obstacle_height_m: params.max_obstacle_height_m,
            max_step_down_m: params.max_step_down_m,
            voxel_m,
        })?,
        obstacle_max_range_m: positive("obstacle_max_range_m", params.obstacle_max_range_m)?,
        footprint_padding: non_negative("footprint_padding_m", params.footprint_padding_m)?,
        inflation: InflationParams {
            inflation_radius_m: non_negative("inflation_radius_m", params.inflation_radius_m)?,
            cost_scaling_factor: non_negative("cost_scaling_factor", params.cost_scaling_factor)?,
        },
    })
}

fn parse_planner(params: &Parameters) -> Result<PlannerParams, ConfigError> {
    let planner = &params.planner;
    let smoother = &planner.smoother;
    Ok(PlannerParams {
        search: SearchParams {
            cost_travel_multiplier: CostTravelMultiplier::parse(planner.cost_travel_multiplier)?,
            // base_navigation counts a cell never seen as free.
            allow_unknown: true,
            max_iterations: at_least_one("planner.max_iterations", planner.max_iterations)?,
        },
        use_final_approach_orientation: planner.use_final_approach_orientation,
        smoother: Some(SmootherParams {
            tolerance_m: non_negative("planner.smoother.tolerance_m", smoother.tolerance_m)?,
            max_iterations: at_least_one(
                "planner.smoother.max_iterations",
                smoother.max_iterations,
            )?,
            w_data: non_negative("planner.smoother.w_data", smoother.w_data)?,
            w_smooth: non_negative("planner.smoother.w_smooth", smoother.w_smooth)?,
            refinements: smoother.refinements,
        }),
    })
}

fn parse_controller(
    params: &Parameters,
    controller_rate: PositiveFinite,
) -> Result<ControllerSettings, ConfigError> {
    let c = &params.controller;
    Ok(ControllerSettings {
        lookahead: VelocityScaledLookahead::parse(
            positive("controller.lookahead_time_s", c.lookahead_time_s)?,
            non_negative("controller.min_lookahead_dist_m", c.min_lookahead_dist_m)?,
            non_negative("controller.max_lookahead_dist_m", c.max_lookahead_dist_m)?,
        )?,
        rotate_to_heading_angular_vel_rad_s: positive(
            "controller.rotate_to_heading_angular_vel_rad_s",
            c.rotate_to_heading_angular_vel_rad_s,
        )?,
        rotate_to_heading_min_angle_rad: non_negative(
            "controller.rotate_to_heading_min_angle_rad",
            c.rotate_to_heading_min_angle_rad,
        )?,
        max_angular_accel_rad_s2: positive(
            "controller.max_angular_accel_rad_s2",
            c.max_angular_accel_rad_s2,
        )?,
        min_approach_linear_velocity_m_s: positive(
            "controller.min_approach_linear_velocity_m_s",
            c.min_approach_linear_velocity_m_s,
        )?,
        approach_velocity_scaling_dist_m: non_negative(
            "controller.approach_velocity_scaling_dist_m",
            c.approach_velocity_scaling_dist_m,
        )?,
        regulated_linear_scaling_min_radius_m: non_negative(
            "controller.regulated_linear_scaling_min_radius_m",
            c.regulated_linear_scaling_min_radius_m,
        )?,
        regulated_linear_scaling_min_speed_m_s: non_negative(
            "controller.regulated_linear_scaling_min_speed_m_s",
            c.regulated_linear_scaling_min_speed_m_s,
        )?,
        cost_scaling_dist_m: positive("controller.cost_scaling_dist_m", c.cost_scaling_dist_m)?,
        cost_scaling_gain: non_negative("controller.cost_scaling_gain", c.cost_scaling_gain)?,
        max_allowed_time_to_collision_up_to_carrot_s: non_negative(
            "controller.max_allowed_time_to_collision_up_to_carrot_s",
            c.max_allowed_time_to_collision_up_to_carrot_s,
        )?,
        controller_frequency_hz: controller_rate,
    })
}

fn parse_moves(params: &Parameters) -> Result<MoveSettings, ConfigError> {
    Ok(MoveSettings {
        min_turn_rate_rad_s: non_negative("min_turn_rate_rad_s", params.min_turn_rate_rad_s)?,
        min_forward_speed_m_s: non_negative("min_forward_speed_m_s", params.min_forward_speed_m_s)?,
        min_backward_speed_m_s: non_negative(
            "min_backward_speed_m_s",
            params.min_backward_speed_m_s,
        )?,
        simulate_ahead_time_s: non_negative("simulate_ahead_time_s", params.simulate_ahead_time_s)?,
    })
}

fn parse_recoveries(params: &Parameters) -> Result<RecoverySettings, ConfigError> {
    let r = &params.recoveries;
    Ok(RecoverySettings {
        turn_angle_rad: positive("recoveries.turn_angle_rad", r.turn_angle_rad)?,
        turn_time_allowance_ns: positive_ns(
            "recoveries.turn_time_allowance_s",
            r.turn_time_allowance_s,
        )?,
        wait_ns: positive_ns("recoveries.wait_s", r.wait_s)?,
        back_up_distance_m: positive("recoveries.back_up_distance_m", r.back_up_distance_m)?,
        back_up_speed_m_s: positive("recoveries.back_up_speed_m_s", r.back_up_speed_m_s)?,
        back_up_time_allowance_ns: positive_ns(
            "recoveries.back_up_time_allowance_s",
            r.back_up_time_allowance_s,
        )?,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use peppygen::parameters::{controller, grid};

    /// The parameters the unit tests and the harness tests run,
    /// `tests/parameters.json`: a 5 m grid of 0.05 m voxels.
    pub(crate) fn parameters() -> Parameters {
        serde_json::from_str(include_str!("../tests/parameters.json"))
            .expect("tests/parameters.json gives every parameter")
    }

    pub(crate) fn config() -> NavigatorConfig {
        NavigatorConfig::parse(&parameters()).expect("the tests' parameters")
    }

    #[test]
    fn the_search_reaches_half_the_side_and_the_controller_steps_every_50_ms() {
        let config = config();
        assert_eq!(
            config
                .follow
                .path_handler
                .max_robot_pose_search_dist_m
                .get(),
            2.5
        );
        assert_eq!(config.step_ns, 50_000_000);
        assert!(config.planner.search.allow_unknown);
        assert_eq!(config.approach_warning(), None);
    }

    // A controller that slows on its approach from farther than half the
    // grid's side gets a setup warning naming both numbers.
    #[test]
    fn an_approach_past_half_the_side_is_warned_of() {
        let parameters = parameters();
        let far = Parameters {
            controller: controller::Controller {
                approach_velocity_scaling_dist_m: 2.6,
                ..parameters.controller.clone()
            },
            ..parameters
        };
        let warning = NavigatorConfig::parse(&far)
            .expect("a warning, no refusal")
            .approach_warning()
            .expect("2.6 m lies past 2.5 m");
        assert!(
            warning.contains("2.6 m") && warning.contains("2.5 m"),
            "{warning}"
        );
    }

    #[test]
    fn a_grid_whose_voxel_map_cannot_be_built_is_refused() {
        let parameters = parameters();
        let wide = Parameters {
            grid: grid::Grid {
                side_m: 7_000.0,
                ..parameters.grid.clone()
            },
            ..parameters
        };
        let error = NavigatorConfig::parse(&wide).expect_err("a map too wide to build");
        assert!(matches!(error, ConfigError::Grid(_)), "{error}");
    }

    #[test]
    fn a_slow_controller_is_refused() {
        let slow = Parameters {
            controller_rate_hz: 2.0,
            ..parameters()
        };
        assert!(matches!(
            NavigatorConfig::parse(&slow),
            Err(ConfigError::ControllerTooSlow { .. })
        ));
    }
}
