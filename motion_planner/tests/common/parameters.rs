//! The parameters the OpenArm v2 launchers give the planner, as their motion_planner
//! fragment sets them. The harness tests reach this file as `common::parameters`, and the
//! node's own tests as `config::launched`.

use peppygen::Parameters;

/// The parameters the OpenArm v2 launchers give the planner.
pub fn parameters() -> Parameters {
    Parameters {
        goal_samples: 10,
        map_max_x_m: 0.66,
        map_max_y_m: 0.82,
        map_max_z_m: 1.36,
        map_min_x_m: -0.66,
        map_min_y_m: -0.82,
        map_min_z_m: 0.04,
        max_iterations: 20_000,
        max_range_m: 1.36,
        max_update_rate_hz: 5.0,
        min_clearance_m: 0.04,
        padding_offset_m: 0.04,
        padding_scale: 1.0,
        planning_attempts: 1,
        start_tolerance_rad: 0.05,
        voxel_size_m: 0.02,
    }
}
