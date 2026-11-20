//! The base the unit tests run on: the parameters of the harness rig's base
//! (tests/helpers/parameters.rs), and its axes.

#[path = "../tests/helpers/parameters.rs"]
mod parameters;
pub use parameters::*;

use crate::config::{AxisSpec, BaseAxes};

pub const AXES: BaseAxes = BaseAxes {
    x: AxisSpec {
        min_velocity: -MAX_LINEAR_SPEED_M_S,
        max_velocity: MAX_LINEAR_SPEED_M_S,
        max_acceleration: LINEAR_ACCELERATION_M_S2,
        max_deceleration: LINEAR_DECELERATION_M_S2,
        deadband: LINEAR_DEADBAND_M_S,
        rest: REST_LINEAR_M_S,
    },
    yaw: AxisSpec {
        min_velocity: -MAX_ANGULAR_SPEED_RAD_S,
        max_velocity: MAX_ANGULAR_SPEED_RAD_S,
        max_acceleration: ANGULAR_ACCELERATION_RAD_S2,
        max_deceleration: ANGULAR_DECELERATION_RAD_S2,
        deadband: ANGULAR_DEADBAND_RAD_S,
        rest: REST_ANGULAR_RAD_S,
    },
};
