//! The base the unit tests run on: the parameters of the harness rig's base
//! (tests/helpers/parameters.rs), and its axes; and botshare_sim's axes.

#[path = "../tests/helpers/parameters.rs"]
mod parameters;
pub use parameters::*;

use std::sync::Arc;
use std::time::Duration;

use control_core::node_clock::NodeClock;
use control_core::robot_tree::{Placer, RobotTree};

use crate::config::{AxisSpec, BaseAxes, SlowestSpeeds};
use crate::move_control::MoveAxes;
use crate::transforms::robot_frame;

/// A placer in the robot frame on an empty robot's tree on the wall clock,
/// which places a value in the robot frame as it is given.
pub fn placer() -> Placer {
    let tree =
        RobotTree::new(Duration::from_secs(10), NodeClock::wall()).expect("a window above 0");
    Placer::new(Arc::new(tree), robot_frame(), Duration::ZERO)
}

pub const AXES: BaseAxes = BaseAxes {
    linear: AxisSpec {
        min_velocity: -MAX_LINEAR_SPEED_M_S,
        max_velocity: MAX_LINEAR_SPEED_M_S,
        max_acceleration: LINEAR_ACCELERATION_M_S2,
        max_deceleration: LINEAR_DECELERATION_M_S2,
        deadband: LINEAR_DEADBAND_M_S,
    },
    angular: AxisSpec {
        min_velocity: -MAX_ANGULAR_SPEED_RAD_S,
        max_velocity: MAX_ANGULAR_SPEED_RAD_S,
        max_acceleration: ANGULAR_ACCELERATION_RAD_S2,
        max_deceleration: ANGULAR_DECELERATION_RAD_S2,
        deadband: ANGULAR_DEADBAND_RAD_S,
    },
};

/// `axes` as move_base plans and steers along them, floored at the rig's
/// slowest turn rate and drive speeds, under the rig's limiter.
pub fn move_axes(axes: &BaseAxes) -> MoveAxes {
    let slowest = SlowestSpeeds {
        turn_rate: MIN_TURN_RATE_RAD_S,
        forward: MIN_FORWARD_SPEED_M_S,
        backward: MIN_BACKWARD_SPEED_M_S,
    };
    axes.move_axes(slowest, 1.0 / SMOOTHING_FREQUENCY_HZ)
}

/// botshare_sim's limits (launchers-hub botshare/fragments/botshare_sim.json5),
/// whose floors lie above one step of the deceleration.
pub const BOTSHARE_SIM_AXES: BaseAxes = BaseAxes {
    linear: AxisSpec {
        min_velocity: -0.2,
        max_velocity: 0.2,
        max_acceleration: 1.0,
        max_deceleration: 2.5,
        deadband: 0.0,
    },
    angular: AxisSpec {
        min_velocity: -0.5,
        max_velocity: 0.5,
        max_acceleration: 3.0,
        max_deceleration: 3.0,
        deadband: 0.0,
    },
};

/// [`BOTSHARE_SIM_AXES`] as move_base plans and steers along them, floored as
/// botshare_sim floors them, under its limiter's rate, which is the rig's.
pub fn botshare_sim_move_axes() -> MoveAxes {
    let slowest = SlowestSpeeds {
        turn_rate: 0.4,
        forward: 0.1,
        backward: 0.1,
    };
    BOTSHARE_SIM_AXES.move_axes(slowest, 1.0 / SMOOTHING_FREQUENCY_HZ)
}
