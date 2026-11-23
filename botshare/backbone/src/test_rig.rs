//! The base the unit tests run on: the parameters of the harness rig's base
//! (tests/helpers/parameters.rs), and its axes.

#[path = "../tests/helpers/parameters.rs"]
mod parameters;
pub use parameters::*;

use std::sync::Arc;
use std::time::Duration;

use control_core::node_clock::NodeClock;
use control_core::robot_tree::{Placer, RobotTree};

use crate::config::{AxisSpec, BaseAxes};
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
