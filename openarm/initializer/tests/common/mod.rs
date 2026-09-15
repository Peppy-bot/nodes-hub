//! What both of this node's tests boot a robot with. Each test binary
//! compiles its own copy, so a builder only one of them calls is unused in
//! the other.
#![allow(dead_code)]

use peppygen::Parameters;
use peppygen::parameters::placement::Placement;

/// A robot that a simulation stands as `model`, wherever it parks it.
pub fn parameters(model: &str) -> Parameters {
    Parameters {
        command_rate_hz: 50,
        model: model.into(),
        placement: Placement {
            auto: true,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            yaw: 0.0,
        },
    }
}

/// The same robot, standing where it is told rather than where the
/// simulation parks it.
pub fn standing_at(model: &str, x: f64, y: f64, z: f64, yaw: f64) -> Parameters {
    Parameters {
        placement: Placement {
            auto: false,
            x,
            y,
            z,
            yaw,
        },
        ..parameters(model)
    }
}
