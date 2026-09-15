//! What both of this node's tests boot a robot with.

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
