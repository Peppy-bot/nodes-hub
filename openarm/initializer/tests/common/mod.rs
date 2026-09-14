//! What both of this node's tests boot a robot with.

use peppygen::Parameters;
use peppygen::parameters::placement::Placement;

/// A robot of the given OpenArm generation, standing wherever a simulation
/// parks it.
pub fn parameters(hardware_version: &str) -> Parameters {
    Parameters {
        command_rate_hz: 50,
        hardware_version: hardware_version.into(),
        placement: Placement {
            auto: true,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            yaw: 0.0,
        },
    }
}
