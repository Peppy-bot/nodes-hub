//! Gripper joint↔motor geometry: the linear opening-fraction↔motor-radian
//! mapping. The motor speaks radians (0 = closed, full open at a signed
//! angle); the wire speaks the opening fraction (0 = closed, 1 = fully open).
//!
//! The angle's magnitude and sign are facts of the generation and the side:
//! the v1.0 prismatic jaws are not mirrored (both sides open toward the same
//! negative motor angle), while the v2.0 revolute pinch fingers are, so each
//! instance carries its own mapping.

use openarm_can::{v10, v20};
use openarm_description::{HardwareVersion, Side};

/// One instance's signed motor mapping, resolved from the generation it
/// drives and the side it sits on.
#[derive(Debug, Clone, Copy)]
pub struct Geometry {
    /// Motor angle (rad) at full open; the closed end is 0.
    open_rad: f64,
}

impl Geometry {
    pub fn new(hardware_version: HardwareVersion, side: Side) -> Self {
        let open_rad = match (hardware_version, side) {
            (HardwareVersion::V1, _) => v10::GRIPPER_OPEN_RAD,
            (HardwareVersion::V2, Side::Left) => v20::GRIPPER_OPEN_RAD,
            (HardwareVersion::V2, Side::Right) => -v20::GRIPPER_OPEN_RAD,
        };
        Self { open_rad }
    }

    /// Opening fraction (0 = closed, 1 = fully open) to signed motor radians.
    pub fn fraction_to_motor_rad(self, fraction: f64) -> f64 {
        fraction * self.open_rad
    }

    /// Signed motor radians to the wire's opening fraction, clamped to
    /// 0..=1 so encoder readings past the calibrated travel cannot leave
    /// the contract's range. Inverse of [`Self::fraction_to_motor_rad`]
    /// within that travel.
    pub fn motor_rad_to_fraction(self, motor_rad: f64) -> f64 {
        (motor_rad / self.open_rad).clamp(0.0, 1.0)
    }

    /// Measured motor torque (N*m at the shaft) mapped into the opening
    /// frame: positive drives toward open on either side, so the wire's
    /// effort sign is side-consistent despite the mirrored v2 motors.
    pub fn motor_torque_to_effort(self, torque: f64) -> f64 {
        torque * self.open_rad.signum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [(HardwareVersion, Side); 4] = [
        (HardwareVersion::V1, Side::Left),
        (HardwareVersion::V1, Side::Right),
        (HardwareVersion::V2, Side::Left),
        (HardwareVersion::V2, Side::Right),
    ];

    fn geometry(hardware_version: HardwareVersion, side: Side) -> Geometry {
        Geometry::new(hardware_version, side)
    }

    #[test]
    fn each_generation_carries_its_own_travel() {
        // Exact: both sides of each assertion are `1.0 * open_rad`.
        assert_eq!(
            geometry(HardwareVersion::V1, Side::Left).fraction_to_motor_rad(1.0),
            v10::GRIPPER_OPEN_RAD
        );
        assert_eq!(
            geometry(HardwareVersion::V2, Side::Left).fraction_to_motor_rad(1.0),
            v20::GRIPPER_OPEN_RAD
        );
        assert_ne!(
            v10::GRIPPER_OPEN_RAD.abs(),
            v20::GRIPPER_OPEN_RAD.abs(),
            "the generations travel different distances, so neither constant may stand in for the other"
        );
    }

    #[test]
    fn v2_sides_open_toward_mirrored_motor_angles() {
        let left = geometry(HardwareVersion::V2, Side::Left).fraction_to_motor_rad(1.0);
        let right = geometry(HardwareVersion::V2, Side::Right).fraction_to_motor_rad(1.0);
        assert!(left > 0.0);
        assert!(right < 0.0);
        assert_eq!(left, -right);
    }

    #[test]
    fn v1_sides_share_one_unmirrored_mapping() {
        let left = geometry(HardwareVersion::V1, Side::Left).fraction_to_motor_rad(1.0);
        let right = geometry(HardwareVersion::V1, Side::Right).fraction_to_motor_rad(1.0);
        assert_eq!(left, right);
        assert!(left < 0.0, "the v1 jaws open toward a negative motor angle");
    }

    #[test]
    fn closed_is_zero_and_mapping_round_trips() {
        for (hardware_version, side) in ALL {
            let geometry = geometry(hardware_version, side);
            assert_eq!(geometry.fraction_to_motor_rad(0.0), 0.0);
            for fraction in [0.25, 1.0 / 3.0, 0.5, 1.0] {
                let back = geometry.motor_rad_to_fraction(geometry.fraction_to_motor_rad(fraction));
                assert!(
                    (back - fraction).abs() < 1e-12,
                    "round trip {fraction} on {hardware_version} {side:?}"
                );
            }
        }
    }

    #[test]
    fn measured_fraction_clamps_to_the_wire_range() {
        for (hardware_version, side) in ALL {
            let geometry = geometry(hardware_version, side);
            let open = geometry.fraction_to_motor_rad(1.0);
            assert_eq!(geometry.motor_rad_to_fraction(open * 1.2), 1.0);
            assert_eq!(geometry.motor_rad_to_fraction(open * -0.1), 0.0);
        }
    }

    #[test]
    fn effort_is_side_consistent_toward_open() {
        // Whatever the mechanism, a torque driving toward this side's open
        // angle reports positive and one driving toward closed reports
        // negative, which is what the gripper_link contract fixes.
        for (hardware_version, side) in ALL {
            let geometry = geometry(hardware_version, side);
            let toward_open = geometry.fraction_to_motor_rad(1.0).signum();
            assert_eq!(
                geometry.motor_torque_to_effort(0.5 * toward_open),
                0.5,
                "toward open on {hardware_version} {side:?}"
            );
            assert_eq!(
                geometry.motor_torque_to_effort(-0.25 * toward_open),
                -0.25,
                "toward closed on {hardware_version} {side:?}"
            );
        }
    }
}
