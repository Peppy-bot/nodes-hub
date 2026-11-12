//! Gripper joint↔motor geometry: the linear opening-fraction↔motor-radian
//! mapping. The motor speaks radians (0 = closed, full open at a signed
//! angle); the wire speaks the opening fraction (0 = closed, 1 = fully open).
//!
//! The parallel jaws' open angle is a fact of their mechanism, and its sign
//! follows the way the description opens the finger the motor drives: the
//! motor turns one way for a finger that opens toward its joint's positive
//! travel and the other way for one that opens toward its negative travel. A
//! pinch finger's motor turns the finger one to one, so its open angle is the
//! finger's travel from where the description shuts it to where it opens it.
//! Each instance carries its own mapping.

use robot_description::{Finger, Gripper};

/// The finger `gripper`'s motor drives: the first its group names, the joint
/// the others follow. None for a gripper that moves no finger.
pub fn driven_finger(gripper: &Gripper) -> Option<&Finger> {
    gripper.fingers.first()
}

/// One instance's signed motor mapping, resolved from its mechanism's travel
/// and the finger it drives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    /// Motor angle (rad) at full open; the closed end is 0.
    open_rad: f64,
}

impl Geometry {
    /// The mapping of a motor that stands at `open_rad` at full open when its
    /// finger opens toward its joint's positive travel, driving `finger`.
    pub fn new(open_rad: f64, finger: &Finger) -> Self {
        Self {
            open_rad: open_rad * (finger.open - finger.closed).signum(),
        }
    }

    /// The mapping of a motor that turns `finger` one to one: at full open it
    /// stands the finger's travel from closed to open from where it stands
    /// closed.
    pub fn one_to_one(finger: &Finger) -> Self {
        Self {
            open_rad: finger.open - finger.closed,
        }
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
    /// frame: positive drives toward open whichever way the motor turns to
    /// open, so the wire's effort sign is the same on every gripper.
    pub fn motor_torque_to_effort(self, torque: f64) -> f64 {
        torque * self.open_rad.signum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openarm_can::v10;

    use crate::hardware::{DRIVEN_MODELS, Mechanism};

    /// One gripper of a bundled model this node drives: its name, its
    /// mechanism, the finger its motor drives and the mapping this node
    /// resolves for it.
    struct Bundled {
        gripper: String,
        mechanism: Mechanism,
        finger: Finger,
        geometry: Geometry,
    }

    /// Every gripper of every model this node drives, in the order of each
    /// description's arms.
    fn every_gripper() -> Vec<Bundled> {
        DRIVEN_MODELS
            .into_iter()
            .flat_map(|(model, mechanism)| {
                robot_description::model(model)
                    .expect("a bundled model")
                    .description()
                    .arms()
                    .iter()
                    .map(|arm| {
                        let finger = driven_finger(&arm.gripper)
                            .expect("a gripper that moves a finger")
                            .clone();
                        Bundled {
                            gripper: arm.gripper.name.clone(),
                            mechanism,
                            geometry: mechanism.geometry(&finger),
                            finger,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The grippers of every model whose mechanism is `mechanism`.
    fn grippers_run_as(mechanism: Mechanism) -> Vec<Bundled> {
        every_gripper()
            .into_iter()
            .filter(|bundled| bundled.mechanism == mechanism)
            .collect()
    }

    /// The full-open motor angle of every gripper run as `mechanism`, in the
    /// order of [`DRIVEN_MODELS`] and of each description's arms.
    fn open_angles(mechanism: Mechanism) -> Vec<f64> {
        let grippers = grippers_run_as(mechanism);
        assert!(!grippers.is_empty(), "a model runs {mechanism:?}");
        grippers
            .iter()
            .map(|bundled| bundled.geometry.fraction_to_motor_rad(1.0))
            .collect()
    }

    // The parallel jaws' motor opens to its mechanism's angle, and each pinch
    // finger's motor to its finger's travel.
    #[test]
    fn each_mechanism_opens_its_motor_to_its_own_travel() {
        for bundled in every_gripper() {
            let open = bundled.geometry.fraction_to_motor_rad(1.0);
            let travel = bundled.finger.open - bundled.finger.closed;
            match bundled.mechanism {
                Mechanism::ParallelJaws => assert_eq!(
                    open.abs(),
                    v10::GRIPPER_OPEN_RAD.abs(),
                    "{}",
                    bundled.gripper
                ),
                Mechanism::PinchFingers => assert_eq!(open, travel, "{}", bundled.gripper),
            }
        }
    }

    #[test]
    fn the_pinch_grippers_open_toward_mirrored_motor_angles() {
        let opens = open_angles(Mechanism::PinchFingers);
        let [first, second] = opens[..] else {
            panic!("two grippers, got {opens:?}");
        };
        assert!(first > 0.0, "{opens:?}");
        assert_eq!(first, -second);
    }

    #[test]
    fn the_parallel_jaws_share_one_unmirrored_mapping() {
        let opens = open_angles(Mechanism::ParallelJaws);
        assert!(
            opens.iter().all(|&open| open == v10::GRIPPER_OPEN_RAD),
            "{opens:?}"
        );
    }

    #[test]
    fn closed_is_zero_and_mapping_round_trips() {
        for bundled in every_gripper() {
            let geometry = bundled.geometry;
            assert_eq!(geometry.fraction_to_motor_rad(0.0), 0.0);
            for fraction in [0.25, 1.0 / 3.0, 0.5, 1.0] {
                let back = geometry.motor_rad_to_fraction(geometry.fraction_to_motor_rad(fraction));
                assert!(
                    (back - fraction).abs() < 1e-12,
                    "round trip {fraction} on {}",
                    bundled.gripper
                );
            }
        }
    }

    #[test]
    fn measured_fraction_clamps_to_the_wire_range() {
        for bundled in every_gripper() {
            let geometry = bundled.geometry;
            let open = geometry.fraction_to_motor_rad(1.0);
            assert_eq!(geometry.motor_rad_to_fraction(open * 1.2), 1.0);
            assert_eq!(geometry.motor_rad_to_fraction(open * -0.1), 0.0);
        }
    }

    #[test]
    fn effort_is_positive_toward_open_on_every_gripper() {
        // Whatever the mechanism, a torque driving toward this gripper's open
        // angle reports positive and one driving toward closed reports
        // negative, which is what the gripper_link contract fixes.
        for bundled in every_gripper() {
            let geometry = bundled.geometry;
            let toward_open = geometry.fraction_to_motor_rad(1.0).signum();
            assert_eq!(
                geometry.motor_torque_to_effort(0.5 * toward_open),
                0.5,
                "toward open on {}",
                bundled.gripper
            );
            assert_eq!(
                geometry.motor_torque_to_effort(-0.25 * toward_open),
                -0.25,
                "toward closed on {}",
                bundled.gripper
            );
        }
    }
}
