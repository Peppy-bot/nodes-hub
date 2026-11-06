//! The robot's measured positions as the planner keeps them: both arms' joints and both
//! grippers' openings at one instant, and the history of them that the map reads at a
//! depth frame's stamp, which may lie some tens of milliseconds before the frame arrives.

use motion_planning_core::JointHistory;
use openarm_description::{ARM_DOF, ArmPair, Side};

/// Both arms' joint positions at one instant.
pub type ArmJoints = ArmPair<[f64; ARM_DOF]>;

/// The values of one measurement: the left arm's joints, the right arm's, then the left
/// and the right gripper's openings.
const VALUES: usize = 2 * ARM_DOF + 2;

/// The measurements of the robot over the history's span.
pub type History = JointHistory<VALUES>;

/// Both arms' joint positions and both grippers' openings at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured {
    pub arms: ArmJoints,
    pub openings: ArmPair<f64>,
}

impl Measured {
    /// The measurement as the history keeps it.
    pub fn values(&self) -> [f64; VALUES] {
        let (left, right) = (self.arms[Side::Left], self.arms[Side::Right]);
        std::array::from_fn(|i| match i {
            i if i < ARM_DOF => left[i],
            i if i < 2 * ARM_DOF => right[i - ARM_DOF],
            i if i == 2 * ARM_DOF => self.openings[Side::Left],
            _ => self.openings[Side::Right],
        })
    }

    /// The measurement whose values the history keeps as `values`.
    pub fn from_values(values: &[f64; VALUES]) -> Self {
        Self {
            arms: ArmPair::new(
                std::array::from_fn(|i| values[i]),
                std::array::from_fn(|i| values[ARM_DOF + i]),
            ),
            openings: ArmPair::new(values[2 * ARM_DOF], values[2 * ARM_DOF + 1]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each value keeps its place through the history: the arms' joints, side by side,
    // and the grippers' openings.
    #[test]
    fn a_measurement_comes_back_from_its_values() {
        let measured = Measured {
            arms: ArmPair::new(
                std::array::from_fn(|i| i as f64),
                std::array::from_fn(|i| 10.0 + i as f64),
            ),
            openings: ArmPair::new(0.25, 0.75),
        };
        let values = measured.values();
        assert_eq!(values[..ARM_DOF], measured.arms[Side::Left]);
        assert_eq!(values[ARM_DOF..2 * ARM_DOF], measured.arms[Side::Right]);
        assert_eq!(values[2 * ARM_DOF..], [0.25, 0.75]);
        assert_eq!(Measured::from_values(&values), measured);
    }
}
