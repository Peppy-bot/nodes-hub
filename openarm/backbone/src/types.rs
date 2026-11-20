//! Shared primitives for the bimanual backbone: the limb tables and every
//! arm's joints in one vector.

use robot_description::{ArmId, PerArm};
use srs_model::{ARM_DOF, JointVec};

use crate::robot::{ARMS, DUAL_DOF, LIMBS};

/// Every arm's joints, in place order.
pub type DualJointVec = [f64; DUAL_DOF];

/// Every arm's joints in one vector, in place order.
pub fn dual_joints(arms: &PerArm<JointVec, ARMS>) -> DualJointVec {
    std::array::from_fn(|i| arms[ArmId::new(i / ARM_DOF)][i % ARM_DOF])
}

/// The joints of `arm` in `q`.
pub fn arm_joints(q: &DualJointVec, arm: ArmId) -> JointVec {
    std::array::from_fn(|i| q[arm.place() * ARM_DOF + i])
}

/// This robot's limb tables as the limb_state contract carries them, wire
/// shaped and in place order.
pub struct LimbNames {
    pub arm_names: Vec<String>,
    pub joints_per_arm: Vec<u32>,
    pub gripper_names: Vec<String>,
}

/// The one reading of the name tables: the labels on every limb_states
/// snapshot and the answer get_limb_names gives.
pub fn limb_names() -> LimbNames {
    LimbNames {
        arm_names: LIMBS.arm_names().map(str::to_owned).to_vec(),
        joints_per_arm: vec![ARM_DOF as u32; ARMS],
        gripper_names: LIMBS.gripper_names().map(str::to_owned).to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::robot::{arm_in, gripper_in};

    #[test]
    fn the_limb_tables_name_each_arm_in_place_order() {
        let names = limb_names();
        for (arm, (arm_name, gripper_name)) in
            PerArm::<(), ARMS>::ids().zip(names.arm_names.iter().zip(&names.gripper_names))
        {
            assert_eq!(arm_in("arm_name", arm_name), Ok(arm));
            assert_eq!(gripper_in("gripper_name", gripper_name), Ok(arm));
        }
        assert_eq!(names.joints_per_arm, [ARM_DOF as u32; ARMS]);
    }

    #[test]
    fn every_arms_joints_split_into_each_arms() {
        let arms =
            PerArm::from_fn(|arm| std::array::from_fn(|j| (10 * arm.place()) as f64 + j as f64));
        let dual = dual_joints(&arms);
        assert_eq!(dual[ARM_DOF - 1], (ARM_DOF - 1) as f64);
        assert_eq!(dual[ARM_DOF], 10.0);
        assert_eq!(PerArm::from_fn(|arm| arm_joints(&dual, arm)), arms);
    }
}
