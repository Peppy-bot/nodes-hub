//! The robot the KER leads: [`ARMS`] arms of [`ARM_DOF`] joints, each with a
//! gripper, in the order of the KER's halves ([`LIMBS`]). A model's description
//! drives this node only when its arms are these, so every per-arm table of
//! this crate, held in place order, meets the arm it names.

use robot_description::NodeArms;

/// The arms the KER leads, one per half.
pub const ARMS: usize = 2;

/// The joints of each arm the KER leads, one encoder channel each.
pub const ARM_DOF: usize = 7;

/// The arms the KER leads, in place order, each by the link ids of the
/// joint_link and gripper_link pairing slots this node streams that arm and its
/// gripper on.
pub const LIMBS: NodeArms<ARM_DOF, ARMS> =
    NodeArms::new([("left_arm", "left_gripper"), ("right_arm", "right_gripper")]);

/// The arms this crate's tests name.
#[cfg(test)]
pub(crate) mod fixtures {
    use robot_description::{ArmId, PerArm};

    use super::*;

    /// The arms, in place order.
    pub(crate) const LEFT: ArmId = ArmId::new(0);
    pub(crate) const RIGHT: ArmId = ArmId::new(1);

    /// The model whose joint structure the KER shares.
    pub(crate) const MODEL: &str = "openarm_v2";

    /// The joint limits of [`MODEL`]'s arms.
    pub(crate) fn follower_limits() -> PerArm<[[f64; 2]; ARM_DOF], ARMS> {
        LIMBS
            .model(MODEL)
            .expect("the KER leads its own model")
            .joint_limits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Both OpenArm models are the robot the KER leads.
    #[test]
    fn both_openarm_models_are_the_robot_the_ker_leads() {
        for model in ["openarm_v1", "openarm_v2"] {
            LIMBS
                .model(model)
                .unwrap_or_else(|error| panic!("{model}: {error}"));
        }
    }
}
