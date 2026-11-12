//! The robot this node is built for: [`ARMS`] arms of [`ARM_DOF`] joints, each with a
//! gripper, named as the pairing slots this node serves them on ([`LIMBS`]). A model's
//! description drives this node only when its arms are these ([`NodeArms::model`]).

use control_core::requested_arms::{place_of_arm, place_of_gripper};
use robot_description::{ArmId, Driven, NodeArms, PerArm, UndrivenModel};
use srs_model::{ARM_DOF, JointVec};

/// The arms this node drives.
pub const ARMS: usize = 2;

/// Every arm's joints, in place order.
pub const DUAL_DOF: usize = ARMS * ARM_DOF;

/// This node's arms, in place order, each by the link ids of the pairing slots this node
/// serves the arm and its gripper on: the names the wire carries.
pub const LIMBS: NodeArms<ARM_DOF, ARMS> =
    NodeArms::new([("left_arm", "left_gripper"), ("right_arm", "right_gripper")]);

/// The arm a request's `field` names by `name`, or the refusal that names the
/// arms of this robot.
pub fn arm_in(field: &str, name: &str) -> Result<ArmId, String> {
    place_of_arm(field, name, &LIMBS.arm_names()).map(ArmId::new)
}

/// The arm whose gripper a request's `field` names by `name`, or the refusal
/// that names the grippers of this robot.
pub fn gripper_in(field: &str, name: &str) -> Result<ArmId, String> {
    place_of_gripper(field, name, &LIMBS.gripper_names()).map(ArmId::new)
}

/// A model whose description this node does not drive.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Undriven(#[from] UndrivenModel),

    #[error("model {model} gives its arm {arm} no {state} posture")]
    NoPosture {
        model: &'static str,
        arm: &'static str,
        state: &'static str,
    },
}

/// Each arm's posture in the state `state` of the description of `driven`: refused when
/// the description gives an arm none.
pub fn postures_of(
    driven: &Driven<ARM_DOF, ARMS>,
    state: &'static str,
) -> Result<PerArm<JointVec, ARMS>, ModelError> {
    driven.state(state).map_err(|arm| ModelError::NoPosture {
        model: driven.model().id(),
        arm,
        state,
    })
}

#[cfg(test)]
mod tests {
    use robot_description::states;

    use super::*;

    // Both OpenArm models drive this node, and a request names each arm and gripper by its
    // description's name; a model of other arms is refused.
    #[test]
    fn both_openarm_models_drive_this_node() {
        for model in ["openarm_v1", "openarm_v2"] {
            let driven = LIMBS.model(model).expect("an OpenArm drives this node");
            assert_eq!(driven.model().id(), model);
            for (place, arm) in driven.description().arms().iter().enumerate() {
                let id = ArmId::new(place);
                assert_eq!(arm_in("arm_name", &arm.name), Ok(id), "{model}");
                assert_eq!(
                    gripper_in("gripper_name", &arm.gripper.name),
                    Ok(id),
                    "{model}"
                );
            }
        }
        assert!(matches!(
            LIMBS.model("so101"),
            Err(UndrivenModel::OtherArms { .. })
        ));
    }

    /// A name that is no arm, or no gripper, of the robot is refused in the words of the
    /// field, naming the robot's arms or grippers.
    #[test]
    fn a_name_of_no_limb_is_refused_naming_the_limbs() {
        let [left, right] = LIMBS.arm_names();
        assert_eq!(
            arm_in("arm_name", "left"),
            Err(format!(
                "arm_name names \"left\", no arm of this robot: its arms are \"{left}\" and \
                 \"{right}\""
            ))
        );
        let [left, right] = LIMBS.gripper_names();
        assert_eq!(
            gripper_in("gripper_name", "left"),
            Err(format!(
                "gripper_name names \"left\", no gripper of this robot: its grippers are \
                 \"{left}\" and \"{right}\""
            ))
        );
    }

    /// Each arm's posture of a state is the description's, and a state the description
    /// gives no arm is refused naming the model and the state.
    #[test]
    fn a_posture_is_the_descriptions_and_a_missing_one_is_refused() {
        let driven = LIMBS.model("openarm_v2").expect("an OpenArm");
        let ready = postures_of(&driven, states::READY).expect("each arm's ready");
        for arm in PerArm::<(), ARMS>::ids() {
            assert_eq!(
                ready[arm].as_slice(),
                driven
                    .description()
                    .state(LIMBS.arm_name(arm), states::READY)
                    .expect("ready")
            );
        }
        let refused = postures_of(&driven, "no_such_state")
            .expect_err("no arm has it")
            .to_string();
        assert_eq!(
            refused,
            "model openarm_v2 gives its arm left_arm no no_such_state posture"
        );
    }
}
