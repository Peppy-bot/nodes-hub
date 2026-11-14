//! The robot this node is built for: [`ARMS`] arms of [`ARM_DOF`] joints, each with a
//! gripper, named as the pairing slots this node serves them on ([`LIMBS`]). A model's
//! description drives this node only when its arms are these.

use robot_description::NodeArms;
use srs_model::ARM_DOF;

/// The arms this node drives.
pub const ARMS: usize = 2;

/// This node's arms, in place order, each by the link ids of the pairing slots this node
/// serves the arm and its gripper on: the names the wire carries.
pub const LIMBS: NodeArms<ARM_DOF, ARMS> =
    NodeArms::new([("left_arm", "left_gripper"), ("right_arm", "right_gripper")]);
