//! Shared primitives for the bimanual backbone: the limb tables, the
//! world-pose wire codec, and every arm's joints in one vector.

use robot_description::{ArmId, PerArm};
use srs_model::nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};
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

/// Norm floor for [`pose_from_wire`]: a quaternion at or below it is refused
/// as naming no rotation at all.
const QUATERNION_MIN_NORM: f64 = 1e-6;

/// Decompose a world-frame pose into the wire `(position, quaternion)` arrays:
/// scalar-last `[x, y, z, w]` out of nalgebra's scalar-first, the outbound
/// mirror of [`pose_from_wire`].
pub fn world_pose_arrays(pose: &Isometry3<f64>) -> ([f64; 3], [f64; 4]) {
    let t = pose.translation.vector;
    let r = pose.rotation;
    ([t.x, t.y, t.z], [r.i, r.j, r.k, r.w])
}

/// Every arm's grasp pose as limb_state and the posture results carry them.
/// The positions have 3 values per arm. The orientations have 4 values per
/// arm, `[x, y, z, w]`. Both are in place order.
pub fn arm_pose_arrays(poses: &PerArm<Isometry3<f64>, ARMS>) -> (Vec<f64>, Vec<f64>) {
    let (positions, orientations): (Vec<[f64; 3]>, Vec<[f64; 4]>) =
        poses.iter().map(world_pose_arrays).unzip();
    (positions.concat(), orientations.concat())
}

/// Parse a world-frame pose off the wire: three finite position components
/// and a finite, normalizable quaternion. Normalized rather than trusted (the
/// wire is four independent floats); a zero-length one is refused rather than
/// read as identity.
pub fn pose_from_wire(
    position: [f64; 3],
    orientation: [f64; 4],
) -> Result<Isometry3<f64>, &'static str> {
    Ok(Isometry3::from_parts(
        position_from_wire(position)?,
        orientation_from_wire(orientation)?,
    ))
}

/// A wire position, or a refusal of one with a non-finite value.
pub fn position_from_wire(position: [f64; 3]) -> Result<Translation3<f64>, &'static str> {
    if !position.iter().all(|v| v.is_finite()) {
        return Err("non-finite values");
    }
    Ok(Translation3::new(position[0], position[1], position[2]))
}

/// A wire orientation, normalized, or a refusal of one with a non-finite value or
/// too short to normalize.
pub fn orientation_from_wire(orientation: [f64; 4]) -> Result<UnitQuaternion<f64>, &'static str> {
    if !orientation.iter().all(|v| v.is_finite()) {
        return Err("non-finite values");
    }
    // The wire is scalar-last [x, y, z, w]; nalgebra's Quaternion is
    // scalar-first, and mixing the two is a silent 90-degree class of error.
    let quaternion = Quaternion::new(
        orientation[3],
        orientation[0],
        orientation[1],
        orientation[2],
    );
    UnitQuaternion::try_new(quaternion, QUATERNION_MIN_NORM).ok_or("an unnormalizable orientation")
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

    // Pins the outbound wire ordering to the inbound one: a scalar-first slip
    // on either arm breaks the round trip.
    #[test]
    fn wire_arrays_round_trip_through_pose_from_wire() {
        let pose = Isometry3::from_parts(
            Translation3::new(0.2, -0.4, 0.9),
            UnitQuaternion::from_euler_angles(0.3, -0.5, 1.1),
        );
        let (position, orientation) = world_pose_arrays(&pose);
        let rebuilt = pose_from_wire(position, orientation).expect("round trip");
        assert!((rebuilt.translation.vector - pose.translation.vector).norm() < 1e-12);
        assert!(rebuilt.rotation.angle_to(&pose.rotation) < 1e-12);
    }
}
