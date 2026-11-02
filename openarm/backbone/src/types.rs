//! Shared primitives for the bimanual backbone: the limb tables, the
//! world-pose wire codec, and both arms' joints in one vector.

use openarm_description::{ARM_DOF, ArmPair, Side};
use srs_model::JointVec;
use srs_model::nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};

/// Joints across both arms, left (0..7) then right (7..14).
pub const DUAL_DOF: usize = 2 * ARM_DOF;

/// Both arms' joints, the left arm's then the right's.
pub type DualJointVec = [f64; DUAL_DOF];

/// Both arms' joints in one vector, the left arm's then the right's.
pub fn dual_joints(arms: &ArmPair<JointVec>) -> DualJointVec {
    std::array::from_fn(|i| match i {
        i if i < ARM_DOF => arms.left[i],
        i => arms.right[i - ARM_DOF],
    })
}

/// The joints of `side`'s arm in `q`.
pub fn arm_joints(q: &DualJointVec, side: Side) -> JointVec {
    let offset = match side {
        Side::Left => 0,
        Side::Right => ARM_DOF,
    };
    std::array::from_fn(|i| q[offset + i])
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

/// Both arms' grasp poses as limb_state and the posture results carry them.
/// The positions have 3 values per arm. The orientations have 4 values per
/// arm, `[x, y, z, w]`. Both are in [`Side::BOTH`] order.
pub fn arm_pose_arrays(poses: &ArmPair<Isometry3<f64>>) -> (Vec<f64>, Vec<f64>) {
    let (positions, orientations): (Vec<[f64; 3]>, Vec<[f64; 4]>) = Side::BOTH
        .into_iter()
        .map(|side| world_pose_arrays(&poses[side]))
        .unzip();
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
    if !position
        .iter()
        .chain(orientation.iter())
        .all(|v| v.is_finite())
    {
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
    let Some(rotation) = UnitQuaternion::try_new(quaternion, QUATERNION_MIN_NORM) else {
        return Err("an unnormalizable orientation");
    };
    Ok(Isometry3::from_parts(
        Translation3::new(position[0], position[1], position[2]),
        rotation,
    ))
}

/// This robot's limb tables as the limb_state contract carries them, wire
/// shaped and in [`Side::BOTH`] order.
pub struct LimbNames {
    pub arm_names: Vec<String>,
    pub joints_per_arm: Vec<u32>,
    pub gripper_names: Vec<String>,
}

/// The one reading of the name tables: the labels on every limb_states
/// snapshot and the answer get_limb_names gives.
pub fn limb_names() -> LimbNames {
    LimbNames {
        arm_names: Side::BOTH.map(|side| side.arm_name().to_owned()).to_vec(),
        joints_per_arm: vec![ARM_DOF as u32; Side::BOTH.len()],
        gripper_names: Side::BOTH
            .map(|side| side.gripper_name().to_owned())
            .to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_limb_tables_name_each_side_in_side_order() {
        let names = limb_names();
        for (side, (arm, gripper)) in Side::BOTH
            .into_iter()
            .zip(names.arm_names.iter().zip(&names.gripper_names))
        {
            assert_eq!(Side::from_arm_name(arm), Ok(side));
            assert_eq!(Side::from_gripper_name(gripper), Ok(side));
        }
        assert_eq!(names.joints_per_arm, [ARM_DOF as u32; 2]);
    }

    #[test]
    fn both_arms_joints_split_into_each_arms() {
        let arms = ArmPair::from_fn(|side| {
            std::array::from_fn(|j| match side {
                Side::Left => j as f64,
                Side::Right => 10.0 + j as f64,
            })
        });
        let dual = dual_joints(&arms);
        assert_eq!(dual[ARM_DOF - 1], (ARM_DOF - 1) as f64);
        assert_eq!(dual[ARM_DOF], 10.0);
        assert_eq!(ArmPair::from_fn(|side| arm_joints(&dual, side)), arms);
    }

    // Pins the outbound wire ordering to the inbound one: a scalar-first slip
    // on either side breaks the round trip.
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
