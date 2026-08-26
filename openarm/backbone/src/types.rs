//! Shared primitives for the bimanual backbone: arm DOF, the joint vector, the
//! arm side identifier, and the world-pose wire decomposition.

use srs_model::nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};

/// Degrees of freedom of one arm, from the description that also supplies the
/// URDF the governor and planner run against.
pub const ARM_DOF: usize = openarm_description::ARM_DOF;

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

/// One joint-space vector (positions, velocities, or torques), j1..j7.
pub type JointVec = [f64; ARM_DOF];

/// Which arm a message addresses. The wire carries the limb's name
/// ("left_arm" / "right_arm"); [`Side::from_arm_name`] parses that at the
/// boundary so the rest of the backbone carries a side it cannot get wrong,
/// never a raw string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    /// This robot's arm names in their fixed left-then-right order: the wire
    /// values limb_motion goals carry and limb_state's arm_names reports.
    pub const ARM_NAMES: [&'static str; 2] = ["left_arm", "right_arm"];

    /// This robot's gripper names, same order and role as [`Self::ARM_NAMES`].
    pub const GRIPPER_NAMES: [&'static str; 2] = ["left_gripper", "right_gripper"];

    /// Refusal for a goal naming no arm of this robot.
    pub const UNKNOWN_ARM_NAME: &'static str =
        r#"unknown arm_name: this robot's arms are "left_arm" and "right_arm""#;

    /// Refusal for a goal naming no gripper of this robot.
    pub const UNKNOWN_GRIPPER_NAME: &'static str =
        r#"unknown gripper_name: this robot's grippers are "left_gripper" and "right_gripper""#;

    /// Parse a wire `arm_name`, or `None` for a name not in [`Self::ARM_NAMES`].
    pub fn from_arm_name(arm_name: &str) -> Option<Self> {
        match arm_name {
            "left_arm" => Some(Side::Left),
            "right_arm" => Some(Side::Right),
            _ => None,
        }
    }

    /// Parse a wire `gripper_name`, or `None` for a name not in
    /// [`Self::GRIPPER_NAMES`].
    pub fn from_gripper_name(gripper_name: &str) -> Option<Self> {
        match gripper_name {
            "left_gripper" => Some(Side::Left),
            "right_gripper" => Some(Side::Right),
            _ => None,
        }
    }

    /// Index into a left-then-right `[T; 2]`.
    pub fn index(self) -> usize {
        match self {
            Side::Left => 0,
            Side::Right => 1,
        }
    }

    /// Label for logs.
    pub fn label(self) -> &'static str {
        match self {
            Side::Left => "left",
            Side::Right => "right",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pins the name tables to the parsers: a name the tables advertise (on
    // limb_state's arm_names/gripper_names) must parse back to its side, and
    // a foreign name must be refused, not misrouted.
    #[test]
    fn advertised_names_parse_and_foreign_names_refuse() {
        assert_eq!(Side::from_arm_name(Side::ARM_NAMES[0]), Some(Side::Left));
        assert_eq!(Side::from_arm_name(Side::ARM_NAMES[1]), Some(Side::Right));
        assert_eq!(
            Side::from_gripper_name(Side::GRIPPER_NAMES[0]),
            Some(Side::Left)
        );
        assert_eq!(
            Side::from_gripper_name(Side::GRIPPER_NAMES[1]),
            Some(Side::Right)
        );
        assert_eq!(Side::from_arm_name("left_gripper"), None);
        assert_eq!(Side::from_gripper_name("left_arm"), None);
        assert_eq!(Side::from_arm_name("LEFT_ARM"), None);
        assert_eq!(Side::from_arm_name(""), None);
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
