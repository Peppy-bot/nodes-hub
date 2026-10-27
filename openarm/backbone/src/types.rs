//! Shared primitives for the bimanual backbone: arm DOF, the joint vector,
//! the world-pose wire decomposition, the pose slack a move is planned and
//! settled within, and the tolerances and times every move goal is parsed by.

use std::time::Duration;

use srs_model::chain_kinematics::{ServoTolerances, ToleranceError};
use srs_model::nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};

use crate::side::{ArmPair, Side};

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

/// The longest a move may be asked to take, and the longest it may be given
/// to settle after its planned end: the bound the limb_motion contract states
/// for every duration and timeout. A plan walks the control ticks of the
/// request, so the bound also bounds its work.
const MAX_MOVE_DURATION_S: f64 = 60.0;

/// A slack around a grasp-point pose: a position (m) and an orientation
/// (rad). A `move_arm` goal is planned within one, judged against the goal
/// pose alone and never the path, and settles within another.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseTolerance {
    pub position_m: f64,
    pub orientation_rad: f64,
}

impl PoseTolerance {
    /// The same slack as the guarded servo's arrival bar, or the refusal of a
    /// value that is not a slack.
    pub fn servo(&self) -> Result<ServoTolerances, ToleranceError> {
        ServoTolerances::new(self.position_m, self.orientation_rad)
    }

    /// Parse the wire pair of the `{field}_position_tolerance_m` and
    /// `{field}_orientation_tolerance_rad` goal fields, each a positive finite
    /// number, or the refusal naming the field.
    pub fn from_wire(field: &str, position_m: f64, orientation_rad: f64) -> Result<Self, String> {
        Ok(Self {
            position_m: tolerance_from_wire(&format!("{field}_position_tolerance_m"), position_m)?,
            orientation_rad: tolerance_from_wire(
                &format!("{field}_orientation_tolerance_rad"),
                orientation_rad,
            )?,
        })
    }

    /// Refuse this settle slack where it lies below `plan`'s on an axis,
    /// naming the field to raise: the plan may sit that far from the pose.
    pub fn covers(&self, plan: &Self) -> Result<(), String> {
        let axes = [
            (self.position_m, plan.position_m, "position_tolerance_m"),
            (
                self.orientation_rad,
                plan.orientation_rad,
                "orientation_tolerance_rad",
            ),
        ];
        match axes.iter().find(|(settle, plan, _)| settle < plan) {
            None => Ok(()),
            Some((_, _, axis)) => Err(format!(
                "settle_{axis} must be at least plan_{axis}: the plan may sit that far from the pose"
            )),
        }
    }
}

/// A Cartesian move as `move_arm` plans it and `check_arm_move` checks it,
/// parsed once off the wire: both carry the same planning fields with the
/// same meaning, and both refuse the same values with the same words.
#[derive(Clone, Copy, Debug)]
pub struct ArmMoveRequest {
    pub side: Side,
    pub target: Isometry3<f64>,
    pub tolerance: PoseTolerance,
    pub duration: Duration,
}

impl ArmMoveRequest {
    pub fn from_wire(
        arm_name: &str,
        position: [f64; 3],
        orientation: [f64; 4],
        duration_s: f64,
        plan_position_tolerance_m: f64,
        plan_orientation_tolerance_rad: f64,
    ) -> Result<Self, String> {
        let side = Side::from_arm_name(arm_name)?;
        let target = pose_from_wire(position, orientation)
            .map_err(|reason| format!("goal pose has {reason}"))?;
        let tolerance = PoseTolerance::from_wire(
            "plan",
            plan_position_tolerance_m,
            plan_orientation_tolerance_rad,
        )?;
        let duration = move_seconds_from_wire("duration_s", duration_s)?;
        Ok(Self {
            side,
            target,
            tolerance,
            duration,
        })
    }
}

/// Parse a tolerance off the wire: a positive finite number, or the refusal
/// naming `field`.
fn tolerance_from_wire(field: &str, value: f64) -> Result<f64, String> {
    if value.is_finite() && value > 0.0 {
        return Ok(value);
    }
    Err(format!("{field} must be a positive number"))
}

/// Parse the per-joint settle tolerances off the wire, one per joint in the
/// arm's joint order: each a positive finite number, or the refusal.
pub fn joint_tolerances_from_wire(tolerances: JointVec) -> Result<JointVec, String> {
    if tolerances.iter().all(|v| v.is_finite() && *v > 0.0) {
        return Ok(tolerances);
    }
    Err("each settle_tolerances_rad entry must be a positive number".to_string())
}

/// Parse a number of seconds a move may be asked for off the wire, `field`
/// being `duration_s` or `settle_timeout_s`: from 0 to
/// [`MAX_MOVE_DURATION_S`], or the refusal naming the field.
pub fn move_seconds_from_wire(field: &str, value: f64) -> Result<Duration, String> {
    if value.is_finite() && (0.0..=MAX_MOVE_DURATION_S).contains(&value) {
        return Ok(Duration::from_secs_f64(value));
    }
    Err(format!(
        "{field} must be a number of seconds from 0 to {MAX_MOVE_DURATION_S:.0}"
    ))
}

/// One joint-space vector (positions, velocities, or torques), j1..j7.
pub type JointVec = [f64; ARM_DOF];

#[cfg(test)]
mod tests {
    use super::*;

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

    // --- Tolerances and times off the wire ---------------------------------

    use hegel::TestCase;
    use hegel::generators as gs;

    fn admitted_tolerance(value: f64) -> bool {
        value.is_finite() && value > 0.0
    }

    // A tolerance off the wire is exactly a positive finite number, kept as it
    // came; anything else is refused naming the field.
    #[hegel::test]
    fn a_tolerance_is_accepted_exactly_when_positive_and_finite(tc: TestCase) {
        let value: f64 = tc.draw(gs::floats());
        match tolerance_from_wire("settle_tolerance", value) {
            Ok(parsed) => {
                assert!(admitted_tolerance(value), "{value} was accepted");
                assert_eq!(parsed, value);
            }
            Err(refusal) => {
                assert!(!admitted_tolerance(value), "{value} was refused");
                assert_eq!(refusal, "settle_tolerance must be a positive number");
            }
        }
    }

    #[test]
    fn non_finite_zero_and_negative_tolerances_are_refused() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1e-9, -1.0] {
            assert!(
                tolerance_from_wire("settle_tolerance", bad).is_err(),
                "tolerance {bad} must be refused"
            );
        }
    }

    // A number of seconds off the wire is exactly a finite value from 0 to the
    // move ceiling, carried as a duration.
    #[hegel::test]
    fn move_seconds_are_accepted_exactly_within_the_ceiling(tc: TestCase) {
        let value: f64 = tc.draw(gs::floats());
        let admitted = value.is_finite() && (0.0..=MAX_MOVE_DURATION_S).contains(&value);
        match move_seconds_from_wire("settle_timeout_s", value) {
            Ok(duration) => {
                assert!(admitted, "{value} s was accepted");
                assert_eq!(duration, Duration::from_secs_f64(value));
            }
            Err(refusal) => {
                assert!(!admitted, "{value} s was refused");
                assert_eq!(
                    refusal,
                    "settle_timeout_s must be a number of seconds from 0 to 60"
                );
            }
        }
    }

    // Per-joint tolerances are accepted exactly when every entry is a
    // positive finite value, kept in order.
    #[hegel::test]
    fn joint_tolerances_need_a_positive_value_per_joint(tc: TestCase) {
        let values: JointVec = tc.draw(gs::arrays(gs::floats()));
        let admitted = values.iter().all(|v| v.is_finite() && *v > 0.0);
        match joint_tolerances_from_wire(values) {
            Ok(parsed) => {
                assert!(admitted, "{values:?} was accepted");
                assert_eq!(parsed, values);
            }
            Err(refusal) => {
                assert!(!admitted, "{values:?} was refused");
                assert_eq!(
                    refusal,
                    "each settle_tolerances_rad entry must be a positive number"
                );
            }
        }
    }

    // A pose slack covers another exactly when it is at least as wide on
    // both axes.
    #[hegel::test]
    fn a_pose_tolerance_covers_another_exactly_when_wider_on_both_axes(tc: TestCase) {
        let [settle_m, plan_m, settle_rad, plan_rad]: [f64; 4] =
            tc.draw(gs::arrays(gs::floats().min_value(0.0).max_value(1.0)));
        let settle = PoseTolerance {
            position_m: settle_m,
            orientation_rad: settle_rad,
        };
        let plan = PoseTolerance {
            position_m: plan_m,
            orientation_rad: plan_rad,
        };
        let expected = if settle_m < plan_m {
            Err(
                "settle_position_tolerance_m must be at least plan_position_tolerance_m: \
                 the plan may sit that far from the pose"
                    .to_string(),
            )
        } else if settle_rad < plan_rad {
            Err("settle_orientation_tolerance_rad must be at least \
                 plan_orientation_tolerance_rad: the plan may sit that far from the pose"
                .to_string())
        } else {
            Ok(())
        };
        assert_eq!(settle.covers(&plan), expected);
    }

    #[test]
    fn a_pose_tolerance_names_the_axis_it_refuses() {
        assert_eq!(
            PoseTolerance::from_wire("plan", 0.0, 0.1).unwrap_err(),
            "plan_position_tolerance_m must be a positive number"
        );
        assert_eq!(
            PoseTolerance::from_wire("settle", 0.01, -1.0).unwrap_err(),
            "settle_orientation_tolerance_rad must be a positive number"
        );
        assert_eq!(
            PoseTolerance::from_wire("plan", 0.005, 0.087),
            Ok(PoseTolerance {
                position_m: 0.005,
                orientation_rad: 0.087,
            })
        );
    }
}
