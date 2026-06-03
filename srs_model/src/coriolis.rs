//! Coriolis + centripetal feedforward torques: the `C(q, q̇) · q̇` vector
//! in the standard manipulator equation
//!
//!   M(q) q̈ + C(q, q̇) q̇ + g(q) = τ.
//!
//! Implemented as a world-frame Recursive Newton-Euler pass with q̈ = 0 and
//! gravity = 0, so all that remains is the Coriolis/centripetal coupling.
//! No mass matrix is materialized.
//!
//! Sign convention matches KDL's `ChainDynParam::JntToCoriolis`; verified
//! by the unit tests against KDL reference values at several (q, q̇) pairs.

use k::nalgebra::Vector3;

use crate::JointVec;
use crate::fk::Posed;

/// Coriolis + centripetal torques at velocity `q̇`. The configuration is baked
/// into the posed `fk` ([`ForwardKinematics::at`](crate::fk::ForwardKinematics::at)),
/// so there is no separate refresh step to forget or race on.
pub fn torques(fk: &Posed, qdot: &JointVec) -> JointVec {
    // Forward pass: propagate angular velocity, angular acceleration, and
    // the linear acceleration of each joint origin and link COM outward.
    // Index 0 is the fixed base (zeros); index i+1 belongs to segment i.
    let mut omega = [Vector3::<f64>::zeros(); crate::ARM_DOF + 1];
    let mut alpha = [Vector3::<f64>::zeros(); crate::ARM_DOF + 1];
    let mut a_origin = [Vector3::<f64>::zeros(); crate::ARM_DOF + 1];
    let mut a_com = [Vector3::<f64>::zeros(); crate::ARM_DOF];

    let mut prev_origin = Vector3::<f64>::zeros();
    for i in 0..crate::ARM_DOF {
        let origin = fk.origin_world(i);
        let qd_h = qdot[i] * fk.axis_world(i);

        omega[i + 1] = omega[i] + qd_h;
        // Full RNEA: α_{i+1} = α_i + ω_i × q̇·ĥ + q̈·ĥ. With q̈ = 0 only
        // the parent-ω cross-coupling with this joint's spin survives.
        alpha[i + 1] = alpha[i] + omega[i].cross(&qd_h);

        // Joint i origin is rigidly attached to link i-1, so its
        // acceleration is computed from the parent's (ω_{i-1}, α_{i-1}).
        let r = origin - prev_origin;
        a_origin[i + 1] = a_origin[i] + alpha[i].cross(&r) + omega[i].cross(&omega[i].cross(&r));
        prev_origin = origin;

        // COM acceleration adds this link's own angular contribution.
        let c = fk.com_world(i) - origin;
        a_com[i] =
            a_origin[i + 1] + alpha[i + 1].cross(&c) + omega[i + 1].cross(&omega[i + 1].cross(&c));
    }

    // Backward pass: accumulate the inertial force and moment each parent
    // must transmit to its child, working from the tip inward. The joint
    // torque is the projection of that moment onto the joint axis.
    let mut f_child = Vector3::<f64>::zeros();
    let mut n_child = Vector3::<f64>::zeros();
    let mut tau = [0.0_f64; crate::ARM_DOF];

    for i in (0..crate::ARM_DOF).rev() {
        let origin = fk.origin_world(i);
        let inertia = fk.inertia_world(i);

        // Inertial force at the COM and moment about the COM.
        let force_com = fk.mass(i) * a_com[i];
        let moment_com = inertia * alpha[i + 1] + omega[i + 1].cross(&(inertia * omega[i + 1]));

        let force_joint = force_com + f_child;

        // Moment about joint i origin = inertial moment about COM
        //   + (COM - origin) × inertial force
        //   + child moment + (child origin - origin) × child force.
        let r_com = fk.com_world(i) - origin;
        let r_child = if i + 1 < crate::ARM_DOF {
            fk.origin_world(i + 1) - origin
        } else {
            // Tip link has no child: f_child and n_child are zero.
            Vector3::zeros()
        };
        let moment_joint = moment_com + r_com.cross(&force_com) + n_child + r_child.cross(&f_child);

        tau[i] = fk.axis_world(i).dot(&moment_joint);

        f_child = force_joint;
        n_child = moment_joint;
    }

    tau
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ARM_DOF;
    use crate::test_support::v1_fk;
    use std::f64::consts::FRAC_PI_2;

    fn coriolis_at(side: &str, q: &JointVec, qdot: &JointVec) -> JointVec {
        let mut fk = v1_fk(side);
        torques(&fk.at(q), qdot)
    }

    // KDL `ChainDynParam::JntToCoriolis` for the same URDF and chain
    // (openarm_body_link0 -> tip). The left and right arms are mirror images,
    // so their torques differ at the same (q, q_dot). Velocities are larger
    // than human-teleop speeds so the Coriolis/centripetal torques clear the
    // 1e-3 Nm noise floor; sub-threshold values are written 0.0. Regenerate
    // both sides with `tools/kdl_reference.cpp`.
    const CASES: [(JointVec, JointVec); 4] = [
        // shoulder spin: pure centripetal load projected back via Christoffel
        ([0.0; ARM_DOF], [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        // elbow spin
        ([0.0; ARM_DOF], [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
        // folded (q4 = pi/2), shoulder + elbow co-rotating: cross-coupling
        ([0.0, 0.0, 0.0, FRAC_PI_2, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]),
        // mixed posture and velocity
        ([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], [1.0, -1.5, 2.0, -2.5, 3.0, -3.5, 4.0]),
    ];

    fn assert_matches_kdl(side: &str, expected: [JointVec; 4]) {
        for ((q, qdot), exp) in CASES.iter().zip(&expected) {
            let tau = coriolis_at(side, q, qdot);
            for i in 0..ARM_DOF {
                assert!(
                    (tau[i] - exp[i]).abs() < 1e-3,
                    "{side} arm, q={q:?} qd={qdot:?}, joint {i}: actual={} expected={}",
                    tau[i],
                    exp[i],
                );
            }
        }
    }

    #[test]
    fn zero_velocity_gives_zero_torque() {
        let tau = coriolis_at("left", &[0.0; ARM_DOF], &[0.0; ARM_DOF]);
        assert_eq!(tau, [0.0; ARM_DOF]);
    }

    #[test]
    fn left_matches_kdl() {
        assert_matches_kdl(
            "left",
            [
                [0.0, -0.0714, 0.0, -0.0193, 0.0, 0.0740, 0.0],
                [-0.0193, -0.0425, 0.0, 0.0, 0.0, 0.0407, 0.0],
                [-0.7152, -0.0104, 0.0, 0.7152, 0.0131, 0.0, -0.0641],
                [-2.0527, -0.9977, 0.4181, 0.4458, 0.0957, 0.0587, 0.1603],
            ],
        );
    }

    #[test]
    fn right_matches_kdl() {
        assert_matches_kdl(
            "right",
            [
                [0.0, 0.0867, 0.0, -0.0193, 0.0, -0.0740, 0.0],
                [0.0193, 0.0430, 0.0, 0.0, 0.0, -0.0407, 0.0],
                [-2.1456, 0.0156, 0.0619, 0.7152, -0.0129, -0.0586, 0.0644],
                [0.0995, -0.7649, 0.1181, -0.2054, -0.0180, -0.0224, 0.0163],
            ],
        );
    }
}
