//! Per-joint torques that hold the arm against gravity at a given posture.

use crate::JointVec;
use crate::fk::Posed;

const GRAVITY_MAGNITUDE: f64 = 9.81;

/// Gravity-compensation torques: the torque each joint must apply to hold the arm
/// against gravity at the posed configuration. The pose is baked into `fk`
/// ([`ForwardKinematics::at`](crate::fk::ForwardKinematics::at)), so there is no
/// configuration argument and no refresh to forget.
pub fn torques(fk: &Posed) -> JointVec {
    // Joint j carries every downstream segment i (j..=last). Gravity on
    // segment i is (0, 0, -m·g_mag); its moment about joint j projected
    // onto the joint axis reduces to m·g_mag·(axis × r).z because gravity
    // has only a z component.
    // Any distal payload (gripper, fingers, tools) is folded into the last
    // segment's inertial at load, so it is carried here.
    std::array::from_fn(|j| {
        let origin_j = fk.origin_world(j);
        let axis_j = fk.axis_world(j);
        (j..crate::ARM_DOF)
            .map(|i| {
                let r = fk.com_world(i) - origin_j;
                fk.mass(i) * GRAVITY_MAGNITUDE * axis_j.cross(&r).z
            })
            .sum()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ARM_DOF;
    use crate::test_support::v1_fk;
    use std::f64::consts::FRAC_PI_2;

    fn gravity_at(side: &str, q: &JointVec) -> JointVec {
        let mut fk = v1_fk(side);
        torques(&fk.at(q))
    }

    // KDL `TreeIdSolver_RNE` over the whole tree with the gripper fingers at home,
    // so the distal payload is included (a serial `ChainDynParam` cannot). Gravity
    // is world -Z; the mirror arms differ at the same posture. Tolerance 1e-3 Nm;
    // sub-threshold values written 0.0. Regenerate with `tools/kdl_reference.cpp`.
    const POSTURES: [JointVec; 4] = [
        [0.0; ARM_DOF],
        [FRAC_PI_2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, FRAC_PI_2, 0.0, 0.0, 0.0],
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
    ];

    fn assert_matches_kdl(side: &str, expected: [JointVec; 4]) {
        for (q, exp) in POSTURES.iter().zip(&expected) {
            let tau = gravity_at(side, q);
            for i in 0..ARM_DOF {
                assert!(
                    (tau[i] - exp[i]).abs() < 1e-3,
                    "{side} arm, q={q:?}, joint {i}: actual={} expected={}",
                    tau[i],
                    exp[i],
                );
            }
        }
    }

    #[test]
    fn left_matches_kdl() {
        assert_matches_kdl(
            "left",
            [
                [0.0983, -0.0515, 0.0, -0.0299, 0.0, 0.0594, -0.0049],
                [10.4090, 0.0, -0.0515, -3.7841, -0.0648, 0.0, 0.4058],
                [-3.7157, -0.0515, 0.0, 3.7841, 0.0648, 0.0, -0.4058],
                [2.6572, -2.4663, 0.2762, -2.1801, 0.0351, -0.1392, 0.3296],
            ],
        );
    }

    #[test]
    fn right_matches_kdl() {
        assert_matches_kdl(
            "right",
            [
                [-0.0983, 0.0780, 0.0, -0.0299, 0.0, -0.0594, 0.0049],
                [10.4090, 0.0, -0.0781, 3.7841, -0.0648, 0.0, 0.4058],
                [3.7157, 0.0780, 0.0, 3.7841, -0.0648, 0.0, 0.4058],
                [-0.4429, -2.0486, 0.2803, -1.1611, 0.0865, -0.1971, 0.0782],
            ],
        );
    }

    #[test]
    fn matches_potential_energy_gradient_both_sides() {
        // Independent of KDL: gravity torque g(q) = ∂U/∂q with U = Σ mᵢ·g·z_comᵢ
        // over the segments (the last one carries the folded distal payload).
        fn potential(fk: &Posed) -> f64 {
            (0..ARM_DOF)
                .map(|i| fk.mass(i) * GRAVITY_MAGNITUDE * fk.com_world(i).z)
                .sum()
        }
        for side in ["left", "right"] {
            let mut fk = v1_fk(side);
            for q in [
                [0.0; ARM_DOF],
                [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
                [-0.5, 0.3, -0.2, 0.8, 0.1, 0.4, -0.3],
            ] {
                let tau = torques(&fk.at(&q));
                let h = 1e-6;
                for i in 0..ARM_DOF {
                    let (mut qp, mut qm) = (q, q);
                    qp[i] += h;
                    qm[i] -= h;
                    let grad = (potential(&fk.at(&qp)) - potential(&fk.at(&qm))) / (2.0 * h);
                    assert!(
                        (tau[i] - grad).abs() < 1e-3,
                        "{side:?} j{i}: tau={} grad={}",
                        tau[i],
                        grad
                    );
                }
            }
        }
    }
}
