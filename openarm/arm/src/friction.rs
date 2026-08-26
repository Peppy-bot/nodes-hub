//! Per-joint friction feedforward torque: a pure function of joint velocity (no
//! rigid-body model needed).
//!
//! Tanh model, per joint:
//!
//!   τ_fric[i] = Fo[i] + Fv[i]·ω[i] + Fc[i]·tanh(k[i]·ω[i])

use crate::JointVec;

/// Per-joint friction-model constants for the tanh model above.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub fc: JointVec,
    pub fv: JointVec,
    pub fo: JointVec,
    pub k: JointVec,
}

/// Friction constants from openarm_teleop's `config/follower.yaml`. This node is a stiff
/// position controller, so it matches the teleop follower role. `follower.yaml` and
/// `leader.yaml` agree on every constant except `Fc[5]` (joint 6: follower 0.093, leader
/// 0.083). The `coef_tmp = 0.1` tanh softening that `ComputeFriction` always applies is
/// folded into `k` (so `k` here is the yaml `k` times 0.1), making the runtime expression
/// `Fo + Fv·ω + Fc·tanh(k·ω)`.
///
/// The enactic teleop config is not versioned, so these apply to both hardware
/// generations. `Fo` is a directional Coulomb-breakaway offset defined in each joint's
/// positive direction; v2.0 reoriented joints 1/3/4, so confirm the per-joint sign
/// convention against the v2 URDF axes at bring-up (a flipped joint would need its `Fo`
/// sign flipped). Not defaulted differently in code because the reference does not.
///
/// These are the physical (full) friction torques, applied directly at full weight
/// (matching the openarm teleop follower). Fo is a non-zero static offset, so at rest the
/// model commands a small directional bias (intentional Coulomb breakaway).
pub const NOMINAL: Params = Params {
    fc: [0.306, 0.306, 0.40, 0.166, 0.050, 0.093, 0.172],
    fv: [0.063, 0.063, 0.604, 0.813, 0.029, 0.072, 0.084],
    fo: [0.088, 0.088, 0.008, -0.058, 0.005, 0.009, -0.059],
    k: [2.8417, 2.8417, 2.9065, 13.0038, 15.1771, 24.2287, 0.7888],
};

/// Friction torque at velocity `qdot` for the given constants.
pub fn torques(p: &Params, qdot: &JointVec) -> JointVec {
    std::array::from_fn(|i| p.fo[i] + p.fv[i] * qdot[i] + p.fc[i] * f64::tanh(p.k[i] * qdot[i]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ARM_DOF;

    /// Self-contained constants for the `torques` math (distinct, non-zero per
    /// joint), independent of the production [`NOMINAL`] values.
    const FIXTURE: Params = Params {
        fc: [0.09, 0.10, 0.12, 0.05, 0.015, 0.025, 0.05],
        fv: [0.02, 0.03, 0.18, 0.24, 0.009, 0.02, 0.025],
        fo: [0.026, 0.027, 0.002, -0.017, 0.0015, 0.0027, -0.018],
        k: [2.84, 2.90, 2.95, 13.0, 15.0, 24.0, 0.79],
    };

    #[test]
    fn at_zero_velocity_equals_offset() {
        let p = &FIXTURE;
        let tau = torques(p, &[0.0; ARM_DOF]);
        // ω=0 → tanh(0)=0, Fv·ω=0, so τ = Fo.
        for (i, &t) in tau.iter().enumerate() {
            assert!(
                (t - p.fo[i]).abs() < 1e-12,
                "joint {i}: tau={t} Fo={}",
                p.fo[i]
            );
        }
    }

    #[test]
    fn at_high_velocity_saturates() {
        let (p, omega) = (&FIXTURE, 100.0);
        let tau = torques(p, &[omega; ARM_DOF]);
        // ω large positive → tanh→+1, so τ ≈ Fo + Fv·ω + Fc.
        for (i, &t) in tau.iter().enumerate() {
            let expected = p.fo[i] + p.fv[i] * omega + p.fc[i];
            assert!(
                (t - expected).abs() < 1e-6,
                "joint {i}: tau={t} expected={expected}"
            );
        }
    }

    #[test]
    fn antisymmetric_about_zero_modulo_offset() {
        // Coulomb + viscous components are odd in ω; only Fo breaks antisymmetry.
        let (p, omega) = (&FIXTURE, 0.5);
        let pos = torques(p, &[omega; ARM_DOF]);
        let neg = torques(p, &[-omega; ARM_DOF]);
        for (i, (&pp, &nn)) in pos.iter().zip(&neg).enumerate() {
            assert!((pp + nn - 2.0 * p.fo[i]).abs() < 1e-9, "joint {i}");
        }
    }

    #[test]
    fn nominal_constants_are_complete_and_finite() {
        for arr in [&NOMINAL.fc, &NOMINAL.fv, &NOMINAL.fo, &NOMINAL.k] {
            assert!(arr.iter().all(|x| x.is_finite()));
        }
    }
}
