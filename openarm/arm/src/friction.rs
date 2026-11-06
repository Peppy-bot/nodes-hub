//! Per-joint friction feedforward torque: a pure function of joint velocity (no
//! rigid-body model needed).
//!
//! Tanh model, per joint:
//!
//!   τ_fric[i] = Fo[i] + Fv[i]·ω[i] + Fc[i]·tanh(k[i]·ω[i])

use openarm_description::{ARM_DOF, HardwareVersion};
use srs_model::JointVec;

/// Per-joint friction-model constants for the tanh model above.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    pub fc: JointVec,
    pub fv: JointVec,
    pub fo: JointVec,
    pub k: JointVec,
}

/// The viscous term of openarm_teleop's `config/follower.yaml`, on both generations.
const FV: JointVec = [0.063, 0.063, 0.604, 0.813, 0.029, 0.072, 0.084];

/// The offset of openarm_teleop's `config/follower.yaml`, on both generations. `Fo` is a
/// directional Coulomb-breakaway offset defined in each joint's positive direction, so at
/// zero velocity the model commands a small directional bias. v2 reoriented joints 1, 3 and
/// 4, and its sign on them is unconfirmed: a flipped joint would need its `Fo` flipped.
const FO: JointVec = [0.088, 0.088, 0.008, -0.058, 0.005, 0.009, -0.059];

/// v1's constants, from `config/follower.yaml`: this node is a stiff position controller,
/// the teleop follower's role. `follower.yaml` and `leader.yaml` agree on every constant
/// except `Fc[5]` (joint 6: follower 0.093, leader 0.083). The `coef_tmp = 0.1` tanh
/// softening that `ComputeFriction` always applies is folded into `k` (so `k` here is the
/// yaml `k` times 0.1).
const V1: Params = Params {
    fc: [0.306, 0.306, 0.40, 0.166, 0.050, 0.093, 0.172],
    fv: FV,
    fo: FO,
    k: [2.8417, 2.8417, 2.9065, 13.0038, 15.1771, 24.2287, 0.7888],
};

/// v2's constants, fit on a real v2 at kp 360 on a wide two-arm joint path at 40 s, 20 s and
/// a duration of 0. `Fc` leaves about zero of the torque a joint's position error holds
/// against the sign of its commanded velocity, times kp, averaged over the paths and both
/// arms. On joints 1 to 4 that torque, with the feedforward's own Coulomb term added, reads
/// within 0.13 Nm at kp 120, 240 and 360. `k` brings the Coulomb term to 90% of `Fc` at
/// 0.0075 rad/s, the command's speed at the median start of the 40 s path once it has moved
/// two encoder counts, so a joint held by stiction breaks free with the command. The
/// slowest starts read 0.0031 to 0.0055 rad/s there and get 55% to 80%. The term reads the
/// commanded velocity, so its slope at zero speed adds no feedback.
const V2: Params = Params {
    fc: [0.83, 0.49, 0.60, 1.09, 0.095, 0.13, 0.11],
    fv: FV,
    fo: FO,
    k: [200.0; ARM_DOF],
};

/// The friction constants of one hardware generation.
pub fn params(version: HardwareVersion) -> Params {
    match version {
        HardwareVersion::V1 => V1,
        HardwareVersion::V2 => V2,
    }
}

/// Friction torque at velocity `dq` for the given constants.
pub fn torques(p: &Params, dq: &JointVec) -> JointVec {
    std::array::from_fn(|i| p.fo[i] + p.fv[i] * dq[i] + p.fc[i] * f64::tanh(p.k[i] * dq[i]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The command's speed at the median start of a 40 s path once it has moved two encoder
    /// counts, and how much of `Fc` the Coulomb term gives by then on v2.
    const MEDIAN_START_RAD_S: f64 = 0.0075;
    const COULOMB_SHARE_AT_MEDIAN_START: f64 = 0.9;

    /// Self-contained constants for the `torques` math (distinct, non-zero per
    /// joint), independent of each generation's constants.
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
    fn each_generation_reads_its_own_constants() {
        assert_eq!(params(HardwareVersion::V1), V1);
        assert_eq!(params(HardwareVersion::V2), V2);
        assert_ne!(V1, V2);
    }

    #[test]
    fn each_generations_constants_are_finite() {
        for p in [V1, V2] {
            for arr in [&p.fc, &p.fv, &p.fo, &p.k] {
                assert!(arr.iter().all(|x| x.is_finite()));
            }
        }
    }

    #[test]
    fn each_generations_coulomb_term_pushes_with_the_motion() {
        for p in [V1, V2] {
            assert!(p.fc.iter().chain(&p.k).all(|&x| x > 0.0));
        }
    }

    #[test]
    fn v2s_coulomb_term_is_near_full_at_a_median_start() {
        let v2 = params(HardwareVersion::V2);
        for (i, &k) in v2.k.iter().enumerate() {
            let share = f64::tanh(k * MEDIAN_START_RAD_S);
            assert!(share >= COULOMB_SHARE_AT_MEDIAN_START, "joint {i}: {share}");
        }
    }
}
