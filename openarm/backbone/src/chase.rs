//! Setpoint shaping: the rate-limited chase that absorbs target jumps, and the
//! clamp that keeps streamed joint targets inside the arm's limits.
//!
//! [`rate_limited`] is the one place a per-tick rate limit is applied. Every
//! chase in the node goes through it (the arm's joints, the grippers' opening, the
//! servo's post-filter re-clamp), so they cannot round or bound differently.

use srs_model::Limit;

use crate::types::{ARM_DOF, JointVec};

/// Advance `from` one tick toward `target`, moving at most `rate * dt`. A
/// target within reach lands on it exactly.
///
/// `rate` and `dt` must be finite and non-negative: the clamp bounds are
/// `[-rate * dt, rate * dt]` and `f64::clamp` panics if they invert, so an
/// unvalidated rate fails loudly at the call site rather than limping. Every
/// current caller's rate is bringup-validated.
pub(crate) fn rate_limited(from: f64, target: f64, rate: f64, dt: f64) -> f64 {
    let max_step = rate * dt;
    from + (target - from).clamp(-max_step, max_step)
}

/// Advance the setpoint one tick toward the target, each joint stepping at
/// most its own `vmax * dt`: the velocity-limited chase that absorbs target
/// jumps.
pub(crate) fn chase_step(
    setpoint: &JointVec,
    target: &JointVec,
    vmax: &JointVec,
    dt: f64,
) -> JointVec {
    std::array::from_fn(|i| rate_limited(setpoint[i], target[i], vmax[i], dt))
}

/// Clamp a streamed target into the arm's joint position limits, so an
/// out-of-range command tracks to the nearest reachable value instead of
/// driving past a limit.
pub(crate) fn clamp_to_limits(q: &JointVec, limits: &[Limit; ARM_DOF]) -> JointVec {
    std::array::from_fn(|i| q[i].clamp(limits[i].lo, limits[i].hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The scalar rate limit the arm chase, the gripper chase and the servo's
    // post-filter clamp all share.
    #[test]
    fn rate_limited_caps_both_directions_and_lands_exactly() {
        assert_eq!(rate_limited(0.0, 1.0, 1.0, 0.01), 0.01);
        assert_eq!(rate_limited(0.0, -1.0, 1.0, 0.01), -0.01);
        // Within one step: land on the target, not near it.
        assert_eq!(rate_limited(0.995, 1.0, 1.0, 0.01), 1.0);
        // Already there: hold, bit-exact.
        assert_eq!(rate_limited(0.3, 0.3, 1.0, 0.01), 0.3);
    }

    #[test]
    fn chase_step_caps_each_joint_at_its_velocity_limit() {
        let setpoint = [0.0; ARM_DOF];
        let mut target = [0.0; ARM_DOF];
        target[0] = 1.0; // far above j1's per-tick step
        target[1] = -1.0; // far below j2's
        let mut vmax = [1.0; ARM_DOF];
        vmax[1] = 2.0;
        let next = chase_step(&setpoint, &target, &vmax, 0.01);
        assert_eq!(next[0], 0.01);
        assert_eq!(next[1], -0.02);
        assert_eq!(next[2..], [0.0; 5]);
    }

    #[test]
    fn chase_step_lands_exactly_on_a_near_target() {
        let mut setpoint = [0.0; ARM_DOF];
        setpoint[0] = 0.995;
        let mut target = [0.0; ARM_DOF];
        target[0] = 1.0; // within one 0.01 step
        let next = chase_step(&setpoint, &target, &[1.0; ARM_DOF], 0.01);
        assert_eq!(next[0], 1.0);
    }

    #[test]
    fn chase_step_holds_when_target_equals_setpoint() {
        let q = [0.3, -0.2, 0.1, 0.5, 0.0, -0.4, 0.2];
        assert_eq!(chase_step(&q, &q, &[1.0; ARM_DOF], 0.01), q);
    }

    #[test]
    fn clamp_to_limits_pins_out_of_range_joints() {
        let limits = [Limit { lo: -1.0, hi: 1.0 }; ARM_DOF];
        let mut q = [0.0; ARM_DOF];
        q[0] = 2.0;
        q[1] = -2.0;
        q[2] = 0.5;
        let clamped = clamp_to_limits(&q, &limits);
        assert_eq!(clamped[0], 1.0);
        assert_eq!(clamped[1], -1.0);
        assert_eq!(clamped[2], 0.5);
    }
}
