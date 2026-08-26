//! How much of a proposed step each governed DOF is permitted to take.
//!
//! An [`Allowance`] is the one currency every limiter speaks: a per-DOF
//! fraction of the proposed motion, `1.0` for all of it and `0.0` to freeze
//! that DOF where it is. Limiters never see each other's output, and
//! [`Limits::add`] combines them by keeping the most restrictive fraction per
//! DOF, so the order they run in cannot change the governed step.
//!
//! [`Limits`] also carries which limiter set each DOF's fraction, so the log
//! line and the operator readout can name the mechanism that is actually
//! binding instead of reconstructing it from flags threaded across the module.

use super::super::{GOV_DOF, Step};

/// Name reported for a DOF no limiter has restricted.
const UNRESTRICTED: &str = "unrestricted";

/// The fraction of its proposed motion each governed DOF may take this tick.
///
/// Construction is total: a non-finite fraction reads as `0.0` (a limiter that
/// cannot decide must never widen a step) and finite values clamp into
/// `[0, 1]`. Every value of this type is therefore a valid attenuation, so no
/// composition of allowances can add motion or reverse it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::governor) struct Allowance([f64; GOV_DOF]);

impl Allowance {
    /// Take all of the proposed motion.
    pub const FULL: Self = Self([1.0; GOV_DOF]);

    /// Freeze every DOF at `prev`.
    pub const FREEZE: Self = Self([0.0; GOV_DOF]);

    pub fn new(fractions: [f64; GOV_DOF]) -> Self {
        Self(std::array::from_fn(|i| {
            let f = fractions[i];
            // A non-finite fraction is a limiter that could not decide; fail
            // safe to freezing that DOF rather than trusting it.
            if f.is_finite() {
                f.clamp(0.0, 1.0)
            } else {
                0.0
            }
        }))
    }

    /// `1.0` where `free` holds, `0.0` elsewhere: the shape of a gate that
    /// stops a limb without shaping it.
    pub fn gate(free: impl Fn(usize) -> bool) -> Self {
        Self::new(std::array::from_fn(|i| if free(i) { 1.0 } else { 0.0 }))
    }

    fn get(&self, dof: usize) -> f64 {
        self.0[dof]
    }

    /// Apply to a step, yielding the governed configuration.
    ///
    /// The endpoints are exact rather than interpolated: interpolating all the
    /// way to `target` is not bit-identical to `target` in floating point, and
    /// both the passthrough paths and the followers require an unrestricted DOF
    /// to carry the commanded value itself, not a value one ulp away from it.
    pub fn apply(&self, step: &Step) -> [f64; GOV_DOF] {
        std::array::from_fn(|i| match self.0[i] {
            f if f >= 1.0 => step.target[i],
            f if f <= 0.0 => step.prev[i],
            f => step.prev[i] + f * (step.target[i] - step.prev[i]),
        })
    }
}

/// The combined allowance of every limiter that ran, plus which one set each
/// DOF's fraction.
#[derive(Clone, Copy, Debug)]
pub(in crate::governor) struct Limits {
    allowance: Allowance,
    binding: [&'static str; GOV_DOF],
}

impl Limits {
    /// Nothing restricted yet.
    pub fn unrestricted() -> Self {
        Self {
            allowance: Allowance::FULL,
            binding: [UNRESTRICTED; GOV_DOF],
        }
    }

    /// Fold in one limiter's allowance, keeping the more restrictive fraction per
    /// DOF. Commutative and associative in the fraction, which is what lets
    /// limiters run in any order.
    ///
    /// Only a strictly tighter allowance claims a DOF's name, so a limiter that
    /// merely ties an existing bound does not take credit for it and the
    /// recorded name is stable as well as the fraction.
    pub fn add(mut self, name: &'static str, allowance: Allowance) -> Self {
        for i in 0..GOV_DOF {
            if allowance.get(i) < self.allowance.get(i) {
                self.allowance.0[i] = allowance.get(i);
                self.binding[i] = name;
            }
        }
        self
    }

    /// Whether any DOF is restricted at all.
    pub fn restricted(&self) -> bool {
        (0..GOV_DOF).any(|i| self.allowance.get(i) < 1.0)
    }

    /// Whether any DOF is denied outright.
    pub fn frozen(&self) -> bool {
        (0..GOV_DOF).any(|i| self.allowance.get(i) <= 0.0)
    }

    /// The limiter binding the most-restricted DOF, for the log line and the
    /// operator readout. `None` when nothing is restricted.
    pub fn tightest(&self) -> Option<&'static str> {
        (0..GOV_DOF)
            .filter(|&i| self.allowance.get(i) < 1.0)
            .min_by(|&a, &b| self.allowance.get(a).total_cmp(&self.allowance.get(b)))
            .map(|i| self.binding[i])
    }

    pub fn apply(&self, step: &Step) -> [f64; GOV_DOF] {
        self.allowance.apply(step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::governor::{DUAL_DOF, LEFT_GRIPPER};

    const A: &str = "a";
    const B: &str = "b";

    fn uniform(f: f64) -> Allowance {
        Allowance::new([f; GOV_DOF])
    }

    fn ramp(base: f64) -> Allowance {
        Allowance::new(std::array::from_fn(|i| base + 0.01 * i as f64))
    }

    fn step(prev: f64, target: f64) -> Step {
        Step {
            prev: [prev; GOV_DOF],
            target: [target; GOV_DOF],
        }
    }

    #[test]
    fn construction_clamps_into_the_unit_range_and_freezes_on_non_finite() {
        let a = Allowance::new(std::array::from_fn(|i| match i {
            0 => -0.5,
            1 => 1.5,
            2 => f64::NAN,
            3 => f64::INFINITY,
            4 => f64::NEG_INFINITY,
            _ => 0.25,
        }));
        assert_eq!(a.get(0), 0.0);
        assert_eq!(a.get(1), 1.0);
        assert_eq!(a.get(2), 0.0, "NaN must fail safe to frozen, not to full");
        assert_eq!(a.get(3), 0.0);
        assert_eq!(a.get(4), 0.0);
        assert_eq!(a.get(5), 0.25);
    }

    #[test]
    fn combining_is_order_independent() {
        let (x, y, z) = (ramp(0.1), uniform(0.5), ramp(0.7));
        let forward = Limits::unrestricted().add(A, x).add(B, y).add("c", z);
        let reverse = Limits::unrestricted().add("c", z).add(B, y).add(A, x);
        for i in 0..GOV_DOF {
            assert_eq!(
                forward.allowance.get(i),
                reverse.allowance.get(i),
                "dof {i}"
            );
        }
    }

    #[test]
    fn a_full_allowance_is_the_identity() {
        let base = Limits::unrestricted().add(A, ramp(0.3));
        let with_full = base.add(B, Allowance::FULL);
        for i in 0..GOV_DOF {
            assert_eq!(with_full.allowance.get(i), base.allowance.get(i));
            assert_eq!(with_full.binding[i], base.binding[i]);
        }
    }

    #[test]
    fn full_allowance_yields_the_target_bit_exactly() {
        // Two joint angles, in range, for which `prev + 1.0 * (target - prev)`
        // lands one ulp off `target`. The passthrough paths and the followers
        // require an unrestricted DOF to carry the commanded value itself, so
        // `apply` returns the endpoint rather than interpolating to it.
        let (prev, target) = (2.1508107542920776, -1.2623442820099424);
        assert!(
            prev + 1.0 * (target - prev) != target,
            "the trap this guards"
        );

        let s = step(prev, target);
        assert_eq!(Allowance::FULL.apply(&s), s.target);
    }

    #[test]
    fn zero_allowance_yields_prev_bit_exactly() {
        let s = step(0.1, 0.3);
        assert_eq!(Allowance::FREEZE.apply(&s), s.prev);
    }

    #[test]
    fn partial_allowance_interpolates() {
        let s = step(0.0, 2.0);
        let governed = uniform(0.25).apply(&s);
        assert!(governed.iter().all(|&q| (q - 0.5).abs() < 1e-12));
    }

    #[test]
    fn a_gate_frees_only_the_dof_it_names() {
        let free = Allowance::gate(|i| i < DUAL_DOF);
        assert_eq!(free.get(0), 1.0);
        assert_eq!(free.get(DUAL_DOF - 1), 1.0);
        assert_eq!(free.get(LEFT_GRIPPER), 0.0);
    }

    #[test]
    fn limits_record_the_limiter_that_bound_each_dof() {
        let limits = Limits::unrestricted().add(A, uniform(0.6)).add(
            B,
            Allowance::new(std::array::from_fn(|i| if i == 0 { 0.2 } else { 0.9 })),
        );
        assert_eq!(limits.allowance.get(0), 0.2);
        assert_eq!(limits.binding[0], B);
        assert_eq!(limits.allowance.get(1), 0.6, "B's 0.9 loses to A's 0.6");
        assert_eq!(limits.binding[1], A);
        assert_eq!(limits.tightest(), Some(B));
    }

    #[test]
    fn a_tie_leaves_the_earlier_limiter_recorded() {
        let limits = Limits::unrestricted()
            .add(A, uniform(0.5))
            .add(B, uniform(0.5));
        assert_eq!(
            limits.binding[0], A,
            "only a strictly tighter allowance claims a DOF"
        );
    }

    #[test]
    fn unrestricted_limits_report_nothing_binding() {
        let limits = Limits::unrestricted().add(A, Allowance::FULL);
        assert!(!limits.restricted());
        assert!(!limits.frozen());
        assert_eq!(limits.tightest(), None);
    }

    #[test]
    fn frozen_is_distinct_from_merely_restricted() {
        let restricted = Limits::unrestricted().add(A, uniform(0.5));
        assert!(restricted.restricted());
        assert!(!restricted.frozen());

        let frozen = Limits::unrestricted().add(A, uniform(0.0));
        assert!(frozen.restricted());
        assert!(frozen.frozen());
    }

    #[test]
    fn the_combined_step_never_leaves_the_span_from_prev_to_target() {
        // The invariant the whole pipeline rests on: a limiter may only pull a
        // DOF back toward `prev`, never past it and never beyond `target`.
        let s = Step {
            prev: [0.0; GOV_DOF],
            target: std::array::from_fn(|i| if i % 2 == 0 { 1.0 } else { -1.0 }),
        };
        let governed = Limits::unrestricted()
            .add(A, ramp(0.1))
            .add(B, uniform(0.35))
            .apply(&s);
        for (i, (&q, &target)) in governed.iter().zip(s.target.iter()).enumerate() {
            let (lo, hi) = (target.min(0.0), target.max(0.0));
            assert!(q >= lo && q <= hi, "dof {i}: {q} outside [{lo}, {hi}]");
        }
    }
}
