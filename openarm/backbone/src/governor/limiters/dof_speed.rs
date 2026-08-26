//! The per-tick travel each governed DOF may take: the arm joint speed cap for
//! a joint, the gripper rate for a finger.
//!
//! The chase upstream already limits every DOF, so on a healthy tick this is
//! inert. It is here because the floor scan sizes its probe count from the same
//! bound: distance is not monotone along a segment, so a step past the bound
//! under-resolves the scan and can step over a pocket and miss a breach.
//! Clamping keeps that precondition true by construction, and keeps the robot
//! under control when something upstream is wrong.

use super::super::GOV_DOF;
use super::allowance::Allowance;
use super::{Limiter, Step};

pub(in crate::governor) struct DofSpeed {
    /// Largest excursion each DOF may take this tick.
    pub max_step: [f64; GOV_DOF],
}

impl Limiter for DofSpeed {
    fn name(&self) -> &'static str {
        "dof-speed"
    }

    fn allow(&self, step: &Step) -> Allowance {
        Allowance::new(std::array::from_fn(|i| {
            let excursion = (step.target[i] - step.prev[i]).abs();
            // `max_step` is positive (both rate bounds are validated above
            // zero), so a motionless DOF takes this branch and never divides.
            if excursion <= self.max_step[i] {
                1.0
            } else {
                self.max_step[i] / excursion
            }
        }))
    }
}
