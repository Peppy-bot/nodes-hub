//! The smoothing of a streamed pose's steps: one Butterworth per joint at MoveIt
//! Servo's default cutoff for the control period, bounding the jerk of the
//! command.

use std::f64::consts::PI;
use std::time::Duration;

use control_core::filters::ButterworthFilter;
use srs_model::ARM_DOF;
use srs_model::JointVec;

/// MoveIt Servo's default first-order low-pass coefficient
/// (`low_pass_filter_coeff`), whose -3 dB cutoff at a control period `Ts` is
/// `atan(1 / coeff) / (pi Ts)`.
const MOVEIT_LOW_PASS_COEFF: f64 = 1.5;

/// One second-order Butterworth per joint, so it smooths at least as much as
/// MoveIt's first-order filter.
pub(super) struct StreamSmoothing([ButterworthFilter; ARM_DOF]);

impl StreamSmoothing {
    /// The smoothing for a control `period`, which is not zero.
    pub(super) fn new(period: Duration) -> Self {
        let period_s = period.as_secs_f64();
        let cutoff_hz = (1.0 / MOVEIT_LOW_PASS_COEFF).atan() / (PI * period_s);
        let filter = ButterworthFilter::from_cutoff(cutoff_hz, period_s)
            .expect("the cutoff is a fixed share of Nyquist, below it at every period");
        Self([filter; ARM_DOF])
    }

    /// Forget every filter's state, so the next step seeds them on itself.
    pub(super) fn reset(&mut self) {
        for filter in &mut self.0 {
            filter.reset();
        }
    }

    /// The smoothed command for `q`.
    pub(super) fn smooth(&mut self, q: &JointVec) -> JointVec {
        std::array::from_fn(|i| self.0[i].filter(q[i]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PERIOD: Duration = Duration::from_millis(10);
    const START: JointVec = [0.1, -0.2, 0.3, 0.9, -0.4, 0.5, 0.6];

    #[test]
    fn every_control_rate_carries_the_cutoff() {
        for rate_hz in [1, 30, 100, 1_000, 10_000] {
            StreamSmoothing::new(Duration::from_secs_f64(1.0 / f64::from(rate_hz)));
        }
    }

    #[test]
    fn the_first_step_passes_through_and_the_next_are_smoothed() {
        let mut smoothing = StreamSmoothing::new(PERIOD);
        assert_eq!(smoothing.smooth(&START), START);
        let far = START.map(|q| q + 1.0);
        let next = smoothing.smooth(&far);
        for i in 0..ARM_DOF {
            assert!(
                next[i] > START[i] && next[i] < far[i],
                "joint {i} went to {} on the way to {}",
                next[i],
                far[i]
            );
        }
    }

    #[test]
    fn a_reset_seeds_the_next_step_on_itself() {
        let mut smoothing = StreamSmoothing::new(PERIOD);
        smoothing.smooth(&START);
        smoothing.reset();
        let far = START.map(|q| q + 1.0);
        assert_eq!(smoothing.smooth(&far), far);
    }
}
