//! Defense in depth against tracking error: the whole tripwire in one place.
//!
//! The commanded barrier shapes only the commanded stream and cannot see how
//! well the arms follow it. This watches the real clearance, holds closing
//! motion when the bodies have actually closed, and releases when they recover.
//!
//! Its hysteresis latch, its thresholds, the clearances it needs and the
//! decision it makes all live here. [`super::super::sense`] calls
//! [`Governor::sense_tripwire`] during the tick's single model-read phase; by
//! the time [`MeasuredTripwire`] votes, the decision is already data.

use tracing::{info, warn};

use super::super::{ArmPair, GovState, Governor, Step, concat, is_left_dof};
use super::Limiter;
use super::allowance::Allowance;

/// The measured-state tripwire trips when the real clearance drops below this
/// fraction of `d_stop`, and releases only once it recovers past the full
/// `d_stop` (hysteresis). Sitting below the commanded floor leaves headroom for
/// tracking jitter at the wall, where the barrier parks the commanded clearance
/// at `d_stop`, so only a genuine divergence trips it. A module constant (not a
/// node parameter); promote it to a parameter when tuning on hardware.
pub(in crate::governor) const MONITOR_TRIP_FRACTION: f64 = 0.5;
// The trip floor must sit strictly inside (0, d_stop) or the hysteresis band
// [trip_floor, d_stop) collapses and the latch logic degrades silently.
const _: () = assert!(MONITOR_TRIP_FRACTION > 0.0 && MONITOR_TRIP_FRACTION < 1.0);

/// Commanded-space clearances the measured-state tripwire needs to decide which
/// side is making the real breach worse. Sampled only while the latch is armed,
/// so an untripped tick costs one measured query and nothing else.
///
/// Judging in the commanded space (against `Sensed::d_prev`, the held
/// setpoint's own clearance) rather than against the measured clearance is what
/// keeps a systematic tracking offset from either waving every closing command
/// through or deadlocking every escape.
pub(in crate::governor) struct Tripwire {
    /// Clearance at the full candidate.
    pub d_cand: f64,
    /// Per side, the clearance of that side moving alone with the other held,
    /// and `None` when the side does not move or its query failed. A failed
    /// query reads as "does not open", which holds rather than frees.
    pub d_solo: ArmPair<Option<f64>>,
}

impl Governor {
    /// Update the hysteresis latch from the measured clearance and, while it is
    /// armed, sample the commanded-space clearances the gate needs.
    ///
    /// A failed measured query leaves the latch untouched and returns `None`
    /// (defer to the commanded barrier), so a bad measurement can neither block
    /// separation nor latch a hold.
    pub(in crate::governor) fn sense_tripwire(
        &mut self,
        prev: &GovState,
        cand: &GovState,
        measured: &GovState,
    ) -> Option<Tripwire> {
        let d_measured = self.distance_at(&concat(measured))?;
        let trip_floor = MONITOR_TRIP_FRACTION * self.d_stop;
        let threshold = if self.monitor_tripped {
            self.d_stop
        } else {
            trip_floor
        };
        let breached = d_measured < threshold;
        if breached != self.monitor_tripped {
            if breached {
                warn!(
                    "collision MONITOR: measured clearance past {trip_floor:+.4} m, blocking approach (separation still allowed)"
                );
            } else {
                info!("collision MONITOR: measured clearance recovered past d_stop, resuming");
            }
            self.monitor_tripped = breached;
        }
        if !breached {
            return None;
        }

        let d_cand = self.distance_at(&concat(cand))?;
        let solo = |side_is_left: bool| GovState {
            arms: ArmPair::new(
                if side_is_left { cand } else { prev }.arms.left,
                if side_is_left { prev } else { cand }.arms.right,
            ),
            grippers: ArmPair::new(
                if side_is_left { cand } else { prev }.grippers.left,
                if side_is_left { prev } else { cand }.grippers.right,
            ),
        };
        let moves = |side_is_left: bool| {
            let (c, p) = (cand, prev);
            if side_is_left {
                c.arms.left != p.arms.left || c.grippers.left != p.grippers.left
            } else {
                c.arms.right != p.arms.right || c.grippers.right != p.grippers.right
            }
        };
        let sample = |g: &mut Self, side_is_left: bool| -> Option<f64> {
            moves(side_is_left)
                .then(|| g.distance_at(&concat(&solo(side_is_left))))
                .flatten()
        };
        let d_solo = ArmPair::new(sample(self, true), sample(self, false));
        Some(Tripwire { d_cand, d_solo })
    }
}

/// Defense in depth against tracking error: the barrier shapes only the
/// commanded stream and cannot see how well the arms follow it, so this holds
/// closing motion whenever the *real* clearance has closed past the tripwire
/// floor, until it recovers.
///
/// The gate is per side (an arm and its gripper opening together): one
/// operator's closing push must not trap the other side's escape. When the
/// joint candidate closes, each side is re-judged with the other held, and any
/// sub-motion that does not worsen the commanded clearance stays free.
///
/// Whether the tripwire is armed at all, and the clearances this reads, are
/// decided in [`super::super::sense`]; by the time it limits, the decision is
/// already data, carried here by construction.
pub(in crate::governor) struct MeasuredTripwire<'tick> {
    /// Clearance of the held setpoint, the commanded-space reference.
    pub d_prev: f64,
    /// `Some` only while the latch is armed.
    pub tripwire: Option<&'tick Tripwire>,
}

impl Limiter for MeasuredTripwire<'_> {
    fn name(&self) -> &'static str {
        "measured-tripwire"
    }

    fn allow(&self, _step: &Step) -> Allowance {
        let Some(tripwire) = self.tripwire else {
            return Allowance::FULL;
        };
        // Judged against the held setpoint's own clearance, in the same
        // (commanded) space: comparing across spaces would pass every closing
        // command under a systematic tracking offset and freeze every escape
        // under the opposite one.
        let opens = |d: Option<f64>| d.filter(|d| *d >= self.d_prev);
        if tripwire.d_cand >= self.d_prev {
            return Allowance::FULL;
        }
        // The joint candidate closes, so free at most one side: the two can
        // each open alone yet still converge on the same gap together.
        match (opens(tripwire.d_solo.left), opens(tripwire.d_solo.right)) {
            (Some(left), Some(right)) => Allowance::gate(|i| is_left_dof(i) == (left >= right)),
            (Some(_), None) => Allowance::gate(is_left_dof),
            (None, Some(_)) => Allowance::gate(|i| !is_left_dof(i)),
            (None, None) => Allowance::FREEZE,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::types::ARM_DOF;

    use super::super::super::{GOV_DOF, LEFT_GRIPPER, RIGHT_GRIPPER};
    use super::*;

    const D_PREV: f64 = 0.004;

    fn allowance_for(d_solo: ArmPair<Option<f64>>, d_cand: f64) -> Allowance {
        let tripwire = Tripwire { d_cand, d_solo };
        let limiter = MeasuredTripwire {
            d_prev: D_PREV,
            tripwire: Some(&tripwire),
        };
        // The gate ignores the step's contents; any finite step will do.
        limiter.allow(&Step {
            prev: [0.0; GOV_DOF],
            target: [0.1; GOV_DOF],
        })
    }

    fn frees(a: &Allowance, left_side: bool) -> bool {
        let arm = if left_side { 0 } else { ARM_DOF };
        let gripper = if left_side {
            LEFT_GRIPPER
        } else {
            RIGHT_GRIPPER
        };
        let step = Step {
            prev: [0.0; GOV_DOF],
            target: [1.0; GOV_DOF],
        };
        let governed = a.apply(&step);
        governed[arm] == 1.0 && governed[gripper] == 1.0
    }

    /// The four-way side-freeing decision, pinned as a table without any
    /// geometry: which side stays free while the real clearance is breached is
    /// the sharpest call the limiter set makes.
    #[test]
    fn the_side_freeing_match_pins_all_four_arms_and_the_tie() {
        let closing = Some(D_PREV - 0.001);
        let opens_a_little = Some(D_PREV + 0.001);
        let opens_more = Some(D_PREV + 0.002);

        // Both sides open alone: the one opening more is freed, the other held.
        let a = allowance_for(ArmPair::new(opens_a_little, opens_more), 0.0);
        assert!(!frees(&a, true) && frees(&a, false));
        let a = allowance_for(ArmPair::new(opens_more, opens_a_little), 0.0);
        assert!(frees(&a, true) && !frees(&a, false));

        // A tie frees the left side (the match's documented >= choice).
        let a = allowance_for(ArmPair::new(opens_more, opens_more), 0.0);
        assert!(frees(&a, true) && !frees(&a, false));

        // Exactly one side opens alone: that side and only that side is free.
        let a = allowance_for(ArmPair::new(opens_more, closing), 0.0);
        assert!(frees(&a, true) && !frees(&a, false));
        let a = allowance_for(ArmPair::new(closing, opens_more), 0.0);
        assert!(!frees(&a, true) && frees(&a, false));

        // A side that does not move (or whose query failed) reads None and is
        // held, exactly like a closing one.
        let a = allowance_for(ArmPair::new(None, opens_more), 0.0);
        assert!(!frees(&a, true) && frees(&a, false));

        // Neither opens alone: everything freezes.
        let a = allowance_for(ArmPair::new(closing, None), 0.0);
        assert_eq!(a, Allowance::FREEZE);
    }

    /// The gate stands down entirely when the joint candidate does not close
    /// on the held setpoint, and when the latch is not armed at all.
    #[test]
    fn a_non_closing_candidate_and_an_unarmed_latch_pass_in_full() {
        let a = allowance_for(ArmPair::new(None, None), D_PREV);
        assert_eq!(a, Allowance::FULL, "candidate at d_prev is not closing");

        let unarmed = MeasuredTripwire {
            d_prev: D_PREV,
            tripwire: None,
        };
        let step = Step {
            prev: [0.0; GOV_DOF],
            target: [0.1; GOV_DOF],
        };
        assert_eq!(unarmed.allow(&step), Allowance::FULL);
    }
}
