//! How long a move is allowed to take, and how that changes underneath it.
//!
//! Both limbs budget a move once, when the goal is admitted, and then execute
//! it against a speed the operator can retune every tick. The budget is what
//! says a move is stuck rather than slow, so it has to move with the speed that
//! drives the motion; these three pieces are that one rule.

/// Grace multiple over a move's nominal duration before the runtime declares it
/// stuck and fails the goal. The nominal proves the unobstructed motion's
/// length; the governor can hold motion off that path, so allow this multiple
/// before aborting. The timeout tracks each move, not a flat ceiling.
pub const MOTION_TIMEOUT_FACTOR: f64 = 2.0;

/// Whether a move that has run `elapsed_s` has overrun its nominal `budget_s`
/// by more than [`MOTION_TIMEOUT_FACTOR`], the runtime abort condition.
pub fn motion_timed_out(elapsed_s: f64, budget_s: f64) -> bool {
    elapsed_s > budget_s * MOTION_TIMEOUT_FACTOR
}

/// A move's deadline together with the speed it was computed against. The two
/// travel as one value because they are only meaningful together: a caller that
/// recorded a new speed beside a budget still expressed in the old one would
/// read a later return to the original speed as a slowdown and extend a
/// deadline that never needed extending.
#[derive(Debug, Clone, Copy)]
pub struct MoveBudget {
    /// Nominal seconds the move should take; [`MOTION_TIMEOUT_FACTOR`] times
    /// this is the abort threshold.
    seconds: f64,
    /// The speed `seconds` was computed against, in that limb's own unit.
    budgeted_at: f64,
}

impl MoveBudget {
    /// A budget of `seconds`, computed against `rate`.
    pub fn new(seconds: f64, rate: f64) -> Self {
        Self {
            seconds,
            budgeted_at: rate,
        }
    }

    /// The nominal duration, for the operator-facing overrun message.
    pub fn seconds(self) -> f64 {
        self.seconds
    }

    /// Whether a move that has run `elapsed_s` has overrun this budget.
    pub fn timed_out(self, elapsed_s: f64) -> bool {
        motion_timed_out(elapsed_s, self.seconds)
    }

    /// This budget under a retuned `rate`. Both limbs budget a move once, at
    /// admission, and then execute it under a speed the operator can change
    /// every tick: the gripper's opening rate and the arm's end-effector speed.
    /// Lowering either drives the motion slower than the deadline it is judged
    /// against, so without this the move aborts short of its target reporting
    /// an overrun it did not cause.
    ///
    /// Only a slowdown moves anything. Speeding up returns `self` untouched,
    /// reference speed included: the move lands sooner anyway, and shrinking a
    /// deadline under a move that has already legitimately spent time at the
    /// old speed would abort it for going fast. The extension scales what is
    /// *left* of the budget, never what is spent, so a move that stops making
    /// progress still runs out of it.
    #[must_use]
    pub fn after_rate_change(self, elapsed_s: f64, rate: f64) -> Self {
        if !(rate.is_finite() && rate > 0.0) || rate >= self.budgeted_at {
            return self;
        }
        Self {
            seconds: elapsed_s + (self.seconds - elapsed_s).max(0.0) * self.budgeted_at / rate,
            budgeted_at: rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The timeout scales with the move's nominal duration, not a flat ceiling:
    // an 8 s nominal tolerates up to 16 s (2x), a 1 s nominal only 2 s.
    #[test]
    fn motion_timeout_scales_with_the_nominal_budget() {
        // Long validated motion gets proportionally longer before it is stuck.
        assert!(!motion_timed_out(15.0, 8.0));
        assert!(motion_timed_out(17.0, 8.0));
        // Short validated motion is held to a short leash.
        assert!(!motion_timed_out(1.9, 1.0));
        assert!(motion_timed_out(2.1, 1.0));
    }

    #[test]
    fn halving_the_rate_mid_move_doubles_what_is_left_of_the_budget() {
        // A quarter spent at the old rate is not rescaled; the remaining three
        // quarters need twice as long.
        let b = MoveBudget::new(1.0, 1.0).after_rate_change(0.25, 0.5);
        assert!((b.seconds() - 1.75).abs() < 1e-12, "got {}", b.seconds());
    }

    #[test]
    fn raising_the_rate_mid_move_never_shortens_the_deadline() {
        assert_eq!(
            MoveBudget::new(1.0, 1.0)
                .after_rate_change(0.9, 4.0)
                .seconds(),
            1.0
        );
    }

    #[test]
    fn returning_to_the_budgeted_rate_after_a_rise_does_not_extend_anything() {
        // A rise leaves the budget expressed in the original rate, so the
        // original rate must not then read as a slowdown against it.
        let b = MoveBudget::new(1.0, 1.0)
            .after_rate_change(0.0, 2.0)
            .after_rate_change(0.0, 1.0);
        assert_eq!(b.seconds(), 1.0, "a round trip must not move the deadline");
    }

    #[test]
    fn a_move_that_stops_progressing_still_runs_out_of_budget() {
        // Only a rate change moves the deadline, so a steady rate leaves it be
        // and the timeout still fires.
        let elapsed = 1.0 * MOTION_TIMEOUT_FACTOR + 0.01;
        let b = MoveBudget::new(1.0, 1.0).after_rate_change(elapsed, 1.0);
        assert_eq!(b.seconds(), 1.0);
        assert!(b.timed_out(elapsed));
    }

    #[test]
    fn an_unusable_rate_leaves_the_budget_alone() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let b = MoveBudget::new(1.0, 1.0).after_rate_change(0.01, bad);
            assert_eq!(b.seconds(), 1.0, "rate {bad}");
        }
    }
}
