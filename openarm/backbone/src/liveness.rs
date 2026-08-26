//! Whether a follower is still delivering state the backbone can act on.
//!
//! A limb the backbone cannot currently see is a limb it must not command. Its
//! held setpoint keeps advancing on the operator's stream while the real arm
//! does whatever an uncommanded arm does (droops, gets power-cycled, gets moved
//! by hand), and the two diverge without bound. When the follower comes back it
//! is handed that setpoint, and a position-controlled arm steps to it in one
//! tick: the larger the divergence, the larger the torque.
//!
//! Two rules close that. A limb whose deliveries have stopped is frozen and its
//! wire goes silent, so the follower holds its own last setpoint (which is
//! about where it actually is) rather than tracking one the backbone can no
//! longer vouch for. And the first delivery after any gap is flagged for
//! re-anchoring, so the held setpoint is reset to the measured pose before the
//! limb moves again, leaving the ordinary per-joint velocity limit to walk it
//! back to the operator's command.

use std::time::{Duration, Instant};

/// How many control periods of silence before a limb is declared stale. A
/// healthy follower delivers every period, so this tolerates four consecutive
/// missed deliveries plus scheduler jitter; a stream that quiet is broken
/// rather than jittery.
const STALE_PERIODS: u32 = 4;

/// The silence a limb is allowed before it must not be acted on.
pub fn stale_limit(cycle_period: Duration) -> Duration {
    cycle_period * STALE_PERIODS
}

/// What the backbone may do with a limb this tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// Deliveries are current; command the limb normally.
    Live,
    /// First delivery after a gap. Re-anchor the held setpoint on the measured
    /// pose before commanding, or the limb steps by however far it drifted
    /// while the backbone could not see it.
    Reanchor,
    /// Nothing delivered within the limit. Freeze the limb and stay silent on
    /// its wire.
    Stale,
}

/// Per-limb delivery watchdog.
#[derive(Debug)]
pub struct Liveness {
    last_delivery: Instant,
    live: bool,
}

impl Liveness {
    /// Start live: the caller seeds a limb from its first measurement before
    /// the loop runs, which is the anchor a `Reanchor` would establish.
    pub fn seeded(now: Instant) -> Self {
        Self {
            last_delivery: now,
            live: true,
        }
    }

    /// Judge the limb, given whether it delivered since the last tick.
    pub fn admit(&mut self, delivered: bool, now: Instant, limit: Duration) -> Admission {
        if delivered {
            self.last_delivery = now;
        }
        // Saturating: a `now` behind `last_delivery` cannot happen on a
        // monotonic clock, and reading it as infinitely stale would be a
        // spurious freeze.
        match (
            now.saturating_duration_since(self.last_delivery) <= limit,
            self.live,
        ) {
            (true, true) => Admission::Live,
            (true, false) => {
                self.live = true;
                Admission::Reanchor
            }
            (false, _) => {
                self.live = false;
                Admission::Stale
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PERIOD: Duration = Duration::from_millis(10);

    fn limit() -> Duration {
        stale_limit(PERIOD)
    }

    #[test]
    fn stale_limit_tolerates_four_missed_deliveries() {
        assert_eq!(limit(), Duration::from_millis(40));
    }

    #[test]
    fn a_delivering_limb_stays_live() {
        let start = Instant::now();
        let mut l = Liveness::seeded(start);
        for tick in 1..=100 {
            let now = start + PERIOD * tick;
            assert_eq!(l.admit(true, now, limit()), Admission::Live);
        }
    }

    #[test]
    fn silence_past_the_limit_goes_stale_and_the_first_delivery_back_re_anchors() {
        let start = Instant::now();
        let mut l = Liveness::seeded(start);

        // Inside the limit, silence is tolerated: jitter is not a fault.
        assert_eq!(l.admit(false, start + PERIOD * 4, limit()), Admission::Live);
        // Past it, the limb is stale.
        assert_eq!(
            l.admit(false, start + PERIOD * 5, limit()),
            Admission::Stale
        );
        assert_eq!(
            l.admit(false, start + PERIOD * 50, limit()),
            Admission::Stale
        );
        // The follower returns: exactly one Reanchor, then ordinary Live.
        let back = start + PERIOD * 51;
        assert_eq!(l.admit(true, back, limit()), Admission::Reanchor);
        assert_eq!(l.admit(true, back + PERIOD, limit()), Admission::Live);
    }

    #[test]
    fn a_one_tick_gap_never_re_anchors() {
        // Re-anchoring mid-motion would discard the governed setpoint and
        // restart from the measured pose, which under normal tracking lag is a
        // step backwards. Only a real gap may trigger it.
        let start = Instant::now();
        let mut l = Liveness::seeded(start);
        assert_eq!(l.admit(false, start + PERIOD, limit()), Admission::Live);
        assert_eq!(l.admit(true, start + PERIOD * 2, limit()), Admission::Live);
    }

    #[test]
    fn repeated_gaps_each_re_anchor() {
        let start = Instant::now();
        let mut l = Liveness::seeded(start);
        let mut now = start;
        for _ in 0..3 {
            now += PERIOD * 5;
            assert_eq!(l.admit(false, now, limit()), Admission::Stale);
            now += PERIOD;
            assert_eq!(l.admit(true, now, limit()), Admission::Reanchor);
        }
    }
}
