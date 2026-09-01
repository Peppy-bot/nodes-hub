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

/// How many follower state periods of silence before a limb is declared
/// stale. A healthy follower delivers every period, so this tolerates four
/// consecutive missed deliveries plus scheduler jitter; a stream that quiet is
/// broken rather than jittery.
const STALE_PERIODS: u32 = 4;

/// The longest silence any follower rate may buy. Past this the watchdog no
/// longer bounds the divergence it exists to catch; the relays' motor_health
/// gate sits at the same 500 ms.
pub const MAX_STALE_LIMIT: Duration = Duration::from_millis(500);

/// How long each delivery-cadence sample runs before it is judged.
const CADENCE_WINDOW: Duration = Duration::from_secs(1);

/// The fraction of the declared follower rate a follower must actually
/// deliver at. A quarter of its deliveries missing is a follower running
/// under the rate it was launched with, not jitter.
const MIN_DELIVERY_FRACTION: f64 = 0.75;

/// A `follower_state_rate_hz` whose four-period window the backbone cannot
/// honour.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error(
    "four deliveries at {rate_hz} Hz span {window_s} s, outside \
     [one control period = {min_s} s, {max_s} s]"
)]
pub struct StaleLimitOutOfRange {
    pub rate_hz: u32,
    pub window_s: f64,
    pub min_s: f64,
    pub max_s: f64,
}

/// The silence a limb is allowed before it must not be acted on: four periods
/// of the follower's declared state rate. Deliveries are observed once per
/// control tick, so a window under one period can never be met; the ceiling
/// is [`MAX_STALE_LIMIT`]. The launcher declares the rate because followers
/// report on their own cadence: the real arms deliver every control period,
/// a rendered simulator only once per frame.
pub fn stale_limit(
    follower_state_rate_hz: u32,
    cycle_period: Duration,
) -> Result<Duration, StaleLimitOutOfRange> {
    let window_s = f64::from(STALE_PERIODS) / f64::from(follower_state_rate_hz);
    let out_of_range = || StaleLimitOutOfRange {
        rate_hz: follower_state_rate_hz,
        window_s,
        min_s: cycle_period.as_secs_f64(),
        max_s: MAX_STALE_LIMIT.as_secs_f64(),
    };
    let limit = Duration::try_from_secs_f64(window_s).map_err(|_| out_of_range())?;
    if limit < cycle_period || limit > MAX_STALE_LIMIT {
        return Err(out_of_range());
    }
    Ok(limit)
}

/// A follower's delivery cadence crossed [`MIN_DELIVERY_FRACTION`] of its
/// declared rate, in either direction, over the last [`CADENCE_WINDOW`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CadenceChange {
    Degraded { delivered: u32, expected: u32 },
    Recovered { delivered: u32, expected: u32 },
}

/// Per-limb delivery-rate watch against the launcher's declared follower
/// rate. The stale limit fires only once a single gap has grown to several
/// periods; this reports a follower that delivers too little overall, so a
/// declared rate the follower cannot keep is visible before it freezes a
/// limb.
#[derive(Debug)]
pub struct Cadence {
    /// Deliveries a window should bring: the declared rate over one window.
    declared: u32,
    window_start: Instant,
    ticks: u32,
    delivered: u32,
    degraded: bool,
}

impl Cadence {
    pub fn new(follower_state_rate_hz: u32, now: Instant) -> Self {
        Self {
            declared: follower_state_rate_hz * CADENCE_WINDOW.as_secs() as u32,
            window_start: now,
            ticks: 0,
            delivered: 0,
            degraded: false,
        }
    }

    /// Count this tick's delivery; at the end of each window, report whether
    /// the follower crossed the cadence threshold. A delivery is observed at
    /// most once per tick, so a follower faster than the control loop is
    /// judged against the ticks instead.
    pub fn observe(&mut self, delivered: bool, now: Instant) -> Option<CadenceChange> {
        self.ticks += 1;
        self.delivered += u32::from(delivered);
        if now.saturating_duration_since(self.window_start) < CADENCE_WINDOW {
            return None;
        }
        let (delivered, expected) = (self.delivered, self.declared.min(self.ticks));
        let degraded = f64::from(delivered) < f64::from(expected) * MIN_DELIVERY_FRACTION;
        let change = match (self.degraded, degraded) {
            (false, true) => Some(CadenceChange::Degraded {
                delivered,
                expected,
            }),
            (true, false) => Some(CadenceChange::Recovered {
                delivered,
                expected,
            }),
            _ => None,
        };
        *self = Self {
            degraded,
            ..Self::new(self.declared, now)
        };
        change
    }
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

    /// A follower delivering every control period, as the real arms do.
    fn limit() -> Duration {
        stale_limit(100, PERIOD).expect("four periods is inside the bounds")
    }

    #[test]
    fn a_follower_at_the_control_rate_gets_four_control_periods() {
        assert_eq!(limit(), Duration::from_millis(40));
    }

    #[test]
    fn a_slower_follower_gets_four_of_its_own_periods() {
        assert_eq!(
            stale_limit(60, PERIOD),
            Ok(Duration::from_secs_f64(4.0 / 60.0))
        );
    }

    #[test]
    fn both_bounds_are_inclusive() {
        assert_eq!(stale_limit(400, PERIOD), Ok(PERIOD));
        assert_eq!(stale_limit(8, PERIOD), Ok(MAX_STALE_LIMIT));
    }

    #[test]
    fn rates_outside_the_bounds_are_refused_with_the_bounds_named() {
        for rate_hz in [401, 7, 1, 0, u32::MAX] {
            let refused = stale_limit(rate_hz, PERIOD).expect_err("must be refused");
            assert_eq!(refused.rate_hz, rate_hz);
            assert_eq!(refused.min_s, 0.01);
            assert_eq!(refused.max_s, 0.5);
        }
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

    /// Drive one window of 100 ticks, delivering on the first `delivered`.
    fn window(cadence: &mut Cadence, start: Instant, delivered: u32) -> Option<CadenceChange> {
        const TICKS: u32 = 100;
        (1..=TICKS)
            .filter_map(|tick| cadence.observe(tick <= delivered, start + PERIOD * tick))
            .next()
    }

    #[test]
    fn a_follower_keeping_its_declared_rate_reports_nothing() {
        let start = Instant::now();
        let mut cadence = Cadence::new(60, start);
        assert_eq!(window(&mut cadence, start, 60), None);
        assert_eq!(window(&mut cadence, start + PERIOD * 100, 46), None);
    }

    #[test]
    fn a_follower_faster_than_the_loop_is_judged_against_the_ticks() {
        let start = Instant::now();
        let mut cadence = Cadence::new(1_000, start);
        assert_eq!(window(&mut cadence, start, 100), None);
    }

    #[test]
    fn nothing_is_judged_before_a_window_has_elapsed() {
        let start = Instant::now();
        let mut cadence = Cadence::new(100, start);
        for tick in 1..100 {
            assert_eq!(cadence.observe(false, start + PERIOD * tick), None);
        }
    }

    #[test]
    fn falling_under_the_declared_rate_reports_once_and_recovering_reports_once() {
        let start = Instant::now();
        let mut cadence = Cadence::new(60, start);
        assert_eq!(
            window(&mut cadence, start, 40),
            Some(CadenceChange::Degraded {
                delivered: 40,
                expected: 60
            })
        );
        assert_eq!(window(&mut cadence, start + PERIOD * 100, 30), None);
        assert_eq!(
            window(&mut cadence, start + PERIOD * 200, 60),
            Some(CadenceChange::Recovered {
                delivered: 60,
                expected: 60
            })
        );
        assert_eq!(window(&mut cadence, start + PERIOD * 300, 60), None);
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
