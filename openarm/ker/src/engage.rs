// Engagement policy: which arms track the KER, and what it takes to get
// there. An arm engages when its trigger is squeezed to the engage opening,
// having first read back above it, so a hand that never let go cannot start
// motion. A gap between usable frames of at least the stale timeout, or a new
// session, disengages both arms and takes that release requirement back.

use std::time::{Duration, Instant};

use openarm_description::Side;
use tracing::info;

use crate::side::{SideFlags, SideValues, label};

/// Consecutive frames a trigger must read open before a squeeze can engage
/// its arm. A checksum-passing frame can still carry a wrong angle, and one
/// spurious open reading beside a held trigger would otherwise arm the latch.
const FRAMES_TO_ARM: u8 = 3;

/// A launcher `engage_trigger_opening` outside [0, 1).
#[derive(Debug, thiserror::Error)]
#[error(
    "engage_trigger_opening is the trigger opening a squeeze must reach: at least 0 and \
     below 1. A released trigger reads 1, so a threshold of 1 never arms and no squeeze \
     engages. Try 0.2, got {0}"
)]
pub struct EngageOpeningOutOfRange(pub f64);

/// The trigger opening at or below which a squeeze engages its arm; strictly
/// below 1 (fully open).
#[derive(Debug, Clone, Copy)]
pub struct EngageOpening(f64);

impl EngageOpening {
    pub fn fraction(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for EngageOpening {
    type Error = EngageOpeningOutOfRange;

    fn try_from(fraction: f64) -> Result<Self, Self::Error> {
        (0.0..1.0)
            .contains(&fraction)
            .then_some(Self(fraction))
            .ok_or(EngageOpeningOutOfRange(fraction))
    }
}

/// Per-arm engagement over the stream of usable frames.
pub struct EngageLatch {
    engage_opening: EngageOpening,
    stale_timeout: Duration,
    engaged: SideFlags,
    /// Consecutive frames each trigger has read open for, counted up to
    /// [`FRAMES_TO_ARM`] and reset by a stall.
    open_frames: [u8; 2],
    last_frame_at: Option<Instant>,
}

impl EngageLatch {
    pub fn new(engage_opening: EngageOpening, stale_timeout: Duration) -> Self {
        Self {
            engage_opening,
            stale_timeout,
            engaged: SideFlags::NONE,
            open_frames: [0; 2],
            last_frame_at: None,
        }
    }

    /// Fold in one usable frame's trigger openings, captured at `at`.
    pub fn update(&mut self, triggers: SideValues, at: Instant) -> SideFlags {
        if self.stalled(at) {
            if self.engaged != SideFlags::NONE {
                info!("KER frames stalled; both arms disengaged, release and squeeze to re-engage");
            }
            self.engaged = SideFlags::NONE;
            self.open_frames = [0; 2];
        }
        let threshold = self.engage_opening.fraction();
        for (index, side) in [Side::Left, Side::Right].into_iter().enumerate() {
            if triggers.side(side) > threshold {
                self.open_frames[index] = self.open_frames[index].saturating_add(1);
            } else if self.open_frames[index] >= FRAMES_TO_ARM && !self.engaged.side(side) {
                self.engaged.set(side, true);
                info!("KER {} arm engaged, tracking the leader", label(side));
            }
        }
        self.last_frame_at = Some(at);
        self.engaged
    }

    fn stalled(&self, at: Instant) -> bool {
        self.last_frame_at
            .is_some_and(|last| at.duration_since(last) >= self.stale_timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STALE: Duration = Duration::from_millis(250);
    const FRAME: Duration = Duration::from_millis(5);
    const ENGAGE_AT: f64 = 0.2;
    const OPEN: f64 = 1.0;
    const SQUEEZED: f64 = 0.02;
    const LEFT: SideFlags = SideFlags {
        left: true,
        right: false,
    };
    const BOTH: SideFlags = SideFlags {
        left: true,
        right: true,
    };

    fn latch() -> EngageLatch {
        EngageLatch::new(EngageOpening::try_from(ENGAGE_AT).expect("in range"), STALE)
    }

    fn triggers(left: f64, right: f64) -> SideValues {
        SideValues { left, right }
    }

    /// Arm both triggers the documented way, returning the time of the last
    /// frame fed in.
    fn arm_both(latch: &mut EngageLatch, from: Instant) -> Instant {
        let mut at = from;
        for _ in 0..FRAMES_TO_ARM {
            latch.update(triggers(OPEN, OPEN), at);
            at += FRAME;
        }
        at
    }

    #[test]
    fn engage_trigger_opening_accepts_only_fractions_below_fully_open() {
        for fraction in [0.0, ENGAGE_AT, 0.999] {
            assert!(EngageOpening::try_from(fraction).is_ok(), "{fraction}");
        }
        for fraction in [-0.01, 1.0, 1.5, f64::NAN, f64::INFINITY] {
            let refused = EngageOpening::try_from(fraction)
                .expect_err("out of range")
                .to_string();
            assert!(refused.contains("engage_trigger_opening"), "{refused}");
        }
    }

    #[test]
    fn a_squeeze_engages_only_its_own_arm_and_release_keeps_it_tracking() {
        let mut latch = latch();
        let at = arm_both(&mut latch, Instant::now());
        assert_eq!(
            latch.update(triggers(OPEN, SQUEEZED), at),
            SideFlags {
                left: false,
                right: true
            }
        );
        assert_eq!(
            latch.update(triggers(OPEN, OPEN), at + FRAME),
            SideFlags {
                left: false,
                right: true
            },
            "releasing keeps the arm tracking"
        );
        assert_eq!(latch.update(triggers(SQUEEZED, OPEN), at + FRAME * 2), BOTH);
    }

    #[test]
    fn the_engage_trigger_opening_itself_engages() {
        let mut latch = latch();
        let at = arm_both(&mut latch, Instant::now());
        assert_eq!(latch.update(triggers(ENGAGE_AT, OPEN), at), LEFT);
    }

    #[test]
    fn a_trigger_held_from_the_first_frame_engages_nothing() {
        let mut latch = latch();
        let mut at = Instant::now();
        for tick in 0..8 {
            assert_eq!(
                latch.update(triggers(SQUEEZED, SQUEEZED), at),
                SideFlags::NONE,
                "tick {tick}: a hand already on the trigger must not engage"
            );
            at += FRAME;
        }
        let at = arm_both(&mut latch, at);
        assert_eq!(latch.update(triggers(SQUEEZED, SQUEEZED), at), BOTH);
    }

    #[test]
    fn one_open_frame_does_not_arm_a_held_trigger() {
        let mut latch = latch();
        let mut at = Instant::now();
        // A single frame reading open, as a corrupt angle would, beside a
        // trigger the operator never released.
        latch.update(triggers(SQUEEZED, SQUEEZED), at);
        at += FRAME;
        latch.update(triggers(OPEN, OPEN), at);
        at += FRAME;
        assert_eq!(
            latch.update(triggers(SQUEEZED, SQUEEZED), at),
            SideFlags::NONE
        );
    }

    #[test]
    fn a_frame_gap_of_the_stale_timeout_disengages_both_arms() {
        let mut latch = latch();
        let at = arm_both(&mut latch, Instant::now());
        latch.update(triggers(SQUEEZED, SQUEEZED), at);
        let just_inside = at + STALE - FRAME;
        assert_eq!(
            latch.update(triggers(SQUEEZED, SQUEEZED), just_inside),
            BOTH
        );
        assert_eq!(
            latch.update(triggers(SQUEEZED, SQUEEZED), just_inside + STALE),
            SideFlags::NONE
        );
    }

    #[test]
    fn a_held_trigger_after_a_stall_stays_disengaged_until_released() {
        let mut latch = latch();
        let at = arm_both(&mut latch, Instant::now());
        latch.update(triggers(SQUEEZED, SQUEEZED), at);
        let returned = at + STALE;
        for tick in 0..FRAMES_TO_ARM + 1 {
            assert_eq!(
                latch.update(triggers(SQUEEZED, SQUEEZED), returned + FRAME * tick as u32),
                SideFlags::NONE,
                "a trigger held across the stall must not re-engage"
            );
        }
        // Only the left trigger is released, so only the left arm can engage.
        let mut at = returned + FRAME * 8;
        for _ in 0..FRAMES_TO_ARM {
            latch.update(triggers(OPEN, SQUEEZED), at);
            at += FRAME;
        }
        assert_eq!(latch.update(triggers(SQUEEZED, SQUEEZED), at), LEFT);
    }
}
