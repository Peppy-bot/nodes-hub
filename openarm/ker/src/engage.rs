// Engagement policy: which arms track the KER, and what it takes to get
// there. An arm engages when its trigger is squeezed to the engage opening,
// having first read back above it, so a hand that never let go cannot start
// motion. A gap between usable frames of at least the stale timeout, or a new
// session, disengages both arms and takes that release requirement back.

use std::time::{Duration, Instant};

use openarm_description::Side;
use tracing::info;

use crate::side::{SideFlags, SideValues, index, label};

/// Consecutive frames a trigger must read open before a squeeze can engage
/// its arm. A checksum-passing frame can still carry a wrong angle, so the
/// run has to be unbroken: a squeezed reading starts the count again.
const FRAMES_TO_ARM: u8 = 3;

/// A launcher `engage_trigger_opening` outside [0, 1).
#[derive(Debug, thiserror::Error)]
#[error(
    "engage_trigger_opening is the trigger opening a squeeze must reach: at least 0 and \
     below 1. A released trigger reads 1, so a threshold of 1 never arms and no squeeze \
     engages. Try 0.2, got {0}"
)]
pub struct EngageTriggerOpeningOutOfRange(pub f64);

/// The trigger opening at or below which a squeeze engages its arm; strictly
/// below 1 (fully open).
#[derive(Debug, Clone, Copy)]
pub struct EngageTriggerOpening(f64);

impl EngageTriggerOpening {
    pub fn fraction(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for EngageTriggerOpening {
    type Error = EngageTriggerOpeningOutOfRange;

    fn try_from(fraction: f64) -> Result<Self, Self::Error> {
        (0.0..1.0)
            .contains(&fraction)
            .then_some(Self(fraction))
            .ok_or(EngageTriggerOpeningOutOfRange(fraction))
    }
}

/// Per-arm engagement over the stream of usable frames.
pub struct EngageLatch {
    engage_trigger_opening: EngageTriggerOpening,
    stale_timeout: Duration,
    engaged: SideFlags,
    /// Consecutive frames each trigger has read open for, held at
    /// [`FRAMES_TO_ARM`] once armed and cleared by a squeeze or a stall.
    open_frames: [u8; 2],
    last_frame_at: Option<Instant>,
}

impl EngageLatch {
    pub fn new(engage_trigger_opening: EngageTriggerOpening, stale_timeout: Duration) -> Self {
        Self {
            engage_trigger_opening,
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
        let threshold = self.engage_trigger_opening.fraction();
        for side in [Side::Left, Side::Right] {
            let open_frames = &mut self.open_frames[index(side)];
            if triggers.side(side) > threshold {
                *open_frames = (*open_frames + 1).min(FRAMES_TO_ARM);
                continue;
            }
            if *open_frames >= FRAMES_TO_ARM && !self.engaged.side(side) {
                self.engaged.set(side, true);
                info!("KER {} arm engaged, tracking the leader", label(side));
            }
            // A squeeze ends the run of open frames, armed or not.
            *open_frames = 0;
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
        EngageLatch::new(
            EngageTriggerOpening::try_from(ENGAGE_AT).expect("in range"),
            STALE,
        )
    }

    fn triggers(left: f64, right: f64) -> SideValues {
        SideValues { left, right }
    }

    /// Feed `frames` open frames to both triggers, returning the time of the
    /// next frame to send.
    fn open_for(latch: &mut EngageLatch, from: Instant, frames: u8) -> Instant {
        let mut at = from;
        for _ in 0..frames {
            latch.update(triggers(OPEN, OPEN), at);
            at += FRAME;
        }
        at
    }

    /// Arm both triggers the documented way.
    fn arm_both(latch: &mut EngageLatch, from: Instant) -> Instant {
        open_for(latch, from, FRAMES_TO_ARM)
    }

    #[test]
    fn engage_trigger_opening_accepts_only_fractions_below_fully_open() {
        for fraction in [0.0, ENGAGE_AT, 0.999] {
            assert!(
                EngageTriggerOpening::try_from(fraction).is_ok(),
                "{fraction}"
            );
        }
        for fraction in [-0.01, 1.0, 1.5, f64::NAN, f64::INFINITY] {
            let refused = EngageTriggerOpening::try_from(fraction)
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
    fn open_frames_scattered_between_squeezes_never_arm() {
        let mut latch = latch();
        let mut at = Instant::now();
        // Twice as many open frames as arming takes, none of them in a run.
        for _ in 0..FRAMES_TO_ARM * 2 {
            latch.update(triggers(OPEN, OPEN), at);
            at += FRAME;
            assert_eq!(
                latch.update(triggers(SQUEEZED, SQUEEZED), at),
                SideFlags::NONE,
                "a squeeze between open frames restarts the run"
            );
            at += FRAME;
        }
    }

    #[test]
    fn arming_takes_the_whole_run_of_open_frames() {
        let mut one_short = latch();
        let at = open_for(&mut one_short, Instant::now(), FRAMES_TO_ARM - 1);
        assert_eq!(
            one_short.update(triggers(SQUEEZED, SQUEEZED), at),
            SideFlags::NONE,
            "one frame short of the run must not engage"
        );

        let mut whole_run = latch();
        let at = open_for(&mut whole_run, Instant::now(), FRAMES_TO_ARM);
        assert_eq!(whole_run.update(triggers(SQUEEZED, SQUEEZED), at), BOTH);
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
        let mut at = returned + FRAME * (FRAMES_TO_ARM as u32 + 2);
        for _ in 0..FRAMES_TO_ARM {
            latch.update(triggers(OPEN, SQUEEZED), at);
            at += FRAME;
        }
        assert_eq!(latch.update(triggers(SQUEEZED, SQUEEZED), at), LEFT);
    }
}
