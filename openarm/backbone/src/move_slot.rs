//! A limb's single-flight move slot: an action handler claims it when it
//! admits a goal on the limb, so a limb runs one move at a time and a second
//! goal on it is refused while the first runs. The handler releases the slot
//! when the goal never reaches the coordinator, and the coordinator releases
//! it with a [`SlotRelease`] for a goal it ends before the goal's move starts
//! (before the seed, on a stop, on a stale follower). The move that runs a
//! goal (an arm's mode in the planner, a `GripperMove` the coordinator holds)
//! holds a [`SlotRelease`], which frees the slot on every end of the move.
//!
//! A stop that finds a slot claimed and no goal of it, neither in flight nor
//! queued, came between the admission of the goal and the tick that takes it.
//! The stop marks the slot ([`MoveSlot::stop_unstarted`]), and the coordinator
//! ends the goal cancelled for the stop's reason when it takes the goal
//! ([`MoveSlot::take_stop`]). Thus a stop also ends a goal that its action
//! holds back: the move of the arms of a move_to_home goal, which waits for
//! the close of the grippers.

use std::sync::{Arc, Mutex};

/// The refusal of a goal, or a plan check, that reaches an arm executing a
/// move: the one message for both, whichever gate answers it.
pub const ARM_BUSY: &str = "arm is already executing a motion";

/// The refusal of a gripper goal that reaches a gripper executing a move.
pub const GRIPPER_BUSY: &str = "gripper is already executing a move";

/// Who holds a move slot.
#[derive(Debug, Default, PartialEq)]
enum Holder {
    /// No goal holds the slot.
    #[default]
    Free,
    /// An admitted goal holds the slot.
    Goal,
    /// An admitted goal holds the slot, and a stop came, for this reason,
    /// before the coordinator took the goal.
    StoppedGoal(String),
}

/// One limb's move slot, shared by the handlers that claim it and the loop
/// that runs the claimed goal.
#[derive(Clone, Default)]
pub struct MoveSlot(Arc<Mutex<Holder>>);

impl MoveSlot {
    /// Run `change` on who holds the slot.
    fn with_holder<T>(&self, change: impl FnOnce(&mut Holder) -> T) -> T {
        change(&mut self.0.lock().expect("slot lock poisoned"))
    }

    /// Claim the slot for an admitted goal: `false` while another goal holds
    /// it.
    #[must_use]
    pub fn claim(&self) -> bool {
        self.with_holder(|holder| {
            if *holder != Holder::Free {
                return false;
            }
            *holder = Holder::Goal;
            true
        })
    }

    /// Whether a goal holds the slot.
    pub fn is_claimed(&self) -> bool {
        self.with_holder(|holder| *holder != Holder::Free)
    }

    /// Free the slot. The mark of a stop goes with the goal that held it.
    pub fn release(&self) {
        self.with_holder(|holder| *holder = Holder::Free);
    }

    /// The release of this slot that the goal holding it carries.
    pub fn release_on_drop(&self) -> SlotRelease {
        SlotRelease(self.clone())
    }

    /// Mark the goal that holds the slot as stopped for `reason`, which names
    /// the stop: for a stop that ended no goal of this slot, so the goal is
    /// admitted and the coordinator has not taken it. A free slot gets no
    /// mark, and a goal that an earlier stop marked keeps that stop's reason.
    pub fn stop_unstarted(&self, reason: &str) {
        self.with_holder(|holder| {
            if *holder == Holder::Goal {
                *holder = Holder::StoppedGoal(reason.to_string());
            }
        });
    }

    /// The reason of the stop that came before the coordinator took the goal
    /// that holds the slot, or `None` when no stop came. The mark is taken:
    /// the goal still holds the slot, for its caller to end it and release.
    #[must_use]
    pub fn take_stop(&self) -> Option<String> {
        self.with_holder(|holder| match std::mem::take(holder) {
            Holder::StoppedGoal(reason) => {
                *holder = Holder::Goal;
                Some(reason)
            }
            unmarked => {
                *holder = unmarked;
                None
            }
        })
    }
}

/// Frees its move slot on drop. A move holds one for its lifetime, so no end
/// of the move (success, failure, cancel, or an unreachable plan) leaves the
/// slot claimed.
#[must_use = "dropping a SlotRelease frees its move slot"]
pub struct SlotRelease(MoveSlot);

impl Drop for SlotRelease {
    fn drop(&mut self) {
        self.0.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slot_is_claimed_once_until_it_is_released() {
        let slot = MoveSlot::default();
        assert!(slot.claim(), "a free slot is claimed");
        assert!(slot.is_claimed());
        assert!(!slot.claim(), "a claimed slot refuses a second claim");
        slot.release();
        assert!(!slot.is_claimed());
        assert!(slot.claim(), "a released slot is claimed again");
    }

    #[test]
    fn a_slot_release_frees_its_slot_on_drop() {
        let slot = MoveSlot::default();
        assert!(slot.claim());
        {
            let _release = slot.release_on_drop();
            assert!(slot.is_claimed());
        }
        assert!(
            !slot.is_claimed(),
            "guard must free the slot on drop, so no move terminal can leak it"
        );
    }

    #[test]
    fn a_stop_marks_the_goal_that_holds_the_slot_and_the_mark_is_taken_once() {
        let slot = MoveSlot::default();
        assert!(slot.claim());
        assert_eq!(slot.take_stop(), None, "no stop came");
        slot.stop_unstarted("stopped: the operator asked");
        assert!(slot.is_claimed(), "the stopped goal still holds the slot");
        assert!(!slot.claim(), "a stopped goal refuses a second claim");
        assert_eq!(
            slot.take_stop(),
            Some("stopped: the operator asked".to_string())
        );
        assert_eq!(slot.take_stop(), None, "the mark is taken once");
        assert!(
            slot.is_claimed(),
            "the goal holds the slot until its release"
        );
    }

    #[test]
    fn a_stop_marks_no_free_slot_and_keeps_the_reason_of_the_first_stop() {
        let slot = MoveSlot::default();
        slot.stop_unstarted("stopped: first");
        assert!(!slot.is_claimed(), "a stop claims nothing");
        assert!(slot.claim(), "a free slot is claimed after a stop");
        assert_eq!(slot.take_stop(), None, "the stop came before the goal");

        slot.stop_unstarted("stopped: first");
        slot.stop_unstarted("stopped: second");
        assert_eq!(slot.take_stop(), Some("stopped: first".to_string()));
    }

    #[test]
    fn the_release_of_a_stopped_goal_leaves_no_mark_for_the_next_goal() {
        let slot = MoveSlot::default();
        assert!(slot.claim());
        slot.stop_unstarted("stopped");
        slot.release();
        assert!(!slot.is_claimed());
        assert!(slot.claim(), "the next goal claims the slot");
        assert_eq!(slot.take_stop(), None, "the next goal is not stopped");
    }
}
