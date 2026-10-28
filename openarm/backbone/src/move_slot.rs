//! A limb's single-flight move slot: an action handler claims it when it
//! admits a goal on the limb, so a limb runs one move at a time and a second
//! goal on it is refused while the first runs. The handler releases the slot
//! when the goal never reaches the coordinator, and the coordinator releases
//! it with a [`SlotRelease`] for a goal it ends before the goal's move starts
//! (before the seed, on a stop, on a stale follower). The move that runs a
//! goal (an arm's mode in the planner, a `GripperMove` the coordinator holds)
//! holds a [`SlotRelease`], which frees the slot on every end of the move.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The refusal of a goal, or a plan check, that reaches an arm executing a
/// move: the one message for both, whichever gate answers it.
pub const ARM_BUSY: &str = "arm is already executing a motion";

/// The refusal of a gripper goal that reaches a gripper executing a move.
pub const GRIPPER_BUSY: &str = "gripper is already executing a move";

/// One limb's move slot, shared by the handlers that claim it and the loop
/// that runs the claimed goal.
#[derive(Clone, Default)]
pub struct MoveSlot(Arc<AtomicBool>);

impl MoveSlot {
    /// Claim the slot for an admitted goal: `false` while another goal holds
    /// it.
    #[must_use]
    pub fn claim(&self) -> bool {
        self.0
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    /// Whether a goal holds the slot.
    pub fn is_claimed(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    /// Free the slot.
    pub fn release(&self) {
        self.0.store(false, Ordering::Release);
    }

    /// The release of this slot that the goal holding it carries.
    pub fn release_on_drop(&self) -> SlotRelease {
        SlotRelease(self.clone())
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
}
