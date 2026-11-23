//! The actions and services a caller reaches the limbs through. Each move
//! goal is admitted here (validated, then its limb claimed) and handed to the
//! planner or the coordinator that runs it; a posture goal also waits for the
//! shares of the limbs it moves and ends with their one result, and an
//! execute_joint_path goal for the share of each of its arms
//! ([`collect_ends`]); the stop and the plan check ask the coordinator and
//! answer what it says. The contact gap every arm goal names is parsed here,
//! the times and the tolerances every move goal names by control_core's
//! parses ([`MoveSeconds`](control_core::move_seconds::MoveSeconds),
//! [`PositiveFinite::parse_field`]), and each action's own fields in its
//! module.

pub mod arm;
pub mod gripper;
pub mod path;
pub mod postures;
pub mod stop;

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use control_core::move_slot::{MoveSlot, SlotRelease};
use control_core::positive_finite::PositiveFinite;
use robot_description::{ArmId, PerArm};
use tokio::sync::mpsc;

use crate::failure::{ARM_BUSY, Failure, FailureKind, GRIPPER_BUSY};
use crate::robot::{ARMS, LIMBS};
use crate::settle::ContactGap;
use crate::streams::LeaderActivity;

/// How long the plan check and the grasp measurement of a posture wait for
/// the coordinator's answer before they give an error: many ticks of the
/// slowest control rate, so only a loop that has stopped ticking runs it out.
const COORDINATOR_REPLY_TIMEOUT: Duration = Duration::from_secs(5);

/// The name an ask of the coordinator gives it in its errors
/// ([`ask_loop`](control_core::asking::ask_loop)).
const COORDINATOR: &str = "coordinator";

/// Parse the `contact_gap_m` of an arm goal off the wire: 0 for no contact
/// check, a positive finite number of meters for a gap, or the refusal.
fn contact_gap_from_wire(value: f64) -> Result<ContactGap, String> {
    if value == 0.0 {
        return Ok(ContactGap::NONE);
    }
    PositiveFinite::parse(value)
        .map(ContactGap::of)
        .map_err(|_| "contact_gap_m must be 0 or a positive number of meters".to_string())
}

/// The words of the refusals of a move on one kind of limb.
struct LimbWords {
    /// The name of the limb of this kind on an arm.
    limb: fn(ArmId) -> &'static str,
    /// The follower of the limb delivers no fresh state: before its first
    /// state, or after it stopped.
    no_fresh_state: &'static str,
    /// A leader streams the limb: the leader has it until its stream stops.
    leader_streaming: &'static str,
    /// The limb executes another move.
    busy: &'static str,
}

/// The refusals of an arm goal, and of a plan check.
const ARM_WORDS: LimbWords = LimbWords {
    limb: |arm| LIMBS.arm_name(arm),
    no_fresh_state: "no fresh follower state to anchor the move from",
    leader_streaming: "a leader is streaming arm commands; release the headset's grip, switch off the browser panel's stream, or stop the leader node, then retry",
    busy: ARM_BUSY,
};

/// The refusals of a gripper goal.
const GRIPPER_WORDS: LimbWords = LimbWords {
    limb: |arm| LIMBS.gripper_name(arm),
    no_fresh_state: "no fresh gripper state to anchor the move from",
    leader_streaming: "a leader is streaming gripper commands; release the headset's grip, switch off the browser panel's stream, or stop the leader node, then retry",
    busy: GRIPPER_BUSY,
};

/// Whether a limb can take a move now, by its follower's freshness and its
/// leader's stream:
/// - its follower delivers no fresh state: `resource_unavailable`;
/// - a leader streams it: `leader_streaming`.
fn check_limb_free(fresh: bool, streaming: bool, words: &LimbWords) -> Result<(), Failure> {
    if !fresh {
        return Err(Failure::new(
            FailureKind::ResourceUnavailable,
            words.no_fresh_state,
        ));
    }
    if streaming {
        return Err(Failure::new(
            FailureKind::LeaderStreaming,
            words.leader_streaming,
        ));
    }
    Ok(())
}

/// Claim the move slot of the limb on each arm of `arms`, in their order,
/// `move_slots` being the slots of the arms or of the grippers: the release
/// of each claim in that order, or the first arm whose slot another move
/// holds. A refusal drops the releases of the claims before it, so it never
/// leaves a slot held.
fn claim_slots(
    arms: &[ArmId],
    move_slots: &PerArm<MoveSlot, ARMS>,
) -> Result<Vec<SlotRelease>, ArmId> {
    arms.iter()
        .map(|&arm| move_slots[arm].claim().ok_or(arm))
        .collect()
}

/// Admit a move on one limb: [`check_limb_free`], then claim the limb's
/// move slot, or `precondition_failed` while the limb executes another move.
/// The goal carries the release of the claim, and the move that runs it then.
fn admit_limb(
    fresh: &AtomicBool,
    leader: &LeaderActivity,
    move_slot: &MoveSlot,
    words: &LimbWords,
) -> Result<SlotRelease, Failure> {
    check_limb_free(fresh.load(Ordering::Acquire), leader.streaming(), words)?;
    move_slot
        .claim()
        .ok_or_else(|| Failure::new(FailureKind::PreconditionFailed, words.busy))
}

/// The ends the shares of a goal report on `ends`, once `pending` have
/// reported or every share's sender is gone. The goal's cancel,
/// `cancel_signal`, raises `cancelled`, which the shares poll.
async fn collect_ends<T>(
    pending: usize,
    ends: &mut mpsc::Receiver<T>,
    cancel_signal: impl Future<Output = ()>,
    cancelled: &AtomicBool,
) -> Vec<T> {
    let mut cancel_signal = std::pin::pin!(cancel_signal);
    let mut cancel_seen = false;
    let mut received = Vec::with_capacity(pending);
    while received.len() < pending {
        tokio::select! {
            () = &mut cancel_signal, if !cancel_seen => {
                cancel_seen = true;
                cancelled.store(true, Ordering::Release);
            }
            end = ends.recv() => match end {
                Some(end) => received.push(end),
                None => break,
            },
        }
    }
    received
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{LEFT, RIGHT};

    /// A limb with no fresh state is unavailable whatever its leader does;
    /// a fresh limb a leader streams is the leader's.
    #[test]
    fn a_limb_is_refused_by_its_state_first_then_by_its_leader() {
        let refused = |kind, message: &str| Err(Failure::new(kind, message));
        assert_eq!(
            check_limb_free(false, true, &ARM_WORDS),
            refused(FailureKind::ResourceUnavailable, ARM_WORDS.no_fresh_state)
        );
        assert_eq!(
            check_limb_free(true, true, &GRIPPER_WORDS),
            refused(FailureKind::LeaderStreaming, GRIPPER_WORDS.leader_streaming)
        );
        assert_eq!(check_limb_free(true, false, &ARM_WORDS), Ok(()));
    }

    /// A free limb is claimed until the release its admission gives drops; a
    /// second move on it is refused busy, and a refused move claims nothing.
    #[test]
    fn admitting_a_move_claims_the_limb_once() {
        let fresh = AtomicBool::new(true);
        let idle = LeaderActivity::default();
        let move_slot = MoveSlot::default();
        let release = admit_limb(&fresh, &idle, &move_slot, &ARM_WORDS).expect("a free limb");
        assert!(move_slot.is_claimed(), "the move holds the limb");
        assert_eq!(
            admit_limb(&fresh, &idle, &move_slot, &ARM_WORDS).err(),
            Some(Failure::new(FailureKind::PreconditionFailed, ARM_BUSY))
        );
        drop(release);
        assert!(!move_slot.is_claimed(), "the release frees the limb");

        let stale = AtomicBool::new(false);
        let free = MoveSlot::default();
        assert_eq!(
            admit_limb(&stale, &idle, &free, &ARM_WORDS)
                .err()
                .map(|f| f.kind),
            Some(FailureKind::ResourceUnavailable)
        );
        assert!(!free.is_claimed(), "a refusal claims nothing");
    }

    /// The claims of a goal on several limbs come back in their order; a busy
    /// limb refuses the goal, naming it, and frees the limbs claimed before it.
    #[test]
    fn claiming_the_slots_of_a_goal_frees_them_all_on_a_busy_one() {
        let move_slots = PerArm::<MoveSlot, ARMS>::from_fn(|_| MoveSlot::default());
        let both = claim_slots(&[LEFT, RIGHT], &move_slots).expect("free slots");
        assert_eq!(both.len(), 2);
        assert!(move_slots.iter().all(MoveSlot::is_claimed));
        drop(both);
        let _busy = move_slots[RIGHT].claim().expect("a free slot");
        assert_eq!(claim_slots(&[LEFT, RIGHT], &move_slots).err(), Some(RIGHT));
        assert!(
            !move_slots[LEFT].is_claimed(),
            "the refusal frees the left arm"
        );
    }

    use hegel::TestCase;
    use hegel::generators as gs;

    // A contact gap off the wire is exactly 0, for no contact check, or a
    // positive finite number of meters, carried as that gap.
    #[hegel::test]
    fn a_contact_gap_is_accepted_exactly_at_0_or_positive_and_finite(tc: TestCase) {
        let value: f64 = tc.draw(gs::floats());
        match contact_gap_from_wire(value) {
            Ok(gap) => {
                let expected =
                    PositiveFinite::parse(value).map_or(ContactGap::NONE, ContactGap::of);
                assert!(value.is_finite() && value >= 0.0, "{value} m was accepted");
                assert_eq!(gap, expected);
            }
            Err(refusal) => {
                assert!(
                    !(value.is_finite() && value >= 0.0),
                    "{value} m was refused"
                );
                assert_eq!(
                    refusal,
                    "contact_gap_m must be 0 or a positive number of meters"
                );
            }
        }
    }

    #[test]
    fn a_contact_gap_of_0_asks_for_no_contact_check_and_a_negative_one_is_refused() {
        assert_eq!(contact_gap_from_wire(0.0), Ok(ContactGap::NONE));
        assert_eq!(contact_gap_from_wire(-0.0), Ok(ContactGap::NONE));
        for bad in [-1e-9, -0.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                contact_gap_from_wire(bad).is_err(),
                "gap {bad} must be refused"
            );
        }
    }
}
