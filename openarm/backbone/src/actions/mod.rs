//! The actions and services a caller reaches the limbs through. Each move
//! goal is admitted here (validated, then its limb claimed) and handed to the
//! planner or the coordinator that runs it; a posture goal also waits for the
//! shares of the limbs it moves and ends with their one result, and an
//! execute_joint_path goal for the share of each of its arms
//! ([`collect_ends`]); the stop and the plan check ask the coordinator and
//! answer what it says. The times and the tolerances every move goal names,
//! and the contact gap every arm goal names, are parsed here, and each
//! action's own fields in its module.

pub mod arm;
pub mod gripper;
pub mod path;
pub mod postures;
pub mod stop;

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use control_core::asking::ask_loop;
use control_core::move_seconds::MoveSeconds;
use control_core::move_slot::MoveSlot;
use control_core::positive_finite::PositiveFinite;
use control_core::sync::wait_for;
use robot_description::{ArmId, PerArm};
use tokio::sync::{mpsc, oneshot};

use crate::coordinator::CoordinatorRequest;
use crate::failure::{ARM_BUSY, Failure, FailureKind, GRIPPER_BUSY};
use crate::robot::{ARMS, LIMBS};
use crate::settle::ContactGap;
use crate::streams::LeaderActivity;

/// How long the plan check and the grasp measurement of a posture wait for
/// the coordinator's answer before they give an error: many ticks of the
/// slowest control rate, so only a loop that has stopped ticking runs it out.
const COORDINATOR_REPLY_TIMEOUT: Duration = Duration::from_secs(5);

/// The longest a move may be asked to take, and the longest it may be given
/// to settle after its planned end: the bound the limb_motion contract states
/// for every duration and timeout. A plan walks the control ticks of the
/// request, so the bound also bounds its work.
pub(crate) const MAX_MOVE_DURATION_S: f64 = 60.0;

/// Parse a tolerance off the wire: a positive finite number, or the refusal
/// naming `field` and the value.
fn tolerance_from_wire(field: &str, value: f64) -> Result<f64, String> {
    PositiveFinite::parse_field(field, value).map(PositiveFinite::get)
}

/// Parse a number of seconds a move may be asked for off the wire, `field`
/// being `duration_s` or `settle_timeout_s`: from 0 to
/// [`MAX_MOVE_DURATION_S`], or the refusal naming the field and the value.
fn move_seconds_from_wire(field: &str, value: f64) -> Result<Duration, String> {
    MoveSeconds::from_wire(field, value, MAX_MOVE_DURATION_S).map(MoveSeconds::duration)
}

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
/// `move_slots` being the slots of the arms or of the grippers, or give the
/// first arm whose slot another move holds. A refusal releases the slots it
/// claimed before it, so it never leaves a slot held.
fn claim_slots(arms: &[ArmId], move_slots: &PerArm<MoveSlot, ARMS>) -> Result<(), ArmId> {
    for (k, &arm) in arms.iter().enumerate() {
        if move_slots[arm].claim() {
            continue;
        }
        for &claimed in &arms[..k] {
            move_slots[claimed].release();
        }
        return Err(arm);
    }
    Ok(())
}

/// Admit a move on one limb: [`check_limb_free`], then claim the limb's
/// move slot, or `precondition_failed` while the limb executes another move.
/// The move that runs the goal releases the slot.
fn admit_limb(
    fresh: &AtomicBool,
    leader: &LeaderActivity,
    move_slot: &MoveSlot,
    words: &LimbWords,
) -> Result<(), Failure> {
    check_limb_free(fresh.load(Ordering::Acquire), leader.streaming(), words)?;
    if !move_slot.claim() {
        return Err(Failure::new(FailureKind::PreconditionFailed, words.busy));
    }
    Ok(())
}

/// Hand `request` to the coordinator and wait for the answer it sends on
/// `answer`, for at most `within` ([`ask_loop`]). The error names what went
/// wrong: the coordinator is not running, it dropped the request, or it did
/// not take and answer it in time.
async fn ask_coordinator<T>(
    requests: &mpsc::Sender<CoordinatorRequest>,
    request: CoordinatorRequest,
    answer: oneshot::Receiver<T>,
    within: Duration,
) -> std::result::Result<T, String> {
    ask_loop("coordinator", requests, request, answer, within).await
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

/// [`ask_coordinator`] from inside a service handler, which the generated
/// code calls synchronously: the wait blocks in place on the runtime.
fn blocking_ask_coordinator<T>(
    requests: &mpsc::Sender<CoordinatorRequest>,
    request: CoordinatorRequest,
    answer: oneshot::Receiver<T>,
    within: Duration,
) -> std::result::Result<T, String> {
    wait_for(ask_coordinator(requests, request, answer, within))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// A free limb is claimed; a second move on it is refused busy, and a
    /// refused move claims nothing.
    #[test]
    fn admitting_a_move_claims_the_limb_once() {
        let fresh = AtomicBool::new(true);
        let idle = LeaderActivity::default();
        let move_slot = MoveSlot::default();
        assert_eq!(admit_limb(&fresh, &idle, &move_slot, &ARM_WORDS), Ok(()));
        assert!(move_slot.is_claimed(), "the move holds the limb");
        assert_eq!(
            admit_limb(&fresh, &idle, &move_slot, &ARM_WORDS),
            Err(Failure::new(FailureKind::PreconditionFailed, ARM_BUSY))
        );

        let stale = AtomicBool::new(false);
        let free = MoveSlot::default();
        assert_eq!(
            admit_limb(&stale, &idle, &free, &ARM_WORDS).map_err(|f| f.kind),
            Err(FailureKind::ResourceUnavailable)
        );
        assert!(!free.is_claimed(), "a refusal claims nothing");
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
