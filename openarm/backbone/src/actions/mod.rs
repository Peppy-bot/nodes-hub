//! The actions and services a caller reaches the arms through. Each move
//! goal is admitted here (validated, then its limb claimed) and handed to the
//! planner or the coordinator that runs it; a posture goal also waits for the
//! shares of both arms and ends with their one result; the stop and the plan
//! check ask the coordinator and answer what it says. The times and the
//! tolerances every move goal names are parsed here, and each action's own
//! fields in its module.

pub mod arm;
pub mod gripper;
pub mod postures;
pub mod stop;

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use control_core::positive_finite::PositiveFinite;
use control_core::sync::wait_for;
use tokio::sync::{mpsc, oneshot};

use crate::coordinator::CoordinatorRequest;
use crate::failure::{Failure, FailureKind};
use crate::move_slot::{ARM_BUSY, GRIPPER_BUSY, MoveSlot};
use crate::streams::LeaderActivity;

/// How long a service waits for the coordinator's answer before it gives an
/// error reply: many ticks of the slowest control rate, so only a loop that
/// has stopped ticking runs it out.
const COORDINATOR_REPLY_TIMEOUT: Duration = Duration::from_secs(5);

/// The longest a move may be asked to take, and the longest it may be given
/// to settle after its planned end: the bound the limb_motion contract states
/// for every duration and timeout. A plan walks the control ticks of the
/// request, so the bound also bounds its work.
const MAX_MOVE_DURATION_S: f64 = 60.0;

/// Parse a tolerance off the wire: a positive finite number, or the refusal
/// naming `field`.
fn tolerance_from_wire(field: &str, value: f64) -> Result<f64, String> {
    PositiveFinite::parse(value)
        .map(PositiveFinite::get)
        .map_err(|_| format!("{field} must be a positive number"))
}

/// Parse a number of seconds a move may be asked for off the wire, `field`
/// being `duration_s` or `settle_timeout_s`: from 0 to
/// [`MAX_MOVE_DURATION_S`], or the refusal naming the field.
fn move_seconds_from_wire(field: &str, value: f64) -> Result<Duration, String> {
    if value.is_finite() && (0.0..=MAX_MOVE_DURATION_S).contains(&value) {
        return Ok(Duration::from_secs_f64(value));
    }
    Err(format!(
        "{field} must be a number of seconds from 0 to {MAX_MOVE_DURATION_S:.0}"
    ))
}

/// The words of the refusals of a move on one kind of limb.
struct LimbWords {
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
    no_fresh_state: "no fresh follower state to anchor the move from",
    leader_streaming: "a leader is streaming arm commands; release the headset's grip, switch off the browser panel's stream, or stop the leader node, then retry",
    busy: ARM_BUSY,
};

/// The refusals of a gripper goal.
const GRIPPER_WORDS: LimbWords = LimbWords {
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
/// `answer`, for at most [`COORDINATOR_REPLY_TIMEOUT`]. The error names
/// what went wrong:
/// - the coordinator is not running;
/// - it has too many requests queued;
/// - it dropped the request;
/// - it did not answer in time.
async fn ask_coordinator<T>(
    requests: &mpsc::Sender<CoordinatorRequest>,
    request: CoordinatorRequest,
    answer: oneshot::Receiver<T>,
) -> std::result::Result<T, String> {
    requests.try_send(request).map_err(|e| match e {
        mpsc::error::TrySendError::Full(_) => {
            "the coordinator has too many requests queued".to_string()
        }
        mpsc::error::TrySendError::Closed(_) => "the coordinator is not running".to_string(),
    })?;
    match tokio::time::timeout(COORDINATOR_REPLY_TIMEOUT, answer).await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(_)) => Err("the coordinator dropped the request".to_string()),
        Err(_) => Err(format!(
            "the coordinator did not answer within {COORDINATOR_REPLY_TIMEOUT:?}"
        )),
    }
}

/// [`ask_coordinator`] from inside a service handler, which the generated
/// code calls synchronously: the wait blocks in place on the runtime.
fn blocking_ask_coordinator<T>(
    requests: &mpsc::Sender<CoordinatorRequest>,
    request: CoordinatorRequest,
    answer: oneshot::Receiver<T>,
) -> std::result::Result<T, String> {
    wait_for(ask_coordinator(requests, request, answer))
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

    // --- Tolerances and times off the wire ---------------------------------

    use hegel::TestCase;
    use hegel::generators as gs;

    fn admitted_tolerance(value: f64) -> bool {
        value.is_finite() && value > 0.0
    }

    // A tolerance off the wire is exactly a positive finite number, kept as it
    // came; anything else is refused naming the field.
    #[hegel::test]
    fn a_tolerance_is_accepted_exactly_when_positive_and_finite(tc: TestCase) {
        let value: f64 = tc.draw(gs::floats());
        match tolerance_from_wire("settle_tolerance", value) {
            Ok(parsed) => {
                assert!(admitted_tolerance(value), "{value} was accepted");
                assert_eq!(parsed, value);
            }
            Err(refusal) => {
                assert!(!admitted_tolerance(value), "{value} was refused");
                assert_eq!(refusal, "settle_tolerance must be a positive number");
            }
        }
    }

    #[test]
    fn non_finite_zero_and_negative_tolerances_are_refused() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1e-9, -1.0] {
            assert!(
                tolerance_from_wire("settle_tolerance", bad).is_err(),
                "tolerance {bad} must be refused"
            );
        }
    }

    // A number of seconds off the wire is exactly a finite value from 0 to the
    // move ceiling, carried as a duration.
    #[hegel::test]
    fn move_seconds_are_accepted_exactly_within_the_ceiling(tc: TestCase) {
        let value: f64 = tc.draw(gs::floats());
        let admitted = value.is_finite() && (0.0..=MAX_MOVE_DURATION_S).contains(&value);
        match move_seconds_from_wire("settle_timeout_s", value) {
            Ok(duration) => {
                assert!(admitted, "{value} s was accepted");
                assert_eq!(duration, Duration::from_secs_f64(value));
            }
            Err(refusal) => {
                assert!(!admitted, "{value} s was refused");
                assert_eq!(
                    refusal,
                    "settle_timeout_s must be a number of seconds from 0 to 60"
                );
            }
        }
    }
}
