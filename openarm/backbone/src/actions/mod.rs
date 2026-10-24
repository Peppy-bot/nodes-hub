pub mod arm;
pub mod gripper;
pub mod postures;
pub mod stop;

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};

use crate::coordinator::CoordinatorRequest;
use crate::failure::{Failure, FailureKind};
use crate::planner::ARM_BUSY;
use crate::streams::LeaderActivity;

/// How long a service waits for the coordinator's answer before it gives an
/// error reply: many ticks of the slowest control rate, so only a loop that
/// has stopped ticking runs it out.
const COORDINATOR_REPLY_TIMEOUT: Duration = Duration::from_secs(5);

/// The words of the refusals of a move on one kind of limb.
pub struct LimbWords {
    /// The follower of the limb delivers no fresh state: before its first
    /// state, or after it stopped.
    pub no_fresh_state: &'static str,
    /// A leader streams the limb: the leader has it until its stream stops.
    pub leader_streaming: &'static str,
    /// The limb executes another move.
    pub busy: &'static str,
}

/// The refusals of an arm goal, and of a plan check.
pub const ARM_WORDS: LimbWords = LimbWords {
    no_fresh_state: "no fresh follower state to anchor the move from",
    leader_streaming: "a leader is streaming arm commands; release the headset's grip, switch off the browser panel's stream, or stop the leader node, then retry",
    busy: ARM_BUSY,
};

/// The refusals of a gripper goal.
pub const GRIPPER_WORDS: LimbWords = LimbWords {
    no_fresh_state: "no fresh gripper state to anchor the move from",
    leader_streaming: "a leader is streaming gripper commands; release the headset's grip, switch off the browser panel's stream, or stop the leader node, then retry",
    busy: "gripper is already executing a move",
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
/// single-flight slot, or `precondition_failed` while the limb executes
/// another move. The matching release rides the move's busy guard.
fn admit_limb(
    fresh: &AtomicBool,
    leader: &LeaderActivity,
    busy: &AtomicBool,
    words: &LimbWords,
) -> Result<(), Failure> {
    check_limb_free(fresh.load(Ordering::Acquire), leader.streaming(), words)?;
    if !claim(busy) {
        return Err(Failure::new(FailureKind::PreconditionFailed, words.busy));
    }
    Ok(())
}

/// Claim a side's single-flight move slot, or report it already busy. Shared
/// by the limb and posture admissions; the matching release rides the move's
/// busy guard.
fn claim(busy: &AtomicBool) -> bool {
    busy.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_ok()
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
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(ask_coordinator(requests, request, answer))
    })
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
        let busy = AtomicBool::new(false);
        assert_eq!(admit_limb(&fresh, &idle, &busy, &ARM_WORDS), Ok(()));
        assert!(busy.load(Ordering::Acquire), "the move holds the limb");
        assert_eq!(
            admit_limb(&fresh, &idle, &busy, &ARM_WORDS),
            Err(Failure::new(FailureKind::PreconditionFailed, ARM_BUSY))
        );

        let stale = AtomicBool::new(false);
        let free = AtomicBool::new(false);
        assert_eq!(
            admit_limb(&stale, &idle, &free, &ARM_WORDS).map_err(|f| f.kind),
            Err(FailureKind::ResourceUnavailable)
        );
        assert!(!free.load(Ordering::Acquire), "a refusal claims nothing");
    }
}
