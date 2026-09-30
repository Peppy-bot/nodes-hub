pub mod arm;
pub mod gripper;
pub mod postures;
pub mod stop;

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};

use crate::coordinator::CoordinatorRequest;

/// How long a service waits for the coordinator's answer before it refuses:
/// many ticks of the slowest control rate, so only a loop that has stopped
/// ticking runs it out.
const COORDINATOR_REPLY_TIMEOUT: Duration = Duration::from_secs(5);

/// Claim a side's single-flight move slot, or report it already busy. Shared
/// by the arm, gripper, and posture admission handlers; the matching release
/// rides the move's busy guard.
fn claim(busy: &AtomicBool) -> bool {
    busy.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_ok()
}

/// Hand `request` to the coordinator and wait for the answer it sends on
/// `answer`, from inside a service handler, which the generated code calls
/// synchronously: the wait blocks in place on the runtime, for at most
/// [`COORDINATOR_REPLY_TIMEOUT`]. The refusal names what went wrong: a
/// coordinator that is not running, one that has too many requests queued,
/// or one that did not answer in time.
fn ask_coordinator<T>(
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
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            match tokio::time::timeout(COORDINATOR_REPLY_TIMEOUT, answer).await {
                Ok(Ok(value)) => Ok(value),
                Ok(Err(_)) => Err("the coordinator dropped the request".to_string()),
                Err(_) => Err(format!(
                    "the coordinator did not answer within {COORDINATOR_REPLY_TIMEOUT:?}"
                )),
            }
        })
    })
}
