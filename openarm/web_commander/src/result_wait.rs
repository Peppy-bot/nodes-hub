//! Shared policy for awaiting a fired goal's result. The backbone owns motion
//! deadlines (each proportional to the move it accepted), so no flat result
//! deadline is imposed here: the commander re-arms a bounded poll until the
//! goal reaches a terminal outcome. A timed-out poll probes the producer's
//! liveness token, so a dead or restarted backbone resolves to `Abandoned`
//! instead of waiting on a result that can no longer arrive.

use std::time::Duration;

use peppygen::Error;

/// One bounded long-poll per re-arm of the result wait. Paces liveness failure
/// detection, not the move: the backbone's own deadline bounds that.
pub const RESULT_POLL: Duration = Duration::from_secs(10);

/// Pause before re-arming after a retryable result error, so a fast-failing
/// transport (an unreachable result service on a still-live producer) is
/// re-polled at a bounded rate instead of spinning.
pub const RESULT_RETRY_DELAY: Duration = Duration::from_millis(250);

/// Whether a result error means "no terminal outcome yet", so the poll should
/// re-arm: it timed out or found the result service unreachable while the
/// producer's liveness token still stands. Anything else is a hard failure.
pub fn result_poll_retryable(error: &Error) -> bool {
    matches!(
        error,
        Error::ActionResultTimeout { .. } | Error::ActionResultUnreachable { .. }
    )
}
