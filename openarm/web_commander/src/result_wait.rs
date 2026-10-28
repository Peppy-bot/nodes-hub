//! Shared policy for awaiting a fired goal's result. The backbone owns motion
//! deadlines (each proportional to the move it accepted), so no flat result
//! deadline is imposed here: the commander re-arms a bounded poll until the
//! goal reaches a terminal outcome. A timed-out poll probes the producer's
//! liveness token, so a dead or restarted backbone resolves to `Abandoned`
//! instead of waiting on a result that can no longer arrive.
//!
//! The limb moves (move_arm, move_arm_joints, move_gripper) share the whole
//! wait: [`wait_for_end`] runs it over a [`GoalHandle`], and [`summarize`]
//! words the outcome. Each move module keeps its own logging, so every line it
//! writes carries that module's target.

use std::future::Future;
use std::time::Duration;

use peppygen::Error;
use peppylib::runtime::CancellationToken;
use tokio_util::sync::CancellationToken as GoalToken;

/// One bounded long-poll per re-arm of the result wait. Paces liveness failure
/// detection, not the move: the backbone's own deadline bounds that.
pub const RESULT_POLL: Duration = Duration::from_secs(10);

/// Pause before re-arming after a retryable result error, so a fast-failing
/// transport (an unreachable result service on a still-live producer) is
/// re-polled at a bounded rate instead of spinning.
pub const RESULT_RETRY_DELAY: Duration = Duration::from_millis(250);

/// Goal-accept round-trip to a pinned producer; answered directly, so this only
/// needs to cover the decider, not a discovery probe.
pub const GOAL_TIMEOUT: Duration = Duration::from_secs(2);

/// After a preempted goal reports done, the backbone needs a beat to release its
/// single-flight gate before it will accept the queued move; the queued goal task
/// waits this long before firing.
const PREEMPT_GRACE: Duration = Duration::from_millis(50);

/// Whether a result error means "no terminal outcome yet", so the poll should
/// re-arm: it timed out or found the result service unreachable while the
/// producer's liveness token still stands. Anything else is a hard failure.
pub fn result_poll_retryable(error: &Error) -> bool {
    matches!(
        error,
        Error::ActionResultTimeout { .. } | Error::ActionResultUnreachable { .. }
    )
}

/// How a limb goal ended, mapped from its action's generated outcome.
pub enum GoalEnd<D> {
    /// The goal succeeded, with the action's result data.
    Completed(D),
    /// The goal failed, as its failure prints.
    Failed(String),
    /// The goal ended cancelled, for this reason.
    Cancelled(String),
    /// The backbone dropped the goal, or is gone.
    Abandoned,
    /// The backbone no longer keeps the goal's result.
    Expired,
}

/// A fired limb goal: its result poll and its best-effort cancel.
pub trait GoalHandle {
    type Data;

    /// Poll for the goal's end, waiting at most `poll`.
    fn end(
        &self,
        poll: Duration,
    ) -> impl Future<Output = peppygen::Result<GoalEnd<Self::Data>>> + Send;

    /// Ask the backbone to cancel the goal.
    fn cancel(&self) -> impl Future<Output = peppygen::Result<()>> + Send;
}

/// Implement [`GoalHandle`] for a limb_motion action's generated handle.
macro_rules! limb_goal_handle {
    ($action:ident) => {
        impl $crate::result_wait::GoalHandle
            for peppygen::consumed_actions::limb_motion::$action::ActionHandle
        {
            type Data = peppygen::consumed_actions::limb_motion::$action::ResultResponseData;

            async fn end(
                &self,
                poll: std::time::Duration,
            ) -> peppygen::Result<$crate::result_wait::GoalEnd<Self::Data>> {
                use peppygen::consumed_actions::limb_motion::$action::ResultOutcome;
                use $crate::result_wait::GoalEnd;
                Ok(match self.get_result(poll).await?.outcome {
                    ResultOutcome::Completed(data) => GoalEnd::Completed(data),
                    ResultOutcome::Failed(failure) => GoalEnd::Failed(failure.to_string()),
                    ResultOutcome::Cancelled { reason, .. } => GoalEnd::Cancelled(reason),
                    ResultOutcome::Abandoned => GoalEnd::Abandoned,
                    ResultOutcome::Expired => GoalEnd::Expired,
                })
            }

            async fn cancel(&self) -> peppygen::Result<()> {
                self.cancel_goal($crate::result_wait::GOAL_TIMEOUT)
                    .await
                    .map(|_| ())
            }
        }
    };
}
pub(crate) use limb_goal_handle;

/// Wait out the preempt grace before firing a queued move. False when the node
/// shuts down first, and the move is dropped.
pub async fn preempt_grace_elapsed(token: &CancellationToken) -> bool {
    tokio::select! {
        _ = token.cancelled() => false,
        _ = tokio::time::sleep(PREEMPT_GRACE) => true,
    }
}

/// Why the wait asked the backbone to cancel the goal.
#[derive(Clone, Copy)]
pub enum CancelTrigger {
    /// The node is shutting down.
    Shutdown,
    /// A newer move preempts this one.
    Preempt,
    /// The result poll failed, so the goal's fate is unknown.
    ResultError,
}

impl CancelTrigger {
    /// The log line for a cancel of this trigger that failed.
    pub fn failure_message(self) -> &'static str {
        match self {
            Self::Shutdown => "shutdown cancel failed",
            Self::Preempt => "preempt cancel failed",
            Self::ResultError => "cancel after result error failed",
        }
    }
}

/// What the wait for a goal's end came to.
pub enum Waited<D> {
    /// The node is shutting down; the goal was cancelled best-effort.
    Shutdown,
    /// The goal ended, or its result poll failed.
    Ended(peppygen::Result<GoalEnd<D>>),
}

/// Await the goal's end, honoring shutdown and, when given, `preempt` (a newer
/// move cancels this goal). The wait re-arms its bounded poll until a terminal
/// outcome; the backbone's own deadline bounds the move itself. Shutdown
/// cancels the goal best-effort so it leaves no unsupervised motion, and so does
/// a result error, which leaves the goal's fate unknown. A cancel that fails is
/// handed to `cancel_failed`.
pub async fn wait_for_end<H: GoalHandle>(
    handle: &H,
    token: &CancellationToken,
    preempt: Option<&GoalToken>,
    cancel_failed: impl Fn(CancelTrigger, &Error),
) -> Waited<H::Data> {
    let cancel_failed = &cancel_failed;
    let cancel = |trigger| async move {
        if let Err(e) = handle.cancel().await {
            cancel_failed(trigger, &e);
        }
    };
    let mut preempted = preempt.is_none();
    let ended = loop {
        let end = handle.end(RESULT_POLL);
        tokio::pin!(end);
        tokio::select! {
            _ = token.cancelled() => {
                cancel(CancelTrigger::Shutdown).await;
                return Waited::Shutdown;
            }
            _ = preempt_signal(preempt), if !preempted => {
                preempted = true;
                cancel(CancelTrigger::Preempt).await;
            }
            ended = &mut end => match ended {
                Err(e) if result_poll_retryable(&e) => {
                    tokio::time::sleep(RESULT_RETRY_DELAY).await;
                }
                ended => break ended,
            }
        }
    };
    if ended.is_err() {
        cancel(CancelTrigger::ResultError).await;
    }
    Waited::Ended(ended)
}

/// Resolves when `preempt` is cancelled; never without one.
async fn preempt_signal(preempt: Option<&GoalToken>) {
    match preempt {
        Some(preempt) => preempt.cancelled().await,
        None => std::future::pending().await,
    }
}

/// Whether the goal succeeded, and the status line naming `action` on the
/// `label` side: every line opens with `{action} ({label})`, `completed`
/// words a success after `: `, and the rest share one wording.
pub fn summarize<D>(
    action: &str,
    label: &str,
    ended: peppygen::Result<GoalEnd<D>>,
    completed: impl FnOnce(D) -> String,
) -> (bool, String) {
    let subject = format!("{action} ({label})");
    match ended {
        Ok(GoalEnd::Completed(data)) => (true, format!("{subject}: {}", completed(data))),
        Ok(GoalEnd::Failed(failure)) => (false, format!("{subject} failed: {failure}")),
        Ok(GoalEnd::Cancelled(reason)) => (false, format!("{subject} cancelled: {reason}")),
        Ok(GoalEnd::Abandoned) => (false, format!("{subject} abandoned by backbone")),
        Ok(GoalEnd::Expired) => (false, format!("{subject} result expired")),
        Err(e) => (false, format!("{subject} result error: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_end_but_a_success_is_worded_the_same_for_each_move() {
        let summary = |ended| {
            summarize("move_arm", "left", Ok(ended), |seconds: f64| {
                format!("success in {seconds:.2}s")
            })
        };
        assert_eq!(
            summary(GoalEnd::Completed(1.5)),
            (true, "move_arm (left): success in 1.50s".to_string())
        );
        assert_eq!(
            summary(GoalEnd::Failed("timeout: 3 cm off".to_string())),
            (
                false,
                "move_arm (left) failed: timeout: 3 cm off".to_string()
            )
        );
        assert_eq!(
            summary(GoalEnd::Cancelled("preempted".to_string())),
            (false, "move_arm (left) cancelled: preempted".to_string())
        );
        assert_eq!(
            summary(GoalEnd::Abandoned),
            (false, "move_arm (left) abandoned by backbone".to_string())
        );
        assert_eq!(
            summary(GoalEnd::Expired),
            (false, "move_arm (left) result expired".to_string())
        );
    }

    #[test]
    fn a_failed_cancel_names_its_trigger() {
        assert_eq!(
            CancelTrigger::Shutdown.failure_message(),
            "shutdown cancel failed"
        );
        assert_eq!(
            CancelTrigger::Preempt.failure_message(),
            "preempt cancel failed"
        );
        assert_eq!(
            CancelTrigger::ResultError.failure_message(),
            "cancel after result error failed"
        );
    }
}
