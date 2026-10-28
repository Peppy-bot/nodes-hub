//! Spawned per start_recording command (the panel's Record button). Fires the
//! recorder's record_episode goal, drains its frame-count feedback into the
//! owner, and reports the saved episode when the goal ends. The stop token is
//! the Stop button: cancelling the goal ends the episode with a save. The
//! recorder also ends episodes on its own (stale source, disk floor); that
//! surfaces here as a cancelled result whose reason names the cause. An
//! episode the recorder cannot save fails, with its failure kind and message.

use std::sync::Arc;
use std::time::Duration;

use control_core::time::sleep_unless_cancelled;
use peppygen::consumed_actions::recorder::record_episode;
use peppygen::consumed_actions::recorder::record_episode::ResultOutcome;
use peppygen::consumed_services::recorder::finish_session;
use peppygen::{CancelCause, NodeRunner};
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::owner::Feedback;
use crate::result_wait::{GOAL_TIMEOUT, RESULT_POLL, RESULT_RETRY_DELAY, result_poll_retryable};

// finish_session finalizes the dataset and completes the mirror before
// answering, which can carry a large upload on a long session.
const FINISH_TIMEOUT: Duration = Duration::from_secs(300);

/// Whether this deployment binds a recorder instance (the slot is
/// zero_or_more); resolved once at startup to gate the panel.
pub fn available(runner: &NodeRunner) -> bool {
    !record_episode::bound_producers(runner).is_empty()
}

pub fn spawn(
    runner: Arc<NodeRunner>,
    feedback: mpsc::Sender<Feedback>,
    token: CancellationToken,
    stop: tokio_util::sync::CancellationToken,
    task: String,
) {
    tokio::spawn(async move {
        run(runner, feedback, token, stop, task).await;
    });
}

async fn run(
    runner: Arc<NodeRunner>,
    feedback: mpsc::Sender<Feedback>,
    token: CancellationToken,
    stop: tokio_util::sync::CancellationToken,
    task: String,
) {
    info!(task, "fire record_episode");

    // The owner gates spawning on `available`, so the slot has a producer.
    let recorders = record_episode::bound_producers(&runner);
    let Some(target) = recorders.first() else {
        finalize(&feedback, false, "no recorder bound").await;
        return;
    };

    let goal = record_episode::GoalRequest { task };
    let mut downstream = match record_episode::ActionHandle::fire_goal(
        &runner,
        target,
        GOAL_TIMEOUT,
        goal,
        peppygen::DEFAULT_FEEDBACK_BOUND,
    )
    .await
    {
        Ok(handle) => match handle.refusal {
            None => handle,
            Some(refusal) => {
                finalize(
                    &feedback,
                    false,
                    format!("recorder refused the episode: {refusal}"),
                )
                .await;
                return;
            }
        },
        Err(e) => {
            finalize(&feedback, false, format!("fire_goal failed: {e}")).await;
            return;
        }
    };

    // Drain feedback (one message per recorded frame) until the operator or a
    // shutdown intervenes, or the stream ends on its own: the producer closes
    // it when the goal reaches a terminal state (its own auto-stop included),
    // so a feedback error is that terminal signal, not a failure of its own.
    // Cancelling happens outside the loop because the feedback future holds
    // the handle's mutable borrow for the whole select.
    enum Ended {
        Shutdown,
        Stopped,
        Terminal,
    }
    let ended = loop {
        tokio::select! {
            _ = token.cancelled() => break Ended::Shutdown,
            _ = stop.cancelled() => break Ended::Stopped,
            msg = downstream.on_next_feedback_message() => match msg {
                Ok(fb) => {
                    let _ = feedback
                        .send(Feedback::RecordProgress { frames: fb.frames_recorded })
                        .await;
                }
                Err(_) => break Ended::Terminal,
            }
        }
    };
    match ended {
        Ended::Shutdown => {
            // Best-effort stop-and-save so shutdown does not drop frames.
            if let Err(e) = downstream.cancel_goal(GOAL_TIMEOUT).await {
                warn!(error = %e, "shutdown cancel failed");
            }
            finalize(&feedback, false, "shutting down; episode stopped").await;
            return;
        }
        Ended::Stopped => {
            // Cancel = stop and save; the result below reports the episode.
            if let Err(e) = downstream.cancel_goal(GOAL_TIMEOUT).await {
                warn!(error = %e, "stop cancel failed");
            }
        }
        Ended::Terminal => {}
    }

    let Some(outcome) = episode_result(|| downstream.get_result(RESULT_POLL), &token).await else {
        finalize(&feedback, false, "shutting down; episode result unknown").await;
        return;
    };

    let (success, summary) = match outcome {
        Ok(r) => match r.outcome {
            ResultOutcome::Cancelled { reason, data } => {
                episode_summary(&data, recorder_reason(reason))
            }
            ResultOutcome::Completed(data) => episode_summary(&data, None),
            ResultOutcome::Failed(failure) => (false, format!("episode failed: {failure}")),
            ResultOutcome::Abandoned => (false, "recorder abandoned the episode".to_string()),
            ResultOutcome::Expired => (false, "episode result expired".to_string()),
        },
        Err(e) => (false, format!("episode result error: {e}")),
    };
    finalize(&feedback, success, summary).await;
}

/// Call the recorder's finish_session (finalize + mirror + fresh session in
/// place) and report the outcome to the owner.
pub fn spawn_finish(runner: Arc<NodeRunner>, feedback: mpsc::Sender<Feedback>) {
    tokio::spawn(async move {
        let recorders = finish_session::bound_producers(&runner);
        let Some(target) = recorders.first() else {
            finish_done(&feedback, "no recorder bound".to_string()).await;
            return;
        };
        let summary = match finish_session::poll(&runner, target, FINISH_TIMEOUT).await {
            Ok(reply) => format!(
                "session {} finished and ready for replay",
                reply.data.session
            ),
            Err(e) => match finish_session::Failure::of(&e) {
                Some(failure) => format!("session not finished: {failure}"),
                None => format!("finish_session failed: {e}"),
            },
        };
        finish_done(&feedback, summary).await;
    });
}

async fn finish_done(feedback: &mpsc::Sender<Feedback>, summary: String) {
    info!(%summary, "finish_session done");
    let _ = feedback.send(Feedback::SessionFinished { summary }).await;
}

// The reason of a cancelled episode, unless it is the reason of the
// commander's own cancel: then the operator stopped it, and the recorder gave
// no cause of its own.
fn recorder_reason(reason: String) -> Option<String> {
    (reason != CancelCause::Caller.reason()).then_some(reason)
}

// The one-line outcome for the status bar, from the recorder's result fields
// and the reason the recorder gave when it ended the episode itself. The
// outcome is a success only for an episode the operator stopped whose videos
// hold every frame: an episode whose videos miss frames should not be trained
// on.
fn episode_summary(
    data: &record_episode::ResultResponseData,
    recorder_reason: Option<String>,
) -> (bool, String) {
    let ending = match &recorder_reason {
        Some(reason) => format!(" ({reason})"),
        None => String::new(),
    };
    let missing = match data.video_frames_missing {
        0 => String::new(),
        count => format!("; {count} video frame(s) missing, do not train on it"),
    };
    (
        recorder_reason.is_none() && data.video_frames_missing == 0,
        format!(
            "episode {} saved, {} frames{ending}{missing}",
            data.episode_index, data.frames_recorded
        ),
    )
}

/// The episode's result, polled with `get_result` and re-polled
/// [`RESULT_RETRY_DELAY`] after each retryable error; `None` when `token`
/// cancels first, while a poll or a retry delay is under way.
async fn episode_result<R, F>(
    mut get_result: impl FnMut() -> F,
    token: &CancellationToken,
) -> Option<peppygen::Result<R>>
where
    F: Future<Output = peppygen::Result<R>>,
{
    loop {
        let result = tokio::select! {
            _ = token.cancelled() => return None,
            result = get_result() => result,
        };
        match result {
            Err(e) if result_poll_retryable(&e) => {
                if !sleep_unless_cancelled(RESULT_RETRY_DELAY, token.cancelled()).await {
                    return None;
                }
            }
            result => return Some(result),
        }
    }
}

// Report the episode outcome to the owner, which clears the recording state and
// writes the status line; a dropped channel means the owner is gone (shutdown).
async fn finalize(feedback: &mpsc::Sender<Feedback>, success: bool, summary: impl Into<String>) {
    let summary = summary.into();
    if success {
        info!(%summary, "record_episode done");
    } else {
        warn!(%summary, "record_episode done");
    }
    let _ = feedback.send(Feedback::RecordDone { summary }).await;
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use peppygen::consumed_actions::recorder::record_episode::ResultResponseData;

    use super::{episode_result, episode_summary, recorder_reason};
    use crate::result_wait::RESULT_RETRY_DELAY;
    use crate::test_support::retryable_result_error;
    use peppylib::runtime::CancellationToken;

    /// On a paused clock that moves only when every task waits: the delay
    /// is the clock's, not the host's.
    #[tokio::test(start_paused = true)]
    async fn a_cancel_while_a_retry_is_waited_out_ends_the_result_wait_at_once() {
        let token = CancellationToken::default();
        let start = tokio::time::Instant::now();
        let polls = Cell::new(0);
        let poll = || {
            polls.set(polls.get() + 1);
            std::future::ready(Err::<(), _>(retryable_result_error()))
        };
        let cancel_once_polled = async {
            while polls.get() == 0 {
                tokio::task::yield_now().await;
            }
            token.cancel();
        };
        let (result, ()) = tokio::join!(episode_result(poll, &token), cancel_once_polled);
        assert!(result.is_none());
        assert_eq!(polls.get(), 1);
        assert!(tokio::time::Instant::now() - start < RESULT_RETRY_DELAY);
    }

    fn saved(
        episode_index: i64,
        frames_recorded: u64,
        video_frames_missing: u64,
    ) -> ResultResponseData {
        ResultResponseData {
            episode_index,
            frames_recorded,
            video_frames_missing,
        }
    }

    #[test]
    fn the_commanders_own_cancel_gives_no_recorder_reason() {
        assert_eq!(recorder_reason("cancelled by its caller".to_string()), None);
        assert_eq!(
            recorder_reason("source went stale".to_string()),
            Some("source went stale".to_string())
        );
    }

    #[test]
    fn saved_episode_reports_index_and_frames() {
        let (ok, msg) = episode_summary(&saved(4, 120, 0), None);
        assert!(ok);
        assert_eq!(msg, "episode 4 saved, 120 frames");
    }

    #[test]
    fn auto_stop_reason_rides_the_summary() {
        let (ok, msg) = episode_summary(&saved(2, 55, 0), Some("source went stale".into()));
        assert!(!ok);
        assert_eq!(msg, "episode 2 saved, 55 frames (source went stale)");
    }

    #[test]
    fn missing_video_frames_ride_the_summary_and_are_not_a_success() {
        let (ok, msg) = episode_summary(&saved(4, 120, 3), None);
        assert!(!ok);
        assert_eq!(
            msg,
            "episode 4 saved, 120 frames; 3 video frame(s) missing, do not train on it"
        );

        let (ok, msg) = episode_summary(&saved(2, 55, 1), Some("source went stale".into()));
        assert!(!ok);
        assert_eq!(
            msg,
            "episode 2 saved, 55 frames (source went stale); 1 video frame(s) missing, do not train on it"
        );
    }
}
