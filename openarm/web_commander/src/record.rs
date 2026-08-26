// Spawned per start_recording command (the panel's Record button). Fires the
// recorder's record_episode goal, drains its frame-count feedback into the
// owner, and reports the saved episode when the goal ends. The stop token is
// the Stop button: cancelling the goal ends the episode with a save. The
// recorder also ends episodes on its own (stale source, disk floor); that
// surfaces here as a result with an error message.

use std::sync::Arc;
use std::time::Duration;

use peppygen::NodeRunner;
use peppygen::QoSProfile;
use peppygen::consumed_actions::recorder::record_episode;
use peppygen::consumed_actions::recorder::record_episode::ResultOutcome;
use peppygen::consumed_services::recorder::finish_session;
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::owner::Feedback;
use crate::result_wait::{RESULT_POLL, RESULT_RETRY_DELAY, result_poll_retryable};

// Goal-accept round-trip to a pinned producer; answered directly, so this only
// needs to cover the decider, not a discovery probe.
const GOAL_TIMEOUT: Duration = Duration::from_secs(2);
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
    let Some(target) = record_episode::bound_producers(&runner).first() else {
        finalize(&feedback, false, "no recorder bound").await;
        return;
    };

    let goal = record_episode::GoalRequest { task };
    let mut downstream = match record_episode::ActionHandle::fire_goal(
        &runner,
        target,
        GOAL_TIMEOUT,
        goal,
        QoSProfile::Standard,
    )
    .await
    {
        Ok(handle) if handle.accepted => handle,
        Ok(handle) => {
            let reason = handle.reason.unwrap_or_else(|| "no reason given".into());
            finalize(
                &feedback,
                false,
                format!("recorder refused the episode: {reason}"),
            )
            .await;
            return;
        }
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

    let outcome = loop {
        let result_fut = downstream.get_result(RESULT_POLL);
        tokio::pin!(result_fut);
        tokio::select! {
            _ = token.cancelled() => {
                finalize(&feedback, false, "shutting down; episode result unknown").await;
                return;
            }
            result = &mut result_fut => match result {
                Err(e) if result_poll_retryable(&e) => {
                    tokio::time::sleep(RESULT_RETRY_DELAY).await;
                }
                result => break result,
            }
        }
    };

    let (success, summary) = match outcome {
        Ok(r) => match r.outcome {
            ResultOutcome::Completed(data) | ResultOutcome::Cancelled(data) => episode_summary(
                data.episode_index,
                data.frames_recorded,
                data.discarded,
                data.error,
            ),
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
        let Some(target) = finish_session::bound_producers(&runner).first() else {
            finish_done(&feedback, "no recorder bound".to_string()).await;
            return;
        };
        let summary = match finish_session::poll(&runner, target, FINISH_TIMEOUT).await {
            Ok(reply) => match reply.data.error {
                Some(reason) => format!("session not finished: {reason}"),
                None => format!(
                    "session {} finished and ready for replay",
                    reply.data.session
                ),
            },
            Err(e) => format!("finish_session failed: {e}"),
        };
        finish_done(&feedback, summary).await;
    });
}

async fn finish_done(feedback: &mpsc::Sender<Feedback>, summary: String) {
    info!(%summary, "finish_session done");
    let _ = feedback.send(Feedback::SessionFinished { summary }).await;
}

// The one-line outcome for the status bar, from the recorder's result fields.
fn episode_summary(
    episode_index: i64,
    frames: u64,
    discarded: bool,
    error: Option<String>,
) -> (bool, String) {
    let ending = match &error {
        Some(reason) => format!(" ({reason})"),
        None => String::new(),
    };
    if discarded {
        (
            false,
            format!("episode discarded after {frames} frames{ending}"),
        )
    } else {
        (
            error.is_none(),
            format!("episode {episode_index} saved, {frames} frames{ending}"),
        )
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
    use super::episode_summary;

    #[test]
    fn saved_episode_reports_index_and_frames() {
        let (ok, msg) = episode_summary(4, 120, false, None);
        assert!(ok);
        assert_eq!(msg, "episode 4 saved, 120 frames");
    }

    #[test]
    fn auto_stop_reason_rides_the_summary() {
        let (ok, msg) = episode_summary(2, 55, false, Some("source went stale".into()));
        assert!(!ok);
        assert_eq!(msg, "episode 2 saved, 55 frames (source went stale)");
    }

    #[test]
    fn discarded_episode_is_a_failure() {
        let (ok, msg) = episode_summary(-1, 9, true, None);
        assert!(!ok);
        assert_eq!(msg, "episode discarded after 9 frames");
    }
}
