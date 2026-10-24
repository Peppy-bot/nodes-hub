// Spawned per fire_arm_pose command (the panel's Execute-pose in Actions mode). Fires
// the limb_motion slot's Cartesian move_arm (planned straight-line pose move), then
// reports the outcome to the owner. Each goal is its own task; cancel-aware so a shutdown can't
// wedge an in-flight goal, and preempt-aware so a new move can cancel it.

use std::sync::Arc;
use std::time::Duration;

use peppygen::NodeRunner;
use peppygen::consumed_actions::limb_motion::move_arm as limb_motion_move_arm;
use peppygen::consumed_actions::limb_motion::move_arm::ResultOutcome;
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::owner::{Feedback, PREEMPT_GRACE};
use crate::pose::{PLAN_ANGLE_TOL_RAD, PLAN_POS_TOL_M};
use crate::result_wait::{RESULT_POLL, RESULT_RETRY_DELAY, result_poll_retryable};
use crate::state::Side;

// Goal-accept round-trip to a pinned producer; answered directly, so this only needs to
// cover the decider, not a discovery probe.
const GOAL_TIMEOUT: Duration = Duration::from_secs(2);

/// One discrete pose move, as fired at the backbone: the side, the world-frame target
/// (position m, orientation quaternion `[x, y, z, w]`), the requested duration
/// (0 = fastest safe), and the settle tolerances (m, rad) and timeout (s) the
/// move completes under.
#[derive(Clone, Copy)]
pub struct Goal {
    pub side: Side,
    pub position: [f64; 3],
    pub orientation: [f64; 4],
    pub duration_s: f64,
    pub settle_position_tolerance_m: f64,
    pub settle_orientation_tolerance_rad: f64,
    pub settle_timeout_s: f64,
}

/// Fire `goal`; `grace` waits the preempt grace first (set when this move was
/// queued behind the goal it cancelled).
pub fn spawn(
    runner: Arc<NodeRunner>,
    feedback: mpsc::Sender<Feedback>,
    token: CancellationToken,
    preempt: tokio_util::sync::CancellationToken,
    goal: Goal,
    grace: bool,
) {
    tokio::spawn(async move {
        run(runner, feedback, token, preempt, goal, grace).await;
    });
}

async fn run(
    runner: Arc<NodeRunner>,
    feedback: mpsc::Sender<Feedback>,
    token: CancellationToken,
    preempt: tokio_util::sync::CancellationToken,
    goal: Goal,
    grace: bool,
) {
    let Goal {
        side,
        position,
        orientation,
        duration_s,
        settle_position_tolerance_m,
        settle_orientation_tolerance_rad,
        settle_timeout_s,
    } = goal;
    // A queued preempt fires only after the backbone releases its single-flight gate.
    if grace {
        tokio::select! {
            _ = token.cancelled() => return finalize(&feedback, side, false, "shutting down; move dropped").await,
            _ = tokio::time::sleep(PREEMPT_GRACE) => {}
        }
    }
    let label = side.label();
    info!(side = label, ?position, ?orientation, "fire move_arm");

    let goal = limb_motion_move_arm::GoalRequest {
        arm_name: side.arm_name().to_string(),
        position,
        orientation,
        duration_s,
        plan_position_tolerance_m: PLAN_POS_TOL_M,
        plan_orientation_tolerance_rad: PLAN_ANGLE_TOL_RAD,
        settle_position_tolerance_m,
        settle_orientation_tolerance_rad,
        settle_timeout_s,
    };

    let downstream = match limb_motion_move_arm::ActionHandle::fire_goal(
        &runner,
        limb_motion_move_arm::bound_producer(&runner),
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
                    side,
                    false,
                    format!("backbone refused the pose goal: {refusal}"),
                )
                .await;
                return;
            }
        },
        Err(e) => {
            finalize(&feedback, side, false, format!("fire_goal failed: {e}")).await;
            return;
        }
    };

    // Await the move result, honoring preempt (a new Execute cancels this goal) and
    // shutdown. Live progress is shown from the arm_states stream. The wait re-arms
    // its bounded poll until a terminal outcome (see result_wait); the backbone's
    // proportional deadline bounds the move itself.
    let mut preempted = false;
    let outcome = loop {
        let result_fut = downstream.get_result(RESULT_POLL);
        tokio::pin!(result_fut);
        tokio::select! {
            _ = token.cancelled() => {
                // Best-effort cancel so shutdown leaves no unsupervised motion.
                if let Err(e) = downstream.cancel_goal(GOAL_TIMEOUT).await {
                    warn!(side = side.label(), error = %e, "shutdown cancel failed");
                }
                finalize(&feedback, side, false, "shutting down; move cancelled").await;
                return;
            }
            _ = preempt.cancelled(), if !preempted => {
                preempted = true;
                if let Err(e) = downstream.cancel_goal(GOAL_TIMEOUT).await {
                    warn!(side = side.label(), error = %e, "preempt cancel failed");
                }
            }
            result = &mut result_fut => match result {
                Err(e) if result_poll_retryable(&e) => {
                    tokio::time::sleep(RESULT_RETRY_DELAY).await;
                }
                result => break result,
            }
        }
    };
    // A result error leaves the goal's fate unknown; cancel best-effort so a move
    // somehow still running does not continue unsupervised.
    if outcome.is_err()
        && let Err(e) = downstream.cancel_goal(GOAL_TIMEOUT).await
    {
        warn!(side = side.label(), error = %e, "cancel after result error failed");
    }
    // The backbone judges arrival: a completed move means the measured grasp
    // point settled within the goal's tolerances, and a failure's kind and
    // message say why not, such as `unreachable`, or `timeout` and how far
    // off it stood.
    let (success, summary) = match outcome {
        Ok(r) => match r.outcome {
            ResultOutcome::Completed(data) => (
                true,
                format!("move_arm ({label}): success in {:.2}s", data.action_time),
            ),
            ResultOutcome::Failed(failure) => {
                (false, format!("move_arm ({label}) failed: {failure}"))
            }
            ResultOutcome::Cancelled { reason, .. } => {
                (false, format!("move_arm ({label}) cancelled: {reason}"))
            }
            ResultOutcome::Abandoned => {
                (false, format!("move_arm ({label}) abandoned by backbone"))
            }
            ResultOutcome::Expired => (false, format!("move_arm ({label}) result expired")),
        },
        Err(e) => (false, format!("move_arm ({label}) result error: {e}")),
    };
    finalize(&feedback, side, success, summary).await;
}

// Report the move outcome to the owner, which clears the in-flight slot and writes the
// status line; a dropped channel means the owner is gone (shutdown), so ignore it.
async fn finalize(
    feedback: &mpsc::Sender<Feedback>,
    side: Side,
    success: bool,
    summary: impl Into<String>,
) {
    let summary = summary.into();
    if success {
        info!(side = side.label(), %summary, "move_arm done");
    } else {
        warn!(side = side.label(), %summary, "move_arm done");
    }
    let _ = feedback.send(Feedback::ArmGoalDone { side, summary }).await;
}
