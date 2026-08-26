// Spawned per fire_arm_pose command (the panel's Execute-pose in Actions mode). Fires
// the limb_motion slot's Cartesian move_arm (planned straight-line pose move), then
// reports the outcome to the owner. Each goal is its own task; cancel-aware so a shutdown can't
// wedge an in-flight goal, and preempt-aware so a new move can cancel it.

use std::sync::Arc;
use std::time::Duration;

use peppygen::NodeRunner;
use peppygen::QoSProfile;
use peppygen::consumed_actions::limb_motion::move_arm as limb_motion_move_arm;
use peppygen::consumed_actions::limb_motion::move_arm::ResultOutcome;
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::owner::{Feedback, PREEMPT_GRACE};
use crate::pose::{REACHED_ANGLE_TOL_RAD, REACHED_POS_TOL_M, dist3, quat_angle};
use crate::result_wait::{RESULT_POLL, RESULT_RETRY_DELAY, result_poll_retryable};
use crate::state::Side;

// Goal-accept round-trip to a pinned producer; answered directly, so this only needs to
// cover the decider, not a discovery probe.
const GOAL_TIMEOUT: Duration = Duration::from_secs(2);

/// One discrete pose move, as fired at the backbone: the side, the world-frame target
/// (position m, orientation quaternion `[x, y, z, w]`), the requested duration
/// (0 = fastest safe), and whether to wait the preempt grace first (set when this
/// move was queued behind the goal it cancelled).
pub struct Goal {
    pub side: Side,
    pub position: [f64; 3],
    pub orientation: [f64; 4],
    pub duration_s: f64,
    pub grace: bool,
}

pub fn spawn(
    runner: Arc<NodeRunner>,
    feedback: mpsc::Sender<Feedback>,
    token: CancellationToken,
    preempt: tokio_util::sync::CancellationToken,
    goal: Goal,
) {
    tokio::spawn(async move {
        run(runner, feedback, token, preempt, goal).await;
    });
}

async fn run(
    runner: Arc<NodeRunner>,
    feedback: mpsc::Sender<Feedback>,
    token: CancellationToken,
    preempt: tokio_util::sync::CancellationToken,
    goal: Goal,
) {
    let Goal {
        side,
        position,
        orientation,
        duration_s,
        grace,
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
    };

    let downstream = match limb_motion_move_arm::ActionHandle::fire_goal(
        &runner,
        limb_motion_move_arm::bound_producer(&runner),
        GOAL_TIMEOUT,
        goal,
        QoSProfile::SensorData,
    )
    .await
    {
        Ok(handle) if handle.accepted => handle,
        Ok(handle) => {
            let reason = handle.reason.unwrap_or_else(|| "no reason given".into());
            finalize(
                &feedback,
                side,
                false,
                format!("backbone rejected the pose goal: {reason}"),
            )
            .await;
            return;
        }
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
    let (success, summary) = match outcome {
        Ok(r) => match r.outcome {
            ResultOutcome::Completed(data) => {
                if !data.success {
                    (
                        false,
                        format!("move_arm ({label}) failed: {}", data.message),
                    )
                } else {
                    // Confirm the arm actually reached the commanded pose, not just that
                    // the trajectory finished (a governor stop finishes it too).
                    let pos_err = dist3(data.final_position, position);
                    let rot_err = quat_angle(data.final_orientation, orientation);
                    if pos_err <= REACHED_POS_TOL_M && rot_err <= REACHED_ANGLE_TOL_RAD {
                        (
                            true,
                            format!("move_arm ({label}): success in {:.2}s", data.action_time),
                        )
                    } else {
                        (
                            false,
                            format!(
                                "move_arm ({label}) ended {:.0} mm / {:.1} deg off target (blocked?)",
                                pos_err * 1000.0,
                                rot_err.to_degrees()
                            ),
                        )
                    }
                }
            }
            ResultOutcome::Cancelled(data) => (
                false,
                format!("move_arm ({label}) cancelled: {}", data.message),
            ),
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
