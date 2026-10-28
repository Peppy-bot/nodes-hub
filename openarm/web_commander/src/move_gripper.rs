//! Spawned per fire_gripper command (the gripper card's Execute in Actions mode). Fires
//! the limb_motion slot's move_gripper (a discrete governed open/close), then reports the
//! outcome to the owner. Cancel-aware so a shutdown can't wedge an in-flight goal. A second Execute
//! is refused while one is in flight (the owner gates it), so this needs no per-goal
//! preempt the way the longer arm moves do.

use std::sync::Arc;

use openarm_description::Side;
use peppygen::NodeRunner;
use peppygen::consumed_actions::limb_motion::move_gripper as limb_motion_move_gripper;
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::owner::Feedback;
use crate::result_wait::{GOAL_TIMEOUT, Waited, limb_goal_handle, summarize, wait_for_end};

limb_goal_handle!(move_gripper);

/// One discrete gripper move, as fired at the backbone: the side, the opening
/// fraction, the effort cap in its wire encoding (0 = no preference), and the
/// longest (s) the backbone waits after the planned end of the travel for the
/// jaws to stand still.
pub struct Goal {
    pub side: Side,
    pub opening: f64,
    pub max_effort: f64,
    pub settle_timeout_s: f64,
}

pub fn spawn(
    runner: Arc<NodeRunner>,
    feedback: mpsc::Sender<Feedback>,
    token: CancellationToken,
    goal: Goal,
) {
    tokio::spawn(async move {
        run(runner, feedback, token, goal).await;
    });
}

async fn run(
    runner: Arc<NodeRunner>,
    feedback: mpsc::Sender<Feedback>,
    token: CancellationToken,
    goal: Goal,
) {
    let Goal {
        side,
        opening,
        max_effort,
        settle_timeout_s,
    } = goal;
    let label = side.to_string();
    info!(
        side = label.as_str(),
        opening, max_effort, "fire move_gripper"
    );

    let goal = limb_motion_move_gripper::GoalRequest {
        gripper_name: side.gripper_name().to_string(),
        opening,
        max_effort,
        settle_timeout_s,
    };

    let downstream = match limb_motion_move_gripper::ActionHandle::fire_goal(
        &runner,
        limb_motion_move_gripper::bound_producer(&runner),
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
                    format!("backbone refused the gripper goal: {refusal}"),
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

    // Await the result, honoring shutdown. A rejected concurrent goal cannot happen
    // here: the owner refuses a second Execute while one is in flight, so unlike the
    // arm moves there is no preempt branch. The wait re-arms its bounded poll until
    // a terminal outcome (see result_wait); the backbone's own deadline bounds the
    // move itself.
    let ended = match wait_for_end(
        &downstream,
        &token,
        None,
        |trigger, e| warn!(side = label.as_str(), error = %e, "{}", trigger.failure_message()),
    )
    .await
    {
        Waited::Shutdown => {
            finalize(&feedback, side, false, "shutting down; move cancelled").await;
            return;
        }
        Waited::Ended(ended) => ended,
    };
    let (success, summary) = summarize("move_gripper", &label, ended, |data| {
        // The backbone reports where the jaws measured once they stood
        // still, or settle_timeout_s after the planned end of the travel:
        // on the target, or where an object or the effort cap stopped
        // them, which a grasp reads as its result.
        let settle = if data.settled {
            ""
        } else {
            ", the jaws had not stood still by the settle timeout"
        };
        format!(
            "success in {:.2}s, opening {:.2}{settle}",
            data.action_time, data.final_opening
        )
    });
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
    let label = side.to_string();
    if success {
        info!(side = label.as_str(), %summary, "move_gripper done");
    } else {
        warn!(side = label.as_str(), %summary, "move_gripper done");
    }
    let _ = feedback
        .send(Feedback::GripperGoalDone { side, summary })
        .await;
}
