//! Spawned per fire_arm command (the panel's Home/Ready parks, as discrete governed
//! moves). Fires the limb_motion slot's move_arm_joints, then reports the outcome to the
//! owner. Each goal is its own task; cancel-aware so a shutdown can't wedge an in-flight
//! goal, and preempt-aware so a new move can cancel it.

use std::sync::Arc;

use openarm_description::{ARM_DOF, Side};
use peppygen::NodeRunner;
use peppygen::consumed_actions::limb_motion::move_arm_joints as limb_motion_move_arm_joints;
use peppylib::runtime::CancellationToken;
use srs_model::JointVec;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::owner::Feedback;
use crate::result_wait::{
    GOAL_TIMEOUT, Waited, limb_goal_handle, preempt_grace_elapsed, summarize, wait_for_end,
};

limb_goal_handle!(move_arm_joints);

/// One discrete joint move, as fired at the backbone: the side, the 7-joint target
/// (rad), the requested duration (0 = fastest safe), and the settle tolerance
/// (rad, the one bar every joint is held to) and timeout (s) the move
/// completes under.
#[derive(Clone, Copy)]
pub struct Goal {
    pub side: Side,
    pub joint_positions: JointVec,
    pub duration_s: f64,
    pub settle_tolerance_rad: f64,
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
        joint_positions,
        duration_s,
        settle_tolerance_rad,
        settle_timeout_s,
    } = goal;
    // A queued preempt fires only after the backbone releases its single-flight gate.
    if grace && !preempt_grace_elapsed(&token).await {
        return finalize(&feedback, side, false, "shutting down; move dropped").await;
    }
    let label = side.to_string();
    info!(
        side = label.as_str(),
        ?joint_positions,
        "fire move_arm_joints"
    );

    let goal = limb_motion_move_arm_joints::GoalRequest {
        arm_name: side.arm_name().to_string(),
        joint_positions,
        duration_s,
        settle_tolerances_rad: [settle_tolerance_rad; ARM_DOF],
        settle_timeout_s,
    };

    // The launcher-pinned, cardinality-one limb_motion slot provides the explicit
    // target used for this goal and its feedback/cancel/result lifecycle.
    let downstream = match limb_motion_move_arm_joints::ActionHandle::fire_goal(
        &runner,
        limb_motion_move_arm_joints::bound_producer(&runner),
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
                    format!("backbone refused the goal: {refusal}"),
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

    // Await the move result, honoring preempt (a new move cancels this goal) and
    // shutdown. There is no feedback to drain: live progress is shown from the
    // joint_states stream (see joint_states.rs). The wait re-arms its bounded poll
    // until a terminal outcome (see result_wait); the backbone's own deadline
    // bounds the move itself.
    let ended = match wait_for_end(
        &downstream,
        &token,
        Some(&preempt),
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
    // The backbone judges arrival: a completed move means the measured joints
    // settled within the goal's tolerance, and a failure's kind and message
    // say why not, such as `timeout` and how far off they stood.
    let (success, summary) = summarize("move_arm_joints", &label, ended, |data| {
        format!("success in {:.2}s", data.action_time)
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
        info!(side = label.as_str(), %summary, "move_arm_joints done");
    } else {
        warn!(side = label.as_str(), %summary, "move_arm_joints done");
    }
    let _ = feedback.send(Feedback::ArmGoalDone { side, summary }).await;
}
