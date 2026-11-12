//! Spawned per fire_arm_pose command (the panel's Execute-pose in Actions mode). Fires
//! the limb_motion slot's Cartesian move_arm (planned straight-line pose move), then
//! reports the outcome to the owner. Each goal is its own task; cancel-aware so a shutdown can't
//! wedge an in-flight goal, and preempt-aware so a new move can cancel it.

use std::sync::Arc;

use peppygen::NodeRunner;
use peppygen::consumed_actions::limb_motion::move_arm as limb_motion_move_arm;
use peppylib::runtime::CancellationToken;
use robot_description::ArmId;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::owner::Feedback;
use crate::result_wait::{
    GOAL_TIMEOUT, NO_CONTACT_CHECK_M, Waited, limb_goal_handle, preempt_grace_elapsed, summarize,
    wait_for_end,
};
use crate::robot;

limb_goal_handle!(move_arm);

/// One discrete pose move, as fired at the backbone: the arm, the world-frame target
/// (position m, orientation quaternion `[x, y, z, w]`), the requested duration
/// (0 = fastest safe), and the settle tolerances (m, rad) and timeout (s) the
/// move completes under.
#[derive(Clone, Copy)]
pub struct Goal {
    pub arm: ArmId,
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
        arm,
        position,
        orientation,
        duration_s,
        settle_position_tolerance_m,
        settle_orientation_tolerance_rad,
        settle_timeout_s,
    } = goal;
    // A queued preempt fires only after the backbone releases its single-flight gate.
    if grace && !preempt_grace_elapsed(&token).await {
        return finalize(&feedback, arm, false, "shutting down; move dropped").await;
    }
    let label = robot::LIMBS.arm_name(arm);
    info!(arm = label, ?position, ?orientation, "fire move_arm");

    let goal = limb_motion_move_arm::GoalRequest {
        arm_name: label.to_string(),
        position,
        orientation: Some(orientation),
        duration_s,
        settle_position_tolerance_m,
        settle_orientation_tolerance_rad,
        settle_timeout_s,
        contact_gap_m: NO_CONTACT_CHECK_M,
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
                    arm,
                    false,
                    format!("backbone refused the pose goal: {refusal}"),
                )
                .await;
                return;
            }
        },
        Err(e) => {
            finalize(&feedback, arm, false, format!("fire_goal failed: {e}")).await;
            return;
        }
    };

    // Await the move result, honoring preempt (a new Execute cancels this goal) and
    // shutdown. Live progress is shown from the joint_states stream. The wait re-arms
    // its bounded poll until a terminal outcome (see result_wait); the backbone's
    // proportional deadline bounds the move itself.
    let ended = match wait_for_end(
        &downstream,
        &token,
        Some(&preempt),
        |trigger, e| warn!(arm = label, error = %e, "{}", trigger.failure_message()),
    )
    .await
    {
        Waited::Shutdown => {
            finalize(&feedback, arm, false, "shutting down; move cancelled").await;
            return;
        }
        Waited::Ended(ended) => ended,
    };
    // The backbone judges arrival: a completed move means the measured grasp
    // point settled within the goal's tolerances, and a failure's kind and
    // message say why not, such as `unreachable`, or `timeout` and how far
    // off it stood.
    let (success, summary) = summarize("move_arm", label, ended, |data| {
        format!("success in {:.2}s", data.action_time)
    });
    finalize(&feedback, arm, success, summary).await;
}

// Report the move outcome to the owner, which clears the in-flight slot and writes the
// status line; a dropped channel means the owner is gone (shutdown), so ignore it.
async fn finalize(
    feedback: &mpsc::Sender<Feedback>,
    arm: ArmId,
    success: bool,
    summary: impl Into<String>,
) {
    let summary = summary.into();
    let label = robot::LIMBS.arm_name(arm);
    if success {
        info!(arm = label, %summary, "move_arm done");
    } else {
        warn!(arm = label, %summary, "move_arm done");
    }
    let _ = feedback.send(Feedback::ArmGoalDone { arm, summary }).await;
}
