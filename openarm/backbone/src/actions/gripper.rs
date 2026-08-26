//! Gripper move-action admission: the `move_gripper` handler the backbone exposes.
//! Mirrors the arm move admission exactly: validate the goal (gripper_name,
//! finiteness, the [0, 1] opening range, a non-negative effort cap), claim
//! the side's single-flight slot, and
//! hand the accepted goal to the coordinator over its gripper goal channel. The
//! coordinator runs the motion through the same per-tick governing as every
//! other DOF, completes the goal on measured convergence, and releases the busy
//! slot at the terminal.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use peppygen::exposed_actions::limb_motion::move_gripper::{ActionHandle, GoalDecision};
use peppygen::{NodeRunner, Result};
use tokio::sync::mpsc;
use tracing::error;

use crate::actions::claim;
use crate::coordinator::GripperGoal;
use crate::types::Side;

/// Expose `move_gripper`: validate + claim, then hand the goal to the
/// coordinator. The coordinator releases the busy slot when the move ends.
pub async fn run_move_gripper(
    runner: Arc<NodeRunner>,
    goal_txs: [mpsc::Sender<GripperGoal>; 2],
    busy: [Arc<AtomicBool>; 2],
) -> Result<()> {
    let mut handle = ActionHandle::expose(&runner).await?;
    loop {
        let accepted = handle
            .handle_goal_next_request(|req| {
                let d = &req.data;
                let Some(idx) = Side::from_gripper_name(&d.gripper_name).map(Side::index) else {
                    return Ok(GoalDecision::reject(Side::UNKNOWN_GRIPPER_NAME));
                };
                if !d.opening.is_finite() {
                    return Ok(GoalDecision::reject("non-finite gripper opening"));
                }
                if !(0.0..=1.0).contains(&d.opening) {
                    return Ok(GoalDecision::reject(format!(
                        "opening {} outside [0, 1]",
                        d.opening
                    )));
                }
                if !d.max_effort.is_finite() || d.max_effort < 0.0 {
                    return Ok(GoalDecision::reject(format!(
                        "max_effort {} is not a non-negative finite value",
                        d.max_effort
                    )));
                }
                if !claim(&busy[idx]) {
                    return Ok(GoalDecision::reject("gripper is already executing a move"));
                }
                Ok(GoalDecision::accept())
            })
            .await?;
        let Some(ctx) = accepted else { return Ok(()) };
        let idx = Side::from_gripper_name(&ctx.request().data.gripper_name)
            .map(Side::index)
            .expect("validated on accept");
        let opening = ctx.request().data.opening;
        // The wire's 0 means no preference: the follower's ceiling stays in charge.
        let max_effort = ctx.request().data.max_effort;
        let max_effort = (max_effort > 0.0).then_some(max_effort);
        if goal_txs[idx]
            .send(GripperGoal {
                opening,
                max_effort,
                ctx,
            })
            .await
            .is_err()
        {
            busy[idx].store(false, std::sync::atomic::Ordering::Release);
            error!("move_gripper: coordinator channel closed");
            return Ok(());
        }
    }
}
