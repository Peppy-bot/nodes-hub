//! Gripper move-action admission: the `move_gripper` handler the backbone
//! exposes. Mirrors the arm move admission exactly: validate the goal
//! (gripper_name, finiteness, the [0, 1] opening range, a non-negative effort
//! cap, a settle timeout from 0 to 60 s), refuse it while a leader streams
//! the gripper, claim the side's single-flight slot, and hand the accepted
//! goal to the coordinator over its gripper goal channel. The coordinator
//! runs the motion through the same per-tick governing as every other DOF,
//! completes the goal once the jaws stand still after the planned end of the
//! commanded travel or at the goal's settle timeout, and releases the busy
//! slot at the terminal.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use peppygen::exposed_actions::limb_motion::move_gripper::{
    ActionHandle, GoalDecision, GoalRequestData,
};
use peppygen::{NodeRunner, Result};
use tokio::sync::mpsc;
use tracing::error;

use crate::actions::GRIPPER_LEADER_STREAMING;
use crate::actions::claim;
use crate::coordinator::GripperGoal;
use crate::streams::LeaderActivity;
use crate::types::{Side, move_seconds_from_wire};

/// A `move_gripper` goal parsed off the wire: the side, a finite opening in
/// [0, 1], the effort cap (`None` for the wire's 0, no preference, which
/// leaves the follower's configured ceiling in charge), and the settle
/// timeout.
struct GripperGoalRequest {
    side: Side,
    opening: f64,
    max_effort: Option<f64>,
    settle_timeout: Duration,
}

fn gripper_goal_request(d: &GoalRequestData) -> std::result::Result<GripperGoalRequest, String> {
    let side = Side::from_gripper_name(&d.gripper_name)
        .ok_or_else(|| Side::UNKNOWN_GRIPPER_NAME.to_string())?;
    if !d.opening.is_finite() {
        return Err("non-finite gripper opening".to_string());
    }
    if !(0.0..=1.0).contains(&d.opening) {
        return Err(format!("opening {} outside [0, 1]", d.opening));
    }
    if !d.max_effort.is_finite() || d.max_effort < 0.0 {
        return Err(format!(
            "max_effort {} is not a non-negative finite value",
            d.max_effort
        ));
    }
    let settle_timeout = move_seconds_from_wire("settle_timeout_s", d.settle_timeout_s)?;
    Ok(GripperGoalRequest {
        side,
        opening: d.opening,
        max_effort: (d.max_effort > 0.0).then_some(d.max_effort),
        settle_timeout,
    })
}

/// Expose `move_gripper`: validate + claim, then hand the goal to the
/// coordinator. The coordinator releases the busy slot when the move ends.
pub async fn run_move_gripper(
    runner: Arc<NodeRunner>,
    goal_txs: [mpsc::Sender<GripperGoal>; 2],
    busy: [Arc<AtomicBool>; 2],
    activity: [LeaderActivity; 2],
) -> Result<()> {
    let mut handle = ActionHandle::expose(&runner).await?;
    loop {
        let accepted = handle
            .handle_goal_next_request(|req| {
                let side = match gripper_goal_request(&req.data) {
                    Ok(request) => request.side,
                    Err(reason) => return Ok(GoalDecision::reject(reason)),
                };
                if activity[side.index()].streaming() {
                    return Ok(GoalDecision::reject(GRIPPER_LEADER_STREAMING));
                }
                if !claim(&busy[side.index()]) {
                    return Ok(GoalDecision::reject("gripper is already executing a move"));
                }
                Ok(GoalDecision::accept())
            })
            .await?;
        let Some(ctx) = accepted else { return Ok(()) };
        let GripperGoalRequest {
            side,
            opening,
            max_effort,
            settle_timeout,
        } = gripper_goal_request(&ctx.request().data).expect("validated on accept");
        let idx = side.index();
        if goal_txs[idx]
            .send(GripperGoal {
                opening,
                max_effort,
                settle_timeout,
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
