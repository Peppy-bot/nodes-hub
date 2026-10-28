//! Gripper move-action admission: the `move_gripper` handler the backbone
//! exposes. Mirrors the arm move admission exactly: validate the goal
//! (gripper_name, finiteness, the [0, 1] opening range, a non-negative effort
//! cap, a settle timeout from 0 to 60 s), refusing a value it does not take
//! with `rejected`, refuse it while the gripper's follower delivers no fresh
//! state or a leader streams the gripper, claim the side's move slot, and
//! hand the accepted goal to the coordinator over its gripper goal channel.
//! The coordinator runs the motion through the same per-tick governing as
//! every other DOF, completes the goal once the jaws stand still after the
//! planned end of the commanded travel or at the goal's settle timeout, and
//! releases the move slot at the terminal.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use openarm_description::{ArmPair, Side};
use peppygen::exposed_actions::limb_motion::move_gripper::{
    ActionHandle, GoalDecision, GoalRequestData,
};
use peppygen::{NodeRunner, Result};
use tokio::sync::mpsc;
use tracing::error;

use crate::actions::{GRIPPER_WORDS, admit_limb, move_seconds_from_wire};
use crate::failure::{Failure, FailureKind};
use crate::gripper_move::GripperGoal;
use crate::move_slot::MoveSlot;
use crate::streams::LeaderActivity;

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
    let side = Side::from_gripper_name(&d.gripper_name).map_err(|e| e.to_string())?;
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
/// coordinator. The coordinator releases the move slot when the move ends.
pub async fn run_move_gripper(
    runner: Arc<NodeRunner>,
    goal_txs: ArmPair<mpsc::Sender<GripperGoal>>,
    move_slots: ArmPair<MoveSlot>,
    fresh: ArmPair<Arc<AtomicBool>>,
    activity: ArmPair<LeaderActivity>,
) -> Result<()> {
    let mut handle = ActionHandle::expose(&runner).await?;
    loop {
        let accepted = handle
            .handle_goal_next_request(|req| {
                let admitted = gripper_goal_request(&req.data)
                    .map_err(|reason| Failure::new(FailureKind::Rejected, reason))
                    .and_then(|request| {
                        let side = request.side;
                        admit_limb(
                            &fresh[side],
                            &activity[side],
                            &move_slots[side],
                            &GRIPPER_WORDS,
                        )
                    });
                Ok(match admitted {
                    Ok(()) => GoalDecision::accept(),
                    Err(f) => GoalDecision::refuse(f.kind.into(), f.message),
                })
            })
            .await?;
        let Some(ctx) = accepted else { return Ok(()) };
        let GripperGoalRequest {
            side,
            opening,
            max_effort,
            settle_timeout,
        } = gripper_goal_request(&ctx.request().data).expect("validated on accept");
        if goal_txs[side]
            .send(GripperGoal {
                opening,
                max_effort,
                settle_timeout,
                ctx,
            })
            .await
            .is_err()
        {
            move_slots[side].release();
            error!("move_gripper: coordinator channel closed");
            return Ok(());
        }
    }
}
