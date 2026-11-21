//! Gripper move-action admission: the `move_gripper` handler the backbone
//! exposes. Mirrors the arm move admission exactly: validate the goal
//! (gripper_name, finiteness, the [0, 1] opening range, a non-negative effort
//! cap, a settle timeout from 0 to 60 s), refusing a value it does not take
//! with `rejected`, refuse it while the gripper's follower delivers no fresh
//! state or a leader streams the gripper, claim the arm's move slot, and
//! hand the accepted goal to the coordinator over its gripper goal channel.
//! The coordinator runs the motion through the same per-tick governing as
//! every other DOF, completes the goal once the jaws stand still after the
//! planned end of the commanded travel or at the goal's settle timeout, and
//! releases the move slot at the terminal.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::robot::{ARMS, gripper_in};
use control_core::move_seconds::MoveSeconds;
use control_core::move_slot::MoveSlot;
use control_core::sync::lock_ignoring_poison;
use peppygen::exposed_actions::limb_motion::move_gripper::{
    ActionHandle, GoalDecision, GoalRequestData,
};
use peppygen::{NodeRunner, Result};
use robot_description::{ArmId, PerArm};
use tokio::sync::mpsc;
use tracing::error;

use crate::actions::{GRIPPER_WORDS, admit_limb};
use crate::failure::{Failure, FailureKind};
use crate::gripper_move::{EffortCap, GripperGoal, GripperReply};
use crate::streams::LeaderActivity;

/// A `move_gripper` goal parsed off the wire: the arm, a finite opening in
/// [0, 1], the effort cap (`None` for the wire's 0, no preference, which
/// leaves the follower's configured ceiling in charge), and the settle
/// timeout.
struct GripperGoalRequest {
    arm: ArmId,
    opening: f64,
    max_effort: Option<f64>,
    settle_timeout: Duration,
}

fn gripper_goal_request(d: &GoalRequestData) -> std::result::Result<GripperGoalRequest, String> {
    let arm = gripper_in("gripper_name", &d.gripper_name)?;
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
    let settle_timeout = MoveSeconds::from_wire("settle_timeout_s", d.settle_timeout_s)?.duration();
    Ok(GripperGoalRequest {
        arm,
        opening: d.opening,
        max_effort: (d.max_effort > 0.0).then_some(d.max_effort),
        settle_timeout,
    })
}

/// Expose `move_gripper`: validate + claim, then hand the goal to the
/// coordinator with the release of its claim. The coordinator releases the
/// move slot when the move ends.
pub async fn run_move_gripper(
    runner: Arc<NodeRunner>,
    goal_txs: PerArm<mpsc::Sender<GripperGoal>, ARMS>,
    move_slots: PerArm<MoveSlot, ARMS>,
    fresh: PerArm<Arc<AtomicBool>, ARMS>,
    activity: PerArm<LeaderActivity, ARMS>,
) -> Result<()> {
    let mut handle = ActionHandle::expose(&runner).await?;
    loop {
        let accepted_goal = Mutex::new(None);
        let accepted = handle
            .handle_goal_next_request(|req| {
                let admitted = gripper_goal_request(&req.data)
                    .map_err(|reason| Failure::new(FailureKind::Rejected, reason))
                    .and_then(|request| {
                        let arm = request.arm;
                        let move_slot = admit_limb(
                            &fresh[arm],
                            &activity[arm],
                            &move_slots[arm],
                            &GRIPPER_WORDS,
                        )?;
                        Ok((request, move_slot))
                    });
                Ok(match admitted {
                    Ok(goal) => {
                        *lock_ignoring_poison(&accepted_goal) = Some(goal);
                        GoalDecision::accept()
                    }
                    Err(f) => GoalDecision::refuse(f.kind.into(), f.message),
                })
            })
            .await?;
        let Some(ctx) = accepted else { return Ok(()) };
        let (
            GripperGoalRequest {
                arm,
                opening,
                max_effort,
                settle_timeout,
            },
            move_slot,
        ) = lock_ignoring_poison(&accepted_goal)
            .take()
            .expect("the decider keeps the goal it accepts");
        let goal = GripperGoal {
            opening,
            effort: EffortCap::Named(max_effort),
            settle_timeout,
            travel_timeout: None,
            reply: GripperReply::MoveGripper(Box::new(ctx)),
            move_slot,
        };
        if goal_txs[arm].send(goal).await.is_err() {
            error!("move_gripper: coordinator channel closed");
            return Ok(());
        }
    }
}
