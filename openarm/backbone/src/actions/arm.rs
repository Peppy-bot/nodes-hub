//! Arm move-action admission: the `move_arm_joints` and `move_arm` handlers the
//! backbone exposes to the commander. Each validates the goal (arm_name, finiteness,
//! duration, and joint limits for joint moves) and claims the target arm's
//! single-flight slot, then hands the accepted goal to that arm's planner over
//! its goal channel. The planner runs the motion - governed against the other
//! arm - completes the goal, and releases the busy slot at the terminal.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use peppygen::exposed_actions::limb_motion::{move_arm, move_arm_joints};
use peppygen::{NodeRunner, Result};
use srs_model::Limit;
use tokio::sync::mpsc;
use tracing::error;

use crate::planner::{Goal, JointReply};
use crate::types::{ARM_DOF, JointVec, Side, pose_from_wire};

use crate::actions::claim;

fn target_in_limits(q: &JointVec, limits: &[Limit; ARM_DOF]) -> bool {
    q.iter().zip(limits).all(|(&v, l)| v >= l.lo && v <= l.hi)
}

/// Expose `move_arm_joints`: validate + claim, then hand the goal to the arm's
/// planner. The planner releases the busy slot when the move ends.
pub async fn run_move_arm_joints(
    runner: Arc<NodeRunner>,
    goal_txs: [mpsc::Sender<Goal>; 2],
    busy: [Arc<AtomicBool>; 2],
    limits: [[Limit; ARM_DOF]; 2],
) -> Result<()> {
    let mut handle = move_arm_joints::ActionHandle::expose(&runner).await?;
    loop {
        let accepted = handle
            .handle_goal_next_request(|req| {
                let d = &req.data;
                let Some(idx) = Side::from_arm_name(&d.arm_name).map(Side::index) else {
                    return Ok(move_arm_joints::GoalDecision::reject(
                        Side::UNKNOWN_ARM_NAME,
                    ));
                };
                if !d.joint_positions.iter().all(|v| v.is_finite()) {
                    return Ok(move_arm_joints::GoalDecision::reject(
                        "non-finite joint target",
                    ));
                }
                if !(d.duration_s.is_finite() && d.duration_s >= 0.0) {
                    return Ok(move_arm_joints::GoalDecision::reject("invalid duration"));
                }
                if !target_in_limits(&d.joint_positions, &limits[idx]) {
                    return Ok(move_arm_joints::GoalDecision::reject(
                        "target out of joint limits",
                    ));
                }
                if !claim(&busy[idx]) {
                    return Ok(move_arm_joints::GoalDecision::reject(
                        "arm is already executing a motion",
                    ));
                }
                Ok(move_arm_joints::GoalDecision::accept())
            })
            .await?;
        let Some(ctx) = accepted else { return Ok(()) };
        let idx = Side::from_arm_name(&ctx.request().data.arm_name)
            .map(Side::index)
            .expect("validated on accept");
        let target = ctx.request().data.joint_positions;
        let duration_s = ctx.request().data.duration_s;
        if goal_txs[idx]
            .send(Goal::Joint {
                target,
                duration_s,
                reply: JointReply::MoveArmJoints(Box::new(ctx)),
            })
            .await
            .is_err()
        {
            busy[idx].store(false, Ordering::Release);
            error!("move_arm_joints: coordinator channel closed");
            return Ok(());
        }
    }
}

/// Expose `move_arm` (Cartesian): validate + claim, then hand the goal to the
/// arm's planner, which plans IK along the path and runs it governed.
pub async fn run_move_arm(
    runner: Arc<NodeRunner>,
    goal_txs: [mpsc::Sender<Goal>; 2],
    busy: [Arc<AtomicBool>; 2],
) -> Result<()> {
    let mut handle = move_arm::ActionHandle::expose(&runner).await?;
    loop {
        let accepted = handle
            .handle_goal_next_request(|req| {
                let d = &req.data;
                let Some(idx) = Side::from_arm_name(&d.arm_name).map(Side::index) else {
                    return Ok(move_arm::GoalDecision::reject(Side::UNKNOWN_ARM_NAME));
                };
                if let Err(reason) = pose_from_wire(d.position, d.orientation) {
                    return Ok(move_arm::GoalDecision::reject(format!(
                        "goal pose has {reason}"
                    )));
                }
                if !(d.duration_s.is_finite() && d.duration_s >= 0.0) {
                    return Ok(move_arm::GoalDecision::reject("invalid duration"));
                }
                if !claim(&busy[idx]) {
                    return Ok(move_arm::GoalDecision::reject(
                        "arm is already executing a motion",
                    ));
                }
                Ok(move_arm::GoalDecision::accept())
            })
            .await?;
        let Some(ctx) = accepted else { return Ok(()) };
        let idx = Side::from_arm_name(&ctx.request().data.arm_name)
            .map(Side::index)
            .expect("validated on accept");
        let target = pose_from_wire(ctx.request().data.position, ctx.request().data.orientation)
            .expect("validated on accept");
        let duration_s = ctx.request().data.duration_s;
        if goal_txs[idx]
            .send(Goal::Cartesian {
                target,
                duration_s,
                ctx: Box::new(ctx),
            })
            .await
            .is_err()
        {
            busy[idx].store(false, Ordering::Release);
            error!("move_arm: coordinator channel closed");
            return Ok(());
        }
    }
}
