//! Arm move-action admission: the `move_arm_joints` and `move_arm` handlers the
//! backbone exposes to the commander, and the `check_arm_move` service that
//! asks whether a Cartesian goal has a plan without moving. Each validates
//! the goal (arm_name, finiteness, duration, the settle it asks for, and
//! joint limits for joint moves) and refuses it while a leader streams the
//! arm; a move claims the target arm's single-flight slot, then hands the
//! accepted goal to that arm's planner over its goal channel. The planner
//! runs the motion - governed against the other arm - settles it, completes
//! the goal, and releases the busy slot at the terminal. The check claims
//! nothing: the coordinator asks the planner and answers the duration or the
//! refusal.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use peppygen::exposed_actions::limb_motion::{move_arm, move_arm_joints};
use peppygen::exposed_services::limb_motion::check_arm_move;
use peppygen::{NodeRunner, Result};
use srs_model::Limit;
use tokio::sync::{mpsc, oneshot};
use tracing::error;

use crate::coordinator::CoordinatorRequest;
use crate::planner::{ARM_BUSY, Goal, JointReply};
use crate::settle::Settle;
use crate::types::{
    ARM_DOF, ArmMoveRequest, JointVec, PoseTolerance, Side, joint_tolerances_from_wire,
    move_seconds_from_wire,
};

use crate::actions::{ARM_LEADER_STREAMING, blocking_ask_coordinator, claim};
use crate::streams::LeaderActivity;

fn target_in_limits(q: &JointVec, limits: &[Limit; ARM_DOF]) -> bool {
    q.iter().zip(limits).all(|(&v, l)| v >= l.lo && v <= l.hi)
}

/// A `move_arm` goal parsed off the wire: the move to plan, and the settle it
/// asks for, whose slack covers the plan's on each axis.
struct MoveArmGoal {
    request: ArmMoveRequest,
    settle: Settle<PoseTolerance>,
}

fn move_arm_goal(d: &move_arm::GoalRequestData) -> std::result::Result<MoveArmGoal, String> {
    let request = ArmMoveRequest::from_wire(
        &d.arm_name,
        d.position,
        d.orientation,
        d.duration_s,
        d.plan_position_tolerance_m,
        d.plan_orientation_tolerance_rad,
    )?;
    let tolerance = PoseTolerance::from_wire(
        "settle",
        d.settle_position_tolerance_m,
        d.settle_orientation_tolerance_rad,
    )?;
    tolerance.covers(&request.tolerance)?;
    let timeout = move_seconds_from_wire("settle_timeout_s", d.settle_timeout_s)?;
    Ok(MoveArmGoal {
        request,
        settle: Settle { tolerance, timeout },
    })
}

/// A `move_arm_joints` goal parsed off the wire: a finite target inside the
/// arm's joint limits, the duration, and the settle it asks for.
struct JointMoveRequest {
    side: Side,
    target: JointVec,
    duration: Duration,
    settle: Settle<JointVec>,
}

fn joint_move_request(
    d: &move_arm_joints::GoalRequestData,
    limits: &[[Limit; ARM_DOF]; 2],
) -> std::result::Result<JointMoveRequest, String> {
    let side =
        Side::from_arm_name(&d.arm_name).ok_or_else(|| Side::UNKNOWN_ARM_NAME.to_string())?;
    if !d.joint_positions.iter().all(|v| v.is_finite()) {
        return Err("non-finite joint target".to_string());
    }
    let duration = move_seconds_from_wire("duration_s", d.duration_s)?;
    if !target_in_limits(&d.joint_positions, &limits[side.index()]) {
        return Err("target out of joint limits".to_string());
    }
    let settle = Settle {
        tolerance: joint_tolerances_from_wire(d.settle_tolerances_rad)?,
        timeout: move_seconds_from_wire("settle_timeout_s", d.settle_timeout_s)?,
    };
    Ok(JointMoveRequest {
        side,
        target: d.joint_positions,
        duration,
        settle,
    })
}

/// Expose `move_arm_joints`: validate + claim, then hand the goal to the arm's
/// planner. The planner releases the busy slot when the move ends.
pub async fn run_move_arm_joints(
    runner: Arc<NodeRunner>,
    goal_txs: [mpsc::Sender<Goal>; 2],
    busy: [Arc<AtomicBool>; 2],
    activity: [LeaderActivity; 2],
    limits: [[Limit; ARM_DOF]; 2],
) -> Result<()> {
    let mut handle = move_arm_joints::ActionHandle::expose(&runner).await?;
    loop {
        let accepted = handle
            .handle_goal_next_request(|req| {
                let request = match joint_move_request(&req.data, &limits) {
                    Ok(request) => request,
                    Err(reason) => return Ok(move_arm_joints::GoalDecision::reject(reason)),
                };
                if activity[request.side.index()].streaming() {
                    return Ok(move_arm_joints::GoalDecision::reject(ARM_LEADER_STREAMING));
                }
                if !claim(&busy[request.side.index()]) {
                    return Ok(move_arm_joints::GoalDecision::reject(ARM_BUSY));
                }
                Ok(move_arm_joints::GoalDecision::accept())
            })
            .await?;
        let Some(ctx) = accepted else { return Ok(()) };
        let request =
            joint_move_request(&ctx.request().data, &limits).expect("validated on accept");
        let idx = request.side.index();
        if goal_txs[idx]
            .send(Goal::Joint {
                target: request.target,
                duration: request.duration,
                reply: JointReply::MoveArmJoints {
                    ctx: Box::new(ctx),
                    settle: request.settle,
                },
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
    activity: [LeaderActivity; 2],
) -> Result<()> {
    let mut handle = move_arm::ActionHandle::expose(&runner).await?;
    loop {
        let accepted = handle
            .handle_goal_next_request(|req| {
                let side = match move_arm_goal(&req.data) {
                    Ok(goal) => goal.request.side,
                    Err(reason) => return Ok(move_arm::GoalDecision::reject(reason)),
                };
                if activity[side.index()].streaming() {
                    return Ok(move_arm::GoalDecision::reject(ARM_LEADER_STREAMING));
                }
                if !claim(&busy[side.index()]) {
                    return Ok(move_arm::GoalDecision::reject(ARM_BUSY));
                }
                Ok(move_arm::GoalDecision::accept())
            })
            .await?;
        let Some(ctx) = accepted else { return Ok(()) };
        let MoveArmGoal { request, settle } =
            move_arm_goal(&ctx.request().data).expect("validated on accept");
        let idx = request.side.index();
        if goal_txs[idx]
            .send(Goal::Cartesian {
                request,
                settle,
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

/// Expose `check_arm_move`: validate the planning fields as `move_arm` does,
/// then ask the coordinator, which asks the arm's planner for a plan from the
/// held setpoint and moves nothing. The answer is the duration the move would
/// take, or the words a `move_arm` refusal gives.
pub async fn run_check_arm_move(
    runner: Arc<NodeRunner>,
    requests: mpsc::Sender<CoordinatorRequest>,
    activity: [LeaderActivity; 2],
) -> Result<()> {
    loop {
        check_arm_move::handle_next_request(&runner, |req| {
            let d = &req.data;
            let request = ArmMoveRequest::from_wire(
                &d.arm_name,
                d.position,
                d.orientation,
                d.duration_s,
                d.plan_position_tolerance_m,
                d.plan_orientation_tolerance_rad,
            );
            let request = match request {
                Ok(request) => request,
                Err(reason) => return Ok(check_arm_move::Response::new(false, reason, 0.0)),
            };
            if activity[request.side.index()].streaming() {
                return Ok(check_arm_move::Response::new(
                    false,
                    ARM_LEADER_STREAMING.to_string(),
                    0.0,
                ));
            }
            let (reply, answer) = oneshot::channel();
            let asked = blocking_ask_coordinator(
                &requests,
                CoordinatorRequest::CheckArmMove { request, reply },
                answer,
            );
            Ok(match asked {
                Ok(Ok(duration)) => check_arm_move::Response::new(
                    true,
                    format!("a plan reaches the pose in {:.3} s", duration.as_secs_f64()),
                    duration.as_secs_f64(),
                ),
                Ok(Err(refusal)) => check_arm_move::Response::new(false, refusal, 0.0),
                Err(refusal) => {
                    error!("check_arm_move: {refusal}");
                    check_arm_move::Response::new(false, refusal, 0.0)
                }
            })
        })
        .await?;
    }
}
