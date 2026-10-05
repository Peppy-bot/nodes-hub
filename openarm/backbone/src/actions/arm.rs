//! Arm move-action admission: the `move_arm_joints` and `move_arm` handlers the
//! backbone exposes to the commander, and the `check_arm_move` service that
//! asks whether a Cartesian goal has a plan without moving. Each validates
//! the goal (arm_name, finiteness, duration, and joint limits for joint
//! moves); a move claims the target arm's single-flight slot, then hands the
//! accepted goal to that arm's planner over its goal channel. The planner
//! runs the motion - governed against the other arm - completes the goal, and
//! releases the busy slot at the terminal. The check claims nothing: the
//! coordinator asks the planner and answers the duration or the refusal.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use peppygen::exposed_actions::limb_motion::{move_arm, move_arm_joints};
use peppygen::exposed_services::limb_motion::check_arm_move;
use peppygen::{NodeRunner, Result};
use srs_model::Limit;
use srs_model::nalgebra::Isometry3;
use tokio::sync::{mpsc, oneshot};
use tracing::error;

use crate::coordinator::CoordinatorRequest;
use crate::planner::{ARM_BUSY, Goal, JointReply};
use crate::types::{ARM_DOF, JointVec, PlanTolerance, Side, pose_from_wire};

use crate::actions::{blocking_ask_coordinator, claim};

fn target_in_limits(q: &JointVec, limits: &[Limit; ARM_DOF]) -> bool {
    q.iter().zip(limits).all(|(&v, l)| v >= l.lo && v <= l.hi)
}

/// A Cartesian move as `move_arm` takes it and `check_arm_move` checks it,
/// parsed once off the wire: both carry the same fields with the same
/// meaning, and both refuse the same values with the same words.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ArmMoveRequest {
    pub side: Side,
    pub target: Isometry3<f64>,
    pub tolerance: PlanTolerance,
    pub duration_s: f64,
}

impl ArmMoveRequest {
    pub(crate) fn from_wire(
        arm_name: &str,
        position: [f64; 3],
        orientation: [f64; 4],
        duration_s: f64,
        plan_position_tolerance_m: f64,
        plan_orientation_tolerance_rad: f64,
    ) -> std::result::Result<Self, String> {
        let side =
            Side::from_arm_name(arm_name).ok_or_else(|| Side::UNKNOWN_ARM_NAME.to_string())?;
        let target = pose_from_wire(position, orientation)
            .map_err(|reason| format!("goal pose has {reason}"))?;
        let tolerance =
            PlanTolerance::from_wire(plan_position_tolerance_m, plan_orientation_tolerance_rad)
                .map_err(|reason| format!("goal has {reason}"))?;
        if !(duration_s.is_finite() && duration_s >= 0.0) {
            return Err("invalid duration".to_string());
        }
        Ok(Self {
            side,
            target,
            tolerance,
            duration_s,
        })
    }
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
                    return Ok(move_arm_joints::GoalDecision::reject(ARM_BUSY));
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

/// The `move_arm` goal's fields as [`ArmMoveRequest::from_wire`] takes them.
fn move_arm_request(d: &move_arm::GoalRequestData) -> std::result::Result<ArmMoveRequest, String> {
    ArmMoveRequest::from_wire(
        &d.arm_name,
        d.position,
        d.orientation,
        d.duration_s,
        d.plan_position_tolerance_m,
        d.plan_orientation_tolerance_rad,
    )
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
                let request = match move_arm_request(&req.data) {
                    Ok(request) => request,
                    Err(reason) => return Ok(move_arm::GoalDecision::reject(reason)),
                };
                if !claim(&busy[request.side.index()]) {
                    return Ok(move_arm::GoalDecision::reject(ARM_BUSY));
                }
                Ok(move_arm::GoalDecision::accept())
            })
            .await?;
        let Some(ctx) = accepted else { return Ok(()) };
        let request = move_arm_request(&ctx.request().data).expect("validated on accept");
        let idx = request.side.index();
        if goal_txs[idx]
            .send(Goal::Cartesian {
                target: request.target,
                tolerance: request.tolerance,
                duration_s: request.duration_s,
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

/// Expose `check_arm_move`: validate the fields as `move_arm` does, then ask
/// the coordinator, which asks the arm's planner for a plan from the held
/// setpoint and moves nothing. The answer is the duration the move would
/// take, or the words a `move_arm` refusal gives.
pub async fn run_check_arm_move(
    runner: Arc<NodeRunner>,
    requests: mpsc::Sender<CoordinatorRequest>,
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
            let (reply, answer) = oneshot::channel();
            let asked = blocking_ask_coordinator(
                &requests,
                CoordinatorRequest::CheckArmMove { request, reply },
                answer,
            );
            Ok(match asked {
                Ok(Ok(duration_s)) => check_arm_move::Response::new(
                    true,
                    format!("a plan reaches the pose in {duration_s:.3} s"),
                    duration_s,
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
