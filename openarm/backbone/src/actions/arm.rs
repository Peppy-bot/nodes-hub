//! Arm move-action admission: the `move_arm_joints` and `move_arm` handlers the
//! backbone exposes to the commander, and the `check_arm_move` service that
//! asks whether a Cartesian goal has a plan without moving. Each validates
//! the goal (arm_name, finiteness, duration, the settle it asks for, and
//! joint limits for joint moves), refusing a value it does not take with
//! `rejected`, and refuses it while the arm's follower delivers no fresh
//! state or a leader streams the arm ([`admit_limb`]); a move claims the
//! target arm's single-flight slot, then hands the accepted goal to that
//! arm's planner over its goal channel. The planner runs the motion -
//! governed against the other arm - settles it, ends the goal, and releases
//! the busy slot at the terminal. The check claims nothing: the coordinator
//! asks the planner and answers whether a plan exists, or the refusal a
//! goal would get, as an error reply.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use peppygen::exposed_actions::limb_motion::{move_arm, move_arm_joints};
use peppygen::exposed_services::limb_motion::check_arm_move;
use peppygen::{NodeRunner, Result};
use srs_model::Limit;
use tokio::sync::{mpsc, oneshot};
use tracing::error;

use crate::actions::{ARM_WORDS, admit_limb, blocking_ask_coordinator, check_limb_free};
use crate::arm_pair::ArmPair;
use crate::coordinator::CoordinatorRequest;
use crate::failure::{Failure, FailureKind};
use crate::planner::{Goal, JointReply, PlanCheck};
use crate::settle::Settle;
use crate::streams::LeaderActivity;
use crate::types::{
    ARM_DOF, ArmMoveRequest, JointVec, PoseTolerance, Side, joint_tolerances_from_wire,
    move_seconds_from_wire,
};

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
    limits: &ArmPair<[Limit; ARM_DOF]>,
) -> std::result::Result<JointMoveRequest, String> {
    let side =
        Side::from_arm_name(&d.arm_name).ok_or_else(|| Side::UNKNOWN_ARM_NAME.to_string())?;
    if !d.joint_positions.iter().all(|v| v.is_finite()) {
        return Err("non-finite joint target".to_string());
    }
    let duration = move_seconds_from_wire("duration_s", d.duration_s)?;
    if !target_in_limits(&d.joint_positions, &limits[side]) {
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
    goal_txs: ArmPair<mpsc::Sender<Goal>>,
    busy: ArmPair<Arc<AtomicBool>>,
    fresh: ArmPair<Arc<AtomicBool>>,
    activity: ArmPair<LeaderActivity>,
    limits: ArmPair<[Limit; ARM_DOF]>,
) -> Result<()> {
    let mut handle = move_arm_joints::ActionHandle::expose(&runner).await?;
    loop {
        let accepted = handle
            .handle_goal_next_request(|req| {
                let admitted = joint_move_request(&req.data, &limits)
                    .map_err(|reason| Failure::new(FailureKind::Rejected, reason))
                    .and_then(|request| {
                        let side = request.side;
                        admit_limb(&fresh[side], &activity[side], &busy[side], &ARM_WORDS)
                    });
                Ok(match admitted {
                    Ok(()) => move_arm_joints::GoalDecision::accept(),
                    Err(f) => move_arm_joints::GoalDecision::refuse(f.kind.into(), f.message),
                })
            })
            .await?;
        let Some(ctx) = accepted else { return Ok(()) };
        let request =
            joint_move_request(&ctx.request().data, &limits).expect("validated on accept");
        let side = request.side;
        if goal_txs[side]
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
            busy[side].store(false, Ordering::Release);
            error!("move_arm_joints: coordinator channel closed");
            return Ok(());
        }
    }
}

/// Expose `move_arm` (Cartesian): validate + claim, then hand the goal to the
/// arm's planner, which plans IK along the path and runs it governed.
pub async fn run_move_arm(
    runner: Arc<NodeRunner>,
    goal_txs: ArmPair<mpsc::Sender<Goal>>,
    busy: ArmPair<Arc<AtomicBool>>,
    fresh: ArmPair<Arc<AtomicBool>>,
    activity: ArmPair<LeaderActivity>,
) -> Result<()> {
    let mut handle = move_arm::ActionHandle::expose(&runner).await?;
    loop {
        let accepted = handle
            .handle_goal_next_request(|req| {
                let admitted = move_arm_goal(&req.data)
                    .map_err(|reason| Failure::new(FailureKind::Rejected, reason))
                    .and_then(|goal| {
                        let side = goal.request.side;
                        admit_limb(&fresh[side], &activity[side], &busy[side], &ARM_WORDS)
                    });
                Ok(match admitted {
                    Ok(()) => move_arm::GoalDecision::accept(),
                    Err(f) => move_arm::GoalDecision::refuse(f.kind.into(), f.message),
                })
            })
            .await?;
        let Some(ctx) = accepted else { return Ok(()) };
        let MoveArmGoal { request, settle } =
            move_arm_goal(&ctx.request().data).expect("validated on accept");
        let side = request.side;
        if goal_txs[side]
            .send(Goal::Cartesian {
                request,
                settle,
                ctx: Box::new(ctx),
            })
            .await
            .is_err()
        {
            busy[side].store(false, Ordering::Release);
            error!("move_arm: coordinator channel closed");
            return Ok(());
        }
    }
}

/// Expose `check_arm_move`: validate the planning fields as `move_arm` does,
/// give an error reply where `move_arm` refuses (a value it does not take,
/// no fresh follower state, a streaming leader, the arm busy), then ask the
/// coordinator, which asks the arm's planner for a plan from the held
/// setpoint and moves nothing. The answer is whether a plan exists: the time
/// the move would take, or the words a `move_arm` failure with `unreachable`
/// gives. A coordinator that does not answer gives `attempt_failed`.
pub async fn run_check_arm_move(
    runner: Arc<NodeRunner>,
    requests: mpsc::Sender<CoordinatorRequest>,
    fresh: ArmPair<Arc<AtomicBool>>,
    activity: ArmPair<LeaderActivity>,
) -> Result<()> {
    loop {
        check_arm_move::handle_next_request(&runner, |req| {
            check_plan(&requests, &fresh, &activity, &req.data)
                .map_err(|f| check_arm_move::Failure::new(f.kind.into(), f.message))
        })
        .await?;
    }
}

/// The answer to one plan check, or the refusal a `move_arm` goal with these
/// fields would get.
fn check_plan(
    requests: &mpsc::Sender<CoordinatorRequest>,
    fresh: &ArmPair<Arc<AtomicBool>>,
    activity: &ArmPair<LeaderActivity>,
    d: &check_arm_move::RequestData,
) -> std::result::Result<check_arm_move::Response, Failure> {
    let request = ArmMoveRequest::from_wire(
        &d.arm_name,
        d.position,
        d.orientation,
        d.duration_s,
        d.plan_position_tolerance_m,
        d.plan_orientation_tolerance_rad,
    )
    .map_err(|reason| Failure::new(FailureKind::Rejected, reason))?;
    let side = request.side;
    check_limb_free(
        fresh[side].load(Ordering::Acquire),
        activity[side].streaming(),
        &ARM_WORDS,
    )?;
    let (reply, answer) = oneshot::channel();
    let checked = blocking_ask_coordinator(
        requests,
        CoordinatorRequest::CheckArmMove { request, reply },
        answer,
    )
    .map_err(|reason| {
        error!("check_arm_move: {reason}");
        Failure::new(FailureKind::AttemptFailed, reason)
    })??;
    Ok(match checked {
        PlanCheck::Reachable(duration) => {
            check_arm_move::Response::new(true, String::new(), duration.as_secs_f64())
        }
        PlanCheck::Unreachable(reason) => check_arm_move::Response::new(false, reason, 0.0),
    })
}
