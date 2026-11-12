//! Arm move-action admission: the `move_arm_joints` and `move_arm` handlers the
//! backbone exposes to the commander, and the `check_arm_move` service that
//! asks whether a Cartesian goal has a plan without moving. Each validates
//! the goal (arm_name, finiteness, duration, the settle it asks for, and
//! joint limits for joint moves), refusing a value it does not take with
//! `rejected`, and refuses it while the arm's follower delivers no fresh
//! state or a leader streams the arm ([`admit_limb`]); a move claims the
//! target arm's move slot, then hands the accepted goal to that arm's
//! planner over its goal channel. The planner runs the motion - governed
//! against the other arm - settles it, ends the goal, and releases the move
//! slot at the terminal. The check claims nothing: the coordinator
//! asks the planner and answers whether a plan exists, or the refusal a
//! goal would get, as an error reply.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::robot::{ARMS, arm_in};
use control_core::positive_finite::PositiveFinite;
use control_core::serving;
use peppygen::exposed_actions::limb_motion::{move_arm, move_arm_joints};
use peppygen::exposed_services::limb_motion::check_arm_move;
use peppygen::{NodeRunner, Result};
use peppylib::runtime::CancellationToken;
use robot_description::{ArmId, PerArm};
use srs_model::ARM_DOF;
use srs_model::{JointVec, Limit, take_targets};
use tokio::sync::{mpsc, oneshot};
use tracing::error;

use crate::actions::{
    ARM_WORDS, admit_limb, blocking_ask_coordinator, check_limb_free, move_seconds_from_wire,
    tolerance_from_wire,
};
use crate::coordinator::CoordinatorRequest;
use crate::failure::{Failure, FailureKind};
use crate::move_slot::MoveSlot;
use crate::planner::{ArmMoveRequest, Goal, JointReply, PlanCheck};
use crate::settle::{PoseTolerance, Settle};
use crate::streams::LeaderActivity;
use crate::types::{orientation_from_wire, position_from_wire};

/// Parse the wire pair of the `settle_position_tolerance_m` and
/// `settle_orientation_tolerance_rad` fields, each a positive finite number, or the
/// refusal naming the field.
fn settle_tolerance_from_wire(
    position_m: f64,
    orientation_rad: f64,
) -> std::result::Result<PoseTolerance, String> {
    Ok(PoseTolerance {
        position_m: tolerance_from_wire("settle_position_tolerance_m", position_m)?,
        orientation_rad: tolerance_from_wire("settle_orientation_tolerance_rad", orientation_rad)?,
    })
}

/// Parse the per-joint settle tolerances off the wire, one per joint in the
/// arm's joint order: each a positive finite number, or the refusal.
pub(super) fn joint_tolerances_from_wire(
    tolerances: JointVec,
) -> std::result::Result<JointVec, String> {
    if tolerances.iter().all(|&v| PositiveFinite::parse(v).is_ok()) {
        return Ok(tolerances);
    }
    Err("each settle_tolerances_rad entry must be a positive number".to_string())
}

/// Parse the fields `move_arm` and `check_arm_move` share into the
/// [`ArmMoveRequest`] both take, or the refusal both give.
fn arm_move_request(
    arm_name: &str,
    position: [f64; 3],
    orientation: Option<[f64; 4]>,
    duration_s: f64,
) -> std::result::Result<ArmMoveRequest, String> {
    let goal_pose_has = |reason| format!("goal pose has {reason}");
    let arm = arm_in("arm_name", arm_name)?;
    let position = position_from_wire(position).map_err(goal_pose_has)?;
    let orientation = orientation
        .map(orientation_from_wire)
        .transpose()
        .map_err(goal_pose_has)?;
    let duration = move_seconds_from_wire("duration_s", duration_s)?;
    Ok(ArmMoveRequest {
        arm,
        position,
        orientation,
        duration,
    })
}

/// A `move_arm` goal parsed off the wire: the move to plan, and the settle it
/// asks for.
struct MoveArmGoal {
    request: ArmMoveRequest,
    settle: Settle<PoseTolerance>,
}

fn move_arm_goal(d: &move_arm::GoalRequestData) -> std::result::Result<MoveArmGoal, String> {
    let request = arm_move_request(&d.arm_name, d.position, d.orientation, d.duration_s)?;
    let tolerance = settle_tolerance_from_wire(
        d.settle_position_tolerance_m,
        d.settle_orientation_tolerance_rad,
    )?;
    let timeout = move_seconds_from_wire("settle_timeout_s", d.settle_timeout_s)?;
    Ok(MoveArmGoal {
        request,
        settle: Settle { tolerance, timeout },
    })
}

/// A `move_arm_joints` goal parsed off the wire: a finite target inside the
/// arm's joint limits ([`take_targets`]), the duration, and the settle it
/// asks for.
struct JointMoveRequest {
    arm: ArmId,
    target: JointVec,
    duration: Duration,
    settle: Settle<JointVec>,
}

fn joint_move_request(
    d: &move_arm_joints::GoalRequestData,
    limits: &PerArm<[Limit; ARM_DOF], ARMS>,
) -> std::result::Result<JointMoveRequest, String> {
    let arm = arm_in("arm_name", &d.arm_name)?;
    if !d.joint_positions.iter().all(|v| v.is_finite()) {
        return Err("non-finite joint target".to_string());
    }
    let duration = move_seconds_from_wire("duration_s", d.duration_s)?;
    let target = take_targets(&limits[arm], &d.joint_positions)
        .map_err(|outside| format!("the target puts {outside}"))?;
    let settle = Settle {
        tolerance: joint_tolerances_from_wire(d.settle_tolerances_rad)?,
        timeout: move_seconds_from_wire("settle_timeout_s", d.settle_timeout_s)?,
    };
    Ok(JointMoveRequest {
        arm,
        target,
        duration,
        settle,
    })
}

/// Expose `move_arm_joints`: validate + claim, then hand the goal to the arm's
/// planner. The planner releases the move slot when the move ends.
pub async fn run_move_arm_joints(
    runner: Arc<NodeRunner>,
    goal_txs: PerArm<mpsc::Sender<Goal>, ARMS>,
    move_slots: PerArm<MoveSlot, ARMS>,
    fresh: PerArm<Arc<AtomicBool>, ARMS>,
    activity: PerArm<LeaderActivity, ARMS>,
    limits: PerArm<[Limit; ARM_DOF], ARMS>,
) -> Result<()> {
    let mut handle = move_arm_joints::ActionHandle::expose(&runner).await?;
    loop {
        let accepted = handle
            .handle_goal_next_request(|req| {
                let admitted = joint_move_request(&req.data, &limits)
                    .map_err(|reason| Failure::new(FailureKind::Rejected, reason))
                    .and_then(|request| {
                        let arm = request.arm;
                        admit_limb(&fresh[arm], &activity[arm], &move_slots[arm], &ARM_WORDS)
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
        let arm = request.arm;
        if goal_txs[arm]
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
            move_slots[arm].release();
            error!("move_arm_joints: coordinator channel closed");
            return Ok(());
        }
    }
}

/// Expose `move_arm` (Cartesian): validate + claim, then hand the goal to the
/// arm's planner, which plans IK along the path and runs it governed.
pub async fn run_move_arm(
    runner: Arc<NodeRunner>,
    goal_txs: PerArm<mpsc::Sender<Goal>, ARMS>,
    move_slots: PerArm<MoveSlot, ARMS>,
    fresh: PerArm<Arc<AtomicBool>, ARMS>,
    activity: PerArm<LeaderActivity, ARMS>,
) -> Result<()> {
    let mut handle = move_arm::ActionHandle::expose(&runner).await?;
    loop {
        let accepted = handle
            .handle_goal_next_request(|req| {
                let admitted = move_arm_goal(&req.data)
                    .map_err(|reason| Failure::new(FailureKind::Rejected, reason))
                    .and_then(|goal| {
                        let arm = goal.request.arm;
                        admit_limb(&fresh[arm], &activity[arm], &move_slots[arm], &ARM_WORDS)
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
        let arm = request.arm;
        if goal_txs[arm]
            .send(Goal::Cartesian {
                request,
                settle,
                ctx: Box::new(ctx),
            })
            .await
            .is_err()
        {
            move_slots[arm].release();
            error!("move_arm: coordinator channel closed");
            return Ok(());
        }
    }
}

/// Expose `check_arm_move`: validate its fields as `move_arm` does,
/// give an error reply where `move_arm` refuses (a value it does not take,
/// no fresh follower state, a streaming leader, the arm busy), then ask the
/// coordinator, which asks the arm's planner for a plan from the held
/// setpoint and moves nothing. The answer is whether a plan exists: the time
/// the move would take, or the words of the `move_arm` failure. A coordinator
/// that does not answer gives `attempt_failed`. Serves until `token` cancels.
pub async fn run_check_arm_move(
    runner: Arc<NodeRunner>,
    token: CancellationToken,
    requests: mpsc::Sender<CoordinatorRequest>,
    fresh: PerArm<Arc<AtomicBool>, ARMS>,
    activity: PerArm<LeaderActivity, ARMS>,
) {
    serving::serve("check_arm_move", token.cancelled(), || {
        check_arm_move::handle_requests(&runner, |req| {
            check_plan(&requests, &fresh, &activity, &req.data)
                .map_err(|f| check_arm_move::Failure::new(f.kind.into(), f.message))
        })
    })
    .await;
}

/// The answer to one plan check, or the refusal a `move_arm` goal with these
/// fields would get.
fn check_plan(
    requests: &mpsc::Sender<CoordinatorRequest>,
    fresh: &PerArm<Arc<AtomicBool>, ARMS>,
    activity: &PerArm<LeaderActivity, ARMS>,
    d: &check_arm_move::RequestData,
) -> std::result::Result<check_arm_move::Response, Failure> {
    let request = arm_move_request(&d.arm_name, d.position, d.orientation, d.duration_s)
        .map_err(|reason| Failure::new(FailureKind::Rejected, reason))?;
    let arm = request.arm;
    check_limb_free(
        fresh[arm].load(Ordering::Acquire),
        activity[arm].streaming(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use robot_description::states;
    use srs_model::TARGET_SLACK;

    use crate::robot::LIMBS;
    use crate::test_support::{LEFT, V2, postures};

    use hegel::TestCase;
    use hegel::generators as gs;

    // Per-joint tolerances are accepted exactly when every entry is a
    // positive finite value, kept in order.
    #[hegel::test]
    fn joint_tolerances_need_a_positive_value_per_joint(tc: TestCase) {
        let values: JointVec = tc.draw(gs::arrays(gs::floats()));
        let admitted = values.iter().all(|v| v.is_finite() && *v > 0.0);
        match joint_tolerances_from_wire(values) {
            Ok(parsed) => {
                assert!(admitted, "{values:?} was accepted");
                assert_eq!(parsed, values);
            }
            Err(refusal) => {
                assert!(!admitted, "{values:?} was refused");
                assert_eq!(
                    refusal,
                    "each settle_tolerances_rad entry must be a positive number"
                );
            }
        }
    }

    /// An elbow measured a hair under its lower limit, as a world saved with
    /// the arm straight can hold it, is taken at its limit; one farther under
    /// it is refused with the joint, its position and its limits.
    #[test]
    fn a_joint_target_a_hair_outside_its_limits_is_taken_at_the_limit_and_one_farther_out_refused()
    {
        let limits = PerArm::from_fn(|arm| crate::test_support::bundled_arm(V2, arm).limits());
        let elbow = limits[LEFT][3];
        let mut goal = move_arm_joints::GoalRequestData {
            arm_name: LIMBS.arm_name(LEFT).to_string(),
            joint_positions: postures(V2, states::HOME)[LEFT],
            duration_s: 1.0,
            settle_tolerances_rad: [0.01; ARM_DOF],
            settle_timeout_s: 1.0,
        };
        let farther_out = elbow.lo - 1.1 * TARGET_SLACK;
        goal.joint_positions[3] = farther_out;
        assert_eq!(
            joint_move_request(&goal, &limits).err(),
            Some(format!(
                "the target puts joint 4 at {farther_out} rad, outside its limits [{}, {}]",
                elbow.lo, elbow.hi
            ))
        );
        for position in [
            elbow.lo - TARGET_SLACK / 2.0,
            elbow.lo,
            elbow.lo - 0.9 * TARGET_SLACK,
        ] {
            goal.joint_positions[3] = position;
            let taken = joint_move_request(&goal, &limits).unwrap().target;
            assert_eq!(taken[3], elbow.lo, "{position}");
        }
        let inside = (elbow.lo + elbow.hi) / 2.0;
        goal.joint_positions[3] = inside;
        assert_eq!(
            joint_move_request(&goal, &limits).unwrap().target[3],
            inside
        );
    }

    #[test]
    fn a_settle_tolerance_names_the_axis_it_refuses() {
        assert_eq!(
            settle_tolerance_from_wire(0.0, 0.1).unwrap_err(),
            "settle_position_tolerance_m must be a positive number"
        );
        assert_eq!(
            settle_tolerance_from_wire(0.01, -1.0).unwrap_err(),
            "settle_orientation_tolerance_rad must be a positive number"
        );
        assert_eq!(
            settle_tolerance_from_wire(0.005, 0.087),
            Ok(PoseTolerance {
                position_m: 0.005,
                orientation_rad: 0.087,
            })
        );
    }
}
