//! Arm move-action admission: the `move_arm_joints` and `move_arm` handlers the
//! backbone exposes to the commander, and the `check_arm_move` service that
//! asks whether a Cartesian goal has a plan without moving. Each validates
//! the goal (arm_name, finiteness, duration, the settle it asks for, and
//! joint limits for joint moves), refusing a value it does not take with
//! `rejected`; a Cartesian goal or check then has its target placed in the
//! robot frame by the robot's tree ([`Placer`]), refused with the failure of
//! the lookup when the tree cannot place it. Each refuses it while the arm's
//! follower delivers no fresh state or a leader streams the arm
//! ([`admit_limb`]); a move claims the target arm's move slot, then hands
//! the accepted goal, its target as placed, to that arm's planner over its
//! goal channel. The planner runs the motion - governed against the other
//! arm - settles it, ends the goal, and releases the move slot at the
//! terminal. The check claims nothing: the coordinator asks the planner and
//! answers whether a plan exists, or the refusal a goal would get, as an
//! error reply.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use crate::robot::{ARMS, arm_in};
use control_core::asking::ask_loop;
use control_core::geometry::{PoseTarget, Stamped};
use control_core::move_seconds::MoveSeconds;
use control_core::move_slot::MoveSlot;
use control_core::positive_finite::PositiveFinite;
use control_core::robot_tree::{Placer, Unplaced};
use control_core::serving;
use control_core::sync::{lock_ignoring_poison, wait_for};
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
    ARM_WORDS, COORDINATOR, COORDINATOR_REPLY_TIMEOUT, admit_limb, check_limb_free,
    contact_gap_from_wire,
};
use crate::coordinator::CoordinatorRequest;
use crate::failure::{Failure, FailureKind, TargetFailure};
use crate::planner::{ArmMoveRequest, Goal, JointReply, PlanCheck};
use crate::settle::{ContactGap, PoseTolerance, Settle};
use crate::streams::LeaderActivity;

/// Parse the wire pair of the `settle_position_tolerance_m` and
/// `settle_orientation_tolerance_rad` fields, each a positive finite number, or the
/// refusal naming the field.
fn settle_tolerance_from_wire(
    position_m: f64,
    orientation_rad: f64,
) -> std::result::Result<PoseTolerance, String> {
    Ok(PoseTolerance {
        position_m: PositiveFinite::parse_field("settle_position_tolerance_m", position_m)?.get(),
        orientation_rad: PositiveFinite::parse_field(
            "settle_orientation_tolerance_rad",
            orientation_rad,
        )?
        .get(),
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

/// The target of a `move_arm` goal or a `check_arm_move` request, field for
/// field: each member generates a type of its own.
struct TargetFields<'a> {
    frame: &'a str,
    timestamp: Option<SystemTime>,
    position: [f64; 3],
    orientation: Option<[f64; 4]>,
}

/// A refusal of a value the backbone does not take.
fn rejected(reason: String) -> Failure {
    Failure::new(FailureKind::Rejected, reason)
}

/// The fields `move_arm` and `check_arm_move` share, parsed off the wire:
/// the arm, the target in its frame at its timestamp, and the duration.
struct ArmMoveFields {
    arm: ArmId,
    target: Stamped<PoseTarget>,
    duration: Duration,
}

/// Parse the fields `move_arm` and `check_arm_move` share, or the refusal
/// both give with `rejected` for a value the backbone does not take.
fn arm_move_fields(
    arm_name: &str,
    target: TargetFields<'_>,
    duration_s: f64,
) -> std::result::Result<ArmMoveFields, Failure> {
    let arm = arm_in("arm_name", arm_name).map_err(rejected)?;
    let target = PoseTarget::parse(target.position, target.orientation)
        .and_then(|value| Stamped::parse(target.frame, target.timestamp, value))
        .map_err(|e| rejected(format!("the target: {e}")))?;
    let duration = MoveSeconds::from_wire("duration_s", duration_s)
        .map_err(rejected)?
        .duration();
    Ok(ArmMoveFields {
        arm,
        target,
        duration,
    })
}

/// The [`ArmMoveRequest`] of `fields`, its target placed in the robot frame
/// by `placer`, or the failure of the lookup.
fn placed_request(
    placer: &Placer,
    fields: ArmMoveFields,
) -> std::result::Result<ArmMoveRequest, Unplaced> {
    let target = placer
        .place_now(&fields.target)
        .map_err(|e| Unplaced::of("the target", &e))?
        .value;
    Ok(ArmMoveRequest {
        arm: fields.arm,
        position: target.position().into(),
        orientation: target.orientation().map(Into::into),
        duration: fields.duration,
    })
}

/// A `move_arm` goal parsed off the wire: the move to plan, its target
/// placed in the robot frame, the settle it asks for, and its contact gap.
struct MoveArmGoal {
    request: ArmMoveRequest,
    settle: Settle<PoseTolerance>,
    contact: ContactGap,
}

/// Parse a `move_arm` goal, every value first, then place its target, or
/// the refusal: `rejected` for a value the backbone does not take, then the
/// failure of the lookup.
fn move_arm_goal(
    placer: &Placer,
    d: &move_arm::GoalRequestData,
) -> std::result::Result<MoveArmGoal, TargetFailure> {
    let target = TargetFields {
        frame: placer.frame_of(d.target.frame.as_deref()),
        timestamp: d.target.timestamp,
        position: d.target.position,
        orientation: d.target.orientation,
    };
    let fields = arm_move_fields(&d.arm_name, target, d.duration_s)?;
    let tolerance = settle_tolerance_from_wire(
        d.settle_position_tolerance_m,
        d.settle_orientation_tolerance_rad,
    )
    .map_err(rejected)?;
    let timeout = MoveSeconds::from_wire("settle_timeout_s", d.settle_timeout_s)
        .map_err(rejected)?
        .duration();
    let contact = contact_gap_from_wire(d.contact_gap_m).map_err(rejected)?;
    Ok(MoveArmGoal {
        request: placed_request(placer, fields)?,
        settle: Settle { tolerance, timeout },
        contact,
    })
}

/// A `move_arm_joints` goal parsed off the wire: a finite target inside the
/// arm's joint limits ([`take_targets`]), the duration, the settle it asks
/// for, and its contact gap.
struct JointMoveRequest {
    arm: ArmId,
    target: JointVec,
    duration: Duration,
    settle: Settle<JointVec>,
    contact: ContactGap,
}

fn joint_move_request(
    d: &move_arm_joints::GoalRequestData,
    limits: &PerArm<[Limit; ARM_DOF], ARMS>,
) -> std::result::Result<JointMoveRequest, String> {
    let arm = arm_in("arm_name", &d.arm_name)?;
    if !d.joint_positions.iter().all(|v| v.is_finite()) {
        return Err("non-finite joint target".to_string());
    }
    let duration = MoveSeconds::from_wire("duration_s", d.duration_s)?.duration();
    let target = take_targets(&limits[arm], &d.joint_positions)
        .map_err(|outside| format!("the target puts {outside}"))?;
    let settle = Settle {
        tolerance: joint_tolerances_from_wire(d.settle_tolerances_rad)?,
        timeout: MoveSeconds::from_wire("settle_timeout_s", d.settle_timeout_s)?.duration(),
    };
    Ok(JointMoveRequest {
        arm,
        target,
        duration,
        settle,
        contact: contact_gap_from_wire(d.contact_gap_m)?,
    })
}

/// Expose `move_arm_joints`: validate + claim, then hand the goal to the arm's
/// planner with the release of its claim. The planner releases the move slot
/// when the move ends.
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
        let accepted_goal = Mutex::new(None);
        let accepted = handle
            .handle_goal_next_request(|req| {
                let admitted = joint_move_request(&req.data, &limits)
                    .map_err(|reason| Failure::new(FailureKind::Rejected, reason))
                    .and_then(|request| {
                        let arm = request.arm;
                        let move_slot =
                            admit_limb(&fresh[arm], &activity[arm], &move_slots[arm], &ARM_WORDS)?;
                        Ok((request, move_slot))
                    });
                Ok(match admitted {
                    Ok(goal) => {
                        *lock_ignoring_poison(&accepted_goal) = Some(goal);
                        move_arm_joints::GoalDecision::accept()
                    }
                    Err(f) => move_arm_joints::GoalDecision::refuse(f.kind.into(), f.message),
                })
            })
            .await?;
        let Some(ctx) = accepted else { return Ok(()) };
        let (request, move_slot) = lock_ignoring_poison(&accepted_goal)
            .take()
            .expect("the decider keeps the goal it accepts");
        let goal = Goal::Joint {
            target: request.target,
            duration: request.duration,
            reply: JointReply::MoveArmJoints {
                ctx: Box::new(ctx),
                settle: request.settle,
                contact: request.contact,
            },
            move_slot,
        };
        if goal_txs[request.arm].send(goal).await.is_err() {
            error!("move_arm_joints: coordinator channel closed");
            return Ok(());
        }
    }
}

/// Expose `move_arm` (Cartesian): validate, place its target + claim, then
/// hand the goal to the arm's planner, which plans IK along the path and runs
/// it governed. The goal carries its target as the decider placed it, the
/// tree placing a target once, when the backbone accepts its goal, and the
/// release of its claim.
pub async fn run_move_arm(
    runner: Arc<NodeRunner>,
    placer: Arc<Placer>,
    goal_txs: PerArm<mpsc::Sender<Goal>, ARMS>,
    move_slots: PerArm<MoveSlot, ARMS>,
    fresh: PerArm<Arc<AtomicBool>, ARMS>,
    activity: PerArm<LeaderActivity, ARMS>,
) -> Result<()> {
    let mut handle = move_arm::ActionHandle::expose(&runner).await?;
    loop {
        let accepted_goal = Mutex::new(None);
        let accepted = handle
            .handle_goal_next_request(|req| {
                let admitted = move_arm_goal(&placer, &req.data).and_then(|goal| {
                    let arm = goal.request.arm;
                    let move_slot =
                        admit_limb(&fresh[arm], &activity[arm], &move_slots[arm], &ARM_WORDS)?;
                    Ok((goal, move_slot))
                });
                Ok(match admitted {
                    Ok(goal) => {
                        *lock_ignoring_poison(&accepted_goal) = Some(goal);
                        move_arm::GoalDecision::accept()
                    }
                    Err(f) => {
                        let (kind, message) = f.for_member(move_arm::FailureKind::parse);
                        move_arm::GoalDecision::refuse(kind, message)
                    }
                })
            })
            .await?;
        let Some(ctx) = accepted else { return Ok(()) };
        let (
            MoveArmGoal {
                request,
                settle,
                contact,
            },
            move_slot,
        ) = lock_ignoring_poison(&accepted_goal)
            .take()
            .expect("the decider keeps the goal it accepts");
        let goal = Goal::Cartesian {
            request,
            settle,
            contact,
            ctx: Box::new(ctx),
            move_slot,
        };
        if goal_txs[request.arm].send(goal).await.is_err() {
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
    placer: Arc<Placer>,
    requests: mpsc::Sender<CoordinatorRequest>,
    fresh: PerArm<Arc<AtomicBool>, ARMS>,
    activity: PerArm<LeaderActivity, ARMS>,
) {
    serving::serve("check_arm_move", token.cancelled(), || {
        check_arm_move::handle_requests(&runner, |req| {
            check_plan(&placer, &requests, &fresh, &activity, &req.data).map_err(|f| {
                let (kind, message) = f.for_member(check_arm_move::FailureKind::parse);
                check_arm_move::Failure::new(kind, message)
            })
        })
    })
    .await;
}

/// The answer to one plan check, or the refusal a `move_arm` goal with these
/// fields would get.
fn check_plan(
    placer: &Placer,
    requests: &mpsc::Sender<CoordinatorRequest>,
    fresh: &PerArm<Arc<AtomicBool>, ARMS>,
    activity: &PerArm<LeaderActivity, ARMS>,
    d: &check_arm_move::RequestData,
) -> std::result::Result<check_arm_move::Response, TargetFailure> {
    let target = TargetFields {
        frame: placer.frame_of(d.target.frame.as_deref()),
        timestamp: d.target.timestamp,
        position: d.target.position,
        orientation: d.target.orientation,
    };
    let request = placed_request(placer, arm_move_fields(&d.arm_name, target, d.duration_s)?)?;
    let arm = request.arm;
    check_limb_free(
        fresh[arm].load(Ordering::Acquire),
        activity[arm].streaming(),
        &ARM_WORDS,
    )?;
    let (reply, answer) = oneshot::channel();
    let checked = wait_for(ask_loop(
        COORDINATOR,
        requests,
        CoordinatorRequest::CheckArmMove { request, reply },
        answer,
        COORDINATOR_REPLY_TIMEOUT,
    ))
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
    use control_core::geometry::GeometryError;
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
            contact_gap_m: 0.0,
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

    /// A target in the robot frame at the latest time, as a caller gives one.
    fn target(position: [f64; 3], orientation: Option<[f64; 4]>) -> TargetFields<'static> {
        TargetFields {
            frame: "base_link",
            timestamp: None,
            position,
            orientation,
        }
    }

    // A target is parsed before the tree places it: a unit quaternion,
    // normalized, or none; one whose squared norm lies 0.01 or more from 1, or
    // with no frame, is refused with `rejected`, the message naming the target
    // and what to send.
    #[test]
    fn a_target_no_contract_takes_is_refused_before_it_is_placed() {
        let fields = arm_move_fields("left_arm", target([0.3, 0.1, 0.2], None), 1.0)
            .expect("a position alone");
        assert_eq!(fields.target.value().orientation(), None);
        let typed = arm_move_fields(
            "left_arm",
            target([0.3, 0.1, 0.2], Some([0.0, 0.0, 0.0, 1.004])),
            1.0,
        )
        .expect("a quaternion within tf2's tolerance");
        let [_, _, _, w] = typed
            .target
            .value()
            .orientation()
            .expect("one")
            .quaternion();
        assert!((w - 1.0).abs() < 1e-15, "normalized: {w}");
        let off_unit = arm_move_fields(
            "left_arm",
            target([0.3, 0.1, 0.2], Some([0.0, 0.0, 0.0, 1.02])),
            1.0,
        )
        .err()
        .expect("a target no contract takes");
        assert_eq!(off_unit.kind, FailureKind::Rejected);
        assert!(
            off_unit
                .message
                .starts_with("the target: the orientation [0, 0, 0, 1.02] has a squared norm"),
            "{}",
            off_unit.message
        );
        let nowhere = TargetFields {
            frame: "",
            ..target([0.3, 0.1, 0.2], None)
        };
        let refused = arm_move_fields("left_arm", nowhere, 1.0)
            .err()
            .expect("a target with no frame");
        assert_eq!(
            refused,
            Failure::new(
                FailureKind::Rejected,
                format!("the target: {}", GeometryError::NoFrame)
            )
        );
    }

    #[test]
    fn a_settle_tolerance_names_the_axis_it_refuses() {
        assert_eq!(
            settle_tolerance_from_wire(0.0, 0.1).unwrap_err(),
            "settle_position_tolerance_m must be a positive finite number, got 0"
        );
        assert_eq!(
            settle_tolerance_from_wire(0.01, -1.0).unwrap_err(),
            "settle_orientation_tolerance_rad must be a positive finite number, got -1"
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
