//! The postures contract's moves (move_to_ready, move_to_home): claim both
//! arms, hand each planner an ordinary joint goal to the posture, and end the
//! one action goal from both terminals. Cancel flips a shared flag the moves
//! poll, so each arm stops the way a cancelled joint move stops; the stop
//! service ends both arms' moves the same way, and the goal ends cancelled
//! for the stop's reason. The two actions share the arms' single-flight
//! slots, so a posture goal arriving while the other posture runs is refused
//! with `precondition_failed`.
//!
//! A goal that completes or ends cancelled gives the grasp pose of each arm,
//! in limb_state's arm order. The coordinator measures both arms then, after
//! the shares of both arms ended. When an arm has no fresh measurement then,
//! the result gives no pose. A goal that fails gives no result.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use peppygen::exposed_actions::postures::{move_to_home, move_to_ready};
use peppygen::{NodeRunner, Result};
use srs_model::nalgebra::Isometry3;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, warn};

use crate::actions::{ARM_WORDS, ask_coordinator, check_limb_free, claim};
use crate::coordinator::{CoordinatorRequest, MeasuredGrasp};
use crate::failure::{Failure, FailureKind};
use crate::planner::{Goal, JointReply, Outcome, ReadyReply, Unmeasured};
use crate::side::{ArmPair, Side};
use crate::streams::LeaderActivity;
use crate::types::{arm_pose_arrays, limb_names, move_seconds_from_wire};

/// Claim both arms' single-flight slots, or refuse with
/// `precondition_failed`, naming the busy arm. A failure on the second claim
/// unwinds the first, so a refusal never leaves a slot held.
fn claim_both(busy: &ArmPair<Arc<AtomicBool>>) -> std::result::Result<(), Failure> {
    let busy_arm = |side: Side| {
        let message = format!("the {} {}", side.label(), ARM_WORDS.busy);
        Failure::new(FailureKind::PreconditionFailed, message)
    };
    if !claim(&busy[Side::Left]) {
        return Err(busy_arm(Side::Left));
    }
    if !claim(&busy[Side::Right]) {
        busy[Side::Left].store(false, Ordering::Release);
        return Err(busy_arm(Side::Right));
    }
    Ok(())
}

/// Admit a posture goal of `duration_s`: a duration the robot takes
/// (`rejected` otherwise), both arms free ([`check_limb_free`] over both),
/// then both arms claimed.
fn admit_posture(
    duration_s: f64,
    fresh: &ArmPair<Arc<AtomicBool>>,
    activity: &ArmPair<LeaderActivity>,
    busy: &ArmPair<Arc<AtomicBool>>,
) -> std::result::Result<(), Failure> {
    move_seconds_from_wire("duration_s", duration_s)
        .map_err(|reason| Failure::new(FailureKind::Rejected, reason))?;
    check_limb_free(
        fresh.iter().all(|fresh| fresh.load(Ordering::Acquire)),
        activity.iter().any(LeaderActivity::streaming),
        &ARM_WORDS,
    )?;
    claim_both(busy)
}

/// How a posture goal ends.
#[derive(Debug, PartialEq)]
enum PostureEnd {
    /// Both moves ran their time out. The governor can hold an arm short and
    /// an object can stop it, so it does not mean the arms arrived.
    Completed,
    /// Cancelled for this reason: by the goal's caller, or by the stop.
    Cancelled(String),
    /// A move failed, or a planner did not take or answer its share.
    Failed(Failure),
}

/// Judge the end of a posture move from what actually came back: the goals
/// dispatched, the outcomes received, and `cancel`, the reason of the goal's
/// own cancel. The goal's own cancel wins whatever the outcomes say; a share
/// the stop ended ends the goal cancelled for the stop's reason; otherwise
/// the goal completes when both arms were dispatched and both shares
/// completed, and fails with the first thing that went wrong.
fn summarize(pending: usize, outcomes: &[Outcome], cancel: Option<String>) -> PostureEnd {
    if let Some(reason) = cancel {
        return PostureEnd::Cancelled(reason);
    }
    let stop = outcomes.iter().find_map(|outcome| match outcome {
        Outcome::Cancelled(reason) => Some(reason),
        _ => None,
    });
    if let Some(reason) = stop {
        return PostureEnd::Cancelled(reason.clone());
    }
    let planner_failure = |message: &str| Failure::new(FailureKind::AttemptFailed, message);
    if pending < 2 {
        return PostureEnd::Failed(planner_failure("an arm's planner is unavailable"));
    }
    if outcomes.len() < pending {
        return PostureEnd::Failed(planner_failure("a planner dropped the move"));
    }
    let failure = outcomes.iter().find_map(|outcome| match outcome {
        Outcome::Failed(failure) => Some(failure),
        _ => None,
    });
    match failure {
        Some(failure) => PostureEnd::Failed(failure.clone()),
        None => PostureEnd::Completed,
    }
}

/// The poses a posture result gives: the grasp pose of each arm in
/// [`Side::BOTH`] order, or none.
#[derive(Debug)]
struct ArmPoses {
    arm_names: Vec<String>,
    positions: Vec<f64>,
    orientations: Vec<f64>,
}

/// What the coordinator answers when the goal asks for the grasp pose of
/// each arm, or why it did not answer.
type GraspAnswer = std::result::Result<ArmPair<MeasuredGrasp>, String>;

/// Ask the coordinator for the grasp pose of each arm, measured when it
/// serves the request.
async fn measure_grasps(requests: &mpsc::Sender<CoordinatorRequest>) -> GraspAnswer {
    let (reply, answer) = oneshot::channel();
    ask_coordinator(
        requests,
        CoordinatorRequest::MeasureGrasps { reply },
        answer,
    )
    .await
}

/// The grasp pose of each arm in `answer`, or why the result gives none:
/// - the coordinator did not answer;
/// - the seed wait is not over, so the robot has not measured every limb;
/// - the follower of an arm stopped reporting.
fn measured_poses(answer: GraspAnswer) -> std::result::Result<ArmPair<Isometry3<f64>>, String> {
    let grasps = answer?;
    let pose = |side: Side| {
        let arm = side.arm_name();
        let grasp: MeasuredGrasp = grasps[side];
        grasp.map_err(|unmeasured| match unmeasured {
            Unmeasured::NotYet => "the robot has not measured every limb yet".to_string(),
            Unmeasured::Stale => format!("{arm} stopped reporting its joints"),
        })
    };
    ArmPair::from_fn(pose).transpose()
}

/// The poses of a posture result from `answer`: every arm's grasp pose, or
/// no pose at all when an arm has no fresh measurement or the coordinator
/// does not answer. The log of `action` says why there is none.
fn arm_poses(answer: GraspAnswer, action: &str) -> ArmPoses {
    match measured_poses(answer) {
        Ok(poses) => {
            let (positions, orientations) = arm_pose_arrays(&poses);
            ArmPoses {
                arm_names: limb_names().arm_names,
                positions,
                orientations,
            }
        }
        Err(reason) => {
            warn!("{action}: the result gives no arm poses: {reason}");
            ArmPoses {
                arm_names: Vec::new(),
                positions: Vec::new(),
                orientations: Vec::new(),
            }
        }
    }
}

/// Expand one posture action's run loop: expose it, claim both arms per
/// accepted goal, and run both joint moves. Once both report, end the goal:
/// a goal that completes or ends cancelled gives the grasp poses that the
/// coordinator measures then.
/// One goal at a time; a goal arriving mid-move waits unread until this one
/// ends, then claims the freed arms. Written once here because the two
/// generated action modules carry distinct types with an identical surface.
macro_rules! posture_runner {
    ($fn_name:ident, $action:ident, $name:literal, $posture:expr) => {
        pub async fn $fn_name(
            runner: Arc<NodeRunner>,
            goal_txs: ArmPair<mpsc::Sender<Goal>>,
            busy: ArmPair<Arc<AtomicBool>>,
            fresh: ArmPair<Arc<AtomicBool>>,
            activity: ArmPair<LeaderActivity>,
            requests: mpsc::Sender<CoordinatorRequest>,
        ) -> Result<()> {
            let mut handle = $action::ActionHandle::expose(&runner).await?;
            loop {
                let accepted = handle
                    .handle_goal_next_request(|req| {
                        Ok(
                            match admit_posture(req.data.duration_s, &fresh, &activity, &busy) {
                                Ok(()) => $action::GoalDecision::accept(),
                                Err(f) => $action::GoalDecision::refuse(f.kind.into(), f.message),
                            },
                        )
                    })
                    .await?;
                let Some(ctx) = accepted else { return Ok(()) };
                let duration = move_seconds_from_wire("duration_s", ctx.request().data.duration_s)
                    .expect("validated on accept");

                let cancelled = Arc::new(AtomicBool::new(false));
                let (done_tx, mut done_rx) = mpsc::channel::<Outcome>(2);
                let mut pending = 0usize;
                for side in Side::BOTH {
                    let goal = Goal::Joint {
                        target: $posture(side.model()),
                        duration,
                        reply: JointReply::Ready(ReadyReply {
                            done_tx: done_tx.clone(),
                            cancelled: cancelled.clone(),
                        }),
                    };
                    if goal_txs[side].send(goal).await.is_err() {
                        // The planner is gone; release the claim its goal
                        // would have held.
                        busy[side].store(false, Ordering::Release);
                        error!("{}: {} goal channel closed", $name, side.label());
                    } else {
                        pending += 1;
                    }
                }
                drop(done_tx);

                let mut outcomes: Vec<Outcome> = Vec::with_capacity(pending);
                let mut cancel_seen = false;
                while outcomes.len() < pending {
                    tokio::select! {
                        _ = ctx.cancel_signal(), if !cancel_seen => {
                            cancel_seen = true;
                            cancelled.store(true, Ordering::Release);
                        }
                        received = done_rx.recv() => match received {
                            Some(outcome) => outcomes.push(outcome),
                            None => break,
                        },
                    }
                }

                let cancel = ctx.cancel_cause().map(|cause| cause.reason().to_string());
                let ended = match summarize(pending, &outcomes, cancel) {
                    PostureEnd::Failed(failure) => {
                        ctx.fail(failure.kind.into(), failure.message).await
                    }
                    PostureEnd::Completed => {
                        let poses = arm_poses(measure_grasps(&requests).await, $name);
                        ctx.complete(poses.arm_names, poses.positions, poses.orientations)
                            .await
                    }
                    PostureEnd::Cancelled(reason) => {
                        let poses = arm_poses(measure_grasps(&requests).await, $name);
                        ctx.complete_cancelled(
                            reason,
                            poses.arm_names,
                            poses.positions,
                            poses.orientations,
                        )
                        .await
                    }
                };
                if let Err(e) = ended {
                    error!("{}: end: {e}", $name);
                }
            }
        }
    };
}

posture_runner!(
    run_move_to_ready,
    move_to_ready,
    "move_to_ready",
    openarm_description::ready
);
posture_runner!(
    run_move_to_home,
    move_to_home,
    "move_to_home",
    openarm_description::home
);

#[cfg(test)]
mod tests {
    use super::*;

    use srs_model::nalgebra::Vector3;

    #[test]
    fn claiming_both_arms_names_the_busy_arm_and_holds_no_slot_it_refused() {
        let slots = |left, right| {
            ArmPair::new(
                Arc::new(AtomicBool::new(left)),
                Arc::new(AtomicBool::new(right)),
            )
        };
        let held = |busy: &ArmPair<Arc<AtomicBool>>| {
            ArmPair::from_fn(|side| busy[side].load(Ordering::Acquire))
        };
        let refused = |message: &str| Err(Failure::new(FailureKind::PreconditionFailed, message));

        let free = slots(false, false);
        assert_eq!(claim_both(&free), Ok(()));
        assert_eq!(held(&free), ArmPair::new(true, true));

        let left_busy = slots(true, false);
        assert_eq!(
            claim_both(&left_busy),
            refused("the left arm is already executing a motion")
        );
        assert_eq!(held(&left_busy), ArmPair::new(true, false));

        let right_busy = slots(false, true);
        assert_eq!(
            claim_both(&right_busy),
            refused("the right arm is already executing a motion")
        );
        assert_eq!(held(&right_busy), ArmPair::new(false, true));
    }

    #[test]
    fn both_postures_sit_inside_both_generations_joint_limits() {
        // The planner sends postures unclamped, so an out-of-limit posture
        // would reach the arms; this pin, against the same floored model the
        // planner clamps with, is what stands in for a clamp. The description
        // pins the same against its own floored joint_limits; this covers the
        // srs_model arm actually used.
        use openarm_description::{HardwareVersion, JointPosture, home, ready};
        for version in [HardwareVersion::V1, HardwareVersion::V2] {
            for side in Side::BOTH {
                let model = crate::arm_model(version, side.model())
                    .expect("build arm from the bundled URDF");
                let limits = model.limits();
                let postures = [
                    (
                        "ready",
                        ready as fn(openarm_description::Side) -> JointPosture,
                    ),
                    ("home", home),
                ];
                for (name, posture) in postures {
                    let q_all = posture(side.model());
                    for (j, (&q, limit)) in q_all.iter().zip(&limits).enumerate() {
                        assert!(
                            q >= limit.lo && q <= limit.hi,
                            "{version:?} {} {name} j{}: {q} outside [{}, {}]",
                            side.label(),
                            j + 1,
                            limit.lo,
                            limit.hi
                        );
                    }
                }
            }
        }
    }

    /// A grasp pose that tells the arms apart. y is 0.2 m on the left and
    /// -0.2 m on the right. Each turns about z by its own y in radians.
    fn grasp_of(side: Side) -> Isometry3<f64> {
        let y = match side {
            Side::Left => 0.2,
            Side::Right => -0.2,
        };
        Isometry3::new(Vector3::new(0.3, y, 0.25), Vector3::new(0.0, 0.0, y))
    }

    /// The coordinator's answer with each arm measured at [`grasp_of`].
    fn both_measured() -> GraspAnswer {
        Ok(ArmPair::from_fn(|side| Ok(grasp_of(side))))
    }

    /// An arm's share that failed with `kind` and `message`.
    fn failed(kind: FailureKind, message: &str) -> Outcome {
        Outcome::Failed(Failure::new(kind, message))
    }

    /// The shares of both arms, ended by the stop service.
    fn stopped_outcomes() -> [Outcome; 2] {
        [
            Outcome::Cancelled("stopped: operator".to_string()),
            Outcome::Cancelled("stopped: operator".to_string()),
        ]
    }

    const CALLER: &str = "cancelled by its caller";

    #[test]
    fn both_complete_shares_complete_the_goal() {
        let outcomes = [Outcome::Complete, Outcome::Complete];
        assert_eq!(summarize(2, &outcomes, None), PostureEnd::Completed);
    }

    #[test]
    fn a_failed_arm_fails_the_goal_with_its_failure() {
        let stale = Failure::new(
            FailureKind::ResourceUnavailable,
            "right: the follower stopped reporting",
        );
        let outcomes = [Outcome::Complete, Outcome::Failed(stale.clone())];
        assert_eq!(summarize(2, &outcomes, None), PostureEnd::Failed(stale));
    }

    #[test]
    fn a_dropped_planner_reply_fails_with_attempt_failed() {
        let outcomes = [Outcome::Complete];
        assert_eq!(
            summarize(2, &outcomes, None),
            PostureEnd::Failed(Failure::new(
                FailureKind::AttemptFailed,
                "a planner dropped the move"
            ))
        );
    }

    #[test]
    fn no_dispatched_goal_is_a_planner_failure() {
        let unavailable = PostureEnd::Failed(Failure::new(
            FailureKind::AttemptFailed,
            "an arm's planner is unavailable",
        ));
        assert_eq!(summarize(0, &[], None), unavailable);
        assert_eq!(summarize(1, &[Outcome::Complete], None), unavailable);
    }

    #[test]
    fn two_failures_report_the_first_received() {
        let outcomes = [
            failed(FailureKind::AttemptFailed, "left: IK failed mid-trajectory"),
            failed(
                FailureKind::ResourceUnavailable,
                "right: the follower stopped reporting",
            ),
        ];
        assert_eq!(
            summarize(2, &outcomes, None),
            PostureEnd::Failed(Failure::new(
                FailureKind::AttemptFailed,
                "left: IK failed mid-trajectory"
            ))
        );
    }

    #[test]
    fn a_cancel_with_nothing_pending_still_ends_cancelled() {
        assert_eq!(
            summarize(0, &[], Some(CALLER.to_string())),
            PostureEnd::Cancelled(CALLER.to_string())
        );
    }

    /// The stop service ended both arms' shares. The goal ends cancelled
    /// for the stop's reason.
    #[test]
    fn a_stopped_arm_ends_the_goal_cancelled_for_the_stops_reason() {
        assert_eq!(
            summarize(2, &stopped_outcomes(), None),
            PostureEnd::Cancelled("stopped: operator".to_string())
        );
    }

    /// The goal's own cancel wins over what its shares say: both completed,
    /// one failed, or the stop ended them.
    #[test]
    fn the_goals_own_cancel_wins_over_every_share() {
        let completed = [Outcome::Complete, Outcome::Complete];
        let one_failed = [
            Outcome::Complete,
            failed(
                FailureKind::AttemptFailed,
                "right: IK failed mid-trajectory",
            ),
        ];
        for outcomes in [&completed[..], &one_failed[..], &stopped_outcomes()[..]] {
            assert_eq!(
                summarize(2, outcomes, Some(CALLER.to_string())),
                PostureEnd::Cancelled(CALLER.to_string())
            );
        }
    }

    /// The result gives the arms in limb_state's order, each with its own
    /// pose. Each orientation is `[x, y, z, w]`.
    #[test]
    fn the_posture_poses_come_in_arm_name_order() {
        let poses = arm_poses(both_measured(), "move_to_ready");
        let (positions, orientations) = arm_pose_arrays(&ArmPair::from_fn(grasp_of));
        assert_eq!(poses.arm_names, ["left_arm", "right_arm"]);
        assert_eq!(poses.positions, positions);
        assert_eq!(poses.orientations, orientations);
        let (sin, cos) = 0.1f64.sin_cos();
        let expected = [0.0, 0.0, sin, cos, 0.0, 0.0, -sin, cos];
        for (i, (got, want)) in poses.orientations.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "orientations[{i}] = {got}, expected {want}"
            );
        }
    }

    /// When an arm has no fresh measurement, or the coordinator does not
    /// answer, the result gives no pose for any arm:
    /// - the seed wait is not over, so neither arm has a pose;
    /// - the follower of the right arm stopped reporting;
    /// - the coordinator is not running.
    #[test]
    fn an_arm_without_a_measured_pose_empties_the_posture_arrays() {
        let unmeasured = [
            (
                Ok(ArmPair::new(
                    Err(Unmeasured::NotYet),
                    Err(Unmeasured::NotYet),
                )),
                "the robot has not measured every limb yet",
            ),
            (
                Ok(ArmPair::new(
                    Ok(grasp_of(Side::Left)),
                    Err(Unmeasured::Stale),
                )),
                "right_arm stopped reporting its joints",
            ),
            (
                Err("the coordinator is not running".to_string()),
                "the coordinator is not running",
            ),
        ];
        for (answer, reason) in unmeasured {
            assert_eq!(measured_poses(answer.clone()), Err(reason.to_string()));
            let poses = arm_poses(answer, "move_to_ready");
            assert!(poses.arm_names.is_empty(), "{:?}", poses.arm_names);
            assert!(poses.positions.is_empty() && poses.orientations.is_empty());
        }
    }
}
