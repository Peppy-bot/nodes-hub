//! The postures contract's moves (move_to_ready, move_to_home): claim both
//! arms, hand each planner an ordinary joint goal to the posture, and
//! complete the one action goal from both terminals. Cancel flips a shared
//! flag the moves poll, so each arm stops the way a cancelled joint move
//! stops; the stop service ends both arms' moves the same way, and the goal
//! ends as cancelled with the stop's message. The two actions share the
//! arms' single-flight slots, so a posture goal arriving while the other
//! posture runs is rejected busy.
//!
//! Whatever the terminal, the result gives the grasp pose of each arm, in
//! limb_state's arm order. The coordinator measures both arms when the
//! posture goal completes, after the shares of both arms ended. When an arm
//! has no fresh measurement then, the result gives no pose, and its message
//! says why.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use peppygen::exposed_actions::postures::{move_to_home, move_to_ready};
use peppygen::{NodeRunner, Result};
use srs_model::nalgebra::Isometry3;
use tokio::sync::{mpsc, oneshot};
use tracing::error;

use crate::actions::{ask_coordinator, claim};
use crate::arm_pair::ArmPair;
use crate::coordinator::{CoordinatorRequest, MeasuredGrasp};
use crate::planner::{Goal, JointReply, ReadyOutcome, ReadyReply, Unmeasured};
use crate::types::{Side, arm_pose_arrays, limb_names};

/// Claim both arms' single-flight slots, or name the busy arm. A failure on
/// the second claim unwinds the first, so a refusal never leaves a slot held.
fn claim_both(busy: &[Arc<AtomicBool>; 2]) -> std::result::Result<(), &'static str> {
    if !claim(&busy[Side::Left.index()]) {
        return Err("the left arm is already executing a motion");
    }
    if !claim(&busy[Side::Right.index()]) {
        busy[Side::Left.index()].store(false, Ordering::Release);
        return Err("the right arm is already executing a motion");
    }
    Ok(())
}

/// How the whole-robot goal completes.
#[derive(Debug, PartialEq, Eq)]
enum Terminal {
    Success,
    Failed,
    Cancelled,
}

/// Judge the end of a posture move from what actually came back: the goals
/// dispatched, the outcomes received, and whether a cancel was seen. Cancel
/// wins whatever the outcomes say; a stop ends the goal as cancelled with
/// the stop's message; otherwise success requires both arms dispatched and
/// both outcomes successful (reported as `done`), and the message names the
/// first thing that went wrong.
fn summarize(
    pending: usize,
    outcomes: &[ReadyOutcome],
    cancelled: bool,
    done: &str,
) -> (Terminal, String) {
    if cancelled {
        return (Terminal::Cancelled, "goal cancelled".to_string());
    }
    if let Some(stopped) = outcomes.iter().find(|o| o.stopped) {
        return (Terminal::Cancelled, stopped.message.clone());
    }
    if pending < 2 {
        return (
            Terminal::Failed,
            "an arm's planner is unavailable".to_string(),
        );
    }
    if outcomes.len() < pending {
        return (Terminal::Failed, "a planner dropped the move".to_string());
    }
    match outcomes.iter().find(|o| !o.success) {
        Some(failed) => (Terminal::Failed, failed.message.clone()),
        None => (Terminal::Success, done.to_string()),
    }
}

/// What a posture goal completes with.
#[derive(Debug)]
struct PostureResult {
    terminal: Terminal,
    message: String,
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
/// - an arm has not measured its joints;
/// - the follower of an arm stopped reporting.
fn measured_poses(answer: GraspAnswer) -> std::result::Result<ArmPair<Isometry3<f64>>, String> {
    let grasps = answer?;
    let pose = |side: Side| {
        let arm = side.arm_name();
        let grasp: MeasuredGrasp = *grasps.get(side);
        grasp.map_err(|unmeasured| match unmeasured {
            Unmeasured::NotYet => format!("{arm} has not measured its joints"),
            Unmeasured::Stale => format!("{arm} stopped reporting its joints"),
        })
    };
    Ok(ArmPair::new(pose(Side::Left)?, pose(Side::Right)?))
}

/// The result of a posture goal:
/// - the terminal and the message that [`summarize`] gives;
/// - the grasp pose of every arm in `answer`, in [`Side::ARM_NAMES`] order.
///
/// The result gives the poses whatever the terminal. When an arm has no
/// grasp pose, the three arrays are empty and the message says why.
fn posture_result(
    pending: usize,
    outcomes: &[ReadyOutcome],
    cancelled: bool,
    done: &str,
    answer: GraspAnswer,
) -> PostureResult {
    let (terminal, message) = summarize(pending, outcomes, cancelled, done);
    match measured_poses(answer) {
        Ok(poses) => {
            let (positions, orientations) = arm_pose_arrays(&poses);
            PostureResult {
                terminal,
                message,
                arm_names: limb_names().arm_names,
                positions,
                orientations,
            }
        }
        Err(reason) => PostureResult {
            terminal,
            message: format!("{message}; no arm poses: {reason}"),
            arm_names: Vec::new(),
            positions: Vec::new(),
            orientations: Vec::new(),
        },
    }
}

/// Ceiling on a requested posture duration: the move claims both arms'
/// single-flight slots for its whole run, so an absurd request must not pin
/// the robot until a manual cancel.
const MAX_REQUESTED_DURATION_S: f64 = 600.0;

/// Whether a posture goal of `duration_s` is admitted: a duration from 0 s
/// to [`MAX_REQUESTED_DURATION_S`], both included. A NaN or an infinite
/// duration is refused.
fn admissible_duration(duration_s: f64) -> bool {
    duration_s.is_finite() && (0.0..=MAX_REQUESTED_DURATION_S).contains(&duration_s)
}

/// The message of a move_to_ready that succeeds. Success says that both
/// arms' moves ran their time out, not that the arms arrived. The governor
/// can hold an arm short, and an object can stop it.
const READY_DONE: &str = "the move to ready ran its time";

/// The message of a move_to_home that succeeds, worded as [`READY_DONE`].
const HOME_DONE: &str = "the move to home ran its time";

/// Expand one posture action's run loop: expose it, claim both arms per
/// accepted goal, and run both joint moves. Once both report, complete the
/// goal with the grasp poses that the coordinator measures then.
/// One goal at a time; a goal arriving mid-move waits unread until this one
/// completes, then claims the freed arms. Written once here because the two
/// generated action modules carry distinct types with an identical surface.
macro_rules! posture_runner {
    ($fn_name:ident, $action:ident, $name:literal, $posture:expr, $done:expr) => {
        pub async fn $fn_name(
            runner: Arc<NodeRunner>,
            goal_txs: [mpsc::Sender<Goal>; 2],
            busy: [Arc<AtomicBool>; 2],
            requests: mpsc::Sender<CoordinatorRequest>,
        ) -> Result<()> {
            let mut handle = $action::ActionHandle::expose(&runner).await?;
            loop {
                let accepted = handle
                    .handle_goal_next_request(|req| {
                        if !admissible_duration(req.data.duration_s) {
                            return Ok($action::GoalDecision::reject("invalid duration"));
                        }
                        if let Err(reason) = claim_both(&busy) {
                            return Ok($action::GoalDecision::reject(reason));
                        }
                        Ok($action::GoalDecision::accept())
                    })
                    .await?;
                let Some(ctx) = accepted else { return Ok(()) };
                let duration_s = ctx.request().data.duration_s;

                let cancelled = Arc::new(AtomicBool::new(false));
                let (done_tx, mut done_rx) = mpsc::channel::<ReadyOutcome>(2);
                let mut pending = 0usize;
                for side in [Side::Left, Side::Right] {
                    let idx = side.index();
                    let goal = Goal::Joint {
                        target: $posture(side.model()),
                        duration_s,
                        reply: JointReply::Ready(ReadyReply {
                            done_tx: done_tx.clone(),
                            cancelled: cancelled.clone(),
                        }),
                    };
                    if goal_txs[idx].send(goal).await.is_err() {
                        // The planner is gone; release the claim its goal
                        // would have held.
                        busy[idx].store(false, Ordering::Release);
                        error!("{}: {} goal channel closed", $name, side.label());
                    } else {
                        pending += 1;
                    }
                }
                drop(done_tx);

                let mut outcomes: Vec<ReadyOutcome> = Vec::with_capacity(pending);
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

                let grasps = measure_grasps(&requests).await;
                let PostureResult {
                    terminal,
                    message,
                    arm_names,
                    positions,
                    orientations,
                } = posture_result(
                    pending,
                    &outcomes,
                    cancel_seen || ctx.is_cancelled(),
                    $done,
                    grasps,
                );
                let result = match terminal {
                    Terminal::Success | Terminal::Failed => {
                        let success = terminal == Terminal::Success;
                        ctx.complete(success, message, arm_names, positions, orientations)
                            .await
                    }
                    Terminal::Cancelled => {
                        ctx.complete_cancelled(false, message, arm_names, positions, orientations)
                            .await
                    }
                };
                if let Err(e) = result {
                    error!("{}: complete: {e}", $name);
                }
            }
        }
    };
}

posture_runner!(
    run_move_to_ready,
    move_to_ready,
    "move_to_ready",
    openarm_description::ready,
    READY_DONE
);
posture_runner!(
    run_move_to_home,
    move_to_home,
    "move_to_home",
    openarm_description::home,
    HOME_DONE
);

#[cfg(test)]
mod tests {
    use super::*;

    use srs_model::nalgebra::Vector3;

    #[test]
    fn both_postures_sit_inside_both_generations_joint_limits() {
        // The planner sends postures unclamped, so an out-of-limit posture
        // would reach the arms; this pin, against the same floored model the
        // planner clamps with, is what stands in for a clamp. The description
        // pins the same against its own floored joint_limits; this covers the
        // srs_model arm actually used.
        use openarm_description::{HardwareVersion, JointPosture, home, ready};
        for version in [HardwareVersion::V1, HardwareVersion::V2] {
            for side in [Side::Left, Side::Right] {
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

    /// The duration ceiling admits 0 s and 600 s, and refuses a duration
    /// just past either edge, a NaN and an infinite one.
    #[test]
    fn a_posture_duration_is_admitted_from_zero_to_six_hundred_seconds() {
        for admitted in [0.0, 1.0, 600.0] {
            assert!(admissible_duration(admitted), "{admitted} s is refused");
        }
        for refused in [-1e-6, 600.000001, f64::NAN, f64::INFINITY] {
            assert!(!admissible_duration(refused), "{refused} s is admitted");
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
        Ok(ArmPair::new(
            Ok(grasp_of(Side::Left)),
            Ok(grasp_of(Side::Right)),
        ))
    }

    /// An arm's share that ended with `success` and `message`.
    fn outcome(success: bool, message: &str) -> ReadyOutcome {
        ReadyOutcome {
            success,
            message: message.to_string(),
            stopped: false,
        }
    }

    /// The shares of both arms, ended by the stop service.
    fn stopped_outcomes() -> [ReadyOutcome; 2] {
        [
            ReadyOutcome {
                stopped: true,
                ..outcome(false, "left: stopped: operator")
            },
            ReadyOutcome {
                stopped: true,
                ..outcome(false, "right: stopped: operator")
            },
        ]
    }

    /// Assert that `result` gives each arm at [`grasp_of`], in arm-name
    /// order.
    fn assert_both_poses(result: &PostureResult) {
        let (positions, orientations) =
            arm_pose_arrays(&ArmPair::new(grasp_of(Side::Left), grasp_of(Side::Right)));
        assert_eq!(result.arm_names, ["left_arm", "right_arm"]);
        assert_eq!(result.positions, positions);
        assert_eq!(result.orientations, orientations);
    }

    /// Assert that `result` gives no arm pose.
    fn assert_no_poses(result: &PostureResult) {
        assert!(result.arm_names.is_empty(), "{:?}", result.arm_names);
        assert!(result.positions.is_empty() && result.orientations.is_empty());
    }

    #[test]
    fn both_successful_outcomes_complete_successfully() {
        let outcomes = [
            outcome(true, "trajectory complete"),
            outcome(true, "trajectory complete"),
        ];
        let (terminal, message) = summarize(2, &outcomes, false, READY_DONE);
        assert_eq!(terminal, Terminal::Success);
        assert_eq!(message, READY_DONE);
    }

    #[test]
    fn a_failed_arm_fails_the_goal_with_its_message() {
        let outcomes = [
            outcome(true, "trajectory complete"),
            outcome(false, "goal cancelled"),
        ];
        let (terminal, message) = summarize(2, &outcomes, false, HOME_DONE);
        assert_eq!(terminal, Terminal::Failed);
        assert_eq!(message, "goal cancelled");
    }

    #[test]
    fn a_dropped_planner_reply_fails_with_a_matching_message() {
        // done_rx closed after one success: success and message must derive
        // from the same predicate, so this cannot read READY_DONE.
        let outcomes = [outcome(true, "trajectory complete")];
        let (terminal, message) = summarize(2, &outcomes, false, READY_DONE);
        assert_eq!(terminal, Terminal::Failed);
        assert_eq!(message, "a planner dropped the move");
    }

    #[test]
    fn no_dispatched_goal_is_a_planner_failure() {
        let (terminal, message) = summarize(0, &[], false, READY_DONE);
        assert_eq!(terminal, Terminal::Failed);
        assert_eq!(message, "an arm's planner is unavailable");
        let only_left = [outcome(true, "trajectory complete")];
        let (terminal, _) = summarize(1, &only_left, false, "");
        assert_eq!(terminal, Terminal::Failed);
    }

    #[test]
    fn two_failures_report_the_first_received() {
        let outcomes = [
            outcome(false, "left: IK failed mid-trajectory"),
            outcome(false, "right: motion timed out"),
        ];
        let (terminal, message) = summarize(2, &outcomes, false, READY_DONE);
        assert_eq!(terminal, Terminal::Failed);
        assert_eq!(message, "left: IK failed mid-trajectory");
    }

    #[test]
    fn a_cancel_with_nothing_pending_still_completes_cancelled() {
        let (terminal, message) = summarize(0, &[], true, READY_DONE);
        assert_eq!(terminal, Terminal::Cancelled);
        assert_eq!(message, "goal cancelled");
    }

    #[test]
    fn a_stopped_arm_ends_the_goal_cancelled_with_the_stops_message() {
        // The stop service ended the left arm's share; the right arm's share
        // ended with it. The goal ends as cancelled, naming the stop.
        let (terminal, message) = summarize(2, &stopped_outcomes(), false, READY_DONE);
        assert_eq!(terminal, Terminal::Cancelled);
        assert_eq!(message, "left: stopped: operator");
    }

    #[test]
    fn a_cancel_after_both_arms_succeeded_reads_as_cancelled() {
        let outcomes = [
            outcome(true, "trajectory complete"),
            outcome(true, "trajectory complete"),
        ];
        let (terminal, message) = summarize(2, &outcomes, true, READY_DONE);
        assert_eq!(terminal, Terminal::Cancelled);
        assert_eq!(message, "goal cancelled");
    }

    /// The result gives the arms in limb_state's order, each with its own
    /// pose. Each orientation is `[x, y, z, w]`.
    #[test]
    fn the_posture_poses_come_in_arm_name_order() {
        let outcomes = [
            outcome(true, "trajectory complete"),
            outcome(true, "trajectory complete"),
        ];
        let result = posture_result(2, &outcomes, false, READY_DONE, both_measured());
        assert_eq!(result.terminal, Terminal::Success);
        assert_eq!(result.message, READY_DONE);
        assert_both_poses(&result);
        let (sin, cos) = 0.1f64.sin_cos();
        let expected = [0.0, 0.0, sin, cos, 0.0, 0.0, -sin, cos];
        for (i, (got, want)) in result.orientations.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "orientations[{i}] = {got}, expected {want}"
            );
        }
    }

    /// A posture that ends failed, cancelled or stopped still gives where
    /// each arm is when the goal completes. So does a posture whose planner
    /// dropped the move.
    #[test]
    fn a_posture_that_does_not_succeed_still_reports_both_poses() {
        let failed = [
            outcome(false, "left: IK failed mid-trajectory"),
            outcome(true, "trajectory complete"),
        ];
        let stopped = stopped_outcomes();
        let dropped = [outcome(true, "trajectory complete")];
        for (outcomes, cancelled, terminal) in [
            (&failed[..], false, Terminal::Failed),
            (&failed[..], true, Terminal::Cancelled),
            (&stopped[..], false, Terminal::Cancelled),
            (&dropped[..], false, Terminal::Failed),
        ] {
            let result = posture_result(2, outcomes, cancelled, READY_DONE, both_measured());
            assert_eq!(result.terminal, terminal);
            assert_both_poses(&result);
        }
    }

    /// When one arm has no fresh measurement, or the coordinator does not
    /// answer, the result gives no pose for any arm. The message adds why:
    /// - the left arm has not measured its joints;
    /// - the follower of the right arm stopped reporting;
    /// - the coordinator is not running.
    #[test]
    fn an_arm_without_a_measured_pose_empties_the_posture_arrays() {
        let succeeded = [
            outcome(true, "trajectory complete"),
            outcome(true, "trajectory complete"),
        ];
        let unmeasured = [
            (
                Ok(ArmPair::new(
                    Err(Unmeasured::NotYet),
                    Ok(grasp_of(Side::Right)),
                )),
                "left_arm has not measured its joints",
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
            let result = posture_result(2, &succeeded, false, READY_DONE, answer);
            assert_eq!(result.terminal, Terminal::Success);
            assert_eq!(
                result.message,
                format!("{READY_DONE}; no arm poses: {reason}")
            );
            assert_no_poses(&result);
        }
    }
}
