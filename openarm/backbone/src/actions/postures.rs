//! The postures contract's moves (move_to_ready, move_to_home): claim both
//! arms, hand each planner an ordinary joint goal to the posture, and
//! complete the one action goal from both terminals. Cancel flips a shared
//! flag the moves poll, so each arm stops the way a cancelled joint move
//! stops; the stop service ends both arms' moves the same way, and the goal
//! ends as cancelled with the stop's message. The two actions share the
//! arms' single-flight slots, so a posture goal arriving while the other
//! posture runs is rejected busy. Whatever the terminal, the result gives
//! the grasp pose of each arm from the joints it measured when its share
//! ended, in limb_state's arm order; when an arm has no such pose, the
//! result gives no pose and its message says why.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use peppygen::exposed_actions::postures::{move_to_home, move_to_ready};
use peppygen::{NodeRunner, Result};
use srs_model::nalgebra::Isometry3;
use tokio::sync::mpsc;
use tracing::error;

use crate::actions::claim;
use crate::arm_pair::ArmPair;
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

/// The grasp pose of each arm when its share ended, or why an arm has
/// none: its share did not report (its planner is unavailable or dropped
/// the move), the arm had not measured its joints, or its follower stopped
/// reporting.
fn measured_grasps(
    outcomes: &[ReadyOutcome],
) -> std::result::Result<ArmPair<Isometry3<f64>>, String> {
    let grasp = |side: Side| {
        let arm = Side::ARM_NAMES[side.index()];
        let outcome = outcomes
            .iter()
            .find(|o| o.side == side)
            .ok_or_else(|| format!("{arm} did not report the end of its move"))?;
        outcome.grasp.map_err(|unmeasured| match unmeasured {
            Unmeasured::NotYet => format!("{arm} has not measured its joints"),
            Unmeasured::Stale => format!("{arm} stopped reporting its joints"),
        })
    };
    Ok(ArmPair::new(grasp(Side::Left)?, grasp(Side::Right)?))
}

/// The result of a posture goal: the terminal and message [`summarize`]
/// gives, and the grasp pose of every arm in [`Side::ARM_NAMES`] order,
/// whatever the order the shares ended in and whatever the terminal. When
/// an arm has no grasp pose, the three arrays are empty and the message
/// says why.
fn posture_result(
    pending: usize,
    outcomes: &[ReadyOutcome],
    cancelled: bool,
    done: &str,
) -> PostureResult {
    let (terminal, message) = summarize(pending, outcomes, cancelled, done);
    match measured_grasps(outcomes) {
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

/// The message of a move_to_ready that succeeds. Success says that both
/// arms' moves ran their time out, not that the arms arrived: the governor
/// can hold an arm short, and an object can stop it.
const READY_DONE: &str = "the move to ready ran its time";

/// The message of a move_to_home that succeeds, worded as [`READY_DONE`].
const HOME_DONE: &str = "the move to home ran its time";

/// Expand one posture action's run loop: expose it, claim both arms per
/// accepted goal, run both joint moves, complete the goal once both report.
/// One goal at a time; a goal arriving mid-move waits unread until this one
/// completes, then claims the freed arms. Written once here because the two
/// generated action modules carry distinct types with an identical surface.
macro_rules! posture_runner {
    ($fn_name:ident, $action:ident, $name:literal, $posture:expr, $done:expr) => {
        pub async fn $fn_name(
            runner: Arc<NodeRunner>,
            goal_txs: [mpsc::Sender<Goal>; 2],
            busy: [Arc<AtomicBool>; 2],
        ) -> Result<()> {
            let mut handle = $action::ActionHandle::expose(&runner).await?;
            loop {
                let accepted = handle
                    .handle_goal_next_request(|req| {
                        let duration_s = req.data.duration_s;
                        if !(duration_s.is_finite()
                            && (0.0..=MAX_REQUESTED_DURATION_S).contains(&duration_s))
                        {
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

                let PostureResult {
                    terminal,
                    message,
                    arm_names,
                    positions,
                    orientations,
                } = posture_result(pending, &outcomes, cancel_seen || ctx.is_cancelled(), $done);
                let result = match terminal {
                    Terminal::Success => {
                        ctx.complete(true, message, arm_names, positions, orientations)
                            .await
                    }
                    Terminal::Failed => {
                        ctx.complete(false, message, arm_names, positions, orientations)
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

    /// A grasp pose that tells the arms apart: y is 0.2 m on the left and
    /// -0.2 m on the right, and each turns about z by its own y in radians.
    fn grasp_of(side: Side) -> Isometry3<f64> {
        let y = match side {
            Side::Left => 0.2,
            Side::Right => -0.2,
        };
        Isometry3::new(Vector3::new(0.3, y, 0.25), Vector3::new(0.0, 0.0, y))
    }

    /// The share of `side` that ended with `success` and `message`, its arm
    /// measured at [`grasp_of`].
    fn outcome(side: Side, success: bool, message: &str) -> ReadyOutcome {
        ReadyOutcome {
            side,
            success,
            message: message.to_string(),
            stopped: false,
            grasp: Ok(grasp_of(side)),
        }
    }

    #[test]
    fn both_successful_outcomes_complete_successfully() {
        let outcomes = [
            outcome(Side::Left, true, "trajectory complete"),
            outcome(Side::Right, true, "trajectory complete"),
        ];
        let (terminal, message) = summarize(2, &outcomes, false, READY_DONE);
        assert_eq!(terminal, Terminal::Success);
        assert_eq!(message, READY_DONE);
    }

    #[test]
    fn a_failed_arm_fails_the_goal_with_its_message() {
        let outcomes = [
            outcome(Side::Left, true, "trajectory complete"),
            outcome(Side::Right, false, "goal cancelled"),
        ];
        let (terminal, message) = summarize(2, &outcomes, false, HOME_DONE);
        assert_eq!(terminal, Terminal::Failed);
        assert_eq!(message, "goal cancelled");
    }

    #[test]
    fn a_dropped_planner_reply_fails_with_a_matching_message() {
        // done_rx closed after one success: success and message must derive
        // from the same predicate, so this cannot read READY_DONE.
        let outcomes = [outcome(Side::Left, true, "trajectory complete")];
        let (terminal, message) = summarize(2, &outcomes, false, READY_DONE);
        assert_eq!(terminal, Terminal::Failed);
        assert_eq!(message, "a planner dropped the move");
    }

    #[test]
    fn no_dispatched_goal_is_a_planner_failure() {
        let (terminal, message) = summarize(0, &[], false, READY_DONE);
        assert_eq!(terminal, Terminal::Failed);
        assert_eq!(message, "an arm's planner is unavailable");
        let only_left = [outcome(Side::Left, true, "trajectory complete")];
        let (terminal, _) = summarize(1, &only_left, false, "");
        assert_eq!(terminal, Terminal::Failed);
    }

    #[test]
    fn two_failures_report_the_first_received() {
        let outcomes = [
            outcome(Side::Left, false, "left: IK failed mid-trajectory"),
            outcome(Side::Right, false, "right: motion timed out"),
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
        let outcomes = [
            ReadyOutcome {
                stopped: true,
                ..outcome(Side::Left, false, "left: stopped: operator")
            },
            ReadyOutcome {
                stopped: true,
                ..outcome(Side::Right, false, "right: stopped: operator")
            },
        ];
        let (terminal, message) = summarize(2, &outcomes, false, READY_DONE);
        assert_eq!(terminal, Terminal::Cancelled);
        assert_eq!(message, "left: stopped: operator");
    }

    #[test]
    fn a_cancel_after_both_arms_succeeded_reads_as_cancelled() {
        let outcomes = [
            outcome(Side::Left, true, "trajectory complete"),
            outcome(Side::Right, true, "trajectory complete"),
        ];
        let (terminal, message) = summarize(2, &outcomes, true, READY_DONE);
        assert_eq!(terminal, Terminal::Cancelled);
        assert_eq!(message, "goal cancelled");
    }

    /// The result gives the arms in limb_state's order, each with its own
    /// pose, whatever the order their shares ended in: here the right arm's
    /// share ends first.
    #[test]
    fn the_posture_poses_come_in_arm_name_order() {
        let outcomes = [
            outcome(Side::Right, true, "trajectory complete"),
            outcome(Side::Left, true, "trajectory complete"),
        ];
        let result = posture_result(2, &outcomes, false, READY_DONE);
        assert_eq!(result.terminal, Terminal::Success);
        assert_eq!(result.message, READY_DONE);
        assert_eq!(result.arm_names, ["left_arm", "right_arm"]);
        assert_eq!(result.positions, [0.3, 0.2, 0.25, 0.3, -0.2, 0.25]);
        let (sin, cos) = 0.1f64.sin_cos();
        let expected = [0.0, 0.0, sin, cos, 0.0, 0.0, -sin, cos];
        assert_eq!(result.orientations.len(), expected.len());
        for (i, (got, want)) in result.orientations.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "orientations[{i}] = {got}, expected {want}"
            );
        }
    }

    /// A posture that ends failed, cancelled or stopped still gives where
    /// each arm ended.
    #[test]
    fn a_posture_that_does_not_succeed_still_reports_both_poses() {
        let failed = [
            outcome(Side::Left, false, "left: IK failed mid-trajectory"),
            outcome(Side::Right, true, "trajectory complete"),
        ];
        let stopped = [
            ReadyOutcome {
                stopped: true,
                ..outcome(Side::Left, false, "left: stopped: operator")
            },
            ReadyOutcome {
                stopped: true,
                ..outcome(Side::Right, false, "right: stopped: operator")
            },
        ];
        for (outcomes, cancelled, terminal) in [
            (&failed, false, Terminal::Failed),
            (&failed, true, Terminal::Cancelled),
            (&stopped, false, Terminal::Cancelled),
        ] {
            let result = posture_result(2, outcomes, cancelled, READY_DONE);
            assert_eq!(result.terminal, terminal);
            assert_eq!(result.arm_names, ["left_arm", "right_arm"]);
            assert_eq!(result.positions, [0.3, 0.2, 0.25, 0.3, -0.2, 0.25]);
            assert_eq!(result.orientations.len(), 8);
        }
    }

    /// When one arm has no measured pose, the result gives no pose for any
    /// arm, and the message adds why: here the right arm's share never
    /// reported, then the left arm had not measured its joints, then the
    /// right arm's follower stopped reporting.
    #[test]
    fn an_arm_without_a_measured_pose_empties_the_posture_arrays() {
        let dropped = [outcome(Side::Left, true, "trajectory complete")];
        let result = posture_result(2, &dropped, false, READY_DONE);
        assert_eq!(result.terminal, Terminal::Failed);
        assert_eq!(
            result.message,
            "a planner dropped the move; no arm poses: right_arm did not report the end of \
             its move"
        );
        assert!(result.arm_names.is_empty());
        assert!(result.positions.is_empty() && result.orientations.is_empty());

        let refusal = "the follower has not reported its first state yet";
        let unmeasured = [
            outcome(Side::Right, false, &format!("right: {refusal}")),
            ReadyOutcome {
                grasp: Err(Unmeasured::NotYet),
                ..outcome(Side::Left, false, &format!("left: {refusal}"))
            },
        ];
        let result = posture_result(2, &unmeasured, false, READY_DONE);
        assert_eq!(result.terminal, Terminal::Failed);
        assert_eq!(
            result.message,
            format!("right: {refusal}; no arm poses: left_arm has not measured its joints")
        );
        assert!(result.arm_names.is_empty());
        assert!(result.positions.is_empty() && result.orientations.is_empty());

        let stale = [
            outcome(Side::Left, true, "trajectory complete"),
            ReadyOutcome {
                grasp: Err(Unmeasured::Stale),
                ..outcome(Side::Right, false, "right: the follower stopped reporting")
            },
        ];
        let result = posture_result(2, &stale, false, READY_DONE);
        assert_eq!(result.terminal, Terminal::Failed);
        assert_eq!(
            result.message,
            "right: the follower stopped reporting; no arm poses: right_arm stopped reporting \
             its joints"
        );
        assert!(result.arm_names.is_empty());
        assert!(result.positions.is_empty() && result.orientations.is_empty());
    }
}
