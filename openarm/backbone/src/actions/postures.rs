//! The postures contract's moves (move_to_ready, move_to_home): claim both
//! arms, hand each planner an ordinary joint goal to the posture, and
//! complete the one action goal from both terminals. Cancel flips a shared
//! flag the moves poll, so each arm stops the way a cancelled joint move
//! stops. The two actions share the arms' single-flight slots, so a posture
//! goal arriving while the other posture runs is rejected busy.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use peppygen::exposed_actions::postures::{move_to_home, move_to_ready};
use peppygen::{NodeRunner, Result};
use tokio::sync::mpsc;
use tracing::error;

use crate::actions::claim;
use crate::planner::{Goal, JointReply, ReadyOutcome, ReadyReply};
use crate::types::Side;

/// This crate's side in the description's vocabulary.
fn model_side(side: Side) -> openarm_description::Side {
    match side {
        Side::Left => openarm_description::Side::Left,
        Side::Right => openarm_description::Side::Right,
    }
}

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
/// wins whatever the outcomes say; otherwise success requires both arms
/// dispatched and both outcomes successful (reported as `done`), and the
/// message names the first thing that went wrong.
fn summarize(
    pending: usize,
    outcomes: &[ReadyOutcome],
    cancelled: bool,
    done: &str,
) -> (Terminal, String) {
    if cancelled {
        return (Terminal::Cancelled, "goal cancelled".to_string());
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

/// Ceiling on a requested posture duration: the move claims both arms'
/// single-flight slots for its whole run, so an absurd request must not pin
/// the robot until a manual cancel.
const MAX_REQUESTED_DURATION_S: f64 = 600.0;

/// Expand one posture action's run loop: expose it, claim both arms per
/// accepted goal, run both joint moves, complete the goal once both report.
/// One goal at a time; a goal arriving mid-move waits unread until this one
/// completes, then claims the freed arms. Written once here because the two
/// generated action modules carry distinct types with an identical surface.
macro_rules! posture_runner {
    ($fn_name:ident, $action:ident, $name:literal, $posture:expr, $done:literal) => {
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
                        target: $posture(model_side(side)),
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

                let (terminal, message) =
                    summarize(pending, &outcomes, cancel_seen || ctx.is_cancelled(), $done);
                let result = match terminal {
                    Terminal::Success => ctx.complete(true, message).await,
                    Terminal::Failed => ctx.complete(false, message).await,
                    Terminal::Cancelled => ctx.complete_cancelled(false, message).await,
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
    "both arms at ready"
);
posture_runner!(
    run_move_to_home,
    move_to_home,
    "move_to_home",
    openarm_description::home,
    "both arms at home"
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_side_maps_this_crate_onto_the_description() {
        assert_eq!(model_side(Side::Left), openarm_description::Side::Left);
        assert_eq!(model_side(Side::Right), openarm_description::Side::Right);
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
            for side in [Side::Left, Side::Right] {
                let model = crate::arm_model(version, model_side(side))
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
                    let q_all = posture(model_side(side));
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

    fn outcome(success: bool, message: &str) -> ReadyOutcome {
        ReadyOutcome {
            success,
            message: message.to_string(),
        }
    }

    #[test]
    fn both_successful_outcomes_complete_successfully() {
        let outcomes = [
            outcome(true, "trajectory complete"),
            outcome(true, "trajectory complete"),
        ];
        let (terminal, message) = summarize(2, &outcomes, false, "both arms at ready");
        assert_eq!(terminal, Terminal::Success);
        assert_eq!(message, "both arms at ready");
    }

    #[test]
    fn a_failed_arm_fails_the_goal_with_its_message() {
        let outcomes = [
            outcome(true, "trajectory complete"),
            outcome(false, "goal cancelled"),
        ];
        let (terminal, message) = summarize(2, &outcomes, false, "both arms at home");
        assert_eq!(terminal, Terminal::Failed);
        assert_eq!(message, "goal cancelled");
    }

    #[test]
    fn a_dropped_planner_reply_fails_with_a_matching_message() {
        // done_rx closed after one success: success and message must derive
        // from the same predicate, so this cannot read "both arms at ready".
        let outcomes = [outcome(true, "trajectory complete")];
        let (terminal, message) = summarize(2, &outcomes, false, "both arms at ready");
        assert_eq!(terminal, Terminal::Failed);
        assert_eq!(message, "a planner dropped the move");
    }

    #[test]
    fn no_dispatched_goal_is_a_planner_failure() {
        let (terminal, message) = summarize(0, &[], false, "both arms at ready");
        assert_eq!(terminal, Terminal::Failed);
        assert_eq!(message, "an arm's planner is unavailable");
        let (terminal, _) = summarize(1, &[outcome(true, "trajectory complete")], false, "");
        assert_eq!(terminal, Terminal::Failed);
    }

    #[test]
    fn two_failures_report_the_first_received() {
        let outcomes = [
            outcome(false, "left: IK failed mid-trajectory"),
            outcome(false, "right: motion timed out"),
        ];
        let (terminal, message) = summarize(2, &outcomes, false, "both arms at ready");
        assert_eq!(terminal, Terminal::Failed);
        assert_eq!(message, "left: IK failed mid-trajectory");
    }

    #[test]
    fn a_cancel_with_nothing_pending_still_completes_cancelled() {
        let (terminal, message) = summarize(0, &[], true, "both arms at ready");
        assert_eq!(terminal, Terminal::Cancelled);
        assert_eq!(message, "goal cancelled");
    }

    #[test]
    fn a_cancel_after_both_arms_succeeded_reads_as_cancelled() {
        let outcomes = [
            outcome(true, "trajectory complete"),
            outcome(true, "trajectory complete"),
        ];
        let (terminal, message) = summarize(2, &outcomes, true, "both arms at ready");
        assert_eq!(terminal, Terminal::Cancelled);
        assert_eq!(message, "goal cancelled");
    }
}
