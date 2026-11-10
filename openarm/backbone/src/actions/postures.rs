//! The postures contract's moves (move_to_ready, move_to_home). Each claims
//! the limbs it moves, runs their shares of the goal, and ends the one
//! action goal from their terminals:
//! - move_to_ready claims both arms and hands each planner an ordinary joint
//!   goal to the posture. It sends no command to the grippers.
//! - move_to_home claims both grippers and both arms. It closes both
//!   grippers, a gripper move each that the coordinator runs, then hands each
//!   planner its joint goal to the posture. The arms keep their claims
//!   through the close, and they move only once both closes completed: jaws
//!   that an item stops close on it, and the arms go home with it.
//!
//! Cancel flips a shared flag the shares poll, so each limb stops the way a
//! cancelled move of its own stops; the stop service ends the shares in
//! flight the same way, and the goal ends cancelled for the stop's reason.
//! A cancel, a stop or a failure during the close ends the goal before the
//! arms move. The two actions share the arms' move slots, so a posture goal
//! arriving while the other posture runs is refused with
//! `precondition_failed`.
//!
//! A goal that completes or ends cancelled gives the grasp pose of each arm,
//! in limb_state's arm order. The coordinator measures both arms then, after
//! the shares of both arms ended. When an arm has no fresh measurement then,
//! the result gives no pose. A goal that fails gives no result.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use openarm_description::{ArmPair, JointPosture, Side};
use peppygen::exposed_actions::postures::{move_to_home, move_to_ready};
use peppygen::{NodeRunner, Result};
use srs_model::nalgebra::Isometry3;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, warn};

use crate::actions::{
    ARM_WORDS, GRIPPER_WORDS, LimbWords, ask_coordinator, check_limb_free, claim_slots,
    collect_ends, move_seconds_from_wire,
};
use crate::coordinator::{CoordinatorRequest, GraspMeasurement};
use crate::failure::{Failure, FailureKind};
use crate::gripper_move::GripperGoal;
use crate::move_slot::MoveSlot;
use crate::planner::{Goal, JointReply, Outcome, PostureShare, Unmeasured};
use crate::streams::LeaderActivity;
use crate::types::{arm_pose_arrays, limb_names};

/// What a posture action needs of one kind of limb, the arms or the
/// grippers, by side: the queue that takes its goals, its move slots,
/// whether its followers deliver fresh state, and its leaders.
pub struct Limbs<G> {
    pub goal_txs: ArmPair<mpsc::Sender<G>>,
    pub move_slots: ArmPair<MoveSlot>,
    pub fresh: ArmPair<Arc<AtomicBool>>,
    pub activity: ArmPair<LeaderActivity>,
}

// Written by hand: a derive would ask for goals that clone, and each field
// clones whatever the goal.
impl<G> Clone for Limbs<G> {
    fn clone(&self) -> Self {
        Self {
            goal_txs: self.goal_txs.clone(),
            move_slots: self.move_slots.clone(),
            fresh: self.fresh.clone(),
            activity: self.activity.clone(),
        }
    }
}

impl<G> Limbs<G> {
    /// Whether both limbs can take a move now ([`check_limb_free`] over
    /// both), in `words`.
    fn check_free(&self, words: &LimbWords) -> std::result::Result<(), Failure> {
        check_limb_free(
            self.fresh.iter().all(|fresh| fresh.load(Ordering::Acquire)),
            self.activity.iter().any(LeaderActivity::streaming),
            words,
        )
    }

    /// Claim the move slots of both limbs, or refuse with
    /// `precondition_failed`, naming the busy limb in `words`. A failure on
    /// the second claim unwinds the first, so a refusal never leaves a slot
    /// held.
    fn claim_both(&self, words: &LimbWords) -> std::result::Result<(), Failure> {
        claim_slots(&Side::BOTH, &self.move_slots).map_err(|side| {
            let message = format!("the {side} {}", words.busy);
            Failure::new(FailureKind::PreconditionFailed, message)
        })
    }

    /// Free the move slots of both limbs.
    fn release_both(&self) {
        for move_slot in self.move_slots.iter() {
            move_slot.release();
        }
    }
}

/// Admit a posture goal of `duration_s` that moves both arms and first
/// closes `grippers`, when it has some: a duration the robot takes
/// (`rejected` otherwise), every limb that the goal moves free
/// ([`Limbs::check_free`]), then every one of them claimed, the grippers
/// ahead of the arms. A refusal leaves no slot held.
fn admit_posture(
    duration_s: f64,
    arms: &Limbs<Goal>,
    grippers: Option<&Limbs<GripperGoal>>,
) -> std::result::Result<(), Failure> {
    move_seconds_from_wire("duration_s", duration_s)
        .map_err(|reason| Failure::new(FailureKind::Rejected, reason))?;
    arms.check_free(&ARM_WORDS)?;
    let Some(grippers) = grippers else {
        return arms.claim_both(&ARM_WORDS);
    };
    grippers.check_free(&GRIPPER_WORDS)?;
    grippers.claim_both(&GRIPPER_WORDS)?;
    arms.claim_both(&ARM_WORDS)
        .inspect_err(|_| grippers.release_both())
}

/// How a posture goal, or its close, ends.
#[derive(Debug, PartialEq)]
enum PostureEnd {
    /// Both limbs ended their shares: both arms settled at the posture
    /// ([`crate::planner::POSTURE_SETTLE`]), or both grippers closed.
    Completed,
    /// Cancelled for this reason: by the goal's caller, or by the stop.
    Cancelled(String),
    /// A share failed, or its limb did not take or answer it.
    Failed(Failure),
}

/// The words of the failures of a posture goal whose shares on one kind of
/// limb did not all start, or did not all report.
struct ShareWords {
    /// A limb did not take its share.
    unavailable: &'static str,
    /// A limb took its share and did not report how it ended.
    dropped: &'static str,
}

/// The failures of the arms' shares, which the planners run.
const ARM_SHARES: ShareWords = ShareWords {
    unavailable: "an arm's planner is unavailable",
    dropped: "a planner dropped the move",
};

/// The failures of the grippers' shares, which the coordinator runs.
const GRIPPER_SHARES: ShareWords = ShareWords {
    unavailable: "the coordinator takes no gripper close",
    dropped: "the coordinator dropped a gripper close",
};

/// The shares of a posture goal on one kind of limb, as they came back: how
/// many were dispatched, and how those that reported ended.
struct Shares {
    pending: usize,
    outcomes: Vec<Outcome>,
}

/// Judge the end of the shares of a posture goal on one kind of limb from
/// what actually came back: the shares dispatched, the outcomes received,
/// and `cancel`, the reason of the goal's own cancel. The goal's own cancel
/// wins whatever the outcomes say; a share the stop ended ends the goal
/// cancelled for the stop's reason; otherwise the shares complete when both
/// limbs were dispatched and both shares completed, and fail with the first
/// thing that went wrong, in `words`.
fn summarize(shares: &Shares, cancel: Option<String>, words: &ShareWords) -> PostureEnd {
    let Shares { pending, outcomes } = shares;
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
    let share_failure = |message: &str| Failure::new(FailureKind::AttemptFailed, message);
    if *pending < 2 {
        return PostureEnd::Failed(share_failure(words.unavailable));
    }
    if outcomes.len() < *pending {
        return PostureEnd::Failed(share_failure(words.dropped));
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

/// The cancel of a posture goal, as the goal context of either postures
/// action gives it: the two generated contexts are distinct types with an
/// identical surface.
trait PostureGoal {
    /// Resolves once the goal received its cancel signal.
    fn cancel_signal(&self) -> impl Future<Output = ()>;

    /// The reason of the cancel the goal received, or `None` before one.
    fn cancel_reason(&self) -> Option<String>;
}

/// Give each limb of `limbs` its share of the goal `ctx` of `action`,
/// `goal_of` making the goal that carries it, and wait until both shares
/// reported how they ended ([`collect_ends`]). The cancel of the goal raises
/// `cancelled`, which the shares poll. A limb whose queue is closed gets no
/// share, and its claim is released.
async fn run_shares<G>(
    action: &str,
    ctx: &impl PostureGoal,
    limbs: &Limbs<G>,
    goal_of: impl Fn(Side, PostureShare) -> G,
    cancelled: &Arc<AtomicBool>,
) -> Shares {
    let (done_tx, mut done_rx) = mpsc::channel::<Outcome>(2);
    let mut pending = 0usize;
    for side in Side::BOTH {
        let share = PostureShare {
            done_tx: done_tx.clone(),
            cancelled: cancelled.clone(),
        };
        if limbs.goal_txs[side]
            .send(goal_of(side, share))
            .await
            .is_err()
        {
            // The limb's runner is gone; release the claim its goal would
            // have held.
            limbs.move_slots[side].release();
            error!("{action}: {side} goal channel closed");
            continue;
        }
        pending += 1;
    }
    drop(done_tx);
    let outcomes = collect_ends(pending, &mut done_rx, ctx.cancel_signal(), cancelled).await;
    Shares { pending, outcomes }
}

/// Run the admitted goal `ctx` of `action`, and give how it ends:
/// - close `grippers`, when the goal has some. A close that does not
///   complete ends the goal there, and the arms, which no planner took, are
///   freed;
/// - move both arms to `posture` over `duration`.
async fn run_posture(
    action: &str,
    ctx: &impl PostureGoal,
    arms: &Limbs<Goal>,
    grippers: Option<&Limbs<GripperGoal>>,
    posture: fn(Side) -> JointPosture,
    duration: Duration,
) -> PostureEnd {
    let cancelled = Arc::new(AtomicBool::new(false));
    if let Some(grippers) = grippers {
        let closes = run_shares(action, ctx, grippers, GripperGoal::close, &cancelled).await;
        let closed = summarize(&closes, ctx.cancel_reason(), &GRIPPER_SHARES);
        if closed != PostureEnd::Completed {
            arms.release_both();
            return closed;
        }
    }
    let arm_goal = |side, share| Goal::Joint {
        target: posture(side),
        duration,
        reply: JointReply::Posture(share),
    };
    let moves = run_shares(action, ctx, arms, arm_goal, &cancelled).await;
    summarize(&moves, ctx.cancel_reason(), &ARM_SHARES)
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
type GraspAnswer = std::result::Result<ArmPair<GraspMeasurement>, String>;

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
        let grasp: GraspMeasurement = grasps[side];
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

/// Expand one posture action's run loop: expose it, admit each goal
/// ([`admit_posture`]), run it ([`run_posture`]) and end it: a goal that
/// completes or ends cancelled gives the grasp poses that the coordinator
/// measures then. `grippers` are the grippers that the action closes ahead
/// of the arms, or `None` for an action that sends them no command.
/// One goal at a time; a goal arriving mid-move waits unread until this one
/// ends, then claims the freed limbs. Written once here because the two
/// generated action modules carry distinct types with an identical surface.
macro_rules! posture_runner {
    ($fn_name:ident, $action:ident, $name:literal, $posture:expr) => {
        impl PostureGoal for $action::GoalContext {
            fn cancel_signal(&self) -> impl Future<Output = ()> {
                $action::GoalContext::cancel_signal(self)
            }

            fn cancel_reason(&self) -> Option<String> {
                self.cancel_cause().map(|cause| cause.reason().to_string())
            }
        }

        pub async fn $fn_name(
            runner: Arc<NodeRunner>,
            arms: Limbs<Goal>,
            grippers: Option<Limbs<GripperGoal>>,
            requests: mpsc::Sender<CoordinatorRequest>,
        ) -> Result<()> {
            let mut handle = $action::ActionHandle::expose(&runner).await?;
            loop {
                let accepted = handle
                    .handle_goal_next_request(|req| {
                        Ok(
                            match admit_posture(req.data.duration_s, &arms, grippers.as_ref()) {
                                Ok(()) => $action::GoalDecision::accept(),
                                Err(f) => $action::GoalDecision::refuse(f.kind.into(), f.message),
                            },
                        )
                    })
                    .await?;
                let Some(ctx) = accepted else { return Ok(()) };
                let duration = move_seconds_from_wire("duration_s", ctx.request().data.duration_s)
                    .expect("validated on accept");

                let end =
                    run_posture($name, &ctx, &arms, grippers.as_ref(), $posture, duration).await;
                let ended = match end {
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

    use crate::gripper_move::GripperReply;
    use crate::test_support::bundled_arm;

    /// The limbs of one kind, each with a fresh follower, no leader and a
    /// free move slot, and the queues that take their goals.
    fn limbs<G>() -> (Limbs<G>, ArmPair<mpsc::Receiver<G>>) {
        let (goal_txs, queues) = ArmPair::from_fn(|_| mpsc::channel(1)).unzip();
        let limbs = Limbs {
            goal_txs,
            move_slots: ArmPair::from_fn(|_| MoveSlot::default()),
            fresh: ArmPair::from_fn(|_| Arc::new(AtomicBool::new(true))),
            activity: ArmPair::from_fn(|_| LeaderActivity::default()),
        };
        (limbs, queues)
    }

    /// The limbs of one kind as [`limbs`] gives them, with no queue: for a
    /// goal that is admitted and not run.
    fn admitted_limbs<G>() -> Limbs<G> {
        limbs().0
    }

    /// Which move slots of `limbs` a goal holds.
    fn held<G>(limbs: &Limbs<G>) -> ArmPair<bool> {
        ArmPair::from_fn(|side| limbs.move_slots[side].is_claimed())
    }

    fn refused(kind: FailureKind, message: &str) -> std::result::Result<(), Failure> {
        Err(Failure::new(kind, message))
    }

    const NONE_HELD: ArmPair<bool> = ArmPair::new(false, false);
    const BOTH_HELD: ArmPair<bool> = ArmPair::new(true, true);

    #[test]
    fn a_goal_that_closes_no_gripper_claims_both_arms_alone() {
        let arms = admitted_limbs();
        assert_eq!(admit_posture(1.0, &arms, None), Ok(()));
        assert_eq!(held(&arms), BOTH_HELD);
    }

    #[test]
    fn a_goal_that_closes_the_grippers_claims_both_grippers_and_both_arms() {
        let (arms, grippers) = (admitted_limbs(), admitted_limbs());
        assert_eq!(admit_posture(1.0, &arms, Some(&grippers)), Ok(()));
        assert_eq!(held(&grippers), BOTH_HELD);
        assert_eq!(held(&arms), BOTH_HELD);
    }

    #[test]
    fn a_busy_arm_refuses_the_goal_by_name_and_no_refused_slot_stays_held() {
        for (busy, message) in [
            (Side::Left, "the left arm is already executing a motion"),
            (Side::Right, "the right arm is already executing a motion"),
        ] {
            let (arms, grippers) = (admitted_limbs(), admitted_limbs());
            assert!(arms.move_slots[busy].claim());
            assert_eq!(
                admit_posture(1.0, &arms, Some(&grippers)),
                refused(FailureKind::PreconditionFailed, message)
            );
            let other_move = ArmPair::from_fn(|side| side == busy);
            assert_eq!(held(&arms), other_move, "the other move keeps its arm");
            assert_eq!(held(&grippers), NONE_HELD);
        }
    }

    #[test]
    fn a_busy_gripper_refuses_the_goal_by_name_and_no_refused_slot_stays_held() {
        for (busy, message) in [
            (Side::Left, "the left gripper is already executing a move"),
            (Side::Right, "the right gripper is already executing a move"),
        ] {
            let (arms, grippers) = (admitted_limbs(), admitted_limbs());
            assert!(grippers.move_slots[busy].claim());
            assert_eq!(
                admit_posture(1.0, &arms, Some(&grippers)),
                refused(FailureKind::PreconditionFailed, message)
            );
            let other_move = ArmPair::from_fn(|side| side == busy);
            assert_eq!(held(&grippers), other_move, "the other move keeps it");
            assert_eq!(held(&arms), NONE_HELD);
        }
    }

    /// A leader that streams a gripper refuses the goal that closes it, and
    /// not a goal that sends the grippers no command.
    #[test]
    fn a_leader_streaming_a_gripper_refuses_the_goal_that_closes_it() {
        let arms = admitted_limbs();
        let mut grippers = admitted_limbs::<GripperGoal>();
        grippers.activity[Side::Right] = LeaderActivity::streaming_always();
        assert_eq!(
            admit_posture(1.0, &arms, Some(&grippers)),
            refused(FailureKind::LeaderStreaming, GRIPPER_WORDS.leader_streaming)
        );
        assert_eq!(held(&grippers), NONE_HELD);
        assert_eq!(held(&arms), NONE_HELD);
        assert_eq!(admit_posture(1.0, &arms, None), Ok(()));
    }

    #[test]
    fn a_gripper_with_no_fresh_state_refuses_the_goal_that_closes_it() {
        let (arms, grippers) = (admitted_limbs(), admitted_limbs::<GripperGoal>());
        grippers.fresh[Side::Left].store(false, Ordering::Release);
        assert_eq!(
            admit_posture(1.0, &arms, Some(&grippers)),
            refused(
                FailureKind::ResourceUnavailable,
                GRIPPER_WORDS.no_fresh_state
            )
        );
        assert_eq!(held(&grippers), NONE_HELD);
        assert_eq!(held(&arms), NONE_HELD);
    }

    /// An arm refuses ahead of a gripper, so both postures refuse a robot
    /// whose arms cannot move in the same words.
    #[test]
    fn an_arm_refuses_the_goal_ahead_of_a_gripper() {
        let mut arms = admitted_limbs();
        let mut grippers = admitted_limbs::<GripperGoal>();
        arms.activity[Side::Left] = LeaderActivity::streaming_always();
        grippers.activity[Side::Left] = LeaderActivity::streaming_always();
        assert_eq!(
            admit_posture(1.0, &arms, Some(&grippers)),
            refused(FailureKind::LeaderStreaming, ARM_WORDS.leader_streaming)
        );
    }

    // The run of an admitted goal. The test plays the coordinator and the
    // planners: it takes each share off its queue and reports how it ends.

    /// A posture goal that its caller does not cancel.
    struct Uncancelled;

    impl PostureGoal for Uncancelled {
        fn cancel_signal(&self) -> impl Future<Output = ()> {
            std::future::pending()
        }

        fn cancel_reason(&self) -> Option<String> {
            None
        }
    }

    /// A posture goal that its caller cancelled.
    struct CancelledByCaller;

    impl PostureGoal for CancelledByCaller {
        fn cancel_signal(&self) -> impl Future<Output = ()> {
            std::future::ready(())
        }

        fn cancel_reason(&self) -> Option<String> {
            Some(CALLER.to_string())
        }
    }

    /// The share of the gripper close that `queue` holds, and the side it
    /// names. The close goes to a closed gripper under no effort cap.
    async fn next_close(queue: &mut mpsc::Receiver<GripperGoal>) -> (Side, PostureShare) {
        let goal = queue.recv().await.expect("a gripper close");
        assert_eq!((goal.opening, goal.max_effort), (0.0, None));
        match goal.reply {
            GripperReply::Close { side, share } => (side, share),
            GripperReply::MoveGripper(_) => panic!("a move_gripper goal, not a close"),
        }
    }

    /// The share of the arm move that `queue` holds, and its target.
    async fn next_arm_move(queue: &mut mpsc::Receiver<Goal>) -> (JointPosture, PostureShare) {
        match queue.recv().await.expect("an arm move") {
            Goal::Joint {
                target,
                reply: JointReply::Posture(share),
                ..
            } => (target, share),
            _ => panic!("not a posture share"),
        }
    }

    /// Whether no goal waits on either queue.
    fn both_empty<G>(queues: &mut ArmPair<mpsc::Receiver<G>>) -> bool {
        Side::BOTH.into_iter().all(|side| {
            matches!(
                queues[side].try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            )
        })
    }

    /// End the close of the gripper on each side with `outcome_of` its side,
    /// once both closes are queued.
    async fn end_closes(
        queues: &mut ArmPair<mpsc::Receiver<GripperGoal>>,
        outcome_of: impl Fn(Side) -> Outcome,
    ) {
        for expected in Side::BOTH {
            let (side, share) = next_close(&mut queues[expected]).await;
            assert_eq!(side, expected);
            share.report(side.gripper_name(), outcome_of(side)).await;
        }
    }

    /// A goal that closes the grippers gives the arms their moves only once
    /// both closes completed, and completes once both arms did.
    #[tokio::test]
    async fn a_goal_that_closes_the_grippers_moves_the_arms_after_both_closes() {
        let (arms, mut arm_queues) = limbs();
        let (grippers, mut gripper_queues) = limbs();
        assert_eq!(admit_posture(0.0, &arms, Some(&grippers)), Ok(()));
        let run = run_posture(
            "move_to_home",
            &Uncancelled,
            &arms,
            Some(&grippers),
            openarm_description::home,
            Duration::ZERO,
        );
        let robot = async {
            let (side, left_close) = next_close(&mut gripper_queues.left).await;
            assert_eq!(side, Side::Left);
            let (side, right_close) = next_close(&mut gripper_queues.right).await;
            assert_eq!(side, Side::Right);
            left_close.report("left_gripper", Outcome::Complete).await;
            tokio::task::yield_now().await;
            assert!(both_empty(&mut arm_queues), "an arm moved ahead of a close");
            right_close.report("right_gripper", Outcome::Complete).await;
            for side in Side::BOTH {
                let (target, share) = next_arm_move(&mut arm_queues[side]).await;
                assert_eq!(target, openarm_description::home(side));
                share.report(side, Outcome::Complete).await;
            }
        };
        let (end, ()) = tokio::join!(run, robot);
        assert_eq!(end, PostureEnd::Completed);
    }

    /// A goal that closes no gripper gives the arms their moves at once.
    #[tokio::test]
    async fn a_goal_that_closes_no_gripper_moves_the_arms_at_once() {
        let (arms, mut arm_queues) = limbs();
        assert_eq!(admit_posture(0.0, &arms, None), Ok(()));
        let run = run_posture(
            "move_to_ready",
            &Uncancelled,
            &arms,
            None,
            openarm_description::ready,
            Duration::ZERO,
        );
        let robot = async {
            for side in Side::BOTH {
                let (target, share) = next_arm_move(&mut arm_queues[side]).await;
                assert_eq!(target, openarm_description::ready(side));
                share.report(side, Outcome::Complete).await;
            }
        };
        let (end, ()) = tokio::join!(run, robot);
        assert_eq!(end, PostureEnd::Completed);
    }

    /// The end of a goal whose closes end with `outcome_of` their side, for
    /// the cancel that `ctx` gives. The arms get no move, and the goal frees
    /// them.
    async fn end_of_a_close_that_ends(
        ctx: &impl PostureGoal,
        outcome_of: impl Fn(Side) -> Outcome,
    ) -> PostureEnd {
        let (arms, mut arm_queues) = limbs();
        let (grippers, mut gripper_queues) = limbs();
        assert_eq!(admit_posture(0.0, &arms, Some(&grippers)), Ok(()));
        let run = run_posture(
            "move_to_home",
            ctx,
            &arms,
            Some(&grippers),
            openarm_description::home,
            Duration::ZERO,
        );
        let (end, ()) = tokio::join!(run, end_closes(&mut gripper_queues, outcome_of));
        assert!(both_empty(&mut arm_queues), "an arm moved");
        assert_eq!(held(&arms), NONE_HELD, "the goal keeps an arm");
        end
    }

    /// A close that fails ends the goal with its failure, which names the
    /// gripper.
    #[tokio::test]
    async fn a_close_that_fails_fails_the_goal_before_the_arms_move() {
        let stale = Failure::new(
            FailureKind::ResourceUnavailable,
            "the follower stopped reporting",
        );
        let outcome_of = |side| match side {
            Side::Left => Outcome::Complete,
            Side::Right => Outcome::Failed(stale.clone()),
        };
        assert_eq!(
            end_of_a_close_that_ends(&Uncancelled, outcome_of).await,
            PostureEnd::Failed(stale.clone().labelled("right_gripper"))
        );
    }

    /// The stop service ended both closes. The goal ends cancelled for the
    /// stop's reason.
    #[tokio::test]
    async fn a_stop_during_the_close_cancels_the_goal_before_the_arms_move() {
        let stopped = |_| Outcome::Cancelled("stopped: operator".to_string());
        assert_eq!(
            end_of_a_close_that_ends(&Uncancelled, stopped).await,
            PostureEnd::Cancelled("stopped: operator".to_string())
        );
    }

    /// The caller cancelled the goal during the close. The goal ends
    /// cancelled for the caller's reason, whatever the closes report.
    #[tokio::test]
    async fn a_cancel_during_the_close_cancels_the_goal_before_the_arms_move() {
        for outcome in [
            Outcome::Complete,
            Outcome::Cancelled("its posture goal was cancelled".to_string()),
        ] {
            assert_eq!(
                end_of_a_close_that_ends(&CancelledByCaller, |_| outcome.clone()).await,
                PostureEnd::Cancelled(CALLER.to_string())
            );
        }
    }

    /// The cancel of the goal raises the flag that its closes poll.
    #[tokio::test]
    async fn the_cancel_of_the_goal_reaches_the_closes() {
        let (arms, _arm_queues) = limbs();
        let (grippers, mut gripper_queues) = limbs();
        assert_eq!(admit_posture(0.0, &arms, Some(&grippers)), Ok(()));
        let run = run_posture(
            "move_to_home",
            &CancelledByCaller,
            &arms,
            Some(&grippers),
            openarm_description::home,
            Duration::ZERO,
        );
        let robot = async {
            for expected in Side::BOTH {
                let (side, share) = next_close(&mut gripper_queues[expected]).await;
                while share.cancel_reason().is_none() {
                    tokio::task::yield_now().await;
                }
                let reason = share.cancel_reason().expect("the cancel reached the close");
                share
                    .report(side.gripper_name(), Outcome::Cancelled(reason))
                    .await;
            }
        };
        let (end, ()) = tokio::join!(run, robot);
        assert_eq!(end, PostureEnd::Cancelled(CALLER.to_string()));
    }

    /// A coordinator that takes no close fails the goal, and the goal frees
    /// every limb.
    #[tokio::test]
    async fn a_coordinator_that_takes_no_close_fails_the_goal() {
        let (arms, mut arm_queues) = limbs();
        let (grippers, gripper_queues) = limbs();
        drop(gripper_queues);
        assert_eq!(admit_posture(0.0, &arms, Some(&grippers)), Ok(()));
        let end = run_posture(
            "move_to_home",
            &Uncancelled,
            &arms,
            Some(&grippers),
            openarm_description::home,
            Duration::ZERO,
        )
        .await;
        assert_eq!(
            end,
            PostureEnd::Failed(Failure::new(
                FailureKind::AttemptFailed,
                "the coordinator takes no gripper close"
            ))
        );
        assert!(both_empty(&mut arm_queues), "an arm moved");
        assert_eq!(held(&grippers), NONE_HELD);
        assert_eq!(held(&arms), NONE_HELD);
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
                let model = bundled_arm(version, side);
                let limits = model.limits();
                let postures = [
                    (
                        "ready",
                        ready as fn(openarm_description::Side) -> JointPosture,
                    ),
                    ("home", home),
                ];
                for (name, posture) in postures {
                    let q_all = posture(side);
                    for (j, (&q, limit)) in q_all.iter().zip(&limits).enumerate() {
                        assert!(
                            q >= limit.lo && q <= limit.hi,
                            "{version:?} {} {name} j{}: {q} outside [{}, {}]",
                            side,
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

    /// How the shares of the arms end, `pending` of them dispatched and
    /// `outcomes` reported.
    fn arms_end(pending: usize, outcomes: &[Outcome], cancel: Option<String>) -> PostureEnd {
        let shares = Shares {
            pending,
            outcomes: outcomes.to_vec(),
        };
        summarize(&shares, cancel, &ARM_SHARES)
    }

    #[test]
    fn both_complete_shares_complete_the_goal() {
        let outcomes = [Outcome::Complete, Outcome::Complete];
        assert_eq!(arms_end(2, &outcomes, None), PostureEnd::Completed);
    }

    #[test]
    fn a_failed_arm_fails_the_goal_with_its_failure() {
        let stale = Failure::new(
            FailureKind::ResourceUnavailable,
            "right: the follower stopped reporting",
        );
        let outcomes = [Outcome::Complete, Outcome::Failed(stale.clone())];
        assert_eq!(arms_end(2, &outcomes, None), PostureEnd::Failed(stale));
    }

    #[test]
    fn a_dropped_planner_reply_fails_with_attempt_failed() {
        let outcomes = [Outcome::Complete];
        assert_eq!(
            arms_end(2, &outcomes, None),
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
        assert_eq!(arms_end(0, &[], None), unavailable);
        assert_eq!(arms_end(1, &[Outcome::Complete], None), unavailable);
    }

    /// The shares of the grippers fail in the coordinator's words.
    #[test]
    fn a_close_that_did_not_start_or_report_fails_in_the_coordinators_words() {
        let closes_end = |pending, outcomes: &[Outcome]| {
            let shares = Shares {
                pending,
                outcomes: outcomes.to_vec(),
            };
            summarize(&shares, None, &GRIPPER_SHARES)
        };
        let failed =
            |message| PostureEnd::Failed(Failure::new(FailureKind::AttemptFailed, message));
        assert_eq!(
            closes_end(1, &[Outcome::Complete]),
            failed("the coordinator takes no gripper close")
        );
        assert_eq!(
            closes_end(2, &[Outcome::Complete]),
            failed("the coordinator dropped a gripper close")
        );
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
            arms_end(2, &outcomes, None),
            PostureEnd::Failed(Failure::new(
                FailureKind::AttemptFailed,
                "left: IK failed mid-trajectory"
            ))
        );
    }

    #[test]
    fn a_cancel_with_nothing_pending_still_ends_cancelled() {
        assert_eq!(
            arms_end(0, &[], Some(CALLER.to_string())),
            PostureEnd::Cancelled(CALLER.to_string())
        );
    }

    /// The stop service ended both arms' shares. The goal ends cancelled
    /// for the stop's reason.
    #[test]
    fn a_stopped_arm_ends_the_goal_cancelled_for_the_stops_reason() {
        assert_eq!(
            arms_end(2, &stopped_outcomes(), None),
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
                arms_end(2, outcomes, Some(CALLER.to_string())),
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
