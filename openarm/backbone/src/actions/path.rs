//! Joint path admission: the `execute_joint_path` handler of limb_motion. It
//! parses its goal into one joint path of the arms it moves (the arm names,
//! the waypoints, the expected starts, the duration and the settle it asks
//! for), refusing a value it does not take with `rejected`, refuses it while
//! a moved arm's follower delivers no fresh state or a leader streams it, and
//! claims the moved arms' move slots ([`admit_arms`]).
//!
//! Each accepted goal runs on its own task. It hands the path to the
//! coordinator and waits until the coordinator takes it: the coordinator
//! starts it on every moved arm at once, or fails it before any arm moves
//! ([`crate::planner::start_path`]), and the stop ends a path it has not
//! started. The task then waits for each arm's share ([`collect_ends`]) and
//! ends the goal from them: a goal that completes or ends cancelled gives
//! each arm's joints as its share reports them and the time since the path
//! started.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::arm_order::ArmOrder;
use crate::robot::{ARMS, LIMBS};
use peppygen::exposed_actions::limb_motion::execute_joint_path::{
    self, ExecuteJointPathResultArmsItem,
};
use peppygen::{NodeRunner, Result};
use robot_description::{ArmId, PerArm};
use srs_model::ARM_DOF;
use srs_model::{JointVec, Limit, take_targets};
use tokio::sync::{mpsc, oneshot};
use tracing::error;

use crate::actions::arm::joint_tolerances_from_wire;
use crate::actions::{
    ARM_WORDS, check_limb_free, claim_slots, collect_ends, contact_gap_from_wire,
    move_seconds_from_wire,
};
use crate::coordinator::CoordinatorRequest;
use crate::failure::{Failure, FailureKind};
use crate::move_slot::MoveSlot;
use crate::planner::{
    JointPath, Outcome, PathGoal, PathOrder, ShareReply, ShareReport, cancelled_reason,
};
use crate::settle::{ContactGap, Settle};
use crate::streams::LeaderActivity;

/// An `execute_joint_path` goal parsed off the wire: the path, the expected
/// start of each arm and the settle each arm's share ends by, these two in
/// the order of the path's arms, and the goal's contact gap.
struct ExecuteJointPathGoal {
    path: JointPath,
    expected_starts: Vec<Option<JointVec>>,
    settle: Settle<Vec<f64>>,
    contact: ContactGap,
}

/// An `execute_joint_path` goal as the path of its arms, in the order of
/// its `arms`: each named once, every arm through as many waypoints, each
/// waypoint finite and inside its arm's joint limits, each expected start
/// empty or a finite position of each joint, each settle tolerance
/// positive; the duration, and the settle timeout.
fn execute_joint_path_goal(
    d: &execute_joint_path::GoalRequestData,
    limits: &PerArm<[Limit; ARM_DOF], ARMS>,
) -> std::result::Result<ExecuteJointPathGoal, String> {
    let names: Vec<&str> = d.arms.iter().map(|arm| arm.arm_name.as_str()).collect();
    let order = ArmOrder::parse(&names)?;
    let arms: Vec<(ArmId, &execute_joint_path::GoalArmsItem)> =
        order.ids().iter().copied().zip(&d.arms).collect();
    let counts: Vec<usize> = d.arms.iter().map(|arm| arm.waypoints.len()).collect();
    let count = order.waypoint_count(&counts)?;
    let taken = arms
        .iter()
        .map(|(arm, item)| {
            item.waypoints
                .iter()
                .enumerate()
                .map(|(k, waypoint)| {
                    waypoint_within_limits(*arm, k, &waypoint.joint_positions, &limits[*arm])
                })
                .collect::<std::result::Result<Vec<JointVec>, String>>()
        })
        .collect::<std::result::Result<Vec<Vec<JointVec>>, String>>()?;
    let waypoints = (0..count)
        .map(|k| taken.iter().flat_map(|arm| arm[k]).collect())
        .collect();
    let expected_starts = arms
        .iter()
        .map(|(arm, item)| expected_start(*arm, &item.expected_start_joint_positions))
        .collect::<std::result::Result<Vec<Option<JointVec>>, String>>()?;
    let tolerance = arms
        .iter()
        .map(|(arm, item)| {
            joint_tolerances_from_wire(item.settle_tolerances_rad)
                .map_err(|reason| format!("{}: {reason}", LIMBS.arm_name(*arm)))
        })
        .collect::<std::result::Result<Vec<JointVec>, String>>()?
        .concat();
    Ok(ExecuteJointPathGoal {
        path: JointPath {
            arms: order,
            waypoints,
            duration: move_seconds_from_wire("duration_s", d.duration_s)?,
        },
        expected_starts,
        settle: Settle {
            tolerance,
            timeout: move_seconds_from_wire("settle_timeout_s", d.settle_timeout_s)?,
        },
        contact: contact_gap_from_wire(d.contact_gap_m)?,
    })
}

/// `q`, waypoint `k` (from 0) of the arm `arm`, inside `limits`
/// ([`take_targets`]), or its refusal where it is not finite or puts a
/// joint outside `limits`, naming the waypoint and the joint.
fn waypoint_within_limits(
    arm: ArmId,
    k: usize,
    q: &JointVec,
    limits: &[Limit; ARM_DOF],
) -> std::result::Result<JointVec, String> {
    let waypoint = format!("waypoint {} of {}", k + 1, LIMBS.arm_name(arm));
    if !q.iter().all(|v| v.is_finite()) {
        return Err(format!("{waypoint} holds a non-finite value"));
    }
    take_targets(limits, q).map_err(|outside| format!("{waypoint} puts {outside}"))
}

/// The expected start of the arm `arm` off the wire: none when
/// `joints` is empty, else a finite position of each joint, or the refusal
/// naming the arm.
fn expected_start(arm: ArmId, joints: &[f64]) -> std::result::Result<Option<JointVec>, String> {
    if joints.is_empty() {
        return Ok(None);
    }
    let field = format!("expected_start_joint_positions of {}", LIMBS.arm_name(arm));
    let start: JointVec = joints.try_into().map_err(|_| {
        format!(
            "{field} holds {} values, where the arm has {ARM_DOF} joints: give none, or one per \
             joint",
            joints.len()
        )
    })?;
    if !start.iter().all(|v| v.is_finite()) {
        return Err(format!("{field} holds a non-finite value"));
    }
    Ok(Some(start))
}

/// Admit a goal on the arms `arms` names: each free in the goal's order
/// ([`check_limb_free`]), then each claimed ([`claim_slots`]), or refused
/// with `precondition_failed` for the first busy arm. On a goal of both arms
/// a refusal names the arm ([`PathOrder::failure_of`]).
fn admit_arms(
    arms: &ArmOrder,
    fresh: &PerArm<Arc<AtomicBool>, ARMS>,
    activity: &PerArm<LeaderActivity, ARMS>,
    move_slots: &PerArm<MoveSlot, ARMS>,
) -> std::result::Result<(), Failure> {
    for &arm in arms.ids() {
        check_limb_free(
            fresh[arm].load(Ordering::Acquire),
            activity[arm].streaming(),
            &ARM_WORDS,
        )
        .map_err(|refusal| arms.failure_of(arm, refusal))?;
    }
    claim_slots(arms.ids(), move_slots).map_err(|busy| {
        let refusal = Failure::new(FailureKind::PreconditionFailed, ARM_WORDS.busy);
        arms.failure_of(busy, refusal)
    })
}

/// How an execute_joint_path goal ends: completed or cancelled for a reason,
/// with the share report of each arm in the order of the goal's arms, or
/// failed.
#[derive(Debug, PartialEq)]
enum PathEnd {
    Completed(Vec<ShareReport>),
    Cancelled(String, Vec<ShareReport>),
    Failed(Failure),
}

/// Judge the end of a goal of the arms `arms` names from the share reports
/// in the order they came back, and `cancel`, the reason of the goal's own
/// cancel. A goal whose every arm reported ends cancelled for its own cancel
/// whatever the shares say, then cancelled for the stop's reason when the
/// stop ended a share; otherwise it completes when every share completed or
/// ended on a contact, and fails with the failure that came back first. A share that fails
/// reports before the shares its failure ends, so that failure is the one
/// that ended the path. A goal one of whose arms never reported fails as a
/// planner that dropped the move. A goal that completes or ends cancelled
/// gives the reports in the order of `arms`.
fn summarize(arms: &ArmOrder, reports: Vec<ShareReport>, cancel: Option<String>) -> PathEnd {
    if reports.len() < arms.ids().len() {
        return PathEnd::Failed(Failure::new(
            FailureKind::AttemptFailed,
            "a planner dropped the move",
        ));
    }
    let failure = reports.iter().find_map(|report| match &report.outcome {
        Outcome::Failed(failure) => Some(failure.clone()),
        _ => None,
    });
    let mut reports = reports;
    reports.sort_by_key(|report| arms.ids().iter().position(|&arm| arm == report.arm));
    let stop = reports.iter().find_map(|report| match &report.outcome {
        Outcome::Cancelled(reason) => Some(reason.clone()),
        _ => None,
    });
    if let Some(reason) = cancel.or(stop) {
        return PathEnd::Cancelled(reason, reports);
    }
    match failure {
        Some(failure) => PathEnd::Failed(failure),
        None => PathEnd::Completed(reports),
    }
}

/// The result of a goal that completes or ends cancelled with `reports`: each
/// arm's joints in their order with whether its own contact ended the move,
/// and the time since the path started.
fn path_result(reports: &[ShareReport]) -> (Vec<ExecuteJointPathResultArmsItem>, f64) {
    let arms = reports
        .iter()
        .map(|report| ExecuteJointPathResultArmsItem {
            arm_name: LIMBS.arm_name(report.arm).to_string(),
            final_joint_positions: report.joints,
            ended_on_contact: report.outcome.ended_on_contact(),
        })
        .collect();
    let action_time = reports
        .iter()
        .map(|report| report.elapsed_s)
        .fold(0.0, f64::max);
    (arms, action_time)
}

/// The result of a goal that ends cancelled with `reports`: as
/// [`path_result`] gives it, and with no arm on contact, as limb_motion has a
/// cancelled move say.
fn cancelled_path_result(reports: &[ShareReport]) -> (Vec<ExecuteJointPathResultArmsItem>, f64) {
    let (arms, action_time) = path_result(reports);
    let arms = arms
        .into_iter()
        .map(|arm| ExecuteJointPathResultArmsItem {
            ended_on_contact: false,
            ..arm
        })
        .collect();
    (arms, action_time)
}

/// The reason a cancelled goal ends with: `reason`, with the note that the
/// result comes from the joints commanded last when an arm of `reports` had
/// no fresh measurement ([`cancelled_reason`]).
fn path_cancel_reason(reason: String, reports: &[ShareReport]) -> String {
    let unmeasured = reports.iter().find(|report| report.measured.is_err());
    match unmeasured {
        Some(report) => cancelled_reason(reason, &report.measured),
        None => reason,
    }
}

/// Hand `goal`, whose arms' slots its admission claimed in `move_slots`, to
/// the coordinator, waiting for it however long the coordinator takes, and
/// give how the goal ends: the coordinator's failure when it does not start
/// the path, else as [`summarize`] judges the share reports. `cancel_signal`
/// resolves on the goal's cancel, and `cancel_reason` reads its reason once
/// the shares ended.
async fn run_path(
    requests: &mpsc::Sender<CoordinatorRequest>,
    goal: ExecuteJointPathGoal,
    move_slots: &PerArm<MoveSlot, ARMS>,
    cancel_signal: impl Future<Output = ()>,
    cancel_reason: impl FnOnce() -> Option<String>,
) -> PathEnd {
    let ExecuteJointPathGoal {
        path,
        expected_starts,
        settle,
        contact,
    } = goal;
    let arms = path.arms.clone();
    let shares = arms.ids().len();
    let cancelled = Arc::new(AtomicBool::new(false));
    let (done_tx, mut done_rx) = mpsc::channel(shares);
    let goal = PathGoal {
        move_slots: arms
            .ids()
            .iter()
            .map(|&arm| move_slots[arm].release_on_drop())
            .collect(),
        path,
        expected_starts,
        settle,
        contact,
        reply: ShareReply {
            done_tx,
            cancelled: cancelled.clone(),
            failed: Arc::new(AtomicBool::new(false)),
        },
    };
    let (started, answer) = oneshot::channel();
    let gone = || Failure::new(FailureKind::AttemptFailed, "the coordinator is not running");
    if requests
        .send(CoordinatorRequest::StartPath { goal, started })
        .await
        .is_err()
    {
        return PathEnd::Failed(gone());
    }
    match answer.await {
        Ok(Ok(())) => {}
        Ok(Err(refusal)) => return PathEnd::Failed(refusal),
        Err(_) => return PathEnd::Failed(gone()),
    }
    let reports = collect_ends(shares, &mut done_rx, cancel_signal, &cancelled).await;
    summarize(&arms, reports, cancel_reason())
}

/// Expose `execute_joint_path`: parse each goal and claim its arms, and run
/// each accepted goal on its own task ([`run_path`]), so a goal on arms
/// another goal moves is refused while that goal runs, and goals on
/// different arms run side by side.
pub async fn run_execute_joint_path(
    runner: Arc<NodeRunner>,
    move_slots: PerArm<MoveSlot, ARMS>,
    fresh: PerArm<Arc<AtomicBool>, ARMS>,
    activity: PerArm<LeaderActivity, ARMS>,
    limits: PerArm<[Limit; ARM_DOF], ARMS>,
    requests: mpsc::Sender<CoordinatorRequest>,
) -> Result<()> {
    let mut handle = execute_joint_path::ActionHandle::expose(&runner).await?;
    loop {
        let accepted = handle
            .handle_goal_next_request(|req| {
                let admitted = execute_joint_path_goal(&req.data, &limits)
                    .map_err(|reason| Failure::new(FailureKind::Rejected, reason))
                    .and_then(|goal| admit_arms(&goal.path.arms, &fresh, &activity, &move_slots));
                Ok(match admitted {
                    Ok(()) => execute_joint_path::GoalDecision::accept(),
                    Err(f) => execute_joint_path::GoalDecision::refuse(f.kind.into(), f.message),
                })
            })
            .await?;
        let Some(ctx) = accepted else { return Ok(()) };
        let goal =
            execute_joint_path_goal(&ctx.request().data, &limits).expect("validated on accept");
        let requests = requests.clone();
        let move_slots = move_slots.clone();
        tokio::spawn(async move {
            let ending = run_path(&requests, goal, &move_slots, ctx.cancel_signal(), || {
                ctx.cancel_cause().map(|cause| cause.reason().to_string())
            })
            .await;
            let ended = match ending {
                PathEnd::Failed(failure) => ctx.fail(failure.kind.into(), failure.message).await,
                PathEnd::Completed(reports) => {
                    let (arms, action_time) = path_result(&reports);
                    ctx.complete(arms, action_time).await
                }
                PathEnd::Cancelled(reason, reports) => {
                    let (arms, action_time) = cancelled_path_result(&reports);
                    let reason = path_cancel_reason(reason, &reports);
                    ctx.complete_cancelled(reason, arms, action_time).await
                }
            };
            if let Err(e) = ended {
                error!("execute_joint_path: end: {e}");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{LEFT, RIGHT};

    use execute_joint_path::{GoalArmsItem, GoalArmsItemWaypointsItem, GoalRequestData};

    use control_core::positive_finite::PositiveFinite;

    use crate::failure::stale_failure;
    use crate::planner::OTHER_SHARE_FAILED;
    use srs_model::TARGET_SLACK;

    use crate::robot::arm_in;
    use crate::test_support::{V1, bundled_arm};

    /// Both arms' joints, the left arm's `left` in every joint and the right
    /// arm's `right`. From 0.05 to 0.17 rad, every joint of [`V1`] lies within
    /// its limits on both arms.
    fn joints(left: f64, right: f64) -> PerArm<JointVec, ARMS> {
        PerArm::new([[left; ARM_DOF], [right; ARM_DOF]])
    }

    /// v1's joint limits of both arms.
    fn limits() -> PerArm<[Limit; ARM_DOF], ARMS> {
        PerArm::from_fn(|arm| bundled_arm(V1, arm).limits())
    }

    /// The goal of the arm `arm_name` through `waypoints`, expecting its
    /// start at `expected_start` when given, its settle tolerances 0.03 rad
    /// on every joint. An arm name of no arm takes the left arm's joints.
    fn arm(
        arm_name: &str,
        waypoints: &[PerArm<JointVec, ARMS>],
        expected_start: Option<PerArm<JointVec, ARMS>>,
    ) -> GoalArmsItem {
        let arm = arm_in("arm_name", arm_name).unwrap_or(LEFT);
        GoalArmsItem {
            arm_name: arm_name.to_string(),
            waypoints: waypoints
                .iter()
                .map(|joints| GoalArmsItemWaypointsItem {
                    joint_positions: joints[arm],
                })
                .collect(),
            expected_start_joint_positions: expected_start
                .map_or_else(Vec::new, |joints| joints[arm].to_vec()),
            settle_tolerances_rad: [0.03; ARM_DOF],
        }
    }

    /// A goal of the arms `arm_names` names, in that order, each through
    /// `waypoints` (see [`arm`]).
    fn goal(
        arm_names: &[&str],
        waypoints: &[PerArm<JointVec, ARMS>],
        expected_start: Option<PerArm<JointVec, ARMS>>,
    ) -> GoalRequestData {
        GoalRequestData {
            arms: arm_names
                .iter()
                .map(|name| arm(name, waypoints, expected_start))
                .collect(),
            duration_s: 1.0,
            settle_timeout_s: 0.5,
            contact_gap_m: 0.0,
        }
    }

    /// A path of the arms in the order the goal's arms name them: named right
    /// first, each waypoint, the expected starts and the settle give each arm
    /// its own part of the goal.
    #[test]
    fn each_arm_gives_its_part_of_the_path_in_the_order_of_the_goals_arms() {
        let waypoints = [joints(0.10, 0.12), joints(0.14, 0.16)];
        let start = joints(0.06, 0.08);
        for names in [["left_arm", "right_arm"], ["right_arm", "left_arm"]] {
            let request = goal(&names, &waypoints, Some(start));
            let parsed =
                execute_joint_path_goal(&request, &limits()).expect("a goal within the limits");
            let arms = &parsed.path.arms;
            let named = names.map(|name| arm_in("arm_name", name).expect("an arm"));
            assert_eq!(arms.ids(), named);
            for (parsed, sent) in parsed.path.waypoints.iter().zip(&waypoints) {
                let in_order: Vec<f64> = named.iter().flat_map(|&arm| sent[arm]).collect();
                assert_eq!(*parsed, in_order);
            }
            assert_eq!(parsed.expected_starts, named.map(|arm| Some(start[arm])));
            assert_eq!(parsed.settle.tolerance, vec![0.03; 2 * ARM_DOF]);
        }
    }

    /// A waypoint whose elbow lies a hair under its lower limit, as a
    /// measured pose of the straight arm can, runs at the limit; its other
    /// joints run as sent.
    #[test]
    fn a_waypoint_a_hair_outside_its_limits_runs_at_the_limit() {
        let elbow = limits()[LEFT][3];
        let mut waypoints = [joints(0.10, 0.12)];
        waypoints[0][LEFT][3] = elbow.lo - TARGET_SLACK / 2.0;
        let request = goal(&["left_arm"], &waypoints, None);
        let parsed = execute_joint_path_goal(&request, &limits()).expect("a goal at the limits");
        let mut expected = [0.10; ARM_DOF];
        expected[3] = elbow.lo;
        assert_eq!(parsed.path.waypoints, [expected.to_vec()]);
    }

    /// A goal of one arm moves that arm alone, and an arm with no expected
    /// start is not checked while another arm gives one.
    #[test]
    fn a_goal_of_one_arm_moves_it_alone_and_an_empty_expected_start_checks_nothing() {
        let request = goal(&["right_arm"], &[joints(0.10, 0.12)], None);
        let parsed = execute_joint_path_goal(&request, &limits()).expect("a goal of the right arm");
        assert_eq!(parsed.path.arms.ids(), [RIGHT]);
        assert_eq!(parsed.path.waypoints, [vec![0.12; ARM_DOF]]);
        assert_eq!(parsed.expected_starts, [None]);

        let mut both = goal(&["left_arm", "right_arm"], &[joints(0.10, 0.12)], None);
        both.arms[1] = arm("right_arm", &[joints(0.10, 0.12)], Some(joints(0.0, 0.07)));
        let parsed = execute_joint_path_goal(&both, &limits()).expect("a goal of both arms");
        assert_eq!(parsed.expected_starts, [None, Some([0.07; ARM_DOF])]);
    }

    /// Each value the goal does not take is refused, naming what to change:
    /// no arm, an arm of no robot, an arm named twice, no waypoint, arms
    /// through different numbers of waypoints, a non-finite value, a joint
    /// past its limit, an expected start of the wrong length, a tolerance
    /// that is not positive, and a time past the ceiling.
    #[test]
    fn a_goal_the_backbone_does_not_take_is_refused_naming_its_field() {
        let waypoints = [joints(0.10, 0.12), joints(0.14, 0.16)];
        let both = ["left_arm", "right_arm"];
        let refusal = |goal: GoalRequestData| {
            execute_joint_path_goal(&goal, &limits())
                .err()
                .expect("the goal is refused")
        };
        assert!(refusal(goal(&[], &waypoints, None)).starts_with("arms holds no arm"));
        assert_eq!(
            refusal(goal(&["left_arm", "left_arm"], &waypoints, None)),
            "arms names \"left_arm\" twice: name each arm once"
        );
        assert!(refusal(goal(&["left_arm", "head"], &waypoints, None)).contains("\"head\""));
        assert_eq!(
            refusal(goal(&both, &[], None)),
            "left_arm holds no waypoint: a joint path passes through one or more"
        );

        let mut uneven = goal(&both, &waypoints, None);
        uneven.arms[1].waypoints.pop();
        assert_eq!(
            refusal(uneven),
            "right_arm holds 1 waypoints, where left_arm holds 2: every arm passes through as \
             many"
        );

        let mut not_finite = goal(&both, &waypoints, None);
        not_finite.arms[1].waypoints[0].joint_positions[3] = f64::NAN;
        assert_eq!(
            refusal(not_finite),
            "waypoint 1 of right_arm holds a non-finite value"
        );
        // v1's right shoulder and both elbows stop above -1 rad.
        let past_right = [waypoints[0], joints(0.14, -1.0)];
        let refused = refusal(goal(&["right_arm", "left_arm"], &past_right, None));
        assert!(
            refused.starts_with("waypoint 2 of right_arm puts joint ")
                && refused.contains(" at -1 rad, outside its limits ["),
            "{refused}"
        );

        let mut short_start = goal(&both, &waypoints, None);
        short_start.arms[0].expected_start_joint_positions = vec![0.1; ARM_DOF - 1];
        assert_eq!(
            refusal(short_start),
            "expected_start_joint_positions of left_arm holds 6 values, where the arm has 7 \
             joints: give none, or one per joint"
        );
        let mut start_not_finite = goal(&both, &waypoints, None);
        start_not_finite.arms[1].expected_start_joint_positions = vec![f64::INFINITY; ARM_DOF];
        assert_eq!(
            refusal(start_not_finite),
            "expected_start_joint_positions of right_arm holds a non-finite value"
        );

        let mut zero_tolerance = goal(&both, &waypoints, None);
        zero_tolerance.arms[1].settle_tolerances_rad[2] = 0.0;
        assert_eq!(
            refusal(zero_tolerance),
            "right_arm: each settle_tolerances_rad entry must be a positive number"
        );
        let mut slow = goal(&both, &waypoints, None);
        slow.duration_s = 61.0;
        assert!(refusal(slow).starts_with("duration_s must be"));
        let mut late = goal(&both, &waypoints, None);
        late.settle_timeout_s = -1.0;
        assert!(refusal(late).starts_with("settle_timeout_s must be"));
        let mut negative_gap = goal(&both, &waypoints, None);
        negative_gap.contact_gap_m = -0.005;
        assert_eq!(
            refusal(negative_gap),
            "contact_gap_m must be 0 or a positive number of meters"
        );
    }

    /// The goal's contact gap is the gap of every arm of its path, and a gap
    /// of 0 asks for no contact check.
    #[test]
    fn the_contact_gap_of_a_goal_is_parsed_off_its_one_field() {
        let both = ["left_arm", "right_arm"];
        let mut request = goal(&both, &[joints(0.10, 0.12)], None);
        let parsed = |request: &GoalRequestData| {
            execute_joint_path_goal(request, &limits()).expect("a goal within the limits")
        };
        assert_eq!(parsed(&request).contact, ContactGap::NONE);
        request.contact_gap_m = 0.005;
        let gap = PositiveFinite::parse(0.005).expect("a positive gap");
        assert_eq!(parsed(&request).contact, ContactGap::of(gap));
    }

    /// A goal on an arm whose follower is stale, that a leader streams, or
    /// that runs another move is refused, naming the arm on a goal of both
    /// arms and in the words of one arm on a goal of that arm alone, and
    /// leaves every slot as it found it; a goal of free arms claims them.
    #[test]
    fn a_goal_on_an_arm_that_is_not_free_is_refused_naming_the_arm() {
        let both = ArmOrder::parse(&["left_arm", "right_arm"]).expect("both arms");
        let right = ArmOrder::parse(&["right_arm"]).expect("the right arm");
        let move_slots = PerArm::from_fn(|_| MoveSlot::default());
        let fresh = PerArm::from_fn(|_| Arc::new(AtomicBool::new(true)));
        let idle = PerArm::from_fn(|_| LeaderActivity::default());

        let right_stale = PerArm::new([
            Arc::new(AtomicBool::new(true)),
            Arc::new(AtomicBool::new(false)),
        ]);
        let stale = Failure::new(FailureKind::ResourceUnavailable, ARM_WORDS.no_fresh_state);
        assert_eq!(
            admit_arms(&both, &right_stale, &idle, &move_slots),
            Err(stale.clone().labelled(LIMBS.arm_name(RIGHT)))
        );
        assert_eq!(
            admit_arms(&right, &right_stale, &idle, &move_slots),
            Err(stale)
        );

        let left_streamed = PerArm::from_fn(|_| LeaderActivity::default());
        left_streamed[LEFT].note();
        assert_eq!(
            admit_arms(&both, &fresh, &left_streamed, &move_slots),
            Err(
                Failure::new(FailureKind::LeaderStreaming, ARM_WORDS.leader_streaming)
                    .labelled(LIMBS.arm_name(LEFT))
            )
        );
        assert!(
            move_slots.iter().all(|slot| !slot.is_claimed()),
            "a refused goal claims no arm"
        );

        let right_busy = PerArm::from_fn(|_| MoveSlot::default());
        assert!(
            right_busy[RIGHT].claim(),
            "another move holds the right arm"
        );
        assert_eq!(
            admit_arms(&both, &fresh, &idle, &right_busy),
            Err(Failure::new(
                FailureKind::PreconditionFailed,
                "right_arm: arm is already executing a motion"
            ))
        );
        assert!(
            !right_busy[LEFT].is_claimed(),
            "the refusal releases the left arm it claimed"
        );
        assert_eq!(
            admit_arms(&right, &fresh, &idle, &right_busy),
            Err(Failure::new(
                FailureKind::PreconditionFailed,
                ARM_WORDS.busy
            ))
        );

        assert_eq!(admit_arms(&both, &fresh, &idle, &move_slots), Ok(()));
        assert!(move_slots.iter().all(MoveSlot::is_claimed));
    }

    /// A report of the arm `arm` ending on `outcome`, `elapsed_s` after
    /// its path started, measured at `joints` when `fresh`.
    fn report(arm: ArmId, outcome: Outcome, joints: JointVec, fresh: bool) -> ShareReport {
        ShareReport {
            arm,
            outcome,
            joints,
            measured: if fresh {
                Ok(joints)
            } else {
                Err(crate::planner::Unmeasured::Stale)
            },
            elapsed_s: 1.5,
        }
    }

    /// A goal of both arms whose right follower goes stale fails with the
    /// stale arm's failure, which comes back before the failure it causes on
    /// the left arm's share, though the goal names the left arm first.
    #[test]
    fn a_goal_fails_with_the_failure_that_ended_its_path() {
        let arms = ArmOrder::parse(&["left_arm", "right_arm"]).expect("both arms");
        let stale = stale_failure().labelled(LIMBS.arm_name(RIGHT));
        let caused = Failure::new(FailureKind::AttemptFailed, OTHER_SHARE_FAILED)
            .labelled(LIMBS.arm_name(LEFT));
        let arrived = vec![
            report(RIGHT, Outcome::Failed(stale.clone()), [0.2; ARM_DOF], false),
            report(LEFT, Outcome::Failed(caused), [0.1; ARM_DOF], true),
        ];
        assert_eq!(summarize(&arms, arrived, None), PathEnd::Failed(stale));
    }

    /// The goal of both arms completes once both shares complete, its
    /// result in the order of the goal's arms whatever order the shares
    /// ended in; a share the stop ended ends it cancelled for the stop's
    /// reason; the goal's own cancel wins over every share; a failure fails
    /// it; and a share that never reported fails it as a planner that
    /// dropped the move.
    #[test]
    fn a_goal_ends_as_its_shares_report_it_in_the_order_of_its_arms() {
        let right_first = ArmOrder::parse(&["right_arm", "left_arm"]).expect("both arms");
        let (left, right) = ([0.1; ARM_DOF], [0.2; ARM_DOF]);
        let completed = vec![
            report(LEFT, Outcome::Complete, left, true),
            report(RIGHT, Outcome::Complete, right, true),
        ];
        let PathEnd::Completed(reports) = summarize(&right_first, completed.clone(), None) else {
            panic!("both shares completed");
        };
        let named: Vec<(String, JointVec)> = path_result(&reports)
            .0
            .into_iter()
            .map(|arm| (arm.arm_name, arm.final_joint_positions))
            .collect();
        assert_eq!(
            named,
            [
                ("right_arm".to_string(), right),
                ("left_arm".to_string(), left)
            ]
        );
        assert_eq!(path_result(&reports).1, 1.5);

        let stopped = "stopped: operator".to_string();
        let stopped_reports = vec![
            report(LEFT, Outcome::Cancelled(stopped.clone()), left, true),
            report(RIGHT, Outcome::Cancelled(stopped.clone()), right, false),
        ];
        let PathEnd::Cancelled(reason, reports) =
            summarize(&right_first, stopped_reports.clone(), None)
        else {
            panic!("the stop ended both shares");
        };
        assert_eq!(reason, stopped);
        assert_eq!(
            path_cancel_reason(reason, &reports),
            "stopped: operator; the result comes from the joints commanded last"
        );
        let caller = "cancelled by its caller".to_string();
        assert!(matches!(
            summarize(&right_first, stopped_reports, Some(caller.clone())),
            PathEnd::Cancelled(reason, _) if reason == caller
        ));

        // The left arm's contact completes the goal for both arms, and the
        // result says whose contact it was.
        let contact_reports = vec![
            report(LEFT, Outcome::Contact { this_arm: true }, left, true),
            report(RIGHT, Outcome::Contact { this_arm: false }, right, true),
        ];
        let PathEnd::Completed(reports) = summarize(&right_first, contact_reports.clone(), None)
        else {
            panic!("a contact completes the goal");
        };
        let on_contact: Vec<(String, bool)> = path_result(&reports)
            .0
            .into_iter()
            .map(|arm| (arm.arm_name, arm.ended_on_contact))
            .collect();
        assert_eq!(
            on_contact,
            [
                ("right_arm".to_string(), false),
                ("left_arm".to_string(), true)
            ]
        );
        assert!(
            path_result(&completed)
                .0
                .iter()
                .all(|arm| !arm.ended_on_contact),
            "a path that settled ended on no contact"
        );
        // The goal's own cancel wins over the shares, and a cancelled goal
        // says no arm ended on contact.
        let PathEnd::Cancelled(_, reports) =
            summarize(&right_first, contact_reports, Some(caller.clone()))
        else {
            panic!("the goal's own cancel ends it cancelled");
        };
        assert!(
            cancelled_path_result(&reports)
                .0
                .iter()
                .all(|arm| !arm.ended_on_contact),
            "a cancelled goal ended on no contact"
        );

        let failure = Failure::new(FailureKind::Timeout, "left_arm: still short");
        let one_failed = vec![
            report(LEFT, Outcome::Failed(failure.clone()), left, true),
            report(RIGHT, Outcome::Complete, right, true),
        ];
        assert_eq!(
            summarize(&right_first, one_failed, None),
            PathEnd::Failed(failure)
        );
        assert_eq!(
            summarize(&right_first, completed[..1].to_vec(), None),
            PathEnd::Failed(Failure::new(
                FailureKind::AttemptFailed,
                "a planner dropped the move"
            ))
        );
    }
}
