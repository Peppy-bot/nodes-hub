//! plan_cartesian_path over the generated harness, as a caller sees it: a path that checks
//! collisions waits for the camera's first depth frame and a path that does not answers
//! before it; each waypoint comes with the grasp-point pose limb_state reports there; a
//! line toward a table the camera sees ends before the table and covers its whole length with
//! collisions unchecked; a start in contact fails, a cancel ends the goal cancelled, and every
//! refusal comes at a goal's admission.

mod common;

use chain_kinematics::nalgebra::{Isometry3, Translation3, Vector3};
use motion_planner::node::NO_DEPTH_FRAME;
use openarm_description::{ARM_DOF, Side};
use peppygen::exposed_actions::motion_planning::plan_cartesian_path::GoalWaypointsItem;
use peppygen::fixtures::exposed_actions::motion_planning::plan_cartesian_path::{
    CancelState, Failure, FailureKind, GoalRequestData, ResultData, ResultOutcome, send_goal,
};
use peppygen::fixtures::harness::Harness;

use common::{
    LEFT_READY, OPENARM_V2, OVER_THE_TABLE, RIGHT_READY, Solid, TIMEOUT, UNDER_THE_TABLE, along,
    arms, at_ms, facing_the_robot, looking_at, publish_arms, render, start_with_camera,
    start_without_camera, table_across, take_frame,
};

/// A path of the left arm's grasp point from where it stands through `poses`.
fn path_through(poses: &[Isometry3<f64>], avoid_collisions: bool) -> GoalRequestData {
    GoalRequestData {
        arm_name: "left_arm".to_string(),
        start_joint_positions: Vec::new(),
        link_name: "openarm_left_tcp".to_string(),
        waypoints: poses
            .iter()
            .map(|pose| GoalWaypointsItem {
                position: pose.translation.vector.into(),
                orientation: [
                    pose.rotation.i,
                    pose.rotation.j,
                    pose.rotation.k,
                    pose.rotation.w,
                ],
            })
            .collect(),
        avoid_collisions,
    }
}

/// The grasp point's pose with the left arm at `q`.
fn hand(q: &[f64; ARM_DOF]) -> Isometry3<f64> {
    OPENARM_V2.arms[Side::Left]
        .geometry
        .chain()
        .world_ee_pose(q)
}

/// `pose` moved `m` along z, its orientation kept.
fn raised(pose: &Isometry3<f64>, m: f64) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::from(pose.translation.vector + Vector3::z() * m),
        pose.rotation,
    )
}

/// How a path ends without a result.
#[derive(Debug)]
enum Ended {
    /// The node refused the goal: it took nothing.
    Refused(Failure),
    /// The node took the goal and failed it.
    Failed(Failure),
}

/// How a path ends: its result, or the failure it was refused or failed with.
async fn path(harness: &Harness, goal: &GoalRequestData) -> Result<ResultData, Ended> {
    let handle = send_goal(harness, goal, peppygen::DEFAULT_FEEDBACK_BOUND, TIMEOUT)
        .await
        .expect("the goal is sent");
    if let Some(refusal) = handle.refusal {
        return Err(Ended::Refused(refusal));
    }
    match handle
        .get_result(TIMEOUT)
        .await
        .expect("the result arrives")
        .outcome
    {
        ResultOutcome::Completed(data) => Ok(data),
        ResultOutcome::Failed(failure) => Err(Ended::Failed(failure)),
        other => panic!("the path ends {other:?}"),
    }
}

/// The failure a goal is refused with; panics on any other end.
async fn refusal(harness: &Harness, goal: &GoalRequestData) -> Failure {
    match path(harness, goal).await {
        Err(Ended::Refused(failure)) => failure,
        other => panic!("the goal is refused, not {other:?}"),
    }
}

// Before the camera's first depth frame a path that checks collisions is refused, and one
// that does not covers its line, with no map stamp, each waypoint holding the grasp-point
// pose there. After the frame, the path that checks collisions answers the frame's stamp.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_path_that_checks_collisions_waits_for_a_depth_frame_and_one_that_does_not_answers()
-> peppygen::Result<()> {
    let (harness, mocks) = start_with_camera(&facing_the_robot()).await?;
    publish_arms(&mocks, at_ms(0), &arms(LEFT_READY, RIGHT_READY)).await?;
    let poses = [raised(&hand(&LEFT_READY), 0.05)];

    let refused = refusal(&harness, &path_through(&poses, true)).await;
    assert!(
        refused.kind == FailureKind::PreconditionFailed
            && refused.message.starts_with(NO_DEPTH_FRAME),
        "{refused}"
    );

    let unchecked = path(&harness, &path_through(&poses, false))
        .await
        .expect("a path");
    assert_eq!(unchecked.fraction, 1.0);
    assert_eq!(unchecked.map_stamp, None);
    assert_eq!(unchecked.start_joint_positions, LEFT_READY.to_vec());
    for waypoint in &unchecked.waypoints {
        let q: [f64; ARM_DOF] = waypoint.joint_positions.as_slice().try_into().expect("7");
        let pose = hand(&q);
        assert_eq!(waypoint.position, <[f64; 3]>::from(pose.translation.vector));
        let rotation = pose.rotation;
        assert_eq!(
            waypoint.orientation,
            [rotation.i, rotation.j, rotation.k, rotation.w]
        );
    }

    let nothing = vec![None; (common::WIDTH * common::HEIGHT) as usize];
    take_frame(&harness, &mocks, at_ms(0), &nothing, LEFT_READY).await?;
    let checked = path(&harness, &path_through(&poses, true))
        .await
        .expect("a path");
    assert_eq!(checked.fraction, 1.0);
    assert_eq!(checked.map_stamp, Some(at_ms(0)));
    harness.shutdown().await
}

// A line down toward a table the camera sees ends where the arm comes to the clearance from
// the table, past its start; with collisions unchecked it covers the whole line.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_line_toward_a_table_ends_before_it_unless_collisions_go_unchecked()
-> peppygen::Result<()> {
    let start = hand(&OVER_THE_TABLE);
    let here = start.translation.vector;
    let top = here.z - 0.2;
    let table: Solid = (
        [here.x - 0.2, here.y - 0.2, top - 0.1],
        [here.x + 0.2, here.y + 0.2, top],
    );
    let camera = looking_at(
        Vector3::new(here.x + 0.5, here.y, top + 0.7),
        Vector3::new(here.x, here.y, top),
    );
    let (harness, mocks) = start_with_camera(&camera).await?;
    publish_arms(&mocks, at_ms(0), &arms(OVER_THE_TABLE, RIGHT_READY)).await?;
    take_frame(
        &harness,
        &mocks,
        at_ms(0),
        &render(&camera, None, &[table]),
        OVER_THE_TABLE,
    )
    .await?;
    let poses = [raised(&start, -0.15)];

    let blocked = path(&harness, &path_through(&poses, true))
        .await
        .expect("a path");
    assert!(
        blocked.fraction > 0.0 && blocked.fraction < 1.0,
        "the line toward the table covers {}",
        blocked.fraction
    );

    let unchecked = path(&harness, &path_through(&poses, false))
        .await
        .expect("a path");
    assert_eq!(unchecked.fraction, 1.0);
    harness.shutdown().await
}

// A start in contact with a table the camera sees fails once taken, naming the pair; with
// collisions unchecked, the same start walks.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_start_in_contact_fails_unless_collisions_go_unchecked() -> peppygen::Result<()> {
    let (table, crossing_t) = table_across(&UNDER_THE_TABLE, &OVER_THE_TABLE);
    let ([x0, y0, _], [x1, y1, top]) = table;
    let (x, y) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let camera = looking_at(Vector3::new(x + 0.5, y, top + 0.7), Vector3::new(x, y, top));
    let (harness, mocks) = start_with_camera(&camera).await?;
    publish_arms(&mocks, at_ms(0), &arms(UNDER_THE_TABLE, RIGHT_READY)).await?;
    take_frame(
        &harness,
        &mocks,
        at_ms(0),
        &render(&camera, None, &[table]),
        UNDER_THE_TABLE,
    )
    .await?;
    let crossing = along(&UNDER_THE_TABLE, &OVER_THE_TABLE, crossing_t);
    let from_contact = |avoid_collisions: bool| GoalRequestData {
        start_joint_positions: crossing.to_vec(),
        ..path_through(&[raised(&hand(&crossing), 0.05)], avoid_collisions)
    };
    let Err(Ended::Failed(failed)) = path(&harness, &from_contact(true)).await else {
        panic!("a path from a start in contact fails once taken");
    };
    assert_eq!(failed.kind, FailureKind::StartInCollision, "{failed}");
    assert!(
        failed.message.contains("touches the occupancy map"),
        "{failed}"
    );
    assert!(
        path(&harness, &from_contact(false)).await.is_ok(),
        "the same start with collisions unchecked walks"
    );
    harness.shutdown().await
}

/// Lines of 1 cm the next test sends: at each step's work, a run far longer than a cancel takes
/// to arrive on any host.
const MANY_LINES: usize = 20_000;

// A cancel of a path of many lines ends the goal cancelled.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cancel_ends_the_goal_cancelled() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(LEFT_READY, RIGHT_READY)).await?;
    let here = hand(&LEFT_READY);
    let there = raised(&here, 0.01);
    let poses: Vec<Isometry3<f64>> = (0..MANY_LINES)
        .map(|k| if k % 2 == 0 { there } else { here })
        .collect();
    let handle = send_goal(
        &harness,
        &path_through(&poses, false),
        peppygen::DEFAULT_FEEDBACK_BOUND,
        TIMEOUT,
    )
    .await?;
    assert!(handle.accepted, "{:?}", handle.refusal);
    assert_eq!(handle.cancel_goal(TIMEOUT).await?, CancelState::Signalled);
    let outcome = handle.get_result(TIMEOUT).await?.outcome;
    assert!(
        matches!(outcome, ResultOutcome::Cancelled { .. }),
        "{outcome:?}"
    );
    harness.shutdown().await
}

// Every refusal comes at a goal's admission: rejected for a link that is not fixed to the grasp
// link and for no pose to pass through, and precondition_failed for a start outside a joint
// limit by more than the start tolerance.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_refusal_comes_before_acceptance() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(LEFT_READY, RIGHT_READY)).await?;
    let inner = GoalRequestData {
        link_name: "openarm_left_link3".to_string(),
        ..path_through(&[hand(&LEFT_READY)], true)
    };
    let refused = refusal(&harness, &inner).await;
    assert_eq!(refused.kind, FailureKind::Rejected, "{refused}");
    let refused = refusal(&harness, &path_through(&[], true)).await;
    assert_eq!(refused.kind, FailureKind::Rejected, "{refused}");
    let mut outside = LEFT_READY;
    outside[3] = -1.0;
    let from_outside = GoalRequestData {
        start_joint_positions: outside.to_vec(),
        ..path_through(&[hand(&LEFT_READY)], true)
    };
    let refused = refusal(&harness, &from_outside).await;
    assert_eq!(refused.kind, FailureKind::PreconditionFailed, "{refused}");
    harness.shutdown().await
}
