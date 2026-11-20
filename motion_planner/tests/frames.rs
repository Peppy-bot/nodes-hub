//! Goals, requests and objects in their frames, over the generated harness, against a
//! robot's tree whose `odom` to robot frame transform moves: each record is placed by the
//! tree at its timestamp, a pose in `odom` and the same pose in the robot frame at the time
//! it was read land at one place, and a robot-frame pose with no timestamp lands where the
//! robot frame stands when the goal is taken; a record the tree cannot place is refused
//! with the lookup's failure kind before anything is planned; an object given in odom stays
//! where odom holds it as the base drives, and a depth frame cuts it out where it stood at the
//! frame's stamp; a goal of one arm in the other arm's grasp link is held against that link
//! where the plan leaves it; and a path orientation of an arm the plan leaves standing, in the
//! moving arm's grasp link, is held at each waypoint against where that link stands.

mod common;

use std::time::{Duration, SystemTime};

use chain_kinematics::nalgebra::{Isometry3, Vector3};
use control_core::geometry::pose_arrays;
use peppygen::exposed_actions::motion_planning::plan_cartesian_path::GoalWaypointsItem;
use peppygen::exposed_actions::motion_planning::plan_joint_path::{
    GoalArmsItem, GoalGoalJointConstraintsItem, GoalGoalPositionConstraintsItem,
    GoalPathOrientationConstraintsItem,
};
use peppygen::exposed_services::motion_planning::apply_collision_objects::{
    ApplyCollisionObjectsObjectsItem, ApplyCollisionObjectsObjectsItemShapesItem,
};
use peppygen::fixtures::exposed_actions::motion_planning::plan_cartesian_path;
use peppygen::fixtures::exposed_actions::motion_planning::plan_joint_path::{
    FailureKind, GoalRequestData, ResultData,
};
use peppygen::fixtures::exposed_services::motion_planning::{
    apply_collision_objects, get_planning_scene,
};
use peppygen::fixtures::harness::{Harness, Mocks};
use robot_description::ArmId;

use common::{
    ARM_DOF, Ended, LEFT, OPENARM_V2, PLANNING_TIME_S, RIGHT, TIMEOUT, arms, at_ms, check,
    left_elbow, left_path, left_ready, looking_at, move_frame, next_map_where, over_the_table,
    parameters, plan, plan_to, publish_arms, render, right_ready, robot_frame, start_with_camera,
    start_without_camera, start_without_the_rig, table_across, take_frame, under_the_table,
};

/// The frame that holds still while the robot's base moves.
const ODOM: &str = "odom";

/// How far (m) a position goal's grasp point may end from its target.
const WITHIN_M: f64 = 0.01;

/// The rounding of the arithmetic in a comparison of two placements of one pose (m).
const ROUNDING_M: f64 = 1e-9;

/// When a report read the robot frame, standing at `ODOM`'s origin, and when the robot frame
/// stands where it moved to, `MOVED` along `ODOM`'s x.
fn read_at() -> SystemTime {
    at_ms(0)
}
fn moved_at() -> SystemTime {
    at_ms(500)
}
const MOVED: [f64; 3] = [0.3, 0.0, 0.0];

/// The grasp pose of `arm` of OpenArm v2 at `q`, in the robot frame.
fn grasp(arm: ArmId, q: &[f64; ARM_DOF]) -> Isometry3<f64> {
    OPENARM_V2.arms[arm].geometry.chain().root_ee_pose(q)
}

/// The node over the harness, with no camera, the arms at Ready, and a robot's tree whose
/// robot frame stood at `ODOM`'s origin at `read_at` and at `MOVED` at `moved_at`, once the
/// tree holds both.
async fn moved_base() -> peppygen::Result<(Harness, Mocks)> {
    let (mut harness, mocks) = start_without_the_rig(&arms(left_ready(), right_ready())).await?;
    move_frame(
        &mocks,
        read_at(),
        ODOM,
        &robot_frame(),
        &Isometry3::identity(),
    )
    .await?;
    drive(&mut harness, &mocks, moved_at(), Vector3::from(MOVED)).await?;
    Ok((harness, mocks))
}

/// Publishes the robot frame at `position` in `ODOM` at `at`, and waits until the tree holds
/// it: occupancy_map gives the time of the robot frame's newest place.
async fn drive(
    harness: &mut Harness,
    mocks: &Mocks,
    at: SystemTime,
    position: Vector3<f64>,
) -> peppygen::Result<()> {
    let pose = Isometry3::translation(position.x, position.y, position.z);
    move_frame(mocks, at, ODOM, &robot_frame(), &pose).await?;
    next_map_where(harness, |message| message.timestamp == at).await;
    Ok(())
}

/// A plan of `arm` alone, its grasp point to within `WITHIN_M` of `position` in `frame` at
/// `timestamp`, none for the latest time.
fn grasp_goal(
    arm: ArmId,
    frame: &str,
    timestamp: Option<SystemTime>,
    position: [f64; 3],
) -> GoalRequestData {
    let model = &OPENARM_V2.arms[arm];
    GoalRequestData {
        arms: vec![GoalArmsItem {
            arm_name: model.name.clone(),
            start_joint_positions: Vec::new(),
        }],
        goal_joint_constraints: Vec::new(),
        goal_position_constraints: vec![GoalGoalPositionConstraintsItem {
            link_name: model.grasp_link.clone(),
            frame: Some(frame.to_string()),
            timestamp,
            position,
            tolerance_m: WITHIN_M,
        }],
        goal_orientation_constraints: Vec::new(),
        path_orientation_constraints: Vec::new(),
        planner: "rrt_connect".to_string(),
        refiner: "none".to_string(),
        allowed_planning_time_s: PLANNING_TIME_S,
    }
}

/// The joints of the last waypoint of the answer's arm at `k`.
fn last_waypoint(result: &ResultData, k: usize) -> [f64; ARM_DOF] {
    result.arms[k]
        .waypoints
        .last()
        .expect("a waypoint")
        .joint_positions
        .as_slice()
        .try_into()
        .expect("a position for each joint")
}

/// Asserts that `goal` plans the left arm's grasp point to within `WITHIN_M` of `target`, in
/// the robot frame, and that the answer names the robot frame.
async fn plans_to(harness: &Harness, goal: &GoalRequestData, target: &Vector3<f64>) {
    let result = plan(harness, goal)
        .await
        .unwrap_or_else(|ended| panic!("{ended:?}"));
    assert_eq!(result.frame, robot_frame());
    let reached = grasp(LEFT, &last_waypoint(&result, 0)).translation.vector;
    assert!(
        (reached - target).norm() <= WITHIN_M + ROUNDING_M,
        "the grasp point ends at {reached:?}, {} m from {target:?}",
        (reached - target).norm()
    );
}

// Criterion 4: a place on the floor given in odom at the latest time, the same place given
// in the robot frame at the time it was read, before the base moved, and the robot-frame
// pose of that place with no timestamp, where the robot frame stands when the goal is taken,
// are planned to one place. The robot frame's pose read before the move, taken where the
// robot frame stands now, lies the move's length away.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pose_is_placed_where_its_frame_stood_at_its_time() -> peppygen::Result<()> {
    let (harness, _mocks) = moved_base().await?;
    let target = grasp(LEFT, &left_elbow(1.3)).translation.vector;
    let in_odom: [f64; 3] = (target + Vector3::from(MOVED)).into();
    plans_to(&harness, &grasp_goal(LEFT, ODOM, None, in_odom), &target).await;
    plans_to(
        &harness,
        &grasp_goal(LEFT, &robot_frame(), Some(read_at()), in_odom),
        &target,
    )
    .await;
    plans_to(
        &harness,
        &grasp_goal(LEFT, &robot_frame(), None, target.into()),
        &target,
    )
    .await;
    assert!(
        Vector3::from(in_odom).metric_distance(&target) > 0.2,
        "the read pose taken in the robot frame now lies elsewhere"
    );
    harness.shutdown().await
}

// Criterion 5: a record in a frame the tree does not hold is refused with unknown_frame, and
// a robot-frame record at a time the tree held, which its window has left behind since, with
// extrapolation, at admission, before anything is planned or applied: a goal's constraint, a
// Cartesian path's waypoints and an object of the planning scene alike, the object's refusal
// naming it. The tree holds the robot frame at `read_at` until it takes a place of the robot
// frame more than cache_window_s later and then another, as tf2's buffer drops the stamps past
// the window of its newest when it takes the next.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_record_the_tree_cannot_place_is_refused_before_anything_moves() -> peppygen::Result<()> {
    let (mut harness, mocks) = moved_base().await?;
    let window = Duration::from_secs_f64(parameters().cache_window_s);
    let past_the_window = read_at() + window + Duration::from_secs(1);
    for at in [
        past_the_window,
        past_the_window + Duration::from_millis(500),
    ] {
        drive(&mut harness, &mocks, at, Vector3::from(MOVED)).await?;
    }
    let target: [f64; 3] = grasp(LEFT, &left_elbow(1.3)).translation.vector.into();
    let cases = [
        ("nowhere".to_string(), None, FailureKind::UnknownFrame),
        (robot_frame(), Some(read_at()), FailureKind::Extrapolation),
    ];
    for (frame, timestamp, kind) in cases {
        let goal = match plan(&harness, &grasp_goal(LEFT, &frame, timestamp, target)).await {
            Err(Ended::Refused(failure)) => failure,
            other => panic!("{frame} at {timestamp:?}: a refusal at admission, not {other:?}"),
        };
        assert_eq!(goal.kind, kind, "{goal}");
        assert!(goal.message.contains(&frame), "{goal}");

        let model = &OPENARM_V2.arms[LEFT];
        let (position, orientation) = pose_arrays(&grasp(LEFT, &left_ready()));
        let path = plan_cartesian_path::GoalRequestData {
            arm_name: model.name.clone(),
            start_joint_positions: Vec::new(),
            link_name: model.grasp_link.clone(),
            frame: Some(frame.clone()),
            timestamp,
            waypoints: vec![GoalWaypointsItem {
                position,
                orientation,
            }],
            avoid_collisions: false,
        };
        let handle = plan_cartesian_path::send_goal(
            &harness,
            &path,
            peppygen::DEFAULT_FEEDBACK_BOUND,
            TIMEOUT,
        )
        .await
        .expect("the goal is sent");
        let waypoints = handle.refusal.expect("a refusal at admission");
        assert_eq!(waypoints.kind.to_string(), kind.to_string(), "{waypoints}");

        let object = apply_collision_objects::RequestData {
            objects: vec![table_in(&frame, timestamp, [0.4, 0.0, 0.2])],
        };
        let refused = apply_collision_objects::poll(&harness, &object, TIMEOUT)
            .await
            .map_err(|e| apply_collision_objects::Failure::of(&e).unwrap_or_else(|| panic!("{e}")))
            .expect_err("the object is refused");
        assert_eq!(refused.kind.to_string(), kind.to_string(), "{refused}");
        assert!(
            refused
                .message
                .starts_with("the robot's tree cannot place the object \"table\": "),
            "{refused}"
        );
    }
    let scene = get_planning_scene::poll(&harness, TIMEOUT)
        .await
        .expect("the scene");
    assert!(scene.objects.is_empty(), "{:?}", scene.objects);
    harness.shutdown().await
}

/// An `operation` on the box `id` of `sides` (m), centered at `center` in `frame` at
/// `timestamp`, none for the latest time.
fn box_in(
    operation: &str,
    id: &str,
    (frame, timestamp): (&str, Option<SystemTime>),
    center: [f64; 3],
    sides: [f64; 3],
) -> ApplyCollisionObjectsObjectsItem {
    ApplyCollisionObjectsObjectsItem {
        id: id.to_string(),
        operation: operation.to_string(),
        frame: Some(frame.to_string()),
        timestamp,
        position: Some(center),
        orientation: Some([0.0, 0.0, 0.0, 1.0]),
        shapes: Some(vec![ApplyCollisionObjectsObjectsItemShapesItem {
            kind: "box".to_string(),
            dimensions: sides.to_vec(),
            mesh_vertices: Vec::new(),
            mesh_triangles: Vec::new(),
            position: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],
        }]),
    }
}

/// An add of a 10 cm box, `table`, at `position` in `frame` at `timestamp`, none for the
/// latest time.
fn table_in(
    frame: &str,
    timestamp: Option<SystemTime>,
    position: [f64; 3],
) -> ApplyCollisionObjectsObjectsItem {
    box_in("add", "table", (frame, timestamp), position, [0.1; 3])
}

/// When the robot frame stands `DRIVEN` past `MOVED` along `ODOM`'s x, after the base drove.
fn driven_at() -> SystemTime {
    at_ms(1_000)
}
const DRIVEN: [f64; 3] = [0.5, 0.0, 0.0];

/// Asserts that get_planning_scene answers the one object, `table`, at `expected` in the robot
/// frame, at `at`.
async fn the_table_stands_at(harness: &Harness, at: SystemTime, expected: Vector3<f64>) {
    let scene = get_planning_scene::poll(harness, TIMEOUT)
        .await
        .expect("the scene");
    assert_eq!(
        (scene.frame.as_str(), scene.timestamp),
        (robot_frame().as_str(), at)
    );
    let [table] = scene.objects.as_slice() else {
        panic!("one object: {:?}", scene.objects);
    };
    assert!(
        Vector3::from(table.position).metric_distance(&expected) <= ROUNDING_M,
        "{:?} against {expected:?}",
        table.position
    );
}

// An object given in odom is held where odom holds it, as MoveIt holds a collision object in
// its planning frame, and each check and get_planning_scene place it in the robot frame at the
// latest time. Added `DRIVEN` ahead of the left hand, it stands clear of the arm standing at
// Ready; once the base drives `DRIVEN` forward, it stands on the hand, and a check of the arm
// standing still finds the hand in it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_object_stays_where_odom_holds_it_as_the_base_drives() -> peppygen::Result<()> {
    let (mut harness, mocks) = moved_base().await?;
    let hand = grasp(LEFT, &left_ready()).translation.vector;
    let in_odom: [f64; 3] = (hand + Vector3::from(MOVED) + Vector3::from(DRIVEN)).into();
    let object = apply_collision_objects::RequestData {
        objects: vec![table_in(ODOM, None, in_odom)],
    };
    apply_collision_objects::poll(&harness, &object, TIMEOUT)
        .await
        .expect("the object is added");
    let standing = left_path(&[], &[left_ready()]);
    the_table_stands_at(&harness, moved_at(), hand + Vector3::from(DRIVEN)).await;
    let apart = check(&harness, &standing).await.expect("an answer");
    assert!(apart.clear, "{}", apart.reason);

    let driven = Vector3::from(MOVED) + Vector3::from(DRIVEN);
    drive(&mut harness, &mocks, driven_at(), driven).await?;
    the_table_stands_at(&harness, driven_at(), hand).await;
    let on_the_hand = check(&harness, &standing).await.expect("an answer");
    assert!(!on_the_hand.clear, "the hand stands in the object");
    harness.shutdown().await
}

// A depth frame cuts each object out where it stood in the robot frame at the frame's stamp,
// placed from odom, where the node holds it. A box the size of the table the camera sees, held
// in odom `DRIVEN` ahead of the table, covers it while the base stands `DRIVEN` forward, from
// 1 s to 1.6 s, and stands clear of it at the latest time, 3 s, with the base back where it
// started: three frames taken in that while mark nothing inside the box, so once the box is
// removed the left arm's straight path from under the table to over it passes, and three
// frames taken after that mark the table, which the path then fails on.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_frame_cuts_an_object_out_where_it_stood_at_the_frames_stamp() -> peppygen::Result<()> {
    let (table, _) = table_across(&under_the_table(), &over_the_table());
    let ([x0, y0, z0], [x1, y1, top]) = table;
    let center = Vector3::new((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + top) / 2.0);
    let camera = looking_at(
        Vector3::new(center.x + 0.5, center.y, top + 0.7),
        Vector3::new(center.x, center.y, top),
    );
    let (mut harness, mocks) = start_with_camera(&camera).await?;
    let driven = Isometry3::translation(DRIVEN[0], DRIVEN[1], DRIVEN[2]);
    for (ms, base) in [
        (1_000, driven),
        (1_600, driven),
        (3_000, Isometry3::identity()),
    ] {
        move_frame(&mocks, at_ms(ms), ODOM, &robot_frame(), &base).await?;
    }
    next_map_where(&mut harness, |message| message.timestamp == at_ms(3_000)).await;
    let in_odom = (ODOM, None);
    let ahead: [f64; 3] = (center + Vector3::from(DRIVEN)).into();
    let sides = [x1 - x0, y1 - y0, top - z0];
    let apply = async |item: ApplyCollisionObjectsObjectsItem| {
        let objects = apply_collision_objects::RequestData {
            objects: vec![item],
        };
        apply_collision_objects::poll(&harness, &objects, TIMEOUT).await
    };
    apply(box_in("add", "cover", in_odom, ahead, sides))
        .await
        .expect("the box is added");
    let (start, seen) = (
        arms(under_the_table(), right_ready()),
        render(&camera, None, &[table]),
    );
    let take_three = async |from_ms: u64| -> peppygen::Result<()> {
        for at in [from_ms, from_ms + 200, from_ms + 400].map(at_ms) {
            publish_arms(&mocks, at, &start).await?;
            take_frame(&harness, &mocks, at, &seen, under_the_table()).await?;
        }
        Ok(())
    };
    let straight = left_path(&[], &[over_the_table()]);

    take_three(1_100).await?;
    apply(box_in("remove", "cover", in_odom, ahead, sides))
        .await
        .expect("the box is removed");
    let passes = check(&harness, &straight).await.expect("an answer");
    assert!(passes.clear, "nothing marked inside the box: {passes:?}");
    take_three(2_000).await?;
    let blocked = check(&harness, &straight).await.expect("an answer");
    assert!(
        !blocked.clear && blocked.reason.contains(" from the map and the objects, "),
        "the table marked: {blocked:?}"
    );
    harness.shutdown().await
}

/// A plan of both arms: `jointed` to within 0.01 rad of `to`, and `held`'s grasp point to
/// within `WITHIN_M` of `position` in the grasp link of `jointed`.
fn held_in_the_other_hand(
    jointed: ArmId,
    to: &[f64; ARM_DOF],
    held: ArmId,
    position: [f64; 3],
) -> GoalRequestData {
    let (jointed_model, held_model) = (&OPENARM_V2.arms[jointed], &OPENARM_V2.arms[held]);
    GoalRequestData {
        arms: [LEFT, RIGHT]
            .map(|arm| GoalArmsItem {
                arm_name: OPENARM_V2.arms[arm].name.clone(),
                start_joint_positions: Vec::new(),
            })
            .to_vec(),
        goal_joint_constraints: (jointed_model.joint_names.iter().zip(to))
            .map(|(name, &joint_position)| GoalGoalJointConstraintsItem {
                joint_name: name.clone(),
                joint_position,
                tolerance_above: 0.01,
                tolerance_below: 0.01,
            })
            .collect(),
        goal_position_constraints: vec![GoalGoalPositionConstraintsItem {
            link_name: held_model.grasp_link.clone(),
            frame: Some(jointed_model.grasp_link.clone()),
            timestamp: None,
            position,
            tolerance_m: WITHIN_M,
        }],
        goal_orientation_constraints: Vec::new(),
        path_orientation_constraints: Vec::new(),
        planner: "rrt_connect".to_string(),
        refiner: "none".to_string(),
        allowed_planning_time_s: PLANNING_TIME_S,
    }
}

/// The right arm at Ready with its elbow moved two fifths of the way to straight.
fn right_lowered() -> [f64; ARM_DOF] {
    let mut q = right_ready();
    q[3] *= 0.6;
    q
}

// Criterion 6: a two-arm plan whose goal for one arm lies in the other arm's grasp link
// holds that goal against the link where the plan leaves it, as MoveIt holds a goal in a
// robot link: the right hand's goal in the left grasp link, and the left hand's in the right
// grasp link, which the plan samples first. The goal taken where the other hand starts lies
// elsewhere.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_in_the_other_arms_grasp_link_is_held_against_it() -> peppygen::Result<()> {
    let (harness, _mocks) = moved_base().await?;
    let cases = [
        (LEFT, left_elbow(1.3), left_ready(), RIGHT, right_lowered()),
        (RIGHT, right_lowered(), right_ready(), LEFT, left_elbow(1.3)),
    ];
    for (jointed, to, from, held, reached) in cases {
        let in_the_hand = grasp(jointed, &to)
            .inverse_transform_point(&grasp(held, &reached).translation.vector.into());
        let goal = held_in_the_other_hand(jointed, &to, held, in_the_hand.coords.into());
        let result = plan(&harness, &goal)
            .await
            .unwrap_or_else(|ended| panic!("{held:?} in {jointed:?}: {ended:?}"));
        let (jointed_end, held_end) = (
            last_waypoint(&result, jointed.place()),
            last_waypoint(&result, held.place()),
        );
        let wanted = grasp(jointed, &jointed_end) * in_the_hand;
        let ended = grasp(held, &held_end).translation.vector;
        assert!(
            ended.metric_distance(&wanted.coords) <= WITHIN_M + ROUNDING_M,
            "{held:?}'s grasp point ends {} m from its goal in {jointed:?}'s grasp link",
            ended.metric_distance(&wanted.coords)
        );
        let at_the_start = grasp(jointed, &from) * in_the_hand;
        assert!(
            at_the_start.coords.metric_distance(&wanted.coords) > 0.05,
            "the goal placed where the other hand starts lies elsewhere"
        );
    }
    harness.shutdown().await
}

// A path orientation of the right grasp link, which stands outside a plan of the left arm, in
// the left grasp link's frame, is held at each waypoint against where the left hand stands, as
// MoveIt judges a constraint on a link outside its planning group with every joint outside the
// group at the start state's (state_validity_checker.cpp:58, :102-107; kinematic_constraint.cpp
// :728-735): the right hand ends turned from the left hand's frame by the left hand's swing,
// within the tolerance at every waypoint.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_standing_hands_orientation_is_held_in_the_moving_hands_frame() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(left_ready(), right_ready())).await?;
    let (left_start, right_standing) = (grasp(LEFT, &left_ready()), grasp(RIGHT, &right_ready()));
    let to = left_elbow(1.3);
    let swing = left_start.rotation.angle_to(&grasp(LEFT, &to).rotation);
    assert!(swing > 0.1, "the left hand swings {swing} rad");
    let relative = left_start.inverse() * right_standing;
    let tolerance = 2.0 * swing;
    let (left_model, right_model) = (&OPENARM_V2.arms[LEFT], &OPENARM_V2.arms[RIGHT]);
    let goal = GoalRequestData {
        arms: [LEFT, RIGHT]
            .map(|arm| GoalArmsItem {
                arm_name: OPENARM_V2.arms[arm].name.clone(),
                start_joint_positions: Vec::new(),
            })
            .to_vec(),
        path_orientation_constraints: vec![GoalPathOrientationConstraintsItem {
            link_name: right_model.grasp_link.clone(),
            frame: Some(left_model.grasp_link.clone()),
            timestamp: None,
            orientation: pose_arrays(&relative).1,
            tolerance_rad: tolerance,
            free_axis: "none".to_string(),
        }],
        ..plan_to(&to)
    };
    let result = plan(&harness, &goal)
        .await
        .unwrap_or_else(|ended| panic!("{ended:?}"));
    let off = |left: &[f64; ARM_DOF]| {
        (grasp(LEFT, left).rotation * relative.rotation).angle_to(&right_standing.rotation)
    };
    for waypoint in &result.arms[0].waypoints {
        let left: [f64; ARM_DOF] = waypoint
            .joint_positions
            .as_slice()
            .try_into()
            .expect("a position for each joint");
        assert!(
            off(&left) <= tolerance,
            "the right hand stands {} rad off its orientation in the left hand at {left:?}",
            off(&left)
        );
    }
    assert!(
        result.arms[1]
            .waypoints
            .iter()
            .all(|waypoint| waypoint.joint_positions == right_ready().to_vec()),
        "the right arm stands where it starts"
    );
    let at_the_end = off(&last_waypoint(&result, 0));
    assert!(
        at_the_end >= swing / 2.0,
        "the left hand's swing turns the frame the right hand is held in: {at_the_end} of {swing}"
    );
    harness.shutdown().await
}
