//! The node over the generated harness, as a caller sees motion_planning: plans and
//! checks wait for the perception camera's first depth frame and for the next after the
//! map is cleared; the arms' own points mark nothing while they move; a check passes a
//! planned path and stops before a table the camera sees; a frame of an alignment the
//! camera's geometry does not describe asks the camera again; a goal holds any link of the
//! arm, and the grasp point free about its approach axis; a path holds an orientation of the
//! grasp link at each waypoint, and a start or a goal that breaks one fails with no_path; a
//! cancel ends the goal
//! cancelled; every refusal comes at a goal's admission; and describe_arms names the
//! joints, the limits limb_motion enforces, the links goals take, and the gripper's links
//! a held object may touch.

mod common;

use chain_kinematics::nalgebra::{Point3, Quaternion, UnitQuaternion, Vector3};
use motion_planner::node::NO_DEPTH_FRAME;
use motion_planner::request::{MAX_PLANNING_TIME_S, Named, Planner, Refiner};
use peppygen::exposed_actions::motion_planning::plan_joint_path::{
    GoalArmsItem, GoalGoalOrientationConstraintsItem, GoalGoalPositionConstraintsItem,
    GoalPathOrientationConstraintsItem,
};
use peppygen::fixtures::exposed_actions::motion_planning::plan_joint_path::{
    CancelState, FailureKind, GoalRequestData, ResultData, ResultOutcome, send_goal,
};
use peppygen::fixtures::exposed_services::motion_planning::{
    check_joint_path, clear_occupancy_map, describe_arms,
};
use peppygen::fixtures::harness::{Config, Harness};
use robot_description::{ArmId, states};

use common::{
    ARM_DOF, BOTH, CLEAR_OR_ALLOW_WAY_OUT, Ended, LEFT, OPENARM_V2, RIGHT, TIMEOUT, along, arms,
    at_ms, check, describe_camera, facing_the_robot, left_elbow, left_path, left_ready,
    over_the_table, parameters, plan, plan_from, plan_to, planned_waypoints, posture, posture_of,
    publish_arms, publish_arms_with_grippers_at, render, right_ready, start, start_described,
    start_identified_as, start_with_camera, start_without_camera, start_without_camera_with,
    start_without_the_rig, take_frame, take_frame_aligned, under_a_seen_table, under_the_table,
    vacant_camera, wait_for_the_arms,
};

/// How near the forearm ends to a pose a plan holds it at: meters, and radians.
const FOREARM_POSITION_M: f64 = 0.01;
const FOREARM_ORIENTATION_RAD: f64 = 0.05;
/// How near the grasp point ends to its position, and its approach axis to the
/// target's: meters, and radians.
const GRASP_POSITION_M: f64 = 0.01;
const APPROACH_AXIS_RAD: f64 = 0.05;
/// How near an axis describe_arms names lies to the grasp frame's axis the model gives
/// (rad).
const GRASP_AXIS_RAD: f64 = 0.01;

// Before the camera's first depth frame a plan and a check are refused; after it a
// plan answers its path, each waypoint with the grasp-point pose limb_state reports
// there, and the frame's stamp. After the map is cleared both are refused again until
// the next frame.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_plan_or_a_check_waits_for_a_depth_frame() -> peppygen::Result<()> {
    let (harness, mocks) = start_with_camera(&facing_the_robot()).await?;
    let start = arms(left_ready(), right_ready());
    let goal = left_elbow(1.3);
    let refused = |ended: Result<ResultData, Ended>| {
        let Err(Ended::Refused(failure)) = ended else {
            panic!("a plan before a frame is refused, not {ended:?}");
        };
        assert!(
            failure.kind == FailureKind::PreconditionFailed
                && failure.message.starts_with(NO_DEPTH_FRAME),
            "{failure}"
        );
    };
    let waiting = |answer: Result<check_joint_path::Response, check_joint_path::Failure>| {
        let failure = answer.expect_err("a check before a frame is refused");
        assert!(
            failure.kind == check_joint_path::FailureKind::PreconditionFailed
                && failure.message.starts_with(NO_DEPTH_FRAME),
            "{failure}"
        );
    };

    publish_arms(&mocks, at_ms(0), &start).await?;
    refused(plan(&harness, &plan_to(&goal)).await);
    waiting(check(&harness, &left_path(&left_ready(), &[goal])).await);

    let nothing = vec![None; (common::WIDTH * common::HEIGHT) as usize];
    take_frame(&harness, &mocks, at_ms(0), &nothing, left_ready()).await?;
    let planned = plan(&harness, &plan_to(&goal)).await.expect("a path");
    assert_eq!(planned.map_stamp, Some(at_ms(0)));
    let arm = &planned.arms[0];
    assert_eq!(arm.arm_name, "left_arm");
    assert_eq!(arm.start_joint_positions, left_ready().to_vec());
    let last = planned_waypoints(&planned).pop().expect("a waypoint");
    assert!(
        (0..ARM_DOF).all(|j| (last[j] - goal[j]).abs() <= 0.01 + 1e-9),
        "the path ends at {last:?}, not within 0.01 of {goal:?}"
    );
    let chain = OPENARM_V2.arms[LEFT].geometry.chain();
    for waypoint in &arm.waypoints {
        let q: [f64; ARM_DOF] = waypoint.joint_positions.as_slice().try_into().expect("7");
        let pose = chain.world_ee_pose(&q);
        let rotation = pose.rotation;
        assert_eq!(waypoint.position, <[f64; 3]>::from(pose.translation.vector));
        assert_eq!(
            waypoint.orientation,
            [rotation.i, rotation.j, rotation.k, rotation.w]
        );
    }

    clear_occupancy_map::poll(&harness, TIMEOUT).await?;
    refused(plan(&harness, &plan_to(&goal)).await);
    waiting(check(&harness, &left_path(&left_ready(), &[goal])).await);

    publish_arms(&mocks, at_ms(1_000), &start).await?;
    take_frame(&harness, &mocks, at_ms(1_000), &nothing, left_ready()).await?;
    assert_eq!(
        plan(&harness, &plan_to(&goal))
            .await
            .expect("a path")
            .map_stamp,
        Some(at_ms(1_000))
    );
    harness.shutdown().await
}

// A depth frame whose align_mode the camera's geometry does not describe is skipped, and
// the node asks the camera again: once the camera describes that alignment, such a
// frame fills the map.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_frame_of_another_alignment_asks_the_camera_again() -> peppygen::Result<()> {
    let camera = facing_the_robot();
    let (harness, mocks) = start_with_camera(&camera).await?;
    describe_camera(&mocks, &camera, "none")?;
    publish_arms(&mocks, at_ms(0), &arms(left_ready(), right_ready())).await?;
    let nothing = vec![None; (common::WIDTH * common::HEIGHT) as usize];
    take_frame_aligned(&harness, &mocks, at_ms(0), &nothing, left_ready(), "none").await?;
    let lens = mocks
        .deps
        .geometry
        .as_ref()
        .expect("the geometry slot is linked");
    assert_eq!(
        lens.get_depth_intrinsics.captured_count(),
        2,
        "the node describes the camera once more"
    );
    harness.shutdown().await
}

// While the left arm bends its elbow in view of the camera, a frame taken halfway
// marks nothing where the arm stood at that instant: the path back through it passes.
// After a clear, the same frame stamped at an instant the arm stood as before it moved
// is cut out there, marks the arm, and the path fails.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_arms_own_points_mark_nothing_while_they_move() -> peppygen::Result<()> {
    let camera = facing_the_robot();
    let (harness, mocks) = start_with_camera(&camera).await?;
    let (before, after, halfway) = (left_elbow(0.6), left_elbow(1.6), left_elbow(1.1));
    publish_arms(&mocks, at_ms(0), &arms(before, right_ready())).await?;
    publish_arms(&mocks, at_ms(100), &arms(after, right_ready())).await?;
    let frame = render(&camera, Some(&arms(halfway, right_ready())), &[]);
    assert!(
        frame.iter().flatten().count() > 100,
        "the camera sees the arms"
    );

    take_frame(&harness, &mocks, at_ms(50), &frame, after).await?;
    let back = check(&harness, &left_path(&[], &[before]))
        .await
        .expect("an answer");
    assert!(
        back.clear && back.clear_waypoints == 1 && back.reason.is_empty(),
        "the path back through the arm's own points: {back:?}"
    );
    assert_eq!(back.map_stamp, Some(at_ms(50)));

    clear_occupancy_map::poll(&harness, TIMEOUT).await?;
    publish_arms(&mocks, at_ms(1_000), &arms(before, right_ready())).await?;
    publish_arms(&mocks, at_ms(1_100), &arms(after, right_ready())).await?;
    take_frame(&harness, &mocks, at_ms(1_000), &frame, after).await?;
    let marked = check(&harness, &left_path(&[], &[before]))
        .await
        .expect("an answer");
    assert!(
        !marked.clear && marked.clear_waypoints == 0 && marked.reason.contains("the occupancy map"),
        "the arm's points cut out where it did not stand: {marked:?}"
    );
    harness.shutdown().await
}

// With a table the camera sees between the left hand under it, the elbow open, and the
// hand over it, the elbow bent: the straight path through the table fails from the
// start, and through a waypoint below, the elbow opened further, passes that waypoint
// alone. A waypoint with the elbow outside its limit under the table, and a start in
// contact with the table, fail; a plan from that start fails with start_in_collision. A
// planned path around the table passes whole.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_check_passes_a_planned_path_and_stops_before_the_table() -> peppygen::Result<()> {
    let (under_q, over_q) = (under_the_table(), over_the_table());
    let (harness, _mocks, crossing_t) = under_a_seen_table().await?;

    let through = check(&harness, &left_path(&[], &[over_q]))
        .await
        .expect("an answer");
    assert!(
        !through.clear
            && through.clear_waypoints == 0
            && through.reason.starts_with("on the run from the start, ")
            && through.reason.contains(" from the map and the objects, "),
        "the straight path through the table: {through:?}"
    );
    let below = left_elbow(0.3);
    assert!(
        check(&harness, &left_path(&[], &[below]))
            .await
            .expect("an answer")
            .clear,
        "the path down from under the table"
    );
    let stopped = check(&harness, &left_path(&[], &[below, over_q]))
        .await
        .expect("an answer");
    assert!(
        !stopped.clear
            && stopped.clear_waypoints == 1
            && stopped.reason.starts_with("on the run from waypoint 1, ")
            && stopped.reason.contains(" from the map and the objects, "),
        "the path through the table after a waypoint below it: {stopped:?}"
    );

    // The elbow opened past its lower limit, the hand moving down from under the table.
    let mut beyond = below;
    beyond[3] = OPENARM_V2.arms[LEFT].geometry.chain().limits()[3].lo - 0.1;
    let outside = check(&harness, &left_path(&[], &[below, beyond]))
        .await
        .expect("an answer");
    assert!(
        !outside.clear
            && outside.clear_waypoints == 1
            && outside.reason.contains("outside its limits"),
        "a waypoint outside a joint limit: {outside:?}"
    );
    let crossing_q = along(&under_q, &over_q, crossing_t);
    let in_contact = check(&harness, &left_path(&crossing_q, &[over_q]))
        .await
        .expect("an answer");
    assert!(
        !in_contact.clear
            && in_contact.clear_waypoints == 0
            && in_contact.reason.contains("touches the occupancy map")
            && in_contact.reason.ends_with(CLEAR_OR_ALLOW_WAY_OUT),
        "a start in contact with the table: {in_contact:?}"
    );
    let Err(Ended::Failed(collided)) = plan(&harness, &plan_from(&crossing_q, &over_q)).await
    else {
        panic!("a plan from a start in contact fails once taken");
    };
    assert_eq!(collided.kind, FailureKind::StartInCollision, "{collided}");
    assert!(
        collided.message.contains("touches the occupancy map")
            && collided.message.ends_with(CLEAR_OR_ALLOW_WAY_OUT),
        "{collided}"
    );

    let planned = plan(&harness, &plan_to(&over_q))
        .await
        .expect("a path around the table");
    let waypoints = planned_waypoints(&planned);
    let around = check(
        &harness,
        &left_path(&planned.arms[0].start_joint_positions, &waypoints),
    )
    .await
    .expect("an answer");
    assert!(
        around.clear && around.clear_waypoints as usize == waypoints.len(),
        "the planned path around the table: {around:?}"
    );
    harness.shutdown().await
}

/// The axis of a grasp link's frame as describe_arms names it, such as "+z".
fn named_axis(name: &str) -> Vector3<f64> {
    let (sign, letter) = name.split_at(1);
    let unit = match letter {
        "x" => Vector3::x(),
        "y" => Vector3::y(),
        "z" => Vector3::z(),
        other => panic!("{name:?} names no axis {other:?}"),
    };
    match sign {
        "+" => unit,
        "-" => -unit,
        other => panic!("{name:?} has no sign {other:?}"),
    }
}

/// In the grasp frame of the arm on `side`, by OpenArm v2's model: the direction from
/// the middle of the gripper's finger joints to the grasp point, along which the
/// gripper approaches, and the direction from one finger joint to the other, along
/// which its jaws close.
fn grasp_axes(side: ArmId) -> (Vector3<f64>, Vector3<f64>) {
    let chain = OPENARM_V2.arms[side].geometry.chain();
    let q = posture(side, states::READY);
    let grasp = chain.world_ee_pose(&q);
    let posed = chain.at(&q);
    let joints: Vec<Vector3<f64>> = OPENARM_V2
        .own
        .fingers(side, &OPENARM_V2.arms[side].geometry)
        .expect("the arm's fingers")
        .into_iter()
        .map(|finger| {
            let world = posed.link_pose_world(finger.segment) * finger.offset;
            (grasp.inverse() * Point3::from(world.translation.vector)).coords
        })
        .collect();
    let [a, b] = joints.as_slice() else {
        panic!("the gripper has two fingers, not {}", joints.len());
    };
    (-((a + b) / 2.0).normalize(), (b - a).normalize())
}

// An OpenArm v1 plans with no planner change: its description alone gives the planner its
// arms, which bear the names of OpenArm v2's, and a joint goal of its left arm plans.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn openarm_v1_plans_from_its_description_alone() -> peppygen::Result<()> {
    let config = Config {
        camera_mounts_vacant: true,
        workspace_vacant: true,
        ..vacant_camera()
    };
    let (harness, mocks, _fault) = start_identified_as(config, "openarm_v1").await?;
    let v1 = robot_description::model("openarm_v1")
        .expect("robot_description bundles OpenArm v1")
        .description();
    let names: Vec<&str> = v1.arms().iter().map(|arm| arm.name.as_str()).collect();
    let v2_names: Vec<&str> = BOTH
        .iter()
        .map(|arm| OPENARM_V2.arms[*arm].name.as_str())
        .collect();
    assert_eq!(names, v2_names);
    let ready = |arm| posture_of("openarm_v1", arm, states::READY);
    publish_arms(&mocks, at_ms(0), &arms(ready(LEFT), ready(common::RIGHT))).await?;
    wait_for_the_arms(&harness, ready(LEFT)).await;
    let mut goal = ready(LEFT);
    goal[3] += 0.3;
    let planned = plan(&harness, &plan_to(&goal)).await.expect("a path");
    let last = planned_waypoints(&planned)
        .last()
        .copied()
        .expect("a path ends");
    assert!(
        (0..ARM_DOF).all(|j| (last[j] - goal[j]).abs() <= 0.01 + 1e-9),
        "{last:?} against {goal:?}"
    );
    harness.shutdown().await
}

// Each OpenArm plans its left arm from Home to Ready and from Ready to Home, its grippers shut and
// open: its Home stands clear of the planner's own clearance at every opening.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn each_openarm_plans_to_and_from_home_with_its_grippers_shut_and_open()
-> peppygen::Result<()> {
    for model in ["openarm_v1", "openarm_v2"] {
        for opening in [0.0, 1.0] {
            for (start, end) in [(states::HOME, states::READY), (states::READY, states::HOME)] {
                let config = Config {
                    camera_mounts_vacant: true,
                    workspace_vacant: true,
                    ..vacant_camera()
                };
                let (harness, mocks, _fault) = start_identified_as(config, model).await?;
                let at = |arm, name| posture_of(model, arm, name);
                let stand = arms(at(LEFT, start), at(RIGHT, start));
                publish_arms_with_grippers_at(&mocks, at_ms(0), &stand, opening).await?;
                wait_for_the_arms(&harness, at(LEFT, start)).await;
                let goal = at(LEFT, end);
                let planned = plan(&harness, &plan_to(&goal))
                    .await
                    .unwrap_or_else(|ended| {
                        panic!("{model}, grippers at {opening}: {start} to {end}: {ended:?}")
                    });
                let last = planned_waypoints(&planned)
                    .last()
                    .copied()
                    .expect("a path ends");
                assert!(
                    (0..ARM_DOF).all(|j| (last[j] - goal[j]).abs() <= 0.01 + 1e-9),
                    "{model}, grippers at {opening}: {start} to {end} ends at {last:?}"
                );
                harness.shutdown().await?;
            }
        }
    }
    Ok(())
}

// A robot whose description names no arm ends the node with a fault that names the cause.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_description_of_no_arm_ends_the_node_with_a_fault() -> peppygen::Result<()> {
    let config = Config {
        parameters: Some(parameters()),
        ..Config::default()
    };
    let bundled = robot_description::model("so101").expect("robot_description bundles SO-101");
    let srdf = bundled.srdf();
    let start = srdf.find("<end_effector").expect("an end effector");
    let end = srdf[start..].find("/>").expect("a closed element") + start + 2;
    let armless = format!("{}{}", &srdf[..start], &srdf[end..]);
    let (harness, _mocks, fault) = start_described(
        config,
        "so101",
        peppygen::mock::deps::identity::get_description::ResponseData::new(
            bundled.urdf().to_string(),
            armless,
            Vec::new(),
            "edited".to_string(),
        ),
    )
    .await?;
    tokio::time::timeout(
        TIMEOUT,
        harness.node_runner().cancellation_token().cancelled(),
    )
    .await
    .expect("the fault shuts the node down");
    let fault = fault.take().expect("the node ends with a fault");
    assert!(
        fault.to_string().contains(
            "the motion planner for so101: the description names no arm: an arm is an SRDF \
             group of one chain whose tip is an end effector's parent link"
        ),
        "{fault}"
    );
    harness.shutdown().await
}

// A description the planner cannot read ends the node with a fault that names the model
// and the cause, after one ask.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_description_the_planner_cannot_read_ends_the_node_with_a_fault() -> peppygen::Result<()>
{
    let config = Config {
        parameters: Some(parameters()),
        ..Config::default()
    };
    let bundled = robot_description::model("so101").expect("robot_description bundles SO-101");
    let (harness, mocks, fault) = start_described(
        config,
        "so101",
        peppygen::mock::deps::identity::get_description::ResponseData::new(
            "not a URDF".to_string(),
            bundled.srdf().to_string(),
            Vec::new(),
            "edited".to_string(),
        ),
    )
    .await?;
    tokio::time::timeout(
        TIMEOUT,
        harness.node_runner().cancellation_token().cancelled(),
    )
    .await
    .expect("the fault shuts the node down");
    let fault = fault.take().expect("the node ends with a fault");
    assert!(
        fault.to_string().starts_with("the description of so101: "),
        "{fault}"
    );
    assert_eq!(mocks.deps.identity.get_description.captured_count(), 1);
    harness.shutdown().await
}

// describe_arms names each arm's joints in limb_state's order with the limits the
// backbone enforces, its gripper, its links, its grasp link and the grasp frame's axes
// where the model's fingers set them, and the body's links: those of the model the
// robot's identity names, which the planner asked once before it served. A planner with
// no camera linked answers it, and plans and checks around the robot's own body with
// no map stamp.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn describe_arms_names_the_joints_limits_and_links() -> peppygen::Result<()> {
    let (harness, mocks) = start_without_camera(&arms(left_ready(), right_ready())).await?;
    assert_eq!(mocks.deps.identity.get_identity.captured_count(), 1);
    let described = describe_arms::poll(&harness, TIMEOUT).await?;
    let description = robot_description::model("openarm_v2")
        .expect("robot_description bundles OpenArm v2")
        .description();
    let urdf_limit = |joint: &str| {
        let limit = &description
            .robot()
            .joints
            .iter()
            .find(|candidate| candidate.name == joint)
            .expect("an arm joint of the URDF")
            .limit;
        [limit.lower, limit.upper]
    };
    assert_eq!(
        described
            .arms
            .iter()
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>(),
        description
            .arms()
            .iter()
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>()
    );
    for ((arm, from), id) in described.arms.iter().zip(description.arms()).zip(BOTH) {
        assert_eq!(arm.gripper_name, from.gripper.name);
        assert_eq!(arm.joint_names, from.joints);
        let limits: Vec<[f64; 2]> = from.joints.iter().map(|joint| urdf_limit(joint)).collect();
        assert_eq!(
            arm.joint_lower_limits,
            limits.iter().map(|l| l[0]).collect::<Vec<_>>()
        );
        assert_eq!(
            arm.joint_upper_limits,
            limits.iter().map(|l| l[1]).collect::<Vec<_>>()
        );
        assert_eq!(arm.grasp_link, from.grasp_link);
        let (approach, closing) = grasp_axes(id);
        assert!(
            named_axis(&arm.approach_axis).angle(&approach) <= GRASP_AXIS_RAD,
            "the approach axis {} against {approach:?}",
            arm.approach_axis
        );
        let named = named_axis(&arm.closing_axis);
        assert!(
            named.angle(&closing).min(named.angle(&-closing)) <= GRASP_AXIS_RAD,
            "the closing axis {} against {closing:?}",
            arm.closing_axis
        );
        assert_eq!(arm.gripper_links, OPENARM_V2.arms[id].gripper_links);
        for finger in &from.gripper.fingers {
            assert!(arm.gripper_links.contains(&finger.link), "{}", finger.link);
        }
        assert!(
            arm.gripper_links.iter().all(|l| arm.link_names.contains(l)),
            "{:?}",
            arm.link_names
        );
    }
    let mut body = described.body_link_names.clone();
    body.sort_unstable();
    let mut modeled: Vec<&str> = OPENARM_V2.body.link_names().collect();
    modeled.sort_unstable();
    assert_eq!(body, modeled);

    let goal = left_elbow(1.3);
    let answered = check(&harness, &left_path(&[], &[goal]))
        .await
        .expect("an answer");
    assert!(
        answered.clear && answered.map_stamp.is_none(),
        "{answered:?}"
    );
    let planned = plan(&harness, &plan_to(&goal)).await.expect("a path");
    assert_eq!(planned.map_stamp, None);
    harness.shutdown().await
}

// A position and an orientation constraint on a link of the arm above its grasp point,
// the forearm, give a path whose last waypoint puts the forearm there.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_plan_holds_a_link_of_the_arm() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(left_ready(), right_ready())).await?;
    let forearm = |q: &[f64; ARM_DOF]| {
        OPENARM_V2.arms[LEFT]
            .geometry
            .chain()
            .at(q)
            .link_pose_world(3)
    };
    let target = forearm(&left_elbow(1.3));
    let rotation = target.rotation;
    let goal = GoalRequestData {
        goal_joint_constraints: Vec::new(),
        goal_position_constraints: vec![GoalGoalPositionConstraintsItem {
            link_name: "openarm_left_link4".to_string(),
            position: target.translation.vector.into(),
            tolerance_m: FOREARM_POSITION_M,
        }],
        goal_orientation_constraints: vec![GoalGoalOrientationConstraintsItem {
            link_name: "openarm_left_link4".to_string(),
            orientation: [rotation.i, rotation.j, rotation.k, rotation.w],
            tolerance_rad: FOREARM_ORIENTATION_RAD,
            free_axis: "none".to_string(),
        }],
        ..plan_to(&left_ready())
    };
    let planned = plan(&harness, &goal).await.expect("a path");
    let last = planned_waypoints(&planned).pop().expect("a waypoint");
    let reached = forearm(&last);
    assert!(
        (reached.translation.vector - target.translation.vector).norm() <= FOREARM_POSITION_M
            && reached.rotation.angle_to(&target.rotation) <= FOREARM_ORIENTATION_RAD,
        "the forearm ends at {reached}, not at {target}"
    );
    harness.shutdown().await
}

// A position and an orientation constraint on the grasp link with its approach axis
// free give a path whose last waypoint holds the grasp point at the position and the
// approach axis along the target's, turned about it as the plan finds.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_plan_holds_the_grasp_point_free_about_its_approach_axis() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(left_ready(), right_ready())).await?;
    let reached = OPENARM_V2.arms[LEFT]
        .geometry
        .chain()
        .world_ee_pose(&left_elbow(1.3));
    let turned = reached.rotation * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 1.0);
    let goal = GoalRequestData {
        goal_joint_constraints: Vec::new(),
        goal_position_constraints: vec![GoalGoalPositionConstraintsItem {
            link_name: "openarm_left_tcp".to_string(),
            position: reached.translation.vector.into(),
            tolerance_m: GRASP_POSITION_M,
        }],
        goal_orientation_constraints: vec![GoalGoalOrientationConstraintsItem {
            link_name: "openarm_left_tcp".to_string(),
            orientation: [turned.i, turned.j, turned.k, turned.w],
            tolerance_rad: APPROACH_AXIS_RAD,
            free_axis: "z".to_string(),
        }],
        ..plan_to(&left_ready())
    };
    let planned = plan(&harness, &goal).await.expect("a path");
    let last = planned.arms[0].waypoints.last().expect("a waypoint");
    let [x, y, z, w] = last.orientation;
    let ended = UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z));
    let approach = |rotation: &UnitQuaternion<f64>| rotation * Vector3::z();
    let off_axis = approach(&ended).angle(&approach(&turned));
    let off_position = (Vector3::from(last.position) - reached.translation.vector).norm();
    assert!(
        off_position <= GRASP_POSITION_M && off_axis <= APPROACH_AXIS_RAD,
        "the grasp point ends {off_position} m from its position, its approach axis {off_axis} rad off"
    );
    harness.shutdown().await
}

/// The tolerance of the path orientations held here (rad).
const PATH_TOLERANCE_RAD: f64 = 0.1;

/// The left grasp link held within `PATH_TOLERANCE_RAD` of where `q` turns it, free about
/// `free_axis`.
fn holding_the_left_hand(
    q: &[f64; ARM_DOF],
    free_axis: &str,
) -> GoalPathOrientationConstraintsItem {
    let rotation = OPENARM_V2.arms[LEFT]
        .geometry
        .chain()
        .world_ee_pose(q)
        .rotation;
    GoalPathOrientationConstraintsItem {
        link_name: "openarm_left_tcp".to_string(),
        orientation: [rotation.i, rotation.j, rotation.k, rotation.w],
        tolerance_rad: PATH_TOLERANCE_RAD,
        free_axis: free_axis.to_string(),
    }
}

// A path orientation of the grasp link, whole or free about an axis, holds at each waypoint of
// a plan whose goal names a position alone, which takes the orientation of the constraint.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_plan_holds_a_path_orientation_at_each_waypoint() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(left_ready(), right_ready())).await?;
    let chain = OPENARM_V2.arms[LEFT].geometry.chain();
    let from = chain.world_ee_pose(&left_ready());
    let position = from.translation.vector + Vector3::new(0.0, -0.05, 0.05);
    for free_axis in ["none", "z"] {
        let goal = GoalRequestData {
            goal_joint_constraints: Vec::new(),
            goal_position_constraints: vec![GoalGoalPositionConstraintsItem {
                link_name: "openarm_left_tcp".to_string(),
                position: position.into(),
                tolerance_m: GRASP_POSITION_M,
            }],
            path_orientation_constraints: vec![holding_the_left_hand(&left_ready(), free_axis)],
            ..plan_to(&left_ready())
        };
        let planned = plan(&harness, &goal).await.expect("a path");
        let axis = Vector3::z();
        for waypoint in &planned.arms[0].waypoints {
            let [x, y, z, w] = waypoint.orientation;
            let turned = UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z));
            let error = if free_axis == "none" {
                from.rotation.angle_to(&turned)
            } else {
                (from.rotation * axis).angle(&(turned * axis))
            };
            assert!(
                error <= PATH_TOLERANCE_RAD,
                "free about {free_axis}: a waypoint stands {error} rad from the orientation"
            );
        }
    }
    harness.shutdown().await
}

// A start past a path orientation's tolerance, an arm that stays where it starts past one, and a
// goal of joints past one each fail with no_path, and the message names the link.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_start_or_a_goal_that_breaks_a_path_orientation_fails_with_no_path()
-> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(left_ready(), right_ready())).await?;
    let elsewhere = left_elbow(1.3);
    let of_the_elbow = holding_the_left_hand(&elsewhere, "none");
    let of_ready = holding_the_left_hand(&left_ready(), "none");
    let starting_past_it = GoalRequestData {
        path_orientation_constraints: vec![of_the_elbow.clone()],
        ..plan_to(&left_ready())
    };
    let ending_past_it = GoalRequestData {
        path_orientation_constraints: vec![of_ready],
        ..plan_to(&elsewhere)
    };
    let staying_past_it = {
        let mut goal = plan_to(&left_ready());
        goal.arms.push(GoalArmsItem {
            arm_name: "right_arm".to_string(),
            start_joint_positions: Vec::new(),
        });
        goal.goal_joint_constraints.clear();
        goal.path_orientation_constraints = vec![of_the_elbow];
        goal.goal_joint_constraints.extend(
            OPENARM_V2.arms[RIGHT]
                .joint_names
                .iter()
                .zip(&right_ready())
                .map(|(name, &position)| {
                    peppygen::exposed_actions::motion_planning::plan_joint_path::GoalGoalJointConstraintsItem {
                        joint_name: name.clone(),
                        position,
                        tolerance_above: 0.01,
                        tolerance_below: 0.01,
                    }
                }),
        );
        goal
    };
    for (what, goal, says) in [
        (
            "a start",
            starting_past_it,
            "the start holds openarm_left_tcp",
        ),
        ("a goal", ending_past_it, "each breaking a path orientation"),
        (
            "an arm that stays",
            staying_past_it,
            "and its arm stays where it starts",
        ),
    ] {
        let Err(Ended::Failed(failure)) = plan(&harness, &goal).await else {
            panic!("{what} past a path orientation fails");
        };
        assert_eq!(failure.kind, FailureKind::NoPath, "{what}: {failure}");
        assert!(failure.to_string().contains(says), "{what}: {failure}");
    }
    harness.shutdown().await
}

/// A grasp point 5 m before the robot, out of either arm's reach.
const OUT_OF_REACH: [f64; 3] = [5.0, 0.0, 1.0];

// A cancel of a plan that nothing else ends before it arrives ends the goal cancelled:
// the plan's goal sampling searches for a grasp point out of reach, which no search
// meets, until its 1000 samples or its deadline, the longest a plan may take. The stop
// the cancel sets ends the plan itself (`node::tests::a_set_stop_ends_the_plan_stopped`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cancel_ends_the_goal_cancelled() -> peppygen::Result<()> {
    let (harness, _mocks) =
        start_without_camera_with(parameters(), &arms(left_ready(), right_ready())).await?;
    let long = GoalRequestData {
        allowed_planning_time_s: MAX_PLANNING_TIME_S,
        goal_joint_constraints: Vec::new(),
        goal_position_constraints: vec![GoalGoalPositionConstraintsItem {
            link_name: "openarm_left_tcp".to_string(),
            position: OUT_OF_REACH,
            tolerance_m: GRASP_POSITION_M,
        }],
        ..plan_to(&left_ready())
    };
    let handle = send_goal(&harness, &long, peppygen::DEFAULT_FEEDBACK_BOUND, TIMEOUT).await?;
    assert!(handle.accepted, "{:?}", handle.refusal);
    assert_eq!(handle.cancel_goal(TIMEOUT).await?, CancelState::Signalled);
    let outcome = handle.get_result(TIMEOUT).await?.outcome;
    assert!(
        matches!(outcome, ResultOutcome::Cancelled { .. }),
        "{outcome:?}"
    );
    harness.shutdown().await
}

// With the four camera slots vacant, as on a robot without a camera, the map stays empty
// and a plan answers its path with no map stamp at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_robot_without_a_camera_plans_without_a_map() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(left_ready(), right_ready())).await?;
    assert_plans_without_a_map(&harness).await;
    harness.shutdown().await
}

// With the camera and geometry vacant and camera_mounts and workspace bound, as on
// OpenArm without its camera rig, the map stays empty too.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_robot_without_its_camera_rig_plans_without_a_map() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_the_rig(&arms(left_ready(), right_ready())).await?;
    assert_plans_without_a_map(&harness).await;
    harness.shutdown().await
}

/// Asserts that a plan of the left arm answers its path with no map stamp.
async fn assert_plans_without_a_map(harness: &Harness) {
    let planned = plan(harness, &plan_to(&left_elbow(1.3)))
        .await
        .expect("a path");
    assert_eq!(planned.map_stamp, None);
}

/// The planning time of each plan of `each_planner_and_refiner_answers_a_path_to_the_goal`:
/// an `rrt_star` plan plans all of it.
const EACH_PLANNER_TIME_S: f64 = 1.0;

// Each planner, its path refined by each refiner, answers a path of the left arm that ends
// within its goal: on a robot without a camera, a goal of the left elbow planned by
// rrt_connect and by rrt_star, each refined by none and by chomp.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn each_planner_and_refiner_answers_a_path_to_the_goal() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(left_ready(), right_ready())).await?;
    for planner in Planner::ALL {
        for refiner in Refiner::ALL {
            let request = GoalRequestData {
                planner: planner.name().to_string(),
                refiner: refiner.name().to_string(),
                allowed_planning_time_s: EACH_PLANNER_TIME_S,
                ..plan_to(&left_elbow(1.3))
            };
            let planned = plan(&harness, &request)
                .await
                .unwrap_or_else(|ended| panic!("{planner:?} and {refiner:?}: {ended:?}"));
            let end = *planned_waypoints(&planned).last().expect("a waypoint");
            assert!(
                request
                    .goal_joint_constraints
                    .iter()
                    .zip(&end)
                    .all(|(goal, &at)| {
                        goal.position - goal.tolerance_below <= at
                            && at <= goal.position + goal.tolerance_above
                    }),
                "{planner:?} and {refiner:?} end at {end:?}"
            );
        }
    }
    harness.shutdown().await
}

// With the camera and geometry bound and camera_mounts vacant, the map cannot place the
// camera: every check and plan is refused, the message naming the slots to bind.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_camera_without_its_mounts_refuses_every_plan_and_check() -> peppygen::Result<()> {
    let config = Config {
        parameters: Some(parameters()),
        camera_mounts_vacant: true,
        ..Config::default()
    };
    let (harness, mocks) = start(config).await?;
    publish_arms(&mocks, at_ms(0), &arms(left_ready(), right_ready())).await?;
    let reason = format!(
        "{NO_DEPTH_FRAME}: the launcher binds the planner's camera, geometry and workspace \
         and leaves its camera_mounts vacant; bind camera, geometry, camera_mounts and \
         workspace, or leave camera and geometry vacant"
    );
    let failure = check(&harness, &left_path(&left_ready(), &[left_elbow(1.3)]))
        .await
        .expect_err("a check is refused");
    assert_eq!(
        (failure.kind, failure.message),
        (
            check_joint_path::FailureKind::PreconditionFailed,
            reason.clone()
        )
    );
    let Err(Ended::Refused(refusal)) = plan(&harness, &plan_to(&left_elbow(1.3))).await else {
        panic!("a plan is refused");
    };
    assert_eq!(
        (refusal.kind, refusal.message),
        (FailureKind::PreconditionFailed, reason)
    );
    harness.shutdown().await
}

// Each refusal the contract states comes at the goal's admission, the node taking
// nothing: a goal that names an arm twice, of a joint the arm lacks, of a joint target outside its
// limits, of a skewed orientation, of a planning time out of range, and from a start
// outside a joint limit by more than the start tolerance.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_refusal_comes_before_acceptance() -> peppygen::Result<()> {
    let (harness, _mocks) = start_without_camera(&arms(left_ready(), right_ready())).await?;
    let goal = left_elbow(1.3);
    let refused = |ended: Result<ResultData, Ended>| match ended {
        Err(Ended::Refused(failure)) => failure.kind,
        other => panic!("taken: {other:?}"),
    };

    let mut twice = plan_to(&goal);
    twice.arms.push(GoalArmsItem {
        arm_name: "left_arm".to_string(),
        start_joint_positions: Vec::new(),
    });
    let mut missing = plan_to(&goal);
    missing.goal_joint_constraints[0].joint_name = "openarm_left_joint8".to_string();
    let mut beyond = plan_to(&goal);
    beyond.goal_joint_constraints[3].position = 3.0;
    let skewed = GoalRequestData {
        goal_joint_constraints: Vec::new(),
        goal_orientation_constraints: vec![GoalGoalOrientationConstraintsItem {
            link_name: "openarm_left_tcp".to_string(),
            orientation: [0.0, 0.0, 0.0, 1.1],
            tolerance_rad: 0.1,
            free_axis: "none".to_string(),
        }],
        ..plan_to(&goal)
    };
    let slow = GoalRequestData {
        allowed_planning_time_s: 61.0,
        ..plan_to(&goal)
    };
    for rejected in [twice, missing, beyond, skewed, slow] {
        assert_eq!(
            refused(plan(&harness, &rejected).await),
            FailureKind::Rejected
        );
    }

    let mut below = left_ready();
    below[3] = OPENARM_V2.arms[LEFT].geometry.chain().limits()[3].lo - 0.1;
    assert_eq!(
        refused(plan(&harness, &plan_from(&below, &goal)).await),
        FailureKind::PreconditionFailed
    );
    harness.shutdown().await
}
