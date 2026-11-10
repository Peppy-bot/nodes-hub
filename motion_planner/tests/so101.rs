//! The planner of SO-101, whose robot_identity names so101: one five-joint arm with one jaw
//! and no depth camera, planning around the objects of its planning scene and clear of its own
//! links, its map empty.

mod common;

use std::time::Duration;

use chain_kinematics::nalgebra::{Isometry3, UnitQuaternion, Vector3};
use chain_kinematics::{Chain, ChainSpec, JointSelection};
use motion_planning_core::pose_parts;
use peppygen::Parameters;
use peppygen::exposed_actions::motion_planning::plan_cartesian_path::GoalWaypointsItem;
use peppygen::exposed_actions::motion_planning::plan_joint_path::{
    GoalArmsItem, GoalGoalJointConstraintsItem, GoalGoalOrientationConstraintsItem,
    GoalGoalPositionConstraintsItem,
};
use peppygen::exposed_services::motion_planning::apply_collision_objects::{
    ApplyCollisionObjectsObjectsItem as Object, ApplyCollisionObjectsObjectsItemShapesItem as Shape,
};
use peppygen::exposed_services::motion_planning::check_joint_path::{
    CheckJointPathArmsItem, CheckJointPathArmsItemWaypointsItem,
};
use peppygen::fixtures::exposed_actions::motion_planning::plan_cartesian_path as cartesian;
use peppygen::fixtures::exposed_actions::motion_planning::plan_joint_path::{
    FailureKind, GoalRequestData,
};
use peppygen::fixtures::exposed_services::motion_planning::{
    apply_attached_collision_object, apply_collision_objects, check_joint_path, describe_arms,
};
use peppygen::fixtures::harness::{Config, Harness, Mocks};
use peppygen::mock::deps::limb_state;
use so101_description::{ARM_DOF, ARM_LIMB, BASE_LINK, GRIPPER_LIMB, JAW_LINK, TCP_LINK, URDF};

use common::{
    Ended, OWN_BODY_WAY_OUT, PLANNING_TIME_S, TIMEOUT, at_ms, check, plan, start_identified_as,
    vacant_camera,
};

/// The arm turned to one side of the base, every joint off its limits.
const START: [f64; ARM_DOF] = [-0.9, -0.6, 0.9, 0.4, 0.0];
/// The arm swung to the other side of the base, a turn of the pan alone.
const ACROSS: [f64; ARM_DOF] = [0.9, -0.6, 0.9, 0.4, 0.0];
/// The pan halfway between `START` and `ACROSS`.
const HALFWAY: [f64; ARM_DOF] = [0.0, -0.6, 0.9, 0.4, 0.0];
/// A configuration whose pose the goals of a position or a pose name.
const REACHED: [f64; ARM_DOF] = [0.5, -0.3, 0.6, 0.2, 0.0];
/// The arm folded over its base as it rests, its own links touching in its self model.
const FOLDED: [f64; ARM_DOF] = [-0.05, -1.72, 1.68, 1.24, 0.32];
/// The arm at home (so101_description's postures.py, `HOME_POSITIONS_RAD`): `FOLDED`
/// unfolded out of contact.
const HOME: [f64; ARM_DOF] = [-0.05, -1.51, 1.47, 1.03, 0.32];
/// A posture whose grasp point stands where a box of `BOX_SIDE_M` around it comes within the
/// clearance of the base, and stands more than the clearance from each link it may not touch,
/// the arm's own links farther apart than the clearance a plan keeps between them.
const BOX_AT_THE_BASE: [f64; ARM_DOF] = [0.0, 1.15, 0.01, 1.54, 0.0];
/// A turn of the frame about its x axis that lies past the orientations the arm holds at
/// `REACHED`'s position by more than the goals' tolerance (rad).
const FAR_TURN_RAD: f64 = 0.8;
const JOINTS: [&str; ARM_DOF] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
];
/// The side of the boxes these tests enter (m).
const BOX_SIDE_M: f64 = 0.06;
/// How near its position a goal holds the grasp point (m).
const POSITION_TOLERANCE_M: f64 = 0.001;

/// SO-101's arm from its base link to its grasp point.
fn chain() -> Chain<ARM_DOF> {
    Chain::from_urdf(
        &urdf_rs::read_from_string(URDF).expect("the URDF parses"),
        &ChainSpec {
            base_link: Some(BASE_LINK),
            tip_link: TCP_LINK,
            joints: JointSelection::PathOrder,
        },
    )
    .expect("SO-101's arm")
}

fn tool_pose(q: &[f64; ARM_DOF]) -> Isometry3<f64> {
    chain().world_ee_pose(q)
}

fn tool_point(q: &[f64; ARM_DOF]) -> [f64; 3] {
    tool_pose(q).translation.vector.into()
}

/// The parameters SO-101's launchers give the planner, as its motion_planner fragment sets
/// them.
fn so101_parameters() -> Parameters {
    Parameters {
        goal_samples: 10,
        map_max_x_m: 0.52,
        map_max_y_m: 0.48,
        map_max_z_m: 0.56,
        map_min_x_m: -0.38,
        map_min_y_m: -0.48,
        map_min_z_m: -0.26,
        max_iterations: 20_000,
        max_range_m: 1.36,
        max_update_rate_hz: 5.0,
        min_clearance_m: 0.02,
        padding_offset_m: 0.02,
        padding_scale: 1.0,
        planning_attempts: 1,
        start_tolerance_rad: 0.05,
        voxel_size_m: 0.02,
    }
}

/// The planner of SO-101 over the harness, no camera slot bound, once its arm's joint
/// positions at `q` have arrived.
async fn started_at(q: &[f64; ARM_DOF]) -> peppygen::Result<(Harness, Mocks)> {
    let config = Config {
        parameters: Some(so101_parameters()),
        camera_mounts_vacant: true,
        workspace_vacant: true,
        ..vacant_camera()
    };
    let (harness, mocks, _fault) = start_identified_as(config, "so101").await?;
    mocks
        .deps
        .limb_state
        .limb_states
        .publish(&limb_state::limb_states::Message {
            timestamp: at_ms(0),
            arm_names: vec![ARM_LIMB.to_string()],
            joints_per_arm: vec![ARM_DOF as u32],
            joint_positions: q.to_vec(),
            positions: vec![0.0; 3],
            orientations: vec![0.0, 0.0, 0.0, 1.0],
            gripper_names: vec![GRIPPER_LIMB.to_string()],
            gripper_openings: vec![0.0],
        })
        .await?;
    for _ in 0..TIMEOUT.as_secs() {
        match check(&harness, &path_through(&[*q])).await {
            Ok(_) => return Ok((harness, mocks)),
            Err(failure) => assert_eq!(
                failure.kind,
                check_joint_path::FailureKind::PreconditionFailed,
                "{failure}"
            ),
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    panic!("the arm's joint positions never arrived");
}

/// The check of the arm from where it stands through `waypoints`.
fn path_through(waypoints: &[[f64; ARM_DOF]]) -> check_joint_path::RequestData {
    check_joint_path::RequestData {
        arms: vec![CheckJointPathArmsItem {
            arm_name: ARM_LIMB.to_string(),
            start_joint_positions: Vec::new(),
            waypoints: waypoints
                .iter()
                .map(|q| CheckJointPathArmsItemWaypointsItem {
                    joint_positions: q.to_vec(),
                })
                .collect(),
        }],
    }
}

/// The answer of a check the planner takes.
async fn checked(harness: &Harness, waypoints: &[[f64; ARM_DOF]]) -> check_joint_path::Response {
    check(harness, &path_through(waypoints))
        .await
        .expect("an answer")
}

/// The plan of the arm from where it stands to the goal the lists name.
fn goal_of(
    joints: Vec<GoalGoalJointConstraintsItem>,
    positions: Vec<GoalGoalPositionConstraintsItem>,
    orientations: Vec<GoalGoalOrientationConstraintsItem>,
) -> GoalRequestData {
    GoalRequestData {
        arms: vec![GoalArmsItem {
            arm_name: ARM_LIMB.to_string(),
            start_joint_positions: Vec::new(),
        }],
        goal_joint_constraints: joints,
        goal_position_constraints: positions,
        goal_orientation_constraints: orientations,
        path_orientation_constraints: Vec::new(),
        objective: "any".to_string(),
        allowed_planning_time_s: PLANNING_TIME_S,
    }
}

/// The goal of every joint within 0.01 rad of `q`.
fn joints_at(q: &[f64; ARM_DOF]) -> GoalRequestData {
    goal_of(
        JOINTS
            .iter()
            .zip(q)
            .map(|(joint, position)| GoalGoalJointConstraintsItem {
                joint_name: joint.to_string(),
                position: *position,
                tolerance_above: 0.01,
                tolerance_below: 0.01,
            })
            .collect(),
        Vec::new(),
        Vec::new(),
    )
}

/// The goal of the grasp point within `POSITION_TOLERANCE_M` of `position`.
fn position_constraint(position: [f64; 3]) -> GoalGoalPositionConstraintsItem {
    GoalGoalPositionConstraintsItem {
        link_name: TCP_LINK.to_string(),
        position,
        tolerance_m: POSITION_TOLERANCE_M,
    }
}

/// The goal of the pose of `REACHED`, its orientation turned `turn_rad` about x, held within
/// `tolerance_rad` and free about `free_axis`.
fn turned_goal(turn_rad: f64, tolerance_rad: f64, free_axis: &str) -> GoalRequestData {
    let pose = tool_pose(&REACHED);
    let turned = pose.rotation * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), turn_rad);
    let (_, orientation) = pose_parts(&Isometry3::from_parts(pose.translation, turned));
    goal_of(
        Vec::new(),
        vec![position_constraint(tool_point(&REACHED))],
        vec![GoalGoalOrientationConstraintsItem {
            link_name: TCP_LINK.to_string(),
            orientation,
            tolerance_rad,
            free_axis: free_axis.to_string(),
        }],
    )
}

/// Enters a box of `BOX_SIDE_M` as the object `id`, centered at `center`.
async fn enter_box(harness: &Harness, id: &str, center: [f64; 3]) {
    apply_collision_objects::poll(
        harness,
        &apply_collision_objects::RequestData {
            objects: vec![Object {
                id: id.to_string(),
                operation: "add".to_string(),
                position: center,
                orientation: [0.0, 0.0, 0.0, 1.0],
                shapes: vec![Shape {
                    kind: "box".to_string(),
                    dimensions: vec![BOX_SIDE_M; 3],
                    mesh_vertices: Vec::new(),
                    mesh_triangles: Vec::new(),
                    position: [0.0; 3],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                }],
            }],
        },
        TIMEOUT,
    )
    .await
    .expect("the box enters");
}

// describe_arms names SO-101's one arm by its limb names, and its base as the body.
#[tokio::test(flavor = "multi_thread")]
async fn it_describes_its_one_arm() -> peppygen::Result<()> {
    let (harness, _mocks) = started_at(&START).await?;
    let described = describe_arms::poll(&harness, TIMEOUT).await?;
    let [arm] = &described.arms[..] else {
        panic!("{:?}", described.arms);
    };
    assert_eq!(
        (arm.name.as_str(), arm.gripper_name.as_str()),
        (ARM_LIMB, GRIPPER_LIMB)
    );
    assert_eq!(arm.joint_names, JOINTS);
    assert_eq!(arm.grasp_link, TCP_LINK);
    assert_eq!(arm.gripper_links, ["gripper_link", JAW_LINK]);
    assert_eq!(described.body_link_names, [BASE_LINK]);
    Ok(())
}

// The arm folded over its base touches itself: a check of it fails, the reason naming the
// links that touch and the home posture to plan from, and answers no map stamp, its map
// empty. A plan from there is refused with the same way out.
#[tokio::test(flavor = "multi_thread")]
async fn it_keeps_its_own_links_apart() -> peppygen::Result<()> {
    let (harness, _mocks) = started_at(&FOLDED).await?;
    let folded = checked(&harness, &[FOLDED]).await;
    assert!(!folded.clear, "the folded arm touches itself");
    assert!(
        folded.reason.starts_with("the start is in contact: ")
            && folded.reason.contains(" touches ")
            && folded.reason.ends_with(OWN_BODY_WAY_OUT),
        "{}",
        folded.reason
    );
    assert_eq!(folded.map_stamp, None);
    let Err(Ended::Failed(refused)) = plan(&harness, &joints_at(&REACHED)).await else {
        panic!("a plan from the folded arm fails once taken");
    };
    assert_eq!(refused.kind, FailureKind::StartInCollision, "{refused}");
    assert!(refused.message.ends_with(OWN_BODY_WAY_OUT), "{refused}");
    Ok(())
}

// From home, the rest pose unfolded out of contact, a plan has a path.
#[tokio::test(flavor = "multi_thread")]
async fn a_plan_from_home_has_a_path() -> peppygen::Result<()> {
    let (harness, _mocks) = started_at(&HOME).await?;
    plan(&harness, &joints_at(&REACHED))
        .await
        .expect("a plan from home");
    Ok(())
}

// A box on the straight joint path fails the check of that path, the reason saying the arm
// comes too near the objects. A plan to the same goal goes around it, and its path passes the
// check.
#[tokio::test(flavor = "multi_thread")]
async fn it_plans_around_a_box_the_straight_path_meets() -> peppygen::Result<()> {
    let (harness, _mocks) = started_at(&START).await?;
    let free = checked(&harness, &[ACROSS]).await;
    assert!(free.clear, "{}", free.reason);
    enter_box(&harness, "box", tool_point(&HALFWAY)).await;
    let met = checked(&harness, &[ACROSS]).await;
    assert!(!met.clear, "the straight path keeps clear of the box");
    assert!(met.reason.contains("the objects"), "{}", met.reason);
    let answer = plan(&harness, &joints_at(&ACROSS))
        .await
        .expect("a plan around the box");
    let waypoints: Vec<[f64; ARM_DOF]> = answer.arms[0]
        .waypoints
        .iter()
        .map(|w| w.joint_positions.clone().try_into().expect("five joints"))
        .collect();
    assert!(waypoints.len() > 1, "a path with a corner: {waypoints:?}");
    let around = checked(&harness, &waypoints).await;
    assert!(around.clear, "{}", around.reason);
    Ok(())
}

// A goal of a position alone plans, the grasp point ending within its tolerance.
#[tokio::test(flavor = "multi_thread")]
async fn it_plans_to_a_position_alone() -> peppygen::Result<()> {
    let (harness, _mocks) = started_at(&START).await?;
    let position = tool_point(&REACHED);
    let answer = plan(
        &harness,
        &goal_of(Vec::new(), vec![position_constraint(position)], Vec::new()),
    )
    .await
    .expect("a plan to the position");
    let end = answer.arms[0].waypoints.last().expect("a last waypoint");
    let apart = (Vector3::from(end.position) - Vector3::from(position)).norm();
    assert!(apart <= POSITION_TOLERANCE_M, "{apart} m from the goal");
    Ok(())
}

// A pose whose orientation lies off those the arm holds at its position plans within a
// tolerance that reaches one it holds, and a tolerance tighter than the arm can come fails with
// unreachable. A turn far past the tolerance fails with unreachable, and plans with its axis
// held and the frame free about it.
#[tokio::test(flavor = "multi_thread")]
async fn a_tolerance_or_a_free_axis_reaches_an_orientation_off_those_it_holds()
-> peppygen::Result<()> {
    let (harness, _mocks) = started_at(&START).await?;
    plan(&harness, &turned_goal(0.05, 0.1, "none"))
        .await
        .expect("a plan within 0.1 rad");
    for goal in [
        turned_goal(0.05, 1e-4, "none"),
        turned_goal(FAR_TURN_RAD, 0.1, "none"),
    ] {
        match plan(&harness, &goal).await {
            Err(Ended::Failed(failure)) => assert_eq!(failure.kind, FailureKind::Unreachable),
            other => panic!("no plan: {other:?}"),
        }
    }
    plan(&harness, &turned_goal(FAR_TURN_RAD, 0.1, "x"))
        .await
        .expect("a plan with x held");
    Ok(())
}

// An object the hand holds keeps clear of the base: the arm at a posture that brings the box
// within the clearance of the base, and of no link it may not touch, fails a check holding the
// box, and passes it with nothing held.
#[tokio::test(flavor = "multi_thread")]
async fn a_held_object_keeps_clear_of_the_base() -> peppygen::Result<()> {
    let (harness, _mocks) = started_at(&BOX_AT_THE_BASE).await?;
    let empty = checked(&harness, &[BOX_AT_THE_BASE]).await;
    assert!(empty.clear, "{}", empty.reason);
    enter_box(&harness, "held", tool_point(&BOX_AT_THE_BASE)).await;
    apply_attached_collision_object::poll(
        &harness,
        &apply_attached_collision_object::RequestData {
            object_id: "held".to_string(),
            link_name: TCP_LINK.to_string(),
            touch_links: vec!["gripper_link".to_string(), JAW_LINK.to_string()],
            operation: "attach".to_string(),
        },
        TIMEOUT,
    )
    .await
    .expect("the box attaches");
    let held = checked(&harness, &[BOX_AT_THE_BASE]).await;
    assert!(
        !held.clear,
        "the held box keeps its clearance from the base"
    );
    assert!(
        held.reason.contains("the map and the objects"),
        "{}",
        held.reason
    );
    Ok(())
}

/// The fraction of the Cartesian path of the grasp point from `START` along `by`, its
/// orientation kept.
async fn fraction_along(harness: &Harness, by: [f64; 3]) -> f64 {
    let pose = tool_pose(&START);
    let goal = cartesian::GoalRequestData {
        arm_name: ARM_LIMB.to_string(),
        start_joint_positions: Vec::new(),
        link_name: TCP_LINK.to_string(),
        waypoints: vec![GoalWaypointsItem {
            position: (pose.translation.vector + Vector3::from(by)).into(),
            orientation: pose_parts(&pose).1,
        }],
        avoid_collisions: true,
    };
    let handle = cartesian::send_goal(harness, &goal, peppygen::DEFAULT_FEEDBACK_BOUND, TIMEOUT)
        .await
        .expect("the goal is sent");
    assert!(handle.refusal.is_none(), "{:?}", handle.refusal);
    match handle
        .get_result(TIMEOUT)
        .await
        .expect("the result arrives")
        .outcome
    {
        cartesian::ResultOutcome::Completed(data) => data.fraction,
        other => panic!("the path ends {other:?}"),
    }
}

// A Cartesian path covers a line in the arm's plane, and ends short on one across it, where
// five joints cannot hold the orientation.
#[tokio::test(flavor = "multi_thread")]
async fn a_cartesian_path_ends_where_five_joints_cannot_hold_the_orientation()
-> peppygen::Result<()> {
    let (harness, _mocks) = started_at(&START).await?;
    assert_eq!(fraction_along(&harness, [0.0, 0.0, 0.03]).await, 1.0);
    let across = fraction_along(&harness, [0.0, 0.03, 0.0]).await;
    assert!(across < 1.0, "{across}");
    Ok(())
}

// A start of another count than the arm's five joints is refused, saying what to send, and
// so is a name of no arm of this robot, naming its one arm.
#[tokio::test(flavor = "multi_thread")]
async fn a_request_names_its_one_arm_and_its_five_joints() -> peppygen::Result<()> {
    let (harness, _mocks) = started_at(&START).await?;
    let refused = |goal: GoalRequestData| {
        let harness = &harness;
        async move {
            match plan(harness, &goal).await {
                Err(Ended::Refused(failure)) => failure.message,
                other => panic!("the goal is refused, not {other:?}"),
            }
        }
    };
    let mut seven = joints_at(&ACROSS);
    seven.arms[0].start_joint_positions = vec![0.0; 7];
    assert_eq!(
        refused(seven).await,
        "start_joint_positions of arm has 7 values; send one for each of its 5 joints, or none \
         to start where the arm stands"
    );
    let mut other = joints_at(&ACROSS);
    other.arms[0].arm_name = "left_arm".to_string();
    assert_eq!(
        refused(other).await,
        "arms names \"left_arm\", no arm of this robot: its arm is \"arm\""
    );
    Ok(())
}
