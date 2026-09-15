//! The seat over the wire, the engine played by its generated mock: the node
//! attaches as its model, publishes every state the engine feeds back on the
//! limb it measures, carries the relays' setpoints across in its commands,
//! reads why the engine ended its stay, and never commands an engine that
//! refused it.
//!
//! Readiness comes after the seat, never before it: the robot serves no
//! `is_ready` until the simulation has stood it in the scene.

use std::time::{Duration, SystemTime};

use peppygen::consumed_actions::simulation::attach::{
    SimulationRobotAttachActionFeedbackMessageArmsItem as ArmState,
    SimulationRobotAttachActionFeedbackMessageGrippersItem as GripperState,
};
use peppygen::fixtures::exposed_services::robot_ready::is_ready as robot_is_ready;
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::simulation::{attach, command};
use peppygen::mock::pairings::{left_arm_link, right_gripper_link};

mod common;

/// How long the wire may take for any one exchange.
const WIRE: Duration = Duration::from_secs(10);
/// How long a service that nothing serves is waited on before concluding it
/// is not being served.
const UNSERVED: Duration = Duration::from_millis(500);
/// Joints of one OpenArm arm, as the engine reports them.
const ARM_DOF: usize = 7;
/// Commands drained while a setpoint published beside a tick lands in one.
const COMMANDS_TO_LAND: usize = 50;

/// A robot that joins a simulation: the `simulation` slot is bound, so the
/// node takes its seat before it does anything else.
fn joining_a_simulation(model: &str) -> Config {
    Config {
        parameters: Some(common::parameters(model)),
        ..Config::default()
    }
}

/// The engine's answer: the model's limbs listed right before left, which
/// is not the order of the node's slots.
fn seated() -> attach::GoalResponseData {
    attach::GoalResponseData {
        arm_names: vec!["right".into(), "left".into()],
        arm_joints: vec![ARM_DOF as u32, ARM_DOF as u32],
        gripper_names: vec!["right".into(), "left".into()],
    }
}

/// A measured state in the engine's order: the right limbs first.
fn state(left_joint: f64, right_joint: f64) -> attach::FeedbackMessage {
    let arm = |joint| ArmState {
        positions: vec![joint; ARM_DOF],
        velocities: vec![0.0; ARM_DOF],
    };
    let gripper = |opening| GripperState {
        opening,
        effort: 0.0,
    };
    attach::FeedbackMessage {
        timestamp: SystemTime::now(),
        arms: vec![arm(right_joint), arm(left_joint)],
        grippers: vec![gripper(0.25), gripper(0.75)],
    }
}

fn accepted() -> command::ResponseData {
    command::ResponseData {
        success: true,
        joining: false,
        message: String::new(),
    }
}

/// The engine is still standing this robot, so it wrote no limb. A robot
/// commanding through its own join reads the flag, not the wording.
fn joining() -> command::ResponseData {
    command::ResponseData {
        success: false,
        joining: true,
        message: "the engine has not stood this robot yet".into(),
    }
}

async fn next_state<T>(
    next: impl Future<Output = peppygen::Result<Option<T>>>,
) -> peppygen::Result<T> {
    Ok(tokio::time::timeout(WIRE, next)
        .await
        .expect("a state within the wire's time")?
        .expect("the mock's session is open"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_robot_takes_its_seat_and_drives_its_limbs_through_it() -> peppygen::Result<()> {
    let (harness, mut mocks) = Harness::start_with(
        joining_a_simulation("openarm_v2"),
        openarm_initializer::setup,
    )
    .await?;

    let mut simulation = mocks
        .deps
        .simulation
        .take()
        .expect("a robot joining a simulation has one bound");

    // The node attaches as its model, on a spot of the engine's choosing.
    let goal = simulation.attach.next_goal(WIRE).await?;
    assert_eq!(goal.request.model, "openarm_v2");
    assert!(goal.request.placement.is_none());

    // Readiness waits on the seat, so a robot the engine has not stood yet
    // answers nothing: the limbs it would aggregate cannot be live either.
    assert!(
        robot_is_ready::poll(&harness, UNSERVED).await.is_err(),
        "an unseated robot serves no readiness"
    );
    let seat = goal.accept(seated()).await?;

    // Its commands arrive at the command rate, holding every limb until a
    // relay says otherwise.
    let (request, responder) = simulation.command.next_request(WIRE).await?;
    assert_eq!(request.arms.len(), 2);
    assert!(request.arms.iter().all(|arm| arm.positions.is_empty()));
    assert!(request.grippers.iter().all(|gripper| !gripper.commanded));
    responder.respond(accepted()).await?;

    // Seated, the robot serves readiness, and reports not-ready while its
    // limbs are the silent mocks the seat's own test leaves them.
    assert!(
        !robot_is_ready::poll(&harness, WIRE).await?.ready,
        "a seated robot whose limbs are not up is not ready"
    );

    // A state fed back lands on the limb it measures, whichever position
    // the engine listed that limb at.
    seat.publish_feedback(&state(0.1, 0.9)).await?;
    let left = next_state(mocks.pairings.left_arm_link.joint_states.next()).await?;
    assert_eq!(left.positions, vec![0.1; ARM_DOF]);
    let right = next_state(mocks.pairings.right_arm_link.joint_states.next()).await?;
    assert_eq!(right.positions, vec![0.9; ARM_DOF]);
    let left_gripper = next_state(mocks.pairings.left_gripper_link.gripper_states.next()).await?;
    assert_eq!(left_gripper.opening, 0.75);
    let right_gripper = next_state(mocks.pairings.right_gripper_link.gripper_states.next()).await?;
    assert_eq!(right_gripper.opening, 0.25);
    assert_eq!(
        (left_gripper.max_effort, right_gripper.max_effort),
        (0.0, 0.0),
        "a gripper nothing has commanded is under no effort control"
    );

    // A relay's setpoint goes out in a command, at the engine's index for
    // that limb.
    mocks
        .pairings
        .left_arm_link
        .joint_setpoints
        .publish(&left_arm_link::joint_setpoints::Message {
            timestamp: SystemTime::now(),
            positions: vec![0.5; ARM_DOF],
            velocities: Vec::new(),
            efforts: Vec::new(),
        })
        .await?;
    mocks
        .pairings
        .right_gripper_link
        .gripper_setpoints
        .publish(&right_gripper_link::gripper_setpoints::Message {
            timestamp: SystemTime::now(),
            opening: 0.3,
            max_effort: 2.0,
        })
        .await?;
    let mut landed = false;
    for tick in 0..COMMANDS_TO_LAND {
        let (request, responder) = simulation.command.next_request(WIRE).await?;
        // The engine answers the first command as still standing this robot.
        // The robot keeps commanding through that, so the setpoints below
        // still land.
        responder
            .respond(if tick == 0 { joining() } else { accepted() })
            .await?;
        if request.arms[1].positions == vec![0.5; ARM_DOF] && request.grippers[0].commanded {
            assert!(
                request.arms[0].positions.is_empty(),
                "the right arm was never commanded"
            );
            assert_eq!(request.grippers[0].opening, 0.3);
            assert_eq!(request.grippers[0].max_effort, 2.0);
            assert!(
                !request.grippers[1].commanded,
                "the left gripper was never commanded"
            );
            landed = true;
            break;
        }
    }
    assert!(landed, "the setpoints went out in a command");

    // The ceiling the engine now holds reaches the leader that set it, and
    // the gripper nobody capped still reports none.
    seat.publish_feedback(&state(0.1, 0.9)).await?;
    let right_gripper = next_state(mocks.pairings.right_gripper_link.gripper_states.next()).await?;
    assert_eq!(right_gripper.max_effort, 2.0);
    let left_gripper = next_state(mocks.pairings.left_gripper_link.gripper_states.next()).await?;
    assert_eq!(left_gripper.max_effort, 0.0);

    // The engine ends the stay: the node's life ends with its seat.
    seat.complete(&attach::ResultResponseData {
        success: false,
        message: "the robot's lease lapsed".into(),
    })
    .await?;
    let token = harness.node_runner().cancellation_token().clone();
    tokio::time::timeout(WIRE, token.cancelled())
        .await
        .expect("the node stops when its seat ends");
    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_stands_where_it_is_told() -> peppygen::Result<()> {
    let placed = Config {
        parameters: Some(common::standing_at("openarm_v2", 1.0, -2.0, 0.0, 0.5)),
        ..joining_a_simulation("openarm_v2")
    };
    let (harness, mut mocks) = Harness::start_with(placed, openarm_initializer::setup).await?;
    let mut simulation = mocks
        .deps
        .simulation
        .take()
        .expect("a robot joining a simulation has one bound");
    let goal = simulation.attach.next_goal(WIRE).await?;
    let spot = goal
        .request
        .placement
        .clone()
        .expect("this robot named its spot");
    assert_eq!((spot.position, spot.yaw), ([1.0, -2.0, 0.0], 0.5));
    goal.reject(Some("that is enough of this robot"), None)
        .await?;
    let _ = harness.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_seated_robot_names_the_model_the_simulation_stands() -> peppygen::Result<()> {
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation(""), openarm_initializer::setup).await?;
    let mut simulation = mocks
        .deps
        .simulation
        .take()
        .expect("a robot joining a simulation has one bound");
    assert!(
        simulation.attach.next_goal(UNSERVED).await.is_err(),
        "a robot with no model to stand never attaches"
    );
    let failure = harness
        .shutdown()
        .await
        .expect_err("a robot with no model fails to start")
        .to_string();
    assert!(
        failure.contains("names no model for it to stand")
            && failure.contains("such as openarm_v2"),
        "{failure}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refused_robot_never_commands_the_engine() -> peppygen::Result<()> {
    let (harness, mut mocks) = Harness::start_with(
        joining_a_simulation("openarm_v1"),
        openarm_initializer::setup,
    )
    .await?;
    let mut simulation = mocks
        .deps
        .simulation
        .take()
        .expect("a robot joining a simulation has one bound");
    let goal = simulation.attach.next_goal(WIRE).await?;
    assert_eq!(goal.request.model, "openarm_v1");
    goal.reject(Some("robot 'other' stands within 1.5 m"), None)
        .await?;

    // Setup fails, so no command loop ever runs.
    let deadline = tokio::time::Instant::now() + WIRE;
    while !harness.setup_finished() {
        assert!(tokio::time::Instant::now() < deadline, "setup returned");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        simulation.command.next_request(UNSERVED).await.is_err(),
        "a refused robot sends no command"
    );
    // A robot the simulation would not stand reports no readiness either.
    assert!(
        robot_is_ready::poll(&harness, UNSERVED).await.is_err(),
        "a refused robot serves no readiness"
    );
    // The node's failure carries the engine's reason.
    let failure = harness
        .shutdown()
        .await
        .expect_err("a refused robot fails to start")
        .to_string();
    assert!(
        failure.contains("the simulation refused this robot: robot 'other' stands within 1.5 m"),
        "{failure}"
    );
    Ok(())
}
