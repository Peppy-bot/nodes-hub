//! The seat over the wire, the engine played by its generated mock: the
//! node attaches as its model, publishes every state the engine feeds back
//! on the limb it measures, carries the relays' setpoints across in its
//! commands, reads why the engine ended its stay, and never commands an
//! engine that refused it.

use std::time::{Duration, SystemTime};

use peppygen::Parameters;
use peppygen::consumed_actions::engine::attach::{
    SimulationRobotAttachActionFeedbackMessageArmsItem as ArmState,
    SimulationRobotAttachActionFeedbackMessageGrippersItem as GripperState,
};
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::engine::{attach, command};
use peppygen::mock::pairings::{left_arm, right_gripper};
use peppygen::parameters::placement::Placement;

/// How long the wire may take for any one exchange.
const WIRE: Duration = Duration::from_secs(10);
/// Joints of one OpenArm arm, as the engine reports them.
const ARM_DOF: usize = 7;
/// Commands drained while a setpoint published beside a tick lands in one.
const COMMANDS_TO_LAND: usize = 50;

fn parameters(hardware_version: &str) -> Parameters {
    Parameters {
        command_rate_hz: 50,
        hardware_version: hardware_version.into(),
        placement: Placement {
            auto: true,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            yaw: 0.0,
        },
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
        message: String::new(),
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
        Config {
            parameters: Some(parameters("v2")),
            ..Config::default()
        },
        openarm_sim_attachment::setup,
    )
    .await?;

    // The node attaches as its model, on a spot of the engine's choosing.
    let goal = mocks.deps.engine.attach.next_goal(WIRE).await?;
    assert_eq!(goal.request.model, "openarm_v2");
    assert!(goal.request.placement.is_none());
    let seat = goal.accept(seated()).await?;

    // Its commands arrive at the command rate, holding every limb until a
    // relay says otherwise.
    let (request, responder) = mocks.deps.engine.command.next_request(WIRE).await?;
    assert_eq!(request.arms.len(), 2);
    assert!(request.arms.iter().all(|arm| arm.positions.is_empty()));
    assert!(request.grippers.iter().all(|gripper| !gripper.commanded));
    responder.respond(accepted()).await?;

    // A state fed back lands on the limb it measures, whichever position
    // the engine listed that limb at.
    seat.publish_feedback(&state(0.1, 0.9)).await?;
    let left = next_state(mocks.pairings.left_arm.joint_states.next()).await?;
    assert_eq!(left.positions, vec![0.1; ARM_DOF]);
    let right = next_state(mocks.pairings.right_arm.joint_states.next()).await?;
    assert_eq!(right.positions, vec![0.9; ARM_DOF]);
    let left_gripper = next_state(mocks.pairings.left_gripper.gripper_states.next()).await?;
    assert_eq!(left_gripper.opening, 0.75);
    let right_gripper = next_state(mocks.pairings.right_gripper.gripper_states.next()).await?;
    assert_eq!(right_gripper.opening, 0.25);

    // A relay's setpoint goes out in a command, at the engine's index for
    // that limb.
    mocks
        .pairings
        .left_arm
        .joint_setpoints
        .publish(&left_arm::joint_setpoints::Message {
            timestamp: SystemTime::now(),
            positions: vec![0.5; ARM_DOF],
            velocities: Vec::new(),
            efforts: Vec::new(),
        })
        .await?;
    mocks
        .pairings
        .right_gripper
        .gripper_setpoints
        .publish(&right_gripper::gripper_setpoints::Message {
            timestamp: SystemTime::now(),
            opening: 0.3,
            max_effort: 2.0,
        })
        .await?;
    let mut landed = false;
    for _ in 0..COMMANDS_TO_LAND {
        let (request, responder) = mocks.deps.engine.command.next_request(WIRE).await?;
        responder.respond(accepted()).await?;
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
async fn a_refused_robot_never_commands_the_engine() -> peppygen::Result<()> {
    let (harness, mut mocks) = Harness::start_with(
        Config {
            parameters: Some(parameters("v1")),
            ..Config::default()
        },
        openarm_sim_attachment::setup,
    )
    .await?;
    let goal = mocks.deps.engine.attach.next_goal(WIRE).await?;
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
        mocks
            .deps
            .engine
            .command
            .next_request(Duration::from_millis(500))
            .await
            .is_err(),
        "a refused robot sends no command"
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
