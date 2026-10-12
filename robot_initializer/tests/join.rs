//! Joining a simulation over the wire, the engine played by its generated
//! mock: the node attaches as the name it runs under with the model it is,
//! stands where it is told, starts only once the engine stands it, says so
//! while it waits, fails with the engine's reason or its own when the robot
//! does not stand, cancels every goal the engine accepted when it stops, and
//! serves no readiness for a robot that does not stand.
//!
//! The tests of the wait for standing run it on a clock the test drives, on
//! the stand budget of the node's manifest, so no test reaches a deadline it
//! does not spend itself.

use std::time::Duration;

use peppygen::fixtures::exposed_services::robot_ready::is_ready as robot_is_ready;
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::simulation::attach;

mod common;
use common::{
    ScriptedClock, await_setup_return, capture_logs, joining_a_simulation, logged,
    on_its_own_hardware, scripted_clock, setup_on, shutdown_once_setup_returns, simulation_of,
    stood,
};

/// How long the wire may take for any one exchange.
const WIRE: Duration = Duration::from_secs(10);
/// How long a service that nothing serves is waited on before concluding it
/// is not being served.
const UNSERVED: Duration = Duration::from_millis(500);
/// How long a standing robot is watched serving readiness while its goal
/// runs on.
const A_LONG_STAY: Duration = Duration::from_secs(3);
/// The stand budget of the node's manifest: its setup budget of 180 s less
/// the 5.5 s it keeps to leave the simulation.
const STAND_BUDGET: &str = "174.5 s";

/// The engine's result for a robot it took out because its goal was
/// cancelled.
fn left(message: &str) -> attach::ResultResponseData {
    attach::ResultResponseData {
        success: true,
        message: message.into(),
    }
}

/// Stops the node, and plays the engine that takes its robot out: the node
/// cancels the robot's goal, and the engine ends it as cancelled. Answers
/// the node's own account of its setup. That the shutdown hook waits for the
/// robot to leave is the unit tests' of the hook, on paused time.
async fn stop_and_take_out(harness: Harness, stay: &attach::ActiveGoal) -> peppygen::Result<()> {
    let stopping = tokio::spawn(harness.shutdown());
    tokio::time::timeout(WIRE, stay.cancel_signal())
        .await
        .expect("a stopping robot cancels its goal");
    stay.complete_cancelled(&left("the robot left the scene"))
        .await?;
    tokio::time::timeout(WIRE, stopping)
        .await
        .expect("a robot the engine took out stops")
        .expect("the stopping task ran to its end")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_robot_joins_the_scene_under_its_own_name() -> peppygen::Result<()> {
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), robot_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);

    // The node attaches as the model the launcher wrote, under the name it
    // runs as, on a spot of the engine's choosing.
    let goal = simulation.attach.next_goal(WIRE).await?;
    assert_eq!(goal.request.model, "openarm_v2");
    assert_eq!(goal.request.robot, harness.instance_id());
    assert!(goal.request.placement.is_none());

    // Readiness waits on the scene, so a robot the engine has not stood yet
    // answers nothing.
    assert!(
        robot_is_ready::poll(&harness, UNSERVED).await.is_err(),
        "a robot the scene has not stood serves no readiness"
    );

    let stay = goal.accept(stood()).await?;
    stay.publish_feedback(&attach::FeedbackMessage { standing: true })
        .await?;

    // Standing, the robot serves readiness, and reports not-ready while the
    // engine's own readiness is the silent mock this test leaves it.
    assert!(
        !robot_is_ready::poll(&harness, WIRE).await?.ready,
        "a robot its simulation has not answered for is not ready"
    );

    // The engine takes the robot out, which ends the node with it.
    stay.complete(&attach::ResultResponseData {
        success: true,
        message: "the scene was cleared".into(),
    })
    .await?;
    let deadline = tokio::time::Instant::now() + WIRE;
    while robot_is_ready::poll(&harness, UNSERVED).await.is_ok() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "a robot taken out of the scene stops serving readiness"
        );
    }
    let _ = harness.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_starts_once_the_simulation_stands_it_and_says_so_while_it_waits()
-> peppygen::Result<()> {
    capture_logs();
    let (clock, scripted) = scripted_clock();
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), setup_on(scripted)).await?;
    let mut simulation = simulation_of(&mut mocks);
    let robot = harness.instance_id().to_owned();
    let goal = simulation.attach.next_goal(WIRE).await?;

    // The wait for the engine's answer spends the stand budget too, so the
    // robot says that it does not stand yet before the engine answers.
    clock.progress(Duration::from_secs(10));
    logged(&format!(
        "'{robot}' does not stand yet: 10 s of {STAND_BUDGET}"
    ))
    .await;

    // Admitted, the robot says how long the simulation has to stand it, and
    // goes on saying that it does not stand yet.
    let stay = goal.accept(stood()).await?;
    logged(&format!(
        "'{robot}' was admitted to the simulation as openarm_v2; it starts once the simulation stands it, within 164.5 s"
    ))
    .await;
    clock.progress(Duration::from_secs(20));
    logged(&format!(
        "'{robot}' does not stand yet: 20 s of {STAND_BUDGET}"
    ))
    .await;
    assert!(
        !harness.setup_finished(),
        "an admitted robot that does not stand yet has not started"
    );

    // The engine stands the robot: its setup returns, and it serves
    // readiness from then on.
    stay.publish_feedback(&attach::FeedbackMessage { standing: true })
        .await?;
    logged(&format!(
        "'{robot}' stands in the simulation, with arms [right, left] and grippers [right, left]"
    ))
    .await;
    await_setup_return(&harness).await;
    robot_is_ready::poll(&harness, WIRE).await?;

    stop_and_take_out(harness, &stay).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_the_engine_ends_before_it_stands_fails_with_the_engines_reason()
-> peppygen::Result<()> {
    let (_clock, scripted) = scripted_clock();
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), setup_on(scripted)).await?;
    let mut simulation = simulation_of(&mut mocks);
    let stay = simulation
        .attach
        .next_goal(WIRE)
        .await?
        .accept(stood())
        .await?;

    // The engine admitted the robot and then could not stand it: the goal
    // ends before the robot ever stood, and the node fails with the
    // engine's reason.
    stay.complete(&attach::ResultResponseData {
        success: false,
        message: "its files could not be fetched".into(),
    })
    .await?;
    let failure = shutdown_once_setup_returns(harness)
        .await
        .expect_err("a robot the engine did not stand fails to start")
        .to_string();
    assert!(
        failure.contains("the simulation did not stand this robot: its files could not be fetched"),
        "{failure}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_whose_engine_is_gone_before_it_stands_fails() -> peppygen::Result<()> {
    let (_clock, scripted) = scripted_clock();
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), setup_on(scripted)).await?;
    let mut simulation = simulation_of(&mut mocks);
    let _stay = simulation
        .attach
        .next_goal(WIRE)
        .await?
        .accept(stood())
        .await?;

    // The engine dies before it stands the robot.
    simulation.stop();
    let failure = shutdown_once_setup_returns(harness)
        .await
        .expect_err("a robot whose engine is gone fails to start")
        .to_string();
    assert!(
        failure.contains("the simulation did not stand this robot: the simulation is gone"),
        "{failure}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_that_does_not_stand_within_its_budget_leaves_and_fails() -> peppygen::Result<()> {
    capture_logs();
    let (clock, scripted) = scripted_clock();
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), setup_on(scripted)).await?;
    let mut simulation = simulation_of(&mut mocks);
    let robot = harness.instance_id().to_owned();
    let stay = simulation
        .attach
        .next_goal(WIRE)
        .await?
        .accept(stood())
        .await?;
    logged(&format!("'{robot}' was admitted to the simulation")).await;

    // The budget runs out: the robot leaves the simulation, and fails.
    clock.spend();
    tokio::time::timeout(WIRE, stay.cancel_signal())
        .await
        .expect("a robot out of budget cancels its goal");
    stay.complete_cancelled(&left("the robot left before it stood"))
        .await?;
    let failure = shutdown_once_setup_returns(harness)
        .await
        .expect_err("a robot that did not stand in time fails to start")
        .to_string();
    assert!(
        failure.contains(&format!(
            "the simulation did not stand this robot within {STAND_BUDGET}"
        )),
        "{failure}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_out_of_budget_before_the_engine_answers_fails_and_cancels_the_goal_it_accepts()
-> peppygen::Result<()> {
    let (clock, scripted) = scripted_clock();
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), setup_on(scripted)).await?;
    let mut simulation = simulation_of(&mut mocks);
    let goal = simulation.attach.next_goal(WIRE).await?;

    // The budget runs out while the engine has not answered: the setup
    // fails at once, and the node waits for the answer.
    clock.spend();
    await_setup_return(&harness).await;

    // The engine accepts the goal after all, and the node cancels it at
    // once, so no robot stays in the scene for a node that failed.
    let stay = goal.accept(stood()).await?;
    tokio::time::timeout(WIRE, stay.cancel_signal())
        .await
        .expect("a robot out of budget cancels the goal the engine accepts");
    stay.complete_cancelled(&left("the robot left before it stood"))
        .await?;
    let failure = harness
        .shutdown()
        .await
        .expect_err("a robot that did not stand in time fails to start")
        .to_string();
    assert!(
        failure.contains(&format!(
            "the simulation did not stand this robot within {STAND_BUDGET}"
        )),
        "{failure}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_stopped_before_the_engine_answers_cancels_the_goal_it_accepts()
-> peppygen::Result<()> {
    let (_clock, scripted) = scripted_clock();
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), setup_on(scripted)).await?;
    let mut simulation = simulation_of(&mut mocks);
    let goal = simulation.attach.next_goal(WIRE).await?;

    // The node stops while the engine has not answered: its setup ends at
    // once, and the node waits for the answer.
    harness.node_runner().cancellation_token().cancel();
    await_setup_return(&harness).await;

    // The engine accepts the goal, and the node cancels it at once, so no
    // robot stays in the scene for a node that stopped.
    let stay = goal.accept(stood()).await?;
    tokio::time::timeout(WIRE, stay.cancel_signal())
        .await
        .expect("a stopped robot cancels the goal the engine accepts");
    stay.complete_cancelled(&left("the robot left before it stood"))
        .await?;

    // A stop is no failure.
    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_stopped_before_it_stands_leaves_the_scene() -> peppygen::Result<()> {
    capture_logs();
    let (_clock, scripted) = scripted_clock();
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), setup_on(scripted)).await?;
    let mut simulation = simulation_of(&mut mocks);
    let robot = harness.instance_id().to_owned();
    let stay = simulation
        .attach
        .next_goal(WIRE)
        .await?
        .accept(stood())
        .await?;
    logged(&format!("'{robot}' was admitted to the simulation")).await;

    // Admitted and not standing yet, the robot is stopped: it takes itself
    // out, and the stop is no failure.
    stop_and_take_out(harness, &stay).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_on_its_own_hardware_joins_no_simulation_and_starts_at_once() -> peppygen::Result<()>
{
    let (harness, _mocks) = Harness::start_with(on_its_own_hardware(), |params, runner| {
        robot_initializer::setup_with_clock(params, runner, |_budget| -> ScriptedClock {
            unreachable!("a robot with no simulation waits for no stand")
        })
    })
    .await?;

    await_setup_return(&harness).await;
    robot_is_ready::poll(&harness, WIRE).await?;
    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_stays_until_the_engine_ends_its_goal() -> peppygen::Result<()> {
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), robot_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);
    let stay = simulation
        .attach
        .next_goal(WIRE)
        .await?
        .accept(stood())
        .await?;

    // The engine says the robot stands and keeps the goal open: the robot
    // is in the scene, and stays, serving readiness.
    stay.publish_feedback(&attach::FeedbackMessage { standing: true })
        .await?;
    await_setup_return(&harness).await;
    tokio::time::sleep(A_LONG_STAY).await;
    assert!(
        robot_is_ready::poll(&harness, WIRE).await.is_ok(),
        "a robot whose goal still runs keeps serving readiness"
    );

    // The engine ends the goal: the robot is out of the scene, and the node
    // stops with it.
    stay.complete(&attach::ResultResponseData {
        success: false,
        message: "the scene was cleared".into(),
    })
    .await?;
    stopped_within(&harness, WIRE).await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_whose_engine_is_gone_stops() -> peppygen::Result<()> {
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), robot_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);
    let stay = simulation
        .attach
        .next_goal(WIRE)
        .await?
        .accept(stood())
        .await?;
    stay.publish_feedback(&attach::FeedbackMessage { standing: true })
        .await?;
    await_setup_return(&harness).await;

    // The engine dies without ending the goal: the stay is over all the
    // same, and the node stops.
    simulation.stop();
    stopped_within(&harness, WIRE).await;
    Ok(())
}

/// Waits for the node to stop serving readiness, which is how a stay that
/// ended shows from outside.
async fn stopped_within(harness: &Harness, within: Duration) {
    let deadline = tokio::time::Instant::now() + within;
    while robot_is_ready::poll(harness, WIRE).await.is_ok() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "a robot whose stay ended stops serving readiness"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stopping_robot_waits_for_the_engine_to_take_it_out() -> peppygen::Result<()> {
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), robot_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);
    let stay = simulation
        .attach
        .next_goal(WIRE)
        .await?
        .accept(stood())
        .await?;
    stay.publish_feedback(&attach::FeedbackMessage { standing: true })
        .await?;
    await_setup_return(&harness).await;

    // Stopping the node cancels the robot's goal, and the node stops only
    // once the engine's result says the robot is out.
    stop_and_take_out(harness, &stay).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_stands_where_it_is_told() -> peppygen::Result<()> {
    let placed = Config {
        parameters: Some(common::standing_at("openarm_v2", 1.0, -2.0, 0.0, 0.5)),
        ..joining_a_simulation("openarm_v2")
    };
    let (harness, mut mocks) = Harness::start_with(placed, robot_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);
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
async fn a_robot_attaches_as_the_model_it_was_launched_as() -> peppygen::Result<()> {
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("so101"), robot_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);

    // The model reaches the engine as the launcher wrote it: which models
    // exist is the engine's to say, and this node adds nothing to the name.
    let goal = simulation.attach.next_goal(WIRE).await?;
    assert_eq!(goal.request.model, "so101");
    goal.reject(Some("that is enough of this robot"), None)
        .await?;
    let _ = harness.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_of_no_model_is_refused() -> peppygen::Result<()> {
    for blank in ["", "  "] {
        let (harness, mut mocks) =
            Harness::start_with(joining_a_simulation(blank), robot_initializer::setup).await?;
        let mut simulation = simulation_of(&mut mocks);

        // Setup has returned, so a goal it never fired never arrives.
        await_setup_return(&harness).await;
        assert!(
            simulation.attach.next_goal(UNSERVED).await.is_err(),
            "a robot of no model never attaches"
        );
        let failure = harness
            .shutdown()
            .await
            .expect_err("a robot of no model fails to start")
            .to_string();
        assert!(
            failure.contains("model names the robot's model"),
            "{failure}"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refused_robot_serves_no_readiness() -> peppygen::Result<()> {
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v1"), robot_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);
    let goal = simulation.attach.next_goal(WIRE).await?;
    assert_eq!(goal.request.model, "openarm_v1");
    goal.reject(Some("robot 'other' stands within 1.5 m"), None)
        .await?;

    // Setup fails, so the readiness service never starts.
    await_setup_return(&harness).await;
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
