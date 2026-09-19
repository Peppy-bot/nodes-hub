//! Joining a simulation over the wire, the engine played by its generated
//! mock: the node attaches as the name it runs under with the model it is,
//! stands where it is told, reads why the engine ended its stay, waits for
//! the engine to take it out when it stops, and serves no readiness for a
//! robot the engine refused to stand.

use std::time::Duration;

use peppygen::fixtures::exposed_services::robot_ready::is_ready as robot_is_ready;
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::simulation::attach;

mod common;
use common::{
    await_setup_return, joining_a_simulation, shutdown_once_setup_returns, simulation_of, stood,
};

/// How long the wire may take for any one exchange.
const WIRE: Duration = Duration::from_secs(10);
/// How long a service that nothing serves is waited on before concluding it
/// is not being served.
const UNSERVED: Duration = Duration::from_millis(500);
/// How long a standing robot is watched serving readiness while its goal
/// runs on.
const A_LONG_STAY: Duration = Duration::from_secs(3);
/// How long the engine takes to take a robot out, shorter than the node waits
/// for its result, so a node that stops without waiting shows.
const TAKING_OUT: Duration = Duration::from_secs(1);

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_robot_joins_the_scene_under_its_own_name() -> peppygen::Result<()> {
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("v2"), openarm_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);

    // The node attaches as the model of its generation, under the name it
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
async fn a_robot_stays_until_the_engine_ends_its_goal() -> peppygen::Result<()> {
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("v2"), openarm_initializer::setup).await?;
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
        Harness::start_with(joining_a_simulation("v2"), openarm_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);
    let stay = simulation
        .attach
        .next_goal(WIRE)
        .await?
        .accept(stood())
        .await?;
    stay.publish_feedback(&attach::FeedbackMessage { standing: true })
        .await?;

    // The engine dies without ending the goal: the stay is over all the
    // same, and the node stops.
    simulation.stop();
    stopped_within(&harness, WIRE).await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_the_scene_could_not_stand_stops() -> peppygen::Result<()> {
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("v2"), openarm_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);
    let stay = simulation
        .attach
        .next_goal(WIRE)
        .await?
        .accept(stood())
        .await?;

    // Admitted, the robot serves readiness (the launch's health check
    // waits for it) while the engine stands it.
    assert!(
        !robot_is_ready::poll(&harness, WIRE).await?.ready,
        "a robot the engine is standing is not ready yet"
    );

    // The engine could not stand it: the goal ends before the robot ever
    // stood, and the node stops.
    stay.complete(&attach::ResultResponseData {
        success: false,
        message: "the scene could not stand the robot: its files are missing".into(),
    })
    .await?;
    stopped_within(&harness, WIRE).await;
    let _ = harness.shutdown().await;
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
        Harness::start_with(joining_a_simulation("v2"), openarm_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);
    let stay = simulation
        .attach
        .next_goal(WIRE)
        .await?
        .accept(stood())
        .await?;
    stay.publish_feedback(&attach::FeedbackMessage { standing: true })
        .await?;

    // Stopping the node cancels the robot's goal, and the node stops only
    // once the engine's result says the robot is out.
    let stopping = tokio::spawn(harness.shutdown());
    tokio::time::timeout(WIRE, stay.cancel_signal())
        .await
        .expect("a stopping robot cancels its goal");
    tokio::time::sleep(TAKING_OUT).await;
    assert!(
        !stopping.is_finished(),
        "a stopping robot waits for the engine to take it out"
    );
    stay.complete_cancelled(&attach::ResultResponseData {
        success: true,
        message: "the robot left the scene".into(),
    })
    .await?;
    let _ = tokio::time::timeout(WIRE, stopping)
        .await
        .expect("a robot the engine took out stops")
        .expect("the stopping task ran to its end");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_stands_where_it_is_told() -> peppygen::Result<()> {
    let placed = Config {
        parameters: Some(common::standing_at("v2", 1.0, -2.0, 0.0, 0.5)),
        ..joining_a_simulation("v2")
    };
    let (harness, mut mocks) = Harness::start_with(placed, openarm_initializer::setup).await?;
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
async fn a_robot_of_no_known_generation_is_refused() -> peppygen::Result<()> {
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("v3"), openarm_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);
    assert!(
        simulation.attach.next_goal(UNSERVED).await.is_err(),
        "a robot of no known generation never attaches"
    );
    let failure = shutdown_once_setup_returns(harness)
        .await
        .expect_err("a robot of no known generation fails to start")
        .to_string();
    assert!(
        failure.contains("hardware_version names an OpenArm generation, v1 or v2"),
        "{failure}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refused_robot_serves_no_readiness() -> peppygen::Result<()> {
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("v1"), openarm_initializer::setup).await?;
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
