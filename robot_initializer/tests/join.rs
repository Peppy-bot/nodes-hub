//! Joining a simulation over the wire, the engine played by its generated
//! mock: the node attaches as the name it runs under with the model it is,
//! stands where it is told, starts only once the engine stands it, says so
//! while it waits, attaches again while the engine refuses it with `busy`,
//! fails with the engine's failure kind and reason or its own when the
//! robot does not stand, cancels every goal the engine accepted when it
//! stops, serves no readiness for a robot that does not stand, and says how
//! the stay of a robot that stood ended.
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
    ScriptedClock, await_setup_return, capture_logs, joining_a_simulation, logged, logged_times,
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

/// The reason the engine ends the goal of a robot it took out because its
/// caller cancelled the goal.
const CANCELLED_BY_ITS_CALLER: &str = "cancelled by its caller";

/// What the engine says when it refuses a robot with `busy` because the
/// robot of its name still leaves the scene.
const STILL_LEAVING: &str = "'alpha' is leaving the world; attach again once it is out";

/// Stops the node, and plays the engine that takes its robot out: the node
/// cancels the robot's goal, and the engine ends it as cancelled. Answers
/// the node's own account of its setup. That the shutdown hook waits for the
/// robot to leave is the unit tests' of the hook, on paused time.
async fn stop_and_take_out(harness: Harness, stay: &attach::ActiveGoal) -> peppygen::Result<()> {
    let stopping = tokio::spawn(harness.shutdown());
    tokio::time::timeout(WIRE, stay.cancel_signal())
        .await
        .expect("a stopping robot cancels its goal");
    stay.complete_cancelled(CANCELLED_BY_ITS_CALLER).await?;
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
    stay.fail(
        attach::FailureKind::ResourceUnavailable,
        "the scene was cleared",
    )
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
    // fails before the robot ever stood, and the node fails with the
    // engine's failure kind and reason.
    stay.fail(
        attach::FailureKind::AttemptFailed,
        "its files could not be fetched",
    )
    .await?;
    let failure = shutdown_once_setup_returns(harness)
        .await
        .expect_err("a robot the engine did not stand fails to start")
        .to_string();
    assert!(
        failure.contains(
            "the simulation did not stand this robot: attempt_failed: its files could not be fetched"
        ),
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
    stay.complete_cancelled(CANCELLED_BY_ITS_CALLER).await?;
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
    stay.complete_cancelled(CANCELLED_BY_ITS_CALLER).await?;
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
    stay.complete_cancelled(CANCELLED_BY_ITS_CALLER).await?;

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
async fn a_robot_refused_with_busy_attaches_again_and_stands() -> peppygen::Result<()> {
    capture_logs();
    let (clock, scripted) = scripted_clock();
    let placed = Config {
        parameters: Some(common::standing_at("openarm_v2", 1.0, -2.0, 0.0, 0.5)),
        ..joining_a_simulation("openarm_v2")
    };
    let (harness, mut mocks) = Harness::start_with(placed, setup_on(scripted)).await?;
    let mut simulation = simulation_of(&mut mocks);
    let robot = harness.instance_id().to_owned();

    // The robot of its name still leaves the scene: the engine refuses it
    // with `busy`, and the robot says that it waits.
    simulation
        .attach
        .next_goal(WIRE)
        .await?
        .refuse(attach::FailureKind::Busy, STILL_LEAVING)
        .await?;
    logged(&format!(
        "the simulation cannot take '{robot}' yet (busy: {STILL_LEAVING}); it attaches again in 1 s"
    ))
    .await;
    assert!(
        simulation.attach.next_goal(UNSERVED).await.is_err(),
        "the robot attaches again only once its wait is over"
    );

    // While it waits, it goes on saying that it does not stand yet.
    clock.progress(Duration::from_secs(10));
    logged(&format!(
        "'{robot}' does not stand yet: 10 s of {STAND_BUDGET}"
    ))
    .await;
    assert!(
        !harness.setup_finished(),
        "a robot that waits for its name has not failed"
    );

    // Its wait over, the robot sends the same goal again, and the engine
    // admits it and stands it.
    clock.resend();
    let again = simulation.attach.next_goal(WIRE).await?;
    assert_eq!(again.request.robot, robot);
    assert_eq!(again.request.model, "openarm_v2");
    let spot = again
        .request
        .placement
        .clone()
        .expect("the robot names its spot again");
    assert_eq!((spot.position, spot.yaw), ([1.0, -2.0, 0.0], 0.5));
    let stay = again.accept(stood()).await?;
    stay.publish_feedback(&attach::FeedbackMessage { standing: true })
        .await?;
    await_setup_return(&harness).await;
    robot_is_ready::poll(&harness, WIRE).await?;

    stop_and_take_out(harness, &stay).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_refused_for_another_reason_after_it_waited_fails_at_once() -> peppygen::Result<()>
{
    let (clock, scripted) = scripted_clock();
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), setup_on(scripted)).await?;
    let mut simulation = simulation_of(&mut mocks);
    simulation
        .attach
        .next_goal(WIRE)
        .await?
        .refuse(attach::FailureKind::Busy, STILL_LEAVING)
        .await?;

    // Once the robot of its name is out, another robot stands too close:
    // a refusal that does not pass, so the robot fails without waiting.
    clock.resend();
    simulation
        .attach
        .next_goal(WIRE)
        .await?
        .refuse(
            attach::FailureKind::PreconditionFailed,
            "robot 'other' stands within 1.5 m",
        )
        .await?;
    await_setup_return(&harness).await;
    clock.resend();
    assert!(
        simulation.attach.next_goal(UNSERVED).await.is_err(),
        "a robot refused for good attaches no more"
    );
    let failure = harness
        .shutdown()
        .await
        .expect_err("a refused robot fails to start")
        .to_string();
    assert!(
        failure.contains(
            "the simulation refused this robot: precondition_failed: robot 'other' stands within 1.5 m"
        ),
        "{failure}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_refused_with_busy_until_its_budget_is_spent_fails() -> peppygen::Result<()> {
    capture_logs();
    let (clock, scripted) = scripted_clock();
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), setup_on(scripted)).await?;
    let mut simulation = simulation_of(&mut mocks);
    let waits = format!("the simulation cannot take '{}' yet", harness.instance_id());
    for _ in 0..2 {
        simulation
            .attach
            .next_goal(WIRE)
            .await?
            .refuse(attach::FailureKind::Busy, STILL_LEAVING)
            .await?;
        clock.resend();
    }
    simulation
        .attach
        .next_goal(WIRE)
        .await?
        .refuse(attach::FailureKind::Busy, STILL_LEAVING)
        .await?;

    // The robot of its name is still not out when the stand budget is
    // spent, while the robot waits after its third goal: the robot fails,
    // with the engine's last refusal.
    logged_times(&waits, 3).await;
    clock.spend();
    await_setup_return(&harness).await;
    clock.resend();
    assert!(
        simulation.attach.next_goal(UNSERVED).await.is_err(),
        "a robot out of budget attaches no more"
    );
    let failure = harness
        .shutdown()
        .await
        .expect_err("a robot that did not stand in time fails to start")
        .to_string();
    assert!(
        failure.contains(&format!(
            "the simulation did not stand this robot within {STAND_BUDGET}: until then it refused this robot with busy: {STILL_LEAVING}"
        )),
        "{failure}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_stopped_while_it_waits_to_attach_again_stops() -> peppygen::Result<()> {
    capture_logs();
    let (clock, scripted) = scripted_clock();
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), setup_on(scripted)).await?;
    let mut simulation = simulation_of(&mut mocks);
    let robot = harness.instance_id().to_owned();
    simulation
        .attach
        .next_goal(WIRE)
        .await?
        .refuse(attach::FailureKind::Busy, STILL_LEAVING)
        .await?;
    logged(&format!("the simulation cannot take '{robot}' yet")).await;

    // The node stops while it waits: the wait ends, and nothing more is
    // sent, even once the wait would have been over.
    harness.node_runner().cancellation_token().cancel();
    await_setup_return(&harness).await;
    clock.resend();
    assert!(
        simulation.attach.next_goal(UNSERVED).await.is_err(),
        "a stopped robot attaches no more"
    );

    // A stop is no failure.
    harness.shutdown().await
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

    // The engine takes the robot out: its goal fails, the robot is out of
    // the scene, and the node stops with it.
    stay.fail(
        attach::FailureKind::ResourceUnavailable,
        "the scene was cleared",
    )
    .await?;
    stopped_within(&harness, WIRE).await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_the_engine_takes_out_says_why_and_stops() -> peppygen::Result<()> {
    capture_logs();
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), robot_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);
    let robot = harness.instance_id().to_owned();
    let stay = simulation
        .attach
        .next_goal(WIRE)
        .await?
        .accept(stood())
        .await?;
    stay.publish_feedback(&attach::FeedbackMessage { standing: true })
        .await?;
    await_setup_return(&harness).await;

    // The engine takes the robot out for a reason of its own: the node
    // logs the failure kind and the reason, and stops.
    stay.fail(
        attach::FailureKind::ResourceUnavailable,
        "none of this robot's limbs were paired for 2.0s",
    )
    .await?;
    logged(&format!(
        "the simulation took '{robot}' out of the scene: resource_unavailable: none of this robot's limbs were paired for 2.0s"
    ))
    .await;
    stopped_within(&harness, WIRE).await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_another_goal_of_its_copy_holds_says_so_and_stops() -> peppygen::Result<()> {
    capture_logs();
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("openarm_v2"), robot_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);
    let robot = harness.instance_id().to_owned();
    let stay = simulation
        .attach
        .next_goal(WIRE)
        .await?
        .accept(stood())
        .await?;
    stay.publish_feedback(&attach::FeedbackMessage { standing: true })
        .await?;
    await_setup_return(&harness).await;

    // Another goal of the same copy holds the robot now: this goal ends
    // cancelled, with that reason, and the node stops. The robot stays in
    // the scene for the goal that holds it.
    stay.complete_cancelled("another goal of the same caller holds the robot now")
        .await?;
    logged(&format!(
        "the stay of '{robot}' in the simulation ended: another goal of the same caller holds the robot now"
    ))
    .await;
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
    goal.refuse(
        attach::FailureKind::Rejected,
        "that is enough of this robot",
    )
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
    goal.refuse(
        attach::FailureKind::Rejected,
        "that is enough of this robot",
    )
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
    goal.refuse(
        attach::FailureKind::PreconditionFailed,
        "robot 'other' stands within 1.5 m",
    )
    .await?;

    // Setup fails, so the readiness service never starts.
    await_setup_return(&harness).await;
    assert!(
        robot_is_ready::poll(&harness, UNSERVED).await.is_err(),
        "a refused robot serves no readiness"
    );
    // The node's failure carries the engine's failure kind and reason.
    let failure = harness
        .shutdown()
        .await
        .expect_err("a refused robot fails to start")
        .to_string();
    assert!(
        failure.contains(
            "the simulation refused this robot: precondition_failed: robot 'other' stands within 1.5 m"
        ),
        "{failure}"
    );
    Ok(())
}
