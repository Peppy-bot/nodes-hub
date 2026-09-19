//! Integration tests over the generated harness: the node in-process, and
//! whatever answers for its readiness played by generated mocks over the
//! real wire: the drivers of a real robot's limbs, or the simulation that
//! stands a simulated one.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use peppygen::fixtures::exposed_services::robot_ready::is_ready as robot_is_ready;
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::limbs::is_ready::ResponseData as LimbReady;
use peppygen::mock::deps::simulation::attach;
use peppygen::mock::deps::simulation::is_ready::ResponseData as SimulationReady;

mod common;
use common::{
    LIMBS, joining_a_simulation, on_its_own_hardware, shutdown_once_setup_returns, simulation_of,
    stood,
};

/// How long each mock waits parked for the node's next poll. The node polls
/// every 500ms, so this only expires once the harness is gone.
const PUMP_TIMEOUT: Duration = Duration::from_secs(60);

/// Answers every `is_ready` poll from the node with the current value of
/// `flag`, until the mock's session closes.
fn pump_is_ready(
    service: peppygen::mock::deps::limbs::is_ready::Service,
    flag: &Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    let flag = Arc::clone(flag);
    tokio::spawn(async move {
        while let Ok(responder) = service.next_request(PUMP_TIMEOUT).await {
            let ready = flag.load(Ordering::SeqCst);
            if responder.respond(LimbReady { ready }).await.is_err() {
                break;
            }
        }
    })
}

/// The same, for the simulation's answer.
fn pump_simulation_is_ready(
    service: peppygen::mock::deps::simulation::is_ready::Service,
    flag: &Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    let flag = Arc::clone(flag);
    tokio::spawn(async move {
        while let Ok(responder) = service.next_request(PUMP_TIMEOUT).await {
            let ready = flag.load(Ordering::SeqCst);
            if responder.respond(SimulationReady { ready }).await.is_err() {
                break;
            }
        }
    })
}

/// Polls the node's exposed `is_ready` until it reports `want`. The node
/// re-checks its limbs every 500ms, so the deadline bounds a handful of its
/// poll passes.
async fn poll_until(harness: &Harness, want: bool, deadline: Duration) -> peppygen::Result<()> {
    let end = tokio::time::Instant::now() + deadline;
    loop {
        let response = robot_is_ready::poll(harness, Duration::from_secs(2)).await?;
        if response.ready == want {
            return Ok(());
        }
        assert!(
            tokio::time::Instant::now() < end,
            "node never reported ready={want}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reports_ready_only_when_every_limb_is() -> peppygen::Result<()> {
    let (harness, mocks) =
        Harness::start_with(on_its_own_hardware(), openarm_initializer::setup).await?;

    let flags: Vec<Arc<AtomicBool>> = [true, true, false, true]
        .into_iter()
        .map(|ready| Arc::new(AtomicBool::new(ready)))
        .collect();
    assert_eq!(mocks.deps.limbs.len(), LIMBS);
    for (mock, flag) in mocks.deps.limbs.into_iter().zip(&flags) {
        pump_is_ready(mock.is_ready, flag);
    }

    // Three of four limbs ready: the robot must keep reporting not-ready
    // across full poll passes (a pass exists once every limb was polled).
    let response = robot_is_ready::poll(&harness, Duration::from_secs(2)).await?;
    assert!(!response.ready);

    // The last limb comes up: the aggregate flips within a poll pass.
    flags[2].store(true, Ordering::SeqCst);
    poll_until(&harness, true, Duration::from_secs(10)).await?;

    // A limb reporting not-ready flips the robot back: readiness is
    // re-polled every pass.
    flags[1].store(false, Ordering::SeqCst);
    poll_until(&harness, false, Duration::from_secs(10)).await?;

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn losing_a_limb_flips_back_to_not_ready() -> peppygen::Result<()> {
    let (harness, mocks) =
        Harness::start_with(on_its_own_hardware(), openarm_initializer::setup).await?;

    let always = Arc::new(AtomicBool::new(true));
    let mut limbs = mocks.deps.limbs.into_iter();
    // One limb serves from a scripted queue, so the whole Mock stays intact
    // for `stop()` below.
    let dying = limbs.next().expect("four limb mocks");
    for _ in 0..60 {
        dying.is_ready.enqueue_response(LimbReady { ready: true })?;
    }
    for mock in limbs {
        pump_is_ready(mock.is_ready, &always);
    }

    poll_until(&harness, true, Duration::from_secs(10)).await?;

    // The limb dies: its queryable disappears with its session, the node's
    // next poll of it fails, and the aggregate must drop.
    dying.stop();
    poll_until(&harness, false, Duration::from_secs(15)).await?;

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_simulated_robot_is_as_ready_as_its_simulation_says() -> peppygen::Result<()> {
    let (harness, mut mocks) =
        Harness::start_with(joining_a_simulation("v2"), openarm_initializer::setup).await?;
    let mut simulation = simulation_of(&mut mocks);
    let stay = simulation
        .attach
        .next_goal(Duration::from_secs(10))
        .await?
        .accept(stood())
        .await?;
    stay.publish_feedback(&attach::FeedbackMessage { standing: true })
        .await?;
    assert!(
        mocks.deps.limbs.is_empty(),
        "a simulated robot has no drivers of its own"
    );

    let flag = Arc::new(AtomicBool::new(false));
    pump_simulation_is_ready(simulation.is_ready, &flag);

    // The simulation has not stood every limb yet: the robot is not ready.
    poll_until(&harness, false, Duration::from_secs(10)).await?;

    // The simulation holds every limb pair: the robot is ready.
    flag.store(true, Ordering::SeqCst);
    poll_until(&harness, true, Duration::from_secs(10)).await?;

    // A limb pair dissolves: the simulation's answer flips the robot back.
    flag.store(false, Ordering::SeqCst);
    poll_until(&harness, false, Duration::from_secs(10)).await?;

    let _ = harness.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_nothing_answers_for_is_refused() -> peppygen::Result<()> {
    let unanswered = Config {
        limbs_instances: 0,
        ..on_its_own_hardware()
    };
    let (harness, _mocks) = Harness::start_with(unanswered, openarm_initializer::setup).await?;
    let failure = shutdown_once_setup_returns(harness)
        .await
        .expect_err("a robot nothing answers for fails to start")
        .to_string();
    assert!(
        failure.contains("nothing answers for this robot's readiness")
            && failure.contains("bind `limbs`")
            && failure.contains("or `simulation`"),
        "{failure}"
    );
    Ok(())
}
