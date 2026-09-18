//! The robot's readiness gate: the robot is ready only while everything that
//! answers for it reports ready. On a real robot that is the driver of each
//! limb, bound on `limbs`; on a simulated one it is the simulation standing
//! it, bound on `simulation`, which answers for every limb at once.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use peppygen::consumed_services::limbs::is_ready as limb_is_ready;
use peppygen::consumed_services::simulation::is_ready as simulation_is_ready;
use peppygen::exposed_services::robot_ready::is_ready;
use peppygen::{NodeRunner, Result};
use peppylib::runtime::CancellationToken;

use crate::refused;

// How often the poller re-checks every source, and how long each poll waits
// before treating an unreachable source as not-ready.
const POLL_INTERVAL: Duration = Duration::from_millis(500);
const POLL_TIMEOUT: Duration = Duration::from_secs(2);

/// Starts serving the robot-level `is_ready`, for as long as the node runs.
/// A robot nothing answers for is refused: there is nothing to report.
pub fn serve(runner: Arc<NodeRunner>) -> Result<()> {
    if simulation_is_ready::bound_producer(&runner).is_none()
        && limb_is_ready::bound_producers(&runner).is_empty()
    {
        return Err(refused(
            "nothing answers for this robot's readiness: bind `limbs` to the drivers of its limbs, or `simulation` to the simulation that stands it",
        ));
    }
    let token = runner.cancellation_token().clone();
    tokio::spawn(run(runner, token));
    Ok(())
}

async fn run(runner: Arc<NodeRunner>, token: CancellationToken) {
    // The generated is_ready handler closure is synchronous, so it cannot poll
    // the sources itself. A background task caches their aggregate readiness
    // here and the handler just reads it.
    let ready = Arc::new(AtomicBool::new(false));
    tokio::spawn(poll_sources(runner.clone(), ready.clone(), token.clone()));

    tracing::info!("is_ready service started");
    loop {
        tokio::select! {
            _ = token.cancelled() => {
                tracing::info!("is_ready service shutting down");
                break;
            }
            result = is_ready::handle_next_request(&runner, |_req| {
                Ok(is_ready::Response::new(ready.load(Ordering::SeqCst)))
            }) => {
                if let Err(e) = result {
                    tracing::warn!("is_ready handler error: {e}");
                }
            }
        }
    }
}

async fn poll_sources(runner: Arc<NodeRunner>, ready: Arc<AtomicBool>, token: CancellationToken) {
    // Every pass re-polls every source, so a source that dies flips the
    // robot back to not-ready; the loop sleeps POLL_INTERVAL after each
    // pass.
    loop {
        ready.store(every_source_ready(&runner).await, Ordering::SeqCst);
        tokio::select! {
            _ = token.cancelled() => break,
            _ = tokio::time::sleep(POLL_INTERVAL) => {}
        }
    }
}

/// Whether everything answering for this robot reports ready. Polled
/// concurrently, so one unreachable source costs a pass one POLL_TIMEOUT
/// in all.
async fn every_source_ready(runner: &NodeRunner) -> bool {
    let (simulation, limbs) = futures::join!(simulation_ready(runner), every_limb_ready(runner));
    simulation && limbs
}

/// Whether the simulation standing this robot reports it ready: standing,
/// with every limb pair held. True for a robot that stands in none.
async fn simulation_ready(runner: &NodeRunner) -> bool {
    match simulation_is_ready::bound_producer(runner) {
        None => true,
        Some(simulation) => matches!(
            simulation_is_ready::poll(runner, simulation, POLL_TIMEOUT).await,
            Ok(response) if response.data.ready
        ),
    }
}

/// Whether every driver of this robot's limbs reports ready. True for a
/// robot with no drivers of its own.
async fn every_limb_ready(runner: &NodeRunner) -> bool {
    let polls = limb_is_ready::bound_producers(runner)
        .iter()
        .map(|limb| async move {
            matches!(
                limb_is_ready::poll(runner, limb, POLL_TIMEOUT).await,
                Ok(response) if response.data.ready
            )
        });
    futures::future::join_all(polls)
        .await
        .into_iter()
        .all(|ready| ready)
}
