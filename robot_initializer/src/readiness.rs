//! The robot's readiness gate: the robot is ready only while everything that
//! answers for it reports ready. On a real robot that is whatever drivers
//! answer for its limbs, bound on `limbs`: one per limb on an OpenArm, the
//! single `so101_follower` on an SO-101. Each poll first tells a driver the
//! robot's model (`set_model`), which the driver configures for the first time
//! it is told, then asks it whether it is ready; a driver whose polls fail is
//! logged once for each run of one reply. On a
//! simulated one it is the simulation standing it, bound on `simulation`,
//! which answers for every limb at once.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use control_core::serving;
use control_core::time::sleep_unless_cancelled;
use peppygen::consumed_services::limbs::is_ready as limb_is_ready;
use peppygen::consumed_services::limbs::set_model as limb_set_model;
use peppygen::consumed_services::simulation::is_ready as simulation_is_ready;
use peppygen::exposed_services::robot_ready::is_ready;
use peppygen::{NodeRunner, Result};
use peppylib::messaging::ProducerRef;
use peppylib::runtime::CancellationToken;

use crate::refused;

// How often the poller re-checks every source, and how long each poll waits
// before treating an unreachable source as not-ready.
const POLL_INTERVAL: Duration = Duration::from_millis(500);
const POLL_TIMEOUT: Duration = Duration::from_secs(2);

/// Starts serving the robot-level `is_ready` of a robot of `model`, for as long
/// as the node runs. A robot nothing answers for is refused: there is nothing
/// to report.
pub fn serve(runner: Arc<NodeRunner>, model: &'static str) -> Result<()> {
    if simulation_is_ready::bound_producer(&runner).is_none()
        && limb_is_ready::bound_producers(&runner).is_empty()
    {
        return Err(refused(
            "nothing answers for this robot's readiness: bind `limbs` to the drivers of its limbs, or `simulation` to the simulation that stands it",
        ));
    }
    let token = runner.cancellation_token().clone();
    tokio::spawn(run(runner, model, token));
    Ok(())
}

async fn run(runner: Arc<NodeRunner>, model: &'static str, token: CancellationToken) {
    // The generated is_ready handler closure is synchronous, so it cannot poll
    // the sources itself. A background task caches their aggregate readiness
    // here and the handler just reads it.
    let ready = Arc::new(AtomicBool::new(false));
    tokio::spawn(poll_sources(
        runner.clone(),
        model,
        ready.clone(),
        token.clone(),
    ));

    tracing::info!("is_ready service started");
    serving::serve("is_ready", token.cancelled(), || {
        is_ready::handle_requests(&runner, |_req| {
            Ok(is_ready::Response::new(ready.load(Ordering::SeqCst)))
        })
    })
    .await;
}

async fn poll_sources(
    runner: Arc<NodeRunner>,
    model: &'static str,
    ready: Arc<AtomicBool>,
    token: CancellationToken,
) {
    // Every pass re-polls every source, so a source that dies flips the
    // robot back to not-ready; the loop sleeps POLL_INTERVAL after each
    // pass.
    let mut failures = LimbFailures::default();
    loop {
        let every_ready = every_source_ready(&runner, model, &mut failures).await;
        ready.store(every_ready, Ordering::SeqCst);
        if !sleep_unless_cancelled(POLL_INTERVAL, token.cancelled()).await {
            break;
        }
    }
}

/// Whether everything answering for this robot reports ready. Polled
/// concurrently, so one unreachable source costs a pass one POLL_TIMEOUT
/// in all.
async fn every_source_ready(runner: &NodeRunner, model: &str, failures: &mut LimbFailures) -> bool {
    let (simulation, limbs) = futures::join!(
        simulation_ready(runner),
        every_limb_ready(runner, model, failures)
    );
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

/// Whether every driver of this robot's limbs reports ready as a part of a
/// robot of `model`: each is told `model`, then asked. True for a robot with
/// no drivers of its own. Each reply goes to `failures`.
async fn every_limb_ready(runner: &NodeRunner, model: &str, failures: &mut LimbFailures) -> bool {
    let limbs = limb_is_ready::bound_producers(runner);
    let polls = limbs
        .iter()
        .map(|limb| async move { (limb, limb_ready(runner, limb, model).await) });
    let replies = futures::future::join_all(polls).await;
    for (limb, reply) in &replies {
        failures.note(limb, reply);
    }
    replies.iter().all(|(_, reply)| matches!(reply, Ok(true)))
}

/// The reply of one driver of a limb told `model`, then asked whether it is
/// ready: the failure of either call, else whether it is ready.
async fn limb_ready(runner: &NodeRunner, limb: &ProducerRef, model: &str) -> Result<bool> {
    let told = limb_set_model::Request {
        model: model.to_owned(),
    };
    limb_set_model::poll(runner, limb, POLL_TIMEOUT, told).await?;
    let asked = limb_is_ready::poll(runner, limb, POLL_TIMEOUT).await?;
    Ok(asked.data.ready)
}

/// The failed reply each driver of the robot's limbs gave last, once logged: a
/// driver whose polls keep failing with one reply is logged once, and again when
/// its reply changes.
#[derive(Default)]
struct LimbFailures(HashMap<ProducerRef, String>);

impl LimbFailures {
    /// Notes the reply of one poll of `limb`: a failure other than the one the
    /// limb gave last is logged, and an answer ends the run.
    fn note(&mut self, limb: &ProducerRef, reply: &Result<bool>) {
        let Err(failure) = reply else {
            self.0.remove(limb);
            return;
        };
        let failure = failure.to_string();
        if self.0.get(limb) == Some(&failure) {
            return;
        }
        tracing::warn!(
            "the driver {} on {} of a limb of this robot fails its readiness poll, \
             suppressing repeats until its reply changes: {failure}",
            limb.instance_id,
            limb.core_node
        );
        self.0.insert(limb.clone(), failure);
    }
}
