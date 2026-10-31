//! The workflow director: it checks workflows (`workflow/v1`) against the
//! contracts of the fleet listing and runs them (`run/v1`) on the robots of
//! the stack under robot leases.
//!
//! The director follows the fleet with one `fleet.changes` goal and holds
//! its leases with one `fleet.lease` goal, for the life of the node. A run
//! binds its roles to robots of a fresh fleet listing, takes their leases,
//! and calls the members its workflow names through the codec API, so a
//! workflow can name contracts this node was not built with. The engine
//! runs one lane per role on the run clock. The journal of each run keeps
//! its inputs, its events and its end, and gives its replay. A trial
//! (`trial/v1`) runs a bundle on physical robots, a person preparing and
//! judging each run, and keeps a sign-off record when its runs meet the
//! thresholds of the bundle; an unattended run on physical robots runs only
//! as such a record admits it.

#![forbid(unsafe_code)]

use std::sync::{Arc, Mutex, OnceLock};

use peppygen::{NodeRunner, Parameters, Result};
use tokio::sync::{broadcast, watch};

mod alerts;
mod checker;
mod clock;
mod config;
mod contract;
mod director;
mod engine;
mod fleet;
mod journal;
mod leases;
mod live;
mod replay;
mod serve;
mod sign_off;
#[cfg(test)]
mod testing;
mod trial;

use clock::ClockSource;
use config::Config;
use director::{Director, EVENTS_KEPT};
use fleet::Snapshot;
use journal::Journals;
use leases::Leases;
use sign_off::SignOffs;
use trial::Trials;

/// The director's own error: a fact about its parameters or its machine,
/// which is what a launcher reads back when the node refuses to start.
pub(crate) fn refused(message: impl Into<String>) -> peppygen::Error {
    peppygen::Error::Node(std::io::Error::other(message.into()).into())
}

/// The fault that ends the node after its setup: the end of its lease
/// goal, whose leases end with it, since a process runs one lease goal, or
/// the end of the simulation clock its runs read. The first raise shuts the
/// node down, and `main` exits with the fault as its error, which the
/// restart policy `on_failure` restarts.
#[derive(Clone, Debug, Default)]
pub struct Fault(Arc<OnceLock<String>>);

impl Fault {
    /// Records `message`, unless a fault came first, and shuts the node down.
    pub(crate) fn raise(&self, runner: &NodeRunner, message: &str) {
        tracing::error!("{message}");
        let _ = self.0.set(message.to_owned());
        runner.cancellation_token().cancel();
    }

    /// The end of the node: the error of its fault, if one was raised.
    pub fn end(&self) -> Result<()> {
        self.0
            .get()
            .map_or(Ok(()), |message| Err(refused(message.clone())))
    }
}

pub async fn setup(params: Parameters, runner: Arc<NodeRunner>, fault: Fault) -> Result<()> {
    let config = Config::new(
        &params.allowed_contracts,
        params.poll_s,
        params.cancel_timeout_s,
        &params.journal_root,
        params.journal_max_runs,
        params.journal_max_mib,
    )
    .map_err(refused)?;
    let journals = Journals::open(
        &config.journal_root,
        config.journal_max_runs,
        config.journal_max_bytes,
    )
    .map_err(|error| {
        refused(format!(
            "the journals under journal_root `{}` did not open: {error}",
            config.journal_root.display()
        ))
    })?;
    let sign_offs = SignOffs::open(&config.journal_root).map_err(|error| {
        refused(format!(
            "the sign-off records under journal_root `{}` did not read: {error}",
            config.journal_root.display()
        ))
    })?;
    let (trial_records, trials) = Trials::open(&config.journal_root).map_err(|error| {
        refused(format!(
            "the trial records under journal_root `{}` did not read: {error}",
            config.journal_root.display()
        ))
    })?;
    let clock = ClockSource::of(&runner, fault.clone()).await?;
    let leases = Leases::start(Arc::clone(&runner), fault).await?;
    let (snapshots_tx, snapshots) = watch::channel(Arc::new(Snapshot::default()));
    tokio::spawn(fleet::follow(Arc::clone(&runner), snapshots_tx));
    let (events, _) = broadcast::channel(EVENTS_KEPT);
    let director = Arc::new(Director {
        runner,
        config,
        leases,
        snapshots,
        journals,
        sign_offs,
        trial_records,
        trials: Mutex::new(Default::default()),
        runs: Mutex::new(Default::default()),
        teaching: Mutex::new(Default::default()),
        events,
        clock,
    });
    director.restore_trials(trials);
    serve::publish_run_events(&director).await?;
    director.restore().await.map_err(|error| {
        refused(format!(
            "the journals of earlier runs did not read: {error}"
        ))
    })?;
    serve::serve(Arc::clone(&director)).await
}
