//! Fixtures shared by the backbone's integration tests.
//!
//! Each integration test is its own crate, so items unused by one suite are
//! expected; the allow keeps clippy's -D warnings gate meaningful for the rest.
#![allow(dead_code)]

pub mod fixtures;
pub mod params;

use std::time::Duration;

use control_core::fault::Fault;
use peppygen::fixtures::harness::{Config, Harness, Mocks};

/// The model of the tests' robot, as its initializer names it: v1 hardware.
pub const MODEL: &str = "openarm_v1";

/// How long a test waits for the node's `setup` to return.
const SETUP_BUDGET: Duration = Duration::from_secs(30);

/// How often a test checks whether the node's `setup` has returned.
const SETUP_POLL: Duration = Duration::from_millis(50);

/// Boots the node over the harness as `config` binds it, with the fault that
/// ends it once its setup has returned.
pub async fn start(config: Config) -> peppygen::Result<(Harness, Mocks, Fault)> {
    let fault = Fault::default();
    let raised = fault.clone();
    let (harness, mocks) = Harness::start_with(config, move |params, runner| {
        openarm_backbone::setup(params, runner, raised)
    })
    .await?;
    Ok((harness, mocks, fault))
}

/// Waits for the node's `setup` to return, failing the test after
/// `SETUP_BUDGET`.
pub async fn await_setup_return(harness: &Harness) {
    let deadline = tokio::time::Instant::now() + SETUP_BUDGET;
    while !harness.setup_finished() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "setup must return within {SETUP_BUDGET:?}"
        );
        tokio::time::sleep(SETUP_POLL).await;
    }
}

/// Tears the harness down once the node's `setup` has returned, so a setup
/// error reaches the caller. Teardown aborts a setup still running after the
/// shutdown grace and reports it as a clean stop.
pub async fn shutdown_once_setup_returns(harness: Harness) -> peppygen::Result<()> {
    await_setup_return(&harness).await;
    harness.shutdown().await
}
