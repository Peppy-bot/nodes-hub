//! The node's setup over the generated harness, on a CAN interface the machine
//! lacks.

use std::time::Duration;

use control_core::fault::Fault;
use peppygen::Parameters;
use peppygen::fixtures::harness::{Config, Harness};

/// How long the test waits for the node's `setup` to return.
const SETUP_BUDGET: Duration = Duration::from_secs(30);

/// How often the test checks whether the node's `setup` has returned.
const SETUP_POLL: Duration = Duration::from_millis(50);

/// The interface the launch names, which no machine has.
const MISSING_INTERFACE: &str = "no_such_can";

/// A launch the node accepts in every parameter but its interface.
fn on_the_missing_interface() -> Parameters {
    Parameters {
        can_interface: MISSING_INTERFACE.to_string(),
        control_rate_hz: 100,
        gripper_id: 0,
        max_effort_nm: 3.0,
        speed_rad_s: 25.0,
        state_rate_hz: 100,
    }
}

/// `error` and each error it comes from, joined by ": ".
fn with_sources(error: &(dyn std::error::Error + 'static)) -> String {
    std::iter::successors(Some(error), |&error| error.source())
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
}

/// The interface does not depend on the robot's model, so the launch is refused
/// before the robot's initializer names one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_launch_on_a_missing_can_interface_fails_its_setup_naming_it() -> peppygen::Result<()> {
    std::fs::create_dir_all(instance_lock::LOCK_DIR).expect("the instance lock directory");
    let config = Config {
        parameters: Some(on_the_missing_interface()),
        ..Config::default()
    };
    let (harness, _mocks) = Harness::start_with(config, |params, runner| {
        openarm_gripper::setup(params, runner, Fault::default())
    })
    .await?;
    let deadline = tokio::time::Instant::now() + SETUP_BUDGET;
    while !harness.setup_finished() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "setup must return within {SETUP_BUDGET:?}"
        );
        tokio::time::sleep(SETUP_POLL).await;
    }
    let failure = harness
        .shutdown()
        .await
        .expect_err("a launch on a missing CAN interface fails to start");
    let failure = with_sources(&failure);
    assert!(
        failure.contains(&format!(
            "failed to open CAN interface '{MISSING_INTERFACE}'"
        )),
        "{failure}"
    );
    Ok(())
}
