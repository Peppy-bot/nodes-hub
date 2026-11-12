//! A panel whose robot's initializer names a model of other arms ends with a
//! fault that names the served model and the models of the panel's shape.
//!
//! One booting test per binary: `limits::init_limits` is once-per-process.

mod helpers;

use std::time::Duration;

use peppygen::fixtures::harness::Config;

const PANEL_PORT: u16 = 18638;

/// How long the test waits for the fault to shut the node down.
const FAULT_BUDGET: Duration = Duration::from_secs(30);

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_model_the_panel_does_not_drive_ends_the_node_naming_the_driven_ones()
-> peppygen::Result<()> {
    let (harness, mocks, fault) = helpers::boot(Config {
        parameters: Some(helpers::test_parameters(PANEL_PORT)),
        ..Config::default()
    })
    .await?;
    helpers::answer_identity(&mocks, "so101").await?;
    tokio::time::timeout(
        FAULT_BUDGET,
        harness.node_runner().cancellation_token().cancelled(),
    )
    .await
    .expect("the fault shuts the node down");
    let reason = fault.take().expect("the node ends with a fault");
    assert_eq!(
        reason.to_string(),
        "the model the robot's initializer serves"
    );
    let undriven = reason
        .source()
        .expect("the reason names the model")
        .to_string();
    assert!(
        undriven.starts_with("model so101 describes")
            && undriven.ends_with("the models of that shape are \"openarm_v1\" and \"openarm_v2\""),
        "{undriven}"
    );
    harness.shutdown().await
}
