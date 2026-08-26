//! The governor controls the panel streams must reach the wire in the fields
//! they name. `build_message` takes four f64s positionally, so a caller that
//! transposes two of them still compiles and still publishes: the swap only
//! shows up as a robot moving at another control's value. This drives the
//! panel with a distinct value per field and reads the emission back. The EE
//! speed cap and the gripper opening rate are adjacent f64 arguments, so a
//! transposition of those two is exactly what this pins.
//!
//! One booting test per binary: `ui::init_limits` is once-per-process.

mod helpers;

use std::time::Duration;

use peppygen::fixtures::harness::{Config, Harness};

const PANEL_PORT: u16 = 18637;

/// Distinct, and none a plausible neighbour of another: any transposition
/// changes a value rather than landing on the same number twice.
const D_STOP: f64 = 0.011;
const D_SAFE: f64 = 0.022;
const MAX_EE_VELOCITY_M_S: f64 = 0.333;
const MAX_GRIPPER_RATE_FRAC_S: f64 = 5.555;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_governor_control_reaches_the_field_it_names() -> peppygen::Result<()> {
    helpers::set_panel_env(PANEL_PORT);
    let (mut harness, _mocks) = Harness::start_with(
        Config {
            parameters: Some(helpers::test_parameters()),
            ..Config::default()
        },
        openarm_commander::setup,
    )
    .await?;

    let mut ws = helpers::WsClient::connect(PANEL_PORT).await;
    ws.next_snapshot(Duration::from_secs(10), "first snapshot")
        .await;

    let command = serde_json::json!({
        "cmd": "set_governor_params",
        "d_stop": D_STOP,
        "d_safe": D_SAFE,
        "max_ee_velocity_m_s": MAX_EE_VELOCITY_M_S,
        "max_gripper_rate_frac_s": MAX_GRIPPER_RATE_FRAC_S,
    });
    ws.send_text(&command.to_string())
        .await
        .expect("send set_governor_params");

    // The panel re-publishes every tick, so the first emissions still carry the
    // launch values; wait for the one that carries what was just set.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let msg = loop {
        assert!(
            tokio::time::Instant::now() < deadline,
            "no governor_control carrying the commanded values arrived"
        );
        let msg = tokio::time::timeout(
            Duration::from_secs(5),
            harness.emitted.governor_governor_control.next(),
        )
        .await
        .expect("governor_control emission")?
        .expect("fixture session open");
        if msg.d_stop == D_STOP {
            break msg;
        }
    };

    assert_eq!(msg.d_stop, D_STOP, "d_stop");
    assert_eq!(msg.d_safe, D_SAFE, "d_safe");
    assert_eq!(
        msg.max_ee_velocity_m_s, MAX_EE_VELOCITY_M_S,
        "max_ee_velocity_m_s"
    );
    assert_eq!(
        msg.max_gripper_rate_frac_s, MAX_GRIPPER_RATE_FRAC_S,
        "max_gripper_rate_frac_s"
    );

    harness.shutdown().await
}
