//! Boot with every zero_or_more slot empty (no recorder, no motor_health, no
//! alerts producers), only the required backbone mock bound: the node must
//! come up cleanly, stream `governor_control` from its launch parameters, keep
//! every pairing setpoint stream silent while the deadmen are off, and render
//! the unbound slots as unbound on its panel.
//!
//! One booting test per binary: `ui::init_limits` is once-per-process.

mod helpers;

use std::time::Duration;

use peppygen::fixtures::harness::{Config, Harness};

const PANEL_PORT: u16 = 18631;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn boots_clean_with_empty_zero_or_more_slots() -> peppygen::Result<()> {
    helpers::set_panel_env(PANEL_PORT);
    let (mut harness, mut mocks) = Harness::start_with(
        Config {
            parameters: Some(helpers::test_parameters()),
            ..Config::default()
        },
        openarm_commander::setup,
    )
    .await?;

    // The governor stream has no deadman: it re-publishes the launch controls
    // at command_rate_hz from the first tick, so the very next message must
    // carry the parameters verbatim.
    let governor = tokio::time::timeout(
        Duration::from_secs(20),
        harness.emitted.governor_governor_control.next(),
    )
    .await
    .expect("governor_control must stream from boot")?
    .expect("governor_control subscription should be open");
    assert!(governor.collision_governor_enabled);
    assert!((governor.d_stop - 0.005).abs() < 1e-12);
    assert!((governor.d_safe - 0.02).abs() < 1e-12);
    assert!((governor.max_ee_velocity_m_s - 0.5).abs() < 1e-12);

    // With every deadman off the command frame holds None per side, so the
    // pairing setpoint streams must publish nothing. Bounded absence check:
    // ~20 governor ticks pass in this window while the arm stays silent.
    let quiet = tokio::time::timeout(
        Duration::from_millis(400),
        mocks.pairings.left_arm.joint_setpoints.next(),
    )
    .await;
    assert!(
        quiet.is_err(),
        "left_arm joint_setpoints must stay silent while disabled"
    );
    let quiet = tokio::time::timeout(
        Duration::from_millis(400),
        mocks.pairings.right_gripper.gripper_setpoints.next(),
    )
    .await;
    assert!(
        quiet.is_err(),
        "right_gripper gripper_setpoints must stay silent while disabled"
    );

    // The panel is the node's own surface for slot binding: an empty recorder
    // slot hides the recorder card, and the unwired health/alerts slots must
    // say "not wired" rather than implying a healthy robot.
    let mut ws = helpers::WsClient::connect(PANEL_PORT).await;
    let snapshot = ws
        .snapshot_until(Duration::from_secs(10), "unbound-slot snapshot", |s| {
            s.get("recorder").is_some()
        })
        .await;
    assert!(snapshot["recorder"].is_null(), "no recorder bound");
    assert_eq!(snapshot["health"]["bound"], false);
    assert_eq!(snapshot["health"]["left"]["status"], "not_wired");
    assert_eq!(snapshot["health"]["right"]["status"], "not_wired");
    assert_eq!(snapshot["alerts_bound"], false);
    assert_eq!(snapshot["left_enabled"], false);
    assert_eq!(snapshot["right_enabled"], false);
    assert!(snapshot["proximity"].is_null(), "no backbone report yet");

    harness.shutdown().await
}
