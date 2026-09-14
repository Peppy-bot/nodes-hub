//! Consumed-topic flow with one producer bound per zero_or_more contract slot:
//! the collision_status mock's stream must surface as the panel's
//! proximity readout, the alerts mock's alert must render in the alerts list,
//! and both contract slots must report as bound.
//!
//! motor_health uses the same producer identity as the observed left arm.
//! An unrelated producer's reports are dropped at the parse boundary.
//!
//! One booting test per binary: `ui::init_limits` is once-per-process.

mod helpers;

use std::time::{Duration, SystemTime};

use peppygen::consumed_topics::alerts::alerts;
use peppygen::consumed_topics::collision_status::collision_status;
use peppygen::consumed_topics::motor_health::motor_health;
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::observed::observed_left_arm;

const PANEL_PORT: u16 = 18633;

fn proximity_msg() -> collision_status::Message {
    collision_status::Message {
        timestamp: SystemTime::now(),
        distance: 0.0123,
        link_a: "left_link4".to_string(),
        link_b: "right_link4".to_string(),
        throttled: true,
        stopped: false,
    }
}

fn alert_msg() -> alerts::Message {
    alerts::Message {
        timestamp: SystemTime::now(),
        source: "left arm j2".to_string(),
        kind: "motor_overload".to_string(),
        severity: 2,
        message: "holding 93% of rated torque".to_string(),
    }
}

/// A well-formed 7-motor arm report, attributed through its observed slot.
fn arm_health_msg() -> motor_health::Message {
    motor_health::Message {
        timestamp: SystemTime::now(),
        level: vec![0; 7],
        effort_fraction_rated: vec![0.4; 7],
        effort_fraction_rated_sustained: vec![0.3; 7],
        effort_fraction_peak: vec![0.2; 7],
        driver_temp_c: vec![41.0; 7],
        winding_temp_c: vec![37.0; 7],
    }
}

/// Re-publishes `publish` and takes one panel snapshot per pass until `done`
/// holds, bounded by 10s and named by `what`: the panel re-renders on its own
/// ~100 ms cadence, so each pass observes the latest publish without sleeping
/// for it. Panics with the last snapshot seen on expiry.
async fn republish_until<P, PFut>(
    ws: &mut helpers::WsClient,
    what: &str,
    mut publish: P,
    mut done: impl FnMut(&serde_json::Value) -> bool,
) -> peppygen::Result<serde_json::Value>
where
    P: FnMut() -> PFut,
    PFut: std::future::Future<Output = peppygen::Result<()>>,
{
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        publish().await?;
        let snapshot = ws
            .next_snapshot(Duration::from_secs(5), "snapshot pass")
            .await;
        if done(&snapshot) {
            return Ok(snapshot);
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what}; last snapshot: {snapshot}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consumed_topics_surface_on_the_panel() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start_with(
        Config {
            parameters: Some(helpers::test_parameters(PANEL_PORT)),
            motor_health_instance_ids: vec![
                observed_left_arm::MOCK_INSTANCE_ID.to_string(),
                "alpha_right_arm_inst".to_string(),
            ],
            alerts_instances: 1,
            ..Config::default()
        },
        openarm_web_commander::setup,
    )
    .await?;

    // Slot binding is resolved once at owner start, so the very first
    // snapshot must already report both contract slots as wired.
    let mut ws = helpers::WsClient::connect(PANEL_PORT).await;
    let first = ws
        .next_snapshot(Duration::from_secs(10), "first snapshot")
        .await;
    assert_eq!(first["health"]["bound"], true);
    assert_eq!(first["alerts_bound"], true);

    // collision_status -> proximity readout. The readout goes stale 500 ms
    // after the last report, so re-publish each snapshot pass (~100 ms), as
    // the backbone's ~20 Hz stream would.
    let snapshot = republish_until(
        &mut ws,
        "proximity never rendered",
        || {
            let msg = proximity_msg();
            let publisher = &mocks.deps.collision_status.collision_status;
            async move { publisher.publish(&msg).await }
        },
        |s| !s["proximity"].is_null(),
    )
    .await?;
    let proximity = &snapshot["proximity"];
    assert_eq!(proximity["distance"].as_f64(), Some(0.0123));
    assert_eq!(proximity["link_a"], "left_link4");
    assert_eq!(proximity["link_b"], "right_link4");
    assert_eq!(proximity["throttled"], true);
    assert_eq!(proximity["stopped"], false);

    // alerts -> the alerts list, attributed and severity-tagged. Re-publish
    // with a fresh timestamp per pass so the entry cannot age out mid-poll.
    let snapshot = republish_until(
        &mut ws,
        "the alert never rendered",
        || {
            let msg = alert_msg();
            let publisher = &mocks.deps.alerts[0].alerts;
            async move { publisher.publish(&msg).await }
        },
        |s| s["alerts"].as_array().is_some_and(|a| !a.is_empty()),
    )
    .await?;
    let alert = &snapshot["alerts"][0];
    assert_eq!(alert["source"], "left arm j2");
    assert_eq!(alert["severity"], 2);
    assert_eq!(alert["message"], "holding 93% of rated torque");

    // The observed left arm's health renders one row per motor.
    let live = republish_until(
        &mut ws,
        "left health never went live",
        || {
            let msg = arm_health_msg();
            let publisher = &mocks.deps.motor_health[0].motor_health;
            async move { publisher.publish(&msg).await }
        },
        |s| s["health"]["left"]["status"] == "live",
    )
    .await?;
    // A live side renders its arm rows plus the (silent) gripper component's
    // placeholder row: 7 + 1.
    assert_eq!(
        live["health"]["left"]["motors"]
            .as_array()
            .expect("left motor rows")
            .len(),
        8,
        "seven arm motor rows plus the gripper placeholder"
    );

    // A producer outside the observed limbs must leave the right side empty.
    for _ in 0..10 {
        mocks.deps.motor_health[1]
            .motor_health
            .publish(&arm_health_msg())
            .await?;
        let snapshot = ws
            .next_snapshot(Duration::from_secs(5), "snapshot pass")
            .await;
        let status = snapshot["health"]["right"]["status"]
            .as_str()
            .expect("right status");
        assert!(
            status == "pending" || status == "not_reporting",
            "right health must stay empty for an unobserved producer, got {status}"
        );
        assert!(
            snapshot["health"]["right"]["motors"]
                .as_array()
                .is_some_and(Vec::is_empty)
        );
    }

    harness.shutdown().await
}
