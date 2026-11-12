//! Observed-source flow: the harness seeds one mock source per pairing
//! observer slot; publishing `joint_states` / `gripper_states` from those
//! sources must surface as the panel's live feedback, and once measurements
//! exist the operator can enable an arm's deadman, at which point the node
//! streams that arm's setpoints on the leader pairing slots (received here by
//! the pairing peer mocks).
//!
//! One booting test per binary: `limits::init_limits` is once-per-process.

mod helpers;

use std::time::{Duration, SystemTime};

use peppygen::fixtures::harness::Config;
use peppygen::paired_topics::observed_left_arm::joint_states;
use peppygen::paired_topics::observed_left_gripper::gripper_states;

const PANEL_PORT: u16 = 18632;

const MEASURED_JOINTS: [f64; 7] = [0.1, -0.2, 0.3, 0.9, -0.1, 0.2, 0.05];
const MEASURED_OPENING: f64 = 0.42;
const EFFORT_CEILING: f64 = 2.5;

fn joints_of(value: &serde_json::Value) -> Option<Vec<f64>> {
    value
        .as_array()?
        .iter()
        .map(serde_json::Value::as_f64)
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn observed_states_feed_the_panel_and_gate_streaming() -> peppygen::Result<()> {
    let (harness, mut mocks) = helpers::start(Config {
        parameters: Some(helpers::test_parameters(PANEL_PORT)),
        ..Config::default()
    })
    .await?;

    // The observed sources publish the followers' measured state. The first
    // publish waits for the node's observation subscription, so nothing races.
    mocks
        .observed
        .observed_left_arm
        .joint_states
        .publish(&joint_states::Message {
            timestamp: SystemTime::now(),
            positions: MEASURED_JOINTS.to_vec(),
            velocities: Vec::new(),
            efforts: Vec::new(),
        })
        .await?;
    mocks
        .observed
        .observed_left_gripper
        .gripper_states
        .publish(&gripper_states::Message {
            timestamp: SystemTime::now(),
            opening: MEASURED_OPENING,
            effort: 0.0,
            max_effort: EFFORT_CEILING,
        })
        .await?;

    // The panel must render both measurements as the left arm's feedback
    // while the never-fed right arm stays null.
    let mut ws = helpers::WsClient::connect(PANEL_PORT).await;
    let snapshot = ws
        .snapshot_until(Duration::from_secs(10), "left arm feedback", |s| {
            let left = helpers::arm(s, "left_arm");
            joints_of(&left["arm"]["feedback"])
                .is_some_and(|j| helpers::approx_eq(&j, &MEASURED_JOINTS))
                && left["gripper"]["feedback"].as_f64() == Some(MEASURED_OPENING)
        })
        .await;
    assert_eq!(
        helpers::arm(&snapshot, "left_arm")["gripper"]["effort_ceiling"].as_f64(),
        Some(EFFORT_CEILING),
        "the gripper's reported ceiling must reach the panel"
    );
    let right = helpers::arm(&snapshot, "right_arm");
    assert!(
        right["arm"]["feedback"].is_null(),
        "the unfed arm must stay without feedback"
    );
    assert!(right["gripper"]["feedback"].is_null());

    // Measurements exist, so the deadman may arm: the target was established
    // from the first measured pose and the gripper seeds from measured, and
    // the command stream starts publishing exactly those on the left slots.
    ws.send_text(r#"{"cmd":"set_enabled","arm":"left_arm","on":true}"#)
        .await
        .expect("send set_enabled");
    ws.snapshot_until(Duration::from_secs(10), "left arm enabled", |s| {
        helpers::arm(s, "left_arm")["enabled"] == true
    })
    .await;

    let setpoint = tokio::time::timeout(
        Duration::from_secs(10),
        mocks.pairings.left_arm.joint_setpoints.next(),
    )
    .await
    .expect("enabled left arm must stream joint_setpoints")?
    .expect("pairing subscription should be open");
    assert!(
        helpers::approx_eq(&setpoint.positions, &MEASURED_JOINTS),
        "the streamed setpoint must hold the established target: {:?}",
        setpoint.positions
    );
    assert!(setpoint.velocities.is_empty());
    assert!(setpoint.efforts.is_empty());

    let gripper_setpoint = tokio::time::timeout(
        Duration::from_secs(10),
        mocks.pairings.left_gripper.gripper_setpoints.next(),
    )
    .await
    .expect("enabled left gripper must stream gripper_setpoints")?
    .expect("pairing subscription should be open");
    assert!((gripper_setpoint.opening - MEASURED_OPENING).abs() < 1e-9);
    assert!(
        gripper_setpoint.max_effort.abs() < 1e-12,
        "no operator cap set: the wire encodes 0 (no preference)"
    );

    // The right arm never armed, so its slots stay silent even while the
    // left streams (bounded absence check, ~20 left-arm ticks long).
    let quiet = tokio::time::timeout(
        Duration::from_millis(400),
        mocks.pairings.right_arm.joint_setpoints.next(),
    )
    .await;
    assert!(
        quiet.is_err(),
        "right_arm joint_setpoints must stay silent while disabled"
    );

    harness.shutdown().await
}
