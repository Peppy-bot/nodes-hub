//! Operator-driven consumed actions and services, end to end over the panel's
//! WebSocket (the node's real command surface): fire_arm must land a
//! move_arm_joints goal on the limb_motion mock; the record button must drive the
//! recorder mock's record_episode goal through feedback, Stop (= cancel with
//! save), and result; Finish must call the recorder's finish_session service.
//!
//! One booting test per binary: `ui::init_limits` is once-per-process.

mod helpers;

use std::time::Duration;

use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::limb_motion::move_arm_joints;
use peppygen::mock::deps::recorder::{finish_session, record_episode};

const PANEL_PORT: u16 = 18634;

/// Well inside the v2 URDF ranges (j2's range is only ±~0.17 rad, and j4 must
/// sit above the elbow's singularity floor), so the panel's clamp passes them
/// through verbatim.
const TARGET_JOINTS: [f64; 7] = [0.1, 0.1, -0.2, 0.9, 0.1, -0.1, 0.0];

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn panel_commands_drive_backbone_and_recorder() -> peppygen::Result<()> {
    helpers::set_panel_env(PANEL_PORT);
    let (harness, mut mocks) = Harness::start_with(
        Config {
            parameters: Some(helpers::test_parameters()),
            recorder_instances: 1,
            ..Config::default()
        },
        openarm_commander::setup,
    )
    .await?;

    let mut ws = helpers::WsClient::connect(PANEL_PORT).await;
    let first = ws
        .next_snapshot(Duration::from_secs(10), "first snapshot")
        .await;
    assert!(
        first["recorder"].is_object(),
        "a bound recorder must show its panel"
    );
    assert_eq!(first["recorder"]["recording"], false);
    assert_eq!(first["recorder"]["finishing"], false);

    // --- fire_arm -> move_arm_joints goal on the limb_motion producer --------
    let command = serde_json::json!({
        "cmd": "fire_arm",
        "side": "left",
        "joints": TARGET_JOINTS,
        "duration_s": 0.0,
    });
    ws.send_text(&command.to_string())
        .await
        .expect("send fire_arm");

    let pending = mocks
        .deps
        .limb_motion
        .move_arm_joints
        .next_goal(Duration::from_secs(10))
        .await?;
    assert_eq!(
        pending.request.arm_name, "left_arm",
        "left arm goes out under its wire name"
    );
    assert!(
        helpers::approx_eq(&pending.request.joint_positions, &TARGET_JOINTS),
        "the goal must carry the panel's joints: {:?}",
        pending.request.joint_positions
    );
    assert_eq!(pending.request.duration_s, 0.0, "0 = fastest safe");
    let goal_joints = pending.request.joint_positions;
    // The node's goal-accept timeout is 2 s, so admit before observing state.
    let active = pending.accept().await?;

    ws.snapshot_until(Duration::from_secs(10), "left move in flight", |s| {
        s["left_arm"]["in_flight"] == true
    })
    .await;

    // Finishing exactly on target makes the node's reached-check pass.
    active
        .complete(&move_arm_joints::ResultResponseData {
            success: true,
            message: String::new(),
            final_joint_positions: goal_joints,
            action_time: 0.42,
        })
        .await?;
    ws.snapshot_until(Duration::from_secs(15), "left move done", |s| {
        s["left_arm"]["in_flight"] == false
            && s["status"]
                .as_str()
                .is_some_and(|t| t.contains("move_arm_joints (left): success"))
    })
    .await;

    // --- record button -> record_episode goal on the recorder ---------------
    ws.send_text(r#"{"cmd":"start_recording","task":"pick the red cube"}"#)
        .await
        .expect("send start_recording");

    let recorder = &mut mocks.deps.recorder[0];
    let pending = recorder
        .record_episode
        .next_goal(Duration::from_secs(10))
        .await?;
    assert_eq!(pending.request.task, "pick the red cube");
    let active = pending.accept().await?;

    active
        .publish_feedback(&record_episode::FeedbackMessage {
            frames_recorded: 7,
            disk_free_bytes: 50_000_000_000,
        })
        .await?;
    ws.snapshot_until(Duration::from_secs(10), "recording with frames", |s| {
        s["recorder"]["recording"] == true && s["recorder"]["frames"] == 7
    })
    .await;

    // Stop = cancel-with-save: the node must request cancellation of the
    // in-flight goal, and the cancelled result reports the saved episode.
    ws.send_text(r#"{"cmd":"stop_recording"}"#)
        .await
        .expect("send stop_recording");
    tokio::time::timeout(Duration::from_secs(10), active.cancel_signal())
        .await
        .expect("Stop must cancel the record goal");
    active
        .complete_cancelled(&record_episode::ResultResponseData {
            episode_index: 3,
            frames_recorded: 7,
            discarded: false,
            error: None,
        })
        .await?;
    ws.snapshot_until(Duration::from_secs(15), "episode saved", |s| {
        s["recorder"]["recording"] == false
            && s["status"]
                .as_str()
                .is_some_and(|t| t.contains("episode 3 saved, 7 frames"))
    })
    .await;

    // --- finish button -> finish_session service on the recorder ------------
    ws.send_text(r#"{"cmd":"finish_session"}"#)
        .await
        .expect("send finish_session");
    let responder = mocks.deps.recorder[0]
        .finish_session
        .next_request(Duration::from_secs(10))
        .await?;
    responder
        .respond(finish_session::ResponseData {
            session: "session_007".to_string(),
            error: None,
        })
        .await?;
    ws.snapshot_until(Duration::from_secs(15), "session finished", |s| {
        s["recorder"]["finishing"] == false
            && s["status"]
                .as_str()
                .is_some_and(|t| t.contains("session session_007 finished"))
    })
    .await;

    harness.shutdown().await
}
