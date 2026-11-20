//! The Cartesian arm move end to end over the panel's WebSocket: a pose whose
//! orientation is no unit quaternion does not fire and the status says what to
//! send; firing a pose must land a `move_arm` goal on the limb_motion mock
//! whose target is the pose in the robot frame with no timestamp, carrying the
//! operator's settle tolerances and timeout, the
//! backbone's verdict must come back as the status as it stands, a pose the
//! panel's own exact IK cannot solve must still reach the backbone, whose
//! failure surfaces as the status, and a refusal of the backbone surfaces as
//! the status with its kind.
//!
//! One booting test per binary: `limits::init_limits` is once-per-process.

mod helpers;

use std::time::{Duration, SystemTime};

use control_core::geometry::Pose;
use peppygen::consumed_actions::limb_motion::move_arm::ResultResponseDataFinalPose;
use peppygen::fixtures::harness::Config;
use peppygen::mock::deps::limb_motion::{move_arm, move_arm_joints};

const PANEL_PORT: u16 = 18637;

/// Inside the v2 left arm's URDF ranges with the elbow bent, so the FK pose it
/// produces is one the panel's own IK can re-solve for its preview.
const SETTLED_JOINTS: [f64; 7] = [0.1, 0.1, -0.2, 0.9, 0.1, -0.1, 0.0];

/// The settle fields the operator types for the pose move.
const SETTLE_POSITION_TOLERANCE_M: f64 = 0.012;
const SETTLE_ORIENTATION_TOLERANCE_RAD: f64 = 0.12;
const SETTLE_TIMEOUT_S: f64 = 1.5;

/// The panel's command that fires a pose move of the left arm to `position`
/// and `orientation` [x, y, z, w], the fastest, with the settle fields above.
fn left_pose_command(position: [f64; 3], orientation: [f64; 4]) -> String {
    serde_json::json!({
        "cmd": "fire_arm_pose",
        "arm": "left_arm",
        "position": position,
        "orientation": orientation,
        "duration_s": 0.0,
        "settle_position_tolerance_m": SETTLE_POSITION_TOLERANCE_M,
        "settle_orientation_tolerance_rad": SETTLE_ORIENTATION_TOLERANCE_RAD,
        "settle_timeout_s": SETTLE_TIMEOUT_S,
    })
    .to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn firing_a_pose_carries_the_settle_fields_and_reports_the_backbone_verdict()
-> peppygen::Result<()> {
    let (harness, mut mocks) = helpers::start(Config {
        parameters: Some(helpers::test_parameters(PANEL_PORT)),
        ..Config::default()
    })
    .await?;

    let mut ws = helpers::WsClient::connect(PANEL_PORT).await;
    ws.next_snapshot(Duration::from_secs(10), "first snapshot")
        .await;

    // The panel's target stands at every joint 0, the straight arm, until a
    // measured pose or a move sets it. Settle the arm at a bent elbow first, so
    // the pose the panel reports is one it can re-solve. This is the panel's own
    // surface, not a back door.
    ws.send_text(
        &serde_json::json!({
            "cmd": "fire_arm",
            "arm": "left_arm",
            "joints": SETTLED_JOINTS,
            "duration_s": 0.0,
            "settle_tolerance_rad": 0.1,
            "settle_timeout_s": 2.0,
        })
        .to_string(),
    )
    .await
    .expect("send fire_arm");
    let settling = mocks
        .deps
        .limb_motion
        .move_arm_joints
        .next_goal(Duration::from_secs(10))
        .await?;
    settling
        .accept()
        .await?
        .complete(&move_arm_joints::ResultResponseData {
            final_joint_positions: SETTLED_JOINTS,
            ended_on_contact: false,
            action_time: 0.1,
        })
        .await?;
    // The joint move's verdict is its last status, so a later refusal stands.
    let settled = ws
        .snapshot_until(Duration::from_secs(15), "settled at a bent elbow", |s| {
            helpers::arm(s, "left_arm")["arm"]["in_flight"] == false
                && s["status"]
                    .as_str()
                    .is_some_and(|t| t.contains("move_arm_joints (left_arm): success"))
        })
        .await;

    // --- an orientation that is no unit quaternion does not fire ------------
    let numbers = |key: &str| -> Vec<f64> {
        helpers::arm(&settled, "left_arm")["arm"][key]
            .as_array()
            .unwrap_or_else(|| panic!("the panel publishes {key}"))
            .iter()
            .map(|v| v.as_f64().expect("pose components are numbers"))
            .collect()
    };
    let pose = numbers("pose");
    let position = [pose[0], pose[1], pose[2]];
    let orientation: [f64; 4] = numbers("orientation")
        .try_into()
        .expect("the panel publishes a quaternion");

    let doubled = [0.0, 0.0, 0.0, 2.0];
    ws.send_text(&left_pose_command(position, doubled))
        .await
        .expect("send fire_arm_pose");
    let refusal = Pose::parse(position, doubled).expect_err("a quaternion of length 2");
    let not_unit = format!("invalid pose, not firing: {refusal}");
    ws.snapshot_until(Duration::from_secs(10), "non-unit pose refused", |s| {
        s["status"].as_str().is_some_and(|t| t.contains(&not_unit))
    })
    .await;

    // --- fire_arm_pose -> move_arm goal carrying the plan and settle fields --
    ws.send_text(&left_pose_command(position, orientation))
        .await
        .expect("send fire_arm_pose");

    let pending = mocks
        .deps
        .limb_motion
        .move_arm
        .next_goal(Duration::from_secs(10))
        .await?;
    assert_eq!(pending.request.arm_name, "left_arm");
    let target = &pending.request.target;
    let robot_frame = robot_description::model(helpers::MODEL)
        .expect("a bundled model")
        .description()
        .root_link()
        .to_owned();
    assert_eq!(
        target.frame.as_deref(),
        Some(robot_frame.as_str()),
        "the target names the robot frame"
    );
    assert_eq!(target.timestamp, None, "the target names the latest time");
    assert_eq!(
        target.position, position,
        "the target is the composed position"
    );
    let sent = target.orientation.expect("the target names an orientation");
    for (part, composed) in sent.iter().zip(orientation) {
        assert!(
            (part - composed).abs() < 1e-12,
            "the target is the composed orientation: {sent:?} vs {orientation:?}"
        );
    }
    assert_eq!(
        (
            pending.request.settle_position_tolerance_m,
            pending.request.settle_orientation_tolerance_rad,
            pending.request.settle_timeout_s,
        ),
        (
            SETTLE_POSITION_TOLERANCE_M,
            SETTLE_ORIENTATION_TOLERANCE_RAD,
            SETTLE_TIMEOUT_S
        ),
        "the goal must carry the operator's settle fields as typed"
    );
    assert_eq!(
        pending.request.contact_gap_m, 0.0,
        "a panel move asks for no contact check"
    );
    let active = pending.accept().await?;

    ws.snapshot_until(Duration::from_secs(10), "left pose move in flight", |s| {
        helpers::arm(s, "left_arm")["arm"]["in_flight"] == true
    })
    .await;

    // The backbone judged the settle; the panel reports its verdict as it stands.
    active
        .complete(&move_arm::ResultResponseData {
            final_pose: ResultResponseDataFinalPose {
                frame: robot_frame,
                timestamp: SystemTime::now(),
                position,
                orientation,
            },
            ended_on_contact: false,
            action_time: 0.42,
        })
        .await?;

    ws.snapshot_until(Duration::from_secs(15), "left pose move done", |s| {
        helpers::arm(s, "left_arm")["arm"]["in_flight"] == false
            && s["status"]
                .as_str()
                .is_some_and(|t| t.contains("move_arm (left_arm): success"))
    })
    .await;

    // --- a failed settle surfaces the backbone's failure ---------------------
    ws.send_text(&left_pose_command(position, orientation))
        .await
        .expect("send fire_arm_pose");
    let pending = mocks
        .deps
        .limb_motion
        .move_arm
        .next_goal(Duration::from_secs(10))
        .await?;
    let active = pending.accept().await?;
    active
        .fail(
            move_arm::FailureKind::Timeout,
            "still 8.0 mm / 0.02 rad from the target, 2.0 s after the planned end",
        )
        .await?;
    let timed_out = "move_arm (left_arm) failed: timeout: still 8.0 mm / 0.02 rad from the target";
    ws.snapshot_until(Duration::from_secs(15), "failed settle surfaced", |s| {
        helpers::arm(s, "left_arm")["arm"]["in_flight"] == false
            && s["status"].as_str().is_some_and(|t| t.contains(timed_out))
    })
    .await;

    // --- no exact solution still fires; the backbone's failure surfaces -----
    // Two metres out has no exact IK solution, so the preview cannot be
    // solved; the goal must reach the backbone anyway, and its failure must come
    // back as the operator status.
    let far = [position[0] + 2.0, position[1], position[2]];
    ws.send_text(&left_pose_command(far, orientation))
        .await
        .expect("send far fire_arm_pose");

    let pending = mocks
        .deps
        .limb_motion
        .move_arm
        .next_goal(Duration::from_secs(10))
        .await?;
    let active = pending.accept().await?;
    active
        .fail(move_arm::FailureKind::Unreachable, "goal pose not planned")
        .await?;
    let unreachable = "move_arm (left_arm) failed: unreachable: goal pose not planned";
    ws.snapshot_until(Duration::from_secs(15), "backbone failure surfaced", |s| {
        helpers::arm(s, "left_arm")["arm"]["in_flight"] == false
            && s["status"]
                .as_str()
                .is_some_and(|t| t.contains(unreachable))
    })
    .await;

    // --- a refusal surfaces with its kind ------------------------------------
    ws.send_text(&left_pose_command(position, orientation))
        .await
        .expect("send fire_arm_pose");
    let pending = mocks
        .deps
        .limb_motion
        .move_arm
        .next_goal(Duration::from_secs(10))
        .await?;
    pending
        .refuse(
            move_arm::FailureKind::LeaderStreaming,
            "a leader is streaming arm commands",
        )
        .await?;
    let refused = "backbone refused the pose goal: leader_streaming: a leader is streaming";
    ws.snapshot_until(Duration::from_secs(15), "backbone refusal surfaced", |s| {
        helpers::arm(s, "left_arm")["arm"]["in_flight"] == false
            && s["status"].as_str().is_some_and(|t| t.contains(refused))
    })
    .await;

    harness.shutdown().await
}
