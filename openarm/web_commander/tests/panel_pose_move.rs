//! The Cartesian arm move end to end over the panel's WebSocket: firing a pose
//! must land a `move_arm` goal on the limb_motion mock carrying the panel's
//! plan tolerances and the operator's settle tolerances and timeout, the
//! backbone's verdict must come back as the status as it stands, and a pose
//! the panel's own exact IK cannot solve must still reach the backbone, whose
//! refusal surfaces as the status.
//!
//! One booting test per binary: `ui::init_limits` is once-per-process.

mod helpers;

use std::time::Duration;

use openarm_web_commander::{PLAN_ANGLE_TOL_RAD, PLAN_POS_TOL_M};
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::limb_motion::{move_arm, move_arm_joints};
use srs_model::nalgebra::UnitQuaternion;

const PANEL_PORT: u16 = 18637;

/// Well inside the v2 URDF ranges and clear of the elbow singularity floor, so
/// the FK pose it produces is one the panel's own IK can re-solve for its
/// preview.
const SETTLED_JOINTS: [f64; 7] = [0.1, 0.1, -0.2, 0.9, 0.1, -0.1, 0.0];

/// The settle fields the operator types for the pose move, each covering the
/// panel's plan tolerance of its axis as the backbone requires.
const SETTLE_POSITION_TOLERANCE_M: f64 = 0.012;
const SETTLE_ORIENTATION_TOLERANCE_RAD: f64 = 0.12;
const SETTLE_TIMEOUT_S: f64 = 1.5;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn firing_a_pose_carries_the_plan_and_settle_fields_and_reports_the_backbone_verdict()
-> peppygen::Result<()> {
    let (harness, mut mocks) = Harness::start_with(
        Config {
            parameters: Some(helpers::test_parameters(PANEL_PORT)),
            ..Config::default()
        },
        openarm_web_commander::setup,
    )
    .await?;

    let mut ws = helpers::WsClient::connect(PANEL_PORT).await;
    ws.next_snapshot(Duration::from_secs(10), "first snapshot")
        .await;

    // Settle the arm off the singularity first, so the pose the panel reports is
    // one it can re-solve. This is the panel's own surface, not a back door.
    ws.send_text(
        &serde_json::json!({
            "cmd": "fire_arm",
            "side": "left",
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
            success: true,
            message: String::new(),
            final_joint_positions: SETTLED_JOINTS,
            action_time: 0.1,
        })
        .await?;
    let settled = ws
        .snapshot_until(
            Duration::from_secs(15),
            "settled off the singularity",
            |s| s["left_arm"]["in_flight"] == false,
        )
        .await;

    // --- fire_arm_pose -> move_arm goal carrying the plan and settle fields --
    let pose: Vec<f64> = settled["left_arm"]["pose"]
        .as_array()
        .expect("the panel publishes a pose")
        .iter()
        .map(|v| v.as_f64().expect("pose components are numbers"))
        .collect();
    let position = [pose[0], pose[1], pose[2]];
    let rotation = UnitQuaternion::from_euler_angles(pose[3], pose[4], pose[5]);

    ws.send_text(
        &serde_json::json!({
            "cmd": "fire_arm_pose",
            "side": "left",
            "position": position,
            "orientation": [rotation.i, rotation.j, rotation.k, rotation.w],
            "duration_s": 0.0,
            "settle_position_tolerance_m": SETTLE_POSITION_TOLERANCE_M,
            "settle_orientation_tolerance_rad": SETTLE_ORIENTATION_TOLERANCE_RAD,
            "settle_timeout_s": SETTLE_TIMEOUT_S,
        })
        .to_string(),
    )
    .await
    .expect("send fire_arm_pose");

    let pending = mocks
        .deps
        .limb_motion
        .move_arm
        .next_goal(Duration::from_secs(10))
        .await?;
    assert_eq!(pending.request.arm_name, "left_arm");
    assert_eq!(
        pending.request.plan_position_tolerance_m, PLAN_POS_TOL_M,
        "the panel must ask the backbone to plan to its own position slack"
    );
    assert_eq!(
        pending.request.plan_orientation_tolerance_rad, PLAN_ANGLE_TOL_RAD,
        "the panel must ask the backbone to plan to its own orientation slack"
    );
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
    let (goal_position, goal_orientation) = (pending.request.position, pending.request.orientation);
    let active = pending.accept().await?;

    ws.snapshot_until(Duration::from_secs(10), "left pose move in flight", |s| {
        s["left_arm"]["in_flight"] == true
    })
    .await;

    // The backbone judged the settle; the panel reports its verdict as it stands.
    active
        .complete(&move_arm::ResultResponseData {
            success: true,
            message: "cartesian move complete".to_string(),
            final_position: goal_position,
            final_orientation: goal_orientation,
            action_time: 0.42,
        })
        .await?;

    ws.snapshot_until(Duration::from_secs(15), "left pose move done", |s| {
        s["left_arm"]["in_flight"] == false
            && s["status"]
                .as_str()
                .is_some_and(|t| t.contains("move_arm (left): success"))
    })
    .await;

    // --- a failed settle surfaces the backbone's message --------------------
    ws.send_text(
        &serde_json::json!({
            "cmd": "fire_arm_pose",
            "side": "left",
            "position": position,
            "orientation": [rotation.i, rotation.j, rotation.k, rotation.w],
            "duration_s": 0.0,
            "settle_position_tolerance_m": SETTLE_POSITION_TOLERANCE_M,
            "settle_orientation_tolerance_rad": SETTLE_ORIENTATION_TOLERANCE_RAD,
            "settle_timeout_s": SETTLE_TIMEOUT_S,
        })
        .to_string(),
    )
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
        .complete(&move_arm::ResultResponseData {
            success: false,
            message: "still 8 mm / 0.02 rad from the target 2.0 s after the planned end"
                .to_string(),
            final_position: goal_position,
            final_orientation: goal_orientation,
            action_time: 2.9,
        })
        .await?;
    ws.snapshot_until(Duration::from_secs(15), "failed settle surfaced", |s| {
        s["left_arm"]["in_flight"] == false
            && s["status"].as_str().is_some_and(|t| {
                t.contains("move_arm (left) failed: still 8 mm / 0.02 rad from the target")
            })
    })
    .await;

    // --- no exact solution still fires; the backbone's refusal surfaces -----
    // Two metres out has no exact IK solution, so the preview cannot be
    // solved; the goal must reach the backbone anyway, and its decision, the
    // one naming the plan tolerance, must come back as the operator status.
    ws.send_text(
        &serde_json::json!({
            "cmd": "fire_arm_pose",
            "side": "left",
            "position": [position[0] + 2.0, position[1], position[2]],
            "orientation": [rotation.i, rotation.j, rotation.k, rotation.w],
            "duration_s": 0.0,
            "settle_position_tolerance_m": SETTLE_POSITION_TOLERANCE_M,
            "settle_orientation_tolerance_rad": SETTLE_ORIENTATION_TOLERANCE_RAD,
            "settle_timeout_s": SETTLE_TIMEOUT_S,
        })
        .to_string(),
    )
    .await
    .expect("send far fire_arm_pose");

    let pending = mocks
        .deps
        .limb_motion
        .move_arm
        .next_goal(Duration::from_secs(10))
        .await?;
    pending
        .reject(Some("goal pose unreachable within 5.0 mm / 5.0 deg"))
        .await?;
    ws.snapshot_until(Duration::from_secs(15), "backbone refusal surfaced", |s| {
        s["left_arm"]["in_flight"] == false
            && s["status"].as_str().is_some_and(|t| {
                t.contains("backbone rejected the pose goal") && t.contains("5.0 mm")
            })
    })
    .await;

    harness.shutdown().await
}
