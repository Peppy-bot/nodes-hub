//! The Cartesian arm move end to end over the panel's WebSocket: firing a pose
//! must land a `move_arm` goal on the limb_motion mock carrying the panel's
//! plan tolerances, and the arrival re-check must judge the returned pose
//! against the separate, wider result bar.
//!
//! One booting test per binary: `ui::init_limits` is once-per-process.

mod helpers;

use std::time::Duration;

use openarm_commander::{PLAN_ANGLE_TOL_RAD, PLAN_POS_TOL_M, REACHED_ORIENTATION_TOL_RAD};
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::limb_motion::{move_arm, move_arm_joints};
use srs_model::nalgebra::{Quaternion, UnitQuaternion};

const PANEL_PORT: u16 = 18637;

/// Well inside the v2 URDF ranges and clear of the elbow singularity floor, so
/// the FK pose it produces is one the panel's own IK can re-solve. The zero
/// configuration is not: its pose is on the singularity and the panel refuses
/// it before firing.
const SETTLED_JOINTS: [f64; 7] = [0.1, 0.1, -0.2, 0.9, 0.1, -0.1, 0.0];

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn firing_a_pose_carries_the_plan_tolerances_and_judges_arrival_separately()
-> peppygen::Result<()> {
    let (harness, mut mocks) = Harness::start_with(
        Config {
            parameters: Some(helpers::test_parameters(PANEL_PORT)),
            ..Config::default()
        },
        openarm_commander::setup,
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

    // --- fire_arm_pose -> move_arm goal carrying the plan tolerances --------
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
    let (goal_position, goal_orientation) = (pending.request.position, pending.request.orientation);
    let active = pending.accept().await?;

    ws.snapshot_until(Duration::from_secs(10), "left pose move in flight", |s| {
        s["left_arm"]["in_flight"] == true
    })
    .await;

    // Land past the plan slack but inside the arrival bar: an ordinary move that
    // used its planning budget. Half the arrival bar keeps it clear of both
    // edges, and it must read as success, which is the whole reason the two bars
    // are separate numbers.
    let landed = UnitQuaternion::from_euler_angles(0.0, 0.0, REACHED_ORIENTATION_TOL_RAD / 2.0)
        * UnitQuaternion::new_normalize(Quaternion::new(
            goal_orientation[3],
            goal_orientation[0],
            goal_orientation[1],
            goal_orientation[2],
        ));
    active
        .complete(&move_arm::ResultResponseData {
            success: true,
            message: String::new(),
            final_position: goal_position,
            final_orientation: [landed.i, landed.j, landed.k, landed.w],
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

    harness.shutdown().await
}
