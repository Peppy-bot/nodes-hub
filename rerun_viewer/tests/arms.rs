//! The arms over the wire: limb_state's snapshots become grasp points, trails and plots,
//! with the joints named by the planner when it is linked.

mod common;

use common::{
    TIMEOUT, bound, drew_at, every_optional_link_vacant, publish_once_subscribed, right_arm, start,
    two_arms,
};
use rerun_viewer::config::default_parameters;
use rerun_viewer::draw::CUBES_NOTE_PATH;
use rerun_viewer::testing::Drawn;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_note_on_the_cubes_is_drawn_at_the_start() -> Result<(), String> {
    let (harness, _mocks, canvas) = start(every_optional_link_vacant()).await?;
    canvas
        .wait_until(TIMEOUT, "the note", |drawn| drew_at(drawn, CUBES_NOTE_PATH))
        .await?;
    assert!(matches!(&canvas.at(CUBES_NOTE_PATH)[0], Drawn::Note { .. }));
    harness.shutdown().await.map_err(|e| e.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_snapshot_draws_the_arms_with_every_optional_link_vacant() -> Result<(), String> {
    let (harness, mocks, canvas) = start(every_optional_link_vacant()).await?;
    publish_once_subscribed(
        mocks
            .deps
            .limb_state
            .limb_states
            .wait_for_subscriber(TIMEOUT),
        mocks.deps.limb_state.limb_states.publish(&two_arms()),
    )
    .await?;
    canvas
        .wait_until(TIMEOUT, "the gripper's opening", |drawn| {
            drew_at(drawn, "plots/grippers/right_gripper")
        })
        .await?;
    let Drawn::Frame { pose, .. } = &canvas.at("world/arms/right_arm/grasp")[0] else {
        panic!("the grasp point is a frame")
    };
    assert_eq!(pose.origin(), [0.3, -0.2, 0.4]);
    assert!(matches!(
        &canvas.at("plots/joints/left_arm/joint_1")[0],
        Drawn::Scalar { value, .. } if *value == 0.2
    ));
    assert!(matches!(
        &canvas.at("plots/grippers/right_gripper")[0],
        Drawn::Scalar { value, .. } if *value == 0.5
    ));
    harness.shutdown().await.map_err(|e| e.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_joints_take_the_planners_names() -> Result<(), String> {
    let (harness, mocks, canvas) = start(bound(&["motion_planning"], default_parameters())).await?;
    let planner = mocks
        .deps
        .motion_planning
        .as_ref()
        .expect("the planner is linked");
    planner
        .describe_arms
        .next_request(TIMEOUT)
        .await
        .map_err(|e| e.to_string())?
        .respond(right_arm())
        .await
        .map_err(|e| e.to_string())?;
    // The scene is asked after the arms: its first ask shows the arms are known.
    planner
        .get_planning_scene
        .next_request(TIMEOUT)
        .await
        .map_err(|e| e.to_string())?
        .respond(common::scene(vec![], vec![]))
        .await
        .map_err(|e| e.to_string())?;
    canvas
        .wait_until(TIMEOUT, "the scene", |drawn| {
            drew_at(drawn, "scene/allowed_pairs")
        })
        .await?;
    publish_once_subscribed(
        mocks
            .deps
            .limb_state
            .limb_states
            .wait_for_subscriber(TIMEOUT),
        mocks.deps.limb_state.limb_states.publish(&two_arms()),
    )
    .await?;
    canvas
        .wait_until(TIMEOUT, "the named joint", |drawn| {
            drew_at(drawn, "plots/joints/right_arm/openarm_right_joint1")
        })
        .await?;
    harness.shutdown().await.map_err(|e| e.to_string())
}
