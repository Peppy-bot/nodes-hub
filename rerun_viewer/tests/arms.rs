//! The arms over the wire: limb_state's snapshots become grasp points, trails and plots,
//! with the joints named by the planner when it is linked.

mod common;

use common::{
    GRASP_PATH, ODOM_FRAME, ROBOT_FRAME, TIMEOUT, bound, drew_at, every_optional_link_vacant,
    hold_the_robot_frame, logged, publish_arms_until_drawn, publish_once_subscribed, right_arm,
    start, two_arms,
};
use peppygen::consumed_topics::limb_state::limb_states;
use rerun_viewer::config::default_parameters;
use rerun_viewer::draw::CUBES_NOTE_PATH;
use rerun_viewer::drawn::Drawn;

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
        .wait_until(TIMEOUT, "the grippers' openings", |drawn| {
            drawn
                .iter()
                .any(|call| matches!(call, Drawn::Scalars { path, .. } if path == "plots/grippers"))
        })
        .await?;
    let Drawn::Frame { pose, .. } = &canvas.at("world/arms/right_arm/grasp")[0] else {
        panic!("the grasp point is a frame")
    };
    assert_eq!(pose.position(), [0.3, -0.2, 0.4]);
    // Each plot takes the names of its lines, then a value for each.
    let names = |names: &[&str]| -> Vec<String> { names.iter().map(|n| n.to_string()).collect() };
    assert!(matches!(
        &canvas.at("plots/joints/left_arm")[..],
        [Drawn::SeriesNames { names: lines, .. }, Drawn::Scalars { values, .. }]
            if lines == &names(&["joint_0", "joint_1"]) && values == &[0.1, 0.2]
    ));
    assert!(matches!(
        &canvas.at("plots/grippers")[..],
        [Drawn::SeriesNames { names: lines, .. }, Drawn::Scalars { values, .. }]
            if lines == &names(&["right_gripper"]) && values == &[0.5]
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
            drawn.iter().any(|call| {
                matches!(
                    call,
                    Drawn::SeriesNames { path, names }
                        if path == "plots/joints/right_arm" && names == &["openarm_right_joint1"]
                )
            })
        })
        .await?;
    harness.shutdown().await.map_err(|e| e.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_snapshot_in_another_frame_than_the_robot_frame_is_dropped_and_logged()
-> Result<(), String> {
    let (harness, mocks, canvas) = start(bound(&["sensor_mounts"], default_parameters())).await?;
    hold_the_robot_frame(&mocks, &canvas).await?;
    let mut elsewhere = two_arms();
    elsewhere.frame = ODOM_FRAME.to_string();
    elsewhere.grasp_points[1].position = [9.0, 9.0, 9.0];
    // The publishes share one mock publisher, so their order holds: a snapshot drawn after
    // it proves the drop before it.
    mocks
        .deps
        .limb_state
        .limb_states
        .publish(&elsewhere)
        .await
        .map_err(|e| e.to_string())?;
    publish_arms_until_drawn(&mocks, &canvas, &two_arms()).await?;
    logged(&["limb_states names the frame", ODOM_FRAME, ROBOT_FRAME]).await?;
    let Some(Drawn::Frame { pose, .. }) = canvas.at(GRASP_PATH).last().cloned() else {
        panic!("the grasp point is a frame")
    };
    assert_eq!(pose.position(), [0.3, -0.2, 0.4]);
    let Some(Drawn::LineStrip { points, .. }) =
        canvas.at("world/arms/right_arm/trail").last().cloned()
    else {
        panic!("the trail is a line strip")
    };
    assert!(
        points.iter().all(|point| point == &[0.3, -0.2, 0.4]),
        "the dropped snapshot adds no point to the trail: {points:?}"
    );
    harness.shutdown().await.map_err(|e| e.to_string())
}

// With the sensor_mounts link vacant the viewer reads no robot frame: a snapshot is drawn in
// the frame it names, which limb_state's contract gives as the robot frame.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn with_the_sensor_mounts_link_vacant_a_snapshot_is_drawn_in_the_frame_it_names()
-> Result<(), String> {
    let (harness, mocks, canvas) = start(every_optional_link_vacant()).await?;
    let limb_states = &mocks.deps.limb_state.limb_states;
    let in_odom = limb_states::Message {
        frame: ODOM_FRAME.to_string(),
        ..two_arms()
    };
    publish_once_subscribed(
        limb_states.wait_for_subscriber(TIMEOUT),
        limb_states.publish(&in_odom),
    )
    .await?;
    canvas
        .wait_until(TIMEOUT, "the grasp point", |drawn| {
            drew_at(drawn, GRASP_PATH)
        })
        .await?;
    harness.shutdown().await.map_err(|e| e.to_string())
}
