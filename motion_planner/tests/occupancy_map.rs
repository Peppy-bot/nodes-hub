//! The map over the generated harness, as a subscriber of occupancy_map sees it: before
//! a depth frame reaches the map, each message says why the map waits and holds no
//! cube; once the camera has seen a table, the cubes stand on the table, in the robot frame;
//! a planner that reads no camera waits on nothing.

mod common;

use std::time::SystemTime;

use chain_kinematics::nalgebra::Vector3;
use common::{
    arms, at_ms, looking_at, next_map_where, over_the_table, publish_arms, right_ready,
    robot_frame, start_with_camera, start_without_camera, table_across, under_a_seen_table,
    under_the_table,
};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn before_a_frame_each_message_says_why_the_map_waits() -> peppygen::Result<()> {
    let camera = looking_at(Vector3::new(0.8, 0.3, 1.0), Vector3::new(0.3, 0.3, 0.3));
    let (mut harness, mocks) = start_with_camera(&camera).await?;
    publish_arms(&mocks, at_ms(0), &arms(under_the_table(), right_ready())).await?;
    let message = next_map_where(&mut harness, |_| true).await;
    assert!(!message.waiting.is_empty(), "the map says why it waits");
    assert!(message.centers.is_empty(), "{:?}", message.centers);
    assert_eq!(message.voxel_size_m, 0.02);
    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn once_the_camera_has_seen_a_table_the_cubes_stand_on_it() -> peppygen::Result<()> {
    let started_at = SystemTime::now();
    let (mut harness, _mocks, _crossing) = under_a_seen_table().await?;
    let ([x0, y0, z0], [x1, y1, top]) = table_across(&under_the_table(), &over_the_table()).0;
    let message = next_map_where(&mut harness, |message| {
        message.waiting.is_empty() && !message.centers.is_empty()
    })
    .await;
    // The map stands in the robot frame, which no moving transform places on OpenArm, so it
    // holds at every time: the message gives the time the node read the map.
    assert_eq!(message.frame, robot_frame());
    assert!(
        (started_at..=SystemTime::now()).contains(&message.timestamp),
        "{:?} after {started_at:?}",
        message.timestamp
    );
    // A cube whose center lies within half a cube of the table's box touches the table.
    let half = message.voxel_size_m / 2.0;
    for [x, y, z] in message.centers.iter().map(|center| center.position) {
        assert!(
            (x0 - half..=x1 + half).contains(&x)
                && (y0 - half..=y1 + half).contains(&y)
                && (z0 - half..=top + half).contains(&z),
            "the cube at ({x}, {y}, {z}) is off the table from ({x0}, {y0}, {z0}) to ({x1}, \
             {y1}, {top})"
        );
    }
    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_planner_that_reads_no_camera_waits_on_nothing() -> peppygen::Result<()> {
    let (mut harness, _mocks) =
        start_without_camera(&arms(under_the_table(), right_ready())).await?;
    let message = next_map_where(&mut harness, |_| true).await;
    assert_eq!(message.waiting, "");
    assert!(message.centers.is_empty(), "{:?}", message.centers);
    harness.shutdown().await
}
