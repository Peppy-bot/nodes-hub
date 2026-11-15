//! The map over the generated harness, as a subscriber of occupancy_map sees it: before
//! a depth frame reaches the map, each message says why the map waits and holds no
//! cube; once the camera has seen a table, the cubes stand on the table; a planner that
//! reads no camera waits on nothing.

mod common;

use chain_kinematics::nalgebra::Vector3;
use common::{
    TIMEOUT, arms, at_ms, looking_at, over_the_table, publish_arms, right_ready, start_with_camera,
    start_without_camera, table_across, under_a_seen_table, under_the_table,
};
use peppygen::fixtures::emitted_topics::motion_planning::occupancy_map::Message;
use peppygen::fixtures::harness::Harness;

/// The next message of occupancy_map that `wanted` holds of. The timeout only bounds a
/// test that fails.
async fn next_where(harness: &mut Harness, wanted: impl Fn(&Message) -> bool) -> Message {
    let subscription = &mut harness.emitted.motion_planning_occupancy_map;
    tokio::time::timeout(TIMEOUT, async {
        loop {
            let message = subscription
                .next()
                .await
                .expect("a message of occupancy_map")
                .expect("the harness's session is open");
            if wanted(&message) {
                return message;
            }
        }
    })
    .await
    .expect("the message within the timeout")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn before_a_frame_each_message_says_why_the_map_waits() -> peppygen::Result<()> {
    let camera = looking_at(Vector3::new(0.8, 0.3, 1.0), Vector3::new(0.3, 0.3, 0.3));
    let (mut harness, mocks) = start_with_camera(&camera).await?;
    publish_arms(&mocks, at_ms(0), &arms(under_the_table(), right_ready())).await?;
    let message = next_where(&mut harness, |_| true).await;
    assert!(!message.waiting.is_empty(), "the map says why it waits");
    assert!(message.centers.is_empty(), "{:?}", message.centers);
    assert_eq!(message.voxel_size_m, 0.02);
    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn once_the_camera_has_seen_a_table_the_cubes_stand_on_it() -> peppygen::Result<()> {
    let (mut harness, _mocks, _crossing) = under_a_seen_table().await?;
    let ([x0, y0, z0], [x1, y1, top]) = table_across(&under_the_table(), &over_the_table()).0;
    let message = next_where(&mut harness, |message| {
        message.waiting.is_empty() && !message.centers.is_empty()
    })
    .await;
    let (centers, rest) = message.centers.as_chunks::<3>();
    assert!(rest.is_empty(), "three coordinates a cube");
    // A cube whose center lies within half a cube of the table's box touches the table.
    let half = message.voxel_size_m / 2.0;
    for &[x, y, z] in centers {
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
    let message = next_where(&mut harness, |_| true).await;
    assert_eq!(message.waiting, "");
    assert!(message.centers.is_empty(), "{:?}", message.centers);
    harness.shutdown().await
}
