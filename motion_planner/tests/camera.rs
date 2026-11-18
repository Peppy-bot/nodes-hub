//! The perception camera placed by the robot's tree at the stamp of each depth frame: a
//! frame whose camera transform has not arrived waits for it, up to lookup_timeout_s, and
//! one the tree cannot place once the wait runs out is skipped, naming the lookup's
//! failure.

use motion_planner::node::NO_DEPTH_FRAME;
use peppygen::Parameters;
use peppygen::fixtures::harness::{Harness, Mocks};

mod common;
use common::{
    ALIGNED, HEIGHT, TIMEOUT, WIDTH, arms, at_ms, camera_rig, check, describe_camera,
    facing_the_robot, left_path, left_ready, move_the_camera, parameters, publish_arms,
    publish_frame, refused_naming, right_ready, start, wait_for_the_arms,
};

/// The node over the harness with the perception camera linked and its description
/// scripted, a depth frame waiting up to `lookup_timeout_s` for the robot's tree, and the
/// arms measured standing at 0 ms and at 100 ms.
async fn started_waiting_up_to(lookup_timeout_s: f64) -> peppygen::Result<(Harness, Mocks)> {
    let (harness, mocks) = start(camera_rig(Parameters {
        lookup_timeout_s,
        ..parameters()
    }))
    .await?;
    describe_camera(&mocks, ALIGNED)?;
    let stand = arms(left_ready(), right_ready());
    publish_arms(&mocks, at_ms(0), &stand).await?;
    publish_arms(&mocks, at_ms(100), &stand).await?;
    Ok((harness, mocks))
}

/// A depth frame of nothing in view.
fn nothing() -> Vec<Option<[f64; 3]>> {
    vec![None; (WIDTH * HEIGHT) as usize]
}

// A camera the tree moves is placed at the frame's stamp: the frame of 100 ms waits while
// the tree holds the camera at 0 ms alone, and the map takes it once the camera's transform
// of 100 ms arrives.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_depth_frame_waits_for_the_tree_to_place_its_camera() -> peppygen::Result<()> {
    let (harness, mocks) = started_waiting_up_to(TIMEOUT.as_secs_f64()).await?;
    let camera = facing_the_robot();
    move_the_camera(&mocks, at_ms(0), &camera).await?;
    publish_frame(&mocks, at_ms(100), &nothing(), ALIGNED).await?;
    let waiting = check(&harness, &left_path(&[], &[left_ready()]))
        .await
        .expect_err("the frame waits for the camera's transform of 100 ms");
    assert!(waiting.message.starts_with(NO_DEPTH_FRAME), "{waiting}");

    move_the_camera(&mocks, at_ms(100), &camera).await?;
    let taken = wait_for_the_arms(&harness, left_ready()).await;
    assert_eq!(taken.map_stamp, Some(at_ms(100)));
    harness.shutdown().await
}

// With no wait, a frame the tree cannot place at its stamp is skipped, and the refusal of
// a check names the lookup's failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_depth_frame_the_tree_cannot_place_is_skipped_naming_the_failure() -> peppygen::Result<()>
{
    let (harness, mocks) = started_waiting_up_to(0.0).await?;
    move_the_camera(&mocks, at_ms(0), &facing_the_robot()).await?;
    publish_frame(&mocks, at_ms(100), &nothing(), ALIGNED).await?;
    let skipped = refused_naming(
        &harness,
        "a depth frame was skipped: the robot's tree places no camera frame \
         chest_optical_frame in world at its stamp: ",
    )
    .await;
    assert!(
        skipped.message.contains("world -> chest_optical_frame"),
        "{skipped}"
    );
    harness.shutdown().await
}
