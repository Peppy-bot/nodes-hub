//! The depth camera placed by the robot's tree at the stamp of each depth frame, by the
//! frame its header names: a frame whose camera transform has not arrived waits for it, up
//! to lookup_timeout_s, and one the tree cannot place once the wait runs out is skipped,
//! naming the lookup's failure kind, unknown_frame for a frame the tree does not hold. The
//! geometry's depth_to_color pose parses as every pose of the contracts does.

use chain_kinematics::nalgebra::Isometry3;
use motion_planner::node::NO_DEPTH_FRAME;
use peppygen::Parameters;
use peppygen::fixtures::harness::{Harness, Mocks};

mod common;
use common::{
    ALIGNED, CAMERA_FRAME, HEIGHT, TIMEOUT, WIDTH, arms, at_ms, camera_rig, check,
    describe_camera_with_depth_turned, facing_the_robot, fix_frame, left_path, left_ready,
    move_the_camera, parameters, publish_arms, publish_frame, refused_naming, right_ready,
    robot_frame, start, wait_for_the_arms,
};

/// The node over the harness with the depth camera linked and its description
/// scripted, a depth frame waiting up to `lookup_timeout_s` for the robot's tree, and the
/// arms measured standing at 0 ms and at 100 ms.
async fn started_waiting_up_to(lookup_timeout_s: f64) -> peppygen::Result<(Harness, Mocks)> {
    started_with_depth_turned(lookup_timeout_s, [0.0, 0.0, 0.0, 1.0]).await
}

/// `started_waiting_up_to`, the geometry turning the depth optical frame by `orientation` in
/// the color one.
async fn started_with_depth_turned(
    lookup_timeout_s: f64,
    orientation: [f64; 4],
) -> peppygen::Result<(Harness, Mocks)> {
    let (harness, mocks) = start(camera_rig(Parameters {
        lookup_timeout_s,
        ..parameters()
    }))
    .await?;
    describe_camera_with_depth_turned(&mocks, ALIGNED, orientation)?;
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

// The geometry's depth_to_color pose parses as each pose of the contracts does: an orientation
// [0, 0, 0, 1.001], whose squared norm lies 0.002 from 1, within the 0.01 tf2 takes, is taken
// normalized, and the map takes the camera's frames.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_depth_pose_takes_an_orientation_tf2_takes() -> peppygen::Result<()> {
    let (harness, mocks) =
        started_with_depth_turned(TIMEOUT.as_secs_f64(), [0.0, 0.0, 0.0, 1.001]).await?;
    move_the_camera(&mocks, at_ms(100), &facing_the_robot()).await?;
    publish_frame(&mocks, at_ms(100), &nothing(), ALIGNED).await?;
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
        &format!(
            "a depth frame was skipped: the robot's tree places no frame {CAMERA_FRAME} in {} \
             at its stamp: extrapolation: ",
            robot_frame()
        ),
    )
    .await;
    assert!(
        skipped
            .message
            .contains(&format!("{} -> {CAMERA_FRAME}", robot_frame())),
        "{skipped}"
    );
    harness.shutdown().await
}

// A frame whose header names a frame the robot's tree does not hold is dropped, its skip
// naming unknown_frame, as a consumer places a picture by the frame it names.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_depth_frame_in_a_frame_the_tree_does_not_hold_is_dropped_as_unknown_frame()
-> peppygen::Result<()> {
    let (harness, mocks) = started_waiting_up_to(0.0).await?;
    fix_frame(&mocks, &robot_frame(), "table", &Isometry3::identity()).await?;
    publish_frame(&mocks, at_ms(100), &nothing(), ALIGNED).await?;
    refused_naming(
        &harness,
        &format!(
            "a depth frame was skipped: the robot's tree places no frame {CAMERA_FRAME} in {} \
             at its stamp: unknown_frame: no transform of the tree names the frame \
             {CAMERA_FRAME}",
            robot_frame()
        ),
    )
    .await;
    harness.shutdown().await
}
