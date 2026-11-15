//! The camera over the wire: the color and depth frames become images, and the depth
//! frames points in the robot frame, through the geometry and the pose the camera's
//! services give.

mod common;

use std::time::{Duration, SystemTime};

use common::{IDENTITY, TIMEOUT, bound, drew_at, publish_once_subscribed, start};
use peppygen::consumed_topics::camera::depth_stream::{
    Message as DepthFrame, MessageHeader as DepthHeader,
};
use peppygen::consumed_topics::camera::video_stream::{
    Message as ColorFrame, MessageHeader as ColorHeader,
};
use peppygen::mock::deps::camera::depth_stream_info;
use peppygen::mock::deps::camera_mounts::get_camera_poses;
use peppygen::mock::deps::geometry::{get_depth_intrinsics, get_depth_to_color_extrinsics};
use rerun_viewer::config::default_parameters;
use rerun_viewer::draw::camera::{COLOR_PATH, DEPTH_PATH, POINTS_PATH};
use rerun_viewer::model::camera::ColorImage;
use rerun_viewer::testing::Drawn;

const ALIGN: &str = "depth_to_color";

/// A 4 by 3 depth stream with a focal length of 2 pixels and its principal point at
/// (1.5, 1), half a meter a count, on the color frame; the chest camera stands at
/// (1, 0, 0) of the robot, turned as the robot frame, and carried by `carried_by`
/// (empty: fixed to the base). camera_mounts answers `pose_answers` asks.
fn script_the_camera(
    mocks: &peppygen::fixtures::harness::Mocks,
    carried_by: &str,
    pose_answers: usize,
) -> Result<(), String> {
    let camera = mocks.deps.camera.as_ref().expect("the camera is linked");
    let geometry = mocks
        .deps
        .geometry
        .as_ref()
        .expect("the geometry is linked");
    let mounts = mocks
        .deps
        .camera_mounts
        .as_ref()
        .expect("camera_mounts is linked");
    camera
        .depth_stream_info
        .enqueue_response(depth_stream_info::ResponseData::new(
            4,
            3,
            15,
            "z16".to_string(),
            0.5,
        ))
        .map_err(|e| e.to_string())?;
    geometry
        .get_depth_intrinsics
        .enqueue_response(get_depth_intrinsics::ResponseData::new(
            4,
            3,
            2.0,
            2.0,
            1.5,
            1.0,
            "none".to_string(),
            vec![],
            "z".to_string(),
            0.1,
            10.0,
            ALIGN.to_string(),
        ))
        .map_err(|e| e.to_string())?;
    geometry
        .get_depth_to_color_extrinsics
        .enqueue_response(get_depth_to_color_extrinsics::ResponseData::new(
            ALIGN.to_string(),
            [0.0; 3],
            IDENTITY,
        ))
        .map_err(|e| e.to_string())?;
    for _ in 0..pose_answers {
        mounts
            .get_camera_poses
            .enqueue_response(get_camera_poses::ResponseData::new(
                SystemTime::UNIX_EPOCH,
                vec!["chest".to_string()],
                vec![1.0, 0.0, 0.0],
                IDENTITY.to_vec(),
                vec![carried_by.to_string()],
            ))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The number of clouds drawn so far.
fn clouds(drawn: &[Drawn]) -> usize {
    drawn
        .iter()
        .filter(|call| call.path() == Some(POINTS_PATH))
        .count()
}

/// A frame of 4 by 3 samples of 2 counts: 1 m everywhere.
fn depth_frame() -> DepthFrame {
    DepthFrame {
        header: DepthHeader {
            timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(20),
            frame_id: 1,
            align_mode: ALIGN.to_string(),
        },
        encoding: "z16".to_string(),
        width: 4,
        height: 3,
        frame: [2_u16; 12].iter().flat_map(|s| s.to_le_bytes()).collect(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_depth_frame_becomes_an_image_and_points_in_the_robot_frame() -> Result<(), String> {
    let mut parameters = default_parameters();
    parameters.cloud_stride = 2;
    let links = ["camera", "geometry", "camera_mounts"];
    let (harness, mocks, canvas) = start(bound(&links, parameters)).await?;
    script_the_camera(&mocks, "", 1)?;
    let camera = mocks.deps.camera.as_ref().expect("the camera is linked");
    publish_once_subscribed(
        camera.depth_stream.wait_for_subscriber(TIMEOUT),
        camera.depth_stream.publish(&depth_frame()),
    )
    .await?;
    canvas
        .wait_until(TIMEOUT, "the points", |drawn| drew_at(drawn, POINTS_PATH))
        .await?;
    // Pixels (0, 0), (2, 0), (0, 2) and (2, 2) read ((u - 1.5) / 2, (v - 1) / 2, 1) in
    // the camera, which stands 1 m along the robot's x.
    let Drawn::Points { points, .. } = &canvas.at(POINTS_PATH)[0] else {
        panic!("points")
    };
    assert_eq!(
        points,
        &[
            [0.25, -0.5, 1.0],
            [1.25, -0.5, 1.0],
            [0.25, 0.5, 1.0],
            [1.25, 0.5, 1.0]
        ]
    );
    let Drawn::DepthImage { image, .. } = &canvas.at(DEPTH_PATH)[0] else {
        panic!("a depth image")
    };
    assert_eq!(
        (image.width(), image.height(), image.depth_unit().get()),
        (4, 3, 0.5)
    );
    harness.shutdown().await.map_err(|e| e.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_fixed_camera_is_placed_once_and_a_carried_one_for_each_cloud() -> Result<(), String> {
    for (carried_by, asks) in [("", 1), ("left_arm", 2)] {
        let mut parameters = default_parameters();
        // 1 us between two clouds: each frame, published after the cloud of the one
        // before is drawn, makes its own cloud.
        parameters.cloud_rate_hz = 1e6;
        let links = ["camera", "geometry", "camera_mounts"];
        let (harness, mocks, canvas) = start(bound(&links, parameters)).await?;
        script_the_camera(&mocks, carried_by, 2)?;
        let camera = mocks.deps.camera.as_ref().expect("the camera is linked");
        for count in 1..=2 {
            publish_once_subscribed(
                camera.depth_stream.wait_for_subscriber(TIMEOUT),
                camera.depth_stream.publish(&depth_frame()),
            )
            .await?;
            canvas
                .wait_until(TIMEOUT, "the next cloud", |drawn| clouds(drawn) == count)
                .await?;
        }
        let mounts = mocks
            .deps
            .camera_mounts
            .as_ref()
            .expect("camera_mounts is linked");
        assert_eq!(
            mounts.get_camera_poses.captured_count(),
            asks,
            "carried by {carried_by:?}"
        );
        harness.shutdown().await.map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_bgr8_frame_becomes_an_rgb_image() -> Result<(), String> {
    let links = ["camera", "geometry", "camera_mounts"];
    let (harness, mocks, canvas) = start(bound(&links, default_parameters())).await?;
    let camera = mocks.deps.camera.as_ref().expect("the camera is linked");
    publish_once_subscribed(
        camera.video_stream.wait_for_subscriber(TIMEOUT),
        camera.video_stream.publish(&ColorFrame {
            header: ColorHeader {
                timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(20),
                frame_id: 1,
                align_mode: ALIGN.to_string(),
            },
            encoding: "bgr8".to_string(),
            width: 1,
            height: 1,
            frame: vec![10, 20, 30],
        }),
    )
    .await?;
    canvas
        .wait_until(TIMEOUT, "the color image", |drawn| {
            drew_at(drawn, COLOR_PATH)
        })
        .await?;
    let Drawn::ColorImage { image, .. } = &canvas.at(COLOR_PATH)[0] else {
        panic!("a color image")
    };
    assert_eq!(
        image,
        &ColorImage::Rgb {
            width: 1,
            height: 1,
            pixels: vec![30, 20, 10]
        }
    );
    harness.shutdown().await.map_err(|e| e.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_the_camera_mounts_link_the_depth_image_is_drawn_and_no_points()
-> Result<(), String> {
    let (harness, mocks, canvas) =
        start(bound(&["camera", "geometry"], default_parameters())).await?;
    script_camera_unit(&mocks)?;
    let camera = mocks.deps.camera.as_ref().expect("the camera is linked");
    publish_once_subscribed(
        camera.depth_stream.wait_for_subscriber(TIMEOUT),
        camera.depth_stream.publish(&depth_frame()),
    )
    .await?;
    canvas
        .wait_until(TIMEOUT, "the depth image", |drawn| {
            drew_at(drawn, DEPTH_PATH)
        })
        .await?;
    assert!(canvas.at(POINTS_PATH).is_empty());
    harness.shutdown().await.map_err(|e| e.to_string())
}

fn script_camera_unit(mocks: &peppygen::fixtures::harness::Mocks) -> Result<(), String> {
    mocks
        .deps
        .camera
        .as_ref()
        .expect("the camera is linked")
        .depth_stream_info
        .enqueue_response(depth_stream_info::ResponseData::new(
            4,
            3,
            15,
            "z16".to_string(),
            0.5,
        ))
        .map_err(|e| e.to_string())
}
