//! The camera over the wire: the color and depth frames become images, and the depth
//! frames points in the robot frame, through the geometry the camera's services give and
//! the frame each depth frame names, which the robot's tree places at that frame's time.

mod common;

use std::time::{Duration, SystemTime};

use common::{
    CAMERA_FRAME, IDENTITY, ROBOT_FRAME, TIMEOUT, bound, drew_at, hold_the_robot_frame, logged,
    publish_once_subscribed, sensor_mounts, start,
};
use peppygen::consumed_services::geometry::get_depth_to_color_extrinsics::ResponseDataDepthToColor as DepthToColor;
use peppygen::consumed_topics::camera::depth_stream::{
    Message as DepthFrame, MessageHeader as DepthHeader,
};
use peppygen::consumed_topics::camera::video_stream::{
    Message as ColorFrame, MessageHeader as ColorHeader,
};
use peppygen::consumed_topics::transforms::static_transforms::{
    Message as StaticTransforms, MessageTransformsItem as StaticTransform,
};
use peppygen::consumed_topics::transforms::transforms::{
    Message as Transforms, MessageTransformsItem as MovingTransform,
};
use peppygen::fixtures::harness::Mocks;
use peppygen::mock::deps::camera::depth_stream_info;
use peppygen::mock::deps::geometry::{get_depth_intrinsics, get_depth_to_color_extrinsics};
use rerun_viewer::config::default_parameters;
use rerun_viewer::draw::camera::{COLOR_PATH, DEPTH_PATH, POINTS_PATH};
use rerun_viewer::drawn::Drawn;
use rerun_viewer::model::camera::ColorImage;
use rerun_viewer::testing::FakeCanvas;

const ALIGN: &str = "depth_to_color";

/// The links the points need: the camera, its geometry, the robot frame and the robot's
/// tree.
const CLOUD_LINKS: [&str; 4] = ["camera", "geometry", "sensor_mounts", "transforms"];

/// A 4 by 3 depth stream in the frame `CAMERA_FRAME` with a focal length of 2 pixels and
/// its principal point at (1.5, 1), half a meter a count, on the color frame; sensor_mounts
/// names the robot frame.
async fn script_the_camera(mocks: &Mocks, canvas: &FakeCanvas) -> Result<(), String> {
    script_the_camera_in(mocks, canvas, CAMERA_FRAME).await
}

/// The camera of `script_the_camera`, its depth stream placed in the frame `frame`.
async fn script_the_camera_in(
    mocks: &Mocks,
    canvas: &FakeCanvas,
    frame: &str,
) -> Result<(), String> {
    script_the_geometry_in(mocks, frame)?;
    hold_the_robot_frame(mocks, canvas).await
}

/// The depth stream and its geometry of `script_the_camera`, placed in the frame `frame`.
fn script_the_geometry_in(mocks: &Mocks, frame: &str) -> Result<(), String> {
    let camera = mocks.deps.camera.as_ref().expect("the camera is linked");
    let geometry = mocks
        .deps
        .geometry
        .as_ref()
        .expect("the geometry is linked");
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
            DepthToColor {
                frame: frame.to_string(),
                position: [0.0; 3],
                orientation: IDENTITY,
            },
        ))
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// The publisher of the robot's tree.
fn tree_publisher(mocks: &Mocks) -> &peppygen::mock::deps::transforms::Mock {
    mocks
        .deps
        .transforms
        .first()
        .expect("the transforms link has a publisher")
}

/// Publishes the fixed transforms `transforms`, each a parent, a child and the child's
/// position in the parent, which it is turned as.
async fn publish_fixed(mocks: &Mocks, transforms: &[(&str, &str, [f64; 3])]) -> Result<(), String> {
    let tree = tree_publisher(mocks);
    publish_once_subscribed(
        tree.static_transforms.wait_for_subscriber(TIMEOUT),
        tree.static_transforms.publish(&StaticTransforms {
            transforms: transforms
                .iter()
                .map(|&(parent, child, position)| StaticTransform {
                    frame: parent.to_string(),
                    child_frame: child.to_string(),
                    position,
                    orientation: IDENTITY,
                })
                .collect(),
        }),
    )
    .await
}

/// Publishes the arm link `x` m along the robot's x at `seconds`.
async fn publish_arm_at(mocks: &Mocks, seconds: u32, x: f64) -> Result<(), String> {
    let tree = tree_publisher(mocks);
    publish_once_subscribed(
        tree.transforms.wait_for_subscriber(TIMEOUT),
        tree.transforms.publish(&Transforms {
            timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(u64::from(seconds)),
            transforms: vec![MovingTransform {
                frame: ROBOT_FRAME.to_string(),
                child_frame: "arm".to_string(),
                position: [x, 0.0, 0.0],
                orientation: IDENTITY,
            }],
            start_timestamp: SystemTime::UNIX_EPOCH,
        }),
    )
    .await
}

/// The number of clouds drawn so far.
fn clouds(drawn: &[Drawn]) -> usize {
    drawn
        .iter()
        .filter(|call| call.path() == Some(POINTS_PATH))
        .count()
}

/// A frame of 4 by 3 samples of 2 counts, 1 m everywhere, in `CAMERA_FRAME`, taken at 20 s.
fn depth_frame() -> DepthFrame {
    depth_frame_at(20)
}

/// A frame of 4 by 3 samples of 2 counts, 1 m everywhere, in `CAMERA_FRAME`, taken at
/// `seconds`.
fn depth_frame_at(seconds: u32) -> DepthFrame {
    depth_frame_in(CAMERA_FRAME, seconds)
}

/// A frame of 4 by 3 samples of 2 counts, 1 m everywhere, in the frame `frame`, taken at
/// `seconds`.
fn depth_frame_in(frame: &str, seconds: u32) -> DepthFrame {
    DepthFrame {
        header: DepthHeader {
            timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(u64::from(seconds)),
            frame: frame.to_string(),
            capture_index: seconds,
            align_mode: ALIGN.to_string(),
        },
        encoding: "z16".to_string(),
        width: 4,
        height: 3,
        data: [2_u16; 12].iter().flat_map(|s| s.to_le_bytes()).collect(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_depth_frame_becomes_an_image_and_points_where_the_tree_places_its_frame()
-> Result<(), String> {
    let mut parameters = default_parameters();
    parameters.cloud_stride = 2;
    parameters.depth_image = true;
    let (harness, mocks, canvas) = start(bound(&CLOUD_LINKS, parameters)).await?;
    script_the_camera(&mocks, &canvas).await?;
    publish_fixed(&mocks, &[(ROBOT_FRAME, CAMERA_FRAME, [1.0, 0.0, 0.0])]).await?;
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
    // the frame the depth frame names, which the tree stands 1 m along the robot's x.
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
async fn by_default_a_depth_frame_becomes_points_and_no_depth_image() -> Result<(), String> {
    let (harness, mocks, canvas) = start(bound(&CLOUD_LINKS, default_parameters())).await?;
    script_the_camera(&mocks, &canvas).await?;
    publish_fixed(&mocks, &[(ROBOT_FRAME, CAMERA_FRAME, [1.0, 0.0, 0.0])]).await?;
    let camera = mocks.deps.camera.as_ref().expect("the camera is linked");
    publish_once_subscribed(
        camera.depth_stream.wait_for_subscriber(TIMEOUT),
        camera.depth_stream.publish(&depth_frame()),
    )
    .await?;
    canvas
        .wait_until(TIMEOUT, "the points", |drawn| drew_at(drawn, POINTS_PATH))
        .await?;
    // A frame's depth image is drawn before its points, when it is drawn.
    assert!(canvas.at(DEPTH_PATH).is_empty());
    harness.shutdown().await.map_err(|e| e.to_string())
}

/// The x of the first point of the cloud `index` drawn, the pixel (0, 0), which stands
/// 1.25 m behind the camera's x.
fn first_x(canvas: &rerun_viewer::testing::FakeCanvas, index: usize) -> f32 {
    let Drawn::Points { points, .. } = &canvas.at(POINTS_PATH)[index] else {
        panic!("points")
    };
    points[0][0]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_carried_camera_is_placed_where_it_stood_when_each_frame_was_taken() -> Result<(), String>
{
    let mut parameters = default_parameters();
    // 1 us between two clouds: each frame makes its own cloud.
    parameters.cloud_rate_hz = 1e6;
    let (harness, mocks, canvas) = start(bound(&CLOUD_LINKS, parameters)).await?;
    script_the_camera(&mocks, &canvas).await?;
    // The camera rides the arm link, which the tree moves.
    publish_fixed(&mocks, &[("arm", CAMERA_FRAME, [0.0; 3])]).await?;
    let camera = mocks.deps.camera.as_ref().expect("the camera is linked");
    for (count, (seconds, x)) in [(20, 1.0), (21, 2.0)].into_iter().enumerate() {
        // The frame waits for the arm's transform of its time, which comes after it.
        publish_once_subscribed(
            camera.depth_stream.wait_for_subscriber(TIMEOUT),
            camera.depth_stream.publish(&depth_frame_at(seconds)),
        )
        .await?;
        publish_arm_at(&mocks, seconds, x).await?;
        canvas
            .wait_until(TIMEOUT, "the frame's cloud", |drawn| {
                clouds(drawn) == count + 1
            })
            .await?;
        assert_eq!(
            f64::from(first_x(&canvas, count)),
            x - 0.75,
            "the frame at {seconds} s"
        );
    }
    // The robot frame is asked once.
    assert_eq!(sensor_mounts(&mocks).describe_sensors.captured_count(), 1);
    harness.shutdown().await.map_err(|e| e.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_newer_frame_takes_the_place_of_one_that_waits() -> Result<(), String> {
    let mut parameters = default_parameters();
    parameters.cloud_rate_hz = 1e6;
    parameters.image_rate_hz = 1e6;
    parameters.depth_image = true;
    let (harness, mocks, canvas) = start(bound(&CLOUD_LINKS, parameters)).await?;
    script_the_camera(&mocks, &canvas).await?;
    publish_fixed(&mocks, &[("arm", CAMERA_FRAME, [0.0; 3])]).await?;
    let camera = mocks.deps.camera.as_ref().expect("the camera is linked");
    publish_once_subscribed(
        camera.depth_stream.wait_for_subscriber(TIMEOUT),
        camera.depth_stream.publish(&depth_frame_at(20)),
    )
    .await?;
    camera
        .depth_stream
        .publish(&depth_frame_at(21))
        .await
        .map_err(|e| e.to_string())?;
    // A frame's image is drawn once the frame went to the cloud maker.
    canvas
        .wait_until(TIMEOUT, "both depth images", |drawn| {
            drawn
                .iter()
                .filter(|call| call.path() == Some(DEPTH_PATH))
                .count()
                == 2
        })
        .await?;
    // The frame of 21 s is in the watch before the arm's transforms come, so it takes the
    // place of the frame of 20 s, which waits for the arm's transform of its time or was
    // not taken yet: the transforms of 20 s and 21 s then place the frame of 21 s alone.
    publish_arm_at(&mocks, 20, 1.0).await?;
    publish_arm_at(&mocks, 21, 2.0).await?;
    canvas
        .wait_until(TIMEOUT, "a cloud", |drawn| clouds(drawn) == 1)
        .await?;
    assert_eq!(
        f64::from(first_x(&canvas, 0)),
        2.0 - 0.75,
        "the cloud of the frame at 21 s"
    );
    harness.shutdown().await.map_err(|e| e.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_depth_frame_in_a_frame_the_tree_does_not_hold_draws_no_points_and_logs_unknown_frame()
-> Result<(), String> {
    const WRIST_FRAME: &str = "wrist_color_optical_frame";
    let mut parameters = default_parameters();
    parameters.lookup_timeout_s = 0.0;
    let (harness, mocks, canvas) = start(bound(&CLOUD_LINKS, parameters)).await?;
    script_the_camera_in(&mocks, &canvas, WRIST_FRAME).await?;
    // The tree holds the chest camera's frame alone.
    publish_fixed(&mocks, &[(ROBOT_FRAME, CAMERA_FRAME, [1.0, 0.0, 0.0])]).await?;
    let camera = mocks.deps.camera.as_ref().expect("the camera is linked");
    publish_once_subscribed(
        camera.depth_stream.wait_for_subscriber(TIMEOUT),
        camera
            .depth_stream
            .publish(&depth_frame_in(WRIST_FRAME, 20)),
    )
    .await?;
    logged(&["unknown_frame", WRIST_FRAME]).await?;
    assert!(canvas.at(POINTS_PATH).is_empty());
    harness.shutdown().await.map_err(|e| e.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_depth_frame_in_another_frame_than_the_geometry_draws_no_points() -> Result<(), String> {
    const OTHER_FRAME: &str = "head_color_optical_frame";
    let (harness, mocks, canvas) = start(bound(&CLOUD_LINKS, default_parameters())).await?;
    script_the_camera(&mocks, &canvas).await?;
    publish_fixed(&mocks, &[(ROBOT_FRAME, CAMERA_FRAME, [1.0, 0.0, 0.0])]).await?;
    let camera = mocks.deps.camera.as_ref().expect("the camera is linked");
    publish_once_subscribed(
        camera.depth_stream.wait_for_subscriber(TIMEOUT),
        camera
            .depth_stream
            .publish(&depth_frame_in(OTHER_FRAME, 20)),
    )
    .await?;
    logged(&["the depth frames name the frame", OTHER_FRAME, CAMERA_FRAME]).await?;
    assert!(canvas.at(POINTS_PATH).is_empty());
    harness.shutdown().await.map_err(|e| e.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_frame_wider_than_image_max_width_is_drawn_shrunk() -> Result<(), String> {
    let mut parameters = default_parameters();
    parameters.image_max_width = 2;
    let (harness, mocks, canvas) = start(bound(&CLOUD_LINKS, parameters)).await?;
    let camera = mocks.deps.camera.as_ref().expect("the camera is linked");
    publish_once_subscribed(
        camera.video_stream.wait_for_subscriber(TIMEOUT),
        camera.video_stream.publish(&ColorFrame {
            header: ColorHeader {
                timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(20),
                frame: CAMERA_FRAME.to_string(),
                capture_index: 1,
                align_mode: ALIGN.to_string(),
            },
            encoding: "rgb8".to_string(),
            // 4 by 2 pixels: the left half 10, the right half 200.
            width: 4,
            height: 2,
            data: [[10_u8; 6], [200; 6], [10; 6], [200; 6]].concat(),
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
            width: 2,
            height: 1,
            pixels: vec![10, 10, 10, 200, 200, 200]
        }
    );
    harness.shutdown().await.map_err(|e| e.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_bgr8_frame_becomes_an_rgb_image() -> Result<(), String> {
    let (harness, mocks, canvas) = start(bound(&CLOUD_LINKS, default_parameters())).await?;
    let camera = mocks.deps.camera.as_ref().expect("the camera is linked");
    publish_once_subscribed(
        camera.video_stream.wait_for_subscriber(TIMEOUT),
        camera.video_stream.publish(&ColorFrame {
            header: ColorHeader {
                timestamp: SystemTime::UNIX_EPOCH + Duration::from_secs(20),
                frame: CAMERA_FRAME.to_string(),
                capture_index: 1,
                align_mode: ALIGN.to_string(),
            },
            encoding: "bgr8".to_string(),
            width: 1,
            height: 1,
            data: vec![10, 20, 30],
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
async fn without_the_sensor_mounts_and_transforms_links_the_depth_image_is_drawn_and_no_points()
-> Result<(), String> {
    let mut parameters = default_parameters();
    parameters.depth_image = true;
    let (harness, mocks, canvas) = start(bound(&["camera", "geometry"], parameters)).await?;
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
