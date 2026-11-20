//! Integration tests over the generated harness: the node in-process, replaying its
//! bundled video, with its sensor_mounts link bound to the generated mock or vacant, and its
//! pictures read the way a consumer reads them.

use std::time::Duration;

use peppygen::Parameters;
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::sensor_mounts::describe_sensors::ResponseData;
use peppygen::parameters::video::{Video, VideoResolution};

/// Bounds every wait for a request or a picture. Generous: the assertions are about what
/// arrives, never about how fast.
const TIMEOUT: Duration = Duration::from_secs(20);

/// The name the camera runs under. The harness runs the node outside any copy, so the
/// instance id is the name as it is.
const CAMERA: &str = "wrist_left";

/// The frame sensor_mounts gives the camera.
const CAMERA_FRAME: &str = "wrist_left_color_optical_frame";

/// How many pictures each test reads.
const PICTURES: usize = 3;

/// A small picture, so each test decodes and sends little.
fn parameters() -> Parameters {
    Parameters {
        device_path: "/dev/video0".to_owned(),
        video: Video {
            camera_encoding: "mjpeg".to_owned(),
            frame_rate: 30,
            resolution: VideoResolution {
                width: 64,
                height: 36,
            },
            topic_encoding: "rgb8".to_owned(),
        },
    }
}

fn config(instance_id: &str) -> Config {
    Config {
        parameters: Some(parameters()),
        instance_id: Some(instance_id.to_owned()),
        ..Config::default()
    }
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

/// A robot with a chest camera, this wrist camera on its left arm, and a lidar.
fn sensors() -> ResponseData {
    ResponseData {
        robot_frame: "base_footprint".to_owned(),
        sensor_names: strings(&["chest", CAMERA, "lidar"]),
        sensor_kinds: strings(&["rgbd_camera", "rgb_camera", "lidar_2d"]),
        sensor_frames: strings(&["chest_color_optical_frame", CAMERA_FRAME, "lidar_link"]),
        carried_by: strings(&["", "left_arm", ""]),
    }
}

/// The frame and the capture index of the next `PICTURES` pictures the node emits.
async fn next_pictures(harness: &mut Harness) -> Vec<(String, u32)> {
    let mut pictures = Vec::new();
    while pictures.len() < PICTURES {
        let picture = tokio::time::timeout(TIMEOUT, harness.emitted.camera_video_stream.next())
            .await
            .expect("no picture reached the contract surface")
            .expect("video_stream reads")
            .expect("video_stream subscription is open");
        pictures.push((picture.header.frame, picture.header.capture_index));
    }
    pictures
}

/// Waits until the node's setup has returned.
async fn setup_returned(harness: &Harness) {
    let deadline = tokio::time::Instant::now() + TIMEOUT;
    while !harness.setup_finished() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the setup never returned"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_picture_names_the_frame_sensor_mounts_gives_the_camera() -> peppygen::Result<()> {
    let (mut harness, mocks) =
        Harness::start_with(config(CAMERA), uvc_camera_rust_mock::setup).await?;
    let mounts = mocks
        .deps
        .sensor_mounts
        .expect("a bound sensor_mounts link starts its mock");
    mounts
        .describe_sensors
        .next_request(TIMEOUT)
        .await?
        .respond(sensors())
        .await?;

    let pictures = next_pictures(&mut harness).await;
    assert!(
        pictures.iter().all(|(frame, _)| frame == CAMERA_FRAME),
        "{pictures:?}"
    );
    assert!(
        pictures.windows(2).all(|pair| pair[0].1 < pair[1].1),
        "the capture index rises from picture to picture: {pictures:?}"
    );
    assert_eq!(mounts.describe_sensors.captured_count(), 1);
    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_its_sensor_mounts_link_vacant_every_picture_names_the_cameras_name()
-> peppygen::Result<()> {
    let config = Config {
        sensor_mounts_vacant: true,
        ..config(CAMERA)
    };
    let (mut harness, mocks) = Harness::start_with(config, uvc_camera_rust_mock::setup).await?;
    assert!(
        mocks.deps.sensor_mounts.is_none(),
        "a vacant boot starts no sensor_mounts mock"
    );

    let pictures = next_pictures(&mut harness).await;
    assert!(
        pictures.iter().all(|(frame, _)| frame == CAMERA),
        "{pictures:?}"
    );
    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_camera_sensor_mounts_does_not_list_fails_its_start() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start_with(config("head"), uvc_camera_rust_mock::setup).await?;
    mocks
        .deps
        .sensor_mounts
        .as_ref()
        .expect("a bound sensor_mounts link starts its mock")
        .describe_sensors
        .next_request(TIMEOUT)
        .await?
        .respond(sensors())
        .await?;

    setup_returned(&harness).await;
    let refusal = harness
        .shutdown()
        .await
        .expect_err("the start fails")
        .to_string();
    assert!(
        refusal.contains(
            "sensor_mounts' describe_sensors names no sensor \"head\" (its sensor_names: chest, \
             wrist_left, lidar): run the camera under one of those names"
        ),
        "{refusal}"
    );
    Ok(())
}
