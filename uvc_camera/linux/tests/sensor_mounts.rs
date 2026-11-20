//! The node's start over the generated harness, with a device path that holds no device and
//! its sensor_mounts link bound to the generated mock or vacant: the node reads the frame every
//! picture names before it opens the camera, so each start ends at the camera's open, or before
//! it when a bound sensor_mounts places no camera of the node's name.

use std::time::Duration;

use peppygen::Parameters;
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::sensor_mounts::describe_sensors::ResponseData;
use peppygen::parameters::video::{Video, VideoResolution};

/// Bounds every wait for a request or the end of the start. Generous: the assertions are
/// about what happens, never about how fast.
const TIMEOUT: Duration = Duration::from_secs(20);

/// The name the camera runs under. The harness runs the node outside any copy, so the
/// instance id is the name as it is.
const CAMERA: &str = "wrist_left";

/// A device path that holds no device, so the camera's open fails.
const NO_DEVICE: &str = "/dev/peppy_test_no_camera";

fn config(instance_id: &str) -> Config {
    Config {
        parameters: Some(Parameters {
            device_path: NO_DEVICE.to_owned(),
            video: Video {
                camera_encoding: "mjpeg".to_owned(),
                frame_rate: 30,
                resolution: VideoResolution {
                    width: 640,
                    height: 480,
                },
                topic_encoding: "rgb8".to_owned(),
            },
        }),
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
        sensor_frames: strings(&[
            "chest_color_optical_frame",
            "wrist_left_color_optical_frame",
            "lidar_link",
        ]),
        carried_by: strings(&["", "left_arm", ""]),
    }
}

/// The error the node's start failed with, once its setup returned.
async fn start_error(harness: Harness) -> String {
    let deadline = tokio::time::Instant::now() + TIMEOUT;
    while !harness.setup_finished() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the setup never returned"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    harness
        .shutdown()
        .await
        .expect_err("the start fails")
        .to_string()
}

/// The node running as `instance_id` with its sensor_mounts link bound to the robot
/// `sensors` describes, its ask of describe_sensors answered: how many asks reached
/// sensor_mounts, and the error its start failed with.
async fn failed_start(instance_id: &str) -> peppygen::Result<(usize, String)> {
    let (harness, mocks) =
        Harness::start_with(config(instance_id), uvc_camera_linux::setup).await?;
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
    let refusal = start_error(harness).await;
    Ok((mounts.describe_sensors.captured_count(), refusal))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_camera_sensor_mounts_lists_reads_its_frame_and_then_opens_its_device()
-> peppygen::Result<()> {
    let (asks, refusal) = failed_start(CAMERA).await?;
    assert_eq!(asks, 1);
    assert!(
        refusal.contains(&format!("Device {NO_DEVICE} does not exist")),
        "{refusal}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_camera_with_its_sensor_mounts_link_vacant_opens_its_device_asking_nothing()
-> peppygen::Result<()> {
    let config = Config {
        sensor_mounts_vacant: true,
        ..config(CAMERA)
    };
    let (harness, mocks) = Harness::start_with(config, uvc_camera_linux::setup).await?;
    assert!(
        mocks.deps.sensor_mounts.is_none(),
        "a vacant boot starts no sensor_mounts mock"
    );
    let refusal = start_error(harness).await;
    assert!(
        refusal.contains(&format!("Device {NO_DEVICE} does not exist")),
        "{refusal}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_camera_sensor_mounts_does_not_list_fails_its_start_before_it_opens_its_device()
-> peppygen::Result<()> {
    let (asks, refusal) = failed_start("head").await?;
    assert_eq!(asks, 1);
    assert!(
        refusal.contains(
            "sensor_mounts' describe_sensors names no sensor \"head\" (its sensor_names: chest, \
             wrist_left, lidar): run the camera under one of those names, or bind its \
             sensor_mounts link to the robot that carries it"
        ),
        "{refusal}"
    );
    assert!(!refusal.contains(NO_DEVICE), "{refusal}");
    Ok(())
}
