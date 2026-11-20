//! Where the robot carries its sensors, over the wire: describe_sensors
//! answers the robot frame, the root link of the model's URDF, and each
//! camera of the bundled model in the URDF's order, with the contract it
//! streams on, its optical frame and the arm that carries it; a model with no
//! camera answers empty arrays.

use std::time::Duration;

use peppygen::fixtures::exposed_services::sensor_mounts::describe_sensors;
use peppygen::fixtures::harness::{Config, Harness};

mod common;
use common::{an_so101_on_its_own_hardware, on_its_own_hardware, parameters};

/// How long the wire may take for any one exchange.
const WIRE: Duration = Duration::from_secs(10);

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_openarm_v2_names_its_chest_camera_and_the_camera_each_arm_carries()
-> peppygen::Result<()> {
    let (harness, _mocks) =
        Harness::start_with(on_its_own_hardware(), robot_initializer::setup).await?;

    let answer = describe_sensors::poll(&harness, WIRE).await?;
    assert_eq!(answer.robot_frame, "base_link");
    assert_eq!(answer.sensor_names, ["wrist_left", "wrist_right", "chest"]);
    assert_eq!(
        answer.sensor_kinds,
        ["rgb_camera", "rgb_camera", "rgbd_camera"]
    );
    assert_eq!(
        answer.sensor_frames,
        [
            "wrist_left_optical_frame",
            "wrist_right_optical_frame",
            "chest_optical_frame"
        ]
    );
    assert_eq!(answer.carried_by, ["left_arm", "right_arm", ""]);

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_so101_names_the_wrist_camera_its_arm_carries() -> peppygen::Result<()> {
    let (harness, _mocks) =
        Harness::start_with(an_so101_on_its_own_hardware(), robot_initializer::setup).await?;

    let answer = describe_sensors::poll(&harness, WIRE).await?;
    assert_eq!(answer.robot_frame, "base_link");
    assert_eq!(answer.sensor_names, ["wrist"]);
    assert_eq!(answer.sensor_kinds, ["rgb_camera"]);
    assert_eq!(answer.sensor_frames, ["wrist_optical_frame"]);
    assert_eq!(answer.carried_by, ["arm"]);

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_robot_whose_model_carries_no_camera_answers_empty_arrays() -> peppygen::Result<()> {
    let config = Config {
        parameters: Some(parameters("openarm_v1")),
        ..on_its_own_hardware()
    };
    let (harness, _mocks) = Harness::start_with(config, robot_initializer::setup).await?;

    let answer = describe_sensors::poll(&harness, WIRE).await?;
    assert_eq!(answer.robot_frame, "base_link");
    assert!(answer.sensor_names.is_empty() && answer.sensor_kinds.is_empty());
    assert!(answer.sensor_frames.is_empty() && answer.carried_by.is_empty());

    harness.shutdown().await
}
