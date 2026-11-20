//! Where this robot carries its sensors, from its model alone, as
//! sensor_mounts' describe_sensors answers it: the robot frame, the root link
//! of the model's URDF, each camera of the model, then each 2-D lidar, each in
//! the URDF's order, with the contract it streams on, its frame and the arm
//! that carries it.
//! Where a sensor stands is the transforms', and the answer carries no pose.
//! The node works the answer out once, when it starts, and gives it for as
//! long as it runs.

use std::sync::Arc;

use control_core::serving;
use peppygen::NodeRunner;
use peppygen::exposed_services::sensor_mounts::describe_sensors;
use peppylib::runtime::CancellationToken;
use robot_description::{CameraFrame, RobotDescription};

/// The contract a camera that gives color alone streams on.
const RGB_CAMERA: &str = "rgb_camera";

/// The contract a camera that also measures depth streams on.
const RGBD_CAMERA: &str = "rgbd_camera";

/// The contract a 2-D lidar streams on.
const LIDAR_2D: &str = "lidar_2d";

/// The carrier sensor_mounts names for a sensor that no arm carries.
const NO_ARM: &str = "";

/// One sensor of the robot, as describe_sensors lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sensor {
    /// The name the sensor runs under, its sensor's name in the model.
    pub name: String,
    /// The contract the sensor streams on, which defines its frame.
    pub kind: &'static str,
    /// The sensor's frame: a link of the URDF.
    pub frame: String,
    /// The arm that carries the sensor ([`RobotDescription::arm_carrying`]),
    /// or [`NO_ARM`].
    pub carried_by: String,
}

/// The answer of describe_sensors for one robot, fixed for as long as the
/// node runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SensorMounts {
    /// The robot frame: the root link of the URDF.
    pub robot_frame: String,
    pub sensors: Vec<Sensor>,
}

impl SensorMounts {
    /// The sensors of `description`: each camera of its URDF, then each 2-D
    /// lidar, each in the URDF's order.
    pub fn of(description: &RobotDescription) -> Self {
        let cameras = description.cameras().iter().map(|camera| Sensor {
            name: camera.name.clone(),
            kind: kind_of(camera),
            frame: camera.frame(),
            carried_by: carrier_of(description, &camera.parent_link).to_owned(),
        });
        let lidars = description.lidars().iter().map(|lidar| Sensor {
            name: lidar.name.clone(),
            kind: LIDAR_2D,
            frame: lidar.frame.clone(),
            carried_by: carrier_of(description, &lidar.frame).to_owned(),
        });
        Self {
            robot_frame: description.root_link().to_owned(),
            sensors: cameras.chain(lidars).collect(),
        }
    }

    /// The answer of describe_sensors: the arrays of the sensor_mounts
    /// contract, index-aligned by sensor.
    fn response(&self) -> describe_sensors::Response {
        describe_sensors::Response::new(
            self.robot_frame.clone(),
            self.sensors.iter().map(|s| s.name.clone()).collect(),
            self.sensors.iter().map(|s| s.kind.to_owned()).collect(),
            self.sensors.iter().map(|s| s.frame.clone()).collect(),
            self.sensors.iter().map(|s| s.carried_by.clone()).collect(),
        )
    }
}

/// The contract `camera` streams on: rgbd_camera for a camera the URDF gives
/// a depth sensor, rgb_camera for one that gives color alone.
fn kind_of(camera: &CameraFrame) -> &'static str {
    match camera.depth_range {
        Some(_) => RGBD_CAMERA,
        None => RGB_CAMERA,
    }
}

/// The name of the arm that carries the sensor's link `link`, or [`NO_ARM`].
fn carrier_of<'a>(description: &'a RobotDescription, link: &str) -> &'a str {
    description
        .arm_carrying(link)
        .expect("a sensor's link is a link of the URDF")
        .map_or(NO_ARM, |arm| arm.name.as_str())
}

/// Starts serving describe_sensors, for as long as the node runs.
pub fn serve(mounts: SensorMounts, runner: Arc<NodeRunner>) {
    let token = runner.cancellation_token().clone();
    tokio::spawn(run(mounts, runner, token));
}

async fn run(mounts: SensorMounts, runner: Arc<NodeRunner>, token: CancellationToken) {
    tracing::info!(
        "describe_sensors service started: robot frame {}, {} sensors",
        mounts.robot_frame,
        mounts.sensors.len()
    );
    let response = mounts.response();
    serving::serve("describe_sensors", token.cancelled(), || {
        describe_sensors::handle_requests(&runner, |_req| Ok(response.clone()))
    })
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_answer_lists_each_sensor_at_one_index_in_every_array() {
        let sensor = |name: &str, kind, carried_by: &str| Sensor {
            name: name.to_owned(),
            kind,
            frame: format!("{name}_optical_frame"),
            carried_by: carried_by.to_owned(),
        };
        let answer = SensorMounts {
            robot_frame: "base_link".to_owned(),
            sensors: vec![
                sensor("wrist", RGB_CAMERA, "left_arm"),
                sensor("chest", RGBD_CAMERA, NO_ARM),
            ],
        }
        .response();
        assert_eq!(answer.robot_frame, "base_link");
        assert_eq!(answer.sensor_names, ["wrist", "chest"]);
        assert_eq!(answer.sensor_kinds, [RGB_CAMERA, RGBD_CAMERA]);
        assert_eq!(
            answer.sensor_frames,
            ["wrist_optical_frame", "chest_optical_frame"]
        );
        assert_eq!(answer.carried_by, ["left_arm", ""]);
    }

    #[test]
    fn each_lidar_follows_the_cameras_on_its_own_frame_with_the_arm_that_carries_it() {
        let model = robot_description::model("openarm_v2").expect("a bundled model");
        let lidar = |name: &str, parent: &str| {
            format!(
                r#"  <link name="{name}_frame"/>
  <joint name="{name}_joint" type="fixed"><parent link="{parent}"/><child link="{name}_frame"/></joint>
  <gazebo reference="{name}_frame">
    <sensor name="{name}" type="gpu_lidar"><lidar><range><min>0.05</min><max>40</max></range></lidar></sensor>
  </gazebo>
"#
            )
        };
        let lidars = lidar("chest_lidar", "openarm_body_link0")
            + &lidar("wrist_lidar", "openarm_left_ee_base_link");
        let urdf = model
            .urdf()
            .replacen("</robot>", &format!("{lidars}</robot>"), 1);
        let description =
            RobotDescription::parse(&urdf, model.srdf()).expect("a URDF with two lidars");
        let mounts = SensorMounts::of(&description);
        let lidars: Vec<_> = mounts.sensors[3..]
            .iter()
            .map(|s| {
                (
                    s.name.as_str(),
                    s.kind,
                    s.frame.as_str(),
                    s.carried_by.as_str(),
                )
            })
            .collect();
        assert_eq!(
            lidars,
            [
                ("chest_lidar", LIDAR_2D, "chest_lidar_frame", NO_ARM),
                ("wrist_lidar", LIDAR_2D, "wrist_lidar_frame", "left_arm"),
            ]
        );
        assert_eq!(
            mounts.sensors[..3],
            SensorMounts::of(&model.description()).sensors,
            "the cameras come first, as the model without lidars lists them"
        );
    }
}
