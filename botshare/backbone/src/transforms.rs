//! The transforms the base measures, as it publishes them on transforms, each
//! set in a message of its own: with each base_states, the transform odom to
//! base_footprint where the odometry places the robot frame, under the
//! state's timestamp and started at its odometry_start_timestamp; and at each
//! wheel measurement the odometry takes, the transforms of the two wheel
//! joints at their measured angles, from the model's URDF, under the
//! measurement's timestamp and started when the node started. The model's
//! fixed joints are the robot initializer's.

use std::collections::HashMap;
use std::time::SystemTime;

use control_core::planar_pose::PlanarPose;
use frame_tree::arithmetic::quaternion_of_yaw;
use frame_tree::{FrameName, FrameTransform, MessageError, MovingMessage};
use peppygen::NodeRunner;
use peppygen::emitted_topics::transforms::transforms;
use peppylib::{Payload, TopicPublisher};
use robot_description::{LinkTransform, LinkTransformError, RobotDescription};

/// The odometry frame, as base_state names it on transforms.
pub const ODOMETRY_FRAME: &str = "odom";

/// The robot frame of a mobile base, the root link of its URDF, as
/// transforms names it.
pub const ROBOT_FRAME: &str = "base_footprint";

/// The odometry frame, as the robot's tree names it.
pub fn odometry_frame() -> FrameName {
    FrameName::parse(ODOMETRY_FRAME).expect("odom is a frame name")
}

/// The robot frame, as the robot's tree names it.
pub fn robot_frame() -> FrameName {
    FrameName::parse(ROBOT_FRAME).expect("base_footprint is a frame name")
}

/// Why a set of the base's transforms makes no message.
#[derive(Debug, thiserror::Error)]
pub enum TransformsError {
    #[error(transparent)]
    Transform(#[from] LinkTransformError),
    #[error(transparent)]
    Message(#[from] MessageError),
}

/// The two wheel joints of a model, [left, right], with the description
/// that places their links.
#[derive(Debug)]
pub struct JointTransforms {
    description: RobotDescription,
    joints: [String; 2],
    /// When the node started, on the wall clock: the start of every
    /// transform of the wheel joints.
    start: SystemTime,
}

impl JointTransforms {
    /// The wheel joints `joints`, [left, right], of the model `description`
    /// describes, every transform of them started at `start`.
    pub fn new(description: RobotDescription, joints: [String; 2], start: SystemTime) -> Self {
        Self {
            description,
            joints,
            start,
        }
    }

    /// The message of the wheel joints' transforms at `angles` (rad), [left,
    /// right], measured at `timestamp`: each wheel's link at its joint's
    /// origin, turned about its axle by its angle, in the URDF's order of
    /// joints.
    pub fn message_at(
        &self,
        timestamp: SystemTime,
        angles: [f64; 2],
    ) -> Result<MovingMessage, TransformsError> {
        let positions: HashMap<&str, f64> =
            self.joints.iter().map(String::as_str).zip(angles).collect();
        let transforms = self
            .description
            .transforms_at(&positions)
            .iter()
            .map(LinkTransform::frame_transform)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MovingMessage::from_transforms(
            timestamp,
            &transforms,
            self.start,
        )?)
    }
}

/// `pose` in space, as the messages of the base carry a pose: its position
/// on the floor at height 0 and its heading's unit quaternion, [x, y, z, w],
/// as tf2's `setRPY(0, 0, yaw)` makes it, the quaternion diff_drive_controller
/// publishes.
pub fn position_and_orientation(pose: PlanarPose) -> ([f64; 3], [f64; 4]) {
    ([pose.x(), pose.y(), 0.0], quaternion_of_yaw(pose.yaw()))
}

/// The message of the transform odom to base_footprint at `pose`, measured at
/// `timestamp` and started at `odometry_start`: the pose on the floor, its
/// heading a turn about +z ([`position_and_orientation`]).
pub fn odometry_message(
    timestamp: SystemTime,
    pose: PlanarPose,
    odometry_start: SystemTime,
) -> Result<MovingMessage, TransformsError> {
    let (position, orientation) = position_and_orientation(pose);
    let transform = FrameTransform::parse(ODOMETRY_FRAME, ROBOT_FRAME, position, orientation)
        .expect("a planar pose is finite, its quaternion a unit one");
    Ok(MovingMessage::from_transforms(
        timestamp,
        &[transform],
        odometry_start,
    )?)
}

pub async fn declare_publisher(runner: &NodeRunner) -> peppygen::Result<TopicPublisher> {
    transforms::declare_publisher(runner).await
}

/// `message` as the transforms topic carries it.
pub fn payload(message: MovingMessage) -> peppygen::Result<Payload> {
    let MovingMessage {
        timestamp,
        transforms: records,
        start_timestamp,
    } = message;
    let records = records
        .into_iter()
        .map(|t| transforms::MessageTransformsItem {
            frame: t.frame,
            child_frame: t.child_frame,
            position: t.position,
            orientation: t.orientation,
        })
        .collect();
    transforms::build_message(timestamp, records, start_timestamp)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    const MEASURED: SystemTime = SystemTime::UNIX_EPOCH;

    fn started() -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000)
    }

    fn botshare_wheels() -> JointTransforms {
        let description = robot_description::model("botshare_base")
            .expect("the bundled base")
            .description();
        JointTransforms::new(
            description,
            [
                "left_wheel_joint".to_owned(),
                "right_wheel_joint".to_owned(),
            ],
            started(),
        )
    }

    // The odometry's pose becomes the transform odom to base_footprint: its
    // position on the floor, its yaw a turn about +z, under the state's
    // timestamp and the odometry's start.
    #[test]
    fn the_odometry_places_base_footprint_in_odom() {
        let pose = PlanarPose::new(1.5, -0.5, std::f64::consts::FRAC_PI_2).expect("a pose");
        let message = odometry_message(MEASURED, pose, started()).expect("a message");
        let [transform] = message.transforms.as_slice() else {
            panic!("one transform: {:?}", message.transforms);
        };
        assert_eq!(
            (transform.frame.as_str(), transform.child_frame.as_str()),
            ("odom", "base_footprint")
        );
        assert_eq!(transform.position, [1.5, -0.5, 0.0]);
        let half = std::f64::consts::FRAC_1_SQRT_2;
        for (actual, expected) in transform.orientation.iter().zip([0.0, 0.0, half, half]) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "{:?}",
                transform.orientation
            );
        }
        assert_eq!(
            (message.timestamp, message.start_timestamp),
            (MEASURED, started())
        );
    }

    // A pose's orientation is setRPY's quaternion of its heading, its sign
    // included: at -2.5 rad, [0, 0, -0.949, 0.315].
    #[test]
    fn a_pose_carries_the_setrpy_quaternion_of_its_heading() {
        let pose = PlanarPose::new(1.0, -2.0, -2.5).expect("a pose");
        let (position, orientation) = position_and_orientation(pose);
        assert_eq!(position, [1.0, -2.0, 0.0]);
        assert_eq!(orientation, quaternion_of_yaw(-2.5));
        for (actual, expected) in orientation.iter().zip([0.0, 0.0, -0.949, 0.315]) {
            assert!((actual - expected).abs() < 5e-4, "{orientation:?}");
        }
    }

    // The wheel joints' transforms, left then right as the URDF orders them,
    // each link on base_link at its joint's origin, turned about +y by its
    // angle, under the node's start.
    #[test]
    fn the_wheel_angles_turn_each_wheel_link_about_its_axle() {
        let message = botshare_wheels()
            .message_at(MEASURED, [std::f64::consts::PI, 0.0])
            .expect("a message");
        let [left, right] = message.transforms.as_slice() else {
            panic!("two transforms: {:?}", message.transforms);
        };
        assert_eq!(
            [&left.child_frame, &right.child_frame],
            ["left_wheel_link", "right_wheel_link"]
        );
        assert_eq!([&left.frame, &right.frame], ["base_link", "base_link"]);
        for (transform, expected) in [(left, [0.0, 0.2, 0.0]), (right, [0.0, -0.2, 0.0])] {
            for (actual, expected) in transform.position.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-9, "{:?}", transform.position);
            }
        }
        // A half turn about +y, and no turn: [0, 1, 0, 0] (or its negative)
        // and [0, 0, 0, 1].
        let [x, y, z, w] = left.orientation;
        assert!(y.abs() > 1.0 - 1e-9 && x.abs() + z.abs() + w.abs() < 1e-9);
        assert_eq!(right.orientation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(message.start_timestamp, started());
    }
}
