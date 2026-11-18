//! The transforms this backbone publishes on transforms: at each limb_states
//! snapshot, under the snapshot's timestamp, the transform of every arm joint
//! at its measured position and of every finger joint at the position its
//! gripper's measured opening puts it, from the robot's model, as one
//! frame_tree [`MovingMessage`]. Composed from the robot frame with the
//! transforms of the fixed joints, which the robot's initializer publishes,
//! they place each arm's grasp link where the snapshot places its grasp
//! point. Every transform starts at the bringup a process runs once, and
//! never starts over while the process runs.

use std::collections::HashMap;
use std::time::SystemTime;

use frame_tree::{MessageError, MovingMessage};
use robot_description::{LinkTransform, LinkTransformError, PerArm, RobotDescription};
use srs_model::JointVec;

use crate::robot::ARMS;

/// Why the transforms of a snapshot make no message.
#[derive(Debug, thiserror::Error)]
pub enum TransformsError {
    #[error(transparent)]
    Transform(#[from] LinkTransformError),

    #[error(transparent)]
    Message(#[from] MessageError),
}

/// The transforms of the joints this backbone measures, from its robot's
/// model.
pub struct JointTransforms {
    description: &'static RobotDescription,
    /// When this process brought the backbone up, on the wall clock: the
    /// start of every transform it publishes.
    start: SystemTime,
}

impl JointTransforms {
    /// The transforms of the robot `description` models, every one started at
    /// `start`.
    pub fn new(description: &'static RobotDescription, start: SystemTime) -> Self {
        Self { description, start }
    }

    /// The message of the transforms measured at `timestamp`: of every arm
    /// joint at `joints` and of every finger joint at the opening `openings`
    /// gives its gripper, in the URDF's order of joints. A joint whose
    /// transform the transforms contract does not take, such as one at a
    /// position that is no number, makes no message, and the error names it.
    pub fn message_at(
        &self,
        timestamp: SystemTime,
        joints: &PerArm<JointVec, ARMS>,
        openings: &PerArm<f64, ARMS>,
    ) -> Result<MovingMessage, TransformsError> {
        let transforms = self
            .at(joints, openings)
            .iter()
            .map(LinkTransform::frame_transform)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MovingMessage::from_transforms(
            timestamp,
            &transforms,
            self.start,
        )?)
    }

    /// The transform of every arm joint at `joints` and of every finger joint
    /// at the opening `openings` gives its gripper, in the URDF's order of
    /// joints.
    fn at(
        &self,
        joints: &PerArm<JointVec, ARMS>,
        openings: &PerArm<f64, ARMS>,
    ) -> Vec<LinkTransform<'static>> {
        let arms = self.description.arms();
        let positions: HashMap<&str, f64> = PerArm::<(), ARMS>::ids()
            .flat_map(|arm| {
                let described = &arms[arm.place()];
                described
                    .joints
                    .iter()
                    .map(String::as_str)
                    .zip(joints[arm])
                    .chain(described.gripper.finger_positions_at(openings[arm]))
            })
            .collect();
        self.description.transforms_at(&positions)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use robot_description::ArmId;
    use srs_model::nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};

    use super::*;
    use crate::robot::LIMBS;
    use crate::test_support::{V1, V2, bundled_arm, description};

    /// When the transforms of the tests start: a time of its own, apart from
    /// [`MEASURED`].
    fn start() -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000)
    }

    /// When the snapshots of the tests are measured.
    const MEASURED: SystemTime = SystemTime::UNIX_EPOCH;

    /// The place of the child frame of transform `index` of `message` in its
    /// parent.
    fn pose_in(message: &MovingMessage, index: usize) -> Isometry3<f64> {
        let [x, y, z] = std::array::from_fn(|axis| message.positions[3 * index + axis]);
        let [i, j, k, w] = std::array::from_fn(|axis| message.orientations[4 * index + axis]);
        Isometry3::from_parts(
            Translation3::new(x, y, z),
            UnitQuaternion::from_quaternion(Quaternion::new(w, i, j, k)),
        )
    }

    /// Where `link` stands in the robot frame: the transforms of `message`
    /// and of the model's fixed joints composed from the root down to it.
    fn place_of(
        link: &str,
        message: &MovingMessage,
        description: &RobotDescription,
    ) -> Isometry3<f64> {
        let fixed = description.fixed_transforms();
        let mut place = Isometry3::identity();
        let mut at = link.to_owned();
        loop {
            if let Some(index) = message.child_frames.iter().position(|child| *child == at) {
                place = pose_in(message, index) * place;
                at = message.parent_frames[index].clone();
                continue;
            }
            let Some(transform) = fixed.iter().find(|transform| transform.child == at) else {
                return place;
            };
            place = transform.pose * place;
            at = transform.parent.to_owned();
        }
    }

    /// A drawn posture of `arm` of `model`, within its joints' limits.
    fn drawn_posture(tc: &hegel::TestCase, model: &str, arm: ArmId) -> JointVec {
        let limits = bundled_arm(model, arm).limits();
        std::array::from_fn(|i| {
            tc.draw(
                hegel::generators::floats()
                    .min_value(limits[i].lo)
                    .max_value(limits[i].hi),
            )
        })
    }

    // Down each arm, the message at drawn joints and openings places the
    // grasp link where the arm model the backbone reports limb_states from
    // places the grasp point.
    #[hegel::test(test_cases = 40)]
    fn composed_down_an_arm_the_transforms_place_its_grasp_point(tc: hegel::TestCase) {
        let model = if tc.draw(hegel::generators::booleans()) {
            V2
        } else {
            V1
        };
        let joints = PerArm::from_fn(|arm| drawn_posture(&tc, model, arm));
        let openings =
            PerArm::from_fn(|_| tc.draw(hegel::generators::floats().min_value(0.0).max_value(1.0)));
        let description = description(model);
        let message = JointTransforms::new(description, start())
            .message_at(MEASURED, &joints, &openings)
            .expect("the transforms of a measurement within the joints' limits");
        for arm in PerArm::<(), ARMS>::ids() {
            let built = bundled_arm(model, arm);
            let reported = built.world_pose(&built.at(&joints[arm]).ee_pose());
            let grasp_link = &description.arms()[arm.place()].grasp_link;
            let composed = place_of(grasp_link, &message, description);
            let miss = (composed.translation.vector - reported.translation.vector).norm();
            let turn = composed.rotation.angle_to(&reported.rotation);
            assert!(
                miss <= 1e-9 && turn <= 1e-9,
                "{model} {}: {miss} m, {turn} rad",
                LIMBS.arm_name(arm)
            );
        }
    }

    #[test]
    fn the_message_carries_the_measurement_time_and_the_start() {
        let measured = SystemTime::UNIX_EPOCH + Duration::from_nanos(1_763_400_000_123_456_789);
        let message = JointTransforms::new(description(V2), start())
            .message_at(
                measured,
                &PerArm::from_fn(|_| [0.0; 7]),
                &PerArm::from_fn(|_| 0.5),
            )
            .expect("a message");
        assert_eq!(message.timestamp, measured);
        assert_eq!(message.start_timestamp, start());
    }

    #[test]
    fn every_arm_and_finger_joint_has_one_transform_and_no_fixed_joint_has_one() {
        for model in [V1, V2] {
            let description = description(model);
            let message = JointTransforms::new(description, start())
                .message_at(
                    MEASURED,
                    &PerArm::from_fn(|_| [0.0; 7]),
                    &PerArm::from_fn(|_| 0.5),
                )
                .expect("a message");
            let mut children: Vec<&str> = message.child_frames.iter().map(String::as_str).collect();
            let mut measured: Vec<&str> = description
                .arms()
                .iter()
                .flat_map(|arm| {
                    arm.joints
                        .iter()
                        .chain(arm.gripper.fingers.iter().map(|finger| &finger.joint))
                })
                .map(|joint| {
                    description
                        .robot()
                        .joints
                        .iter()
                        .find(|urdf| urdf.name == *joint)
                        .expect("a joint of the URDF")
                        .child
                        .link
                        .as_str()
                })
                .collect();
            children.sort_unstable();
            measured.sort_unstable();
            assert_eq!(children, measured, "{model}");
            let fixed = description.fixed_transforms();
            assert!(
                fixed.iter().all(|f| !children.contains(&f.child)),
                "{model}"
            );
        }
    }

    // Each gripper's opening moves its own fingers alone.
    #[test]
    fn a_gripper_opening_moves_the_fingers_of_that_gripper_alone() {
        let description = description(V2);
        let transforms = JointTransforms::new(description, start());
        let joints = PerArm::from_fn(|_| [0.0; 7]);
        let shut = transforms
            .message_at(MEASURED, &joints, &PerArm::new([0.0, 1.0]))
            .expect("a message");
        let opened = transforms
            .message_at(MEASURED, &joints, &PerArm::new([1.0, 1.0]))
            .expect("a message");
        let pose_of = |message: &MovingMessage, link: &str| {
            let index = message
                .child_frames
                .iter()
                .position(|child| child == link)
                .expect("a finger's transform");
            pose_in(message, index)
        };
        for arm in PerArm::<(), ARMS>::ids() {
            for finger in &description.arms()[arm.place()].gripper.fingers {
                let moved = pose_of(&shut, &finger.link) != pose_of(&opened, &finger.link);
                assert_eq!(moved, arm.place() == 0, "{}", finger.joint);
            }
        }
    }

    // A joint at a position that is no number makes no message, and the
    // error names the joint's links.
    #[test]
    fn a_joint_at_no_number_makes_no_message_and_is_named() {
        let description = description(V2);
        let joints = PerArm::new([[f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0; 7]]);
        let refused = JointTransforms::new(description, start())
            .message_at(MEASURED, &joints, &PerArm::from_fn(|_| 0.5))
            .expect_err("a joint at NaN");
        let first = description
            .robot()
            .joints
            .iter()
            .find(|joint| joint.name == description.arms()[0].joints[0])
            .expect("the left arm's first joint");
        let TransformsError::Transform(LinkTransformError { parent, child, .. }) = refused else {
            panic!("the refusal names a transform: {refused}");
        };
        assert_eq!(
            (parent.as_str(), child.as_str()),
            (first.parent.link.as_str(), first.child.link.as_str())
        );
    }
}
