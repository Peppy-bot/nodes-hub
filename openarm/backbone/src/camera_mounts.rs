//! The camera_mounts contract: where the robot's design carries each camera,
//! in the robot frame, now. The description gives each camera's optical frame
//! (+Z along the view, +Y down, as camera_geometry defines it) as a link of
//! the URDF fixed to its parent link; this module places each one, once at
//! bringup, in the frame that carries it: the robot frame for a camera fixed
//! to the base, the grasp point's frame for a camera an arm carries.
//! A request then composes the carried ones with the grasp poses the
//! coordinator measured last, the ones limb_states publishes, and answers
//! under that measurement's stamp. Served from bringup: until the coordinator
//! has measured the arms the caller gets an error reply with
//! `precondition_failed`. The workspace tools read the
//! same placements, with the depths the description gives each depth camera,
//! through [`CameraMounts::design_cameras`].

use std::sync::Arc;
use std::time::SystemTime;

use control_core::serving;
use peppygen::NodeRunner;
use peppygen::exposed_services::camera_mounts::get_camera_poses;
use peppylib::runtime::CancellationToken;
use robot_description::{ArmId, CameraFrame, DepthRange, PerArm, RobotDescription};
use srs_model::chain_kinematics::Tree;
use srs_model::nalgebra::{Isometry3, Translation3, UnitQuaternion};
use tokio::sync::watch;
use tracing::info;

use crate::robot::{ARMS, LIMBS};
use crate::types::world_pose_arrays;

/// The message of the error reply while the coordinator has measured
/// nothing yet.
const NOT_MEASURED_YET: &str = "the robot has not measured its joints yet";

/// The grasp poses the coordinator measured last, in the robot frame, and
/// when: what the carried cameras' poses are composed with.
#[derive(Clone, Debug)]
pub struct MeasuredGrasps {
    pub timestamp: SystemTime,
    pub poses: PerArm<Isometry3<f64>, ARMS>,
}

/// What carries a camera.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carrier {
    /// The robot's base: the camera has one pose for the life of the robot.
    Base,
    /// One arm's grasp point: the camera moves with that arm.
    Arm(ArmId),
}

/// One camera, resolved: its optical frame in its carrier's frame, and the
/// depths it measures.
#[derive(Clone, Debug)]
struct ResolvedMount {
    name: &'static str,
    carrier: Carrier,
    pose: Isometry3<f64>,
    depth_range: Option<DepthRange>,
}

/// One camera as the design fixes it, whatever the arms do: what the
/// workspace tools read to find the perception camera and place its view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DesignCamera {
    pub name: &'static str,
    /// The depths the camera measures, as the description gives them; none
    /// for a camera that gives color alone.
    pub depth_range: Option<DepthRange>,
    /// The camera's optical frame in the robot frame when the base carries
    /// it; none when an arm carries it, so that it moves with that arm.
    pub fixed_pose: Option<Isometry3<f64>>,
}

/// The pose of one camera at one measurement, as the service reports it.
#[derive(Clone, Debug, PartialEq)]
struct CameraPose {
    name: &'static str,
    /// The arm that carries the camera, or "" for one fixed to the base.
    carried_by: &'static str,
    /// The camera's optical frame in the robot frame.
    pose: Isometry3<f64>,
}

/// A mount the robot's description carries that this backbone cannot place.
#[derive(Debug, thiserror::Error)]
pub enum CameraMountError {
    #[error(
        "camera {camera} is fixed to {link}, which joint {joint} moves: only the base and \
         the links the grasp points hang off carry a camera"
    )]
    MovingLink {
        camera: String,
        link: String,
        joint: String,
    },
}

/// The robot's cameras, each placed in the frame that carries it.
#[derive(Clone, Debug)]
pub struct CameraMounts {
    mounts: Vec<ResolvedMount>,
}

impl CameraMounts {
    /// Resolves every mount of `description` against its URDF. A camera
    /// whose link the grasp point of one arm hangs off is carried by that
    /// arm, in the grasp point's frame; a camera whose link is fixed to the
    /// root is carried by the base, in the robot frame; any other mount is
    /// refused.
    pub fn resolve(description: &'static RobotDescription) -> Result<Self, CameraMountError> {
        let tree = Tree::from_robot(description.robot()).expect("a description is one tree");
        let mounts = description
            .cameras()
            .iter()
            .map(|frame| resolve_mount(&tree, description, frame))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { mounts })
    }

    /// The pose of every camera in the robot frame at the grasp poses
    /// `grasps`, in the description's order.
    fn poses(&self, grasps: &PerArm<Isometry3<f64>, ARMS>) -> Vec<CameraPose> {
        self.mounts
            .iter()
            .map(|mount| {
                let (carried_by, pose) = match mount.carrier {
                    Carrier::Base => ("", mount.pose),
                    Carrier::Arm(arm) => (LIMBS.arm_name(arm), grasps[arm] * mount.pose),
                };
                CameraPose {
                    name: mount.name,
                    carried_by,
                    pose,
                }
            })
            .collect()
    }

    /// Every camera as the design fixes it, in the description's order.
    pub fn design_cameras(&self) -> Vec<DesignCamera> {
        self.mounts
            .iter()
            .map(|mount| DesignCamera {
                name: mount.name,
                depth_range: mount.depth_range,
                fixed_pose: match mount.carrier {
                    Carrier::Base => Some(mount.pose),
                    Carrier::Arm(_) => None,
                },
            })
            .collect()
    }
}

/// The optical frame `frame` in the link it is fixed to: its URDF origin, the
/// roll, pitch and yaw about the link's fixed axes.
fn frame_pose(frame: &CameraFrame) -> Isometry3<f64> {
    let [x, y, z] = frame.xyz;
    let [roll, pitch, yaw] = frame.rpy;
    Isometry3::from_parts(
        Translation3::new(x, y, z),
        UnitQuaternion::from_euler_angles(roll, pitch, yaw),
    )
}

/// One camera of `description` in its carrier's frame: its optical frame,
/// composed below the link it is fixed to as that link stands in its
/// carrier's frame.
fn resolve_mount(
    tree: &Tree,
    description: &RobotDescription,
    frame: &'static CameraFrame,
) -> Result<ResolvedMount, CameraMountError> {
    let name = frame.name.as_str();
    let depth_range = frame.depth_range;
    let in_link = frame_pose(frame);
    let link = tree
        .link_index(&frame.parent_link)
        .expect("a camera frame's parent link is a link of the URDF its tree holds");
    for arm in PerArm::<(), ARMS>::ids() {
        // The grasp point hangs off the camera's link over fixed joints: the
        // camera in the grasp point's frame is the tool transform undone,
        // then the mount.
        let grasp_link = &description.arms()[arm.place()].grasp_link;
        if let Some(tool) = tree.fixed_path_to(link, grasp_link) {
            return Ok(ResolvedMount {
                name,
                carrier: Carrier::Arm(arm),
                pose: tool.inverse() * in_link,
                depth_range,
            });
        }
    }
    let link_in_root = if link == tree.root() {
        Isometry3::identity()
    } else {
        tree.fixed_path_to(tree.root(), &frame.parent_link)
            .ok_or_else(|| CameraMountError::MovingLink {
                camera: name.to_string(),
                link: frame.parent_link.clone(),
                joint: moving_joint_above(tree, link),
            })?
    };
    Ok(ResolvedMount {
        name,
        carrier: Carrier::Base,
        pose: link_in_root * in_link,
        depth_range,
    })
}

/// The name of the first joint between the root and `link` that moves.
fn moving_joint_above(tree: &Tree, link: usize) -> String {
    tree.path_to(link)
        .into_iter()
        .map(|joint| tree.joint(joint))
        .find(|joint| joint.kind.is_movable())
        .map(|joint| joint.name.clone())
        .expect("a link no fixed path reaches hangs below a moving joint")
}

/// The answer to one request: every camera's pose at `measured`, or
/// `precondition_failed` while nothing has been measured.
fn answer(
    mounts: &CameraMounts,
    measured: Option<&MeasuredGrasps>,
) -> Result<get_camera_poses::Response, get_camera_poses::Failure> {
    let Some(measured) = measured else {
        return Err(get_camera_poses::Failure::new(
            get_camera_poses::FailureKind::PreconditionFailed,
            NOT_MEASURED_YET,
        ));
    };
    let poses = mounts.poses(&measured.poses);
    let mut positions = Vec::with_capacity(poses.len() * 3);
    let mut orientations = Vec::with_capacity(poses.len() * 4);
    for camera in &poses {
        let (position, orientation) = world_pose_arrays(&camera.pose);
        positions.extend(position);
        orientations.extend(orientation);
    }
    Ok(get_camera_poses::Response::new(
        measured.timestamp,
        poses.iter().map(|camera| camera.name.to_string()).collect(),
        positions,
        orientations,
        poses
            .iter()
            .map(|camera| camera.carried_by.to_string())
            .collect(),
    ))
}

/// Serve `get_camera_poses` for the life of the node, from the grasp poses
/// the coordinator measured last.
pub async fn serve(
    runner: Arc<NodeRunner>,
    token: CancellationToken,
    mounts: Arc<CameraMounts>,
    measured: watch::Receiver<Option<MeasuredGrasps>>,
) {
    info!("camera mounts: {} cameras", mounts.mounts.len());
    serving::serve("get_camera_poses", token.cancelled(), || {
        get_camera_poses::handle_requests(&runner, |_request| {
            answer(&mounts, measured.borrow().as_ref())
        })
    })
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_description::states;
    use srs_model::JointVec;
    use srs_model::nalgebra::Vector3;

    use crate::test_support::{LEFT, RIGHT, V1, V2, bundled_arm, description, postures};

    fn v2_mounts() -> CameraMounts {
        CameraMounts::resolve(description(V2)).expect("the v2 mounts resolve")
    }

    /// The camera `name` of [`V2`]'s description.
    fn v2_camera(name: &str) -> &'static CameraFrame {
        description(V2)
            .cameras()
            .iter()
            .find(|camera| camera.name == name)
            .expect("a camera of the description")
    }

    /// The names of the cameras of [`V2`]'s description, in its order.
    fn v2_camera_names() -> Vec<&'static str> {
        description(V2)
            .cameras()
            .iter()
            .map(|camera| camera.name.as_str())
            .collect()
    }

    /// The grasp pose of `arm` at `q`, in the robot frame.
    fn grasp(arm: ArmId, q: &JointVec) -> Isometry3<f64> {
        let arm = bundled_arm(V2, arm);
        arm.world_pose(&arm.at(q).ee_pose())
    }

    /// The camera `carried_by` carries: the name of an arm, or "" for the base.
    fn carried_by<'a>(poses: &'a [CameraPose], carrier: &str) -> &'a CameraPose {
        poses
            .iter()
            .find(|camera| camera.carried_by == carrier)
            .unwrap_or_else(|| panic!("a camera carried by {carrier:?}"))
    }

    /// The pose of `link` in the robot frame, composed by hand from the URDF's
    /// joint origins over the fixed joints above it.
    fn fixed_link_in_root(link: &str) -> Isometry3<f64> {
        let robot = description(V2).robot();
        let Some(joint) = robot.joints.iter().find(|joint| joint.child.link == link) else {
            return Isometry3::identity();
        };
        assert_eq!(
            joint.joint_type,
            robot_description::urdf_rs::JointType::Fixed,
            "{}",
            joint.name
        );
        let [x, y, z] = *joint.origin.xyz;
        let [roll, pitch, yaw] = *joint.origin.rpy;
        fixed_link_in_root(&joint.parent.link)
            * Isometry3::from_parts(
                Translation3::new(x, y, z),
                UnitQuaternion::from_euler_angles(roll, pitch, yaw),
            )
    }

    #[test]
    fn v1_carries_no_camera() {
        let mounts = CameraMounts::resolve(description(V1)).expect("resolves");
        let grasps = PerArm::from_fn(|_| Isometry3::identity());
        assert!(mounts.poses(&grasps).is_empty());
    }

    #[test]
    fn the_base_camera_is_its_optical_frame_in_the_robot_frame() {
        let mounts = v2_mounts();
        let grasps = PerArm::from_fn(|_| Isometry3::identity());
        let poses = mounts.poses(&grasps);
        assert_eq!(
            poses.iter().map(|c| c.name).collect::<Vec<_>>(),
            v2_camera_names()
        );
        let base = carried_by(&poses, "");
        let frame = v2_camera(base.name);
        let expected = fixed_link_in_root(&frame.parent_link) * frame_pose(frame);
        assert!(
            (base.pose.translation.vector - expected.translation.vector).norm() < 1e-9,
            "{:?} vs {:?}",
            base.pose.translation,
            expected.translation
        );
        assert!(base.pose.rotation.angle_to(&expected.rotation) < 1e-9);
        // It faces the robot's front and looks down.
        let view = base.pose.rotation * Vector3::z();
        assert!(view.x > 0.4 && view.z < -0.8, "{view}");
    }

    #[test]
    fn the_design_fixes_the_base_camera_where_it_is_reported_and_no_arm_camera() {
        let mounts = v2_mounts();
        let cameras = mounts.design_cameras();
        assert_eq!(
            cameras.iter().map(|c| c.name).collect::<Vec<_>>(),
            v2_camera_names()
        );
        let reported = mounts.poses(&PerArm::from_fn(|_| Isometry3::identity()));
        for camera in &cameras {
            let carrier = reported
                .iter()
                .find(|pose| pose.name == camera.name)
                .expect("reported");
            if carrier.carried_by.is_empty() {
                assert_eq!(camera.fixed_pose, Some(carrier.pose), "{}", camera.name);
                let described = v2_camera(camera.name).depth_range;
                assert_eq!(camera.depth_range, described, "{}", camera.name);
                assert!(camera.depth_range.is_some(), "{} gives depth", camera.name);
                continue;
            }
            assert_eq!(
                (camera.fixed_pose, camera.depth_range),
                (None, None),
                "{}",
                camera.name
            );
        }
        assert!(
            CameraMounts::resolve(description(V1))
                .expect("resolves")
                .design_cameras()
                .is_empty()
        );
    }

    #[test]
    fn an_arm_camera_moves_with_its_arm_and_matches_the_urdf_composed_by_hand() {
        let mounts = v2_mounts();
        let home = postures(V2, states::HOME);
        let bent: JointVec = [0.3, -0.8, 0.2, 1.2, -0.4, 0.5, 0.1];
        let at_home = mounts.poses(&PerArm::from_fn(|arm| grasp(arm, &home[arm])));
        let at_bent = mounts.poses(&PerArm::new([
            grasp(LEFT, &bent),
            grasp(RIGHT, &home[RIGHT]),
        ]));
        let left_home = carried_by(&at_home, LIMBS.arm_name(LEFT));
        let left_bent = carried_by(&at_bent, LIMBS.arm_name(LEFT));
        assert!(
            (left_home.pose.translation.vector - left_bent.pose.translation.vector).norm() > 0.1,
            "the left arm's camera moved with its arm"
        );
        assert_eq!(
            carried_by(&at_home, LIMBS.arm_name(RIGHT)).pose,
            carried_by(&at_bent, LIMBS.arm_name(RIGHT)).pose,
            "the right arm did not move, nor its camera"
        );

        // By hand: the tip link's pose from the chain, then the optical
        // frame in that link.
        let arm = bundled_arm(V2, LEFT);
        let tip_in_root = arm.world_pose(&arm.at(&bent).tip_pose());
        let frame = v2_camera(left_bent.name);
        let expected = tip_in_root * frame_pose(frame);
        assert!(
            (left_bent.pose.translation.vector - expected.translation.vector).norm() < 1e-9,
            "{:?} vs {:?}",
            left_bent.pose.translation,
            expected.translation
        );
        assert!(left_bent.pose.rotation.angle_to(&expected.rotation) < 1e-9);
    }

    #[test]
    fn the_answer_lists_every_camera_under_the_measurement_stamp_or_refuses() {
        let mounts = v2_mounts();
        let refused = answer(&mounts, None).expect_err("nothing is measured yet");
        assert_eq!(
            refused,
            get_camera_poses::Failure::new(
                get_camera_poses::FailureKind::PreconditionFailed,
                NOT_MEASURED_YET
            )
        );

        let home = postures(V2, states::HOME);
        let measured = MeasuredGrasps {
            timestamp: SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(7),
            poses: PerArm::from_fn(|arm| grasp(arm, &home[arm])),
        };
        let reply = answer(&mounts, Some(&measured)).expect("the arms are measured");
        assert_eq!(reply.timestamp, measured.timestamp);
        assert_eq!(reply.camera_names, v2_camera_names());
        let cameras = v2_camera_names().len();
        assert_eq!(reply.carried_by.len(), cameras);
        for arm in PerArm::<(), ARMS>::ids() {
            assert_eq!(
                reply
                    .carried_by
                    .iter()
                    .filter(|c| *c == LIMBS.arm_name(arm))
                    .count(),
                1,
                "{} carries one camera",
                LIMBS.arm_name(arm)
            );
        }
        assert_eq!(reply.positions.len(), 3 * cameras);
        assert_eq!(reply.orientations.len(), 4 * cameras);
        for quat in reply.orientations.chunks(4) {
            let norm = quat.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-9);
        }
        // The arms' cameras mirror across the robot's center line at home.
        let position_of = |arm: ArmId| {
            let k = reply
                .carried_by
                .iter()
                .position(|c| c == LIMBS.arm_name(arm))
                .expect("an arm's camera");
            [0, 1, 2].map(|axis| reply.positions[3 * k + axis])
        };
        let (left, right) = (position_of(LEFT), position_of(RIGHT));
        assert!((left[0] - right[0]).abs() < 1e-6);
        assert!((left[1] + right[1]).abs() < 1e-6);
        assert!((left[2] - right[2]).abs() < 1e-6);
    }
}
