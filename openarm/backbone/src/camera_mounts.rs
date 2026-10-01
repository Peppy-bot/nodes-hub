//! The camera_mounts contract: where the robot's design carries each camera,
//! in the robot frame, now. The description lists each camera's pose in the
//! frame of the URDF link it is fixed to, looking along its own -Z with +Y
//! up; this module turns each one, once at bringup, into the pose of the
//! camera's optical frame (+Z along the view, +Y down, as camera_geometry
//! defines it) in the frame that carries it: the robot frame for a camera
//! fixed to the base, the grasp point's frame for a camera an arm carries.
//! A request then composes the carried ones with the grasp poses the
//! coordinator measured last, the ones limb_states publishes, and answers
//! under that measurement's stamp. Served from bringup: until the coordinator
//! has measured the arms the answer says so.

use std::f64::consts::PI;
use std::sync::Arc;
use std::time::SystemTime;

use openarm_description::{CameraMount, HardwareVersion};
use peppygen::NodeRunner;
use peppygen::exposed_services::camera_mounts::get_camera_poses;
use peppylib::runtime::CancellationToken;
use srs_model::chain_kinematics::Tree;
use srs_model::nalgebra::{Isometry3, UnitQuaternion, Vector3};
use tokio::sync::watch;
use tracing::{error, info};

use crate::arm_pair::ArmPair;
use crate::types::{Side, pose_from_wire, world_pose_arrays};

/// What the service answers while the coordinator has measured nothing yet.
const NOT_MEASURED_YET: &str = "the robot has not measured its joints yet";

/// The grasp poses the coordinator measured last, in the robot frame, and
/// when: what the carried cameras' poses are composed with.
#[derive(Clone, Debug)]
pub struct MeasuredGrasps {
    pub timestamp: SystemTime,
    pub poses: ArmPair<Isometry3<f64>>,
}

/// What carries a camera.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carrier {
    /// The robot's base: the camera has one pose for the life of the robot.
    Base,
    /// One arm's grasp point: the camera moves with that arm.
    Arm(Side),
}

/// One camera, resolved: its optical frame in its carrier's frame.
#[derive(Clone, Debug)]
struct ResolvedMount {
    name: &'static str,
    carrier: Carrier,
    pose: Isometry3<f64>,
}

/// The pose of one camera at one measurement, as the service reports it.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraPose {
    pub name: &'static str,
    /// The arm that carries the camera, or "" for one fixed to the base.
    pub carried_by: &'static str,
    /// The camera's optical frame in the robot frame.
    pub pose: Isometry3<f64>,
}

/// A mount the robot's description carries that this backbone cannot place.
#[derive(Debug, thiserror::Error)]
pub enum CameraMountError {
    #[error("camera {camera} is fixed to {link}, which the URDF does not carry")]
    UnknownLink { camera: String, link: String },

    #[error(
        "camera {camera} is fixed to {link}, which joint {joint} moves: only the base and \
         the links the grasp points hang off carry a camera"
    )]
    MovingLink {
        camera: String,
        link: String,
        joint: String,
    },

    #[error("the URDF names no tool-centre-point joint for the {side} arm")]
    NoToolJoint { side: &'static str },
}

/// The robot's cameras, each placed in the frame that carries it.
#[derive(Clone, Debug)]
pub struct CameraMounts {
    mounts: Vec<ResolvedMount>,
}

impl CameraMounts {
    /// Resolves every mount of `version` against its URDF. A camera whose
    /// link the grasp point of one arm hangs off is carried by that arm, in
    /// the grasp point's frame; a camera whose link is fixed to the root is
    /// carried by the base, in the robot frame; any other mount is refused.
    pub fn resolve(version: HardwareVersion) -> Result<Self, CameraMountError> {
        let robot = urdf_rs::read_from_string(version.urdf()).expect("bundled URDF must parse");
        let tree = Tree::from_robot(&robot).expect("bundled URDF must be one tree");
        for side in [Side::Left, Side::Right] {
            if tree.link_index(version.tcp_link(side.model())).is_none() {
                return Err(CameraMountError::NoToolJoint { side: side.label() });
            }
        }
        let mounts = version
            .camera_mounts()
            .iter()
            .map(|mount| resolve_mount(&tree, version, mount))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { mounts })
    }

    /// The pose of every camera in the robot frame at the grasp poses
    /// `grasps`, in the description's order.
    pub fn poses(&self, grasps: &ArmPair<Isometry3<f64>>) -> Vec<CameraPose> {
        self.mounts
            .iter()
            .map(|mount| {
                let (carried_by, pose) = match mount.carrier {
                    Carrier::Base => ("", mount.pose),
                    Carrier::Arm(side) => {
                        (Side::ARM_NAMES[side.index()], grasps.get(side) * mount.pose)
                    }
                };
                CameraPose {
                    name: mount.name,
                    carried_by,
                    pose,
                }
            })
            .collect()
    }
}

/// The description's pose of `mount` in the link that carries it.
fn mount_pose(mount: &CameraMount) -> Isometry3<f64> {
    let [w, x, y, z] = mount.quat_wxyz;
    pose_from_wire(mount.position, [x, y, z, w]).expect("the description's mount pose is a pose")
}

/// One mount in its carrier's frame: the description's pose, turned half a
/// turn about X into the optical frame, composed below the link it is fixed
/// to as that link stands in its carrier's frame.
fn resolve_mount(
    tree: &Tree,
    version: HardwareVersion,
    mount: &CameraMount,
) -> Result<ResolvedMount, CameraMountError> {
    let in_link = mount_pose(mount) * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI);
    let Some(link) = tree.link_index(mount.parent_link) else {
        return Err(CameraMountError::UnknownLink {
            camera: mount.name.to_string(),
            link: mount.parent_link.to_string(),
        });
    };
    for side in [Side::Left, Side::Right] {
        // The grasp point hangs off the camera's link over fixed joints: the
        // camera in the grasp point's frame is the tool transform undone,
        // then the mount.
        if let Some(tool) = tree.fixed_path_to(link, version.tcp_link(side.model())) {
            return Ok(ResolvedMount {
                name: mount.name,
                carrier: Carrier::Arm(side),
                pose: tool.inverse() * in_link,
            });
        }
    }
    let link_in_root = if link == tree.root() {
        Isometry3::identity()
    } else {
        tree.fixed_path_to(tree.root(), mount.parent_link)
            .ok_or_else(|| CameraMountError::MovingLink {
                camera: mount.name.to_string(),
                link: mount.parent_link.to_string(),
                joint: moving_joint_above(tree, link),
            })?
    };
    Ok(ResolvedMount {
        name: mount.name,
        carrier: Carrier::Base,
        pose: link_in_root * in_link,
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

/// The answer to one request: every camera's pose at `measured`, or the
/// refusal while nothing has been measured.
fn answer(mounts: &CameraMounts, measured: Option<&MeasuredGrasps>) -> get_camera_poses::Response {
    let Some(measured) = measured else {
        return get_camera_poses::Response::new(
            false,
            NOT_MEASURED_YET.to_string(),
            SystemTime::UNIX_EPOCH,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
    };
    let poses = mounts.poses(&measured.poses);
    let mut positions = Vec::with_capacity(poses.len() * 3);
    let mut orientations = Vec::with_capacity(poses.len() * 4);
    for camera in &poses {
        let (position, orientation) = world_pose_arrays(&camera.pose);
        positions.extend(position);
        orientations.extend(orientation);
    }
    get_camera_poses::Response::new(
        true,
        format!("{} cameras", poses.len()),
        measured.timestamp,
        poses.iter().map(|camera| camera.name.to_string()).collect(),
        positions,
        orientations,
        poses
            .iter()
            .map(|camera| camera.carried_by.to_string())
            .collect(),
    )
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
    while !token.is_cancelled() {
        if let Err(e) = get_camera_poses::handle_next_request(&runner, |_request| {
            Ok(answer(&mounts, measured.borrow().as_ref()))
        })
        .await
        {
            error!("get_camera_poses: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::JointVec;

    fn v2_mounts() -> CameraMounts {
        CameraMounts::resolve(HardwareVersion::V2).expect("the v2 mounts resolve")
    }

    fn v2_arm(side: Side) -> srs_model::Arm {
        crate::arm_model(HardwareVersion::V2, side.model()).expect("build the arm")
    }

    /// The grasp pose of `side` at `q`, in the robot frame.
    fn grasp(side: Side, q: &JointVec) -> Isometry3<f64> {
        let arm = v2_arm(side);
        arm.world_pose(&arm.at(q).ee_pose())
    }

    fn by_name<'a>(poses: &'a [CameraPose], name: &str) -> &'a CameraPose {
        poses
            .iter()
            .find(|camera| camera.name == name)
            .unwrap_or_else(|| panic!("{name} is listed"))
    }

    #[test]
    fn v1_carries_no_camera() {
        let mounts = CameraMounts::resolve(HardwareVersion::V1).expect("resolves");
        let grasps = ArmPair::new(Isometry3::identity(), Isometry3::identity());
        assert!(mounts.poses(&grasps).is_empty());
    }

    #[test]
    fn the_chest_camera_is_the_models_pose_turned_into_the_optical_frame() {
        let mounts = v2_mounts();
        let grasps = ArmPair::new(Isometry3::identity(), Isometry3::identity());
        let poses = mounts.poses(&grasps);
        assert_eq!(
            poses.iter().map(|c| c.name).collect::<Vec<_>>(),
            ["wrist_left", "wrist_right", "chest"]
        );
        let chest = by_name(&poses, "chest");
        assert_eq!(chest.carried_by, "");
        let (position, _) = world_pose_arrays(&chest.pose);
        assert_eq!(position, [0.0792, 0.0315, 0.7941]);
        // The model's camera looks along its own -Z; the optical frame's +Z
        // is that same direction, and its +Y points down the image where
        // the model's +Y pointed up it.
        let model_rotation = mount_pose(&HardwareVersion::V2.camera_mounts()[2]).rotation;
        let view = model_rotation * -Vector3::z();
        let image_up = model_rotation * Vector3::y();
        let optical = chest.pose.rotation;
        assert!((optical * Vector3::z() - view).norm() < 1e-9);
        assert!((optical * Vector3::y() + image_up).norm() < 1e-9);
        // It faces the robot's front and looks down.
        assert!(view.x > 0.4 && view.z < -0.8, "{view}");
    }

    #[test]
    fn a_wrist_camera_moves_with_its_arm_and_matches_the_urdf_composed_by_hand() {
        let mounts = v2_mounts();
        let home: JointVec = [0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0];
        let bent: JointVec = [0.3, -0.8, 0.2, 1.2, -0.4, 0.5, 0.1];
        let at_home = mounts.poses(&ArmPair::new(
            grasp(Side::Left, &home),
            grasp(Side::Right, &home),
        ));
        let at_bent = mounts.poses(&ArmPair::new(
            grasp(Side::Left, &bent),
            grasp(Side::Right, &home),
        ));
        let left_home = by_name(&at_home, "wrist_left");
        let left_bent = by_name(&at_bent, "wrist_left");
        assert_eq!(left_home.carried_by, "left_arm");
        assert!(
            (left_home.pose.translation.vector - left_bent.pose.translation.vector).norm() > 0.1,
            "the left wrist camera moved with its arm"
        );
        assert_eq!(
            by_name(&at_home, "wrist_right").pose,
            by_name(&at_bent, "wrist_right").pose,
            "the right arm did not move, nor its camera"
        );

        // By hand: the tip link's pose from the chain, then the model's
        // mount in that link, then the half turn into the optical frame.
        let arm = v2_arm(Side::Left);
        let tip_in_root = arm.world_pose(&arm.at(&bent).tip_pose());
        let expected = tip_in_root
            * mount_pose(&HardwareVersion::V2.camera_mounts()[0])
            * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI);
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
        let refused = answer(&mounts, None);
        assert!(!refused.success);
        assert_eq!(refused.message, NOT_MEASURED_YET);
        assert!(refused.camera_names.is_empty() && refused.positions.is_empty());

        let home: JointVec = [0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0];
        let measured = MeasuredGrasps {
            timestamp: SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(7),
            poses: ArmPair::new(grasp(Side::Left, &home), grasp(Side::Right, &home)),
        };
        let reply = answer(&mounts, Some(&measured));
        assert!(reply.success, "{}", reply.message);
        assert_eq!(reply.timestamp, measured.timestamp);
        assert_eq!(reply.camera_names, ["wrist_left", "wrist_right", "chest"]);
        assert_eq!(reply.carried_by, ["left_arm", "right_arm", ""]);
        assert_eq!(reply.positions.len(), 9);
        assert_eq!(reply.orientations.len(), 12);
        for quat in reply.orientations.chunks(4) {
            let norm = quat.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-9);
        }
        // The wrists mirror across the robot's centre line at home.
        assert!((reply.positions[0] - reply.positions[3]).abs() < 1e-6);
        assert!((reply.positions[1] + reply.positions[4]).abs() < 1e-6);
        assert!((reply.positions[2] - reply.positions[5]).abs() < 1e-6);
    }
}
