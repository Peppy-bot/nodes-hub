//! The robots the planner plans for, named by the robot_identity of its initializer, and
//! the model each gives: its arms with the limits limb_motion enforces, as a plan checks
//! them, the links describe_arms names, the body fixed around the arms, and the
//! self-collision model.

use std::fmt;
use std::str::FromStr;

use chain_kinematics::{ChainSpec, JointSelection};
use collision_geometry::chain::{BodyGeometry, CarriedPart, ChainGeometry, Rider};
use motion_planning_core::{Arm, RiderOffTheArm};
use occupancy_map::{Frame, FrameError};
use openarm_description::{ARM_DOF, ArmPair, HardwareVersion, Side};
use self_collision_model::{CollisionError, SelfCollisionModel};

use crate::fingers::{FingerFault, riders};

/// A robot the planner plans for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Robot {
    /// OpenArm v2: two seven-joint arms, each with a pinch gripper, on a torso.
    OpenArmV2,
}

impl Robot {
    const ALL: [Self; 1] = [Self::OpenArmV2];

    /// The name robot_identity gives it.
    const fn name(self) -> &'static str {
        match self {
            Self::OpenArmV2 => "openarm_v2",
        }
    }

    /// The robot's model, its collision pieces fitted to its meshes.
    pub fn model(self) -> Result<RobotModel, ModelError> {
        match self {
            Self::OpenArmV2 => openarm(HardwareVersion::V2),
        }
    }
}

impl fmt::Display for Robot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A robot_identity robot that names no robot the planner plans for.
#[derive(Debug, PartialEq, thiserror::Error)]
#[error("no robot is named {name:?}; the robots are {}", robot_names())]
pub struct UnknownRobot {
    name: String,
}

fn robot_names() -> String {
    Robot::ALL
        .map(|robot| format!("{:?}", robot.name()))
        .join(", ")
}

impl FromStr for Robot {
    type Err = UnknownRobot;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|robot| robot.name() == name)
            .ok_or_else(|| UnknownRobot {
                name: name.to_string(),
            })
    }
}

/// One arm of a robot.
pub struct ArmModel {
    pub side: Side,
    /// The arm's limb name, and its gripper's.
    pub name: &'static str,
    pub gripper_name: &'static str,
    /// The arm's chain from its base link to its grasp point, with the joint limits
    /// limb_motion enforces, and the collision pieces of its links. Its chain gives the
    /// grasp-point pose limb_state reports.
    pub geometry: ChainGeometry<ARM_DOF>,
    /// The arm's joints, in limb_state's order.
    pub joint_names: Vec<String>,
    /// Every link the arm's first joint moves, its gripper's included.
    pub link_names: Vec<String>,
    /// The link at the grasp point.
    pub grasp_link: &'static str,
    /// The axes of the grasp link's frame along which the gripper approaches an item
    /// and along which its jaws close.
    pub approach_axis: &'static str,
    pub closing_axis: &'static str,
    /// The gripper's links, which a held object may touch: every link the arm's last
    /// joint moves that has collision pieces.
    pub gripper_links: Vec<String>,
}

/// A robot's model.
pub struct RobotModel {
    /// The robot frame: the URDF's root link, where the arms and the body are placed.
    pub frame: Frame,
    pub arms: ArmPair<ArmModel>,
    /// The links fixed around the arms.
    pub body: BodyGeometry,
    pub self_model: SelfCollisionModel,
}

/// Why a robot's model cannot be built.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("the robot's meshes: {0}")]
    Meshes(#[from] std::io::Error),
    #[error("the robot's meshes are written to a directory whose path is not UTF-8")]
    MeshesPath,
    #[error("the robot's collision pieces: {0}")]
    Geometry(#[from] collision_geometry::GeometryError),
    #[error("the robot frame: {0}")]
    Frame(#[from] FrameError),
    #[error(transparent)]
    SelfModel(#[from] CollisionError),
    #[error("a finger of the self-collision model: {0}")]
    Finger(#[from] FingerFault),
}

/// OpenArm `version`'s model. Each arm's chain runs from its base link to its grasp point,
/// with the URDF's joint limits, the elbow's lower limit raised to its singularity floor as
/// `HardwareVersion::arm_model` raises it. A finger of the self-collision model that rides
/// no joint of its arm stops the model.
fn openarm(version: HardwareVersion) -> Result<RobotModel, ModelError> {
    let meshes = tempfile::tempdir()?;
    version.write_meshes_to(meshes.path())?;
    let meshes_dir = meshes.path().to_str().ok_or(ModelError::MeshesPath)?;
    let arm = |side: Side| -> Result<ArmModel, ModelError> {
        let geometry = ChainGeometry::<ARM_DOF>::from_urdf(
            version.urdf(),
            meshes_dir,
            &ChainSpec {
                base_link: Some(version.base_link(side)),
                tip_link: version.tcp_link(side),
                joints: JointSelection::PathOrder,
            },
        )?
        .with_lower_floor(
            version.elbow_joint_index(),
            version.elbow_singularity_floor_rad(),
        );
        let joint_names = (0..ARM_DOF)
            .map(|i| geometry.chain().joint_name(i).to_string())
            .collect();
        Ok(ArmModel {
            side,
            name: side.arm_name(),
            gripper_name: side.gripper_name(),
            joint_names,
            link_names: moved_by(&geometry, 0),
            grasp_link: version.tcp_link(side),
            // The grasp frame of both generations: +z out of the gripper, y on the jaw
            // closing axis.
            approach_axis: "+z",
            closing_axis: "+y",
            gripper_links: with_pieces(&geometry, moved_by(&geometry, ARM_DOF - 1)),
            geometry,
        })
    };
    let body = BodyGeometry::from_urdf(
        version.urdf(),
        meshes_dir,
        &[(version.torso_body(), version.torso_regions())],
    )?;
    let arms = ArmPair::new(arm(Side::Left)?, arm(Side::Right)?);
    let tree = arms[Side::Left].geometry.chain().tree();
    let self_model = self_collision_model::openarm(version, meshes_dir)?;
    for arm in arms.iter() {
        fingers_fit(&self_model, arm)?;
    }
    Ok(RobotModel {
        frame: Frame::parse(&tree.link(tree.root()).name)?,
        arms,
        body,
        self_model,
    })
}

/// Refuses a finger of `model` that does not fit `arm`: one the model does not hang off
/// the link of the arm that carries it, or one that rides no joint of the arm, naming
/// the finger.
fn fingers_fit(model: &SelfCollisionModel, arm: &ArmModel) -> Result<(), FingerFault> {
    on_the_arm(&arm.geometry, &riders(model, &arm.geometry, arm.side)?)?;
    Ok(())
}

/// Refuses a finger of `fingers` that rides no joint of the arm whose links `geometry`
/// holds, naming the finger.
fn on_the_arm(geometry: &ChainGeometry<ARM_DOF>, fingers: &[Rider]) -> Result<(), RiderOffTheArm> {
    Arm::new(geometry, None).riding(fingers).map(|_| ())
}

/// The links of `links` that have collision pieces in `geometry`, on its chain or carried
/// by it.
fn with_pieces(geometry: &ChainGeometry<ARM_DOF>, links: Vec<String>) -> Vec<String> {
    let has_pieces = |link: &str| {
        geometry
            .links()
            .iter()
            .any(|chained| chained.name() == link && !chained.pieces().is_empty())
            || geometry
                .carried()
                .iter()
                .map(CarriedPart::rider)
                .any(|carried| carried.link == link && !carried.pieces.is_empty())
    };
    links.into_iter().filter(|link| has_pieces(link)).collect()
}

/// The link that joint `joint` of `geometry`'s chain moves, then every link below it.
fn moved_by(geometry: &ChainGeometry<ARM_DOF>, joint: usize) -> Vec<String> {
    let chain = geometry.chain();
    let tree = chain.tree();
    let link = chain.at(&[0.0; ARM_DOF]).link_name(joint).to_string();
    let index = tree
        .link_index(&link)
        .expect("a link of the chain is a link of its tree");
    std::iter::once(link)
        .chain(
            tree.subtree_from(index, &|_| 0.0)
                .into_iter()
                .map(|(below, _)| tree.link(below).name.clone()),
        )
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use std::sync::LazyLock;

    /// OpenArm v2's model, built once for the unit tests.
    pub(crate) static OPENARM_V2: LazyLock<RobotModel> =
        LazyLock::new(|| Robot::OpenArmV2.model().expect("OpenArm v2's model"));

    // The parameter names each robot, and a name of none is refused with the names.
    #[test]
    fn the_robot_parameter_names_a_robot() {
        assert_eq!("openarm_v2".parse(), Ok(Robot::OpenArmV2));
        assert_eq!(
            "openarm_v1".parse::<Robot>().unwrap_err().to_string(),
            "no robot is named \"openarm_v1\"; the robots are \"openarm_v2\""
        );
    }

    // Each arm names its joints in limb_state's order, its links from the first joint's
    // down to the grasp point, and its gripper's links from the last joint's that have
    // collision pieces: the grasp point's frame, which has none, is no gripper link.
    #[test]
    fn each_arm_names_its_joints_and_links() {
        let left = &OPENARM_V2.arms[Side::Left];
        assert_eq!(
            left.joint_names,
            (1..=7)
                .map(|j| format!("openarm_left_joint{j}"))
                .collect::<Vec<_>>()
        );
        let mut links = left.link_names.clone();
        links.sort_unstable();
        let mut expected: Vec<String> = (1..=6)
            .map(|l| format!("openarm_left_link{l}"))
            .chain(
                ["ee_base_link", "ee_link1", "ee_link2", "tcp"]
                    .map(|l| format!("openarm_left_{l}")),
            )
            .collect();
        expected.sort_unstable();
        assert_eq!(links, expected);
        let mut gripper = left.gripper_links.clone();
        gripper.sort_unstable();
        assert_eq!(
            gripper,
            ["ee_base_link", "ee_link1", "ee_link2"].map(|l| format!("openarm_left_{l}"))
        );
        assert_eq!(left.grasp_link, "openarm_left_tcp");
        assert_eq!((left.name, left.gripper_name), ("left_arm", "left_gripper"));
        let mut body: Vec<&str> = OPENARM_V2.body.link_names().collect();
        body.sort_unstable();
        assert_eq!(
            body,
            [
                "openarm_body_link0",
                "openarm_head_camera",
                "openarm_left_base_link",
                "openarm_right_base_link"
            ]
        );
    }

    // Each arm's two fingers fit it. A finger the self model hangs off the other arm, or
    // one past the arm's last joint, stops the model, the message naming it.
    #[test]
    fn a_finger_off_its_arm_stops_the_model() {
        let model = &OPENARM_V2.self_model;
        for arm in OPENARM_V2.arms.iter() {
            assert_eq!(fingers_fit(model, arm), Ok(()));
            let fingers = riders(model, &arm.geometry, arm.side).expect("the arm's fingers");
            assert_eq!(fingers.len(), 2, "{:?}", arm.side);
            let finger = fingers[0].link;
            let crossed = ModelError::from(FingerFault::from(
                riders(model, &arm.geometry, arm.side.other())
                    .err()
                    .expect("the other arm's fault"),
            ));
            assert!(
                crossed.to_string().starts_with(&format!(
                    "a finger of the self-collision model: the self model holds '{finger}' as \
                     no finger of the "
                )),
                "{crossed}"
            );
            let off = Rider {
                segment: ARM_DOF,
                ..fingers[0]
            };
            assert_eq!(
                ModelError::from(FingerFault::from(
                    on_the_arm(&arm.geometry, &[off]).unwrap_err()
                ))
                .to_string(),
                format!(
                    "a finger of the self-collision model: {finger} rides the child link of \
                     joint {ARM_DOF}, and the arm's joints run from 0 to {}",
                    ARM_DOF - 1
                )
            );
        }
    }

    // The limits a plan keeps are the ones the backbone enforces, the elbow held off its
    // floor, and the chain a plan checks puts the grasp point where the backbone's arm
    // model reports it.
    #[test]
    fn a_plan_keeps_the_backbones_limits_and_grasp_point() {
        for arm in OPENARM_V2.arms.iter() {
            let backbones = HardwareVersion::V2
                .arm_model(arm.side)
                .expect("the backbone's arm model builds");
            let planned = arm.geometry.chain().limits();
            let enforced = backbones.limits();
            for j in 0..ARM_DOF {
                assert_eq!(
                    [planned[j].lo, planned[j].hi],
                    [enforced[j].lo, enforced[j].hi],
                    "joint {j}"
                );
            }
            let bent = [0.3, 0.2, -0.4, 1.1, 0.5, -0.3, 0.7];
            let planned = arm.geometry.chain().world_ee_pose(&bent);
            let reported = backbones.world_ee_pose(&bent);
            assert!(
                (planned.translation.vector - reported.translation.vector).norm() < 1e-12
                    && planned.rotation.angle_to(&reported.rotation) < 1e-12,
                "{planned} against {reported}"
            );
        }
    }
}
