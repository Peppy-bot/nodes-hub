//! The robots the planner plans for, named by the robot_identity of its initializer, and
//! the model each gives: its arms with the limits limb_motion enforces, as a plan checks
//! them, the links describe_arms names, the body fixed around the arms, and the robot's own
//! body.

use std::fmt;
use std::str::FromStr;

use chain_kinematics::{ChainSpec, JointSelection};
use collision_geometry::chain::{BodyGeometry, CarriedPart, ChainGeometry, Rider, RiderBody};
use motion_planning_core::{Arm, RiderOffTheArm};
use occupancy_map::{Frame, FrameError};
use openarm_description::{ARM_DOF, ARMS, DUAL_DOF, HardwareVersion, Side};
use self_collision_model::CollisionError;
use so101_description::{ARM_DOF as SO101_DOF, ARM_LIMB, BASE_LINK, GRIPPER_LIMB, TCP_LINK, URDF};

use crate::arm_set::{ArmId, PerArm};
use crate::fingers::{FingerFault, riders_of_the_arms};
use crate::own_body::OwnBody;

/// A robot the planner plans for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Robot {
    /// OpenArm v2: two seven-joint arms, each with a pinch gripper, on a torso.
    OpenArmV2,
    /// SO-101: one five-joint arm with a one-jawed gripper on a base.
    So101,
}

impl Robot {
    const ALL: [Self; 2] = [Self::OpenArmV2, Self::So101];

    /// The name robot_identity gives it.
    const fn name(self) -> &'static str {
        match self {
            Self::OpenArmV2 => "openarm_v2",
            Self::So101 => "so101",
        }
    }
}

impl fmt::Display for Robot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A model name, as robot_identity gives it, of no robot the planner plans for.
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
pub struct ArmModel<const N: usize> {
    /// The arm's place in the robot's model.
    pub id: ArmId,
    /// The arm's limb name, and its gripper's.
    pub name: &'static str,
    pub gripper_name: &'static str,
    /// The arm's chain from its base link to its grasp point, with the joint limits
    /// limb_motion enforces, and the collision pieces of its links. Its chain gives the
    /// grasp-point pose limb_state reports.
    pub geometry: ChainGeometry<N>,
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

/// A robot's model, of `K` arms of `N` joints, `D` in all.
pub struct RobotModel<const N: usize, const D: usize, const K: usize> {
    /// The robot frame: the URDF's root link, where the arms and the body are placed.
    pub frame: Frame,
    pub arms: PerArm<ArmModel<N>>,
    /// The links fixed around the arms.
    pub body: BodyGeometry,
    pub own: OwnBody<N, K>,
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
    #[error("a finger of the robot's own body: {0}")]
    Finger(#[from] FingerFault),
}

/// OpenArm v2's model. Each arm's chain runs from its base link to its grasp point, with the
/// URDF's joint limits, the elbow's lower limit raised to its singularity floor as
/// `HardwareVersion::arm_model` raises it. A finger of the self-collision model that rides
/// no joint of its arm stops the model.
pub fn openarm_v2() -> Result<RobotModel<ARM_DOF, DUAL_DOF, ARMS>, ModelError> {
    let version = HardwareVersion::V2;
    let meshes = tempfile::tempdir()?;
    version.write_meshes_to(meshes.path())?;
    let meshes_dir = meshes.path().to_str().ok_or(ModelError::MeshesPath)?;
    let arm = |side: Side| -> Result<ArmModel<ARM_DOF>, ModelError> {
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
        Ok(ArmModel {
            id: side.into(),
            name: side.arm_name(),
            gripper_name: side.gripper_name(),
            joint_names: joint_names(&geometry),
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
    let arms = PerArm::both(arm(Side::Left)?, arm(Side::Right)?);
    let own = OwnBody {
        model: self_collision_model::openarm(version, meshes_dir)?,
    };
    built(arms, body, own)
}

/// SO-101's model. Its arm's chain runs from the base link, the robot frame, to its grasp
/// point, and its one jaw rides the gripper link on a joint the chain does not drive.
pub fn so101() -> Result<RobotModel<SO101_DOF, SO101_DOF, 1>, ModelError> {
    let meshes = tempfile::tempdir()?;
    so101_description::write_meshes_to(meshes.path())?;
    let meshes_dir = meshes.path().to_str().ok_or(ModelError::MeshesPath)?;
    let geometry = ChainGeometry::<SO101_DOF>::from_urdf(
        URDF,
        meshes_dir,
        &ChainSpec {
            base_link: Some(BASE_LINK),
            tip_link: TCP_LINK,
            joints: JointSelection::PathOrder,
        },
    )?;
    let arm = ArmModel {
        id: ArmId::new(0),
        name: ARM_LIMB,
        gripper_name: GRIPPER_LIMB,
        joint_names: joint_names(&geometry),
        link_names: moved_by(&geometry, 0),
        grasp_link: TCP_LINK,
        // The URDF's grasp frame turns the gripper link's half a turn about y, so +z points
        // out of the gripper, and the jaw's joint turns it across +x.
        approach_axis: "+z",
        closing_axis: "+x",
        gripper_links: with_pieces(&geometry, moved_by(&geometry, SO101_DOF - 1)),
        geometry,
    };
    let body = BodyGeometry::from_urdf(URDF, meshes_dir, &[])?;
    let own = OwnBody {
        model: self_collision_model::so101(meshes_dir)?,
    };
    built(PerArm::new(vec![arm]), body, own)
}

/// The robot of `arms`, `body` and `own`, its frame the root of the first arm's URDF. A
/// finger of `own` that does not fit its arm stops the model.
fn built<const N: usize, const D: usize, const K: usize>(
    arms: PerArm<ArmModel<N>>,
    body: BodyGeometry,
    own: OwnBody<N, K>,
) -> Result<RobotModel<N, D, K>, ModelError> {
    for (id, arm) in arms.ids().zip(arms.iter()) {
        assert_eq!(arm.id, id, "{} stands at its id's place", arm.name);
    }
    let fingers = riders_of_the_arms(&own, &arms)?;
    for (arm, fingers) in arms.iter().zip(fingers.iter()) {
        on_the_arm(arm, fingers).map_err(FingerFault::from)?;
    }
    let frame = {
        let tree = arms[ArmId::new(0)].geometry.chain().tree();
        Frame::parse(&tree.link(tree.root()).name)?
    };
    Ok(RobotModel {
        frame,
        arms,
        body,
        own,
    })
}

/// The joints of `geometry`'s chain, in limb_state's order.
fn joint_names<const N: usize>(geometry: &ChainGeometry<N>) -> Vec<String> {
    (0..N)
        .map(|i| geometry.chain().joint_name(i).to_string())
        .collect()
}

/// Refuses a finger of `fingers` that rides no joint of `arm`, naming the finger.
fn on_the_arm<const N: usize>(arm: &ArmModel<N>, fingers: &[Rider]) -> Result<(), RiderOffTheArm> {
    Arm::new(arm.name, &arm.geometry)
        .riding(fingers)
        .map(|_| ())
}

/// The links of `links` that have collision pieces in `geometry`, on its chain or carried
/// by it.
fn with_pieces<const N: usize>(geometry: &ChainGeometry<N>, links: Vec<String>) -> Vec<String> {
    let has_pieces = |link: &str| {
        geometry
            .links()
            .iter()
            .any(|chained| chained.name() == link && !chained.pieces().is_empty())
            || geometry
                .carried()
                .iter()
                .map(CarriedPart::rider)
                .any(|carried| {
                    carried.name == link
                        && matches!(carried.body, RiderBody::Pieces(pieces) if !pieces.is_empty())
                })
    };
    links.into_iter().filter(|link| has_pieces(link)).collect()
}

/// The link that joint `joint` of `geometry`'s chain moves, then every link below it.
fn moved_by<const N: usize>(geometry: &ChainGeometry<N>, joint: usize) -> Vec<String> {
    let chain = geometry.chain();
    let tree = chain.tree();
    let link = chain.at(&[0.0; N]).link_name(joint).to_string();
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
    pub(crate) static OPENARM_V2: LazyLock<RobotModel<ARM_DOF, DUAL_DOF, 2>> =
        LazyLock::new(|| openarm_v2().expect("OpenArm v2's model"));
    /// SO-101's model, built once for the unit tests.
    static SO101: LazyLock<RobotModel<SO101_DOF, SO101_DOF, 1>> =
        LazyLock::new(|| so101().expect("SO-101's model"));

    // Each robot parses from the model name robot_identity gives it, and a name of none
    // is refused with the names.
    #[test]
    fn a_model_name_names_a_robot() {
        assert_eq!("openarm_v2".parse(), Ok(Robot::OpenArmV2));
        assert_eq!("so101".parse(), Ok(Robot::So101));
        assert_eq!(
            "openarm_v1".parse::<Robot>().unwrap_err().to_string(),
            "no robot is named \"openarm_v1\"; the robots are \"openarm_v2\", \"so101\""
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
        let own = &OPENARM_V2.own;
        for arm in OPENARM_V2.arms.iter() {
            let side = arm.id.side();
            let fingers = own
                .fingers(arm.id, &arm.geometry)
                .expect("the arm's fingers");
            assert_eq!(on_the_arm(arm, &fingers), Ok(()));
            assert_eq!(fingers.len(), 2, "{side:?}");
            let finger = fingers[0].name;
            let crossed = ModelError::from(
                own.fingers(side.other().into(), &arm.geometry)
                    .expect_err("the other arm's fault"),
            );
            assert!(
                crossed.to_string().starts_with(&format!(
                    "a finger of the robot's own body: the self model holds '{finger}' as \
                     no finger of chain "
                )),
                "{crossed}"
            );
            let off = Rider {
                segment: ARM_DOF,
                ..fingers[0]
            };
            assert_eq!(
                ModelError::from(FingerFault::from(on_the_arm(arm, &[off]).unwrap_err()))
                    .to_string(),
                format!(
                    "a finger of the robot's own body: {finger} rides the child link of \
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
                .arm_model(arm.id.side())
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

    // SO-101's one arm names its five joints in limb_state's order, its links from the
    // shoulder's down to the grasp point, and its gripper's links, the gripper link and the
    // jaw; the base is its body.
    #[test]
    fn so101s_arm_names_its_joints_and_links() {
        let [arm] = &SO101.arms.iter().collect::<Vec<_>>()[..] else {
            panic!("SO-101 has one arm");
        };
        assert_eq!((arm.name, arm.gripper_name), ("arm", "gripper"));
        assert_eq!(
            arm.joint_names,
            [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll"
            ]
        );
        let mut links = arm.link_names.clone();
        links.sort_unstable();
        assert_eq!(
            links,
            [
                "gripper_frame_link",
                "gripper_link",
                "lower_arm_link",
                "moving_jaw_so101_v1_link",
                "shoulder_link",
                "upper_arm_link",
                "wrist_link"
            ]
        );
        let mut gripper = arm.gripper_links.clone();
        gripper.sort_unstable();
        assert_eq!(gripper, ["gripper_link", "moving_jaw_so101_v1_link"]);
        assert_eq!(arm.grasp_link, "gripper_frame_link");
        assert_eq!(SO101.body.link_names().collect::<Vec<_>>(), ["base_link"]);
    }

    // At the opening that puts the jaw's joint at 0, the jaw hangs where the URDF's joint origin
    // puts it, as the chain carries it.
    #[test]
    fn so101s_jaw_hangs_off_the_gripper_link_at_its_joints_origin() {
        let arm = &SO101.arms[ArmId::new(0)];
        let [part] = arm.geometry.carried() else {
            panic!("SO-101's arm carries its jaw");
        };
        let mut own = SO101.own.clone();
        own.set_openings(&PerArm::new(vec![so101_description::jaw_opening(0.0)]));
        let fingers = own.fingers(arm.id, &arm.geometry).expect("the jaw");
        let at_zero = part.rider().offset;
        assert_eq!(fingers[0].segment, part.rider().segment);
        assert!(
            (fingers[0].offset.translation.vector - at_zero.translation.vector).norm() < 1e-12
                && fingers[0].offset.rotation.angle_to(&at_zero.rotation) < 1e-9,
            "{} against {at_zero}",
            fingers[0].offset
        );
    }

    // A part the arm carries beside its jaw, which the self model does not hold, stops the
    // model, the message naming the part.
    #[test]
    fn a_part_beside_the_jaw_stops_the_model() {
        let urdf = URDF.replace(
            "</robot>",
            r#"<link name="beside_the_jaw">
                 <collision><geometry><box size="0.01 0.01 0.01"/></geometry></collision>
               </link>
               <joint name="beside_the_jaw_joint" type="revolute">
                 <parent link="gripper_link"/>
                 <child link="beside_the_jaw"/>
                 <axis xyz="0 0 1"/>
                 <limit lower="-1" upper="1" effort="1" velocity="1"/>
               </joint>
             </robot>"#,
        );
        let meshes = tempfile::tempdir().expect("a temporary directory");
        so101_description::write_meshes_to(meshes.path()).expect("the meshes write");
        let geometry = ChainGeometry::<SO101_DOF>::from_urdf(
            &urdf,
            meshes.path().to_str().expect("the directory is UTF-8"),
            &ChainSpec {
                base_link: Some(BASE_LINK),
                tip_link: TCP_LINK,
                joints: JointSelection::PathOrder,
            },
        )
        .expect("the arm fits");
        assert_eq!(
            ModelError::from(
                SO101
                    .own
                    .fingers(ArmId::new(0), &geometry)
                    .expect_err("a part beside the jaw")
            )
            .to_string(),
            "a finger of the robot's own body: the self model holds no body 'beside_the_jaw'; \
             give the parts of this robot's chain"
        );
    }

    // The jaw swings with the gripper's opening: its piece stands farther from the grasp point
    // open than shut, and shuts along the grasp frame's +x, the closing axis. The grasp frame's
    // +z leads away from the gripper link, the approach axis.
    #[test]
    fn so101s_grasp_axes_are_the_ones_its_jaw_and_gripper_turn_on() {
        use chain_kinematics::nalgebra::{Isometry3, Point3, Vector3};

        let arm = &SO101.arms[ArmId::new(0)];
        let tree = arm.geometry.chain().tree();
        let gripper = tree.link_index("gripper_link").expect("the gripper link");
        let tool: Isometry3<f64> = tree
            .fixed_path_to(gripper, TCP_LINK)
            .expect("the grasp point is fixed to the gripper link");
        let center_at = |opening: f64| {
            let mut own = SO101.own.clone();
            own.set_openings(&PerArm::new(vec![opening]));
            let fingers = own.fingers(arm.id, &arm.geometry).expect("the jaw");
            let [jaw] = &fingers[..] else {
                panic!("SO-101 has one jaw");
            };
            assert_eq!(jaw.name, so101_description::JAW_LINK);
            let RiderBody::Pieces([piece, ..]) = jaw.body else {
                panic!("the jaw is a part of the robot");
            };
            let corners = piece.vertices();
            let sum = corners
                .iter()
                .fold(Vector3::zeros(), |sum, c| sum + c.coords);
            tool.inverse() * jaw.offset * Point3::from(sum / corners.len() as f64)
        };
        let (shut, open) = (center_at(0.0), center_at(1.0));
        assert!(
            shut.coords.norm() < open.coords.norm(),
            "{shut} against {open}"
        );
        assert_eq!(arm.closing_axis, "+x");
        assert!(shut.x > open.x, "{shut} against {open}");
        assert_eq!(arm.approach_axis, "+z");
        let gripper_origin = tool.inverse() * Point3::origin();
        assert!(gripper_origin.z < 0.0, "{gripper_origin}");
    }
}
