//! The robot the planner plans for, as the robot_identity of its initializer describes it:
//! its arms, each an arm group of the description's SRDF, with the limits limb_motion
//! enforces, as a plan checks them, the links describe_arms names, the body fixed around
//! the arms, and the robot's own body. The planner's libraries fix an arm's joint count
//! and the robot's arm count when they compile, so the planner builds the model of a
//! description at the shape of its arms, which robot_description's `on_shape` picks.

use std::collections::HashMap;

use chain_kinematics::nalgebra::{Isometry3, Vector3};
use chain_kinematics::{ChainSpec, JointKind, JointSelection, Tree};
use collision_geometry::chain::{BodyGeometry, CarriedPart, ChainGeometry, Rider, RiderBody};
use motion_planning_core::{Arm, RiderOffTheArm};
use occupancy_map::{Frame, FrameError};
use robot_description::{ArmId, PerArm, RobotDescription};
use self_collision_model::{CollisionError, SelfCollisionModel};

use crate::fingers::{FingerFault, riders_of_the_arms};
use crate::own_body::OwnBody;

/// The axis of an arm's grasp frame along which its gripper approaches an item: the grasp
/// link's +z, which points from the grasp point away from the hand, as robot_description's
/// README states under "Adding a robot" and its bundled-models test checks for every
/// bundled model.
const APPROACH_AXIS: &str = "+z";

/// How far, in radians, a gripper's closing axis may lie from the axis of the grasp frame
/// it is reported as: a description writes its angles to a few decimals (a quarter turn as
/// 1.5708 lies 3.7e-6 rad from a quarter turn), so its frame axes come out that far off;
/// a closing axis farther off lies between the frame's axes.
const FRAME_AXIS_TOLERANCE_RAD: f64 = 1e-3;

/// One arm of a robot.
pub struct ArmModel<const N: usize> {
    /// The arm's place among the description's arms.
    pub id: ArmId,
    /// The arm's limb name, its SRDF group's, and its gripper's.
    pub name: String,
    pub gripper_name: String,
    /// The arm's chain from its base link to its grasp point, with the joint limits
    /// limb_motion enforces, and the collision pieces of its links. Its chain gives the
    /// grasp-point pose limb_state reports.
    pub geometry: ChainGeometry<N>,
    /// The arm's joints, in limb_state's order.
    pub joint_names: Vec<String>,
    /// Every link the arm's first joint moves, its gripper's included.
    pub link_names: Vec<String>,
    /// The link at the grasp point.
    pub grasp_link: String,
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
    pub arms: PerArm<ArmModel<N>, K>,
    /// The links fixed around the arms.
    pub body: BodyGeometry,
    pub own: OwnBody<N, K>,
}

/// Why a robot's model cannot be built.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("the robot's collision pieces: {0}")]
    Geometry(#[from] collision_geometry::GeometryError),
    #[error("the robot frame: {0}")]
    Frame(#[from] FrameError),
    #[error(transparent)]
    SelfModel(#[from] CollisionError),
    #[error("a finger of the robot's own body: {0}")]
    Finger(#[from] FingerFault),
    #[error(
        "{arm} starts at {base}, which a moving joint carries; the planner plans arms whose \
         base stands fixed to the robot's root"
    )]
    MovingBase { arm: String, base: String },
    #[error("{gripper} moves no finger; the planner plans grippers that move one or more")]
    NoFinger { gripper: String },
    #[error(
        "the finger {finger} of {gripper} carries {below} on a moving joint; the planner plans \
         grippers whose fingers are one link each"
    )]
    MultiLinkFinger {
        gripper: String,
        finger: String,
        below: String,
    },
    #[error(
        "{gripper} closes along {axis:?} in the grasp frame of {arm}, which lies along no \
         axis of the frame"
    )]
    ClosingAxis {
        arm: String,
        gripper: String,
        axis: [f64; 3],
    },
}

/// The model of the robot `description` describes, of `K` arms of `N` joints, `D` in all,
/// its collision meshes read from `meshes_dir`. Each arm's chain runs from its base link to
/// its grasp link over its joints, with the URDF's limits. A finger of the self model that
/// rides no joint of its arm stops the model.
///
/// # Panics
///
/// When the description does not hold `K` arms of `N` joints each.
pub fn model<const N: usize, const D: usize, const K: usize>(
    description: &RobotDescription,
    meshes_dir: &str,
) -> Result<RobotModel<N, D, K>, ModelError> {
    const { assert!(D == N * K, "the robot's joints are its arms' joints") };
    let described = description.arms();
    assert!(
        described.len() == K && described.iter().all(|arm| arm.joints.len() == N),
        "the description holds {K} arms of {N} joints each"
    );
    let arms = PerArm::from_fn(|id| arm_model(description, meshes_dir, id, &described[id.place()]))
        .transpose()?;
    let body = BodyGeometry::from_urdf(description.urdf(), meshes_dir)?;
    let own = OwnBody {
        model: SelfCollisionModel::from_description(description, meshes_dir)?,
    };
    built(arms, body, own)
}

/// The model of `arm`, at `id` among the arms of `description`.
fn arm_model<const N: usize>(
    description: &RobotDescription,
    meshes_dir: &str,
    id: ArmId,
    arm: &robot_description::Arm,
) -> Result<ArmModel<N>, ModelError> {
    let joints: Vec<&str> = arm.joints.iter().map(String::as_str).collect();
    let geometry = ChainGeometry::<N>::from_urdf(
        description.urdf(),
        meshes_dir,
        &ChainSpec {
            base_link: Some(&arm.base_link),
            tip_link: &arm.grasp_link,
            joints: JointSelection::Named(&joints),
        },
    )?;
    let tree = geometry.chain().tree();
    stands_fixed(tree, arm)?;
    let closing_axis = closing_axis(tree, arm)?;
    Ok(ArmModel {
        id,
        name: arm.name.clone(),
        gripper_name: arm.gripper.name.clone(),
        joint_names: arm.joints.clone(),
        link_names: moved_by(&geometry, 0),
        grasp_link: arm.grasp_link.clone(),
        approach_axis: APPROACH_AXIS,
        closing_axis,
        gripper_links: with_pieces(&geometry, moved_by(&geometry, N - 1)),
        geometry,
    })
}

/// Refuses an arm whose base a moving joint carries, or whose gripper has a finger that
/// carries another moving link.
fn stands_fixed(tree: &Tree, arm: &robot_description::Arm) -> Result<(), ModelError> {
    let root = tree.root();
    if tree.link(root).name != arm.base_link && tree.fixed_path_to(root, &arm.base_link).is_none() {
        return Err(ModelError::MovingBase {
            arm: arm.name.clone(),
            base: arm.base_link.clone(),
        });
    }
    for finger in &arm.gripper.fingers {
        let link = tree
            .link_index(&finger.link)
            .expect("a finger's link is a link of the URDF");
        if let Some(below) = moving_below(tree, link) {
            return Err(ModelError::MultiLinkFinger {
                gripper: arm.gripper.name.clone(),
                finger: finger.link.clone(),
                below,
            });
        }
    }
    Ok(())
}

/// The first link below `link` that a moving joint carries, if one is.
fn moving_below(tree: &Tree, link: usize) -> Option<String> {
    tree.children_of(link).iter().find_map(|&joint| {
        let joint = tree.joint(joint);
        if joint.kind.is_movable() {
            return Some(tree.link(joint.child).name.clone());
        }
        moving_below(tree, joint.child)
    })
}

/// The axis of the grasp frame of `arm` along which its gripper closes, from the gripper's
/// first finger at the zero of every joint: a sliding finger's axis, or, for a turning
/// finger, the approach axis crossed with its axis; reported as the positive frame axis it
/// lies along.
fn closing_axis(tree: &Tree, arm: &robot_description::Arm) -> Result<&'static str, ModelError> {
    let finger = arm
        .gripper
        .fingers
        .first()
        .ok_or_else(|| ModelError::NoFinger {
            gripper: arm.gripper.name.clone(),
        })?;
    let root = tree.root();
    let poses: HashMap<usize, Isometry3<f64>> = tree
        .subtree_from(root, &|_| 0.0)
        .into_iter()
        .chain([(root, Isometry3::identity())])
        .collect();
    let pose_of = |link: &str| {
        let index = tree
            .link_index(link)
            .expect("a described link is a link of the URDF");
        poses[&index]
    };
    let joint = tree
        .joints()
        .iter()
        .find(|joint| joint.name == finger.joint)
        .expect("a finger's joint is a joint of the URDF");
    let to_grasp = pose_of(&arm.grasp_link).rotation.inverse() * pose_of(&finger.link).rotation;
    let closing = match joint.kind {
        JointKind::Prismatic { axis } => to_grasp * axis.into_inner(),
        JointKind::Revolute { axis } => Vector3::z().cross(&(to_grasp * axis.into_inner())),
        JointKind::Fixed => unreachable!("a finger's joint moves"),
    };
    frame_axis(&closing).ok_or_else(|| ModelError::ClosingAxis {
        arm: arm.name.clone(),
        gripper: arm.gripper.name.clone(),
        axis: closing.into(),
    })
}

/// The positive axis of a frame that `direction` lies along, either way, within
/// [`FRAME_AXIS_TOLERANCE_RAD`].
fn frame_axis(direction: &Vector3<f64>) -> Option<&'static str> {
    let unit = direction.try_normalize(0.0)?;
    [
        ("+x", Vector3::x()),
        ("+y", Vector3::y()),
        ("+z", Vector3::z()),
    ]
    .into_iter()
    .find(|(_, axis)| unit.dot(axis).abs() >= FRAME_AXIS_TOLERANCE_RAD.cos())
    .map(|(name, _)| name)
}

/// The robot of `arms`, `body` and `own`, its frame the root of the first arm's URDF. A
/// finger of `own` that does not fit its arm stops the model.
fn built<const N: usize, const D: usize, const K: usize>(
    arms: PerArm<ArmModel<N>, K>,
    body: BodyGeometry,
    own: OwnBody<N, K>,
) -> Result<RobotModel<N, D, K>, ModelError> {
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

/// Refuses a finger of `fingers` that rides no joint of `arm`, naming the finger.
fn on_the_arm<const N: usize>(arm: &ArmModel<N>, fingers: &[Rider]) -> Result<(), RiderOffTheArm> {
    Arm::new(&arm.name, &arm.geometry)
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
                        && matches!(
                            carried.body,
                            RiderBody::Part(part) if !part.pieces().is_empty()
                        )
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

    use chain_kinematics::nalgebra::Point3;
    use robot_description::{OnShape, on_shape};

    /// The description `robot_description` bundles as `id`, edited by `urdf` and `srdf`,
    /// with the bundled meshes written to a directory of their own.
    pub(crate) struct Described {
        pub description: RobotDescription,
        meshes: tempfile::TempDir,
    }

    impl Described {
        pub fn edited(
            id: &str,
            urdf: impl FnOnce(&str) -> String,
            srdf: impl FnOnce(&str) -> String,
        ) -> Self {
            let bundled = robot_description::model(id).expect("robot_description bundles it");
            let meshes = tempfile::tempdir().expect("a temporary directory");
            bundled
                .write_meshes_to(meshes.path())
                .expect("the meshes write");
            let description = RobotDescription::parse(&urdf(bundled.urdf()), &srdf(bundled.srdf()))
                .expect("the edited description parses");
            Self {
                description,
                meshes,
            }
        }

        pub fn bundled(id: &str) -> Self {
            Self::edited(id, str::to_string, str::to_string)
        }

        pub fn meshes_dir(&self) -> &str {
            self.meshes.path().to_str().expect("the directory is UTF-8")
        }

        pub fn model<const N: usize, const D: usize, const K: usize>(
            &self,
        ) -> Result<RobotModel<N, D, K>, ModelError> {
            model(&self.description, self.meshes_dir())
        }
    }

    /// The posture `name` of `arm` of the bundled description `id`, as its SRDF gives it.
    pub(crate) fn posture<const N: usize>(id: &str, arm: ArmId, name: &str) -> [f64; N] {
        let description = robot_description::model(id)
            .expect("robot_description bundles it")
            .description();
        let group = &description.arms()[arm.place()].name;
        description
            .state(group, name)
            .unwrap_or_else(|| panic!("{group} of {id} has the posture {name}"))
            .try_into()
            .expect("a posture holds a value for each joint of the arm")
    }

    /// The joints of each arm of OpenArm v2, the robot the planner's unit tests run, and of
    /// both.
    pub(crate) const ARM_DOF: usize = 7;
    pub(crate) const ARMS: usize = 2;
    pub(crate) const DUAL_DOF: usize = ARMS * ARM_DOF;
    /// OpenArm v2's arms, in the order of its description: the left, then the right.
    pub(crate) const LEFT: ArmId = ArmId::new(0);
    pub(crate) const RIGHT: ArmId = ArmId::new(1);
    pub(crate) const BOTH: [ArmId; ARMS] = [LEFT, RIGHT];

    /// The ready posture of `arm` of OpenArm v2, as its description gives it.
    pub(crate) fn ready(arm: ArmId) -> [f64; ARM_DOF] {
        posture("openarm_v2", arm, robot_description::states::READY)
    }

    /// The home posture of `arm` of OpenArm v2, as its description gives it.
    pub(crate) fn home(arm: ArmId) -> [f64; ARM_DOF] {
        posture("openarm_v2", arm, robot_description::states::HOME)
    }

    // The unit tests' names for OpenArm v2's arms are the arms of its description.
    #[test]
    fn openarm_v2_lists_its_left_arm_first() {
        let names: Vec<&str> = BOTH
            .iter()
            .map(|arm| OPENARM_V2.arms[*arm].name.as_str())
            .collect();
        assert_eq!(names, ["left_arm", "right_arm"]);
    }

    /// The model of the bundled description `id`.
    pub(crate) fn bundled<const N: usize, const D: usize, const K: usize>(
        id: &str,
    ) -> RobotModel<N, D, K> {
        Described::bundled(id)
            .model()
            .unwrap_or_else(|error| panic!("{id}: {error}"))
    }

    /// OpenArm v2's model, built once for the unit tests.
    pub(crate) static OPENARM_V2: LazyLock<RobotModel<ARM_DOF, DUAL_DOF, ARMS>> =
        LazyLock::new(|| bundled("openarm_v2"));
    /// OpenArm v1's model.
    static OPENARM_V1: LazyLock<RobotModel<ARM_DOF, DUAL_DOF, ARMS>> =
        LazyLock::new(|| bundled("openarm_v1"));
    /// SO-101's model.
    static SO101: LazyLock<RobotModel<5, 5, 1>> = LazyLock::new(|| bundled("so101"));

    /// The SRDF `srdf` without the states of `group`.
    fn without_states(srdf: &str, group: &str) -> String {
        let opening = format!("group=\"{group}\">");
        srdf.split("<group_state ")
            .enumerate()
            .map(|(index, part)| match index {
                0 => part.to_string(),
                _ if part.contains(&opening) => {
                    let end = part.find("</group_state>").expect("a closed state")
                        + "</group_state>".len();
                    part[end..].to_string()
                }
                _ => format!("<group_state {part}"),
            })
            .collect()
    }

    /// The checks of a bundled model: each arm is its description's, its joints in
    /// limb_state's order, its links from the first joint's down, the grasp link among them,
    /// and its gripper's links, the links of those the last joint moves that have collision
    /// pieces, which the grasp frame's link has none of; and the body is every link with
    /// collision pieces fixed to the root.
    struct EachArmIsItsDescriptions<'d> {
        id: &'d str,
        described: &'d Described,
    }

    impl OnShape for EachArmIsItsDescriptions<'_> {
        type Output = ();

        fn on_shape<const N: usize, const D: usize, const K: usize>(self) {
            let Self { id, described } = self;
            let model = described
                .model::<N, D, K>()
                .unwrap_or_else(|error| panic!("{id}: {error}"));
            for (arm, from) in model.arms.iter().zip(described.description.arms()) {
                assert_eq!(arm.name, from.name, "{id}");
                assert_eq!(arm.gripper_name, from.gripper.name, "{id}");
                assert_eq!(arm.joint_names, from.joints, "{id}");
                assert_eq!(arm.grasp_link, from.grasp_link, "{id}");
                assert!(arm.link_names.contains(&arm.grasp_link), "{id}");
                assert!(!arm.gripper_links.contains(&arm.grasp_link), "{id}");
                assert!(!arm.gripper_links.is_empty(), "{id}");
                for link in &arm.gripper_links {
                    assert!(arm.link_names.contains(link), "{id}: {link}");
                }
                for finger in &from.gripper.fingers {
                    assert!(
                        arm.gripper_links.contains(&finger.link),
                        "{id}: {}",
                        finger.link
                    );
                }
            }
            let tree = model.arms[ArmId::new(0)].geometry.chain().tree();
            for link in model.body.link_names() {
                let root = tree.root();
                assert!(
                    tree.link(root).name == link || tree.fixed_path_to(root, link).is_some(),
                    "{id}: {link} stands fixed to the root"
                );
            }
        }
    }

    #[test]
    fn each_bundled_models_arms_are_its_descriptions() {
        for bundled in robot_description::models() {
            let id = bundled.id();
            let described = Described::bundled(id);
            on_shape(
                &described.description,
                EachArmIsItsDescriptions {
                    id,
                    described: &described,
                },
            )
            .unwrap_or_else(|error| panic!("{id}: {error}"));
        }
    }

    // Each gripper closes along the axis its first finger moves along, or turns across:
    // OpenArm's fingers slide or turn across the grasp frame's y, SO-101's jaw across its x.
    #[test]
    fn each_gripper_closes_along_the_axis_its_finger_moves() {
        for arm in OPENARM_V1.arms.iter().chain(OPENARM_V2.arms.iter()) {
            assert_eq!(arm.closing_axis, "+y");
        }
        let [arm] = &SO101.arms.iter().collect::<Vec<_>>()[..] else {
            panic!("SO-101 has one arm");
        };
        assert_eq!(arm.closing_axis, "+x");
    }

    // An arm whose base a moving joint carries, a gripper finger that carries a moving
    // link, a gripper that moves no finger and one that closes between the frame's axes
    // are refused, naming them.
    #[test]
    fn a_description_the_planner_cannot_plan_for_is_refused() {
        let refused = |described: Described| -> String {
            match described.model::<5, 5, 1>() {
                Ok(_) => panic!("the model builds"),
                Err(error) => error.to_string(),
            }
        };
        let carried = Described::edited(
            "so101",
            |urdf| {
                urdf.replace(
                    "</robot>",
                    r#"<link name="pad"/>
                       <joint name="pad_joint" type="revolute">
                         <parent link="moving_jaw_so101_v1_link"/><child link="pad"/>
                         <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
                       </joint>
                     </robot>"#,
                )
            },
            str::to_string,
        );
        assert_eq!(
            refused(carried),
            "the finger moving_jaw_so101_v1_link of gripper carries pad on a moving joint; the \
             planner plans grippers whose fingers are one link each"
        );
        let fingerless = Described::edited("so101", str::to_string, |srdf| {
            without_states(srdf, "gripper").replace("<joint name=\"gripper\"/>", "")
        });
        assert_eq!(
            refused(fingerless),
            "gripper moves no finger; the planner plans grippers that move one or more"
        );
        let tilted = Described::edited(
            "so101",
            |urdf| {
                let joint = urdf
                    .find("<joint name=\"gripper\"")
                    .expect("the jaw's joint");
                let axis = urdf[joint..].find("<axis").expect("its axis") + joint;
                let end = urdf[axis..].find("/>").expect("a closed axis") + axis + 2;
                format!("{}<axis xyz=\"0.6 0 0.8\"/>{}", &urdf[..axis], &urdf[end..])
            },
            str::to_string,
        );
        assert!(
            refused(tilted).starts_with("gripper closes along ["),
            "the tilted jaw's refusal names its axis"
        );
    }

    // An arm whose chain starts at a link a moving joint carries is refused.
    #[test]
    fn an_arm_on_a_moving_base_is_refused() {
        let riding = Described::edited("so101", str::to_string, |srdf| {
            without_states(srdf, "arm")
                .replace("base_link=\"base_link\"", "base_link=\"shoulder_link\"")
        });
        let error = match riding.model::<4, 4, 1>() {
            Ok(_) => panic!("the model builds"),
            Err(error) => error.to_string(),
        };
        assert_eq!(
            error,
            "arm starts at shoulder_link, which a moving joint carries; the planner plans arms \
             whose base stands fixed to the robot's root"
        );
    }

    // Each arm's two fingers fit it. A finger the self model hangs off the other arm, or
    // one past the arm's last joint, stops the model, the message naming it.
    #[test]
    fn a_finger_off_its_arm_stops_the_model() {
        let own = &OPENARM_V2.own;
        for arm in OPENARM_V2.arms.iter() {
            let fingers = own
                .fingers(arm.id, &arm.geometry)
                .expect("the arm's fingers");
            assert_eq!(on_the_arm(arm, &fingers), Ok(()));
            assert_eq!(fingers.len(), 2, "{}", arm.name);
            let finger = fingers[0].name;
            let other = ArmId::new(1 - arm.id.place());
            let crossed = ModelError::from(
                own.fingers(other, &arm.geometry)
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
                segment: 7,
                ..fingers[0]
            };
            assert_eq!(
                ModelError::from(FingerFault::from(on_the_arm(arm, &[off]).unwrap_err()))
                    .to_string(),
                format!(
                    "a finger of the robot's own body: {finger} rides the child link of \
                     joint 7, and the arm's joints run from 0 to 6"
                )
            );
        }
    }

    // The limits a plan keeps are the URDF's, which the backbone enforces, and the chain a
    // plan checks puts the grasp point where the closed-form model of the same arm reports it.
    #[test]
    fn a_plan_keeps_the_urdfs_limits_and_the_closed_form_grasp_point() {
        let described = Described::bundled("openarm_v2");
        let robot = described.description.robot();
        for (arm, from) in OPENARM_V2.arms.iter().zip(described.description.arms()) {
            let planned = arm.geometry.chain().limits();
            for (j, name) in from.joints.iter().enumerate() {
                let limit = &robot
                    .joints
                    .iter()
                    .find(|joint| joint.name == *name)
                    .expect("an arm joint of the URDF")
                    .limit;
                assert_eq!(
                    [planned[j].lo, planned[j].hi],
                    [limit.lower, limit.upper],
                    "{name}"
                );
            }
            let closed_form =
                srs_model::Arm::from_urdf(described.description.urdf(), &from.base_link)
                    .expect("the closed-form arm builds")
                    .with_tool_link(&from.grasp_link)
                    .expect("the grasp link is the arm's tool");
            let bent = [0.3, 0.2, -0.4, 1.1, 0.5, -0.3, 0.7];
            let planned = arm.geometry.chain().world_ee_pose(&bent);
            let reported = closed_form.world_ee_pose(&bent);
            assert!(
                (planned.translation.vector - reported.translation.vector).norm() < 1e-12
                    && planned.rotation.angle_to(&reported.rotation) < 1e-12,
                "{planned} against {reported}"
            );
        }
    }

    /// The opening of SO-101's gripper that puts its jaw's joint at 0.
    fn so101_jaw_at_zero() -> f64 {
        let described = Described::bundled("so101");
        let [finger] = &described.description.arms()[0].gripper.fingers[..] else {
            panic!("SO-101's gripper moves one jaw");
        };
        (0.0 - finger.closed) / (finger.open - finger.closed)
    }

    // At the opening that puts the jaw's joint at 0, the jaw hangs where the URDF's joint
    // origin puts it, as the chain carries it.
    #[test]
    fn so101s_jaw_hangs_off_the_gripper_link_at_its_joints_origin() {
        let arm = &SO101.arms[ArmId::new(0)];
        let [part] = arm.geometry.carried() else {
            panic!("SO-101's arm carries its jaw");
        };
        let mut own = SO101.own.clone();
        own.set_openings(&PerArm::new([so101_jaw_at_zero()]));
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
        let beside = Described::edited(
            "so101",
            |urdf| {
                urdf.replace(
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
                )
            },
            str::to_string,
        );
        let arm = &beside.description.arms()[0];
        let joints: Vec<&str> = arm.joints.iter().map(String::as_str).collect();
        let geometry = ChainGeometry::<5>::from_urdf(
            beside.description.urdf(),
            beside.meshes_dir(),
            &ChainSpec {
                base_link: Some(&arm.base_link),
                tip_link: &arm.grasp_link,
                joints: JointSelection::Named(&joints),
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
        let described = Described::bundled("so101");
        let from = &described.description.arms()[0];
        let arm = &SO101.arms[ArmId::new(0)];
        let tree = arm.geometry.chain().tree();
        let gripper = tree.link_index("gripper_link").expect("the gripper link");
        let tool: Isometry3<f64> = tree
            .fixed_path_to(gripper, &from.grasp_link)
            .expect("the grasp point is fixed to the gripper link");
        let center_at = |opening: f64| {
            let mut own = SO101.own.clone();
            own.set_openings(&PerArm::new([opening]));
            let fingers = own.fingers(arm.id, &arm.geometry).expect("the jaw");
            let [jaw] = &fingers[..] else {
                panic!("SO-101 has one jaw");
            };
            assert_eq!(jaw.name, from.gripper.fingers[0].link);
            let RiderBody::Part(part) = jaw.body else {
                panic!("the jaw is a part of the robot");
            };
            let [piece, ..] = part.pieces() else {
                panic!("the jaw holds a piece");
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
        let gripper_origin = tool.inverse() * Point3::origin();
        assert!(gripper_origin.z < 0.0, "{gripper_origin}");
    }
}
