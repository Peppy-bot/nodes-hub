//! The robot's body as its description gives it: the collision shapes of each link, the
//! shapes that the motion planner fits its own collision pieces around, and where each
//! link stands for a snapshot of limb_state, by forward kinematics over the
//! description's tree of joints.

use std::collections::HashMap;

use camera_core::pose::{Pose, PoseError};
use chain_kinematics::nalgebra::{Isometry3, Translation3, UnitQuaternion};
use chain_kinematics::{Tree, TreeError};
use robot_description::urdf_rs::{self, Geometry};
use robot_description::{DescriptionError, RobotDescription, mesh_name};

use crate::model::arms::ArmsSnapshot;
use crate::model::placement::translation;
use crate::model::scene::{PlacedShape, ShapeKind};
use crate::model::stl::{StlError, parse_stl};

/// Why a description gives no body to draw.
#[derive(Debug, thiserror::Error)]
pub enum BodyError {
    #[error("{0}")]
    Description(#[from] DescriptionError),
    #[error("{0}")]
    Tree(#[from] TreeError),
    #[error("the mesh {0} is given twice")]
    MeshTwice(String),
    #[error("link {link}: the mesh {mesh} is not among the meshes given")]
    MissingMesh { link: String, mesh: String },
    #[error("link {link}: the mesh {mesh}: {reason}")]
    Mesh {
        link: String,
        mesh: String,
        reason: StlError,
    },
    #[error("link {link}: a {what} of {value}; a shape's sizes are above 0 and finite")]
    Size {
        link: String,
        what: &'static str,
        value: f64,
    },
    #[error("link {link}: the origin of a collision shape: {reason}")]
    Origin { link: String, reason: PoseError },
}

/// Why the links cannot be placed for a snapshot.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PlacingError {
    #[error("limb_states has no arm {0}, which the robot's description names")]
    MissingArm(String),
    #[error(
        "limb_states gives {got} joints for {arm}, whose chain in the description has {expected}"
    )]
    JointCount {
        arm: String,
        expected: usize,
        got: usize,
    },
    #[error("limb_states has no gripper {0}, which the robot's description names")]
    MissingGripper(String),
    #[error("the opening of {gripper} is {opening}, not a finite number")]
    Opening { gripper: String, opening: f64 },
    #[error("link {link}: {reason}")]
    Pose { link: String, reason: String },
}

/// A link that has collision shapes.
#[derive(Debug)]
pub struct BodyLink {
    pub name: String,
    /// Its collision shapes, each in the link's frame.
    pub shapes: Vec<PlacedShape>,
    /// Its index in the tree.
    index: usize,
}

/// An arm of the description: its name, and the tree's index of each joint of its chain
/// that moves of its own, base to tip, in the order of limb_state.
#[derive(Debug)]
struct DrivenArm {
    name: String,
    joints: Vec<usize>,
}

/// A gripper of the description: its name, and each of its finger joints.
#[derive(Debug)]
struct DrivenGripper {
    name: String,
    fingers: Vec<DrivenFinger>,
}

/// A finger joint: the tree's index of the joint, and its positions with the gripper
/// closed and open. It moves in proportion to the gripper's opening.
#[derive(Debug)]
struct DrivenFinger {
    joint: usize,
    closed: f64,
    open: f64,
}

/// The robot's body: each link with collision shapes, and the joints that limb_state
/// drives. A joint that moves and that neither an arm nor a gripper drives stands at 0.
#[derive(Debug)]
pub struct RobotBody {
    tree: Tree,
    /// In the URDF's order.
    links: Vec<BodyLink>,
    arms: Vec<DrivenArm>,
    grippers: Vec<DrivenGripper>,
}

impl RobotBody {
    /// The body of the description `urdf` and `srdf` (robot_description's rules), with
    /// `meshes`, each collision mesh that the URDF names by its file name. A mesh given
    /// twice, a mesh the URDF names that `meshes` lacks or that is no binary STL, and a
    /// box, a cylinder, a sphere or a capsule whose size is not above 0 are refused. A
    /// capsule is drawn as its cylinder and the spheres at its two ends.
    pub fn parse(urdf: &str, srdf: &str, meshes: &[(&str, &[u8])]) -> Result<Self, BodyError> {
        let description = RobotDescription::parse(urdf, srdf)?;
        let tree = Tree::from_robot(description.robot())?;
        let mut by_name = HashMap::with_capacity(meshes.len());
        for (name, bytes) in meshes {
            if by_name.insert(*name, *bytes).is_some() {
                return Err(BodyError::MeshTwice(name.to_string()));
            }
        }
        let mut links = Vec::new();
        for link in &description.robot().links {
            if link.collision.is_empty() {
                continue;
            }
            let mut shapes = Vec::new();
            for collision in &link.collision {
                shapes.extend(collision_shapes(&link.name, collision, &by_name)?);
            }
            let index = tree
                .link_index(&link.name)
                .expect("the tree holds each link of the URDF it is read from");
            links.push(BodyLink {
                name: link.name.clone(),
                shapes,
                index,
            });
        }
        let joints: HashMap<&str, usize> = tree
            .joints()
            .iter()
            .enumerate()
            .map(|(index, joint)| (joint.name.as_str(), index))
            .collect();
        // robot_description holds each joint of an arm and of a gripper to the URDF that
        // the tree is read from.
        let joint_index = |name: &String| {
            *joints
                .get(name.as_str())
                .expect("the tree holds each joint the description names")
        };
        let mut arms = Vec::new();
        let mut grippers = Vec::new();
        for arm in description.arms() {
            arms.push(DrivenArm {
                name: arm.name.clone(),
                joints: arm.joints.iter().map(joint_index).collect(),
            });
            let fingers: Vec<DrivenFinger> = arm
                .gripper
                .fingers
                .iter()
                .map(|finger| DrivenFinger {
                    joint: joint_index(&finger.joint),
                    closed: finger.closed,
                    open: finger.open,
                })
                .collect();
            if !fingers.is_empty() {
                grippers.push(DrivenGripper {
                    name: arm.gripper.name.clone(),
                    fingers,
                });
            }
        }
        Ok(Self {
            tree,
            links,
            arms,
            grippers,
        })
    }

    /// Each link that has collision shapes, in the URDF's order.
    pub fn links(&self) -> &[BodyLink] {
        &self.links
    }

    /// Where each link with collision shapes stands in the robot frame for `snapshot`:
    /// each arm's joints at the snapshot's positions, each finger at the gripper's
    /// opening, clamped to 0 (closed) to 1 (open). Refused when the snapshot lacks an arm
    /// or a gripper of the description, or gives an arm another number of joints. An
    /// arm or a gripper of the snapshot that the description does not name moves no link.
    pub fn link_poses(
        &self,
        snapshot: &ArmsSnapshot,
    ) -> Result<Vec<(&BodyLink, Pose)>, PlacingError> {
        let mut positions = vec![0.0; self.tree.joints().len()];
        for arm in &self.arms {
            let state = snapshot
                .arms
                .iter()
                .find(|state| state.name == arm.name)
                .ok_or_else(|| PlacingError::MissingArm(arm.name.clone()))?;
            if state.joint_positions.len() != arm.joints.len() {
                return Err(PlacingError::JointCount {
                    arm: arm.name.clone(),
                    expected: arm.joints.len(),
                    got: state.joint_positions.len(),
                });
            }
            for (&joint, &position) in arm.joints.iter().zip(&state.joint_positions) {
                positions[joint] = position;
            }
        }
        for gripper in &self.grippers {
            let state = snapshot
                .grippers
                .iter()
                .find(|state| state.name == gripper.name)
                .ok_or_else(|| PlacingError::MissingGripper(gripper.name.clone()))?;
            if !state.opening.is_finite() {
                return Err(PlacingError::Opening {
                    gripper: gripper.name.clone(),
                    opening: state.opening,
                });
            }
            let opening = state.opening.clamp(0.0, 1.0);
            for finger in &gripper.fingers {
                positions[finger.joint] = finger.closed + opening * (finger.open - finger.closed);
            }
        }
        let mut in_robot = vec![Isometry3::identity(); self.tree.link_count()];
        for (link, placed) in self
            .tree
            .subtree_from(self.tree.root(), &|joint| positions[joint])
        {
            in_robot[link] = placed;
        }
        self.links
            .iter()
            .map(|link| {
                pose_of(&in_robot[link.index])
                    .map(|pose| (link, pose))
                    .map_err(|reason| PlacingError::Pose {
                        link: link.name.clone(),
                        reason: reason.to_string(),
                    })
            })
            .collect()
    }
}

/// The shapes of one collision element of the link `link`, each in the link's frame.
fn collision_shapes(
    link: &str,
    collision: &urdf_rs::Collision,
    meshes: &HashMap<&str, &[u8]>,
) -> Result<Vec<PlacedShape>, BodyError> {
    let origin = urdf_pose(&collision.origin).map_err(|reason| BodyError::Origin {
        link: link.to_string(),
        reason,
    })?;
    let size = |what: &'static str, value: f64| {
        if value > 0.0 && value.is_finite() {
            Ok(value)
        } else {
            Err(BodyError::Size {
                link: link.to_string(),
                what,
                value,
            })
        }
    };
    let at_origin = |kind| PlacedShape { kind, pose: origin };
    let shapes = match &collision.geometry {
        Geometry::Box { size: sides } => vec![at_origin(ShapeKind::Box {
            size: [
                size("side along x", sides[0])?,
                size("side along y", sides[1])?,
                size("side along z", sides[2])?,
            ],
        })],
        Geometry::Cylinder { radius, length } => vec![at_origin(ShapeKind::Cylinder {
            height: size("length", *length)?,
            radius: size("radius", *radius)?,
        })],
        Geometry::Sphere { radius } => vec![at_origin(ShapeKind::Sphere {
            radius: size("radius", *radius)?,
        })],
        Geometry::Capsule { radius, length } => {
            let (radius, length) = (size("radius", *radius)?, size("length", *length)?);
            let end = |z: f64| PlacedShape {
                kind: ShapeKind::Sphere { radius },
                pose: origin.compose(&translation([0.0, 0.0, z])),
            };
            vec![
                at_origin(ShapeKind::Cylinder {
                    height: length,
                    radius,
                }),
                end(length / 2.0),
                end(-length / 2.0),
            ]
        }
        Geometry::Mesh { filename, scale } => {
            let mesh = mesh_name(filename);
            let bytes = meshes.get(mesh).ok_or_else(|| BodyError::MissingMesh {
                link: link.to_string(),
                mesh: mesh.to_string(),
            })?;
            let kind =
                parse_stl(bytes, scale.map_or([1.0; 3], |scale| scale.0)).map_err(|reason| {
                    BodyError::Mesh {
                        link: link.to_string(),
                        mesh: mesh.to_string(),
                        reason,
                    }
                })?;
            vec![at_origin(kind)]
        }
    };
    Ok(shapes)
}

/// A URDF `<origin>`: its `xyz`, turned by the roll, pitch and yaw of its `rpy` about the
/// fixed x, y and z axes, as chain_kinematics reads a joint's origin.
fn urdf_pose(origin: &urdf_rs::Pose) -> Result<Pose, PoseError> {
    let [x, y, z] = origin.xyz.0;
    let [roll, pitch, yaw] = origin.rpy.0;
    pose_of(&Isometry3::from_parts(
        Translation3::new(x, y, z),
        UnitQuaternion::from_euler_angles(roll, pitch, yaw),
    ))
}

fn pose_of(isometry: &Isometry3<f64>) -> Result<Pose, PoseError> {
    let position = isometry.translation.vector;
    let turn = isometry.rotation.quaternion();
    Pose::new(
        [position.x, position.y, position.z],
        [turn.i, turn.j, turn.k, turn.w],
    )
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;
    use std::time::SystemTime;

    use chain_kinematics::{Chain, ChainSpec, JointSelection};

    use super::*;
    use crate::model::arms::{ArmState, GripperState};
    use crate::model::placement::Placement;
    use crate::model::placement::tests::close;
    use crate::testing::{MINI_SRDF, MINI_URDF, mini, mini_meshes};

    /// A snapshot of mini's arm at `lift` and `reach`, its gripper at `opening`.
    fn snapshot(lift: f64, reach: f64, opening: f64) -> ArmsSnapshot {
        ArmsSnapshot {
            at: SystemTime::UNIX_EPOCH,
            arms: vec![ArmState {
                name: "arm".to_string(),
                joint_positions: vec![lift, reach],
                grasp: Placement::parse([0.0; 3], [0.0, 0.0, 0.0, 1.0]).unwrap(),
            }],
            grippers: vec![GripperState {
                name: "gripper".to_string(),
                opening,
            }],
        }
    }

    fn pose_of_link<'a>(poses: &'a [(&BodyLink, Pose)], name: &str) -> &'a Pose {
        &poses
            .iter()
            .find(|(link, _)| link.name == name)
            .expect("a link with shapes")
            .1
    }

    #[test]
    fn each_link_with_collisions_takes_its_shapes_in_its_own_frame() {
        let body = mini();
        let names: Vec<&str> = body.links().iter().map(|link| link.name.as_str()).collect();
        assert_eq!(names, ["base", "upper", "hand", "finger"], "grasp has none");
        let base = &body.links()[0].shapes;
        assert_eq!(
            base[0].kind,
            ShapeKind::Box {
                size: [0.2, 0.2, 0.1]
            }
        );
        assert_eq!(base[0].pose.origin(), [0.0, 0.0, 0.05]);
        let upper = &body.links()[1].shapes[0];
        assert_eq!(
            upper.kind,
            ShapeKind::Cylinder {
                height: 0.2,
                radius: 0.03
            }
        );
        // The URDF's mirror across y: the vertices flip, the triangle turns around.
        let tenth = f64::from(0.1_f32);
        assert_eq!(
            body.links()[2].shapes[0].kind,
            ShapeKind::Mesh {
                vertices: vec![[0.0, 0.0, 0.0], [tenth, 0.0, 0.0], [0.0, -tenth, 0.0]],
                triangles: vec![[0, 2, 1]],
            }
        );
    }

    #[test]
    fn a_capsule_is_its_cylinder_and_a_sphere_at_each_end() {
        let body = mini();
        let capsule = &body.links()[0].shapes[1..];
        assert_eq!(
            capsule[0].kind,
            ShapeKind::Cylinder {
                height: 0.04,
                radius: 0.01
            }
        );
        // The capsule lies along the base's x (pitched a quarter turn), 0.1 m out.
        assert!(close(capsule[0].pose.origin(), [0.1, 0.0, 0.0], 1e-12));
        assert_eq!(capsule[1].kind, ShapeKind::Sphere { radius: 0.01 });
        assert!(close(capsule[1].pose.origin(), [0.12, 0.0, 0.0], 1e-12));
        assert!(close(capsule[2].pose.origin(), [0.08, 0.0, 0.0], 1e-12));
    }

    #[test]
    fn the_links_stand_where_the_joints_and_the_opening_put_them() {
        let body = mini();
        // lift turns the upper arm a quarter turn about z; reach turns the hand a
        // quarter turn about its y, which then points along -x of the robot.
        let poses = body
            .link_poses(&snapshot(FRAC_PI_2, FRAC_PI_2, 0.5))
            .expect("mini's arm and gripper");
        assert_eq!(pose_of_link(&poses, "base").origin(), [0.0, 0.0, 0.0]);
        assert!(close(
            pose_of_link(&poses, "upper").origin(),
            [0.0, 0.0, 0.1],
            1e-12
        ));
        let hand = pose_of_link(&poses, "hand");
        assert!(close(hand.origin(), [0.0, 0.0, 0.3], 1e-12));
        // The hand's x points down, its y along -x of the robot.
        assert!(close(hand.apply([1.0, 0.0, 0.0]), [0.0, 0.0, -0.7], 1e-12));
        assert!(close(hand.apply([0.0, 1.0, 0.0]), [-1.0, 0.0, 0.3], 1e-12));
        // Half open: the finger slides 0.02 m along the hand's y, 0.1 m along its x.
        assert!(close(
            pose_of_link(&poses, "finger").origin(),
            [-0.02, 0.0, 0.2],
            1e-12
        ));
    }

    #[test]
    fn an_opening_past_the_ends_is_clamped() {
        let body = mini();
        let poses = body.link_poses(&snapshot(0.0, 0.0, 1.5)).unwrap();
        assert!(close(
            pose_of_link(&poses, "finger").origin(),
            [0.1, 0.04, 0.3],
            1e-12
        ));
        let poses = body.link_poses(&snapshot(0.0, 0.0, -0.5)).unwrap();
        assert!(close(
            pose_of_link(&poses, "finger").origin(),
            [0.1, 0.0, 0.3],
            1e-12
        ));
    }

    #[test]
    fn a_snapshot_that_does_not_fit_the_description_is_refused() {
        let body = mini();
        let mut other = snapshot(0.0, 0.0, 0.5);
        other.arms[0].name = "left_arm".to_string();
        assert_eq!(
            body.link_poses(&other).unwrap_err(),
            PlacingError::MissingArm("arm".to_string())
        );
        let mut short = snapshot(0.0, 0.0, 0.5);
        short.arms[0].joint_positions.pop();
        assert_eq!(
            body.link_poses(&short).unwrap_err(),
            PlacingError::JointCount {
                arm: "arm".to_string(),
                expected: 2,
                got: 1
            }
        );
        let mut no_gripper = snapshot(0.0, 0.0, 0.5);
        no_gripper.grippers.clear();
        assert_eq!(
            body.link_poses(&no_gripper).unwrap_err(),
            PlacingError::MissingGripper("gripper".to_string())
        );
        assert!(matches!(
            body.link_poses(&snapshot(0.0, 0.0, f64::NAN)).unwrap_err(),
            PlacingError::Opening { gripper, .. } if gripper == "gripper"
        ));
        assert!(matches!(
            body.link_poses(&snapshot(f64::NAN, 0.0, 0.5)).unwrap_err(),
            PlacingError::Pose { link, .. } if link == "upper"
        ));
    }

    #[test]
    fn an_extra_arm_of_the_snapshot_moves_nothing() {
        let body = mini();
        let mut more = snapshot(0.0, 0.0, 0.5);
        more.arms.push(ArmState {
            name: "spare_arm".to_string(),
            joint_positions: vec![1.0],
            grasp: more.arms[0].grasp,
        });
        assert_eq!(body.link_poses(&more).unwrap().len(), 4);
    }

    #[test]
    fn a_mesh_that_is_missing_twice_or_unreadable_is_refused() {
        assert!(matches!(
            RobotBody::parse(MINI_URDF, MINI_SRDF, &[]),
            Err(BodyError::MissingMesh { link, mesh }) if link == "hand" && mesh == "hand.stl"
        ));
        let meshes = mini_meshes();
        let (name, bytes) = (meshes[0].0.as_str(), meshes[0].1.as_slice());
        assert!(matches!(
            RobotBody::parse(MINI_URDF, MINI_SRDF, &[(name, bytes), (name, bytes)]),
            Err(BodyError::MeshTwice(twice)) if twice == "hand.stl"
        ));
        assert!(matches!(
            RobotBody::parse(MINI_URDF, MINI_SRDF, &[(name, &bytes[..bytes.len() - 1])]),
            Err(BodyError::Mesh {
                reason: StlError::Size { .. },
                ..
            })
        ));
    }

    #[test]
    fn a_shape_size_that_is_not_above_0_is_refused() {
        let flat = MINI_URDF.replace(r#"radius="0.03""#, r#"radius="0""#);
        let meshes = mini_meshes();
        let meshes = [(meshes[0].0.as_str(), meshes[0].1.as_slice())];
        assert!(matches!(
            RobotBody::parse(&flat, MINI_SRDF, &meshes),
            Err(BodyError::Size { link, what: "radius", .. }) if link == "upper"
        ));
    }

    /// True when `pose` and `expected` place the frame within `tolerance` (m), its origin
    /// and the tips of its unit axes.
    fn same_pose(pose: &Pose, expected: &Isometry3<f64>, tolerance: f64) -> bool {
        [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            .into_iter()
            .all(|point| {
                let moved = expected.transform_point(&point.into());
                close(pose.apply(point), [moved.x, moved.y, moved.z], tolerance)
            })
    }

    #[test]
    fn the_links_of_openarm_v2_stand_where_chain_kinematics_puts_them() {
        let model = robot_description::model("openarm_v2").expect("a bundled model");
        let body = RobotBody::parse(model.urdf(), model.srdf(), model.meshes())
            .expect("openarm_v2's description");
        assert_eq!(body.links().len(), 22, "each link with collision meshes");
        let left = [-0.2, 0.4, -0.5, 0.9, 0.6, -0.7, 0.1];
        let right = [0.3, -0.4, 0.5, 1.1, -0.6, 0.7, -0.2];
        let arm = |name: &str, joints: &[f64]| ArmState {
            name: name.to_string(),
            joint_positions: joints.to_vec(),
            grasp: Placement::parse([0.0; 3], [0.0, 0.0, 0.0, 1.0]).unwrap(),
        };
        let gripper = |name: &str| GripperState {
            name: name.to_string(),
            opening: 0.5,
        };
        let snapshot = ArmsSnapshot {
            at: SystemTime::UNIX_EPOCH,
            arms: vec![arm("left_arm", &left), arm("right_arm", &right)],
            grippers: vec![gripper("left_gripper"), gripper("right_gripper")],
        };
        let poses = body.link_poses(&snapshot).expect("openarm_v2's arms");
        let robot = urdf_rs::read_from_string(model.urdf()).unwrap();
        for (tcp, joints) in [("openarm_left_tcp", left), ("openarm_right_tcp", right)] {
            let chain = Chain::<7>::from_urdf(
                &robot,
                &ChainSpec {
                    base_link: None,
                    tip_link: tcp,
                    joints: JointSelection::PathOrder,
                },
            )
            .unwrap();
            let posed = chain.at(&joints);
            for segment in 0..7 {
                let link = posed.link_name(segment);
                assert!(
                    same_pose(
                        pose_of_link(&poses, link),
                        &posed.link_pose_world(segment),
                        1e-9
                    ),
                    "{link}"
                );
            }
        }
    }

    #[test]
    fn a_description_that_does_not_read_is_refused() {
        assert!(matches!(
            RobotBody::parse("<robot", MINI_SRDF, &[]),
            Err(BodyError::Description(_))
        ));
    }
}
