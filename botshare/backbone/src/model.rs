//! The differential base a robot model's description gives, the model the
//! robot's initializer names on robot_identity's `get_identity`: its wheels
//! and the outline of its parts on the floor. Each is read from the model's
//! URDF in the frame of its root link, the robot frame, base_footprint, with
//! every joint at zero:
//!
//! - the wheels are the description's differential base
//!   ([`RobotDescription::differential_base`]): its two wheel joints, each
//!   wheel's radius and their separation;
//! - the outline is the rectangle on the floor that holds every collision
//!   element of every link, boxes, cylinders and spheres.

use control_core::footprint::{Footprint, FootprintError};
use control_core::odometry::WheelGeometry;
use control_core::positive_finite::{NotPositiveFinite, PositiveFinite};
use robot_description::urdf_rs::{Geometry, Joint, Robot};
use robot_description::{Model, NoDifferentialBase, RobotDescription};

use crate::geometry::{Frame, Quaternion, rotate};
use crate::transforms::{JointTransforms, ROBOT_FRAME};

/// What a model's description gives the base.
#[derive(Debug)]
pub struct BaseModel {
    pub wheels: WheelGeometry,
    /// The outline's corners, counterclockwise from the front right one.
    pub footprint: Footprint,
    /// The wheel joints, [left, right], whose transforms the node publishes.
    pub joint_transforms: JointTransforms,
}

/// The base of a bundled model, with the model it is built from.
#[derive(Debug)]
pub struct BundledBase {
    pub model: &'static Model,
    pub base: BaseModel,
}

/// Why a description gives no differential base this node drives.
#[derive(Debug, thiserror::Error)]
pub enum BaseError {
    #[error(transparent)]
    NoWheels(#[from] NoDifferentialBase),
    #[error(
        "its root link is {root}: the root link of a mobile base is its robot frame, base_footprint, which odom places on transforms; name the URDF's root link base_footprint"
    )]
    RobotFrame { root: String },
    #[error("the wheel geometry's {length} {source}")]
    Wheels {
        length: &'static str,
        source: NotPositiveFinite,
    },
    #[error(
        "link {link}: the outline holds boxes, cylinders and spheres, and one of its collision elements is none of them"
    )]
    OutlineShape { link: String },
    #[error("no link has collision geometry, so the base has no outline")]
    NoOutline,
    #[error("the outline: {0}")]
    Footprint(FootprintError),
}

/// The geometry of two wheels `separation_m` apart, the left of radius
/// `left_radius_m` and the right of `right_radius_m`, each length above 0.
pub fn wheel_geometry(
    separation_m: f64,
    left_radius_m: f64,
    right_radius_m: f64,
) -> Result<WheelGeometry, BaseError> {
    let length = |length, value| {
        PositiveFinite::parse(value).map_err(|source| BaseError::Wheels { length, source })
    };
    Ok(WheelGeometry {
        wheel_separation_m: length("wheel_separation", separation_m)?,
        left_wheel_radius_m: length("left_wheel_radius", left_radius_m)?,
        right_wheel_radius_m: length("right_wheel_radius", right_radius_m)?,
    })
}

/// Why this node takes no robot of a model.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Unknown(#[from] robot_description::UnknownModel),
    #[error("model {model} is no differential base this node drives: {reason}")]
    NoBase { model: String, reason: BaseError },
}

impl BaseModel {
    /// The base of the bundled model `model`.
    pub fn of(model: &str) -> Result<BundledBase, ModelError> {
        let bundled = robot_description::model(model)?;
        let base =
            BaseModel::parse(bundled.description()).map_err(|reason| ModelError::NoBase {
                model: model.to_owned(),
                reason,
            })?;
        Ok(BundledBase {
            model: bundled,
            base,
        })
    }

    /// The base `description` gives.
    pub fn parse(description: RobotDescription) -> Result<BaseModel, BaseError> {
        let base = description.differential_base()?;
        let root = description.root_link();
        if root != ROBOT_FRAME {
            return Err(BaseError::RobotFrame {
                root: root.to_owned(),
            });
        }
        Ok(BaseModel {
            wheels: wheel_geometry(base.separation_m, base.left.radius_m, base.right.radius_m)?,
            footprint: outline(description.robot())?,
            joint_transforms: JointTransforms::new(
                description,
                [base.left.joint, base.right.joint],
            ),
        })
    }
}

/// The joint `link` hangs on; none for the root.
fn joint_above<'r>(robot: &'r Robot, link: &str) -> Option<&'r Joint> {
    robot.joints.iter().find(|joint| joint.child.link == link)
}

/// The joints from `link` up to the root, nearest first.
fn joints_to_root<'r>(robot: &'r Robot, link: &str) -> Vec<&'r Joint> {
    std::iter::successors(joint_above(robot, link), |joint| {
        joint_above(robot, &joint.parent.link)
    })
    .collect()
}

/// `link`'s frame in the root's, every joint on the way at zero.
fn link_frame(robot: &Robot, link: &str) -> Frame {
    joints_to_root(robot, link)
        .into_iter()
        .fold(Frame::IDENTITY, |placed, joint| {
            Frame::of_origin(joint.origin.xyz.0, joint.origin.rpy.0).then(placed)
        })
}

/// How far a collision element of `geometry`, turned by `orientation`,
/// reaches from its origin along the robot frame's x and y.
fn reach(geometry: &Geometry, orientation: Quaternion) -> Option<[f64; 2]> {
    let axis = |unit: [f64; 3]| rotate(orientation, unit);
    let [x, y, z] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].map(axis);
    match *geometry {
        Geometry::Sphere { radius } => Some([radius, radius]),
        Geometry::Box { ref size } => {
            let half = size.0.map(|side| side / 2.0);
            let along = |i: usize| {
                [x, y, z]
                    .iter()
                    .zip(half)
                    .map(|(column, half)| column[i].abs() * half)
                    .sum::<f64>()
            };
            Some([along(0), along(1)])
        }
        Geometry::Cylinder { radius, length } => {
            // The axis's own reach along i, and the disc's: its radius times
            // the part of the axis across i, which the two other components
            // give.
            let along = |i: usize| {
                let across = (0..3)
                    .filter(|&j| j != i)
                    .map(|j| z[j] * z[j])
                    .sum::<f64>()
                    .sqrt();
                z[i].abs() * length / 2.0 + radius * across
            };
            Some([along(0), along(1)])
        }
        _ => None,
    }
}

/// The rectangle on the floor that holds every collision element of
/// `robot`, its corners counterclockwise from the front right one.
fn outline(robot: &Robot) -> Result<Footprint, BaseError> {
    let mut bounds: Option<[f64; 4]> = None;
    for link in &robot.links {
        let frame = link_frame(robot, &link.name);
        for collision in &link.collision {
            let placed = frame.then(Frame::of_origin(
                collision.origin.xyz.0,
                collision.origin.rpy.0,
            ));
            let [reach_x, reach_y] =
                reach(&collision.geometry, placed.orientation).ok_or_else(|| {
                    BaseError::OutlineShape {
                        link: link.name.clone(),
                    }
                })?;
            let [cx, cy, _] = placed.position;
            let part = [cx + reach_x, cx - reach_x, cy + reach_y, cy - reach_y];
            bounds = Some(bounds.map_or(part, |[front, rear, left, right]| {
                [
                    front.max(part[0]),
                    rear.min(part[1]),
                    left.max(part[2]),
                    right.min(part[3]),
                ]
            }));
        }
    }
    let [front, rear, left, right] = bounds.ok_or(BaseError::NoOutline)?;
    Footprint::parse(&[[front, right], [front, left], [rear, left], [rear, right]])
        .map_err(BaseError::Footprint)
}

#[cfg(test)]
mod tests;
