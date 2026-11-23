//! The differential base a robot model's description gives, the model the
//! robot's initializer names on robot_identity's `get_identity`: its wheels
//! and the outline of its parts on the floor. Each is read from the model's
//! URDF in the frame of its root link, the robot frame, base_footprint, with
//! every joint at zero:
//!
//! - the wheels are the description's differential base
//!   ([`RobotDescription::differential_base`]): its two wheel joints, each
//!   wheel's radius and their separation;
//! - the outline is the description's ([`RobotDescription::outline`]): the
//!   rectangle on the floor that holds every collision element of every link,
//!   boxes, cylinders and spheres.

use control_core::footprint::{Footprint, FootprintError};
use control_core::odometry::{WheelGeometry, WheelGeometryError};
use robot_description::{Model, NoDifferentialBase, NoOutline, RobotDescription};

use crate::transforms::ROBOT_FRAME;

/// What a model's description gives the base.
#[derive(Debug)]
pub struct BaseModel {
    pub wheels: WheelGeometry,
    /// The outline's corners, counterclockwise from the front right one.
    pub footprint: Footprint,
    /// The wheel joints, [left, right], whose transforms the node publishes.
    pub wheel_joints: [String; 2],
    /// The description, which places the wheel joints' links.
    pub description: RobotDescription,
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
    #[error(transparent)]
    Wheels(#[from] WheelGeometryError),
    #[error(transparent)]
    Outline(#[from] NoOutline),
    #[error("the outline: {0}")]
    Footprint(FootprintError),
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
        let outline = description.outline()?;
        Ok(BaseModel {
            wheels: WheelGeometry::parse(
                base.separation_m,
                base.left.radius_m,
                base.right.radius_m,
            )?,
            footprint: Footprint::parse(&outline.corners()).map_err(BaseError::Footprint)?,
            wheel_joints: [base.left.joint, base.right.joint],
            description,
        })
    }
}

#[cfg(test)]
mod tests;
