//! The grippers' fingers where they hang at the grippers' openings: the parts each arm's
//! chain carries, placed by the self model at its side's opening. A plan carries the
//! planned arm's fingers as riders of its hand, with the shapes of the objects the
//! planning scene attaches to its grasp link, a goal may name a finger, and the map cuts
//! every finger and every attached object out of a depth frame. The robot's model
//! refuses fingers that do not fit their arms when it is built.

use collision_geometry::chain::{ChainGeometry, Rider};
use motion_planning_core::{PlanningScene, RiderOffTheArm};
use openarm_description::{ARM_DOF, ArmPair, Side};
use self_collision_model::{FingerError, SelfCollisionModel};

use crate::robot::ArmModel;

/// Why a finger of the self model does not fit its arm: the model does not hang it off the
/// link that carries it, or it rides no joint of the arm.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum FingerFault {
    #[error(transparent)]
    Placed(#[from] FingerError),
    #[error(transparent)]
    OffTheArm(#[from] RiderOffTheArm),
}

/// The fingers of the arm on `side`, whose links `geometry` holds, as riders of its links
/// at the opening of `model`.
pub fn riders<'g>(
    model: &SelfCollisionModel,
    geometry: &'g ChainGeometry<ARM_DOF>,
    side: Side,
) -> Result<Vec<Rider<'g>>, FingerError> {
    model.fingers(side, geometry.carried())
}

/// The fingers of both arms of `arms` at the openings of `model`.
pub fn riders_of_both_arms<'g>(
    model: &SelfCollisionModel,
    arms: &'g ArmPair<ArmModel>,
) -> Result<ArmPair<Vec<Rider<'g>>>, FingerError> {
    ArmPair::from_fn(|side| riders(model, &arms[side].geometry, side)).transpose()
}

/// Every body the arm `arm` carries: `fingers`, its fingers, and the shapes of the objects
/// `scene` attaches to its grasp link.
pub fn with_held_objects<'a>(
    fingers: Vec<Rider<'a>>,
    arm: &'a ArmModel,
    scene: &'a PlanningScene,
) -> Vec<Rider<'a>> {
    fingers
        .into_iter()
        .chain(scene.riders(arm.grasp_link, &arm.geometry))
        .collect()
}
