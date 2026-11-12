//! The grippers' fingers where they hang at the grippers' openings: the parts each arm's
//! chain carries, placed by the robot's own body at its gripper's opening. A plan carries the
//! planned arm's fingers as riders of its hand, with the shapes of the objects the
//! planning scene attaches to its grasp link, a goal may name a finger, and the map cuts
//! every finger and every attached object out of a depth frame. The robot's model
//! refuses fingers that do not fit their arms when it is built.

use collision_geometry::chain::Rider;
use motion_planning_core::{PlanningScene, RiderOffTheArm};
use robot_description::PerArm;
use self_collision_model::FingerError;

use crate::own_body::OwnBody;
use crate::robot::ArmModel;

/// Why a finger of the robot's own body does not fit its arm: the model does not hang it off
/// the link that carries it, or it rides no joint of the arm.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum FingerFault {
    #[error(transparent)]
    Placed(#[from] FingerError),
    #[error(transparent)]
    OffTheArm(#[from] RiderOffTheArm),
}

/// The fingers of every arm of `arms` at the openings of `own`.
pub fn riders_of_the_arms<'g, const N: usize, const K: usize>(
    own: &OwnBody<N, K>,
    arms: &'g PerArm<ArmModel<N>, K>,
) -> Result<PerArm<Vec<Rider<'g>>, K>, FingerFault> {
    PerArm::from_fn(|arm| own.fingers(arm, &arms[arm].geometry)).transpose()
}

/// Every body the arm `arm` carries: `fingers`, its fingers, and the shapes of the objects
/// `scene` attaches to its grasp link.
pub fn with_held_objects<'a, const N: usize>(
    fingers: Vec<Rider<'a>>,
    arm: &'a ArmModel<N>,
    scene: &'a PlanningScene,
) -> Vec<Rider<'a>> {
    fingers
        .into_iter()
        .chain(scene.riders(&arm.grasp_link, &arm.geometry))
        .collect()
}
