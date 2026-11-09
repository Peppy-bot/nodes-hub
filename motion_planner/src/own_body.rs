//! The robot's own body as a plan meets it: its self-collision model, of `K` arms of `N`
//! joints, which places the grippers' fingers at their openings, gives the self model of the
//! arms a plan moves, and places the pieces of the robot that stand still, which an object
//! an arm carries keeps clear of.

use std::num::NonZeroUsize;

use collision_geometry::chain::{ChainGeometry, Rider};
use collision_geometry::gjk::Placed;
use self_collision_model::SelfCollisionModel;

use crate::arm_set::{ArmId, PerArm};
use crate::fingers::FingerFault;
use crate::history::ArmJoints;
use crate::self_model::{EveryArm, HeldOtherArms};

/// A robot's own body: its self-collision model, of `K` arms of `N` joints, an arm's chain
/// at the arm's place in the robot's model.
#[derive(Clone)]
pub struct OwnBody<const N: usize, const K: usize> {
    pub model: SelfCollisionModel<N, K>,
}

impl<const N: usize, const K: usize> OwnBody<N, K> {
    /// Puts the grippers at `openings`.
    pub fn set_openings(&mut self, openings: &PerArm<f64>) {
        self.model.set_gripper_openings(&in_model_order(openings));
    }

    /// The fingers of `arm`, whose links `geometry` holds, as riders of its links at its
    /// gripper's opening.
    pub fn fingers<'g>(
        &self,
        arm: ArmId,
        geometry: &'g ChainGeometry<N>,
    ) -> Result<Vec<Rider<'g>>, FingerFault> {
        Ok(self.model.fingers(arm.place(), geometry.carried())?)
    }

    /// The self model of a plan of `arm`, every other arm held at `standing`, over `copies`
    /// copies of the model.
    pub fn of_one_arm(
        &self,
        copies: NonZeroUsize,
        arm: ArmId,
        standing: &ArmJoints<N>,
    ) -> HeldOtherArms<N, K> {
        HeldOtherArms::new(&self.model, copies, arm, in_model_order(standing))
    }

    /// The self model of a plan of every arm, a state of `D` joints holding each arm's in
    /// turn, over `copies` copies of the model.
    pub fn of_every_arm<const D: usize>(&self, copies: NonZeroUsize) -> EveryArm<N, K, D> {
        EveryArm::new(&self.model, copies)
    }

    /// Each piece of the robot with its arms at `standing`, by the name of its link.
    pub fn standing_pieces(&mut self, standing: &ArmJoints<N>) -> Vec<(&str, Placed<'_>)> {
        self.model
            .world_pieces(&in_model_order(standing))
            .expect("the history holds finite joint positions")
            .into_iter()
            .flat_map(|(link, pieces)| pieces.into_iter().map(move |piece| (link, piece)))
            .collect()
    }
}

/// The value of each of the `K` arms of `values`, an arm's at its place in the robot's model.
fn in_model_order<T: Copy, const K: usize>(values: &PerArm<T>) -> [T; K] {
    assert_eq!(
        values.iter().count(),
        K,
        "a value for each arm of the robot"
    );
    std::array::from_fn(|place| values[ArmId::new(place)])
}
