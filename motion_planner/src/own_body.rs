//! The robot's own body as a plan meets it: its self-collision model, of `K` arms of `N`
//! joints, which places the grippers' fingers at their openings, gives the self model of the
//! arms a plan moves, and places the pieces of the robot that stand still, which an object
//! an arm carries keeps clear of.

use std::num::NonZeroUsize;

use collision_geometry::chain::{ChainGeometry, Rider};
use collision_geometry::gjk::Placed;
use robot_description::{ArmId, PerArm};
use self_collision_model::SelfCollisionModel;

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
    pub fn set_openings(&mut self, openings: &PerArm<f64, K>) {
        self.model.set_gripper_openings(&openings.into_array());
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
        standing: &ArmJoints<N, K>,
    ) -> HeldOtherArms<N, K> {
        HeldOtherArms::new(&self.model, copies, arm, standing.into_array())
    }

    /// The self model of a plan of every arm, a state of `D` joints holding each arm's in
    /// turn, over `copies` copies of the model.
    pub fn of_every_arm<const D: usize>(&self, copies: NonZeroUsize) -> EveryArm<N, K, D> {
        EveryArm::new(&self.model, copies)
    }

    /// Each piece of the robot with its arms at `standing`, by the name of its link.
    pub fn standing_pieces(&mut self, standing: &ArmJoints<N, K>) -> Vec<(&str, Placed<'_>)> {
        self.model
            .world_pieces(&standing.into_array())
            .expect("the history holds finite joint positions")
            .into_iter()
            .flat_map(|(link, pieces)| pieces.into_iter().map(move |piece| (link, piece)))
            .collect()
    }
}
