//! The pre-projection model read: every clearance and gradient the limiters
//! and the projection decide on is sampled here into an immutable [`Sensed`]
//! before any of them run.
//!
//! The model rewrites its finger placement on every query
//! ([`BimanualCollisionModel::set_gripper_openings`]), so a design where
//! limiters query it directly would make their order load-bearing: whoever
//! asks last leaves the model configured for whoever asks next, and the
//! gradient becomes one taken at the wrong finger placement. Sampling once
//! removes that coupling for everything that decides on the snapshot.
//!
//! The clip stage still probes the model live: the floor scan's whole job is
//! to check configurations no snapshot can anticipate. That is safe because
//! [`super::model::ConfiguredModel`] is the only door to the model and every
//! query through it takes the whole configuration, so a read at a stale
//! placement is unrepresentable rather than merely avoided.

use bimanual_collision_model::CollisionError;
use tracing::error;

use super::limiters::Tripwire;
use super::{ARM_DOF, DUAL_DOF, GOV_DOF, GovState, Governor, LEFT_GRIPPER, concat};
use super::{NearestPair, RIGHT_GRIPPER};

/// Everything the collision stages read, sampled once at `prev`.
pub(super) struct Sensed {
    /// Signed surface clearance at `prev` (m; negative is penetration).
    pub d_prev: f64,
    /// `d(clearance)/d(dof)` at `prev`. `None` in deep penetration, where the
    /// witness points coincide and no separating direction exists: the barrier
    /// stands down there and the floor scan alone guards the step, so the
    /// operator can still drive out of the contact.
    pub grad: Option<[f64; GOV_DOF]>,
    /// The nearest checked pair at `prev`, for the log line and the readout.
    pub pair: NearestPair,
    /// `Some` only while the measured-state tripwire is latched.
    pub tripwire: Option<Tripwire>,
}

impl Governor {
    /// Sample the model at `prev` and, while the tripwire is armed, at the
    /// commanded configurations it needs. `None` when the clearance at `prev`
    /// cannot be obtained at all, which the caller turns into a hold.
    pub(super) fn sense(
        &mut self,
        prev: &GovState,
        cand: &GovState,
        measured: &GovState,
    ) -> Option<Sensed> {
        // The tripwire samples the measured state first; every later query in
        // this function re-places the fingers, so nothing may read the model
        // between these calls and their use.
        let tripwire = self.sense_tripwire(prev, cand, measured);

        match self.model.gradient(&concat(prev)) {
            Ok(g) => {
                let mut grad = [0.0; GOV_DOF];
                grad[..ARM_DOF].copy_from_slice(&g.grad_left);
                grad[ARM_DOF..DUAL_DOF].copy_from_slice(&g.grad_right);
                grad[LEFT_GRIPPER] = g.grad_openings[0];
                grad[RIGHT_GRIPPER] = g.grad_openings[1];
                Some(Sensed {
                    d_prev: g.proximity.distance,
                    grad: Some(grad),
                    pair: NearestPair {
                        distance: g.proximity.distance,
                        link_a: g.proximity.link_a.to_string(),
                        link_b: g.proximity.link_b.to_string(),
                    },
                    tripwire,
                })
            }
            Err(CollisionError::WitnessesCoincide { .. }) => {
                // Deep penetration. There is no gradient to steer on, but there
                // is still a distance to hold, so report the reading without one
                // rather than failing: freezing here would trap the operator
                // inside the collision.
                let pair = self.proximity(prev)?;
                Some(Sensed {
                    d_prev: pair.distance,
                    grad: None,
                    pair,
                    tripwire,
                })
            }
            Err(e) => {
                // NonFinite / NoPairs cannot arise from a finite, governed prev
                // with pairs configured; treat as a fault and hold rather than
                // steer on it.
                error!("collision: distance_gradient: {e}; holding");
                None
            }
        }
    }
}
