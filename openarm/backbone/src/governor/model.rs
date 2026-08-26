//! The governor's only door to the collision model.
//!
//! The underlying model keys distance reads off an ambient finger placement
//! (`set_gripper_openings` then `min_distance`), which makes "query at a stale
//! placement" representable. This wrapper removes that state from the
//! governor's vocabulary: the raw model is private, every query takes the
//! whole governed configuration, and the placement happens inside the call
//! that uses it. A read at the wrong placement is unrepresentable here, not
//! merely avoided.
//!
//! The model still takes `&mut self` (its query scratch is the library's), so
//! queries are sequential by construction; what this boundary adds is that
//! their *order* no longer matters.

use std::time::Instant;

use bimanual_collision_model::{BimanualCollisionModel, CollisionError, DistanceGradient};
use tracing::warn;

use crate::streams::warn_throttled;

use super::{GOV_DOF, JointVec, NearestPair, split};

pub(super) struct ConfiguredModel {
    inner: BimanualCollisionModel,
    /// Throttles the failed-query log. A failed query stays fail-safe (`None`
    /// reads as a breach in the scan and as "no pair" in the readout), but a
    /// persistently failing model must be visible in the log, not
    /// indistinguishable from geometry.
    last_query_error: Option<Instant>,
}

impl ConfiguredModel {
    pub fn new(inner: BimanualCollisionModel) -> Self {
        Self {
            inner,
            last_query_error: None,
        }
    }

    fn min_distance_at(
        &mut self,
        q: &[f64; GOV_DOF],
    ) -> Option<bimanual_collision_model::Proximity<'_>> {
        let s = split(q);
        self.inner
            .set_gripper_openings(s.grippers.left, s.grippers.right);
        match self.inner.min_distance(&s.arms.left, &s.arms.right) {
            Ok(p) => Some(p),
            Err(e) => {
                warn_throttled(&mut self.last_query_error, || {
                    warn!("collision model query failed (treated as a breach): {e}");
                });
                None
            }
        }
    }

    /// Signed clearance at a governed configuration. `None` on a query error
    /// so callers fail safe.
    pub fn distance(&mut self, q: &[f64; GOV_DOF]) -> Option<f64> {
        self.min_distance_at(q).map(|p| p.distance)
    }

    /// The nearest checked pair at a governed configuration, for the readout.
    pub fn proximity(&mut self, q: &[f64; GOV_DOF]) -> Option<NearestPair> {
        self.min_distance_at(q).map(|p| NearestPair {
            distance: p.distance,
            link_a: p.link_a.to_string(),
            link_b: p.link_b.to_string(),
        })
    }

    /// `d(clearance)/d(dof)` at a governed configuration, with the proximity
    /// it was taken at.
    pub fn gradient(&mut self, q: &[f64; GOV_DOF]) -> Result<DistanceGradient<'_>, CollisionError> {
        let s = split(q);
        self.inner
            .set_gripper_openings(s.grippers.left, s.grippers.right);
        self.inner.distance_gradient(&s.arms.left, &s.arms.right)
    }

    /// Upper bound (m) on how much the clearance can change over a step.
    /// Placement-free by the library's construction (precomputed levers), so
    /// it takes only the step.
    pub fn step_bound(&self, dq_left: &JointVec, dq_right: &JointVec, dopenings: &[f64; 2]) -> f64 {
        self.inner
            .clearance_step_bound(dq_left, dq_right, dopenings)
    }
}
