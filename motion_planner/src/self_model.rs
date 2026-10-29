//! The robot's own body as the planned arm's self model: the pairs of the
//! self-collision model that the planned arm's bodies ride, with the planned arm at the
//! configuration asked about and the other arm held where it stands. A plan asks from
//! each of its attempts' threads at once, so each asks a copy of its own.

use std::num::NonZeroUsize;
use std::sync::{Mutex, MutexGuard, PoisonError, TryLockError};

use motion_planning_core::SelfModel;
use openarm_description::{ARM_DOF, Side};
use self_collision_model::{CollisionError, Proximity, SelfCollisionModel};

/// The self model of the arm on `side`, the other arm at `other`, over copies of one
/// snapshot of the robot's self-collision model.
pub struct HeldOtherArm {
    copies: Vec<Mutex<SelfCollisionModel>>,
    side: Side,
    other: [f64; ARM_DOF],
}

impl HeldOtherArm {
    /// The self model over `copies` copies of `model`: as many threads as that ask at
    /// once, each on a copy no other holds.
    pub fn new(
        model: &SelfCollisionModel,
        copies: NonZeroUsize,
        side: Side,
        other: [f64; ARM_DOF],
    ) -> Self {
        Self {
            copies: (0..copies.get())
                .map(|_| Mutex::new(model.clone()))
                .collect(),
            side,
            other,
        }
    }

    /// A copy no other thread holds; the first copy, once free, when every copy is held.
    fn free_copy(&self) -> MutexGuard<'_, SelfCollisionModel> {
        self.copies
            .iter()
            .find_map(|copy| match copy.try_lock() {
                Ok(guard) => Some(guard),
                Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
                Err(TryLockError::WouldBlock) => None,
            })
            .unwrap_or_else(|| {
                self.copies[0]
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
            })
    }

    /// Runs `read` on the planned arm's nearest pair with the arm at `q`.
    fn nearest<T>(
        &self,
        q: &[f64; ARM_DOF],
        read: impl FnOnce(Result<Proximity<'_>, CollisionError>) -> T,
    ) -> T {
        let (left, right) = match self.side {
            Side::Left => (q, &self.other),
            Side::Right => (&self.other, q),
        };
        let mut model = self.free_copy();
        read(model.min_distance_of_arm(self.side, left, right))
    }
}

impl SelfModel<ARM_DOF> for HeldOtherArm {
    /// The smallest distance over the checked pairs the planned arm rides; a
    /// configuration the model cannot measure, a non-finite one, reads as a collision.
    fn clearance(&self, q: &[f64; ARM_DOF]) -> f64 {
        self.nearest(q, |nearest| {
            nearest.map_or(f64::NEG_INFINITY, |proximity| proximity.distance)
        })
    }

    /// The links of the nearest pair; a configuration the model cannot measure names
    /// why in place of the pair.
    fn nearest_pair(&self, q: &[f64; ARM_DOF]) -> (String, String) {
        self.nearest(q, |nearest| match nearest {
            Ok(proximity) => (proximity.link_a.to_string(), proximity.link_b.to_string()),
            Err(e) => (format!("the self model ({e})"), "the arm".to_string()),
        })
    }
}
