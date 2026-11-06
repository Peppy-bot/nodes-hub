//! The robot's own body as the self model of the arms a plan moves: the pairs of the
//! self-collision model that a moving arm's bodies ride, with the moving arms at the state
//! asked about and an arm the plan does not move held where it stands. A plan asks from
//! each of its attempts' threads at once, so each asks a copy of its own.

use std::num::NonZeroUsize;
use std::sync::{Mutex, MutexGuard, PoisonError, TryLockError};

use motion_planning_core::{Arms, Moving, SelfModel};
use openarm_description::{ARM_DOF, DUAL_DOF, Side};
use self_collision_model::{CollisionError, Proximity, SelfCollisionModel};

/// Copies of one snapshot of the robot's self-collision model: as many threads as that ask
/// at once, each on a copy no other holds.
struct Copies(Vec<Mutex<SelfCollisionModel>>);

impl Copies {
    fn new(model: &SelfCollisionModel, copies: NonZeroUsize) -> Self {
        Self(
            (0..copies.get())
                .map(|_| Mutex::new(model.clone()))
                .collect(),
        )
    }

    /// A copy no other thread holds; the first copy, once free, when every copy is held.
    fn free_copy(&self) -> MutexGuard<'_, SelfCollisionModel> {
        self.0
            .iter()
            .find_map(|copy| match copy.try_lock() {
                Ok(guard) => Some(guard),
                Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
                Err(TryLockError::WouldBlock) => None,
            })
            .unwrap_or_else(|| self.0[0].lock().unwrap_or_else(PoisonError::into_inner))
    }
}

/// The smallest distance of `nearest`; a configuration the model cannot measure, a
/// non-finite one, reads as a collision.
fn clearance_of(nearest: Result<Proximity<'_>, CollisionError>) -> f64 {
    nearest.map_or(f64::NEG_INFINITY, |proximity| proximity.distance)
}

/// The links of the pair of `nearest`; a configuration the model cannot measure names why
/// in place of the pair.
fn pair_of(nearest: Result<Proximity<'_>, CollisionError>) -> (String, String) {
    match nearest {
        Ok(proximity) => (proximity.link_a.to_string(), proximity.link_b.to_string()),
        Err(e) => (
            format!("the self model ({e})"),
            "the planned arms".to_string(),
        ),
    }
}

/// The self model of a plan of the arm on `side`, the other arm held at `other`: the
/// checked pairs the moving arm rides.
pub struct HeldOtherArm {
    copies: Copies,
    side: Side,
    other: [f64; ARM_DOF],
}

impl HeldOtherArm {
    /// The self model over `copies` copies of `model`.
    pub fn new(
        model: &SelfCollisionModel,
        copies: NonZeroUsize,
        side: Side,
        other: [f64; ARM_DOF],
    ) -> Self {
        Self {
            copies: Copies::new(model, copies),
            side,
            other,
        }
    }

    /// Runs `read` on the moving arm's nearest pair with the arm at `q`.
    fn nearest<T>(
        &self,
        q: &[f64; ARM_DOF],
        read: impl FnOnce(Result<Proximity<'_>, CollisionError>) -> T,
    ) -> T {
        let (left, right) = match self.side {
            Side::Left => (q, &self.other),
            Side::Right => (&self.other, q),
        };
        let mut model = self.copies.free_copy();
        read(model.min_distance_of_arm(self.side, left, right))
    }
}

impl SelfModel<ARM_DOF> for HeldOtherArm {
    fn clearance(&self, q: &[f64; ARM_DOF], _: Moving) -> f64 {
        self.nearest(q, clearance_of)
    }

    fn nearest_pair(&self, q: &[f64; ARM_DOF], _: Moving) -> (String, String) {
        self.nearest(q, pair_of)
    }
}

/// The self model of a plan of both arms, a state holding the left arm's joints and then
/// the right's: every checked pair, or those a turn of one arm judges, the left arm's
/// without the right arm and the right arm's with the left.
pub struct BothArms {
    copies: Copies,
}

impl BothArms {
    /// The self model over `copies` copies of `model`.
    pub fn new(model: &SelfCollisionModel, copies: NonZeroUsize) -> Self {
        Self {
            copies: Copies::new(model, copies),
        }
    }

    /// Runs `read` on the robot's nearest pair that `moving` judges, with the arms at `q`.
    fn nearest<T>(
        &self,
        q: &[f64; DUAL_DOF],
        moving: Moving,
        read: impl FnOnce(Result<Proximity<'_>, CollisionError>) -> T,
    ) -> T {
        let (left, right) = (
            Arms::<ARM_DOF, DUAL_DOF>::slice(q, 0),
            Arms::<ARM_DOF, DUAL_DOF>::slice(q, 1),
        );
        let mut model = self.copies.free_copy();
        read(match moving {
            Moving::All => model.min_distance(&left, &right),
            Moving::Arm(0) => model.min_distance_of_arm_alone(Side::Left, &left, &right),
            Moving::Arm(1) => model.min_distance_of_arm(Side::Right, &left, &right),
            Moving::Arm(k) => panic!("arm {k} of a plan of the two arms"),
        })
    }
}

impl SelfModel<DUAL_DOF> for BothArms {
    fn clearance(&self, q: &[f64; DUAL_DOF], moving: Moving) -> f64 {
        self.nearest(q, moving, clearance_of)
    }

    fn nearest_pair(&self, q: &[f64; DUAL_DOF], moving: Moving) -> (String, String) {
        self.nearest(q, moving, pair_of)
    }
}

#[cfg(test)]
mod tests {
    use openarm_description::{home, ready};

    use super::*;
    use crate::robot::Robot;

    // A turn of the left arm, first in the state, reads its pairs without the right arm,
    // and a turn of the right arm its pairs with the left: the left arm at Ready and the
    // right at its home posture, where those stand nearest at different distances.
    #[test]
    fn a_turn_of_one_arm_of_both_reads_its_pairs_with_the_arms_before_it() {
        let robot = Robot::OpenArmV2.model().expect("OpenArm v2's model");
        let (left, right) = (ready(Side::Left), home(Side::Right));
        let q = Arms::<ARM_DOF, DUAL_DOF>::join(&[left, right]);
        let both = BothArms::new(&robot.self_model, NonZeroUsize::MIN);
        let mut model = robot.self_model.clone();
        let left_alone = model
            .min_distance_of_arm_alone(Side::Left, &left, &right)
            .expect("a measured state")
            .distance;
        let of_right = model
            .min_distance_of_arm(Side::Right, &left, &right)
            .expect("a measured state")
            .distance;
        let of_all = model
            .min_distance(&left, &right)
            .expect("a measured state")
            .distance;
        assert_ne!(left_alone, of_right);
        assert_eq!(both.clearance(&q, Moving::Arm(0)), left_alone);
        assert_eq!(both.clearance(&q, Moving::Arm(1)), of_right);
        assert_eq!(both.clearance(&q, Moving::All), of_all);
    }
}
