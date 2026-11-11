//! A robot's own body as the self model of the arms a plan moves: the pairs of the
//! self-collision model that a moving arm's bodies ride, with the moving arms at the state
//! asked about and an arm the plan does not move held where it stands. A plan asks from
//! each of its attempts' threads at once, so each asks a copy of its own.

use std::num::NonZeroUsize;
use std::sync::{Mutex, MutexGuard, PoisonError, TryLockError};

use motion_planning_core::{Arms, Moving, SelfModel};
use self_collision_model::{CollisionError, Joints, Proximity, SelfCollisionModel};

use crate::arm_set::ArmId;

/// Copies of one snapshot of the robot's self-collision model: as many threads as that ask
/// at once, each on a copy no other holds.
struct Copies<const N: usize, const K: usize>(Vec<Mutex<SelfCollisionModel<N, K>>>);

impl<const N: usize, const K: usize> Copies<N, K> {
    fn new(model: &SelfCollisionModel<N, K>, copies: NonZeroUsize) -> Self {
        Self(
            (0..copies.get())
                .map(|_| Mutex::new(model.clone()))
                .collect(),
        )
    }

    /// A copy no other thread holds; the first copy, once free, when every copy is held.
    fn free_copy(&self) -> MutexGuard<'_, SelfCollisionModel<N, K>> {
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

/// The distance `read` holds; a configuration the model cannot measure, a non-finite one,
/// reads as a collision.
fn clearance_of(read: Result<f64, CollisionError>) -> f64 {
    read.unwrap_or(f64::NEG_INFINITY)
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

/// The self model of a plan of one arm of a robot of `K` arms, every other arm held where it
/// stands: the checked pairs the moving arm rides.
pub struct HeldOtherArms<const N: usize, const K: usize> {
    copies: Copies<N, K>,
    arm: usize,
    standing: Joints<N, K>,
}

impl<const N: usize, const K: usize> HeldOtherArms<N, K> {
    /// The self model of a plan of `arm` over `copies` copies of `model`, every arm at its
    /// joints of `standing` but the one the plan moves.
    pub fn new(
        model: &SelfCollisionModel<N, K>,
        copies: NonZeroUsize,
        arm: ArmId,
        standing: Joints<N, K>,
    ) -> Self {
        Self {
            copies: Copies::new(model, copies),
            arm: arm.place(),
            standing,
        }
    }

    /// Every arm's joints, the moving arm at `q`.
    fn at(&self, q: &[f64; N]) -> Joints<N, K> {
        std::array::from_fn(|k| if k == self.arm { *q } else { self.standing[k] })
    }
}

impl<const N: usize, const K: usize> SelfModel<N> for HeldOtherArms<N, K> {
    fn clearance(&self, q: &[f64; N], _: Moving, within: f64) -> f64 {
        let mut model = self.copies.free_copy();
        clearance_of(model.min_distance_of_chain_within(self.arm, &self.at(q), within))
    }

    fn nearest_pair(&self, q: &[f64; N], _: Moving) -> (String, String) {
        let mut model = self.copies.free_copy();
        pair_of(model.min_distance_of_chain(self.arm, &self.at(q)))
    }
}

/// The self model of a plan of every arm of a robot of `K` arms, a state of `D` joints
/// holding each arm's in turn: every checked pair, or those a turn of one arm judges, the
/// last arm's with every other arm and, of two arms, the first's without the second.
pub struct EveryArm<const N: usize, const K: usize, const D: usize> {
    copies: Copies<N, K>,
}

impl<const N: usize, const K: usize, const D: usize> EveryArm<N, K, D> {
    /// The self model over `copies` copies of `model`.
    pub fn new(model: &SelfCollisionModel<N, K>, copies: NonZeroUsize) -> Self {
        const {
            assert!(D == N * K, "a state holds the joints of every arm");
            assert!(
                K <= 2,
                "a turn of one arm reads the last arm's pairs, and the first's of two"
            );
        };
        Self {
            copies: Copies::new(model, copies),
        }
    }

    /// Each arm's joints of the state `q`.
    fn at(q: &[f64; D]) -> Joints<N, K> {
        std::array::from_fn(|k| Arms::<N, D>::slice(q, k))
    }

    /// Refuses a turn of an arm whose pairs the self model reads for no turn.
    fn no_turn_of(k: usize) -> ! {
        panic!(
            "a turn of arm {k} of a plan of {K} arms: the self model reads the last arm's, \
             and the first's of two"
        )
    }
}

impl<const N: usize, const K: usize, const D: usize> SelfModel<D> for EveryArm<N, K, D> {
    fn clearance(&self, q: &[f64; D], moving: Moving, within: f64) -> f64 {
        let at = Self::at(q);
        let mut model = self.copies.free_copy();
        clearance_of(match moving {
            Moving::All => model.min_distance_within(&at, within),
            Moving::Arm(k) if k + 1 == K => model.min_distance_of_chain_within(k, &at, within),
            Moving::Arm(0) if K == 2 => model.min_distance_of_chain_alone_within(0, &at, within),
            Moving::Arm(k) => Self::no_turn_of(k),
        })
    }

    fn nearest_pair(&self, q: &[f64; D], moving: Moving) -> (String, String) {
        let at = Self::at(q);
        let mut model = self.copies.free_copy();
        pair_of(match moving {
            Moving::All => model.min_distance(&at),
            Moving::Arm(k) if k + 1 == K => model.min_distance_of_chain(k, &at),
            Moving::Arm(0) if K == 2 => model.min_distance_of_chain_alone(0, &at),
            Moving::Arm(k) => Self::no_turn_of(k),
        })
    }
}

#[cfg(test)]
mod tests {
    use openarm_description::{ARM_DOF, DUAL_DOF, Side, home, ready};
    use so101_description::ARM_DOF as SO101_DOF;

    use super::*;
    use crate::arm_set::PerArm;
    use crate::robot::{openarm_v2, so101};

    /// A bound under the distance of every pair of the robots at the postures of the tests.
    const UNDER_EVERY_PAIR_M: f64 = -1.0;

    /// Checks that `model` reads `exact`, the distance the self-collision model measures,
    /// with no bound, and at `UNDER_EVERY_PAIR_M` a lower bound under it, the model's
    /// reading at that bound.
    fn reads<const D: usize>(model: &dyn SelfModel<D>, q: &[f64; D], moving: Moving, exact: f64) {
        assert_eq!(model.clearance(q, moving, f64::INFINITY), exact);
        let bounded = model.clearance(q, moving, UNDER_EVERY_PAIR_M);
        assert!(
            (UNDER_EVERY_PAIR_M..exact).contains(&bounded),
            "{moving:?}: {bounded} against {exact}"
        );
    }

    // A turn of the left arm, first in the state, reads its pairs without the right arm,
    // and a turn of the right arm its pairs with the left: the left arm at Ready and the
    // right at its home posture, where those stand nearest at different distances.
    #[test]
    fn a_turn_of_one_arm_of_both_reads_its_pairs_with_the_arms_before_it() {
        let robot = openarm_v2().expect("OpenArm v2's model");
        let at = [ready(Side::Left), home(Side::Right)];
        let q = Arms::<ARM_DOF, DUAL_DOF>::join(&at);
        let both = robot.own.of_every_arm::<DUAL_DOF>(NonZeroUsize::MIN);
        let mut model = robot.own.model.clone();
        let distance = |nearest: Result<Proximity<'_>, CollisionError>| {
            nearest.expect("a measured state").distance
        };
        let left_alone = distance(model.min_distance_of_chain_alone(0, &at));
        let of_right = distance(model.min_distance_of_chain(1, &at));
        let of_all = distance(model.min_distance(&at));
        assert_ne!(left_alone, of_right);
        reads(&both, &q, Moving::Arm(0), left_alone);
        reads(&both, &q, Moving::Arm(1), of_right);
        reads(&both, &q, Moving::All, of_all);
    }

    // A plan of one arm reads the pairs it rides with the other arm where it stands, which
    // a plan of both reads for the arm last in the state.
    #[test]
    fn a_plan_of_one_arm_reads_its_pairs_with_the_other_arm_where_it_stands() {
        let robot = openarm_v2().expect("OpenArm v2's model");
        let at = [ready(Side::Left), home(Side::Right)];
        let held = robot.own.of_one_arm(
            NonZeroUsize::MIN,
            Side::Right.into(),
            &PerArm::new(at.to_vec()),
        );
        let mut model = robot.own.model.clone();
        let of_right = model
            .min_distance_of_chain(1, &at)
            .expect("a measured state")
            .distance;
        reads(&held, &at[1], Moving::All, of_right);
    }

    // SO-101's one arm reads every checked pair, planned alone or as every arm.
    #[test]
    fn a_plan_of_so101s_arm_reads_every_pair() {
        let robot = so101().expect("SO-101's model");
        let q = [0.3, -0.4, 0.9, 0.2, -0.1];
        let mut model = robot.own.model.clone();
        let of_all = model.min_distance(&[q]).expect("a measured state").distance;
        let alone = robot
            .own
            .of_one_arm(NonZeroUsize::MIN, ArmId::new(0), &PerArm::new(vec![q]));
        let every = robot.own.of_every_arm::<SO101_DOF>(NonZeroUsize::MIN);
        reads(&alone, &q, Moving::All, of_all);
        reads(&every, &q, Moving::Arm(0), of_all);
        reads(&every, &q, Moving::All, of_all);
    }
}
