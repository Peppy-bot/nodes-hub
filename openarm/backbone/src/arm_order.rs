//! The arms a request names, in its order: the order its joint arrays hold the arms'
//! joints in, [`ARM_DOF`] for each arm.

use control_core::requested_arms::{common_waypoint_count, places_in_order};
use robot_description::{ArmId, PerArm};
use srs_model::{ARM_DOF, JointVec};

use crate::robot::{ARMS, LIMBS};

/// The arms a request names, in its order, each once.
#[derive(Clone, Debug)]
pub struct ArmOrder(Vec<ArmId>);

impl ArmOrder {
    /// Parse the arm names of a request's `arms`, in their order: one or more
    /// of this robot's arm names, none named twice. A refusal names what is
    /// wrong in the words of the field.
    pub fn parse(arm_names_requested: &[&str]) -> Result<Self, String> {
        let places = places_in_order(arm_names_requested, &LIMBS.arm_names())?;
        Ok(Self(places.into_iter().map(ArmId::new).collect()))
    }

    /// The arms, in the request's order.
    pub fn ids(&self) -> &[ArmId] {
        &self.0
    }

    /// How many joint positions an array in this order holds.
    fn joint_count(&self) -> usize {
        self.0.len() * ARM_DOF
    }

    /// The number of waypoints every arm of this order holds, given each
    /// arm's `counts` in this order: refused unless the first holds one or
    /// more and every other as many.
    ///
    /// # Panics
    ///
    /// When `counts` does not hold one count for each arm of this order.
    pub fn waypoint_count(&self, counts: &[usize]) -> Result<usize, String> {
        assert_eq!(counts.len(), self.0.len(), "one count for each arm");
        let named: Vec<(&str, usize)> = self
            .0
            .iter()
            .map(|&arm| LIMBS.arm_name(arm))
            .zip(counts.iter().copied())
            .collect();
        common_waypoint_count(&named)
    }

    /// Each arm of this order with its joints in `array`, an array in this
    /// order.
    ///
    /// # Panics
    ///
    /// When `array` does not hold [`Self::joint_count`] joints.
    fn split<'a>(&'a self, array: &'a [f64]) -> impl Iterator<Item = (ArmId, JointVec)> + 'a {
        assert_eq!(array.len(), self.joint_count(), "an array in this order");
        self.0
            .iter()
            .zip(array.chunks_exact(ARM_DOF))
            .map(|(&arm, joints)| (arm, joints.try_into().expect("an arm's share of the array")))
    }

    /// The joints of `arm` in `array`, an array in this order: none for an
    /// arm this order does not name.
    pub fn joints_of(&self, array: &[f64], arm: ArmId) -> Option<JointVec> {
        self.split(array)
            .find_map(|(named, joints)| (named == arm).then_some(joints))
    }

    /// `held` with each arm of this order at its joints in `array`.
    pub fn placed(&self, array: &[f64], held: &PerArm<JointVec, ARMS>) -> PerArm<JointVec, ARMS> {
        PerArm::from_fn(|arm| self.joints_of(array, arm).unwrap_or(held[arm]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{LEFT, RIGHT};

    /// Where the tests hold the arms, and where an array takes them.
    const HELD: PerArm<JointVec, ARMS> = PerArm::new([
        [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0],
        [0.0, 0.8, 0.0, 1.2, 0.0, 0.0, 0.0],
    ]);
    const TO: PerArm<JointVec, ARMS> = PerArm::new([
        [0.0, -1.2, 0.0, 1.2, 0.0, 0.0, 0.0],
        [0.0, 0.8, 0.0, 1.3, 0.0, 0.0, 0.0],
    ]);

    fn order(names: &[&str]) -> ArmOrder {
        ArmOrder::parse(names).expect("the arms of this robot, each once")
    }

    /// The array in `order` of the arms at `joints`.
    fn array_of(order: &ArmOrder, joints: &PerArm<JointVec, ARMS>) -> Vec<f64> {
        order.ids().iter().flat_map(|&arm| joints[arm]).collect()
    }

    /// A request's arms are this robot's, by the names its arms go by, in the request's
    /// order; a refusal names this robot's arms in their words. The rules of a list of arms
    /// and of their waypoints are control_core's (`requested_arms`).
    #[test]
    fn a_request_names_this_robots_arms_in_its_order() {
        let [left, right] = LIMBS.arm_names();
        assert_eq!(order(&[right]).ids(), [RIGHT]);
        assert_eq!(order(&[right, left]).ids(), [RIGHT, LEFT]);
        assert_eq!(
            ArmOrder::parse(&["head"]).expect_err("no arm of this robot"),
            format!(
                "arms names \"head\", no arm of this robot: its arms are \"{left}\" and \
                 \"{right}\""
            )
        );
        assert_eq!(
            order(&[right, left]).waypoint_count(&[2, 0]),
            Err(format!(
                "{left} holds 0 waypoints, where {right} holds 2: every arm passes through as \
                 many"
            ))
        );
    }

    /// An array in an order holds the named arms' joints in that order: named
    /// right first, its first joints are the right arm's, and the arms it does
    /// not name keep where they are held.
    #[test]
    fn an_array_holds_each_named_arm_in_the_order_of_the_requests_arms() {
        let [left, right] = LIMBS.arm_names();
        let right_first = order(&[right, left]);
        let array = array_of(&right_first, &TO);
        assert_eq!(array[..ARM_DOF], TO[RIGHT]);
        assert_eq!(array[ARM_DOF..], TO[LEFT]);
        assert_eq!(right_first.joint_count(), ARMS * ARM_DOF);
        assert_eq!(right_first.placed(&array, &HELD), TO);

        let left_only = order(&[left]);
        let array = array_of(&left_only, &TO);
        assert_eq!(array, TO[LEFT]);
        assert_eq!(
            left_only.placed(&array, &HELD),
            PerArm::new([TO[LEFT], HELD[RIGHT]])
        );
        assert_eq!(left_only.joints_of(&array, RIGHT), None);
    }
}
