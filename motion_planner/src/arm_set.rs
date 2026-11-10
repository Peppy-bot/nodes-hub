//! The arms of a robot, one or more: an arm by its place in the robot's model, and a value
//! for each arm.

use std::ops::{Index, IndexMut};

use openarm_description::Side;

/// An arm of a robot, by its place in the robot's model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArmId(usize);

impl ArmId {
    pub const fn new(place: usize) -> Self {
        Self(place)
    }

    /// The arm's place in the robot's model.
    pub fn place(self) -> usize {
        self.0
    }

    /// The side this arm stands on in OpenArm's model, which holds the left arm first.
    pub fn side(self) -> Side {
        Side::BOTH[self.0]
    }
}

impl From<Side> for ArmId {
    /// The arm on `side` of OpenArm's model, which holds the left arm first.
    fn from(side: Side) -> Self {
        Self(side.index())
    }
}

/// A value for each arm of a robot, in the order of its model.
#[derive(Debug, Clone, PartialEq)]
pub struct PerArm<T>(Vec<T>);

impl<T> PerArm<T> {
    /// The values of a robot of as many arms, in the order of its model.
    pub fn new(values: Vec<T>) -> Self {
        Self(values)
    }

    /// The values of a robot of two arms.
    pub fn both(first: T, second: T) -> Self {
        Self(vec![first, second])
    }

    /// The robot's arms.
    pub fn ids(&self) -> impl Iterator<Item = ArmId> + use<T> {
        (0..self.0.len()).map(ArmId)
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    /// A value for each arm, from each arm and its value here.
    pub fn each<'a, U>(&'a self, mut f: impl FnMut(ArmId, &'a T) -> U) -> PerArm<U> {
        PerArm(
            self.0
                .iter()
                .enumerate()
                .map(|(place, value)| f(ArmId(place), value))
                .collect(),
        )
    }

    /// Each value mapped, with its arm.
    pub fn map<U>(self, mut f: impl FnMut(ArmId, T) -> U) -> PerArm<U> {
        PerArm(
            self.0
                .into_iter()
                .enumerate()
                .map(|(place, value)| f(ArmId(place), value))
                .collect(),
        )
    }
}

impl<T> IntoIterator for PerArm<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T, A: Into<ArmId>> Index<A> for PerArm<T> {
    type Output = T;

    fn index(&self, arm: A) -> &T {
        &self.0[arm.into().0]
    }
}

impl<T, A: Into<ArmId>> IndexMut<A> for PerArm<T> {
    fn index_mut(&mut self, arm: A) -> &mut T {
        &mut self.0[arm.into().0]
    }
}

impl<T> PerArm<Option<T>> {
    /// The values, where every arm has one.
    pub fn transpose(self) -> Option<PerArm<T>> {
        self.0.into_iter().collect::<Option<Vec<T>>>().map(PerArm)
    }
}

impl<T, E> PerArm<Result<T, E>> {
    /// The values, or the first arm's error.
    pub fn transpose(self) -> Result<PerArm<T>, E> {
        self.0
            .into_iter()
            .collect::<Result<Vec<T>, E>>()
            .map(PerArm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // An arm reads and writes its own value, and `each` and `map` keep the arms' order.
    #[test]
    fn each_arm_holds_its_own_value() {
        let mut values = PerArm::both(1, 2);
        let second = values.ids().last().expect("two arms");
        values[second] = 5;
        assert_eq!(values[ArmId::new(0)], 1);
        assert_eq!(values[ArmId::new(1)], 5);
        assert_eq!(
            values.each(|arm, value| (arm.0, *value)),
            PerArm::both((0, 1), (1, 5))
        );
        assert_eq!(values.map(|_, value| value * 2), PerArm::both(2, 10));
    }

    // Values gather where every arm has one, and an arm with none, or the first error, ends
    // the gathering.
    #[test]
    fn values_gather_where_every_arm_has_one() {
        assert_eq!(
            PerArm::both(Some(1), Some(2)).transpose(),
            Some(PerArm::both(1, 2))
        );
        assert_eq!(PerArm::both(Some(1), None).transpose(), None);
        assert_eq!(
            PerArm::both(Err::<i32, _>("first"), Err("second")).transpose(),
            Err("first")
        );
    }
}
