//! A value held once per arm. The backbone works on the two arms together almost
//! everywhere (governing, streaming, the per-tick math), so it keeps per-arm
//! values in an [`ArmPair`], indexed by [`Side`].

use std::ops::{Index, IndexMut};

use crate::types::Side;

/// A `left`/`right` pair of the same type: joint vectors, planners, the per-arm
/// channel bundles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmPair<T> {
    pub left: T,
    pub right: T,
}

impl<T> ArmPair<T> {
    pub fn new(left: T, right: T) -> Self {
        Self { left, right }
    }

    /// The values, in [`Side::BOTH`] order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        [&self.left, &self.right].into_iter()
    }

    /// The pair holding `f(side)` for each side, in [`Side::BOTH`] order: `f`
    /// runs for the left side first.
    pub fn from_fn(mut f: impl FnMut(Side) -> T) -> Self {
        let left = f(Side::Left);
        let right = f(Side::Right);
        Self { left, right }
    }

    /// The pair holding `f(side, value)` for each side's value, in
    /// [`Side::BOTH`] order.
    pub fn map<U>(self, mut f: impl FnMut(Side, T) -> U) -> ArmPair<U> {
        let left = f(Side::Left, self.left);
        let right = f(Side::Right, self.right);
        ArmPair { left, right }
    }

    /// The pair holding each side's value of `self` and of `other` together.
    pub fn zip<U>(self, other: ArmPair<U>) -> ArmPair<(T, U)> {
        ArmPair::new((self.left, other.left), (self.right, other.right))
    }
}

impl<A, B> ArmPair<(A, B)> {
    /// The pair of first values and the pair of second values.
    pub fn unzip(self) -> (ArmPair<A>, ArmPair<B>) {
        let ArmPair {
            left: (left_a, left_b),
            right: (right_a, right_b),
        } = self;
        (ArmPair::new(left_a, right_a), ArmPair::new(left_b, right_b))
    }
}

impl<T> Index<Side> for ArmPair<T> {
    type Output = T;

    fn index(&self, side: Side) -> &T {
        match side {
            Side::Left => &self.left,
            Side::Right => &self.right,
        }
    }
}

impl<T> IndexMut<Side> for ArmPair<T> {
    fn index_mut(&mut self, side: Side) -> &mut T {
        match side {
            Side::Left => &mut self.left,
            Side::Right => &mut self.right,
        }
    }
}

impl<T, E> ArmPair<Result<T, E>> {
    /// Both values, or the first error in [`Side::BOTH`] order.
    pub fn transpose(self) -> Result<ArmPair<T>, E> {
        Ok(ArmPair::new(self.left?, self.right?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_side_reads_and_writes_its_own_value() {
        let mut pair = ArmPair::new("left", "right");
        assert_eq!(pair[Side::Left], "left");
        assert_eq!(pair[Side::Right], "right");
        pair[Side::Left] = "new left";
        assert_eq!(pair, ArmPair::new("new left", "right"));
        pair[Side::Right] = "new right";
        assert_eq!(pair, ArmPair::new("new left", "new right"));
    }

    #[test]
    fn iter_walks_the_values_left_first() {
        assert_eq!(ArmPair::new(1, 2).iter().collect::<Vec<_>>(), [&1, &2]);
    }

    #[test]
    fn transpose_gives_both_values_or_the_first_error() {
        let pair = |left: Result<i32, &'static str>, right| ArmPair::new(left, right);
        assert_eq!(pair(Ok(1), Ok(2)).transpose(), Ok(ArmPair::new(1, 2)));
        assert_eq!(pair(Ok(1), Err("right")).transpose(), Err("right"));
        assert_eq!(pair(Err("left"), Ok(2)).transpose(), Err("left"));
        assert_eq!(pair(Err("left"), Err("right")).transpose(), Err("left"));
    }

    #[test]
    fn from_fn_builds_each_side_from_its_own_side_left_first() {
        let mut calls = Vec::new();
        let pair = ArmPair::from_fn(|side| {
            calls.push(side);
            side.label()
        });
        assert_eq!(pair, ArmPair::new("left", "right"));
        assert_eq!(calls, Side::BOTH);
    }

    #[test]
    fn map_hands_each_value_its_own_side_left_first() {
        let mut calls = Vec::new();
        let pair = ArmPair::new(1, 2).map(|side, value| {
            calls.push(side);
            format!("{} {value}", side.label())
        });
        assert_eq!(
            pair,
            ArmPair::new("left 1".to_owned(), "right 2".to_owned())
        );
        assert_eq!(calls, Side::BOTH);
    }

    #[test]
    fn zip_and_unzip_keep_each_value_on_its_side() {
        let zipped = ArmPair::new(1, 2).zip(ArmPair::new("a", "b"));
        assert_eq!(zipped, ArmPair::new((1, "a"), (2, "b")));
        assert_eq!(zipped.unzip(), (ArmPair::new(1, 2), ArmPair::new("a", "b")));
    }
}
