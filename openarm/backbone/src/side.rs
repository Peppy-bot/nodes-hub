//! A side of the robot and a value held once per side. The backbone works on
//! the two arms together almost everywhere (governing, streaming, the per-tick
//! math), so it keeps per-arm values in an [`ArmPair`], indexed by [`Side`].

use std::ops::{Index, IndexMut};

/// Which arm a message addresses. The wire carries the limb's name
/// ("left_arm" / "right_arm"); [`Side::from_arm_name`] parses that at the
/// boundary so the rest of the backbone carries a side it cannot get wrong,
/// never a raw string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    /// Both sides, left then right: the order of every per-arm table on the
    /// wire, and the order [`ArmPair::from_fn`] builds a pair in.
    pub const BOTH: [Side; 2] = [Side::Left, Side::Right];

    /// Refusal for a goal naming no arm of this robot.
    pub const UNKNOWN_ARM_NAME: &'static str =
        r#"unknown arm_name: this robot's arms are "left_arm" and "right_arm""#;

    /// Refusal for a goal naming no gripper of this robot.
    pub const UNKNOWN_GRIPPER_NAME: &'static str =
        r#"unknown gripper_name: this robot's grippers are "left_gripper" and "right_gripper""#;

    /// Parse a wire `arm_name`, or `None` for a name no side's
    /// [`Self::arm_name`] gives.
    pub fn from_arm_name(arm_name: &str) -> Option<Self> {
        Side::BOTH
            .into_iter()
            .find(|side| side.arm_name() == arm_name)
    }

    /// Parse a wire `gripper_name`, or `None` for a name no side's
    /// [`Self::gripper_name`] gives.
    pub fn from_gripper_name(gripper_name: &str) -> Option<Self> {
        Side::BOTH
            .into_iter()
            .find(|side| side.gripper_name() == gripper_name)
    }

    /// This side in the description's vocabulary.
    pub fn model(self) -> openarm_description::Side {
        match self {
            Side::Left => openarm_description::Side::Left,
            Side::Right => openarm_description::Side::Right,
        }
    }

    /// This side's arm name on the wire: the value limb_motion goals carry
    /// and limb_state's arm_names reports.
    pub fn arm_name(self) -> &'static str {
        match self {
            Side::Left => "left_arm",
            Side::Right => "right_arm",
        }
    }

    /// This side's gripper name on the wire, in the same role as
    /// [`Self::arm_name`].
    pub fn gripper_name(self) -> &'static str {
        match self {
            Side::Left => "left_gripper",
            Side::Right => "right_gripper",
        }
    }

    /// The side's name in logs and in the messages of refusals and failures.
    pub fn label(self) -> &'static str {
        match self {
            Side::Left => "left",
            Side::Right => "right",
        }
    }
}

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

impl<T> ArmPair<Option<T>> {
    /// Both values, or `None` when either side holds none.
    pub fn transpose(self) -> Option<ArmPair<T>> {
        Some(ArmPair::new(self.left?, self.right?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn side_model_maps_this_crate_onto_the_description() {
        assert_eq!(Side::Left.model(), openarm_description::Side::Left);
        assert_eq!(Side::Right.model(), openarm_description::Side::Right);
    }

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
    fn transpose_gives_both_values_or_none() {
        assert_eq!(
            ArmPair::new(Some(1), Some(2)).transpose(),
            Some(ArmPair::new(1, 2))
        );
        assert_eq!(ArmPair::new(Some(1), None).transpose(), None);
        assert_eq!(ArmPair::new(None, Some(2)).transpose(), None);
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
