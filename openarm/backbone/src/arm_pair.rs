//! A value held once per arm. The backbone works on the two arms together almost
//! everywhere (governing, streaming, the per-tick math), so a named `left`/`right`
//! pair reads far better than `[_; 2]` indexed by 0/1 or a bare tuple.

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
}
