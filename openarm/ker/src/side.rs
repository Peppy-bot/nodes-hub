// The per-side vocabulary this node shares: which arms are in a state, one
// value per arm, and the spelling of a side in the operator log.

use openarm_description::Side;

/// One flag per arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SideFlags {
    pub left: bool,
    pub right: bool,
}

impl SideFlags {
    /// Neither side set.
    pub const NONE: Self = Self {
        left: false,
        right: false,
    };

    pub fn side(self, side: Side) -> bool {
        match side {
            Side::Left => self.left,
            Side::Right => self.right,
        }
    }

    pub(crate) fn set(&mut self, side: Side, value: bool) {
        match side {
            Side::Left => self.left = value,
            Side::Right => self.right = value,
        }
    }
}

/// One value per arm.
#[derive(Debug, Clone, Copy)]
pub struct SideValues {
    pub left: f64,
    pub right: f64,
}

impl SideValues {
    pub(crate) fn side(self, side: Side) -> f64 {
        match side {
            Side::Left => self.left,
            Side::Right => self.right,
        }
    }

    /// Both values scaled, which is how a trigger opening becomes the gripper
    /// opening commanded from it.
    pub(crate) fn scaled(self, factor: f64) -> Self {
        Self {
            left: self.left * factor,
            right: self.right * factor,
        }
    }
}

/// One arm's name for the operator log, the spelling every line in this node
/// uses.
pub(crate) fn label(side: Side) -> &'static str {
    match side {
        Side::Left => "left",
        Side::Right => "right",
    }
}
