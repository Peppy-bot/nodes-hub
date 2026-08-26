//! What the leading node commands an arm with, and which kind this instance
//! listens to.
//!
//! An arm is led in joint space (`joint_link`) or Cartesian space
//! (`pose_link`), never both: [`UpstreamMode`] is parsed once at bringup and
//! decides which listener is spawned, so nothing downstream can observe a
//! contested arm and [`Upstream`] needs no third variant. A paired slot of
//! the unfollowed kind is never read; bringup warns when it sees one.

use std::fmt;
use std::str::FromStr;

use srs_model::nalgebra::Isometry3;

use crate::types::JointVec;

/// Which upstream slot kind this instance follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpstreamMode {
    /// The `joint_link` follower slots: the leader has already solved the arm.
    Joints,
    /// The `pose_link` follower slots: the leader streams a pose, this
    /// backbone solves it.
    Pose,
}

impl UpstreamMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Joints => "joints",
            Self::Pose => "pose",
        }
    }
}

/// A launcher `upstream_mode` this node has no listener for.
#[derive(Debug, thiserror::Error)]
#[error("unknown upstream_mode {0:?}: expected \"joints\" or \"pose\"")]
pub struct UnknownUpstreamMode(pub String);

impl FromStr for UpstreamMode {
    type Err = UnknownUpstreamMode;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "joints" => Ok(Self::Joints),
            "pose" => Ok(Self::Pose),
            other => Err(UnknownUpstreamMode(other.to_string())),
        }
    }
}

impl fmt::Display for UpstreamMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// One arm's latest upstream command. Both variants are parsed and finite at
/// the wire boundary, so the planner never re-validates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Upstream {
    /// Joint positions to chase directly, in the arm's fixed joint order.
    Joints(JointVec),
    /// A world-frame end-effector pose to servo toward.
    Pose(Isometry3<f64>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_modes_round_trip_through_their_labels() {
        for mode in [UpstreamMode::Joints, UpstreamMode::Pose] {
            assert_eq!(mode.label().parse::<UpstreamMode>().unwrap(), mode);
        }
    }

    #[test]
    fn an_unknown_mode_is_refused_and_names_what_it_expected() {
        let err = "cartesian".parse::<UpstreamMode>().unwrap_err();
        assert_eq!(err.0, "cartesian");
        let shown = err.to_string();
        assert!(
            shown.contains("joints") && shown.contains("pose"),
            "{shown}"
        );
        assert!("Joints".parse::<UpstreamMode>().is_err());
        assert!("".parse::<UpstreamMode>().is_err());
    }
}
