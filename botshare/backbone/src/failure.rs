//! Why the base refuses or fails a goal: a failure kind that peppy or the
//! base_motion contract declares, and a message for a person. drive_base and
//! move_base each have a generated `FailureKind` with these names, and the
//! admission and the control loop pass one failure for both, so the node
//! keeps its own kind and gives each member its own. A goal the robot's tree
//! cannot place is refused with the kind of the placement
//! (control_core's [`Unplaced`](control_core::robot_tree::Unplaced)), through
//! [`TargetFailure`].

use control_core::robot_tree;
use peppygen::exposed_actions::base_motion::{drive_base, move_base};

/// A failure kind of the base's goals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    /// A value the base does not take.
    Rejected,
    /// A valid goal the base cannot take now: another goal runs, or the
    /// base still comes to rest from one.
    PreconditionFailed,
    /// The wheels deliver no fresh state: before their first state and while
    /// they are silent; the base's odometry started over; or the base's
    /// clock went back.
    ResourceUnavailable,
    /// A goal did not settle by its settle timeout.
    Timeout,
    /// A leader streams setpoints to the base.
    LeaderStreaming,
}

/// Why a goal is refused or failed.
pub type Failure = robot_tree::Failure<FailureKind>;

/// Why the base refuses a goal: a failure of the goals, or a goal the robot's
/// tree cannot place.
pub type TargetFailure = robot_tree::TargetFailure<FailureKind>;

control_core::member_failure_kinds!(
    FailureKind {
        Rejected,
        PreconditionFailed,
        ResourceUnavailable,
        Timeout,
        LeaderStreaming,
    } => drive_base::FailureKind, move_base::FailureKind
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_member_gets_the_kind_of_the_same_name() {
        let kinds = [
            (FailureKind::Rejected, "rejected"),
            (FailureKind::PreconditionFailed, "precondition_failed"),
            (FailureKind::ResourceUnavailable, "resource_unavailable"),
            (FailureKind::Timeout, "timeout"),
            (FailureKind::LeaderStreaming, "leader_streaming"),
        ];
        for (kind, name) in kinds {
            assert_eq!(drive_base::FailureKind::from(kind).as_str(), name);
            assert_eq!(move_base::FailureKind::from(kind).as_str(), name);
        }
    }
}
