//! Why the base refuses or fails a goal: a failure kind that peppy or the
//! base_motion contract declares, and a message for a person. drive_base and
//! move_base each have a generated `FailureKind` with these names, and the
//! admission and the control loop pass one failure for both, so the node
//! keeps its own kind and gives each member its own. A goal the robot's tree
//! cannot place is refused with the kind of the placement ([`Unplaced`]),
//! through [`TargetFailure`].

use control_core::robot_tree::PlaceError;
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    pub kind: FailureKind,
    pub message: String,
}

impl Failure {
    pub fn new(kind: FailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

/// The failure kind of a placement in the robot's tree, as control_core's
/// [`PlaceError::failure_kind`] names it: the kind of the lookup, as
/// transform_lookup names the failures of a lookup, or `rejected` for a
/// placed value no contract takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaceKind(&'static str);

impl PlaceKind {
    pub fn of(error: &PlaceError) -> Self {
        Self(error.failure_kind())
    }
}

/// Why the robot's tree cannot place a goal: the kind of the placement, and a
/// message that names the goal's value, then the cause as transform_lookup's
/// message does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unplaced {
    pub kind: PlaceKind,
    pub message: String,
}

impl Unplaced {
    /// The failure of the placement `error` of the value `what` names.
    pub fn of(what: impl std::fmt::Display, error: &PlaceError) -> Self {
        Self {
            kind: PlaceKind::of(error),
            message: format!("{what}: {error}"),
        }
    }
}

/// Why the base refuses a goal: a failure of the goals, or a goal the robot's
/// tree cannot place.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetFailure {
    Failed(Failure),
    Unplaced(Unplaced),
}

impl From<Failure> for TargetFailure {
    fn from(failure: Failure) -> Self {
        Self::Failed(failure)
    }
}

impl From<Unplaced> for TargetFailure {
    fn from(unplaced: Unplaced) -> Self {
        Self::Unplaced(unplaced)
    }
}

impl TargetFailure {
    /// The kind `Member`, the generated kind of a base_motion member, gives
    /// this failure, and its message.
    pub fn for_member<Member: From<FailureKind> + From<PlaceKind>>(self) -> (Member, String) {
        match self {
            Self::Failed(failure) => (failure.kind.into(), failure.message),
            Self::Unplaced(unplaced) => (unplaced.kind.into(), unplaced.message),
        }
    }
}

/// Gives the generated `FailureKind` of `member` the kind of the same name.
macro_rules! member_failure_kinds {
    ($member:ty) => {
        impl From<FailureKind> for $member {
            fn from(kind: FailureKind) -> Self {
                match kind {
                    FailureKind::Rejected => Self::Rejected,
                    FailureKind::PreconditionFailed => Self::PreconditionFailed,
                    FailureKind::ResourceUnavailable => Self::ResourceUnavailable,
                    FailureKind::Timeout => Self::Timeout,
                    FailureKind::LeaderStreaming => Self::LeaderStreaming,
                }
            }
        }
    };
}

member_failure_kinds!(drive_base::FailureKind);
member_failure_kinds!(move_base::FailureKind);

/// Gives the generated `FailureKind` of `member` each kind of a placement by
/// the same name.
macro_rules! place_kinds {
    ($member:ty) => {
        impl From<PlaceKind> for $member {
            fn from(kind: PlaceKind) -> Self {
                Self::parse(kind.0)
                    .expect("a base_motion member lists each failure kind of a placement")
            }
        }
    };
}

// Both members place their goal and list the kinds of a lookup and `rejected`.
place_kinds!(drive_base::FailureKind);
place_kinds!(move_base::FailureKind);

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
