//! Why the backbone refuses or fails a move, or gives an error reply on a
//! plan check: a failure kind that peppy or the limb_motion and postures
//! contracts declare, and a message for a person. Each generated member has
//! a `FailureKind` of its own with these names, and the moves pass one
//! failure between them (a posture share ends its posture goal), so the
//! backbone keeps its own kind and gives each member its own.

use peppygen::exposed_actions::limb_motion::{move_arm, move_arm_joints, move_gripper};
use peppygen::exposed_actions::postures::{move_to_home, move_to_ready};
use peppygen::exposed_services::limb_motion::check_arm_move;

/// A failure kind of the moves and of `check_arm_move`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    /// A limb name or a value the backbone does not take.
    Rejected,
    /// A valid goal the state of the limb does not allow now: the limb
    /// executes another move.
    PreconditionFailed,
    /// The follower of a limb delivers no fresh state: before its first
    /// state, or after it stopped reporting.
    ResourceUnavailable,
    /// The move cannot go on, and no other kind says why.
    AttemptFailed,
    /// An arm move did not settle by its timeout.
    Timeout,
    /// No plan reaches the target.
    Unreachable,
    /// A leader streams a limb of the goal.
    LeaderStreaming,
}

/// Why a move is refused or failed.
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

    /// The same failure with `label` before its message, such as the side
    /// of the arm whose share of a posture goal failed.
    pub fn labelled(self, label: &str) -> Self {
        Self {
            message: format!("{label}: {}", self.message),
            ..self
        }
    }
}

/// Gives each listed generated `FailureKind` the kind of the same name.
macro_rules! member_failure_kinds {
    ($($member:ty),+ $(,)?) => {$(
        impl From<FailureKind> for $member {
            fn from(kind: FailureKind) -> Self {
                match kind {
                    FailureKind::Rejected => Self::Rejected,
                    FailureKind::PreconditionFailed => Self::PreconditionFailed,
                    FailureKind::ResourceUnavailable => Self::ResourceUnavailable,
                    FailureKind::AttemptFailed => Self::AttemptFailed,
                    FailureKind::Timeout => Self::Timeout,
                    FailureKind::Unreachable => Self::Unreachable,
                    FailureKind::LeaderStreaming => Self::LeaderStreaming,
                }
            }
        }
    )+};
}

member_failure_kinds!(
    move_arm_joints::FailureKind,
    move_arm::FailureKind,
    move_gripper::FailureKind,
    move_to_ready::FailureKind,
    move_to_home::FailureKind,
    check_arm_move::FailureKind,
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
            (FailureKind::AttemptFailed, "attempt_failed"),
            (FailureKind::Timeout, "timeout"),
            (FailureKind::Unreachable, "unreachable"),
            (FailureKind::LeaderStreaming, "leader_streaming"),
        ];
        for (kind, name) in kinds {
            let names = [
                move_arm_joints::FailureKind::from(kind).as_str(),
                move_arm::FailureKind::from(kind).as_str(),
                move_gripper::FailureKind::from(kind).as_str(),
                move_to_ready::FailureKind::from(kind).as_str(),
                move_to_home::FailureKind::from(kind).as_str(),
                check_arm_move::FailureKind::from(kind).as_str(),
            ];
            assert_eq!(names, [name; 6], "{kind:?}");
        }
    }

    #[test]
    fn a_labelled_failure_keeps_its_kind_and_names_the_label_first() {
        let failure = Failure::new(FailureKind::Timeout, "still 0.2 rad from the target");
        assert_eq!(
            failure.labelled("left"),
            Failure::new(FailureKind::Timeout, "left: still 0.2 rad from the target")
        );
    }
}
