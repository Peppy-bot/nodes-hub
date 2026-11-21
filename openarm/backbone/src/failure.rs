//! Why the backbone refuses or fails a move, or gives an error reply on a
//! plan check or a check of points: a failure kind that peppy or the
//! limb_motion, postures and workspace contracts declare, and a message for a
//! person. Each generated member has a `FailureKind` of its own with these
//! names, and the moves pass one failure between them (a posture share ends
//! its posture goal, and a share of a joint path its execute_joint_path goal),
//! so the backbone keeps its own kind and gives each member its own. A
//! command the robot's tree cannot place fails with the kind of the placement
//! (control_core's [`Unplaced`](control_core::robot_tree::Unplaced)), which
//! only the members that place a frame take:
//! `move_arm` and `check_arm_move` through [`TargetFailure`], and
//! `check_positions`. A move along a path the backbone plans or judges fails
//! with `unreachable` where no path reaches its target, which only the
//! members that run such a path take: `move_arm` and `execute_joint_path`
//! through [`PathFailure`].

use control_core::robot_tree;
use peppygen::exposed_actions::limb_motion::{
    execute_joint_path, move_arm, move_arm_joints, move_gripper,
};
use peppygen::exposed_actions::postures::{move_to_home, move_to_ready};
use peppygen::exposed_services::limb_motion::check_arm_move;

/// A failure kind of the moves and of `check_arm_move`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    /// A limb name or a value the backbone does not take.
    Rejected,
    /// A valid goal the state of the limb does not allow now: the limb
    /// executes another move, or an arm of a joint path stands past the start
    /// tolerance from the path's expected start, when the coordinator takes
    /// the goal or at the path's run start, or at its run start from where it
    /// stood when the goal was taken, or still moves after the path's last
    /// judgment.
    PreconditionFailed,
    /// The follower of a limb delivers no fresh state: before its first
    /// state, or after it stopped reporting.
    ResourceUnavailable,
    /// The move cannot go on, and no other kind says why.
    AttemptFailed,
    /// An arm move did not settle by its timeout.
    Timeout,
    /// A leader streams a limb of the goal.
    LeaderStreaming,
}

/// Why a move is refused or failed.
pub type Failure = robot_tree::Failure<FailureKind>;

/// Why the backbone refuses a `move_arm` goal or gives an error reply on a
/// `check_arm_move` request, the moves that place a target: a failure of the
/// moves, or a target the robot's tree cannot place.
pub type TargetFailure = robot_tree::TargetFailure<FailureKind>;

/// The refusal of a goal, or a plan check, that reaches an arm executing a
/// move: the one message for both, whichever gate answers it.
pub const ARM_BUSY: &str = "arm is already executing a motion";

/// The refusal of a gripper goal that reaches a gripper executing a move.
pub const GRIPPER_BUSY: &str = "gripper is already executing a move";

/// The failure kind `unreachable` of a move along a path the backbone plans
/// or judges: no path it planned or judged reaches the target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnreachableKind;

/// Why the backbone fails a move along a path it plans or judges, the moves
/// of `move_arm` and `execute_joint_path`: a failure of the moves, or
/// `unreachable`, with the message that says why no path it planned or
/// judged reaches the target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PathFailure {
    Failed(Failure),
    Unreachable(String),
}

impl From<Failure> for PathFailure {
    fn from(failure: Failure) -> Self {
        Self::Failed(failure)
    }
}

impl PathFailure {
    /// The kind `Member`, the generated kind of a member that runs a path the
    /// backbone plans or judges, gives this failure, and its message.
    pub fn for_member<Member: From<FailureKind> + From<UnreachableKind>>(self) -> (Member, String) {
        match self {
            Self::Failed(failure) => (failure.kind.into(), failure.message),
            Self::Unreachable(message) => (UnreachableKind.into(), message),
        }
    }
}

/// The failure of a goal that reaches a limb whose follower stream has gone
/// stale, and of a move whose follower goes stale while it runs.
pub fn stale_failure() -> Failure {
    Failure::new(
        FailureKind::ResourceUnavailable,
        "the follower stopped reporting",
    )
}

/// Gives the generated `FailureKind` of `member` its `unreachable`, for a
/// member that runs a path the backbone plans or judges.
macro_rules! unreachable_kind {
    ($member:ty) => {
        impl From<UnreachableKind> for $member {
            fn from(_: UnreachableKind) -> Self {
                Self::Unreachable
            }
        }
    };
}

// The Cartesian move and the joint path run a path the backbone plans or
// judges, and list `unreachable`.
unreachable_kind!(move_arm::FailureKind);
unreachable_kind!(execute_joint_path::FailureKind);

control_core::member_failure_kinds!(
    FailureKind {
        Rejected,
        PreconditionFailed,
        ResourceUnavailable,
        AttemptFailed,
        Timeout,
        LeaderStreaming,
    } => execute_joint_path::FailureKind,
        move_arm::FailureKind,
        move_to_ready::FailureKind,
        move_to_home::FailureKind,
        move_arm_joints::FailureKind,
        move_gripper::FailureKind,
        check_arm_move::FailureKind,
);

#[cfg(test)]
mod tests {
    use super::*;
    use control_core::geometry::GeometryError;
    use control_core::robot_tree::{PlaceError, PlaceKind, Unplaced};
    use frame_tree::{
        BrokenTree, FrameName, HeldStamps, LookupError, LookupTime, Stamp, TransformName,
    };
    use peppygen::exposed_services::workspace::check_positions;

    use crate::robot::LIMBS;
    use crate::test_support::LEFT;

    /// Each member gets the kind of the same name.
    #[test]
    fn each_member_gets_the_kind_of_the_same_name() {
        let kinds = [
            (FailureKind::Rejected, "rejected"),
            (FailureKind::PreconditionFailed, "precondition_failed"),
            (FailureKind::ResourceUnavailable, "resource_unavailable"),
            (FailureKind::AttemptFailed, "attempt_failed"),
            (FailureKind::Timeout, "timeout"),
            (FailureKind::LeaderStreaming, "leader_streaming"),
        ];
        for (kind, name) in kinds {
            let members = [
                execute_joint_path::FailureKind::from(kind).as_str(),
                move_arm::FailureKind::from(kind).as_str(),
                move_to_ready::FailureKind::from(kind).as_str(),
                move_to_home::FailureKind::from(kind).as_str(),
                move_arm_joints::FailureKind::from(kind).as_str(),
                move_gripper::FailureKind::from(kind).as_str(),
                check_arm_move::FailureKind::from(kind).as_str(),
            ];
            assert_eq!(members, [name; 7], "{kind:?}");
        }
    }

    /// A member that runs a path the backbone plans or judges gets
    /// `unreachable`, with the message that says why no path reaches the
    /// target.
    #[test]
    fn a_member_that_runs_a_path_gets_unreachable() {
        let unreachable = PathFailure::Unreachable("blocked".to_string());
        let (moving, moving_message) = unreachable.clone().for_member::<move_arm::FailureKind>();
        let (path, path_message) = unreachable.for_member::<execute_joint_path::FailureKind>();
        assert_eq!([moving.as_str(), path.as_str()], ["unreachable"; 2]);
        assert_eq!([moving_message, path_message], ["blocked", "blocked"]);
    }

    /// A failure of the moves reaches a member that runs a path with the kind
    /// of the same name.
    #[test]
    fn a_failure_of_the_moves_reaches_a_member_that_runs_a_path_by_its_name() {
        let failed = PathFailure::from(Failure::new(FailureKind::Timeout, "unsettled"));
        let (kind, message) = failed.for_member::<execute_joint_path::FailureKind>();
        assert_eq!(
            (kind, message.as_str()),
            (execute_joint_path::FailureKind::Timeout, "unsettled")
        );
    }

    fn frame(name: &str) -> FrameName {
        FrameName::parse(name).expect("a frame name")
    }

    /// A failure of each kind of a placement, with the name the contracts give it.
    fn each_place_error() -> [(PlaceError, &'static str); 5] {
        let transform = TransformName {
            parent: frame("odom"),
            child: frame("base_footprint"),
        };
        [
            (
                PlaceError::Lookup(LookupError::UnknownFrame {
                    frame: frame("gripper"),
                    held: Vec::new(),
                }),
                "unknown_frame",
            ),
            (
                PlaceError::Lookup(LookupError::NoChain {
                    target_frame: frame("base_link"),
                    source_frame: frame("odom"),
                }),
                "no_chain",
            ),
            (
                PlaceError::Lookup(LookupError::Extrapolation {
                    transform: transform.clone(),
                    requested: LookupTime::At(Stamp::from_nanos(5)),
                    held: HeldStamps::Empty,
                }),
                "extrapolation",
            ),
            (
                PlaceError::Lookup(LookupError::BrokenTree(BrokenTree::Cycle {
                    transforms: vec![transform],
                })),
                "broken_tree",
            ),
            (PlaceError::Value(GeometryError::NoFrame), "rejected"),
        ]
    }

    /// A member that places a frame gets each kind of a placement by the same
    /// name, and the message that names the command, then the placement's
    /// cause.
    #[test]
    fn a_member_that_places_a_frame_gets_each_kind_of_a_placement() {
        for (error, name) in each_place_error() {
            let unplaced = TargetFailure::from(Unplaced::of("the target", &error));
            let (move_arm, moving) = unplaced.clone().for_member(move_arm::FailureKind::parse);
            let (check_arm_move, checking) =
                unplaced.for_member(check_arm_move::FailureKind::parse);
            let check_positions =
                PlaceKind::of(&error).of_member(check_positions::FailureKind::parse);
            assert_eq!(
                [
                    move_arm.as_str(),
                    check_arm_move.as_str(),
                    check_positions.as_str()
                ],
                [name; 3],
                "{error}"
            );
            let cause = format!("the target: {error}");
            assert_eq!([moving, checking], [cause.clone(), cause]);
        }
    }

    /// A failure of the moves reaches a member that places a target with the
    /// kind of the same name.
    #[test]
    fn a_failure_of_the_moves_reaches_a_member_that_places_a_target_by_its_name() {
        let failed = TargetFailure::from(Failure::new(FailureKind::LeaderStreaming, "streaming"));
        let (kind, message) = failed.for_member(move_arm::FailureKind::parse);
        assert_eq!(
            (kind, message.as_str()),
            (move_arm::FailureKind::LeaderStreaming, "streaming")
        );
    }

    #[test]
    fn a_labelled_failure_keeps_its_kind_and_names_the_label_first() {
        let failure = Failure::new(FailureKind::Timeout, "still 0.2 rad from the target");
        assert_eq!(
            failure.labelled(LIMBS.arm_name(LEFT)),
            Failure::new(
                FailureKind::Timeout,
                format!("{}: still 0.2 rad from the target", LIMBS.arm_name(LEFT))
            )
        );
    }
}
