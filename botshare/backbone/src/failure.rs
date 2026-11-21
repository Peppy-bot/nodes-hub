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
    use control_core::geometry::GeometryError;
    use control_core::robot_tree::{PlaceError, Unplaced};
    use frame_tree::{
        BrokenTree, FrameName, HeldStamps, LookupError, LookupTime, Stamp, TransformName,
    };

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

    /// Each member gets each kind of a placement by the same name, and the
    /// message that names the value placed, then the placement's cause.
    #[test]
    fn each_member_gets_each_kind_of_a_placement() {
        for (error, name) in each_place_error() {
            let (drive, driving) = TargetFailure::from(Unplaced::of("the velocity", &error))
                .for_member(drive_base::FailureKind::parse);
            let (movement, moving) = TargetFailure::from(Unplaced::of("the target", &error))
                .for_member(move_base::FailureKind::parse);
            assert_eq!([drive.as_str(), movement.as_str()], [name; 2], "{error}");
            assert_eq!(
                [driving, moving],
                [
                    format!("the velocity: {error}"),
                    format!("the target: {error}")
                ]
            );
        }
    }
}
