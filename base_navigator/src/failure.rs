//! Why the navigator refuses or fails a goal: a failure kind that peppy or
//! base_navigation declares, and a message for a person. Each of the four
//! actions has a generated `FailureKind`; the admission and the control loop
//! pass the node's own kinds, and each member gets the kind of the same name.
//!
//! [`FailureKind`] holds the kinds every action of base_navigation declares,
//! and [`NavigationKind`] adds those navigate_to_pose alone declares: the
//! kinds of its plans and of its map. A spin, a drive_on_heading and a
//! back_up fail with a [`Failure`], whose kind holds none of a navigation's
//! own. A navigation the robot's tree cannot place is refused with the kind
//! of the placement (control_core's
//! [`Unplaced`](control_core::robot_tree::Unplaced)), through
//! [`TargetFailure`].

use control_core::robot_tree;
use peppygen::exposed_actions::base_navigation::{
    back_up, drive_on_heading, navigate_to_pose, spin,
};

/// A failure kind of every goal of base_navigation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    /// A value the navigator does not take.
    Rejected,
    /// A valid goal the base cannot take now: describe_base names commanded
    /// odometry, or drive_states reports a hold or a limb outside the
    /// footprint.
    PreconditionFailed,
    /// The base or a sensor delivers nothing fresh: describe_base's answer,
    /// drive_states, a sensor's readings, or the odometry, which started over.
    ResourceUnavailable,
    /// A goal's timeout passed before the odometry measured its whole turn
    /// or distance.
    Timeout,
    /// The goal cannot go on for another reason.
    AttemptFailed,
    /// The moving base found no way forward.
    Blocked,
}

/// Why a spin, a drive_on_heading or a back_up is refused or fails, and why
/// a goal of any action is refused before its own checks.
pub type Failure = robot_tree::Failure<FailureKind>;

control_core::member_failure_kinds!(
    FailureKind {
        Rejected,
        PreconditionFailed,
        ResourceUnavailable,
        Timeout,
        AttemptFailed,
        Blocked,
    } => navigate_to_pose::FailureKind, spin::FailureKind, drive_on_heading::FailureKind,
        back_up::FailureKind
);

/// A failure kind of a navigation: a kind of every goal, or one of its plans
/// and its map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavigationKind {
    Goal(FailureKind),
    /// The goal is in collision with an obstacle the sensors have seen.
    Unreachable,
    /// The base stands in collision with an obstacle the sensors have seen.
    StartInCollision,
    /// No path joins the base to the goal, after the recoveries.
    NoPath,
    /// A navigation whose target is in the map frame has no loaded map, the
    /// base is not localized in it, or its loaded map changed.
    NotLocalized,
}

impl From<FailureKind> for NavigationKind {
    fn from(kind: FailureKind) -> Self {
        Self::Goal(kind)
    }
}

impl From<NavigationKind> for navigate_to_pose::FailureKind {
    fn from(kind: NavigationKind) -> Self {
        match kind {
            NavigationKind::Goal(kind) => kind.into(),
            NavigationKind::Unreachable => Self::Unreachable,
            NavigationKind::StartInCollision => Self::StartInCollision,
            NavigationKind::NoPath => Self::NoPath,
            NavigationKind::NotLocalized => Self::NotLocalized,
        }
    }
}

/// Why a navigation is refused or fails.
pub type NavigationFailure = robot_tree::Failure<NavigationKind>;

/// Why a navigation is refused: a failure of the navigation, or a target the
/// robot's tree cannot place.
pub type TargetFailure = robot_tree::TargetFailure<NavigationKind>;

/// How a goal's own step ends it: completed at its end, failed with
/// `timeout` once its timeout_s passed, the message saying why, or failed
/// for another cause, with a failure of the goal's kinds `K`. The control
/// loop ends a goal cancelled, and fails it for the causes every goal shares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ended<K> {
    Completed,
    TimedOut(String),
    Failed(robot_tree::Failure<K>),
}

/// `failure`, of a kind every goal shares, as the failure of a navigation.
pub fn of_navigation(failure: Failure) -> NavigationFailure {
    NavigationFailure::new(failure.kind.into(), failure.message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use control_core::robot_tree::{PlaceKind, Unplaced};

    const SHARED: [(FailureKind, &str); 6] = [
        (FailureKind::Rejected, "rejected"),
        (FailureKind::PreconditionFailed, "precondition_failed"),
        (FailureKind::ResourceUnavailable, "resource_unavailable"),
        (FailureKind::Timeout, "timeout"),
        (FailureKind::AttemptFailed, "attempt_failed"),
        (FailureKind::Blocked, "blocked"),
    ];

    #[test]
    fn each_member_gets_the_shared_kind_of_the_same_name() {
        for (kind, name) in SHARED {
            assert_eq!(navigate_to_pose::FailureKind::from(kind).as_str(), name);
            assert_eq!(spin::FailureKind::from(kind).as_str(), name);
            assert_eq!(drive_on_heading::FailureKind::from(kind).as_str(), name);
            assert_eq!(back_up::FailureKind::from(kind).as_str(), name);
        }
    }

    #[test]
    fn a_navigation_gets_its_own_kinds_by_name() {
        let own = [
            (NavigationKind::Unreachable, "unreachable"),
            (NavigationKind::StartInCollision, "start_in_collision"),
            (NavigationKind::NoPath, "no_path"),
            (NavigationKind::NotLocalized, "not_localized"),
            (NavigationKind::Goal(FailureKind::Blocked), "blocked"),
        ];
        for (kind, name) in own {
            assert_eq!(navigate_to_pose::FailureKind::from(kind).as_str(), name);
        }
    }

    /// A navigation gets each kind of a placement ([`PlaceKind::ALL`]) by the
    /// same name, and the message of the placement.
    #[test]
    fn a_navigation_gets_each_kind_of_a_placement() {
        for kind in PlaceKind::ALL {
            let unplaced = TargetFailure::from(Unplaced {
                kind,
                message: "the target: unplaced".to_string(),
            });
            let (member, message) = unplaced.for_member(navigate_to_pose::FailureKind::parse);
            assert_eq!(member.as_str(), kind.name());
            assert_eq!(message, "the target: unplaced");
        }
    }
}
