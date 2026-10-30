//! The failures of a step and how they propagate: a pass failure ends a pass
//! of the nearest enclosing loop, and a lane failure ends its lane through
//! every loop.

use std::fmt;

use workflow_documents::result::{AlertCause, Anomaly};

/// How a failure propagates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Class {
    /// Ends a pass of the nearest enclosing loop; a retryable one first takes
    /// the step's retries.
    Pass { retryable: bool },
    /// Ends the lane through every loop.
    Lane,
}

/// The failure of a step: its kind, its message, its class, and the cause
/// that the incident of the failure names.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Failure {
    pub kind: String,
    pub message: String,
    pub class: Class,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cause: Option<Box<Cause>>,
}

/// What caused a failure of the kind `anomaly` or `alert`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cause {
    /// The anomaly that the monitor gave.
    Anomaly(Anomaly),
    /// The alert of a robot, or the producer whose alert set went stale.
    Alert(AlertCause),
}

impl Failure {
    /// A failure of `kind`, classed by [`class_of`].
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Failure {
        let kind = kind.into();
        let class = class_of(&kind);
        Failure {
            kind,
            message: message.into(),
            class,
            cause: None,
        }
    }

    /// This failure, caused by `cause`.
    pub fn because(self, cause: Cause) -> Failure {
        Failure {
            cause: Some(Box::new(cause)),
            ..self
        }
    }

    pub fn of(kind: DirectorKind, message: impl Into<String>) -> Failure {
        Failure::new(kind.name(), message)
    }

    pub fn is_retryable(&self) -> bool {
        self.class == Class::Pass { retryable: true }
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

/// The kinds that only the director gives, for the constructs of a workflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectorKind {
    ConditionFailed,
    BoundReached,
    InvariantViolated,
    BreakerOpen,
    StaleObservation,
    Stopped,
    Anomaly,
    Alert,
    SignalUnreachable,
    ClockStopped,
}

impl DirectorKind {
    pub fn name(self) -> &'static str {
        match self {
            DirectorKind::ConditionFailed => "condition_failed",
            DirectorKind::BoundReached => "bound_reached",
            DirectorKind::InvariantViolated => "invariant_violated",
            DirectorKind::BreakerOpen => "breaker_open",
            DirectorKind::StaleObservation => "stale_observation",
            DirectorKind::Stopped => "stopped",
            DirectorKind::Anomaly => "anomaly",
            DirectorKind::Alert => "alert",
            DirectorKind::SignalUnreachable => "signal_unreachable",
            DirectorKind::ClockStopped => "clock_stopped",
        }
    }

    fn class(self) -> Class {
        match self {
            DirectorKind::ConditionFailed | DirectorKind::BoundReached => {
                Class::Pass { retryable: false }
            }
            _ => Class::Lane,
        }
    }

    const ALL: [DirectorKind; 10] = [
        DirectorKind::ConditionFailed,
        DirectorKind::BoundReached,
        DirectorKind::InvariantViolated,
        DirectorKind::BreakerOpen,
        DirectorKind::StaleObservation,
        DirectorKind::Stopped,
        DirectorKind::Anomaly,
        DirectorKind::Alert,
        DirectorKind::SignalUnreachable,
        DirectorKind::ClockStopped,
    ];
}

/// The kinds of peppy that the director gives itself.
pub const ATTEMPT_FAILED: &str = "attempt_failed";
pub const TIMEOUT: &str = "timeout";
pub const PRECONDITION_FAILED: &str = "precondition_failed";
pub const RESOURCE_UNAVAILABLE: &str = "resource_unavailable";
pub const REJECTED: &str = "rejected";

/// The kind a backbone refuses a goal with while a leader streams its limb.
pub const LEADER_STREAMING: &str = "leader_streaming";

/// The class of a failure kind: the kinds of peppy, of the robot contracts
/// and of the director by their tables, and every other kind a contract
/// declares as a lane failure.
fn class_of(kind: &str) -> Class {
    if let Some(director) = DirectorKind::ALL.iter().find(|k| k.name() == kind) {
        return director.class();
    }
    match kind {
        ATTEMPT_FAILED | "unverified" | "provider_failed" => Class::Pass { retryable: true },
        TIMEOUT | "impossible" | "budget_exhausted" | "unreachable" | "item_lost" | "not_found" => {
            Class::Pass { retryable: false }
        }
        _ => Class::Lane,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_kind_takes_the_class_of_its_table() {
        assert_eq!(class_of("attempt_failed"), Class::Pass { retryable: true });
        assert_eq!(class_of("unverified"), Class::Pass { retryable: true });
        assert_eq!(class_of("timeout"), Class::Pass { retryable: false });
        assert_eq!(class_of("not_found"), Class::Pass { retryable: false });
        assert_eq!(class_of("bound_reached"), Class::Pass { retryable: false });
        assert_eq!(class_of("precondition_failed"), Class::Lane);
        assert_eq!(class_of("leader_streaming"), Class::Lane);
        assert_eq!(class_of("breaker_open"), Class::Lane);
        assert_eq!(class_of("a_kind_no_table_names"), Class::Lane);
    }
}
