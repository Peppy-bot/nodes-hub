//! What the engine asks of the world: the run clock and wall time, calls on
//! producers, observation reads, the leases of the run and the events it
//! gives. The live host serves it with the codec clients and the fleet; the
//! tests serve it with a script.

use std::future::Future;
use tokio_util::sync::CancellationToken;

use serde_json::Value as Json;
use workflow_documents::{EventKind, Name, RobotName};

use super::failure::Failure;
use super::plan::Target;

/// How a call ended.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallEnd {
    /// A service gave its response, or a goal completed with its result.
    Completed(Json),
    /// A goal was refused, or a service gave an error reply.
    Refused(Failure),
    /// A goal failed.
    Failed(Failure),
    /// A goal ended cancelled: `asked` when the director sent the cancel.
    Cancelled { asked: bool, reason: String },
    /// The goal did not end within the cancel timeout after its cancel: the
    /// robot is in an unknown state.
    Unknown,
    /// `fleet.check_move` refused the call: the run's leases ended, or the
    /// stack is halted.
    MoveRefused { halted: bool, message: String },
}

/// An event of a run, which the host gives on the progress of the goal of the
/// run, on `run_events` and in the journal, where it is written as it
/// serializes.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Event {
    pub role: String,
    #[serde(rename = "event")]
    pub kind: EventKind,
    pub step: String,
    pub failure_kind: String,
    pub message: String,
    pub details: Json,
}

/// How a read of an observation goes.
pub enum Joining {
    /// As the engine finds it: with the read of another reader at its
    /// instant when there is one, else on its own.
    AsFound,
    /// On its own, as the journal of a replayed run gives it.
    Own,
    /// With the answer of another reader's read, as the journal of a
    /// replayed run gives it.
    Took(Result<Json, Failure>),
}

/// How a read that joined another reader's read in flight goes on, as the
/// journal keeps it.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Join {
    /// It waits for the answer of that read.
    Waits,
    /// It took the answer.
    Took(Result<Json, Failure>),
    /// The read it joined was dropped, and it reads itself.
    ReadsItself,
}

/// The world as the engine of one run sees it. Every future a host gives
/// can be dropped: the engine drops the call of a service it no longer
/// waits for, and drops a timer it no longer needs.
pub trait Host: Send + Sync {
    /// The run clock, in nanoseconds.
    fn now(&self) -> u64;

    /// Wall time, in nanoseconds.
    fn wall_now(&self) -> u64;

    /// Resolves at the first instant of the run clock at or after `at`.
    fn until(&self, at: u64) -> impl Future<Output = ()> + Send;

    /// Sends a call and gives how it ended. A goal that `cancel` cancels
    /// gets its cancel, and the host waits for its end up to the cancel
    /// timeout; a call of a member with the mark moves_robot asks
    /// `fleet.check_move` first.
    fn call(
        &self,
        target: &Target,
        request: Json,
        cancel: CancellationToken,
    ) -> impl Future<Output = CallEnd> + Send;

    /// The latest message of a topic and when it arrived on the run clock,
    /// or `None` before the first.
    fn latest(&self, target: &Target) -> Option<(Json, u64)>;

    /// Resolves once the topic of `target` has given a message.
    fn message_arrives(&self, target: &Target) -> impl Future<Output = ()> + Send;

    /// How the read `nth` of the observation `name` in the run goes.
    fn joining(&self, name: &Name, nth: u64) -> impl Future<Output = Joining> + Send;

    /// The read `nth` of the observation `name`, which joined another
    /// reader's read in flight, goes on as `join` says.
    fn joined(&self, name: &Name, nth: u64, join: &Join);

    /// Opens or closes the lease of a robot of the run for a take-over, or
    /// gives why the daemon did not.
    fn set_lease_open(
        &self,
        robot: &RobotName,
        open: bool,
    ) -> impl Future<Output = Result<(), String>> + Send;

    fn event(&self, event: Event);
}
