//! The `stop` service of robot_stop, the robot stop the robot's MCP
//! endpoint calls: it ends every planned move in flight on the robot, whoever
//! started it, through the coordinator, and answers the limbs whose move was
//! ended. The moves of both motion contracts run on the coordinator's tick, so
//! the request is handed to it and its reply awaited
//! ([`CoordinatorRequest::Stop`]); each goal a stop ends is cancelled with a
//! reason that names the stop, and each arm whose move it ends holds where it
//! stands.
//! A coordinator that does not take and answer the stop within the
//! `stop_answer_timeout_s` parameter cannot carry it out: the caller gets an
//! error reply with `attempt_failed`, and the coordinator runs the stop once
//! it takes it.

use std::sync::Arc;
use std::time::Duration;

use control_core::asking::ask_loop;
use control_core::serving;
use control_core::sync::wait_for;
use peppygen::NodeRunner;
use peppygen::exposed_services::robot_stop::stop;
use peppylib::runtime::CancellationToken;
use tokio::sync::{mpsc, oneshot};
use tracing::error;

use crate::actions::COORDINATOR;
use crate::coordinator::CoordinatorRequest;

/// Hand a stop to the coordinator, which ends the moves at the start of its
/// next tick, and return the limbs it stopped, or why it could not, the
/// coordinator taken and answered within `answer_timeout` ([`ask_loop`]).
fn stop_moves(
    requests: &mpsc::Sender<CoordinatorRequest>,
    reason: String,
    answer_timeout: Duration,
) -> Result<Vec<String>, String> {
    let (reply, answer) = oneshot::channel();
    wait_for(ask_loop(
        COORDINATOR,
        requests,
        CoordinatorRequest::Stop { reason, reply },
        answer,
        answer_timeout,
    ))
    .inspect_err(|reason| error!("stop: {reason}"))
}

/// Serve robot_stop's `stop` until `token` cancels: [`stop_moves`] within
/// `answer_timeout`, answered with the limbs it stopped, or an error reply
/// with `attempt_failed`.
pub async fn run_robot_stop(
    runner: Arc<NodeRunner>,
    token: CancellationToken,
    requests: mpsc::Sender<CoordinatorRequest>,
    answer_timeout: Duration,
) {
    serving::serve("robot_stop", token.cancelled(), || {
        stop::handle_requests(&runner, |request| {
            stop_moves(&requests, request.data.reason, answer_timeout)
                .map(stop::Response::new)
                .map_err(|reason| stop::Failure::new(stop::FailureKind::AttemptFailed, reason))
        })
    })
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A coordinator that takes no request within the answer time: the stop
    /// fails naming that time, and stays queued for the coordinator to run
    /// once it takes it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stop_the_coordinator_does_not_answer_in_its_answer_time_fails_and_stays_queued() {
        let (requests, mut queued) = mpsc::channel(1);
        let answer_timeout = Duration::from_millis(10);
        assert_eq!(
            stop_moves(&requests, "the operator asked".to_string(), answer_timeout),
            Err(format!(
                "the coordinator did not answer within {answer_timeout:?}: call again, and check \
                 the node's log for what holds up the coordinator"
            ))
        );
        let Ok(CoordinatorRequest::Stop { reason, .. }) = queued.try_recv() else {
            panic!("the stop stays queued for the coordinator");
        };
        assert_eq!(reason, "the operator asked");
    }
}
