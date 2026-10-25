//! The `stop` service of limb_motion: ends every planned move in flight on
//! the robot, whoever started it, through the coordinator, and answers the
//! limbs whose move was ended. The moves of both contracts run on the
//! coordinator's tick, so the request is handed to it and its reply awaited
//! ([`CoordinatorRequest::Stop`]); each goal a stop ends is cancelled with a
//! reason that names the stop, and the arm holds where it was last governed.
//! A coordinator that does not answer cannot carry the stop out: the caller
//! gets an error reply with `attempt_failed`.

use std::sync::Arc;

use peppygen::exposed_services::limb_motion::stop;
use peppygen::{NodeRunner, Result};
use tokio::sync::{mpsc, oneshot};
use tracing::error;

use crate::actions::blocking_ask_coordinator;
use crate::coordinator::CoordinatorRequest;

/// Expose `stop`: every request goes to the coordinator, which ends the
/// moves at the start of its next tick and answers the limbs it stopped.
pub async fn run_stop(
    runner: Arc<NodeRunner>,
    requests: mpsc::Sender<CoordinatorRequest>,
) -> Result<()> {
    loop {
        stop::handle_next_request(&runner, |request| {
            let (reply, answer) = oneshot::channel();
            let stopped = blocking_ask_coordinator(
                &requests,
                CoordinatorRequest::Stop {
                    reason: request.data.reason,
                    reply,
                },
                answer,
            )
            .map_err(|reason| {
                error!("stop: {reason}");
                stop::Failure::new(stop::FailureKind::AttemptFailed, reason)
            })?;
            Ok(stop::Response::new(stopped))
        })
        .await?;
    }
}
