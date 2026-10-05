//! The `stop` service of limb_motion: ends every planned move in flight on
//! the robot, whoever started it, through the coordinator, and answers the
//! limbs whose move was ended. The moves of both contracts run on the
//! coordinator's tick, so the request is handed to it and its reply awaited
//! ([`CoordinatorRequest::Stop`]); a goal a stop ends completes as cancelled
//! with the stop's message, and the arm holds where it was last governed.

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
            let asked = blocking_ask_coordinator(
                &requests,
                CoordinatorRequest::Stop {
                    reason: request.data.reason,
                    reply,
                },
                answer,
            );
            Ok(match asked {
                Ok(stopped) => {
                    let message = if stopped.is_empty() {
                        "nothing was moving".to_string()
                    } else {
                        format!("stopped {}", stopped.join(", "))
                    };
                    stop::Response::new(true, message, stopped)
                }
                Err(refusal) => {
                    error!("stop: {refusal}");
                    stop::Response::new(false, refusal, Vec::new())
                }
            })
        })
        .await?;
    }
}
