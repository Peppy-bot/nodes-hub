//! base_navigation's clear_grid: it hands the clearing to the control loop,
//! which empties the grid between two steps, and answers once the loop took
//! it, within a timeout; an error reply, with `attempt_failed`, says why the
//! loop did not.

use std::sync::Arc;
use std::time::Duration;

use control_core::asking::ask_loop;
use control_core::serving;
use control_core::sync::wait_for;
use peppygen::NodeRunner;
use peppygen::exposed_services::base_navigation::clear_grid;
use tokio::sync::{mpsc, oneshot};
use tracing::error;

use crate::control::Event;

/// How long a clearing waits for the control loop to take and answer it:
/// many steps of the slowest loop, so only a loop that stopped answering runs
/// it out.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(5);

/// Serve clear_grid until the node cancels.
pub async fn serve_clear_grid(runner: Arc<NodeRunner>, events: mpsc::Sender<Event>) {
    let token = runner.cancellation_token().clone();
    serving::serve("clear_grid", token.cancelled(), || {
        clear_grid::handle_requests(&runner, |_request| {
            let (reply, answer) = oneshot::channel();
            wait_for(ask_loop(
                "control loop",
                &events,
                Event::ClearGrid { reply },
                answer,
                ANSWER_TIMEOUT,
            ))
            .inspect_err(|reason| error!("clear_grid: {reason}"))
            .map_err(|reason| {
                clear_grid::Failure::new(clear_grid::FailureKind::AttemptFailed, reason)
            })
        })
    })
    .await;
}
