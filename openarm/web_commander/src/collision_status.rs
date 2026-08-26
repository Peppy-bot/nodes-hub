// Live self-collision proximity for the UI. Consumes the collision_status
// contract's stream and reports the nearest-pair distance and link names to the
// owner, so the panel can show how close the arms are and color it against the
// governor band.

use std::sync::Arc;
use std::time::Instant;

use peppygen::NodeRunner;
use peppygen::consumed_topics::collision_status::collision_status;
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::error;

use crate::owner::Feedback;
use crate::state::{Disposition, Proximity};

pub async fn run(
    runner: Arc<NodeRunner>,
    feedback: mpsc::Sender<Feedback>,
    token: CancellationToken,
) {
    let mut subscription = match collision_status::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => {
            error!(error = %e, "collision_status subscribe");
            return;
        }
    };
    loop {
        let received = tokio::select! {
            _ = token.cancelled() => return,
            received = subscription.next() => received,
        };
        let (_producer, msg) = match received {
            Ok(Some(pair)) => pair,
            Ok(None) => return,
            Err(e) => {
                error!(error = %e, "collision_status receive");
                continue;
            }
        };
        let proximity = Proximity {
            distance: msg.distance,
            link_a: msg.link_a,
            link_b: msg.link_b,
            disposition: Disposition::from_wire(msg.throttled, msg.stopped),
            received_at: Instant::now(),
        };
        if feedback.send(Feedback::Proximity(proximity)).await.is_err() {
            return; // the owner is gone; nothing left to report to
        }
    }
}
