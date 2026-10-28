//! Live self-collision proximity for the UI. Consumes the collision_status
//! contract's stream and reports the nearest-pair distance and link names to the
//! owner, so the panel can show how close the arms are and color it against the
//! governor band.

use std::sync::Arc;
use std::time::Instant;

use peppygen::NodeRunner;
use peppygen::consumed_topics::collision_status::collision_status;
use peppylib::messaging::ProducerRef;
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::error;

use crate::consumer;
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
    forward_proximity(async move || subscription.next().await, feedback, token).await;
}

/// Forward the proximity each message `next_message` receives reports to
/// the owner, through the shared consumer loop.
async fn forward_proximity(
    next_message: impl AsyncFnMut()
        -> peppygen::Result<Option<(ProducerRef, collision_status::Message)>>,
    feedback: mpsc::Sender<Feedback>,
    token: CancellationToken,
) {
    consumer::forward_parsed(
        "collision_status",
        token,
        feedback,
        next_message,
        |_producer, msg| Ok(Feedback::Proximity(proximity(msg))),
    )
    .await;
}

/// The proximity one collision_status message reports, received now.
fn proximity(msg: &collision_status::Message) -> Proximity {
    Proximity {
        distance: msg.distance,
        link_a: msg.link_a.clone(),
        link_b: msg.link_b.clone(),
        disposition: Disposition::from_wire(msg.throttled, msg.stopped),
        received_at: Instant::now(),
    }
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use super::*;

    #[tokio::test]
    async fn each_message_reaches_the_owner_as_the_proximity_it_reports() {
        let (feedback, mut owner) = mpsc::channel(1);
        let mut message = Some(collision_status::Message {
            timestamp: SystemTime::UNIX_EPOCH,
            distance: 0.012,
            link_a: "left_link5".to_string(),
            link_b: "right_link6".to_string(),
            throttled: true,
            stopped: false,
        });
        let next_message = async || {
            Ok(message
                .take()
                .map(|msg| (ProducerRef::new("core", "collision_ctrl"), msg)))
        };
        forward_proximity(next_message, feedback, CancellationToken::default()).await;
        let Some(Feedback::Proximity(proximity)) = owner.recv().await else {
            panic!("the owner gets the message's proximity");
        };
        assert_eq!(proximity.distance, 0.012);
        assert_eq!(proximity.link_a, "left_link5");
        assert_eq!(proximity.link_b, "right_link6");
        assert_eq!(proximity.disposition, Disposition::Throttled);
    }
}
