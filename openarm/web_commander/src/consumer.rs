//! The shared consumer loop for owner-feedback topics: receive from a
//! bound-set subscription, parse each message or drop it with a throttled
//! warn, and forward the parsed feedback to the state owner.

use std::time::{Duration, SystemTime};

use control_core::throttle::Throttle;
use peppylib::messaging::ProducerRef;
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::{error, warn};

use crate::owner::Feedback;
use crate::state::REJECT_WARN_PERIOD;

/// Pause after a receive error before retrying, so a persistently failing
/// subscription cannot hot-spin its task or flood the log.
const RECEIVE_ERROR_BACKOFF: Duration = Duration::from_millis(100);

/// The daemon-resolved time (sim time under a simulated clock), for judging
/// a wire timestamp's age on the timeline it was written from. Errs until the
/// clock resolves (in sim mode, until the first tick is observed).
pub fn clock_now() -> Result<SystemTime, String> {
    let ns = peppygen::clock::now_ns().map_err(|e| format!("daemon clock unavailable: {e}"))?;
    Ok(SystemTime::UNIX_EPOCH + Duration::from_nanos(ns))
}

/// A bound-set subscription's receive end, unifying the generated nominal
/// subscription types over [`forward_parsed`]. The returned future is Send
/// so the loop can run under `tokio::spawn`.
pub trait Subscription {
    type Message;
    fn recv(
        &mut self,
    ) -> impl Future<Output = peppygen::Result<Option<(ProducerRef, Self::Message)>>> + Send;
}

/// Drive one subscription until shutdown: each received message is parsed
/// into owner feedback or dropped with a reason, warned at most once per
/// [`REJECT_WARN_PERIOD`]. The parser receives the sending producer, the
/// transport-authenticated identity a bound-set topic attributes reports by.
/// Returns when the token cancels, the subscription ends, or the owner is
/// gone.
pub async fn forward_parsed<S: Subscription>(
    topic: &'static str,
    token: CancellationToken,
    feedback: mpsc::Sender<Feedback>,
    mut subscription: S,
    mut parse: impl FnMut(&ProducerRef, &S::Message) -> Result<Feedback, String>,
) {
    let mut reject_warn = Throttle::new(REJECT_WARN_PERIOD);
    loop {
        let received = tokio::select! {
            _ = token.cancelled() => return,
            received = subscription.recv() => received,
        };
        let (producer, msg) = match received {
            Ok(Some(pair)) => pair,
            Ok(None) => return,
            Err(e) => {
                error!(error = %e, "{topic} receive");
                tokio::select! {
                    _ = token.cancelled() => return,
                    _ = tokio::time::sleep(RECEIVE_ERROR_BACKOFF) => {}
                }
                continue;
            }
        };
        match parse(&producer, &msg) {
            Ok(event) => {
                if feedback.send(event).await.is_err() {
                    return; // the owner is gone; nothing left to report to
                }
            }
            Err(reason) => {
                if reject_warn.admit() {
                    warn!("{topic}: dropping message: {reason}");
                }
            }
        }
    }
}
