//! The boundary every consumed feed crosses: the shared consumer loop for
//! owner-feedback topics (receive from a bound-set subscription, parse each
//! message or drop it with a throttled warn, and forward the parsed feedback
//! to the state owner), and the policy that loop and the alerts listener
//! judge a message by: how often a rejection warns, how long a report stays
//! renderable, and which wire timestamps arrive already too old.

use std::time::{Duration, Instant, SystemTime};

use control_core::throttle::Throttle;
use control_core::time::RECEIVE_ERROR_BACKOFF;
use peppylib::messaging::ProducerRef;
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::{error, warn};

use crate::owner::Feedback;
use crate::state::Validity;

/// How often a listener may warn about a producer it is rejecting: enough to
/// notice a persistently malformed producer, not enough to bury the log at
/// that producer's own rate.
pub const REJECT_WARN_PERIOD: Duration = Duration::from_secs(1);

/// The motor_health contract's producer cadence mandate: every producer
/// reports each component at least this often.
const HEALTH_REPORT_PERIOD: Duration = Duration::from_millis(500);

/// Health reports age out three contract periods (1500 ms) after receipt:
/// one missed report is transport jitter, three in a row is a producer gone
/// quiet, and the panel falls to not-reporting rather than latching.
pub const HEALTH_STALE_AFTER: Duration = HEALTH_REPORT_PERIOD.saturating_mul(3);

/// The alert contract's producer cadence ceiling: every producer publishes
/// its active set at least this often while it runs.
const ALERT_SET_PERIOD: Duration = Duration::from_millis(2000);

/// A producer's alert set ages out three cadence periods (6000 ms) after
/// receipt, the same three-missed-periods patience as [`HEALTH_STALE_AFTER`].
/// An aged-out set means the producer went quiet and its conditions are
/// unknown; the contract says only an omitted entry says cleared.
pub const ALERT_STALE_AFTER: Duration = ALERT_SET_PERIOD.saturating_mul(3);

/// Slack allowed between a wire timestamp and this consumer's clock before the
/// timestamp counts as pre-aged. Producers timestamp from the daemon-resolved clock
/// this node also reads, so the allowance absorbs scheduling skew, not clock
/// disagreement.
pub const TIMESTAMP_SKEW_ALLOWANCE: Duration = Duration::from_millis(500);

/// Parse a wire timestamp into a report's [`Validity`], rejecting a timestamp already
/// older than the aging window plus [`TIMESTAMP_SKEW_ALLOWANCE`] on the
/// daemon-resolved clock both ends read. A consumer working through a backlog
/// re-stamps queued reports as fresh at receipt; the timestamp check refuses what is already older than the window.
pub fn parse_timestamp_validity(
    timestamp: SystemTime,
    clock_now: SystemTime,
    received_at: Instant,
    live_for: Duration,
) -> Result<Validity, String> {
    // A timestamp ahead of the clock reads as age zero: forward skew is not backlog.
    let age = clock_now
        .duration_since(timestamp)
        .unwrap_or(Duration::ZERO);
    if age > live_for + TIMESTAMP_SKEW_ALLOWANCE {
        return Err(format!(
            "timestamp {} ms old is past its {} ms window",
            age.as_millis(),
            live_for.as_millis()
        ));
    }
    Ok(Validity::new(received_at, live_for))
}

/// Now on this instance's bound clock: the timeline its messages are
/// stamped and aged on. Errors until the clock delivers its first tick.
pub fn clock_now() -> Result<SystemTime, String> {
    let ns = peppygen::clock::now_ns().map_err(|e| format!("clock not ready: {e}"))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aging_windows_are_three_contract_periods() {
        assert_eq!(HEALTH_STALE_AFTER, Duration::from_millis(1500));
        assert_eq!(ALERT_STALE_AFTER, Duration::from_millis(6000));
    }

    #[test]
    fn a_pre_aged_timestamp_rejects_and_a_fresh_or_future_one_parses() {
        let clock = SystemTime::now();
        let t0 = Instant::now();
        let parse = |timestamp| parse_timestamp_validity(timestamp, clock, t0, HEALTH_STALE_AFTER);
        assert!(parse(clock).is_ok());
        assert!(
            parse(clock - (HEALTH_STALE_AFTER + TIMESTAMP_SKEW_ALLOWANCE)).is_ok(),
            "exactly at the allowance still parses"
        );
        assert!(
            parse(
                clock - (HEALTH_STALE_AFTER + TIMESTAMP_SKEW_ALLOWANCE + Duration::from_millis(1))
            )
            .is_err(),
            "a backlogged report must not be re-stamped fresh"
        );
        assert!(
            parse(clock + Duration::from_secs(5)).is_ok(),
            "forward skew is not backlog"
        );
    }
}
