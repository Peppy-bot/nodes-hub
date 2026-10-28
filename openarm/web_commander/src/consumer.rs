//! The boundary every consumed feed crosses: the shared consumer loops for
//! owner-feedback topics (receive from a bound-set subscription, parse each
//! message or drop it with a throttled warn, and forward the parsed feedback
//! to the state owner; or receive a two-sided observed feed and forward each
//! side's measurement), and the policy those loops and the alerts listener
//! judge a message by: how often a rejection warns, how long a report stays
//! renderable, and which wire timestamps arrive already too old.

use std::fmt;
use std::ops::ControlFlow;
use std::time::{Duration, Instant, SystemTime};

use control_core::receiving::{Tagged, receive, tagged};
use control_core::throttle::Throttle;
use openarm_description::Side;
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

/// Drive one bound-set subscription until shutdown: `next_message` receives
/// the next message with the producer that sent it (a generated
/// subscription's `next`), and each message is parsed into owner feedback or
/// dropped with a reason, warned at most once per [`REJECT_WARN_PERIOD`]. The
/// parser receives the sending producer, the transport-authenticated identity
/// a bound-set topic attributes reports by. Returns when the token cancels,
/// the subscription ends, or the owner is gone.
pub async fn forward_parsed<M>(
    topic: &'static str,
    token: CancellationToken,
    feedback: mpsc::Sender<Feedback>,
    next_message: impl AsyncFnMut() -> peppygen::Result<Option<(ProducerRef, M)>>,
    mut parse: impl FnMut(&ProducerRef, &M) -> Result<Feedback, String>,
) {
    let mut reject_warn = Throttle::new(REJECT_WARN_PERIOD);
    let rejecting = topic.to_owned();
    receive(
        topic,
        token.cancelled(),
        next_message,
        async move |(producer, msg)| match parse(&producer, &msg) {
            Ok(event) => forwarded(feedback.send(event).await),
            Err(reason) => {
                if reject_warn.admit() {
                    warn!("{rejecting}: dropping message: {reason}");
                }
                ControlFlow::Continue(())
            }
        },
    )
    .await;
}

/// One slot of a two-sided observed feed: the name of the slot, for the log,
/// and the side whose follower it observes.
pub struct ObservedSlot {
    pub name: &'static str,
    pub side: Side,
}

impl fmt::Display for ObservedSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name)
    }
}

/// One receive of a two-sided observed feed: a message with its slot, `None`
/// once a slot's observation closes, or a receive error tagged with its slot.
pub type ObservedReceive<M> =
    Result<Option<(ObservedSlot, M)>, Tagged<ObservedSlot, Box<peppygen::Error>>>;

/// One receive on `slot`'s subscription of a two-sided observed feed, tagged
/// with the slot through [`tagged`]. A closed observation is logged with
/// `closed`, the words that name what stops, and ends the feed.
pub fn observed<M>(
    slot: ObservedSlot,
    closed: &str,
    received: peppygen::Result<Option<M>>,
) -> ObservedReceive<M> {
    if matches!(received, Ok(None)) {
        error!(slot = slot.name, "{closed}");
    }
    tagged(slot, received.map_err(Box::new))
}

/// Report a two-sided observed feed to the owner until `token` cancels,
/// either slot's observation closes, or the owner is gone: `next_message`
/// selects over both slots' subscriptions through [`observed`], so whichever
/// slot delivers next wins and the other stays queued in its own
/// subscription, and `to_feedback` turns one slot's message into owner
/// feedback, or drops it with its own warning.
pub async fn forward_observed<M>(
    topic: &'static str,
    token: CancellationToken,
    feedback: mpsc::Sender<Feedback>,
    next_message: impl AsyncFnMut() -> ObservedReceive<M>,
    mut to_feedback: impl FnMut(ObservedSlot, M) -> Option<Feedback>,
) {
    receive(
        topic,
        token.cancelled(),
        next_message,
        async move |(slot, message)| match to_feedback(slot, message) {
            Some(event) => forwarded(feedback.send(event).await),
            None => ControlFlow::Continue(()),
        },
    )
    .await;
}

/// Whether a feed goes on after a send to the owner: a send that fails means
/// the owner is gone, with nothing left to report to.
pub fn forwarded(sent: Result<(), mpsc::error::SendError<Feedback>>) -> ControlFlow<()> {
    match sent {
        Ok(()) => ControlFlow::Continue(()),
        Err(_) => ControlFlow::Break(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEFT: ObservedSlot = ObservedSlot {
        name: "observed_left_arm",
        side: Side::Left,
    };

    /// The error of a receive whose message did not decode.
    fn receive_error() -> peppygen::Error {
        peppygen::Error::Io(std::io::Error::other("the message does not decode"))
    }

    #[test]
    fn a_closed_observation_ends_the_feed() {
        assert!(matches!(
            observed(LEFT, "observation closed", Ok::<Option<()>, _>(None)),
            Ok(None)
        ));
    }

    #[test]
    fn a_message_and_an_error_carry_their_slot() {
        let Ok(Some((slot, message))) = observed(LEFT, "observation closed", Ok(Some(7))) else {
            panic!("the message comes through");
        };
        assert_eq!(
            (slot.name, slot.side, message),
            ("observed_left_arm", Side::Left, 7)
        );
        let Err(error) = observed::<()>(LEFT, "observation closed", Err(receive_error())) else {
            panic!("the error comes through");
        };
        assert_eq!(error.tag.side, Side::Left);
        assert_eq!(
            error.to_string(),
            format!("observed_left_arm: {}", receive_error())
        );
    }

    #[tokio::test]
    async fn a_feed_ends_once_the_owner_is_gone() {
        let (feedback, owner) = mpsc::channel(1);
        let sent = feedback.send(Feedback::AlertProducers(Vec::new())).await;
        assert_eq!(forwarded(sent), ControlFlow::Continue(()));
        drop(owner);
        let sent = feedback.send(Feedback::AlertProducers(Vec::new())).await;
        assert_eq!(forwarded(sent), ControlFlow::Break(()));
    }

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
