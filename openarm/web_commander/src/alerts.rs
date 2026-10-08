//! Operator alerts for the UI. Consumes every producer bound to the alerts
//! slot (zero_or_more), parses each alert set, and hands it to the owner,
//! which holds one set per producer. A set is dropped when its producer
//! leaves the slot or when the producer stops refreshing it.
//!
//! The subscription and the slot's bound set are read by one task, so the
//! owner receives a set and the membership it was sent under in the order
//! they happened. Channel order is causal order, which is what keeps a
//! departed producer's alerts off the panel.
//!
//! The producer is the transport-authenticated instance, so no producer can
//! replace or clear another's alert through the wire strings.

use std::sync::Arc;

use control_core::motor_health::AlertSeverity;
use control_core::throttle::Throttle;
use peppygen::NodeRunner;
use peppygen::consumed_topics::alerts::alerts;
use peppylib::messaging::ProducerRef;
use peppylib::runtime::{CancellationToken, watch_bound_set};
use tokio::sync::mpsc;
use tracing::{error, warn};

use crate::consumer;
use crate::owner::Feedback;
use crate::state::{ALERT_STALE_AFTER, Alert, AlertSet, REJECT_WARN_PERIOD};

/// The slot this node consumes alerts on, as its manifest declares it.
///
/// peppygen exposes the slot's producers but no watch of them, so this name
/// is written out here. A rename in the manifest regenerates the former and
/// leaves this behind, so a slot this node cannot watch stops the node: the
/// operator's alert surface is not something to run without.
const SLOT: &str = "alerts";

/// The most alerts one message may list before the rest are dropped.
///
/// The slot takes any producer, so the panel's list is as long as its
/// producers assert. A bound sized above every motor of a two-arm robot
/// keeps a runaway source generator off the panel.
const MAX_ACTIVE_ALERTS: usize = 64;

/// Whether this deployment binds any alerts producer.
pub fn available(runner: &NodeRunner) -> bool {
    !alerts::bound_producers(runner).is_empty()
}

pub async fn run(
    runner: Arc<NodeRunner>,
    feedback: mpsc::Sender<Feedback>,
    token: CancellationToken,
) {
    let producers = match watch_bound_set(&runner, SLOT) {
        Ok(producers) => producers,
        Err(e) => {
            error!(error = %e, "alerts bound set: the `{SLOT}` slot is not this node's");
            return token.cancel();
        }
    };
    let mut subscription = match alerts::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => {
            error!(error = %e, "alerts subscribe");
            return;
        }
    };
    let mut producers = producers;
    let mut reject_warn = Throttle::new(REJECT_WARN_PERIOD);
    // The membership in hand before the first set, so a producer that was
    // already bound renders its alerts.
    if !send_producers(&feedback, &mut producers).await {
        return;
    }
    loop {
        let received = tokio::select! {
            _ = token.cancelled() => return,
            changed = producers.changed() => {
                if changed.is_err() {
                    return;
                }
                if !send_producers(&feedback, &mut producers).await {
                    return;
                }
                continue;
            }
            received = subscription.next() => received,
        };
        let (producer, msg) = match received {
            Ok(Some(pair)) => pair,
            Ok(None) => return,
            Err(e) => {
                error!(error = %e, "alerts receive");
                tokio::select! {
                    _ = token.cancelled() => return,
                    _ = tokio::time::sleep(consumer::RECEIVE_ERROR_BACKOFF) => {}
                }
                continue;
            }
        };
        // An unresolved daemon clock cannot certify a set's age, so the set
        // drops on the same throttled-warn path as a malformed one.
        let parsed = consumer::clock_now()
            .and_then(|clock_now| parse_alert_set(&msg, clock_now, std::time::Instant::now()));
        match parsed {
            Ok(Parsed { set, refused }) => {
                if let Some(reason) = refused.filter(|_| reject_warn.admit()) {
                    warn!(producer = %producer.instance_id, "alerts entry dropped: {reason}");
                }
                let feedback_sent = feedback
                    .send(Feedback::AlertSet {
                        producer: producer.clone(),
                        set,
                    })
                    .await;
                if feedback_sent.is_err() {
                    return;
                }
            }
            Err(e) => {
                if reject_warn.admit() {
                    warn!(producer = %producer.instance_id, "alerts set dropped: {e}");
                }
            }
        }
    }
}

/// Tell the owner which producers the slot holds, so the alerts of one that
/// left go with it. Answers false when the owner is gone.
async fn send_producers(
    feedback: &mpsc::Sender<Feedback>,
    producers: &mut tokio::sync::watch::Receiver<peppylib::messaging::BoundSetState>,
) -> bool {
    let held: Vec<ProducerRef> = producers
        .borrow_and_update()
        .producers
        .iter()
        .map(|member| member.producer.clone())
        .collect();
    feedback.send(Feedback::AlertProducers(held)).await.is_ok()
}

/// One parsed set, and the first entry it dropped.
struct Parsed {
    set: AlertSet,
    refused: Option<String>,
}

/// Parse one wire alert set.
///
/// An entry with no identity, a severity outside the contract's scale, or an
/// identity another entry already carries is dropped and the rest of the set
/// stands: a producer's one bad entry must not blank the alerts of the motors
/// it reported correctly. A timestamp already past the aging window belongs
/// to the message rather than to any entry, so it refuses the whole set and a
/// backlogged consumer cannot re-stamp an old set as current.
fn parse_alert_set(
    msg: &alerts::Message,
    clock_now: std::time::SystemTime,
    received_at: std::time::Instant,
) -> Result<Parsed, String> {
    let validity = crate::state::parse_timestamp_validity(
        msg.timestamp,
        clock_now,
        received_at,
        ALERT_STALE_AFTER,
    )?;
    let mut alerts: Vec<Alert> = Vec::with_capacity(msg.active.len().min(MAX_ACTIVE_ALERTS));
    let mut seen: Vec<(&str, &str)> = Vec::with_capacity(alerts.capacity());
    let mut refused: Option<String> = None;
    let mut refuse = |reason: String| {
        if refused.is_none() {
            refused = Some(reason);
        }
    };
    for item in &msg.active {
        if alerts.len() == MAX_ACTIVE_ALERTS {
            refuse(format!("more than {MAX_ACTIVE_ALERTS} alerts listed"));
            break;
        }
        if item.source.is_empty() || item.kind.is_empty() {
            refuse("empty source or kind".to_string());
            continue;
        }
        let Some(severity) = AlertSeverity::from_wire(item.severity) else {
            refuse(format!("undefined severity {}", item.severity));
            continue;
        };
        if seen.contains(&(item.source.as_str(), item.kind.as_str())) {
            refuse(format!("`{}` `{}` listed twice", item.source, item.kind));
            continue;
        }
        seen.push((item.source.as_str(), item.kind.as_str()));
        alerts.push(Alert {
            source: item.source.clone(),
            severity,
            message: item.message.clone(),
        });
    }
    Ok(Parsed {
        set: AlertSet { alerts, validity },
        refused,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant, SystemTime};

    fn item(source: &str, severity: u8) -> alerts::MessageActiveItem {
        alerts::MessageActiveItem {
            source: source.to_string(),
            kind: "motor_condition".to_string(),
            severity,
            message: "holding 93% of rated torque".to_string(),
        }
    }

    fn parse(active: Vec<alerts::MessageActiveItem>) -> Parsed {
        let now = SystemTime::now();
        let msg = alerts::Message {
            timestamp: now,
            active,
        };
        parse_alert_set(&msg, now, Instant::now()).expect("a fresh timestamp parses")
    }

    #[test]
    fn a_well_formed_set_parses_including_an_empty_one() {
        let parsed = parse(vec![item("left arm j2", 2), item("left arm j5", 1)]);
        assert!(parsed.refused.is_none());
        assert_eq!(parsed.set.alerts.len(), 2);
        assert_eq!(parsed.set.alerts[0].source, "left arm j2");
        assert_eq!(parsed.set.alerts[0].severity, AlertSeverity::Critical);
        assert_eq!(parsed.set.alerts[1].severity, AlertSeverity::Warning);
        assert!(parse(Vec::new()).set.alerts.is_empty());
    }

    #[test]
    fn two_kinds_on_one_source_both_parse() {
        let mut other = item("left arm j2", 1);
        other.kind = "encoder".to_string();
        let parsed = parse(vec![item("left arm j2", 2), other]);
        assert!(parsed.refused.is_none(), "{:?}", parsed.refused);
        assert_eq!(parsed.set.alerts.len(), 2, "identity is (source, kind)");
    }

    #[test]
    fn a_bad_entry_is_dropped_and_the_rest_of_the_set_stands() {
        // A producer's one bad entry must not blank the motors it reported
        // correctly: the operator needs the fault that is real.
        for bad in [item("", 3), item("left arm j9", 0), item("left arm j9", 4)] {
            let parsed = parse(vec![bad, item("left arm j5", 3)]);
            assert!(parsed.refused.is_some(), "the entry is reported");
            assert_eq!(parsed.set.alerts.len(), 1, "the good entry stands");
            assert_eq!(parsed.set.alerts[0].source, "left arm j5");
        }
        let mut no_kind = item("left arm j9", 3);
        no_kind.kind = String::new();
        let parsed = parse(vec![no_kind, item("left arm j5", 3)]);
        assert!(parsed.refused.is_some());
        assert_eq!(parsed.set.alerts.len(), 1);
    }

    #[test]
    fn a_repeated_identity_keeps_the_first_entry() {
        let mut second = item("left arm j2", 1);
        second.message = "later".to_string();
        let parsed = parse(vec![item("left arm j2", 2), second]);
        assert!(parsed.refused.is_some());
        assert_eq!(parsed.set.alerts.len(), 1);
        assert_eq!(
            parsed.set.alerts[0].severity,
            AlertSeverity::Critical,
            "the first entry stands"
        );
    }

    #[test]
    fn a_runaway_set_is_cut_at_the_bound() {
        let long: Vec<_> = (0..MAX_ACTIVE_ALERTS + 10)
            .map(|i| item(&format!("joint {i}"), 1))
            .collect();
        let parsed = parse(long);
        assert_eq!(parsed.set.alerts.len(), MAX_ACTIVE_ALERTS);
        assert!(parsed.refused.is_some());
    }

    #[test]
    fn a_set_already_past_its_window_is_refused_whole() {
        // A backlogged consumer must not re-stamp an old set as current.
        let clock_now = SystemTime::now();
        let msg = alerts::Message {
            timestamp: clock_now - ALERT_STALE_AFTER - Duration::from_secs(1),
            active: vec![item("left arm j2", 2)],
        };
        assert!(parse_alert_set(&msg, clock_now, Instant::now()).is_err());
    }
}
