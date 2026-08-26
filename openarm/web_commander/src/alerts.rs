//! Operator alerts for the UI. Consumes every producer bound to the alerts
//! slot (zero_or_more), validates each message, and hands it to the owner,
//! which holds one alert per (producer, source, kind) and drops it on a
//! severity-0 clear or when its producer stops re-emitting. The producer is
//! the transport-authenticated instance, so no producer can replace or
//! clear another's alert through the wire strings.

use std::sync::Arc;
use std::time::{Instant, SystemTime};

use peppygen::NodeRunner;
use peppygen::consumed_topics::alerts::alerts;
use peppylib::messaging::ProducerRef;
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::error;

use crate::consumer;
use crate::owner::Feedback;
use crate::state::{ALERT_STALE_AFTER, Alert, parse_timestamp_validity};

/// The alert contract's severity ceiling: 0 clear, 1 warning, 2 critical,
/// 3 fault.
const MAX_ALERT_SEVERITY: u8 = 3;

/// Whether this deployment binds any alerts producer.
pub fn available(runner: &NodeRunner) -> bool {
    !alerts::bound_producers(runner).is_empty()
}

impl consumer::Subscription for alerts::Subscription {
    type Message = alerts::Message;
    async fn recv(&mut self) -> peppygen::Result<Option<(ProducerRef, Self::Message)>> {
        self.next().await
    }
}

pub async fn run(
    runner: Arc<NodeRunner>,
    feedback: mpsc::Sender<Feedback>,
    token: CancellationToken,
) {
    let subscription = match alerts::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => {
            error!(error = %e, "alerts subscribe");
            return;
        }
    };
    consumer::forward_parsed(
        "alerts",
        token,
        feedback,
        subscription,
        // An unresolved daemon clock cannot certify a timestamp's age, so the
        // alert drops on the same throttled-warn path as a malformed one.
        |producer, msg| {
            parse_alert(producer, msg, consumer::clock_now()?, Instant::now()).map(Feedback::Alert)
        },
    )
    .await;
}

/// Parse one wire alert: a non-empty identity, a defined severity, and a
/// timestamp not already past the aging window (a backlogged consumer must not
/// re-stamp a stale alert as fresh). The producing instance becomes part of
/// the alert's identity, scoping replaces and clears to their own producer.
fn parse_alert(
    producer: &ProducerRef,
    msg: &alerts::Message,
    clock_now: SystemTime,
    received_at: Instant,
) -> Result<Alert, String> {
    if msg.source.is_empty() || msg.kind.is_empty() {
        return Err("empty source or kind".to_string());
    }
    if msg.severity > MAX_ALERT_SEVERITY {
        return Err(format!("undefined severity {}", msg.severity));
    }
    Ok(Alert {
        producer: format!("{}/{}", producer.core_node, producer.instance_id),
        source: msg.source.clone(),
        kind: msg.kind.clone(),
        severity: msg.severity,
        message: msg.message.clone(),
        validity: parse_timestamp_validity(
            msg.timestamp,
            clock_now,
            received_at,
            ALERT_STALE_AFTER,
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::TIMESTAMP_SKEW_ALLOWANCE;
    use std::time::Duration;

    fn msg(severity: u8) -> alerts::Message {
        alerts::Message {
            timestamp: SystemTime::now(),
            source: "left arm j2".to_string(),
            kind: "motor_overload".to_string(),
            severity,
            message: "holding 93% of rated torque".to_string(),
        }
    }

    /// Parse against a clock equal to the timestamp, so only shape can fail.
    fn parse_fresh(msg: &alerts::Message) -> Result<Alert, String> {
        let producer = ProducerRef::new("core", "left_arm_inst");
        parse_alert(&producer, msg, msg.timestamp, Instant::now())
    }

    #[test]
    fn well_formed_alerts_parse_including_clears() {
        let raised = parse_fresh(&msg(2)).unwrap();
        assert_eq!(raised.severity, 2);
        assert_eq!(raised.source, "left arm j2");
        assert_eq!(raised.producer, "core/left_arm_inst");
        let cleared = parse_fresh(&msg(0)).unwrap();
        assert_eq!(cleared.severity, 0);
    }

    #[test]
    fn malformed_alerts_reject() {
        let mut empty_source = msg(1);
        empty_source.source = String::new();
        assert!(parse_fresh(&empty_source).is_err());

        let mut empty_kind = msg(1);
        empty_kind.kind = String::new();
        assert!(parse_fresh(&empty_kind).is_err());

        assert!(parse_fresh(&msg(4)).is_err());
    }

    #[test]
    fn a_pre_aged_timestamp_rejects_the_alert() {
        let producer = ProducerRef::new("core", "left_arm_inst");
        let m = msg(2);
        let just_inside = m.timestamp + ALERT_STALE_AFTER + TIMESTAMP_SKEW_ALLOWANCE;
        assert!(parse_alert(&producer, &m, just_inside, Instant::now()).is_ok());
        let past = just_inside + Duration::from_millis(1);
        assert!(parse_alert(&producer, &m, past, Instant::now()).is_err());
    }
}
