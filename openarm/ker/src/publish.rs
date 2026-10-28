//! Always-on command publisher, the same shape as the commander's: for each
//! side, one task streams the arm setpoint at command_rate_hz on that limb's
//! joint_link pairing slot and one streams the commanded gripper opening on
//! its gripper_link slot (the slot is the side, so no id demux); the backbone
//! governs everything before it reaches a follower. A tick publishes nothing
//! when the newest sample is missing or stale, or its arm is not engaged, so
//! that limb holds at its last governed setpoints: skipping is the deadman. Re-publishing an
//! unchanged sample every tick keeps the stream trivially fresh for a backbone
//! that starts mid-session.
//!
//! Each side+stream runs its own publish task on its own interval, owning its
//! pairing slot's publisher. Zenoh publish resolves synchronously, and each
//! task publishes on its own tick, so neither side's setpoint waits behind the
//! other's.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use control_core::setpoint_stream::{SetpointSink, stream_setpoints};
use openarm_description::{ArmPair, Side};
use peppygen::NodeRunner;
use peppygen::paired_topics::{left_arm, left_gripper, right_arm, right_gripper};
use peppylib::runtime::CancellationToken;
use peppylib::{Payload, TopicPublisher};
use tokio::sync::watch;

use crate::reader::KerSample;

/// Now on this instance's bound clock: the timeline its messages are
/// stamped and aged on. Errors until the clock delivers its first tick.
fn clock_now() -> Result<SystemTime, String> {
    let ns = peppygen::clock::now_ns().map_err(|e| format!("clock not ready: {e}"))?;
    Ok(UNIX_EPOCH + Duration::from_nanos(ns))
}

type BuildJointSetpoint = fn(SystemTime, Vec<f64>, Vec<f64>, Vec<f64>) -> peppygen::Result<Payload>;
type BuildGripperSetpoint = fn(SystemTime, f64, f64) -> peppygen::Result<Payload>;

/// Each arm's pairing slot message builder.
const BUILD_ARM_SETPOINT: ArmPair<BuildJointSetpoint> = ArmPair::new(
    left_arm::joint_setpoints::build_message,
    right_arm::joint_setpoints::build_message,
);
/// Each gripper's pairing slot message builder.
const BUILD_GRIPPER_SETPOINT: ArmPair<BuildGripperSetpoint> = ArmPair::new(
    left_gripper::gripper_setpoints::build_message,
    right_gripper::gripper_setpoints::build_message,
);

/// Why the publisher stopped commanding. The supervisor decides what that
/// means for the node; this only reports.
#[derive(Debug, thiserror::Error)]
pub enum PublishFault {
    #[error("declare the pairing setpoint publishers")]
    Declare(#[source] peppygen::Error),

    #[error("a command stream task died")]
    StreamTask(#[source] tokio::task::JoinError),
}

pub async fn run(
    runner: Arc<NodeRunner>,
    rx: watch::Receiver<Option<KerSample>>,
    command_period: Duration,
    stale_timeout: Duration,
    token: CancellationToken,
) -> Result<(), PublishFault> {
    // A failed publisher declaration leaves the node connected to the device
    // but unable to command anything. One publisher per pairing slot;
    // publishing while unbound is a legal no-op.
    let (left_arm_pub, right_arm_pub, left_gripper_pub, right_gripper_pub) = tokio::try_join!(
        left_arm::joint_setpoints::declare_publisher(&runner),
        right_arm::joint_setpoints::declare_publisher(&runner),
        left_gripper::gripper_setpoints::declare_publisher(&runner),
        right_gripper::gripper_setpoints::declare_publisher(&runner),
    )
    .map_err(PublishFault::Declare)?;
    let publishers = ArmPair::new(left_arm_pub, right_arm_pub)
        .zip(ArmPair::new(left_gripper_pub, right_gripper_pub));

    let mut tasks = tokio::task::JoinSet::new();

    for (side, (arm_pub, gripper_pub)) in publishers {
        // Arm: velocities and efforts stay empty; the backbone shapes its own
        // velocity feedforward over the governed stream.
        let build_arm = BUILD_ARM_SETPOINT[side];
        let sample_rx = rx.clone();
        tasks.spawn(stream_setpoints(
            SetpointPublisher(arm_pub),
            command_period,
            cancelled(token.clone()),
            format!("{side} arm"),
            move || {
                let target = streamable(&sample_rx, stale_timeout, side)?.joints[side];
                Some(clock_now().and_then(|timestamp| {
                    build_arm(timestamp, target.to_vec(), Vec::new(), Vec::new())
                        .map_err(|e| e.to_string())
                }))
            },
        ));
        // Gripper: stream the commanded opening while its arm streams. The
        // leader trigger carries no effort source: max_effort 0 (no
        // preference) leaves the follower's ceiling in charge.
        let build_gripper = BUILD_GRIPPER_SETPOINT[side];
        let sample_rx = rx.clone();
        tasks.spawn(stream_setpoints(
            SetpointPublisher(gripper_pub),
            command_period,
            cancelled(token.clone()),
            format!("{side} gripper"),
            move || {
                let opening = streamable(&sample_rx, stale_timeout, side)?.gripper_openings[side];
                Some(clock_now().and_then(|timestamp| {
                    build_gripper(timestamp, opening, 0.0).map_err(|e| e.to_string())
                }))
            },
        ));
    }
    // join_next surfaces tasks in completion order, so a panicked stream is
    // seen immediately. A dead channel would silently hold its side while the
    // node reports healthy, which is worse than a restart.
    while let Some(result) = tasks.join_next().await {
        result.map_err(PublishFault::StreamTask)?;
    }
    Ok(())
}

/// The newest sample if `side` should stream: present, that arm engaged, and
/// fresher than the stale timeout. `None` skips the tick, which holds the limb.
fn streamable(
    rx: &watch::Receiver<Option<KerSample>>,
    stale_timeout: Duration,
    side: Side,
) -> Option<KerSample> {
    let sample = rx.borrow().clone()?;
    (sample.engaged[side] && sample.received_at.elapsed() < stale_timeout).then_some(sample)
}

/// One pairing slot's publisher, as the sink of its setpoint stream.
struct SetpointPublisher(TopicPublisher);

impl SetpointSink for SetpointPublisher {
    type Message = Payload;

    async fn send(&self, message: Payload) -> Result<(), String> {
        self.0.publish(message).await.map_err(|e| e.to_string())
    }
}

/// Resolves once `token` is cancelled.
async fn cancelled(token: CancellationToken) {
    token.cancelled().await;
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::engage::fixtures::{BOTH_ENGAGED, STALE};

    const RIGHT_ENGAGED: ArmPair<bool> = ArmPair::new(false, true);
    /// Distinguishable per side, so a swapped accessor cannot pass.
    const LEFT_OPENING: f64 = 0.25;
    const RIGHT_OPENING: f64 = 0.75;

    fn sample(engaged: ArmPair<bool>, age: Duration) -> KerSample {
        KerSample {
            joints: ArmPair::new([0.1; 7], [0.2; 7]),
            gripper_openings: ArmPair::new(LEFT_OPENING, RIGHT_OPENING),
            engaged,
            received_at: Instant::now() - age,
        }
    }

    #[test]
    fn streams_only_fresh_samples_for_an_engaged_arm() {
        let (tx, rx) = watch::channel(None);
        assert!(
            streamable(&rx, STALE, Side::Right).is_none(),
            "no sample yet"
        );

        tx.send(Some(sample(RIGHT_ENGAGED, Duration::ZERO)))
            .unwrap();
        assert!(streamable(&rx, STALE, Side::Right).is_some());
        assert!(
            streamable(&rx, STALE, Side::Left).is_none(),
            "an unengaged arm holds"
        );

        tx.send(Some(sample(RIGHT_ENGAGED, STALE))).unwrap();
        assert!(streamable(&rx, STALE, Side::Right).is_none(), "stale holds");

        tx.send(None).unwrap();
        assert!(
            streamable(&rx, STALE, Side::Right).is_none(),
            "device loss holds"
        );
    }

    #[test]
    fn the_stale_window_holds_at_its_own_edge() {
        let (tx, rx) = watch::channel(None);
        tx.send(Some(sample(
            BOTH_ENGAGED,
            STALE - Duration::from_millis(100),
        )))
        .unwrap();
        assert!(
            streamable(&rx, STALE, Side::Left).is_some(),
            "inside the window still streams"
        );

        tx.send(Some(sample(
            BOTH_ENGAGED,
            STALE + Duration::from_millis(100),
        )))
        .unwrap();
        assert!(
            streamable(&rx, STALE, Side::Left).is_none(),
            "past the window holds"
        );
    }

    #[test]
    fn each_side_reads_its_own_values() {
        let (tx, rx) = watch::channel(None);
        tx.send(Some(sample(BOTH_ENGAGED, Duration::ZERO))).unwrap();
        let left = streamable(&rx, STALE, Side::Left).expect("engaged");
        let right = streamable(&rx, STALE, Side::Right).expect("engaged");
        assert_eq!(left.gripper_openings[Side::Left], LEFT_OPENING);
        assert_eq!(right.gripper_openings[Side::Right], RIGHT_OPENING);
        assert_eq!(left.joints[Side::Left], [0.1; 7]);
        assert_eq!(right.joints[Side::Right], [0.2; 7]);
    }
}
