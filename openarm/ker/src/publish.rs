// Always-on command publisher, the same shape as the commander's: for each
// side, one task streams the arm setpoint at command_rate_hz on that limb's
// joint_link pairing slot and one streams the trigger opening on its
// gripper_link slot (the slot is the side, so no id demux); the backbone
// governs everything before it reaches a follower. A tick publishes nothing
// when the newest sample is missing or stale, or its arm is not engaged, so
// that limb holds at its last governed setpoints: skipping is the deadman. Re-publishing an
// unchanged sample every tick keeps the stream trivially fresh for a backbone
// that starts mid-session.
//
// Each side+stream runs its own publish task on its own interval, cloning the
// shared per-topic publisher. A single shared loop publishing Left then Right
// would leave Right permanently second (zenoh publish resolves synchronously),
// so independent tasks avoid that bias.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use openarm_description::Side;
use peppygen::NodeRunner;
use peppygen::paired_topics::{left_arm, left_gripper, right_arm, right_gripper};
use peppylib::runtime::CancellationToken;
use peppylib::{Payload, TopicPublisher};
use tokio::sync::watch;
use tokio::time::MissedTickBehavior;
use tracing::warn;

use crate::reader::KerSample;

/// Pairing timestamp from the daemon-resolved clock, so the backbone ages
/// setpoints on the same timeline it reads. Errors until the clock delivers
/// its first tick.
fn pairing_timestamp() -> Result<SystemTime, String> {
    let ns = peppygen::clock::now_ns().map_err(|e| format!("clock not ready: {e}"))?;
    Ok(UNIX_EPOCH + Duration::from_nanos(ns))
}

type BuildJointSetpoint = fn(SystemTime, Vec<f64>, Vec<f64>, Vec<f64>) -> peppygen::Result<Payload>;
type BuildGripperSetpoint = fn(SystemTime, f64, f64) -> peppygen::Result<Payload>;

fn label(side: Side) -> &'static str {
    match side {
        Side::Left => "left",
        Side::Right => "right",
    }
}

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

    let mut tasks = tokio::task::JoinSet::new();

    for (side, arm_pub, build_arm, gripper_pub, build_gripper) in [
        (
            Side::Left,
            left_arm_pub,
            left_arm::joint_setpoints::build_message as BuildJointSetpoint,
            left_gripper_pub,
            left_gripper::gripper_setpoints::build_message as BuildGripperSetpoint,
        ),
        (
            Side::Right,
            right_arm_pub,
            right_arm::joint_setpoints::build_message as BuildJointSetpoint,
            right_gripper_pub,
            right_gripper::gripper_setpoints::build_message as BuildGripperSetpoint,
        ),
    ] {
        // Arm: velocities and efforts stay empty; the backbone shapes its own
        // velocity feedforward over the governed stream.
        let sample_rx = rx.clone();
        tasks.spawn(stream_setpoints(
            arm_pub,
            command_period,
            token.clone(),
            format!("{} arm", label(side)),
            move || {
                let target = streamable(&sample_rx, stale_timeout, side)?.joints(side);
                Some(pairing_timestamp().and_then(|timestamp| {
                    build_arm(timestamp, target.to_vec(), Vec::new(), Vec::new())
                        .map_err(|e| e.to_string())
                }))
            },
        ));
        // Gripper: stream the trigger opening fraction while its arm streams. The leader trigger carries no effort
        // source: max_effort 0 (no preference) leaves the follower's ceiling
        // in charge.
        let sample_rx = rx.clone();
        tasks.spawn(stream_setpoints(
            gripper_pub,
            command_period,
            token.clone(),
            format!("{} gripper", label(side)),
            move || {
                let opening = streamable(&sample_rx, stale_timeout, side)?.opening(side);
                Some(pairing_timestamp().and_then(|timestamp| {
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
    (sample.engaged.side(side) && sample.received_at.elapsed() < stale_timeout).then_some(sample)
}

// Publish the latest setpoint from `next_message` every `period`, skipping a
// tick whenever it returns None. Failures latch so a stuck channel warns once,
// not every tick. The period arrives already validated, so this side never
// divides by a rate it has to trust.
async fn stream_setpoints(
    publisher: TopicPublisher,
    period: Duration,
    token: CancellationToken,
    label: String,
    mut next_message: impl FnMut() -> Option<Result<Payload, String>>,
) {
    // interval (not sleep) so the publish cadence holds at the commanded rate
    // instead of drifting by the per-tick work time; Delay avoids a catch-up
    // burst after a scheduling hiccup.
    let mut ticker = tokio::time::interval(period);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut failing = false;

    loop {
        tokio::select! {
            _ = token.cancelled() => return,
            _ = ticker.tick() => {}
        }

        let Some(built) = next_message() else {
            continue;
        };
        let result = match built {
            Ok(msg) => publisher.publish(msg).await.map_err(|e| e.to_string()),
            Err(e) => Err(e),
        };
        match result {
            Ok(()) => failing = false,
            Err(e) if !failing => {
                failing = true;
                warn!("{label} command publish failing, suppressing repeats: {e}");
            }
            Err(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::reader::Engaged;

    const STALE: Duration = Duration::from_millis(250);
    const RIGHT_ONLY: Engaged = Engaged {
        left: false,
        right: true,
    };

    fn sample(engaged: Engaged, age: Duration) -> KerSample {
        KerSample {
            left_joints: [0.0; 7],
            right_joints: [0.0; 7],
            left_opening: 0.0,
            right_opening: 0.0,
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

        tx.send(Some(sample(RIGHT_ONLY, Duration::ZERO))).unwrap();
        assert!(streamable(&rx, STALE, Side::Right).is_some());
        assert!(
            streamable(&rx, STALE, Side::Left).is_none(),
            "an unengaged arm holds"
        );

        tx.send(Some(sample(RIGHT_ONLY, STALE))).unwrap();
        assert!(streamable(&rx, STALE, Side::Right).is_none(), "stale holds");

        tx.send(None).unwrap();
        assert!(
            streamable(&rx, STALE, Side::Right).is_none(),
            "device loss holds"
        );
    }
}
