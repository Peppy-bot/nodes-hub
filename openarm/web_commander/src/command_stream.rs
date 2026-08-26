// Always-on command publisher. Reads the owner's per-tick `CommandFrame` and streams
// each enabled side's arm setpoint and gripper opening on that limb's pairing
// slot toward the backbone (the slot is the side, so no id demux), plus the
// operator's governor controls on `governor_control`. The backbone governs each
// stream and re-streams the governed value the followers track. A side with
// `None` in the frame has its deadman off, so nothing is published and the
// backbone holds that side at its last governed setpoint. Re-publishing every
// tick (even an unchanged frame) keeps the stream trivially fresh for a
// backbone that starts or re-pairs mid-session.
//
// The owner is the sole writer of state and the sole jog integrator; these tasks only
// forward the latest frame. Each side+stream runs its own publish task on its own
// interval, cloning the shared per-topic publisher and the frame receiver. A single
// shared loop publishing Left then Right would leave Right permanently second (zenoh
// publish resolves synchronously), so independent tasks avoid that bias.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use peppygen::NodeRunner;
use peppygen::emitted_topics::governor::governor_control;
use peppygen::paired_topics::{left_arm, left_gripper, right_arm, right_gripper};
use peppylib::runtime::CancellationToken;
use peppylib::{Payload, TopicPublisher};
use tokio::sync::watch;
use tokio::time::MissedTickBehavior;
use tracing::{error, warn};

use crate::owner::CommandFrame;
use crate::state::Side;

/// Pairing timestamp from the daemon-resolved clock (sim time under a simulated
/// clock), so the backbone ages setpoints on the same timeline it reads.
/// Errors until the clock delivers its first tick.
fn pairing_timestamp() -> Result<SystemTime, String> {
    let ns = peppygen::clock::now_ns().map_err(|e| format!("clock not ready: {e}"))?;
    Ok(UNIX_EPOCH + Duration::from_nanos(ns))
}

type BuildJointSetpoint = fn(SystemTime, Vec<f64>, Vec<f64>, Vec<f64>) -> peppygen::Result<Payload>;
type BuildGripperSetpoint = fn(SystemTime, f64, f64) -> peppygen::Result<Payload>;

pub async fn run(
    runner: Arc<NodeRunner>,
    command_period: Duration,
    token: CancellationToken,
    frame_rx: watch::Receiver<CommandFrame>,
) {
    // A failed publisher declaration leaves the node serving UI/health but unable to
    // command anything, so cancel the node to restart it rather than returning quietly.
    // One publisher per pairing slot; publishing while unbound is a legal no-op,
    // so a monitor-only deployment simply streams nothing.
    let (left_arm_pub, right_arm_pub, left_gripper_pub, right_gripper_pub) = match tokio::try_join!(
        left_arm::joint_setpoints::declare_publisher(&runner),
        right_arm::joint_setpoints::declare_publisher(&runner),
        left_gripper::gripper_setpoints::declare_publisher(&runner),
        right_gripper::gripper_setpoints::declare_publisher(&runner),
    ) {
        Ok(pubs) => pubs,
        Err(e) => {
            error!("declare pairing setpoint publishers: {e}");
            return token.cancel();
        }
    };
    let governor_pub = match governor_control::declare_publisher(&runner).await {
        Ok(p) => p,
        Err(e) => {
            error!("declare governor_control publisher: {e}");
            return token.cancel();
        }
    };

    let mut tasks = tokio::task::JoinSet::new();

    // Governor controls: no deadman, so this always publishes the latest frame.
    let governor_rx = frame_rx.clone();
    tasks.spawn(stream_setpoints(
        governor_pub,
        command_period,
        token.clone(),
        "governor control".to_string(),
        move || {
            let g = governor_rx.borrow().governor;
            Some(
                governor_control::build_message(
                    g.collision_enabled,
                    g.d_stop,
                    g.d_safe,
                    g.max_ee_velocity_m_s,
                    g.max_gripper_rate_frac_s,
                )
                .map_err(|e| e.to_string()),
            )
        },
    ));
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
        // Arm: stream the 7-joint setpoint while enabled (None in the frame =
        // disabled). Velocities and efforts stay empty: the backbone shapes its
        // own velocity feedforward over the governed stream.
        let arm_rx = frame_rx.clone();
        tasks.spawn(stream_setpoints(
            arm_pub,
            command_period,
            token.clone(),
            format!("{} arm", side.label()),
            move || {
                let joints = arm_rx.borrow().arms[side]?;
                Some(pairing_timestamp().and_then(|timestamp| {
                    build_arm(timestamp, joints.to_vec(), Vec::new(), Vec::new())
                        .map_err(|e| e.to_string())
                }))
            },
        ));
        // Gripper: stream the opening fraction and the operator's effort cap
        // while enabled (mirror of the arm stream above).
        let gripper_rx = frame_rx.clone();
        tasks.spawn(stream_setpoints(
            gripper_pub,
            command_period,
            token.clone(),
            format!("{} gripper", side.label()),
            move || {
                let frame = gripper_rx.borrow().grippers[side]?;
                Some(pairing_timestamp().and_then(|timestamp| {
                    build_gripper(timestamp, frame.opening, frame.max_effort)
                        .map_err(|e| e.to_string())
                }))
            },
        ));
    }
    // join_next surfaces tasks in completion order, so a panicked stream is seen
    // immediately. A dead channel would silently hold its side while the node reports
    // healthy, which is worse than a restart: cancel the node.
    while let Some(result) = tasks.join_next().await {
        if let Err(e) = result {
            error!("command stream task died: {e}; cancelling the node");
            token.cancel();
        }
    }
}

// Publish the latest setpoint from `next_message` every `period`, skipping a tick
// whenever it returns None (the side is disabled). Failures latch so a stuck side warns
// once, not every tick. The period arrives already validated, so this side never
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
