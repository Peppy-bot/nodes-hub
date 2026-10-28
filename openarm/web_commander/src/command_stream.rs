//! Always-on command publisher. Reads the owner's per-tick `CommandFrame` and streams
//! each enabled side's arm setpoint and gripper opening on that limb's pairing
//! slot toward the backbone (the slot is the side, so no id demux), plus the
//! operator's governor controls on `governor_control`. The backbone governs each
//! stream and re-streams the governed value the followers track. A side with
//! `None` in the frame has its deadman off, so nothing is published and the
//! backbone holds that side at its last governed setpoint. Re-publishing every
//! tick (even an unchanged frame) keeps the stream trivially fresh for a
//! backbone that starts or re-pairs mid-session.
//!
//! The owner is the sole writer of state and the sole jog integrator; these tasks only
//! forward the latest frame. Each side+stream runs its own publish task on its own
//! interval, owning its pairing slot's publisher and a clone of the frame receiver.
//! Zenoh publish resolves synchronously, and each task publishes on its own tick, so
//! neither side's setpoint waits behind the other's.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use control_core::report_once::ReportOnce;
use control_core::setpoint_stream::{SetpointSink, stream_setpoints};
use openarm_description::{ArmPair, Side};
use peppygen::NodeRunner;
use peppygen::emitted_topics::governor::governor_control;
use peppygen::paired_topics::{left_arm, left_gripper, right_arm, right_gripper};
use peppylib::runtime::CancellationToken;
use peppylib::{Payload, TopicPublisher};
use srs_model::JointVec;
use tokio::sync::watch;
use tracing::error;

use crate::consumer::clock_now;
use crate::state::{GovernorSettings, Limb, UiState};

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

/// The setpoints to stream this tick. `None` for a side means its deadman is off, so
/// the publisher emits nothing and the backbone holds the last setpoint. Recomputed
/// every command tick and read by the publisher tasks on their own cadence.
#[derive(Clone, Copy, Debug)]
pub struct CommandFrame {
    pub arms: ArmPair<Option<JointVec>>,
    pub grippers: ArmPair<Option<GripperFrame>>,
    pub governor: GovernorSettings,
}

/// One gripper's streamed setpoint: the opening fraction and the operator's
/// effort cap in its wire encoding (0 = no preference).
#[derive(Clone, Copy, Debug)]
pub struct GripperFrame {
    pub opening: f64,
    pub max_effort: f64,
}

impl CommandFrame {
    pub(crate) fn from_state(s: &UiState) -> Self {
        let arm = |side: Side| s.side_active(side).then_some(s.arms[side].joints);
        let grip = |side: Side| {
            s.side_active(side).then_some(GripperFrame {
                opening: s.grippers[side].opening,
                max_effort: s.grippers[side].max_effort.unwrap_or(0.0),
            })
        };
        Self {
            arms: ArmPair::from_fn(arm),
            grippers: ArmPair::from_fn(grip),
            governor: s.governor,
        }
    }
}

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
    let publishers = ArmPair::new(left_arm_pub, right_arm_pub)
        .zip(ArmPair::new(left_gripper_pub, right_gripper_pub));
    let governor_pub = match governor_control::declare_publisher(&runner).await {
        Ok(p) => p,
        Err(e) => {
            error!("declare governor_control publisher: {e}");
            return token.cancel();
        }
    };

    let mut tasks = tokio::task::JoinSet::new();

    // Governor controls: a latched setting. The topic retains the newest
    // message, so a backbone that starts later reads the setting in force
    // and the operator need not touch a control to resend it.
    tasks.spawn(stream_governor_control(
        governor_pub,
        frame_rx.clone(),
        token.clone(),
    ));
    for (side, (arm_pub, gripper_pub)) in publishers {
        // Arm: stream the 7-joint setpoint while enabled (None in the frame =
        // disabled). Velocities and efforts stay empty: the backbone shapes its
        // own velocity feedforward over the governed stream.
        let build_arm = BUILD_ARM_SETPOINT[side];
        let arm_rx = frame_rx.clone();
        tasks.spawn(stream_setpoints(
            SetpointPublisher(arm_pub),
            command_period,
            cancelled(token.clone()),
            format!("{side} {}", Limb::Arm),
            move || {
                let joints = arm_rx.borrow().arms[side]?;
                Some(clock_now().and_then(|timestamp| {
                    build_arm(timestamp, joints.to_vec(), Vec::new(), Vec::new())
                        .map_err(|e| e.to_string())
                }))
            },
        ));
        // Gripper: stream the opening fraction and the operator's effort cap
        // while enabled (mirror of the arm stream above).
        let build_gripper = BUILD_GRIPPER_SETPOINT[side];
        let gripper_rx = frame_rx.clone();
        tasks.spawn(stream_setpoints(
            SetpointPublisher(gripper_pub),
            command_period,
            cancelled(token.clone()),
            format!("{side} {}", Limb::Gripper),
            move || {
                let frame = gripper_rx.borrow().grippers[side]?;
                Some(clock_now().and_then(|timestamp| {
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

/// Publish the operator's governor controls whenever they change, until
/// shutdown.
///
/// The first frame always goes out, so the retained message exists before
/// the operator touches anything. A frame that repeats the setting in force
/// publishes nothing: each message is a change the backbone governs by until
/// the next, and the topic holds the last one for whoever subscribes later.
async fn stream_governor_control(
    publisher: TopicPublisher,
    mut frame_rx: watch::Receiver<CommandFrame>,
    token: CancellationToken,
) {
    let mut sent: Option<GovernorSettings> = None;
    let publish_failures = ReportOnce::default();
    loop {
        let governor = frame_rx.borrow_and_update().governor;
        if !sent
            .as_ref()
            .is_some_and(|last| last.same_setting(&governor))
        {
            let built = governor_control::build_message(
                governor.collision_enabled,
                governor.d_stop,
                governor.d_safe,
                governor.max_ee_velocity_m_s,
                governor.max_gripper_rate_frac_s,
            )
            .map_err(|e| e.to_string());
            let result = match built {
                Ok(msg) => publisher.publish(msg).await.map_err(|e| e.to_string()),
                Err(e) => Err(e),
            };
            match result {
                // Only a delivered setting is recorded, so a failed publish
                // is sent again on the next frame.
                Ok(()) => {
                    sent = Some(governor);
                    publish_failures.clear();
                }
                Err(e) => {
                    if publish_failures.first_of_run() {
                        error!("governor control publish failing: {e}");
                    }
                }
            }
        }
        tokio::select! {
            _ = token.cancelled() => return,
            changed = frame_rx.changed() => {
                if changed.is_err() {
                    return;
                }
            }
        }
    }
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
