//! Stream plumbing for the gripper, both directions of the gripper_link
//! pairing with the backbone (its `backbone` slot):
//!
//! - A listener for the streamed opening setpoints that keeps the latest one
//!   in a watch channel for the follow loop. Subscribing while unpaired is
//!   legal: the subscription stays silent until a backbone pairs, and only
//!   the paired peer's messages surface, so there is no gripper_id filter. A
//!   setpoint whose opening is non-finite, or whose effort cap the wire
//!   cannot carry, is dropped rather than driving the gripper.
//! - An always-on gripper_states publisher: emits the measured opening,
//!   effort, and effort ceiling at state_rate_hz regardless of mode on the
//!   pairing's `gripper_states` topic (a legal no-op while unpaired; any
//!   monitor observes the pairing). Reads the motor's cached state (no CAN
//!   traffic of its own), so it never contends with the follow loop for the
//!   bus; the follow loop refreshes that cache every tick, so the reading is
//!   always current.

use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use control_core::latched_log::LatchedLog;
use control_core::receiving::receive;
use control_core::sync::lock_ignoring_poison;
use control_core::time::sleep_unless_cancelled;
use peppygen::NodeRunner;
use peppygen::paired_topics::backbone;
use peppylib::runtime::CancellationToken;
use tokio::sync::watch;
use tracing::{error, warn};

use crate::hardware::{Gripper, MaxEffortNm};
use crate::health::clock_now;

/// The latest streamed setpoint for this gripper: the opening the follow
/// loop drives toward.
#[derive(Clone, Copy)]
pub struct GripperSetpoint {
    pub opening: f64,
    /// Commanded effort cap; `None` when the wire carried no preference (0),
    /// so the configured ceiling applies.
    pub max_effort: Option<MaxEffortNm>,
}

/// Keep the latest setpoint the paired backbone streams in `latest` until
/// the subscription closes or `token` cancels.
pub async fn run_setpoint_listener(
    runner: Arc<NodeRunner>,
    latest: watch::Sender<Option<GripperSetpoint>>,
    token: CancellationToken,
) {
    let mut subscription = match backbone::gripper_setpoints::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => {
            error!(error = %e, "gripper_setpoints subscribe");
            return;
        }
    };
    receive(
        "gripper_setpoints",
        token.cancelled(),
        async move || subscription.next().await,
        async move |(_peer, msg)| {
            if !msg.opening.is_finite() {
                warn!(
                    "gripper_setpoints: dropping a setpoint with a non-finite opening {}",
                    msg.opening
                );
                return ControlFlow::Continue(());
            }
            let max_effort = match MaxEffortNm::parse(msg.max_effort) {
                Ok(max_effort) => max_effort,
                Err(reason) => {
                    warn!("gripper_setpoints: dropping a setpoint, {reason}");
                    return ControlFlow::Continue(());
                }
            };
            latest.send_replace(Some(GripperSetpoint {
                opening: msg.opening,
                max_effort,
            }));
            ControlFlow::Continue(())
        },
    )
    .await;
}

pub async fn run_state_publisher(
    runner: Arc<NodeRunner>,
    period: Duration,
    gripper: Arc<Mutex<Gripper>>,
    token: CancellationToken,
) {
    let peer_pub = match backbone::gripper_states::declare_publisher(&runner).await {
        Ok(p) => p,
        Err(e) => return error!("declare paired gripper_states publisher: {e}"),
    };
    let peer_warn = LatchedLog::warn("paired gripper_states publish");
    loop {
        if !sleep_unless_cancelled(period, token.cancelled()).await {
            return;
        }
        let state = lock_ignoring_poison(&gripper).wire_state();
        // Skip a poisoned sample rather than publishing NaN/Inf to consumers,
        // matching the finiteness guards on the command paths.
        if !state.opening.is_finite() || !state.effort.is_finite() {
            warn!(
                "gripper_states: skipping a non-finite motor sample, opening {}, effort {}",
                state.opening, state.effort
            );
            continue;
        }
        let peer_result = match clock_now().and_then(|timestamp| {
            backbone::gripper_states::build_message(
                timestamp,
                state.opening,
                state.effort,
                state.max_effort,
            )
            .map_err(|e| e.to_string())
        }) {
            Ok(msg) => peer_pub.publish(msg).await.map_err(|e| e.to_string()),
            Err(e) => Err(e),
        };
        match peer_result {
            Ok(()) => peer_warn.success(),
            Err(e) => peer_warn.failure(e),
        }
    }
}
