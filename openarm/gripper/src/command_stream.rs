// Listens for streamed opening setpoints from the paired backbone (the
// `backbone` pairing slot of gripper_link) and keeps the latest one
// in a watch channel for the follow loop. Subscribing while unpaired is legal:
// the subscription stays silent until a backbone pairs, and only the paired
// peer's messages surface, so there is no gripper_id filter. A setpoint whose
// opening is non-finite, or whose effort cap the wire cannot carry, is dropped
// rather than driving the gripper. stream.rs is the return direction; this is
// the command direction.

use std::sync::Arc;

use peppygen::NodeRunner;
use peppygen::paired_topics::backbone;
use peppylib::runtime::CancellationToken;
use tokio::sync::watch;
use tracing::{error, warn};

use crate::hardware::MaxEffortNm;

#[derive(Clone, Copy)]
pub struct GripperCommand {
    pub opening: f64,
    /// Commanded effort cap; `None` when the wire carried no preference (0),
    /// so the configured ceiling applies.
    pub max_effort: Option<MaxEffortNm>,
}

pub async fn run(
    runner: Arc<NodeRunner>,
    latest: watch::Sender<Option<GripperCommand>>,
    token: CancellationToken,
) {
    let mut subscription = match backbone::gripper_setpoints::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => {
            error!(error = %e, "gripper_setpoints subscribe");
            return;
        }
    };
    loop {
        let received = tokio::select! {
            _ = token.cancelled() => return,
            received = subscription.next() => received,
        };
        let (_peer, msg) = match received {
            Ok(Some(pair)) => pair,
            Ok(None) => return,
            Err(e) => {
                error!(error = %e, "gripper_setpoints receive");
                continue;
            }
        };
        if !msg.opening.is_finite() {
            warn!(
                "gripper_setpoints: dropping a setpoint with a non-finite opening {}",
                msg.opening
            );
            continue;
        }
        let max_effort = match MaxEffortNm::parse(msg.max_effort) {
            Ok(max_effort) => max_effort,
            Err(reason) => {
                warn!("gripper_setpoints: dropping a setpoint, {reason}");
                continue;
            }
        };
        latest.send_replace(Some(GripperCommand {
            opening: msg.opening,
            max_effort,
        }));
    }
}
