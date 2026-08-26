// Live arm joint state for the UI. Observes each side's backbone<->arm pairing
// and reports the follower's measured positions to the owner. The observer slot
// fixes the side (an observed pairing delivers only its one follower), so there
// is no id demux. It runs continuously and works in every deployment, including
// ones where another node commands; the owner decides how to fold each
// measurement into the target.

use std::sync::Arc;

use peppygen::NodeRunner;
use peppygen::paired_topics::observed_left_arm::joint_states as observed_left_arm_joint_states;
use peppygen::paired_topics::observed_right_arm::joint_states as observed_right_arm_joint_states;
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::{error, warn};

use crate::owner::Feedback;
use crate::state::{ARM_DOF, Side};

/// Pause after a receive error before retrying, so a persistently broken subscription
/// cannot spin a listener at full CPU or flood the log at the stream rate (shared with
/// the gripper-states listener).
pub(crate) const RECEIVE_ERROR_BACKOFF: std::time::Duration = std::time::Duration::from_millis(100);

pub async fn run(
    runner: Arc<NodeRunner>,
    feedback: mpsc::Sender<Feedback>,
    token: CancellationToken,
) {
    let mut left_subscription = match observed_left_arm_joint_states::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => {
            error!(error = %e, "observed_left_arm joint_states subscribe");
            return;
        }
    };
    let mut right_subscription = match observed_right_arm_joint_states::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => {
            error!(error = %e, "observed_right_arm joint_states subscribe");
            return;
        }
    };
    loop {
        let (slot, side, received) = tokio::select! {
            _ = token.cancelled() => return,
            received = left_subscription.next() => (
                "observed_left_arm",
                Side::Left,
                received.map(|pair| pair.map(|(_source, msg)| msg.positions)),
            ),
            received = right_subscription.next() => (
                "observed_right_arm",
                Side::Right,
                received.map(|pair| pair.map(|(_source, msg)| msg.positions)),
            ),
        };
        let positions = match received {
            Ok(Some(positions)) => positions,
            Ok(None) => {
                error!(
                    slot,
                    "joint_states observation closed; live joint readouts stopped"
                );
                return;
            }
            Err(e) => {
                error!(error = %e, slot, "joint_states receive");
                tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
                continue;
            }
        };
        // The generic contract carries a Vec; this panel drives fixed 7-joint
        // arms, so anything else is malformed and dropped.
        let Ok(joints) = <[f64; ARM_DOF]>::try_from(positions) else {
            warn!(
                slot,
                "joint_states: dropping message with a non-arm joint count"
            );
            continue;
        };
        if feedback
            .send(Feedback::ArmMeasured { side, joints })
            .await
            .is_err()
        {
            return; // the owner is gone; nothing left to report to
        }
    }
}
