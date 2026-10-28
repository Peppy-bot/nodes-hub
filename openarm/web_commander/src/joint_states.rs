//! Live arm joint state for the UI. Observes each side's backbone<->arm pairing
//! and reports the follower's measured positions to the owner. The observer slot
//! fixes the side (an observed pairing delivers only its one follower), so there
//! is no id demux. It runs continuously and works in every deployment, including
//! ones where another node commands; the owner decides how to fold each
//! measurement into the target.

use std::sync::Arc;

use openarm_description::Side;
use peppygen::NodeRunner;
use peppygen::paired_topics::observed_left_arm::joint_states as observed_left_arm_joint_states;
use peppygen::paired_topics::observed_right_arm::joint_states as observed_right_arm_joint_states;
use peppylib::runtime::CancellationToken;
use srs_model::JointVec;
use tokio::sync::mpsc;
use tracing::{error, warn};

use crate::consumer::{self, ObservedSlot};
use crate::owner::Feedback;

const LEFT: ObservedSlot = ObservedSlot {
    name: "observed_left_arm",
    side: Side::Left,
};

const RIGHT: ObservedSlot = ObservedSlot {
    name: "observed_right_arm",
    side: Side::Right,
};

const CLOSED: &str = "joint_states observation closed; live joint readouts stopped";

/// Report each side's measured joints to the owner until `token` cancels, an
/// observation closes, or the owner is gone.
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
    consumer::forward_observed(
        "joint_states",
        token,
        feedback,
        async move || {
            tokio::select! {
                received = left_subscription.next() => consumer::observed(
                    LEFT,
                    CLOSED,
                    received.map(|pair| pair.map(|(_source, msg)| msg.positions)),
                ),
                received = right_subscription.next() => consumer::observed(
                    RIGHT,
                    CLOSED,
                    received.map(|pair| pair.map(|(_source, msg)| msg.positions)),
                ),
            }
        },
        arm_measured,
    )
    .await;
}

/// The owner feedback one slot's measured positions make. The generic
/// contract carries a Vec; this panel drives fixed 7-joint arms, so anything
/// else is malformed and dropped.
fn arm_measured(slot: ObservedSlot, positions: Vec<f64>) -> Option<Feedback> {
    let Ok(joints) = JointVec::try_from(positions) else {
        warn!(
            slot = slot.name,
            "joint_states: dropping message with a non-arm joint count"
        );
        return None;
    };
    Some(Feedback::ArmMeasured {
        side: slot.side,
        joints,
    })
}

#[cfg(test)]
mod tests {
    use openarm_description::ARM_DOF;

    use super::*;

    #[test]
    fn a_non_arm_joint_count_is_dropped() {
        assert!(arm_measured(LEFT, vec![0.1; ARM_DOF - 1]).is_none());
        let Some(Feedback::ArmMeasured { side, joints }) = arm_measured(RIGHT, vec![0.1; ARM_DOF])
        else {
            panic!("seven joints reach the owner");
        };
        assert_eq!((side, joints), (Side::Right, [0.1; ARM_DOF]));
    }
}
