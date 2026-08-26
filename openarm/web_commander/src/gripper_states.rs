// Live gripper opening and effort ceiling for the UI. Observes each side's
// backbone<->gripper pairing and reports the follower's measured opening and
// its reported effort ceiling to the owner. The observer slot fixes the side,
// so there is no id demux. Mirrors joint_states.rs for the arm: it runs
// continuously, so the panel shows live aperture whether or not a move is in
// flight.

use std::sync::Arc;

use peppygen::NodeRunner;
use peppygen::paired_topics::observed_left_gripper::gripper_states as observed_left_gripper_gripper_states;
use peppygen::paired_topics::observed_right_gripper::gripper_states as observed_right_gripper_gripper_states;
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::error;

use crate::joint_states::RECEIVE_ERROR_BACKOFF;
use crate::owner::Feedback;
use crate::state::Side;

pub async fn run(
    runner: Arc<NodeRunner>,
    feedback: mpsc::Sender<Feedback>,
    token: CancellationToken,
) {
    let mut left_subscription = match observed_left_gripper_gripper_states::subscribe(&runner).await
    {
        Ok(subscription) => subscription,
        Err(e) => {
            error!(error = %e, "observed_left_gripper gripper_states subscribe");
            return;
        }
    };
    let mut right_subscription =
        match observed_right_gripper_gripper_states::subscribe(&runner).await {
            Ok(subscription) => subscription,
            Err(e) => {
                error!(error = %e, "observed_right_gripper gripper_states subscribe");
                return;
            }
        };
    loop {
        let (slot, side, received) = tokio::select! {
            _ = token.cancelled() => return,
            received = left_subscription.next() => (
                "observed_left_gripper",
                Side::Left,
                received.map(|pair| {
                    pair.map(|(_source, msg)| (msg.opening, msg.max_effort))
                }),
            ),
            received = right_subscription.next() => (
                "observed_right_gripper",
                Side::Right,
                received.map(|pair| {
                    pair.map(|(_source, msg)| (msg.opening, msg.max_effort))
                }),
            ),
        };
        let (opening, max_effort) = match received {
            Ok(Some(pair)) => pair,
            Ok(None) => {
                error!(
                    slot,
                    "gripper_states observation closed; live gripper readouts stopped"
                );
                return;
            }
            Err(e) => {
                error!(error = %e, slot, "gripper_states receive");
                tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
                continue;
            }
        };
        if feedback
            .send(Feedback::GripperMeasured {
                side,
                opening,
                max_effort,
            })
            .await
            .is_err()
        {
            return;
        }
    }
}
