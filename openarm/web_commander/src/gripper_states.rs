//! Live gripper opening and effort ceiling for the UI. Observes each side's
//! backbone<->gripper pairing and reports the follower's measured opening and
//! its reported effort ceiling to the owner. The observer slot fixes the side,
//! so there is no id demux. Mirrors joint_states.rs for the arm: it runs
//! continuously, so the panel shows live aperture whether or not a move is in
//! flight.

use std::sync::Arc;

use openarm_description::Side;
use peppygen::NodeRunner;
use peppygen::paired_topics::observed_left_gripper::gripper_states as observed_left_gripper_gripper_states;
use peppygen::paired_topics::observed_right_gripper::gripper_states as observed_right_gripper_gripper_states;
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::error;

use crate::consumer::{self, ObservedSlot};
use crate::owner::Feedback;

const LEFT: ObservedSlot = ObservedSlot {
    name: "observed_left_gripper",
    side: Side::Left,
};

const RIGHT: ObservedSlot = ObservedSlot {
    name: "observed_right_gripper",
    side: Side::Right,
};

const CLOSED: &str = "gripper_states observation closed; live gripper readouts stopped";

/// Report each side's measured opening and effort ceiling to the owner until
/// `token` cancels, an observation closes, or the owner is gone.
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
    consumer::forward_observed(
        "gripper_states",
        token,
        feedback,
        async move || {
            tokio::select! {
                received = left_subscription.next() => consumer::observed(
                    LEFT,
                    CLOSED,
                    received.map(|pair| pair.map(|(_source, msg)| (msg.opening, msg.max_effort))),
                ),
                received = right_subscription.next() => consumer::observed(
                    RIGHT,
                    CLOSED,
                    received.map(|pair| pair.map(|(_source, msg)| (msg.opening, msg.max_effort))),
                ),
            }
        },
        |slot, (opening, max_effort)| {
            Some(Feedback::GripperMeasured {
                side: slot.side,
                opening,
                max_effort,
            })
        },
    )
    .await;
}
