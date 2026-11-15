//! The model of the robot the panel drives, as its initializer names it on
//! robot_identity's `get_identity`. The panel asks after its setup has returned, since
//! its initializer may start after it, and asks until the initializer answers.

use std::sync::Arc;
use std::time::Duration;

use control_core::asking;
use peppygen::NodeRunner;
use peppygen::consumed_services::robot_identity::get_identity;

/// How long the initializer may take to answer one ask.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(5);

/// The model id the robot's initializer names, or `None` once the node is cancelled
/// before the initializer answers.
pub async fn model(runner: &Arc<NodeRunner>) -> Option<String> {
    let asker = runner.clone();
    asking::ask(
        "robot_identity",
        runner.cancellation_token().cancelled(),
        async move || {
            get_identity::poll(&asker, get_identity::bound_producer(&asker), ANSWER_TIMEOUT)
                .await
                .map(|answer| answer.data.model)
        },
    )
    .await
}
