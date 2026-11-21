//! Bringup gate: the backbone waits for the launcher-bound robot_initializer
//! to report the robot ready before it runs its goals, its streams and the
//! robot stop, so it never commands wheels the robot has not brought up.

use std::sync::Arc;
use std::time::Duration;

use control_core::asking::ask_until_ready;
use peppygen::NodeRunner;
use peppygen::consumed_services::robot_init::is_ready as robot_init_is_ready;
use tracing::info;

const SERVICE_TIMEOUT: Duration = Duration::from_secs(5);

/// Wait until the launcher-bound robot_initializer reports ready
/// ([`ask_until_ready`]), so the backbone can expose its own actions; false
/// once the node is cancelled first.
pub async fn wait_until_ready(runner: &Arc<NodeRunner>) -> bool {
    let asker = runner.clone();
    let ready = ask_until_ready(
        "robot_init is_ready",
        runner.cancellation_token().cancelled(),
        async move || {
            robot_init_is_ready::poll(
                &asker,
                robot_init_is_ready::bound_producer(&asker),
                SERVICE_TIMEOUT,
            )
            .await
            .map(|response| response.data.ready)
        },
    )
    .await;
    if ready {
        info!("robot_initializer reported ready");
    }
    ready
}
