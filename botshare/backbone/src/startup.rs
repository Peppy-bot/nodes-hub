//! Bringup gate: the backbone waits for the launcher-bound robot_initializer
//! to report the robot ready before it runs its goals, its streams and the
//! robot stop, so it never commands wheels the robot has not brought up.

use std::time::Duration;

use control_core::time::sleep_unless_cancelled;
use peppygen::NodeRunner;
use peppygen::consumed_services::robot_init::is_ready as robot_init_is_ready;
use peppylib::runtime::CancellationToken;
use tracing::{info, warn};

const IS_READY_POLL_INTERVAL: Duration = Duration::from_millis(500);
const SERVICE_TIMEOUT: Duration = Duration::from_secs(5);

/// Block until the launcher-bound robot_initializer reports ready, then return
/// so the backbone can expose its own actions.
pub async fn wait_until_ready(runner: &NodeRunner, token: &CancellationToken) {
    loop {
        match robot_init_is_ready::poll(
            runner,
            robot_init_is_ready::bound_producer(runner),
            SERVICE_TIMEOUT,
        )
        .await
        {
            Ok(resp) if resp.data.ready => {
                info!("robot_initializer reported ready");
                return;
            }
            Ok(_) => {}
            Err(e) => warn!(error = %e, "is_ready poll failed; retrying"),
        }
        if !sleep_unless_cancelled(IS_READY_POLL_INTERVAL, token.cancelled()).await {
            return;
        }
    }
}
