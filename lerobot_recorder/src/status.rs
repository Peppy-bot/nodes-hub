//! recorder_status publisher: the manager updates a watch; this task
//! republishes it at status_rate_hz with live disk headroom.

use std::sync::Arc;

use peppygen::NodeRunner;
use peppygen::emitted_topics::recorder::v1::recorder_status;
use peppylib::runtime::CancellationToken;
use tokio::sync::watch;
use tracing::warn;

use crate::config::Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusState {
    Idle,
    Recording,
    Error,
}

impl StatusState {
    fn wire(self) -> &'static str {
        match self {
            StatusState::Idle => "idle",
            StatusState::Recording => "recording",
            StatusState::Error => "error",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Status {
    pub state: StatusState,
    pub episode_index: i64,
    pub frames: u64,
    pub task: String,
    pub message: String,
}

impl Status {
    pub fn initial() -> Self {
        Status {
            state: StatusState::Idle,
            episode_index: -1,
            frames: 0,
            task: String::new(),
            message: String::new(),
        }
    }
}

pub async fn run(
    runner: Arc<NodeRunner>,
    config: Config,
    status: watch::Receiver<Status>,
    token: CancellationToken,
) {
    let publisher = match recorder_status::declare_publisher(&runner).await {
        Ok(p) => p,
        Err(e) => {
            warn!("declare recorder_status publisher: {e}");
            return;
        }
    };
    let mut ticker = tokio::time::interval(config.status_period);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            _ = token.cancelled() => return,
            _ = ticker.tick() => {}
        }
        let current = status.borrow().clone();
        let disk_free = fs2::available_space(&config.output_root).unwrap_or(0);
        let payload = recorder_status::build_message(
            current.state.wire().to_string(),
            current.episode_index,
            current.frames,
            current.task,
            current.message,
            disk_free,
        );
        match payload {
            Ok(payload) => {
                if let Err(e) = publisher.publish(payload).await {
                    warn!("publish recorder_status: {e}");
                }
            }
            Err(e) => warn!("build recorder_status: {e}"),
        }
    }
}
