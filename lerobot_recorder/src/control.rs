//! Exposed episode services. The generated handlers are synchronous and
//! handle one request per call; each bridges into the manager's command
//! channel and waits briefly for its reply.

use std::sync::Arc;
use std::time::Duration;

use peppygen::NodeRunner;
use peppygen::exposed_services::recorder::v1::{start_episode, stop_episode};
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::warn;

use crate::episode::Command;

/// Bounded by one sampler period in practice; generous to absorb an episode
/// end (which waits on the video encoder).
const REPLY_TIMEOUT: Duration = Duration::from_secs(90);

fn busy<T>(make: impl FnOnce(String) -> T) -> T {
    make("recorder busy".to_string())
}

pub async fn start_service(
    runner: Arc<NodeRunner>,
    commands: mpsc::Sender<Command>,
    token: CancellationToken,
) {
    loop {
        let commands = commands.clone();
        let served = tokio::select! {
            _ = token.cancelled() => return,
            served = start_episode::handle_next_request(&runner, move |request| {
                let task = match request.data.task.trim() {
                    "" => None,
                    task => Some(task.to_string()),
                };
                let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
                let sent = commands
                    .try_send(Command::Start { task, reply: reply_tx })
                    .is_ok();
                let reply = if sent {
                    tokio::task::block_in_place(|| reply_rx.recv_timeout(REPLY_TIMEOUT).ok())
                } else {
                    None
                };
                let reply = reply.unwrap_or_else(|| busy(|message| crate::episode::StartReply {
                    accepted: false,
                    episode_index: -1,
                    message,
                }));
                Ok(start_episode::Response::new(
                    reply.accepted,
                    reply.episode_index,
                    reply.message,
                ))
            }) => served,
        };
        if let Err(e) = served {
            warn!("start_episode service: {e}");
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

pub async fn stop_service(
    runner: Arc<NodeRunner>,
    commands: mpsc::Sender<Command>,
    token: CancellationToken,
) {
    loop {
        let commands = commands.clone();
        let served = tokio::select! {
            _ = token.cancelled() => return,
            served = stop_episode::handle_next_request(&runner, move |request| {
                let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
                let sent = commands
                    .try_send(Command::Stop { save: request.data.save, reply: reply_tx })
                    .is_ok();
                let reply = if sent {
                    tokio::task::block_in_place(|| reply_rx.recv_timeout(REPLY_TIMEOUT).ok())
                } else {
                    None
                };
                let reply = reply.unwrap_or_else(|| busy(|message| crate::episode::StopReply {
                    accepted: false,
                    episode_index: -1,
                    frames: 0,
                    message,
                }));
                Ok(stop_episode::Response::new(
                    reply.accepted,
                    reply.episode_index,
                    reply.frames,
                    reply.message,
                ))
            }) => served,
        };
        if let Err(e) = served {
            warn!("stop_episode service: {e}");
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}
