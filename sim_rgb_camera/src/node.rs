// The relay loops and their assembly: frames forward to the contract
// surface, the stream descriptions feed the info services, the hardware
// controls refuse, and `setup` wires them together.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use peppygen::emitted_topics::camera::video_stream as camera_video;
use peppygen::exposed_services::camera::{
    set_brightness, set_contrast, set_exposure, set_gain, set_white_balance, video_stream_info,
};
use peppygen::paired_topics::engine::{stream_info, video_stream as engine_video};
use peppygen::{NodeRunner, Parameters, Result};
use peppylib::runtime::CancellationToken;
use tracing::{error, info, warn};

/// Set when a relay leg ends on its own. `setup` returns as soon as the legs
/// are spawned, so the flag, not its return value, is what tells the binary a
/// leg died: without it the process exits clean and the stack shows a node
/// that stopped relaying as a healthy finish.
static LEG_DIED: AtomicBool = AtomicBool::new(false);

/// Whether a relay leg ended on its own.
pub fn leg_died() -> bool {
    LEG_DIED.load(Ordering::SeqCst)
}

/// Pause after a receive error before retrying, so a persistently broken
/// subscription cannot hot-spin the relay or flood the log.
const RECEIVE_ERROR_BACKOFF: Duration = Duration::from_millis(100);

const UNSUPPORTED_MESSAGE: &str = "not adjustable in sim";

/// current_value in a refused control response: the no-value sentinel uvc and
/// zed answer when no usable hardware value exists.
const NO_CURRENT_VALUE: i32 = -1;

/// The engine's latest stream description; zeros until the first one arrives.
#[derive(Clone, Default)]
struct StreamDescription {
    width: u32,
    height: u32,
    frames_per_second: u8,
    encoding: String,
}

type SharedDescription = Arc<Mutex<StreamDescription>>;

/// Logs the first error of a run and suppresses the rest until something
/// succeeds, then waits out the backoff. A loop whose only outcome is an error
/// retries at the backoff interval, so without this it writes ten lines a
/// second for the life of the node.
#[derive(Default)]
struct RepeatedError {
    reported: bool,
}

impl RepeatedError {
    /// Reports one failure, then backs off before the caller retries.
    async fn report(&mut self, what: &str, error: impl std::fmt::Display) {
        if !self.reported {
            self.reported = true;
            error!("{what} failing, suppressing repeats: {error}");
        }
        tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
    }

    /// Ends the run, so the next failure is reported again.
    fn clear(&mut self) {
        self.reported = false;
    }
}

/// Forward the engine's frames to the contract video_stream.
async fn relay_frames(runner: Arc<NodeRunner>, token: CancellationToken) {
    let mut sub = match engine_video::subscribe(&runner).await {
        Ok(s) => s,
        Err(e) => return error!("engine video_stream subscribe: {e}"),
    };
    let publisher = match camera_video::declare_publisher(&runner).await {
        Ok(p) => p,
        Err(e) => return error!("declare video_stream publisher: {e}"),
    };
    let mut failing = false;
    let mut dropping = false;
    let mut first = true;
    let mut receive_errors = RepeatedError::default();
    loop {
        let received = tokio::select! {
            _ = token.cancelled() => return,
            received = sub.next() => received,
        };
        let msg = match received {
            Ok(Some((_, msg))) => msg,
            Ok(None) => return,
            Err(e) => {
                receive_errors
                    .report("engine video_stream receive", e)
                    .await;
                continue;
            }
        };
        receive_errors.clear();
        if !timestamp_is_valid(msg.header.timestamp) {
            if !dropping {
                dropping = true;
                warn!(
                    "dropping frames with invalid timestamps, suppressing repeats (first {:?})",
                    msg.header.timestamp
                );
            }
            continue;
        }
        dropping = false;
        let header = camera_video::MessageHeader {
            timestamp: msg.header.timestamp,
            frame_id: msg.header.frame_id,
        };
        let result = match camera_video::build_message(
            header,
            msg.encoding,
            msg.width,
            msg.height,
            msg.frame,
        ) {
            Ok(payload) => publisher.publish(payload).await.map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        };
        match result {
            Ok(()) => {
                failing = false;
                if first {
                    first = false;
                    info!("first frame relayed from the engine");
                }
            }
            Err(e) if !failing => {
                failing = true;
                warn!("video_stream publish failing, suppressing repeats: {e}");
            }
            Err(_) => {}
        }
    }
}

fn timestamp_is_valid(timestamp: std::time::SystemTime) -> bool {
    timestamp > std::time::SystemTime::UNIX_EPOCH
}

/// Track the engine's latest stream description for the info service.
async fn track_stream_info(
    runner: Arc<NodeRunner>,
    description: SharedDescription,
    token: CancellationToken,
) {
    let mut sub = match stream_info::subscribe(&runner).await {
        Ok(s) => s,
        Err(e) => return error!("engine stream_info subscribe: {e}"),
    };
    let mut receive_errors = RepeatedError::default();
    loop {
        let received = tokio::select! {
            _ = token.cancelled() => return,
            received = sub.next() => received,
        };
        match received {
            Ok(Some((_, msg))) => {
                *description
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = StreamDescription {
                    width: msg.width,
                    height: msg.height,
                    frames_per_second: msg.frames_per_second,
                    encoding: msg.encoding,
                };
            }
            Ok(None) => return,
            Err(e) => {
                receive_errors.report("engine stream_info receive", e).await;
            }
        }
    }
}

/// Answer one stream-info service from the shared stream description.
macro_rules! spawn_info_service {
    ($runner:expr, $description:expr, $service:ident, $respond:expr) => {{
        let runner = $runner.clone();
        let description = $description.clone();
        tokio::spawn(async move {
            let cancel = runner.cancellation_token().clone();
            let mut service_errors = RepeatedError::default();
            loop {
                let result = tokio::select! {
                    _ = cancel.cancelled() => break,
                    result = $service::handle_next_request(&runner, |_req| {
                        let d = description
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .clone();
                        Ok($respond(d))
                    }) => result,
                };
                match result {
                    Ok(()) => service_errors.clear(),
                    Err(e) => service_errors.report(stringify!($service), e).await,
                }
            }
        });
    }};
}

/// Every hardware control refuses: a rendered stream has no sensor to adjust.
macro_rules! spawn_refusing_control {
    ($runner:expr, $service:ident) => {{
        let runner = $runner.clone();
        tokio::spawn(async move {
            let cancel = runner.cancellation_token().clone();
            let mut service_errors = RepeatedError::default();
            loop {
                let result = tokio::select! {
                    _ = cancel.cancelled() => break,
                    result = $service::handle_next_request(&runner, |_req| {
                        Ok($service::Response::new(
                            false,
                            UNSUPPORTED_MESSAGE.to_string(),
                            NO_CURRENT_VALUE,
                        ))
                    }) => result,
                };
                match result {
                    Ok(()) => service_errors.clear(),
                    Err(e) => service_errors.report(stringify!($service), e).await,
                }
            }
        });
    }};
}

/// The node's entry point: the exact closure `NodeBuilder::run` used to get,
/// named so the test harness can boot the node in-process.
pub async fn setup(_params: Parameters, node_runner: Arc<NodeRunner>) -> Result<()> {
    let token = node_runner.cancellation_token().clone();
    let description: SharedDescription = Arc::new(Mutex::new(StreamDescription::default()));

    spawn_info_service!(
        node_runner,
        description,
        video_stream_info,
        |d: StreamDescription| {
            video_stream_info::Response::new(d.width, d.height, d.frames_per_second, d.encoding)
        }
    );
    spawn_refusing_control!(node_runner, set_exposure);
    spawn_refusing_control!(node_runner, set_white_balance);
    spawn_refusing_control!(node_runner, set_gain);
    spawn_refusing_control!(node_runner, set_brightness);
    spawn_refusing_control!(node_runner, set_contrast);

    let frames = tokio::spawn(relay_frames(node_runner.clone(), token.clone()));
    let info = tokio::spawn(track_stream_info(
        node_runner.clone(),
        description,
        token.clone(),
    ));
    // A dead relay leg would hold its direction silently while the node
    // reports healthy; flag it and cancel, so the process exits with a
    // failure the stack shows instead of a clean finish.
    tokio::spawn(async move {
        tokio::select! {
            _ = frames => {}
            _ = info => {}
        }
        if !token.is_cancelled() {
            LEG_DIED.store(true, Ordering::SeqCst);
            token.cancel();
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_guard_rejects_epoch_and_earlier() {
        use std::time::{Duration, SystemTime};

        assert!(timestamp_is_valid(
            SystemTime::UNIX_EPOCH + Duration::from_secs(1)
        ));
        assert!(!timestamp_is_valid(SystemTime::UNIX_EPOCH));
        assert!(!timestamp_is_valid(
            SystemTime::UNIX_EPOCH - Duration::from_secs(1)
        ));
    }
}
