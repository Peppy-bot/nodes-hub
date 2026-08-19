// The relay loops and their assembly: frames forward to the contract
// surface, the stream descriptions feed the info services, the hardware
// controls refuse, and `setup` wires them together.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use peppygen::emitted_topics::camera::{
    depth_stream as camera_depth, video_stream as camera_video,
};
use peppygen::exposed_services::camera::{
    depth_stream_info, set_color_brightness, set_color_contrast, set_color_exposure,
    set_color_gain, set_color_white_balance, video_stream_info,
};
use peppygen::paired_topics::engine::{
    depth_stream as engine_depth, stream_info, video_stream as engine_video,
};
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
    depth_width: u32,
    depth_height: u32,
    depth_encoding: String,
    depth_unit: f32,
}

type SharedDescription = Arc<Mutex<StreamDescription>>;

fn timestamp_is_valid(timestamp: std::time::SystemTime) -> bool {
    timestamp > std::time::SystemTime::UNIX_EPOCH
}

fn depth_unit_is_valid(depth_unit: f32) -> bool {
    depth_unit.is_finite() && depth_unit > 0.0
}

/// Forward one engine stream to its contract twin; color and depth legs are
/// the same conversation over different topics.
macro_rules! relay_stream {
    ($fn_name:ident, $consume:ident, $emit:ident, $label:literal) => {
        async fn $fn_name(runner: Arc<NodeRunner>, token: CancellationToken) {
            let mut sub = match $consume::subscribe(&runner).await {
                Ok(s) => s,
                Err(e) => return error!(concat!("engine ", $label, " subscribe: {}"), e),
            };
            let publisher = match $emit::declare_publisher(&runner).await {
                Ok(p) => p,
                Err(e) => return error!(concat!("declare ", $label, " publisher: {}"), e),
            };
            let mut failing = false;
            let mut dropping = false;
            let mut first = true;
            loop {
                let received = tokio::select! {
                    _ = token.cancelled() => return,
                    received = sub.next() => received,
                };
                let msg = match received {
                    Ok(Some((_, msg))) => msg,
                    Ok(None) => return,
                    Err(e) => {
                        error!(concat!("engine ", $label, " receive: {}"), e);
                        tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
                        continue;
                    }
                };
                if !timestamp_is_valid(msg.header.timestamp) {
                    if !dropping {
                        dropping = true;
                        warn!(
                            concat!(
                                "dropping ",
                                $label,
                                " frames with invalid timestamps, suppressing repeats (first {:?})"
                            ),
                            msg.header.timestamp
                        );
                    }
                    continue;
                }
                dropping = false;
                let header = $emit::MessageHeader {
                    timestamp: msg.header.timestamp,
                    frame_id: msg.header.frame_id,
                    align_mode: msg.header.align_mode,
                };
                let result = match $emit::build_message(
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
                            info!(concat!("first ", $label, " frame relayed from the engine"));
                        }
                    }
                    Err(e) if !failing => {
                        failing = true;
                        warn!(
                            concat!($label, " publish failing, suppressing repeats: {}"),
                            e
                        );
                    }
                    Err(_) => {}
                }
            }
        }
    };
}

relay_stream!(relay_video, engine_video, camera_video, "video_stream");
relay_stream!(relay_depth, engine_depth, camera_depth, "depth_stream");

/// Track the engine's latest stream description for the info services.
async fn track_stream_info(
    runner: Arc<NodeRunner>,
    description: SharedDescription,
    token: CancellationToken,
) {
    let mut sub = match stream_info::subscribe(&runner).await {
        Ok(s) => s,
        Err(e) => return error!("engine stream_info subscribe: {e}"),
    };
    let mut rejecting = false;
    loop {
        let received = tokio::select! {
            _ = token.cancelled() => return,
            received = sub.next() => received,
        };
        match received {
            Ok(Some((_, msg))) => {
                if !depth_unit_is_valid(msg.depth_unit) {
                    if !rejecting {
                        rejecting = true;
                        warn!(
                            "ignoring stream_info with invalid depth_unit {}, suppressing repeats",
                            msg.depth_unit
                        );
                    }
                    continue;
                }
                rejecting = false;
                *description
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = StreamDescription {
                    width: msg.width,
                    height: msg.height,
                    frames_per_second: msg.frames_per_second,
                    encoding: msg.encoding,
                    depth_width: msg.depth_width,
                    depth_height: msg.depth_height,
                    depth_encoding: msg.depth_encoding,
                    depth_unit: msg.depth_unit,
                };
            }
            Ok(None) => return,
            Err(e) => {
                error!("engine stream_info receive: {e}");
                tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
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
                if let Err(e) = result {
                    error!(concat!(stringify!($service), ": {}"), e);
                    tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
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
                if let Err(e) = result {
                    error!(concat!(stringify!($service), ": {}"), e);
                    tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
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
    spawn_info_service!(
        node_runner,
        description,
        depth_stream_info,
        |d: StreamDescription| {
            depth_stream_info::Response::new(
                d.depth_width,
                d.depth_height,
                d.frames_per_second,
                d.depth_encoding,
                d.depth_unit,
            )
        }
    );
    spawn_refusing_control!(node_runner, set_color_exposure);
    spawn_refusing_control!(node_runner, set_color_white_balance);
    spawn_refusing_control!(node_runner, set_color_gain);
    spawn_refusing_control!(node_runner, set_color_brightness);
    spawn_refusing_control!(node_runner, set_color_contrast);

    let video = tokio::spawn(relay_video(node_runner.clone(), token.clone()));
    let depth = tokio::spawn(relay_depth(node_runner.clone(), token.clone()));
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
            _ = video => {}
            _ = depth => {}
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

    #[test]
    fn depth_unit_guard_rejects_nonpositive_and_nonfinite() {
        assert!(depth_unit_is_valid(0.001));
        assert!(!depth_unit_is_valid(0.0));
        assert!(!depth_unit_is_valid(-0.0));
        assert!(!depth_unit_is_valid(-0.001));
        assert!(!depth_unit_is_valid(f32::NAN));
        assert!(!depth_unit_is_valid(f32::INFINITY));
        assert!(!depth_unit_is_valid(f32::NEG_INFINITY));
    }
}
