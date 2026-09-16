// The relay loops and their assembly: frames forward to the contract
// surface, the stream descriptions feed the info services, the colour
// controls and the profile forward to the simulation's camera response model
// under the camera slot this relay views, and `setup` wires them together.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use peppygen::consumed_services::control::{
    describe_camera, reset_camera as control_reset_camera, set_camera_brightness,
    set_camera_contrast, set_camera_exposure, set_camera_gain, set_camera_white_balance,
};
use peppygen::emitted_topics::camera::{
    depth_stream as camera_depth, video_stream as camera_video,
};
use peppygen::exposed_services::camera::{
    depth_stream_info, set_color_brightness, set_color_contrast, set_color_exposure,
    set_color_gain, set_color_white_balance, video_stream_info,
};
use peppygen::exposed_services::profile::{get_camera_profile, reset_camera};
use peppygen::paired_topics::simulation::{
    depth_stream as simulation_depth, stream_info, video_stream as simulation_video,
};
use peppygen::{NodeRunner, Parameters, ProducerRef, Result};
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

/// Bounds every call forwarded to the simulation's camera response model. A
/// control is one round trip through the simulation's device model, so an
/// answer later than this means the model is gone, and the caller gets a
/// refusal carrying the timeout rather than a service that hangs.
const CONTROL_TIMEOUT: Duration = Duration::from_secs(5);

/// The refusal every forwarded call (a control, the profile, the reset)
/// answers while the control slot is vacant: the launch linked this relay
/// to a simulation without a camera response model, so there is nothing
/// behind the controls to adjust, describe or reset.
const NO_RESPONSE_MODEL_MESSAGE: &str =
    "no camera response model is linked: nothing to adjust, describe or reset";

/// The refusal every control and profile call answers while the simulation
/// pairing is not established: the camera slot a forwarded request must name
/// is the pairing peer's link id, and there is no peer yet.
const NOT_PAIRED_MESSAGE: &str = "not paired to its simulation camera yet";

/// current_value in a refused control response: the no-value sentinel uvc and
/// zed answer when no usable hardware value exists.
const NO_CURRENT_VALUE: i32 = -1;

/// The simulation's latest stream description; zeros until the first one arrives.
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

/// The outcome of routing one call to the simulation's camera response
/// model: the model's answer, relayed verbatim, or the message the caller
/// gets for why the call never reached a model.
type Forwarded<T> = std::result::Result<T, String>;

/// The camera slot this relay views, as the simulation names it: the link id
/// of the peer slot on the simulation pairing. The simulation renders each
/// camera on a slot of its own and its response model keys every control on
/// that slot's name, so the pairing peer is the camera's whole identity and
/// the relay carries no id of its own. The colour and depth streams share
/// the one pairing, so either topic's pin names the same peer.
fn camera_slot(runner: &NodeRunner) -> Forwarded<String> {
    match simulation_video::paired(runner) {
        Ok(Some(peer)) => Ok(peer.peer_link_id),
        Ok(None) => Err(NOT_PAIRED_MESSAGE.to_string()),
        Err(e) => Err(format!("simulation pairing state unavailable: {e}")),
    }
}

/// Runs a forwarded poll to completion from inside a synchronous service
/// handler. The generated handlers take a plain closure, while a poll is
/// awaitable; `block_in_place` hands this worker's queue to another runtime
/// thread for the duration, so the relay legs and the other services keep
/// running while this one waits on the simulation. Legal only on a
/// multi-thread runtime, which `NodeBuilder::run` and the harness both
/// provide.
fn wait_for<T>(future: impl Future<Output = T>) -> T {
    tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(future))
}

/// The route from a contract call to the simulation's camera response model,
/// shared by every forwarding service task.
struct ControlRoute {
    runner: Arc<NodeRunner>,
    /// Whether the last forwarded call failed in transport. The first failure
    /// is logged and the rest are suppressed until a call goes through: a
    /// caller retrying against a simulation that is gone would otherwise
    /// write a line per attempt.
    transport_failing: AtomicBool,
}

impl ControlRoute {
    fn new(runner: Arc<NodeRunner>) -> Self {
        Self {
            runner,
            transport_failing: AtomicBool::new(false),
        }
    }

    /// Forwards one call. `bound` is the control slot's producer, `call`
    /// polls the response model for the camera slot on it, and the model's
    /// answer relays verbatim. Every way the call cannot reach the model is
    /// an `Err` carrying the caller's message: the vacant slot first, since
    /// the binding is fixed for the life of the node and no pairing changes
    /// it; then the pairing, which is the transient condition; then the
    /// transport.
    fn forward<T, Fut>(
        &self,
        bound: Option<&ProducerRef>,
        call: impl FnOnce(String, ProducerRef) -> Fut,
    ) -> Forwarded<T>
    where
        Fut: Future<Output = Result<T>>,
    {
        let Some(target) = bound else {
            return Err(NO_RESPONSE_MODEL_MESSAGE.to_string());
        };
        let camera = camera_slot(&self.runner)?;
        match wait_for(call(camera, target.clone())) {
            Ok(answer) => {
                self.transport_failing.store(false, Ordering::SeqCst);
                Ok(answer)
            }
            Err(e) => {
                if !self.transport_failing.swap(true, Ordering::SeqCst) {
                    warn!(
                        "forwarding to the camera response model failing, suppressing repeats: {e}"
                    );
                }
                Err(format!(
                    "forwarding to the simulation's camera response model failed: {e}"
                ))
            }
        }
    }
}

/// Forward one simulation stream to its contract twin; color and depth legs are
/// the same conversation over different topics.
macro_rules! relay_stream {
    ($fn_name:ident, $consume:ident, $emit:ident, $label:literal) => {
        async fn $fn_name(runner: Arc<NodeRunner>, token: CancellationToken) {
            let mut sub = match $consume::subscribe(&runner).await {
                Ok(s) => s,
                Err(e) => return error!(concat!("simulation ", $label, " subscribe: {}"), e),
            };
            let publisher = match $emit::declare_publisher(&runner).await {
                Ok(p) => p,
                Err(e) => return error!(concat!("declare ", $label, " publisher: {}"), e),
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
                            .report(concat!("simulation ", $label, " receive"), e)
                            .await;
                        continue;
                    }
                };
                receive_errors.clear();
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
                            info!(concat!(
                                "first ",
                                $label,
                                " frame relayed from the simulation"
                            ));
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

relay_stream!(relay_video, simulation_video, camera_video, "video_stream");
relay_stream!(relay_depth, simulation_depth, camera_depth, "depth_stream");

/// Track the simulation's latest stream description for the info services.
async fn track_stream_info(
    runner: Arc<NodeRunner>,
    description: SharedDescription,
    token: CancellationToken,
) {
    let mut sub = match stream_info::subscribe(&runner).await {
        Ok(s) => s,
        Err(e) => return error!("simulation stream_info subscribe: {e}"),
    };
    let mut rejecting = false;
    let mut receive_errors = RepeatedError::default();
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
                receive_errors
                    .report("simulation stream_info receive", e)
                    .await;
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

/// One service whose every request routes through the control route:
/// `$answer` maps the request to the response, forwarding inside.
macro_rules! spawn_forwarding_service {
    ($route:expr, $service:ident, $answer:expr) => {{
        let route: Arc<ControlRoute> = $route.clone();
        tokio::spawn(async move {
            let runner = route.runner.clone();
            let cancel = runner.cancellation_token().clone();
            let mut service_errors = RepeatedError::default();
            loop {
                let result = tokio::select! {
                    _ = cancel.cancelled() => break,
                    result = $service::handle_next_request(&runner, |request| {
                        Ok(($answer)(&route, request))
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

/// One colour control forwarded to its response-model twin: the request's
/// fields go through under the camera slot's name and the model's answer
/// relays verbatim, success or not; a call that never reached the model
/// answers false and the reason. `$reading` names the response's reading
/// field beside the fallback a refusal carries in it: a temperature for
/// white balance, a value for the other controls, the profile JSON for
/// get_camera_profile; reset_camera carries none.
macro_rules! spawn_forwarding_control {
    ($route:expr, $service:ident => $control:ident, [$($field:ident),*] $(, $reading:ident = $fallback:expr)?) => {
        spawn_forwarding_service!(
            $route,
            $service,
            |route: &ControlRoute, _request: $service::Request| {
                $(let $field = _request.data.$field;)*
                let answer = route.forward(
                    $control::bound_producer(&route.runner),
                    |camera, target| async move {
                        let request = $control::Request::new(camera $(, $field)*);
                        $control::poll(&route.runner, &target, CONTROL_TIMEOUT, request)
                            .await
                            .map(|response| response.data)
                    },
                );
                match answer {
                    Ok(answer) => {
                        $service::Response::new(answer.success, answer.message $(, answer.$reading)?)
                    }
                    Err(message) => $service::Response::new(false, message $(, $fallback)?),
                }
            }
        )
    };
}

/// The node's entry point: the exact closure `NodeBuilder::run` used to get,
/// named so the test harness can boot the node in-process.
pub async fn setup(_params: Parameters, node_runner: Arc<NodeRunner>) -> Result<()> {
    let token = node_runner.cancellation_token().clone();
    let description: SharedDescription = Arc::new(Mutex::new(StreamDescription::default()));
    let route = Arc::new(ControlRoute::new(node_runner.clone()));
    if describe_camera::bound_producer(&node_runner).is_some() {
        info!("camera response model linked: colour controls forward to the simulation");
    } else {
        info!("no camera response model linked: every colour control refuses");
    }

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
    spawn_forwarding_control!(
        route,
        set_color_exposure => set_camera_exposure,
        [mode, value],
        current_value = NO_CURRENT_VALUE
    );
    spawn_forwarding_control!(
        route,
        set_color_white_balance => set_camera_white_balance,
        [mode, temperature],
        current_temperature = NO_CURRENT_VALUE
    );
    spawn_forwarding_control!(route, set_color_gain => set_camera_gain, [value], current_value = NO_CURRENT_VALUE);
    spawn_forwarding_control!(
        route,
        set_color_brightness => set_camera_brightness,
        [value],
        current_value = NO_CURRENT_VALUE
    );
    spawn_forwarding_control!(
        route,
        set_color_contrast => set_camera_contrast,
        [value],
        current_value = NO_CURRENT_VALUE
    );
    // The profile is the simulation's description of the device this camera
    // stands for, and the reset puts every control back to its defaults:
    // both live with the response model, so both forward to it.
    spawn_forwarding_control!(
        route,
        get_camera_profile => describe_camera,
        [],
        profile_json = String::new()
    );
    spawn_forwarding_control!(route, reset_camera => control_reset_camera, []);

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
