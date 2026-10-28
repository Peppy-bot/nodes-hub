//! The relay loops and their assembly: frames forward to the contract
//! surface, the stream descriptions feed the info services, the camera
//! geometry feeds the geometry services, the colour controls and the profile
//! forward to the simulation's camera response model under the camera's name,
//! which is this relay's name in its copy, and `setup` wires them together.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use camera_core::geometry::{
    depth_range_is_valid, depth_unit_is_valid, pinhole_is_valid, pose_is_valid,
};
use camera_core::relay::{
    CONTROL_TIMEOUT, NO_GEOMETRY_MESSAGE, NO_RESPONSE_MODEL_MESSAGE, TransportFailure, Unanswered,
    camera_name, timestamp_is_valid,
};
use control_core::latched_log::LatchedLog;
use control_core::report_once::ReportOnce;
use control_core::sync::{lock_ignoring_poison, wait_for};
use control_core::time::RECEIVE_ERROR_BACKOFF;
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
use peppygen::exposed_services::geometry::{
    get_color_intrinsics, get_depth_intrinsics, get_depth_to_color_extrinsics,
};
use peppygen::exposed_services::profile::{get_camera_profile, reset_camera};
use peppygen::paired_topics::simulation::{
    depth_stream as simulation_depth, geometry as simulation_geometry, stream_info,
    video_stream as simulation_video,
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

/// The simulation's latest camera geometry, kept as it came; none until the
/// first one arrives.
type SharedGeometry = Arc<Mutex<Option<simulation_geometry::Message>>>;

/// The part of a geometry that makes it unusable, if any. A consumer turns a
/// pixel and a depth into a position with these numbers, so one bad part
/// spoils every position: the whole geometry is ignored, as a stream
/// description with an unusable depth_unit is.
fn unusable_part(geometry: &simulation_geometry::Message) -> Option<&'static str> {
    let g = geometry;
    if !pinhole_is_valid(g.width, g.height, g.fx, g.fy, g.cx, g.cy) {
        Some("colour pinhole model")
    } else if !pinhole_is_valid(
        g.depth_width,
        g.depth_height,
        g.depth_fx,
        g.depth_fy,
        g.depth_cx,
        g.depth_cy,
    ) {
        Some("depth pinhole model")
    } else if !depth_range_is_valid(g.min_depth_m, g.max_depth_m) {
        Some("depth range")
    } else if !pose_is_valid(&g.depth_to_color_position, &g.depth_to_color_orientation) {
        Some("depth-to-colour pose")
    } else {
        None
    }
}

/// get_color_intrinsics: the colour stream's pinhole model as the simulation
/// published it.
fn color_intrinsics(g: simulation_geometry::Message) -> get_color_intrinsics::Response {
    get_color_intrinsics::Response::new(
        g.width,
        g.height,
        g.fx,
        g.fy,
        g.cx,
        g.cy,
        g.distortion_model,
        g.distortion,
    )
}

/// get_depth_intrinsics: the depth stream's own pinhole model, what its
/// samples measure and the alignment the answer holds under.
fn depth_intrinsics(g: simulation_geometry::Message) -> get_depth_intrinsics::Response {
    get_depth_intrinsics::Response::new(
        g.depth_width,
        g.depth_height,
        g.depth_fx,
        g.depth_fy,
        g.depth_cx,
        g.depth_cy,
        g.depth_distortion_model,
        g.depth_distortion,
        g.depth_model,
        g.min_depth_m,
        g.max_depth_m,
        g.align_mode,
    )
}

/// get_depth_to_color_extrinsics: where the depth optical frame sits in the
/// colour one, under the same alignment get_depth_intrinsics names. Both
/// answers come from the one geometry message, so they always agree.
fn depth_to_color_extrinsics(
    g: simulation_geometry::Message,
) -> get_depth_to_color_extrinsics::Response {
    get_depth_to_color_extrinsics::Response::new(
        g.align_mode,
        g.depth_to_color_position,
        g.depth_to_color_orientation,
    )
}

/// Why a poll of the model failed with `error`. `refusal` is the error reply
/// of the model that `error` holds, read by the member the poll went to, or
/// `None` when `error` is no error reply.
fn unanswered_of(error: &peppygen::Error, refusal: Option<(&'static str, String)>) -> Unanswered {
    if let Some((kind, message)) = refusal {
        return Unanswered::Refused { kind, message };
    }
    let failure = match error {
        peppygen::Error::ServiceTimeout { .. } => TransportFailure::TimedOut,
        peppygen::Error::ServiceUnreachable { .. } => TransportFailure::Unreachable,
        _ => TransportFailure::Broken,
    };
    Unanswered::of_transport(failure, error)
}

/// The error reply of the exposed member `$service` for a forwarded call
/// that has no answer, `$unanswered`. An error reply of the model passes on
/// with its kind and its message: the members of sim_camera_control and the
/// members this relay exposes declare the kinds of peppy alone, so a kind of
/// the one is a kind of the other. A vacant control slot gives `rejected`,
/// as a model gives for a control it does not model; a relay that names no
/// camera gives `precondition_failed`; a model that does not answer in time
/// gives `timeout`; a model that is not there gives `resource_unavailable`;
/// any other failure gives `attempt_failed`.
macro_rules! relayed_failure {
    ($service:ident, $unanswered:expr) => {
        match $unanswered {
            Unanswered::Refused { kind, message } => match $service::FailureKind::parse(kind) {
                Some(kind) => $service::Failure::new(kind, message),
                None => $service::Failure::new(
                    $service::FailureKind::AttemptFailed,
                    format!(
                        "the simulation's camera response model gave the failure kind `{kind}`, which {} does not declare: {message}",
                        stringify!($service)
                    ),
                ),
            },
            Unanswered::NoModel => $service::Failure::new(
                $service::FailureKind::Rejected,
                NO_RESPONSE_MODEL_MESSAGE,
            ),
            Unanswered::NoCamera(message) => {
                $service::Failure::new($service::FailureKind::PreconditionFailed, message)
            }
            Unanswered::Transport {
                failure: TransportFailure::TimedOut,
                message,
            } => $service::Failure::new($service::FailureKind::Timeout, message),
            Unanswered::Transport {
                failure: TransportFailure::Unreachable,
                message,
            } => $service::Failure::new($service::FailureKind::ResourceUnavailable, message),
            Unanswered::Transport {
                failure: TransportFailure::Broken,
                message,
            } => $service::Failure::new($service::FailureKind::AttemptFailed, message),
        }
    };
}

/// The route from a contract call to the simulation's camera response model,
/// shared by every forwarding service task.
struct ControlRoute {
    runner: Arc<NodeRunner>,
    /// The camera every forwarded request names, worked out once: the copy
    /// and the instance id are fixed for the life of the node.
    camera: std::result::Result<String, String>,
    /// The run of forwarded calls failing in transport. The first failure of
    /// the run is logged and the rest are suppressed until a call reaches the
    /// model, so a caller retrying against a simulation that is gone leaves
    /// one line for the whole run.
    transport_failures: LatchedLog,
}

impl ControlRoute {
    fn new(runner: Arc<NodeRunner>) -> Self {
        let camera = camera_name(runner.copy(), runner.processor().bound_instance_id());
        Self {
            runner,
            camera,
            transport_failures: LatchedLog::warn("forwarding to the camera response model"),
        }
    }

    /// Forwards one call. `bound` is the control slot's producer, `call`
    /// polls the response model for this relay's camera on it, and the
    /// model's answer relays verbatim. `refusal_of` reads the model's error
    /// reply in the error of a poll that failed: its kind, by its wire name,
    /// and its message. Every way the call gets no answer is an `Err` that
    /// says why: the vacant slot first, then a relay whose id names no
    /// camera, both fixed for the life of the node; then the model's error
    /// reply or the transport. The pairing is no condition here: the model
    /// tells the relay's robot by the pair the relay holds, so a relay
    /// holding none gets the model's own error reply, relayed like any other.
    fn forward<T, Fut>(
        &self,
        bound: Option<&ProducerRef>,
        call: impl FnOnce(String, ProducerRef) -> Fut,
        refusal_of: impl FnOnce(&peppygen::Error) -> Option<(&'static str, String)>,
    ) -> std::result::Result<T, Unanswered>
    where
        Fut: Future<Output = Result<T>>,
    {
        let Some(target) = bound else {
            return Err(Unanswered::NoModel);
        };
        let camera = self.camera.clone().map_err(Unanswered::NoCamera)?;
        match wait_for(call(camera, target.clone())) {
            Ok(answer) => {
                self.transport_failures.success();
                Ok(answer)
            }
            Err(error) => Err(self.unanswered(&error, refusal_of(&error))),
        }
    }

    /// Why a poll failed with `error`, logging the first of a run of
    /// failures in transport.
    fn unanswered(
        &self,
        error: &peppygen::Error,
        refusal: Option<(&'static str, String)>,
    ) -> Unanswered {
        let unanswered = unanswered_of(error, refusal);
        if unanswered.in_transport() {
            self.transport_failures.failure(error);
        } else {
            self.transport_failures.success();
        }
        unanswered
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
            let publish_failures = LatchedLog::warn(concat!($label, " publish"));
            let invalid_timestamps = ReportOnce::default();
            let mut first = true;
            let receive_errors = LatchedLog::error(concat!("simulation ", $label, " receive"));
            loop {
                let received = tokio::select! {
                    _ = token.cancelled() => return,
                    received = sub.next() => received,
                };
                let msg = match received {
                    Ok(Some((_, msg))) => msg,
                    Ok(None) => return,
                    Err(e) => {
                        receive_errors.failure(e);
                        tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
                        continue;
                    }
                };
                receive_errors.success();
                if !timestamp_is_valid(msg.header.timestamp) {
                    if invalid_timestamps.first_of_run() {
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
                invalid_timestamps.clear();
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
                        publish_failures.success();
                        if first {
                            first = false;
                            info!(concat!(
                                "first ",
                                $label,
                                " frame relayed from the simulation"
                            ));
                        }
                    }
                    Err(e) => publish_failures.failure(e),
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
    let invalid_stream_infos = ReportOnce::default();
    let receive_errors = LatchedLog::error("simulation stream_info receive");
    loop {
        let received = tokio::select! {
            _ = token.cancelled() => return,
            received = sub.next() => received,
        };
        match received {
            Ok(Some((_, msg))) => {
                if !depth_unit_is_valid(msg.depth_unit) {
                    if invalid_stream_infos.first_of_run() {
                        warn!(
                            "ignoring stream_info with invalid depth_unit {}, suppressing repeats",
                            msg.depth_unit
                        );
                    }
                    continue;
                }
                invalid_stream_infos.clear();
                *lock_ignoring_poison(&description) = StreamDescription {
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
                receive_errors.failure(e);
                tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
            }
        }
    }
}

/// Track the simulation's latest camera geometry for the geometry services.
async fn track_geometry(
    runner: Arc<NodeRunner>,
    geometry: SharedGeometry,
    token: CancellationToken,
) {
    let mut sub = match simulation_geometry::subscribe(&runner).await {
        Ok(s) => s,
        Err(e) => return error!("simulation geometry subscribe: {e}"),
    };
    let unusable_geometries = ReportOnce::default();
    let receive_errors = LatchedLog::error("simulation geometry receive");
    loop {
        let received = tokio::select! {
            _ = token.cancelled() => return,
            received = sub.next() => received,
        };
        match received {
            Ok(Some((_, msg))) => {
                if let Some(part) = unusable_part(&msg) {
                    if unusable_geometries.first_of_run() {
                        warn!("ignoring geometry with an unusable {part}, suppressing repeats");
                    }
                    continue;
                }
                unusable_geometries.clear();
                *lock_ignoring_poison(&geometry) = Some(msg);
            }
            Ok(None) => return,
            Err(e) => {
                receive_errors.failure(e);
                tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
            }
        }
    }
}

/// Answer every request of `$service` with `$respond`, which maps the
/// request to the response or the failure of an error reply, until the node
/// stops. The first error of a run of failed requests is logged.
macro_rules! spawn_service {
    ($runner:expr, $service:ident, $respond:expr) => {{
        let runner = $runner.clone();
        let respond = $respond;
        tokio::spawn(async move {
            let cancel = runner.cancellation_token().clone();
            let service_errors = LatchedLog::error(stringify!($service));
            loop {
                let result = tokio::select! {
                    _ = cancel.cancelled() => break,
                    result = $service::handle_next_request(&runner, &respond) => result,
                };
                match result {
                    Ok(()) => service_errors.success(),
                    Err(e) => {
                        service_errors.failure(e);
                        tokio::time::sleep(RECEIVE_ERROR_BACKOFF).await;
                    }
                }
            }
        });
    }};
}

/// Answer one service from what the simulation last said in `$said`:
/// a stream-info service from the shared stream description, a geometry
/// service from the shared geometry. `$respond` gives the response, or the
/// failure of an error reply.
macro_rules! spawn_info_service {
    ($runner:expr, $said:expr, $service:ident, $respond:expr) => {{
        let said = $said.clone();
        spawn_service!($runner, $service, move |_request| {
            let latest = lock_ignoring_poison(&said).clone();
            $respond(latest)
        })
    }};
}

/// Answer one geometry service with `$answer` from the simulation's latest
/// geometry. A camera that has not heard from its simulation yet gives an
/// error reply of the kind `precondition_failed`.
macro_rules! spawn_geometry_service {
    ($runner:expr, $geometry:expr, $service:ident, $answer:expr) => {
        spawn_info_service!($runner, $geometry, $service, |geometry: Option<_>| {
            geometry.map($answer).ok_or_else(|| {
                $service::Failure::new(
                    $service::FailureKind::PreconditionFailed,
                    NO_GEOMETRY_MESSAGE,
                )
            })
        })
    };
}

/// One service whose every request routes through the control route:
/// `$answer` maps the request to the response or the failure of an error
/// reply, forwarding inside.
macro_rules! spawn_forwarding_service {
    ($route:expr, $service:ident, $answer:expr) => {{
        let route: Arc<ControlRoute> = $route.clone();
        let runner = route.runner.clone();
        spawn_service!(runner, $service, move |request| ($answer)(&route, request))
    }};
}

/// One colour control forwarded to its response-model twin: the request's
/// fields go through under the camera's name, and the model's response
/// relays verbatim, as does the kind and the message of its error reply. A
/// call that gets no answer of the model gets the error reply
/// `relayed_failure!` names. `$reading` names the one field of the response: a temperature for
/// white balance, a value for the other controls, the profile JSON for
/// get_camera_profile; reset_camera answers none.
macro_rules! spawn_forwarding_control {
    ($route:expr, $service:ident => $control:ident, [$($field:ident),*] $(, $reading:ident)?) => {
        spawn_forwarding_service!(
            $route,
            $service,
            |route: &ControlRoute, _request: $service::Request| {
                $(let $field = _request.data.$field;)*
                route
                    .forward(
                        $control::bound_producer(&route.runner),
                        |camera, target| async move {
                            let request =
                                $control::Request::new(String::new(), camera $(, $field)*);
                            $control::poll(&route.runner, &target, CONTROL_TIMEOUT, request)
                                .await
                                $(.map(|response| response.data.$reading))?
                        },
                        |error| {
                            $control::Failure::of(error)
                                .map(|failure| (failure.kind.as_str(), failure.message))
                        },
                    )
                    $(.map(|$reading| $service::Response::new($reading)))?
                    .map_err(|unanswered| relayed_failure!($service, unanswered))
            }
        )
    };
}

/// The node's entry point: the exact closure `NodeBuilder::run` used to get,
/// named so the test harness can boot the node in-process.
pub async fn setup(_params: Parameters, node_runner: Arc<NodeRunner>) -> Result<()> {
    let token = node_runner.cancellation_token().clone();
    let description: SharedDescription = Arc::new(Mutex::new(StreamDescription::default()));
    let geometry: SharedGeometry = Arc::new(Mutex::new(None));
    let route = Arc::new(ControlRoute::new(node_runner.clone()));
    match &route.camera {
        Ok(camera) => info!("this relay is camera '{camera}'"),
        Err(unnamed) => warn!("every forwarded call gets an error reply: {unnamed}"),
    }
    if describe_camera::bound_producer(&node_runner).is_some() {
        info!("camera response model linked: colour controls forward to the simulation");
    } else {
        info!("no camera response model linked: every colour control gets an error reply");
    }

    spawn_info_service!(
        node_runner,
        description,
        video_stream_info,
        |d: StreamDescription| {
            Ok(video_stream_info::Response::new(
                d.width,
                d.height,
                d.frames_per_second,
                d.encoding,
            ))
        }
    );
    spawn_info_service!(
        node_runner,
        description,
        depth_stream_info,
        |d: StreamDescription| {
            Ok(depth_stream_info::Response::new(
                d.depth_width,
                d.depth_height,
                d.frames_per_second,
                d.depth_encoding,
                d.depth_unit,
            ))
        }
    );
    spawn_geometry_service!(
        node_runner,
        geometry,
        get_color_intrinsics,
        color_intrinsics
    );
    spawn_geometry_service!(
        node_runner,
        geometry,
        get_depth_intrinsics,
        depth_intrinsics
    );
    spawn_geometry_service!(
        node_runner,
        geometry,
        get_depth_to_color_extrinsics,
        depth_to_color_extrinsics
    );
    spawn_forwarding_control!(
        route,
        set_color_exposure => set_camera_exposure,
        [mode, value],
        current_value
    );
    spawn_forwarding_control!(
        route,
        set_color_white_balance => set_camera_white_balance,
        [mode, temperature],
        current_temperature
    );
    spawn_forwarding_control!(route, set_color_gain => set_camera_gain, [value], current_value);
    spawn_forwarding_control!(
        route,
        set_color_brightness => set_camera_brightness,
        [value],
        current_value
    );
    spawn_forwarding_control!(
        route,
        set_color_contrast => set_camera_contrast,
        [value],
        current_value
    );
    // The profile is the simulation's description of the device this camera
    // stands for, and the reset puts every control back to its defaults:
    // both live with the response model, so both forward to it.
    spawn_forwarding_control!(route, get_camera_profile => describe_camera, [], profile_json);
    spawn_forwarding_control!(route, reset_camera => control_reset_camera, []);

    let video = tokio::spawn(relay_video(node_runner.clone(), token.clone()));
    let depth = tokio::spawn(relay_depth(node_runner.clone(), token.clone()));
    let info = tokio::spawn(track_stream_info(
        node_runner.clone(),
        description,
        token.clone(),
    ));
    let geometry = tokio::spawn(track_geometry(node_runner.clone(), geometry, token.clone()));
    // A dead relay leg would hold its direction silently while the node
    // reports healthy; flag it and cancel, so the process exits with a
    // failure the stack shows instead of a clean finish.
    tokio::spawn(async move {
        tokio::select! {
            _ = video => {}
            _ = depth => {}
            _ = info => {}
            _ = geometry => {}
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

    /// The chest camera as the engines publish it: 1280x720 colour, 640x360
    /// depth from the same 52 degree view, aligned.
    fn chest_geometry() -> simulation_geometry::Message {
        simulation_geometry::Message {
            width: 1280,
            height: 720,
            fx: 738.1094,
            fy: 738.1094,
            cx: 639.5,
            cy: 359.5,
            distortion_model: "none".to_string(),
            distortion: Vec::new(),
            depth_width: 640,
            depth_height: 360,
            depth_fx: 369.0547,
            depth_fy: 369.0547,
            depth_cx: 319.5,
            depth_cy: 179.5,
            depth_distortion_model: "none".to_string(),
            depth_distortion: Vec::new(),
            depth_model: "z".to_string(),
            min_depth_m: 0.1,
            max_depth_m: 10.0,
            align_mode: "depth_to_color".to_string(),
            depth_to_color_position: [0.0, 0.0, 0.0],
            depth_to_color_orientation: [0.0, 0.0, 0.0, 1.0],
        }
    }

    #[test]
    fn a_geometry_is_unusable_by_its_first_bad_part() {
        assert_eq!(unusable_part(&chest_geometry()), None);
        assert_eq!(
            unusable_part(&simulation_geometry::Message {
                fx: 0.0,
                ..chest_geometry()
            }),
            Some("colour pinhole model")
        );
        assert_eq!(
            unusable_part(&simulation_geometry::Message {
                depth_width: 0,
                ..chest_geometry()
            }),
            Some("depth pinhole model")
        );
        assert_eq!(
            unusable_part(&simulation_geometry::Message {
                max_depth_m: 0.05,
                ..chest_geometry()
            }),
            Some("depth range")
        );
        assert_eq!(
            unusable_part(&simulation_geometry::Message {
                depth_to_color_orientation: [0.0; 4],
                ..chest_geometry()
            }),
            Some("depth-to-colour pose")
        );
    }

    #[test]
    fn each_geometry_answer_reads_its_own_stream() {
        // The depth grid is not the colour grid: an answer that read the
        // colour numbers for depth would place every point at twice its
        // distance from the optical axis.
        let color = color_intrinsics(chest_geometry());
        assert_eq!((color.width, color.height), (1280, 720));
        assert_eq!((color.fx, color.cx, color.cy), (738.1094, 639.5, 359.5));

        let depth = depth_intrinsics(chest_geometry());
        assert_eq!((depth.width, depth.height), (640, 360));
        assert_eq!((depth.fx, depth.cx, depth.cy), (369.0547, 319.5, 179.5));
        assert_eq!(depth.depth_model, "z");
        assert_eq!((depth.min_depth_m, depth.max_depth_m), (0.1, 10.0));

        // The two depth answers name the same alignment.
        let pose = depth_to_color_extrinsics(chest_geometry());
        assert_eq!(pose.align_mode, depth.align_mode);
        assert_eq!(pose.depth_to_color_orientation, [0.0, 0.0, 0.0, 1.0]);
    }

    /// The failure kind of the error reply that `unanswered` gives
    /// set_color_gain, by its wire name, and its message.
    fn relayed(unanswered: Unanswered) -> (&'static str, String) {
        let failure = relayed_failure!(set_color_gain, unanswered);
        (failure.kind.as_str(), failure.message)
    }

    #[test]
    fn an_error_reply_of_the_model_passes_on_with_its_kind_and_its_message() {
        for kind in set_camera_gain::FailureKind::ALL {
            let error = peppygen::Error::Deserialization("an error reply".to_owned());
            let unanswered = unanswered_of(
                &error,
                Some((kind.as_str(), "gain 9000 is above 255".to_owned())),
            );
            assert!(!unanswered.in_transport(), "an error reply is an answer");
            assert_eq!(
                relayed(unanswered),
                (kind.as_str(), "gain 9000 is above 255".to_owned())
            );
        }
    }

    #[test]
    fn a_call_that_gets_no_answer_gives_the_kind_of_its_cause() {
        let timed_out = unanswered_of(
            &peppygen::Error::ServiceTimeout {
                instance_id: Some("sim".to_owned()),
                service_name: "set_camera_gain".to_owned(),
            },
            None,
        );
        let unreachable = unanswered_of(
            &peppygen::Error::ServiceUnreachable {
                instance_id: Some("sim".to_owned()),
                service_name: "set_camera_gain".to_owned(),
            },
            None,
        );
        let broken = unanswered_of(
            &peppygen::Error::Deserialization("a response that does not decode".to_owned()),
            None,
        );
        for unanswered in [&timed_out, &unreachable, &broken] {
            assert!(unanswered.in_transport(), "{unanswered:?}");
        }
        let (kind, message) = relayed(timed_out);
        assert_eq!(kind, "timeout");
        assert!(
            message.starts_with("forwarding to the simulation's camera response model failed: ")
                && message.contains("timed out"),
            "{message}"
        );
        assert_eq!(relayed(unreachable).0, "resource_unavailable");
        let (kind, message) = relayed(broken);
        assert_eq!(kind, "attempt_failed");
        assert!(
            message.contains("a response that does not decode"),
            "{message}"
        );

        // Causes fixed for the life of the node.
        assert_eq!(
            relayed(Unanswered::NoModel),
            ("rejected", NO_RESPONSE_MODEL_MESSAGE.to_owned())
        );
        assert_eq!(
            relayed(Unanswered::NoCamera(
                "this relay names no camera".to_owned()
            )),
            (
                "precondition_failed",
                "this relay names no camera".to_owned()
            )
        );
        assert!(!Unanswered::NoModel.in_transport());
    }
}
