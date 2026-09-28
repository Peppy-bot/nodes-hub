// The relay loops and their assembly: frames forward to the contract
// surface, the stream descriptions feed the info service, the camera
// geometry feeds the geometry services, the camera controls and the profile
// forward to the simulation's camera response model under the camera's name,
// which is this relay's name in its copy, and `setup` wires them together.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use peppygen::consumed_services::control::{
    describe_camera, reset_camera as control_reset_camera, set_camera_brightness,
    set_camera_contrast, set_camera_exposure, set_camera_gain, set_camera_white_balance,
};
use peppygen::emitted_topics::camera::video_stream as camera_video;
use peppygen::exposed_services::camera::{
    set_brightness, set_contrast, set_exposure, set_gain, set_white_balance, video_stream_info,
};
use peppygen::exposed_services::geometry::{
    get_color_intrinsics, get_depth_intrinsics, get_depth_to_color_extrinsics,
};
use peppygen::exposed_services::profile::{get_camera_profile, reset_camera};
use peppygen::paired_topics::simulation::{
    geometry as simulation_geometry, stream_info, video_stream as simulation_video,
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

/// current_value in a refused control response: the no-value sentinel uvc and
/// zed answer when no usable hardware value exists.
const NO_CURRENT_VALUE: i32 = -1;

/// The refusal get_color_intrinsics answers until the simulation's first
/// geometry arrives. The relay has no camera model of its own, and zeros
/// handed over as a pinhole model would have a consumer divide by them.
const NO_GEOMETRY_MESSAGE: &str = "no camera geometry received from the simulation yet";

/// What get_color_intrinsics says beside the numbers it hands over: they are
/// the simulation's, relayed as they came.
const GEOMETRY_MESSAGE: &str = "geometry as published by the simulation";

/// The refusal the two depth services of camera_geometry always answer. The
/// contract requires them of every camera, and a colour camera has no depth
/// stream to describe or to place against its colour one.
const NO_DEPTH_STREAM_MESSAGE: &str = "a colour camera has no depth stream";

/// The simulation's latest stream description; zeros until the first one arrives.
#[derive(Clone, Default)]
struct StreamDescription {
    width: u32,
    height: u32,
    frames_per_second: u8,
    encoding: String,
}

type SharedDescription = Arc<Mutex<StreamDescription>>;

/// The simulation's latest camera geometry, kept as it came; none until the
/// first one arrives.
type SharedGeometry = Arc<Mutex<Option<simulation_geometry::Message>>>;

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

/// The camera this relay is, as the simulation names it: the relay's name
/// in its copy. A launch mints a copy's instance ids as `<copy>_<id>`, so
/// relay `alpha_wrist_left` of copy `alpha` is camera `wrist_left`, and a
/// relay launched outside a copy runs under the id the launcher wrote, which
/// is the camera's name as it is. The simulation's slot holds a pair per
/// camera of its kind, so the slot says nothing of which camera a pair is;
/// the relay's requests name no robot, which the simulation tells by the
/// pair the relay holds.
fn camera_name(copy: Option<&str>, instance_id: &str) -> Forwarded<String> {
    let Some(copy) = copy else {
        return Ok(instance_id.to_owned());
    };
    instance_id
        .strip_prefix(copy)
        .and_then(|rest| rest.strip_prefix('_'))
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            format!(
                "this relay names no camera: its instance id {instance_id:?} is not `{copy}_` followed by the camera's name in copy {copy:?}"
            )
        })
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
    /// The camera every forwarded request names, worked out once: the copy
    /// and the instance id are fixed for the life of the node.
    camera: Forwarded<String>,
    /// Whether the last forwarded call failed in transport. The first failure
    /// is logged and the rest are suppressed until a call goes through: a
    /// caller retrying against a simulation that is gone would otherwise
    /// write a line per attempt.
    transport_failing: AtomicBool,
}

impl ControlRoute {
    fn new(runner: Arc<NodeRunner>) -> Self {
        let camera = camera_name(runner.copy(), runner.processor().bound_instance_id());
        Self {
            runner,
            camera,
            transport_failing: AtomicBool::new(false),
        }
    }

    /// Forwards one call. `bound` is the control slot's producer, `call`
    /// polls the response model for this relay's camera on it, and the
    /// model's answer relays verbatim. Every way the call cannot reach the
    /// model is an `Err` carrying the caller's message: the vacant slot
    /// first, then a relay whose id names no camera, both fixed for the life
    /// of the node; then the transport. The pairing is no condition here:
    /// the model tells the relay's robot by the pair the relay holds, so a
    /// relay holding none gets the model's own refusal, relayed like any
    /// other answer.
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
        let camera = self.camera.clone()?;
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

/// Forward the simulation's frames to the contract video_stream.
async fn relay_frames(runner: Arc<NodeRunner>, token: CancellationToken) {
    let mut sub = match simulation_video::subscribe(&runner).await {
        Ok(s) => s,
        Err(e) => return error!("simulation video_stream subscribe: {e}"),
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
                    .report("simulation video_stream receive", e)
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
                    info!("first frame relayed from the simulation");
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

/// A pinhole model a consumer can use: an image with a size, focal lengths it
/// can divide by and a principal point that is a number.
fn pinhole_is_valid(width: u32, height: u32, fx: f64, fy: f64, cx: f64, cy: f64) -> bool {
    width > 0
        && height > 0
        && fx.is_finite()
        && fx > 0.0
        && fy.is_finite()
        && fy > 0.0
        && cx.is_finite()
        && cy.is_finite()
}

/// get_color_intrinsics: the stream's pinhole model as the simulation
/// published it, or the refusal while no geometry has arrived.
fn color_intrinsics(
    geometry: Option<simulation_geometry::Message>,
) -> get_color_intrinsics::Response {
    match geometry {
        Some(g) => get_color_intrinsics::Response::new(
            true,
            GEOMETRY_MESSAGE.to_string(),
            g.width,
            g.height,
            g.fx,
            g.fy,
            g.cx,
            g.cy,
            g.distortion_model,
            g.distortion,
        ),
        None => get_color_intrinsics::Response::new(
            false,
            NO_GEOMETRY_MESSAGE.to_string(),
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            String::new(),
            Vec::new(),
        ),
    }
}

/// get_depth_intrinsics on a colour camera: there is no depth stream to
/// describe, whatever the simulation has said.
fn depth_intrinsics(
    _geometry: Option<simulation_geometry::Message>,
) -> get_depth_intrinsics::Response {
    get_depth_intrinsics::Response::new(
        false,
        NO_DEPTH_STREAM_MESSAGE.to_string(),
        0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        String::new(),
        Vec::new(),
        String::new(),
        0.0,
        0.0,
        String::new(),
    )
}

/// get_depth_to_color_extrinsics on a colour camera: with no depth stream
/// there is nothing to place against the colour one.
fn depth_to_color_extrinsics(
    _geometry: Option<simulation_geometry::Message>,
) -> get_depth_to_color_extrinsics::Response {
    get_depth_to_color_extrinsics::Response::new(
        false,
        NO_DEPTH_STREAM_MESSAGE.to_string(),
        String::new(),
        [0.0; 3],
        [0.0; 4],
    )
}

/// Track the simulation's latest stream description for the info service.
async fn track_stream_info(
    runner: Arc<NodeRunner>,
    description: SharedDescription,
    token: CancellationToken,
) {
    let mut sub = match stream_info::subscribe(&runner).await {
        Ok(s) => s,
        Err(e) => return error!("simulation stream_info subscribe: {e}"),
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
                receive_errors
                    .report("simulation stream_info receive", e)
                    .await;
            }
        }
    }
}

/// Track the simulation's latest camera geometry for get_color_intrinsics.
async fn track_geometry(
    runner: Arc<NodeRunner>,
    geometry: SharedGeometry,
    token: CancellationToken,
) {
    let mut sub = match simulation_geometry::subscribe(&runner).await {
        Ok(s) => s,
        Err(e) => return error!("simulation geometry subscribe: {e}"),
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
                // A consumer turns a pixel into a ray with these numbers, so
                // a model it cannot divide by is ignored whole.
                if !pinhole_is_valid(msg.width, msg.height, msg.fx, msg.fy, msg.cx, msg.cy) {
                    if !rejecting {
                        rejecting = true;
                        warn!(
                            "ignoring geometry with an unusable pinhole model, suppressing repeats"
                        );
                    }
                    continue;
                }
                rejecting = false;
                *geometry
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(msg);
            }
            Ok(None) => return,
            Err(e) => {
                receive_errors
                    .report("simulation geometry receive", e)
                    .await;
            }
        }
    }
}

/// Answer one service from what the simulation last said: the stream-info
/// service from the shared stream description, a geometry service from the
/// shared geometry.
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

/// One camera control forwarded to its response-model twin: the request's
/// fields go through under the camera's name and the model's answer
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
                        let request =
                            $control::Request::new(String::new(), camera $(, $field)*);
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
    let geometry: SharedGeometry = Arc::new(Mutex::new(None));
    let route = Arc::new(ControlRoute::new(node_runner.clone()));
    match &route.camera {
        Ok(camera) => info!("this relay is camera '{camera}'"),
        Err(unnamed) => warn!("every forwarded call refuses: {unnamed}"),
    }
    if describe_camera::bound_producer(&node_runner).is_some() {
        info!("camera response model linked: controls forward to the simulation");
    } else {
        info!("no camera response model linked: every control refuses");
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
        geometry,
        get_color_intrinsics,
        color_intrinsics
    );
    spawn_info_service!(
        node_runner,
        geometry,
        get_depth_intrinsics,
        depth_intrinsics
    );
    spawn_info_service!(
        node_runner,
        geometry,
        get_depth_to_color_extrinsics,
        depth_to_color_extrinsics
    );
    spawn_forwarding_control!(route, set_exposure => set_camera_exposure, [mode, value], current_value = NO_CURRENT_VALUE);
    spawn_forwarding_control!(
        route,
        set_white_balance => set_camera_white_balance,
        [mode, temperature],
        current_temperature = NO_CURRENT_VALUE
    );
    spawn_forwarding_control!(route, set_gain => set_camera_gain, [value], current_value = NO_CURRENT_VALUE);
    spawn_forwarding_control!(route, set_brightness => set_camera_brightness, [value], current_value = NO_CURRENT_VALUE);
    spawn_forwarding_control!(route, set_contrast => set_camera_contrast, [value], current_value = NO_CURRENT_VALUE);
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

    let frames = tokio::spawn(relay_frames(node_runner.clone(), token.clone()));
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
            _ = frames => {}
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

    #[test]
    fn a_relay_in_a_copy_is_the_camera_its_id_names_after_the_copy() {
        assert_eq!(
            camera_name(Some("alpha"), "alpha_wrist_left"),
            Ok("wrist_left".to_owned())
        );
        // Only the copy's own prefix comes off: a camera may carry the
        // copy's name in its own.
        assert_eq!(
            camera_name(Some("alpha"), "alpha_alpha_front"),
            Ok("alpha_front".to_owned())
        );
    }

    #[test]
    fn a_relay_outside_a_copy_is_the_camera_its_id_names() {
        assert_eq!(camera_name(None, "wrist_left"), Ok("wrist_left".to_owned()));
        assert_eq!(
            camera_name(None, "alpha_wrist_left"),
            Ok("alpha_wrist_left".to_owned())
        );
    }

    #[test]
    fn an_id_that_is_no_name_in_its_copy_names_no_camera() {
        // Another copy's prefix, the copy's name run into the camera's, the
        // copy's name alone, and the bare prefix.
        for instance_id in ["bravo_wrist_left", "alphawrist_left", "alpha", "alpha_"] {
            let unnamed = camera_name(Some("alpha"), instance_id).unwrap_err();
            assert!(
                unnamed.contains("names no camera") && unnamed.contains(instance_id),
                "{unnamed}"
            );
        }
    }

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

    /// A wrist camera as the engines publish it: 960x600 from a 66 degree
    /// view.
    fn wrist_geometry() -> simulation_geometry::Message {
        simulation_geometry::Message {
            width: 960,
            height: 600,
            fx: 461.9595,
            fy: 461.9595,
            cx: 479.5,
            cy: 299.5,
            distortion_model: "none".to_string(),
            distortion: Vec::new(),
        }
    }

    #[test]
    fn pinhole_guard_wants_a_size_and_focal_lengths_to_divide_by() {
        assert!(pinhole_is_valid(960, 600, 461.9, 461.9, 479.5, 299.5));
        // A principal point off the image is still a camera: a cropped or
        // shifted sensor has one.
        assert!(pinhole_is_valid(960, 600, 461.9, 461.9, -12.0, 900.0));
        assert!(!pinhole_is_valid(0, 600, 461.9, 461.9, 479.5, 299.5));
        assert!(!pinhole_is_valid(960, 0, 461.9, 461.9, 479.5, 299.5));
        for focal in [0.0, -461.9, f64::NAN, f64::INFINITY] {
            assert!(!pinhole_is_valid(960, 600, focal, 461.9, 479.5, 299.5));
            assert!(!pinhole_is_valid(960, 600, 461.9, focal, 479.5, 299.5));
        }
        assert!(!pinhole_is_valid(960, 600, 461.9, 461.9, f64::NAN, 299.5));
        assert!(!pinhole_is_valid(
            960,
            600,
            461.9,
            461.9,
            479.5,
            f64::INFINITY
        ));
    }

    #[test]
    fn color_intrinsics_refuse_until_a_geometry_arrives() {
        let refused = color_intrinsics(None);
        assert!(!refused.success);
        assert_eq!(refused.message, NO_GEOMETRY_MESSAGE);
        assert_eq!((refused.width, refused.height), (0, 0));

        let answered = color_intrinsics(Some(wrist_geometry()));
        assert!(answered.success);
        assert_eq!((answered.width, answered.height), (960, 600));
        assert_eq!(
            (answered.fx, answered.cx, answered.cy),
            (461.9595, 479.5, 299.5)
        );
        assert_eq!(answered.distortion_model, "none");
    }

    #[test]
    fn the_depth_answers_refuse_whatever_the_simulation_said() {
        for geometry in [None, Some(wrist_geometry())] {
            let depth = depth_intrinsics(geometry.clone());
            assert!(!depth.success);
            assert_eq!(depth.message, NO_DEPTH_STREAM_MESSAGE);
            assert_eq!((depth.width, depth.height), (0, 0));

            let pose = depth_to_color_extrinsics(geometry);
            assert!(!pose.success);
            assert_eq!(pose.message, NO_DEPTH_STREAM_MESSAGE);
            assert_eq!(pose.depth_to_color_orientation, [0.0; 4]);
        }
    }
}
