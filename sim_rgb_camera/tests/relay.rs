//! Integration tests over the generated harness: the node in-process, the
//! simulation peer played by the generated pairing mock over the real wire,
//! the camera response model played by the generated control mock, and the
//! contract surface observed the way a consumer sees it.

use std::time::{Duration, SystemTime};

use peppygen::fixtures::exposed_services::camera::{
    set_brightness, set_contrast, set_exposure, set_gain, set_white_balance, video_stream_info,
};
use peppygen::fixtures::exposed_services::geometry::{
    get_color_intrinsics, get_depth_intrinsics, get_depth_to_color_extrinsics,
};
use peppygen::fixtures::exposed_services::profile::{get_camera_profile, reset_camera};
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::control as control_mock;
use peppygen::mock::pairings::simulation::{
    self as simulation_mock, geometry as simulation_geometry, stream_info as simulation_info,
    video_stream as simulation_video,
};
use peppygen::paired_topics::simulation::video_stream::MessageHeader;

/// Bounds every poll and every wait for a relayed frame. Generous: the
/// assertions are about what arrives, never about how fast.
const TIMEOUT: Duration = Duration::from_secs(10);

/// A refused control answers with the no-value sentinel rather than inventing
/// a reading, matching what the uvc and zed nodes answer.
const NO_CURRENT_VALUE: i32 = -1;

/// The refusal the node answers on every control while the control slot is
/// vacant, verbatim: a caller reads it to learn there is no model to adjust.
const NO_RESPONSE_MODEL_MESSAGE: &str =
    "no camera response model is linked: nothing to adjust, describe or reset";

/// The profile the control mock describes the camera with; opaque to the
/// relay, which must hand it over untouched.
const PROFILE_JSON: &str =
    r#"{"device":"sim","label":"Sim color camera","encoding":"rgb8","controls":{}}"#;

/// The refusal get_color_intrinsics answers before the simulation's first
/// geometry, verbatim: a consumer reads it to learn it is early, not broken.
const NO_GEOMETRY_MESSAGE: &str = "no camera geometry received from the simulation yet";

/// The refusal the two depth services always answer, verbatim: a consumer
/// reads it to learn this camera will never have a depth stream.
const NO_DEPTH_STREAM_MESSAGE: &str = "a colour camera has no depth stream";

/// Width carried only by the geometries a test expects to be rejected, so
/// adopting one is visible.
const REJECTED_WIDTH: u32 = 999;

/// How long to let rejected geometries settle before reading the served one
/// back. The harness wire delivers in-process, so this is a wide margin over
/// the delivery it waits out, not a guess at it.
const REJECTION_SETTLE: Duration = Duration::from_millis(500);

/// A wrist camera as the engines publish it: 960x600 from a 66 degree view.
fn geometry() -> simulation_geometry::Message {
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

/// Polls get_color_intrinsics until `ready` accepts the response: the
/// geometry arrives on its own leg, so one round trip does not order the two.
async fn color_intrinsics_once(
    harness: &Harness,
    ready: impl Fn(&get_color_intrinsics::Response) -> bool,
) -> peppygen::Result<get_color_intrinsics::Response> {
    let deadline = tokio::time::Instant::now() + TIMEOUT;
    loop {
        let response = get_color_intrinsics::poll(harness, TIMEOUT).await?;
        if ready(&response) {
            return Ok(response);
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the simulation's geometry never reached get_color_intrinsics"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn frame(timestamp: SystemTime, frame_id: u32, fill: u8) -> simulation_video::Message {
    simulation_video::Message {
        header: MessageHeader {
            timestamp,
            frame_id,
        },
        encoding: "rgb8".to_string(),
        width: 4,
        height: 2,
        frame: vec![fill; 4 * 2 * 3],
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn relays_frames_verbatim_and_drops_invalid_timestamps() -> peppygen::Result<()> {
    let (mut harness, mocks) = Harness::start(sim_rgb_camera::setup).await?;

    // A frame stamped at the epoch is what a simulation publishes before its
    // clock resolves; the relay must drop it rather than forward a sample no
    // consumer can age. The two publishes share one mock publisher, so their
    // order holds: the first frame to surface proves both the drop and the
    // relay.
    mocks
        .pairings
        .simulation
        .video_stream
        .publish(&frame(SystemTime::UNIX_EPOCH, 7, 0x11))
        .await?;
    let timestamp = SystemTime::UNIX_EPOCH + Duration::from_secs(1_780_000_000);
    let sent = frame(timestamp, 42, 0xA5);
    mocks
        .pairings
        .simulation
        .video_stream
        .publish(&sent)
        .await?;

    let relayed = tokio::time::timeout(TIMEOUT, harness.emitted.camera_video_stream.next())
        .await
        .expect("no frame reached the contract surface")?
        .expect("video_stream subscription should be open");

    // Verbatim: a relay that restamped or renumbered would make consumers age
    // samples on the relay's clock instead of the simulation's capture time.
    assert_eq!(relayed.header.timestamp, timestamp);
    assert_eq!(relayed.header.frame_id, 42);
    assert_eq!(relayed.encoding, sent.encoding);
    assert_eq!(relayed.width, sent.width);
    assert_eq!(relayed.height, sent.height);
    assert_eq!(relayed.frame, sent.frame);

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_info_service_answers_from_the_simulation_description() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start(sim_rgb_camera::setup).await?;

    // Before the simulation describes its stream there is nothing to report, and
    // the relay says so with zeros rather than a guess: a consumer gating on
    // a usable size refuses, which is the correct answer this early.
    let response = video_stream_info::poll(&harness, TIMEOUT).await?;
    assert_eq!(response.width, 0);
    assert_eq!(response.height, 0);
    assert_eq!(response.frames_per_second, 0);
    assert!(response.encoding.is_empty());

    mocks
        .pairings
        .simulation
        .stream_info
        .publish(&simulation_info::Message {
            width: 960,
            height: 600,
            frames_per_second: 15,
            encoding: "rgb8".to_string(),
        })
        .await?;

    // The description arrives on its own leg, so poll until it lands rather
    // than assuming one round trip ordered the two.
    let deadline = tokio::time::Instant::now() + TIMEOUT;
    loop {
        let response = video_stream_info::poll(&harness, TIMEOUT).await?;
        if response.width != 0 {
            assert_eq!(response.width, 960);
            assert_eq!(response.height, 600);
            assert_eq!(response.frames_per_second, 15);
            assert_eq!(response.encoding, "rgb8");
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the simulation's stream description never reached the info service"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_control_refuses_without_a_response_model() -> peppygen::Result<()> {
    // A launch on a simulation without a camera response model writes the
    // control slot vacant; the pairing itself is established as ever.
    let config = Config {
        control_vacant: true,
        ..Default::default()
    };
    let (harness, mocks) = Harness::start_with(config, sim_rgb_camera::setup).await?;
    assert!(
        mocks.deps.control.is_none(),
        "a vacant boot must start no control mock"
    );

    // The contract requires these services to exist, and with no model linked
    // there is nothing behind them. Refusing is the honest answer; answering
    // success would tell a caller its adjustment took effect. The message
    // names the cause verbatim, so a caller can tell this apart from a model
    // that refused a value.
    // The reading field is named for what the control adjusts, so the caller
    // names it: white balance reports a temperature where the rest report a
    // value.
    macro_rules! assert_refuses {
        ($service:ident, $request:expr, $reading:ident) => {{
            let response = $service::poll(&harness, &$request, TIMEOUT).await?;
            assert!(
                !response.success,
                concat!(
                    stringify!($service),
                    " must refuse without a response model"
                )
            );
            assert_eq!(response.message, NO_RESPONSE_MODEL_MESSAGE);
            assert_eq!(response.$reading, NO_CURRENT_VALUE);
        }};
    }

    assert_refuses!(
        set_exposure,
        set_exposure::RequestData {
            mode: "manual".to_string(),
            value: 100,
        },
        current_value
    );
    assert_refuses!(
        set_white_balance,
        set_white_balance::RequestData {
            mode: "manual".to_string(),
            temperature: 4000,
        },
        current_temperature
    );
    assert_refuses!(set_gain, set_gain::RequestData { value: 10 }, current_value);
    assert_refuses!(
        set_brightness,
        set_brightness::RequestData { value: 10 },
        current_value
    );
    assert_refuses!(
        set_contrast,
        set_contrast::RequestData { value: 10 },
        current_value
    );

    // The profile is the model's description of the device, so without a
    // model there is none: the answer says so and carries no JSON a caller
    // could mistake for one.
    let profile = get_camera_profile::poll(&harness, TIMEOUT).await?;
    assert!(!profile.success);
    assert!(
        profile
            .message
            .contains("no camera response model is linked")
    );
    assert!(profile.profile_json.is_empty());

    let reset = reset_camera::poll(&harness, TIMEOUT).await?;
    assert!(!reset.success);
    assert!(reset.message.contains("no camera response model is linked"));

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_control_forwards_to_the_response_model_under_the_camera_slot() -> peppygen::Result<()>
{
    // The default boot binds the control slot to the mock model and seeds
    // the simulation pairing, so `paired()` names the mock peer's slot from
    // the first request on.
    let (harness, mocks) = Harness::start(sim_rgb_camera::setup).await?;
    let control = mocks
        .deps
        .control
        .as_ref()
        .expect("the default boot binds the control slot");

    // Each control goes to its twin on the model with the request's own
    // fields under the camera slot's name, and the model's answer comes back
    // verbatim: a relay that rewrote the message or the reading would hide
    // what the device model decided. The scripted answers are served by the
    // mock as the requests arrive, and every request is captured before its
    // answer leaves, so the capture is complete once the poll returns.
    macro_rules! assert_forwards {
        ($service:ident, $twin:ident, $request:expr, $answer:expr, $reading:ident, $expect:expr) => {{
            control.$twin.enqueue_response($answer)?;
            let response = $service::poll(&harness, &$request, TIMEOUT).await?;
            assert!(
                response.success,
                concat!(stringify!($service), " must relay the model's success")
            );
            assert_eq!(response.message, $answer.message);
            assert_eq!(response.$reading, $answer.$reading);
            let captured = control.$twin.captured()?;
            assert_eq!(captured.len(), 1, stringify!($twin));
            assert_eq!(captured[0].camera, simulation_mock::PEER_LINK_ID);
            // The relay names no robot: the simulation tells its robot by
            // the pair the relay holds.
            assert_eq!(captured[0].robot, "");
            $expect(&captured[0]);
        }};
    }

    assert_forwards!(
        set_exposure,
        set_camera_exposure,
        set_exposure::RequestData {
            mode: "manual".to_string(),
            value: 250,
        },
        control_mock::set_camera_exposure::ResponseData::new(true, "exposure set".to_string(), 250),
        current_value,
        |request: &control_mock::set_camera_exposure::Request| {
            assert_eq!(request.mode, "manual");
            assert_eq!(request.value, 250);
        }
    );
    assert_forwards!(
        set_white_balance,
        set_camera_white_balance,
        set_white_balance::RequestData {
            mode: "auto".to_string(),
            temperature: 0,
        },
        control_mock::set_camera_white_balance::ResponseData::new(
            true,
            "white balance auto".to_string(),
            4600
        ),
        current_temperature,
        |request: &control_mock::set_camera_white_balance::Request| {
            assert_eq!(request.mode, "auto");
            assert_eq!(request.temperature, 0);
        }
    );
    assert_forwards!(
        set_gain,
        set_camera_gain,
        set_gain::RequestData { value: 12 },
        control_mock::set_camera_gain::ResponseData::new(true, "gain set".to_string(), 12),
        current_value,
        |request: &control_mock::set_camera_gain::Request| assert_eq!(request.value, 12)
    );
    assert_forwards!(
        set_brightness,
        set_camera_brightness,
        set_brightness::RequestData { value: 64 },
        control_mock::set_camera_brightness::ResponseData::new(
            true,
            "brightness set".to_string(),
            64
        ),
        current_value,
        |request: &control_mock::set_camera_brightness::Request| assert_eq!(request.value, 64)
    );
    assert_forwards!(
        set_contrast,
        set_camera_contrast,
        set_contrast::RequestData { value: 40 },
        control_mock::set_camera_contrast::ResponseData::new(true, "contrast set".to_string(), 40),
        current_value,
        |request: &control_mock::set_camera_contrast::Request| assert_eq!(request.value, 40)
    );

    // The profile is whatever the model describes, byte for byte: the relay
    // has no device of its own to describe.
    control
        .describe_camera
        .enqueue_response(control_mock::describe_camera::ResponseData::new(
            true,
            "described".to_string(),
            PROFILE_JSON.to_string(),
        ))?;
    let profile = get_camera_profile::poll(&harness, TIMEOUT).await?;
    assert!(profile.success);
    assert_eq!(profile.message, "described");
    assert_eq!(profile.profile_json, PROFILE_JSON);
    let described = control.describe_camera.captured()?;
    assert_eq!(described.len(), 1);
    assert_eq!(described[0].camera, simulation_mock::PEER_LINK_ID);

    control
        .reset_camera
        .enqueue_response(control_mock::reset_camera::ResponseData::new(
            true,
            "defaults restored".to_string(),
        ))?;
    let reset = reset_camera::poll(&harness, TIMEOUT).await?;
    assert!(reset.success);
    assert_eq!(reset.message, "defaults restored");
    let resets = control.reset_camera.captured()?;
    assert_eq!(resets.len(), 1);
    assert_eq!(resets[0].camera, simulation_mock::PEER_LINK_ID);

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refusal_from_the_response_model_passes_through_unchanged() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start(sim_rgb_camera::setup).await?;
    let control = mocks
        .deps
        .control
        .as_ref()
        .expect("the default boot binds the control slot");

    // A model refusing a value answers false with the reading it kept. The
    // relay must pass that through as is: substituting its own message or the
    // no-value sentinel would hide both why the value was refused and what
    // the camera is still set to.
    let refusal = control_mock::set_camera_gain::ResponseData::new(
        false,
        "gain 9000 is above the device maximum of 255".to_string(),
        128,
    );
    control.set_camera_gain.enqueue_response(refusal.clone())?;
    let response =
        set_gain::poll(&harness, &set_gain::RequestData { value: 9000 }, TIMEOUT).await?;
    assert!(!response.success);
    assert_eq!(response.message, refusal.message);
    assert_eq!(response.current_value, 128);

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_response_model_failure_answers_a_refusal_with_the_error() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start(sim_rgb_camera::setup).await?;
    let control = mocks
        .deps
        .control
        .as_ref()
        .expect("the default boot binds the control slot");

    // An unscripted request parks at the mock until the test answers it, so
    // the relay's poll and the mock's failure run side by side: the poll
    // returns only once the failure is delivered, and the relay must turn it
    // into a refusal that carries the reason rather than an error of its own.
    let (response, served) = tokio::join!(
        set_contrast::poll(&harness, &set_contrast::RequestData { value: 40 }, TIMEOUT),
        async {
            let (request, responder) = control.set_camera_contrast.next_request(TIMEOUT).await?;
            assert_eq!(request.camera, simulation_mock::PEER_LINK_ID);
            responder.respond_error("device model crashed").await
        }
    );
    served?;
    let response = response?;
    assert!(!response.success);
    assert!(
        response.message.contains("device model crashed"),
        "the model's failure must reach the caller: {}",
        response.message
    );
    assert_eq!(response.current_value, NO_CURRENT_VALUE);

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn color_intrinsics_answer_from_the_simulation_geometry() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start(sim_rgb_camera::setup).await?;

    // Before the simulation says where the pixels point there is no model to
    // hand over. Zeros under success would have a consumer divide by a focal
    // length of zero, so the service refuses and says why.
    let early = get_color_intrinsics::poll(&harness, TIMEOUT).await?;
    assert!(!early.success);
    assert_eq!(early.message, NO_GEOMETRY_MESSAGE);
    assert_eq!(early.fx, 0.0);

    mocks
        .pairings
        .simulation
        .geometry
        .publish(&geometry())
        .await?;

    let color = color_intrinsics_once(&harness, |r| r.success).await?;
    assert_eq!((color.width, color.height), (960, 600));
    assert_eq!((color.fx, color.fy), (461.9595, 461.9595));
    assert_eq!((color.cx, color.cy), (479.5, 299.5));
    assert_eq!(color.distortion_model, "none");
    assert!(color.distortion.is_empty());

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_depth_services_refuse_on_a_colour_camera() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start(sim_rgb_camera::setup).await?;

    // camera_geometry asks every implementer for the two depth services, and
    // a colour camera has nothing behind them. The refusal names that cause,
    // so a consumer can tell it apart from a geometry that is merely late:
    // it is the same before the geometry arrives and after.
    macro_rules! assert_no_depth_stream {
        () => {{
            let depth = get_depth_intrinsics::poll(&harness, TIMEOUT).await?;
            assert!(!depth.success);
            assert_eq!(depth.message, NO_DEPTH_STREAM_MESSAGE);
            assert_eq!((depth.width, depth.height), (0, 0));
            assert!(depth.depth_model.is_empty());

            let pose = get_depth_to_color_extrinsics::poll(&harness, TIMEOUT).await?;
            assert!(!pose.success);
            assert_eq!(pose.message, NO_DEPTH_STREAM_MESSAGE);
            assert_eq!(pose.depth_to_color_orientation, [0.0; 4]);
        }};
    }

    assert_no_depth_stream!();
    mocks
        .pairings
        .simulation
        .geometry
        .publish(&geometry())
        .await?;
    color_intrinsics_once(&harness, |r| r.success).await?;
    assert_no_depth_stream!();

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unusable_geometry_is_ignored() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start(sim_rgb_camera::setup).await?;

    // Establish a usable geometry first, so a rejection can be told apart
    // from never having had one.
    mocks
        .pairings
        .simulation
        .geometry
        .publish(&geometry())
        .await?;
    color_intrinsics_once(&harness, |r| r.success).await?;

    // A consumer divides by the focal lengths, so a model without usable ones
    // is ignored whole; each bad one carries a width the good one does not,
    // so adopting it would show.
    for (fx, fy) in [(0.0, 461.9595), (461.9595, f64::NAN), (-461.9595, 461.9595)] {
        mocks
            .pairings
            .simulation
            .geometry
            .publish(&simulation_geometry::Message {
                width: REJECTED_WIDTH,
                fx,
                fy,
                ..geometry()
            })
            .await?;
    }

    // These are the last messages on the leg, so whatever the service holds
    // after the settle is what the rejections left it.
    tokio::time::sleep(REJECTION_SETTLE).await;
    let after = get_color_intrinsics::poll(&harness, TIMEOUT).await?;
    assert!(after.success);
    assert_ne!(
        after.width, REJECTED_WIDTH,
        "an unusable geometry was adopted"
    );
    assert_eq!(after.fx, 461.9595);

    // A later good geometry must still land, which proves the loop stayed
    // alive through the rejections rather than having ended.
    mocks
        .pairings
        .simulation
        .geometry
        .publish(&simulation_geometry::Message {
            width: 1280,
            ..geometry()
        })
        .await?;
    let settled = color_intrinsics_once(&harness, |r| r.width == 1280).await?;
    assert_eq!(settled.width, 1280);

    harness.shutdown().await
}
