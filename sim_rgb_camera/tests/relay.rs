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
    geometry as simulation_geometry, stream_info as simulation_info,
    video_stream as simulation_video,
};
use peppygen::paired_topics::simulation::video_stream::MessageHeader;

/// Bounds every poll and every wait for a relayed frame. Generous: the
/// assertions are about what arrives, never about how fast.
const TIMEOUT: Duration = Duration::from_secs(10);

/// The id the forwarding test runs the relay under. The harness runs a node
/// outside any copy, so the id is the camera's name as it is; a copy's prefix
/// coming off is covered by the node's own tests.
const CAMERA: &str = "wrist_left";

/// The message of the error reply the node gives on every control while the
/// control slot is vacant, verbatim: a caller reads it to learn there is no
/// model to adjust.
const NO_RESPONSE_MODEL_MESSAGE: &str =
    "no camera response model is linked: nothing to adjust, describe or reset";

/// The profile the control mock describes the camera with; opaque to the
/// relay, which must hand it over untouched.
const PROFILE_JSON: &str =
    r#"{"device":"sim","label":"Sim color camera","encoding":"rgb8","controls":{}}"#;

/// The message of the error reply get_color_intrinsics gives before the
/// simulation's first geometry, verbatim: a consumer reads it to learn it is
/// early, not broken.
const NO_GEOMETRY_MESSAGE: &str = "no camera geometry received from the simulation yet";

/// The message of the error reply the two depth services always give,
/// verbatim: a consumer reads it to learn this camera will never have a depth
/// stream.
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
/// The error reply of a relay that has no geometry yet is polled again.
async fn color_intrinsics_once(
    harness: &Harness,
    ready: impl Fn(&get_color_intrinsics::Response) -> bool,
) -> peppygen::Result<get_color_intrinsics::Response> {
    let deadline = tokio::time::Instant::now() + TIMEOUT;
    loop {
        match get_color_intrinsics::poll(harness, TIMEOUT).await {
            Ok(response) if ready(&response) => return Ok(response),
            Ok(_) => {}
            Err(error) => {
                let early = get_color_intrinsics::Failure::of(&error)
                    .is_some_and(|failure| failure.message == NO_GEOMETRY_MESSAGE);
                if !early {
                    return Err(error);
                }
            }
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
    // there is nothing behind them. An error reply is the honest answer; a
    // response would tell a caller its adjustment took effect. Its kind is
    // the one a model gives for a control it does not model, and its message
    // names the cause verbatim, so a caller can tell it apart from a model
    // that refused a value.
    macro_rules! assert_refuses {
        ($service:ident, $request:expr) => {{
            let error = $service::poll(&harness, &$request, TIMEOUT)
                .await
                .expect_err(concat!(
                    stringify!($service),
                    " must give an error reply without a response model"
                ));
            assert_eq!(
                $service::Failure::of(&error),
                Some($service::Failure::new(
                    $service::FailureKind::Rejected,
                    NO_RESPONSE_MODEL_MESSAGE
                ))
            );
        }};
    }

    assert_refuses!(
        set_exposure,
        set_exposure::RequestData {
            mode: "manual".to_string(),
            value: 100,
        }
    );
    assert_refuses!(
        set_white_balance,
        set_white_balance::RequestData {
            mode: "manual".to_string(),
            temperature: 4000,
        }
    );
    assert_refuses!(set_gain, set_gain::RequestData { value: 10 });
    assert_refuses!(set_brightness, set_brightness::RequestData { value: 10 });
    assert_refuses!(set_contrast, set_contrast::RequestData { value: 10 });

    // The profile is the model's description of the device, so without a
    // model there is none, and nothing to reset either.
    let profile = get_camera_profile::poll(&harness, TIMEOUT)
        .await
        .expect_err("no profile without a response model");
    assert_eq!(
        get_camera_profile::Failure::of(&profile),
        Some(get_camera_profile::Failure::new(
            get_camera_profile::FailureKind::Rejected,
            NO_RESPONSE_MODEL_MESSAGE
        ))
    );
    let reset = reset_camera::poll(&harness, TIMEOUT)
        .await
        .expect_err("no reset without a response model");
    assert_eq!(
        reset_camera::Failure::of(&reset),
        Some(reset_camera::Failure::new(
            reset_camera::FailureKind::Rejected,
            NO_RESPONSE_MODEL_MESSAGE
        ))
    );

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_control_forwards_to_the_response_model_under_the_camera_name() -> peppygen::Result<()>
{
    // The default boot binds the control slot to the mock model, and the
    // relay runs under the camera's name, which is all a request's `camera`
    // comes from: the slot the relay views says nothing of it.
    let config = Config {
        instance_id: Some(CAMERA.to_string()),
        ..Default::default()
    };
    let (harness, mocks) = Harness::start_with(config, sim_rgb_camera::setup).await?;
    let control = mocks
        .deps
        .control
        .as_ref()
        .expect("the default boot binds the control slot");

    // Each control goes to its twin on the model with the request's own
    // fields under the camera's name, and the model's answer comes back
    // verbatim: a relay that rewrote the reading would hide what the device
    // model decided. The scripted answers are served by the
    // mock as the requests arrive, and every request is captured before its
    // answer leaves, so the capture is complete once the poll returns.
    macro_rules! assert_forwards {
        ($service:ident, $twin:ident, $request:expr, $answer:expr, $reading:ident, $expect:expr) => {{
            control.$twin.enqueue_response($answer)?;
            let response = $service::poll(&harness, &$request, TIMEOUT).await?;
            assert_eq!(response.$reading, $answer.$reading);
            let captured = control.$twin.captured()?;
            assert_eq!(captured.len(), 1, stringify!($twin));
            assert_eq!(captured[0].camera, CAMERA);
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
        control_mock::set_camera_exposure::ResponseData::new(250),
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
        control_mock::set_camera_white_balance::ResponseData::new(4600),
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
        control_mock::set_camera_gain::ResponseData::new(12),
        current_value,
        |request: &control_mock::set_camera_gain::Request| assert_eq!(request.value, 12)
    );
    assert_forwards!(
        set_brightness,
        set_camera_brightness,
        set_brightness::RequestData { value: 64 },
        control_mock::set_camera_brightness::ResponseData::new(64),
        current_value,
        |request: &control_mock::set_camera_brightness::Request| assert_eq!(request.value, 64)
    );
    assert_forwards!(
        set_contrast,
        set_camera_contrast,
        set_contrast::RequestData { value: 40 },
        control_mock::set_camera_contrast::ResponseData::new(40),
        current_value,
        |request: &control_mock::set_camera_contrast::Request| assert_eq!(request.value, 40)
    );

    // The profile is whatever the model describes, byte for byte: the relay
    // has no device of its own to describe.
    control
        .describe_camera
        .enqueue_response(control_mock::describe_camera::ResponseData::new(
            PROFILE_JSON.to_string(),
        ))?;
    let profile = get_camera_profile::poll(&harness, TIMEOUT).await?;
    assert_eq!(profile.profile_json, PROFILE_JSON);
    let described = control.describe_camera.captured()?;
    assert_eq!(described.len(), 1);
    assert_eq!(described[0].camera, CAMERA);

    control.reset_camera.enqueue_response()?;
    reset_camera::poll(&harness, TIMEOUT).await?;
    let resets = control.reset_camera.captured()?;
    assert_eq!(resets.len(), 1);
    assert_eq!(resets[0].camera, CAMERA);

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_error_reply_of_the_response_model_passes_on_with_its_kind_and_its_message()
-> peppygen::Result<()> {
    let (harness, mocks) = Harness::start(sim_rgb_camera::setup).await?;
    let control = mocks
        .deps
        .control
        .as_ref()
        .expect("the default boot binds the control slot");

    // An unscripted request parks at the mock until the test answers it, so
    // the relay's poll and the mock's error reply run side by side: the poll
    // returns only once the reply is delivered. The relay passes the reply on
    // as it is: a kind or a message of its own would hide why the model
    // refused.
    for kind in control_mock::set_camera_gain::FailureKind::ALL {
        let message = format!("the device model gives {kind}");
        let (relayed, served) = tokio::join!(
            set_gain::poll(&harness, &set_gain::RequestData { value: 9000 }, TIMEOUT),
            async {
                let (request, responder) = control.set_camera_gain.next_request(TIMEOUT).await?;
                assert_eq!(request.value, 9000);
                responder.respond_failure(kind, message.clone()).await
            }
        );
        served?;
        let error = relayed.expect_err("the model gave an error reply");
        let failure = set_gain::Failure::of(&error).expect("the relay gives an error reply");
        assert_eq!(failure.kind.as_str(), kind.as_str());
        assert_eq!(failure.message, message);
    }

    // The profile and the reset pass on the model's error replies the same
    // way.
    let (relayed, served) = tokio::join!(get_camera_profile::poll(&harness, TIMEOUT), async {
        let (_, responder) = control.describe_camera.next_request(TIMEOUT).await?;
        responder
            .respond_failure(
                control_mock::describe_camera::FailureKind::PreconditionFailed,
                "the camera is not attached to its simulation yet",
            )
            .await
    });
    served?;
    assert_eq!(
        get_camera_profile::Failure::of(&relayed.expect_err("the model gave an error reply")),
        Some(get_camera_profile::Failure::new(
            get_camera_profile::FailureKind::PreconditionFailed,
            "the camera is not attached to its simulation yet"
        ))
    );
    let (relayed, served) = tokio::join!(reset_camera::poll(&harness, TIMEOUT), async {
        let (_, responder) = control.reset_camera.next_request(TIMEOUT).await?;
        responder
            .respond_failure(
                control_mock::reset_camera::FailureKind::ResourceUnavailable,
                "the robot of this camera is gone",
            )
            .await
    });
    served?;
    assert_eq!(
        reset_camera::Failure::of(&relayed.expect_err("the model gave an error reply")),
        Some(reset_camera::Failure::new(
            reset_camera::FailureKind::ResourceUnavailable,
            "the robot of this camera is gone"
        ))
    );

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn color_intrinsics_answer_from_the_simulation_geometry() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start(sim_rgb_camera::setup).await?;

    // Before the simulation says where the pixels point there is no model to
    // hand over. Zeros in a response would have a consumer divide by a focal
    // length of zero, so the service gives an error reply and says why.
    let early = get_color_intrinsics::poll(&harness, TIMEOUT)
        .await
        .expect_err("no geometry before the simulation's first");
    assert_eq!(
        get_color_intrinsics::Failure::of(&early),
        Some(get_color_intrinsics::Failure::new(
            get_color_intrinsics::FailureKind::PreconditionFailed,
            NO_GEOMETRY_MESSAGE
        ))
    );

    mocks
        .pairings
        .simulation
        .geometry
        .publish(&geometry())
        .await?;

    let color = color_intrinsics_once(&harness, |_| true).await?;
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
    // a colour camera has nothing behind them. The error reply is `rejected`
    // and names that cause, so a consumer can tell it apart from a geometry
    // that is merely late: it is the same before the geometry arrives and
    // after.
    macro_rules! assert_no_depth_stream {
        () => {{
            let depth = get_depth_intrinsics::poll(&harness, TIMEOUT)
                .await
                .expect_err("a colour camera has no depth intrinsics");
            assert_eq!(
                get_depth_intrinsics::Failure::of(&depth),
                Some(get_depth_intrinsics::Failure::new(
                    get_depth_intrinsics::FailureKind::Rejected,
                    NO_DEPTH_STREAM_MESSAGE
                ))
            );

            let pose = get_depth_to_color_extrinsics::poll(&harness, TIMEOUT)
                .await
                .expect_err("a colour camera has no depth-to-colour pose");
            assert_eq!(
                get_depth_to_color_extrinsics::Failure::of(&pose),
                Some(get_depth_to_color_extrinsics::Failure::new(
                    get_depth_to_color_extrinsics::FailureKind::Rejected,
                    NO_DEPTH_STREAM_MESSAGE
                ))
            );
        }};
    }

    assert_no_depth_stream!();
    mocks
        .pairings
        .simulation
        .geometry
        .publish(&geometry())
        .await?;
    color_intrinsics_once(&harness, |_| true).await?;
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
    color_intrinsics_once(&harness, |_| true).await?;

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
