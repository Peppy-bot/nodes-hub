//! Integration tests over the generated harness: the node in-process, the
//! simulation peer played by the generated pairing mock over the real wire,
//! the camera response model played by the generated control mock, and the
//! contract surface observed the way a consumer sees it.

use std::time::{Duration, SystemTime};

use peppygen::fixtures::exposed_services::camera::{
    set_brightness, set_contrast, set_exposure, set_gain, set_white_balance, video_stream_info,
};
use peppygen::fixtures::exposed_services::profile::{get_camera_profile, reset_camera};
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::control as control_mock;
use peppygen::mock::pairings::simulation::{
    self as simulation_mock, stream_info as simulation_info, video_stream as simulation_video,
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
