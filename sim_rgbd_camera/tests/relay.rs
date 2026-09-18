//! Integration tests over the generated harness: the node in-process, the
//! simulation peer played by the generated pairing mock over the real wire,
//! the camera response model played by the generated control mock, and the
//! contract surface observed the way a consumer sees it.

use std::time::{Duration, SystemTime};

use peppygen::fixtures::exposed_services::camera::{
    depth_stream_info, set_color_brightness, set_color_contrast, set_color_exposure,
    set_color_gain, set_color_white_balance, video_stream_info,
};
use peppygen::fixtures::exposed_services::profile::{get_camera_profile, reset_camera};
use peppygen::fixtures::harness::{Config, Harness};
use peppygen::mock::deps::control as control_mock;
use peppygen::mock::pairings::simulation::{
    self as simulation_mock, depth_stream as simulation_depth, stream_info as simulation_info,
    video_stream as simulation_video,
};
use peppygen::paired_topics::simulation::{
    depth_stream::MessageHeader as DepthHeader, video_stream::MessageHeader as ColorHeader,
};

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
    r#"{"device":"sim","label":"Sim RGB-D camera","encoding":"rgb8","controls":{}}"#;

/// The simulation renders one camera, so color and depth are aligned by
/// construction and say so.
const ALIGN_MODE: &str = "depth_to_color";

/// Metres per depth LSB on the wire, the value the simulation publishes and the
/// depth info service must hand back unrounded.
const DEPTH_UNIT: f32 = 0.001;

/// Colour width carried only by the descriptions a test expects to be
/// rejected, so adopting one is visible on a second field.
const REJECTED_WIDTH: u32 = 999;

/// How long to let rejected descriptions settle before reading the served one
/// back. The harness wire delivers in-process, so this is a wide margin over
/// the delivery it waits out, not a guess at it.
const REJECTION_SETTLE: Duration = Duration::from_millis(500);

fn description(depth_unit: f32) -> simulation_info::Message {
    simulation_info::Message {
        width: 1280,
        height: 720,
        frames_per_second: 15,
        encoding: "rgb8".to_string(),
        depth_width: 640,
        depth_height: 360,
        depth_encoding: "z16".to_string(),
        depth_unit,
    }
}

/// Polls `service` until `ready` accepts the response, so a test never assumes
/// one round trip ordered the info leg against the service leg.
macro_rules! poll_until {
    ($service:ident, $harness:expr, $ready:expr) => {{
        let deadline = tokio::time::Instant::now() + TIMEOUT;
        loop {
            let response = $service::poll($harness, TIMEOUT).await?;
            if $ready(&response) {
                break response;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                concat!(
                    stringify!($service),
                    " never reflected the simulation's description"
                )
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }};
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn relays_both_streams_verbatim_and_drops_invalid_timestamps() -> peppygen::Result<()> {
    let (mut harness, mocks) = Harness::start(sim_rgbd_camera::setup).await?;

    // A frame stamped at the epoch is what a simulation publishes before its
    // clock resolves; both legs must drop it rather than forward a sample no
    // consumer can age. Each leg publishes its bad frame first through the
    // same mock publisher, so the first frame to surface proves the drop.
    let timestamp = SystemTime::UNIX_EPOCH + Duration::from_secs(1_780_000_000);

    mocks
        .pairings
        .simulation
        .video_stream
        .publish(&simulation_video::Message {
            header: ColorHeader {
                timestamp: SystemTime::UNIX_EPOCH,
                frame_id: 7,
                align_mode: ALIGN_MODE.to_string(),
            },
            encoding: "rgb8".to_string(),
            width: 4,
            height: 2,
            frame: vec![0x11; 4 * 2 * 3],
        })
        .await?;
    let color = simulation_video::Message {
        header: ColorHeader {
            timestamp,
            frame_id: 42,
            align_mode: ALIGN_MODE.to_string(),
        },
        encoding: "rgb8".to_string(),
        width: 4,
        height: 2,
        frame: vec![0xA5; 4 * 2 * 3],
    };
    mocks
        .pairings
        .simulation
        .video_stream
        .publish(&color)
        .await?;

    mocks
        .pairings
        .simulation
        .depth_stream
        .publish(&simulation_depth::Message {
            header: DepthHeader {
                timestamp: SystemTime::UNIX_EPOCH,
                frame_id: 7,
                align_mode: ALIGN_MODE.to_string(),
            },
            encoding: "z16".to_string(),
            width: 2,
            height: 2,
            frame: vec![0x00; 2 * 2 * 2],
        })
        .await?;
    // Depth carries the same frame_id as the color frame it was captured
    // with, which is what lets a consumer pair them.
    let depth = simulation_depth::Message {
        header: DepthHeader {
            timestamp,
            frame_id: 42,
            align_mode: ALIGN_MODE.to_string(),
        },
        encoding: "z16".to_string(),
        width: 2,
        height: 2,
        frame: vec![0x34, 0x12, 0x78, 0x56, 0xBC, 0x9A, 0xF0, 0xDE],
    };
    mocks
        .pairings
        .simulation
        .depth_stream
        .publish(&depth)
        .await?;

    let relayed_color = tokio::time::timeout(TIMEOUT, harness.emitted.camera_video_stream.next())
        .await
        .expect("no color frame reached the contract surface")?
        .expect("video_stream subscription should be open");
    assert_eq!(relayed_color.header.timestamp, timestamp);
    assert_eq!(relayed_color.header.frame_id, 42);
    assert_eq!(relayed_color.header.align_mode, ALIGN_MODE);
    assert_eq!(relayed_color.encoding, color.encoding);
    assert_eq!(relayed_color.width, color.width);
    assert_eq!(relayed_color.height, color.height);
    assert_eq!(relayed_color.frame, color.frame);

    let relayed_depth = tokio::time::timeout(TIMEOUT, harness.emitted.camera_depth_stream.next())
        .await
        .expect("no depth frame reached the contract surface")?
        .expect("depth_stream subscription should be open");
    assert_eq!(relayed_depth.header.timestamp, timestamp);
    assert_eq!(relayed_depth.header.frame_id, 42);
    assert_eq!(relayed_depth.header.align_mode, ALIGN_MODE);
    assert_eq!(relayed_depth.encoding, depth.encoding);
    assert_eq!(relayed_depth.width, depth.width);
    assert_eq!(relayed_depth.height, depth.height);
    // Byte for byte: the depth wire format is little-endian u16, and a relay
    // that reordered or rescaled would silently change every reading.
    assert_eq!(relayed_depth.frame, depth.frame);

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_info_services_answer_from_the_simulation_description() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start(sim_rgbd_camera::setup).await?;

    // Before the simulation describes its stream, depth_unit is zero: a recorder
    // reading it refuses rather than scaling every depth reading by a guess.
    let depth = depth_stream_info::poll(&harness, TIMEOUT).await?;
    assert_eq!(depth.depth_unit, 0.0);
    assert_eq!(depth.width, 0);

    mocks
        .pairings
        .simulation
        .stream_info
        .publish(&description(DEPTH_UNIT))
        .await?;

    let color = poll_until!(
        video_stream_info,
        &harness,
        |r: &video_stream_info::Response| r.width != 0
    );
    assert_eq!(color.width, 1280);
    assert_eq!(color.height, 720);
    assert_eq!(color.frames_per_second, 15);
    assert_eq!(color.encoding, "rgb8");

    let depth = poll_until!(
        depth_stream_info,
        &harness,
        |r: &depth_stream_info::Response| r.width != 0
    );
    // The depth service describes the depth stream, not the color one: a relay
    // that answered with the color geometry would have a consumer decode
    // 1280x720 worth of bytes out of a 640x360 frame.
    assert_eq!(depth.width, 640);
    assert_eq!(depth.height, 360);
    assert_eq!(depth.frames_per_second, 15);
    assert_eq!(depth.encoding, "z16");
    assert_eq!(depth.depth_unit, DEPTH_UNIT);

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_description_with_an_unusable_depth_unit_is_ignored() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start(sim_rgbd_camera::setup).await?;

    // Establish a usable description first, so a rejection can be told apart
    // from never having had one.
    mocks
        .pairings
        .simulation
        .stream_info
        .publish(&description(DEPTH_UNIT))
        .await?;
    let accepted = poll_until!(
        depth_stream_info,
        &harness,
        |r: &depth_stream_info::Response| r.depth_unit != 0.0
    );
    assert_eq!(accepted.depth_unit, DEPTH_UNIT);

    // A depth_unit that is zero, negative or non-finite scales every reading
    // into nonsense. The relay must ignore the whole description rather than
    // adopt it, so each bad one also carries a width the good one does not:
    // adopting it would show up on either field.
    for unusable in [0.0, -0.001, f32::NAN, f32::INFINITY] {
        mocks
            .pairings
            .simulation
            .stream_info
            .publish(&simulation_info::Message {
                width: REJECTED_WIDTH,
                depth_unit: unusable,
                ..description(DEPTH_UNIT)
            })
            .await?;
    }

    // These are the last messages on the leg, so nothing later can restore the
    // served description: whatever it holds after the settle is what the
    // rejections left it. The wire delivers in milliseconds in-process, so the
    // window is orders of magnitude longer than the delivery it waits out.
    tokio::time::sleep(REJECTION_SETTLE).await;
    let after = depth_stream_info::poll(&harness, TIMEOUT).await?;
    assert_eq!(
        after.depth_unit, DEPTH_UNIT,
        "an unusable depth_unit was adopted"
    );
    assert_ne!(
        after.width, REJECTED_WIDTH,
        "a description rejected for its depth_unit was adopted anyway"
    );

    // A later good description must still land, which proves the loop stayed
    // alive through the rejections rather than having ended.
    mocks
        .pairings
        .simulation
        .stream_info
        .publish(&simulation_info::Message {
            depth_unit: 0.002,
            ..description(DEPTH_UNIT)
        })
        .await?;
    let settled = poll_until!(
        depth_stream_info,
        &harness,
        |r: &depth_stream_info::Response| r.depth_unit == 0.002
    );
    assert_eq!(settled.depth_unit, 0.002);

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
    let (harness, mocks) = Harness::start_with(config, sim_rgbd_camera::setup).await?;
    assert!(
        mocks.deps.control.is_none(),
        "a vacant boot must start no control mock"
    );

    // The contract requires these services to exist, and with no model linked
    // there is nothing behind them. Refusing is the honest answer; answering
    // success would tell a caller its adjustment took effect. The message
    // names the cause verbatim, so a caller can tell this apart from a model
    // that refused a value.
    //
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
        set_color_exposure,
        set_color_exposure::RequestData {
            mode: "manual".to_string(),
            value: 100,
        },
        current_value
    );
    assert_refuses!(
        set_color_white_balance,
        set_color_white_balance::RequestData {
            mode: "manual".to_string(),
            temperature: 4000,
        },
        current_temperature
    );
    assert_refuses!(
        set_color_gain,
        set_color_gain::RequestData { value: 10 },
        current_value
    );
    assert_refuses!(
        set_color_brightness,
        set_color_brightness::RequestData { value: 10 },
        current_value
    );
    assert_refuses!(
        set_color_contrast,
        set_color_contrast::RequestData { value: 10 },
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
    let (harness, mocks) = Harness::start(sim_rgbd_camera::setup).await?;
    let control = mocks
        .deps
        .control
        .as_ref()
        .expect("the default boot binds the control slot");

    // Each colour control goes to its twin on the model with the request's
    // own fields under the camera slot's name, and the model's answer comes
    // back verbatim: a relay that rewrote the message or the reading would
    // hide what the device model decided. The scripted answers are served by
    // the mock as the requests arrive, and every request is captured before
    // its answer leaves, so the capture is complete once the poll returns.
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
        set_color_exposure,
        set_camera_exposure,
        set_color_exposure::RequestData {
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
        set_color_white_balance,
        set_camera_white_balance,
        set_color_white_balance::RequestData {
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
        set_color_gain,
        set_camera_gain,
        set_color_gain::RequestData { value: 12 },
        control_mock::set_camera_gain::ResponseData::new(true, "gain set".to_string(), 12),
        current_value,
        |request: &control_mock::set_camera_gain::Request| assert_eq!(request.value, 12)
    );
    assert_forwards!(
        set_color_brightness,
        set_camera_brightness,
        set_color_brightness::RequestData { value: 64 },
        control_mock::set_camera_brightness::ResponseData::new(
            true,
            "brightness set".to_string(),
            64
        ),
        current_value,
        |request: &control_mock::set_camera_brightness::Request| assert_eq!(request.value, 64)
    );
    assert_forwards!(
        set_color_contrast,
        set_camera_contrast,
        set_color_contrast::RequestData { value: 40 },
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
    let (harness, mocks) = Harness::start(sim_rgbd_camera::setup).await?;
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
    let response = set_color_gain::poll(
        &harness,
        &set_color_gain::RequestData { value: 9000 },
        TIMEOUT,
    )
    .await?;
    assert!(!response.success);
    assert_eq!(response.message, refusal.message);
    assert_eq!(response.current_value, 128);

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_response_model_failure_answers_a_refusal_with_the_error() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start(sim_rgbd_camera::setup).await?;
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
        set_color_contrast::poll(
            &harness,
            &set_color_contrast::RequestData { value: 40 },
            TIMEOUT
        ),
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
