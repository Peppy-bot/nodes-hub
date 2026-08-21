//! Integration tests over the generated harness: the node in-process, the
//! engine peer played by the generated pairing mock over the real wire, and
//! the contract surface observed the way a consumer sees it.

use std::time::{Duration, SystemTime};

use peppygen::fixtures::exposed_services::camera::{
    depth_stream_info, set_color_brightness, set_color_contrast, set_color_exposure,
    set_color_gain, set_color_white_balance, video_stream_info,
};
use peppygen::fixtures::harness::Harness;
use peppygen::mock::pairings::engine::{
    depth_stream as engine_depth, stream_info as engine_info, video_stream as engine_video,
};
use peppygen::paired_topics::engine::{
    depth_stream::MessageHeader as DepthHeader, video_stream::MessageHeader as ColorHeader,
};

/// Bounds every poll and every wait for a relayed frame. Generous: the
/// assertions are about what arrives, never about how fast.
const TIMEOUT: Duration = Duration::from_secs(10);

/// A refused control answers with the no-value sentinel rather than inventing
/// a reading, matching what the uvc and zed nodes answer.
const NO_CURRENT_VALUE: i32 = -1;

/// The engine renders one camera, so color and depth are aligned by
/// construction and say so.
const ALIGN_MODE: &str = "depth_to_color";

/// Metres per depth LSB on the wire, the value the engine publishes and the
/// depth info service must hand back unrounded.
const DEPTH_UNIT: f32 = 0.001;

/// Colour width carried only by the descriptions a test expects to be
/// rejected, so adopting one is visible on a second field.
const REJECTED_WIDTH: u32 = 999;

/// How long to let rejected descriptions settle before reading the served one
/// back. The harness wire delivers in-process, so this is a wide margin over
/// the delivery it waits out, not a guess at it.
const REJECTION_SETTLE: Duration = Duration::from_millis(500);

fn description(depth_unit: f32) -> engine_info::Message {
    engine_info::Message {
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
                    " never reflected the engine's description"
                )
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }};
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn relays_both_streams_verbatim_and_drops_invalid_timestamps() -> peppygen::Result<()> {
    let (mut harness, mocks) = Harness::start(sim_rgbd_camera::setup).await?;

    // A frame stamped at the epoch is what an engine publishes before its
    // clock resolves; both legs must drop it rather than forward a sample no
    // consumer can age. Each leg publishes its bad frame first through the
    // same mock publisher, so the first frame to surface proves the drop.
    let timestamp = SystemTime::UNIX_EPOCH + Duration::from_secs(1_780_000_000);

    mocks
        .pairings
        .engine
        .video_stream
        .publish(&engine_video::Message {
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
    let color = engine_video::Message {
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
    mocks.pairings.engine.video_stream.publish(&color).await?;

    mocks
        .pairings
        .engine
        .depth_stream
        .publish(&engine_depth::Message {
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
    let depth = engine_depth::Message {
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
    mocks.pairings.engine.depth_stream.publish(&depth).await?;

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
async fn both_info_services_answer_from_the_engine_description() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start(sim_rgbd_camera::setup).await?;

    // Before the engine describes its stream, depth_unit is zero: a recorder
    // reading it refuses rather than scaling every depth reading by a guess.
    let depth = depth_stream_info::poll(&harness, TIMEOUT).await?;
    assert_eq!(depth.depth_unit, 0.0);
    assert_eq!(depth.width, 0);

    mocks
        .pairings
        .engine
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
        .engine
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
            .engine
            .stream_info
            .publish(&engine_info::Message {
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
        .engine
        .stream_info
        .publish(&engine_info::Message {
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
async fn every_hardware_control_refuses() -> peppygen::Result<()> {
    let (harness, _mocks) = Harness::start(sim_rgbd_camera::setup).await?;

    // The contract requires these services to exist, and a rendered stream has
    // no sensor behind them. Refusing is the honest answer; answering success
    // would tell a caller its adjustment took effect.
    //
    // The reading field is named for what the control adjusts, so the caller
    // names it: white balance reports a temperature where the rest report a
    // value.
    macro_rules! assert_refuses {
        ($service:ident, $request:expr, $reading:ident) => {{
            let response = $service::poll(&harness, &$request, TIMEOUT).await?;
            assert!(
                !response.success,
                concat!(stringify!($service), " must refuse in sim")
            );
            assert!(!response.message.is_empty());
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

    harness.shutdown().await
}
