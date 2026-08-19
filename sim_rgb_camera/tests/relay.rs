//! Integration tests over the generated harness: the node in-process, the
//! engine peer played by the generated pairing mock over the real wire, and
//! the contract surface observed the way a consumer sees it.

use std::time::{Duration, SystemTime};

use peppygen::fixtures::exposed_services::camera::{
    set_brightness, set_contrast, set_exposure, set_gain, set_white_balance, video_stream_info,
};
use peppygen::fixtures::harness::Harness;
use peppygen::mock::pairings::engine::{stream_info as engine_info, video_stream as engine_video};
use peppygen::paired_topics::engine::video_stream::MessageHeader;

/// Bounds every poll and every wait for a relayed frame. Generous: the
/// assertions are about what arrives, never about how fast.
const TIMEOUT: Duration = Duration::from_secs(10);

/// A refused control answers with the no-value sentinel rather than inventing
/// a reading, matching what the uvc and zed nodes answer.
const NO_CURRENT_VALUE: i32 = -1;

fn frame(timestamp: SystemTime, frame_id: u32, fill: u8) -> engine_video::Message {
    engine_video::Message {
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

    // A frame stamped at the epoch is what an engine publishes before its
    // clock resolves; the relay must drop it rather than forward a sample no
    // consumer can age. The two publishes share one mock publisher, so their
    // order holds: the first frame to surface proves both the drop and the
    // relay.
    mocks
        .pairings
        .engine
        .video_stream
        .publish(&frame(SystemTime::UNIX_EPOCH, 7, 0x11))
        .await?;
    let timestamp = SystemTime::UNIX_EPOCH + Duration::from_secs(1_780_000_000);
    let sent = frame(timestamp, 42, 0xA5);
    mocks.pairings.engine.video_stream.publish(&sent).await?;

    let relayed = tokio::time::timeout(TIMEOUT, harness.emitted.camera_video_stream.next())
        .await
        .expect("no frame reached the contract surface")?
        .expect("video_stream subscription should be open");

    // Verbatim: a relay that restamped or renumbered would make consumers age
    // samples on the relay's clock instead of the engine's capture time.
    assert_eq!(relayed.header.timestamp, timestamp);
    assert_eq!(relayed.header.frame_id, 42);
    assert_eq!(relayed.encoding, sent.encoding);
    assert_eq!(relayed.width, sent.width);
    assert_eq!(relayed.height, sent.height);
    assert_eq!(relayed.frame, sent.frame);

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_info_service_answers_from_the_engine_description() -> peppygen::Result<()> {
    let (harness, mocks) = Harness::start(sim_rgb_camera::setup).await?;

    // Before the engine describes its stream there is nothing to report, and
    // the relay says so with zeros rather than a guess: a consumer gating on
    // a usable size refuses, which is the correct answer this early.
    let response = video_stream_info::poll(&harness, TIMEOUT).await?;
    assert_eq!(response.width, 0);
    assert_eq!(response.height, 0);
    assert_eq!(response.frames_per_second, 0);
    assert!(response.encoding.is_empty());

    mocks
        .pairings
        .engine
        .stream_info
        .publish(&engine_info::Message {
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
            "the engine's stream description never reached the info service"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    harness.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_hardware_control_refuses() -> peppygen::Result<()> {
    let (harness, _mocks) = Harness::start(sim_rgb_camera::setup).await?;

    // The contract requires these services to exist, and a rendered stream has
    // no sensor behind them. Refusing is the honest answer; answering success
    // would tell a caller its adjustment took effect.
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

    harness.shutdown().await
}
