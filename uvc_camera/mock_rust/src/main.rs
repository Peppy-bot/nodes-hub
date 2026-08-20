use ffmpeg::format::Pixel;
use ffmpeg::software::scaling::{Context as ScalerContext, Flags as ScalerFlags};
use ffmpeg::util::frame::video::Video as VideoFrame;
use ffmpeg_next as ffmpeg;
use peppygen::emitted_topics::camera::video_stream::{self, MessageHeader};
use peppygen::exposed_services::camera::{
    set_brightness, set_contrast, set_exposure, set_gain, set_white_balance, video_stream_info,
};
use peppygen::parameters::{self};
use peppygen::{NodeBuilder, Parameters, Result, StandaloneConfig};
use peppylib::runtime::CancellationToken;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Timestamp from the daemon-resolved clock; identical to the OS clock in wall
/// mode, the simulator's time under sim time.
fn timestamp_now() -> Result<SystemTime> {
    let ns = peppygen::clock::now_ns()?;
    Ok(UNIX_EPOCH + Duration::from_nanos(ns))
}

/// Everything this mock refuses to run on, or stops for.
///
/// Every variant names the file, parameter or step it failed at and keeps the
/// error it came from, so this list is the one place to read what a launch can
/// be rejected for.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("frame_rate must be greater than zero")]
    FrameRate,

    #[error(
        "invalid encoding '{0}'. This camera node outputs RGB24 data, so encoding must be 'rgb8' or 'rgb'"
    )]
    Encoding(String),

    #[error("read the working directory")]
    WorkingDirectory(#[source] std::io::Error),

    #[error("video file not found: {0}")]
    AssetMissing(PathBuf),

    #[error("read {path}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("parse the mock parameters at {path}")]
    MockParameters {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("initialize FFmpeg")]
    FfmpegInit(#[source] ffmpeg::Error),

    #[error("open the video at {path}")]
    OpenVideo {
        path: PathBuf,
        #[source]
        source: ffmpeg::Error,
    },

    #[error("no video stream in {0}")]
    NoVideoStream(PathBuf),

    /// The rate is served to consumers through `video_stream_info`, so a
    /// container that declares none is refused rather than answered with a
    /// stand-in the video does not actually run at.
    #[error("the video at {0} declares no frame rate")]
    NoSourceFrameRate(PathBuf),

    #[error("libdav1d decoder not found; install libdav1d-dev")]
    NoDecoder,

    #[error("set up the video decoder")]
    Decoder(#[source] ffmpeg::Error),

    #[error("create the {width}x{height} RGB24 scaler")]
    Scaler {
        width: u32,
        height: u32,
        #[source]
        source: ffmpeg::Error,
    },

    /// The runtime's own failures pass through unchanged rather than being
    /// re-wrapped, so a messaging or config error keeps the variant it was
    /// raised as.
    #[error(transparent)]
    Runtime(#[from] peppygen::Error),
}

impl From<NodeError> for peppygen::Error {
    /// The one place this node's refusals meet the runtime's error type.
    ///
    /// A runtime error passes back unchanged; everything else is this node's own
    /// and travels as `Error::Node`, which keeps the wrapped error reachable
    /// through `Error::source` rather than flattening it to a message.
    fn from(e: NodeError) -> Self {
        match e {
            NodeError::Runtime(e) => e,
            other => peppygen::Error::Node(Box::new(other)),
        }
    }
}

/// This node's own results, distinct from `peppygen::Result`, which the runtime
/// takes at the boundary and which is what bare `Result` means in this crate.
type NodeResult<T = ()> = std::result::Result<T, NodeError>;

/// The gap between published frames.
///
/// Refuses a zero rate, which would divide by zero, and keeps microsecond
/// resolution so a high rate is not truncated to a whole millisecond (120 fps
/// is 8333 us, not 8 ms).
fn frame_period(frame_rate: u16) -> NodeResult<Duration> {
    (frame_rate > 0)
        .then(|| Duration::from_micros(1_000_000 / u64::from(frame_rate)))
        .ok_or(NodeError::FrameRate)
}

/// The asset this mock replays. Missing or unreadable, it is a setup failure to
/// report, not a reason to unwind.
fn video_asset_path() -> NodeResult<PathBuf> {
    let path = std::env::current_dir()
        .map_err(NodeError::WorkingDirectory)?
        .join("assets")
        .join("robot.mp4");
    path.exists()
        .then_some(path.clone())
        .ok_or(NodeError::AssetMissing(path))
}

fn get_source_video_fps(video_path: &Path) -> NodeResult<u8> {
    let input = open_video(video_path)?;

    let video_stream = input
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| NodeError::NoVideoStream(video_path.to_path_buf()))?;

    let source_fps = video_stream.avg_frame_rate();
    let (num, den) = (source_fps.numerator(), source_fps.denominator());
    (num > 0 && den > 0)
        .then(|| (f64::from(num) / f64::from(den)).round() as u8)
        .ok_or_else(|| NodeError::NoSourceFrameRate(video_path.to_path_buf()))
}

/// Open a video file, naming it in the failure. The one place a path becomes an
/// ffmpeg input, so the probe and the decode loop report a bad file alike.
fn open_video(path: &Path) -> NodeResult<ffmpeg::format::context::Input> {
    ffmpeg::format::input(path).map_err(|source| NodeError::OpenVideo {
        path: path.to_path_buf(),
        source,
    })
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    ffmpeg::init().map_err(NodeError::FfmpegInit)?;

    // Probe source video to get its actual frame rate
    let video_path = video_asset_path()?;
    let source_fps = get_source_video_fps(&video_path)?;
    tracing::info!("Detected source video frame rate: {} fps", source_fps);

    // Load parameters from mock file for standalone execution
    let mock_params_path = std::env::current_dir()
        .map_err(NodeError::WorkingDirectory)?
        .join("mock_parameters.json");
    let mock_params_json =
        fs::read_to_string(&mock_params_path).map_err(|source| NodeError::ReadFile {
            path: mock_params_path.clone(),
            source,
        })?;
    let mock_params: Parameters =
        serde_json::from_str(&mock_params_json).map_err(|source| NodeError::MockParameters {
            path: mock_params_path,
            source,
        })?;

    // Fallback configuration for standalone execution (e.g., `cargo run`).
    // Ignored when the node is launched by the peppy daemon, which provides its own parameters.
    let standalone_config = StandaloneConfig::new().with_parameters(&mock_params);

    NodeBuilder::new()
        // Fallback configuration for standalone execution (e.g., `cargo run`).
        // Ignored when the node is launched by the peppy daemon, which provides its own parameters.
        .standalone(standalone_config)
        .run(move |args: Parameters, node_runner| async move {
            let video_params = args.video.clone();

            tracing::info!(
                "Video params: {}x{} @ {} fps, encoding: {}",
                video_params.resolution.width,
                video_params.resolution.height,
                video_params.frame_rate,
                video_params.topic_encoding
            );

            // Validate encoding before spawning - this node outputs RGB24 format data
            let encoding = &video_params.topic_encoding;
            if encoding != "rgb8" && encoding != "rgb" {
                return Err(NodeError::Encoding(encoding.clone()).into());
            }

            // Service to expose camera info - use the actual source fps
            let service_node_runner = Arc::clone(&node_runner);
            let service_video_params = video_params.clone();
            let service_cancel_token = node_runner.cancellation_token().clone();
            let actual_fps = source_fps;
            tokio::spawn(async move {
                listen_for_video_stream_info_requests(
                    service_node_runner,
                    service_video_params,
                    actual_fps,
                    service_cancel_token,
                )
                .await;
            });

            spawn_control_acks(&node_runner);

            // The synchronized clock stamping every emission: the OS clock in
            // wall mode, the simulator's time under sim time.
            peppygen::clock::init(&node_runner).await?;

            // Long running tasks should always be spawned in a different thread
            let cancel_token = node_runner.cancellation_token().clone();
            // Log when the shutdown/cancel signal is received so it is visible in
            // the node's stdout.
            node_runner.on_shutdown(async move {
                tracing::info!("Shutdown signal received");
            });
            // A loop that never started publishes nothing while the node
            // answers services as healthy; take the node down with the reason
            // rather than log and stand.
            tokio::spawn(async move {
                if let Err(e) =
                    run_video_loop(node_runner, video_params, cancel_token.clone()).await
                {
                    tracing::error!("video loop stopped: {e}");
                    cancel_token.cancel();
                }
            });

            Ok(())
        })
}

async fn run_video_loop(
    node_runner: Arc<peppygen::NodeRunner>,
    video_params: parameters::video::Video,
    cancel_token: CancellationToken,
) -> Result<()> {
    tracing::info!("Starting video loop...");
    let video_path = video_asset_path()?;
    tracing::info!("Video file found: {}", video_path.display());

    let mut frame_id: u32 = 0;
    let mut last_print_time = Instant::now();

    let width = video_params.resolution.width as u32;
    let height = video_params.resolution.height as u32;
    let encoding = video_params.topic_encoding.clone();
    let frame_period = frame_period(video_params.frame_rate)?;

    // The blocking ffmpeg decode runs on a dedicated std::thread: unlike work
    // on the tokio blocking pool, such a thread never delays process exit at
    // shutdown (it checks the token per packet/frame, and the bounded channel
    // errors out once this receiving task is gone). This task stays fully
    // async so it always parks at an .await and observes the token.
    let (frame_tx, mut frame_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(1);
    let decode_video_path = video_path.clone();
    let decode_cancel_token = cancel_token.clone();
    std::thread::spawn(move || {
        if let Err(e) = decode_frames(
            &decode_video_path,
            width,
            height,
            &frame_tx,
            &decode_cancel_token,
        ) {
            // A decode that never started produces no frames, and a closed
            // channel alone is indistinguishable from a clean stop: without the
            // cancel the node would sit up and answer services forever with
            // nothing on the topic. Log the reason, then take the node down.
            tracing::error!("decode thread stopped: {e}");
            decode_cancel_token.cancel();
        }
    });

    // Declare the publisher once; every publish below is then lock-free.
    let publisher = video_stream::declare_publisher(&node_runner).await?;

    loop {
        let data = tokio::select! {
            _ = cancel_token.cancelled() => {
                tracing::info!("Shutdown requested, stopping video loop");
                return Ok(());
            }
            frame = frame_rx.recv() => match frame {
                Some(data) => data,
                // The decode thread exited. On a failure it logged the reason
                // and cancelled the node; on a clean stop there is nothing to
                // add.
                None => return Ok(()),
            },
        };

        let timestamp = match timestamp_now() {
            Ok(timestamp) => timestamp,
            Err(e) => {
                // Sim mode before the first tick: skip rather than mis-stamp.
                tracing::debug!("skipping frame: {e}");
                continue;
            }
        };
        let header = MessageHeader {
            timestamp,
            frame_id,
        };

        let payload =
            match video_stream::build_message(header, encoding.clone(), width, height, data) {
                Ok(payload) => payload,
                Err(e) => {
                    tracing::error!("Failed to build frame message: {e:?}");
                    continue;
                }
            };
        if let Err(e) = publisher.publish(payload).await {
            tracing::error!("Failed to emit frame: {e:?}");
        }
        if last_print_time.elapsed().as_secs() >= 3 {
            tracing::info!("Emitted frame {}", frame_id);
            last_print_time = Instant::now();
        }

        frame_id = frame_id.wrapping_add(1);

        tokio::select! {
            _ = cancel_token.cancelled() => {
                tracing::info!("Shutdown requested, stopping video loop");
                return Ok(());
            }
            _ = tokio::time::sleep(frame_period) => {}
        }
    }
}

/// Decode the looping source video and push scaled RGB frames into `frame_tx`.
///
/// Runs on a dedicated std::thread because ffmpeg decoding is blocking.
/// `blocking_send` on the bounded channel paces decoding against the emit
/// loop, and returns an error (ending this thread) once the receiver is
/// dropped at shutdown.
fn decode_frames(
    video_path: &Path,
    width: u32,
    height: u32,
    frame_tx: &tokio::sync::mpsc::Sender<Vec<u8>>,
    cancel_token: &CancellationToken,
) -> NodeResult {
    loop {
        if cancel_token.is_cancelled() {
            return Ok(());
        }

        tracing::info!("Opening video file for playback...");
        let mut input = open_video(video_path)?;

        let video_stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| NodeError::NoVideoStream(video_path.to_path_buf()))?;
        let video_stream_index = video_stream.index();

        // Use software decoder (libdav1d) to avoid hardware acceleration issues
        let codec = ffmpeg::decoder::find_by_name("libdav1d").ok_or(NodeError::NoDecoder)?;

        let mut context_decoder =
            ffmpeg::codec::Context::from_parameters(video_stream.parameters())
                .map_err(NodeError::Decoder)?;

        // Disable threading to avoid potential hardware acceleration paths
        context_decoder.set_threading(ffmpeg::threading::Config::default());

        let mut decoder = context_decoder
            .decoder()
            .open_as(codec)
            .map_err(NodeError::Decoder)?
            .video()
            .map_err(NodeError::Decoder)?;

        let mut scaler = ScalerContext::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            Pixel::RGB24,
            width,
            height,
            ScalerFlags::BILINEAR,
        )
        .map_err(|source| NodeError::Scaler {
            width,
            height,
            source,
        })?;

        let mut receive_and_send_frames =
            |decoder: &mut ffmpeg::decoder::Video| -> std::result::Result<(), ffmpeg::Error> {
                let mut decoded_frame = VideoFrame::empty();
                while decoder.receive_frame(&mut decoded_frame).is_ok() {
                    if cancel_token.is_cancelled() {
                        return Ok(());
                    }
                    let mut rgb_frame = VideoFrame::empty();
                    scaler.run(&decoded_frame, &mut rgb_frame)?;

                    let data: Vec<u8> = rgb_frame.data(0).to_vec();
                    if frame_tx.blocking_send(data).is_err() {
                        return Ok(());
                    }
                }
                Ok(())
            };

        for (stream, packet) in input.packets() {
            if cancel_token.is_cancelled() || frame_tx.is_closed() {
                return Ok(());
            }
            if stream.index() == video_stream_index {
                // EAGAIN is the decoder asking to be drained first; every
                // other failure would repeat on each packet, decoding the
                // file at full speed forever while publishing nothing.
                match decoder.send_packet(&packet) {
                    Ok(())
                    | Err(ffmpeg::Error::Other {
                        errno: ffmpeg::util::error::EAGAIN,
                    }) => {}
                    Err(source) => return Err(NodeError::Decoder(source)),
                }
                receive_and_send_frames(&mut decoder).map_err(NodeError::Decoder)?;
            }
        }

        // Flush the decoder. EOF on an already-flushed decoder is its normal
        // answer here; a scaler failure in the drain is the same fault it is
        // mid-stream.
        decoder.send_eof().ok();
        receive_and_send_frames(&mut decoder).map_err(NodeError::Decoder)?;

        // Loop restarts - video will be reopened from the beginning
        tracing::info!("Video ended, restarting from beginning...");
    }
}

/// Answer the five camera control services the `rgb_camera` contract requires.
///
/// The mock has no hardware to adjust, so each one acknowledges the request and
/// echoes the value back. Leaving them unanswered would hang any caller until
/// its timeout, and a stack developed against the mock would only discover the
/// gap when swapped onto real hardware.
fn spawn_control_acks(node_runner: &Arc<peppygen::NodeRunner>) {
    macro_rules! spawn_ack {
        ($service:ident, $request_field:ident) => {{
            let runner = Arc::clone(node_runner);
            let cancel_token = node_runner.cancellation_token().clone();
            tokio::spawn(async move {
                loop {
                    let result = tokio::select! {
                        _ = cancel_token.cancelled() => break,
                        result = $service::handle_next_request(&runner, |request| {
                            Ok($service::Response::new(
                                true,
                                concat!("mock: ", stringify!($service), " acknowledged").to_string(),
                                request.data.$request_field,
                            ))
                        }) => result,
                    };
                    if let Err(e) = result {
                        tracing::error!("{} service error: {e:?}", stringify!($service));
                    }
                }
            });
        }};
    }

    spawn_ack!(set_exposure, value);
    spawn_ack!(set_white_balance, temperature);
    spawn_ack!(set_gain, value);
    spawn_ack!(set_brightness, value);
    spawn_ack!(set_contrast, value);
}

async fn listen_for_video_stream_info_requests(
    node_runner: Arc<peppygen::NodeRunner>,
    video_params: parameters::video::Video,
    actual_fps: u8,
    cancel_token: CancellationToken,
) {
    loop {
        let params = video_params.clone();
        let fps = actual_fps;
        tokio::select! {
            _ = cancel_token.cancelled() => break,
            result = video_stream_info::handle_next_request(&node_runner, move |_request| {
                Ok(video_stream_info::Response::new(
                    params.resolution.width as u32,
                    params.resolution.height as u32,
                    fps,
                    params.topic_encoding.clone(),
                ))
            }) => {
                if let Err(e) = result {
                    tracing::error!("video_stream_info service error: {e:?}");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_frame_rate_is_refused_not_divided_by() {
        // 1000 / frame_rate panics on zero, and the parameter schema constrains
        // the representation but not the sign or the zero.
        let refusal = frame_period(0).expect_err("a zero frame rate must be refused");
        assert!(
            refusal.to_string().contains("frame_rate"),
            "the refusal must name the parameter, got {refusal}"
        );
    }

    #[test]
    fn a_high_frame_rate_keeps_sub_millisecond_resolution() {
        // The previous whole-millisecond division truncated 120 fps to 8 ms,
        // which runs the mock 4% fast; at 1001 fps it truncated to zero.
        assert_eq!(
            frame_period(120).expect("120 fps"),
            Duration::from_micros(8_333)
        );
        assert_eq!(
            frame_period(30).expect("30 fps"),
            Duration::from_micros(33_333)
        );
        assert_eq!(frame_period(1).expect("1 fps"), Duration::from_secs(1));
    }
}
