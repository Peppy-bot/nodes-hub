use ffmpeg::format::Pixel;
use ffmpeg::software::scaling::{Context as ScalerContext, Flags as ScalerFlags};
use ffmpeg::util::frame::video::Video as VideoFrame;
use ffmpeg_next as ffmpeg;
use peppygen::emitted_topics::rgb_camera::v1::video_stream::{self, MessageHeader};
use peppygen::exposed_services::rgb_camera::v1::video_stream_info;
use peppygen::parameters::{self};
use peppygen::{NodeBuilder, Parameters, Result, StandaloneConfig};
use peppylib::runtime::CancellationToken;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime};

fn get_source_video_fps(video_path: &PathBuf) -> u8 {
    let input = ffmpeg::format::input(video_path)
        .unwrap_or_else(|e| panic!("Failed to open video file '{}': {e}", video_path.display()));

    let video_stream = input
        .streams()
        .best(ffmpeg::media::Type::Video)
        .expect("No video stream found");

    let source_fps = video_stream.avg_frame_rate();
    if source_fps.numerator() > 0 && source_fps.denominator() > 0 {
        (source_fps.numerator() as f64 / source_fps.denominator() as f64).round() as u8
    } else {
        30 // Default fallback
    }
}

fn main() -> Result<()> {
    ffmpeg::init().expect("Failed to initialize FFmpeg");

    // Probe source video to get its actual frame rate
    let video_path = std::env::current_dir()
        .expect("Failed to get current working directory")
        .join("assets")
        .join("robot.mp4");

    if !video_path.exists() {
        panic!("Video file not found: {}", video_path.display());
    }

    let source_fps = get_source_video_fps(&video_path);
    println!(
        "[uvc_camera] Detected source video frame rate: {} fps",
        source_fps
    );

    // Load parameters from mock file for standalone execution
    let mock_params_path = std::env::current_dir()
        .expect("Failed to get current working directory")
        .join("mock_parameters.json");
    let mock_params_json = fs::read_to_string(&mock_params_path)
        .unwrap_or_else(|e| panic!("Failed to read '{}': {e}", mock_params_path.display()));
    let mock_params: Parameters = serde_json::from_str(&mock_params_json)
        .unwrap_or_else(|e| panic!("Failed to parse '{}': {e}", mock_params_path.display()));

    // Fallback configuration for standalone execution (e.g., `cargo run`).
    // Ignored when the node is launched by the peppy daemon, which provides its own parameters.
    let standalone_config = StandaloneConfig::new().with_parameters(&mock_params);

    NodeBuilder::new()
        // Fallback configuration for standalone execution (e.g., `cargo run`).
        // Ignored when the node is launched by the peppy daemon, which provides its own parameters.
        .standalone(standalone_config)
        .run(move |args: Parameters, node_runner| async move {
        let video_params = args.video.clone();

        println!(
            "[uvc_camera] Video params: {}x{} @ {} fps, encoding: {}",
            video_params.resolution.width,
            video_params.resolution.height,
            video_params.frame_rate,
            video_params.topic_encoding
        );

        // Validate encoding before spawning - this node outputs RGB24 format data
        let encoding = &video_params.topic_encoding;
        if encoding != "rgb8" && encoding != "rgb" {
            panic!(
                "Invalid encoding '{}'. This camera node outputs RGB24 data, so encoding must be 'rgb8' or 'rgb'",
                encoding
            );
        }

        // Service to expose camera info - use the actual source fps
        let service_node_runner = Arc::clone(&node_runner);
        let service_video_params = video_params.clone();
        let service_cancel_token = node_runner.cancellation_token().clone();
        let actual_fps = source_fps;
        tokio::spawn(async move {
            listen_for_video_stream_info_requests(service_node_runner, service_video_params, actual_fps, service_cancel_token).await;
        });

        // Long running tasks should always be spawned in a different thread
        let cancel_token = node_runner.cancellation_token().clone();
        // Log when the shutdown/cancel signal is received so it is visible in
        // the node's stdout.
        node_runner.on_shutdown(async move {
            println!("[uvc_camera] Shutdown signal received");
        });
        tokio::spawn(async move {
            if let Err(e) = run_video_loop(node_runner, video_params, cancel_token).await {
                tracing::error!("Video loop error: {e:?}");
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
    println!("[uvc_camera] Starting video loop...");
    let video_path = std::env::current_dir()
        .expect("Failed to get current working directory")
        .join("assets")
        .join("robot.mp4");

    if !video_path.exists() {
        panic!("Video file not found: {}", video_path.display());
    }
    println!("[uvc_camera] Video file found: {}", video_path.display());

    let mut frame_id: u32 = 0;
    let mut last_print_time = Instant::now();

    let width = video_params.resolution.width as u32;
    let height = video_params.resolution.height as u32;
    let encoding = video_params.topic_encoding.clone();
    let frame_duration_ms = 1000 / video_params.frame_rate as u64;

    // The blocking ffmpeg decode runs on a dedicated std::thread: unlike work
    // on the tokio blocking pool, such a thread never delays process exit at
    // shutdown (it checks the token per packet/frame, and the bounded channel
    // errors out once this receiving task is gone). This task stays fully
    // async so it always parks at an .await and observes the token.
    let (frame_tx, mut frame_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(1);
    let decode_video_path = video_path.clone();
    let decode_cancel_token = cancel_token.clone();
    std::thread::spawn(move || {
        decode_frames(
            &decode_video_path,
            width,
            height,
            &frame_tx,
            &decode_cancel_token,
        );
    });

    // Declare the publisher once; every publish below is then lock-free.
    let publisher = video_stream::declare_publisher(&node_runner).await?;

    loop {
        let data = tokio::select! {
            _ = cancel_token.cancelled() => {
                println!("[uvc_camera] Shutdown requested, stopping video loop");
                return Ok(());
            }
            frame = frame_rx.recv() => match frame {
                Some(data) => data,
                // The decode thread exited (cancellation or panic)
                None => return Ok(()),
            },
        };

        let header = MessageHeader {
            stamp: SystemTime::now(),
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
            println!("[uvc_camera] Emitted frame {}", frame_id);
            last_print_time = Instant::now();
        }

        frame_id = frame_id.wrapping_add(1);

        tokio::select! {
            _ = cancel_token.cancelled() => {
                println!("[uvc_camera] Shutdown requested, stopping video loop");
                return Ok(());
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(frame_duration_ms)) => {}
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
    video_path: &PathBuf,
    width: u32,
    height: u32,
    frame_tx: &tokio::sync::mpsc::Sender<Vec<u8>>,
    cancel_token: &CancellationToken,
) {
    loop {
        if cancel_token.is_cancelled() {
            return;
        }

        println!("[uvc_camera] Opening video file for playback...");
        let mut input = ffmpeg::format::input(video_path).unwrap_or_else(|e| {
            panic!("Failed to open video file '{}': {e}", video_path.display())
        });

        let video_stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("No video stream found");
        let video_stream_index = video_stream.index();

        // Use software decoder (libdav1d) to avoid hardware acceleration issues
        let codec = ffmpeg::decoder::find_by_name("libdav1d")
            .expect("libdav1d decoder not found - install libdav1d-dev");

        let mut context_decoder =
            ffmpeg::codec::Context::from_parameters(video_stream.parameters())
                .expect("Failed to create codec context");

        // Disable threading to avoid potential hardware acceleration paths
        context_decoder.set_threading(ffmpeg::threading::Config::default());

        let mut decoder = context_decoder
            .decoder()
            .open_as(codec)
            .expect("Failed to open decoder")
            .video()
            .expect("Failed to create video decoder");

        let mut scaler = ScalerContext::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            Pixel::RGB24,
            width,
            height,
            ScalerFlags::BILINEAR,
        )
        .expect("Failed to create scaler");

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
                return;
            }
            if stream.index() == video_stream_index {
                decoder.send_packet(&packet).ok();
                receive_and_send_frames(&mut decoder).ok();
            }
        }

        // Flush the decoder
        decoder.send_eof().ok();
        receive_and_send_frames(&mut decoder).ok();

        // Loop restarts - video will be reopened from the beginning
        println!("[uvc_camera] Video ended, restarting from beginning...");
    }
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
                    tracing::error!("get_camera_info service error: {e:?}");
                }
            }
        }
    }
}
