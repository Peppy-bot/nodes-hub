use ffmpeg::format::Pixel;
use ffmpeg::software::scaling::{Context as ScalerContext, Flags as ScalerFlags};
use ffmpeg::util::frame::video::Video as VideoFrame;
use ffmpeg_next as ffmpeg;
use peppygen::emitted_topics::video_stream::{self, MessageHeader};
use peppygen::exposed_services::video_stream_info;
use peppygen::parameters::{
    self,
};
use peppygen::{NodeBuilder, Parameters, Result, StandaloneConfig};
use peppylib::runtime::CancellationToken;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use std::fs;

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
        let actual_fps = source_fps;
        tokio::spawn(async move {
            listen_for_video_stream_info_requests(service_node_runner, service_video_params, actual_fps).await;
        });

        // Long running tasks should always be spawned in a different thread
        let cancel_token = node_runner.cancellation_token().clone();
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

    loop {
        if cancel_token.is_cancelled() {
            println!("[uvc_camera] Shutdown requested, stopping video loop");
            return Ok(());
        }

        println!("[uvc_camera] Opening video file for playback...");
        let mut input = ffmpeg::format::input(&video_path).unwrap_or_else(|e| {
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

        let mut receive_and_emit_frames =
            |decoder: &mut ffmpeg::decoder::Video| -> std::result::Result<(), ffmpeg::Error> {
                let mut decoded_frame = VideoFrame::empty();
                while decoder.receive_frame(&mut decoded_frame).is_ok() {
                    let mut rgb_frame = VideoFrame::empty();
                    scaler.run(&decoded_frame, &mut rgb_frame)?;

                    let data: Vec<u8> = rgb_frame.data(0).to_vec();

                    let header = MessageHeader {
                        stamp: SystemTime::now(),
                        frame_id,
                    };

                    // Use blocking emit since we're in a sync closure
                    let node_runner = Arc::clone(&node_runner);
                    let encoding = encoding.clone();
                    let current_frame_id = frame_id;
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            video_stream::emit(&node_runner, header, encoding, width, height, data)
                                .await
                                .expect("Failed to emit frame");
                        });
                    });
                    if last_print_time.elapsed().as_secs() >= 3 {
                        println!("[uvc_camera] Emitted frame {}", current_frame_id);
                        last_print_time = Instant::now();
                    }

                    frame_id = frame_id.wrapping_add(1);

                    std::thread::sleep(std::time::Duration::from_millis(frame_duration_ms));
                }
                Ok(())
            };

        for (stream, packet) in input.packets() {
            if cancel_token.is_cancelled() {
                println!("[uvc_camera] Shutdown requested, stopping video loop");
                return Ok(());
            }
            if stream.index() == video_stream_index {
                decoder.send_packet(&packet).ok();
                receive_and_emit_frames(&mut decoder).ok();
            }
        }

        // Flush the decoder
        decoder.send_eof().ok();
        receive_and_emit_frames(&mut decoder).ok();

        // Loop restarts - video will be reopened from the beginning
        println!("[uvc_camera] Video ended, restarting from beginning...");
    }
}

async fn listen_for_video_stream_info_requests(
    node_runner: Arc<peppygen::NodeRunner>,
    video_params: parameters::video::Video,
    actual_fps: u8,
) {
    loop {
        let params = video_params.clone();
        let fps = actual_fps;
        if let Err(e) = video_stream_info::handle_next_request(&node_runner, move |_request| {
            Ok(video_stream_info::Response::new(
                params.resolution.width as u32,
                params.resolution.height as u32,
                fps,
                params.topic_encoding.clone(),
            ))
        })
        .await
        {
            tracing::error!("get_camera_info service error: {e:?}");
        }
    }
}
