use std::sync::Arc;

use peppygen::consumed_services::camera::video_stream_info as camera_video_stream_info;
use peppygen::consumed_topics::camera::video_stream as camera_video_stream;
use peppygen::{NodeBuilder, NodeRunner, Parameters, Result};
use peppylib::runtime::CancellationToken;

use ffmpeg_next::Rational;
use ffmpeg_next::format::Pixel;
use ffmpeg_next::util::frame::video::Video as VideoFrame;

fn main() -> Result<()> {
    ffmpeg_next::init().expect("Failed to initialize FFmpeg");

    NodeBuilder::new().run(|args: Parameters, node_runner| async move {
        let video_duration_seconds = args.video_duration_seconds;

        // Log when the shutdown/cancel signal is received so it is visible in
        // the node's stdout.
        node_runner.on_shutdown(async move {
            println!("[uvc_camera_video_reconstruction] Shutdown signal received");
        });

        tokio::spawn(record_video(node_runner, video_duration_seconds));

        Ok(())
    })
}

async fn record_video(node_runner: Arc<NodeRunner>, video_duration_seconds: u32) {
    let token = node_runner.cancellation_token().clone();

    let camera_info = loop {
        let response = tokio::select! {
            _ = token.cancelled() => return,
            response = camera_video_stream_info::poll(
                &node_runner,
                camera_video_stream_info::bound_producer(&node_runner),
                std::time::Duration::from_secs(5),
            ) => response,
        };

        match response {
            Ok(response) => {
                println!(
                    "Locked onto camera instance_id: {} — {}x{} @ {} fps, encoding: {}",
                    response.instance_id,
                    response.data.width,
                    response.data.height,
                    response.data.frames_per_second,
                    response.data.encoding
                );
                break response.data;
            }
            Err(e) => {
                eprintln!("Failed to get camera info: {}, retrying...", e);
                tokio::select! {
                    _ = token.cancelled() => return,
                    _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {}
                }
            }
        }
    };

    let total_frames = video_duration_seconds * camera_info.frames_per_second as u32;
    println!(
        "Recording {} frames ({} seconds at {} fps)...",
        total_frames, video_duration_seconds, camera_info.frames_per_second
    );

    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(total_frames as usize);

    // Subscribe once; the held subscription buffers frames in order, so the
    // recording loop never misses a frame published between iterations.
    let mut subscription = match camera_video_stream::subscribe(&node_runner).await {
        Ok(subscription) => subscription,
        Err(e) => {
            eprintln!("Failed to subscribe to camera stream: {}", e);
            return;
        }
    };

    for frame_num in 0..total_frames {
        let received = tokio::select! {
            _ = token.cancelled() => return,
            received = subscription.next() => received,
        };
        match received {
            Ok(Some((_producer, message))) => {
                frames.push(message.frame);
                if (frame_num + 1) % camera_info.frames_per_second as u32 == 0 {
                    println!(
                        "Recorded {}/{} frames ({} seconds)",
                        frame_num + 1,
                        total_frames,
                        (frame_num + 1) / camera_info.frames_per_second as u32
                    );
                }
            }
            Ok(None) => break,
            Err(e) => {
                eprintln!("Failed to receive frame: {}", e);
            }
        }
    }

    println!("Recording complete. Encoding video...");

    match encode_video(
        &frames,
        camera_info.width,
        camera_info.height,
        camera_info.frames_per_second,
        &token,
    ) {
        Ok(path) => println!("Video saved to: {}", path),
        Err(e) => eprintln!("Failed to encode video: {}", e),
    }
}

fn encode_video(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
    fps: u8,
    token: &CancellationToken,
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    let output_dir = std::path::PathBuf::from("/tmp/video_reconstruction");
    std::fs::create_dir_all(&output_dir)?;
    let output_path = output_dir.join("reconstructed_video.mp4");
    let output_path_str = output_path.to_string_lossy().to_string();

    let mut output = ffmpeg_next::format::output(&output_path)?;

    let codec =
        ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::H264).ok_or("H264 encoder not found")?;

    let encoder_time_base = Rational::new(1, fps as i32);

    let mut encoder = ffmpeg_next::codec::context::Context::new_with_codec(codec)
        .encoder()
        .video()?;

    encoder.set_width(width);
    encoder.set_height(height);
    encoder.set_format(Pixel::YUV420P);
    encoder.set_time_base(encoder_time_base);
    encoder.set_frame_rate(Some(Rational::new(fps as i32, 1)));

    let encoder = encoder.open_as(codec)?;

    let stream_index = {
        let mut output_stream = output.add_stream(codec)?;
        output_stream.set_parameters(&encoder);
        output_stream.index()
    };

    output.write_header()?;

    // Get the stream's time_base after write_header (muxer may have changed it)
    let stream_time_base = output.stream(stream_index).unwrap().time_base();

    let mut encoder = encoder;

    let mut scaler = ffmpeg_next::software::scaling::Context::get(
        Pixel::RGB24,
        width,
        height,
        Pixel::YUV420P,
        width,
        height,
        ffmpeg_next::software::scaling::Flags::BILINEAR,
    )?;

    for (i, frame_data) in frames.iter().enumerate() {
        // This synchronous encode runs as one poll on the async pool, and the
        // tokio Runtime drop at shutdown blocks until that poll finishes — so
        // the encode must observe the token per frame to stay well inside the
        // shutdown grace window.
        if token.is_cancelled() {
            return Err("shutdown requested; aborting video encode".into());
        }

        let mut rgb_frame = VideoFrame::new(Pixel::RGB24, width, height);
        rgb_frame.data_mut(0).copy_from_slice(frame_data);

        let mut yuv_frame = VideoFrame::empty();
        scaler.run(&rgb_frame, &mut yuv_frame)?;
        yuv_frame.set_pts(Some(i as i64));

        encoder.send_frame(&yuv_frame)?;

        let mut packet = ffmpeg_next::Packet::empty();
        while encoder.receive_packet(&mut packet).is_ok() {
            packet.set_stream(stream_index);
            packet.rescale_ts(encoder_time_base, stream_time_base);
            packet.write_interleaved(&mut output)?;
        }
    }

    encoder.send_eof()?;

    let mut packet = ffmpeg_next::Packet::empty();
    while encoder.receive_packet(&mut packet).is_ok() {
        packet.set_stream(stream_index);
        packet.rescale_ts(encoder_time_base, stream_time_base);
        packet.write_interleaved(&mut output)?;
    }

    output.write_trailer()?;

    println!(
        "Video encoding complete: {}x{} @ {} fps, saved to {}",
        width, height, fps, output_path_str
    );

    Ok(output_path_str)
}
