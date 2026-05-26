use std::sync::Arc;
use std::time::Duration;

use peppygen::consumed_services::{
    camera_set_brightness, camera_set_contrast, camera_set_exposure, camera_set_gain,
    camera_set_white_balance, camera_video_stream_info,
};
use peppygen::consumed_topics::camera_video_stream;
use peppygen::{NodeBuilder, NodeRunner, Parameters, Result};

use ffmpeg_next::Rational;
use ffmpeg_next::format::Pixel;
use ffmpeg_next::util::frame::video::Video as VideoFrame;

fn main() -> Result<()> {
    ffmpeg_next::init().expect("Failed to initialize FFmpeg");

    NodeBuilder::new().run(|_args: Parameters, node_runner| async move {
        tokio::spawn(record_video(node_runner));
        Ok(())
    })
}

async fn record_video(node_runner: Arc<NodeRunner>) {
    let camera_info = loop {
        match camera_video_stream_info::poll(&node_runner, Duration::from_secs(5)).await {
            Ok(response) => {
                println!(
                    "Camera info: {}x{} @ {} fps, encoding: {}",
                    response.data.width,
                    response.data.height,
                    response.data.frames_per_second,
                    response.data.encoding
                );
                break response.data;
            }
            Err(e) => {
                eprintln!("Failed to get camera info: {}, retrying...", e);
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    };

    let fps = camera_info.frames_per_second;
    let mut all_frames: Vec<Vec<u8>> = Vec::new();

    // set_exposure: test manual/200, restore to auto/0
    println!("Testing set_exposure...");
    let _ = camera_set_exposure::poll(
        &node_runner,
        Duration::from_secs(5),
        camera_set_exposure::Request {
            mode: "manual".to_string(),
            value: 200,
        },
    )
    .await;
    all_frames.extend(record_seconds(&node_runner, fps, 3).await);
    let _ = camera_set_exposure::poll(
        &node_runner,
        Duration::from_secs(5),
        camera_set_exposure::Request {
            mode: "auto".to_string(),
            value: 0,
        },
    )
    .await;

    // set_white_balance: test manual/6500K, restore to auto/0
    println!("Testing set_white_balance...");
    let _ = camera_set_white_balance::poll(
        &node_runner,
        Duration::from_secs(5),
        camera_set_white_balance::Request {
            mode: "manual".to_string(),
            temperature: 6500,
        },
    )
    .await;
    all_frames.extend(record_seconds(&node_runner, fps, 3).await);
    let _ = camera_set_white_balance::poll(
        &node_runner,
        Duration::from_secs(5),
        camera_set_white_balance::Request {
            mode: "auto".to_string(),
            temperature: 0,
        },
    )
    .await;

    // set_gain: test 100, restore to 0
    println!("Testing set_gain...");
    let _ = camera_set_gain::poll(
        &node_runner,
        Duration::from_secs(5),
        camera_set_gain::Request { value: 100 },
    )
    .await;
    all_frames.extend(record_seconds(&node_runner, fps, 3).await);
    let _ = camera_set_gain::poll(
        &node_runner,
        Duration::from_secs(5),
        camera_set_gain::Request { value: 0 },
    )
    .await;

    // set_brightness: test 100, restore to 0
    println!("Testing set_brightness...");
    let _ = camera_set_brightness::poll(
        &node_runner,
        Duration::from_secs(5),
        camera_set_brightness::Request { value: 100 },
    )
    .await;
    all_frames.extend(record_seconds(&node_runner, fps, 3).await);
    let _ = camera_set_brightness::poll(
        &node_runner,
        Duration::from_secs(5),
        camera_set_brightness::Request { value: 0 },
    )
    .await;

    // set_contrast: test 100, restore to 0
    println!("Testing set_contrast...");
    let _ = camera_set_contrast::poll(
        &node_runner,
        Duration::from_secs(5),
        camera_set_contrast::Request { value: 100 },
    )
    .await;
    all_frames.extend(record_seconds(&node_runner, fps, 3).await);
    let _ = camera_set_contrast::poll(
        &node_runner,
        Duration::from_secs(5),
        camera_set_contrast::Request { value: 0 },
    )
    .await;

    println!("Recording complete. Encoding video...");

    match encode_video(&all_frames, camera_info.width, camera_info.height, fps) {
        Ok(path) => println!("Video saved to: {}", path),
        Err(e) => eprintln!("Failed to encode video: {}", e),
    }
}

async fn record_seconds(node_runner: &Arc<NodeRunner>, fps: u8, seconds: u32) -> Vec<Vec<u8>> {
    let frame_count = fps as u32 * seconds;
    let mut frames = Vec::with_capacity(frame_count as usize);
    for frame_num in 0..frame_count {
        match camera_video_stream::on_next_message_received(node_runner, None).await {
            Ok((_instance_id, message)) => {
                frames.push(message.frame);
                println!("  Frame {}/{}", frame_num + 1, frame_count);
            }
            Err(e) => {
                eprintln!("Failed to receive frame: {}", e);
            }
        }
    }
    frames
}

fn encode_video(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
    fps: u8,
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;
    let temp_path = temp_dir.keep();
    let output_path = temp_path.join("camera_controls_testing.mp4");
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
