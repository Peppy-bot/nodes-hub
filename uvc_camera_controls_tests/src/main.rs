use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use peppygen::consumed_services::{
    camera_set_brightness, camera_set_contrast, camera_set_exposure, camera_set_gain,
    camera_set_white_balance, camera_video_stream_info,
};
use peppygen::consumed_topics::camera_video_stream;
use peppygen::{NodeBuilder, NodeRunner, Parameters, Result};
use peppylib::runtime::CancellationToken;

use ffmpeg_next::Rational;
use ffmpeg_next::format::Pixel;
use ffmpeg_next::util::frame::video::Video as VideoFrame;

/// Timeout for the restore polls issued from shutdown hooks: all hooks share
/// one grace window (default 3s), so each call must stay well under it (the
/// 5s timeout used on the normal path would exceed the whole window).
const HOOK_RESTORE_TIMEOUT: Duration = Duration::from_secs(1);

fn main() -> Result<()> {
    ffmpeg_next::init().expect("Failed to initialize FFmpeg");

    NodeBuilder::new().run(|_args: Parameters, node_runner| async move {
        tokio::spawn(record_video(node_runner));
        Ok(())
    })
}

async fn record_video(node_runner: Arc<NodeRunner>) {
    let token = node_runner.cancellation_token();

    let camera_info = loop {
        let response = tokio::select! {
            _ = token.cancelled() => return,
            response = camera_video_stream_info::poll(&node_runner, Duration::from_secs(5)) => response,
        };
        match response {
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
                tokio::select! {
                    _ = token.cancelled() => return,
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {}
                }
            }
        }
    };

    let fps = camera_info.frames_per_second;
    let mut all_frames: Vec<Vec<u8>> = Vec::new();

    // set_exposure: test manual/200, restore to auto/0.
    // Each control registers its restore as a shutdown hook before mutating
    // it: a stop landing mid-test drops this task before the inline restore
    // runs, and only hooks are awaited by the runtime. Registration order
    // matches mutation order, so the reverse-order hooks restore the most
    // recently mutated control first.
    println!("Testing set_exposure...");
    let exposure_needs_restore = register_restore_hook(&node_runner, |runner| async move {
        restore_exposure(&runner, HOOK_RESTORE_TIMEOUT).await;
    });
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
    if token.is_cancelled() {
        return;
    }
    if restore_exposure(&node_runner, Duration::from_secs(5)).await {
        exposure_needs_restore.store(false, Ordering::SeqCst);
    }

    // set_white_balance: test manual/6500K, restore to auto/0
    println!("Testing set_white_balance...");
    let white_balance_needs_restore = register_restore_hook(&node_runner, |runner| async move {
        restore_white_balance(&runner, HOOK_RESTORE_TIMEOUT).await;
    });
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
    if token.is_cancelled() {
        return;
    }
    if restore_white_balance(&node_runner, Duration::from_secs(5)).await {
        white_balance_needs_restore.store(false, Ordering::SeqCst);
    }

    // set_gain: test 100, restore to 0
    println!("Testing set_gain...");
    let gain_needs_restore = register_restore_hook(&node_runner, |runner| async move {
        restore_gain(&runner, HOOK_RESTORE_TIMEOUT).await;
    });
    let _ = camera_set_gain::poll(
        &node_runner,
        Duration::from_secs(5),
        camera_set_gain::Request { value: 100 },
    )
    .await;
    all_frames.extend(record_seconds(&node_runner, fps, 3).await);
    if token.is_cancelled() {
        return;
    }
    if restore_gain(&node_runner, Duration::from_secs(5)).await {
        gain_needs_restore.store(false, Ordering::SeqCst);
    }

    // set_brightness: test 100, restore to 0
    println!("Testing set_brightness...");
    let brightness_needs_restore = register_restore_hook(&node_runner, |runner| async move {
        restore_brightness(&runner, HOOK_RESTORE_TIMEOUT).await;
    });
    let _ = camera_set_brightness::poll(
        &node_runner,
        Duration::from_secs(5),
        camera_set_brightness::Request { value: 100 },
    )
    .await;
    all_frames.extend(record_seconds(&node_runner, fps, 3).await);
    if token.is_cancelled() {
        return;
    }
    if restore_brightness(&node_runner, Duration::from_secs(5)).await {
        brightness_needs_restore.store(false, Ordering::SeqCst);
    }

    // set_contrast: test 100, restore to 0
    println!("Testing set_contrast...");
    let contrast_needs_restore = register_restore_hook(&node_runner, |runner| async move {
        restore_contrast(&runner, HOOK_RESTORE_TIMEOUT).await;
    });
    let _ = camera_set_contrast::poll(
        &node_runner,
        Duration::from_secs(5),
        camera_set_contrast::Request { value: 100 },
    )
    .await;
    all_frames.extend(record_seconds(&node_runner, fps, 3).await);
    if token.is_cancelled() {
        return;
    }
    if restore_contrast(&node_runner, Duration::from_secs(5)).await {
        contrast_needs_restore.store(false, Ordering::SeqCst);
    }

    if token.is_cancelled() {
        return;
    }

    println!("Recording complete. Encoding video...");

    match encode_video(&all_frames, camera_info.width, camera_info.height, fps, token) {
        Ok(path) => println!("Video saved to: {}", path),
        Err(e) => eprintln!("Failed to encode video: {}", e),
    }
}

async fn record_seconds(node_runner: &Arc<NodeRunner>, fps: u8, seconds: u32) -> Vec<Vec<u8>> {
    let token = node_runner.cancellation_token();
    let frame_count = fps as u32 * seconds;
    let mut frames = Vec::with_capacity(frame_count as usize);
    for frame_num in 0..frame_count {
        let received = tokio::select! {
            _ = token.cancelled() => break,
            received = camera_video_stream::on_next_message_received(node_runner) => received,
        };
        match received {
            Ok((_producer, message)) => {
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

/// Register a shutdown hook that restores a camera control to its default
/// unless the normal-path restore already succeeded. Returns the flag the
/// normal path clears once its own restore goes through, making the hook a
/// no-op.
fn register_restore_hook<F, Fut>(node_runner: &Arc<NodeRunner>, restore: F) -> Arc<AtomicBool>
where
    F: FnOnce(Arc<NodeRunner>) -> Fut + Send + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    let needs_restore = Arc::new(AtomicBool::new(true));
    let flag = Arc::clone(&needs_restore);
    let runner = Arc::clone(node_runner);
    node_runner.on_shutdown(async move {
        if flag.load(Ordering::SeqCst) {
            restore(runner).await;
        }
    });
    needs_restore
}

/// Each restore_* helper sets a control back to its default and returns
/// whether the service call succeeded.
async fn restore_exposure(node_runner: &NodeRunner, timeout: Duration) -> bool {
    camera_set_exposure::poll(
        node_runner,
        timeout,
        camera_set_exposure::Request {
            mode: "auto".to_string(),
            value: 0,
        },
    )
    .await
    .is_ok()
}

async fn restore_white_balance(node_runner: &NodeRunner, timeout: Duration) -> bool {
    camera_set_white_balance::poll(
        node_runner,
        timeout,
        camera_set_white_balance::Request {
            mode: "auto".to_string(),
            temperature: 0,
        },
    )
    .await
    .is_ok()
}

async fn restore_gain(node_runner: &NodeRunner, timeout: Duration) -> bool {
    camera_set_gain::poll(node_runner, timeout, camera_set_gain::Request { value: 0 })
        .await
        .is_ok()
}

async fn restore_brightness(node_runner: &NodeRunner, timeout: Duration) -> bool {
    camera_set_brightness::poll(
        node_runner,
        timeout,
        camera_set_brightness::Request { value: 0 },
    )
    .await
    .is_ok()
}

async fn restore_contrast(node_runner: &NodeRunner, timeout: Duration) -> bool {
    camera_set_contrast::poll(
        node_runner,
        timeout,
        camera_set_contrast::Request { value: 0 },
    )
    .await
    .is_ok()
}

fn encode_video(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
    fps: u8,
    token: &CancellationToken,
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
        // This encode is synchronous CPU work on a runtime worker thread, so
        // it cannot be dropped at an await point: check the token between
        // frames so a stop arriving mid-encode aborts within one frame
        // instead of blocking runtime teardown past the shutdown grace
        // window.
        if token.is_cancelled() {
            return Err("encode aborted: node is shutting down".into());
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
