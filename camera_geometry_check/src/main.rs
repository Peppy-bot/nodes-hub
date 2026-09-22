//! camera_geometry_check: asks a camera where its pixels point, turns one
//! depth pixel near a corner of the image into a point, and prints it for
//! someone to measure against. See README.md for the procedure.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use camera_geometry_check::{Pinhole, Pose, median_reading_m, pixel_at};
use peppygen::consumed_services::camera::{depth_stream_info, video_stream_info};
use peppygen::consumed_services::geometry::{
    get_color_intrinsics, get_depth_intrinsics, get_depth_to_color_extrinsics,
};
use peppygen::consumed_topics::camera::depth_stream;
use peppygen::{NodeBuilder, NodeRunner, Parameters, Result};

/// Bounds every service call. A camera that does not answer within it is
/// reported as such and asked again on the next report.
const SERVICE_TIMEOUT: Duration = Duration::from_secs(5);

/// The latest depth frame, kept whole: the report reads one window of it.
type LatestDepth = Arc<Mutex<Option<depth_stream::Message>>>;

fn main() -> Result<()> {
    NodeBuilder::new().run(|params: Parameters, node_runner| async move {
        let latest: LatestDepth = Arc::new(Mutex::new(None));
        tokio::spawn(follow_depth(node_runner.clone(), latest.clone()));
        tokio::spawn(report_forever(node_runner, params, latest));
        Ok(())
    })
}

async fn follow_depth(runner: Arc<NodeRunner>, latest: LatestDepth) {
    let cancel = runner.cancellation_token().clone();
    let mut subscription = match depth_stream::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => return eprintln!("depth_stream subscribe: {e}"),
    };
    loop {
        let received = tokio::select! {
            _ = cancel.cancelled() => return,
            received = subscription.next() => received,
        };
        match received {
            Ok(Some((_producer, message))) => {
                *latest.lock().unwrap_or_else(|p| p.into_inner()) = Some(message);
            }
            Ok(None) => return,
            Err(e) => {
                eprintln!("depth_stream receive: {e}");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

async fn report_forever(runner: Arc<NodeRunner>, params: Parameters, latest: LatestDepth) {
    let cancel = runner.cancellation_token().clone();
    let period = Duration::from_secs_f64(params.period_s.max(1.0));
    loop {
        println!("==== camera_geometry_check ====");
        if let Err(reason) = report(&runner, &params, &latest).await {
            println!("no point this time: {reason}");
        }
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = tokio::time::sleep(period) => {}
        }
    }
}

/// One report: the camera's answers as they came, how they agree with the
/// streams, and the point the chosen depth pixel shows.
async fn report(
    runner: &Arc<NodeRunner>,
    params: &Parameters,
    latest: &LatestDepth,
) -> std::result::Result<(), String> {
    let color_info = video_stream_info::poll(
        runner,
        video_stream_info::bound_producer(runner),
        SERVICE_TIMEOUT,
    )
    .await
    .map_err(|e| format!("video_stream_info: {e}"))?
    .data;
    let depth_info = depth_stream_info::poll(
        runner,
        depth_stream_info::bound_producer(runner),
        SERVICE_TIMEOUT,
    )
    .await
    .map_err(|e| format!("depth_stream_info: {e}"))?
    .data;
    let color = get_color_intrinsics::poll(
        runner,
        get_color_intrinsics::bound_producer(runner),
        SERVICE_TIMEOUT,
    )
    .await
    .map_err(|e| format!("get_color_intrinsics: {e}"))?
    .data;
    let depth = get_depth_intrinsics::poll(
        runner,
        get_depth_intrinsics::bound_producer(runner),
        SERVICE_TIMEOUT,
    )
    .await
    .map_err(|e| format!("get_depth_intrinsics: {e}"))?
    .data;
    let extrinsics = get_depth_to_color_extrinsics::poll(
        runner,
        get_depth_to_color_extrinsics::bound_producer(runner),
        SERVICE_TIMEOUT,
    )
    .await
    .map_err(|e| format!("get_depth_to_color_extrinsics: {e}"))?
    .data;

    // The answers as they came, to read beside what the device's own tools
    // print (README.md says which).
    println!(
        "get_color_intrinsics          success {} ({}) | {}x{} fx {} fy {} cx {} cy {} | {} {:?}",
        color.success,
        color.message,
        color.width,
        color.height,
        color.fx,
        color.fy,
        color.cx,
        color.cy,
        color.distortion_model,
        color.distortion,
    );
    println!(
        "get_depth_intrinsics          success {} ({}) | {}x{} fx {} fy {} cx {} cy {} | {} {:?} | {} {} m to {} m | {}",
        depth.success,
        depth.message,
        depth.width,
        depth.height,
        depth.fx,
        depth.fy,
        depth.cx,
        depth.cy,
        depth.distortion_model,
        depth.distortion,
        depth.depth_model,
        depth.min_depth_m,
        depth.max_depth_m,
        depth.align_mode,
    );
    println!(
        "get_depth_to_color_extrinsics success {} ({}) | {} | t {:?} q {:?}",
        extrinsics.success,
        extrinsics.message,
        extrinsics.align_mode,
        extrinsics.depth_to_color_position,
        extrinsics.depth_to_color_orientation,
    );
    println!(
        "stream infos                  colour {}x{} | depth {}x{} unit {} m",
        color_info.width,
        color_info.height,
        depth_info.width,
        depth_info.height,
        depth_info.depth_unit,
    );

    let frame = latest
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
        .ok_or("no depth frame received yet")?;
    println!(
        "checks                        colour size = video_stream_info: {} | depth size = depth frame: {} | \
         align_mode = frame header: {} | both depth answers name one mode: {}",
        (color.width, color.height) == (color_info.width, color_info.height),
        (depth.width, depth.height) == (frame.width, frame.height),
        depth.align_mode == frame.header.align_mode,
        depth.align_mode == extrinsics.align_mode,
    );
    if !depth.success {
        return Err(format!(
            "the camera refuses its depth intrinsics: {}",
            depth.message
        ));
    }

    let pixel = (
        pixel_at(params.pixel_u_fraction, frame.width),
        pixel_at(params.pixel_v_fraction, frame.height),
    );
    let (reading_m, readings) = median_reading_m(
        &frame.frame,
        frame.width,
        frame.height,
        pixel,
        params.window,
        depth_info.depth_unit,
    )?
    .ok_or_else(|| {
        format!(
            "depth pixel ({}, {}) has no reading in its window: put the target there",
            pixel.0, pixel.1
        )
    })?;
    let depth_camera = Pinhole {
        width: depth.width,
        height: depth.height,
        fx: depth.fx,
        fy: depth.fy,
        cx: depth.cx,
        cy: depth.cy,
        distortion_model: depth.distortion_model,
        distortion: depth.distortion,
    };
    let in_depth_frame = depth_camera.deproject(
        f64::from(pixel.0),
        f64::from(pixel.1),
        reading_m,
        &depth.depth_model,
    )?;
    println!(
        "depth pixel ({}, {}) of frame {} reads {:.4} m (median of {} readings)",
        pixel.0, pixel.1, frame.header.frame_id, reading_m, readings,
    );
    println!(
        "  in the depth optical frame : X {:+.4} m (right)  Y {:+.4} m (down)  Z {:+.4} m (forward)",
        in_depth_frame[0], in_depth_frame[1], in_depth_frame[2],
    );

    if !(color.success && extrinsics.success) {
        println!(
            "  in the colour frame        : not answered (colour: {} | extrinsics: {})",
            color.message, extrinsics.message
        );
        return Ok(());
    }
    let in_color_frame = Pose {
        position: extrinsics.depth_to_color_position,
        orientation: extrinsics.depth_to_color_orientation,
    }
    .apply(in_depth_frame);
    let color_camera = Pinhole {
        width: color.width,
        height: color.height,
        fx: color.fx,
        fy: color.fy,
        cx: color.cx,
        cy: color.cy,
        distortion_model: color.distortion_model,
        distortion: color.distortion,
    };
    let (u, v) = color_camera.project(in_color_frame)?;
    println!(
        "  in the colour optical frame: X {:+.4} m (right)  Y {:+.4} m (down)  Z {:+.4} m (forward)",
        in_color_frame[0], in_color_frame[1], in_color_frame[2],
    );
    println!(
        "  it shows in the colour image at pixel ({u:.1}, {v:.1}): the target must be what that pixel shows"
    );
    Ok(())
}
