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

/// One camera_geometry answer: the response, or the error reply the camera
/// gave, as `kind: message`.
type Answer<T> = std::result::Result<T, String>;

/// The answer of one camera_geometry call to `service`: a response or an
/// error reply, read by `reply_of`, the `Failure::of` of the service. A call
/// that got no answer at all is the error, naming the service.
fn answer_of<T, F: std::fmt::Display>(
    service: &str,
    polled: Result<T>,
    reply_of: impl Fn(&peppygen::Error) -> Option<F>,
) -> std::result::Result<Answer<T>, String> {
    match polled {
        Ok(response) => Ok(Ok(response)),
        Err(e) => match reply_of(&e) {
            Some(failure) => Ok(Err(failure.to_string())),
            None => Err(format!("{service}: {e}")),
        },
    }
}

/// A check of the `checks` line: its verdict, or `not answered` when an
/// answer it reads is an error reply.
fn verdict(check: Option<bool>) -> String {
    check.map_or_else(|| "not answered".to_string(), |holds| holds.to_string())
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
    let color = answer_of(
        "get_color_intrinsics",
        get_color_intrinsics::poll(
            runner,
            get_color_intrinsics::bound_producer(runner),
            SERVICE_TIMEOUT,
        )
        .await
        .map(|response| response.data),
        get_color_intrinsics::Failure::of,
    )?;
    let depth = answer_of(
        "get_depth_intrinsics",
        get_depth_intrinsics::poll(
            runner,
            get_depth_intrinsics::bound_producer(runner),
            SERVICE_TIMEOUT,
        )
        .await
        .map(|response| response.data),
        get_depth_intrinsics::Failure::of,
    )?;
    let extrinsics = answer_of(
        "get_depth_to_color_extrinsics",
        get_depth_to_color_extrinsics::poll(
            runner,
            get_depth_to_color_extrinsics::bound_producer(runner),
            SERVICE_TIMEOUT,
        )
        .await
        .map(|response| response.data),
        get_depth_to_color_extrinsics::Failure::of,
    )?;

    // The answers as they came, to read beside what the device's own tools
    // print (README.md says which).
    match &color {
        Ok(color) => println!(
            "get_color_intrinsics          {}x{} fx {} fy {} cx {} cy {} | {} {:?}",
            color.width,
            color.height,
            color.fx,
            color.fy,
            color.cx,
            color.cy,
            color.distortion_model,
            color.distortion,
        ),
        Err(reply) => println!("get_color_intrinsics          error reply: {reply}"),
    }
    match &depth {
        Ok(depth) => println!(
            "get_depth_intrinsics          {}x{} fx {} fy {} cx {} cy {} | {} {:?} | {} {} m to {} m | {}",
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
        ),
        Err(reply) => println!("get_depth_intrinsics          error reply: {reply}"),
    }
    match &extrinsics {
        Ok(extrinsics) => println!(
            "get_depth_to_color_extrinsics {} | t {:?} q {:?}",
            extrinsics.align_mode,
            extrinsics.depth_to_color_position,
            extrinsics.depth_to_color_orientation,
        ),
        Err(reply) => println!("get_depth_to_color_extrinsics error reply: {reply}"),
    }
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
    let answered_depth = depth.as_ref().ok();
    println!(
        "checks                        colour size = video_stream_info: {} | depth size = depth frame: {} | \
         align_mode = frame header: {} | both depth answers name one mode: {}",
        verdict(
            color
                .as_ref()
                .ok()
                .map(|color| (color.width, color.height) == (color_info.width, color_info.height))
        ),
        verdict(
            answered_depth.map(|depth| (depth.width, depth.height) == (frame.width, frame.height))
        ),
        verdict(answered_depth.map(|depth| depth.align_mode == frame.header.align_mode)),
        verdict(
            answered_depth
                .zip(extrinsics.as_ref().ok())
                .map(|(depth, extrinsics)| depth.align_mode == extrinsics.align_mode)
        ),
    );
    let depth =
        depth.map_err(|reply| format!("the camera refuses its depth intrinsics: {reply}"))?;

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

    let (color, extrinsics) = match (color, extrinsics) {
        (Ok(color), Ok(extrinsics)) => (color, extrinsics),
        (color, extrinsics) => {
            let answered = "answered".to_string();
            println!(
                "  in the colour frame        : not answered (colour: {} | extrinsics: {})",
                color.err().unwrap_or_else(|| answered.clone()),
                extrinsics.err().unwrap_or(answered),
            );
            return Ok(());
        }
    };
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
