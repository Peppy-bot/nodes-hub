//! The camera: its color and depth frames, drawn at most at the image rate, and its depth
//! frames turned into points at most at the cloud rate. The geometry of the camera and the
//! robot frame are asked when a frame first needs them, and asked again when an ask fails,
//! so a camera that answers late is drawn once it answers. The robot's tree places the frame
//! each depth frame names, at that frame's time, in the robot frame.

use std::num::NonZeroU32;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use camera_core::depth_camera::{
    DepthCamera, DepthCameraError, DepthExtrinsics, DepthIntrinsics, read_depth_camera,
};
use control_core::latched_log::LatchedLog;
use control_core::positive_finite::PositiveFinite;
use control_core::receiving;
use control_core::robot_tree::RobotTree;
use control_core::throttle::Throttle;
use frame_tree::{FrameName, LookupError, LookupTime, Stamp, Transform};
use peppygen::NodeRunner;
use peppygen::consumed_services::camera::depth_stream_info;
use peppygen::consumed_services::geometry::{get_depth_intrinsics, get_depth_to_color_extrinsics};
use peppygen::consumed_services::sensor_mounts::describe_sensors;
use peppygen::consumed_topics::camera::{depth_stream, video_stream};
use tokio::sync::watch;
use tracing::{error, info};

use crate::canvas::Canvas;
use crate::draw::camera::{draw_cloud, draw_color_image, draw_depth_image};
use crate::model::camera::{ColorImage, DepthImage, strided_points};
use crate::wire::{SERVICE_TIMEOUT, Shared, sendable, stopped};

/// What the depth follower draws, and how often.
#[derive(Debug, Clone)]
pub struct DepthSettings {
    pub image_interval: Duration,
    /// Whether the depth image is drawn; the points are drawn either way.
    pub depth_image: bool,
    pub cloud_interval: Duration,
    pub cloud_stride: NonZeroU32,
    /// How long a frame turned into points waits for the tree to place the frame it names
    /// at its time.
    pub lookup_timeout: Duration,
}

/// A depth unit of `depth_stream_info` that the viewer cannot scale samples by.
#[derive(Debug, thiserror::Error, PartialEq)]
#[error("depth_stream_info: depth_unit {0} is not a positive number of meters")]
pub struct DepthUnitError(pub f32);

/// The meters a count of the depth stream is.
pub fn parse_depth_unit(
    info: &depth_stream_info::ResponseData,
) -> Result<PositiveFinite, DepthUnitError> {
    PositiveFinite::parse(f64::from(info.depth_unit)).map_err(|_| DepthUnitError(info.depth_unit))
}

/// The depth camera `intrinsics` and `extrinsics` describe, read by camera_core.
fn parse_depth_camera(
    intrinsics: &get_depth_intrinsics::ResponseData,
    extrinsics: &get_depth_to_color_extrinsics::ResponseData,
) -> Result<DepthCamera, DepthCameraError> {
    let depth_to_color = &extrinsics.depth_to_color;
    read_depth_camera(
        &DepthIntrinsics {
            width: intrinsics.width,
            height: intrinsics.height,
            fx: intrinsics.fx,
            fy: intrinsics.fy,
            cx: intrinsics.cx,
            cy: intrinsics.cy,
            distortion_model: &intrinsics.distortion_model,
            distortion: &intrinsics.distortion,
            depth_model: &intrinsics.depth_model,
            align_mode: &intrinsics.align_mode,
        },
        &DepthExtrinsics {
            frame: &depth_to_color.frame,
            position: depth_to_color.position,
            orientation: depth_to_color.orientation,
            align_mode: &extrinsics.align_mode,
        },
    )
}

/// Draws the color frames, at most one each image interval, each shrunk to at most
/// `max_width` pixels wide (`ColorImage::shrunk_to`), until the node stops.
pub async fn follow_color(
    runner: Arc<NodeRunner>,
    canvas: Arc<dyn Canvas>,
    image_interval: Duration,
    max_width: NonZeroU32,
) {
    let cancelled = stopped(&runner);
    let mut subscription = match sendable(video_stream::subscribe(&runner)).await {
        Ok(subscription) => subscription,
        Err(e) => {
            error!("video_stream: no subscription, no color image drawn: {e}");
            return;
        }
    };
    let mut follower = ColorFollower {
        canvas,
        throttle: Throttle::new(image_interval),
        max_width,
        unreadable: LatchedLog::warn("reading video_stream"),
    };
    receiving::receive(
        "video_stream",
        cancelled,
        async move || subscription.next().await,
        async move |(_producer, message)| {
            follower.on_frame(message);
            ControlFlow::Continue(())
        },
    )
    .await;
}

/// The color follower's state: when it last drew.
struct ColorFollower {
    canvas: Arc<dyn Canvas>,
    throttle: Throttle,
    max_width: NonZeroU32,
    unreadable: LatchedLog,
}

impl ColorFollower {
    fn on_frame(&mut self, message: video_stream::Message) {
        if !self.throttle.admit_at(Instant::now()) {
            return;
        }
        let at = message.header.timestamp;
        match ColorImage::parse(
            &message.encoding,
            message.width,
            message.height,
            message.data,
        ) {
            Ok(image) => {
                self.unreadable.success();
                let image = image.shrunk_to(self.max_width);
                draw_color_image(self.canvas.as_ref(), at, &image);
            }
            Err(e) => self.unreadable.failure(e),
        }
    }
}

/// Draws the depth frames, at most one each image interval, and their points, at most
/// once each cloud interval, until the node stops. The points of a frame wait for `tree`
/// to place the frame it names at its time, up to the lookup timeout, and a newer frame
/// admitted for points takes the place of one that waits. With no tree, or with the
/// geometry or the sensor_mounts link vacant, the viewer draws no points.
pub async fn follow_depth(
    runner: Arc<NodeRunner>,
    canvas: Arc<dyn Canvas>,
    settings: DepthSettings,
    tree: Option<Arc<RobotTree>>,
    shared: Arc<Shared>,
) {
    let cancelled = stopped(&runner);
    let mut subscription = match sendable(depth_stream::subscribe(&runner)).await {
        Ok(subscription) => subscription,
        Err(e) => {
            error!("depth_stream: no subscription, no depth drawn: {e}");
            return;
        }
    };
    let linked = get_depth_intrinsics::bound_producer(&runner).is_some()
        && describe_sensors::bound_producer(&runner).is_some();
    let clouds = match tree {
        Some(tree) if linked => {
            let (clouds, waiting) = watch::channel(None);
            let maker = CloudMaker {
                runner: runner.clone(),
                canvas: canvas.clone(),
                tree,
                settings: settings.clone(),
                camera: None,
                shared,
                failures: LatchedLog::warn("drawing the depth cloud"),
            };
            tokio::spawn(make_clouds(maker, waiting));
            Some(clouds)
        }
        _ => {
            info!(
                "the geometry, the sensor_mounts or the transforms link is vacant: no points drawn"
            );
            None
        }
    };
    let mut follower = DepthFollower {
        image_throttle: Throttle::new(settings.image_interval),
        cloud_throttle: Throttle::new(settings.cloud_interval),
        runner,
        canvas,
        settings,
        clouds,
        depth_unit: None,
        unreadable: LatchedLog::warn("reading depth_stream"),
        unit_failures: LatchedLog::warn("asking depth_stream_info"),
    };
    receiving::receive(
        "depth_stream",
        cancelled,
        async move || subscription.next().await,
        async move |(_producer, message)| {
            follower.on_frame(message).await;
            ControlFlow::Continue(())
        },
    )
    .await;
}

/// A depth frame admitted for points: its image, its alignment, the frame it names and when
/// it was taken.
struct CloudFrame {
    image: Arc<DepthImage>,
    align_mode: String,
    frame: String,
    at: SystemTime,
}

/// The depth follower's state: what it has learned of the depth stream so far.
struct DepthFollower {
    runner: Arc<NodeRunner>,
    canvas: Arc<dyn Canvas>,
    settings: DepthSettings,
    image_throttle: Throttle,
    cloud_throttle: Throttle,
    /// Where the frames admitted for points go, the newest taking the place of one that
    /// waits; none when the viewer draws no points.
    clouds: Option<watch::Sender<Option<Arc<CloudFrame>>>>,
    depth_unit: Option<PositiveFinite>,
    unreadable: LatchedLog,
    unit_failures: LatchedLog,
}

impl DepthFollower {
    async fn on_frame(&mut self, message: depth_stream::Message) {
        let now = Instant::now();
        let draw_image = self.settings.depth_image && self.image_throttle.admit_at(now);
        let make_cloud = self.clouds.is_some() && self.cloud_throttle.admit_at(now);
        if !draw_image && !make_cloud {
            return;
        }
        let depth_unit = match self.known_depth_unit().await {
            Ok(unit) => {
                self.unit_failures.success();
                unit
            }
            Err(reason) => {
                self.unit_failures.failure(reason);
                return;
            }
        };
        let at = message.header.timestamp;
        let parsed = DepthImage::parse(
            &message.encoding,
            message.width,
            message.height,
            message.data,
            depth_unit,
        );
        let image = match parsed {
            Ok(image) => {
                self.unreadable.success();
                image
            }
            Err(e) => {
                self.unreadable.failure(e);
                return;
            }
        };
        // The frame goes to the cloud maker before its image is drawn, so a drawn image
        // tells that its frame was handed over.
        let image = Arc::new(image);
        if let Some(clouds) = self.clouds.as_ref().filter(|_| make_cloud) {
            clouds.send_replace(Some(Arc::new(CloudFrame {
                image: image.clone(),
                align_mode: message.header.align_mode,
                frame: message.header.frame,
                at,
            })));
        }
        if draw_image {
            draw_depth_image(self.canvas.as_ref(), at, &image);
        }
    }

    /// The meters a count of the depth stream is: asked once, and again after a failed
    /// ask.
    async fn known_depth_unit(&mut self) -> Result<PositiveFinite, String> {
        if let Some(unit) = self.depth_unit {
            return Ok(unit);
        }
        let unit = self.ask_depth_unit().await?;
        self.depth_unit = Some(unit);
        Ok(unit)
    }

    async fn ask_depth_unit(&self) -> Result<PositiveFinite, String> {
        let camera =
            depth_stream_info::bound_producer(&self.runner).ok_or("the camera link is vacant")?;
        let info = sendable(depth_stream_info::poll(
            &self.runner,
            camera,
            SERVICE_TIMEOUT,
        ))
        .await
        .map_err(|e| format!("depth_stream_info: {e}"))?;
        parse_depth_unit(&info.data).map_err(|e| e.to_string())
    }
}

/// Turns the frames `waiting` hands it into points until the node stops: for each, it
/// learns what it does not know of the camera, waits for the tree to place the frame it
/// names at its time, and draws the points. A frame whose place the tree does not give is
/// dropped, the failure logged by its kind as transform_lookup names it. A newer frame that
/// comes while it waits takes the place of the one that waits.
async fn make_clouds(mut maker: CloudMaker, mut waiting: watch::Receiver<Option<Arc<CloudFrame>>>) {
    let mut cancelled = std::pin::pin!(stopped(&maker.runner));
    let mut newer: Option<Arc<CloudFrame>> = None;
    loop {
        let frame = match newer.take() {
            Some(frame) => frame,
            None => {
                tokio::select! {
                    biased;
                    () = &mut cancelled => return,
                    changed = waiting.changed() => {
                        if changed.is_err() {
                            return;
                        }
                    }
                }
                let Some(frame) = waiting.borrow_and_update().clone() else {
                    continue;
                };
                frame
            }
        };
        let learned = tokio::select! {
            biased;
            () = &mut cancelled => return,
            learned = maker.learn(&frame) => learned,
        };
        let frames = match learned {
            Ok(frames) => frames,
            Err(reason) => {
                maker.failures.failure(reason);
                continue;
            }
        };
        let time = match Stamp::from_time(frame.at) {
            Ok(stamp) => LookupTime::At(stamp),
            Err(e) => {
                maker
                    .failures
                    .failure(format!("the depth frame's time: {e}"));
                continue;
            }
        };
        let placed = tokio::select! {
            biased;
            () = &mut cancelled => return,
            changed = waiting.changed() => {
                if changed.is_err() {
                    return;
                }
                newer = waiting.borrow_and_update().clone();
                continue;
            }
            placed = maker.tree.lookup_within(
                &frames.robot_frame,
                &frames.camera_frame,
                time,
                maker.settings.lookup_timeout,
            ) => placed.map_err(|e| failure_of(&e)),
        };
        match placed.and_then(|found| maker.draw(&frame, &found.placement)) {
            Ok(()) => maker.failures.success(),
            Err(reason) => maker.failures.failure(reason),
        }
    }
}

/// The failure of a lookup, by its kind as transform_lookup names it.
fn failure_of(error: &LookupError) -> String {
    format!("{}: {error}", error.failure_kind())
}

/// The frames a depth frame's points are placed between: the frame it names, the camera's,
/// in the robot frame.
struct CloudFrames {
    robot_frame: FrameName,
    camera_frame: FrameName,
}

/// What the cloud maker has learned of the camera so far.
struct CloudMaker {
    runner: Arc<NodeRunner>,
    canvas: Arc<dyn Canvas>,
    tree: Arc<RobotTree>,
    settings: DepthSettings,
    camera: Option<DepthCamera>,
    /// What the followers share, the robot frame among it.
    shared: Arc<Shared>,
    failures: LatchedLog,
}

impl CloudMaker {
    /// The frames `depth`'s points are placed between, asking first what it does not know of
    /// the camera: its geometry, when not known yet or when the frames changed their
    /// alignment, their size or the frame they name; refused while the robot frame is
    /// unknown.
    async fn learn(&mut self, depth: &CloudFrame) -> Result<CloudFrames, String> {
        let stale = self.camera.as_ref().is_none_or(|camera| {
            camera
                .admits_frame(&depth.align_mode, &depth.frame)
                .is_err()
                || !depth.image.fits(&camera.lens)
        });
        if stale {
            let asked = self.ask_depth_camera().await?;
            asked
                .admits_frame(&depth.align_mode, &depth.frame)
                .map_err(|e| e.to_string())?;
            self.camera = Some(asked);
        }
        let camera_frame = self
            .camera
            .as_ref()
            .map(|camera| camera.frame.clone())
            .ok_or("the camera's geometry is not known")?;
        let robot_frame = self.shared.robot_frame(&self.runner)?;
        Ok(CloudFrames {
            robot_frame,
            camera_frame,
        })
    }

    /// Draws the points of `frame`, the frame it names standing at `placement` in the
    /// robot frame.
    fn draw(&self, frame: &CloudFrame, placement: &frame_tree::Placement) -> Result<(), String> {
        let camera = self
            .camera
            .as_ref()
            .ok_or("the camera's geometry is not known")?;
        let color_in_robot = Transform::new(placement.position(), placement.orientation())
            .map_err(|e| format!("the camera's place in the robot frame: {e}"))?;
        let depth_in_robot = camera.depth_in_robot(&color_in_robot);
        let points = strided_points(
            &frame.image,
            camera,
            &depth_in_robot,
            self.settings.cloud_stride,
        )
        .map_err(|e| e.to_string())?;
        draw_cloud(self.canvas.as_ref(), frame.at, &points);
        Ok(())
    }

    async fn ask_depth_camera(&self) -> Result<DepthCamera, String> {
        let geometry = get_depth_intrinsics::bound_producer(&self.runner)
            .ok_or("the geometry link is vacant")?;
        let intrinsics = sendable(get_depth_intrinsics::poll(
            &self.runner,
            geometry,
            SERVICE_TIMEOUT,
        ))
        .await
        .map_err(|e| format!("get_depth_intrinsics: {e}"))?;
        let extrinsics = sendable(get_depth_to_color_extrinsics::poll(
            &self.runner,
            geometry,
            SERVICE_TIMEOUT,
        ))
        .await
        .map_err(|e| format!("get_depth_to_color_extrinsics: {e}"))?;
        parse_depth_camera(&intrinsics.data, &extrinsics.data).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::placement::tests::close;
    use camera_core::pinhole::{DepthModel, Intrinsics, Pinhole};

    fn intrinsics() -> get_depth_intrinsics::ResponseData {
        get_depth_intrinsics::ResponseData {
            width: 4,
            height: 3,
            fx: 2.0,
            fy: 2.0,
            cx: 1.5,
            cy: 1.0,
            distortion_model: "none".to_string(),
            distortion: vec![],
            depth_model: "z".to_string(),
            min_depth_m: 0.1,
            max_depth_m: 10.0,
            align_mode: "depth_to_color".to_string(),
        }
    }

    fn extrinsics() -> get_depth_to_color_extrinsics::ResponseData {
        get_depth_to_color_extrinsics::ResponseData {
            align_mode: "depth_to_color".to_string(),
            depth_to_color: get_depth_to_color_extrinsics::ResponseDataDepthToColor {
                frame: "chest_color_optical_frame".to_string(),
                position: [0.0; 3],
                orientation: [0.0, 0.0, 0.0, 1.0],
            },
        }
    }

    // Each field of the two answers reaches the depth camera: a lens whose focal lengths and
    // principal point differ on each axis, the depth model, the frame, the alignment, and
    // the depth optical frame 0.1 m along the color one's x.
    #[test]
    fn the_geometry_makes_the_depth_camera_in_the_frame_the_extrinsics_name() {
        let lens = get_depth_intrinsics::ResponseData {
            fy: 3.0,
            cy: 1.25,
            ..intrinsics()
        };
        let mut offset = extrinsics();
        offset.depth_to_color.position = [0.1, 0.0, 0.0];
        let camera = parse_depth_camera(&lens, &offset).expect("valid");
        let expected = Pinhole::parse(&Intrinsics {
            width: 4,
            height: 3,
            fx: 2.0,
            fy: 3.0,
            cx: 1.5,
            cy: 1.25,
            distortion_model: "none",
            distortion: &[],
        })
        .expect("a valid model");
        assert_eq!(camera.lens, expected);
        assert_eq!(camera.depth_model, DepthModel::Z);
        assert_eq!(camera.frame.as_str(), "chest_color_optical_frame");
        assert_eq!(camera.align_mode, "depth_to_color");
        assert!(close(
            camera.depth_in_color.apply([0.0; 3]),
            [0.1, 0.0, 0.0],
            1e-12
        ));
    }

    // Each answer's align_mode reaches its own side of camera_core's refusal of two
    // alignments.
    #[test]
    fn geometry_of_two_alignments_is_refused_naming_each_answers() {
        let mut other = extrinsics();
        other.align_mode = "none".to_string();
        let refused = parse_depth_camera(&intrinsics(), &other).expect_err("two alignments");
        assert!(
            matches!(
                &refused,
                DepthCameraError::AlignModes { intrinsics, extrinsics }
                    if intrinsics == "depth_to_color" && extrinsics == "none"
            ),
            "{refused}"
        );
    }

    #[test]
    fn a_depth_unit_of_zero_is_refused() {
        let info = depth_stream_info::ResponseData {
            width: 4,
            height: 3,
            frames_per_second: 15,
            encoding: "z16".to_string(),
            depth_unit: 0.0,
        };
        assert_eq!(parse_depth_unit(&info), Err(DepthUnitError(0.0)));
    }
}
