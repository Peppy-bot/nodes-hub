//! The camera: its color and depth frames, drawn at most at the image rate, and its depth
//! frames turned into points at most at the cloud rate. The geometry of the camera and its
//! frame on the robot are asked when a frame first needs them, and asked again when an ask
//! fails, so a camera that answers late is drawn once it answers. The robot's tree places
//! the camera at the time of each frame turned into points.

use std::num::NonZeroU32;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use camera_core::pinhole::{DepthModel, Intrinsics, Pinhole};
use camera_core::pose::Pose;
use camera_core::sensor_mounts::{CameraFrames, DescribedSensors};
use control_core::latched_log::LatchedLog;
use control_core::positive_finite::PositiveFinite;
use control_core::receiving;
use control_core::robot_tree::RobotTree;
use control_core::throttle::Throttle;
use frame_tree::{LookupTime, Stamp};
use peppygen::NodeRunner;
use peppygen::consumed_services::camera::depth_stream_info;
use peppygen::consumed_services::geometry::{get_depth_intrinsics, get_depth_to_color_extrinsics};
use peppygen::consumed_services::sensor_mounts::describe_sensors;
use peppygen::consumed_topics::camera::{depth_stream, video_stream};
use tokio::sync::watch;
use tracing::{error, info};

use crate::canvas::Canvas;
use crate::draw::camera::{draw_cloud, draw_color_image, draw_depth_image};
use crate::model::camera::{ColorImage, DepthCamera, DepthImage, strided_points};
use crate::wire::{SERVICE_TIMEOUT, sendable, stopped};

/// What the depth follower draws, and how often.
#[derive(Debug, Clone)]
pub struct DepthSettings {
    pub image_interval: Duration,
    /// Whether the depth image is drawn; the points are drawn either way.
    pub depth_image: bool,
    pub cloud_interval: Duration,
    pub cloud_stride: NonZeroU32,
    /// The camera of sensor_mounts that the camera link is.
    pub camera_name: String,
    /// How long a frame turned into points waits for the tree to place the camera at its
    /// time.
    pub lookup_timeout: Duration,
}

/// An answer of the camera's services that the viewer cannot use.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CameraError {
    #[error("depth_stream_info: depth_unit {0} is not a positive number of meters")]
    DepthUnit(f32),
    #[error("get_depth_intrinsics: {0}")]
    Intrinsics(String),
    #[error("get_depth_to_color_extrinsics: {0}")]
    Extrinsics(String),
    #[error(
        "get_depth_intrinsics describes align_mode {intrinsics:?} and \
         get_depth_to_color_extrinsics {extrinsics:?}"
    )]
    AlignModes {
        intrinsics: String,
        extrinsics: String,
    },
}

/// The meters a count of the depth stream is.
pub fn parse_depth_unit(
    info: &depth_stream_info::ResponseData,
) -> Result<PositiveFinite, CameraError> {
    PositiveFinite::parse(f64::from(info.depth_unit))
        .map_err(|_| CameraError::DepthUnit(info.depth_unit))
}

/// The depth stream as `intrinsics` and `extrinsics` describe it.
pub fn parse_depth_camera(
    intrinsics: &get_depth_intrinsics::ResponseData,
    extrinsics: &get_depth_to_color_extrinsics::ResponseData,
) -> Result<DepthCamera, CameraError> {
    if intrinsics.align_mode != extrinsics.align_mode {
        return Err(CameraError::AlignModes {
            intrinsics: intrinsics.align_mode.clone(),
            extrinsics: extrinsics.align_mode.clone(),
        });
    }
    let lens = Pinhole::parse(&Intrinsics {
        width: intrinsics.width,
        height: intrinsics.height,
        fx: intrinsics.fx,
        fy: intrinsics.fy,
        cx: intrinsics.cx,
        cy: intrinsics.cy,
        distortion_model: &intrinsics.distortion_model,
        distortion: &intrinsics.distortion,
    })
    .map_err(|e| CameraError::Intrinsics(e.to_string()))?;
    let depth_model = DepthModel::parse(&intrinsics.depth_model)
        .map_err(|e| CameraError::Intrinsics(e.to_string()))?;
    let depth_in_color = Pose::new(
        extrinsics.depth_to_color_position,
        extrinsics.depth_to_color_orientation,
    )
    .map_err(|e| CameraError::Extrinsics(e.to_string()))?;
    Ok(DepthCamera {
        lens,
        depth_model,
        depth_in_color,
        align_mode: intrinsics.align_mode.clone(),
    })
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
            message.frame,
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
/// to place the camera at the frame's time, up to the lookup timeout, and a newer frame
/// admitted for points takes the place of one that waits. With no tree, or with the
/// geometry or the sensor_mounts link vacant, the viewer draws no points.
pub async fn follow_depth(
    runner: Arc<NodeRunner>,
    canvas: Arc<dyn Canvas>,
    settings: DepthSettings,
    tree: Option<Arc<RobotTree>>,
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
                frames: None,
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

/// A depth frame admitted for points: its image, its alignment and when it was taken.
struct CloudFrame {
    image: Arc<DepthImage>,
    align_mode: String,
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
            message.frame,
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
/// learns what it does not know of the camera, waits for the tree to place the camera at
/// the frame's time, and draws the points. A newer frame that comes while it waits takes
/// the place of the one that waits.
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
            ) => placed.map_err(|e| e.to_string()),
        };
        match placed.and_then(|found| maker.draw(&frame, &found.placement)) {
            Ok(()) => maker.failures.success(),
            Err(reason) => maker.failures.failure(reason),
        }
    }
}

/// What the cloud maker has learned of the camera so far.
struct CloudMaker {
    runner: Arc<NodeRunner>,
    canvas: Arc<dyn Canvas>,
    tree: Arc<RobotTree>,
    settings: DepthSettings,
    camera: Option<DepthCamera>,
    /// The robot frame and the camera's frame: asked once, and again after a failed ask.
    frames: Option<CameraFrames>,
    failures: LatchedLog,
}

impl CloudMaker {
    /// The robot frame and the camera's frame, asking first what it does not know of the
    /// camera for `frame`: its geometry, when not known yet or when the frames changed
    /// their alignment or their size, and its frames.
    async fn learn(&mut self, frame: &CloudFrame) -> Result<CameraFrames, String> {
        let stale = self.camera.as_ref().is_none_or(|camera| {
            camera.align_mode != frame.align_mode || !camera.fits(&frame.image)
        });
        if stale {
            let asked = self.ask_depth_camera().await?;
            if asked.align_mode != frame.align_mode {
                return Err(format!(
                    "the depth frames carry align_mode {:?}, the geometry describes {:?}",
                    frame.align_mode, asked.align_mode
                ));
            }
            self.camera = Some(asked);
        }
        if let Some(frames) = &self.frames {
            return Ok(frames.clone());
        }
        let frames = self.ask_camera_frames().await?;
        self.frames = Some(frames.clone());
        Ok(frames)
    }

    /// Draws the points of `frame`, the camera's color optical frame standing at
    /// `placement` in the robot frame.
    fn draw(&self, frame: &CloudFrame, placement: &frame_tree::Placement) -> Result<(), String> {
        let camera = self
            .camera
            .as_ref()
            .ok_or("the camera's geometry is not known")?;
        let color_in_robot = Pose::new(placement.position(), placement.orientation())
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

    async fn ask_camera_frames(&self) -> Result<CameraFrames, String> {
        let mounts = describe_sensors::bound_producer(&self.runner)
            .ok_or("the sensor_mounts link is vacant")?;
        let sensors = sendable(describe_sensors::poll(
            &self.runner,
            mounts,
            SERVICE_TIMEOUT,
        ))
        .await
        .map_err(|e| format!("describe_sensors: {e}"))?;
        let sensors = &sensors.data;
        let described = DescribedSensors {
            robot_frame: &sensors.robot_frame,
            sensor_names: &sensors.sensor_names,
            sensor_kinds: &sensors.sensor_kinds,
            sensor_frames: &sensors.sensor_frames,
            carried_by: &sensors.carried_by,
        };
        CameraFrames::of(&described, &self.settings.camera_name)
            .map_err(|e| format!("describe_sensors: {e}"))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn intrinsics() -> get_depth_intrinsics::ResponseData {
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

    pub(crate) fn extrinsics() -> get_depth_to_color_extrinsics::ResponseData {
        get_depth_to_color_extrinsics::ResponseData {
            align_mode: "depth_to_color".to_string(),
            depth_to_color_position: [0.0; 3],
            depth_to_color_orientation: [0.0, 0.0, 0.0, 1.0],
        }
    }

    #[test]
    fn the_geometry_makes_the_depth_camera() {
        let camera = parse_depth_camera(&intrinsics(), &extrinsics()).expect("valid");
        assert_eq!(camera.lens.width(), 4);
        assert_eq!(camera.depth_model, DepthModel::Z);
        assert_eq!(camera.align_mode, "depth_to_color");
    }

    #[test]
    fn geometry_the_viewer_cannot_use_is_refused() {
        let mut other = extrinsics();
        other.align_mode = "none".to_string();
        assert!(matches!(
            parse_depth_camera(&intrinsics(), &other),
            Err(CameraError::AlignModes { .. })
        ));
        let mut flat = intrinsics();
        flat.fx = 0.0;
        assert!(matches!(
            parse_depth_camera(&flat, &extrinsics()),
            Err(CameraError::Intrinsics(_))
        ));
        let mut unknown = intrinsics();
        unknown.depth_model = "disparity".to_string();
        assert!(matches!(
            parse_depth_camera(&unknown, &extrinsics()),
            Err(CameraError::Intrinsics(_))
        ));
        let mut long = extrinsics();
        long.depth_to_color_orientation = [0.0, 0.0, 0.0, 2.0];
        assert!(matches!(
            parse_depth_camera(&intrinsics(), &long),
            Err(CameraError::Extrinsics(_))
        ));
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
        assert_eq!(parse_depth_unit(&info), Err(CameraError::DepthUnit(0.0)));
    }
}
