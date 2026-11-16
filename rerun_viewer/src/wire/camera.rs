//! The camera: its color and depth frames, drawn at most at the image rate, and its depth
//! frames turned into points at most at the cloud rate. The geometry and the
//! pose of the camera are asked when a frame first needs them, and asked again when an
//! ask fails, so a camera that answers late is drawn once it answers.

use std::num::NonZeroU32;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use camera_core::pinhole::{DepthModel, Intrinsics, Pinhole};
use camera_core::pose::Pose;
use control_core::latched_log::LatchedLog;
use control_core::positive_finite::PositiveFinite;
use control_core::receiving;
use control_core::throttle::Throttle;
use peppygen::NodeRunner;
use peppygen::consumed_services::camera::depth_stream_info;
use peppygen::consumed_services::camera_mounts::get_camera_poses;
use peppygen::consumed_services::geometry::{get_depth_intrinsics, get_depth_to_color_extrinsics};
use peppygen::consumed_topics::camera::{depth_stream, video_stream};
use tracing::{error, info};

use crate::canvas::Canvas;
use crate::draw::camera::{draw_cloud, draw_color_image, draw_depth_image};
use crate::model::camera::{ColorImage, DepthCamera, DepthImage, strided_points};
use crate::model::placement::Placement;
use crate::wire::{SERVICE_TIMEOUT, sendable, stopped};

/// What the depth follower draws, and how often.
#[derive(Debug, Clone)]
pub struct DepthSettings {
    pub image_interval: Duration,
    /// Whether the depth image is drawn; the points are drawn either way.
    pub depth_image: bool,
    pub cloud_interval: Duration,
    pub cloud_stride: NonZeroU32,
    /// The camera of camera_mounts that the camera link is.
    pub camera_name: String,
}

/// Where camera_mounts places the camera: its color optical frame in the robot frame,
/// and the arm that carries it, empty for a camera fixed to the base.
#[derive(Debug, Clone)]
pub struct CameraMount {
    pub color_in_robot: Pose,
    pub carried_by: String,
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
    #[error(
        "get_camera_poses gives {names} names, {positions} position values, \
         {orientations} orientation values and {carriers} carriers: 3, 4 and 1 a camera"
    )]
    PoseCounts {
        names: usize,
        positions: usize,
        orientations: usize,
        carriers: usize,
    },
    #[error("get_camera_poses places no camera named {0:?}")]
    NoSuchCamera(String),
    #[error("get_camera_poses: the pose of {name}: {reason}")]
    Pose { name: String, reason: String },
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

/// Where `poses` place the camera `name`.
pub fn parse_camera_mount(
    poses: &get_camera_poses::ResponseData,
    name: &str,
) -> Result<CameraMount, CameraError> {
    let (positions, partial_position) = poses.positions.as_chunks::<3>();
    let (orientations, partial_orientation) = poses.orientations.as_chunks::<4>();
    let count = poses.camera_names.len();
    if !partial_position.is_empty()
        || !partial_orientation.is_empty()
        || positions.len() != count
        || orientations.len() != count
        || poses.carried_by.len() != count
    {
        return Err(CameraError::PoseCounts {
            names: count,
            positions: poses.positions.len(),
            orientations: poses.orientations.len(),
            carriers: poses.carried_by.len(),
        });
    }
    let index = poses
        .camera_names
        .iter()
        .position(|named| named == name)
        .ok_or_else(|| CameraError::NoSuchCamera(name.to_string()))?;
    let placement =
        Placement::parse(positions[index], orientations[index]).map_err(|e| CameraError::Pose {
            name: name.to_string(),
            reason: e.to_string(),
        })?;
    Ok(CameraMount {
        color_in_robot: placement.pose,
        carried_by: poses.carried_by[index].clone(),
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
/// once each cloud interval, until the node stops.
pub async fn follow_depth(
    runner: Arc<NodeRunner>,
    canvas: Arc<dyn Canvas>,
    settings: DepthSettings,
) {
    let cancelled = stopped(&runner);
    let mut subscription = match sendable(depth_stream::subscribe(&runner)).await {
        Ok(subscription) => subscription,
        Err(e) => {
            error!("depth_stream: no subscription, no depth drawn: {e}");
            return;
        }
    };
    let cloud_linked = get_depth_intrinsics::bound_producer(&runner).is_some()
        && get_camera_poses::bound_producer(&runner).is_some();
    if !cloud_linked {
        info!("the geometry or the camera_mounts link is vacant: no points drawn");
    }
    let mut follower = DepthFollower {
        image_throttle: Throttle::new(settings.image_interval),
        cloud_throttle: Throttle::new(settings.cloud_interval),
        runner,
        canvas,
        settings,
        cloud_linked,
        depth_unit: None,
        camera: None,
        mount: None,
        unreadable: LatchedLog::warn("reading depth_stream"),
        unit_failures: LatchedLog::warn("asking depth_stream_info"),
        cloud_failures: LatchedLog::warn("drawing the depth cloud"),
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

/// The depth follower's state: what it has learned of the camera so far.
struct DepthFollower {
    runner: Arc<NodeRunner>,
    canvas: Arc<dyn Canvas>,
    settings: DepthSettings,
    /// Whether the geometry and the camera_mounts links are bound.
    cloud_linked: bool,
    image_throttle: Throttle,
    cloud_throttle: Throttle,
    depth_unit: Option<PositiveFinite>,
    camera: Option<DepthCamera>,
    /// Kept for a camera fixed to the base; a carried camera is asked for each cloud.
    mount: Option<CameraMount>,
    unreadable: LatchedLog,
    unit_failures: LatchedLog,
    cloud_failures: LatchedLog,
}

impl DepthFollower {
    async fn on_frame(&mut self, message: depth_stream::Message) {
        let now = Instant::now();
        let draw_image = self.settings.depth_image && self.image_throttle.admit_at(now);
        let make_cloud = self.cloud_linked && self.cloud_throttle.admit_at(now);
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
        let align_mode = message.header.align_mode;
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
        if draw_image {
            draw_depth_image(self.canvas.as_ref(), at, &image);
        }
        if make_cloud {
            match self.cloud_of(&image, &align_mode, at).await {
                Ok(()) => self.cloud_failures.success(),
                Err(reason) => self.cloud_failures.failure(reason),
            }
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

    /// Draws the points of `image`, a frame of the alignment `align_mode`
    /// taken at `at`, asking first what it does not know of the camera.
    async fn cloud_of(
        &mut self,
        image: &DepthImage,
        align_mode: &str,
        at: SystemTime,
    ) -> Result<(), String> {
        // Asked when not known yet, and again when the frames' alignment or size changed.
        let stale = self
            .camera
            .as_ref()
            .is_none_or(|camera| camera.align_mode != align_mode || !camera.fits(image));
        if stale {
            let asked = self.ask_depth_camera().await?;
            if asked.align_mode != align_mode {
                return Err(format!(
                    "the depth frames carry align_mode {align_mode:?}, the geometry describes {:?}",
                    asked.align_mode
                ));
            }
            self.camera = Some(asked);
        }
        let mount = self.current_mount().await?;
        let camera = self.camera.as_ref().expect("the geometry is asked above");
        let depth_in_robot = camera.depth_in_robot(&mount.color_in_robot);
        let points = strided_points(image, camera, &depth_in_robot, self.settings.cloud_stride)
            .map_err(|e| e.to_string())?;
        draw_cloud(self.canvas.as_ref(), at, &points);
        Ok(())
    }

    /// Where the camera stands now: kept for a camera fixed to the base, asked each time
    /// for a camera an arm carries.
    async fn current_mount(&mut self) -> Result<CameraMount, String> {
        if let Some(mount) = &self.mount
            && mount.carried_by.is_empty()
        {
            return Ok(mount.clone());
        }
        let mount = self.ask_camera_mount().await?;
        self.mount = Some(mount.clone());
        Ok(mount)
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

    async fn ask_camera_mount(&self) -> Result<CameraMount, String> {
        let mounts = get_camera_poses::bound_producer(&self.runner)
            .ok_or("the camera_mounts link is vacant")?;
        let poses = sendable(get_camera_poses::poll(
            &self.runner,
            mounts,
            SERVICE_TIMEOUT,
        ))
        .await
        .map_err(|e| format!("get_camera_poses: {e}"))?;
        parse_camera_mount(&poses.data, &self.settings.camera_name).map_err(|e| e.to_string())
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

    pub(crate) fn poses() -> get_camera_poses::ResponseData {
        get_camera_poses::ResponseData {
            timestamp: SystemTime::UNIX_EPOCH,
            camera_names: vec!["wrist_left".to_string(), "chest".to_string()],
            positions: vec![0.0, 0.0, 0.0, 0.1, 0.0, 1.2],
            orientations: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            carried_by: vec!["left_arm".to_string(), String::new()],
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

    #[test]
    fn the_camera_is_found_by_its_name() {
        let mount = parse_camera_mount(&poses(), "chest").expect("the chest camera");
        assert_eq!(mount.color_in_robot.origin(), [0.1, 0.0, 1.2]);
        assert!(mount.carried_by.is_empty());
        let wrist = parse_camera_mount(&poses(), "wrist_left").expect("the wrist camera");
        assert_eq!(wrist.carried_by, "left_arm");
        assert_eq!(
            parse_camera_mount(&poses(), "head").unwrap_err(),
            CameraError::NoSuchCamera("head".to_string())
        );
    }

    #[test]
    fn poses_whose_arrays_do_not_fit_the_names_are_refused() {
        let mut short = poses();
        short.carried_by.pop();
        assert!(matches!(
            parse_camera_mount(&short, "chest"),
            Err(CameraError::PoseCounts { carriers: 1, .. })
        ));
    }
}
