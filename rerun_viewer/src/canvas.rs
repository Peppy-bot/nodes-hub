//! What the drawers draw on: the `Canvas` trait, in the words of this node, and
//! `RerunCanvas`, the Rerun viewer that implements it. This file is the only code that
//! calls Rerun; tests draw on the `FakeCanvas` of `testing.rs`.
//!
//! Rerun keeps the time of a recording for each thread, so a drawer sets the time and
//! draws in one synchronous call, with no `.await` between, which keeps the task on one
//! thread.

use std::net::{IpAddr, TcpListener};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use camera_core::pose::Pose;
use control_core::latched_log::LatchedLog;
use rerun::external::image::ExtendedColorType;
use rerun::external::image::codecs::jpeg::JpegEncoder;
use rerun::grpc_server::GrpcServerSink;
use rerun::sink::FileSink;
use rerun::web_viewer::WebViewerConfig;
use rerun::{
    AsComponents, Boxes3D, Clear, Color, Cylinders3D, DepthImage as RerunDepthImage, Ellipsoids3D,
    EncodedImage, FillMode, LineStrips3D, MemoryLimit, Mesh3D, Points3D, RecordingStream,
    RecordingStreamBuilder, Scalars, ServerOptions, TextDocument, TextLog, Transform3D,
    TransformAxes3D, ViewCoordinates,
};
use tracing::info;

use crate::config::ConfigError;
use crate::model::camera::{ColorImage, DepthImage};
use crate::model::placement::rotation_columns;
use crate::model::scene::{PlacedShape, ShapeKind};

/// The timeline every entry of the viewer is on.
pub const TIMELINE: &str = "time";

/// The quality (1 to 100) of the JPEG a raw color image is sent as.
const JPEG_QUALITY: u8 = 80;

/// A color, red, green, blue and opacity.
pub type Rgba = [u8; 4];

/// How a shape is drawn: its color, and solid or as lines only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Look {
    pub color: Rgba,
    pub wireframe: bool,
}

/// What the drawers draw on. A path names an entity of the viewer, its parts separated
/// by `/`; a pose places an entity in its parent's frame, and the entities under it
/// move with it.
pub trait Canvas: Send + Sync {
    /// Puts what follows, on this thread, at `at` on the timeline.
    fn set_time(&self, at: SystemTime);
    /// A text that stands for the whole recording, whatever the time.
    fn note(&self, path: &str, markdown: &str);
    /// A text that holds from now until the next text at `path`.
    fn text(&self, path: &str, markdown: &str);
    /// One line of a log.
    fn log_line(&self, path: &str, line: &str);
    /// Places the entity at `pose` and draws its axes, each `axis_length` long.
    fn frame(&self, path: &str, pose: &Pose, axis_length: f32);
    /// Places the entity at `pose`, and draws nothing there.
    fn place(&self, path: &str, pose: &Pose);
    /// Places the entity at the shape's pose and draws the shape there.
    fn shape(&self, path: &str, shape: &PlacedShape, look: Look);
    /// Draws the shape as `shape` does, for the whole recording, whatever the time: a
    /// part of the robot that moves only with the entity above it.
    fn static_shape(&self, path: &str, shape: &PlacedShape, look: Look);
    fn points(&self, path: &str, points: &[[f32; 3]], color: Rgba, radius: f32);
    /// Cubes of side `side` around `centers`.
    fn cubes(&self, path: &str, centers: &[[f32; 3]], side: f32, color: Rgba);
    fn line_strip(&self, path: &str, points: &[[f64; 3]], color: Rgba);
    /// One value of a plot.
    fn scalar(&self, path: &str, value: f64);
    fn color_image(&self, path: &str, image: &ColorImage);
    fn depth_image(&self, path: &str, image: &DepthImage);
    /// Removes the entity and every entity under it from now on.
    fn clear(&self, path: &str);
}

/// Where and how the viewer serves, parsed from the node's parameters.
#[derive(Debug)]
pub struct ViewerConfig {
    pub bind_ip: IpAddr,
    pub web_port: u16,
    pub grpc_port: u16,
    pub memory_limit: MemoryLimit,
    pub recording_path: Option<PathBuf>,
}

impl ViewerConfig {
    pub fn parse(
        bind_ip: &str,
        web_port: u16,
        grpc_port: u16,
        memory_limit: &str,
        recording_path: &str,
    ) -> Result<Self, ConfigError> {
        let bind_ip = bind_ip.parse::<IpAddr>().map_err(|_| {
            ConfigError::new("bind_ip", bind_ip, "it is an IP address, such as 127.0.0.1")
        })?;
        if web_port == 0 {
            return Err(ConfigError::new(
                "web_port",
                web_port,
                "it is a port, 1 or more",
            ));
        }
        if grpc_port == 0 || grpc_port == web_port {
            return Err(ConfigError::new(
                "grpc_port",
                grpc_port,
                "it is a port, 1 or more, other than web_port",
            ));
        }
        let memory_limit = MemoryLimit::parse(memory_limit).map_err(|reason| {
            ConfigError::new(
                "memory_limit",
                memory_limit,
                format!("it is a size such as 1GB or a share such as 25%: {reason}"),
            )
        })?;
        let recording_path = (!recording_path.is_empty()).then(|| PathBuf::from(recording_path));
        Ok(Self {
            bind_ip,
            web_port,
            grpc_port,
            memory_limit,
            recording_path,
        })
    }

    /// The address of the viewer's page on the host `host`, connected to the viewer's
    /// data.
    pub fn page_url(&self, host: &str) -> String {
        format!(
            "http://{host}:{}/?url=rerun%2Bhttp%3A%2F%2F{host}%3A{}%2Fproxy",
            self.web_port, self.grpc_port
        )
    }

    /// The origin a browser gives the viewer's page: its address and web port. A viewer
    /// that serves on every address of the machine takes the page from any of them.
    pub fn page_origin(&self) -> String {
        if self.bind_ip.is_unspecified() {
            return format!("http://*:{}", self.web_port);
        }
        format!("http://{}:{}", host_of(self.bind_ip), self.web_port)
    }
}

/// `ip` as the host part of an address: an IPv6 address in brackets.
pub fn host_of(ip: IpAddr) -> String {
    match ip {
        IpAddr::V4(v4) => v4.to_string(),
        IpAddr::V6(v6) => format!("[{v6}]"),
    }
}

/// Why the viewer did not start.
#[derive(Debug, thiserror::Error)]
pub enum CanvasError {
    #[error("the viewer's data server: {0}")]
    Data(String),
    #[error("the recording file: {0}")]
    File(String),
    #[error("the viewer's web page: {0}")]
    Page(String),
}

/// The Rerun viewer: a recording that goes to a gRPC server, and to a file when the
/// parameters name one, and the web page that shows it, served for the life of the node.
pub struct RerunCanvas {
    recording: RecordingStream,
    failures: LatchedLog,
}

impl RerunCanvas {
    /// Starts the data server and the web page that `config` describes.
    pub fn serve(config: &ViewerConfig) -> Result<Self, CanvasError> {
        let bind_ip = config.bind_ip.to_string();
        let options = ServerOptions {
            memory_limit: config.memory_limit,
            // The browser sends the page's origin with each read of the data.
            cors_allowed_origins: vec![config.page_origin()],
            ..ServerOptions::default()
        };
        // The data server binds its port on a thread of its own and only logs a failure,
        // so the port is tried here first: a port in use stops the node.
        drop(
            TcpListener::bind((config.bind_ip, config.grpc_port))
                .map_err(|e| CanvasError::Data(format!("{bind_ip}:{}: {e}", config.grpc_port)))?,
        );
        // The data server reads `<host>:<port>`, so an IPv6 address takes brackets there.
        let data = GrpcServerSink::new(&host_of(config.bind_ip), config.grpc_port, options)
            .map_err(|e| CanvasError::Data(e.to_string()))?;
        let builder = RecordingStreamBuilder::new("rerun_viewer");
        let recording = match &config.recording_path {
            Some(path) => {
                let file = FileSink::new(path).map_err(|e| CanvasError::File(e.to_string()))?;
                builder.set_sinks((data, file))
            }
            None => builder.set_sinks((data,)),
        }
        .map_err(|e| CanvasError::Data(e.to_string()))?;
        rerun::serve_web_viewer(WebViewerConfig {
            bind_ip: bind_ip.clone(),
            web_port: config.web_port.into(),
            open_browser: false,
            ..WebViewerConfig::default()
        })
        .map_err(|e| CanvasError::Page(e.to_string()))?
        .detach();
        recording
            .log_static("world", &ViewCoordinates::FLU())
            .map_err(|e| CanvasError::Data(e.to_string()))?;
        let host = if config.bind_ip.is_unspecified() {
            "<an address of this machine>".to_string()
        } else {
            host_of(config.bind_ip)
        };
        info!("the viewer serves at {}", config.page_url(&host));
        Ok(Self {
            recording,
            failures: LatchedLog::warn("drawing in the viewer"),
        })
    }

    /// Logs `entity` at `path` at the time of the thread, noting a failure once for
    /// each run of failures.
    fn log(&self, path: &str, entity: &dyn AsComponents) {
        self.noted(path, self.recording.log(path, entity));
    }

    /// Logs `entity` at `path` for the whole recording, whatever the time. The data
    /// server keeps such entities for a page that opens late, past the memory limit
    /// for the other data.
    fn log_static(&self, path: &str, entity: &dyn AsComponents) {
        self.noted(path, self.recording.log_static(path, entity));
    }

    fn noted(&self, path: &str, logged: rerun::RecordingStreamResult<()>) {
        match logged {
            Ok(()) => self.failures.success(),
            Err(e) => self.failures.failure(format!("{path}: {e}")),
        }
    }

    /// Logs `shape` at `path`, placed at its pose, with `log`.
    fn log_shape(
        &self,
        path: &str,
        shape: &PlacedShape,
        look: Look,
        log: fn(&Self, &str, &dyn AsComponents),
    ) {
        log(self, path, &transform_of(&shape.pose));
        let color = rerun_color(look.color);
        match &shape.kind {
            ShapeKind::Box { size } => log(
                self,
                path,
                &Boxes3D::from_sizes([size.map(|side| side as f32)])
                    .with_colors([color])
                    .with_fill_mode(fill_mode(look)),
            ),
            ShapeKind::Sphere { radius } => log(
                self,
                path,
                &Ellipsoids3D::from_radii([*radius as f32])
                    .with_colors([color])
                    .with_fill_mode(fill_mode(look)),
            ),
            ShapeKind::Cylinder { height, radius } => log(
                self,
                path,
                &Cylinders3D::from_lengths_and_radii([*height as f32], [*radius as f32])
                    .with_colors([color])
                    .with_fill_mode(fill_mode(look)),
            ),
            ShapeKind::Mesh {
                vertices,
                triangles,
            } => log(
                self,
                path,
                &Mesh3D::new(vertices.iter().map(|vertex| vertex.map(|c| c as f32)))
                    .with_triangle_indices(triangles.iter().copied())
                    .with_albedo_factor(color.0),
            ),
        }
    }
}

fn rerun_color([r, g, b, a]: Rgba) -> Color {
    Color::from_unmultiplied_rgba(r, g, b, a)
}

fn fill_mode(look: Look) -> FillMode {
    if look.wireframe {
        FillMode::MajorWireframe
    } else {
        FillMode::Solid
    }
}

/// The transform that places an entity at `pose`.
fn transform_of(pose: &Pose) -> Transform3D {
    let columns = rotation_columns(pose).map(|column| column.map(|c| c as f32));
    let origin = pose.origin().map(|c| c as f32);
    Transform3D::from_translation_mat3x3(origin, columns)
}

/// `image` as a JPEG: a raw image encoded, a JPEG as it came.
fn jpeg_of(image: &ColorImage) -> Result<Vec<u8>, String> {
    match image {
        ColorImage::Jpeg(bytes) => Ok(bytes.clone()),
        ColorImage::Rgb {
            width,
            height,
            pixels,
        } => {
            let mut jpeg = Vec::new();
            JpegEncoder::new_with_quality(&mut jpeg, JPEG_QUALITY)
                .encode(pixels, *width, *height, ExtendedColorType::Rgb8)
                .map_err(|e| e.to_string())?;
            Ok(jpeg)
        }
    }
}

impl Canvas for RerunCanvas {
    fn set_time(&self, at: SystemTime) {
        let seconds = match at.duration_since(UNIX_EPOCH) {
            Ok(since) => since.as_secs_f64(),
            Err(before) => -before.duration().as_secs_f64(),
        };
        self.recording
            .set_timestamp_secs_since_epoch(TIMELINE, seconds);
    }

    fn note(&self, path: &str, markdown: &str) {
        self.log_static(path, &TextDocument::from_markdown(markdown));
    }

    fn text(&self, path: &str, markdown: &str) {
        self.log(path, &TextDocument::from_markdown(markdown));
    }

    fn log_line(&self, path: &str, line: &str) {
        self.log(path, &TextLog::new(line));
    }

    fn frame(&self, path: &str, pose: &Pose, axis_length: f32) {
        self.place(path, pose);
        self.log(path, &TransformAxes3D::new(axis_length));
    }

    fn place(&self, path: &str, pose: &Pose) {
        self.log(path, &transform_of(pose));
    }

    fn shape(&self, path: &str, shape: &PlacedShape, look: Look) {
        self.log_shape(path, shape, look, Self::log);
    }

    fn static_shape(&self, path: &str, shape: &PlacedShape, look: Look) {
        self.log_shape(path, shape, look, Self::log_static);
    }

    fn points(&self, path: &str, points: &[[f32; 3]], color: Rgba, radius: f32) {
        self.log(
            path,
            &Points3D::new(points.iter().copied())
                .with_colors([rerun_color(color)])
                .with_radii([radius]),
        );
    }

    fn cubes(&self, path: &str, centers: &[[f32; 3]], side: f32, color: Rgba) {
        let half = side / 2.0;
        self.log(
            path,
            &Boxes3D::from_centers_and_half_sizes(
                centers.iter().copied(),
                centers.iter().map(|_| [half, half, half]),
            )
            .with_colors([rerun_color(color)])
            .with_fill_mode(FillMode::Solid),
        );
    }

    fn line_strip(&self, path: &str, points: &[[f64; 3]], color: Rgba) {
        let strip: Vec<[f32; 3]> = points.iter().map(|point| point.map(|c| c as f32)).collect();
        self.log(
            path,
            &LineStrips3D::new([strip]).with_colors([rerun_color(color)]),
        );
    }

    fn scalar(&self, path: &str, value: f64) {
        self.log(path, &Scalars::single(value));
    }

    fn color_image(&self, path: &str, image: &ColorImage) {
        match jpeg_of(image) {
            Ok(jpeg) => self.log(path, &EncodedImage::from_file_contents(jpeg)),
            Err(e) => self.failures.failure(format!("{path}: {e}")),
        }
    }

    fn depth_image(&self, path: &str, image: &DepthImage) {
        self.log(
            path,
            &RerunDepthImage::from_gray16(
                image.samples().to_vec(),
                [image.width(), image.height()],
            )
            .with_meter(1.0 / image.depth_unit().get() as f32),
        );
    }

    fn clear(&self, path: &str) {
        self.log(path, &Clear::recursive());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(
        bind_ip: &str,
        web: u16,
        grpc: u16,
        memory: &str,
    ) -> Result<ViewerConfig, ConfigError> {
        ViewerConfig::parse(bind_ip, web, grpc, memory, "")
    }

    #[test]
    fn the_defaults_parse() {
        let config = parse("127.0.0.1", 9090, 9876, "1GB").expect("valid");
        assert_eq!(config.bind_ip.to_string(), "127.0.0.1");
        assert_eq!(config.recording_path, None);
        assert_eq!(
            config.page_url("192.0.2.7"),
            "http://192.0.2.7:9090/?url=rerun%2Bhttp%3A%2F%2F192.0.2.7%3A9876%2Fproxy"
        );
    }

    #[test]
    fn the_page_is_allowed_from_its_own_origin_only() {
        let config = parse("192.0.2.7", 9090, 9876, "1GB").unwrap();
        assert_eq!(config.page_origin(), "http://192.0.2.7:9090");
        let everywhere = parse("0.0.0.0", 9090, 9876, "1GB").unwrap();
        assert_eq!(everywhere.page_origin(), "http://*:9090");
    }

    #[test]
    fn an_ipv6_address_takes_brackets() {
        let config = parse("::1", 9090, 9876, "1GB").unwrap();
        assert_eq!(config.page_origin(), "http://[::1]:9090");
        assert_eq!(
            config.page_url(&host_of(config.bind_ip)),
            "http://[::1]:9090/?url=rerun%2Bhttp%3A%2F%2F[::1]%3A9876%2Fproxy"
        );
    }

    #[test]
    fn a_recording_path_is_kept() {
        let config = ViewerConfig::parse("0.0.0.0", 9090, 9876, "25%", "/tmp/run.rrd").unwrap();
        assert_eq!(config.recording_path, Some(PathBuf::from("/tmp/run.rrd")));
    }

    #[test]
    fn an_address_a_port_or_a_memory_limit_out_of_range_is_refused() {
        assert_eq!(
            parse("localhost", 9090, 9876, "1GB").unwrap_err().name,
            "bind_ip"
        );
        assert_eq!(
            parse("127.0.0.1", 0, 9876, "1GB").unwrap_err().name,
            "web_port"
        );
        assert_eq!(
            parse("127.0.0.1", 9090, 0, "1GB").unwrap_err().name,
            "grpc_port"
        );
        assert_eq!(
            parse("127.0.0.1", 9090, 9090, "1GB").unwrap_err().name,
            "grpc_port"
        );
        assert_eq!(
            parse("127.0.0.1", 9090, 9876, "lots").unwrap_err().name,
            "memory_limit"
        );
    }

    #[test]
    fn a_raw_image_becomes_a_jpeg_and_a_jpeg_stays_as_it_is() {
        let raw = ColorImage::Rgb {
            width: 2,
            height: 2,
            pixels: vec![255; 12],
        };
        let jpeg = jpeg_of(&raw).expect("an encodable image");
        assert_eq!(
            &jpeg[..2],
            &[0xFF, 0xD8],
            "a JPEG starts with its SOI marker"
        );
        assert_eq!(jpeg_of(&ColorImage::Jpeg(vec![1, 2])).unwrap(), vec![1, 2]);
    }
}
