//! The canvas of the tests: it records each call, and a test waits until what it
//! expects is drawn, woken by each draw rather than by a clock.

use std::sync::{Mutex, PoisonError};
use std::time::{Duration, SystemTime};

use camera_core::pose::Pose;
use tokio::sync::watch;

use crate::canvas::{Canvas, Look, Rgba};
use crate::model::camera::{ColorImage, DepthImage};
use crate::model::scene::PlacedShape;

/// One call to the canvas.
#[derive(Debug, Clone)]
pub enum Drawn {
    Time(SystemTime),
    Note {
        path: String,
        markdown: String,
    },
    Text {
        path: String,
        markdown: String,
    },
    LogLine {
        path: String,
        line: String,
    },
    Frame {
        path: String,
        pose: Pose,
        axis_length: f32,
    },
    Shape {
        path: String,
        shape: PlacedShape,
        look: Look,
    },
    Points {
        path: String,
        points: Vec<[f32; 3]>,
        color: Rgba,
        radius: f32,
    },
    Cubes {
        path: String,
        centers: Vec<[f32; 3]>,
        side: f32,
        color: Rgba,
    },
    LineStrip {
        path: String,
        points: Vec<[f64; 3]>,
        color: Rgba,
    },
    Scalar {
        path: String,
        value: f64,
    },
    ColorImage {
        path: String,
        image: ColorImage,
    },
    DepthImage {
        path: String,
        image: DepthImage,
    },
    Clear {
        path: String,
    },
}

impl Drawn {
    /// The entity the call drew at; none for a time.
    pub fn path(&self) -> Option<&str> {
        match self {
            Self::Time(_) => None,
            Self::Note { path, .. }
            | Self::Text { path, .. }
            | Self::LogLine { path, .. }
            | Self::Frame { path, .. }
            | Self::Shape { path, .. }
            | Self::Points { path, .. }
            | Self::Cubes { path, .. }
            | Self::LineStrip { path, .. }
            | Self::Scalar { path, .. }
            | Self::ColorImage { path, .. }
            | Self::DepthImage { path, .. }
            | Self::Clear { path } => Some(path),
        }
    }
}

/// A canvas that records each call, in order.
pub struct FakeCanvas {
    drawn: Mutex<Vec<Drawn>>,
    /// The number of calls so far, which wakes a waiting test.
    count: watch::Sender<usize>,
}

impl Default for FakeCanvas {
    fn default() -> Self {
        Self {
            drawn: Mutex::new(Vec::new()),
            count: watch::Sender::new(0),
        }
    }
}

impl FakeCanvas {
    /// Every call so far, in order.
    pub fn drawn(&self) -> Vec<Drawn> {
        self.drawn
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Every call so far at `path`, in order.
    pub fn at(&self, path: &str) -> Vec<Drawn> {
        self.drawn()
            .into_iter()
            .filter(|drawn| drawn.path() == Some(path))
            .collect()
    }

    /// The lines logged at `path`, in order.
    pub fn log_lines(&self, path: &str) -> Vec<String> {
        self.at(path)
            .into_iter()
            .filter_map(|drawn| match drawn {
                Drawn::LogLine { line, .. } => Some(line),
                _ => None,
            })
            .collect()
    }

    /// Waits until `done` holds of the calls so far, woken by each call. An error after
    /// `timeout`, which only bounds a test that fails, names `what`.
    pub async fn wait_until(
        &self,
        timeout: Duration,
        what: &str,
        done: impl Fn(&[Drawn]) -> bool,
    ) -> Result<(), String> {
        let mut count = self.count.subscribe();
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if done(&self.drawn()) {
                return Ok(());
            }
            match tokio::time::timeout_at(deadline, count.changed()).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) => return Err(format!("the canvas closed before {what}")),
                Err(_) => {
                    return Err(format!(
                        "{what} not drawn within {timeout:?}; drawn: {:#?}",
                        self.drawn()
                    ));
                }
            }
        }
    }

    fn record(&self, drawn: Drawn) {
        self.drawn
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(drawn);
        self.count.send_modify(|count| *count += 1);
    }
}

impl Canvas for FakeCanvas {
    fn set_time(&self, at: SystemTime) {
        self.record(Drawn::Time(at));
    }

    fn note(&self, path: &str, markdown: &str) {
        self.record(Drawn::Note {
            path: path.to_string(),
            markdown: markdown.to_string(),
        });
    }

    fn text(&self, path: &str, markdown: &str) {
        self.record(Drawn::Text {
            path: path.to_string(),
            markdown: markdown.to_string(),
        });
    }

    fn log_line(&self, path: &str, line: &str) {
        self.record(Drawn::LogLine {
            path: path.to_string(),
            line: line.to_string(),
        });
    }

    fn frame(&self, path: &str, pose: &Pose, axis_length: f32) {
        self.record(Drawn::Frame {
            path: path.to_string(),
            pose: *pose,
            axis_length,
        });
    }

    fn shape(&self, path: &str, shape: &PlacedShape, look: Look) {
        self.record(Drawn::Shape {
            path: path.to_string(),
            shape: shape.clone(),
            look,
        });
    }

    fn points(&self, path: &str, points: &[[f32; 3]], color: Rgba, radius: f32) {
        self.record(Drawn::Points {
            path: path.to_string(),
            points: points.to_vec(),
            color,
            radius,
        });
    }

    fn cubes(&self, path: &str, centers: &[[f32; 3]], side: f32, color: Rgba) {
        self.record(Drawn::Cubes {
            path: path.to_string(),
            centers: centers.to_vec(),
            side,
            color,
        });
    }

    fn line_strip(&self, path: &str, points: &[[f64; 3]], color: Rgba) {
        self.record(Drawn::LineStrip {
            path: path.to_string(),
            points: points.to_vec(),
            color,
        });
    }

    fn scalar(&self, path: &str, value: f64) {
        self.record(Drawn::Scalar {
            path: path.to_string(),
            value,
        });
    }

    fn color_image(&self, path: &str, image: &ColorImage) {
        self.record(Drawn::ColorImage {
            path: path.to_string(),
            image: image.clone(),
        });
    }

    fn depth_image(&self, path: &str, image: &DepthImage) {
        self.record(Drawn::DepthImage {
            path: path.to_string(),
            image: image.clone(),
        });
    }

    fn clear(&self, path: &str) {
        self.record(Drawn::Clear {
            path: path.to_string(),
        });
    }
}
