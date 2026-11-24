//! A call to a canvas as a value, `Drawn`: what the fake canvas of the tests records,
//! and what the live canvas keeps until the canvas under it takes it. Also when two
//! draws are the same for the eye (`Drawn::same_as`), which is how the live canvas
//! sends only what changes the view, and `DrawnCanvas`, which gives a `Canvas` to any
//! type that takes `Drawn` values.

use std::time::SystemTime;

use frame_tree::Transform;

use crate::canvas::{Canvas, Look, Rgba};
use crate::model::camera::{ColorImage, DepthImage};
use crate::model::placement::rotation_columns;
use crate::model::scene::PlacedShape;

/// One call to a canvas, as a value: what the fake canvas of the tests records, and
/// what the live canvas keeps until the canvas under it takes it.
#[derive(Debug, Clone)]
pub enum Drawn {
    Time(SystemTime),
    Flush,
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
        pose: Transform,
        axis_length: f32,
    },
    Place {
        path: String,
        pose: Transform,
    },
    Shape {
        path: String,
        shape: PlacedShape,
        look: Look,
    },
    StaticShape {
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
    Label {
        path: String,
        position: [f64; 3],
        text: String,
        color: Rgba,
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
    Scalars {
        path: String,
        values: Vec<f64>,
    },
    SeriesNames {
        path: String,
        names: Vec<String>,
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
    /// The entity the call drew at; none for a time and a flush.
    pub fn path(&self) -> Option<&str> {
        match self {
            Self::Time(_) | Self::Flush => None,
            Self::Note { path, .. }
            | Self::Text { path, .. }
            | Self::LogLine { path, .. }
            | Self::Frame { path, .. }
            | Self::Place { path, .. }
            | Self::Shape { path, .. }
            | Self::StaticShape { path, .. }
            | Self::Points { path, .. }
            | Self::Label { path, .. }
            | Self::Cubes { path, .. }
            | Self::LineStrip { path, .. }
            | Self::Scalars { path, .. }
            | Self::SeriesNames { path, .. }
            | Self::ColorImage { path, .. }
            | Self::DepthImage { path, .. }
            | Self::Clear { path } => Some(path),
        }
    }

    /// Whether the call stands for the whole recording, whatever the time: the data
    /// server keeps such a draw for a page that opens late.
    pub fn is_static(&self) -> bool {
        matches!(
            self,
            Self::Note { .. } | Self::StaticShape { .. } | Self::SeriesNames { .. }
        )
    }

    /// Whether the call draws at the same entity what `other` draws, as far as an eye
    /// tells: a pose, a point of a line and a value of a plot within `SAME_WITHIN` of the
    /// other's, and everything else equal. A log line is an event: it is the same as no
    /// other. A draw that is the same as the one before it changes nothing in the view.
    pub fn same_as(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Note { path, markdown },
                Self::Note {
                    path: other_path,
                    markdown: other_markdown,
                },
            )
            | (
                Self::Text { path, markdown },
                Self::Text {
                    path: other_path,
                    markdown: other_markdown,
                },
            ) => path == other_path && markdown == other_markdown,
            (
                Self::Frame {
                    path,
                    pose,
                    axis_length,
                },
                Self::Frame {
                    path: other_path,
                    pose: other_pose,
                    axis_length: other_length,
                },
            ) => path == other_path && axis_length == other_length && same_pose(pose, other_pose),
            (
                Self::Place { path, pose },
                Self::Place {
                    path: other_path,
                    pose: other_pose,
                },
            ) => path == other_path && same_pose(pose, other_pose),
            (
                Self::Shape { path, shape, look },
                Self::Shape {
                    path: other_path,
                    shape: other_shape,
                    look: other_look,
                },
            )
            | (
                Self::StaticShape { path, shape, look },
                Self::StaticShape {
                    path: other_path,
                    shape: other_shape,
                    look: other_look,
                },
            ) => {
                path == other_path
                    && look == other_look
                    && shape.kind == other_shape.kind
                    && same_pose(&shape.pose, &other_shape.pose)
            }
            (
                Self::Points {
                    path,
                    points,
                    color,
                    radius,
                },
                Self::Points {
                    path: other_path,
                    points: other_points,
                    color: other_color,
                    radius: other_radius,
                },
            ) => {
                path == other_path
                    && color == other_color
                    && radius == other_radius
                    && points == other_points
            }
            (
                Self::Label {
                    path,
                    position,
                    text,
                    color,
                },
                Self::Label {
                    path: other_path,
                    position: other_position,
                    text: other_text,
                    color: other_color,
                },
            ) => {
                path == other_path
                    && text == other_text
                    && color == other_color
                    && same_numbers(position, other_position)
            }
            (
                Self::Cubes {
                    path,
                    centers,
                    side,
                    color,
                },
                Self::Cubes {
                    path: other_path,
                    centers: other_centers,
                    side: other_side,
                    color: other_color,
                },
            ) => {
                path == other_path
                    && color == other_color
                    && side == other_side
                    && centers == other_centers
            }
            (
                Self::LineStrip {
                    path,
                    points,
                    color,
                },
                Self::LineStrip {
                    path: other_path,
                    points: other_points,
                    color: other_color,
                },
            ) => {
                path == other_path
                    && color == other_color
                    && points.len() == other_points.len()
                    && points
                        .iter()
                        .zip(other_points)
                        .all(|(point, other)| same_numbers(point, other))
            }
            (
                Self::Scalars { path, values },
                Self::Scalars {
                    path: other_path,
                    values: other_values,
                },
            ) => path == other_path && same_numbers(values, other_values),
            (
                Self::SeriesNames { path, names },
                Self::SeriesNames {
                    path: other_path,
                    names: other_names,
                },
            ) => path == other_path && names == other_names,
            (
                Self::ColorImage { path, image },
                Self::ColorImage {
                    path: other_path,
                    image: other_image,
                },
            ) => path == other_path && image == other_image,
            (
                Self::DepthImage { path, image },
                Self::DepthImage {
                    path: other_path,
                    image: other_image,
                },
            ) => path == other_path && image == other_image,
            (Self::Clear { path }, Self::Clear { path: other_path }) => path == other_path,
            _ => false,
        }
    }

    /// Makes the call again, on `canvas`.
    pub fn replay(&self, canvas: &dyn Canvas) {
        match self {
            Self::Time(at) => canvas.set_time(*at),
            Self::Flush => canvas.flush(),
            Self::Note { path, markdown } => canvas.note(path, markdown),
            Self::Text { path, markdown } => canvas.text(path, markdown),
            Self::LogLine { path, line } => canvas.log_line(path, line),
            Self::Frame {
                path,
                pose,
                axis_length,
            } => canvas.frame(path, pose, *axis_length),
            Self::Place { path, pose } => canvas.place(path, pose),
            Self::Shape { path, shape, look } => canvas.shape(path, shape, *look),
            Self::StaticShape { path, shape, look } => canvas.static_shape(path, shape, *look),
            Self::Points {
                path,
                points,
                color,
                radius,
            } => canvas.points(path, points, *color, *radius),
            Self::Label {
                path,
                position,
                text,
                color,
            } => canvas.label(path, *position, text, *color),
            Self::Cubes {
                path,
                centers,
                side,
                color,
            } => canvas.cubes(path, centers, *side, *color),
            Self::LineStrip {
                path,
                points,
                color,
            } => canvas.line_strip(path, points, *color),
            Self::Scalars { path, values } => canvas.scalars(path, values),
            Self::SeriesNames { path, names } => canvas.series_names(path, names),
            Self::ColorImage { path, image } => canvas.color_image(path, image),
            Self::DepthImage { path, image } => canvas.depth_image(path, image),
            Self::Clear { path } => canvas.clear(path),
        }
    }
}

/// How near two poses, two points of a line or two values of a plot are for their draws
/// to be the same (`Drawn::same_as`): 0.1 mm, and 1e-4 of a rotation's unit vectors
/// (about 0.006 degrees) or of a plot's unit.
pub const SAME_WITHIN: f64 = 1e-4;

fn same_numbers(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= SAME_WITHIN)
}

/// Whether the positions and the rotations of two poses are within `SAME_WITHIN`.
fn same_pose(a: &Transform, b: &Transform) -> bool {
    same_numbers(&a.position(), &b.position())
        && rotation_columns(a)
            .iter()
            .zip(&rotation_columns(b))
            .all(|(column, other)| same_numbers(column, other))
}

/// A canvas that takes each call as its [`Drawn`] value.
pub trait DrawnCanvas: Send + Sync {
    fn take(&self, call: Drawn);
}

impl<T: DrawnCanvas> Canvas for T {
    fn set_time(&self, at: SystemTime) {
        self.take(Drawn::Time(at));
    }

    fn note(&self, path: &str, markdown: &str) {
        self.take(Drawn::Note {
            path: path.to_string(),
            markdown: markdown.to_string(),
        });
    }

    fn text(&self, path: &str, markdown: &str) {
        self.take(Drawn::Text {
            path: path.to_string(),
            markdown: markdown.to_string(),
        });
    }

    fn log_line(&self, path: &str, line: &str) {
        self.take(Drawn::LogLine {
            path: path.to_string(),
            line: line.to_string(),
        });
    }

    fn frame(&self, path: &str, pose: &Transform, axis_length: f32) {
        self.take(Drawn::Frame {
            path: path.to_string(),
            pose: *pose,
            axis_length,
        });
    }

    fn place(&self, path: &str, pose: &Transform) {
        self.take(Drawn::Place {
            path: path.to_string(),
            pose: *pose,
        });
    }

    fn shape(&self, path: &str, shape: &PlacedShape, look: Look) {
        self.take(Drawn::Shape {
            path: path.to_string(),
            shape: shape.clone(),
            look,
        });
    }

    fn static_shape(&self, path: &str, shape: &PlacedShape, look: Look) {
        self.take(Drawn::StaticShape {
            path: path.to_string(),
            shape: shape.clone(),
            look,
        });
    }

    fn points(&self, path: &str, points: &[[f32; 3]], color: Rgba, radius: f32) {
        self.take(Drawn::Points {
            path: path.to_string(),
            points: points.to_vec(),
            color,
            radius,
        });
    }

    fn label(&self, path: &str, position: [f64; 3], text: &str, color: Rgba) {
        self.take(Drawn::Label {
            path: path.to_string(),
            position,
            text: text.to_string(),
            color,
        });
    }

    fn cubes(&self, path: &str, centers: &[[f32; 3]], side: f32, color: Rgba) {
        self.take(Drawn::Cubes {
            path: path.to_string(),
            centers: centers.to_vec(),
            side,
            color,
        });
    }

    fn line_strip(&self, path: &str, points: &[[f64; 3]], color: Rgba) {
        self.take(Drawn::LineStrip {
            path: path.to_string(),
            points: points.to_vec(),
            color,
        });
    }

    fn scalars(&self, path: &str, values: &[f64]) {
        self.take(Drawn::Scalars {
            path: path.to_string(),
            values: values.to_vec(),
        });
    }

    fn series_names(&self, path: &str, names: &[String]) {
        self.take(Drawn::SeriesNames {
            path: path.to_string(),
            names: names.to_vec(),
        });
    }

    fn color_image(&self, path: &str, image: &ColorImage) {
        self.take(Drawn::ColorImage {
            path: path.to_string(),
            image: image.clone(),
        });
    }

    fn depth_image(&self, path: &str, image: &DepthImage) {
        self.take(Drawn::DepthImage {
            path: path.to_string(),
            image: image.clone(),
        });
    }

    fn clear(&self, path: &str) {
        self.take(Drawn::Clear {
            path: path.to_string(),
        });
    }

    fn flush(&self) {
        self.take(Drawn::Flush);
    }
}

#[cfg(test)]
mod tests {
    use frame_tree::UnitQuaternion;

    use super::*;
    use crate::model::placement::translation;

    fn place(path: &str, x: f64) -> Drawn {
        Drawn::Place {
            path: path.to_string(),
            pose: translation([x, 0.0, 0.0]),
        }
    }

    #[test]
    fn a_pose_within_a_tenth_of_a_millimetre_is_the_same_draw() {
        let at = place("world/robot/link1", 0.3);
        assert!(at.same_as(&place("world/robot/link1", 0.30005)));
        assert!(!at.same_as(&place("world/robot/link1", 0.301)));
        assert!(
            !at.same_as(&place("world/robot/link2", 0.3)),
            "another entity"
        );
        // A quarter turn about z moves no origin.
        let turned = Drawn::Place {
            path: "world/robot/link1".to_string(),
            pose: Transform::new(
                [0.3, 0.0, 0.0],
                UnitQuaternion::from_yaw(std::f64::consts::FRAC_PI_2).unwrap(),
            )
            .unwrap(),
        };
        assert!(!at.same_as(&turned));
    }

    #[test]
    fn the_values_of_a_plot_within_the_tolerance_are_the_same_draw() {
        let values = |values: &[f64]| Drawn::Scalars {
            path: "plots/joints/left_arm".to_string(),
            values: values.to_vec(),
        };
        assert!(values(&[0.5, -0.25]).same_as(&values(&[0.50005, -0.25])));
        assert!(!values(&[0.5, -0.25]).same_as(&values(&[0.501, -0.25])));
        assert!(!values(&[0.5, -0.25]).same_as(&values(&[0.5])));
    }

    #[test]
    fn points_are_the_same_draw_only_when_each_is_equal() {
        let points = |points: &[[f32; 3]]| Drawn::Points {
            path: "world/camera/points".to_string(),
            points: points.to_vec(),
            color: [1, 2, 3, 255],
            radius: 0.004,
        };
        let cloud = points(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        assert!(cloud.same_as(&points(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])));
        assert!(!cloud.same_as(&points(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.7]])));
    }

    #[test]
    fn a_log_line_is_the_same_as_no_other_and_only_some_draws_stand_for_the_whole_recording() {
        let line = || Drawn::LogLine {
            path: "scene/changes".to_string(),
            line: "moved brain/lime_1".to_string(),
        };
        assert!(!line().same_as(&line()));
        let note = Drawn::Note {
            path: "notes/cubes".to_string(),
            markdown: "a note".to_string(),
        };
        assert!(note.same_as(&note.clone()));
        assert!(note.is_static());
        assert!(!place("world/robot/link1", 0.3).is_static());
        assert!(!line().is_static());
    }
}
