//! The canvas of the tests: it records each call, and a test waits until what it
//! expects is drawn, woken by each draw rather than by a clock. Also the small robot
//! that the tests describe, with numbers a reader can check by hand.

use std::sync::{Mutex, PoisonError};
use std::time::{Duration, SystemTime};

use camera_core::pose::Pose;
use tokio::sync::watch;

use crate::canvas::{Canvas, Look, Rgba};
use crate::model::camera::{ColorImage, DepthImage};
use crate::model::robot::RobotBody;
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
    Place {
        path: String,
        pose: Pose,
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
            | Self::Place { path, .. }
            | Self::Shape { path, .. }
            | Self::StaticShape { path, .. }
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

    fn place(&self, path: &str, pose: &Pose) {
        self.record(Drawn::Place {
            path: path.to_string(),
            pose: *pose,
        });
    }

    fn shape(&self, path: &str, shape: &PlacedShape, look: Look) {
        self.record(Drawn::Shape {
            path: path.to_string(),
            shape: shape.clone(),
            look,
        });
    }

    fn static_shape(&self, path: &str, shape: &PlacedShape, look: Look) {
        self.record(Drawn::StaticShape {
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

/// The bytes of a binary STL of `triangles`, each three vertices, with zero normals.
pub fn stl_bytes(triangles: &[[[f32; 3]; 3]]) -> Vec<u8> {
    let mut bytes = vec![0; 80];
    let count = u32::try_from(triangles.len()).expect("a test mesh of few triangles");
    bytes.extend_from_slice(&count.to_le_bytes());
    for triangle in triangles {
        bytes.extend_from_slice(&[0; 12]);
        for coordinate in triangle.iter().flatten() {
            bytes.extend_from_slice(&coordinate.to_le_bytes());
        }
        bytes.extend_from_slice(&[0; 2]);
    }
    bytes
}

/// The URDF of a small robot, `mini`: a base box, an upper arm (a cylinder) that `lift`
/// turns about z, 0.1 m up, a hand (a mesh, mirrored across y) that `reach` turns about
/// y, 0.2 m up the upper arm, a grasp point and a finger (a sphere) 0.1 m along the
/// hand's x, the finger sliding along the hand's y (`pinch`, 0 closed, 0.04 m open), and
/// a capsule on the base.
pub const MINI_URDF: &str = r#"<robot name="mini">
  <link name="base">
    <collision><origin xyz="0 0 0.05"/><geometry><box size="0.2 0.2 0.1"/></geometry></collision>
    <collision><origin xyz="0.1 0 0" rpy="0 1.5707963267948966 0"/><geometry><capsule radius="0.01" length="0.04"/></geometry></collision>
  </link>
  <link name="upper">
    <collision><origin xyz="0 0 0.1"/><geometry><cylinder radius="0.03" length="0.2"/></geometry></collision>
  </link>
  <link name="hand">
    <collision><geometry><mesh filename="package://mini/meshes/hand.stl" scale="1 -1 1"/></geometry></collision>
  </link>
  <link name="grasp"/>
  <link name="finger">
    <collision><geometry><sphere radius="0.01"/></geometry></collision>
  </link>
  <joint name="lift" type="revolute">
    <origin xyz="0 0 0.1"/><parent link="base"/><child link="upper"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="reach" type="revolute">
    <origin xyz="0 0 0.2"/><parent link="upper"/><child link="hand"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="grasp_joint" type="fixed">
    <origin xyz="0.1 0 0"/><parent link="hand"/><child link="grasp"/>
  </joint>
  <joint name="pinch" type="prismatic">
    <origin xyz="0.1 0 0"/><parent link="hand"/><child link="finger"/><axis xyz="0 1 0"/>
    <limit lower="0" upper="0.04" effort="1" velocity="1"/>
  </joint>
</robot>"#;

/// The SRDF of `mini`: the arm `arm` from the base to the grasp point, and the gripper
/// `gripper` of the finger.
pub const MINI_SRDF: &str = r#"<robot name="mini">
  <group name="arm"><chain base_link="base" tip_link="grasp"/></group>
  <group name="gripper"><joint name="pinch"/></group>
  <group_state name="closed" group="gripper"><joint name="pinch" value="0"/></group_state>
  <group_state name="open" group="gripper"><joint name="pinch" value="0.04"/></group_state>
  <end_effector name="ee" parent_link="grasp" group="gripper" parent_group="arm"/>
</robot>"#;

/// The one triangle of `mini`'s hand mesh, before the URDF's mirror.
pub const MINI_HAND_TRIANGLE: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]];

/// The meshes of `mini`, by file name, as robot_identity gives them.
pub fn mini_meshes() -> Vec<(String, Vec<u8>)> {
    vec![("hand.stl".to_string(), stl_bytes(&[MINI_HAND_TRIANGLE]))]
}

/// The body of `mini`.
pub fn mini() -> RobotBody {
    let meshes = mini_meshes();
    let meshes: Vec<(&str, &[u8])> = meshes
        .iter()
        .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
        .collect();
    RobotBody::parse(MINI_URDF, MINI_SRDF, &meshes).expect("mini's description")
}
