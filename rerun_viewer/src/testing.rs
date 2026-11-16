//! The canvas of the tests: it records each call, and a test waits until what it
//! expects is drawn, woken by each draw rather than by a clock. Also the small robot
//! that the tests describe, with numbers a reader can check by hand.

use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use tokio::sync::watch;

use crate::drawn::{Drawn, DrawnCanvas};
use crate::model::robot::RobotBody;

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

impl DrawnCanvas for FakeCanvas {
    fn take(&self, call: Drawn) {
        self.record(call);
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
