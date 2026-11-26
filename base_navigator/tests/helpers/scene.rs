//! The world the rig's sensors render: boxes on the floor in the world frame,
//! boxes turned about the vertical such as the base's own deck, tables, slabs
//! of floor around a drop; a pinhole depth camera that renders the depth at
//! each pixel as the distance along its axis to the first box its ray enters;
//! and a 2-D lidar's fan, each ray's range the distance along it to the first
//! box it enters. The world frame is where the base's odometry started at the
//! rig's start, so it is odom until a test starts the odometry over.

use camera_core::laser_scan::{NO_RETURN_MARK_M, UNMEASURED_MARK_M};
use frame_tree::arithmetic::{Quaternion, product, quaternion_of_yaw, rotate};

/// A rigid motion of space: a turn by a unit quaternion, then a shift.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose3 {
    pub position: [f64; 3],
    pub orientation: Quaternion,
}

impl Pose3 {
    /// The robot frame on the floor at (`x`, `y`), turned `yaw` about +z.
    pub fn planar(x: f64, y: f64, yaw: f64) -> Self {
        Self {
            position: [x, y, 0.0],
            orientation: quaternion_of_yaw(yaw),
        }
    }

    /// `inner` placed in this pose: a point goes through `inner` first.
    pub fn compose(&self, inner: &Pose3) -> Pose3 {
        Pose3 {
            position: self.apply(inner.position),
            orientation: product(self.orientation, inner.orientation),
        }
    }

    /// `point`, given in this pose's frame, in the frame this pose is in.
    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        let turned = rotate(self.orientation, point);
        std::array::from_fn(|i| self.position[i] + turned[i])
    }

    /// `vector`, given in this pose's frame, turned into the frame this pose
    /// is in.
    pub fn turn(&self, vector: [f64; 3]) -> [f64; 3] {
        rotate(self.orientation, vector)
    }
}

/// An axis-aligned box from `min` to `max` (m), in the world frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box3 {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Box3 {
    pub const fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }

    /// Whether the floor point `point` lies inside the box's outline.
    pub fn covers(&self, point: [f64; 2]) -> bool {
        (0..2).all(|i| self.min[i] <= point[i] && point[i] <= self.max[i])
    }

    /// The distance along `direction` from `origin` at which the ray enters
    /// the box, when it enters it ahead of the origin.
    fn entry(&self, origin: [f64; 3], direction: [f64; 3]) -> Option<f64> {
        let mut near = f64::NEG_INFINITY;
        let mut far = f64::INFINITY;
        let axes = origin
            .iter()
            .zip(&direction)
            .zip(self.min.iter().zip(&self.max));
        for ((&from, &along), (&low, &high)) in axes {
            if along == 0.0 {
                if from < low || from > high {
                    return None;
                }
                continue;
            }
            let (a, b) = ((low - from) / along, (high - from) / along);
            near = near.max(a.min(b));
            far = far.min(a.max(b));
        }
        (near <= far && near > 0.0).then_some(near)
    }
}

/// A box turned by `yaw` (rad) about the vertical through its `center`, of
/// half sizes `half` (m), in the world frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurnedBox {
    pub center: [f64; 3],
    pub half: [f64; 3],
    pub yaw: f64,
}

impl TurnedBox {
    /// The distance along `direction` from `origin` at which the ray enters
    /// the box, when it enters it ahead of the origin.
    fn entry(&self, origin: [f64; 3], direction: [f64; 3]) -> Option<f64> {
        let (sin, cos) = (-self.yaw).sin_cos();
        let turn = |v: [f64; 3]| [cos * v[0] - sin * v[1], sin * v[0] + cos * v[1], v[2]];
        let local = turn(std::array::from_fn(|i| origin[i] - self.center[i]));
        let half = self.half;
        Box3::new([-half[0], -half[1], -half[2]], half).entry(local, turn(direction))
    }
}

/// How far the floor reaches from the world's origin on each side (m), past
/// every camera's farthest depth from anywhere a test drives.
const FLOOR_REACH_M: f64 = 30.0;

/// The floor under the rig: its top at z 0.
pub const FLOOR: Box3 = Box3::new(
    [-FLOOR_REACH_M, -FLOOR_REACH_M, -1.0],
    [FLOOR_REACH_M, FLOOR_REACH_M, 0.0],
);

/// The boxes the cameras render.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    pub boxes: Vec<Box3>,
}

impl Scene {
    /// The floor alone.
    pub fn open() -> Self {
        Self { boxes: vec![FLOOR] }
    }

    /// The floor and `boxes` on it.
    pub fn with(boxes: &[Box3]) -> Self {
        Self {
            boxes: [&[FLOOR][..], boxes].concat(),
        }
    }

    /// The distance along `direction` from `origin` at which the ray first
    /// enters a box of the scene or of `turned`; infinite for a ray that enters
    /// none.
    pub fn first_entry(&self, turned: &[TurnedBox], origin: [f64; 3], direction: [f64; 3]) -> f64 {
        self.boxes
            .iter()
            .filter_map(|b| b.entry(origin, direction))
            .chain(turned.iter().filter_map(|b| b.entry(origin, direction)))
            .fold(f64::INFINITY, f64::min)
    }

    /// A table whose top spans from `min` to `max` of the floor plane at
    /// `top_m` up, 0.05 m thick, on four legs 0.05 m square at its corners.
    pub fn table(min: [f64; 2], max: [f64; 2], top_m: f64) -> Vec<Box3> {
        let leg =
            |x: f64, y: f64| Box3::new([x - 0.025, y - 0.025, 0.0], [x + 0.025, y + 0.025, top_m]);
        vec![
            Box3::new([min[0], min[1], top_m], [max[0], max[1], top_m + 0.05]),
            leg(min[0], min[1]),
            leg(min[0], max[1]),
            leg(max[0], min[1]),
            leg(max[0], max[1]),
        ]
    }

    /// The floor with a pit `depth_m` deep over the rectangle from `min` to
    /// `max` (m) of the floor plane: four slabs of floor around it and the
    /// pit's own floor below.
    pub fn with_pit(min: [f64; 2], max: [f64; 2], depth_m: f64) -> Self {
        let r = FLOOR_REACH_M;
        let slab =
            |x: [f64; 2], y: [f64; 2], top: f64| Box3::new([x[0], y[0], -1.0], [x[1], y[1], top]);
        Self {
            boxes: vec![
                slab([-r, min[0]], [-r, r], 0.0),
                slab([max[0], r], [-r, r], 0.0),
                slab([min[0], max[0]], [-r, min[1]], 0.0),
                slab([min[0], max[0]], [max[1], r], 0.0),
                slab([min[0], max[0]], [min[1], max[1]], -depth_m),
            ],
        }
    }
}

/// The outline on the floor of the pit from `min` to `max`, as
/// [`Scene::with_pit`] digs it.
pub fn pit(min: [f64; 2], max: [f64; 2]) -> Box3 {
    Box3::new([min[0], min[1], -1.0], [max[0], max[1], 0.0])
}

/// A pinhole depth camera: its image size and lens (pixels), its unit (m a
/// count) and the farthest depth it reports (m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthLens {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub unit_m: f32,
    pub max_depth_m: f64,
}

impl DepthLens {
    /// The z16 frame the camera whose optical frame stands at `optical` in
    /// the world renders of `scene` and `turned`, row by row: a pixel whose
    /// ray enters no box within the farthest depth reads 0.
    pub fn render(&self, optical: &Pose3, scene: &Scene, turned: &[TurnedBox]) -> Vec<u8> {
        let origin = optical.position;
        (0..self.height)
            .flat_map(|v| (0..self.width).map(move |u| (u, v)))
            .flat_map(|(u, v)| {
                // The pixel's ray at unit depth: the distance along it is the
                // depth along the camera's axis.
                let direction = optical.turn([
                    (f64::from(u) - self.cx) / self.fx,
                    (f64::from(v) - self.cy) / self.fy,
                    1.0,
                ]);
                let depth = scene.first_entry(turned, origin, direction);
                let sample = if depth <= self.max_depth_m {
                    (depth / f64::from(self.unit_m)).round() as u16
                } else {
                    0
                };
                sample.to_le_bytes()
            })
            .collect()
    }
}

/// A 2-D lidar's fan, as lidar_2d's scan gives it: `rays` rays from
/// `min_angle_rad` (rad) by `angle_increment_rad`, `time_increment_s` (s) from
/// one ray to the next, below 0 for a lidar that sweeps clockwise, and the
/// nearest and farthest ranges it measures (m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LidarLens {
    pub min_angle_rad: f32,
    pub angle_increment_rad: f32,
    pub rays: usize,
    pub time_increment_s: f32,
    pub min_range_m: f32,
    pub max_range_m: f32,
}

impl LidarLens {
    /// Ray `index`'s angle in the lidar's frame (rad), as lidar_2d places it.
    pub fn angle_rad(&self, index: usize) -> f64 {
        f64::from(self.min_angle_rad) + index as f64 * f64::from(self.angle_increment_rad)
    }

    /// Ray `index`'s time after the scan's timestamp (s), before it for a
    /// clockwise sweep.
    pub fn time_s(&self, index: usize) -> f64 {
        f64::from(self.time_increment_s) * index as f64
    }

    /// The range the lidar measures along the ray from `origin` along
    /// `direction`, a unit vector, of `scene` and `turned`: the distance to the
    /// first box the ray enters, lidar_2d's 0 nearer than the nearest range,
    /// and its 10000 at or past the farthest.
    pub fn range(
        &self,
        scene: &Scene,
        turned: &[TurnedBox],
        origin: [f64; 3],
        direction: [f64; 3],
    ) -> f32 {
        let distance = scene.first_entry(turned, origin, direction);
        if distance >= f64::from(self.max_range_m) {
            return NO_RETURN_MARK_M;
        }
        if distance < f64::from(self.min_range_m) {
            return UNMEASURED_MARK_M;
        }
        distance as f32
    }
}
