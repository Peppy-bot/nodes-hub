//! How each value of the model is drawn: the entities of the viewer and the canvas calls
//! that make them. The drawers know the canvas and the model only.
//!
//! The entities, by path:
//! - `world`: the robot frame of limb_motion (x forward, y left, z up, meters).
//! - `world/arms/<arm>/grasp`: each arm's grasp point, with its axes; the objects the
//!   arm carries are drawn under it, in `attached/<id>`.
//! - `world/arms/<arm>/trail`: the past positions of the grasp point.
//! - `world/camera/points`: the camera's depth now, in the robot frame.
//! - `world/planner/map`: the occupied cubes of the motion planner's voxel map, and
//!   `planner/map` what the map holds.
//! - `world/scene/objects/<id>` and `world/scene/shells/<id>`: the planning scene.
//! - `camera/color` and `camera/depth`: the camera's images.
//! - `plots/joints/<arm>/<joint>` and `plots/grippers/<gripper>`: the plots.
//! - `scene/changes` and `scene/allowed_pairs`: the log of the scene and its pairs.
//! - `notes/cubes`: what the cubes and the points are.

pub mod arms;
pub mod camera;
pub mod planner_map;
pub mod scene;

use crate::canvas::Canvas;
use crate::model::scene::PlacedShape;

/// The path of the note on the cubes.
pub const CUBES_NOTE_PATH: &str = "notes/cubes";

/// What the note on the cubes says.
pub const CUBES_NOTE: &str = "The cubes in `world/planner/map` are the motion planner's voxel \
map as it publishes it on occupancy_map, by default once a second: the occupied cubes that its \
plans, Cartesian paths and checks keep their clearance from, cubes the camera cannot see now \
included. `planner/map` says what the map holds. The points in `world/camera/points` are the \
camera's depth now.";

/// The entity of part `index` (such as a shape or its shell) of `id` under `parent`,
/// named with the part's kind: a part that changes kind takes a new entity, and the
/// entity of the old kind is cleared.
pub fn part_path(parent: &str, id: &str, part: &str, index: usize, shape: &PlacedShape) -> String {
    format!("{parent}/{id}/{part}_{index}_{}", shape.kind.name())
}

/// Draws the notes that hold for the whole recording.
pub fn draw_notes(canvas: &dyn Canvas) {
    canvas.note(CUBES_NOTE_PATH, CUBES_NOTE);
}
