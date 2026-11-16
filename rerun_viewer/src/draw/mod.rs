//! How each value of the model is drawn: the entities of the viewer and the canvas calls
//! that make them. The drawers know the canvas and the model only.
//!
//! The entities, by path:
//! - `world`: the robot frame of limb_motion (x forward, y left, z up, meters).
//! - `world/arms/<arm>/grasp`: each arm's grasp point, with its axes; the objects the
//!   arm carries are drawn under it, in `attached/<id>`.
//! - `world/arms/<arm>/trail`: the past positions of the grasp point.
//! - `world/robot/<link>`: each link of the robot's description that has collision
//!   shapes, where the joints of the last snapshot put it; its shapes are drawn under
//!   it, in `collision_<index>_<kind>`.
//! - `world/camera/points`: the camera's depth now, in the robot frame.
//! - `world/planner/map`: the occupied cubes of the motion planner's voxel map, and
//!   `planner/map` what the map holds.
//! - `world/scene/objects/<id>` and `world/scene/shells/<id>`: the planning scene.
//! - `camera/color` and `camera/depth`: the camera's images.
//! - `plots/joints/<arm>`: the positions of the arm's joints, one line for each joint; and
//!   `plots/grippers`: the grippers' openings, one line for each gripper.
//! - `scene/changes` and `scene/allowed_pairs`: the log of the scene and its pairs.
//! - `notes/cubes`: what the cubes and the points are.

pub mod arms;
pub mod camera;
pub mod planner_map;
pub mod robot;
pub mod scene;

use crate::canvas::{Canvas, PageLayout};
use crate::model::scene::PlacedShape;

/// The robot frame: the root of everything drawn in 3D.
pub const WORLD_PATH: &str = "world";

/// The page's layout over the entities of the drawers; the depth image has a view only
/// when the node draws it.
pub fn page_layout(depth_image: bool) -> PageLayout {
    PageLayout {
        world: WORLD_PATH,
        color_image: camera::COLOR_PATH,
        depth_image: depth_image.then_some(camera::DEPTH_PATH),
        joint_plots: arms::JOINT_PLOTS_PATH,
        gripper_plot: arms::GRIPPERS_PATH,
        scene_changes: scene::CHANGES_PATH,
        map_state: planner_map::MAP_STATE_PATH,
        allowed_pairs: scene::ALLOWED_PAIRS_PATH,
        note: CUBES_NOTE_PATH,
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_depth_image_has_a_view_only_when_the_node_draws_it() {
        assert_eq!(page_layout(false).depth_image, None);
        assert_eq!(page_layout(true).depth_image, Some(camera::DEPTH_PATH));
    }
}
