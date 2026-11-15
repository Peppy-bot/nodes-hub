//! The motion planner's voxel map: its occupied cubes in the robot frame, and a text of
//! what it holds.

use crate::canvas::{Canvas, Rgba};
use crate::model::planner_map::PlannerMap;

pub const MAP_PATH: &str = "world/planner/map";
pub const MAP_STATE_PATH: &str = "planner/map";

const CUBE_COLOR: Rgba = [120, 120, 255, 90];

/// Draws `map` at the time the planner read it: its cubes, or none, and its text.
pub fn draw_planner_map(canvas: &dyn Canvas, map: &PlannerMap) {
    canvas.set_time(map.at);
    if map.centers.is_empty() {
        canvas.clear(MAP_PATH);
    } else {
        let side = map.voxel_size.get() as f32;
        canvas.cubes(MAP_PATH, &map.centers, side, CUBE_COLOR);
    }
    canvas.text(MAP_STATE_PATH, &map_state(map));
}

/// What `map` holds, in words.
pub fn map_state(map: &PlannerMap) -> String {
    let held = format!(
        "The planner's map holds {} occupied cubes of {} m.",
        map.centers.len(),
        map.voxel_size.get()
    );
    if map.waiting.is_empty() {
        return held;
    }
    format!("{held} It holds no depth frame: {}.", map.waiting)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use super::*;
    use crate::testing::{Drawn, FakeCanvas};

    fn map(centers: &[f64], waiting: &str) -> PlannerMap {
        let at = SystemTime::UNIX_EPOCH + Duration::from_secs(4);
        PlannerMap::parse(at, 0.02, centers, waiting).expect("a map")
    }

    #[test]
    fn the_cubes_take_the_side_of_the_map_at_the_time_it_was_read() {
        let canvas = FakeCanvas::default();
        draw_planner_map(&canvas, &map(&[0.31, 0.01, 0.51], ""));
        let drawn = canvas.drawn();
        assert!(
            matches!(drawn[0], Drawn::Time(at) if at == SystemTime::UNIX_EPOCH + Duration::from_secs(4))
        );
        assert!(matches!(
            &drawn[1],
            Drawn::Cubes { path, centers, side, .. }
                if path == MAP_PATH && centers == &[[0.31, 0.01, 0.51]] && *side == 0.02
        ));
        assert!(matches!(
            &drawn[2],
            Drawn::Text { path, markdown }
                if path == MAP_STATE_PATH
                    && markdown == "The planner's map holds 1 occupied cubes of 0.02 m."
        ));
    }

    #[test]
    fn a_map_without_cubes_clears_them_and_says_why_it_waits() {
        let canvas = FakeCanvas::default();
        draw_planner_map(&canvas, &map(&[], "no depth frame has reached the map yet"));
        assert!(matches!(&canvas.at(MAP_PATH)[..], [Drawn::Clear { .. }]));
        assert!(matches!(
            &canvas.at(MAP_STATE_PATH)[..],
            [Drawn::Text { markdown, .. }] if markdown == "The planner's map holds 0 occupied \
                cubes of 0.02 m. It holds no depth frame: no depth frame has reached the map yet."
        ));
    }
}
