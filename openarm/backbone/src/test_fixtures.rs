//! Shared test geometry: the obstacle fixtures both the governor's own tests
//! and the coordinator's request tests are written against.
//!
//! They live here rather than in either test module because the two describe
//! the *same* wall standing in front of the *same* robot. Written twice, the
//! copies could drift onto different geometry while both kept passing, and the
//! pair of them would no longer be saying anything about each other.

use bimanual_collision_model::{Obstacle, Point3};

/// The tolerance the fixtures fit at. Boxes hull to their corners at any
/// tolerance, so the value only has to be inside the accepted band.
pub(crate) const FIXTURE_TOLERANCE_M: f64 = 0.001;

/// The eight corners of an axis-aligned box, the cloud an operator sends for a
/// wall.
pub(crate) fn box_points(min: [f64; 3], max: [f64; 3]) -> Vec<Point3<f64>> {
    let mut points = Vec::with_capacity(8);
    for x in [min[0], max[0]] {
        for y in [min[1], max[1]] {
            for z in [min[2], max[2]] {
                points.push(Point3::new(x, y, z));
            }
        }
    }
    points
}

pub(crate) fn obstacle(name: &str, min: [f64; 3], max: [f64; 3]) -> Obstacle {
    Obstacle::fit(name, &box_points(min, max), FIXTURE_TOLERANCE_M).expect("a box bounds a solid")
}

/// A wall standing in front of the robot: clear of both arms at home, and
/// squarely in the way of a forward elbow bend.
pub(crate) fn front_wall() -> Obstacle {
    obstacle("wall", [0.25, -0.9, -0.2], [0.9, 0.9, 1.2])
}

/// A box around the whole robot, so the arms start inside it and no
/// configuration can admit it.
pub(crate) fn swallowing_box(name: &str) -> Obstacle {
    obstacle(name, [-2.0; 3], [2.0; 3])
}
