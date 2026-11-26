//! Poses of the floor plane through frame_tree's transforms: where a transform stands on
//! the floor.

use control_core::planar_pose::PlanarPose;
use frame_tree::Transform;

/// Where `pose` stands on the floor: its origin's x and y, and the heading of its
/// orientation about +Z, as tf2's `getYaw` reads it; none for an origin past the finite
/// numbers, where a composition of two far poses can leave it.
pub fn on_floor(pose: &Transform) -> Option<PlanarPose> {
    let [x, y, _] = pose.position();
    PlanarPose::new(x, y, pose.orientation().yaw()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use frame_tree::UnitQuaternion;
    use hegel::TestCase;
    use hegel::generators as gs;
    use std::f64::consts::PI;

    // A lidar tilted about its own +X on the base stands on the floor where its mount puts
    // it, at the base's heading: the tilt leaves its heading.
    #[hegel::test]
    fn a_tilted_lidar_stands_on_the_floor_at_its_heading(tc: TestCase) {
        let coordinate = || gs::floats().min_value(-10.0).max_value(10.0);
        let (x, y): (f64, f64) = (tc.draw(coordinate()), tc.draw(coordinate()));
        let yaw: f64 = tc.draw(gs::floats().min_value(-3.0).max_value(3.0));
        let tilt: f64 = tc.draw(gs::floats().min_value(-1.0).max_value(1.0));
        let (sin, cos) = (tilt / 2.0).sin_cos();
        let heading = UnitQuaternion::from_yaw(yaw).expect("a finite yaw");
        let base = Transform::new([x, y, 0.0], heading).expect("a transform");
        let tilt = UnitQuaternion::parse([sin, 0.0, 0.0, cos]).expect("a turn about +X");
        let tilted = Transform::new([0.0, 0.0, 0.4], tilt).expect("a transform");
        let floor = on_floor(&base.compose(&tilted)).expect("a pose on the floor");
        assert!((floor.x() - x).abs() < 1e-9 && (floor.y() - y).abs() < 1e-9);
        let turn = (floor.yaw() - yaw + PI).rem_euclid(2.0 * PI) - PI;
        assert!(turn.abs() < 1e-9, "{floor:?} at {yaw}");
    }
}
