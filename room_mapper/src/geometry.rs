//! Poses of the floor plane through frame_tree's transforms: where a transform stands on
//! the floor, and planar poses composed and undone.

use control_core::planar_pose::PlanarPose;
use frame_tree::Transform;
use frame_tree::arithmetic::yaw_of_quaternion;

/// Where `pose` stands on the floor: its origin's x and y, and the heading of its
/// orientation about +Z, as tf2's `getYaw` reads it.
pub fn on_floor(pose: &Transform) -> PlanarPose {
    let [x, y, _] = pose.origin();
    PlanarPose::new(x, y, yaw_of_quaternion(pose.orientation()))
        .expect("a transform's origin and heading are finite")
}

/// `inner`, a pose of the frame that `outer` places, in the frame `outer` is placed in.
pub fn compose(outer: PlanarPose, inner: PlanarPose) -> PlanarPose {
    on_floor(&outer.to_transform().compose(&inner.to_transform()))
}

/// The pose that undoes `pose`: the frame that places `pose`, seen from `pose`.
pub fn inverse(pose: PlanarPose) -> PlanarPose {
    on_floor(&pose.to_transform().inverse())
}

#[cfg(test)]
mod tests {
    use super::*;
    use frame_tree::arithmetic::quaternion_of_yaw;
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
        let base = Transform::new([x, y, 0.0], quaternion_of_yaw(yaw)).expect("a transform");
        let tilted = Transform::new([0.0, 0.0, 0.4], [sin, 0.0, 0.0, cos]).expect("a transform");
        let floor = on_floor(&base.compose(&tilted));
        assert!((floor.x() - x).abs() < 1e-9 && (floor.y() - y).abs() < 1e-9);
        let turn = (floor.yaw() - yaw + PI).rem_euclid(2.0 * PI) - PI;
        assert!(turn.abs() < 1e-9, "{floor:?} at {yaw}");
    }

    #[test]
    fn a_composed_pose_undone_is_the_inner_pose() {
        let outer = PlanarPose::new(1.0, -2.0, 0.7).expect("a pose");
        let inner = PlanarPose::new(0.3, 0.4, -1.2).expect("a pose");
        let back = compose(inverse(outer), compose(outer, inner));
        assert!((back.x() - inner.x()).abs() < 1e-12 && (back.y() - inner.y()).abs() < 1e-12);
        assert!((back.yaw() - inner.yaw()).abs() < 1e-12, "{back:?}");
    }
}
