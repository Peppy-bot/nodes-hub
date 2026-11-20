//! The geometry of a differential base: headings wrapped into (-pi, pi], a
//! pose of the robot frame in a frame fixed to the floor, and a frame placed
//! in its parent's as a URDF origin places a link.

use std::f64::consts::{PI, TAU};

/// `angle` (rad) wrapped into (-pi, pi].
pub fn wrap_angle(angle: f64) -> f64 {
    let turned = angle.rem_euclid(TAU);
    if turned > PI { turned - TAU } else { turned }
}

/// The robot frame placed in a frame fixed to the floor: its origin (m) and
/// its heading (rad), counterclockwise seen from above.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose2 {
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
}

impl Pose2 {
    /// This pose seen from `origin`, a pose of the same frame: the goal's
    /// frame is the robot frame where the base stands when the loop starts
    /// the goal, right after its acceptance.
    pub fn seen_from(self, origin: Pose2) -> Pose2 {
        let (dx, dy) = (self.x - origin.x, self.y - origin.y);
        let (sin, cos) = origin.yaw.sin_cos();
        Pose2 {
            x: cos * dx + sin * dy,
            y: -sin * dx + cos * dy,
            yaw: wrap_angle(self.yaw - origin.yaw),
        }
    }

    /// The distance (m) from this pose's origin to `target`'s.
    pub fn distance_to(self, target: Pose2) -> f64 {
        (target.x - self.x).hypot(target.y - self.y)
    }

    /// The shorter angle (rad) from this pose's heading to `target`'s.
    pub fn yaw_to(self, target: Pose2) -> f64 {
        wrap_angle(target.yaw - self.yaw)
    }
}

/// A unit quaternion [x, y, z, w].
pub type Quaternion = [f64; 4];

/// The orientation a URDF origin's roll, pitch and yaw (rad) give: roll
/// about x, then pitch about y, then yaw about z, each about the parent's
/// fixed axes.
pub fn rpy_orientation([roll, pitch, yaw]: [f64; 3]) -> Quaternion {
    let (sr, cr) = (roll / 2.0).sin_cos();
    let (sp, cp) = (pitch / 2.0).sin_cos();
    let (sy, cy) = (yaw / 2.0).sin_cos();
    [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]
}

/// The rotation `a` after `b`: a * b.
fn quaternion_product(a: Quaternion, b: Quaternion) -> Quaternion {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// `vector` turned by `q`.
pub fn rotate(q: Quaternion, vector: [f64; 3]) -> [f64; 3] {
    let [x, y, z, _] = quaternion_product(
        quaternion_product(q, [vector[0], vector[1], vector[2], 0.0]),
        [-q[0], -q[1], -q[2], q[3]],
    );
    [x, y, z]
}

/// A frame placed in its parent's: its origin (m) and its orientation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub position: [f64; 3],
    pub orientation: Quaternion,
}

impl Frame {
    /// The parent's own frame.
    pub const IDENTITY: Frame = Frame {
        position: [0.0; 3],
        orientation: [0.0, 0.0, 0.0, 1.0],
    };

    /// The frame a URDF origin places in its parent's: at `xyz`, turned by
    /// `rpy` ([`rpy_orientation`]).
    pub fn of_origin(xyz: [f64; 3], rpy: [f64; 3]) -> Frame {
        Frame {
            position: xyz,
            orientation: rpy_orientation(rpy),
        }
    }

    /// `point`, given in this frame, in the parent's.
    pub fn place(self, point: [f64; 3]) -> [f64; 3] {
        let [x, y, z] = rotate(self.orientation, point);
        let [px, py, pz] = self.position;
        [px + x, py + y, pz + z]
    }

    /// `inner`, a frame placed in this one, placed in this one's parent.
    pub fn then(self, inner: Frame) -> Frame {
        Frame {
            position: self.place(inner.position),
            orientation: quaternion_product(self.orientation, inner.orientation),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use hegel::TestCase;
    use hegel::generators as gs;

    fn assert_close(actual: [f64; 3], expected: [f64; 3]) {
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-12, "{actual:?} != {expected:?}");
        }
    }

    #[hegel::test]
    fn a_wrapped_angle_lies_in_the_half_open_turn_and_keeps_its_direction(tc: TestCase) {
        let angle: f64 = tc.draw(gs::floats().min_value(-100.0).max_value(100.0));
        let wrapped = wrap_angle(angle);
        assert!(wrapped > -PI && wrapped <= PI, "{angle} wraps to {wrapped}");
        assert!((wrapped.sin() - angle.sin()).abs() < 1e-9);
        assert!((wrapped.cos() - angle.cos()).abs() < 1e-9);
    }

    #[test]
    fn a_half_turn_wraps_to_plus_pi() {
        assert_eq!(wrap_angle(PI), PI);
        assert_eq!(wrap_angle(-PI), PI);
    }

    #[test]
    fn a_pose_seen_from_a_turned_origin_is_in_that_origin_frame() {
        let origin = Pose2 {
            x: 1.0,
            y: 2.0,
            yaw: PI / 2.0,
        };
        let ahead = Pose2 {
            x: 1.0,
            y: 3.0,
            yaw: PI,
        };
        let seen = ahead.seen_from(origin);
        assert!(
            (seen.x - 1.0).abs() < 1e-12 && seen.y.abs() < 1e-12,
            "{seen:?}"
        );
        assert!((seen.yaw - PI / 2.0).abs() < 1e-12);
    }

    // A URDF origin turns roll about x, then pitch about y, then yaw about z,
    // each about the parent's axes: a quarter turn of yaw after a quarter turn
    // of roll takes the child's y to the parent's z and its x to the parent's y.
    #[test]
    fn an_origin_turns_roll_then_pitch_then_yaw_about_the_parents_axes() {
        let q = rpy_orientation([PI / 2.0, 0.0, PI / 2.0]);
        assert_close(rotate(q, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]);
        assert_close(rotate(q, [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
        let pitched = rpy_orientation([0.0, PI / 2.0, 0.0]);
        assert_close(rotate(pitched, [1.0, 0.0, 0.0]), [0.0, 0.0, -1.0]);
    }

    // A frame placed through two origins places a point as the two origins
    // one after the other do, and the identity places it where it is.
    #[hegel::test]
    fn a_frame_then_another_places_a_point_as_both_in_turn(tc: TestCase) {
        let draw = || tc.draw(gs::floats().min_value(-3.0).max_value(3.0));
        let outer = Frame::of_origin([draw(), draw(), draw()], [draw(), draw(), draw()]);
        let inner = Frame::of_origin([draw(), draw(), draw()], [draw(), draw(), draw()]);
        let point = [draw(), draw(), draw()];
        let [a, b, c] = outer.then(inner).place(point);
        let [x, y, z] = outer.place(inner.place(point));
        assert!(
            (a - x).abs() < 1e-9 && (b - y).abs() < 1e-9 && (c - z).abs() < 1e-9,
            "{:?} != {:?}",
            [a, b, c],
            [x, y, z]
        );
        assert_eq!(Frame::IDENTITY.place(point), point);
    }
}
