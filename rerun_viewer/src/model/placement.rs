//! Where a frame sits in another, as the contracts carry it: a position in meters and a
//! unit quaternion [x, y, z, w].

use control_core::geometry::{GeometryError, Pose};
use frame_tree::Transform;

/// A position and a unit quaternion, kept as they came so that two placements compare
/// equal when the wire gave the same numbers, with the pose they make.
#[derive(Debug, Clone, Copy)]
pub struct Placement {
    pub position: [f64; 3],
    pub orientation: [f64; 4],
    pub pose: Transform,
}

impl PartialEq for Placement {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position && self.orientation == other.orientation
    }
}

impl Placement {
    /// The placement of `position` and `orientation`, a unit quaternion [x, y, z, w], parsed
    /// as control_core parses a pose of the contracts.
    pub fn parse(position: [f64; 3], orientation: [f64; 4]) -> Result<Self, GeometryError> {
        let pose = Pose::parse(position, orientation)?.transform();
        Ok(Self {
            position,
            orientation,
            pose,
        })
    }
}

/// The pose with no turn whose origin is `position`.
pub fn translation(position: [f64; 3]) -> Transform {
    Transform::new(position, [0.0, 0.0, 0.0, 1.0])
        .expect("a finite position and the unit quaternion")
}

/// The columns of the rotation of `pose`: where it takes the unit vectors x, y and z.
pub fn rotation_columns(pose: &Transform) -> [[f64; 3]; 3] {
    let origin = pose.origin();
    let unit = |axis: usize| {
        let mut vector = [0.0; 3];
        vector[axis] = 1.0;
        let moved = pose.apply(vector);
        [
            moved[0] - origin[0],
            moved[1] - origin[1],
            moved[2] - origin[2],
        ]
    };
    [unit(0), unit(1), unit(2)]
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// True when each coordinate of `a` is within `tolerance` of `b`'s.
    pub(crate) fn close(a: [f64; 3], b: [f64; 3], tolerance: f64) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tolerance)
    }

    /// A quarter turn about +z: x goes to y.
    pub(crate) fn quarter_turn_about_z() -> [f64; 4] {
        let half = std::f64::consts::FRAC_PI_4;
        [0.0, 0.0, half.sin(), half.cos()]
    }

    #[test]
    fn two_placements_of_the_same_numbers_are_equal() {
        let a = Placement::parse([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]).unwrap();
        let b = Placement::parse([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]).unwrap();
        let moved = Placement::parse([1.0, 2.0, 3.5], [0.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, moved);
    }

    #[test]
    fn a_quaternion_that_is_not_of_unit_length_makes_no_placement() {
        assert!(Placement::parse([0.0; 3], [0.0, 0.0, 0.0, 2.0]).is_err());
        assert!(Placement::parse([f64::NAN, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]).is_err());
    }

    #[test]
    fn the_columns_of_a_quarter_turn_about_z_take_x_to_y() {
        let pose = Transform::new([5.0, 6.0, 7.0], quarter_turn_about_z()).unwrap();
        let [x, y, z] = rotation_columns(&pose);
        assert!(close(x, [0.0, 1.0, 0.0], 1e-12));
        assert!(close(y, [-1.0, 0.0, 0.0], 1e-12));
        assert!(close(z, [0.0, 0.0, 1.0], 1e-12));
    }
}
