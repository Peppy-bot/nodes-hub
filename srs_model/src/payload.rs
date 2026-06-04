//! The rigid mass distal to the SRS wrist: gripper body, fingers, tools, or any
//! fixed end-effector. The 7-DOF SRS chain ends at the wrist for IK and FK, but
//! gravity and Coriolis must still carry whatever hangs off it.
//!
//! Every link past the wrist is lumped into one rigid body, with movable distal
//! joints (e.g. gripper fingers) frozen at the URDF home pose. A set of bodies
//! that share no relative motion *is* one rigid body, so the lump is exact for
//! the frozen configuration; the only approximation is ignoring finger travel,
//! which is second-order and cancels for a symmetric gripper.
//!
//! The lump is expressed in the tip-link frame and, at load, folded straight
//! into the last segment's inertial ([`Payload::combined_with`]): the payload is
//! rigidly attached to `link7`, so a bigger last link and a separate payload are
//! the same rigid body. Gravity / Coriolis then carry it as part of that segment.

use k::Node;
use k::nalgebra::{Isometry3, Matrix3, Point3, Vector3};

/// Lumped distal rigid body, in the tip-link frame. `inertia` is about `com`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Payload {
    pub mass: f64,
    pub com: Vector3<f64>,
    pub inertia: Matrix3<f64>,
}

impl Payload {
    /// No distal mass. Folding it into a segment leaves the segment unchanged.
    pub(crate) fn none() -> Self {
        Self { mass: 0.0, com: Vector3::zeros(), inertia: Matrix3::zeros() }
    }

    /// Lump every link distal to `tip` into one rigid body in the tip frame. The
    /// full chain must already be posed at home (`update_transforms`), so the
    /// frozen distal link transforms are read straight off the cached poses.
    pub(crate) fn from_distal(tip: &Node<f64>) -> Self {
        let tip_world = tip.world_transform().expect("tip world transform");
        let tip_inv = tip_world.inverse();

        // (mass, COM in tip frame, inertia about that COM in tip frame).
        let mut bodies: Vec<(f64, Vector3<f64>, Matrix3<f64>)> = Vec::new();
        for node in tip.iter_descendants() {
            // `iter_descendants` includes `tip` itself; it is the last SRS
            // segment, already accounted for, not distal mass.
            if node == *tip {
                continue;
            }
            if let Some(body) = distal_body_in_tip(&node, &tip_inv) {
                bodies.push(body);
            }
        }
        compose(&bodies)
    }

    /// Combine this payload with a segment's rigid body, both in the same (tip)
    /// frame with each `inertia` about its own COM, returning the merged body.
    /// Used at load to fold the payload into the last segment's inertial.
    pub(crate) fn combined_with(
        &self,
        mass: f64,
        com: Vector3<f64>,
        inertia: Matrix3<f64>,
    ) -> Payload {
        compose(&[(self.mass, self.com, self.inertia), (mass, com, inertia)])
    }
}

/// One distal node's rigid body, expressed in the tip frame:
/// `(mass, COM, inertia about the COM)`, or `None` if the node is link-less or
/// massless. The link guard and `world_transform()` both lock the node's
/// non-reentrant mutex, so reading them is kept here, in order (guard dropped at
/// the end of the inner block, before the transform), where it cannot overlap and
/// deadlock.
fn distal_body_in_tip(
    node: &Node<f64>,
    tip_inv: &Isometry3<f64>,
) -> Option<(f64, Vector3<f64>, Matrix3<f64>)> {
    let (mass, com_in_link, inertia_in_link) = {
        let guard = node.link();
        let link = guard.as_ref()?;
        let mass = link.inertial.mass;
        if mass == 0.0 {
            return None;
        }
        // Rotate the COM inertia by the inertial's own rpy into the link frame.
        let r = *link.inertial.origin().rotation.to_rotation_matrix().matrix();
        (
            mass,
            link.inertial.origin().translation.vector,
            r * link.inertial.inertia * r.transpose(),
        )
    };

    // Re-express the frozen distal link in the tip frame (constant at home).
    let link_in_tip = tip_inv * node.world_transform().expect("distal world transform");
    let com = link_in_tip.transform_point(&Point3::from(com_in_link)).coords;
    let r = *link_in_tip.rotation.to_rotation_matrix().matrix();
    Some((mass, com, r * inertia_in_link * r.transpose()))
}

/// Combine several rigid bodies into one: total mass, mass-weighted COM, and the
/// inertia about that COM via the parallel-axis theorem. Each body's `inertia`
/// is already about its own COM, in the shared (tip) frame.
fn compose(bodies: &[(f64, Vector3<f64>, Matrix3<f64>)]) -> Payload {
    let mass: f64 = bodies.iter().map(|(m, _, _)| *m).sum();
    if mass == 0.0 {
        return Payload::none();
    }
    let com = bodies.iter().map(|(m, c, _)| *m * c).sum::<Vector3<f64>>() / mass;

    let mut inertia = Matrix3::zeros();
    for (m, c, i) in bodies {
        // Parallel-axis shift from each body's COM to the composite COM:
        // I += m·(‖d‖²·E − d·dᵀ), d = bodyCOM − compositeCOM.
        let d = c - com;
        inertia += i + *m * (d.dot(&d) * Matrix3::identity() - d * d.transpose());
    }
    Payload { mass, com, inertia }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_none() {
        let p = compose(&[]);
        assert_eq!(p.mass, 0.0);
        assert_eq!(p.com, Vector3::zeros());
        assert_eq!(p.inertia, Matrix3::zeros());
    }

    #[test]
    fn single_body_passes_through() {
        let c = Vector3::new(0.1, -0.2, 0.3);
        let i = Matrix3::from_diagonal(&Vector3::new(1.0, 2.0, 3.0));
        let p = compose(&[(0.5, c, i)]);
        assert!((p.mass - 0.5).abs() < 1e-12);
        assert!((p.com - c).norm() < 1e-12);
        assert!((p.inertia - i).norm() < 1e-12);
    }

    #[test]
    fn two_point_masses_combine_to_known_body() {
        // Two equal point masses (zero own-inertia) at ±x about the origin: COM
        // at the midpoint, and the composite inertia is the parallel-axis sum,
        // a thin dumbbell with zero inertia about its own axis (x).
        let m = 0.5;
        let a = Vector3::new(1.0, 0.0, 0.0);
        let b = Vector3::new(-1.0, 0.0, 0.0);
        let p = compose(&[(m, a, Matrix3::zeros()), (m, b, Matrix3::zeros())]);

        assert!((p.mass - 1.0).abs() < 1e-12);
        assert!(p.com.norm() < 1e-12, "COM = {:?}", p.com);
        // Each mass sits 1 m off the COM along x: d=(±1,0,0), so
        // m·(‖d‖²E − d·dᵀ) = m·diag(0,1,1); summed over both = diag(0,1,1).
        let expected = Matrix3::from_diagonal(&Vector3::new(0.0, 1.0, 1.0));
        assert!((p.inertia - expected).norm() < 1e-12, "I = {:?}", p.inertia);
    }

    #[test]
    fn from_distal_lumps_real_gripper_fingers() {
        // Absolute check against the fixture: the two prismatic fingers
        // (0.03602545 kg each) past link7 must be picked up as the distal
        // payload, frozen at home. Expected values are hand-computed from the
        // URDF: finger joint origin z = 0.1025, finger COM (0.0064528, ±0.01702,
        // 0.0219685), so the lumped COM is the mirror-symmetric average.
        use crate::test_support::FIXTURE_URDF;
        let robot = urdf_rs::read_from_string(FIXTURE_URDF).expect("parse fixture");
        let chain = k::Chain::<f64>::from(robot);
        chain.update_transforms();
        let tip = chain.find_link("openarm_left_link7").expect("link7");

        let p = Payload::from_distal(tip);
        let finger_mass = 0.03602545343277134;
        assert!((p.mass - 2.0 * finger_mass).abs() < 1e-12, "mass = {}", p.mass);
        assert!((p.com.x - 0.0064528).abs() < 1e-9, "com.x = {}", p.com.x);
        assert!(p.com.y.abs() < 1e-9, "com.y = {}", p.com.y); // fingers mirror in y
        assert!((p.com.z - (0.1025 + 0.0219685)).abs() < 1e-9, "com.z = {}", p.com.z);
    }

    #[test]
    fn combined_with_merges_a_segment() {
        // Folding a payload point mass at +x into a segment point mass at -x is
        // the same dumbbell as `two_point_masses_combine_to_known_body`.
        let payload = Payload {
            mass: 0.5,
            com: Vector3::new(1.0, 0.0, 0.0),
            inertia: Matrix3::zeros(),
        };
        let merged = payload.combined_with(0.5, Vector3::new(-1.0, 0.0, 0.0), Matrix3::zeros());
        assert!((merged.mass - 1.0).abs() < 1e-12);
        assert!(merged.com.norm() < 1e-12, "COM = {:?}", merged.com);
        let expected = Matrix3::from_diagonal(&Vector3::new(0.0, 1.0, 1.0));
        assert!((merged.inertia - expected).norm() < 1e-12, "I = {:?}", merged.inertia);
    }
}
