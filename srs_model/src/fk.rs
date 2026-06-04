//! Forward kinematics for a 7-DOF SRS arm, built from a URDF via the `k` crate.
//!
//! `k` computes poses in the chain's root (world) frame. This type exposes both:
//! **base-frame** accessors ([`ee_pose`](Posed::ee_pose),
//! `axis_base`, `origin_base`), re-expressed in the arm's own mount link so they
//! line up with the SRS geometry in [`crate::model`] and the IK target frame; and
//! **world-frame** accessors used by [`crate::gravity`] / [`crate::coriolis`], because gravity is a
//! world quantity (acts along world -Z). The fixed `world -> base` transform that
//! relates the two is captured once at load.
//!
//! It is also the *independent* FK used to verify IK: it composes joint
//! transforms straight from the URDF, sharing no code with the analytic solver,
//! so an IK->FK round-trip is a genuine test.

use k::nalgebra::{Isometry3, Matrix3, Point3, Vector3};
use k::{Chain, JointType, Node, SerialChain};

use crate::payload::Payload;
use crate::{ARM_DOF, JointVec};

/// Parsed serial chain (base -> joint7) plus the constant `world -> base`
/// transform and per-segment immutable data (axis, mass, COM, inertia)
/// captured at load. Segment `i` is the link moved by joint `i`; its inertia
/// is in the link frame (V1.0 URDF inertials use identity rpy).
pub struct ForwardKinematics {
    chain: SerialChain<f64>,
    /// The 7 revolute joint nodes in chain order.
    joint_nodes: [Node<f64>; ARM_DOF],
    /// The wrist (tip) node, the EE frame: the link after the 7th revolute joint,
    /// found by walking the chain out from the base.
    tip: Node<f64>,
    /// `world -> base_link`; constant because every joint between them is fixed.
    base_from_world: Isometry3<f64>,
    axes_local: [Vector3<f64>; ARM_DOF],
    masses: [f64; ARM_DOF],
    coms_local: [Vector3<f64>; ARM_DOF],
    inertias_local: [Matrix3<f64>; ARM_DOF],
}

impl ForwardKinematics {
    /// Build the FK chain from a URDF string given only where the SRS arm
    /// *starts* (`base_link`). The wrist (tip) is found by walking exactly
    /// [`ARM_DOF`] revolute joints out from the base, so the 7-DOF SRS invariant
    /// is enforced rather than trusting a hand-entered tip that might disagree.
    /// Everything past the wrist (gripper, fingers, tools) becomes the distal
    /// payload. Agnostic to *which* 7-DOF SRS arm: any URDF + base link the
    /// caller passes (it is not a general N-DOF or non-SRS solver).
    pub fn from_urdf(urdf: &str, base_link: &str) -> Result<Self, String> {
        let robot = urdf_rs::read_from_string(urdf).map_err(|e| format!("parse URDF: {e}"))?;
        Self::from_chain(Chain::<f64>::from(robot), base_link)
    }

    fn from_chain(full: Chain<f64>, base_link: &str) -> Result<Self, String> {
        let base = full
            .find_link(base_link)
            .ok_or_else(|| format!("URDF missing base link '{base_link}'"))?
            .clone();

        // Pose the whole tree at home so the traversal and the frozen distal
        // links are read off consistent world transforms.
        full.update_transforms();

        // The SRS chain is implicit: walk ARM_DOF revolute joints out from the
        // base to find the wrist, then lump everything past it into the distal
        // payload, before reducing to the serial base -> tip chain.
        let tip = find_srs_tip(&base)?;
        let payload = Payload::from_distal(&tip);

        let chain = SerialChain::from_end(&tip);
        let joint_nodes = collect_revolute_nodes(&chain)?;
        let axes_local = std::array::from_fn(|i| match joint_nodes[i].joint().joint_type {
            JointType::Rotational { axis } => *axis.as_ref(),
            _ => unreachable!("collect_revolute_nodes verified revolute"),
        });

        // Per-segment inertial data: segment i is the link rigidly attached
        // downstream of joint i (its `Node`'s link).
        let mut masses = [0.0_f64; ARM_DOF];
        let mut coms_local = [Vector3::zeros(); ARM_DOF];
        let mut inertias_local = [Matrix3::zeros(); ARM_DOF];
        for (i, node) in joint_nodes.iter().enumerate() {
            let guard = node.link();
            let inertial = &guard
                .as_ref()
                .ok_or_else(|| format!("joint {i} node has no link"))?
                .inertial;
            masses[i] = inertial.mass;
            coms_local[i] = inertial.origin().translation.vector;
            // Rotate the inertia into the link frame so a non-identity inertial
            // rpy is handled (V1.0's are all identity, so this is a no-op there).
            let r = *inertial.origin().rotation.to_rotation_matrix().matrix();
            inertias_local[i] = r * inertial.inertia * r.transpose();
        }

        // Fold the distal payload into the last segment. It is rigidly attached
        // to the wrist link, whose frame is segment ARM_DOF-1's frame (the tip is
        // that node), so a bigger last link and a separate payload are the same
        // rigid body. Gravity / Coriolis then carry it.
        if payload.mass > 0.0 {
            let last = ARM_DOF - 1;
            let merged = payload.combined_with(masses[last], coms_local[last], inertias_local[last]);
            masses[last] = merged.mass;
            coms_local[last] = merged.com;
            inertias_local[last] = merged.inertia;
        }

        // Pose the serial chain at home, then read the fixed base-link world transform.
        chain.set_joint_positions_unchecked(&[0.0; ARM_DOF]);
        chain.update_transforms();
        let base_world = base
            .world_transform()
            .ok_or("base link has no world transform")?;

        Ok(Self {
            chain,
            joint_nodes,
            tip,
            base_from_world: base_world.inverse(),
            axes_local,
            masses,
            coms_local,
            inertias_local,
        })
    }

    /// Pose the chain at `q` and return a read-only view of it. Posing needs
    /// `&mut` (the `k` chain mutates in place), but the returned [`Posed`] is the
    /// only way to read the chain, so a configuration is always applied before any
    /// accessor runs, and the `&mut` borrow is held for the view's lifetime so no
    /// read can race a re-pose. "Pose, then read" is thus a type invariant, not a
    /// calling convention.
    pub fn at(&mut self, q: &JointVec) -> Posed<'_> {
        self.chain.set_joint_positions_unchecked(q);
        self.chain.update_transforms();
        Posed { fk: self }
    }
}

/// A [`ForwardKinematics`] posed at one configuration: an immutable, read-only
/// view obtained from [`ForwardKinematics::at`]. Every pose-dependent quantity is
/// read through here, so it can only be queried after a pose has been applied.
pub struct Posed<'a> {
    fk: &'a ForwardKinematics,
}

impl Posed<'_> {
    /// End-effector (tip-link) pose in the arm base frame.
    pub fn ee_pose(&self) -> Isometry3<f64> {
        self.to_base(&self.fk.tip)
    }

    /// World-frame revolute axis of joint `i`, re-expressed in the base frame. A
    /// revolute axis is invariant under its own angle, so rotating the local axis
    /// by the joint's world rotation is exact.
    pub(crate) fn axis_base(&self, i: usize) -> Vector3<f64> {
        self.to_base(&self.fk.joint_nodes[i])
            .rotation
            .transform_vector(&self.fk.axes_local[i])
    }

    /// Origin of joint `i`'s frame in the base frame. A point *on* the joint axis
    /// (used for SRS line geometry, never joint4's offset frame origin alone).
    pub(crate) fn origin_base(&self, i: usize) -> Vector3<f64> {
        self.to_base(&self.fk.joint_nodes[i]).translation.vector
    }

    /// The constant `world -> arm base` transform (this arm's fixed mounting on
    /// the body). Converts world/body-frame targets into the arm base frame.
    pub(crate) fn base_from_world(&self) -> Isometry3<f64> {
        self.fk.base_from_world
    }

    /// URDF position limit `(lower, upper)` of joint `i`, in radians. Falls back
    /// to `±π` if the URDF leaves a joint unlimited (none do for V1.0).
    pub(crate) fn joint_limit(&self, i: usize) -> (f64, f64) {
        match &self.fk.joint_nodes[i].joint().limits {
            Some(range) => (range.min, range.max),
            None => (-std::f64::consts::PI, std::f64::consts::PI),
        }
    }

    // --- World-frame accessors (gravity is world -z; used by `gravity` / `coriolis`). ---
    // These are expressed in the URDF root/world frame used for gravity and for
    // the KDL reference checks. In the bundled OpenArm fixture, that root also
    // happens to coincide with `body_link0`.

    /// Mass of segment `i` (static; URDF parse-time).
    pub(crate) fn mass(&self, i: usize) -> f64 {
        self.fk.masses[i]
    }

    /// World-frame revolute axis of joint `i`.
    pub(crate) fn axis_world(&self, i: usize) -> Vector3<f64> {
        self.joint_world(i)
            .rotation
            .transform_vector(&self.fk.axes_local[i])
    }

    /// World-frame origin of joint `i`.
    pub(crate) fn origin_world(&self, i: usize) -> Vector3<f64> {
        self.joint_world(i).translation.vector
    }

    /// World-frame COM of segment `i`.
    pub(crate) fn com_world(&self, i: usize) -> Vector3<f64> {
        self.joint_world(i)
            .transform_point(&Point3::from(self.fk.coms_local[i]))
            .coords
    }

    /// World-frame inertia tensor of segment `i` about its COM:
    /// `I_world = R · I_local · Rᵀ`.
    pub(crate) fn inertia_world(&self, i: usize) -> Matrix3<f64> {
        let r = *self.joint_world(i).rotation.to_rotation_matrix().matrix();
        r * self.fk.inertias_local[i] * r.transpose()
    }

    fn joint_world(&self, i: usize) -> Isometry3<f64> {
        self.fk.joint_nodes[i]
            .world_transform()
            .expect("node world transform")
    }

    fn to_base(&self, node: &Node<f64>) -> Isometry3<f64> {
        self.fk.base_from_world * node.world_transform().expect("node world transform")
    }
}

/// Walk from `base` down the unique revolute-bearing path and return the link
/// reached after exactly [`ARM_DOF`] revolute joints: the SRS wrist (tip). The
/// arm is serial until the wrist, so at each step exactly one child still leads
/// to a revolute joint; a fixed sensor branch or the (prismatic) gripper is
/// skipped, and a genuine fork (two revolute branches) is rejected as not a
/// single SRS arm.
fn find_srs_tip(base: &Node<f64>) -> Result<Node<f64>, String> {
    let mut node = base.clone();
    let mut revolute = 0;
    while revolute < ARM_DOF {
        // Materialize children before filtering so the parent lock is released
        // before `subtree_has_revolute` locks each child.
        let children: Vec<Node<f64>> = node.children().to_vec();
        let mut arm: Vec<Node<f64>> =
            children.into_iter().filter(subtree_has_revolute).collect();
        node = match arm.len() {
            1 => arm.pop().unwrap(),
            0 => {
                return Err(format!(
                    "chain from base reaches only {revolute} revolute joints; \
                     a 7-DOF SRS arm needs {ARM_DOF}"
                ));
            }
            n => {
                return Err(format!(
                    "ambiguous arm: {n} revolute-bearing branches share one link; \
                     not a single SRS chain"
                ));
            }
        };
        if matches!(node.joint().joint_type, JointType::Rotational { .. }) {
            revolute += 1;
        }
    }
    Ok(node)
}

/// Whether `node` or any of its descendants is reached through a revolute joint:
/// marks the branch that continues the arm, versus a dead fixed mount or the
/// gripper. `iter_descendants` includes `node` itself.
fn subtree_has_revolute(node: &Node<f64>) -> bool {
    node.iter_descendants()
        .any(|d| matches!(d.joint().joint_type, JointType::Rotational { .. }))
}

/// Collect the [`ARM_DOF`] revolute joint nodes of the serial chain in order,
/// rejecting any chain that does not reduce to exactly that. The fixed
/// world/body mounting joints are skipped.
///
/// A clean 7-DOF SRS arm's *only* degrees of freedom are its seven revolute
/// joints. Any other movable joint (e.g. a prismatic DOF interspersed on the
/// path) is rejected: it would otherwise pass the revolute count below while
/// `base_from_world` is frozen at home, silently building the wrong model
/// instead of returning `Err`.
fn collect_revolute_nodes(chain: &SerialChain<f64>) -> Result<[Node<f64>; ARM_DOF], String> {
    if let Some(extra) = chain
        .iter()
        .find(|n| !matches!(n.joint().joint_type, JointType::Fixed | JointType::Rotational { .. }))
    {
        return Err(format!(
            "SRS chain has a non-revolute movable joint '{}': not a 7-DOF revolute arm",
            extra.joint().name
        ));
    }
    let nodes: Vec<Node<f64>> = chain
        .iter()
        .filter(|n| matches!(n.joint().joint_type, JointType::Rotational { .. }))
        .cloned()
        .collect();
    nodes
        .try_into()
        .map_err(|v: Vec<_>| format!("expected {ARM_DOF} revolute joints in the SRS chain, got {}", v.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fk() -> ForwardKinematics {
        crate::test_support::v1_fk("left")
    }

    #[test]
    fn loads_seven_revolute_chain() {
        let mut fk = fk();
        let posed = fk.at(&[0.0; ARM_DOF]);
        for i in 0..ARM_DOF {
            let n = posed.axis_base(i).norm();
            assert!((n - 1.0).abs() < 1e-9, "joint {i} axis not unit: {n}");
        }
    }

    #[test]
    fn home_ee_is_above_shoulder() {
        let mut fk = fk();
        let ee = fk.at(&[0.0; ARM_DOF]).ee_pose();
        // At home the wrist center sits at ~(0, 0.436, 0.1225) in base frame.
        // Mainly a smoke check that the base transform applied.
        let w = ee.translation.vector;
        assert!(
            (w - Vector3::new(0.0, 0.436, 0.1225)).norm() < 1e-3,
            "home EE {w:?} not at expected wrist center",
        );
    }

    #[test]
    fn rejects_urdf_missing_arm_links() {
        // A URDF without the arm links must Err, not panic.
        let urdf = r#"<?xml version="1.0"?><robot name="x"><link name="world"/></robot>"#;
        assert!(ForwardKinematics::from_urdf(urdf, "openarm_left_link0").is_err());
    }

    #[test]
    fn rejects_malformed_urdf() {
        assert!(ForwardKinematics::from_urdf("not even xml", "a").is_err());
    }

    #[test]
    fn rejects_chain_without_seven_revolute_joints() {
        // A prismatic-only arm must Err: walking out from the base finds no
        // revolute joints to reach the wrist, so it is not a 7-DOF SRS arm. (A
        // prismatic joint *interspersed* among the 7 is caught separately by
        // `collect_revolute_nodes`.)
        let urdf = r#"<?xml version="1.0"?><robot name="x">
          <link name="base"/><link name="tip"/>
          <joint name="slide" type="prismatic">
            <parent link="base"/><child link="tip"/>
            <axis xyz="0 0 1"/><origin xyz="0 0 0"/>
            <limit lower="0" upper="1" effort="1" velocity="1"/>
          </joint>
        </robot>"#;
        let err = match ForwardKinematics::from_urdf(urdf, "base") {
            Ok(_) => panic!("expected Err for a prismatic joint"),
            Err(e) => e,
        };
        assert!(err.contains("revolute"), "unexpected error: {err}");
    }
}
