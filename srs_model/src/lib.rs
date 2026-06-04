//! Kinematics + dynamics for a 7-DOF SRS (spherical-revolute-spherical) arm.
//!
//! - [`fk`]: forward kinematics from a URDF (via the `k` crate).
//! - [`model`]: the SRS geometry (shoulder/elbow/wrist centers, link lengths,
//!   joint limits) derived once from the FK chain.
//! - [`ik`]: closed-form arm-angle (Shimizu) inverse kinematics.
//! - [`gravity`] / [`coriolis`]: feedforward dynamics torques for the arm control
//!   loop. Each carries the distal payload (gripper, fingers, tools) lumped into
//!   the last segment, and is validated in tests against KDL `TreeIdSolver_RNE`
//!   reference values (tree inverse dynamics, so the branched gripper is
//!   included); gravity additionally against the potential-energy gradient.
//!
//! Frames: [`fk`] and [`ik`] report poses in the **arm base frame** (the arm's
//! own mounting link). [`gravity`] / [`coriolis`] instead compute in the **world frame**,
//! because gravity is a world quantity (acts along world -Z). The fixed
//! world <-> base mount transform relating the two is captured here and exposed
//! via [`model::ArmModel::base_pose`] / [`world_pose`](model::ArmModel::world_pose),
//! so a caller converting between world and base frames does not redo it.
//!
//! Robot-agnostic: geometry, joint limits and inertials are all derived from
//! whatever URDF the caller passes, and a non-SRS chain is rejected with `Err`.
//! There is no per-robot configuration baked in: the caller supplies the URDF and
//! the `base_link` where the SRS chain starts; the wrist is found by walking 7
//! revolute joints out from it, and everything past the wrist (gripper, fingers,
//! tools) is carried as the distal payload in gravity / Coriolis. Left vs right is
//! a different chain in the same URDF, selected by the base link (the mirror is
//! re-derived from the URDF geometry, never sign-flipped). For gravity to point
//! the right way, the URDF must contain the kinematic tree from the world/body
//! root down to `base_link` (the mount), not just the bare arm chain.
//!
//! Hardware-free: it defines its own [`ARM_DOF`] rather than depending on any
//! hardware crate, so the full IK<->FK round-trip runs under a plain `cargo test`
//! on any host.

pub mod coriolis;
pub mod fk;
pub mod gravity;
pub mod ik;
pub mod model;
mod payload;

/// Degrees of freedom of the arm. The URDF chain is validated against this at
/// load (the closed-form arm-angle redundancy resolution is specific to 7 DOF).
pub const ARM_DOF: usize = 7;

/// One joint-space configuration, j1..j7 in radians.
pub type JointVec = [f64; ARM_DOF];

/// Smallest sine of an angle between two unit axes (or two link directions) we
/// treat as non-degenerate (~1e-6 rad). Below it the perpendicular / cross
/// direction is ill-conditioned to normalize, so the caller bails out (parallel
/// axes, straight arm). Sites that test a `sin²` quantity compare its square.
pub(crate) const PARALLEL_SIN_EPS: f64 = 1e-6;

/// Re-export the linear-algebra types so downstream crates use the same
/// `nalgebra` version `k` was built against.
pub use k::nalgebra;

/// Test fixtures: load a concrete SRS arm (the OpenArm V1.0) from the bundled
/// fixture URDF, giving the agnostic tests a real arm to check against.
/// `side` is `"left"`/`"right"`; left vs right is encoded entirely by the link
/// names, so there is no side parameter beyond which links are selected.
#[cfg(test)]
pub(crate) mod test_support {
    use crate::fk::ForwardKinematics;
    use crate::model::ArmModel;

    /// A concrete SRS arm used only as a test fixture: the real OpenArm V1.0
    /// description, gripper fingers included, so the distal-payload path is
    /// exercised by simply loading it. Production callers pass their own URDF via
    /// the node configuration; this is wired only under `cfg(test)`.
    pub(crate) const FIXTURE_URDF: &str = include_str!("../tests/fixtures/openarm_v10.urdf");

    /// Base link where the fixture's 7-DOF chain for `side` starts.
    fn base(side: &str) -> String {
        format!("openarm_{side}_link0")
    }

    pub(crate) fn v1_fk(side: &str) -> ForwardKinematics {
        ForwardKinematics::from_urdf(FIXTURE_URDF, &base(side)).expect("load fixture fk")
    }

    pub(crate) fn v1_model(side: &str) -> ArmModel {
        ArmModel::from_urdf(FIXTURE_URDF, &base(side)).expect("load fixture model")
    }
}
