//! One arm of the openarm robot (real hardware); instantiate twice, one per
//! side, with a distinct `arm_id`. A follower of the bimanual backbone: it owns the
//! hardware control loop (gravity/Coriolis/friction feedforward from the
//! in-process srs_model, plus MIT control) and tracks the backbone's governed
//! setpoint over the joint_link pairing, reporting measured state back on the
//! same pairing (any monitor observes it). The
//! backbone (openarm_backbone) owns all trajectory generation, stream following, and
//! self-collision governing, so this node carries no motion logic of its own; on
//! shutdown it disables the motors and lets the arm go limp.

#![forbid(unsafe_code)]

mod control;
mod friction;
mod health;
mod node;
mod stream;

/// Degrees of freedom of the arm.
pub const ARM_DOF: usize = openarm_description::ARM_DOF;
/// One joint-space vector (positions, velocities, or torques), j1..j7.
pub use srs_model::JointVec;

pub use node::{NodeError, hard_fault_latched, setup};
