//! One arm of the openarm robot (real hardware); instantiate once per arm of the
//! robot, each with its arm's place as `arm_id`. A follower of the bimanual
//! backbone: it owns the hardware control loop (gravity/Coriolis/friction
//! feedforward from the in-process srs_model, plus MIT control) and tracks the
//! backbone's governed setpoint over the joint_link pairing, reporting measured
//! state back on the same pairing (any monitor observes it). The backbone
//! (openarm_backbone) owns all trajectory generation, stream following, and
//! self-collision governing, so this node carries no motion logic of its own; on
//! shutdown it disables the motors and lets the arm go limp.

#![forbid(unsafe_code)]

mod control;
mod drive;
mod friction;
mod health;
mod node;
mod stream;

pub use node::{NodeError, setup};
