//! One gripper of the openarm robot (real hardware); instantiate twice, one
//! per side, with a distinct `gripper_id`. Both hardware generations run
//! through this node, selected by `hardware_version`: the v1.0 prismatic
//! parallel jaws (MIT position hold, no force control) and the v2.0 revolute
//! pinch fingers (POS_FORCE, with a speed limit, a grip-force ceiling, and
//! force feedback). A follower of the bimanual backbone: it drives the motor
//! toward the opening setpoint the backbone streams over the gripper_link
//! pairing and reports the measured opening, grip effort, and effort ceiling
//! back on the same pairing. It also publishes the motor's health and the
//! operator alerts raised from it. The backbone (openarm_backbone) owns all
//! motion, so this node carries no motion logic of its own; on shutdown it
//! disables the motor.

#![forbid(unsafe_code)]

mod drive;
mod follow;
mod geometry;
mod hardware;
mod health;
mod node;
mod stream;

pub use node::{NodeError, hard_fault_latched, setup};
