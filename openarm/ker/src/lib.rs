//! openarm_ker: operator entry point driven by the OpenArm KER (Kinematic
//! Equivalent Replica), enactic's motorless bimanual leader arm. A dedicated
//! reader thread speaks the M5Stack's framed protocol (USB vendor mode or
//! serial CDC), maps its channels to clamped joint radians and commanded
//! gripper openings, and the publish tasks stream each limb on its joint_link
//! or gripper_link pairing slot (the backbone governs them). An arm engages on
//! a trigger squeeze, the trigger having read open first; for an unengaged
//! arm, a stale KER, or a disconnected one the node emits nothing, so those
//! followers hold their last setpoints.

#![forbid(unsafe_code)]

use openarm_description::Side;

mod mapping;
mod node;
mod protocol;
mod publish;
mod reader;
mod transport;

pub use node::{NodeError, setup, task_failed};

/// One arm's name for the operator log, the spelling every line in this node
/// uses.
pub(crate) fn label(side: Side) -> &'static str {
    match side {
        Side::Left => "left",
        Side::Right => "right",
    }
}
