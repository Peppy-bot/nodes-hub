//! openarm_ker: operator entry point driven by the OpenArm KER (Kinematic
//! Equivalent Replica), enactic's motorless bimanual leader arm. A dedicated
//! reader thread speaks the M5Stack's framed protocol (USB vendor mode or
//! serial CDC), maps encoder channels to calibrated joint radians and trigger
//! openings, and the publish tasks stream them like the commander does:
//! arms on `arm_joint_commands` (the backbone owns governing and safety), grippers
//! on their pairing slots. The thumb button is the engage deadman; while
//! disengaged, stale, or disconnected the node emits nothing and the
//! followers hold their last setpoints.

#![forbid(unsafe_code)]

mod mapping;
mod node;
mod protocol;
mod publish;
mod reader;
mod transport;

pub use node::{NodeError, setup, task_failed};
