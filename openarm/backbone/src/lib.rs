//! openarm_backbone - bimanual coordinator. It owns all arm motion: it
//! consumes the leading node's joint or pose stream (per `upstream_mode`) and
//! exposes the joint / Cartesian move actions, generates the trajectories,
//! runs the self-collision governor over both arms together, and streams the
//! governed per-arm setpoints the arms follow. Grippers run through the
//! backbone the same way: the leading node's gripper stream and move_gripper
//! goals both feed the coordinator, the grippers ride the same governed
//! configuration as the arm joints (a gripper cannot open its
//! fingers into the other arm), and the governed opening streams to each
//! gripper over its gripper_link pairing slot. The governor is URDF-based, so
//! it runs identically for the sim and the real arms.
//!
//! The library crate exists so the harness-driven integration tests can boot
//! the node in-process: [`setup`] is the node's whole entry point, and
//! `src/main.rs` only hands it to `NodeBuilder::run`.

#![forbid(unsafe_code)]

mod actions;
mod arm_pair;
mod chase;
mod coordinator;
mod governor;
mod liveness;
mod motion;
mod node;
mod planner;
mod publish;
mod servo;
mod startup;
mod streams;
mod torso;
mod trajectory;
mod types;
mod upstream;

// Sibling modules' unit tests reach the shared model builder as `crate::arm_model`.
#[cfg(test)]
use node::arm_model;
pub use node::{NodeError, setup};
