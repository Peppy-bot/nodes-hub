// Library crate root: `setup` is the node's real entry point, importable both
// by `main.rs` (which passes it to `NodeBuilder::run`) and by the integration
// tests, whose generated harness boots the node by calling `setup` directly.

#![forbid(unsafe_code)]

mod alerts;
mod collision_status;
mod command_stream;
mod consumer;
mod gestures;
mod gripper_states;
mod joint_states;
mod motor_health;
mod move_arm;
mod move_arm_joints;
mod move_gripper;
mod node;
mod owner;
mod pose;
mod record;
mod result_wait;
mod state;
mod ui;

pub use node::{NodeError, setup, ui_failed};
