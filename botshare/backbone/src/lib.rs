//! botshare_backbone: the robot's backbone of botshare.ai's mobile base,
//! the base alone. It drives the two wheels over joint_link, reports the base's
//! odometry (base_state, drive_link's drive_states) and the transforms it
//! measures (transforms), follows its leader's velocity setpoints
//! (drive_link), and runs the base's planned moves (base_motion) and the
//! robot stop (robot_stop), each goal and setpoint placed by the robot's
//! tree (transforms). The robot's initializer names the robot's model on
//! robot_identity and reports the robot ready on robot_ready, and the model's
//! URDF in robot_description gives the base's wheels and outline.
//!
//! One velocity path carries every motion: the leader's setpoint that the
//! base follows, or the setpoint a goal plans each step, passes
//! control_core's limiter, stepped once per `1 / smoothing_frequency_hz` on
//! the node's clock, and the limited body velocity is mixed into the two
//! wheel rates. Each step the limiter gives a velocity streams one wheel
//! setpoint by control_core's `WheelSetpoint::next`, its anchor bound one step
//! of travel at the fastest wheel rate the base's limits allow. The measured
//! wheel angles feed control_core's wheel odometry.
//!
//! The library crate exists so the harness tests boot the node in-process:
//! [`setup`] is the node's whole entry point, and `src/main.rs` only hands it
//! to `NodeBuilder::run`.

#![forbid(unsafe_code)]

mod actions;
mod clock;
mod config;
mod control;
mod failure;
mod follow;
mod follower;
mod goal;
mod identity;
mod model;
mod move_control;
mod node;
mod rest;
mod services;
mod startup;
mod status;
mod streams;
#[cfg(test)]
mod test_rig;
mod transforms;

pub use node::setup;
