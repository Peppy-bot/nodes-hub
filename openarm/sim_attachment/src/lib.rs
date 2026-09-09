//! One OpenArm's seat in a shared simulation: the limb pairings a robot's
//! relays bind to, held against one seat of the `simulation_robot` contract
//! on the engine.
//!
//! The engine hosts any number of robots and tells them apart by the caller,
//! so everything robot-facing here is the surface an engine offered when it
//! hosted one: four follower slots, setpoints in, measured state back.

#![forbid(unsafe_code)]

mod limbs;
mod node;

pub use node::setup;
