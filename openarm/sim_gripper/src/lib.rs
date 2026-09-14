//! Sim gripper follower for any simulation: a pure relay between its two
//! gripper_link pairings. The backbone's governed opening setpoints (with the
//! operator's effort cap) forward to the simulation's matching limb slot and
//! the simulation's measured state forwards back to the backbone, timestamps
//! untouched, so both peers see the conversation they have with a real
//! counterpart. Non-finite values are dropped, the guard every follower
//! applies at ingestion.

#![forbid(unsafe_code)]

mod node;

pub use node::setup;
