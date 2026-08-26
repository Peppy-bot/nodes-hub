//! Engine-agnostic sim gripper follower: a pure relay between its two
//! gripper_link pairings. The backbone's governed opening setpoints (with the
//! operator's effort cap) forward to the sim engine's matching limb slot and
//! the engine's measured state forwards back to the backbone, timestamps
//! untouched, so both peers see the conversation they would have with a real
//! counterpart. Non-finite values are dropped rather than forwarded, the same
//! guard every follower applies at ingestion.

#![forbid(unsafe_code)]

mod node;

pub use node::setup;
