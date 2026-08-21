//! Sim color camera: a pure relay from a simulation engine's
//! sim_rgb_camera_link pairing to the rgb_camera contract surface. Frames
//! forward with timestamps and frame_ids untouched, so consumers age samples on
//! the capture time; a frame whose timestamp is not after the Unix epoch is
//! dropped rather than forwarded, the same guard recording consumers apply at
//! ingestion. video_stream_info answers from the engine's latest stream
//! description; the hardware control services refuse, because a rendered stream
//! has no sensor to adjust.

#![forbid(unsafe_code)]

mod node;

pub use node::{leg_died, setup};
