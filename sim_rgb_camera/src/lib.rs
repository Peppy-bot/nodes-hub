//! Sim color camera: a relay from a simulation's sim_rgb_camera_link pairing
//! to the rgb_camera and camera_profile contract surfaces. Frames forward
//! with timestamps and frame_ids untouched, so consumers age samples on the
//! capture time; a frame whose timestamp is not after the Unix epoch is
//! dropped rather than forwarded, the same guard recording consumers apply at
//! ingestion. video_stream_info answers from the simulation's latest stream
//! description. The camera controls, the profile and the reset forward to the
//! simulation's camera response model on the optional control slot, named by
//! this relay's name in its copy (its instance id without the copy's prefix),
//! and answer what the model answers; with the slot vacant every one of them
//! refuses.

#![forbid(unsafe_code)]

mod node;

pub use node::{leg_died, setup};
