//! Sim RGB-D camera: a relay from a simulation's sim_rgbd_camera_link pairing
//! to the rgbd_camera and camera_profile contract surfaces. Color and depth
//! frames forward with timestamps, frame_ids and align_mode untouched, so
//! consumers age samples on the capture time and correlate a pair on its shared
//! frame_id; a frame whose timestamp is not after the Unix epoch is dropped
//! rather than forwarded, the same guard recording consumers apply at ingestion.
//! The stream-info services answer from the simulation's latest descriptions.
//! The colour controls, the profile and the reset forward to the simulation's
//! camera response model on the optional control slot, named by this relay's
//! name in its copy (its instance id without the copy's prefix), and answer
//! what the model answers; with the slot vacant every one of them refuses.

#![forbid(unsafe_code)]

mod node;

pub use node::{leg_died, setup};
