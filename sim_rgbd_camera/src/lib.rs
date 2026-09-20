//! Sim RGB-D camera: a relay from a simulation's sim_rgbd_camera_link pairing
//! to the rgbd_camera and camera_profile contract surfaces. Color and depth
//! frames forward with timestamps, frame_ids and align_mode untouched, so
//! consumers age samples on the capture time and correlate a pair on its shared
//! frame_id; a frame whose timestamp is not after the Unix epoch is dropped
//! rather than forwarded, the same guard recording consumers apply at ingestion.
//! The stream-info services answer from the simulation's latest descriptions.
//! The camera_geometry services answer from the simulation's latest geometry,
//! the pinhole model of each stream and the depth stream's pose against the
//! colour one, relayed as it came; until the first arrives they refuse.
//! The colour controls, the profile and the reset forward to the simulation's
//! camera response model on the optional control slot, named by the camera
//! slot this relay views (the pairing peer's link id), and answer what the
//! model answers; with the slot vacant, or while the pairing is not
//! established, every one of them refuses.

#![forbid(unsafe_code)]

mod node;

pub use node::{leg_died, setup};
