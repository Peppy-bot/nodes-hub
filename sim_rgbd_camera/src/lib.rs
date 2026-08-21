//! Sim RGB-D camera: a pure relay from a simulation engine's
//! sim_rgbd_camera_link pairing to the rgbd_camera contract surface. Color and
//! depth frames forward with timestamps, frame_ids and align_mode untouched, so
//! consumers age samples on the capture time and correlate a pair on its shared
//! frame_id; a frame whose timestamp is not after the Unix epoch is dropped
//! rather than forwarded, the same guard recording consumers apply at ingestion.
//! The stream-info services answer from the engine's latest descriptions; the
//! hardware control services refuse, because a rendered stream has no sensor to
//! adjust.

#![forbid(unsafe_code)]

mod node;

pub use node::{leg_died, setup};
