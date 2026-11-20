//! Sim RGB-D camera: a relay from a simulation's sim_rgbd_camera_link pairing
//! to the rgbd_camera and camera_profile contract surfaces. Color and depth
//! pictures forward with their timestamp, capture_index, align_mode and frame
//! untouched, so consumers age samples on the capture time and pair a colour
//! picture with its depth picture on the shared capture_index; the relay holds
//! each picture's frame to the camera's frame, which it reads from
//! sensor_mounts' describe_sensors once at its start by its name in its copy,
//! and drops a picture in another frame, logging the first of a run with both
//! frames and what to fix. The relay relays and serves nothing before it knows
//! the camera's frame. The stream-info services answer from the simulation's
//! latest descriptions. The camera_geometry services answer from the simulation's
//! latest geometry, the pinhole model of each stream relayed as it came and the
//! depth stream's pose against the colour one in the frame the simulation gives
//! it; a geometry whose pose is in another frame than the
//! camera's is ignored. Until the first geometry arrives they give an error
//! reply of the kind `precondition_failed`. The colour controls, the profile and the reset
//! forward to the simulation's camera response model on the optional control
//! slot, named by this relay's name in its copy (its instance id without the
//! copy's prefix), and answer what the model answers, its error replies with
//! their kinds; with the slot vacant every one of them gives an error reply of
//! the kind `rejected`. A relay whose instance id names no camera in its copy,
//! or whose name sensor_mounts gives no camera's frame, fails its start.

#![forbid(unsafe_code)]

mod node;

pub use node::{leg_died, setup};
