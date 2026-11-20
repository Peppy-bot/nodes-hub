//! Sim color camera: a relay from a simulation's sim_rgb_camera_link pairing to
//! the rgb_camera and camera_profile contract surfaces. Pictures forward with
//! their timestamp, capture_index and frame untouched, so consumers age samples
//! on the capture time; the relay holds each picture's frame to the camera's
//! frame, which it reads from sensor_mounts' describe_sensors once at its start
//! by its name in its copy, and drops a picture in another frame, logging the
//! first of a run with both frames and what to fix. The relay relays and serves
//! nothing before it knows the camera's frame.
//! video_stream_info answers from the simulation's latest stream description.
//! Of the camera_geometry services, get_color_intrinsics answers from the
//! simulation's latest geometry, the stream's pinhole model relayed as it came,
//! and gives an error reply of the kind `precondition_failed` until the first
//! arrives; the two depth services always give one of the kind `rejected`, a
//! colour camera having no depth stream. The camera controls, the profile and
//! the reset forward to the simulation's camera response model on the optional
//! control slot, named by this relay's name in its copy (its instance id
//! without the copy's prefix), and answer what the model answers, its error
//! replies with their kinds; with the slot vacant every one of them gives an
//! error reply of the kind `rejected`. A relay whose instance id names no
//! camera in its copy, or whose name sensor_mounts gives no camera's frame,
//! fails its start.

#![forbid(unsafe_code)]

mod node;

pub use node::{leg_died, setup};
