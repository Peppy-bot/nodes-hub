//! The UVC camera node as a library, for the binary and the integration
//! tests: the V4L2 device and its capture loop ([`camera`]), the conversion
//! of a captured frame to the topic's encoding ([`pipeline`]), the rgb_camera
//! and camera_geometry services ([`services`], [`geometry`]), and the
//! configuration and frame types ([`types`]).

pub mod camera;
pub mod geometry;
pub mod pipeline;
pub mod services;
pub mod types;
