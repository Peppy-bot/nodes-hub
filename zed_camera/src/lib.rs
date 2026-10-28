//! Depth from Stereolabs ZED cameras without the ZED SDK.
//!
//! The library half of the node, free of peppy types so the examples can
//! drive real hardware with plain cargo: [`calibration`] parses
//! the per-serial factory geometry, [`resolution`] names the capture modes
//! and their frame rates, [`size`] holds an image size, [`capture`] streams
//! frames and drives controls through the `v4l` crate, [`depth_settings`]
//! validates the stereo-matching settings, [`cv_depth`] rectifies and
//! matches through the maintained opencv crate, and [`geometry`] answers
//! where the published pixels point and turns a disparity into depth.

pub mod calibration;
#[cfg(all(feature = "capture", target_os = "linux"))]
pub mod capture;
#[cfg(feature = "cv")]
pub mod cv_depth;
pub mod depth_settings;
pub mod geometry;
pub mod resolution;
pub mod size;

pub use depth_settings::DepthSettings;
pub use resolution::{LegalFrameRate, Resolution};
pub use size::Size;
