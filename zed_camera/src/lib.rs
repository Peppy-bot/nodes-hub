//! Depth from Stereolabs ZED cameras without the ZED SDK.
//!
//! The library half of the node, free of peppy types so the examples can
//! drive real hardware with plain cargo: [`calibration`] parses
//! the per-serial factory geometry, [`resolution`] names the capture modes,
//! [`capture`] streams frames and drives controls through the `v4l` crate,
//! and [`cv_depth`] rectifies and matches through the maintained opencv
//! crate.

pub mod calibration;
#[cfg(all(feature = "capture", target_os = "linux"))]
pub mod capture;
#[cfg(feature = "cv")]
pub mod cv_depth;
pub mod depth_settings;
pub mod resolution;

pub use depth_settings::DepthSettings;
pub use resolution::Resolution;
