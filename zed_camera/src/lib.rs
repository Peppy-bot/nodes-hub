//! Depth from Stereolabs ZED cameras without the ZED SDK.
//!
//! The library half of the node, free of peppy types so the examples can
//! drive real hardware with plain cargo: [`calibration`] fetches and parses
//! the per-serial factory geometry, [`resolution`] names the capture modes,
//! [`capture`] streams frames and drives controls through the `v4l` crate,
//! and [`cv_depth`] rectifies and matches through the maintained opencv
//! crate.

pub mod calibration;
#[cfg(feature = "capture")]
pub mod capture;
#[cfg(feature = "cv")]
pub mod cv_depth;
pub mod resolution;

pub use resolution::Resolution;
