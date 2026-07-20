//! Depth from Stereolabs ZED cameras without the ZED SDK.
//!
//! The library half of the node, free of peppy types so it can be exercised
//! with plain cargo: [`calibration`] fetches and parses the per-serial
//! factory geometry, [`resolution`] names the capture modes, and [`capture`]
//! streams frames and drives controls through the `v4l` crate.

pub mod calibration;
#[cfg(feature = "capture")]
pub mod capture;
pub mod resolution;

pub use resolution::Resolution;
