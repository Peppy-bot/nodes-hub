//! Depth from Stereolabs ZED cameras without the ZED SDK.
//!
//! The library half of the node, free of peppy types so it can be exercised
//! with plain cargo: [`calibration`] fetches and parses the per-serial
//! factory geometry; [`resolution`] names the capture modes.

pub mod calibration;
pub mod resolution;

pub use resolution::Resolution;
