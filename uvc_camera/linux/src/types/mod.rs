// Domain types for the UVC camera node

pub mod encoding;
pub mod error;
pub mod frame;
pub mod parameters;

// Re-export commonly used types
pub use error::{Error, Result};
pub use frame::{CaptureIndex, Frame};
pub use parameters::{CameraConfig, FrameRate, Resolution};
