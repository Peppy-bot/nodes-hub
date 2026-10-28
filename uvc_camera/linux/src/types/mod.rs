// Domain types for the UVC camera node

pub mod encoding;
pub mod error;
pub mod frame;
pub mod parameters;

// Re-export commonly used types
pub use error::{Error, Result};
pub use frame::{Frame, FrameId};
pub use parameters::{CameraConfig, FrameRate, Resolution};
