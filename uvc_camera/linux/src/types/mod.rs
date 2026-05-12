// Domain types for the UVC camera node

pub mod encoding;
pub mod error;
pub mod frame;
pub mod parameters;

// Re-export commonly used types
pub use encoding::Encoding;
pub use error::{Error, Result};
pub use frame::{Frame, FrameId, FrameInfo};
pub use parameters::{CameraConfig, CameraConfigBuilder, FrameRate, Resolution};
