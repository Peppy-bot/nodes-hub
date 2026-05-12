use super::controls::{CameraControlRequest, ControlResult};
use crate::types::{CameraConfig, Frame, Result};

/// Abstraction for camera devices
///
/// This trait enables testing by allowing mock implementations
pub trait CameraDevice: Send {
    /// Open and configure the camera
    fn open(&mut self, config: &CameraConfig) -> Result<()>;

    /// Capture a single frame (always RGB8 from camera)
    fn capture_frame(&mut self) -> Result<Frame>;

    /// Check if the camera is open
    fn is_open(&self) -> bool;

    /// Apply a camera control request.
    ///
    /// Returns a `ControlResult` indicating success/failure and the current
    /// value after applying the control.  The default implementation reports
    /// that controls are unsupported, which is appropriate for mock devices
    /// that don't need hardware control.
    fn apply_control(&mut self, _request: &CameraControlRequest) -> ControlResult {
        ControlResult::err("Camera controls not supported for this device")
    }
}
