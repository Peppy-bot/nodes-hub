//! Camera control command types and channels
//!
//! This module defines the interface for sending camera control commands
//! from async service handlers to the blocking camera capture loop.

/// Exposure mode for the `set_exposure` service
#[derive(Debug, Clone, PartialEq)]
pub enum ExposureMode {
    Auto,
    Manual,
}

impl TryFrom<&str> for ExposureMode {
    type Error = String;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(ExposureMode::Auto),
            "manual" => Ok(ExposureMode::Manual),
            _ => Err(format!(
                "Invalid exposure mode: '{}', expected 'auto' or 'manual'",
                s
            )),
        }
    }
}

/// White balance mode for the `set_white_balance` service
#[derive(Debug, Clone, PartialEq)]
pub enum WhiteBalanceMode {
    Auto,
    Manual,
}

impl TryFrom<&str> for WhiteBalanceMode {
    type Error = String;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(WhiteBalanceMode::Auto),
            "manual" => Ok(WhiteBalanceMode::Manual),
            _ => Err(format!(
                "Invalid white balance mode: '{}', expected 'auto' or 'manual'",
                s
            )),
        }
    }
}

/// Camera control request payload (without reply channel)
#[derive(Debug, Clone)]
pub enum CameraControlRequest {
    /// Set exposure: auto mode ignores `value`; manual mode uses it
    SetExposure { mode: ExposureMode, value: i32 },
    /// Set white balance: auto mode ignores `temperature`; manual mode uses it
    SetWhiteBalance {
        mode: WhiteBalanceMode,
        temperature: i32,
    },
    /// Set gain level in camera-specific units
    SetGain { value: i32 },
    /// Set brightness level
    SetBrightness { value: i32 },
    /// Set contrast level
    SetContrast { value: i32 },
}

/// Result returned by the capture loop after applying a camera control
#[derive(Debug, Clone)]
pub struct ControlResult {
    pub success: bool,
    pub message: String,
    /// Current value after applying the control; -1 if not applicable or unreadable
    pub current_value: i32,
}

impl ControlResult {
    pub fn ok(message: impl Into<String>, current_value: i32) -> Self {
        Self {
            success: true,
            message: message.into(),
            current_value,
        }
    }

    pub fn err(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            current_value: -1,
        }
    }
}

/// A camera control command sent from a service handler to the capture loop.
///
/// The outer control channel is unbounded, so the sender never blocks on
/// enqueue.  Each command carries a `SyncSender` reply channel (capacity 1)
/// so the capture loop can post the result without blocking even if the
/// service handler has not yet called `recv()`.
pub struct ControlCommand {
    pub request: CameraControlRequest,
    pub reply: std::sync::mpsc::SyncSender<ControlResult>,
}

/// Sender side of the control channel (service handlers hold a clone)
pub type ControlSender = std::sync::mpsc::Sender<ControlCommand>;
/// Receiver side of the control channel (owned by the capture loop)
pub type ControlReceiver = std::sync::mpsc::Receiver<ControlCommand>;

/// Create a new control channel pair
pub fn create_control_channel() -> (ControlSender, ControlReceiver) {
    std::sync::mpsc::channel()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exposure_mode_parse() {
        assert_eq!(ExposureMode::try_from("auto").unwrap(), ExposureMode::Auto);
        assert_eq!(
            ExposureMode::try_from("manual").unwrap(),
            ExposureMode::Manual
        );
        assert_eq!(ExposureMode::try_from("AUTO").unwrap(), ExposureMode::Auto);
        assert!(ExposureMode::try_from("invalid").is_err());
    }

    #[test]
    fn test_white_balance_mode_parse() {
        assert_eq!(
            WhiteBalanceMode::try_from("auto").unwrap(),
            WhiteBalanceMode::Auto
        );
        assert_eq!(
            WhiteBalanceMode::try_from("manual").unwrap(),
            WhiteBalanceMode::Manual
        );
        assert!(WhiteBalanceMode::try_from("bad").is_err());
    }

    #[test]
    fn test_control_result_ok() {
        let r = ControlResult::ok("Brightness set to 128", 128);
        assert!(r.success);
        assert_eq!(r.current_value, 128);
    }

    #[test]
    fn test_control_result_err() {
        let r = ControlResult::err("Not supported");
        assert!(!r.success);
        assert_eq!(r.current_value, -1);
    }
}
