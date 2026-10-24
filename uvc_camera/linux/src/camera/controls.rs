//! Camera control request and result types, shared between the service
//! handlers and the device layer that applies them.

use std::io;

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

/// Reported when a control has no meaningful current value: the mode (e.g.
/// auto exposure) leaves the driver managing it.
pub const VALUE_UNAVAILABLE: i32 = -1;

/// Result of applying a camera control: the value the driver kept
/// ([`VALUE_UNAVAILABLE`] in an auto mode), or why the write gave none.
pub type ControlResult = Result<i32, ControlFailure>;

/// Why a control write gave no value: the failure kind of peppy that says
/// why, and the reason in words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlFailure {
    pub kind: ControlFailureKind,
    pub message: String,
}

/// The failure kinds of peppy that a control write gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlFailureKind {
    /// The device does not offer the control, or does not take the value.
    Rejected,
    /// The control does not take a write now: it is read-only, inactive in
    /// the current mode, or held by another application.
    PreconditionFailed,
    /// The device is gone.
    ResourceUnavailable,
    /// The write failed for another reason.
    AttemptFailed,
}

impl ControlFailure {
    /// A request the node does not take, such as an unknown mode.
    pub fn rejected(message: impl Into<String>) -> Self {
        Self {
            kind: ControlFailureKind::Rejected,
            message: message.into(),
        }
    }

    /// The failure of a write of `what` that the driver refused with
    /// `error`, by the errno that VIDIOC_S_EXT_CTRLS documents for it.
    pub fn of_write(what: &str, error: &io::Error) -> Self {
        let kind = match error.raw_os_error() {
            Some(libc::EINVAL | libc::ERANGE) => ControlFailureKind::Rejected,
            Some(libc::EACCES | libc::EBUSY) => ControlFailureKind::PreconditionFailed,
            Some(libc::ENODEV) => ControlFailureKind::ResourceUnavailable,
            _ => ControlFailureKind::AttemptFailed,
        };
        Self {
            kind,
            message: format!("Failed to set {what}: {error}"),
        }
    }
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

    fn kind_of_errno(errno: i32) -> ControlFailureKind {
        ControlFailure::of_write("gain", &io::Error::from_raw_os_error(errno)).kind
    }

    #[test]
    fn a_control_or_a_value_the_device_does_not_take_is_rejected() {
        assert_eq!(kind_of_errno(libc::EINVAL), ControlFailureKind::Rejected);
        assert_eq!(kind_of_errno(libc::ERANGE), ControlFailureKind::Rejected);
    }

    #[test]
    fn a_control_that_takes_no_write_now_fails_a_precondition() {
        assert_eq!(
            kind_of_errno(libc::EACCES),
            ControlFailureKind::PreconditionFailed
        );
        assert_eq!(
            kind_of_errno(libc::EBUSY),
            ControlFailureKind::PreconditionFailed
        );
    }

    #[test]
    fn a_device_that_is_gone_is_unavailable() {
        assert_eq!(
            kind_of_errno(libc::ENODEV),
            ControlFailureKind::ResourceUnavailable
        );
    }

    #[test]
    fn any_other_refusal_of_the_driver_is_a_failed_attempt() {
        assert_eq!(kind_of_errno(libc::EIO), ControlFailureKind::AttemptFailed);
        let without_errno = io::Error::other("no errno");
        assert_eq!(
            ControlFailure::of_write("gain", &without_errno).kind,
            ControlFailureKind::AttemptFailed
        );
    }

    #[test]
    fn a_failed_write_names_the_control_and_the_cause() {
        let failure =
            ControlFailure::of_write("contrast", &io::Error::from_raw_os_error(libc::ERANGE));
        assert!(failure.message.contains("contrast"), "{}", failure.message);
        assert!(
            failure
                .message
                .contains(&io::Error::from_raw_os_error(libc::ERANGE).to_string()),
            "{}",
            failure.message
        );
    }
}
