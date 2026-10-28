//! Camera control request and result types, shared between the service
//! handlers and the device layer that applies them.

use camera_core::controls::{AutoManualMode, ControlFailure};

/// The mode `name` spells in any case. `control` names the control in the
/// refusal of any other name.
pub(crate) fn parse_mode(name: &str, control: &str) -> Result<AutoManualMode, String> {
    AutoManualMode::from_name(&name.to_lowercase())
        .ok_or_else(|| format!("Invalid {control} mode: '{name}', expected 'auto' or 'manual'"))
}

/// A camera control to apply
#[derive(Debug, Clone)]
pub enum CameraControlRequest {
    /// Set exposure: auto mode ignores `value`; manual mode uses it
    SetExposure { mode: AutoManualMode, value: i32 },
    /// Set white balance: auto mode ignores `temperature`; manual mode uses it
    SetWhiteBalance {
        mode: AutoManualMode,
        temperature: i32,
    },
    /// Set gain level in camera-specific units
    SetGain { value: i32 },
    /// Set brightness level
    SetBrightness { value: i32 },
    /// Set contrast level
    SetContrast { value: i32 },
}

/// Result of applying a camera control: the value the driver kept
/// ([`camera_core::controls::VALUE_UNAVAILABLE`] in an auto mode), or why the
/// write gave none.
pub type ControlResult = Result<i32, ControlFailure>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exposure_mode_parse() {
        assert_eq!(parse_mode("auto", "exposure"), Ok(AutoManualMode::Auto));
        assert_eq!(parse_mode("manual", "exposure"), Ok(AutoManualMode::Manual));
        assert_eq!(parse_mode("AUTO", "exposure"), Ok(AutoManualMode::Auto));
        assert_eq!(
            parse_mode("invalid", "exposure"),
            Err("Invalid exposure mode: 'invalid', expected 'auto' or 'manual'".to_string())
        );
    }

    #[test]
    fn test_white_balance_mode_parse() {
        assert_eq!(
            parse_mode("auto", "white balance"),
            Ok(AutoManualMode::Auto)
        );
        assert_eq!(
            parse_mode("manual", "white balance"),
            Ok(AutoManualMode::Manual)
        );
        assert_eq!(
            parse_mode("bad", "white balance"),
            Err("Invalid white balance mode: 'bad', expected 'auto' or 'manual'".to_string())
        );
    }
}
