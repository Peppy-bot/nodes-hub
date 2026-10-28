//! The modes the node parses from its parameters and requests: the colour
//! encoding with its librealsense format, the RGB-D alignment mode, and the
//! auto/manual mode of a sensor option with a setpoint (exposure,
//! white_balance, ...).
//!
//! All public string forms are case-sensitive (`"auto"`, not `"Auto"`).

use std::fmt;

use camera_core::controls::AutoManualMode;
use camera_core::encoding::ColorEncoding;
use realsense_rust::kind::Rs2Format;

use crate::stream::StreamPair;

/// Human-readable list of the colour formats [`parse_color_format`] accepts.
const SUPPORTED_COLOR_FORMATS: &str = "rgb8/bgr8/yuyv/mjpeg";

/// The colour encoding the `color_format` parameter names.
pub fn parse_color_format(name: &str) -> Result<ColorEncoding, String> {
    ColorEncoding::from_name(name).ok_or_else(|| {
        format!("unknown color format '{name}' (expected {SUPPORTED_COLOR_FORMATS})")
    })
}

/// The librealsense format the colour stream is opened in for `encoding`.
pub fn rs2_format(encoding: ColorEncoding) -> Rs2Format {
    match encoding {
        ColorEncoding::Rgb8 => Rs2Format::Rgb8,
        ColorEncoding::Bgr8 => Rs2Format::Bgr8,
        ColorEncoding::Yuyv => Rs2Format::Yuyv,
        ColorEncoding::Mjpeg => Rs2Format::Mjpeg,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignMode {
    /// Emit raw color + raw depth in their native sensor coordinate frames.
    None,
    /// Warp depth into color's coordinate frame and resolution before emit.
    DepthToColor,
    /// Warp color into depth's coordinate frame and resolution before emit.
    ColorToDepth,
}

impl AlignMode {
    pub const SUPPORTED: &'static str = "none/depth_to_color/color_to_depth";

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::DepthToColor => "depth_to_color",
            Self::ColorToDepth => "color_to_depth",
        }
    }

    /// What each stream is published with under this mode, of what each is
    /// opened with: librealsense warps an aligned stream into the other's
    /// viewpoint and resolution, so the aligned stream takes the target's.
    pub fn published<T: Clone>(self, opened: StreamPair<T>) -> StreamPair<T> {
        match self {
            Self::None => opened,
            Self::DepthToColor => StreamPair {
                color: opened.color.clone(),
                depth: opened.color,
            },
            Self::ColorToDepth => StreamPair {
                color: opened.depth.clone(),
                depth: opened.depth,
            },
        }
    }
}

impl TryFrom<&str> for AlignMode {
    type Error = String;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "none" => Ok(Self::None),
            "depth_to_color" => Ok(Self::DepthToColor),
            "color_to_depth" => Ok(Self::ColorToDepth),
            other => Err(format!(
                "unknown align mode '{other}' (expected {})",
                Self::SUPPORTED
            )),
        }
    }
}

impl fmt::Display for AlignMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The auto/manual mode a request spells. `kind` names the parameter in the
/// refusal of any other spelling (e.g. `"exposure"`, `"white_balance"`).
pub fn parse_mode(s: &str, kind: &str) -> Result<AutoManualMode, String> {
    AutoManualMode::from_name(s)
        .ok_or_else(|| format!("unknown {kind} mode '{s}' (expected 'auto' or 'manual')"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_format_parse_round_trip() {
        for fmt in ColorEncoding::ALL {
            let parsed = parse_color_format(fmt.name()).unwrap();
            assert_eq!(parsed, fmt);
        }
    }

    #[test]
    fn color_format_rejects_unknown() {
        assert!(parse_color_format("z16").is_err());
        assert!(parse_color_format("").is_err());
        assert!(parse_color_format("RGB8").is_err());
    }

    #[test]
    fn align_mode_parse_round_trip() {
        for mode in [
            AlignMode::None,
            AlignMode::DepthToColor,
            AlignMode::ColorToDepth,
        ] {
            let parsed = AlignMode::try_from(mode.as_str()).unwrap();
            assert_eq!(parsed, mode);
        }
    }

    #[test]
    fn align_mode_rejects_unknown() {
        assert!(AlignMode::try_from("color").is_err());
        assert!(AlignMode::try_from("").is_err());
    }

    #[test]
    fn auto_manual_parses_known_strings() {
        assert_eq!(parse_mode("auto", "x"), Ok(AutoManualMode::Auto));
        assert_eq!(parse_mode("manual", "x"), Ok(AutoManualMode::Manual));
    }

    #[test]
    fn auto_manual_is_case_sensitive() {
        assert!(parse_mode("Auto", "x").is_err());
        assert!(parse_mode("AUTO", "x").is_err());
    }

    #[test]
    fn auto_manual_error_carries_kind() {
        let err = parse_mode("foo", "exposure").unwrap_err();
        assert!(err.contains("exposure"), "missing kind: {err}");
        assert!(err.contains("foo"), "missing input: {err}");
    }
}
