//! Typed mode enums: color pixel format, RGB-D alignment mode, and a shared
//! auto/manual mode for sensor options that toggle between automatic control
//! and an explicit setpoint (exposure, white_balance, ...).
//!
//! All public string forms are case-sensitive (`"auto"`, not `"Auto"`).

use std::fmt;

use realsense_rust::kind::Rs2Format;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorFormat {
    /// 8-bit packed `R,G,B`. 3 bytes/pixel. Universal; tensor-friendly.
    Rgb8,
    /// 8-bit packed `B,G,R`. OpenCV's default channel order.
    Bgr8,
    /// YCbCr 4:2:2 packed (Y0,U,Y1,V). 2 bytes/pixel; ~33% wire savings
    /// vs Rgb8.
    Yuyv,
    /// JPEG-compressed (variable, ~10:1 typical). Smallest on the wire but
    /// consumers need a JPEG decoder.
    Mjpeg,
}

impl ColorFormat {
    /// Human-readable list of formats `TryFrom<&str>` accepts. Update
    /// alongside the match arms below.
    pub const SUPPORTED: &'static str = "rgb8/bgr8/yuyv/mjpeg";

    pub fn rs2_format(self) -> Rs2Format {
        match self {
            Self::Rgb8 => Rs2Format::Rgb8,
            Self::Bgr8 => Rs2Format::Bgr8,
            Self::Yuyv => Rs2Format::Yuyv,
            Self::Mjpeg => Rs2Format::Mjpeg,
        }
    }

    pub fn topic_encoding(self) -> &'static str {
        match self {
            Self::Rgb8 => "rgb8",
            Self::Bgr8 => "bgr8",
            Self::Yuyv => "yuyv",
            Self::Mjpeg => "mjpeg",
        }
    }
}

impl TryFrom<&str> for ColorFormat {
    type Error = String;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "rgb8" => Ok(Self::Rgb8),
            "bgr8" => Ok(Self::Bgr8),
            "yuyv" => Ok(Self::Yuyv),
            "mjpeg" => Ok(Self::Mjpeg),
            other => Err(format!(
                "unknown color format '{other}' (expected {})",
                Self::SUPPORTED
            )),
        }
    }
}

impl fmt::Display for ColorFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.topic_encoding())
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

/// Auto/manual toggle for sensor options that pair a mode with a numeric
/// setpoint (exposure, white_balance, ...). 
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoManualMode {
    Auto,
    Manual,
}

impl AutoManualMode {
    pub fn is_auto(self) -> bool {
        matches!(self, Self::Auto)
    }

    /// Parse the wire string. `kind` names the parameter for error messages
    /// (e.g. `"exposure"`, `"white_balance"`).
    pub fn parse(s: &str, kind: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "manual" => Ok(Self::Manual),
            other => Err(format!(
                "unknown {kind} mode '{other}' (expected 'auto' or 'manual')"
            )),
        }
    }
}

impl fmt::Display for AutoManualMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Manual => "manual",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_format_parse_round_trip() {
        for fmt in [
            ColorFormat::Rgb8,
            ColorFormat::Bgr8,
            ColorFormat::Yuyv,
            ColorFormat::Mjpeg,
        ] {
            let parsed = ColorFormat::try_from(fmt.topic_encoding()).unwrap();
            assert_eq!(parsed, fmt);
        }
    }

    #[test]
    fn color_format_rejects_unknown() {
        assert!(ColorFormat::try_from("z16").is_err());
        assert!(ColorFormat::try_from("").is_err());
        assert!(ColorFormat::try_from("RGB8").is_err());
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
        assert_eq!(AutoManualMode::parse("auto", "x"), Ok(AutoManualMode::Auto));
        assert_eq!(AutoManualMode::parse("manual", "x"), Ok(AutoManualMode::Manual));
    }

    #[test]
    fn auto_manual_is_case_sensitive() {
        assert!(AutoManualMode::parse("Auto", "x").is_err());
        assert!(AutoManualMode::parse("AUTO", "x").is_err());
    }

    #[test]
    fn auto_manual_error_carries_kind() {
        let err = AutoManualMode::parse("foo", "exposure").unwrap_err();
        assert!(err.contains("exposure"), "missing kind: {err}");
        assert!(err.contains("foo"), "missing input: {err}");
    }
}
