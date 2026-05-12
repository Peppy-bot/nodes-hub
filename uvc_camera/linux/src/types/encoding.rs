use std::fmt;
use std::str::FromStr;

/// Video encoding format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Rgb8,
    Bgr8,
    Mjpeg,
}

impl FromStr for Encoding {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "rgb8" => Ok(Self::Rgb8),
            "bgr8" => Ok(Self::Bgr8),
            "mjpeg" => Ok(Self::Mjpeg),
            _ => Err(format!(
                "Invalid encoding '{s}'. Supported encodings are: rgb8, bgr8, mjpeg"
            )),
        }
    }
}

impl fmt::Display for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rgb8 => write!(f, "rgb8"),
            Self::Bgr8 => write!(f, "bgr8"),
            Self::Mjpeg => write!(f, "mjpeg"),
        }
    }
}
