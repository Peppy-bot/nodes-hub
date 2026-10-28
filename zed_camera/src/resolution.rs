//! ZED capture modes: per-eye geometry and the frame rates each supports.

use std::fmt;

use crate::size::Size;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Resolution {
    Vga,
    Hd720,
    Hd1080,
    Hd2k,
}

impl Resolution {
    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "vga" => Ok(Self::Vga),
            "hd720" => Ok(Self::Hd720),
            "hd1080" => Ok(Self::Hd1080),
            "hd2k" => Ok(Self::Hd2k),
            other => Err(format!(
                "resolution must be vga|hd720|hd1080|hd2k, got {other:?}"
            )),
        }
    }

    /// The suffix of this mode's sections and keys in a factory calibration.
    pub fn conf_key(self) -> &'static str {
        match self {
            Self::Vga => "VGA",
            Self::Hd720 => "HD",
            Self::Hd1080 => "FHD",
            Self::Hd2k => "2K",
        }
    }

    /// One eye's geometry; the wire carries both eyes side by side.
    pub fn eye_size(self) -> Size {
        let (width, height) = match self {
            Self::Vga => (672, 376),
            Self::Hd720 => (1280, 720),
            Self::Hd1080 => (1920, 1080),
            Self::Hd2k => (2208, 1242),
        };
        Size { width, height }
    }

    fn legal_fps(self) -> &'static [u8] {
        match self {
            Self::Vga => &[15, 30, 60, 100],
            Self::Hd720 => &[15, 30, 60],
            Self::Hd1080 => &[15, 30],
            Self::Hd2k => &[15],
        }
    }

    /// `fps` as a frame rate this mode runs at.
    pub fn frame_rate(self, fps: u32) -> Result<LegalFrameRate, String> {
        self.legal_fps()
            .iter()
            .find(|&&legal| u32::from(legal) == fps)
            .map(|&legal| LegalFrameRate(legal))
            .ok_or_else(|| {
                format!(
                    "frame rate {fps} is not legal for {self}; legal: {:?}",
                    self.legal_fps()
                )
            })
    }
}

/// A frame rate legal for the mode it was parsed against, frames per second.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LegalFrameRate(u8);

impl LegalFrameRate {
    pub fn per_second(self) -> u8 {
        self.0
    }
}

impl fmt::Display for LegalFrameRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Resolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Vga => "vga",
            Self::Hd720 => "hd720",
            Self::Hd1080 => "hd1080",
            Self::Hd2k => "hd2k",
        };
        write!(f, "{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trips_every_mode() {
        for mode in [
            Resolution::Vga,
            Resolution::Hd720,
            Resolution::Hd1080,
            Resolution::Hd2k,
        ] {
            assert_eq!(Resolution::parse(&mode.to_string()).unwrap(), mode);
        }
        assert!(Resolution::parse("4k").is_err());
    }

    #[test]
    fn fps_legality_follows_the_mode_tables() {
        assert_eq!(Resolution::Vga.frame_rate(100).unwrap().per_second(), 100);
        assert!(Resolution::Hd2k.frame_rate(30).is_err());
        assert_eq!(Resolution::Hd720.frame_rate(60).unwrap().per_second(), 60);
    }
}
