//! ZED capture modes: per-eye geometry and the frame rates each supports.

use std::fmt;

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

    /// One eye's geometry; the wire carries both eyes side by side.
    pub fn eye_size(self) -> (u32, u32) {
        match self {
            Self::Vga => (672, 376),
            Self::Hd720 => (1280, 720),
            Self::Hd1080 => (1920, 1080),
            Self::Hd2k => (2208, 1242),
        }
    }

    pub fn legal_fps(self) -> &'static [u32] {
        match self {
            Self::Vga => &[15, 30, 60, 100],
            Self::Hd720 => &[15, 30, 60],
            Self::Hd1080 => &[15, 30],
            Self::Hd2k => &[15],
        }
    }

    pub fn validate_fps(self, fps: u32) -> Result<(), String> {
        if self.legal_fps().contains(&fps) {
            return Ok(());
        }
        Err(format!(
            "frame rate {fps} is not legal for {self}; legal: {:?}",
            self.legal_fps()
        ))
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
        assert!(Resolution::Vga.validate_fps(100).is_ok());
        assert!(Resolution::Hd2k.validate_fps(30).is_err());
        assert!(Resolution::Hd720.validate_fps(60).is_ok());
    }
}
