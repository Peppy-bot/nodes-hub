//! Validated stereo-matching settings. Construction is the only entry, so a
//! `DepthSettings` in hand is always legal; illegal launcher values are
//! rejected instead of silently coerced.

#[derive(Clone, Copy, Debug)]
pub struct DepthSettings {
    min_depth_m: f64,
    block_size: u32,
    downscale: u32,
}

impl DepthSettings {
    pub fn new(min_depth_m: f64, block_size: u32, downscale: u32) -> Result<Self, String> {
        if !min_depth_m.is_finite() || min_depth_m <= 0.0 {
            return Err(format!(
                "min_depth_m must be a positive distance, got {min_depth_m}"
            ));
        }
        if !(3..=11).contains(&block_size) || block_size.is_multiple_of(2) {
            return Err(format!(
                "block_size must be odd and within 3..=11, got {block_size}"
            ));
        }
        if downscale == 0 {
            return Err("downscale must be at least 1".to_string());
        }
        Ok(Self {
            min_depth_m,
            block_size,
            downscale,
        })
    }

    pub fn min_depth_m(self) -> f64 {
        self.min_depth_m
    }

    pub fn block_size(self) -> u32 {
        self.block_size
    }

    pub fn downscale(self) -> u32 {
        self.downscale
    }
}

/// The disparity search range that reaches `min_depth_m` given the rectified
/// focal length at matching scale and the stereo baseline: the smallest
/// multiple of 16 with fx * baseline / range <= min_depth. `max_disparities`
/// bounds the search to the matched image width.
pub fn num_disparities_for(
    fx_at_scale: f64,
    baseline_mm: f64,
    min_depth_m: f64,
    max_disparities: u32,
) -> Result<u32, String> {
    let needed = fx_at_scale * baseline_mm / (min_depth_m * 1000.0);
    let range = 16 * (needed / 16.0).ceil().max(1.0) as u32;
    if range > max_disparities {
        return Err(format!(
            "min_depth_m {min_depth_m} needs a {range}-disparity search, beyond the \
             {max_disparities} this resolution/downscale supports"
        ));
    }
    Ok(range)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_reference_style_settings() {
        assert!(DepthSettings::new(0.3, 3, 2).is_ok());
        assert!(DepthSettings::new(1.0, 11, 1).is_ok());
    }

    #[test]
    fn rejects_illegal_values_instead_of_coercing() {
        assert!(DepthSettings::new(0.0, 3, 2).is_err());
        assert!(DepthSettings::new(f64::NAN, 3, 2).is_err());
        assert!(DepthSettings::new(0.3, 4, 2).is_err()); // even block
        assert!(DepthSettings::new(0.3, 13, 2).is_err()); // block too large
        assert!(DepthSettings::new(0.3, 3, 0).is_err());
    }

    #[test]
    fn disparity_range_derives_from_the_calibration() {
        // SN10383163 hd720 at downscale 2: fx/2 = 386.1, baseline 62.902 mm.
        // A 0.26 m floor needs 93.4 disparities -> rounds up to 96.
        assert_eq!(num_disparities_for(386.1, 62.902, 0.26, 640), Ok(96));
        // Asking for 0.05 m at that geometry blows past the width bound.
        assert!(num_disparities_for(386.1, 62.902, 0.05, 320).is_err());
        // A very distant floor still searches at least one 16-step.
        assert_eq!(num_disparities_for(386.1, 62.902, 100.0, 640), Ok(16));
    }
}
