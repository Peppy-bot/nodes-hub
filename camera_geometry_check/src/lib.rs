//! The arithmetic of camera_geometry_check that is its own, free of peppy
//! types: the depth pixel a fraction of the way across the image, and the
//! median reading around it. The pinhole arithmetic from that reading to a
//! point is camera_core's.

#![forbid(unsafe_code)]

use camera_core::encoding::{DepthFrame, DepthFrameSizeError};

/// The pixel a fraction of the way across an image of `size` pixels: 0 is the
/// first pixel, 1 the last.
pub fn pixel_at(fraction: f64, size: u32) -> u32 {
    let last = f64::from(size.saturating_sub(1));
    (fraction.clamp(0.0, 1.0) * last).round() as u32
}

/// The median reading, in meters, of the square window around pixel (u, v)
/// of a z16 frame, and how many readings it is the median of. 0 is the
/// stream's no-reading value and is left out; a window with no reading at all
/// has no median.
pub fn median_reading_m(
    frame: &[u8],
    width: u32,
    height: u32,
    (u, v): (u32, u32),
    window: u32,
    depth_unit: f32,
) -> Result<Option<(f64, usize)>, DepthFrameSizeError> {
    let frame = DepthFrame::parse(frame, width, height)?;
    let rows = v.saturating_sub(window)..=v.saturating_add(window).min(height.saturating_sub(1));
    let columns = u.saturating_sub(window)..=u.saturating_add(window).min(width.saturating_sub(1));
    let mut readings: Vec<u16> = rows
        .flat_map(|row| columns.clone().map(move |column| (column, row)))
        .filter_map(|(column, row)| frame.sample(column, row))
        .filter(|&sample| sample != 0)
        .collect();
    if readings.is_empty() {
        return Ok(None);
    }
    readings.sort_unstable();
    let median = readings[readings.len() / 2];
    Ok(Some((
        f64::from(median) * f64::from(depth_unit),
        readings.len(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractions_name_pixels_from_the_first_to_the_last() {
        assert_eq!(pixel_at(0.0, 640), 0);
        assert_eq!(pixel_at(1.0, 640), 639);
        assert_eq!(pixel_at(0.5, 641), 320);
        assert_eq!(pixel_at(7.0, 640), 639);
        assert_eq!(pixel_at(-1.0, 640), 0);
        assert_eq!(pixel_at(0.5, 0), 0);
    }

    #[test]
    fn the_median_leaves_out_holes_and_survives_a_flying_pixel() {
        // A 5x5 frame reading 1500 mm, with two holes and one flying pixel.
        let mut samples = [1500_u16; 25];
        samples[6] = 0;
        samples[7] = 0;
        samples[12] = 9000;
        let frame: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let (reading, count) = median_reading_m(&frame, 5, 5, (2, 2), 1, 0.001)
            .unwrap()
            .unwrap();
        assert_eq!(count, 7);
        assert!((reading - 1.5).abs() < 1e-6);
        // At the image's edge the window is what fits.
        let (_, count) = median_reading_m(&frame, 5, 5, (0, 0), 1, 0.001)
            .unwrap()
            .unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn a_window_of_holes_has_no_median_and_a_short_frame_is_refused() {
        let frame = vec![0_u8; 5 * 5 * 2];
        assert_eq!(
            median_reading_m(&frame, 5, 5, (2, 2), 1, 0.001).unwrap(),
            None
        );
        assert!(median_reading_m(&frame[..10], 5, 5, (2, 2), 1, 0.001).is_err());
    }
}
