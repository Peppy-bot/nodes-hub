//! Where the two published streams' pixels point, in the terms of
//! camera_geometry:v1.
//!
//! Nothing here touches OpenCV or the device: it takes the rectified left
//! projection `stereo_rectify` produced and the sizes the node publishes, and
//! answers the pinhole model of each stream. The conventions are OpenCV's on
//! both sides (the centre of the first pixel is (0, 0); +X right, +Y down, +Z
//! forward), so the projection's numbers cross into the contract unchanged.
//!
//! - The colour stream is the rectified left eye, so its model is that
//!   projection as it is, and rectified means without distortion.
//! - The depth stream is matched on the same view resized to
//!   eye / downscale, so its model is the colour one seen through a resized
//!   grid: the same camera, the same viewpoint, fewer pixels. The streams are
//!   aligned by construction, and the depth-to-colour transform is the
//!   identity.
//!
//! A value the contract cannot carry (a focal length that is not a positive
//! number, a depth range that is not one) is refused with the reason rather
//! than answered: a consumer divides by these numbers.

/// The largest reading the depth stream carries, millimetres. The conversion
/// below keeps 65535 out of the stream, so the last z16 value is never a
/// reading.
pub const MAX_Z16_DEPTH_MM: f64 = 65534.0;

/// One stream's pinhole model as camera_geometry:v1 answers it. Both streams
/// are rectified, so neither carries distortion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pinhole {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// The rectified left projection (P1 of `stereo_rectify`) at the eye
/// resolution: the camera the published colour image was remapped into.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RectifiedLeft {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// What the node answers over camera_geometry:v1, fixed for the session: the
/// calibration and the capture mode do not change while it runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    pub color: Pinhole,
    pub depth: Pinhole,
    /// The depths the depth stream can report, metres. A sample outside them
    /// reads 0.
    pub min_depth_m: f64,
    pub max_depth_m: f64,
}

impl Pinhole {
    /// The same camera seen through a grid of another size, as
    /// `imgproc::resize` maps it: pixel i of the resized grid covers
    /// source pixels around (i + 0.5) / s - 0.5, so with pixel centres on
    /// whole numbers the principal point moves to (c + 0.5) * s - 0.5 and the
    /// focal length scales by s. The scale is the ratio of the sizes, per
    /// axis, which is what the resize really applied.
    pub fn resized(&self, width: u32, height: u32) -> Self {
        let sx = f64::from(width) / f64::from(self.width);
        let sy = f64::from(height) / f64::from(self.height);
        Self {
            width,
            height,
            fx: self.fx * sx,
            fy: self.fy * sy,
            cx: (self.cx + 0.5) * sx - 0.5,
            cy: (self.cy + 0.5) * sy - 0.5,
        }
    }
}

impl Geometry {
    /// The geometry of the two published streams. `eye` is the size of the
    /// colour stream, `left` the projection it was rectified into, `depth`
    /// the size of the depth stream, and the depth range is the matcher's, in
    /// millimetres as the matcher counts them.
    pub fn new(
        eye: (u32, u32),
        left: RectifiedLeft,
        depth: (u32, u32),
        min_depth_mm: f64,
        max_depth_mm: f64,
    ) -> Result<Self, String> {
        let RectifiedLeft { fx, fy, cx, cy } = left;
        if eye.0 == 0 || eye.1 == 0 || depth.0 == 0 || depth.1 == 0 {
            return Err(format!(
                "a published stream has no size: colour {}x{}, depth {}x{}",
                eye.0, eye.1, depth.0, depth.1
            ));
        }
        if !(fx.is_finite() && fx > 0.0 && fy.is_finite() && fy > 0.0) {
            return Err(format!(
                "the rectified focal lengths are not positive numbers: fx {fx}, fy {fy}"
            ));
        }
        if !(cx.is_finite() && cy.is_finite()) {
            return Err(format!(
                "the rectified principal point is not a number: cx {cx}, cy {cy}"
            ));
        }
        if !(min_depth_mm.is_finite()
            && max_depth_mm.is_finite()
            && min_depth_mm > 0.0
            && max_depth_mm > min_depth_mm)
        {
            return Err(format!(
                "the matcher's depth range is not a range: {min_depth_mm} mm to {max_depth_mm} mm"
            ));
        }
        let color = Pinhole {
            width: eye.0,
            height: eye.1,
            fx,
            fy,
            cx,
            cy,
        };
        Ok(Self {
            color,
            depth: color.resized(depth.0, depth.1),
            min_depth_m: min_depth_mm / 1000.0,
            max_depth_m: max_depth_mm / 1000.0,
        })
    }
}

/// One disparity sample as millimetres of depth. `numerator` is
/// focal * baseline_mm * 16 and `disparity16` is SGBM's fixed point, in
/// sixteenths of a pixel. 0 is the stream's no-reading value, and everything
/// that is not a depth the stream can carry becomes it: no match (a
/// disparity that is zero or negative), a depth nearer than a millimetre or
/// beyond what z16 holds, and a result that is not a number at all.
pub fn depth_mm(numerator: f64, disparity16: i16) -> u16 {
    if disparity16 <= 0 {
        return 0;
    }
    let mm = numerator / f64::from(disparity16);
    if (1.0..MAX_Z16_DEPTH_MM + 1.0).contains(&mm) {
        mm as u16
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ZED Mini at hd720 after rectification, the magnitudes a real unit
    /// has: a principal point near, not at, the image centre.
    const LEFT: RectifiedLeft = RectifiedLeft {
        fx: 700.25,
        fy: 700.75,
        cx: 652.125,
        cy: 351.5,
    };

    fn geometry(depth: (u32, u32)) -> Geometry {
        Geometry::new((1280, 720), LEFT, depth, 100.0, 65534.0).unwrap()
    }

    #[test]
    fn the_colour_model_is_the_rectified_projection_unchanged() {
        let color = geometry((640, 360)).color;
        assert_eq!((color.width, color.height), (1280, 720));
        assert_eq!((color.fx, color.fy), (LEFT.fx, LEFT.fy));
        assert_eq!((color.cx, color.cy), (LEFT.cx, LEFT.cy));
    }

    #[test]
    fn the_depth_model_is_the_colour_one_through_the_resized_grid() {
        let depth = geometry((640, 360)).depth;
        assert_eq!((depth.width, depth.height), (640, 360));
        assert_eq!((depth.fx, depth.fy), (350.125, 350.375));
        // (c + 0.5) / 2 - 0.5, not c / 2: halving the principal point would
        // be right only for pixel centres on half numbers.
        assert_eq!(depth.cx, (652.125 + 0.5) / 2.0 - 0.5);
        assert_eq!(depth.cy, (351.5 + 0.5) / 2.0 - 0.5);
        assert_ne!(depth.cx, 652.125 / 2.0);
    }

    #[test]
    fn a_depth_stream_at_full_size_shares_the_colour_model() {
        let g = geometry((1280, 720));
        assert_eq!(g.depth, g.color);
    }

    #[test]
    fn a_resized_grid_keeps_every_ray_where_it_was() {
        // A point of the scene projects to the same place in both grids: a
        // pixel of the small grid is the centre of the block it averages.
        let g = geometry((320, 180));
        let (x, y, z) = (0.3, -0.2, 1.5);
        let u_color = g.color.fx * x / z + g.color.cx;
        let v_color = g.color.fy * y / z + g.color.cy;
        let u_depth = g.depth.fx * x / z + g.depth.cx;
        let v_depth = g.depth.fy * y / z + g.depth.cy;
        assert!(((u_color + 0.5) / 4.0 - 0.5 - u_depth).abs() < 1e-9);
        assert!(((v_color + 0.5) / 4.0 - 0.5 - v_depth).abs() < 1e-9);
    }

    #[test]
    fn the_scale_is_the_ratio_of_the_sizes_on_each_axis() {
        // hd2k is 1242 high: a downscale of 4 publishes 310 rows, and the
        // resize scaled the rows by 310 / 1242, not by a quarter.
        let g = Geometry::new((2208, 1242), LEFT, (552, 310), 100.0, 65534.0).unwrap();
        assert_eq!(g.depth.fx, LEFT.fx * 0.25);
        assert_eq!(g.depth.fy, LEFT.fy * 310.0 / 1242.0);
    }

    #[test]
    fn the_depth_range_is_answered_in_metres() {
        let g = Geometry::new((1280, 720), LEFT, (640, 360), 98.5, 65534.0).unwrap();
        assert_eq!(g.min_depth_m, 0.0985);
        assert_eq!(g.max_depth_m, 65.534);
    }

    #[test]
    fn values_a_consumer_cannot_divide_by_are_refused_with_the_reason() {
        for bad in [0.0, -700.0, f64::NAN, f64::INFINITY] {
            let refused = Geometry::new(
                (1280, 720),
                RectifiedLeft { fx: bad, ..LEFT },
                (640, 360),
                100.0,
                65534.0,
            )
            .unwrap_err();
            assert!(refused.contains("focal lengths"), "{refused}");
            assert!(
                Geometry::new(
                    (1280, 720),
                    RectifiedLeft { fy: bad, ..LEFT },
                    (640, 360),
                    100.0,
                    65534.0
                )
                .is_err()
            );
        }
        let refused = Geometry::new(
            (1280, 720),
            RectifiedLeft {
                cx: f64::NAN,
                ..LEFT
            },
            (640, 360),
            100.0,
            65534.0,
        )
        .unwrap_err();
        assert!(refused.contains("principal point"), "{refused}");
        assert!(Geometry::new((1280, 720), LEFT, (0, 360), 100.0, 65534.0).is_err());
        assert!(Geometry::new((0, 720), LEFT, (640, 360), 100.0, 65534.0).is_err());
        for (min, max) in [
            (0.0, 65534.0),
            (100.0, 100.0),
            (f64::NAN, 65534.0),
            (100.0, f64::INFINITY),
        ] {
            let refused = Geometry::new((1280, 720), LEFT, (640, 360), min, max).unwrap_err();
            assert!(refused.contains("depth range"), "{refused}");
        }
    }

    #[test]
    fn a_disparity_becomes_millimetres() {
        // 350 px focal, 63 mm baseline: 16 sixteenths (one pixel) is 22.05 m.
        let numerator = 350.0 * 63.0 * 16.0;
        assert_eq!(depth_mm(numerator, 16), 22050);
        assert_eq!(depth_mm(numerator, 16 * 100), 220);
    }

    #[test]
    fn whatever_is_not_a_depth_reads_zero() {
        let numerator = 350.0 * 63.0 * 16.0;
        // No match.
        assert_eq!(depth_mm(numerator, 0), 0);
        assert_eq!(depth_mm(numerator, -16), 0);
        // Beyond what z16 holds: one sixteenth of a pixel is 352.8 m.
        assert_eq!(depth_mm(numerator, 1), 0);
        // Nearer than a millimetre.
        assert_eq!(depth_mm(100.0, i16::MAX), 0);
        // A numerator that is not a number never becomes a reading.
        assert_eq!(depth_mm(f64::NAN, 16), 0);
        assert_eq!(depth_mm(f64::INFINITY, 16), 0);
        assert_eq!(depth_mm(f64::NEG_INFINITY, 16), 0);
        // The largest reading is kept, the value after it is not.
        assert_eq!(depth_mm(MAX_Z16_DEPTH_MM, 1), 65534);
        assert_eq!(depth_mm(65534.9, 1), 65534);
        assert_eq!(depth_mm(65535.0, 1), 0);
    }
}
