//! The arithmetic of camera_geometry_check, free of peppy types: from a depth
//! pixel and the camera's answers to a point in the camera's optical frame,
//! under the conventions of camera_geometry:v1, which are OpenCV's. The centre
//! of the first pixel is (0, 0); the optical frame has +X to the right of the
//! image, +Y down it and +Z along the view.

#![forbid(unsafe_code)]

/// Iterations of the fixed-point inverse of the Brown-Conrady polynomial,
/// whichever way a model runs it. OpenCV's undistortPoints runs a handful;
/// the lenses this meets converge long before.
const UNDISTORT_ITERATIONS: usize = 20;

/// The lens of a stream, as camera_geometry:v1 names it: none; "plumb_bob",
/// OpenCV's polynomial taking an ideal point to the distorted one; or
/// "inverse_plumb_bob", the same polynomial run from the distorted point to
/// the ideal one.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Lens {
    None,
    PlumbBob([f64; 5]),
    InversePlumbBob([f64; 5]),
}

/// One stream's pinhole model, as a camera_geometry:v1 intrinsics answer
/// carries it.
#[derive(Debug, Clone, PartialEq)]
pub struct Pinhole {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub distortion_model: String,
    pub distortion: Vec<f64>,
}

/// The depth optical frame in the colour optical frame, as
/// get_depth_to_color_extrinsics carries it: metres and a unit quaternion
/// [x, y, z, w].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub position: [f64; 3],
    pub orientation: [f64; 4],
}

impl Pinhole {
    /// The lens this stream's answer names, with its k1, k2, p1, p2, k3.
    /// Anything else is a model this tool cannot apply, refused by name.
    fn lens(&self) -> Result<Lens, String> {
        match (self.distortion_model.as_str(), self.distortion.as_slice()) {
            ("none", []) => Ok(Lens::None),
            ("plumb_bob", &[k1, k2, p1, p2, k3]) => Ok(Lens::PlumbBob([k1, k2, p1, p2, k3])),
            ("inverse_plumb_bob", &[k1, k2, p1, p2, k3]) => {
                Ok(Lens::InversePlumbBob([k1, k2, p1, p2, k3]))
            }
            (model, coefficients) => Err(format!(
                "distortion model {model:?} with {} coefficients is not one camera_geometry names",
                coefficients.len()
            )),
        }
    }

    /// The ideal image point (x, y) at unit depth that pixel (u, v) shows:
    /// the pixel through fx, fy, cx, cy, then the distortion taken off.
    pub fn ray(&self, u: f64, v: f64) -> Result<(f64, f64), String> {
        let (xd, yd) = ((u - self.cx) / self.fx, (v - self.cy) / self.fy);
        Ok(match self.lens()? {
            Lens::None => (xd, yd),
            // The polynomial runs from the distorted point to the ideal one:
            // one application, nothing to invert.
            Lens::InversePlumbBob(k) => apply(&k, xd, yd),
            // It runs the other way, so the ideal point is found by
            // iterating it from the distorted one.
            Lens::PlumbBob(k) => invert(&k, xd, yd),
        })
    }

    /// The pixel that shows a point of this camera's optical frame.
    pub fn project(&self, point: [f64; 3]) -> Result<(f64, f64), String> {
        let [px, py, pz] = point;
        if pz <= 0.0 {
            return Err(format!("the point is not in front of the camera: z {pz}"));
        }
        let (x, y) = (px / pz, py / pz);
        let (x, y) = match self.lens()? {
            Lens::None => (x, y),
            Lens::PlumbBob(k) => apply(&k, x, y),
            // Under the inverse model the polynomial undistorts, so the
            // distorted point is the iteration, from the ideal one.
            Lens::InversePlumbBob(k) => invert(&k, x, y),
        };
        Ok((self.fx * x + self.cx, self.fy * y + self.cy))
    }

    /// The point of this camera's optical frame that depth pixel (u, v)
    /// shows, from its reading in metres. Under "z" the reading is the
    /// distance along the optical axis; under "range" it is the straight
    /// line from the camera to the point.
    pub fn deproject(
        &self,
        u: f64,
        v: f64,
        reading_m: f64,
        depth_model: &str,
    ) -> Result<[f64; 3], String> {
        let (x, y) = self.ray(u, v)?;
        let z = match depth_model {
            "z" => reading_m,
            "range" => reading_m / (x * x + y * y + 1.0).sqrt(),
            other => {
                return Err(format!(
                    "depth model {other:?} is not one camera_geometry names"
                ));
            }
        };
        Ok([x * z, y * z, z])
    }
}

/// The Brown-Conrady polynomial applied once to (x, y).
fn apply(k: &[f64; 5], x: f64, y: f64) -> (f64, f64) {
    let (radial, dx, dy) = distortion_terms(k, x, y);
    (x * radial + dx, y * radial + dy)
}

/// The point the polynomial takes to (xd, yd), found by iterating it from
/// there: the fixed point of x = (xd - tangential) / radial.
fn invert(k: &[f64; 5], xd: f64, yd: f64) -> (f64, f64) {
    let (mut x, mut y) = (xd, yd);
    for _ in 0..UNDISTORT_ITERATIONS {
        let (radial, dx, dy) = distortion_terms(k, x, y);
        x = (xd - dx) / radial;
        y = (yd - dy) / radial;
    }
    (x, y)
}

/// The radial factor and the tangential shift the polynomial applies to the
/// point (x, y), with k1, k2, p1, p2, k3 as OpenCV orders them.
fn distortion_terms(k: &[f64; 5], x: f64, y: f64) -> (f64, f64, f64) {
    let [k1, k2, p1, p2, k3] = *k;
    let r2 = x * x + y * y;
    let radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
    let dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    let dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
    (radial, dx, dy)
}

impl Pose {
    /// A point of the depth optical frame, in the colour optical frame:
    /// R(q) * p + t.
    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let [x, y, z, w] = self.orientation;
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let axis = [x, y, z];
        let t = cross(axis, p).map(|c| 2.0 * c);
        let u = cross(axis, t);
        std::array::from_fn(|i| p[i] + w * t[i] + u[i] + self.position[i])
    }
}

/// The pixel a fraction of the way across an image of `size` pixels: 0 is the
/// first pixel, 1 the last.
pub fn pixel_at(fraction: f64, size: u32) -> u32 {
    let last = f64::from(size.saturating_sub(1));
    (fraction.clamp(0.0, 1.0) * last).round() as u32
}

/// The median reading, in metres, of the square window around pixel (u, v)
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
) -> Result<Option<(f64, usize)>, String> {
    let expected = width as usize * height as usize * 2;
    if frame.len() != expected {
        return Err(format!(
            "the depth frame is {} bytes, a {width}x{height} z16 frame is {expected}",
            frame.len()
        ));
    }
    let mut readings = Vec::new();
    for row in v.saturating_sub(window)..=(v + window).min(height.saturating_sub(1)) {
        for column in u.saturating_sub(window)..=(u + window).min(width.saturating_sub(1)) {
            let at = (row as usize * width as usize + column as usize) * 2;
            let sample = u16::from_le_bytes([frame[at], frame[at + 1]]);
            if sample != 0 {
                readings.push(sample);
            }
        }
    }
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

    fn ideal() -> Pinhole {
        Pinhole {
            width: 640,
            height: 360,
            fx: 369.0547,
            fy: 369.0547,
            cx: 319.5,
            cy: 179.5,
            distortion_model: "none".to_string(),
            distortion: Vec::new(),
        }
    }

    fn distorted() -> Pinhole {
        Pinhole {
            distortion_model: "plumb_bob".to_string(),
            distortion: vec![-0.055, 0.066, -0.0007, 0.0005, -0.021],
            ..ideal()
        }
    }

    fn inverse() -> Pinhole {
        Pinhole {
            distortion_model: "inverse_plumb_bob".to_string(),
            ..distorted()
        }
    }

    fn close(a: [f64; 3], b: [f64; 3], tolerance: f64) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < tolerance)
    }

    #[test]
    fn the_principal_point_looks_straight_ahead() {
        let point = ideal().deproject(319.5, 179.5, 1.25, "z").unwrap();
        assert_eq!(point, [0.0, 0.0, 1.25]);
    }

    #[test]
    fn a_pixel_to_the_right_and_below_is_plus_x_and_plus_y() {
        // One focal length to the right of the principal point is 45 degrees.
        let camera = ideal();
        let point = camera.deproject(319.5 + 369.0547, 179.5, 2.0, "z").unwrap();
        assert!(close(point, [2.0, 0.0, 2.0], 1e-12));
        let point = camera.deproject(319.5, 179.5 + 369.0547, 2.0, "z").unwrap();
        assert!(close(point, [0.0, 2.0, 2.0], 1e-12));
    }

    #[test]
    fn a_range_reading_is_the_straight_line_to_the_point() {
        let camera = ideal();
        let along_the_axis = camera.deproject(319.5 + 369.0547, 179.5, 2.0, "z").unwrap();
        let straight_line = (along_the_axis.iter().map(|c| c * c).sum::<f64>()).sqrt();
        let point = camera
            .deproject(319.5 + 369.0547, 179.5, straight_line, "range")
            .unwrap();
        assert!(close(point, along_the_axis, 1e-12));
        assert!(camera.deproject(0.0, 0.0, 1.0, "inverse").is_err());
    }

    #[test]
    fn deprojecting_what_was_projected_returns_the_point() {
        for camera in [ideal(), distorted(), inverse()] {
            for point in [[0.3, -0.2, 1.5], [-0.6, 0.35, 0.9], [0.0, 0.0, 3.0]] {
                let (u, v) = camera.project(point).unwrap();
                let back = camera.deproject(u, v, point[2], "z").unwrap();
                assert!(close(back, point, 1e-9), "{camera:?} {point:?} {back:?}");
            }
        }
    }

    #[test]
    fn distortion_moves_a_corner_pixel_and_leaves_the_centre() {
        let (ideal, distorted) = (ideal(), distorted());
        let corner = [0.75, 0.42, 1.0];
        let (u0, v0) = ideal.project(corner).unwrap();
        let (u1, v1) = distorted.project(corner).unwrap();
        assert!((u0 - u1).abs() > 1.0 || (v0 - v1).abs() > 1.0);
        let centre = ideal.project([0.0, 0.0, 1.0]).unwrap();
        let (u, v) = distorted.project([0.0, 0.0, 1.0]).unwrap();
        assert!((centre.0 - u).abs() < 1e-12 && (centre.1 - v).abs() < 1e-12);
    }

    #[test]
    fn the_inverse_model_applies_the_polynomial_once_from_the_distorted_point() {
        // inverse_plumb_bob is one application to the distorted point, so
        // the ray is the closed form, and the same numbers under plumb_bob
        // give a different ray, because that model has to be inverted.
        let camera = inverse();
        let (u, v) = (camera.cx + 0.3 * camera.fx, camera.cy - 0.2 * camera.fy);
        let k = [-0.055, 0.066, -0.0007, 0.0005, -0.021];
        let expected = apply(&k, 0.3, -0.2);
        let (x, y) = camera.ray(u, v).unwrap();
        assert!((x - expected.0).abs() < 1e-12 && (y - expected.1).abs() < 1e-12);
        let (xf, yf) = distorted().ray(u, v).unwrap();
        assert!((x - xf).abs() > 1e-4 || (y - yf).abs() > 1e-4);
    }

    #[test]
    fn the_inverse_and_forward_models_agree_for_a_small_lens() {
        // To first order the inverse of the polynomial with coefficients c
        // is the polynomial with -c, so for a small lens the direct inverse
        // model with -c and the iterated forward model with c agree on a
        // ray, and a lens ten times smaller agrees a hundred times better.
        let lens = |scale: f64| {
            [
                -0.5 * scale,
                0.6 * scale,
                -0.07 * scale,
                0.05 * scale,
                -0.2 * scale,
            ]
        };
        let mut gaps = Vec::new();
        for scale in [0.01, 0.001] {
            let forward = Pinhole {
                distortion_model: "plumb_bob".to_string(),
                distortion: lens(scale).to_vec(),
                ..ideal()
            };
            let direct = Pinhole {
                distortion_model: "inverse_plumb_bob".to_string(),
                distortion: lens(-scale).to_vec(),
                ..ideal()
            };
            let (u, v) = (
                ideal().cx + 0.8 * ideal().fx,
                ideal().cy + 0.45 * ideal().fy,
            );
            let (xf, yf) = forward.ray(u, v).unwrap();
            let (xd, yd) = direct.ray(u, v).unwrap();
            let gap = ((xf - xd).powi(2) + (yf - yd).powi(2)).sqrt();
            assert!(gap < 20.0 * scale * scale, "scale {scale}: gap {gap}");
            gaps.push(gap);
        }
        assert!(gaps[0] / gaps[1] > 50.0, "{gaps:?}");
    }

    #[test]
    fn a_model_the_contract_does_not_name_is_refused() {
        let fisheye = Pinhole {
            distortion_model: "kannala_brandt".to_string(),
            distortion: vec![0.1, 0.0, 0.0, 0.0],
            ..ideal()
        };
        assert!(fisheye.ray(10.0, 10.0).is_err());
        let short = Pinhole {
            distortion_model: "plumb_bob".to_string(),
            distortion: vec![0.1, 0.0],
            ..ideal()
        };
        assert!(short.ray(10.0, 10.0).is_err());
        assert!(ideal().project([0.0, 0.0, -1.0]).is_err());
    }

    #[test]
    fn the_pose_takes_a_depth_point_into_the_colour_frame() {
        // A quarter turn about +Z takes +X to +Y, then the offset is added.
        let half = std::f64::consts::FRAC_PI_4;
        let pose = Pose {
            position: [0.015, 0.0, -0.001],
            orientation: [0.0, 0.0, half.sin(), half.cos()],
        };
        assert!(close(
            pose.apply([1.0, 0.0, 2.0]),
            [0.015, 1.0, 1.999],
            1e-12
        ));
        let identity = Pose {
            position: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],
        };
        assert_eq!(identity.apply([0.3, -0.2, 1.5]), [0.3, -0.2, 1.5]);
    }

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
