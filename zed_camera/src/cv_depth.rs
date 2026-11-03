//! Rectify + SGBM depth through the `opencv` crate.
//!
//! At creation it parses the factory `.conf`, builds the per-eye intrinsics
//! and the inter-eye rotation, and computes each eye's remap into its
//! rectified view under the lens model the conf writes: a radial-tangential
//! pair (k1, k2, p1, p2, k3) through `stereo_rectify` and
//! `init_undistort_rectify_map`, a fisheye pair (k1 to k4, no tangential
//! term) through `fisheye_stereo_rectify` and
//! `fisheye_init_undistort_rectify_map`. Per frame it decodes the
//! side-by-side YUYV, remaps both eyes, and runs `StereoSGBM`
//! (`MODE_SGBM_3WAY`, smoothness penalties 24 * block² and 96 * block²),
//! converting fixed-point disparity to millimetres. Every heavy step is a
//! maintained OpenCV call.

use opencv::calib3d::{self, StereoMatcherTrait, StereoSGBM, StereoSGBM_MODE_SGBM_3WAY};
use opencv::core::{
    CV_32FC1, CV_64FC1, Mat, MatTraitConst, Rect, Scalar, Size as CvSize, ToInputArray,
};
use opencv::imgproc::{
    self, COLOR_BGR2GRAY, COLOR_BGR2RGB, COLOR_YUV2BGR_YUYV, INTER_AREA, INTER_LINEAR,
};
use opencv::prelude::MatTraitConstManual;

use crate::calibration::{Lenses, StereoConf};
use crate::depth_settings::{DepthSettings, num_disparities_for};
use crate::geometry::{MAX_Z16_DEPTH_MM, RectifiedLeft, depth_mm as depth_mm_of};
use crate::resolution::Resolution;
use crate::size::Size;

/// Millimetre depth for a fixed geometry, backed by OpenCV.
pub struct CvDepth {
    matcher: opencv::core::Ptr<StereoSGBM>,
    map_left_x: Mat,
    map_left_y: Mat,
    map_right_x: Mat,
    map_right_y: Mat,
    eye_width: i32,
    eye_height: i32,
    downscale: i32,
    /// The rectified left projection the colour image is remapped into.
    rectified_left: RectifiedLeft,
    baseline_mm: f64,
    numerator: f64,
    num_disparities: u32,
}

impl CvDepth {
    /// The matcher for `resolution`'s eye size, through its section of the
    /// factory calibration `conf_text`.
    pub fn create(
        conf_text: &str,
        resolution: Resolution,
        settings: DepthSettings,
    ) -> Result<Self, String> {
        let block = settings.block_size() as i32;
        let downscale = settings.downscale() as i32;
        let eye = resolution.eye_size();
        let conf = StereoConf::from_conf_str(conf_text, resolution)
            .map_err(|e| format!("calibration: {e}"))?;

        let (w, h) = (eye.width as i32, eye.height as i32);
        let size = CvSize::new(w, h);
        let k_left = matrix_3x3(conf.left.fx, conf.left.cx, conf.left.fy, conf.left.cy)?;
        let k_right = matrix_3x3(conf.right.fx, conf.right.cx, conf.right.fy, conf.right.cy)?;

        // Inter-eye rotation from the ZED (rx, cv, rz) Rodrigues vector (1x3),
        // and the translation column (3x1) with baseline on x.
        let rvec = Mat::from_slice(&[conf.rx, conf.cv, conf.rz])
            .map_err(cv)?
            .try_clone()
            .map_err(cv)?;
        let mut rotation = Mat::default();
        calib3d::rodrigues_def(&rvec, &mut rotation).map_err(cv)?;
        let translation = Mat::from_slice(&[conf.baseline, conf.ty, conf.tz])
            .map_err(cv)?
            .reshape(1, 3)
            .map_err(cv)?
            .try_clone()
            .map_err(cv)?;

        let eyes = Eyes {
            k_left,
            k_right,
            rotation,
            translation,
            size,
        };
        let Rectification {
            map_left_x,
            map_left_y,
            map_right_x,
            map_right_y,
            p1,
        } = rectify(&eyes, &conf.lenses)?;

        // Rectified focal length is P1(0,0); disparity to depth uses it, and
        // the disparity search range derives from it and the requested floor.
        let fx = *p1.at_2d::<f64>(0, 0).map_err(cv)?;
        let rectified_left = RectifiedLeft {
            fx,
            fy: *p1.at_2d::<f64>(1, 1).map_err(cv)?,
            cx: *p1.at_2d::<f64>(0, 2).map_err(cv)?,
            cy: *p1.at_2d::<f64>(1, 2).map_err(cv)?,
        };
        let baseline_mm = conf.baseline;
        let numerator = fx / downscale as f64 * baseline_mm * 16.0;
        let num_disp = num_disparities_for(
            fx / downscale as f64,
            baseline_mm,
            settings.min_depth_m(),
            (w / downscale) as u32,
        )? as i32;

        let p1_penalty = 24 * block * block;
        let matcher = StereoSGBM::create(
            0,
            num_disp,
            block,
            p1_penalty,
            4 * p1_penalty,
            96,
            63,
            5,
            255,
            1,
            StereoSGBM_MODE_SGBM_3WAY,
        )
        .map_err(cv)?;

        Ok(Self {
            matcher,
            map_left_x,
            map_left_y,
            map_right_x,
            map_right_y,
            eye_width: w,
            eye_height: h,
            downscale,
            rectified_left,
            baseline_mm,
            numerator,
            num_disparities: num_disp as u32,
        })
    }

    /// The derived disparity search range.
    #[cfg(test)]
    fn num_disparities(&self) -> u32 {
        self.num_disparities
    }

    /// The rectified left projection (P1 of the rectification): the camera the
    /// published colour image shows, at the eye resolution.
    pub fn rectified_left(&self) -> RectifiedLeft {
        self.rectified_left
    }

    /// The nearest depth the derived search range can measure.
    pub fn min_depth_floor_mm(&self) -> f64 {
        self.rectified_left.fx / self.downscale as f64 * self.baseline_mm
            / self.num_disparities as f64
    }

    /// The farthest depth the stream can report: the depth of the smallest
    /// disparity SGBM resolves, a sixteenth of a pixel, or the last z16
    /// reading when that is nearer.
    pub fn max_depth_mm(&self) -> f64 {
        self.numerator.min(MAX_Z16_DEPTH_MM)
    }

    pub fn out_size(&self) -> Size {
        Size {
            width: (self.eye_width / self.downscale) as u32,
            height: (self.eye_height / self.downscale) as u32,
        }
    }

    /// Rectify + match one side-by-side YUYV frame. `left_rgb` is
    /// eye_width*eye_height*3; `depth_mm` matches `out_size()`.
    ///
    /// A buffer that is not that size is refused rather than asserted: `yuyv`
    /// carries whatever the V4L2 driver handed the caller, and a short frame
    /// on the capture path must degrade to a dropped frame, not an unwind.
    pub fn process(
        &mut self,
        yuyv: &[u8],
        left_rgb: &mut [u8],
        depth_mm: &mut [u16],
    ) -> Result<(), String> {
        let (w, h) = (self.eye_width, self.eye_height);
        let (dw, dh) = (w / self.downscale, h / self.downscale);
        require_len("yuyv", yuyv.len(), (2 * w * h * 2) as usize)?;
        require_len("left_rgb", left_rgb.len(), (w * h * 3) as usize)?;
        require_len("depth_mm", depth_mm.len(), (dw * dh) as usize)?;

        // Side-by-side YUYV -> full BGR -> per-eye views.
        let yuyv_mat = Mat::from_slice(yuyv)
            .map_err(cv)?
            .reshape(2, h)
            .map_err(cv)?
            .try_clone()
            .map_err(cv)?;
        let mut frame_bgr = Mat::default();
        imgproc::cvt_color_def(&yuyv_mat, &mut frame_bgr, COLOR_YUV2BGR_YUYV).map_err(cv)?;
        let left_raw = frame_bgr.roi(Rect::new(0, 0, w, h)).map_err(cv)?;
        let right_raw = frame_bgr.roi(Rect::new(w, 0, w, h)).map_err(cv)?;

        let mut left_rect = Mat::default();
        let mut right_rect = Mat::default();
        remap_linear(
            &left_raw,
            &mut left_rect,
            &self.map_left_x,
            &self.map_left_y,
        )?;
        remap_linear(
            &right_raw,
            &mut right_rect,
            &self.map_right_x,
            &self.map_right_y,
        )?;

        let mut left_out = Mat::default();
        imgproc::cvt_color_def(&left_rect, &mut left_out, COLOR_BGR2RGB).map_err(cv)?;
        left_rgb.copy_from_slice(left_out.data_bytes().map_err(cv)?);

        let mut left_gray = Mat::default();
        let mut right_gray = Mat::default();
        imgproc::cvt_color_def(&left_rect, &mut left_gray, COLOR_BGR2GRAY).map_err(cv)?;
        imgproc::cvt_color_def(&right_rect, &mut right_gray, COLOR_BGR2GRAY).map_err(cv)?;
        let (left_gray, right_gray) = if self.downscale > 1 {
            let small = CvSize::new(w / self.downscale, h / self.downscale);
            (
                downscaled(&left_gray, small)?,
                downscaled(&right_gray, small)?,
            )
        } else {
            (left_gray, right_gray)
        };

        let mut disparity16 = Mat::default();
        self.matcher
            .compute(&left_gray, &right_gray, &mut disparity16)
            .map_err(cv)?;

        // depth_mm = (fx / downscale) * baseline_mm * 16 / disparity16.
        // The matcher's output is the one length not fixed by the caller, and
        // zip would silently stop at the shorter side: with the output buffer
        // reused across frames, a short disparity map would leave the tail
        // holding the previous frame's depth, published as current.
        let disp = disparity16.data_typed::<i16>().map_err(cv)?;
        require_len("disparity", disp.len(), depth_mm.len())?;
        for (out, &d) in depth_mm.iter_mut().zip(disp.iter()) {
            *out = depth_mm_of(self.numerator, d);
        }
        Ok(())
    }
}

fn cv(e: opencv::Error) -> String {
    format!("opencv: {e}")
}

/// Refuse a frame buffer that is not the length the geometry calls for, naming
/// the buffer and both lengths so the log identifies the mismatch. Lengths are
/// element counts, which differ between the u8 and u16 buffers.
fn require_len(what: &str, got: usize, want: usize) -> Result<(), String> {
    (got == want)
        .then_some(())
        .ok_or_else(|| format!("{what} length is {got}, expected {want}"))
}

fn matrix_3x3(fx: f64, cx: f64, fy: f64, cy: f64) -> Result<Mat, String> {
    Mat::from_slice_2d(&[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        .and_then(|m| m.try_clone())
        .map_err(cv)
        .inspect(|m| debug_assert_eq!(m.typ(), CV_64FC1))
}

/// Distortion coefficients as the column OpenCV takes.
fn coefficients(values: &[f64]) -> Result<Mat, String> {
    Mat::from_slice(values)
        .map_err(cv)?
        .reshape(1, values.len() as i32)
        .map_err(cv)?
        .try_clone()
        .map_err(cv)
}

/// The stereo pair's geometry that rectification starts from, under any
/// lens model.
struct Eyes {
    k_left: Mat,
    k_right: Mat,
    rotation: Mat,
    translation: Mat,
    size: CvSize,
}

/// Each eye's remap into its rectified view, and the rectified left
/// projection (P1).
struct Rectification {
    map_left_x: Mat,
    map_left_y: Mat,
    map_right_x: Mat,
    map_right_y: Mat,
    p1: Mat,
}

/// Rectification of `eyes` under the lens model of `lenses`, into pinhole
/// views the size of the eye whose every pixel holds the scene: alpha 0 for
/// the radial-tangential model, balance 0 and the field of view unscaled for
/// the fisheye model.
fn rectify(eyes: &Eyes, lenses: &Lenses) -> Result<Rectification, String> {
    let (left, right) = match lenses {
        Lenses::RadialTangential { left, right } => (left.as_slice(), right.as_slice()),
        Lenses::Fisheye { left, right } => (left.as_slice(), right.as_slice()),
    };
    let (dist_left, dist_right) = (coefficients(left)?, coefficients(right)?);
    let (mut r1, mut r2, mut p1, mut p2, mut q) = (
        Mat::default(),
        Mat::default(),
        Mat::default(),
        Mat::default(),
        Mat::default(),
    );
    match lenses {
        Lenses::RadialTangential { .. } => calib3d::stereo_rectify(
            &eyes.k_left,
            &dist_left,
            &eyes.k_right,
            &dist_right,
            eyes.size,
            &eyes.rotation,
            &eyes.translation,
            &mut r1,
            &mut r2,
            &mut p1,
            &mut p2,
            &mut q,
            calib3d::CALIB_ZERO_DISPARITY,
            0.0,
            eyes.size,
            None,
            None,
        ),
        Lenses::Fisheye { .. } => calib3d::fisheye_stereo_rectify(
            &eyes.k_left,
            &dist_left,
            &eyes.k_right,
            &dist_right,
            eyes.size,
            &eyes.rotation,
            &eyes.translation,
            &mut r1,
            &mut r2,
            &mut p1,
            &mut p2,
            &mut q,
            calib3d::CALIB_ZERO_DISPARITY,
            eyes.size,
            0.0,
            1.0,
        ),
    }
    .map_err(cv)?;
    let (map_left_x, map_left_y) =
        undistort_rectify_map(lenses, &eyes.k_left, &dist_left, &r1, &p1, eyes.size)?;
    let (map_right_x, map_right_y) =
        undistort_rectify_map(lenses, &eyes.k_right, &dist_right, &r2, &p2, eyes.size)?;
    Ok(Rectification {
        map_left_x,
        map_left_y,
        map_right_x,
        map_right_y,
        p1,
    })
}

/// One eye's remap (x, y) from its rectifying rotation `r` and projection
/// `p`, under the lens model of `lenses`.
fn undistort_rectify_map(
    lenses: &Lenses,
    k: &Mat,
    dist: &Mat,
    r: &Mat,
    p: &Mat,
    size: CvSize,
) -> Result<(Mat, Mat), String> {
    let (mut map_x, mut map_y) = (Mat::default(), Mat::default());
    match lenses {
        Lenses::RadialTangential { .. } => calib3d::init_undistort_rectify_map(
            k, dist, r, p, size, CV_32FC1, &mut map_x, &mut map_y,
        ),
        Lenses::Fisheye { .. } => calib3d::fisheye_init_undistort_rectify_map(
            k, dist, r, p, size, CV_32FC1, &mut map_x, &mut map_y,
        ),
    }
    .map_err(cv)?;
    Ok((map_x, map_y))
}

fn remap_linear(
    src: &impl ToInputArray,
    dst: &mut Mat,
    map_x: &Mat,
    map_y: &Mat,
) -> Result<(), String> {
    imgproc::remap(
        src,
        dst,
        map_x,
        map_y,
        INTER_LINEAR,
        opencv::core::BORDER_CONSTANT,
        Scalar::default(),
    )
    .map_err(cv)
}

fn downscaled(gray: &Mat, size: CvSize) -> Result<Mat, String> {
    let mut out = Mat::default();
    imgproc::resize(gray, &mut out, size, 0.0, 0.0, INTER_AREA).map_err(cv)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RESOLUTION: Resolution = Resolution::Vga;
    const FOCAL_PX: f64 = 350.0;
    const CX: f64 = 336.0;
    const CY: f64 = 188.0;
    const BASELINE_MM: f64 = 120.0;

    /// A VGA rig with identical eyes, no distortion and no rotation between
    /// them, so rectification leaves both images where they are.
    fn ideal_vga_conf() -> String {
        let eye = format!("fx={FOCAL_PX}\nfy={FOCAL_PX}\ncx={CX}\ncy={CY}\n");
        format!(
            "[LEFT_CAM_VGA]\n{eye}[RIGHT_CAM_VGA]\n{eye}\
             [STEREO]\nBaseline={BASELINE_MM}\nRX_VGA=0\nCV_VGA=0\nRZ_VGA=0\n"
        )
    }

    /// Luma noise in YUYV's limited range, the same for every run.
    fn noise_rows(width: usize, height: usize) -> Vec<Vec<u8>> {
        let mut state = 0x2545_f491_u32;
        (0..height)
            .map(|_| {
                (0..width)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        16 + (state % 220) as u8
                    })
                    .collect()
            })
            .collect()
    }

    /// A side-by-side YUYV frame whose right eye sees the left eye's texture
    /// moved `shift` pixels to the left: a disparity of `shift` everywhere.
    fn side_by_side_yuyv(shift: usize) -> Vec<u8> {
        let eye = RESOLUTION.eye_size();
        let (width, height) = (eye.width as usize, eye.height as usize);
        let texture = noise_rows(width + shift, height);
        let mut frame = Vec::with_capacity(2 * width * height * 2);
        for row in &texture {
            let eyes = row[..width].iter().chain(&row[shift..shift + width]);
            let lumas: Vec<u8> = eyes.copied().collect();
            let (pairs, _) = lumas.as_chunks::<2>();
            for &[y0, y1] in pairs {
                frame.extend([y0, 128, y1, 128]);
            }
        }
        frame
    }

    #[test]
    fn an_ideal_rig_reads_a_uniform_disparity_as_its_depth() {
        let shift = 20;
        let settings = DepthSettings::new(0.5, 5, 1).unwrap();
        let mut depth = CvDepth::create(&ideal_vga_conf(), RESOLUTION, settings)
            .expect("an ideal calibration rectifies");
        let Size {
            width: out_width,
            height: out_height,
        } = depth.out_size();
        let eye = RESOLUTION.eye_size();
        let mut left_rgb = vec![0; (eye.width * eye.height * 3) as usize];
        let mut depth_mm = vec![0; (out_width * out_height) as usize];

        depth
            .process(&side_by_side_yuyv(shift), &mut left_rgb, &mut depth_mm)
            .unwrap();

        // Away from the borders and from the columns left of the search
        // range, where the matcher has no right pixel to compare against.
        let margin = 8;
        let first_column = depth.num_disparities() as usize + margin;
        let rows = margin..out_height as usize - margin;
        let columns = first_column..out_width as usize - margin;
        let inner: Vec<u16> = rows
            .flat_map(|y| columns.clone().map(move |x| (y, x)))
            .map(|(y, x)| depth_mm[y * out_width as usize + x])
            .collect();
        let mut measured: Vec<u16> = inner.iter().copied().filter(|&mm| mm > 0).collect();
        assert!(
            measured.len() * 10 >= inner.len() * 9,
            "only {} of {} inner pixels have a depth",
            measured.len(),
            inner.len()
        );
        measured.sort_unstable();
        let median = f64::from(measured[measured.len() / 2]);
        let expected = FOCAL_PX * BASELINE_MM / shift as f64;
        assert!(
            (median - expected).abs() <= expected * 0.01,
            "median depth {median} mm, expected {expected} mm"
        );
    }

    /// The wall `fisheye_wall_yuyv` draws: square to the rig, this far ahead.
    const WALL_MM: f64 = 1000.0;
    /// The wall's texture cell, 4.2 pixels at the image centre at WALL_MM.
    const WALL_CELL_MM: f64 = 12.0;

    /// The ideal VGA rig's eyes as equidistant fisheye lenses, with no
    /// distortion beyond the equidistant projection.
    fn fisheye_vga_conf() -> String {
        ideal_vga_conf().replace(
            &format!("cy={CY}\n"),
            &format!("cy={CY}\nk1=0\nk2=0\nk3=0\nk4=0\n"),
        )
    }

    /// Value noise on the wall: a hashed luma at each lattice point,
    /// bilinear between them, in YUYV's limited range.
    fn wall_luma(x_mm: f64, y_mm: f64) -> u8 {
        let lattice = |i: f64, j: f64| {
            let mut h = (i as i64 as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ (j as i64 as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
            h ^= h >> 31;
            h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            h ^= h >> 29;
            (h >> 11) as f64 / (1u64 << 53) as f64
        };
        let (gx, gy) = (x_mm / WALL_CELL_MM, y_mm / WALL_CELL_MM);
        let (ix, iy) = (gx.floor(), gy.floor());
        let (fx, fy) = (gx - ix, gy - iy);
        let top = lattice(ix, iy) * (1.0 - fx) + lattice(ix + 1.0, iy) * fx;
        let bottom = lattice(ix, iy + 1.0) * (1.0 - fx) + lattice(ix + 1.0, iy + 1.0) * fx;
        (16.0 + 219.0 * (top * (1.0 - fy) + bottom * fy)) as u8
    }

    /// A side-by-side YUYV frame of the wall WALL_MM ahead, through
    /// equidistant lenses: the image point of a ray at angle θ from the axis
    /// lies FOCAL_PX * θ pixels from the centre. The right eye sits
    /// BASELINE_MM right of the left.
    fn fisheye_wall_yuyv() -> Vec<u8> {
        let eye = RESOLUTION.eye_size();
        let (width, height) = (eye.width as usize, eye.height as usize);
        let mut frame = Vec::with_capacity(2 * width * height * 2);
        for v in 0..height {
            let lumas: Vec<u8> = [0.0, BASELINE_MM]
                .iter()
                .flat_map(|&eye_x_mm| {
                    (0..width).map(move |u| {
                        let (a, b) = ((u as f64 - CX) / FOCAL_PX, (v as f64 - CY) / FOCAL_PX);
                        let theta = a.hypot(b);
                        let tan_per_angle = if theta == 0.0 {
                            1.0
                        } else {
                            theta.tan() / theta
                        };
                        wall_luma(
                            eye_x_mm + WALL_MM * a * tan_per_angle,
                            WALL_MM * b * tan_per_angle,
                        )
                    })
                })
                .collect();
            let (pairs, _) = lumas.as_chunks::<2>();
            for &[y0, y1] in pairs {
                frame.extend([y0, 128, y1, 128]);
            }
        }
        frame
    }

    /// The median depth of the 15-pixel square at each of three columns of
    /// the middle row, a quarter, half and three quarters across: the
    /// depth `conf` reads from `frame`.
    fn depth_across(conf: &str, frame: &[u8]) -> [f64; 3] {
        let settings = DepthSettings::new(0.5, 5, 1).unwrap();
        let mut depth = CvDepth::create(conf, RESOLUTION, settings).expect("the rig rectifies");
        let Size { width, height } = depth.out_size();
        let eye = RESOLUTION.eye_size();
        let mut left_rgb = vec![0; (eye.width * eye.height * 3) as usize];
        let mut depth_mm = vec![0; (width * height) as usize];
        depth.process(frame, &mut left_rgb, &mut depth_mm).unwrap();
        [1, 2, 3].map(|quarter| {
            let (column, row) = (quarter * width as usize / 4, height as usize / 2);
            let mut patch: Vec<u16> = (row - 7..=row + 7)
                .flat_map(|y| (column - 7..=column + 7).map(move |x| (y, x)))
                .map(|(y, x)| depth_mm[y * width as usize + x])
                .filter(|&mm| mm > 0)
                .collect();
            assert!(
                patch.len() >= 15 * 15 / 2,
                "column {column}: {} pixels have a depth",
                patch.len()
            );
            patch.sort_unstable();
            f64::from(patch[patch.len() / 2])
        })
    }

    #[test]
    fn a_fisheye_rig_reads_a_flat_wall_at_one_depth_across_the_image() {
        let frame = fisheye_wall_yuyv();
        // SGBM's sixteenth of a pixel is 0.15 % of the wall's 42-pixel
        // disparity; 2 % holds that and the remap's blur of a 4-pixel cell.
        for (quarter, mm) in depth_across(&fisheye_vga_conf(), &frame)
            .into_iter()
            .enumerate()
        {
            assert!(
                (mm - WALL_MM).abs() <= WALL_MM * 0.02,
                "quarter {}: {mm} mm, the wall is {WALL_MM} mm ahead",
                quarter + 1
            );
        }
        // The same frame rectified as pinhole lenses reads the disparity
        // cos² θ of the true one: 26 % farther at a quarter across, where θ
        // is 0.48 rad.
        let [left, _, right] = depth_across(&ideal_vga_conf(), &frame);
        for mm in [left, right] {
            assert!(
                mm > WALL_MM * 1.1,
                "a pinhole rectification read {mm} mm at a side"
            );
        }
    }
}
