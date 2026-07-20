//! Rectify + SGBM depth through the `opencv` crate.
//!
//! This is the Stereolabs reference recipe (the same one the vendored
//! `initCalibration` runs) expressed directly against OpenCV from Rust: parse
//! the factory `.conf`, build the per-eye intrinsics and the inter-eye
//! rotation, `stereo_rectify` to rectified projections, then
//! `init_undistort_rectify_map` for the remap. Per frame it decodes the
//! side-by-side YUYV, rectifies both eyes, and runs `StereoSGBM`
//! (`MODE_SGBM_3WAY`) with the ZED default penalties, converting fixed-point
//! disparity to millimetres. Every heavy step is a maintained OpenCV call, so
//! the output matches the Stereolabs reference recipe.

use std::path::Path;

use opencv::calib3d::{self, StereoMatcherTrait, StereoSGBM, StereoSGBM_MODE_SGBM_3WAY};
use opencv::core::{CV_32FC1, CV_64FC1, Mat, MatTraitConst, Rect, Scalar, Size, ToInputArray};
use opencv::imgproc::{
    self, COLOR_BGR2GRAY, COLOR_BGR2RGB, COLOR_YUV2BGR_YUYV, INTER_AREA, INTER_LINEAR,
};
use opencv::prelude::MatTraitConstManual;

use crate::calibration::{StereoConf, resolution_key};
use crate::depth_settings::{DepthSettings, num_disparities_for};

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
    fx: f64,
    baseline_mm: f64,
    numerator: f64,
    num_disparities: u32,
}

impl CvDepth {
    pub fn create(
        calib_path: &Path,
        eye_width: u32,
        eye_height: u32,
        settings: DepthSettings,
    ) -> Result<Self, String> {
        let block = settings.block_size() as i32;
        let downscale = settings.downscale() as i32;
        let key = resolution_key(eye_width);
        let conf = StereoConf::load(calib_path, key)
            .map_err(|e| format!("calibration {}: {e}", calib_path.display()))?;

        let (w, h) = (eye_width as i32, eye_height as i32);
        let size = Size::new(w, h);
        let k_left = matrix_3x3(conf.left.fx, conf.left.cx, conf.left.fy, conf.left.cy)?;
        let k_right = matrix_3x3(conf.right.fx, conf.right.cx, conf.right.fy, conf.right.cy)?;
        let dist_left = dist_coeffs(&conf.left)?;
        let dist_right = dist_coeffs(&conf.right)?;

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

        let (mut r1, mut r2, mut p1, mut p2, mut q) = (
            Mat::default(),
            Mat::default(),
            Mat::default(),
            Mat::default(),
            Mat::default(),
        );
        let (mut roi1, mut roi2) = (Rect::default(), Rect::default());
        calib3d::stereo_rectify(
            &k_left,
            &dist_left,
            &k_right,
            &dist_right,
            size,
            &rotation,
            &translation,
            &mut r1,
            &mut r2,
            &mut p1,
            &mut p2,
            &mut q,
            calib3d::CALIB_ZERO_DISPARITY,
            0.0,
            size,
            &mut roi1,
            &mut roi2,
        )
        .map_err(cv)?;

        let (mut map_left_x, mut map_left_y) = (Mat::default(), Mat::default());
        let (mut map_right_x, mut map_right_y) = (Mat::default(), Mat::default());
        calib3d::init_undistort_rectify_map(
            &k_left,
            &dist_left,
            &r1,
            &p1,
            size,
            CV_32FC1,
            &mut map_left_x,
            &mut map_left_y,
        )
        .map_err(cv)?;
        calib3d::init_undistort_rectify_map(
            &k_right,
            &dist_right,
            &r2,
            &p2,
            size,
            CV_32FC1,
            &mut map_right_x,
            &mut map_right_y,
        )
        .map_err(cv)?;

        // Rectified focal length is P1(0,0); disparity to depth uses it, and
        // the disparity search range derives from it and the requested floor.
        let fx = *p1.at_2d::<f64>(0, 0).map_err(cv)?;
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
            fx,
            baseline_mm,
            numerator,
            num_disparities: num_disp as u32,
        })
    }

    pub fn fx(&self) -> f64 {
        self.fx
    }

    /// The derived disparity search range.
    pub fn num_disparities(&self) -> u32 {
        self.num_disparities
    }

    /// The nearest depth the derived search range can measure.
    pub fn min_depth_floor_mm(&self) -> f64 {
        self.fx / self.downscale as f64 * self.baseline_mm / self.num_disparities as f64
    }

    pub fn baseline_mm(&self) -> f64 {
        self.baseline_mm
    }

    pub fn out_size(&self) -> (u32, u32) {
        (
            (self.eye_width / self.downscale) as u32,
            (self.eye_height / self.downscale) as u32,
        )
    }

    /// Rectify + match one side-by-side YUYV frame. `left_rgb` is
    /// eye_width*eye_height*3; `depth_mm` matches `out_size()`.
    pub fn process(
        &mut self,
        yuyv: &[u8],
        left_rgb: &mut [u8],
        depth_mm: &mut [u16],
    ) -> Result<(), String> {
        let (w, h) = (self.eye_width, self.eye_height);
        assert_eq!(yuyv.len(), (2 * w * h * 2) as usize);
        assert_eq!(left_rgb.len(), (w * h * 3) as usize);

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
            let small = Size::new(w / self.downscale, h / self.downscale);
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
        let (dw, dh) = (w / self.downscale, h / self.downscale);
        let disp = disparity16.data_typed::<i16>().map_err(cv)?;
        for (out, &d) in depth_mm.iter_mut().zip(disp.iter()) {
            *out = 0;
            if d > 0 {
                let mm = self.numerator / f64::from(d);
                if (1.0..65535.0).contains(&mm) {
                    *out = mm as u16;
                }
            }
        }
        debug_assert_eq!(depth_mm.len(), (dw * dh) as usize);
        Ok(())
    }
}

fn cv(e: opencv::Error) -> String {
    format!("opencv: {e}")
}

fn matrix_3x3(fx: f64, cx: f64, fy: f64, cy: f64) -> Result<Mat, String> {
    Mat::from_slice_2d(&[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        .and_then(|m| m.try_clone())
        .map_err(cv)
        .inspect(|m| debug_assert_eq!(m.typ(), CV_64FC1))
}

fn dist_coeffs(cam: &crate::calibration::CamConf) -> Result<Mat, String> {
    Mat::from_slice(&[cam.k1, cam.k2, cam.p1, cam.p2, cam.k3])
        .map_err(cv)?
        .reshape(1, 5)
        .map_err(cv)?
        .try_clone()
        .map_err(cv)
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

fn downscaled(gray: &Mat, size: Size) -> Result<Mat, String> {
    let mut out = Mat::default();
    imgproc::resize(gray, &mut out, size, 0.0, 0.0, INTER_AREA).map_err(cv)?;
    Ok(out)
}
