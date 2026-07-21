//! Timing and accuracy for the cv_depth backend.
//!
//! Usage: depth_bench <calibration.conf> [frames]
//!
//! Two legs:
//! - Timing runs the full pipeline through the REAL per-serial calibration;
//!   frame content does not change the work done.
//! - Accuracy runs through a synthetic zero-distortion calibration over scenes
//!   with known disparity (a flat plane and a sub-pixel ramp), scoring valid
//!   density and pixel error against truth. The ideal calibration keeps a
//!   horizontally shifted synthetic scene epipolar-valid, which the real
//!   distortion would not.

use std::fmt::Write as _;
use std::time::Instant;

use zed_camera::DepthSettings;
use zed_camera::cv_depth::CvDepth;

const MODES: [(&str, u32, u32); 3] = [
    ("vga", 672, 376),
    ("hd720", 1280, 720),
    ("hd1080", 1920, 1080),
];
const BLOCK_SIZE: u32 = 3;
const DOWNSCALES: [u32; 2] = [1, 2];
const FLAT_DISPARITY: f64 = 24.0;
const RAMP_MIN: f64 = 8.0;
const RAMP_MAX: f64 = 40.0;
const MARGIN: usize = 8;
const IDEAL_BASELINE: f64 = 63.0;

/// The floor that derives the reference 96-disparity search at this geometry
/// (ideal fx = eye_width), keeping runs comparable across modes.
fn min_depth_for_96_disparities(eye_width: u32, downscale: u32) -> f64 {
    (eye_width / downscale) as f64 * IDEAL_BASELINE / 95.5 / 1000.0
}

#[derive(Clone, Copy)]
enum Scene {
    Flat,
    Ramp,
}

impl Scene {
    fn name(self) -> &'static str {
        match self {
            Scene::Flat => "flat",
            Scene::Ramp => "ramp",
        }
    }

    fn disparity_at(self, col: f64, eye_width: f64) -> f64 {
        match self {
            Scene::Flat => FLAT_DISPARITY,
            Scene::Ramp => RAMP_MIN + (RAMP_MAX - RAMP_MIN) * col / eye_width,
        }
    }
}

fn texture(seed: u32, len: usize) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        })
        .collect()
}

/// Side-by-side YUYV of a textured plane whose disparity follows `scene`.
fn synthetic_yuyv(eye_width: usize, height: usize, scene: Scene) -> Vec<u8> {
    let max_d = RAMP_MAX.max(FLAT_DISPARITY).ceil() as usize + 2;
    let plane = texture(7, (eye_width + max_d) * height);
    let pitch = eye_width + max_d;
    let mut yuyv = vec![128u8; 2 * eye_width * height * 2];
    for row in 0..height {
        for col in 0..eye_width {
            yuyv[(row * 2 * eye_width + col) * 2] = plane[row * pitch + col];
            let src = col as f64 + scene.disparity_at(col as f64, eye_width as f64);
            let (s0, frac) = (src as usize, src.fract());
            let a = f64::from(plane[row * pitch + s0]);
            let b = f64::from(plane[row * pitch + s0 + 1]);
            yuyv[(row * 2 * eye_width + eye_width + col) * 2] = (a * (1.0 - frac) + b * frac) as u8;
        }
    }
    yuyv
}

/// A ZED-format calibration with identical near-ideal pinholes on both eyes and
/// a pure-x baseline: rectification is near-identity, so a horizontally shifted
/// synthetic scene stays epipolar-valid. k1/k2 are negligibly non-zero because
/// the reader treats all-zero distortion as a locale parsing failure.
fn ideal_calibration_conf() -> String {
    let mut conf = format!("[STEREO]\nBaseline={IDEAL_BASELINE}\n");
    for (section, width, height) in [("VGA", 672, 376), ("HD", 1280, 720), ("FHD", 1920, 1080)] {
        for side in ["LEFT", "RIGHT"] {
            let _ = write!(
                conf,
                "[{side}_CAM_{section}]\nfx={w}\nfy={w}\ncx={cx}\ncy={cy}\nk1=0.000001\nk2=0.000001\nk3=0\np1=0\np2=0\n",
                w = width,
                cx = width / 2,
                cy = height / 2,
            );
        }
    }
    conf
}

fn out_size(width: u32, height: u32, downscale: u32) -> (usize, usize) {
    ((width / downscale) as usize, (height / downscale) as usize)
}

fn open(calib: &str, width: u32, height: u32, downscale: u32) -> CvDepth {
    CvDepth::create(
        calib,
        width,
        height,
        DepthSettings::new(
            min_depth_for_96_disparities(width, downscale),
            BLOCK_SIZE,
            downscale,
        )
        .expect("settings"),
    )
    .expect("cv_depth")
}

fn main() {
    let mut args = std::env::args().skip(1);
    let real_calib_path = args
        .next()
        .expect("usage: depth_bench <calibration.conf> [frames]");
    let frames: u32 = args.next().map_or(30, |f| f.parse().expect("frames"));
    let real_calib =
        std::fs::read_to_string(&real_calib_path).expect("read the real calibration file");
    let ideal_calib = ideal_calibration_conf();

    println!("== timing (real calibration: {real_calib_path}) ==");
    println!("mode    downscale  ms/frame");
    for (name, width, height) in MODES {
        let yuyv = synthetic_yuyv(width as usize, height as usize, Scene::Flat);
        for downscale in DOWNSCALES {
            let (dw, dh) = out_size(width, height, downscale);
            let mut left_rgb = vec![0u8; (width * height * 3) as usize];
            let mut depth = vec![0u16; dw * dh];
            let mut backend = open(&real_calib, width, height, downscale);
            backend
                .process(&yuyv, &mut left_rgb, &mut depth)
                .expect("warm"); // warm
            let start = Instant::now();
            for _ in 0..frames {
                backend
                    .process(&yuyv, &mut left_rgb, &mut depth)
                    .expect("process");
            }
            let ms = start.elapsed().as_secs_f64() * 1e3 / f64::from(frames);
            println!("{name:7} {downscale:9}  {ms:8.2}");
        }
    }

    println!();
    println!("== accuracy vs ground truth (ideal calibration) ==");
    println!("mode    ds scene  valid%  mean|err|px  p99px  wrong%(>1px)");
    for (name, width, height) in MODES {
        for scene in [Scene::Flat, Scene::Ramp] {
            let yuyv = synthetic_yuyv(width as usize, height as usize, scene);
            for downscale in DOWNSCALES {
                let (dw, dh) = out_size(width, height, downscale);
                let mut left_rgb = vec![0u8; (width * height * 3) as usize];
                let mut depth = vec![0u16; dw * dh];
                open(&ideal_calib, width, height, downscale)
                    .process(&yuyv, &mut left_rgb, &mut depth)
                    .expect("process");
                let acc = accuracy(&depth, scene, width as usize, dw, dh, downscale as usize);
                println!(
                    "{name:7} {downscale:2} {scene:6} {valid:5}  {mean:10.3}  {p99:5.2}  {wrong:6.2}",
                    scene = scene.name(),
                    valid = acc.valid_pct,
                    mean = acc.mean_err_px,
                    p99 = acc.p99_err_px,
                    wrong = acc.wrong_pct,
                );
            }
        }
    }
}

struct Accuracy {
    valid_pct: usize,
    mean_err_px: f64,
    p99_err_px: f64,
    wrong_pct: f64,
}

/// Score one depth map against the scene's known disparity over the interior.
/// The ideal calibration is pinhole fx = eye_width, baseline = 63, so depth
/// converts back to a disparity in pixels for comparison against truth.
fn accuracy(
    depth: &[u16],
    scene: Scene,
    eye_width: usize,
    dw: usize,
    dh: usize,
    ds: usize,
) -> Accuracy {
    let numer = eye_width as f64 / ds as f64 * IDEAL_BASELINE;
    let to_px = |mm: u16| numer / f64::from(mm);
    let truth_px = |c: usize| {
        let full_col = (c * ds) as f64 + (ds as f64 - 1.0) / 2.0;
        scene.disparity_at(full_col, eye_width as f64) / ds as f64
    };
    let max_truth = RAMP_MAX.max(FLAT_DISPARITY) / ds as f64;
    let col_lo = max_truth.ceil() as usize + MARGIN;
    let interior: Vec<(usize, usize)> = (MARGIN..dh - MARGIN)
        .flat_map(|r| (col_lo..dw - MARGIN).map(move |c| (r, c)))
        .collect();

    let mut errs: Vec<f64> = interior
        .iter()
        .filter(|&&(r, c)| depth[r * dw + c] != 0)
        .map(|&(r, c)| (to_px(depth[r * dw + c]) - truth_px(c)).abs())
        .collect();
    errs.sort_by(f64::total_cmp);
    let n = errs.len().max(1);
    Accuracy {
        valid_pct: errs.len() * 100 / interior.len().max(1),
        mean_err_px: errs.iter().sum::<f64>() / n as f64,
        p99_err_px: errs
            .get(errs.len().saturating_sub(1) * 99 / 100)
            .copied()
            .unwrap_or(0.0),
        wrong_pct: errs.iter().filter(|&&e| e > 1.0).count() as f64 * 100.0 / n as f64,
    }
}
