//! From what librealsense reports to what camera_geometry:v1 answers.
//!
//! Nothing here touches the SDK: `pipeline` copies the device's calibration
//! into the plain types below once, at open, and these functions turn it into
//! the contract's answers for the streams as the current align mode
//! publishes them. librealsense and the contract share their conventions
//! (the centre of the first pixel is (0, 0); +X right, +Y down, +Z forward;
//! an extrinsic takes a point of the first stream into the second), so the
//! SDK's numbers cross unchanged.
//!
//! - Unaligned, each stream has its own pinhole model and the depth-to-colour
//!   transform is the device's.
//! - Aligned, librealsense warps one stream into the other's viewpoint and
//!   resolution, so both streams share the target's model and the transform
//!   is the identity.
//!
//! Whatever the contract cannot carry is refused with the reason, never
//! answered with a guess: a distortion model the contract has no name for, a
//! focal length that is not a positive number, a rotation that is not one.

use crate::modes::AlignMode;

/// Below this a distortion coefficient moves no pixel by a thousandth: the
/// image is an ideal pinhole, whatever model the device names.
const NEGLIGIBLE_COEFFICIENT: f32 = 1e-6;

/// How far a reported rotation may be from orthonormal, per element. The SDK
/// reports single precision.
const ROTATION_TOLERANCE: f64 = 1e-4;

/// The largest sample a z16 depth frame carries.
const MAX_Z16_SAMPLE: f64 = 65535.0;

/// The distortion model librealsense names for a stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Distortion {
    None,
    /// OpenCV's model: the coefficients take an undistorted point to the
    /// distorted one.
    BrownConrady,
    /// A variation of it whose tangential terms use the radially scaled point.
    ModifiedBrownConrady,
    /// The coefficients take a distorted point to the undistorted one.
    InverseBrownConrady,
    FTheta,
    KannalaBrandt4,
    /// A value this node has no name for, as the SDK reported it.
    Unknown(i32),
}

/// One stream's intrinsics as librealsense reports them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamIntrinsics {
    pub width: u32,
    pub height: u32,
    pub fx: f32,
    pub fy: f32,
    pub ppx: f32,
    pub ppy: f32,
    pub distortion: Distortion,
    /// [k1, k2, p1, p2, k3] for the Brown-Conrady family.
    pub coeffs: [f32; 5],
}

/// The extrinsics between two streams as librealsense reports them: a
/// column-major 3x3 rotation and a translation in metres, taking a point of
/// the first stream into the second.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamExtrinsics {
    pub rotation: [f32; 9],
    pub translation: [f32; 3],
}

/// The device's calibration for the two streams the node opened, read once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Calibration {
    pub color: StreamIntrinsics,
    pub depth: StreamIntrinsics,
    pub depth_to_color: StreamExtrinsics,
    /// Metres per z16 sample.
    pub depth_unit: f32,
}

/// One stream's pinhole model as camera_geometry:v1 answers it.
#[derive(Debug, Clone, PartialEq)]
pub struct Pinhole {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub distortion_model: &'static str,
    pub distortion: Vec<f64>,
}

/// The depth optical frame in the colour optical frame: a position in metres
/// and a unit quaternion [x, y, z, w].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub position: [f64; 3],
    pub orientation: [f64; 4],
}

impl Pose {
    pub const IDENTITY: Self = Self {
        position: [0.0, 0.0, 0.0],
        orientation: [0.0, 0.0, 0.0, 1.0],
    };
}

/// What the node answers while `align_mode` is the mode its frames carry.
/// Each part stands or is refused on its own: a colour distortion the
/// contract cannot carry does not take the depth answer with it when the
/// streams are unaligned.
#[derive(Debug, Clone, PartialEq)]
pub struct Published {
    pub align_mode: AlignMode,
    pub color: Result<Pinhole, String>,
    pub depth: Result<Pinhole, String>,
    pub depth_to_color: Result<Pose, String>,
    /// The depths a sample can carry, metres; refused with the depth unit.
    pub depth_range_m: Result<(f64, f64), String>,
}

impl Calibration {
    /// The geometry of the streams as `align_mode` publishes them.
    pub fn published(&self, align_mode: AlignMode) -> Published {
        let (color, depth, depth_to_color) = match align_mode {
            AlignMode::None => (
                pinhole("colour", &self.color),
                pinhole("depth", &self.depth),
                pose(&self.depth_to_color),
            ),
            // Depth is warped into the colour stream's viewpoint, size and
            // distortion: the depth frames are colour-shaped.
            AlignMode::DepthToColor => {
                let target = pinhole("colour", &self.color);
                (target.clone(), target, Ok(Pose::IDENTITY))
            }
            // The inverse: the colour frames are depth-shaped.
            AlignMode::ColorToDepth => {
                let target = pinhole("depth", &self.depth);
                (target.clone(), target, Ok(Pose::IDENTITY))
            }
        };
        Published {
            align_mode,
            color,
            depth,
            depth_to_color,
            depth_range_m: depth_range_m(self.depth_unit),
        }
    }
}

/// The depths a z16 sample can carry at the device's depth unit. The node
/// applies no threshold of its own, so this is the whole range: one sample
/// step to the last sample.
fn depth_range_m(depth_unit: f32) -> Result<(f64, f64), String> {
    if !(depth_unit.is_finite() && depth_unit > 0.0) {
        return Err(format!(
            "the device's depth unit is not a positive number: {depth_unit}"
        ));
    }
    let unit = f64::from(depth_unit);
    Ok((unit, unit * MAX_Z16_SAMPLE))
}

/// One stream's model in the contract's terms. `stream` names it in a refusal.
fn pinhole(stream: &str, sdk: &StreamIntrinsics) -> Result<Pinhole, String> {
    let StreamIntrinsics {
        width,
        height,
        fx,
        fy,
        ppx,
        ppy,
        ..
    } = *sdk;
    if width == 0 || height == 0 {
        return Err(format!("the {stream} stream has no size: {width}x{height}"));
    }
    if !(fx.is_finite() && fx > 0.0 && fy.is_finite() && fy > 0.0) {
        return Err(format!(
            "the {stream} focal lengths are not positive numbers: fx {fx}, fy {fy}"
        ));
    }
    if !(ppx.is_finite() && ppy.is_finite()) {
        return Err(format!(
            "the {stream} principal point is not a number: ppx {ppx}, ppy {ppy}"
        ));
    }
    let (distortion_model, distortion) = distortion(stream, sdk)?;
    Ok(Pinhole {
        width,
        height,
        fx: f64::from(fx),
        fy: f64::from(fy),
        cx: f64::from(ppx),
        cy: f64::from(ppy),
        distortion_model,
        distortion,
    })
}

/// The contract names three models: "none", OpenCV's "plumb_bob" and its
/// "inverse_plumb_bob".
/// - Coefficients that are all negligible leave an ideal pinhole under every
///   Brown-Conrady variant, which is what D4xx depth and most D4xx colour
///   streams report.
/// - librealsense's plain Brown-Conrady is OpenCV's model, coefficient for
///   coefficient, so it crosses as "plumb_bob".
/// - librealsense's Inverse Brown-Conrady is the same polynomial run from
///   the distorted point to the undistorted one, which is what D4xx colour
///   streams name, so it crosses as "inverse_plumb_bob" with the same five
///   coefficients.
/// - The modified variant and the fisheye models are other functions of
///   the same five numbers. Handing them over under either name would have a
///   consumer undistort with the wrong formula, so they refuse.
fn distortion(stream: &str, sdk: &StreamIntrinsics) -> Result<(&'static str, Vec<f64>), String> {
    let coeffs = sdk.coeffs;
    if coeffs.iter().any(|c| !c.is_finite()) {
        return Err(format!(
            "the {stream} distortion coefficients are not numbers: {coeffs:?}"
        ));
    }
    let negligible = coeffs.iter().all(|c| c.abs() < NEGLIGIBLE_COEFFICIENT);
    match sdk.distortion {
        Distortion::None => Ok(("none", Vec::new())),
        Distortion::BrownConrady
        | Distortion::ModifiedBrownConrady
        | Distortion::InverseBrownConrady
            if negligible =>
        {
            Ok(("none", Vec::new()))
        }
        Distortion::BrownConrady => {
            Ok(("plumb_bob", coeffs.iter().map(|c| f64::from(*c)).collect()))
        }
        Distortion::InverseBrownConrady => Ok((
            "inverse_plumb_bob",
            coeffs.iter().map(|c| f64::from(*c)).collect(),
        )),
        other => Err(format!(
            "the {stream} stream is distorted under {other:?} with coefficients {coeffs:?}, \
             which camera_geometry cannot express"
        )),
    }
}

/// The SDK's extrinsics as a position and a unit quaternion. The rotation is
/// column-major: element (row, column) is `rotation[row + 3 * column]`.
fn pose(sdk: &StreamExtrinsics) -> Result<Pose, String> {
    let r = |row: usize, column: usize| f64::from(sdk.rotation[row + 3 * column]);
    if sdk
        .rotation
        .iter()
        .chain(&sdk.translation)
        .any(|v| !v.is_finite())
    {
        return Err(format!(
            "the device's depth-to-colour extrinsics are not numbers: {sdk:?}"
        ));
    }
    // A rotation's columns are unit length and square to each other, and it
    // keeps handedness. Anything else turned into a quaternion would place
    // every depth sample wrongly without saying so.
    for a in 0..3 {
        for b in 0..3 {
            let dot: f64 = (0..3).map(|row| r(row, a) * r(row, b)).sum();
            let expected = if a == b { 1.0 } else { 0.0 };
            if (dot - expected).abs() > ROTATION_TOLERANCE {
                return Err(format!(
                    "the device's depth-to-colour rotation is not a rotation: {:?}",
                    sdk.rotation
                ));
            }
        }
    }
    let determinant = r(0, 0) * (r(1, 1) * r(2, 2) - r(1, 2) * r(2, 1))
        - r(0, 1) * (r(1, 0) * r(2, 2) - r(1, 2) * r(2, 0))
        + r(0, 2) * (r(1, 0) * r(2, 1) - r(1, 1) * r(2, 0));
    if determinant < 0.0 {
        return Err(format!(
            "the device's depth-to-colour rotation mirrors: {:?}",
            sdk.rotation
        ));
    }

    // The branch with the largest divisor, so no case divides by nearly zero.
    let trace = r(0, 0) + r(1, 1) + r(2, 2);
    let (x, y, z, w) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (
            (r(2, 1) - r(1, 2)) / s,
            (r(0, 2) - r(2, 0)) / s,
            (r(1, 0) - r(0, 1)) / s,
            s / 4.0,
        )
    } else if r(0, 0) >= r(1, 1) && r(0, 0) >= r(2, 2) {
        let s = (1.0 + r(0, 0) - r(1, 1) - r(2, 2)).sqrt() * 2.0;
        (
            s / 4.0,
            (r(0, 1) + r(1, 0)) / s,
            (r(0, 2) + r(2, 0)) / s,
            (r(2, 1) - r(1, 2)) / s,
        )
    } else if r(1, 1) >= r(2, 2) {
        let s = (1.0 + r(1, 1) - r(0, 0) - r(2, 2)).sqrt() * 2.0;
        (
            (r(0, 1) + r(1, 0)) / s,
            s / 4.0,
            (r(1, 2) + r(2, 1)) / s,
            (r(0, 2) - r(2, 0)) / s,
        )
    } else {
        let s = (1.0 + r(2, 2) - r(0, 0) - r(1, 1)).sqrt() * 2.0;
        (
            (r(0, 2) + r(2, 0)) / s,
            (r(1, 2) + r(2, 1)) / s,
            s / 4.0,
            (r(1, 0) - r(0, 1)) / s,
        )
    };
    let length = (x * x + y * y + z * z + w * w).sqrt();
    Ok(Pose {
        position: sdk.translation.map(f64::from),
        orientation: [x / length, y / length, z / length, w / length],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A D435 as it reports itself: 1280x720 colour under the inverse model
    /// with zero coefficients, 848x480 rectified depth, the depth imager
    /// 15 mm to the side of the colour one and turned by a fraction of a
    /// degree.
    fn d435() -> Calibration {
        Calibration {
            color: StreamIntrinsics {
                width: 1280,
                height: 720,
                fx: 915.25,
                fy: 914.75,
                ppx: 641.5,
                ppy: 362.25,
                distortion: Distortion::InverseBrownConrady,
                coeffs: [0.0; 5],
            },
            depth: StreamIntrinsics {
                width: 848,
                height: 480,
                fx: 423.5,
                fy: 423.5,
                ppx: 421.75,
                ppy: 238.125,
                distortion: Distortion::BrownConrady,
                coeffs: [0.0; 5],
            },
            depth_to_color: StreamExtrinsics {
                rotation: column_major(rotation_about_z(0.4_f64.to_radians())),
                translation: [0.015, 0.0001, -0.0002],
            },
            depth_unit: 0.001,
        }
    }

    fn rotation_about_z(angle: f64) -> [[f64; 3]; 3] {
        let (s, c) = angle.sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    /// Rows and columns as written, stored the way librealsense stores them.
    fn column_major(m: [[f64; 3]; 3]) -> [f32; 9] {
        let mut out = [0.0_f32; 9];
        for row in 0..3 {
            for column in 0..3 {
                out[row + 3 * column] = m[row][column] as f32;
            }
        }
        out
    }

    /// q * p * q^-1 for a unit quaternion [x, y, z, w].
    fn rotate(q: [f64; 4], p: [f64; 3]) -> [f64; 3] {
        let [x, y, z, w] = q;
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let v = [x, y, z];
        let t = cross(v, p).map(|c| 2.0 * c);
        let u = cross(v, t);
        [
            p[0] + w * t[0] + u[0],
            p[1] + w * t[1] + u[1],
            p[2] + w * t[2] + u[2],
        ]
    }

    #[test]
    fn unaligned_each_stream_keeps_its_own_model_and_the_sdk_values_cross_unchanged() {
        let p = d435().published(AlignMode::None);
        let color = p.color.unwrap();
        assert_eq!((color.width, color.height), (1280, 720));
        assert_eq!(
            (color.fx, color.fy, color.cx, color.cy),
            (915.25, 914.75, 641.5, 362.25)
        );
        let depth = p.depth.unwrap();
        assert_eq!((depth.width, depth.height), (848, 480));
        assert_eq!(
            (depth.fx, depth.fy, depth.cx, depth.cy),
            (423.5, 423.5, 421.75, 238.125)
        );
        // The device's own transform, translation as reported.
        let pose = p.depth_to_color.unwrap();
        assert_eq!(pose.position, [0.015_f32, 0.0001, -0.0002].map(f64::from));
        assert_ne!(pose.orientation, Pose::IDENTITY.orientation);
    }

    #[test]
    fn depth_aligned_to_colour_is_colour_shaped_and_the_transform_is_the_identity() {
        let p = d435().published(AlignMode::DepthToColor);
        assert_eq!(p.depth, p.color);
        assert_eq!(p.depth.unwrap().width, 1280);
        assert_eq!(p.depth_to_color.unwrap(), Pose::IDENTITY);
    }

    #[test]
    fn colour_aligned_to_depth_is_depth_shaped_and_the_transform_is_the_identity() {
        let p = d435().published(AlignMode::ColorToDepth);
        assert_eq!(p.color, p.depth);
        let color = p.color.unwrap();
        assert_eq!((color.width, color.fx), (848, 423.5));
        assert_eq!(p.depth_to_color.unwrap(), Pose::IDENTITY);
    }

    #[test]
    fn the_answer_names_the_mode_it_describes() {
        for mode in [
            AlignMode::None,
            AlignMode::DepthToColor,
            AlignMode::ColorToDepth,
        ] {
            assert_eq!(d435().published(mode).align_mode, mode);
        }
    }

    #[test]
    fn negligible_coefficients_are_an_ideal_pinhole_under_every_brown_conrady_variant() {
        for model in [
            Distortion::None,
            Distortion::BrownConrady,
            Distortion::ModifiedBrownConrady,
            Distortion::InverseBrownConrady,
        ] {
            let mut calibration = d435();
            calibration.color.distortion = model;
            calibration.color.coeffs = [1e-9, 0.0, -1e-8, 0.0, 0.0];
            let color = calibration.published(AlignMode::None).color.unwrap();
            assert_eq!(color.distortion_model, "none", "{model:?}");
            assert!(color.distortion.is_empty());
        }
    }

    #[test]
    fn plain_brown_conrady_crosses_as_plumb_bob_coefficient_for_coefficient() {
        let mut calibration = d435();
        calibration.color.distortion = Distortion::BrownConrady;
        calibration.color.coeffs = [-0.055, 0.066, -0.0007, 0.0005, -0.021];
        let color = calibration.published(AlignMode::None).color.unwrap();
        assert_eq!(color.distortion_model, "plumb_bob");
        assert_eq!(
            color.distortion,
            [-0.055_f32, 0.066, -0.0007, 0.0005, -0.021].map(f64::from)
        );
    }

    #[test]
    fn inverse_brown_conrady_crosses_as_inverse_plumb_bob_coefficient_for_coefficient() {
        // A D455 colour stream as it reports itself: the inverse model with
        // real coefficients. The numbers cross unchanged; only the name says
        // which way the polynomial runs.
        let mut calibration = d435();
        calibration.color.distortion = Distortion::InverseBrownConrady;
        calibration.color.coeffs = [-0.055, 0.066, -0.0007, 0.0005, -0.021];
        let p = calibration.published(AlignMode::None);
        let color = p.color.unwrap();
        assert_eq!(color.distortion_model, "inverse_plumb_bob");
        assert_eq!(
            color.distortion,
            [-0.055_f32, 0.066, -0.0007, 0.0005, -0.021].map(f64::from)
        );
        // Aligned to colour, the depth frames take the colour lens with them.
        let depth = calibration
            .published(AlignMode::DepthToColor)
            .depth
            .unwrap();
        assert_eq!(depth.distortion_model, "inverse_plumb_bob");
        assert_eq!(depth.distortion, color.distortion);
    }

    #[test]
    fn a_distortion_the_contract_cannot_express_refuses_and_says_which() {
        for model in [
            Distortion::ModifiedBrownConrady,
            Distortion::FTheta,
            Distortion::KannalaBrandt4,
            Distortion::Unknown(42),
        ] {
            let mut calibration = d435();
            calibration.color.distortion = model;
            calibration.color.coeffs = [-0.055, 0.066, -0.0007, 0.0005, -0.021];
            let p = calibration.published(AlignMode::None);
            let refused = p.color.unwrap_err();
            assert!(
                refused.contains("colour") && refused.contains("cannot express"),
                "{refused}"
            );
            // Unaligned, the depth stream is its own camera and still answers.
            assert!(p.depth.is_ok());
            // Aligned to colour, the depth frames take the colour distortion
            // with them, so the depth answer refuses as well.
            assert!(
                calibration
                    .published(AlignMode::DepthToColor)
                    .depth
                    .is_err()
            );
            // Aligned to depth, the colour frames are depth-shaped and answer.
            assert!(calibration.published(AlignMode::ColorToDepth).color.is_ok());
        }
    }

    #[test]
    fn values_a_consumer_cannot_divide_by_are_refused_with_the_reason() {
        for bad in [0.0, -915.0, f32::NAN, f32::INFINITY] {
            let mut calibration = d435();
            calibration.color.fx = bad;
            let refused = calibration.published(AlignMode::None).color.unwrap_err();
            assert!(refused.contains("focal lengths"), "{refused}");
        }
        let mut calibration = d435();
        calibration.depth.ppy = f32::NAN;
        assert!(
            calibration
                .published(AlignMode::None)
                .depth
                .unwrap_err()
                .contains("principal point")
        );
        let mut calibration = d435();
        calibration.depth.width = 0;
        assert!(
            calibration
                .published(AlignMode::None)
                .depth
                .unwrap_err()
                .contains("no size")
        );
        let mut calibration = d435();
        calibration.color.coeffs[1] = f32::NAN;
        assert!(calibration.published(AlignMode::None).color.is_err());
    }

    #[test]
    fn the_depth_range_is_what_a_z16_sample_can_carry_at_the_device_unit() {
        let (min, max) = d435().published(AlignMode::None).depth_range_m.unwrap();
        assert_eq!(min, f64::from(0.001_f32));
        assert_eq!(max, f64::from(0.001_f32) * 65535.0);
        for bad in [0.0, -0.001, f32::NAN, f32::INFINITY] {
            let mut calibration = d435();
            calibration.depth_unit = bad;
            assert!(
                calibration
                    .published(AlignMode::None)
                    .depth_range_m
                    .is_err()
            );
        }
    }

    #[test]
    fn the_quaternion_turns_a_point_the_way_the_sdk_rotation_does() {
        // A quarter turn about +Z takes +X to +Y. A row-major reading of the
        // same nine numbers would take it to -Y, so this pins the layout.
        let quarter = StreamExtrinsics {
            rotation: column_major(rotation_about_z(90_f64.to_radians())),
            translation: [0.0; 3],
        };
        let q = pose(&quarter).unwrap().orientation;
        let turned = rotate(q, [1.0, 0.0, 0.0]);
        assert!(
            (turned[0]).abs() < 1e-6 && (turned[1] - 1.0).abs() < 1e-6,
            "{turned:?}"
        );
        let half = std::f64::consts::FRAC_PI_4;
        assert!(
            (q[2] - half.sin()).abs() < 1e-6 && (q[3] - half.cos()).abs() < 1e-6,
            "{q:?}"
        );
    }

    #[test]
    fn the_quaternion_matches_the_matrix_for_turns_about_every_axis() {
        let (a, b, c) = (
            20_f64.to_radians(),
            -35_f64.to_radians(),
            170_f64.to_radians(),
        );
        let rx = [
            [1.0, 0.0, 0.0],
            [0.0, a.cos(), -a.sin()],
            [0.0, a.sin(), a.cos()],
        ];
        let ry = [
            [b.cos(), 0.0, b.sin()],
            [0.0, 1.0, 0.0],
            [-b.sin(), 0.0, b.cos()],
        ];
        let rz = rotation_about_z(c);
        let mul = |m: [[f64; 3]; 3], n: [[f64; 3]; 3]| {
            let mut out = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    out[i][j] = (0..3).map(|k| m[i][k] * n[k][j]).sum();
                }
            }
            out
        };
        // Includes a turn past 120 degrees, where the trace goes negative and
        // the other branches of the conversion are the ones that run.
        for m in [rx, ry, rz, mul(rz, mul(ry, rx)), mul(rx, rz)] {
            let q = pose(&StreamExtrinsics {
                rotation: column_major(m),
                translation: [0.0; 3],
            })
            .unwrap()
            .orientation;
            let length: f64 = q.iter().map(|c| c * c).sum::<f64>().sqrt();
            assert!((length - 1.0).abs() < 1e-9);
            for p in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, -0.2, 1.5]] {
                let by_matrix: [f64; 3] =
                    std::array::from_fn(|i| (0..3).map(|k| m[i][k] * p[k]).sum());
                let by_quaternion = rotate(q, p);
                for i in 0..3 {
                    assert!(
                        (by_matrix[i] - by_quaternion[i]).abs() < 1e-5,
                        "{m:?} {p:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn an_identity_rotation_is_the_identity_quaternion() {
        let identity = StreamExtrinsics {
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation: [0.0; 3],
        };
        assert_eq!(pose(&identity).unwrap(), Pose::IDENTITY);
    }

    #[test]
    fn nine_numbers_that_are_not_a_rotation_are_refused() {
        let zeros = StreamExtrinsics {
            rotation: [0.0; 9],
            translation: [0.0; 3],
        };
        assert!(pose(&zeros).unwrap_err().contains("not a rotation"));
        let scaled = StreamExtrinsics {
            rotation: [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
            translation: [0.0; 3],
        };
        assert!(pose(&scaled).unwrap_err().contains("not a rotation"));
        let mirrored = StreamExtrinsics {
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            translation: [0.0; 3],
        };
        assert!(pose(&mirrored).unwrap_err().contains("mirrors"));
        let mut broken = d435().depth_to_color;
        broken.translation[0] = f32::NAN;
        assert!(pose(&broken).unwrap_err().contains("not numbers"));
    }
}
