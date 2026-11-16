//! The camera: its images as the viewer draws them, and its depth samples as points in
//! the robot frame.

use std::num::NonZeroU32;

use camera_core::encoding::{ColorEncoding, DEPTH_ENCODING, DepthFrame};
use camera_core::pinhole::{DepthModel, Pinhole};
use camera_core::pose::Pose;
use control_core::positive_finite::PositiveFinite;

/// A color image as the viewer draws it: raw RGB pixels, or a JPEG as it came.
#[derive(Debug, Clone, PartialEq)]
pub enum ColorImage {
    /// 3 bytes a pixel, R, G, B, row by row from the top left.
    Rgb {
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    },
    Jpeg(Vec<u8>),
}

/// A color frame the viewer cannot read.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ColorImageError {
    #[error("encoding {0:?} is none of rgb8, bgr8, yuyv and mjpeg")]
    Encoding(String),
    #[error("a {width}x{height} {encoding} frame is {expected} bytes, this one is {got}")]
    Size {
        encoding: ColorEncoding,
        width: u32,
        height: u32,
        expected: usize,
        got: usize,
    },
    #[error("a yuyv frame is an even number of pixels wide, this one is {0}")]
    OddWidth(u32),
}

impl ColorImage {
    /// The image of a frame of `width` by `height` pixels in the encoding `encoding`
    /// names. A bgr8 or yuyv frame becomes RGB; an mjpeg frame stays a JPEG.
    pub fn parse(
        encoding: &str,
        width: u32,
        height: u32,
        frame: Vec<u8>,
    ) -> Result<Self, ColorImageError> {
        let encoding = ColorEncoding::from_name(encoding)
            .ok_or_else(|| ColorImageError::Encoding(encoding.to_string()))?;
        let rgb = |pixels| Self::Rgb {
            width,
            height,
            pixels,
        };
        match encoding {
            ColorEncoding::Mjpeg => Ok(Self::Jpeg(frame)),
            ColorEncoding::Rgb8 => {
                check_size(encoding, width, height, 3, frame.len())?;
                Ok(rgb(frame))
            }
            ColorEncoding::Bgr8 => {
                check_size(encoding, width, height, 3, frame.len())?;
                Ok(rgb(bgr_to_rgb(&frame)))
            }
            ColorEncoding::Yuyv => {
                // A yuyv pair of pixels shares its chroma.
                if !width.is_multiple_of(2) {
                    return Err(ColorImageError::OddWidth(width));
                }
                check_size(encoding, width, height, 2, frame.len())?;
                Ok(rgb(yuyv_to_rgb(&frame)))
            }
        }
    }

    /// The image at most `max_width` pixels wide. Raw pixels wider than that are shrunk
    /// by the smallest whole factor that fits: each pixel is the mean of the block of
    /// factor by factor pixels it stands for, and the columns and the rows past the last
    /// whole block are left out. A JPEG, an image that fits and an image lower than the
    /// factor stay as they are.
    pub fn shrunk_to(self, max_width: NonZeroU32) -> Self {
        let (width, height, pixels) = match self {
            Self::Rgb {
                width,
                height,
                pixels,
            } if width > max_width.get() && height >= width.div_ceil(max_width.get()) => {
                (width, height, pixels)
            }
            as_it_is => return as_it_is,
        };
        let factor = width.div_ceil(max_width.get());
        let (small_width, small_height) = (width / factor, height / factor);
        let block = factor as usize;
        let row_bytes = 3 * width as usize;
        let count = factor * factor;
        let mut small = Vec::with_capacity(3 * small_width as usize * small_height as usize);
        for row in 0..small_height as usize {
            for column in 0..small_width as usize {
                let mut sums = [0_u32; 3];
                for line in 0..block {
                    let start = (row * block + line) * row_bytes + 3 * column * block;
                    for pixel in pixels[start..start + 3 * block].as_chunks::<3>().0 {
                        for (sum, channel) in sums.iter_mut().zip(pixel) {
                            *sum += u32::from(*channel);
                        }
                    }
                }
                // The mean, rounded to the nearest.
                small.extend(sums.map(|sum| ((sum + count / 2) / count) as u8));
            }
        }
        Self::Rgb {
            width: small_width,
            height: small_height,
            pixels: small,
        }
    }
}

/// Whether a frame of `got` bytes is `width` by `height` pixels of `bytes_per_pixel`.
fn check_size(
    encoding: ColorEncoding,
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
    got: usize,
) -> Result<(), ColorImageError> {
    let expected = width as usize * height as usize * bytes_per_pixel;
    if got != expected {
        return Err(ColorImageError::Size {
            encoding,
            width,
            height,
            expected,
            got,
        });
    }
    Ok(())
}

fn bgr_to_rgb(frame: &[u8]) -> Vec<u8> {
    let (pixels, _) = frame.as_chunks::<3>();
    pixels.iter().flat_map(|&[b, g, r]| [r, g, b]).collect()
}

/// RGB from YCbCr 4:2:2 (Y0, U, Y1, V for each pair of pixels), by the BT.601
/// full-range equations.
fn yuyv_to_rgb(frame: &[u8]) -> Vec<u8> {
    let rgb = |y: u8, u: u8, v: u8| {
        let (y, u, v) = (f32::from(y), f32::from(u) - 128.0, f32::from(v) - 128.0);
        let byte = |value: f32| value.round().clamp(0.0, 255.0) as u8;
        [
            byte(y + 1.402 * v),
            byte(y - 0.344_136 * u - 0.714_136 * v),
            byte(y + 1.772 * u),
        ]
    };
    let (pairs, _) = frame.as_chunks::<4>();
    pairs
        .iter()
        .flat_map(|&[y0, u, y1, v]| {
            let [r0, g0, b0] = rgb(y0, u, v);
            let [r1, g1, b1] = rgb(y1, u, v);
            [r0, g0, b0, r1, g1, b1]
        })
        .collect()
}

/// A depth frame the viewer cannot read.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DepthImageError {
    #[error("encoding {0:?}, the viewer reads {DEPTH_ENCODING}")]
    Encoding(String),
    #[error("a {width}x{height} {DEPTH_ENCODING} frame is {expected} bytes, this one is {got}")]
    Size {
        width: u32,
        height: u32,
        expected: usize,
        got: usize,
    },
}

/// A z16 depth frame whose length is its size's, and the meters one count is.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthImage {
    width: u32,
    height: u32,
    /// One little-endian u16 sample a pixel, row by row from the top left.
    samples: Vec<u8>,
    depth_unit: PositiveFinite,
}

impl DepthImage {
    /// The depth image of a frame of `width` by `height` pixels in `encoding`,
    /// `depth_unit` meters a count.
    pub fn parse(
        encoding: &str,
        width: u32,
        height: u32,
        samples: Vec<u8>,
        depth_unit: PositiveFinite,
    ) -> Result<Self, DepthImageError> {
        if encoding != DEPTH_ENCODING {
            return Err(DepthImageError::Encoding(encoding.to_string()));
        }
        let expected = width as usize * height as usize * 2;
        if samples.len() != expected {
            return Err(DepthImageError::Size {
                width,
                height,
                expected,
                got: samples.len(),
            });
        }
        Ok(Self {
            width,
            height,
            samples,
            depth_unit,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn samples(&self) -> &[u8] {
        &self.samples
    }

    pub fn depth_unit(&self) -> PositiveFinite {
        self.depth_unit
    }

    fn frame(&self) -> DepthFrame<'_> {
        DepthFrame::parse(&self.samples, self.width, self.height)
            .expect("parse checked the length against the size")
    }
}

/// The depth stream as the viewer reads it: its lens, how a sample reads as a depth,
/// and where its optical frame sits in the color optical frame.
#[derive(Debug)]
pub struct DepthCamera {
    pub lens: Pinhole,
    pub depth_model: DepthModel,
    pub depth_in_color: Pose,
    /// The alignment the geometry describes, which the frames must carry.
    pub align_mode: String,
}

impl DepthCamera {
    /// Where the depth optical frame stands in the robot frame, the color optical frame
    /// standing at `color_in_robot`: a point goes through the depth-to-color pose first.
    pub fn depth_in_robot(&self, color_in_robot: &Pose) -> Pose {
        color_in_robot.compose(&self.depth_in_color)
    }

    /// Whether `image` is of the lens's size.
    pub fn fits(&self, image: &DepthImage) -> bool {
        image.width == self.lens.width() && image.height == self.lens.height()
    }
}

/// A depth image of another size than its camera's lens.
#[derive(Debug, thiserror::Error, PartialEq)]
#[error("the depth frame is {image_width}x{image_height}, the lens {lens_width}x{lens_height}")]
pub struct LensSizeError {
    pub image_width: u32,
    pub image_height: u32,
    pub lens_width: u32,
    pub lens_height: u32,
}

/// The points that every `stride`-th sample across and down of `image` reads, starting
/// at the first pixel, in the robot frame: the depth optical frame stands at
/// `depth_in_robot`. A sample of 0 reads nothing and makes no point.
pub fn strided_points(
    image: &DepthImage,
    camera: &DepthCamera,
    depth_in_robot: &Pose,
    stride: NonZeroU32,
) -> Result<Vec<[f32; 3]>, LensSizeError> {
    if !camera.fits(image) {
        return Err(LensSizeError {
            image_width: image.width,
            image_height: image.height,
            lens_width: camera.lens.width(),
            lens_height: camera.lens.height(),
        });
    }
    let frame = image.frame();
    let step = stride.get() as usize;
    let mut points = Vec::new();
    for v in (0..image.height).step_by(step) {
        for u in (0..image.width).step_by(step) {
            let sample = frame.sample(u, v).expect("u and v lie inside the frame");
            if sample == 0 {
                continue;
            }
            let reading_m = f64::from(sample) * image.depth_unit.get();
            let in_depth_frame =
                camera
                    .lens
                    .deproject(f64::from(u), f64::from(v), reading_m, camera.depth_model);
            points.push(depth_in_robot.apply(in_depth_frame).map(|c| c as f32));
        }
    }
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::placement::tests::{close, quarter_turn_about_z};
    use crate::model::placement::translation;
    use camera_core::pinhole::Intrinsics;

    /// A 4 by 3 depth camera with a focal length of 2 pixels and its principal point at
    /// (1.5, 1), its depth frame on its color frame.
    fn camera() -> DepthCamera {
        DepthCamera {
            lens: Pinhole::parse(&Intrinsics {
                width: 4,
                height: 3,
                fx: 2.0,
                fy: 2.0,
                cx: 1.5,
                cy: 1.0,
                distortion_model: "none",
                distortion: &[],
            })
            .unwrap(),
            depth_model: DepthModel::Z,
            depth_in_color: translation([0.0; 3]),
            align_mode: "depth_to_color".to_string(),
        }
    }

    /// A 4 by 3 image of `samples`, half a meter a count.
    fn image(samples: [u16; 12]) -> DepthImage {
        let bytes = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        DepthImage::parse("z16", 4, 3, bytes, PositiveFinite::parse(0.5).unwrap()).unwrap()
    }

    fn points_close(got: &[[f32; 3]], expected: &[[f64; 3]]) -> bool {
        got.len() == expected.len()
            && got
                .iter()
                .zip(expected)
                .all(|(g, e)| close(g.map(f64::from), *e, 1e-6))
    }

    #[test]
    fn a_stride_of_one_reads_every_sample_but_the_empty_ones() {
        // Two counts read 1 m. Pixel (u, v) reads ((u - 1.5) / 2, (v - 1) / 2, 1).
        let mut samples = [2_u16; 12];
        samples[1] = 0; // pixel (1, 0)
        let points = strided_points(
            &image(samples),
            &camera(),
            &translation([0.0; 3]),
            NonZeroU32::new(1).unwrap(),
        )
        .unwrap();
        assert_eq!(points.len(), 11);
        assert!(close(points[0].map(f64::from), [-0.75, -0.5, 1.0], 1e-6));
        assert!(close(points[1].map(f64::from), [0.25, -0.5, 1.0], 1e-6));
    }

    #[test]
    fn a_stride_of_two_reads_every_other_column_and_row() {
        // Pixels (0, 0), (2, 0), (0, 2), (2, 2); (2, 2) reads 4 counts, 2 m.
        let mut samples = [2_u16; 12];
        samples[2 * 4 + 2] = 4;
        let points = strided_points(
            &image(samples),
            &camera(),
            &translation([0.0; 3]),
            NonZeroU32::new(2).unwrap(),
        )
        .unwrap();
        assert!(points_close(
            &points,
            &[
                [-0.75, -0.5, 1.0],
                [0.25, -0.5, 1.0],
                [-0.75, 0.5, 1.0],
                [0.5, 1.0, 2.0],
            ]
        ));
    }

    #[test]
    fn the_points_go_through_the_pose_of_the_depth_frame() {
        // A quarter turn about z takes (x, y, z) to (-y, x, z), then the origin adds.
        let samples = [2_u16; 12];
        let depth_in_robot = Pose::new([1.0, 2.0, 3.0], quarter_turn_about_z()).unwrap();
        let points = strided_points(
            &image(samples),
            &camera(),
            &depth_in_robot,
            NonZeroU32::new(4).unwrap(),
        )
        .unwrap();
        // Only pixel (0, 0), which reads (-0.75, -0.5, 1) in the depth frame.
        assert!(points_close(&points, &[[1.5, 1.25, 4.0]]));
    }

    #[test]
    fn an_image_of_another_size_than_the_lens_is_refused() {
        let unit = PositiveFinite::parse(0.5).unwrap();
        let small = DepthImage::parse("z16", 2, 1, vec![0; 4], unit).unwrap();
        assert!(!camera().fits(&small));
        assert_eq!(
            strided_points(
                &small,
                &camera(),
                &translation([0.0; 3]),
                NonZeroU32::new(1).unwrap()
            ),
            Err(LensSizeError {
                image_width: 2,
                image_height: 1,
                lens_width: 4,
                lens_height: 3
            })
        );
    }

    #[test]
    fn a_depth_frame_of_another_encoding_or_length_is_refused() {
        let unit = PositiveFinite::parse(0.001).unwrap();
        assert_eq!(
            DepthImage::parse("rgb8", 1, 1, vec![0; 2], unit),
            Err(DepthImageError::Encoding("rgb8".to_string()))
        );
        assert_eq!(
            DepthImage::parse("z16", 2, 2, vec![0; 7], unit),
            Err(DepthImageError::Size {
                width: 2,
                height: 2,
                expected: 8,
                got: 7
            })
        );
    }

    #[test]
    fn the_depth_frame_goes_through_its_color_frame_first() {
        // The color frame stands at (1, 0, 0) turned a quarter about z; the depth frame
        // stands 0.1 m along the color frame's x, which the turn points along the
        // robot's y. The other order would put the depth frame at (1.1, 0, 0).
        let mut camera = camera();
        camera.depth_in_color = translation([0.1, 0.0, 0.0]);
        let color_in_robot = Pose::new([1.0, 0.0, 0.0], quarter_turn_about_z()).unwrap();
        let depth_in_robot = camera.depth_in_robot(&color_in_robot);
        assert!(close(depth_in_robot.origin(), [1.0, 0.1, 0.0], 1e-12));
    }

    #[test]
    fn an_rgb8_frame_stays_as_it_is_and_a_bgr8_frame_swaps_red_and_blue() {
        let rgb = ColorImage::parse("rgb8", 1, 1, vec![10, 20, 30]).unwrap();
        assert_eq!(
            rgb,
            ColorImage::Rgb {
                width: 1,
                height: 1,
                pixels: vec![10, 20, 30]
            }
        );
        let bgr = ColorImage::parse("bgr8", 1, 1, vec![10, 20, 30]).unwrap();
        assert_eq!(
            bgr,
            ColorImage::Rgb {
                width: 1,
                height: 1,
                pixels: vec![30, 20, 10]
            }
        );
    }

    #[test]
    fn a_yuyv_pair_becomes_two_rgb_pixels() {
        // No chroma (U = V = 128) is gray: each pixel's R, G and B equal its Y.
        let gray = ColorImage::parse("yuyv", 2, 1, vec![50, 128, 200, 128]).unwrap();
        assert_eq!(
            gray,
            ColorImage::Rgb {
                width: 2,
                height: 1,
                pixels: vec![50, 50, 50, 200, 200, 200]
            }
        );
        // V = 228 adds 1.402 * 100 to red and takes 0.714136 * 100 from green.
        let red = ColorImage::parse("yuyv", 2, 1, vec![100, 128, 100, 228]).unwrap();
        let ColorImage::Rgb { pixels, .. } = red else {
            panic!("an RGB image")
        };
        assert_eq!(&pixels[..3], &[240, 29, 100]);
    }

    #[test]
    fn an_mjpeg_frame_stays_a_jpeg() {
        let jpeg = ColorImage::parse("mjpeg", 640, 480, vec![0xFF, 0xD8]).unwrap();
        assert_eq!(jpeg, ColorImage::Jpeg(vec![0xFF, 0xD8]));
    }

    #[test]
    fn a_wide_image_shrinks_to_the_mean_of_each_block() {
        // 4 by 2 pixels, to at most 2 wide: a factor of 2, one row of two pixels.
        #[rustfmt::skip]
        let pixels = vec![
            10, 0, 0,   20, 0, 0,   100, 100, 100,   100, 100, 100,
            30, 0, 0,   40, 0, 2,   100, 100, 100,   104, 100, 100,
        ];
        let wide = ColorImage::Rgb {
            width: 4,
            height: 2,
            pixels,
        };
        assert_eq!(
            wide.shrunk_to(NonZeroU32::new(2).unwrap()),
            ColorImage::Rgb {
                width: 2,
                height: 1,
                // Left: red (10 + 20 + 30 + 40) / 4 = 25, blue 2 / 4 rounds to 1.
                pixels: vec![25, 0, 1, 101, 100, 100],
            }
        );
    }

    #[test]
    fn the_pixels_past_the_last_whole_block_are_left_out() {
        // 5 by 3 pixels, each row 9, 9, 9, 200, 200, to at most 2 wide: a factor of 3,
        // one pixel, the mean of the first 3 by 3 block.
        let row = [9, 9, 9, 9, 9, 9, 9, 9, 9, 200, 200, 200, 200, 200, 200];
        let wide = ColorImage::Rgb {
            width: 5,
            height: 3,
            pixels: row.repeat(3),
        };
        assert_eq!(
            wide.shrunk_to(NonZeroU32::new(2).unwrap()),
            ColorImage::Rgb {
                width: 1,
                height: 1,
                pixels: vec![9, 9, 9],
            }
        );
    }

    #[test]
    fn an_image_that_fits_a_jpeg_and_an_image_lower_than_the_factor_stay_as_they_are() {
        let two = NonZeroU32::new(2).unwrap();
        let fits = ColorImage::Rgb {
            width: 2,
            height: 1,
            pixels: vec![1, 2, 3, 4, 5, 6],
        };
        assert_eq!(fits.clone().shrunk_to(two), fits);
        let jpeg = ColorImage::Jpeg(vec![0xFF, 0xD8]);
        assert_eq!(jpeg.clone().shrunk_to(two), jpeg);
        // 4 by 1 pixels: a factor of 2 leaves no row.
        let low = ColorImage::Rgb {
            width: 4,
            height: 1,
            pixels: vec![7; 12],
        };
        assert_eq!(low.clone().shrunk_to(two), low);
    }

    #[test]
    fn a_frame_of_the_wrong_size_or_encoding_is_refused() {
        assert!(matches!(
            ColorImage::parse("rgb8", 2, 2, vec![0; 11]),
            Err(ColorImageError::Size {
                expected: 12,
                got: 11,
                ..
            })
        ));
        assert_eq!(
            ColorImage::parse("yuyv", 3, 1, vec![0; 6]),
            Err(ColorImageError::OddWidth(3))
        );
        assert_eq!(
            ColorImage::parse("nv12", 2, 2, vec![]),
            Err(ColorImageError::Encoding("nv12".to_string()))
        );
    }
}
