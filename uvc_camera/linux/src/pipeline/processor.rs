use crate::types::{Encoding, Error, Frame, FrameId, Result};

/// JPEG encoding quality (1-100)
const JPEG_QUALITY: u8 = 85;

/// Convert RGB8 to BGR8 by swapping R and B channels
fn rgb_to_bgr(data: &[u8]) -> Vec<u8> {
    data.chunks_exact(3)
        .flat_map(|rgb| [rgb[2], rgb[1], rgb[0]])
        .collect()
}

/// Encode RGB8 data as JPEG
fn encode_jpeg(data: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
    use image::codecs::jpeg::JpegEncoder;
    use image::{ImageBuffer, Rgb};

    let img = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, data.to_vec())
        .ok_or_else(|| Error::EncodingError("Failed to create image buffer".to_string()))?;

    let mut jpeg_data = Vec::new();
    let encoder = JpegEncoder::new_with_quality(std::io::Cursor::new(&mut jpeg_data), quality);
    img.write_with_encoder(encoder)
        .map_err(|e| Error::EncodingError(format!("JPEG encoding failed: {}", e)))?;

    Ok(jpeg_data)
}

/// Decode YUYV 4:2:2 (one Y per pixel, U/V shared per pixel pair) to raw RGB8
/// using limited-range BT.601 coefficients (Y 16..235, chroma 16..240).
/// UVC cameras never signal quantization, and V4L2 resolves that default to
/// limited range for Y'CbCr; OpenCV, GStreamer, and nokhwa decode it the same
/// way, so full-range math here would lift blacks and cap whites.
fn yuyv_to_rgb(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    if !width.is_multiple_of(2) {
        // YUYV packs two horizontal pixels per four bytes; an odd width would
        // pair bytes across row boundaries and flip the chroma phase per row.
        return Err(Error::EncodingError(format!(
            "yuyv requires an even frame width, got {width}"
        )));
    }
    let expected = (width as usize) * (height as usize) * 2;
    if data.len() != expected {
        return Err(Error::EncodingError(format!(
            "yuyv payload is {} bytes, expected {} for {}x{}",
            data.len(),
            expected,
            width,
            height
        )));
    }
    let mut rgb = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for pair in data.chunks_exact(4) {
        let [y0, u, y1, v] = [pair[0], pair[1], pair[2], pair[3]];
        for y in [y0, y1] {
            let y = 1.164_384 * (f32::from(y) - 16.0);
            let u = f32::from(u) - 128.0;
            let v = f32::from(v) - 128.0;
            let r = y + 1.596_027 * v;
            let g = y - 0.391_762 * u - 0.812_968 * v;
            let b = y + 2.017_232 * u;
            rgb.extend([r, g, b].map(|c| c.clamp(0.0, 255.0) as u8));
        }
    }
    Ok(rgb)
}

/// Decode MJPEG data to raw RGB8 at the frame's declared geometry.
///
/// Decoding through `DynamicImage` rather than sizing a buffer by hand: a JPEG
/// carries its own colour type, so a grayscale frame decodes to one byte per
/// pixel, and `ImageDecoder::read_image` panics outright when the buffer does
/// not match `total_bytes()`. This converts whatever the camera sent into the
/// RGB8 the rest of the pipeline expects.
///
/// The JPEG also carries its own dimensions, and the published message
/// advertises the frame's declared ones, so a mismatch (corrupt frame, driver
/// renegotiation mid-stream) must fail here rather than publish a payload that
/// contradicts its geometry.
fn decode_jpeg(data: &[u8], expected_width: u32, expected_height: u32) -> Result<Vec<u8>> {
    let decoded = image::load_from_memory_with_format(data, image::ImageFormat::Jpeg)
        .map_err(|e| Error::EncodingError(format!("Failed to decode JPEG: {}", e)))?;

    if (decoded.width(), decoded.height()) != (expected_width, expected_height) {
        return Err(Error::EncodingError(format!(
            "decoded JPEG is {}x{}, expected {expected_width}x{expected_height}",
            decoded.width(),
            decoded.height(),
        )));
    }

    Ok(decoded.into_rgb8().into_raw())
}

/// Process a raw frame from the camera into the target encoding.
///
/// The conversion is a two-step pipeline:
/// 1. Decode the camera encoding to RGB8 (intermediate representation).
/// 2. Encode RGB8 to the target encoding.
///
/// When camera encoding already matches the target the frame data is
/// passed through unchanged.
pub fn process_frame(frame: Frame, frame_id: FrameId, target_encoding: Encoding) -> Result<Frame> {
    let camera_encoding = frame.encoding();

    // Fast path: no conversion needed
    if camera_encoding == target_encoding {
        return Ok(frame.with_frame_id(frame_id));
    }

    // Step 1: decode camera format to RGB8
    let rgb_data = match camera_encoding {
        Encoding::Rgb8 => frame.data().to_vec(),
        Encoding::Bgr8 => rgb_to_bgr(frame.data()), // BGR→RGB is the same channel swap
        Encoding::Mjpeg => decode_jpeg(frame.data(), frame.width(), frame.height())?,
        Encoding::Yuyv => yuyv_to_rgb(frame.data(), frame.width(), frame.height())?,
    };

    // Step 2: encode RGB8 to target
    let data = match target_encoding {
        Encoding::Rgb8 => rgb_data,
        Encoding::Bgr8 => rgb_to_bgr(&rgb_data),
        Encoding::Mjpeg => encode_jpeg(&rgb_data, frame.width(), frame.height(), JPEG_QUALITY)?,
        Encoding::Yuyv => {
            return Err(Error::EncodingError(
                "yuyv topic_encoding is passthrough-only: it requires yuyv camera_encoding"
                    .to_string(),
            ));
        }
    };

    Ok(frame
        .with_encoding(data, target_encoding)
        .with_frame_id(frame_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn test_process_frame_rgb8() {
        let data = vec![255, 0, 0, 0, 255, 0, 0, 0, 255]; // 3 pixels
        let raw = Frame::from_capture(data.clone(), 3, 1, SystemTime::now(), Encoding::Rgb8);
        let frame = process_frame(raw, FrameId::default(), Encoding::Rgb8).unwrap();

        assert_eq!(frame.data(), &data);
        assert_eq!(frame.width(), 3);
        assert_eq!(frame.height(), 1);
        assert_eq!(frame.encoding(), Encoding::Rgb8);
    }

    #[test]
    fn test_process_frame_bgr8() {
        let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
        let raw = Frame::from_capture(rgb, 3, 1, SystemTime::now(), Encoding::Rgb8);
        let frame = process_frame(raw, FrameId::default(), Encoding::Bgr8).unwrap();

        assert_eq!(frame.data(), &[0, 0, 255, 0, 255, 0, 255, 0, 0]);
        assert_eq!(frame.encoding(), Encoding::Bgr8);
    }

    #[test]
    fn test_process_frame_mjpeg() {
        let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
        let raw = Frame::from_capture(rgb, 3, 1, SystemTime::now(), Encoding::Rgb8);
        let frame = process_frame(raw, FrameId::default(), Encoding::Mjpeg).unwrap();

        // Check JPEG header
        assert!(frame.data().starts_with(&[0xFF, 0xD8]));
        assert_eq!(frame.encoding(), Encoding::Mjpeg);
    }

    #[test]
    fn test_process_frame_yuyv_to_rgb8() {
        // Limited-range mid grey: Y=128, U=V=128 -> 1.164384*(128-16) = 130.4.
        let raw = Frame::from_capture(
            vec![128, 128, 128, 128],
            2,
            1,
            SystemTime::now(),
            Encoding::Yuyv,
        );
        let frame = process_frame(raw, FrameId::default(), Encoding::Rgb8).unwrap();
        assert_eq!(frame.encoding(), Encoding::Rgb8);
        assert_eq!(frame.data(), &[130, 130, 130, 130, 130, 130]);
    }

    #[test]
    fn test_yuyv_range_endpoints() {
        // Limited-range black (Y=16) is RGB 0 and reference white (Y=235) is
        // RGB 255; full-range math would leave them at 16 and 235.
        let black = yuyv_to_rgb(&[16, 128, 16, 128], 2, 1).unwrap();
        assert_eq!(black, vec![0, 0, 0, 0, 0, 0]);
        let white = yuyv_to_rgb(&[235, 128, 235, 128], 2, 1).unwrap();
        assert_eq!(white, vec![255, 255, 255, 255, 255, 255]);
        // Footroom and headroom codes clamp instead of wrapping.
        let sub_black = yuyv_to_rgb(&[0, 128, 0, 128], 2, 1).unwrap();
        assert_eq!(sub_black, vec![0, 0, 0, 0, 0, 0]);
        let super_white = yuyv_to_rgb(&[255, 128, 255, 128], 2, 1).unwrap();
        assert_eq!(super_white, vec![255, 255, 255, 255, 255, 255]);
    }

    #[test]
    fn test_yuyv_rejects_wrong_payload_size() {
        assert!(yuyv_to_rgb(&[0u8; 5], 2, 1).is_err());
    }

    #[test]
    fn test_yuyv_rejects_odd_width() {
        // Odd widths would pair bytes across row boundaries (3x2), and an odd
        // pixel count (3x1) would silently truncate the output buffer.
        assert!(yuyv_to_rgb(&[0u8; 12], 3, 2).is_err());
        assert!(yuyv_to_rgb(&[0u8; 6], 3, 1).is_err());
    }

    #[test]
    fn test_process_frame_yuyv_passthrough() {
        let data = vec![10u8, 20, 30, 40];
        let raw = Frame::from_capture(data.clone(), 2, 1, SystemTime::now(), Encoding::Yuyv);
        let frame = process_frame(raw, FrameId::default(), Encoding::Yuyv).unwrap();
        assert_eq!(frame.data(), &data);
        assert_eq!(frame.encoding(), Encoding::Yuyv);
    }

    #[test]
    fn test_yuyv_target_requires_passthrough() {
        let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
        let raw = Frame::from_capture(rgb, 3, 1, SystemTime::now(), Encoding::Rgb8);
        assert!(process_frame(raw, FrameId::default(), Encoding::Yuyv).is_err());
    }

    #[test]
    fn test_rgb_to_bgr() {
        let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
        let bgr = rgb_to_bgr(&rgb);
        assert_eq!(bgr, vec![0, 0, 255, 0, 255, 0, 255, 0, 0]);

        // Verify the operation is reversible (BGR to RGB is the same)
        let rgb_again = rgb_to_bgr(&bgr);
        assert_eq!(rgb_again, rgb);
    }

    // ── fast path ─────────────────────────────────────────────────────────────

    #[test]
    fn test_process_frame_fast_path_preserves_data() {
        // When camera encoding == topic encoding, data must be bit-for-bit identical.
        let data = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90];
        for enc in [Encoding::Rgb8, Encoding::Bgr8] {
            let raw = Frame::from_capture(data.clone(), 3, 1, SystemTime::now(), enc);
            let frame = process_frame(raw, FrameId::default(), enc).unwrap();
            assert_eq!(frame.data(), &data, "Fast path altered data for {enc:?}");
            assert_eq!(frame.encoding(), enc);
        }
    }

    #[test]
    fn test_process_frame_frame_id_is_set() {
        let data = vec![255u8, 0, 0, 0, 255, 0, 0, 0, 255];
        let raw = Frame::from_capture(data, 3, 1, SystemTime::now(), Encoding::Rgb8);
        let frame_id = FrameId::new(42);
        let frame = process_frame(raw, frame_id, Encoding::Rgb8).unwrap();
        assert_eq!(frame.frame_id(), frame_id);
    }

    // ── BGR8 camera source ────────────────────────────────────────────────────

    #[test]
    fn test_process_frame_bgr8_to_rgb8() {
        // Two pixels: BGR (0,128,255) and (10,20,30)
        // After BGR→RGB swap they become (255,128,0) and (30,20,10)
        let bgr = vec![0u8, 128, 255, 10, 20, 30];
        let raw = Frame::from_capture(bgr, 2, 1, SystemTime::now(), Encoding::Bgr8);
        let frame = process_frame(raw, FrameId::default(), Encoding::Rgb8).unwrap();
        assert_eq!(frame.data(), &[255u8, 128, 0, 30, 20, 10]);
        assert_eq!(frame.encoding(), Encoding::Rgb8);
    }

    #[test]
    fn test_process_frame_bgr8_to_mjpeg() {
        let bgr = vec![0u8; 4 * 3 * 3]; // 4×3 black frame in BGR
        let raw = Frame::from_capture(bgr, 4, 3, SystemTime::now(), Encoding::Bgr8);
        let frame = process_frame(raw, FrameId::default(), Encoding::Mjpeg).unwrap();
        assert!(
            frame.data().starts_with(&[0xFF, 0xD8]),
            "Expected JPEG header"
        );
        assert_eq!(frame.encoding(), Encoding::Mjpeg);
    }

    // ── MJPEG camera source ───────────────────────────────────────────────────

    #[test]
    fn test_process_frame_mjpeg_to_rgb8() {
        // Encode a 1×1 red pixel as JPEG then decode via process_frame.
        // JPEG is lossy so we only check that the red channel dominates.
        let jpeg = encode_jpeg(&[255u8, 0, 0], 1, 1, JPEG_QUALITY).unwrap();
        let raw = Frame::from_capture(jpeg, 1, 1, SystemTime::now(), Encoding::Mjpeg);
        let frame = process_frame(raw, FrameId::default(), Encoding::Rgb8).unwrap();
        assert_eq!(frame.encoding(), Encoding::Rgb8);
        assert_eq!(frame.data().len(), 3);
        assert!(frame.data()[0] > 200, "R channel should be high");
        assert!(frame.data()[1] < 50, "G channel should be low");
        assert!(frame.data()[2] < 50, "B channel should be low");
    }

    #[test]
    fn test_process_frame_mjpeg_to_bgr8() {
        // Encode a 1×1 pure-blue pixel (RGB: 0,0,255) as JPEG then decode to BGR.
        // In BGR output the blue value moves to index 0.
        let jpeg = encode_jpeg(&[0u8, 0, 255], 1, 1, JPEG_QUALITY).unwrap();
        let raw = Frame::from_capture(jpeg, 1, 1, SystemTime::now(), Encoding::Mjpeg);
        let frame = process_frame(raw, FrameId::default(), Encoding::Bgr8).unwrap();
        assert_eq!(frame.encoding(), Encoding::Bgr8);
        assert_eq!(frame.data().len(), 3);
        assert!(
            frame.data()[0] > 200,
            "B channel (index 0 in BGR) should be high"
        );
        assert!(frame.data()[1] < 50, "G channel should be low");
        assert!(
            frame.data()[2] < 50,
            "R channel (index 2 in BGR) should be low"
        );
    }

    #[test]
    fn test_process_frame_grayscale_mjpeg_to_rgb8() {
        // A grayscale JPEG decodes to one byte per pixel, so sizing the output
        // buffer as width*height*3 made `read_image` panic on the capture
        // thread. Expanding to RGB8 must work for any source colour type.
        let gray = image::GrayImage::from_raw(2, 2, vec![0, 64, 128, 255]).unwrap();
        let mut jpeg = Vec::new();
        image::DynamicImage::ImageLuma8(gray)
            .write_to(
                &mut std::io::Cursor::new(&mut jpeg),
                image::ImageFormat::Jpeg,
            )
            .unwrap();

        let raw = Frame::from_capture(jpeg, 2, 2, SystemTime::now(), Encoding::Mjpeg);
        let frame = process_frame(raw, FrameId::default(), Encoding::Rgb8).unwrap();

        assert_eq!(frame.encoding(), Encoding::Rgb8);
        assert_eq!(
            frame.data().len(),
            2 * 2 * 3,
            "grayscale must expand to RGB8"
        );
        // Grey expands to equal channels; JPEG is lossy so allow a little drift.
        for px in frame.data().chunks_exact(3) {
            assert!(px[0].abs_diff(px[1]) <= 2 && px[1].abs_diff(px[2]) <= 2);
        }
    }

    #[test]
    fn test_mjpeg_dimension_mismatch_is_rejected() {
        // The published message advertises the frame's declared geometry, so a
        // JPEG whose embedded dimensions disagree (corrupt frame, driver
        // renegotiation mid-stream) must fail instead of publishing a payload
        // that contradicts its width and height.
        let jpeg = encode_jpeg(&[255u8, 0, 0], 1, 1, JPEG_QUALITY).unwrap();
        let raw = Frame::from_capture(jpeg, 2, 2, SystemTime::now(), Encoding::Mjpeg);
        let err = process_frame(raw, FrameId::default(), Encoding::Rgb8).unwrap_err();
        assert!(
            err.to_string()
                .contains("decoded JPEG is 1x1, expected 2x2"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_process_frame_mjpeg_fast_path() {
        // MJPEG → MJPEG: the encoded bytes must be returned unchanged.
        let jpeg = encode_jpeg(&[128u8, 64, 32], 1, 1, JPEG_QUALITY).unwrap();
        let raw = Frame::from_capture(jpeg.clone(), 1, 1, SystemTime::now(), Encoding::Mjpeg);
        let frame = process_frame(raw, FrameId::default(), Encoding::Mjpeg).unwrap();
        assert_eq!(frame.data(), &jpeg);
        assert_eq!(frame.encoding(), Encoding::Mjpeg);
    }

    #[test]
    fn test_process_frame_rgb8_mjpeg_rgb8_roundtrip() {
        // RGB→MJPEG→RGB: JPEG is lossy, so we check each channel is within a
        // reasonable tolerance (±10) rather than requiring bit-exact equality.
        let original: Vec<u8> = vec![200, 100, 50, 10, 230, 180, 128, 128, 128];
        let width = 3u32;
        let height = 1u32;
        let tolerance = 10u8;

        // Step 1: RGB → MJPEG
        let raw = Frame::from_capture(
            original.clone(),
            width,
            height,
            SystemTime::now(),
            Encoding::Rgb8,
        );
        let mjpeg_frame = process_frame(raw, FrameId::new(1), Encoding::Mjpeg).unwrap();
        assert_eq!(mjpeg_frame.encoding(), Encoding::Mjpeg);
        assert!(mjpeg_frame.data().starts_with(&[0xFF, 0xD8]));

        // Step 2: MJPEG → RGB
        let mjpeg_raw = Frame::from_capture(
            mjpeg_frame.data().to_vec(),
            width,
            height,
            SystemTime::now(),
            Encoding::Mjpeg,
        );
        let rgb_frame = process_frame(mjpeg_raw, FrameId::new(2), Encoding::Rgb8).unwrap();
        assert_eq!(rgb_frame.encoding(), Encoding::Rgb8);
        assert_eq!(rgb_frame.data().len(), original.len());

        for (i, (&orig, &recovered)) in original.iter().zip(rgb_frame.data()).enumerate() {
            let diff = orig.abs_diff(recovered);
            assert!(
                diff <= tolerance,
                "Channel {i}: original={orig}, recovered={recovered}, diff={diff} exceeds tolerance={tolerance}"
            );
        }
    }
}
