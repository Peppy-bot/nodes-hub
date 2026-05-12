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
fn encode_jpeg(data: &[u8], width: u32, height: u32, _quality: u8) -> Result<Vec<u8>> {
    use image::{ImageBuffer, Rgb};

    let img = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, data.to_vec())
        .ok_or_else(|| Error::EncodingError("Failed to create image buffer".to_string()))?;

    let mut jpeg_data = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut jpeg_data),
        image::ImageFormat::Jpeg,
    )
    .map_err(|e| Error::EncodingError(format!("JPEG encoding failed: {}", e)))?;

    Ok(jpeg_data)
}

/// Decode MJPEG data to raw RGB8
fn decode_jpeg(data: &[u8]) -> Result<Vec<u8>> {
    use image::ImageDecoder;
    use image::codecs::jpeg::JpegDecoder;

    let decoder = JpegDecoder::new(std::io::Cursor::new(data))
        .map_err(|e| Error::EncodingError(format!("Failed to create JPEG decoder: {}", e)))?;

    let (width, height) = decoder.dimensions();
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    decoder
        .read_image(&mut rgb)
        .map_err(|e| Error::EncodingError(format!("Failed to decode JPEG: {}", e)))?;

    Ok(rgb)
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
        Encoding::Mjpeg => decode_jpeg(frame.data())?,
    };

    // Step 2: encode RGB8 to target
    let data = match target_encoding {
        Encoding::Rgb8 => rgb_data,
        Encoding::Bgr8 => rgb_to_bgr(&rgb_data),
        Encoding::Mjpeg => encode_jpeg(&rgb_data, frame.width(), frame.height(), JPEG_QUALITY)?,
    };

    Ok(frame
        .with_encoding(data, target_encoding)
        .with_frame_id(frame_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_process_frame_rgb8() {
        let data = vec![255, 0, 0, 0, 255, 0, 0, 0, 255]; // 3 pixels
        let raw = Frame::from_capture(data.clone(), 3, 1, Instant::now(), Encoding::Rgb8);
        let frame = process_frame(raw, FrameId::default(), Encoding::Rgb8).unwrap();

        assert_eq!(frame.data(), &data);
        assert_eq!(frame.width(), 3);
        assert_eq!(frame.height(), 1);
        assert_eq!(frame.encoding(), Encoding::Rgb8);
    }

    #[test]
    fn test_process_frame_bgr8() {
        let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
        let raw = Frame::from_capture(rgb, 3, 1, Instant::now(), Encoding::Rgb8);
        let frame = process_frame(raw, FrameId::default(), Encoding::Bgr8).unwrap();

        assert_eq!(frame.data(), &[0, 0, 255, 0, 255, 0, 255, 0, 0]);
        assert_eq!(frame.encoding(), Encoding::Bgr8);
    }

    #[test]
    fn test_process_frame_mjpeg() {
        let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
        let raw = Frame::from_capture(rgb, 3, 1, Instant::now(), Encoding::Rgb8);
        let frame = process_frame(raw, FrameId::default(), Encoding::Mjpeg).unwrap();

        // Check JPEG header
        assert!(frame.data().starts_with(&[0xFF, 0xD8]));
        assert_eq!(frame.encoding(), Encoding::Mjpeg);
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
            let raw = Frame::from_capture(data.clone(), 3, 1, Instant::now(), enc);
            let frame = process_frame(raw, FrameId::default(), enc).unwrap();
            assert_eq!(frame.data(), &data, "Fast path altered data for {enc:?}");
            assert_eq!(frame.encoding(), enc);
        }
    }

    #[test]
    fn test_process_frame_frame_id_is_set() {
        let data = vec![255u8, 0, 0, 0, 255, 0, 0, 0, 255];
        let raw = Frame::from_capture(data, 3, 1, Instant::now(), Encoding::Rgb8);
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
        let raw = Frame::from_capture(bgr, 2, 1, Instant::now(), Encoding::Bgr8);
        let frame = process_frame(raw, FrameId::default(), Encoding::Rgb8).unwrap();
        assert_eq!(frame.data(), &[255u8, 128, 0, 30, 20, 10]);
        assert_eq!(frame.encoding(), Encoding::Rgb8);
    }

    #[test]
    fn test_process_frame_bgr8_to_mjpeg() {
        let bgr = vec![0u8; 4 * 3 * 3]; // 4×3 black frame in BGR
        let raw = Frame::from_capture(bgr, 4, 3, Instant::now(), Encoding::Bgr8);
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
        let raw = Frame::from_capture(jpeg, 1, 1, Instant::now(), Encoding::Mjpeg);
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
        let raw = Frame::from_capture(jpeg, 1, 1, Instant::now(), Encoding::Mjpeg);
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
    fn test_process_frame_mjpeg_fast_path() {
        // MJPEG → MJPEG: the encoded bytes must be returned unchanged.
        let jpeg = encode_jpeg(&[128u8, 64, 32], 1, 1, JPEG_QUALITY).unwrap();
        let raw = Frame::from_capture(jpeg.clone(), 1, 1, Instant::now(), Encoding::Mjpeg);
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
            Instant::now(),
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
            Instant::now(),
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
