use super::Encoding;
use std::time::SystemTime;

/// Frame identifier (wrapping counter)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FrameId(u32);

impl FrameId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn next(&self) -> Self {
        Self(self.0.wrapping_add(1))
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

/// A captured frame: pixel data plus the metadata published alongside it.
#[derive(Debug, Clone)]
pub struct Frame {
    data: Vec<u8>,
    width: u32,
    height: u32,
    frame_id: FrameId,
    captured_at: SystemTime,
    encoding: Encoding,
}

impl Frame {
    /// Create a frame from raw camera capture with an explicit encoding.
    ///
    /// The encoding must reflect the actual wire format produced by the camera
    /// (e.g. read back from the negotiated `CameraFormat` after `open_stream`).
    ///
    /// `captured_at` is sampled when the driver hands the frame over, not when
    /// it is published, so consumers age samples on capture time rather than on
    /// however long conversion happened to take.
    pub fn from_capture(
        data: Vec<u8>,
        width: u32,
        height: u32,
        captured_at: SystemTime,
        encoding: Encoding,
    ) -> Self {
        Self {
            data,
            width,
            height,
            frame_id: FrameId::default(),
            captured_at,
            encoding,
        }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn frame_id(&self) -> FrameId {
        self.frame_id
    }

    pub fn captured_at(&self) -> SystemTime {
        self.captured_at
    }

    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// Replace the pixel data and encoding, keeping the capture metadata.
    pub fn with_encoding(self, data: Vec<u8>, encoding: Encoding) -> Self {
        Self {
            data,
            encoding,
            ..self
        }
    }

    /// Stamp the frame with its published sequence number.
    pub fn with_frame_id(self, frame_id: FrameId) -> Self {
        Self { frame_id, ..self }
    }
}
