use super::Encoding;
use std::time::Instant;

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

impl From<u32> for FrameId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<FrameId> for u32 {
    fn from(id: FrameId) -> Self {
        id.0
    }
}

/// Frame metadata
#[derive(Debug, Clone)]
pub struct FrameInfo {
    width: u32,
    height: u32,
    frame_id: FrameId,
    timestamp: Instant,
    encoding: Encoding,
}

impl FrameInfo {
    pub fn new(
        width: u32,
        height: u32,
        frame_id: FrameId,
        timestamp: Instant,
        encoding: Encoding,
    ) -> Self {
        Self {
            width,
            height,
            frame_id,
            timestamp,
            encoding,
        }
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

    pub fn timestamp(&self) -> Instant {
        self.timestamp
    }

    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// Create a new FrameInfo with updated encoding
    pub fn with_encoding(&self, encoding: Encoding) -> Self {
        Self { encoding, ..*self }
    }

    /// Update the frame ID
    pub fn with_frame_id(&self, frame_id: FrameId) -> Self {
        Self { frame_id, ..*self }
    }
}

/// Frame with data and metadata
#[derive(Debug, Clone)]
pub struct Frame {
    data: Vec<u8>,
    info: FrameInfo,
}

impl Frame {
    pub fn new(data: Vec<u8>, info: FrameInfo) -> Self {
        Self { data, info }
    }

    /// Create a frame from raw camera capture with an explicit encoding.
    ///
    /// The encoding must reflect the actual wire format produced by the camera
    /// (e.g. read back from the negotiated `CameraFormat` after `open_stream`).
    pub fn from_capture(
        data: Vec<u8>,
        width: u32,
        height: u32,
        timestamp: Instant,
        encoding: Encoding,
    ) -> Self {
        Self {
            data,
            info: FrameInfo::new(width, height, FrameId::default(), timestamp, encoding),
        }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn info(&self) -> &FrameInfo {
        &self.info
    }

    pub fn width(&self) -> u32 {
        self.info.width()
    }

    pub fn height(&self) -> u32 {
        self.info.height()
    }

    pub fn frame_id(&self) -> FrameId {
        self.info.frame_id()
    }

    pub fn timestamp(&self) -> Instant {
        self.info.timestamp()
    }

    pub fn encoding(&self) -> Encoding {
        self.info.encoding()
    }

    /// Convert this frame to a different encoding with new data
    pub fn with_encoding(self, data: Vec<u8>, encoding: Encoding) -> Self {
        Self {
            data,
            info: self.info.with_encoding(encoding),
        }
    }

    /// Update frame ID
    pub fn with_frame_id(self, frame_id: FrameId) -> Self {
        Self {
            data: self.data,
            info: self.info.with_frame_id(frame_id),
        }
    }
}
