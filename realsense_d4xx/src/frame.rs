//! Owned frame payloads handed off from the capture loop to the emit task.
//!
//! Bytes are owned (not borrowed from a librealsense2 frame) so the emit task
//! can outlive any single `wait()` call without holding the pipeline lock.

use std::time::SystemTime;

use crate::modes::AlignMode;

pub struct FrameSet {
    pub frame_id: u32,
    pub stamp: SystemTime,
    /// Alignment the payload was captured under, so each emitted frame is
    /// self-describing even as `set_align_mode` is toggled at runtime.
    pub align_mode: AlignMode,
    /// Color in the configured pixel format (rgb8/bgr8/yuyv/mjpeg).
    pub color: Image,
    /// Z16 little-endian depth; multiply samples by `depth_stream_info.depth_unit`.
    pub depth: Image,
}

pub struct Image {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
}
