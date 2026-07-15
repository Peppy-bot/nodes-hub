//! Small shared types. Producers are identified by their peppy `ProducerRef`
//! (core_node + instance_id); the recorder keys every source and camera by it,
//! so nothing is hard-coded to a robot.

use peppygen::ProducerRef;

/// A stable, hashable key for one bound producer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProducerKey {
    pub core_node: String,
    pub instance_id: String,
}

impl ProducerKey {
    pub fn from_ref(producer: &ProducerRef) -> Self {
        Self {
            core_node: producer.core_node.clone(),
            instance_id: producer.instance_id.clone(),
        }
    }
}

/// Camera pixel encodings the stack's camera contracts emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraEncoding {
    Rgb8,
    Bgr8,
    Yuyv,
    Mjpeg,
    /// 16-bit little-endian depth codes.
    Z16,
}

impl CameraEncoding {
    pub fn parse(wire: &str) -> Option<CameraEncoding> {
        match wire {
            "rgb8" => Some(CameraEncoding::Rgb8),
            "bgr8" => Some(CameraEncoding::Bgr8),
            "yuyv" => Some(CameraEncoding::Yuyv),
            "mjpeg" => Some(CameraEncoding::Mjpeg),
            "z16" => Some(CameraEncoding::Z16),
            _ => None,
        }
    }

    pub fn is_depth(self) -> bool {
        matches!(self, CameraEncoding::Z16)
    }
}

/// One decoded camera frame, shared zero-copy across the cache and sinks.
#[derive(Debug)]
pub struct FrameBuf {
    pub encoding: CameraEncoding,
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_parse_and_depth() {
        assert_eq!(CameraEncoding::parse("rgb8"), Some(CameraEncoding::Rgb8));
        assert_eq!(CameraEncoding::parse("z16"), Some(CameraEncoding::Z16));
        assert!(CameraEncoding::parse("z16").unwrap().is_depth());
        assert!(!CameraEncoding::parse("rgb8").unwrap().is_depth());
        assert_eq!(CameraEncoding::parse("nope"), None);
    }
}
