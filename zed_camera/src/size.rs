//! The size of an image.

/// The size of an image, in pixels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}
