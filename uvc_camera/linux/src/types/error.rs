use std::fmt;

/// Unified error type for the UVC camera node
#[derive(Debug)]
pub enum Error {
    /// Camera hardware errors
    Camera(String),

    /// Image encoding failed
    EncodingError(String),

    /// Device path parsing failed
    InvalidDevicePath(String),

    /// Thread panicked
    ThreadPanic(String),

    /// Generic error with context
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Camera(msg) => write!(f, "Camera error: {msg}"),
            Self::EncodingError(msg) => write!(f, "Encoding error: {msg}"),
            Self::InvalidDevicePath(path) => write!(f, "Invalid device path: {path}"),
            Self::ThreadPanic(msg) => write!(f, "Thread panicked: {msg}"),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// Convenience Result type using our Error
pub type Result<T> = std::result::Result<T, Error>;

/// Convert from anyhow::Error
impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Self::Other(err.to_string())
    }
}

/// Convert from string errors
impl From<String> for Error {
    fn from(s: String) -> Self {
        Self::Other(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Self::Other(s.to_string())
    }
}
