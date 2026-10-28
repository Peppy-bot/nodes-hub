//! Intel RealSense D4xx as an rgbd_camera and camera_geometry producer over
//! librealsense2: [`pipeline`] opens the synchronized colour and depth
//! streams and runs the capture loop, [`frame`] carries a frameset from it
//! to the emit task, [`geometry`] turns the device's calibration into the
//! camera_geometry answers, [`modes`] parses the colour format, the align
//! mode and the auto/manual mode, [`stream`] names the two streams and holds
//! a value per stream, and [`node`] wires them to the generated contract
//! members.

mod frame;
mod geometry;
mod modes;
mod node;
mod pipeline;
mod stream;

pub use node::setup;
