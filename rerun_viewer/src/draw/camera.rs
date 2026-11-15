//! The camera: its color and depth images, and its depth now as points in the robot
//! frame.

use std::time::SystemTime;

use crate::canvas::{Canvas, Rgba};
use crate::model::camera::{ColorImage, DepthImage};

pub const COLOR_PATH: &str = "camera/color";
pub const DEPTH_PATH: &str = "camera/depth";
pub const POINTS_PATH: &str = "world/camera/points";

const POINT_COLOR: Rgba = [80, 200, 255, 255];
const POINT_RADIUS_M: f32 = 0.004;

pub fn draw_color_image(canvas: &dyn Canvas, at: SystemTime, image: &ColorImage) {
    canvas.set_time(at);
    canvas.color_image(COLOR_PATH, image);
}

pub fn draw_depth_image(canvas: &dyn Canvas, at: SystemTime, image: &DepthImage) {
    canvas.set_time(at);
    canvas.depth_image(DEPTH_PATH, image);
}

/// Draws the points of a depth frame taken at `at`.
pub fn draw_cloud(canvas: &dyn Canvas, at: SystemTime, points: &[[f32; 3]]) {
    canvas.set_time(at);
    canvas.points(POINTS_PATH, points, POINT_COLOR, POINT_RADIUS_M);
}
