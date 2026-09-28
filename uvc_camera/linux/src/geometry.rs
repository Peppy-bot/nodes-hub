//! camera_geometry:v1 on a UVC camera.
//!
//! A UVC device carries no calibration: the driver knows the stream's size
//! and nothing about where its pixels point. The contract is implemented so a
//! consumer of a wrist camera meets the same surface on hardware as in
//! simulation, and every member refuses and says why, so the consumer learns
//! there is nothing to use instead of reading zeros as a pinhole model. The
//! intrinsics arrive with calibration-file loading, a change of its own.

use std::sync::Arc;

use peppygen::exposed_services::geometry::{
    get_color_intrinsics, get_depth_intrinsics, get_depth_to_color_extrinsics,
};

/// The refusal of get_color_intrinsics: what is missing and what would fix it.
pub const NO_CALIBRATION_MESSAGE: &str = "no calibration: this camera needs a calibration file";

/// The refusal of the two depth services, in the words the simulated colour
/// camera uses for the same question.
pub const NO_DEPTH_STREAM_MESSAGE: &str = "a colour camera has no depth stream";

pub fn color_intrinsics() -> get_color_intrinsics::Response {
    get_color_intrinsics::Response::new(
        false,
        NO_CALIBRATION_MESSAGE.to_string(),
        0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        String::new(),
        Vec::new(),
    )
}

pub fn depth_intrinsics() -> get_depth_intrinsics::Response {
    get_depth_intrinsics::Response::new(
        false,
        NO_DEPTH_STREAM_MESSAGE.to_string(),
        0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        String::new(),
        Vec::new(),
        String::new(),
        0.0,
        0.0,
        String::new(),
    )
}

pub fn depth_to_color_extrinsics() -> get_depth_to_color_extrinsics::Response {
    get_depth_to_color_extrinsics::Response::new(
        false,
        NO_DEPTH_STREAM_MESSAGE.to_string(),
        String::new(),
        [0.0; 3],
        [0.0; 4],
    )
}

/// One camera_geometry service answering its fixed refusal.
macro_rules! listen_for_geometry_requests {
    ($fn_name:ident, $service:ident, $respond:ident) => {
        pub async fn $fn_name(node_runner: Arc<peppygen::NodeRunner>) {
            let cancel_token = node_runner.cancellation_token().clone();
            loop {
                let result = tokio::select! {
                    _ = cancel_token.cancelled() => break,
                    result = $service::handle_next_request(&node_runner, |_request| {
                        Ok($respond())
                    }) => result,
                };
                if let Err(e) = result {
                    tracing::error!("{} service error: {e:?}", stringify!($service));
                }
            }
        }
    };
}

listen_for_geometry_requests!(
    listen_for_get_color_intrinsics_requests,
    get_color_intrinsics,
    color_intrinsics
);
listen_for_geometry_requests!(
    listen_for_get_depth_intrinsics_requests,
    get_depth_intrinsics,
    depth_intrinsics
);
listen_for_geometry_requests!(
    listen_for_get_depth_to_color_extrinsics_requests,
    get_depth_to_color_extrinsics,
    depth_to_color_extrinsics
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_intrinsics_refuse_for_want_of_a_calibration() {
        let color = color_intrinsics();
        assert!(!color.success);
        assert_eq!(color.message, NO_CALIBRATION_MESSAGE);
        // Nothing a consumer could take for a pinhole model.
        assert_eq!((color.width, color.height), (0, 0));
        assert_eq!(
            (color.fx, color.fy, color.cx, color.cy),
            (0.0, 0.0, 0.0, 0.0)
        );
        assert!(color.distortion_model.is_empty() && color.distortion.is_empty());
    }

    #[test]
    fn the_depth_services_refuse_for_want_of_a_depth_stream() {
        let depth = depth_intrinsics();
        assert!(!depth.success);
        assert_eq!(depth.message, NO_DEPTH_STREAM_MESSAGE);
        assert_eq!((depth.width, depth.height, depth.fx), (0, 0, 0.0));
        assert!(depth.depth_model.is_empty() && depth.align_mode.is_empty());

        let pose = depth_to_color_extrinsics();
        assert!(!pose.success);
        assert_eq!(pose.message, NO_DEPTH_STREAM_MESSAGE);
        assert!(pose.align_mode.is_empty());
        // Not even an identity: all zeros is no rotation at all.
        assert_eq!(pose.depth_to_color_orientation, [0.0; 4]);
    }

    /// A consumer tells "this camera will never answer" from "not yet" by the
    /// message, and the simulated colour camera says the same words.
    #[test]
    fn the_depth_refusal_is_worded_as_the_simulated_colour_camera_words_it() {
        assert_eq!(
            NO_DEPTH_STREAM_MESSAGE,
            "a colour camera has no depth stream"
        );
    }
}
