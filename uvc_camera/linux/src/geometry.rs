//! camera_geometry:v1 on a UVC camera.
//!
//! A UVC device carries no calibration: the driver knows the stream's size
//! and nothing about where its pixels point. The contract is implemented so a
//! consumer of a wrist camera meets the same surface on hardware as in
//! simulation, and every member gives an error reply with `rejected` that
//! says why, so the consumer learns there is nothing to use.

use std::sync::Arc;

use peppygen::exposed_services::geometry::{
    get_color_intrinsics, get_depth_intrinsics, get_depth_to_color_extrinsics,
};

/// The refusal of get_color_intrinsics: what is missing and what would fix it.
pub const NO_CALIBRATION_MESSAGE: &str = "no calibration: this camera needs a calibration file";

/// The refusal of the two depth services, in the words the simulated colour
/// camera uses for the same question.
pub const NO_DEPTH_STREAM_MESSAGE: &str = "a colour camera has no depth stream";

pub fn color_intrinsics() -> Result<get_color_intrinsics::Response, get_color_intrinsics::Failure> {
    Err(get_color_intrinsics::Failure::new(
        get_color_intrinsics::FailureKind::Rejected,
        NO_CALIBRATION_MESSAGE,
    ))
}

pub fn depth_intrinsics() -> Result<get_depth_intrinsics::Response, get_depth_intrinsics::Failure> {
    Err(get_depth_intrinsics::Failure::new(
        get_depth_intrinsics::FailureKind::Rejected,
        NO_DEPTH_STREAM_MESSAGE,
    ))
}

pub fn depth_to_color_extrinsics()
-> Result<get_depth_to_color_extrinsics::Response, get_depth_to_color_extrinsics::Failure> {
    Err(get_depth_to_color_extrinsics::Failure::new(
        get_depth_to_color_extrinsics::FailureKind::Rejected,
        NO_DEPTH_STREAM_MESSAGE,
    ))
}

/// One camera_geometry service answering its fixed error reply.
macro_rules! listen_for_geometry_requests {
    ($fn_name:ident, $service:ident, $respond:ident) => {
        pub async fn $fn_name(node_runner: Arc<peppygen::NodeRunner>) {
            let cancel_token = node_runner.cancellation_token().clone();
            loop {
                let result = tokio::select! {
                    _ = cancel_token.cancelled() => break,
                    result = $service::handle_next_request(&node_runner, |_request| {
                        $respond()
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
        let refusal = color_intrinsics().unwrap_err();
        assert_eq!(refusal.kind, get_color_intrinsics::FailureKind::Rejected);
        assert_eq!(refusal.message, NO_CALIBRATION_MESSAGE);
    }

    #[test]
    fn the_depth_services_refuse_for_want_of_a_depth_stream() {
        let refusal = depth_intrinsics().unwrap_err();
        assert_eq!(refusal.kind, get_depth_intrinsics::FailureKind::Rejected);
        assert_eq!(refusal.message, NO_DEPTH_STREAM_MESSAGE);

        let refusal = depth_to_color_extrinsics().unwrap_err();
        assert_eq!(
            refusal.kind,
            get_depth_to_color_extrinsics::FailureKind::Rejected
        );
        assert_eq!(refusal.message, NO_DEPTH_STREAM_MESSAGE);
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
