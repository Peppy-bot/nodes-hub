//! The UVC camera node as a library, for the binary and the integration
//! tests: the V4L2 device and its capture loop ([`camera`]), the conversion
//! of a captured frame to the topic's encoding ([`pipeline`]), the rgb_camera
//! and camera_geometry services ([`services`], [`geometry`]), and the
//! configuration and frame types ([`types`]).

/// Answer every request of `$service` with `$respond`, which maps the request
/// to the response or the failure of an error reply, until the node stops.
/// Each request that fails is logged.
macro_rules! serve_requests {
    ($node_runner:expr, $service:ident, $respond:expr) => {{
        let cancel_token = $node_runner.cancellation_token().clone();
        loop {
            let result = tokio::select! {
                _ = cancel_token.cancelled() => break,
                result = $service::handle_next_request(&$node_runner, $respond) => result,
            };
            if let Err(e) = result {
                tracing::error!("{} service error: {e:?}", stringify!($service));
            }
        }
    }};
}

pub mod camera;
pub mod geometry;
pub mod pipeline;
pub mod services;
pub mod types;
