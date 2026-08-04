use peppygen::exposed_services::camera::{
    set_brightness, set_contrast, set_exposure, set_gain, set_white_balance, video_stream_info,
};
use std::sync::Arc;

use crate::camera::controls::{
    CameraControlRequest, ControlResult, ExposureMode, WhiteBalanceMode,
};
use crate::camera::v4l_device::{ControlHandle, StreamDescription};
use crate::types::Encoding;

// ─────────────────────────────────────────────────────────────────────────────
// Existing: video_stream_info
// ─────────────────────────────────────────────────────────────────────────────

/// Listen for and handle video stream info service requests.
///
/// Answers with the driver-negotiated stream, not the requested one, so a
/// consumer sizing buffers or validating geometry sees what the wire actually
/// carries.
pub async fn listen_for_video_stream_info_requests(
    node_runner: Arc<peppygen::NodeRunner>,
    stream: StreamDescription,
    topic_encoding: Encoding,
) {
    let cancel_token = node_runner.cancellation_token().clone();
    loop {
        let result = tokio::select! {
            _ = cancel_token.cancelled() => break,
            result = video_stream_info::handle_next_request(&node_runner, |_request| {
                Ok(video_stream_info::Response::new(
                    stream.width,
                    stream.height,
                    stream.frames_per_second,
                    topic_encoding.to_string(),
                ))
            }) => result,
        };
        if let Err(e) = result {
            tracing::error!("video_stream_info service error: {e:?}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a control synchronously and report what actually happened.
///
/// The ioctl is fast but blocking (a USB round trip), so `block_in_place`
/// keeps it legal inside a synchronous handler closure on the runtime. There
/// is no queue and no timeout: by the time the caller has its response, the
/// hardware state is settled.
fn send_control(controls: &ControlHandle, request: CameraControlRequest) -> ControlResult {
    tokio::task::block_in_place(|| controls.apply(&request))
}

// ─────────────────────────────────────────────────────────────────────────────
// set_exposure
// ─────────────────────────────────────────────────────────────────────────────

/// Listen for and handle `set_exposure` service requests
pub async fn listen_for_set_exposure_requests(
    node_runner: Arc<peppygen::NodeRunner>,
    controls: ControlHandle,
) {
    let cancel_token = node_runner.cancellation_token().clone();
    loop {
        let result = tokio::select! {
            _ = cancel_token.cancelled() => break,
            result = set_exposure::handle_next_request(&node_runner, |request| {
                let mode = match ExposureMode::try_from(request.data.mode.as_str()) {
                    Ok(m) => m,
                    Err(err) => {
                        return Ok(set_exposure::Response::new(false, err, -1));
                    }
                };

                let result = send_control(
                    &controls,
                    CameraControlRequest::SetExposure {
                        mode,
                        value: request.data.value,
                    },
                );

                Ok(set_exposure::Response::new(
                    result.success,
                    result.message,
                    result.current_value,
                ))
            }) => result,
        };
        if let Err(e) = result {
            tracing::error!("set_exposure service error: {e:?}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// set_white_balance
// ─────────────────────────────────────────────────────────────────────────────

/// Listen for and handle `set_white_balance` service requests
pub async fn listen_for_set_white_balance_requests(
    node_runner: Arc<peppygen::NodeRunner>,
    controls: ControlHandle,
) {
    let cancel_token = node_runner.cancellation_token().clone();
    loop {
        let result = tokio::select! {
            _ = cancel_token.cancelled() => break,
            result = set_white_balance::handle_next_request(&node_runner, |request| {
                let mode = match WhiteBalanceMode::try_from(request.data.mode.as_str()) {
                    Ok(m) => m,
                    Err(err) => {
                        return Ok(set_white_balance::Response::new(false, err, -1));
                    }
                };

                let result = send_control(
                    &controls,
                    CameraControlRequest::SetWhiteBalance {
                        mode,
                        temperature: request.data.temperature,
                    },
                );

                Ok(set_white_balance::Response::new(
                    result.success,
                    result.message,
                    result.current_value,
                ))
            }) => result,
        };
        if let Err(e) = result {
            tracing::error!("set_white_balance service error: {e:?}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// set_gain
// ─────────────────────────────────────────────────────────────────────────────

/// Listen for and handle `set_gain` service requests
pub async fn listen_for_set_gain_requests(
    node_runner: Arc<peppygen::NodeRunner>,
    controls: ControlHandle,
) {
    let cancel_token = node_runner.cancellation_token().clone();
    loop {
        let result = tokio::select! {
            _ = cancel_token.cancelled() => break,
            result = set_gain::handle_next_request(&node_runner, |request| {
                let result = send_control(
                    &controls,
                    CameraControlRequest::SetGain {
                        value: request.data.value,
                    },
                );

                Ok(set_gain::Response::new(
                    result.success,
                    result.message,
                    result.current_value,
                ))
            }) => result,
        };
        if let Err(e) = result {
            tracing::error!("set_gain service error: {e:?}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// set_brightness
// ─────────────────────────────────────────────────────────────────────────────

/// Listen for and handle `set_brightness` service requests
pub async fn listen_for_set_brightness_requests(
    node_runner: Arc<peppygen::NodeRunner>,
    controls: ControlHandle,
) {
    let cancel_token = node_runner.cancellation_token().clone();
    loop {
        let result = tokio::select! {
            _ = cancel_token.cancelled() => break,
            result = set_brightness::handle_next_request(&node_runner, |request| {
                let result = send_control(
                    &controls,
                    CameraControlRequest::SetBrightness {
                        value: request.data.value,
                    },
                );

                Ok(set_brightness::Response::new(
                    result.success,
                    result.message,
                    result.current_value,
                ))
            }) => result,
        };
        if let Err(e) = result {
            tracing::error!("set_brightness service error: {e:?}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// set_contrast
// ─────────────────────────────────────────────────────────────────────────────

/// Listen for and handle `set_contrast` service requests
pub async fn listen_for_set_contrast_requests(
    node_runner: Arc<peppygen::NodeRunner>,
    controls: ControlHandle,
) {
    let cancel_token = node_runner.cancellation_token().clone();
    loop {
        let result = tokio::select! {
            _ = cancel_token.cancelled() => break,
            result = set_contrast::handle_next_request(&node_runner, |request| {
                let result = send_control(
                    &controls,
                    CameraControlRequest::SetContrast {
                        value: request.data.value,
                    },
                );

                Ok(set_contrast::Response::new(
                    result.success,
                    result.message,
                    result.current_value,
                ))
            }) => result,
        };
        if let Err(e) = result {
            tracing::error!("set_contrast service error: {e:?}");
        }
    }
}
