use peppygen::exposed_services::{
    set_brightness, set_contrast, set_exposure, set_gain, set_white_balance, video_stream_info,
};
use std::sync::Arc;

use crate::camera::controls::{
    CameraControlRequest, ControlCommand, ControlResult, ControlSender, ExposureMode,
    WhiteBalanceMode,
};
use crate::types::CameraConfig;

// ─────────────────────────────────────────────────────────────────────────────
// Existing: video_stream_info
// ─────────────────────────────────────────────────────────────────────────────

/// Listen for and handle video stream info service requests
pub async fn listen_for_video_stream_info_requests(
    node_runner: Arc<peppygen::NodeRunner>,
    config: CameraConfig,
) {
    loop {
        if let Err(e) = video_stream_info::handle_next_request(&node_runner, |_request| {
            Ok(video_stream_info::Response::new(
                config.resolution.width(),
                config.resolution.height(),
                u8::try_from(config.frame_rate.as_u16()).unwrap_or(u8::MAX),
                config.topic_encoding.to_string(),
            ))
        })
        .await
        {
            tracing::error!("video_stream_info service error: {e:?}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Send a control command to the capture loop and wait for the result.
///
/// This function uses `block_in_place` so it can be called from within a
/// synchronous handler closure that is executing on the tokio async runtime.
fn send_control(control_tx: &ControlSender, request: CameraControlRequest) -> ControlResult {
    tokio::task::block_in_place(|| {
        let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel::<ControlResult>(1);
        if control_tx
            .send(ControlCommand {
                request,
                reply: reply_tx,
            })
            .is_err()
        {
            return ControlResult::err("Camera capture loop is not running");
        }
        reply_rx
            .recv()
            .unwrap_or_else(|_| ControlResult::err("Camera channel closed unexpectedly"))
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// set_exposure
// ─────────────────────────────────────────────────────────────────────────────

/// Listen for and handle `set_exposure` service requests
pub async fn listen_for_set_exposure_requests(
    node_runner: Arc<peppygen::NodeRunner>,
    control_tx: ControlSender,
) {
    loop {
        if let Err(e) = set_exposure::handle_next_request(&node_runner, |request| {
            let mode = match ExposureMode::try_from(request.data.mode.as_str()) {
                Ok(m) => m,
                Err(err) => {
                    return Ok(set_exposure::Response::new(false, err, -1));
                }
            };

            let result = send_control(
                &control_tx,
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
        })
        .await
        {
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
    control_tx: ControlSender,
) {
    loop {
        if let Err(e) = set_white_balance::handle_next_request(&node_runner, |request| {
            let mode = match WhiteBalanceMode::try_from(request.data.mode.as_str()) {
                Ok(m) => m,
                Err(err) => {
                    return Ok(set_white_balance::Response::new(false, err, -1));
                }
            };

            let result = send_control(
                &control_tx,
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
        })
        .await
        {
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
    control_tx: ControlSender,
) {
    loop {
        if let Err(e) = set_gain::handle_next_request(&node_runner, |request| {
            let result = send_control(
                &control_tx,
                CameraControlRequest::SetGain {
                    value: request.data.value,
                },
            );

            Ok(set_gain::Response::new(
                result.success,
                result.message,
                result.current_value,
            ))
        })
        .await
        {
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
    control_tx: ControlSender,
) {
    loop {
        if let Err(e) = set_brightness::handle_next_request(&node_runner, |request| {
            let result = send_control(
                &control_tx,
                CameraControlRequest::SetBrightness {
                    value: request.data.value,
                },
            );

            Ok(set_brightness::Response::new(
                result.success,
                result.message,
                result.current_value,
            ))
        })
        .await
        {
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
    control_tx: ControlSender,
) {
    loop {
        if let Err(e) = set_contrast::handle_next_request(&node_runner, |request| {
            let result = send_control(
                &control_tx,
                CameraControlRequest::SetContrast {
                    value: request.data.value,
                },
            );

            Ok(set_contrast::Response::new(
                result.success,
                result.message,
                result.current_value,
            ))
        })
        .await
        {
            tracing::error!("set_contrast service error: {e:?}");
        }
    }
}
