//! The rgb_camera services: the stream description and the camera controls,
//! each applied to the open device.

use peppygen::exposed_services::camera::{
    set_brightness, set_contrast, set_exposure, set_gain, set_white_balance, video_stream_info,
};
use std::sync::Arc;

use camera_core::controls::ControlFailure;
use camera_core::encoding::ColorEncoding;
use control_core::serving;

use crate::camera::controls::{CameraControlRequest, ControlResult, parse_mode};
use crate::camera::v4l_device::{ControlHandle, StreamDescription};

/// Listen for and handle video stream info service requests.
///
/// Answers with the driver-negotiated stream, not the requested one, so a
/// consumer sizing buffers or validating geometry sees what the wire actually
/// carries.
pub async fn listen_for_video_stream_info_requests(
    node_runner: Arc<peppygen::NodeRunner>,
    stream: StreamDescription,
    topic_encoding: ColorEncoding,
) {
    serving::serve(
        "video_stream_info",
        node_runner.cancellation_token().cancelled(),
        || {
            video_stream_info::handle_next_request(&node_runner, |_request| {
                Ok(video_stream_info::Response::new(
                    stream.width,
                    stream.height,
                    stream.frames_per_second,
                    topic_encoding.to_string(),
                ))
            })
        },
    )
    .await;
}

/// Apply a control synchronously and report what actually happened.
///
/// The ioctl is fast but blocking (a USB round trip), so `block_in_place`
/// keeps it legal inside a synchronous handler closure on the runtime. There
/// is no queue and no timeout: by the time the caller has its response, the
/// hardware state is settled.
fn send_control(controls: &ControlHandle, request: CameraControlRequest) -> ControlResult {
    tokio::task::block_in_place(|| controls.apply(&request))
}

/// Listen for and handle the requests of one camera control service.
/// `$control` turns a request into the control to apply, or refuses a request
/// the node does not take. The response is the value the driver kept; a
/// control that gave no value gets an error reply of its failure kind.
macro_rules! listen_for_control_requests {
    ($fn_name:ident, $service:ident, $control:expr) => {
        pub async fn $fn_name(node_runner: Arc<peppygen::NodeRunner>, controls: ControlHandle) {
            let control_of: fn(
                &$service::RequestData,
            ) -> Result<CameraControlRequest, ControlFailure> = $control;
            serving::serve(
                stringify!($service),
                node_runner.cancellation_token().cancelled(),
                || {
                    $service::handle_next_request(&node_runner, |request| {
                        control_of(&request.data)
                            .and_then(|control| send_control(&controls, control))
                            .map($service::Response::new)
                            .map_err(|failure| camera_core::control_failure!($service, failure))
                    })
                },
            )
            .await;
        }
    };
}

listen_for_control_requests!(listen_for_set_exposure_requests, set_exposure, |data| {
    let mode = parse_mode(&data.mode, "exposure").map_err(ControlFailure::rejected)?;
    Ok(CameraControlRequest::SetExposure {
        mode,
        value: data.value,
    })
});

listen_for_control_requests!(
    listen_for_set_white_balance_requests,
    set_white_balance,
    |data| {
        let mode = parse_mode(&data.mode, "white balance").map_err(ControlFailure::rejected)?;
        Ok(CameraControlRequest::SetWhiteBalance {
            mode,
            temperature: data.temperature,
        })
    }
);

listen_for_control_requests!(listen_for_set_gain_requests, set_gain, |data| {
    Ok(CameraControlRequest::SetGain { value: data.value })
});

listen_for_control_requests!(listen_for_set_brightness_requests, set_brightness, |data| {
    Ok(CameraControlRequest::SetBrightness { value: data.value })
});

listen_for_control_requests!(listen_for_set_contrast_requests, set_contrast, |data| {
    Ok(CameraControlRequest::SetContrast { value: data.value })
});
