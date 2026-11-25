//! What is a depth camera's alone: its two producers, bound under one name in the
//! `cameras` and `geometry` slots, and its description, asked in the background again until
//! it answers once describe_sensors gives the camera's frame ([`crate::sensors`]): its depth
//! lens and model, where its depth optical frame stands in its colour optical frame, by
//! camera_core's reader, which must be the frame describe_sensors gives the camera, its depth
//! unit and its farthest depth.

use std::sync::Arc;
use std::time::Duration;

use camera_core::depth_camera::{DepthCamera, DepthExtrinsics, DepthIntrinsics, read_depth_camera};
use control_core::latched_log::LatchedLog;
use control_core::positive_finite::PositiveFinite;
use control_core::time::sleep_unless_cancelled;
use frame_tree::FrameName;
use peppygen::consumed_services::cameras::depth_stream_info;
use peppygen::consumed_services::geometry::{get_depth_intrinsics, get_depth_to_color_extrinsics};
use peppygen::{NodeRunner, ProducerRef};
use tracing::info;

use crate::base::ANSWER_TIMEOUT;
use crate::sensors::{Described, SensorState, Sensors};

/// How long the description waits between two attempts a camera did not
/// answer.
const RETRY: Duration = Duration::from_millis(500);

/// A depth camera's two producers, bound under its name: its depth frames' and its
/// geometry's.
pub struct CameraSlot {
    pub name: String,
    pub camera: ProducerRef,
    pub geometry: ProducerRef,
}

/// The meters of one depth count: a finite number above 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthUnit(f32);

impl DepthUnit {
    /// `depth_unit` of depth_stream_info, or why it is refused.
    pub fn parse(depth_unit: f32) -> Result<Self, String> {
        if depth_unit.is_finite() && depth_unit > 0.0 {
            return Ok(Self(depth_unit));
        }
        Err(format!(
            "depth_stream_info: depth_unit {depth_unit} is not a positive number of meters"
        ))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

/// A described camera: its depth stream, the meters of one depth count, and
/// the farthest depth (m) it reports, beyond which a sample of 0 met nothing.
#[derive(Clone, Debug)]
pub struct Camera {
    pub depth: DepthCamera,
    pub depth_unit: DepthUnit,
    pub range: PositiveFinite,
}

/// Whether `colour_frame`, the colour optical frame camera_geometry gives a
/// camera, is `described_frame`, the one describe_sensors gives it, or why
/// not.
fn frames_agree(described_frame: &str, colour_frame: &str) -> Result<(), String> {
    if described_frame == colour_frame {
        return Ok(());
    }
    Err(format!(
        "get_depth_to_color_extrinsics places the depth frame in {colour_frame:?}, where \
         sensor_mounts' describe_sensors gives the camera the frame {described_frame:?}: give \
         both the camera's colour optical frame"
    ))
}

/// One description of the camera of `slot`, which describe_sensors gives
/// `frame`: its depth stream, by camera_core's reader, its depth unit and its
/// farthest depth.
async fn describe(runner: &NodeRunner, slot: &CameraSlot, frame: &str) -> Result<Camera, String> {
    let intrinsics = get_depth_intrinsics::poll(runner, &slot.geometry, ANSWER_TIMEOUT)
        .await
        .map_err(|e| format!("get_depth_intrinsics: {e}"))?
        .data;
    let extrinsics = get_depth_to_color_extrinsics::poll(runner, &slot.geometry, ANSWER_TIMEOUT)
        .await
        .map_err(|e| format!("get_depth_to_color_extrinsics: {e}"))?
        .data;
    let info = depth_stream_info::poll(runner, &slot.camera, ANSWER_TIMEOUT)
        .await
        .map_err(|e| format!("depth_stream_info: {e}"))?
        .data;
    let depth_to_color = &extrinsics.depth_to_color;
    let depth = read_depth_camera(
        &DepthIntrinsics {
            width: intrinsics.width,
            height: intrinsics.height,
            fx: intrinsics.fx,
            fy: intrinsics.fy,
            cx: intrinsics.cx,
            cy: intrinsics.cy,
            distortion_model: &intrinsics.distortion_model,
            distortion: &intrinsics.distortion,
            depth_model: &intrinsics.depth_model,
            align_mode: &intrinsics.align_mode,
        },
        &DepthExtrinsics {
            frame: &depth_to_color.frame,
            position: depth_to_color.position,
            orientation: depth_to_color.orientation,
            align_mode: &extrinsics.align_mode,
        },
    )
    .map_err(|e| e.to_string())?;
    frames_agree(frame, depth.frame.as_str())?;
    let depth_unit = DepthUnit::parse(info.depth_unit)?;
    let range = PositiveFinite::parse(intrinsics.max_depth_m).map_err(|_| {
        format!(
            "get_depth_intrinsics: max_depth_m {} is not a positive number of meters",
            intrinsics.max_depth_m
        )
    })?;
    info!(
        "depth camera {}: depth {}x{} in {}, align_mode {:?}, range {} m",
        slot.name,
        intrinsics.width,
        intrinsics.height,
        depth.frame.as_str(),
        depth.align_mode,
        range.get()
    );
    Ok(Camera {
        depth,
        depth_unit,
        range,
    })
}

/// Describe the camera of `slot`, the sensor at `source` of `sensors`, in `frame`, the
/// frame describe_sensors gives it, again until it answers or the node cancels.
pub async fn describe_until_answered(
    runner: Arc<NodeRunner>,
    sensors: Arc<Sensors>,
    source: usize,
    slot: CameraSlot,
    frame: FrameName,
) {
    let token = runner.cancellation_token().clone();
    let failing = LatchedLog::warn(format!("depth camera {} description", slot.name));
    loop {
        match describe(&runner, &slot, frame.as_str()).await {
            Ok(camera) => {
                failing.success();
                let described = Described::DepthCamera(Arc::new(camera));
                return sensors.set(source, SensorState::Ready(described));
            }
            Err(reason) => {
                failing.failure(&reason);
                sensors.set(source, SensorState::Describing(reason));
            }
        }
        if !sleep_unless_cancelled(RETRY, token.cancelled()).await {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A camera whose geometry places its depth frame in a frame other than
    // the one describe_sensors gives it is refused, naming both; a depth unit
    // of 0 or below is refused.
    #[test]
    fn a_camera_is_described_in_its_described_frame_with_a_positive_unit() {
        assert_eq!(
            frames_agree("front_color_optical_frame", "front_color_optical_frame"),
            Ok(())
        );
        let refusal =
            frames_agree("front_color_optical_frame", "front_link").expect_err("two frames");
        assert!(
            refusal.contains("\"front_link\"") && refusal.contains("\"front_color_optical_frame\""),
            "{refusal}"
        );
        assert_eq!(DepthUnit::parse(0.001).map(DepthUnit::get), Ok(0.001));
        for refused in [0.0, -0.001, f32::NAN] {
            assert!(DepthUnit::parse(refused).is_err(), "{refused}");
        }
    }
}
