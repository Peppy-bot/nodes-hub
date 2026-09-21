// Mapping: KER encoder channels to follower joint radians, trigger openings
// and commanded gripper openings. Firmware 2.x zeroes every joint against
// enactic's calibration jig and applies each joint's invert and mechanical
// offset itself, so a channel already carries its joint's angle in the
// follower's frame: the map here is the firmware's channel layout plus the
// follower's URDF clamp.
//
// Channels are the device's 1-based CH labels, stored 0-based. Trigger angles
// interpolate from the full-squeeze angle (opening 0) to 0 deg, which a
// released trigger reads (opening 1), and clamp into [0, 1]; the gripper is
// commanded that opening scaled by the launcher's open fraction.

use openarm_description::{ARM_DOF, HardwareVersion, Side};

use crate::protocol::{KerFrame, Metadata};
use crate::side::SideValues;

/// CH01-CH07: the right arm's j1..j7, from firmware 2.0.0's ENCODER_CONFIG.
const RIGHT_JOINT_CHANNELS: [usize; ARM_DOF] = [0, 1, 2, 3, 4, 5, 6];
/// CH09-CH15: the left arm's j1..j7.
const LEFT_JOINT_CHANNELS: [usize; ARM_DOF] = [8, 9, 10, 11, 12, 13, 14];
/// CH08: the right trigger.
const RIGHT_TRIGGER_CHANNEL: usize = 7;
/// CH16: the left trigger.
const LEFT_TRIGGER_CHANNEL: usize = 15;
/// Trigger angle (deg) at a full squeeze; the mirrored mechanisms travel in
/// opposite directions from the 0 deg a released trigger reads.
const RIGHT_TRIGGER_CLOSED_DEG: f64 = -60.0;
const LEFT_TRIGGER_CLOSED_DEG: f64 = 60.0;
/// Angle channels the device's schema must carry, derived from the highest
/// channel these constants read.
const REQUIRED_CHANNELS: usize = required_channels();

const fn required_channels() -> usize {
    let mut highest = RIGHT_TRIGGER_CHANNEL;
    if LEFT_TRIGGER_CHANNEL > highest {
        highest = LEFT_TRIGGER_CHANNEL;
    }
    let mut i = 0;
    while i < ARM_DOF {
        if RIGHT_JOINT_CHANNELS[i] > highest {
            highest = RIGHT_JOINT_CHANNELS[i];
        }
        if LEFT_JOINT_CHANNELS[i] > highest {
            highest = LEFT_JOINT_CHANNELS[i];
        }
        i += 1;
    }
    highest + 1
}
/// The hardware generation whose channel layout this map describes.
const SUPPORTED_HARDWARE_MAJOR: &str = "2";

/// A device this channel map does not describe.
#[derive(Debug, thiserror::Error)]
pub enum DeviceMismatch {
    #[error(
        "KER reports hardware '{found}'; this node maps the hardware \
         {SUPPORTED_HARDWARE_MAJOR}.x channel layout. Connect a \
         {SUPPORTED_HARDWARE_MAJOR}.x KER; `uvx --from openarm_ker openarm-ker-cli ping` \
         prints what a device reports"
    )]
    Hardware { found: String },

    #[error(
        "KER streams {found} channels; this node reads CH01-CH{REQUIRED_CHANNELS:02}. \
         `uvx --from openarm_ker openarm-ker-cli ping` prints the schema the device reports"
    )]
    Channels { found: usize },
}

/// A launcher `gripper_open_fraction` outside (0, 1].
#[derive(Debug, thiserror::Error)]
#[error(
    "gripper_open_fraction is the opening a released trigger commands: greater than 0 and \
     at most 1. Try 0.5, or leave the 1.0 default; got {0}"
)]
pub struct GripperOpenFractionOutOfRange(pub f64);

/// The gripper opening a released trigger commands; a full squeeze closes.
#[derive(Debug, Clone, Copy)]
pub struct GripperOpenFraction(f64);

impl GripperOpenFraction {
    pub fn fraction(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for GripperOpenFraction {
    type Error = GripperOpenFractionOutOfRange;

    fn try_from(fraction: f64) -> Result<Self, Self::Error> {
        (fraction > 0.0 && fraction <= 1.0)
            .then_some(Self(fraction))
            .ok_or(GripperOpenFractionOutOfRange(fraction))
    }
}

/// A frame this node cannot map.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum MapError {
    #[error("device sent a non-finite angle on CH{channel:02}")]
    NonFiniteAngle { channel: usize },
    #[error("frame carries no CH{channel:02}")]
    ChannelMissing { channel: usize },
}

/// One side's channel wiring and the follower's clamp limits.
#[derive(Debug)]
struct ArmMap {
    channels: [usize; ARM_DOF],
    limits: [[f64; 2]; ARM_DOF],
}

impl ArmMap {
    /// Map one frame's channels to this side's clamped joint radians.
    fn joint_radians(&self, angles_deg: &[f32]) -> Result<[f64; ARM_DOF], MapError> {
        let mut joints = [0.0; ARM_DOF];
        for (i, joint) in joints.iter_mut().enumerate() {
            let [lo, hi] = self.limits[i];
            *joint = angle_at(angles_deg, self.channels[i])?
                .to_radians()
                .clamp(lo, hi);
        }
        Ok(joints)
    }
}

/// One trigger's channel and its angle-to-opening travel.
#[derive(Debug)]
struct TriggerMap {
    channel: usize,
    closed_deg: f64,
}

impl TriggerMap {
    /// This frame's trigger opening: 1 released, 0 at a full squeeze.
    fn opening(&self, angles_deg: &[f32]) -> Result<f64, MapError> {
        let angle = angle_at(angles_deg, self.channel)?;
        Ok((1.0 - angle / self.closed_deg).clamp(0.0, 1.0))
    }
}

/// The device's channel map: both arms and both triggers.
#[derive(Debug)]
pub struct ChannelMap {
    left: ArmMap,
    right: ArmMap,
    left_trigger: TriggerMap,
    right_trigger: TriggerMap,
}

impl ChannelMap {
    /// The map for a handshaken device, refused unless its reported hardware
    /// and channel count are the ones these constants describe. The follower
    /// generation supplies the URDF limits that clamp the joints.
    pub fn for_device(
        version: HardwareVersion,
        metadata: &Metadata,
        angle_count: usize,
    ) -> Result<Self, DeviceMismatch> {
        let major = metadata.hardware.split('.').next().unwrap_or_default();
        if major != SUPPORTED_HARDWARE_MAJOR {
            return Err(DeviceMismatch::Hardware {
                found: metadata.hardware.clone(),
            });
        }
        if angle_count < REQUIRED_CHANNELS {
            return Err(DeviceMismatch::Channels { found: angle_count });
        }
        Ok(Self::for_follower(version))
    }

    /// Map one frame's channels: joints clamped into the follower's limits,
    /// each trigger's opening, and the gripper opening each one commands.
    pub fn map(
        &self,
        frame: &KerFrame,
        open_fraction: GripperOpenFraction,
    ) -> Result<MappedFrame, MapError> {
        let angles = &frame.angles_deg;
        let triggers = SideValues {
            left: self.left_trigger.opening(angles)?,
            right: self.right_trigger.opening(angles)?,
        };
        Ok(MappedFrame {
            left_joints: self.left.joint_radians(angles)?,
            right_joints: self.right.joint_radians(angles)?,
            gripper_openings: triggers.scaled(open_fraction.fraction()),
            triggers,
        })
    }

    /// The map alone, for a follower generation.
    fn for_follower(version: HardwareVersion) -> Self {
        Self {
            left: ArmMap {
                channels: LEFT_JOINT_CHANNELS,
                limits: version.joint_limits(Side::Left),
            },
            right: ArmMap {
                channels: RIGHT_JOINT_CHANNELS,
                limits: version.joint_limits(Side::Right),
            },
            left_trigger: TriggerMap {
                channel: LEFT_TRIGGER_CHANNEL,
                closed_deg: LEFT_TRIGGER_CLOSED_DEG,
            },
            right_trigger: TriggerMap {
                channel: RIGHT_TRIGGER_CHANNEL,
                closed_deg: RIGHT_TRIGGER_CLOSED_DEG,
            },
        }
    }
}

/// One frame's mapped values, before engagement decides what streams.
pub struct MappedFrame {
    pub left_joints: [f64; ARM_DOF],
    pub right_joints: [f64; ARM_DOF],
    /// Trigger openings, which engagement reads unscaled.
    pub triggers: SideValues,
    pub gripper_openings: SideValues,
}

fn angle_at(angles_deg: &[f32], channel: usize) -> Result<f64, MapError> {
    // The reader accepted the schema's channel count at handshake, so a miss
    // here means the device changed its schema mid-connection.
    let angle = *angles_deg.get(channel).ok_or(MapError::ChannelMissing {
        channel: channel + 1,
    })? as f64;
    if !angle.is_finite() {
        return Err(MapError::NonFiniteAngle {
            channel: channel + 1,
        });
    }
    Ok(angle)
}

/// Device-side builders this crate's tests share; they need the channel
/// constants above.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    /// The angle a probe frame's first channel carries, and the step between
    /// channels: a window inside every joint limit this map clamps to.
    const PROBE_BASE_DEG: f32 = 3.0;
    const PROBE_STEP_DEG: f32 = 0.3;

    /// The channel count a fake device must stream.
    pub(crate) const CHANNELS: usize = REQUIRED_CHANNELS;
    /// A trigger angle (deg) at a full squeeze, per side.
    pub(crate) const RIGHT_SQUEEZE_DEG: f64 = RIGHT_TRIGGER_CLOSED_DEG;
    pub(crate) const LEFT_SQUEEZE_DEG: f64 = LEFT_TRIGGER_CLOSED_DEG;
    /// The channels each trigger reads, for building a frame.
    pub(crate) const RIGHT_TRIGGER: usize = RIGHT_TRIGGER_CHANNEL;
    pub(crate) const LEFT_TRIGGER: usize = LEFT_TRIGGER_CHANNEL;

    pub(crate) fn channel_map() -> ChannelMap {
        ChannelMap::for_follower(HardwareVersion::V2)
    }

    /// Every channel carries a distinct angle derived from its own 1-based CH
    /// number, inside the tightest joint window this map clamps to (the elbow
    /// floor below, the left shoulder's upper limit above), so every joint
    /// maps to a value of its own and nothing clamps.
    pub(crate) fn numbered_frame() -> Vec<f32> {
        (1..=CHANNELS)
            .map(|c| PROBE_BASE_DEG + c as f32 * PROBE_STEP_DEG)
            .collect()
    }

    /// The joint radians `numbered_frame` maps to on one side, unclamped.
    pub(crate) fn probe_radians(frame: &[f32], first_channel: usize) -> [f64; ARM_DOF] {
        std::array::from_fn(|j| (frame[first_channel + j] as f64).to_radians())
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    fn metadata(hardware: &str) -> Metadata {
        Metadata {
            firmware: "2.0.0".into(),
            hardware: hardware.into(),
            updated: "2026-06-22".into(),
        }
    }

    #[test]
    fn each_joint_reads_its_own_firmware_channel() {
        let map = channel_map();
        let frame = numbered_frame();
        let right = map.right.joint_radians(&frame).expect("mapped");
        let left = map.left.joint_radians(&frame).expect("mapped");
        // CH01..CH07 for the right arm, CH09..CH15 for the left. The probe
        // window is inside every limit, so nothing clamps and a joint reading
        // the wrong channel reads a different number.
        for (j, expected) in probe_radians(&frame, 0).into_iter().enumerate() {
            assert!(
                (right[j] - expected).abs() < 1e-12,
                "right j{}: {} != {expected}",
                j + 1,
                right[j]
            );
        }
        for (j, expected) in probe_radians(&frame, 8).into_iter().enumerate() {
            assert!(
                (left[j] - expected).abs() < 1e-12,
                "left j{}: {} != {expected}",
                j + 1,
                left[j]
            );
        }
    }

    #[test]
    fn joints_clamp_into_the_follower_limits_at_both_ends() {
        let map = channel_map();
        for sweep in [400.0f32, -400.0] {
            let frame = vec![sweep; CHANNELS];
            for (side, arm) in [(Side::Left, &map.left), (Side::Right, &map.right)] {
                let joints = arm.joint_radians(&frame).expect("mapped");
                let limits = HardwareVersion::V2.joint_limits(side);
                for (j, (joint, [lo, hi])) in joints.into_iter().zip(limits).enumerate() {
                    let bound = if sweep > 0.0 { hi } else { lo };
                    assert_eq!(joint, bound, "{side:?} j{} at {sweep} deg", j + 1);
                }
            }
        }
    }

    #[test]
    fn a_released_trigger_is_open_and_a_full_squeeze_is_closed() {
        let map = channel_map();
        let mut frame = vec![0.0f32; CHANNELS];
        assert_eq!(map.right_trigger.opening(&frame).expect("mapped"), 1.0);
        assert_eq!(map.left_trigger.opening(&frame).expect("mapped"), 1.0);

        // The mirrored mechanisms squeeze in opposite directions.
        frame[RIGHT_TRIGGER] = RIGHT_SQUEEZE_DEG as f32 / 2.0;
        frame[LEFT_TRIGGER] = LEFT_SQUEEZE_DEG as f32 / 2.0;
        assert!((map.right_trigger.opening(&frame).expect("mapped") - 0.5).abs() < 1e-12);
        assert!((map.left_trigger.opening(&frame).expect("mapped") - 0.5).abs() < 1e-12);

        // Past the stop, and the wrong way, both clamp.
        frame[RIGHT_TRIGGER] = RIGHT_SQUEEZE_DEG as f32 * 1.25;
        frame[LEFT_TRIGGER] = -5.0;
        assert_eq!(map.right_trigger.opening(&frame).expect("mapped"), 0.0);
        assert_eq!(map.left_trigger.opening(&frame).expect("mapped"), 1.0);
    }

    #[test]
    fn gripper_open_fraction_accepts_only_fractions_in_zero_exclusive_to_one() {
        for fraction in [0.01, 0.5, 1.0] {
            assert!(
                GripperOpenFraction::try_from(fraction).is_ok(),
                "{fraction}"
            );
        }
        for fraction in [0.0, -0.5, 1.01, f64::NAN, f64::INFINITY] {
            let refused = GripperOpenFraction::try_from(fraction)
                .expect_err("out of range")
                .to_string();
            assert!(refused.contains("gripper_open_fraction"), "{refused}");
        }
    }

    #[test]
    fn non_finite_angles_are_rejected_not_clamped() {
        let map = channel_map();
        let mut frame = vec![0.0f32; CHANNELS];
        frame[2] = f32::NAN;
        assert_eq!(
            map.right.joint_radians(&frame),
            Err(MapError::NonFiniteAngle { channel: 3 })
        );
        frame[LEFT_TRIGGER] = f32::INFINITY;
        assert_eq!(
            map.left_trigger.opening(&frame),
            Err(MapError::NonFiniteAngle {
                channel: LEFT_TRIGGER + 1
            })
        );
    }

    #[test]
    fn short_frames_are_rejected() {
        let map = channel_map();
        let frame = vec![0.0f32; 8];
        assert_eq!(
            map.left.joint_radians(&frame),
            Err(MapError::ChannelMissing { channel: 9 })
        );
    }

    #[test]
    fn required_channels_covers_every_channel_read() {
        for channel in LEFT_JOINT_CHANNELS
            .iter()
            .chain(&RIGHT_JOINT_CHANNELS)
            .chain([&LEFT_TRIGGER_CHANNEL, &RIGHT_TRIGGER_CHANNEL])
        {
            assert!(
                REQUIRED_CHANNELS > *channel,
                "CH{:02} is read but not required",
                channel + 1
            );
        }
        assert_eq!(REQUIRED_CHANNELS, 16, "firmware 2.x streams sixteen");
    }

    #[test]
    fn only_a_two_x_device_streaming_every_channel_builds_a_map() {
        assert!(ChannelMap::for_device(HardwareVersion::V2, &metadata("2.0.0"), CHANNELS).is_ok());
        assert!(
            ChannelMap::for_device(HardwareVersion::V2, &metadata("2.1.3"), CHANNELS + 4).is_ok()
        );

        // "20.0.0" holds the comparison to the whole major version.
        for found in ["3.0.0", "1.0.0", "20.0.0", "KER-v2.0.0", ""] {
            let refused = ChannelMap::for_device(HardwareVersion::V2, &metadata(found), CHANNELS)
                .expect_err("unsupported hardware")
                .to_string();
            assert!(refused.contains("hardware"), "{found}: {refused}");
            assert!(refused.contains(found) || found.is_empty(), "{refused}");
        }

        let refused = ChannelMap::for_device(HardwareVersion::V2, &metadata("2.0.0"), CHANNELS - 1)
            .expect_err("short schema")
            .to_string();
        assert!(refused.contains("15 channels"), "{refused}");
    }
}
