// Mapping: KER encoder channels to follower joint radians and trigger
// openings. Firmware 2.0.0 zeroes every joint against enactic's calibration
// jig and applies each joint's invert and mechanical offset itself, so a
// channel already carries its joint's angle in the follower's frame: the map
// here is the firmware's channel layout plus the follower's URDF clamp.
//
// Channels are the device's 1-based CH labels, stored 0-based. Trigger angles
// interpolate from the full-squeeze angle (opening 0) to 0 deg, which a
// released trigger reads (opening 1), and clamp into [0, 1].

use openarm_description::{ARM_DOF, HardwareVersion, Side};

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
/// Angle channels the device's schema must carry for this layout.
pub const REQUIRED_CHANNELS: usize = 16;
/// The hardware generation whose channel layout this map describes.
const SUPPORTED_HARDWARE_MAJOR: &str = "2";

/// A device whose channel layout this map does not describe.
#[derive(Debug, thiserror::Error)]
#[error(
    "KER hardware {found} is unsupported: this node maps hardware {SUPPORTED_HARDWARE_MAJOR}.x \
     channels; run a node build that names this generation"
)]
pub struct UnsupportedHardware {
    pub found: String,
}

/// Accept only the hardware generation the channel constants describe; any
/// other generation fails the launch.
pub fn check_hardware(reported: &str) -> Result<(), UnsupportedHardware> {
    (reported.split('.').next() == Some(SUPPORTED_HARDWARE_MAJOR))
        .then_some(())
        .ok_or_else(|| UnsupportedHardware {
            found: reported.to_string(),
        })
}

/// A frame this node cannot map.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MapError {
    #[error("device sent a non-finite angle on CH{channel:02}")]
    NonFiniteAngle { channel: usize },
    #[error("frame carries no CH{channel:02}")]
    ChannelMissing { channel: usize },
}

/// One side's channel wiring and the follower's clamp limits.
#[derive(Debug, Clone)]
pub struct ArmMap {
    channels: [usize; ARM_DOF],
    limits: [[f64; 2]; ARM_DOF],
}

impl ArmMap {
    /// Map one frame's channels to this side's clamped joint radians.
    pub fn joint_radians(&self, angles_deg: &[f32]) -> Result<[f64; ARM_DOF], MapError> {
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
#[derive(Debug, Clone)]
pub struct TriggerMap {
    channel: usize,
    closed_deg: f64,
}

impl TriggerMap {
    /// Map one frame's trigger angle to an opening fraction in [0, 1].
    pub fn opening(&self, angles_deg: &[f32]) -> Result<f64, MapError> {
        let angle = angle_at(angles_deg, self.channel)?;
        Ok((1.0 - angle / self.closed_deg).clamp(0.0, 1.0))
    }
}

/// The device's channel map: both arms and both triggers.
#[derive(Debug, Clone)]
pub struct Calibration {
    pub left: ArmMap,
    pub right: ArmMap,
    pub left_trigger: TriggerMap,
    pub right_trigger: TriggerMap,
}

impl Calibration {
    /// The map for a follower generation, whose URDF limits clamp the joints.
    pub fn for_follower(version: HardwareVersion) -> Self {
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

fn angle_at(angles_deg: &[f32], channel: usize) -> Result<f64, MapError> {
    // The reader validated the schema's channel count at handshake, so a miss
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

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration() -> Calibration {
        Calibration::for_follower(HardwareVersion::V2)
    }

    /// A frame whose channels each carry their own 1-based CH number, so a
    /// joint reading the wrong channel reads a different number.
    fn numbered_frame() -> Vec<f32> {
        (1..=REQUIRED_CHANNELS).map(|c| c as f32).collect()
    }

    #[test]
    fn each_arm_reads_its_own_firmware_channels() {
        let cal = calibration();
        let frame = numbered_frame();
        let right = cal.right.joint_radians(&frame).expect("mapped");
        let left = cal.left.joint_radians(&frame).expect("mapped");
        // CH01..CH07 for the right arm, CH09..CH15 for the left.
        assert!((right[0] - 1f64.to_radians()).abs() < 1e-12);
        assert!((right[6] - 7f64.to_radians()).abs() < 1e-12);
        assert!((left[0] - 9f64.to_radians()).abs() < 1e-12);
        assert!((left[6] - 15f64.to_radians()).abs() < 1e-12);
    }

    #[test]
    fn joints_clamp_into_the_follower_limits() {
        let cal = calibration();
        let frame = vec![400.0f32; REQUIRED_CHANNELS];
        let joints = cal.right.joint_radians(&frame).expect("mapped");
        let limits = HardwareVersion::V2.joint_limits(Side::Right);
        for (joint, [lo, hi]) in joints.into_iter().zip(limits) {
            assert!(joint <= hi && joint >= lo, "{joint} outside [{lo}, {hi}]");
        }
    }

    #[test]
    fn a_released_trigger_is_open_and_a_full_squeeze_is_closed() {
        let cal = calibration();
        let mut frame = vec![0.0f32; REQUIRED_CHANNELS];
        assert_eq!(cal.right_trigger.opening(&frame).expect("mapped"), 1.0);
        assert_eq!(cal.left_trigger.opening(&frame).expect("mapped"), 1.0);

        // The mirrored mechanisms squeeze in opposite directions.
        frame[RIGHT_TRIGGER_CHANNEL] = -30.0;
        frame[LEFT_TRIGGER_CHANNEL] = 30.0;
        assert!((cal.right_trigger.opening(&frame).expect("mapped") - 0.5).abs() < 1e-12);
        assert!((cal.left_trigger.opening(&frame).expect("mapped") - 0.5).abs() < 1e-12);

        // Past the stop, and the wrong way, both clamp.
        frame[RIGHT_TRIGGER_CHANNEL] = -75.0;
        frame[LEFT_TRIGGER_CHANNEL] = -5.0;
        assert_eq!(cal.right_trigger.opening(&frame).expect("mapped"), 0.0);
        assert_eq!(cal.left_trigger.opening(&frame).expect("mapped"), 1.0);
    }

    #[test]
    fn non_finite_angles_are_rejected_not_clamped() {
        let cal = calibration();
        let mut frame = vec![0.0f32; REQUIRED_CHANNELS];
        frame[2] = f32::NAN;
        assert_eq!(
            cal.right.joint_radians(&frame),
            Err(MapError::NonFiniteAngle { channel: 3 })
        );
        frame[LEFT_TRIGGER_CHANNEL] = f32::INFINITY;
        assert_eq!(
            cal.left_trigger.opening(&frame),
            Err(MapError::NonFiniteAngle { channel: 16 })
        );
    }

    #[test]
    fn short_frames_are_rejected() {
        let cal = calibration();
        let frame = vec![0.0f32; 8];
        assert_eq!(
            cal.left.joint_radians(&frame),
            Err(MapError::ChannelMissing { channel: 9 })
        );
    }

    #[test]
    fn only_the_mapped_hardware_generation_is_accepted() {
        for ok in ["2.0.0", "2.1.3", "2"] {
            assert!(check_hardware(ok).is_ok(), "{ok}");
        }
        for bad in ["3.0.0", "1.0.0", "KER-v1.0.0", ""] {
            assert!(check_hardware(bad).is_err(), "{bad}");
        }
    }
}
