// Calibration: KER encoder channels to follower joint radians and trigger
// openings. The CSV parameter strings are parsed once at startup into
// [`Calibration`] (parse, don't validate); after that boundary every frame
// runs through pure, infallible-shaped conversions that only reject
// non-finite device readings.
//
// Conventions: channels are the device's 1-based CH labels (stored 0-based),
// each side lists its 7 joint channels in follower j1..j7 order, and
// `q = sign * radians(angle) + offset` clamps into the follower's URDF limits.
// Trigger angles interpolate linearly from closed_deg (opening 0) to open_deg
// (opening 1) and clamp into [0, 1]; an inverted device direction is just
// closed_deg > open_deg.

use std::fmt;

use openarm_description::{ARM_DOF, HardwareVersion, Side};

#[derive(Debug, Clone, PartialEq)]
pub enum MapError {
    BadNumber {
        param: &'static str,
        value: String,
    },
    WrongCount {
        param: &'static str,
        expected: usize,
        got: usize,
    },
    BadSign {
        param: &'static str,
        value: String,
    },
    ZeroChannel {
        param: &'static str,
    },
    DuplicateChannel {
        channel: usize,
    },
    EmptyTriggerRange {
        param: &'static str,
    },
    NonFiniteAngle {
        channel: usize,
    },
    ChannelMissing {
        channel: usize,
    },
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadNumber { param, value } => {
                write!(f, "{param}: '{value}' is not a number")
            }
            Self::WrongCount {
                param,
                expected,
                got,
            } => {
                write!(
                    f,
                    "{param}: expected {expected} comma-separated values, got {got}"
                )
            }
            Self::BadSign { param, value } => {
                write!(f, "{param}: sign must be 1 or -1, got '{value}'")
            }
            Self::ZeroChannel { param } => {
                write!(f, "{param}: channels are 1-based, 0 is not a channel")
            }
            Self::DuplicateChannel { channel } => {
                write!(f, "channel CH{channel:02} is wired to more than one joint")
            }
            Self::EmptyTriggerRange { param } => {
                write!(f, "{param}: trigger closed and open angles must differ")
            }
            Self::NonFiniteAngle { channel } => {
                write!(f, "device sent a non-finite angle on CH{channel:02}")
            }
            Self::ChannelMissing { channel } => {
                write!(f, "frame carries no CH{channel:02}")
            }
        }
    }
}

impl std::error::Error for MapError {}

/// One side's channel wiring and calibration, plus the follower clamp limits.
#[derive(Debug, Clone)]
pub struct ArmMap {
    /// 0-based channel index per joint, j1..j7.
    channels: [usize; ARM_DOF],
    signs: [f64; ARM_DOF],
    offsets_rad: [f64; ARM_DOF],
    limits: [[f64; 2]; ARM_DOF],
}

impl ArmMap {
    fn parse(
        params: [(&'static str, &str); 3],
        limits: [[f64; 2]; ARM_DOF],
    ) -> Result<Self, MapError> {
        let [channels_csv, signs_csv, offsets_csv] = params;
        let raw_channels = parse_csv(channels_csv.0, channels_csv.1)?;
        let signs = parse_csv(signs_csv.0, signs_csv.1)?;
        let offsets_deg = parse_csv(offsets_csv.0, offsets_csv.1)?;
        let channels = raw_channels
            .iter()
            .map(|&c| match c {
                c if c.fract() != 0.0 || c < 0.0 => Err(MapError::BadNumber {
                    param: channels_csv.0,
                    value: c.to_string(),
                }),
                0.0 => Err(MapError::ZeroChannel {
                    param: channels_csv.0,
                }),
                c => Ok(c as usize - 1),
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .expect("parse_csv yields ARM_DOF values");
        for (i, &s) in signs.iter().enumerate() {
            if s != 1.0 && s != -1.0 {
                return Err(MapError::BadSign {
                    param: signs_csv.0,
                    value: signs_csv
                        .1
                        .split(',')
                        .nth(i)
                        .unwrap_or("")
                        .trim()
                        .to_string(),
                });
            }
        }
        Ok(Self {
            channels,
            signs,
            offsets_rad: offsets_deg.map(f64::to_radians),
            limits,
        })
    }

    /// Highest 0-based channel this side reads, for the schema size check.
    fn max_channel(&self) -> usize {
        *self.channels.iter().max().expect("ARM_DOF > 0")
    }

    /// Map one frame's channels to this side's clamped joint radians.
    pub fn joint_radians(&self, angles_deg: &[f32]) -> Result<[f64; ARM_DOF], MapError> {
        let mut joints = [0.0; ARM_DOF];
        for (i, joint) in joints.iter_mut().enumerate() {
            let angle = angle_at(angles_deg, self.channels[i])?;
            let [lo, hi] = self.limits[i];
            *joint = (self.signs[i] * angle.to_radians() + self.offsets_rad[i]).clamp(lo, hi);
        }
        Ok(joints)
    }
}

/// One trigger's channel and its linear angle-to-opening calibration.
#[derive(Debug, Clone)]
pub struct TriggerMap {
    /// 0-based channel index.
    channel: usize,
    closed_deg: f64,
    open_deg: f64,
}

impl TriggerMap {
    fn new(
        param: &'static str,
        channel_1based: u32,
        closed_deg: f64,
        open_deg: f64,
    ) -> Result<Self, MapError> {
        if channel_1based == 0 {
            return Err(MapError::ZeroChannel { param });
        }
        if !closed_deg.is_finite() || !open_deg.is_finite() || closed_deg == open_deg {
            return Err(MapError::EmptyTriggerRange { param });
        }
        Ok(Self {
            channel: channel_1based as usize - 1,
            closed_deg,
            open_deg,
        })
    }

    /// Map one frame's trigger angle to an opening fraction in [0, 1].
    pub fn opening(&self, angles_deg: &[f32]) -> Result<f64, MapError> {
        let angle = angle_at(angles_deg, self.channel)?;
        let travel = (angle - self.closed_deg) / (self.open_deg - self.closed_deg);
        Ok(travel.clamp(0.0, 1.0))
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

/// The whole device's parsed calibration: both arms and both triggers, with
/// every referenced channel unique across the device.
#[derive(Debug, Clone)]
pub struct Calibration {
    pub left: ArmMap,
    pub right: ArmMap,
    pub left_trigger: TriggerMap,
    pub right_trigger: TriggerMap,
}

/// The raw calibration parameter strings, one field per node parameter.
pub struct CalibrationParams<'a> {
    pub left_channels: &'a str,
    pub left_signs: &'a str,
    pub left_offsets_deg: &'a str,
    pub right_channels: &'a str,
    pub right_signs: &'a str,
    pub right_offsets_deg: &'a str,
    pub left_trigger_channel: u32,
    pub left_trigger_closed_deg: f64,
    pub left_trigger_open_deg: f64,
    pub right_trigger_channel: u32,
    pub right_trigger_closed_deg: f64,
    pub right_trigger_open_deg: f64,
}

impl Calibration {
    pub fn parse(version: HardwareVersion, p: &CalibrationParams) -> Result<Self, MapError> {
        let left = ArmMap::parse(
            [
                ("left_channels", p.left_channels),
                ("left_signs", p.left_signs),
                ("left_offsets_deg", p.left_offsets_deg),
            ],
            version.joint_limits(Side::Left),
        )?;
        let right = ArmMap::parse(
            [
                ("right_channels", p.right_channels),
                ("right_signs", p.right_signs),
                ("right_offsets_deg", p.right_offsets_deg),
            ],
            version.joint_limits(Side::Right),
        )?;
        let left_trigger = TriggerMap::new(
            "left_trigger",
            p.left_trigger_channel,
            p.left_trigger_closed_deg,
            p.left_trigger_open_deg,
        )?;
        let right_trigger = TriggerMap::new(
            "right_trigger",
            p.right_trigger_channel,
            p.right_trigger_closed_deg,
            p.right_trigger_open_deg,
        )?;

        let all: Vec<usize> = left
            .channels
            .iter()
            .chain(right.channels.iter())
            .copied()
            .chain([left_trigger.channel, right_trigger.channel])
            .collect();
        let mut seen = std::collections::HashSet::new();
        for channel in all {
            if !seen.insert(channel) {
                return Err(MapError::DuplicateChannel {
                    channel: channel + 1,
                });
            }
        }

        Ok(Self {
            left,
            right,
            left_trigger,
            right_trigger,
        })
    }

    /// The number of angle channels the device's schema must carry.
    pub fn required_channels(&self) -> usize {
        [
            self.left.max_channel(),
            self.right.max_channel(),
            self.left_trigger.channel,
            self.right_trigger.channel,
        ]
        .into_iter()
        .max()
        .expect("non-empty")
            + 1
    }
}

fn parse_csv(param: &'static str, csv: &str) -> Result<[f64; ARM_DOF], MapError> {
    let values = csv
        .split(',')
        .map(|s| {
            let s = s.trim();
            s.parse::<f64>().map_err(|_| MapError::BadNumber {
                param,
                value: s.to_string(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    values
        .try_into()
        .map_err(|v: Vec<f64>| MapError::WrongCount {
            param,
            expected: ARM_DOF,
            got: v.len(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> CalibrationParams<'static> {
        CalibrationParams {
            left_channels: "1,2,3,4,5,6,7",
            left_signs: "1,1,1,1,1,1,1",
            left_offsets_deg: "0,0,0,0,0,0,0",
            right_channels: "9,10,11,12,13,14,15",
            right_signs: "1,1,1,1,1,1,1",
            right_offsets_deg: "0,0,0,0,0,0,0",
            left_trigger_channel: 8,
            left_trigger_closed_deg: 0.0,
            left_trigger_open_deg: 40.0,
            right_trigger_channel: 16,
            right_trigger_closed_deg: 40.0,
            right_trigger_open_deg: 0.0,
        }
    }

    fn calibration() -> Calibration {
        Calibration::parse(HardwareVersion::V2, &params()).expect("valid params")
    }

    fn frame(angles: [f32; 16]) -> Vec<f32> {
        angles.to_vec()
    }

    #[test]
    fn identity_map_converts_degrees_to_radians() {
        let cal = calibration();
        let mut angles = [0.0f32; 16];
        angles[0] = 30.0;
        let joints = cal.left.joint_radians(&frame(angles)).expect("finite");
        assert!((joints[0] - 30.0f64.to_radians()).abs() < 1e-9);
        // j4's zero sits below the elbow singularity floor, so it clamps up.
        assert_eq!(
            joints[3],
            HardwareVersion::V2.joint_limits(Side::Left)[3][0]
        );
    }

    #[test]
    fn channel_permutation_signs_and_offsets_apply() {
        let cal = Calibration::parse(
            HardwareVersion::V2,
            &CalibrationParams {
                left_channels: "7,6,5,4,3,2,1",
                left_signs: "-1,1,-1,1,-1,1,-1",
                left_offsets_deg: "10,0,0,90,0,0,0",
                ..params()
            },
        )
        .expect("valid");
        let mut angles = [0.0f32; 16];
        angles[6] = 20.0;
        let joints = cal.left.joint_radians(&frame(angles)).expect("finite");
        // j1 reads CH7 = 20 deg with sign -1 and +10 deg offset: -10 deg.
        assert!((joints[0] - (-10.0f64).to_radians()).abs() < 1e-9);
    }

    #[test]
    fn joints_clamp_into_the_follower_limits() {
        let cal = calibration();
        let joints = cal.left.joint_radians(&frame([1.0e6; 16])).expect("finite");
        let limits = HardwareVersion::V2.joint_limits(Side::Left);
        for (j, &[_, hi]) in joints.iter().zip(limits.iter()) {
            assert_eq!(*j, hi);
        }
    }

    #[test]
    fn non_finite_angles_are_rejected_not_clamped() {
        let cal = calibration();
        let mut angles = [0.0f32; 16];
        angles[2] = f32::NAN;
        assert_eq!(
            cal.left.joint_radians(&frame(angles)),
            Err(MapError::NonFiniteAngle { channel: 3 })
        );
    }

    #[test]
    fn short_frames_are_rejected() {
        let cal = calibration();
        assert!(cal.right.joint_radians(&[0.0; 8]).is_err());
    }

    #[test]
    fn trigger_maps_linearly_and_clamps() {
        let cal = calibration();
        let mut angles = [0.0f32; 16];

        angles[7] = 0.0;
        assert_eq!(cal.left_trigger.opening(&frame(angles)).unwrap(), 0.0);
        angles[7] = 20.0;
        let mid = cal.left_trigger.opening(&frame(angles)).unwrap();
        assert!((mid - 0.5).abs() < 1e-9);
        angles[7] = 40.0;
        assert_eq!(cal.left_trigger.opening(&frame(angles)).unwrap(), 1.0);
        angles[7] = 80.0;
        assert_eq!(
            cal.left_trigger.opening(&frame(angles)).unwrap(),
            1.0,
            "past-open clamps to fully open"
        );
        angles[7] = -10.0;
        assert_eq!(cal.left_trigger.opening(&frame(angles)).unwrap(), 0.0);
    }

    #[test]
    fn inverted_trigger_direction_works() {
        // The right trigger runs 40 deg (closed) down to 0 deg (open).
        let cal = calibration();
        let mut angles = [0.0f32; 16];
        angles[15] = 40.0;
        assert_eq!(cal.right_trigger.opening(&frame(angles)).unwrap(), 0.0);
        angles[15] = 0.0;
        assert_eq!(cal.right_trigger.opening(&frame(angles)).unwrap(), 1.0);
    }

    #[test]
    fn required_channels_covers_the_highest_wired_channel() {
        assert_eq!(calibration().required_channels(), 16);
    }

    #[test]
    fn csv_errors_are_rejected() {
        let cases: Vec<(CalibrationParams, MapError)> = vec![
            (
                CalibrationParams {
                    left_channels: "1,2,3,4,5,6",
                    ..params()
                },
                MapError::WrongCount {
                    param: "left_channels",
                    expected: 7,
                    got: 6,
                },
            ),
            (
                CalibrationParams {
                    left_channels: "1,2,3,4,5,6,x",
                    ..params()
                },
                MapError::BadNumber {
                    param: "left_channels",
                    value: "x".into(),
                },
            ),
            (
                CalibrationParams {
                    left_channels: "0,2,3,4,5,6,7",
                    ..params()
                },
                MapError::ZeroChannel {
                    param: "left_channels",
                },
            ),
            (
                CalibrationParams {
                    left_channels: "1.5,2,3,4,5,6,7",
                    ..params()
                },
                MapError::BadNumber {
                    param: "left_channels",
                    value: "1.5".into(),
                },
            ),
            (
                CalibrationParams {
                    left_signs: "1,1,1,2,1,1,1",
                    ..params()
                },
                MapError::BadSign {
                    param: "left_signs",
                    value: "2".into(),
                },
            ),
            (
                CalibrationParams {
                    left_channels: "1,1,3,4,5,6,7",
                    ..params()
                },
                MapError::DuplicateChannel { channel: 1 },
            ),
            (
                CalibrationParams {
                    right_trigger_channel: 8,
                    ..params()
                },
                MapError::DuplicateChannel { channel: 8 },
            ),
            (
                CalibrationParams {
                    left_trigger_channel: 0,
                    ..params()
                },
                MapError::ZeroChannel {
                    param: "left_trigger",
                },
            ),
            (
                CalibrationParams {
                    left_trigger_open_deg: 0.0,
                    ..params()
                },
                MapError::EmptyTriggerRange {
                    param: "left_trigger",
                },
            ),
        ];
        for (p, expected) in cases {
            assert_eq!(
                Calibration::parse(HardwareVersion::V2, &p).expect_err("must reject"),
                expected
            );
        }
    }

    #[test]
    fn whitespace_in_csvs_is_tolerated() {
        let cal = Calibration::parse(
            HardwareVersion::V2,
            &CalibrationParams {
                left_channels: " 1, 2, 3, 4, 5, 6, 7 ",
                left_signs: " 1, -1, 1, -1, 1, -1, 1 ",
                ..params()
            },
        );
        assert!(cal.is_ok());
    }
}
