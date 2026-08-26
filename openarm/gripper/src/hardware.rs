//! The gripper motor as this node commands it: the CAN driver in its
//! generation's control mode, paired with that instance's opening geometry.
//!
//! The two generations differ only here. The v1.0 prismatic jaws run MIT
//! position hold at fixed gains, a mode carrying no per-command force limit,
//! so that gripper caps no grip effort and reports none on `gripper_states`
//! (its motor torque still reaches the wire as motor_health). The v2.0
//! revolute pinch runs POS_FORCE, a position command bounded by a speed limit
//! and a torque-current ceiling, and reports the force it meets. Every other
//! operation is the same bus behaviour, dispatched once through [`ModeBus`].
//!
//! One type owns the driver, the geometry and the mode's bounds together,
//! resolved from the same `hardware_version`, so no caller can pair a
//! generation with another's mapping or with bounds its mode cannot apply.

use std::num::NonZeroU32;
use std::time::Duration;

use control_core::motor_health::Ratings;
use openarm_can::{
    CanError, EnableFailure, GripperCan, GripperState, Mit, Mode, MotorParam, MotorType, PosForce,
    v10, v20,
};
use openarm_description::{HardwareVersion, Side};

use crate::geometry::Geometry;

/// A motor the datasheet table carries no ratings for.
#[derive(Debug, thiserror::Error)]
#[error("no datasheet ratings for {0:?}")]
pub struct NoDatasheetRatings(pub MotorType);

/// A configured limit the POS_FORCE command fields cannot express.
#[derive(Debug, thiserror::Error)]
pub enum PosForceLimitsError {
    #[error(
        "speed_rad_s must be in [{WIRE_MIN_SPEED_RAD_S}, {WIRE_MAX_SPEED_RAD_S}] rad/s, the \
         range the POS_FORCE speed field can express, got {0}"
    )]
    Speed(f64),

    #[error(
        "max_effort_nm must be in [{smallest_nm}, {peak_nm}] N*m at the shaft, between the \
         smallest cap the POS_FORCE torque-current field can express and the datasheet peak \
         of the {motor_type:?}, got {got}"
    )]
    Effort {
        smallest_nm: f64,
        peak_nm: f64,
        motor_type: MotorType,
        got: f64,
    },

    #[error(transparent)]
    Ratings(#[from] NoDatasheetRatings),
}

/// v1.0 MIT-mode gains, matching the openarm teleop follower
/// (config/follower.yaml gripper entry). Hardcoded, as in the ROS2 reference.
const MIT_KP: f64 = 16.0;
const MIT_KD: f64 = 0.2;

/// Ceiling of the POS_FORCE wire speed field (rad/s at the motor).
/// `pos_force_frame` clamps to it, so a larger configured limit would be
/// silently reduced rather than applied.
const WIRE_MAX_SPEED_RAD_S: f64 = 100.0;

/// One tick of that same field, which carries hundredths and truncates: a
/// limit below this reaches the motor as a full stop.
const WIRE_MIN_SPEED_RAD_S: f64 = 0.01;

/// Smallest torque-current the POS_FORCE field can express: it carries
/// per-unit in ten-thousandths and truncates, so anything under one tick
/// reaches the motor as a cap of zero, permitting no grip force at all.
const WIRE_TORQUE_PU_TICK: f64 = 1.0 / 10_000.0;

/// A commanded grip-force cap (N*m at the shaft), parsed at the wire boundary
/// so the command path cannot be handed a value POS_FORCE would reject.
#[derive(Debug, Clone, Copy)]
pub struct MaxEffortNm(f64);

impl MaxEffortNm {
    /// Parses the wire's `max_effort` field. The contract's 0 means "no
    /// preference", which is absence rather than a cap of zero.
    pub fn parse(nm: f64) -> Result<Option<Self>, String> {
        if !nm.is_finite() {
            return Err(format!("max_effort must be finite, got {nm}"));
        }
        if nm < 0.0 {
            return Err(format!("max_effort must not be negative, got {nm}"));
        }
        Ok((nm > 0.0).then_some(Self(nm)))
    }
}

/// The POS_FORCE bounds every v2 command carries, parsed once at startup.
#[derive(Debug, Clone, Copy)]
pub struct PosForceLimits {
    /// Absolute speed limit (rad/s at the motor).
    speed_rad_s: f64,
    /// Largest grip force this instance applies (N*m at the shaft). A
    /// streamed `max_effort` can only lower a tick's cap beneath it.
    max_effort_nm: f64,
    /// The span the motor's per-unit torque-current field addresses (N*m at
    /// the shaft), resolved once at startup so the single conversion to the
    /// wire's units needs no motor lookup on the control loop.
    full_scale_nm: f64,
    /// The datasheet ratings resolved while bounding `max_effort_nm`, kept so
    /// the one lookup serves both the ceiling and the health filters.
    ratings: Ratings,
}

/// The motor a gripper of this generation drives.
pub fn gripper_motor_type(hardware_version: HardwareVersion) -> MotorType {
    match hardware_version {
        HardwareVersion::V1 => v10::GRIPPER_MOTOR_TYPE,
        HardwareVersion::V2 => v20::GRIPPER_MOTOR_TYPE,
    }
}

impl PosForceLimits {
    /// Rejects the values POS_FORCE cannot carry or the robot cannot mean.
    /// Checked for both generations: the ranges are properties of the
    /// parameters, not of the gripper that happens to read them.
    ///
    /// The floors are the wire's resolution, not zero. A ceiling the field
    /// truncates to nothing would command zero grip force on every tick while
    /// publishing the `max_effort` value the gripper_link contract reserves
    /// for "no effort control", leaving a v2 gripper that cannot grip and
    /// reads on the wire as a v1 that never could; a speed that truncates to
    /// nothing would command a gripper that cannot move. The force ceiling is
    /// bounded above by the motor's datasheet peak rather than by the field's
    /// full scale, which carries quantization headroom the hardware cannot
    /// deliver.
    pub fn new(
        motor_type: MotorType,
        speed_rad_s: f64,
        max_effort_nm: f64,
    ) -> Result<Self, PosForceLimitsError> {
        if !(speed_rad_s.is_finite()
            && (WIRE_MIN_SPEED_RAD_S..=WIRE_MAX_SPEED_RAD_S).contains(&speed_rad_s))
        {
            return Err(PosForceLimitsError::Speed(speed_rad_s));
        }
        let full_scale_nm = torque_full_scale_nm(motor_type);
        let ratings = ratings_of(motor_type)?;
        let peak_nm = ratings.peak_nm();
        let smallest_nm = WIRE_TORQUE_PU_TICK * full_scale_nm;
        if !(max_effort_nm.is_finite() && (smallest_nm..=peak_nm).contains(&max_effort_nm)) {
            return Err(PosForceLimitsError::Effort {
                smallest_nm,
                peak_nm,
                motor_type,
                got: max_effort_nm,
            });
        }
        Ok(Self {
            speed_rad_s,
            max_effort_nm,
            full_scale_nm,
            ratings,
        })
    }

    /// The datasheet ratings resolved when these limits were parsed.
    pub fn ratings(self) -> Ratings {
        self.ratings
    }

    /// The configured ceiling (N*m at the shaft). This is what reaches
    /// gripper_states; a streamed max_effort lowers an individual tick's cap
    /// beneath it without restating the ceiling.
    fn effort_ceiling_nm(self) -> f64 {
        self.max_effort_nm
    }

    /// A commanded cap as the POS_FORCE per-unit torque-current field: the
    /// one place shaft N*m becomes the wire's fraction. Bounded by the
    /// configured ceiling; no commanded preference means the ceiling itself.
    /// Floored at one wire tick, because a positive commanded cap that
    /// truncated to nothing would permit no grip force at all rather than the
    /// small cap it asked for.
    fn torque_pu(self, max_effort: Option<MaxEffortNm>) -> f64 {
        let capped_nm = max_effort.map_or(self.max_effort_nm, |MaxEffortNm(nm)| {
            nm.min(self.max_effort_nm)
        });
        (capped_nm / self.full_scale_nm).max(WIRE_TORQUE_PU_TICK)
    }
}

/// The datasheet ratings this node judges the motor against. The one place a
/// missing rating is turned into a refusal.
fn ratings_of(motor_type: MotorType) -> Result<Ratings, NoDatasheetRatings> {
    motor_type.ratings().ok_or(NoDatasheetRatings(motor_type))
}

/// The span the motor's per-unit torque-current field addresses (N*m at the
/// shaft). Infallible: `TorqueMax` is a scale register, so every motor decodes
/// it.
fn torque_full_scale_nm(motor_type: MotorType) -> f64 {
    motor_type
        .decode_full_scale(MotorParam::TorqueMax)
        .expect("TorqueMax is a scale register, so every motor decodes it")
}

/// The bus operations both control modes answer identically. `openarm_can`
/// implements them once on its mode-generic driver; naming them here is what
/// lets this node dispatch the mode in one place instead of restating every
/// call per variant.
trait ModeBus {
    fn get_state(&self) -> GripperState;
    fn recv_all(&mut self, first_timeout_us: u32) -> Result<(), CanError>;
    fn refresh_all(&mut self) -> Result<(), CanError>;
    fn reenable(&mut self) -> Result<(), CanError>;
    fn disable_all(&mut self) -> Result<(), CanError>;
    fn enable_and_confirm(
        &mut self,
        attempts: NonZeroU32,
        settle: Duration,
        recv_timeout_us: u32,
    ) -> Result<(), EnableFailure>;
}

impl<M: Mode> ModeBus for GripperCan<M> {
    fn get_state(&self) -> GripperState {
        GripperCan::get_state(self)
    }

    fn recv_all(&mut self, first_timeout_us: u32) -> Result<(), CanError> {
        GripperCan::recv_all(self, first_timeout_us)
    }

    fn refresh_all(&mut self) -> Result<(), CanError> {
        GripperCan::refresh_all(self)
    }

    fn reenable(&mut self) -> Result<(), CanError> {
        GripperCan::reenable(self)
    }

    fn disable_all(&mut self) -> Result<(), CanError> {
        GripperCan::disable_all(self)
    }

    fn enable_and_confirm(
        &mut self,
        attempts: NonZeroU32,
        settle: Duration,
        recv_timeout_us: u32,
    ) -> Result<(), EnableFailure> {
        GripperCan::enable_and_confirm(self, attempts, settle, recv_timeout_us)
    }
}

/// The driver in the control mode its generation is commanded through.
enum Drive {
    Mit(GripperCan<Mit>),
    PosForce {
        can: GripperCan<PosForce>,
        limits: PosForceLimits,
    },
}

/// The gripper_states triple this instance reports for one cached sample.
pub struct WireState {
    pub opening: f64,
    pub effort: f64,
    pub max_effort: f64,
}

pub struct Gripper {
    drive: Drive,
    geometry: Geometry,
}

impl Gripper {
    /// Opens the motor in this generation's control mode and resolves this
    /// side's opening geometry. `limits` reaches the wire only on v2, whose
    /// POS_FORCE frames carry them; MIT takes no per-command bounds.
    pub fn open(
        hardware_version: HardwareVersion,
        side: Side,
        can_interface: &str,
        enable_fd: bool,
        limits: PosForceLimits,
    ) -> Result<Self, CanError> {
        let drive = match hardware_version {
            HardwareVersion::V1 => Drive::Mit(GripperCan::open_mit(
                can_interface,
                enable_fd,
                v10::GRIPPER_MOTOR_TYPE,
                v10::GRIPPER_SEND_ID,
                v10::GRIPPER_RECV_ID,
            )?),
            HardwareVersion::V2 => Drive::PosForce {
                can: GripperCan::open_pos_force(
                    can_interface,
                    enable_fd,
                    v20::GRIPPER_MOTOR_TYPE,
                    v20::GRIPPER_SEND_ID,
                    v20::GRIPPER_RECV_ID,
                )?,
                limits,
            },
        };
        Ok(Self {
            drive,
            geometry: Geometry::new(hardware_version, side),
        })
    }

    /// Drive toward `opening` (fraction, 0 = closed, 1 = fully open), capping
    /// the effort at `max_effort` where the mode can. The clamp saturates at
    /// the calibrated travel, which is the wanted behaviour rather than a
    /// distrust of the caller. MIT carries no cap, so v1 ignores it.
    pub fn command(
        &mut self,
        opening: f64,
        max_effort: Option<MaxEffortNm>,
    ) -> Result<(), CanError> {
        let motor_rad = self.geometry.fraction_to_motor_rad(opening.clamp(0.0, 1.0));
        match &mut self.drive {
            Drive::Mit(can) => can.mit_control(MIT_KP, MIT_KD, motor_rad, 0.0, 0.0),
            Drive::PosForce { can, limits } => {
                can.set_position(motor_rad, limits.speed_rad_s, limits.torque_pu(max_effort))
            }
        }
    }

    /// The measured opening, grip effort, and effort ceiling for the cached
    /// sample, in the wire's terms. MIT carries no per-command force cap, so
    /// v1 reports 0 effort and the 0 ceiling the gripper_link contract reads
    /// as "no effort control".
    pub fn wire_state(&self) -> WireState {
        let state = self.get_state();
        let opening = self.geometry.motor_rad_to_fraction(state.position);
        match &self.drive {
            Drive::Mit(_) => WireState {
                opening,
                effort: 0.0,
                max_effort: 0.0,
            },
            Drive::PosForce { limits, .. } => WireState {
                opening,
                // Motor current-derived torque estimate (N*m at the shaft) in
                // the opening frame: positive toward open on either side.
                effort: self.geometry.motor_torque_to_effort(state.torque),
                max_effort: limits.effort_ceiling_nm(),
            },
        }
    }

    /// The mode-independent bus behind whichever mode this gripper opened in.
    fn bus(&self) -> &dyn ModeBus {
        match &self.drive {
            Drive::Mit(can) => can,
            Drive::PosForce { can, .. } => can,
        }
    }

    fn bus_mut(&mut self) -> &mut dyn ModeBus {
        match &mut self.drive {
            Drive::Mit(can) => can,
            Drive::PosForce { can, .. } => can,
        }
    }

    /// Snapshot from the most recent [`recv_all`](Self::recv_all).
    pub fn get_state(&self) -> GripperState {
        self.bus().get_state()
    }

    pub fn recv_all(&mut self, first_timeout_us: u32) -> Result<(), CanError> {
        self.bus_mut().recv_all(first_timeout_us)
    }

    pub fn refresh_all(&mut self) -> Result<(), CanError> {
        self.bus_mut().refresh_all()
    }

    pub fn reenable(&mut self) -> Result<(), CanError> {
        self.bus_mut().reenable()
    }

    pub fn disable_all(&mut self) -> Result<(), CanError> {
        self.bus_mut().disable_all()
    }

    pub fn enable_and_confirm(
        &mut self,
        attempts: NonZeroU32,
        settle: Duration,
        recv_timeout_us: u32,
    ) -> Result<(), EnableFailure> {
        self.bus_mut()
            .enable_and_confirm(attempts, settle, recv_timeout_us)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The DM4310 both generations drive, resolved the way `Gripper` does.
    fn full_scale_nm() -> f64 {
        torque_full_scale_nm(v20::GRIPPER_MOTOR_TYPE)
    }

    /// Both generations drive this motor.
    const MOTOR: MotorType = v20::GRIPPER_MOTOR_TYPE;

    fn limits(speed_rad_s: f64, max_effort_nm: f64) -> PosForceLimits {
        PosForceLimits::new(MOTOR, speed_rad_s, max_effort_nm).expect("valid limits")
    }

    /// The motor's datasheet peak: the largest ceiling `new` accepts.
    fn peak_nm() -> f64 {
        ratings_of(MOTOR)
            .expect("the gripper motor is rated")
            .peak_nm()
    }

    /// The smallest ceiling the torque-current field can express, in N*m.
    fn smallest_nm() -> f64 {
        WIRE_TORQUE_PU_TICK * full_scale_nm()
    }

    fn cap(nm: f64) -> Option<MaxEffortNm> {
        MaxEffortNm::parse(nm).expect("valid cap")
    }

    #[test]
    fn both_generations_drive_the_same_motor_and_full_scale() {
        assert_eq!(v10::GRIPPER_MOTOR_TYPE, v20::GRIPPER_MOTOR_TYPE);
        assert_eq!(full_scale_nm(), 10.0);
    }

    #[test]
    fn limits_reject_values_the_wire_cannot_carry() {
        for speed in [
            0.0,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            WIRE_MAX_SPEED_RAD_S + 1.0,
        ] {
            assert!(
                PosForceLimits::new(MOTOR, speed, 1.5).is_err(),
                "speed {speed}"
            );
        }
        for force in [-1.0, 11.0, 9.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                PosForceLimits::new(MOTOR, 25.0, force).is_err(),
                "force {force} N*m"
            );
        }
        // The wire's own speed bounds are legal, as is the deliverable ceiling.
        assert!(PosForceLimits::new(MOTOR, WIRE_MAX_SPEED_RAD_S, peak_nm()).is_ok());
        assert!(PosForceLimits::new(MOTOR, WIRE_MIN_SPEED_RAD_S, smallest_nm()).is_ok());
    }

    #[test]
    fn a_ceiling_the_motor_cannot_deliver_is_refused() {
        // The torque-current field spans 10 N*m of quantization headroom, but
        // the DM4310's datasheet peak is 7 N*m, so the top 30% of the field
        // would advertise a grip force the hardware cannot reach.
        assert_eq!(full_scale_nm(), 10.0);
        assert_eq!(peak_nm(), 7.0);
        assert!(PosForceLimits::new(MOTOR, 25.0, full_scale_nm()).is_err());
        assert!(PosForceLimits::new(MOTOR, 25.0, peak_nm() + 1e-9).is_err());
        assert!(PosForceLimits::new(MOTOR, 25.0, peak_nm()).is_ok());
    }

    #[test]
    fn the_published_ceiling_never_exceeds_the_datasheet_peak() {
        let published = limits(25.0, peak_nm()).effort_ceiling_nm();
        assert!(
            published <= peak_nm(),
            "published ceiling {published} N*m exceeds the {} N*m peak",
            peak_nm()
        );
    }

    #[test]
    fn the_slowest_accepted_speed_survives_the_wire_encoding() {
        // pos_force_frame encodes rad/s in hundredths with a truncating cast,
        // so the floor must not itself round down to a full stop.
        let encoded = (WIRE_MIN_SPEED_RAD_S * 100.0) as u16;
        assert!(encoded > 0, "the slowest accepted speed encodes as a stop");
    }

    #[test]
    fn a_ceiling_the_wire_would_truncate_away_is_refused() {
        // Zero, and everything the torque-current field rounds down to zero,
        // would grip nothing while publishing the contract's "no effort
        // control". Likewise a speed the field rounds down to a full stop.
        for force in [0.0, f64::MIN_POSITIVE, 5e-324, smallest_nm() / 2.0] {
            assert!(
                PosForceLimits::new(MOTOR, 25.0, force).is_err(),
                "force {force} N*m"
            );
        }
        for speed in [0.0, f64::MIN_POSITIVE, WIRE_MIN_SPEED_RAD_S / 2.0] {
            assert!(
                PosForceLimits::new(MOTOR, speed, 1.5).is_err(),
                "speed {speed}"
            );
        }
    }

    #[test]
    fn a_commanded_cap_is_parsed_at_the_wire_boundary() {
        assert!(MaxEffortNm::parse(f64::NAN).is_err());
        assert!(MaxEffortNm::parse(f64::INFINITY).is_err());
        assert!(MaxEffortNm::parse(f64::NEG_INFINITY).is_err());
        assert!(MaxEffortNm::parse(-1.0).is_err());
        // The wire's 0 is absence, not a cap of zero.
        assert!(MaxEffortNm::parse(0.0).expect("0 is legal").is_none());
        assert!(MaxEffortNm::parse(1.5).expect("1.5 is legal").is_some());
    }

    #[test]
    fn the_published_ceiling_is_the_configured_one() {
        // No conversion on the way out: what the launcher names in N*m is
        // exactly what gripper_states reports.
        assert_eq!(limits(25.0, 7.0).effort_ceiling_nm(), 7.0);
        assert_eq!(limits(25.0, 2.0).effort_ceiling_nm(), 2.0);
    }

    #[test]
    fn commanded_effort_converts_and_respects_the_ceiling() {
        let limits = limits(25.0, 3.0);
        // No preference: the configured ceiling applies, as its wire fraction.
        assert_eq!(limits.torque_pu(None), 0.3);
        // Within the ceiling: the one N*m to per-unit conversion.
        assert_eq!(limits.torque_pu(cap(1.5)), 0.15);
        // Above the ceiling: the ceiling wins.
        assert_eq!(limits.torque_pu(cap(5.0)), 0.3);
    }

    #[test]
    fn every_reachable_torque_cap_survives_the_wire_encoding() {
        let limits = limits(25.0, 3.0);
        // Spans the subnormal floor, the one-tick boundary, and saturation.
        for nm in [
            5e-324,
            f64::MIN_POSITIVE,
            5e-4,
            1e-3,
            2e-3,
            1.5,
            5.0,
            1e9,
            f64::MAX,
        ] {
            let pu = limits.torque_pu(cap(nm));
            assert!(
                (WIRE_TORQUE_PU_TICK..=1.0).contains(&pu),
                "cap {nm} produced a torque {pu} outside the wire range"
            );
            // What pos_force_frame puts on the wire: a truncating cast to
            // ten-thousandths. A positive cap must never encode as zero.
            let encoded = (pu * 10_000.0) as u16;
            assert!(encoded > 0, "cap {nm} encoded as a cap of zero");
        }
        // No preference still means the configured ceiling, not a floor.
        assert_eq!(limits.torque_pu(None), 0.3);
    }
}
