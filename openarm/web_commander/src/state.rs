use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use crate::gestures::BakedGesture;
use crate::pose::Jog;

pub use control_core::motor_health::HealthLevel;
pub use openarm_description::ARM_DOF;
// The gripper axis is the unitless opening fraction (0 = closed, 1 = open);
// this is only the startup default for the gripper target.
pub const GRIPPER_CLOSED: f64 = 0.0;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    /// The wire `arm_name` limb_motion goals carry for this side.
    pub fn arm_name(self) -> &'static str {
        match self {
            Self::Left => "left_arm",
            Self::Right => "right_arm",
        }
    }

    /// The wire `gripper_name` limb_motion goals carry for this side.
    pub fn gripper_name(self) -> &'static str {
        match self {
            Self::Left => "left_gripper",
            Self::Right => "right_gripper",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
        }
    }

    /// The description crate's side selector, for its per-side lookups in the embedded
    /// URDF (the arm chain base link).
    pub fn description(self) -> openarm_description::Side {
        match self {
            Self::Left => openarm_description::Side::Left,
            Self::Right => openarm_description::Side::Right,
        }
    }
}

/// Both sides in a fixed iteration order, for per-side loops.
pub const SIDES: [Side; 2] = [Side::Left, Side::Right];

/// A value stored per side, indexed by [`Side`]: `things[side]` reads or writes the
/// right one, with no left/right accessor split. `Copy` when `T` is, so the small
/// per-tick frames pass by value.
#[derive(Clone, Copy, Debug)]
pub struct BySide<T>([T; 2]);

impl<T> BySide<T> {
    pub const fn new(left: T, right: T) -> Self {
        Self([left, right])
    }
}

impl<T: Clone> BySide<T> {
    pub fn splat(value: T) -> Self {
        Self([value.clone(), value])
    }
}

impl<T> std::ops::Index<Side> for BySide<T> {
    type Output = T;
    fn index(&self, side: Side) -> &T {
        &self.0[side as usize]
    }
}

impl<T> std::ops::IndexMut<Side> for BySide<T> {
    fn index_mut(&mut self, side: Side) -> &mut T {
        &mut self.0[side as usize]
    }
}

#[derive(Clone, Debug)]
pub struct ArmTarget {
    pub joints: [f64; ARM_DOF],
    pub last_feedback: Option<[f64; ARM_DOF]>,
    // Whether `joints` has been initialized from a real measured pose yet. Set once,
    // from the first arm_states feedback, so the target starts where the arm is
    // instead of at the home default; thereafter only streaming and discrete moves
    // move it. Prevents re-seeding the gravity-sagged measured every disable, which
    // ratcheted the arm down across enable/disable cycles.
    pub established: bool,
    pub in_flight: bool,
    // Cancels the in-flight goal so a new Send preempts instead of being
    // rejected by the arm's single-flight gate.
    pub preempt: Option<tokio_util::sync::CancellationToken>,
    // What the operator is actively driving this side toward: a joint target (the setpoint
    // ramps toward it under a velocity/acceleration cap) or a Cartesian jog (stepped toward
    // a world pose one capped increment per tick, held at the reach boundary). None when the
    // side is idle. Arming either space clears the other, and it clears on enable/disable,
    // since the two spaces must not fight.
    pub jog: Option<Jog>,
    // Whether a Cartesian jog is currently held at the reach boundary. Drives one-shot
    // status transitions (blocked <-> moving), so neither message latches or spams.
    pub jog_blocked: bool,
}

impl ArmTarget {
    pub fn home() -> Self {
        Self {
            joints: [0.0; ARM_DOF],
            last_feedback: None,
            established: false,
            in_flight: false,
            preempt: None,
            jog: None,
            jog_blocked: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GripperTarget {
    pub position: f64,
    // Measured gripper opening fraction from the gripper_states stream.
    pub last_feedback: Option<f64>,
    // The operator's effort cap (the gripper's effort unit), sent with both the
    // streamed opening and discrete moves. `None` = no preference (the wire's 0):
    // the gripper's configured ceiling stays in charge.
    pub max_effort: Option<f64>,
    // The gripper's reported effort ceiling from gripper_states: `None` until
    // the first report, 0 = the gripper has no effort control (hides the
    // panel's effort slider).
    pub effort_ceiling: Option<f64>,
    // A discrete move_gripper (Actions mode) is in flight: refuses a second Execute
    // and drives the gripper card's in-flight badge. Streaming mode never sets it.
    pub in_flight: bool,
}

impl GripperTarget {
    pub fn closed() -> Self {
        Self {
            position: GRIPPER_CLOSED,
            last_feedback: None,
            max_effort: None,
            effort_ceiling: None,
            in_flight: false,
        }
    }
}

/// Dataset recorder panel state. `available` is resolved once at startup from
/// the recorder slot's bound producers (the slot is zero_or_more, so a
/// deployment without a recorder hides the panel); `episode` is `Some` while a
/// record goal is in flight.
#[derive(Clone, Debug)]
pub struct RecorderState {
    pub available: bool,
    pub episode: Option<RecordingEpisode>,
    // A finish_session call is in flight (finalize + mirror of the current
    // dataset); gates the panel's Finish button.
    pub finishing: bool,
}

/// One in-flight record_episode goal: the live frame count from its feedback
/// stream and the token that stops it (cancel = stop and save).
#[derive(Clone, Debug)]
pub struct RecordingEpisode {
    pub frames: u64,
    pub stop: tokio_util::sync::CancellationToken,
}

impl RecorderState {
    pub fn unavailable() -> Self {
        Self {
            available: false,
            episode: None,
            finishing: false,
        }
    }
}

/// A gesture in flight: the baked trajectory plus where playback is. `Arc`
/// keeps the per-tick playback advance cheap (each step re-clones the handle,
/// never the trajectory).
#[derive(Clone, Debug)]
pub struct GesturePlayback {
    pub gesture: Arc<BakedGesture>,
    pub phase: GesturePhase,
}

/// Playback phase: a quintic blend from the pose held at start (the retained
/// target, which the backbone is already holding) to each involved track's
/// first sample, then the baked trajectory on a shared clock. `gripper_from`
/// carries the opening measured at start, so a gesture that drives the jaw
/// eases it in over the same blend.
#[derive(Clone, Copy, Debug)]
pub enum GesturePhase {
    LeadIn {
        from: BySide<Option<[f64; ARM_DOF]>>,
        gripper_from: BySide<Option<f64>>,
        t: f64,
        duration_s: f64,
    },
    Playing {
        t: f64,
    },
}

#[derive(Clone, Debug)]
pub struct UiState {
    pub arms: BySide<ArmTarget>,
    pub grippers: BySide<GripperTarget>,
    // The gesture being played, if any. Playback owns its involved sides: their
    // deadmen stay off, discrete fires are refused, and the player writes their
    // arm/gripper targets each tick.
    pub gesture: Option<GesturePlayback>,
    // Streaming deadman, one per side: while false the commander emits no
    // commands for that side's arm or gripper and both targets track the measured
    // pose, so enabling never steps the robot. The arm and gripper share the
    // deadman because the operator enables a whole side at once.
    pub enabled: BySide<bool>,
    // Operator controls for the backbone's self-collision governor, streamed to the
    // backbone on governor_control; the backbone holds its own defaults until the first
    // message. All four launch defaults are node parameters, kept in step with the
    // backbone's, so a deployment tunes startup from one place; the operator then drives
    // them live from the UI.
    pub collision_enabled: bool,
    pub d_stop: f64,
    pub d_safe: f64,
    pub max_ee_velocity_m_s: f64,
    pub max_gripper_rate_frac_s: f64,
    // Joint-slider jog feel, a node parameter so a deployment tunes the ramp without a
    // rebuild: the acceleration the streamed target ramps toward the slider under (the
    // whole jog is acceleration-limited). The backbone still governs the final ramp.
    pub joint_jog_acceleration_rad_s2: f64,
    // The launch governor-enable state, restored on operator disconnect so an
    // operator who turned avoidance off cannot leave the backbone latched ungoverned,
    // while a deployment that launched ungoverned is not force-armed either.
    pub collision_enabled_default: bool,
    // Latest nearest-pair self-collision proximity from the backbone (it carries its own
    // receipt time). `None` until the first message; treated as stale (and rendered
    // n/a) once that receipt time ages past the readout staleness window, so a dead
    // backbone does not leave the last distance latched on the panel.
    pub proximity: Option<Proximity>,
    // The dataset recorder panel; hidden entirely when the deployment binds no
    // recorder instance.
    pub recorder: RecorderState,
    // Latest per-arm motor health: `None` until that side's first report and
    // retained thereafter. The retained entry doubles as the has-ever-reported
    // bit: one aged past its validity renders as not reporting, never as a
    // fresh launch still inside its grace.
    pub health: BySide<Option<ArmHealth>>,
    // Latest per-gripper motor health, keyed, retained, and aged the same way.
    pub gripper_health: BySide<Option<GripperHealth>>,
    // When this state was created; the startup grace for components that have
    // never reported counts from here.
    pub created_at: Instant,
    // Whether this deployment binds any producer to the motor_health slot.
    // The slot is zero_or_more, so a stack that wired nothing receives
    // nothing and would otherwise render exactly like a healthy robot with
    // nothing to say. Resolved once at startup, as the recorder panel is.
    pub health_bound: bool,
    // Same question for the alerts slot, so an empty alert list can say
    // "not wired" rather than implying all clear.
    pub alerts_bound: bool,
    // Active operator alerts, one per (source, kind); severity-0 messages
    // remove theirs, and the view ages out entries whose producer went quiet.
    pub alerts: Vec<Alert>,
    pub status: String,
}

/// How often a listener may warn about a producer it is rejecting: enough to
/// notice a persistently malformed producer, not enough to bury the log at
/// that producer's own rate.
pub const REJECT_WARN_PERIOD: Duration = Duration::from_secs(1);

/// The motor_health contract's producer cadence mandate: every producer
/// reports each component at least this often.
const HEALTH_REPORT_PERIOD: Duration = Duration::from_millis(500);

/// The alerts contract's re-emit ceiling: a producer re-announces each
/// active alert at least this often.
const ALERT_REEMIT_PERIOD: Duration = Duration::from_millis(2000);

/// Health reports age out three contract periods (1500 ms) after receipt:
/// one missed report is transport jitter, three in a row is a producer gone
/// quiet, and the panel falls to not-reporting rather than latching.
pub const HEALTH_STALE_AFTER: Duration = HEALTH_REPORT_PERIOD.saturating_mul(3);

/// Alerts age out three re-emit periods (6000 ms) after receipt, the same
/// three-missed-periods patience as [`HEALTH_STALE_AFTER`].
pub const ALERT_STALE_AFTER: Duration = ALERT_REEMIT_PERIOD.saturating_mul(3);

/// Slack allowed between a wire timestamp and this consumer's clock before the
/// timestamp counts as pre-aged. Producers timestamp from the daemon-resolved clock
/// this node also reads, so the allowance absorbs scheduling skew, not clock
/// disagreement.
pub const TIMESTAMP_SKEW_ALLOWANCE: Duration = Duration::from_millis(500);

/// A report's receipt time plus the fixed window it stays renderable for:
/// the one aging rule shared by health reports and alerts. Anything older
/// than its window stopped being re-emitted and drops instead of latching.
#[derive(Clone, Copy, Debug)]
pub struct Validity {
    received_at: Instant,
    live_for: Duration,
}

impl Validity {
    pub fn new(received_at: Instant, live_for: Duration) -> Self {
        Self {
            received_at,
            live_for,
        }
    }

    pub fn received_at(self) -> Instant {
        self.received_at
    }

    pub fn is_live_at(self, now: Instant) -> bool {
        now.duration_since(self.received_at) < self.live_for
    }
}

/// Parse a wire timestamp into a report's [`Validity`], rejecting a timestamp already
/// older than the aging window plus [`TIMESTAMP_SKEW_ALLOWANCE`] on the
/// daemon-resolved clock both ends read. A consumer working through a backlog
/// re-stamps queued reports as fresh at receipt; the timestamp check refuses what is already older than the window.
pub fn parse_timestamp_validity(
    timestamp: SystemTime,
    clock_now: SystemTime,
    received_at: Instant,
    live_for: Duration,
) -> Result<Validity, String> {
    // A timestamp ahead of the clock reads as age zero: forward skew is not backlog.
    let age = clock_now
        .duration_since(timestamp)
        .unwrap_or(Duration::ZERO);
    if age > live_for + TIMESTAMP_SKEW_ALLOWANCE {
        return Err(format!(
            "timestamp {} ms old is past its {} ms window",
            age.as_millis(),
            live_for.as_millis()
        ));
    }
    Ok(Validity::new(received_at, live_for))
}

/// One operator alert, identified by (producer, source, kind): the producer
/// raises it with a non-zero severity, re-emits actives inside the
/// contract's re-emit ceiling, and retires it with severity 0. One not
/// re-emitted within [`ALERT_STALE_AFTER`] has a quiet producer and drops
/// instead of latching. The producer is part of the identity because source
/// and kind are wire strings: without it, one producer could replace or
/// clear another's alert.
#[derive(Clone, Debug)]
pub struct Alert {
    /// The transport-authenticated producing instance, not a wire string.
    pub producer: String,
    pub source: String,
    pub kind: String,
    pub severity: u8,
    pub message: String,
    pub validity: Validity,
}

/// One motor's health as the panel renders it: the producer-evaluated level
/// plus the readings behind it, each `None` when the producer senses nothing
/// (a sim limb reports levels only).
///
/// The torque fractions divide by different lines: `effort_fraction_rated`
/// and `effort_fraction_rated_sustained` (the filtered value the level
/// judges) are of the continuous thermal rating; `effort_fraction_peak` is
/// of the effective peak, where 1.0 is the cutout line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotorHealthReading {
    pub level: HealthLevel,
    pub effort_fraction_rated: Option<f64>,
    pub effort_fraction_rated_sustained: Option<f64>,
    pub effort_fraction_peak: Option<f64>,
    pub driver_temp_c: Option<f64>,
    pub winding_temp_c: Option<f64>,
}

/// One component's parsed health report: a reading per motor (transposed from
/// the wire's struct-of-arrays at the parse boundary) plus the receipt
/// validity that ages it.
#[derive(Clone, Debug)]
pub struct HealthReport<const MOTORS: usize> {
    pub readings: [MotorHealthReading; MOTORS],
    pub validity: Validity,
}

/// A seven-joint arm's report.
pub type ArmHealth = HealthReport<ARM_DOF>;
/// A gripper's single-motor report.
pub type GripperHealth = HealthReport<1>;

impl<const MOTORS: usize> HealthReport<MOTORS> {
    /// The most severe motor's level.
    pub fn worst(&self) -> HealthLevel {
        const {
            assert!(MOTORS > 0, "a health report covers at least one motor");
        }
        self.readings
            .iter()
            .map(|reading| reading.level)
            .max()
            .expect("MOTORS > 0")
    }
}

/// The backbone's reported nearest checked pair: signed surface distance (m, positive
/// is clearance), the two link names, the governor's disposition of the commanded
/// motion, and the local time it was received (for the readout's staleness check).
#[derive(Clone, Debug)]
pub struct Proximity {
    pub distance: f64,
    pub link_a: String,
    pub link_b: String,
    pub disposition: Disposition,
    pub received_at: Instant,
}

/// The governor's disposition of the commanded motion, parsed from the wire's
/// mutually exclusive booleans. Stopped wins if a producer ever set both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Disposition {
    Clear,
    Throttled,
    Stopped,
}

impl Disposition {
    pub fn from_wire(throttled: bool, stopped: bool) -> Self {
        match (throttled, stopped) {
            (_, true) => Self::Stopped,
            (true, false) => Self::Throttled,
            (false, false) => Self::Clear,
        }
    }
}

impl UiState {
    pub fn new(
        collision_enabled: bool,
        d_stop: f64,
        d_safe: f64,
        max_ee_velocity_m_s: f64,
        max_gripper_rate_frac_s: f64,
        joint_jog_acceleration_rad_s2: f64,
    ) -> Self {
        Self {
            arms: BySide::splat(ArmTarget::home()),
            grippers: BySide::splat(GripperTarget::closed()),
            gesture: None,
            enabled: BySide::splat(false),
            collision_enabled,
            collision_enabled_default: collision_enabled,
            d_stop,
            d_safe,
            max_ee_velocity_m_s,
            max_gripper_rate_frac_s,
            joint_jog_acceleration_rad_s2,
            proximity: None,
            recorder: RecorderState::unavailable(),
            health: BySide::splat(None),
            gripper_health: BySide::splat(None),
            created_at: Instant::now(),
            health_bound: false,
            alerts_bound: false,
            alerts: Vec::new(),
            status: "ready".to_string(),
        }
    }

    /// Fold one received alert in: replace the (producer, source, kind)
    /// entry, or remove it on a severity-0 clear. Entries that outlived
    /// their validity window are purged here too, keyed on the incoming
    /// receipt time, so the list stays bounded even when sources vary.
    pub fn apply_alert(&mut self, alert: Alert) {
        self.alerts.retain(|a| {
            let replaced =
                a.producer == alert.producer && a.source == alert.source && a.kind == alert.kind;
            !replaced && a.validity.is_live_at(alert.validity.received_at())
        });
        if alert.severity > 0 {
            self.alerts.push(alert);
        }
    }

    pub fn set_status(&mut self, message: impl Into<String>) {
        self.status = message.into();
    }

    /// Whether a playing gesture owns this side's targets.
    pub fn gesture_holds(&self, side: Side) -> bool {
        self.gesture
            .as_ref()
            .is_some_and(|p| p.gesture.involves(side))
    }

    /// Whether this side's setpoints should stream: the operator's deadman is on,
    /// or a gesture is driving it.
    pub fn side_active(&self, side: Side) -> bool {
        self.enabled[side] || self.gesture_holds(side)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_defined_level_survives_the_wire_round_trip() {
        // The browser indexes its colour and label tables by this number, so
        // a gap here renders as an unlabelled motor.
        for wire in 0..=4 {
            assert_eq!(HealthLevel::from_wire(wire).unwrap().wire(), wire);
        }
        assert!(HealthLevel::from_wire(5).is_none());
    }

    #[test]
    fn aging_windows_are_three_contract_periods() {
        assert_eq!(HEALTH_STALE_AFTER, Duration::from_millis(1500));
        assert_eq!(ALERT_STALE_AFTER, Duration::from_millis(6000));
    }

    #[test]
    fn a_report_is_live_strictly_inside_its_window() {
        let t0 = Instant::now();
        let validity = Validity::new(t0, HEALTH_STALE_AFTER);
        assert!(validity.is_live_at(t0));
        assert!(validity.is_live_at(t0 + HEALTH_STALE_AFTER - Duration::from_millis(1)));
        assert!(!validity.is_live_at(t0 + HEALTH_STALE_AFTER));
    }

    #[test]
    fn a_pre_aged_timestamp_rejects_and_a_fresh_or_future_one_parses() {
        let clock = SystemTime::now();
        let t0 = Instant::now();
        let parse = |timestamp| parse_timestamp_validity(timestamp, clock, t0, HEALTH_STALE_AFTER);
        assert!(parse(clock).is_ok());
        assert!(
            parse(clock - (HEALTH_STALE_AFTER + TIMESTAMP_SKEW_ALLOWANCE)).is_ok(),
            "exactly at the allowance still parses"
        );
        assert!(
            parse(
                clock - (HEALTH_STALE_AFTER + TIMESTAMP_SKEW_ALLOWANCE + Duration::from_millis(1))
            )
            .is_err(),
            "a backlogged report must not be re-stamped fresh"
        );
        assert!(
            parse(clock + Duration::from_secs(5)).is_ok(),
            "forward skew is not backlog"
        );
    }

    #[test]
    fn worst_reports_the_most_severe_motor() {
        let reading = |level| MotorHealthReading {
            level,
            effort_fraction_rated: None,
            effort_fraction_rated_sustained: None,
            effort_fraction_peak: None,
            driver_temp_c: None,
            winding_temp_c: None,
        };
        let mut readings = [reading(HealthLevel::Nominal); ARM_DOF];
        readings[4] = reading(HealthLevel::Critical);
        let report = HealthReport {
            readings,
            validity: Validity::new(Instant::now(), HEALTH_STALE_AFTER),
        };
        assert_eq!(report.worst(), HealthLevel::Critical);
    }
}
