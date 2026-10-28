//! The panel's state: [`UiState`], which the owner task holds outright (see
//! [`crate::owner`]), with the per-side targets, the governor and jog
//! settings, gesture playback and the recorder panel, and the parsed reports
//! the feeds hand the owner (motor health, alerts, proximity).

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use srs_model::JointVec;

use crate::gestures::BakedGesture;
use crate::pose::{JogCaps, JointJog};

pub use control_core::motor_health::{AlertSeverity, HealthLevel};
use openarm_description::{ARM_DOF, ArmPair, Side};
use peppylib::messaging::ProducerRef;
// The gripper axis is the unitless opening fraction (0 = closed, 1 = open);
// this is only the startup default for the gripper target.
const GRIPPER_CLOSED: f64 = 0.0;

#[derive(Clone, Debug)]
pub struct ArmTarget {
    pub joints: JointVec,
    pub last_feedback: Option<JointVec>,
    // Whether `joints` has been initialized from a real measured pose yet. Set once,
    // from the first joint_states feedback, so the target starts where the arm is
    // instead of at the home default; thereafter only streaming and discrete moves
    // move it. Prevents re-seeding the gravity-sagged measured every disable, which
    // ratcheted the arm down across enable/disable cycles.
    pub established: bool,
    pub in_flight: bool,
    // Cancels the in-flight goal so a new move preempts instead of being
    // rejected by the arm's single-flight gate.
    pub preempt: Option<tokio_util::sync::CancellationToken>,
    // The joint target the operator is driving this side toward: the setpoint ramps
    // toward it under the jog's velocity and acceleration caps. None when the side is
    // idle; it clears on enable/disable.
    pub jog: Option<JointJog>,
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
        }
    }
}

#[derive(Clone, Debug)]
pub struct GripperTarget {
    pub opening: f64,
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
            opening: GRIPPER_CLOSED,
            last_feedback: None,
            max_effort: None,
            effort_ceiling: None,
            in_flight: false,
        }
    }
}

/// A limb of one side: its arm or its gripper.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limb {
    Arm,
    Gripper,
}

impl fmt::Display for Limb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Arm => "arm",
            Self::Gripper => "gripper",
        })
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
        from: ArmPair<Option<JointVec>>,
        gripper_from: ArmPair<Option<f64>>,
        t: f64,
        duration_s: f64,
    },
    Playing {
        t: f64,
    },
}

#[derive(Clone, Debug)]
pub struct UiState {
    pub arms: ArmPair<ArmTarget>,
    pub grippers: ArmPair<GripperTarget>,
    // The gesture being played, if any. Playback owns its involved sides: their
    // deadmen stay off, discrete fires are refused, and the player writes their
    // arm/gripper targets each tick.
    pub gesture: Option<GesturePlayback>,
    // Streaming deadman, one per side: while false the commander emits no
    // commands for that side's arm or gripper. The arm and gripper share the
    // deadman because the operator enables a whole side at once.
    pub enabled: ArmPair<bool>,
    // Operator controls for the backbone's self-collision governor, streamed to the
    // backbone on governor_control; the backbone holds its own defaults until the first
    // message. All five launch defaults are node parameters, kept in step with the
    // backbone's, so a deployment tunes startup from one place; the operator then drives
    // them live from the UI.
    pub governor: GovernorSettings,
    // Joint-slider jog feel, node parameters so a deployment tunes the ramp without a
    // rebuild: the acceleration the streamed target ramps toward the slider under, and the
    // speed it is held at or under. The backbone still governs the final ramp.
    pub jog_caps: JogCaps,
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
    pub arm_health: ArmPair<Option<ArmHealth>>,
    // Latest per-gripper motor health, keyed, retained, and aged the same way.
    pub gripper_health: ArmPair<Option<GripperHealth>>,
    // When this state was created; the startup grace for components that have
    // never reported counts from here.
    pub created_at: Instant,
    // Whether this deployment binds any producer to the motor_health slot.
    // The slot is zero_or_more, so a stack that wired nothing receives
    // nothing and would otherwise render exactly like a healthy robot with
    // nothing to say. Resolved once at startup, as the recorder panel is.
    pub health_bound: bool,
    // Same question for the alerts slot, so an empty alert list can say
    // "not wired" rather than implying all clear. Follows the slot's bound
    // set, which grows when a producer joins after startup.
    pub alerts_bound: bool,
    // The active set each bound producer last published. A producer's next
    // set replaces its whole entry, one that leaves the slot takes its
    // entry with it, and one that stops refreshing ages out.
    pub alerts: BTreeMap<ProducerRef, AlertSet>,
    pub status: String,
}

/// The operator's self-collision governor controls. The command stream
/// publishes them on governor_control when the operator moves one; the topic
/// retains the newest message, so a backbone that starts later reads the
/// controls in force without the operator touching anything.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct GovernorSettings {
    pub collision_enabled: bool,
    pub d_stop: f64,
    pub d_safe: f64,
    pub max_ee_velocity_m_s: f64,
    pub max_gripper_rate_frac_s: f64,
}

impl GovernorSettings {
    /// Whether the panel may stream these controls: a valid band and positive,
    /// finite rates. The backbone validates again before applying.
    pub fn is_streamable(&self) -> bool {
        valid_governor_band(self.d_stop, self.d_safe)
            && [self.max_ee_velocity_m_s, self.max_gripper_rate_frac_s]
                .iter()
                .all(|v| v.is_finite() && *v > 0.0)
    }

    /// Whether this is the same setting as `other`, compared on the bits of
    /// each control so the answer is total: a non-finite value the UI should
    /// never send would otherwise compare unequal to itself and publish the
    /// same setting at the frame rate.
    pub fn same_setting(&self, other: &Self) -> bool {
        self.collision_enabled == other.collision_enabled
            && self.d_stop.to_bits() == other.d_stop.to_bits()
            && self.d_safe.to_bits() == other.d_safe.to_bits()
            && self.max_ee_velocity_m_s.to_bits() == other.max_ee_velocity_m_s.to_bits()
            && self.max_gripper_rate_frac_s.to_bits() == other.max_gripper_rate_frac_s.to_bits()
    }
}

/// A governor band the panel and the backbone both accept: finite, with
/// 0 < d_stop < d_safe.
pub fn valid_governor_band(d_stop: f64, d_safe: f64) -> bool {
    d_stop.is_finite() && d_safe.is_finite() && d_stop > 0.0 && d_stop < d_safe
}

/// A report's receipt time plus the fixed window it stays renderable for.
/// A report older than its window has a quiet producer and drops.
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

    pub fn is_live_at(self, now: Instant) -> bool {
        now.duration_since(self.received_at) < self.live_for
    }
}

/// One operator alert as its producer listed it.
///
/// The producer is not a field: a set belongs to the producer it is filed
/// under, which is the transport-authenticated instance rather than a wire
/// string, so no producer can replace or clear another's alert.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Alert {
    pub source: String,
    pub severity: AlertSeverity,
    pub message: String,
}

/// One producer's active set, with the window it stays renderable for.
#[derive(Clone, Debug)]
pub struct AlertSet {
    pub alerts: Vec<Alert>,
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
    pub fn new(governor: GovernorSettings, jog_caps: JogCaps) -> Self {
        Self {
            arms: ArmPair::from_fn(|_| ArmTarget::home()),
            grippers: ArmPair::from_fn(|_| GripperTarget::closed()),
            gesture: None,
            enabled: ArmPair::new(false, false),
            governor,
            collision_enabled_default: governor.collision_enabled,
            jog_caps,
            proximity: None,
            recorder: RecorderState::unavailable(),
            arm_health: ArmPair::new(None, None),
            gripper_health: ArmPair::new(None, None),
            created_at: Instant::now(),
            health_bound: false,
            alerts_bound: false,
            alerts: BTreeMap::new(),
            status: "ready".to_string(),
        }
    }

    /// Replaces `producer`'s set with the one it published.
    pub fn apply_alert_set(&mut self, producer: ProducerRef, set: AlertSet) {
        self.alerts.insert(producer, set);
    }

    /// Keeps the sets of `producers`, the ones the alerts slot holds, and
    /// records whether the slot holds any producer at all.
    pub fn retain_alert_producers(&mut self, producers: &[ProducerRef]) {
        self.alerts
            .retain(|producer, _| producers.contains(producer));
        self.alerts_bound = !producers.is_empty();
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
    use crate::consumer::HEALTH_STALE_AFTER;
    use crate::test_support::reading;

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
    fn a_report_is_live_strictly_inside_its_window() {
        let t0 = Instant::now();
        let validity = Validity::new(t0, HEALTH_STALE_AFTER);
        assert!(validity.is_live_at(t0));
        assert!(validity.is_live_at(t0 + HEALTH_STALE_AFTER - Duration::from_millis(1)));
        assert!(!validity.is_live_at(t0 + HEALTH_STALE_AFTER));
    }

    #[test]
    fn worst_reports_the_most_severe_motor() {
        let mut readings = [reading(HealthLevel::Nominal); ARM_DOF];
        readings[4] = reading(HealthLevel::Critical);
        let report = HealthReport {
            readings,
            validity: Validity::new(Instant::now(), HEALTH_STALE_AFTER),
        };
        assert_eq!(report.worst(), HealthLevel::Critical);
    }

    #[test]
    fn a_governor_setting_compares_equal_to_itself_including_a_non_finite_one() {
        // The publisher sends only on a change, so a control that compared
        // unequal to itself would publish the setting in force at the frame
        // rate on a reliable topic.
        let setting = GovernorSettings {
            collision_enabled: true,
            d_stop: 0.005,
            d_safe: 0.02,
            max_ee_velocity_m_s: 0.25,
            max_gripper_rate_frac_s: 6.0,
        };
        assert!(setting.same_setting(&setting));

        let nan = GovernorSettings {
            d_stop: f64::NAN,
            ..setting
        };
        assert!(
            nan.same_setting(&nan),
            "a non-finite control is still the same setting as itself"
        );
        assert!(!nan.same_setting(&setting));

        for moved in [
            GovernorSettings {
                collision_enabled: false,
                ..setting
            },
            GovernorSettings {
                d_stop: 0.006,
                ..setting
            },
            GovernorSettings {
                d_safe: 0.03,
                ..setting
            },
            GovernorSettings {
                max_ee_velocity_m_s: 0.3,
                ..setting
            },
            GovernorSettings {
                max_gripper_rate_frac_s: 7.0,
                ..setting
            },
        ] {
            assert!(
                !setting.same_setting(&moved),
                "every control is compared: {moved:?}"
            );
        }
    }

    #[test]
    fn valid_governor_band_boundaries() {
        assert!(valid_governor_band(0.005, 0.02));
        assert!(
            !valid_governor_band(0.02, 0.02),
            "d_stop == d_safe is degenerate"
        );
        assert!(
            !valid_governor_band(0.03, 0.02),
            "d_stop > d_safe is inverted"
        );
        assert!(!valid_governor_band(0.0, 0.02), "non-positive d_stop");
        assert!(!valid_governor_band(f64::NAN, 0.02), "non-finite d_stop");
        let settings = GovernorSettings {
            collision_enabled: true,
            d_stop: 0.005,
            d_safe: 0.02,
            max_ee_velocity_m_s: 1.0,
            max_gripper_rate_frac_s: 6.0,
        };
        assert!(settings.is_streamable());
        assert!(
            !GovernorSettings {
                max_ee_velocity_m_s: 0.0,
                ..settings
            }
            .is_streamable(),
            "non-positive speed cap"
        );
    }
}
