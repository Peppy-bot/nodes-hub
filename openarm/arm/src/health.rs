//! Motor condition telemetry: the per-joint health filters stepped from each
//! tick's decoded state, the publisher that puts the reports and the operator
//! alerts they imply on the wire, and the bring-up resolution of the ratings
//! the filters judge against. Drive policy (re-enable, hard fault) stays in
//! `control`; this module observes and reports.

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use control_core::motor_health::{
    AlertRaiser, DriverTempC, HEALTH_PERIOD, MOTOR_ALERT_KIND, MotorHealth, MotorHealthFilter,
    MotorSample, Ratings, STATE_STALE_AFTER, TEMP_WINDING_CRIT_C, TEMP_WINDING_WARN_C,
    WindingTempC,
};
use control_core::pacer::Pacer;
use control_core::throttle::Throttle;
use openarm_can::{ArmCan, ArmState, EffectiveRatings, MotorParam};
use peppygen::NodeRunner;
use peppygen::emitted_topics::alerts::alerts as alerts_topic;
use peppygen::emitted_topics::motor_health::motor_health;
use peppylib::runtime::CancellationToken;
use tokio::sync::watch;
use tracing::{error, info, warn};

use crate::ARM_DOF;

/// `ARM_MOTOR_TYPES` below is indexed by this crate's joint numbering; the two
/// crates declare their DOF independently, so a skew would be an
/// out-of-bounds panic mid-bring-up rather than a compile error without this.
const _: () = assert!(openarm_can::ARM_MOTOR_TYPES.len() == ARM_DOF);
use crate::control::ticks_within;
use crate::stream::{LatchedWarn, StreamWiring, capture_timestamp};

/// Axes whose motor reports a full scale the decode table disagrees with.
///
/// Each reading on such an axis is wrong by the ratio of the two scales, so
/// the arm refuses to drive rather than judge torque against them. A variant
/// swap (a 24V-config motor where the lineup expects 48V) is what this catches.
#[derive(Debug, thiserror::Error)]
#[error(
    "refusing to drive on mis-scaled readings: {}; retype the lineup or reconfigure the motor",
    .axes.join("; ")
)]
pub struct MisScaledReadings {
    pub axes: Vec<String>,
}

/// How often the tick loop may repeat a clock-outage warning.
const CLOCK_WARN_PERIOD: Duration = Duration::from_secs(1);

/// One tick's health reports, the instantaneous effort fractions they were
/// judged from (against the continuous rating and against the effective
/// peak), and the clock reading they were taken at. The timestamp travels with
/// them so the publisher can tell a fresh capture from one it already
/// published.
#[derive(Clone, Copy)]
pub struct HealthSample {
    pub reports: [MotorHealth; ARM_DOF],
    pub effort_fractions_rated: [f64; ARM_DOF],
    pub effort_fractions_peak: [f64; ARM_DOF],
    pub captured: SystemTime,
}

/// The per-tick health observer: owns the filters, the staleness edge state,
/// and the clock-warning throttle, so the control loop makes one call per
/// tick and stays motion logic.
pub struct Monitor {
    filters: [MotorHealthFilter; ARM_DOF],
    ratings: [Ratings; ARM_DOF],
    stale_after_ticks: u32,
    stale: [bool; ARM_DOF],
    clock_warn: Throttle,
    last_step: Option<Instant>,
}

impl Monitor {
    pub fn new(ratings: [Ratings; ARM_DOF], cycle_period: Duration) -> Self {
        Self {
            filters: std::array::from_fn(|i| MotorHealthFilter::new(ratings[i])),
            ratings,
            stale_after_ticks: ticks_within(STATE_STALE_AFTER, cycle_period),
            stale: [false; ARM_DOF],
            clock_warn: Throttle::new(CLOCK_WARN_PERIOD),
            last_step: None,
        }
    }

    /// Observe one tick's decoded state: mark which joints have gone stale
    /// (returned for the drive policy), step every filter, and hand the
    /// stamped sample to the publisher. The filter integrates wall time, not
    /// tick count: the pacer re-anchors after an overrun, so ticks are only
    /// nominally periodic, and "the last 5 seconds" must mean seconds on a
    /// stopwatch. A clock outage costs this tick's telemetry and nothing
    /// else; the filters keep integrating.
    pub fn observe(&mut self, state: &ArmState, wiring: &StreamWiring) -> [bool; ARM_DOF] {
        let stale = stale_flags(&state.passes_since_state, self.stale_after_ticks);
        self.log_stale_edges(state, &stale);

        let now = Instant::now();
        let dt_s = self
            .last_step
            .map_or(Duration::ZERO, |last| now - last)
            .as_secs_f64();
        self.last_step = Some(now);

        let reports = judge(&mut self.filters, state, &stale, dt_s);
        let (effort_fractions_rated, effort_fractions_peak) = fractions(&self.ratings, state);
        match capture_timestamp() {
            Ok(captured) => {
                wiring.health.send_replace(Some(HealthSample {
                    reports,
                    effort_fractions_rated,
                    effort_fractions_peak,
                    captured,
                }));
            }
            Err(reason) => {
                if self.clock_warn.admit() {
                    warn!("health sample not published: {reason}");
                }
            }
        }
        stale
    }

    fn log_stale_edges(&mut self, state: &ArmState, stale: &[bool; ARM_DOF]) {
        for (joint, (was, is)) in self.stale.iter().zip(stale).enumerate() {
            match (was, is) {
                (false, true) => warn!(
                    "motor j{} sent no state for {} passes",
                    joint + 1,
                    state.passes_since_state[joint]
                ),
                (true, false) => info!("motor j{} state frames recovered", joint + 1),
                _ => {}
            }
        }
        self.stale = *stale;
    }
}

/// Joints whose state frames stopped arriving: past the stale window the
/// cached reading is not a measurement.
fn stale_flags(passes_since_state: &[u32; ARM_DOF], stale_after_ticks: u32) -> [bool; ARM_DOF] {
    std::array::from_fn(|i| passes_since_state[i] >= stale_after_ticks)
}

/// Step each joint's health filter with this tick's decoded state. A stale
/// joint, one whose status carries no condition, or one whose readings
/// decode non-finite is judged silent: the filter reports it as not
/// reporting and carries the readings it was last measured at, so a joint
/// that goes quiet at 96 C does not render as cold. The non-finite case is
/// judged rather than stepped because there is no measurement in it, and
/// feeding it through would poison the sustained average.
fn judge(
    filters: &mut [MotorHealthFilter; ARM_DOF],
    state: &ArmState,
    stale: &[bool; ARM_DOF],
    dt_s: f64,
) -> [MotorHealth; ARM_DOF] {
    std::array::from_fn(|i| {
        let measurable = state.torques[i].is_finite()
            && state.temps_mos_c[i].is_finite()
            && state.temps_rotor_c[i].is_finite();
        match state.statuses[i].condition() {
            Some(condition) if !stale[i] && measurable => filters[i].step(
                MotorSample {
                    torque_nm: state.torques[i],
                    driver_temp: DriverTempC(state.temps_mos_c[i]),
                    winding_temp: WindingTempC(state.temps_rotor_c[i]),
                    condition,
                },
                dt_s,
            ),
            _ => filters[i].silent(),
        }
    })
}

/// This tick's instantaneous |torque| per joint as fractions of the
/// continuous rating and of the effective peak: the raw samples behind the
/// filter's average, carried so a consumer can show the live numbers beside
/// the sustained one the levels judge. A quiet joint's cached torque is its
/// last measurement, the same silence policy the reports carry.
fn fractions(ratings: &[Ratings; ARM_DOF], state: &ArmState) -> ([f64; ARM_DOF], [f64; ARM_DOF]) {
    let rated = std::array::from_fn(|i| state.torques[i].abs() / ratings[i].rated_nm());
    let peak = std::array::from_fn(|i| state.torques[i].abs() / ratings[i].peak_nm());
    (rated, peak)
}

/// Emit the per-motor health reports at [`HEALTH_PERIOD`], forever, and the
/// operator alerts they imply (transitions as they happen, active alerts on
/// the heartbeat resync).
///
/// A sample the control loop has not refreshed is not republished: the
/// cadence is what lets a consumer tell a steady robot from a dead
/// producer.
///
/// On cancellation the loop runs one final round before returning,
/// flushing the terminal verdict the follow loop wrote before it
/// cancelled.
pub async fn run_publisher(
    runner: Arc<NodeRunner>,
    alert_source: String,
    mut health: watch::Receiver<Option<HealthSample>>,
    token: CancellationToken,
) {
    let publisher = match motor_health::declare_publisher(&runner).await {
        Ok(p) => p,
        Err(e) => return error!("declare motor_health publisher: {e}"),
    };
    let alert_publisher = match alerts_topic::declare_publisher(&runner).await {
        Ok(p) => p,
        Err(e) => return error!("declare alerts publisher: {e}"),
    };
    if health.wait_for(|sample| sample.is_some()).await.is_err() {
        return; // control loop gone before any motor reported
    }
    let mut pacer = Pacer::new(HEALTH_PERIOD).expect("HEALTH_PERIOD is non-zero");
    let sources = (1..=ARM_DOF)
        .map(|j| format!("{alert_source} j{j}"))
        .collect();
    let mut raiser = AlertRaiser::new(sources);
    let mut health_warn = LatchedWarn::new("motor_health publish");
    let mut readings_warn = LatchedWarn::new("health readings");
    let mut alert_warn = LatchedWarn::new("alert publish");
    let mut final_round = false;
    let mut last_published: Option<SystemTime> = None;
    loop {
        let sample = health
            .borrow_and_update()
            .expect("the watch is seeded with a sample before this loop starts");
        // The control loop writes a sample every tick, far faster than this
        // cadence, so an unchanged capture time means it has stopped writing.
        // The final round publishes regardless: its job is flushing the
        // terminal verdict.
        let frozen = last_published == Some(sample.captured) && !final_round;
        if !frozen {
            last_published = Some(sample.captured);
            publish_report(&publisher, &sample, &mut health_warn, &mut readings_warn).await;
            publish_due_alerts(
                &alert_publisher,
                &mut raiser,
                &sample.reports,
                &mut alert_warn,
            )
            .await;
        }

        if final_round {
            return;
        }
        // Biased so a cancelled token wins an already-due tick: shutdown
        // gets its flush round before the process exits.
        tokio::select! {
            biased;
            _ = token.cancelled() => final_round = true,
            _ = pacer.pace() => {
                // The watch keeps its value after the sender drops, so a
                // closed channel still owes one round.
                if health.has_changed().is_err() {
                    final_round = true;
                }
            }
        }
    }
}

/// Publish one telemetry message, warning once per failure burst; a degraded
/// message (levels without readings) additionally warns once per burst,
/// because it means a motor carries a non-finite reading, which bring-up's
/// enable confirmation is supposed to make impossible.
async fn publish_report(
    publisher: &peppylib::TopicPublisher,
    sample: &HealthSample,
    publish_warn: &mut LatchedWarn,
    readings_warn: &mut LatchedWarn,
) {
    let (message, degraded) = match build_health_message(sample) {
        Ok(built) => built,
        Err(e) => return publish_warn.failure(&e),
    };
    match degraded {
        true => readings_warn.failure("a motor carries a non-finite reading"),
        false => readings_warn.success(),
    }
    match publisher.publish(message).await {
        Ok(()) => publish_warn.success(),
        Err(e) => publish_warn.failure(&e.to_string()),
    }
}

/// Publish the alert transitions and heartbeat re-emits these reports owe.
/// Alerts commit per successful publish ([`AlertRaiser::mark_sent`]): a
/// failed send stays owed and retries next round, and the heartbeat
/// commits only when the whole round went out.
async fn publish_due_alerts(
    publisher: &peppylib::TopicPublisher,
    raiser: &mut AlertRaiser,
    reports: &[MotorHealth; ARM_DOF],
    warn: &mut LatchedWarn,
) {
    let now = Instant::now();
    let batch = raiser.due(reports, now);
    let mut all_sent = true;
    for pending in &batch.items {
        let result = async {
            let msg = alerts_topic::build_message(
                capture_timestamp()?,
                pending.alert.source.clone(),
                MOTOR_ALERT_KIND.to_string(),
                pending.alert.severity,
                pending.alert.message.clone(),
            )
            .map_err(|e| e.to_string())?;
            publisher.publish(msg).await.map_err(|e| e.to_string())
        }
        .await;
        match result {
            Ok(()) => {
                raiser.mark_sent(pending);
                warn.success();
            }
            Err(e) => {
                all_sent = false;
                warn.failure(&e);
            }
        }
    }
    if batch.heartbeat && all_sent {
        raiser.mark_heartbeat(now);
    }
}

/// One wire message from a tick's reports, and whether it was degraded to
/// levels-only.
///
/// Levels always publish. The reading vectors publish all together or not at
/// all: the contract has no per-joint absence, so one non-finite value (a
/// motor that never reported, which bring-up refuses) empties every reading
/// vector rather than fabricating a number or silencing the six joints still
/// measured. Their levels still name the quiet motor.
fn build_health_message(sample: &HealthSample) -> Result<(peppylib::Payload, bool), String> {
    let readings = [
        sample.effort_fractions_rated.to_vec(),
        sample.reports.map(|r| r.torque_fraction).to_vec(),
        sample.effort_fractions_peak.to_vec(),
        sample.reports.map(|r| r.driver_temp.0).to_vec(),
        sample.reports.map(|r| r.winding_temp.0).to_vec(),
    ];
    let degraded = readings
        .iter()
        .any(|values| values.iter().any(|v| !v.is_finite()));
    let [rated, sustained, peak, driver, winding] = match degraded {
        true => std::array::from_fn(|_| Vec::new()),
        false => readings,
    };
    let payload = motor_health::build_message(
        sample.captured,
        sample.reports.iter().map(|r| r.level.wire()).collect(),
        rated,
        sustained,
        peak,
        driver,
        winding,
    )
    .map_err(|e| e.to_string())?;
    Ok((payload, degraded))
}

/// The torque ratings the health filters judge each joint against: the
/// datasheet ratings, tightened where a motor is configured to trip below
/// its datasheet peak. A joint whose registers cannot be read keeps its
/// datasheet ratings, which is the looser of the two, so a failed read
/// never tightens a limit on evidence that was not gathered.
///
/// Refuses when any motor's reported full scales disagree with the decode
/// table, naming every mismatched axis: each reading on such an axis is
/// wrong by the ratio of the two scales, and a variant swap (a 24V-config
/// motor where the lineup expects 48V) is exactly what this catches.
pub fn resolve_ratings(
    arm: &mut ArmCan,
    recv_timeout_us: u32,
) -> std::result::Result<[Ratings; ARM_DOF], MisScaledReadings> {
    let reads = read_limit_registers(arm, recv_timeout_us);

    let axes = scale_mismatches(&reads);
    if !axes.is_empty() {
        return Err(MisScaledReadings { axes });
    }

    Ok(std::array::from_fn(|joint| {
        warn_late_temp_trip(joint, reads.over_temp[joint]);
        effective_ratings(joint, reads.over_current[joint], reads.torque_max[joint])
    }))
}

/// The limit registers bring-up reads off every motor. A register that could
/// not be read is `None` for every joint and warns once: a failed read is
/// bus flakiness, not evidence about the motor, and the enable confirmation
/// still gates an unresponsive one.
struct LimitRegisters {
    over_current: [Option<f64>; ARM_DOF],
    over_temp: [Option<f64>; ARM_DOF],
    position_max: [Option<f64>; ARM_DOF],
    velocity_max: [Option<f64>; ARM_DOF],
    torque_max: [Option<f64>; ARM_DOF],
}

fn read_limit_registers(arm: &mut ArmCan, recv_timeout_us: u32) -> LimitRegisters {
    let mut read =
        |param: MotorParam, on_failure: &str| match arm.read_param(param, recv_timeout_us) {
            Ok(values) => values,
            Err(e) => {
                warn!("read {param:?}: {e}; {on_failure}");
                [None; ARM_DOF]
            }
        };
    LimitRegisters {
        over_current: read(MotorParam::OverCurrentLimit, "keeping datasheet ratings"),
        over_temp: read(MotorParam::OverTempLimit, "skipping its trip check"),
        position_max: read(MotorParam::PositionMax, "skipping its scale check"),
        velocity_max: read(MotorParam::VelocityMax, "skipping its scale check"),
        torque_max: read(MotorParam::TorqueMax, "keeping datasheet ratings"),
    }
}

/// Every axis whose reported full scale disagrees with the decode table. The
/// motor quantizes each field against its own full scale, so decoding with a
/// different one scales every reading on that axis.
fn scale_mismatches(reads: &LimitRegisters) -> Vec<String> {
    let mut mismatched = Vec::new();
    for joint in 0..ARM_DOF {
        let model = openarm_can::ARM_MOTOR_TYPES[joint];
        let scale_registers = [
            (
                MotorParam::PositionMax,
                reads.position_max[joint],
                "position",
            ),
            (
                MotorParam::VelocityMax,
                reads.velocity_max[joint],
                "velocity",
            ),
            (MotorParam::TorqueMax, reads.torque_max[joint], "torque"),
        ];
        for (param, reported, axis) in scale_registers {
            let Some(reported) = reported else { continue };
            if model.scale_matches(param, reported) == Some(false) {
                mismatched.push(format!(
                    "j{} reports a {reported} {axis} full scale, decoded as {}",
                    joint + 1,
                    model.decode_full_scale(param).expect("scale register")
                ));
            }
        }
    }
    mismatched
}

/// Check one motor's configured over-temperature trip against the winding
/// thresholds, for the same reason the torque trip is checked: a warning
/// that cannot precede the cutout is not a warning.
fn warn_late_temp_trip(joint: usize, over_temp: Option<f64>) {
    match over_temp {
        Some(trip) if !trip.is_finite() => warn!(
            "motor j{} reported a non-finite over-temperature trip; nothing to check it against",
            joint + 1
        ),
        Some(trip) if trip <= TEMP_WINDING_WARN_C => error!(
            "motor j{} cuts out at {trip:.0} C, at or below the {TEMP_WINDING_WARN_C:.0} C winding warning: no warning will precede an over-temperature cutout",
            joint + 1
        ),
        Some(trip) if trip <= TEMP_WINDING_CRIT_C => warn!(
            "motor j{} cuts out at {trip:.0} C, at or below the {TEMP_WINDING_CRIT_C:.0} C winding critical: it will go limp before being reported critical",
            joint + 1
        ),
        _ => {}
    }
}

/// One joint's effective ratings from its trip registers, logged: datasheet
/// when nothing tightens them, the trip point when the motor is configured
/// below its datasheet peak.
fn effective_ratings(joint: usize, over_current: Option<f64>, torque_max: Option<f64>) -> Ratings {
    let datasheet = openarm_can::ARM_MOTOR_TYPES[joint]
        .ratings()
        .expect("every OpenArm arm motor has datasheet ratings");
    let effective = EffectiveRatings::from_registers(datasheet, over_current, torque_max);
    match effective.trip_nm {
        None => warn!(
            "motor j{} did not report its limits; judging it against the datasheet",
            joint + 1
        ),
        Some(trip) if effective.is_tightened() => info!(
            "motor j{} trips at {trip:.1} Nm: the margin-derated trip {:.1} Nm undercuts its {} Nm datasheet peak, warning against the derated one",
            joint + 1,
            effective.ratings.peak_nm(),
            datasheet.peak_nm()
        ),
        Some(trip) if effective.trip_too_low => warn!(
            "motor j{} trips at {trip:.1} Nm, too close to its {} Nm continuous rating to warn early: judging it against the datasheet",
            joint + 1,
            datasheet.rated_nm()
        ),
        Some(_) => {}
    }
    effective.ratings
}

#[cfg(test)]
mod tests {
    use super::*;
    use control_core::motor_health::HealthLevel;
    use openarm_can::MotorStatus;

    /// The lineup is a compiled-in table, so its completeness is enforced
    /// here rather than by the `expect` in `effective_ratings`: this failing
    /// is the only way that expect could ever fire.
    #[test]
    fn every_arm_motor_has_datasheet_ratings() {
        for (joint, motor) in openarm_can::ARM_MOTOR_TYPES.iter().enumerate() {
            assert!(motor.ratings().is_some(), "j{} ({motor:?})", joint + 1);
        }
    }

    /// The lineup's datasheet ratings, standing in for what bring-up reads
    /// off the motors.
    fn datasheet_ratings() -> [Ratings; ARM_DOF] {
        std::array::from_fn(|i| {
            openarm_can::ARM_MOTOR_TYPES[i]
                .ratings()
                .expect("every OpenArm arm motor has datasheet ratings")
        })
    }

    fn filters() -> [MotorHealthFilter; ARM_DOF] {
        std::array::from_fn(|i| MotorHealthFilter::new(datasheet_ratings()[i]))
    }

    fn live_state() -> ArmState {
        ArmState {
            statuses: [MotorStatus::Enabled; ARM_DOF],
            temps_rotor_c: [61.0; ARM_DOF],
            ..Default::default()
        }
    }

    #[test]
    fn a_silent_joint_is_judged_from_its_last_measurement() {
        let mut f = filters();
        judge(&mut f, &live_state(), &[false; ARM_DOF], 0.01);
        let mut stale = [false; ARM_DOF];
        stale[4] = true;
        let reports = judge(&mut f, &live_state(), &stale, 0.01);
        assert_eq!(reports[4].level, HealthLevel::NotReporting);
        assert_eq!(
            reports[4].winding_temp.0, 61.0,
            "a silent joint carries what it was last measured at, not a zero"
        );
        assert_eq!(
            reports[0].level,
            HealthLevel::Nominal,
            "one silence cannot hide the joints still talking"
        );
    }

    #[test]
    fn a_stale_status_is_never_stepped_as_a_measurement() {
        // Whatever the cache still says, a stale joint is silent: the cached
        // Enabled must not keep feeding the filter as fresh readings.
        let mut f = filters();
        let mut stale = [false; ARM_DOF];
        stale[2] = true;
        let reports = judge(&mut f, &live_state(), &stale, 0.01);
        assert_eq!(reports[2].level, HealthLevel::NotReporting);
    }

    #[test]
    fn a_non_finite_reading_is_judged_silent_not_stepped() {
        let mut f = filters();
        let mut state = live_state();
        state.torques[3] = f64::NAN;
        let reports = judge(&mut f, &state, &[false; ARM_DOF], 0.01);
        assert_eq!(reports[3].level, HealthLevel::NotReporting);
        assert_eq!(reports[0].level, HealthLevel::Nominal);
    }

    #[test]
    fn a_disabled_motor_is_never_reported_as_healthy() {
        // Torque and temperature both read fine on a limp joint, so the
        // condition is the only thing that can catch it.
        let mut f = filters();
        let mut state = live_state();
        state.statuses[2] = MotorStatus::Disabled;
        let reports = judge(&mut f, &state, &[false; ARM_DOF], 0.01);
        assert_eq!(reports[2].level, HealthLevel::Fault);
        assert_eq!(reports[0].level, HealthLevel::Nominal);
    }

    #[test]
    fn the_raw_samples_are_the_unfiltered_fractions_of_both_limits() {
        let ratings = datasheet_ratings();
        let mut state = ArmState::default();
        state.torques[0] = -10.0;
        state.torques[6] = 1.5;
        let (rated, peak) = fractions(&ratings, &state);
        assert_eq!(rated[0], 10.0 / ratings[0].rated_nm());
        assert_eq!(rated[6], 1.5 / ratings[6].rated_nm());
        assert_eq!(peak[0], 10.0 / ratings[0].peak_nm());
        assert_eq!(peak[6], 1.5 / ratings[6].peak_nm());
    }

    #[test]
    fn stale_flags_trip_at_the_window() {
        let mut passes = [0u32; ARM_DOF];
        passes[2] = 50;
        passes[3] = 49;
        let stale = stale_flags(&passes, 50);
        assert!(stale[2]);
        assert!(!stale[3]);
        assert!(!stale[0]);
    }

    #[test]
    fn a_full_sample_publishes_every_reading() {
        let mut f = filters();
        let state = live_state();
        let reports = judge(&mut f, &state, &[false; ARM_DOF], 0.01);
        let (effort_fractions_rated, effort_fractions_peak) =
            fractions(&datasheet_ratings(), &state);
        let sample = HealthSample {
            reports,
            effort_fractions_rated,
            effort_fractions_peak,
            captured: SystemTime::UNIX_EPOCH,
        };
        let (_, degraded) = build_health_message(&sample).expect("builds");
        assert!(!degraded, "finite readings publish in full");
    }

    #[test]
    fn a_non_finite_reading_degrades_the_message_to_levels_only() {
        // A never-reported motor has no measurements to carry; the levels
        // still publish (naming it as not reporting) with every reading
        // vector empty, per the contract's not-sensed convention.
        let f = filters();
        let reports: [MotorHealth; ARM_DOF] = std::array::from_fn(|i| f[i].silent());
        let sample = HealthSample {
            reports,
            effort_fractions_rated: [0.0; ARM_DOF],
            effort_fractions_peak: [0.0; ARM_DOF],
            captured: SystemTime::UNIX_EPOCH,
        };
        let (_, degraded) = build_health_message(&sample).expect("still builds");
        assert!(degraded, "NaN readings must degrade, not refuse or encode");
        assert!(
            reports.iter().all(|r| r.level == HealthLevel::NotReporting),
            "the levels still name every quiet motor"
        );
    }

    #[test]
    fn observe_returns_the_stale_joints_and_holds_telemetry_without_a_clock() {
        let (health_tx, health_rx) = tokio::sync::watch::channel(None);
        let (_governed_tx, governed_rx) = tokio::sync::watch::channel(None);
        let (measured_tx, _measured_rx) = tokio::sync::watch::channel(None);
        let wiring = StreamWiring {
            governed: governed_rx,
            measured: measured_tx,
            health: health_tx,
        };
        let mut monitor = Monitor::new(datasheet_ratings(), Duration::from_millis(10));
        let mut state = live_state();
        state.passes_since_state[4] = u32::MAX;
        let stale = monitor.observe(&state, &wiring);
        assert!(stale[4]);
        assert!(!stale[0]);
        // No daemon clock in a unit test: the tick's telemetry is skipped
        // (throttled warn) while the staleness verdict still flows to the
        // drive policy. The publish path itself is covered by
        // build_health_message and the stack's live run.
        assert!(health_rx.borrow().is_none());
    }
}
