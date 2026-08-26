//! Motor condition telemetry and the operator alerts raised from it.
//!
//! Reads the driver's cached state at a fixed cadence (no CAN traffic of its
//! own, so it never contends with the follow loop for the bus), steps the
//! shared health filter, and publishes one-motor motor_health reports plus
//! alert transitions. The gripper is the same DM motor as every arm joint
//! and faults the same way: mid-grasp, dropping whatever it held.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use control_core::motor_health::{
    AlertRaiser, DriverTempC, HEALTH_PERIOD, MOTOR_ALERT_KIND, MotorHealth, MotorHealthFilter,
    MotorSample, Ratings, STATE_STALE_AFTER, WindingTempC,
};
use openarm_can::GripperState;
use peppygen::NodeRunner;
use peppygen::emitted_topics::alerts::alerts;
use peppygen::emitted_topics::motor_health::motor_health;
use peppylib::runtime::CancellationToken;
use tracing::{error, warn};

use crate::drive;
use crate::hardware::Gripper;

/// Warn once per failure burst instead of once per tick: one line when a
/// publish starts failing, silence while it keeps failing, re-armed by the
/// first success.
pub(crate) struct LatchedWarn {
    context: &'static str,
    failing: bool,
}

impl LatchedWarn {
    pub(crate) fn new(context: &'static str) -> Self {
        Self {
            context,
            failing: false,
        }
    }

    pub(crate) fn failure(&mut self, error: &str) {
        if !self.failing {
            self.failing = true;
            warn!("{} failing, suppressing repeats: {error}", self.context);
        }
    }

    pub(crate) fn success(&mut self) {
        self.failing = false;
    }
}

/// Capture timestamp from the daemon-resolved clock (sim time under a simulated
/// clock), so consumers age samples on the same timeline they read. Errors
/// until the clock delivers its first tick.
pub(crate) fn capture_timestamp() -> Result<SystemTime, String> {
    let ns = peppygen::clock::now_ns().map_err(|e| format!("clock not ready: {e}"))?;
    Ok(UNIX_EPOCH + Duration::from_nanos(ns))
}

/// One tick's health for the single gripper motor: the filter's report, fed
/// from the cached driver state, or declared silent when the state is past
/// the stale window (`stale_after` follow-loop receive passes; the follow
/// loop is what runs `recv_all`), the motor has yet to report, or a reading
/// decodes non-finite. The non-finite case is judged rather than stepped
/// because there is no measurement in it, and feeding it through would
/// poison the sustained average.
fn judge(
    state: &GripperState,
    filter: &mut MotorHealthFilter,
    stale_after: u32,
    dt_s: f64,
) -> MotorHealth {
    let stale = state.passes_since_state >= stale_after;
    let measurable =
        state.torque.is_finite() && state.temp_mos_c.is_finite() && state.temp_rotor_c.is_finite();
    match state.status.condition() {
        Some(condition) if !stale && measurable => filter.step(
            MotorSample {
                torque_nm: state.torque,
                driver_temp: DriverTempC(state.temp_mos_c),
                winding_temp: WindingTempC(state.temp_rotor_c),
                condition,
            },
            dt_s,
        ),
        _ => filter.silent(),
    }
}

/// The instantaneous |torque| as fractions of the continuous rating and of
/// the peak: the raw samples behind the filter's average, carried so a
/// consumer can show the live numbers beside the sustained one the level
/// judges.
fn fractions(state: &GripperState, ratings: Ratings) -> (f64, f64) {
    let torque = state.torque.abs();
    (torque / ratings.rated_nm(), torque / ratings.peak_nm())
}

/// Publish health and alerts until shutdown, plus one final round after
/// cancellation.
///
/// The health message and the alert batch fail independently: a bad health
/// round must not swallow the alert transitions or the heartbeat re-emits
/// that keep alerts alive on consumer panels, and vice versa.
///
/// The final round re-reads and re-judges the driver cache, flushing the
/// fault verdict the follow loop wrote there before cancelling the node.
pub async fn run(
    runner: Arc<NodeRunner>,
    alert_source: String,
    gripper: Arc<Mutex<Gripper>>,
    ratings: Ratings,
    cycle_period: Duration,
    token: CancellationToken,
) {
    let health_pub = match motor_health::declare_publisher(&runner).await {
        Ok(p) => p,
        Err(e) => return error!("declare motor_health publisher: {e}"),
    };
    let alert_pub = match alerts::declare_publisher(&runner).await {
        Ok(p) => p,
        Err(e) => return error!("declare alerts publisher: {e}"),
    };
    let stale_after = drive::ticks_within(STATE_STALE_AFTER, cycle_period);
    let mut filter = MotorHealthFilter::new(ratings);
    let mut raiser = AlertRaiser::new(vec![alert_source]);
    let mut ticker = tokio::time::interval(HEALTH_PERIOD);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut health_warn = LatchedWarn::new("motor_health publish");
    let mut readings_warn = LatchedWarn::new("health readings");
    let mut alert_warn = LatchedWarn::new("alert publish");
    let mut last_step: Option<SystemTime> = None;
    let mut final_round = false;
    loop {
        // Biased so a cancelled token wins an already-due tick: shutdown
        // gets its flush round before the process exits.
        tokio::select! {
            biased;
            _ = token.cancelled() => final_round = true,
            _ = ticker.tick() => {}
        }
        // One daemon-clock read per round serves both the filter interval
        // and the published timestamp, so the filter's timeline is the one the
        // timestamps are read on (sim time under a simulated clock). A backward
        // step falls back to the cadence; a paused clock yields dt 0, which
        // the filter takes as no elapsed time.
        let timestamp = match capture_timestamp() {
            Ok(timestamp) => timestamp,
            Err(e) => {
                health_warn.failure(&e);
                if final_round {
                    return;
                }
                continue;
            }
        };
        let dt_s = last_step
            .and_then(|last| timestamp.duration_since(last).ok())
            .unwrap_or(HEALTH_PERIOD)
            .as_secs_f64();
        last_step = Some(timestamp);
        let state = gripper
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_state();
        let report = judge(&state, &mut filter, stale_after, dt_s);
        let (now_rated, now_peak) = fractions(&state, ratings);

        publish_report(
            &health_pub,
            timestamp,
            &report,
            now_rated,
            now_peak,
            &mut health_warn,
            &mut readings_warn,
        )
        .await;
        publish_due_alerts(&alert_pub, &mut raiser, &report, &mut alert_warn).await;

        if final_round {
            return;
        }
    }
}

/// Publish one telemetry message, warning once per failure burst; a degraded
/// message (a level without readings) additionally warns once per burst,
/// because it means the motor carries a non-finite reading, which bring-up's
/// enable confirmation is supposed to make impossible.
async fn publish_report(
    publisher: &peppylib::TopicPublisher,
    timestamp: SystemTime,
    report: &MotorHealth,
    now_rated: f64,
    now_peak: f64,
    publish_warn: &mut LatchedWarn,
    readings_warn: &mut LatchedWarn,
) {
    let built = build_health_message(timestamp, report, now_rated, now_peak);
    let (message, degraded) = match built {
        Ok(built) => built,
        Err(e) => return publish_warn.failure(&e),
    };
    match degraded {
        true => readings_warn.failure("the motor carries a non-finite reading"),
        false => readings_warn.success(),
    }
    match publisher.publish(message).await {
        Ok(()) => publish_warn.success(),
        Err(e) => publish_warn.failure(&e.to_string()),
    }
}

/// Publish the alert transitions and heartbeat re-emits this report owes.
/// Alerts commit per successful publish ([`AlertRaiser::mark_sent`]): a
/// failed send stays owed and retries next round, and the heartbeat
/// commits only when the whole round went out.
async fn publish_due_alerts(
    publisher: &peppylib::TopicPublisher,
    raiser: &mut AlertRaiser,
    report: &MotorHealth,
    warn: &mut LatchedWarn,
) {
    let now = Instant::now();
    let batch = raiser.due(std::slice::from_ref(report), now);
    let mut all_sent = true;
    for pending in &batch.items {
        let result = async {
            let msg = alerts::build_message(
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

/// One wire message from a tick's report, and whether it was degraded to
/// levels-only.
///
/// The level always publishes. The reading vectors publish all together or
/// not at all: the contract has no per-reading absence, so one non-finite
/// value (a motor that never reported, which bring-up refuses) empties every
/// reading vector rather than fabricating a number. The level still names
/// the quiet motor.
fn build_health_message(
    timestamp: SystemTime,
    report: &MotorHealth,
    now_rated: f64,
    now_peak: f64,
) -> Result<(peppylib::Payload, bool), String> {
    let readings = [
        now_rated,
        report.torque_fraction,
        now_peak,
        report.driver_temp.0,
        report.winding_temp.0,
    ];
    let degraded = readings.iter().any(|v| !v.is_finite());
    let [rated, sustained, peak, driver, winding] = match degraded {
        true => std::array::from_fn(|_| Vec::new()),
        false => readings.map(|v| vec![v]),
    };
    let payload = motor_health::build_message(
        timestamp,
        vec![report.level.wire()],
        rated,
        sustained,
        peak,
        driver,
        winding,
    )
    .map_err(|e| e.to_string())?;
    Ok((payload, degraded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use control_core::motor_health::HealthLevel;
    use openarm_can::MotorStatus;

    /// DM4310 datasheet figures, the gripper motor on both hardware versions.
    fn datasheet_ratings() -> Ratings {
        Ratings::new(3.0, 7.0).expect("datasheet figures are valid")
    }

    fn filter() -> MotorHealthFilter {
        MotorHealthFilter::new(datasheet_ratings())
    }

    fn driving(passes_since_state: u32) -> GripperState {
        GripperState {
            position: 0.1,
            velocity: 0.0,
            torque: 0.3,
            status: MotorStatus::Enabled,
            temp_mos_c: 40.0,
            temp_rotor_c: 35.0,
            passes_since_state,
        }
    }

    #[test]
    fn a_fresh_driving_state_reaches_the_filter_with_its_readings() {
        let mut f = filter();
        let report = judge(&driving(0), &mut f, 50, 0.2);
        assert_eq!(report.level, HealthLevel::Nominal);
        let sample_fraction = 0.3 / 3.0;
        assert!(
            report.torque_fraction > 0.0 && report.torque_fraction < sample_fraction,
            "one step moves the average toward the sample, got {}",
            report.torque_fraction
        );
        assert_eq!(report.driver_temp.0, 40.0);
        assert_eq!(report.winding_temp.0, 35.0);
    }

    #[test]
    fn the_raw_samples_are_the_unfiltered_fractions_of_both_limits() {
        let ratings = datasheet_ratings();
        assert_eq!(fractions(&driving(0), ratings), (0.3 / 3.0, 0.3 / 7.0));
        let mut reversed = driving(0);
        reversed.torque = -0.3;
        assert_eq!(fractions(&reversed, ratings), (0.3 / 3.0, 0.3 / 7.0));
    }

    #[test]
    fn a_state_past_the_stale_window_is_judged_silent_not_current() {
        let mut f = filter();
        judge(&driving(0), &mut f, 50, 0.2);
        let report = judge(&driving(50), &mut f, 50, 0.2);
        assert_eq!(report.level, HealthLevel::NotReporting);
        // Silence carries the last-known readings rather than zeros.
        assert_eq!(report.driver_temp.0, 40.0);
    }

    #[test]
    fn a_motor_that_never_reported_is_silent_with_absent_readings() {
        let mut f = filter();
        let state = GripperState::default();
        assert_eq!(state.status, MotorStatus::Unreported);
        let report = judge(&state, &mut f, 50, 0.2);
        assert_eq!(report.level, HealthLevel::NotReporting);
        assert!(
            report.driver_temp.0.is_nan(),
            "nothing measured yet, so there is nothing to carry"
        );
    }

    #[test]
    fn a_non_finite_reading_is_judged_silent_not_stepped() {
        let mut f = filter();
        let mut state = driving(0);
        state.torque = f64::NAN;
        let report = judge(&state, &mut f, 50, 0.2);
        assert_eq!(report.level, HealthLevel::NotReporting);
    }

    #[test]
    fn a_fresh_fault_is_reported_as_a_fault() {
        let mut f = filter();
        let mut state = driving(0);
        state.status = MotorStatus::Fault(openarm_can::FaultKind::Overload);
        let report = judge(&state, &mut f, 50, 0.2);
        assert_eq!(report.level, HealthLevel::Fault);
    }

    #[test]
    fn staleness_outranks_the_cached_status() {
        // A cached fault past the stale window reads NotReporting: the motor
        // stopped talking, which ranks above what it last said. The follow
        // loop, not this telemetry, is what acts on the cached fault.
        let mut f = filter();
        let mut state = driving(50);
        state.status = MotorStatus::Fault(openarm_can::FaultKind::Overload);
        let report = judge(&state, &mut f, 50, 0.2);
        assert_eq!(report.level, HealthLevel::NotReporting);
    }

    #[test]
    fn a_full_report_publishes_every_reading() {
        let mut f = filter();
        let state = driving(0);
        let report = judge(&state, &mut f, 50, 0.2);
        let (now_rated, now_peak) = fractions(&state, datasheet_ratings());
        let (_, degraded) =
            build_health_message(SystemTime::UNIX_EPOCH, &report, now_rated, now_peak)
                .expect("builds");
        assert!(!degraded, "finite readings publish in full");
    }

    #[test]
    fn a_non_finite_reading_degrades_the_message_to_levels_only() {
        // A never-reported motor has no measurements to carry; the level
        // still publishes (naming it as not reporting) with every reading
        // vector empty, per the contract's not-sensed convention.
        let report = filter().silent();
        let (_, degraded) =
            build_health_message(SystemTime::UNIX_EPOCH, &report, 0.0, 0.0).expect("still builds");
        assert!(degraded, "NaN readings must degrade, not refuse or encode");
        assert_eq!(
            report.level,
            HealthLevel::NotReporting,
            "the level still names the quiet motor"
        );
    }
}
