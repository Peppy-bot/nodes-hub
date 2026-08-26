//! Per-component motor health for the UI. Consumes every producer bound to
//! the motor_health slot (zero_or_more; the launcher wires the arm and
//! gripper instances), attributes each report to a side and component by the
//! producing instance's name, parses it into per-motor readings, and hands
//! it to the owner keyed by side, so the panel can badge each motor and
//! banner an overloading component.

use std::sync::Arc;
use std::time::{Instant, SystemTime};

use peppygen::NodeRunner;
use peppygen::consumed_topics::motor_health::motor_health;
use peppylib::messaging::ProducerRef;
use peppylib::runtime::CancellationToken;
use tokio::sync::mpsc;
use tracing::error;

use crate::consumer;
use crate::owner::Feedback;
use crate::state::{
    HEALTH_STALE_AFTER, HealthLevel, HealthReport, MotorHealthReading, Side,
    parse_timestamp_validity,
};

/// Whether this deployment binds any motor_health producer.
pub fn available(runner: &NodeRunner) -> bool {
    !motor_health::bound_producers(runner).is_empty()
}

impl consumer::Subscription for motor_health::Subscription {
    type Message = motor_health::Message;
    async fn recv(&mut self) -> peppygen::Result<Option<(ProducerRef, Self::Message)>> {
        self.next().await
    }
}

pub async fn run(
    runner: Arc<NodeRunner>,
    feedback: mpsc::Sender<Feedback>,
    token: CancellationToken,
) {
    let subscription = match motor_health::subscribe(&runner).await {
        Ok(subscription) => subscription,
        Err(e) => {
            error!(error = %e, "motor_health subscribe");
            return;
        }
    };
    consumer::forward_parsed(
        "motor_health",
        token,
        feedback,
        subscription,
        // An unresolved daemon clock cannot certify a timestamp's age, so the
        // report drops on the same throttled-warn path as a malformed one.
        |producer, msg| parse_report(producer, msg, consumer::clock_now()?, Instant::now()),
    )
    .await;
}

/// The component kind a producing instance reports as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Arm,
    Gripper,
}

/// Which panel slot a producer's reports belong to, read from the instance
/// name the launcher chose: the side from a "left"/"right" token and the
/// kind from an "arm"/"grip" token ("left_arm_inst", "right_grip_inst",
/// "left_gripper"). A name carrying neither or both of either pair is
/// refused by name rather than guessed at.
fn classify(instance: &str) -> Result<(Side, Kind), String> {
    let side = match (instance.contains("left"), instance.contains("right")) {
        (true, false) => Side::Left,
        (false, true) => Side::Right,
        _ => {
            return Err(format!(
                "instance {instance:?} does not name exactly one of left/right"
            ));
        }
    };
    let kind = match (instance.contains("arm"), instance.contains("grip")) {
        (true, false) => Kind::Arm,
        (false, true) => Kind::Gripper,
        _ => {
            return Err(format!(
                "instance {instance:?} does not name exactly one of arm/grip"
            ));
        }
    };
    Ok((side, kind))
}

/// Parse one wire report into the owner feedback it routes to: a producing
/// instance this panel has a slot for, one level per motor of that component
/// each a defined severity, reading vectors either absent (empty) or
/// motor-count-length with finite values, and a timestamp not already past the
/// aging window.
///
/// The producing instance is the component, so a report from hardware this
/// panel does not render (another robot's limb) is dropped here rather than
/// shown in the wrong slot. The kind comes from the instance name, and the
/// vector length is validated against it, so an arm-shaped report from a
/// gripper instance cannot masquerade as the arm.
fn parse_report(
    producer: &ProducerRef,
    msg: &motor_health::Message,
    clock_now: SystemTime,
    received_at: Instant,
) -> Result<Feedback, String> {
    let (side, kind) = classify(&producer.instance_id)?;
    match kind {
        Kind::Arm => Ok(Feedback::MotorHealth {
            side,
            health: Box::new(parse_component(msg, clock_now, received_at)?),
        }),
        Kind::Gripper => Ok(Feedback::GripperMotorHealth {
            side,
            health: parse_component(msg, clock_now, received_at)?,
        }),
    }
}

/// One component's report, transposed from the wire's struct-of-arrays into
/// per-motor readings at this parse boundary so everything downstream works
/// motor-wise.
fn parse_component<const MOTORS: usize>(
    msg: &motor_health::Message,
    clock_now: SystemTime,
    received_at: Instant,
) -> Result<HealthReport<MOTORS>, String> {
    let validity =
        parse_timestamp_validity(msg.timestamp, clock_now, received_at, HEALTH_STALE_AFTER)?;
    let wire_levels: [u8; MOTORS] = msg
        .level
        .as_slice()
        .try_into()
        .map_err(|_| format!("expected {MOTORS} levels, got {}", msg.level.len()))?;
    let mut levels = [HealthLevel::Nominal; MOTORS];
    for (parsed, wire) in levels.iter_mut().zip(&wire_levels) {
        *parsed = HealthLevel::from_wire(*wire).ok_or_else(|| format!("undefined level {wire}"))?;
    }
    let vector = |values, name| parse_reading_vector::<MOTORS>(values, name);
    let effort_fraction_rated = vector(&msg.effort_fraction_rated, "effort_fraction_rated")?;
    let effort_fraction_rated_sustained = vector(
        &msg.effort_fraction_rated_sustained,
        "effort_fraction_rated_sustained",
    )?;
    let effort_fraction_peak = vector(&msg.effort_fraction_peak, "effort_fraction_peak")?;
    let driver_temp_c = vector(&msg.driver_temp_c, "driver_temp_c")?;
    let winding_temp_c = vector(&msg.winding_temp_c, "winding_temp_c")?;
    let readings = std::array::from_fn(|i| MotorHealthReading {
        level: levels[i],
        effort_fraction_rated: effort_fraction_rated.map(|v| v[i]),
        effort_fraction_rated_sustained: effort_fraction_rated_sustained.map(|v| v[i]),
        effort_fraction_peak: effort_fraction_peak.map(|v| v[i]),
        driver_temp_c: driver_temp_c.map(|v| v[i]),
        winding_temp_c: winding_temp_c.map(|v| v[i]),
    });
    Ok(HealthReport { readings, validity })
}

/// A reading vector is empty (the producer senses nothing) or exactly one
/// finite value per motor.
fn parse_reading_vector<const MOTORS: usize>(
    values: &[f64],
    name: &str,
) -> Result<Option<[f64; MOTORS]>, String> {
    if values.is_empty() {
        return Ok(None);
    }
    let full: [f64; MOTORS] = values
        .try_into()
        .map_err(|_| format!("expected 0 or {MOTORS} {name} values, got {}", values.len()))?;
    if !full.iter().all(|v| v.is_finite()) {
        return Err(format!("non-finite {name}"));
    }
    Ok(Some(full))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ARM_DOF, ArmHealth, GripperHealth, TIMESTAMP_SKEW_ALLOWANCE};
    use std::time::Duration;

    fn msg() -> motor_health::Message {
        motor_health::Message {
            timestamp: SystemTime::now(),
            level: vec![0; ARM_DOF],
            effort_fraction_rated: (0..ARM_DOF).map(|i| 0.10 + 0.01 * i as f64).collect(),
            effort_fraction_rated_sustained: (0..ARM_DOF).map(|i| 0.20 + 0.01 * i as f64).collect(),
            effort_fraction_peak: (0..ARM_DOF).map(|i| 0.30 + 0.01 * i as f64).collect(),
            driver_temp_c: (0..ARM_DOF).map(|i| 40.0 + i as f64).collect(),
            winding_temp_c: (0..ARM_DOF).map(|i| 35.0 + i as f64).collect(),
        }
    }

    /// Parse from the named instance against a clock equal to the timestamp, so
    /// only routing and shape can fail.
    fn parse_fresh(instance: &str, msg: &motor_health::Message) -> Result<Feedback, String> {
        let producer = ProducerRef::new("core", instance);
        parse_report(&producer, msg, msg.timestamp, Instant::now())
    }

    /// Unwrap an arm parse, panicking on any other routing.
    fn arm(feedback: Feedback) -> (Side, ArmHealth) {
        match feedback {
            Feedback::MotorHealth { side, health } => (side, *health),
            _ => panic!("routed off the arm slot"),
        }
    }

    fn gripper(feedback: Feedback) -> (Side, GripperHealth) {
        match feedback {
            Feedback::GripperMotorHealth { side, health } => (side, health),
            _ => panic!("routed off the gripper slot"),
        }
    }

    #[test]
    fn instances_classify_by_their_side_and_kind_tokens() {
        assert_eq!(classify("left_arm").unwrap(), (Side::Left, Kind::Arm));
        assert_eq!(
            classify("right_gripper").unwrap(),
            (Side::Right, Kind::Gripper)
        );
        // Substring semantics: a decorated launcher name still classifies.
        // The last two pin the openarm launcher family's instance ids.
        assert_eq!(classify("left_arm_v2").unwrap(), (Side::Left, Kind::Arm));
        assert_eq!(
            classify("right_arm_inst").unwrap(),
            (Side::Right, Kind::Arm)
        );
        assert_eq!(
            classify("left_grip_inst").unwrap(),
            (Side::Left, Kind::Gripper)
        );
    }

    #[test]
    fn an_instance_without_exactly_one_side_and_kind_is_refused() {
        for instance in [
            "arm",
            "left",
            "left_right_arm",
            "left_arm_gripper",
            "camera01",
        ] {
            assert!(classify(instance).is_err(), "{instance:?} must refuse");
        }
    }

    #[test]
    fn a_full_report_transposes_every_vector_per_motor() {
        let (side, health) = arm(parse_fresh("right_arm", &msg()).unwrap());
        assert_eq!(side, Side::Right);
        for (i, reading) in health.readings.iter().enumerate() {
            assert_eq!(reading.level, HealthLevel::Nominal);
            assert_eq!(reading.effort_fraction_rated, Some(0.10 + 0.01 * i as f64));
            assert_eq!(
                reading.effort_fraction_rated_sustained,
                Some(0.20 + 0.01 * i as f64)
            );
            assert_eq!(reading.effort_fraction_peak, Some(0.30 + 0.01 * i as f64));
            assert_eq!(reading.driver_temp_c, Some(40.0 + i as f64));
            assert_eq!(reading.winding_temp_c, Some(35.0 + i as f64));
        }
        assert_eq!(health.worst(), HealthLevel::Nominal);
    }

    #[test]
    fn a_not_sensed_report_parses_with_every_reading_absent() {
        let mut m = msg();
        m.effort_fraction_rated = vec![];
        m.effort_fraction_rated_sustained = vec![];
        m.effort_fraction_peak = vec![];
        m.driver_temp_c = vec![];
        m.winding_temp_c = vec![];
        let (side, health) = arm(parse_fresh("left_arm", &m).unwrap());
        assert_eq!(side, Side::Left);
        for reading in &health.readings {
            assert_eq!(reading.effort_fraction_rated, None);
            assert_eq!(reading.effort_fraction_rated_sustained, None);
            assert_eq!(reading.effort_fraction_peak, None);
            assert_eq!(reading.driver_temp_c, None);
            assert_eq!(reading.winding_temp_c, None);
        }
    }

    #[test]
    fn mixed_sensing_is_legal_per_vector() {
        // A producer that measures torque but not temperature sends empty
        // temperature vectors alongside full torque ones.
        let mut m = msg();
        m.winding_temp_c = vec![];
        let (_, health) = arm(parse_fresh("left_arm", &m).unwrap());
        assert_eq!(health.readings[2].winding_temp_c, None);
        assert_eq!(health.readings[2].driver_temp_c, Some(42.0));
        assert_eq!(
            health.readings[2].effort_fraction_rated,
            Some(0.10 + 0.01 * 2.0)
        );
    }

    #[test]
    fn worst_reports_the_most_severe_joint() {
        let mut m = msg();
        m.level[4] = 2;
        let (_, health) = arm(parse_fresh("left_arm", &m).unwrap());
        assert_eq!(health.worst(), HealthLevel::Critical);
    }

    #[test]
    fn a_silent_joint_does_not_cost_the_arm_its_whole_report() {
        // A motor that stops answering is exactly when the panel is needed;
        // refusing the message would blank the six joints still talking.
        let mut m = msg();
        m.level[5] = 4;
        let (_, health) = arm(parse_fresh("left_arm", &m).unwrap());
        assert_eq!(health.readings[5].level, HealthLevel::NotReporting);
        assert_eq!(health.readings[0].level, HealthLevel::Nominal);
        assert_eq!(health.worst(), HealthLevel::NotReporting);
    }

    #[test]
    fn silence_outranks_every_condition_a_motor_can_report() {
        let mut m = msg();
        m.level = vec![3; ARM_DOF];
        m.level[2] = 4;
        let (_, health) = arm(parse_fresh("left_arm", &m).unwrap());
        assert_eq!(health.worst(), HealthLevel::NotReporting);
    }

    #[test]
    fn a_pre_aged_timestamp_rejects_the_report() {
        // A backlogged consumer must not re-stamp a stale report as fresh.
        let m = msg();
        let producer = ProducerRef::new("core", "left_arm");
        let just_inside = m.timestamp + HEALTH_STALE_AFTER + TIMESTAMP_SKEW_ALLOWANCE;
        assert!(parse_report(&producer, &m, just_inside, Instant::now()).is_ok());
        let past = just_inside + Duration::from_millis(1);
        assert!(parse_report(&producer, &m, past, Instant::now()).is_err());
    }

    #[test]
    fn malformed_reports_reject() {
        let mut short_levels = msg();
        short_levels.level = vec![0; ARM_DOF - 1];
        assert!(parse_fresh("left_arm", &short_levels).is_err());

        let mut bad_level = msg();
        bad_level.level[0] = 5;
        assert!(parse_fresh("left_arm", &bad_level).is_err());

        assert!(
            parse_fresh("left_forklift", &msg()).is_err(),
            "an instance this panel has no slot for is dropped, not guessed"
        );
    }

    #[test]
    fn wrong_length_and_non_finite_readings_reject_per_vector() {
        type Field = fn(&mut motor_health::Message) -> &mut Vec<f64>;
        let fields: [(&str, Field); 5] = [
            ("effort_fraction_rated", |m| &mut m.effort_fraction_rated),
            ("effort_fraction_rated_sustained", |m| {
                &mut m.effort_fraction_rated_sustained
            }),
            ("effort_fraction_peak", |m| &mut m.effort_fraction_peak),
            ("driver_temp_c", |m| &mut m.driver_temp_c),
            ("winding_temp_c", |m| &mut m.winding_temp_c),
        ];
        for (name, field) in fields {
            let mut short = msg();
            field(&mut short).truncate(3);
            assert!(
                parse_fresh("left_arm", &short).is_err(),
                "short {name} must reject"
            );
            for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut poisoned = msg();
                field(&mut poisoned)[2] = bad;
                assert!(
                    parse_fresh("left_arm", &poisoned).is_err(),
                    "{bad} in {name} must reject"
                );
            }
        }
    }

    #[test]
    fn a_gripper_report_parses_as_its_single_motor() {
        let m = motor_health::Message {
            timestamp: SystemTime::now(),
            level: vec![1],
            effort_fraction_rated: vec![0.4],
            effort_fraction_rated_sustained: vec![0.35],
            effort_fraction_peak: vec![0.2],
            driver_temp_c: vec![41.0],
            winding_temp_c: vec![37.0],
        };
        let (side, health) = gripper(parse_fresh("left_gripper", &m).unwrap());
        assert_eq!(side, Side::Left);
        let [reading] = health.readings;
        assert_eq!(reading.level, HealthLevel::Warning);
        assert_eq!(reading.effort_fraction_rated, Some(0.4));
        assert_eq!(reading.effort_fraction_rated_sustained, Some(0.35));
        assert_eq!(reading.effort_fraction_peak, Some(0.2));
    }

    #[test]
    fn a_gripper_report_with_arm_shaped_vectors_rejects() {
        // The kind comes from the instance name and the vector length is
        // validated against it: seven levels from a gripper instance are a
        // producer bug, not an arm report to reroute.
        let mut m = msg();
        assert!(parse_fresh("right_gripper", &m).is_err());
        m.level = vec![0];
        assert!(
            parse_fresh("right_gripper", &m).is_err(),
            "readings must be empty or exactly one"
        );
    }
}
