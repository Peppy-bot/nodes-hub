use super::*;

use crate::test_rig::params as parameters;
use control_core::odometry::WheelGeometry;

/// What spoils one parameter of a base the node takes.
type Spoil = fn(&mut Parameters);

/// Asserts that every spoil of `spoiled` is refused as `refused` says.
fn assert_refused(spoiled: &[(&str, Spoil)], refused: fn(&ConfigError, &str) -> bool) {
    for (field, spoil) in spoiled {
        let mut params = parameters();
        spoil(&mut params);
        match BaseParameters::parse(&params) {
            Err(error) => assert!(refused(&error, field), "{field}: {error:?}"),
            Ok(_) => panic!("{field} is taken"),
        }
    }
}

// The parameters with the bundled base's model: the limiter's step, the
// axes, the rest velocity, the wheels and the outline of the model.
#[test]
fn a_base_of_well_formed_parameters_is_configured_for_the_bundled_base() {
    let parameters = BaseParameters::parse(&parameters()).expect("the parameters are taken");
    let config = parameters
        .configure("botshare_base", SystemTime::UNIX_EPOCH)
        .expect("the bundled base");
    assert_eq!(config.step_ns, 50_000_000);
    assert_eq!(config.axes.linear.profile().positive_speed, 0.5);
    assert_eq!(config.rest.velocity, 0.005);
    assert_eq!(
        config.wheels,
        WheelGeometry::parse(0.4, 0.1, 0.1).expect("a wheel geometry")
    );
    assert_eq!(config.footprint.points().len(), 4);
    let refused = parameters
        .configure("so101", SystemTime::UNIX_EPOCH)
        .expect_err("an arm");
    assert!(
        refused
            .to_string()
            .starts_with("model so101 is no differential base this node drives: "),
        "{refused}"
    );
}

// base_state's rest bound of the turn rate is the turn of the two wheels each
// at the rest velocity, twice rest_velocity over the wheel separation: 0.1
// rad/s for the bundled base's 0.4 m track at 0.02 m/s.
#[test]
fn the_rest_rate_is_the_turn_of_both_wheels_at_the_rest_velocity() {
    let mut params = parameters();
    params.linear_limits.rest_velocity = 0.02;
    let rest = BaseParameters::parse(&params)
        .expect("the parameters are taken")
        .configure("botshare_base", SystemTime::UNIX_EPOCH)
        .expect("the bundled base")
        .rest;
    assert_eq!(rest.velocity, 0.02);
    assert!((rest.rate - 0.1).abs() < 1e-15, "{rest:?}");
    assert_eq!(rest.bounds(), [0.02, 0.0, rest.rate]);
}

#[test]
fn a_positive_number_not_above_zero_is_refused_by_name() {
    assert_refused(
        &[
            ("linear_limits.rest_velocity", |p| {
                p.linear_limits.rest_velocity = 0.0
            }),
            ("linear_limits.rest_velocity", |p| {
                p.linear_limits.rest_velocity = -0.1
            }),
            ("rotate_to_heading_min_angle_rad", |p| {
                p.rotate_to_heading_min_angle_rad = 0.0
            }),
        ],
        |error, field| matches!(error, ConfigError::NotPositive { name, .. } if *name == field),
    );
}

// A stop timeout below 0, too small for a timer to hold, or past what one
// holds is refused naming the parameter, then why.
#[test]
fn a_stop_answer_timeout_no_timer_holds_is_refused() {
    for seconds in [-1.0, 1e-10, 1e300] {
        let mut params = parameters();
        params.stop_answer_timeout_s = seconds;
        let refused = BaseParameters::parse(&params).expect_err("no timer holds it");
        assert_eq!(refused.to_string(), "parameter stop_answer_timeout_s");
        assert!(
            std::error::Error::source(&refused)
                .is_some_and(|why| why.to_string().contains(&format!("{seconds:e}"))
                    || why.to_string().contains(&seconds.to_string())),
            "{refused:?}"
        );
    }
}

#[test]
fn an_axis_that_describe_base_does_not_allow_is_refused_by_name() {
    assert_refused(
        &[
            ("linear_limits", |p| p.linear_limits.min_velocity = 0.1),
            ("linear_limits", |p| p.linear_limits.max_acceleration = 0.0),
            ("angular_limits", |p| p.angular_limits.deadband = 1.5),
        ],
        |error, field| matches!(error, ConfigError::Axis { name, .. } if *name == field),
    );
    assert_refused(
        &[
            ("angular_limits.max_deceleration", |p| {
                p.angular_limits.max_deceleration = -1.0
            }),
            ("linear_limits.deadband", |p| {
                p.linear_limits.deadband = -0.01
            }),
            ("angular_limits.max_acceleration", |p| {
                p.angular_limits.max_acceleration = f64::NAN
            }),
        ],
        |error, field| {
            matches!(error, ConfigError::NotNonNegative { axis, limit, .. }
                if format!("{axis}.{limit}") == field)
        },
    );
}

// move_base drives a base ahead and turns it the shorter way: a base that
// cannot drive ahead, or turns one way alone, is refused, a range of 0 to 0
// with the rig's other limits among them, before the limiter judges those
// limits; a base that cannot drive backward is taken, its profile's backward
// speed 0.
#[test]
fn a_base_that_cannot_drive_ahead_or_turn_both_ways_is_refused() {
    assert_refused(
        &[
            ("linear_limits", |p| {
                p.linear_limits.min_velocity = -0.5;
                p.linear_limits.max_velocity = 0.0;
            }),
            ("linear_limits", |p| {
                p.linear_limits.min_velocity = 0.0;
                p.linear_limits.max_velocity = 0.0;
            }),
        ],
        |error, _| matches!(error, ConfigError::NoForward),
    );
    assert_refused(
        &[
            ("angular_limits", |p| p.angular_limits.min_rate = 0.0),
            ("angular_limits", |p| {
                p.angular_limits.min_rate = -1.5;
                p.angular_limits.max_rate = 0.0;
            }),
            ("angular_limits", |p| {
                p.angular_limits.min_rate = 0.0;
                p.angular_limits.max_rate = 0.0;
            }),
        ],
        |error, _| matches!(error, ConfigError::OneWayTurn),
    );
    let mut ahead_only = parameters();
    ahead_only.linear_limits.min_velocity = 0.0;
    let taken = BaseParameters::parse(&ahead_only).expect("a base that drives ahead alone");
    let profile = taken.axes.linear.profile();
    assert_eq!((profile.positive_speed, profile.negative_speed), (0.5, 0.0));
}

// The limiter steps at a whole number of hertz from 20 to 1000 whose period
// is a whole number of milliseconds; the step is that period.
#[test]
fn a_limiter_frequency_without_a_period_of_whole_milliseconds_is_refused() {
    assert_refused(
        &[
            ("smoothing_frequency_hz", |p| {
                p.smoothing_frequency_hz = 19.9
            }),
            ("smoothing_frequency_hz", |p| {
                p.smoothing_frequency_hz = 30.0
            }),
            ("smoothing_frequency_hz", |p| {
                p.smoothing_frequency_hz = 1_000.1
            }),
        ],
        |error, _| {
            matches!(
                error,
                ConfigError::Smoothing(LimiterError::Frequency(hz))
                    if [19.9, 30.0, 1_000.1].contains(hz)
            )
        },
    );
    assert_refused(
        &[("smoothing_frequency_hz", |p| p.smoothing_frequency_hz = 0.0)],
        |error, field| matches!(error, ConfigError::NotPositive { name, .. } if *name == field),
    );
    for (frequency_hz, step_ns) in [(20.0, 50_000_000), (1_000.0, 1_000_000)] {
        let mut params = parameters();
        params.smoothing_frequency_hz = frequency_hz;
        let taken = BaseParameters::parse(&params).expect("a frequency the limiter steps at");
        assert_eq!(taken.step_ns, step_ns, "{frequency_hz} Hz");
    }
}

#[test]
fn an_empty_window_is_refused() {
    assert_refused(
        &[("velocity_rolling_window_size", |p| {
            p.velocity_rolling_window_size = 0
        })],
        |error, _| matches!(error, ConfigError::EmptyWindow),
    );
}
