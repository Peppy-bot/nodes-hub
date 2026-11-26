use super::*;

use control_core::odometry::WheelGeometry;

use crate::test_rig::{REST_LINEAR_M_S, params as parameters};

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
    assert_eq!(config.move_axes.linear.positive.speed, 0.5);
    assert_eq!(config.rest.velocity, REST_LINEAR_M_S);
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

// move_base ends a turn or a drive at the stop nearest its end, which a floor
// above 0 reaches: a floor of 0 or below, or not finite, is refused by name,
// the message giving Nav2's default of the same floor.
#[test]
fn a_floor_not_above_zero_is_refused_by_name() {
    assert_refused(
        &[
            ("min_turn_rate_rad_s", |p| p.min_turn_rate_rad_s = 0.0),
            ("min_turn_rate_rad_s", |p| p.min_turn_rate_rad_s = f64::NAN),
            ("min_forward_speed_m_s", |p| p.min_forward_speed_m_s = 0.0),
            ("min_forward_speed_m_s", |p| {
                p.min_forward_speed_m_s = f64::INFINITY
            }),
            ("min_backward_speed_m_s", |p| p.min_backward_speed_m_s = 0.0),
            ("min_backward_speed_m_s", |p| {
                p.min_backward_speed_m_s = -0.1
            }),
        ],
        |error, field| matches!(error, ConfigError::FloorNotPositive { name, .. } if *name == field),
    );
    let mut params = parameters();
    params.min_turn_rate_rad_s = 0.0;
    assert_eq!(
        BaseParameters::parse(&params)
            .expect_err("a turn floor of 0")
            .to_string(),
        "parameter min_turn_rate_rad_s: move_base ends a turn or a drive at the stop nearest \
         its end, which a floor above 0 reaches, so give min_turn_rate_rad_s above 0, such as \
         Spin's min_rotational_vel in Nav2's defaults, 0.4"
    );
}

// Each floor is the node's slowest speed of its way, raised to just above the
// deadband and held to the way's speed; the finest yaw tolerance is the
// shortest turn from rest, one step of the limiter toward the floor: at the
// rig's 0.03 rad/s, 1.5 mrad.
#[test]
fn the_move_axes_are_floored_at_the_slowest_speeds() {
    let mut params = parameters();
    params.min_backward_speed_m_s = 0.8;
    let axes = BaseParameters::parse(&params)
        .expect("the parameters are taken")
        .move_axes;
    assert_eq!(
        (axes.linear.positive.floor, axes.linear.negative.floor),
        (0.02, 0.5)
    );
    assert_eq!(
        (axes.angular.positive.floor, axes.angular.negative.floor),
        (0.03, 0.03)
    );
    assert!((axes.finest.yaw - 0.0015).abs() < 1e-15, "{axes:?}");
}

// move_base's drive turns toward its target again past
// rotate_to_heading_min_angle_rad, so an angle below the shortest turn from
// rest is refused, naming both: at the rig's 0.03 rad/s, 1.5 mrad. At a
// floor of 0.4 rad/s and the rig's 3 rad/s^2, botshare_sim's, the first step
// turns at 0.15 rad/s, whose sum lands the shortest turn a few units in the
// last place above 7.5 mrad: 7.5 mrad is taken, and an angle past the slack
// below it is refused, naming 7.5 mrad.
#[test]
fn a_rotate_to_heading_angle_below_the_shortest_turn_is_refused() {
    let refusal = |min_turn_rate: f64, angle: f64| {
        let mut params = parameters();
        params.min_turn_rate_rad_s = min_turn_rate;
        params.rotate_to_heading_min_angle_rad = angle;
        BaseParameters::parse(&params).map(|taken| taken.rotate_to_heading_min_angle_rad)
    };
    let refused = refusal(0.03, 0.001).expect_err("below the shortest turn");
    assert!(
        matches!(refused, ConfigError::AimBelowShortestTurn { angle, .. } if angle == 0.001),
        "{refused:?}"
    );
    let message = |finest: &str, angle: &str| {
        format!(
            "parameter rotate_to_heading_min_angle_rad: move_base's drive turns toward its \
             target again past this bearing, and the shortest turn from rest at angular_limits \
             and min_turn_rate_rad_s is {finest} rad, so give rotate_to_heading_min_angle_rad of \
             at least {finest}, got {angle}"
        )
    };
    assert_eq!(refused.to_string(), message("0.0015", "0.001"));
    assert_eq!(refusal(0.4, 0.0075).expect("the shortest turn"), 0.0075);
    assert_eq!(
        refusal(0.4, 0.007_499_999_985)
            .expect_err("past the slack")
            .to_string(),
        message("0.0075", "0.007499999985")
    );
}

// The shortest turn from rest past 2 rad is refused, naming it and every
// limit that sets it: at a floor of 1.5 rad/s, the rig's fastest turn,
// reached in one step at 30 rad/s^2, and slowed by 0.5 rad/s^2, the turn
// takes 0.075 rad in its first step and 2.2125 rad to stop.
#[test]
fn a_shortest_turn_past_two_radians_is_refused() {
    let mut params = parameters();
    params.min_turn_rate_rad_s = 1.5;
    params.angular_limits.max_acceleration = 30.0;
    params.angular_limits.max_deceleration = 0.5;
    let refused = BaseParameters::parse(&params).expect_err("past 2 rad");
    assert_eq!(
        refused.to_string(),
        "parameter min_turn_rate_rad_s: the shortest turn from rest at smoothing_frequency_hz, \
         angular_limits and min_turn_rate_rad_s is 2.2875 rad, past 2 rad, the longest at which \
         move_base reaches every position tolerance it takes, so give a larger \
         smoothing_frequency_hz or angular_limits max_deceleration, a smaller angular_limits \
         max_acceleration or deadband, or a smaller min_turn_rate_rad_s while it lies above \
         that deadband"
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
    let profile = taken.move_axes.linear;
    assert_eq!((profile.positive.speed, profile.negative.speed), (0.5, 0.0));
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
