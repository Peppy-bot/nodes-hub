use super::*;

use crate::test_rig::AXES;

#[test]
fn grid_points_fall_on_whole_periods() {
    assert_eq!(grid_point_at_or_after(1_000, 50), 1_000);
    assert_eq!(grid_point_at_or_after(1_001, 50), 1_050);
    assert_eq!(grid_point_at_or_after(0, 50), 0);
}

#[test]
fn a_ramp_reaches_its_target_up_to_rounding() {
    let target = PlanarTwist::new(0.3, 0.0, -1.0).unwrap();
    let rounded = PlanarTwist::new(0.1 + 0.2, 0.0, -1.0).unwrap();
    assert!(reached(rounded, target));
    assert!(!reached(PlanarTwist::new(0.29, 0.0, -1.0).unwrap(), target));
}

#[test]
fn a_cancel_ends_a_running_or_completed_goal_cancelled_and_keeps_an_earlier_cancel() {
    let stop = || "stopped".to_string();
    assert!(matches!(
        cancelled(None, stop()),
        (Ending::Cancelled(reason), true) if reason == "stopped"
    ));
    assert!(matches!(
        cancelled(Some(Ending::Completed), stop()),
        (Ending::Cancelled(reason), true) if reason == "stopped"
    ));
    assert!(matches!(
        cancelled(Some(Ending::Cancelled("by its caller".to_string())), stop()),
        (Ending::Cancelled(reason), false) if reason == "by its caller"
    ));
}

#[test]
fn an_epoch_goes_on_with_its_follower_at_or_after_its_last_stamp_within_the_wheels_reach() {
    let state = |follower: &str, timestamp_ns, angle| WheelState {
        positions: [angle; 2],
        timestamp_ns,
        follower: follower.to_string(),
    };
    let epoch = Epoch {
        follower: "wheels".to_string(),
        odometry_start: std::time::SystemTime::UNIX_EPOCH,
        last: state("wheels", 1_000, 0.0),
        sent: None,
        closed: false,
    };
    // A wheel reaches 1 rad a second and 0.5 rad more.
    let reach = |dt_s: f64| dt_s + 0.5;
    assert!(epoch.continues_with(&state("wheels", 1_000, 0.5), reach));
    assert!(
        epoch.continues_with(&state("wheels", 9_000_001_000, 9.5), reach),
        "a silence goes on with the epoch"
    );
    assert!(!epoch.continues_with(&state("wheels", 999, 0.0), reach));
    assert!(!epoch.continues_with(&state("other_wheels", 2_000, 0.0), reach));
    assert!(
        !epoch.continues_with(&state("wheels", 1_001_000, -0.6), reach),
        "wheels past their reach started over"
    );
    let closed = Epoch {
        closed: true,
        ..epoch
    };
    assert!(
        !closed.continues_with(&state("wheels", 2_000, 0.0), reach),
        "a closed epoch takes no measurement"
    );
}

// While the wheels are silent, each step hands the limiter a zero setpoint
// past drive_link's 0.5 s after any leader's setpoint: the base slows at the
// deceleration limit, 0.1 m/s a step here, and once at rest the limiter gives
// nothing.
#[test]
fn a_zero_setpoint_past_its_hold_brings_the_base_to_rest_then_gives_nothing() {
    let step_ns = 50_000_000;
    let mut limiter = AXES.limiter_settings(20.0).expect("the rig's axes").start();
    limiter.take_setpoint(PlanarTwist::new(0.2, 0.0, 0.0).unwrap(), 0);
    for step in 1..=4 {
        limiter.tick(step * step_ns);
    }
    let speeds: Vec<Option<f64>> = (5..=8)
        .map(|step| {
            let now_ns = step * step_ns;
            limiter.take_setpoint(PlanarTwist::new(0.5, 0.0, 0.0).unwrap(), now_ns - 1);
            limiter.take_setpoint(PlanarTwist::ZERO, expired_at(now_ns));
            limiter.tick(now_ns).map(PlanarTwist::linear_x)
        })
        .collect();
    assert!(
        matches!(
            speeds.as_slice(),
            [Some(slower), Some(rest), None, None]
                if (slower - 0.1).abs() < 1e-12 && *rest == 0.0
        ),
        "{speeds:?}"
    );
}

// An admitted goal the loop takes ends cancelled for a stop that came on its
// way, then fails for a clock that went back, then for an odometry that
// started over since its admission, and starts otherwise.
#[test]
fn a_goal_on_its_way_ends_for_a_stop_then_a_clock_back_then_an_odometry_start_over() {
    let failure = |outcome: Option<Outcome>| match outcome {
        Some(Outcome::Failed(failure)) => Some(failure.message),
        _ => None,
    };
    assert!(matches!(
        refusal_at_start(Some("stopped".to_string()), true, true),
        Some(Outcome::Ending(Ending::Cancelled(reason))) if reason == "stopped"
    ));
    assert_eq!(
        failure(refusal_at_start(None, true, true)),
        Some(CLOCK_WENT_BACK.to_string())
    );
    assert_eq!(
        failure(refusal_at_start(None, false, true)),
        Some(ODOMETRY_STARTED_OVER.to_string())
    );
    assert!(refusal_at_start(None, false, false).is_none());
}
