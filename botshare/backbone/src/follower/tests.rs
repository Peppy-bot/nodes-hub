use super::*;

/// `ms` milliseconds on the node's clock, in nanoseconds.
fn ms(ms: i64) -> i64 {
    ms * 1_000_000
}

/// A follower whose wheels report from `ms(0)` on.
fn reporting() -> Follower {
    let mut follower = Follower::default();
    follower.silence_ends(ms(0));
    follower
}

#[test]
fn the_silence_holds_the_base_before_the_wheels_report() {
    let mut follower = Follower::default();
    assert!(
        !follower.setpoint(ms(0), false),
        "dropped before the wheels report"
    );
    assert_eq!(follower.state(ms(0), false), FollowState::BaseSilence);
    assert!(
        follower.streams_at(ms(0)),
        "a dropped setpoint still streams"
    );
}

#[test]
fn the_stream_lasts_until_more_than_half_a_second_passes_with_no_setpoint() {
    let mut follower = reporting();
    assert!(follower.setpoint(ms(100), false));
    assert!(follower.streams_at(ms(600)), "0.5 s exactly still streams");
    assert!(!follower.streams_at(ms(601)));
}

#[test]
fn a_robot_stop_holds_until_more_than_half_a_second_passes_from_it_with_no_setpoint() {
    let mut follower = reporting();
    follower.setpoint(ms(0), false);
    assert!(follower.stop(ms(100)), "the leader streamed at the stop");
    assert!(
        !follower.setpoint(ms(400), false),
        "dropped during the hold"
    );
    assert_eq!(follower.state(ms(900), false), FollowState::RobotStop);
    assert_eq!(
        follower.state(ms(901), false),
        FollowState::Following,
        "0.5 s passed after the last setpoint, from the stop on"
    );
    assert!(follower.setpoint(ms(950), false), "the hold ended for good");
}

#[test]
fn a_robot_stop_with_no_stream_holds_for_half_a_second_from_the_stop() {
    let mut follower = reporting();
    assert!(!follower.stop(ms(100)), "the leader did not stream");
    assert!(
        !follower.setpoint(ms(600), false),
        "within 0.5 s of the stop"
    );
    assert!(
        !follower.setpoint(ms(1_000), false),
        "the setpoint renewed the hold"
    );
    assert!(follower.setpoint(ms(1_501), false));
}

#[test]
fn a_silence_that_starts_while_the_leader_streams_holds_on_until_the_stream_ends() {
    let mut follower = reporting();
    follower.setpoint(ms(0), false);
    follower.silence_starts(ms(200));
    assert!(!follower.setpoint(ms(400), false));
    follower.silence_ends(ms(500));
    assert!(
        !follower.setpoint(ms(800), false),
        "the stream that was live at the silence goes on"
    );
    assert_eq!(follower.state(ms(1_300), false), FollowState::BaseSilence);
    assert_eq!(follower.state(ms(1_301), false), FollowState::Following);
}

#[test]
fn a_silence_that_starts_with_no_stream_holds_the_base_only_while_it_lasts() {
    let mut follower = reporting();
    follower.silence_starts(ms(200));
    assert!(!follower.setpoint(ms(300), false));
    follower.silence_ends(ms(400));
    assert!(follower.setpoint(ms(450), false));
}

#[test]
fn the_holds_are_told_in_order_base_silence_robot_stop_follower_goal() {
    let mut follower = reporting();
    assert_eq!(follower.state(ms(0), true), FollowState::FollowerGoal);
    assert!(!follower.setpoint(ms(0), true), "a goal holds the base");
    follower.stop(ms(10));
    assert_eq!(follower.state(ms(20), true), FollowState::RobotStop);
    follower.silence_starts(ms(30));
    assert_eq!(follower.state(ms(40), true), FollowState::BaseSilence);
    assert_eq!(FollowState::FollowerGoal.as_str(), "follower_goal");
    assert_eq!(FollowState::BaseSilence.as_str(), "base_silence");
}

// After the clock went back, the stream and the robot stop's hold last on
// from the new time, as if the setpoint and the stop came then.
#[test]
fn a_clock_that_went_back_keeps_the_stream_and_the_robot_stop_from_its_time() {
    let mut follower = reporting();
    follower.setpoint(ms(1_000), false);
    follower.stop(ms(1_000));
    follower.clock_went_back(ms(100));
    assert!(
        follower.streams_at(ms(600)),
        "the stream lasts 0.5 s from 100 ms"
    );
    assert!(!follower.streams_at(ms(601)));
    assert_eq!(follower.state(ms(600), false), FollowState::RobotStop);
    assert!(
        follower.setpoint(ms(1_101), false),
        "the stop held until 600 ms"
    );
}
