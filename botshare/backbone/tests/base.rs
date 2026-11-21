//! The differential base booted in-process on a simulated clock the test
//! ticks, against mocks of its wheels and of its leader: the refusals of
//! base_motion, the follower of drive_link's leader and its holds, the robot
//! stop and the cancels, the velocity path, the wheel setpoints, the odometry,
//! its epochs and its transforms, the cadence of base_states and the silence
//! of the wheels, each judged on the node's own clock.

mod helpers;

use std::f64::consts::PI;
use std::num::NonZeroUsize;
use std::time::SystemTime;

use frame_tree::Transform;
use frame_tree::arithmetic::{quaternion_of_yaw, yaw_of_quaternion};

use control_core::odometry::{Odometry, WheelGeometry};
use helpers::{
    ANCHOR_BOUND_RAD, ANGULAR_ACCELERATION_RAD_S2, ANGULAR_DEADBAND_RAD_S,
    ANGULAR_DECELERATION_RAD_S2, ANSWER_BUDGET, BaseState, DT, LEADER_STREAMING_WINDOW_NS,
    LINEAR_ACCELERATION_M_S2, LINEAR_DEADBAND_M_S, LINEAR_DECELERATION_M_S2,
    MAX_ANGULAR_SPEED_RAD_S, MAX_LINEAR_SPEED_M_S, MODEL, REPORTING_SILENT_STEPS,
    REST_ANGULAR_RAD_S, REST_LINEAR_M_S, Rig, START_NS, STEP_NS, WHEEL_LINKS, WHEEL_RADIUS_M,
    WHEEL_SEPARATION_M, drive_goal, forward_speed, move_goal, ns, target, turn_rate, velocity,
};
use peppygen::fixtures::exposed_actions::base_motion::{drive_base, move_base};
use peppygen::fixtures::exposed_services::base_state::describe_base;

/// How much the base speeds up in one step at its acceleration limit (m/s).
const SPEED_UP: f64 = LINEAR_ACCELERATION_M_S2 * DT;

/// How much the base slows in one step at its deceleration limit (m/s).
const SLOW_DOWN: f64 = LINEAR_DECELERATION_M_S2 * DT;

/// The frame odom places: the robot frame, the root link of the model.
const ROBOT_FRAME: [&str; 1] = ["base_footprint"];

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "{what}: {actual} is not {expected}"
    );
}

/// Asserts that the unit quaternion `actual`, [x, y, z, w], gives the
/// rotation of `expected` within 1e-9: equal to it, or its negation.
fn assert_same_rotation(actual: &[f64], expected: [f64; 4], what: &str) {
    let near = |sign: f64| {
        actual
            .iter()
            .zip(expected)
            .all(|(a, e)| (a - sign * e).abs() < 1e-9)
    };
    assert!(
        actual.len() == 4 && (near(1.0) || near(-1.0)),
        "{what}: {actual:?} is not {expected:?}"
    );
}

/// The rotation by `angle` (rad) about the unit axis `axis`, as [x, y, z, w].
fn rotation_about(axis: [f64; 3], angle: f64) -> [f64; 4] {
    let (sin, cos) = (angle / 2.0).sin_cos();
    [axis[0] * sin, axis[1] * sin, axis[2] * sin, cos]
}

/// The wheel angle (rad) one step at `speed` (m/s) turns each wheel by.
fn wheel_step(speed: f64) -> f64 {
    speed / WHEEL_RADIUS_M * DT
}

async fn refusal_of_drive(rig: &Rig, goal: drive_base::GoalRequestData) -> drive_base::Failure {
    rig.send_drive(&goal)
        .await
        .refusal
        .expect("the goal is refused")
}

/// Step to each `(seconds, speed)` after the rig's origin and check the
/// forward speed of the node's wheel setpoint of that step.
async fn assert_speeds(rig: &mut Rig, speeds: &[(f64, f64)]) {
    for &(seconds, speed) in speeds {
        let at = rig.at(seconds);
        rig.step_to(at).await;
        let setpoint = rig.wheel_setpoint_at(at).await;
        assert_close(
            forward_speed(&setpoint),
            speed,
            &format!("the speed at {seconds} s"),
        );
    }
}

/// The follow_state drive_states tells the leader at the rig's time.
async fn follow_state(rig: &mut Rig) -> String {
    rig.drive_state_at(rig.now_ns).await.follow_state
}

// A follower still runs the last setpoint of a backbone that ended: its
// wheels turn as this one starts, with no goal and no setpoint of the leader.
// At their first measurement the base holds them where they stand, a setpoint
// of rate 0 at the measured angles.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_base_holds_its_wheels_where_they_stand_at_their_first_measurement() {
    let mut rig = Rig::boot([0.7, -0.3]).await;
    rig.publish_wheels_turning(START_NS, vec![1.5, 1.5]).await;
    let hold = rig.next_wheel_setpoint().await;
    assert_eq!(
        (hold.positions, hold.velocities),
        ([0.7, -0.3], [0.0, 0.0]),
        "the first wheel setpoint holds the wheels"
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_velocity_outside_the_bases_limits_is_rejected_saying_what_to_send() {
    let rig = Rig::start([0.0, 0.0]).await;
    for (goal, message) in [
        (
            drive_goal(0.51, 1.0),
            "the velocity's linear_velocity x, placed in the robot frame, is 0.51, outside the \
             base's linear_limits along x of -0.5 to 0.5: send a velocity within them",
        ),
        (
            drive_base::GoalRequestData {
                velocity: velocity(0.0, -1.6),
                ..drive_goal(0.0, 1.0)
            },
            "the velocity's angular_velocity z, placed in the robot frame, is -1.6, outside the \
             base's angular_limits of -1.5 to 1.5: send a velocity within them",
        ),
        // A differential base moves along no y: the refusal says to send a
        // velocity with vy 0.
        (
            drive_base::GoalRequestData {
                velocity: peppygen::exposed_actions::base_motion::drive_base::GoalVelocity {
                    linear_velocity: [0.0, 0.1, 0.0],
                    ..velocity(0.0, 0.0)
                },
                ..drive_goal(0.0, 1.0)
            },
            "the velocity's linear_velocity, placed in the robot frame, moves 0.1 m/s along y: \
             send a velocity in the robot frame of [vx, vy, 0] and [0, 0, wz], with vy 0 on a \
             differential base",
        ),
    ] {
        let refused = refusal_of_drive(&rig, goal).await;
        assert_eq!(refused.kind, drive_base::FailureKind::Rejected);
        assert_eq!(refused.message, message);
    }
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_hold_duration_of_0_and_a_settle_timeout_past_60_s_are_rejected() {
    let rig = Rig::start([0.0, 0.0]).await;
    let instant = refusal_of_drive(&rig, drive_goal(0.1, 0.0)).await;
    assert_eq!(instant.kind, drive_base::FailureKind::Rejected);
    assert_eq!(
        instant.message,
        "hold_duration_s must be a number of seconds above 0 and at most 60, got 0"
    );
    let unsettled = refusal_of_drive(
        &rig,
        drive_base::GoalRequestData {
            settle_timeout_s: 61.0,
            ..drive_goal(0.1, 1.0)
        },
    )
    .await;
    assert_eq!(unsettled.kind, drive_base::FailureKind::Rejected);
    assert_eq!(
        unsettled.message,
        "settle_timeout_s must be a number of seconds from 0 to 60, got 61"
    );
    let far_settled = rig
        .send_move(&move_goal(1.0, 0.0, 0.0, 61.0))
        .await
        .refusal
        .expect("a settle timeout past 60 s is refused");
    assert_eq!(far_settled.kind, move_base::FailureKind::Rejected);
    rig.shutdown().await;
}

// The fastest drive of d m straight ahead takes 0.375 s + 2 s/m * d: 0.5 s up
// to 0.5 m/s at 1 m/s^2 and 0.25 s down at 2 m/s^2, over 0.1875 m, and the
// rest at 0.5 m/s. 31 m take 62.375 s, past 60 s; 29 m take 58.375 s.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_move_the_limits_cannot_run_within_60_s_is_rejected() {
    let rig = Rig::start([0.0, 0.0]).await;
    let far = rig
        .send_move(&move_goal(31.0, 0.0, 0.0, 1.0))
        .await
        .refusal
        .expect("a move past 60 s is refused");
    assert_eq!(far.kind, move_base::FailureKind::Rejected);
    assert!(
        far.message.ends_with(
            "s at the base's limits, more than 60 s: split it into moves to nearer targets"
        ),
        "{}",
        far.message
    );
    let near = rig.send_move(&move_goal(29.0, 0.0, 0.0, 1.0)).await;
    assert!(
        near.accepted,
        "a move of 58.4 s is taken: {:?}",
        near.refusal
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_while_the_leader_streams_is_refused_leader_streaming() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    rig.lead_until_it_drives([0.1, 0.0, 0.0], 0.0).await;
    let refused_move = rig
        .send_move(&move_goal(1.0, 0.0, 0.0, 1.0))
        .await
        .refusal
        .expect("a move is refused while the leader streams");
    assert_eq!(refused_move.kind, move_base::FailureKind::LeaderStreaming);
    // The leader streams until more than 0.5 s pass after its setpoint
    // arrived, at the origin: a goal is refused at 0.5 s and taken a step
    // later.
    rig.step_to(rig.at(0.5)).await;
    let still_led = refusal_of_drive(&rig, drive_goal(0.1, 1.0)).await;
    assert_eq!(still_led.kind, drive_base::FailureKind::LeaderStreaming);
    rig.step_to(rig.at(0.55)).await;
    let free = rig.send_drive(&drive_goal(0.1, 1.0)).await;
    assert!(
        free.accepted,
        "0.55 s after the setpoint: {:?}",
        free.refusal
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_with_two_causes_gets_the_kind_of_the_earlier_check() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    rig.lead_until_it_drives([0.1, 0.0, 0.0], 0.0).await;
    // A value the base does not take is checked before the leader.
    let refused = refusal_of_drive(&rig, drive_goal(0.6, 1.0)).await;
    assert_eq!(refused.kind, drive_base::FailureKind::Rejected);
    rig.shutdown().await;
}

// A running move_base holds the base: the follower drops the leader's
// setpoints, drive_states tells the leader so, and the move runs on, as
// drive_link states.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_running_move_base_drops_the_leaders_setpoints_and_runs_on() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    let goal = rig.send_move(&move_goal(1.0, 0.0, 0.0, 5.0)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    // The move drives toward its target, which stays 1 m ahead while the
    // wheels hold still, speeding up at the acceleration limit to the speed
    // limit by 0.5 s. From 0.25 s the leader asks for rest at each step.
    rig.step_to(rig.at(0.25)).await;
    for step in 1..=5 {
        rig.lead([0.0, 0.0, 0.0], 0.0).await;
        rig.step().await;
        assert_eq!(follow_state(&mut rig).await, "follower_goal", "step {step}");
    }
    assert_speeds(
        &mut rig,
        &[(0.5, MAX_LINEAR_SPEED_M_S), (0.55, MAX_LINEAR_SPEED_M_S)],
    )
    .await;
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_leader_setpoint_past_the_limits_is_clamped() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    // Twice the speed limit, a turn rate within its limit and a y the base
    // does not execute.
    let turn = MAX_ANGULAR_SPEED_RAD_S / 2.0;
    rig.lead_until_it_drives([2.0 * MAX_LINEAR_SPEED_M_S, 0.3, 0.0], turn)
        .await;
    // The speed is clamped to its limit, y to 0, and the axes ramp together
    // toward the clamped velocity, the speed at its acceleration limit: each
    // step the turn gains the share of its rate the speed gains of its limit,
    // and both arrive at the tenth step.
    assert_close(
        MAX_LINEAR_SPEED_M_S / SPEED_UP,
        10.0,
        "the steps of the speed's ramp",
    );
    for step in [5, 10] {
        let at = rig.origin_ns + step * STEP_NS;
        rig.step_to(at).await;
        let setpoint = rig.wheel_setpoint_at(at).await;
        let share = step as f64 / 10.0;
        assert_close(
            forward_speed(&setpoint),
            share * MAX_LINEAR_SPEED_M_S,
            &format!("the speed of step {step}"),
        );
        assert_close(
            turn_rate(&setpoint),
            share * turn,
            &format!("the turn rate of step {step}"),
        );
    }
    rig.shutdown().await;
}

// A setpoint with a value that is not finite or with a z other than 0 counts
// as no setpoint, as drive_link states.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_leader_setpoint_not_finite_or_off_the_floor_is_dropped() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    rig.lead_until_it_drives([0.2, 0.0, 0.0], 0.0).await;
    rig.step_to(rig.at(0.25)).await;
    rig.lead([f64::NAN, 0.0, 0.0], 0.0).await;
    rig.lead([0.2, 0.0, 0.0], f64::INFINITY).await;
    rig.lead([0.2, 0.0, 0.3], 0.0).await;
    // The setpoint of the origin holds the base at 0.2 m/s through 0.5 s,
    // and the base slows from the next step on: the dropped setpoints reached
    // neither the limiter nor the leader's stream.
    assert_speeds(&mut rig, &[(0.5, 0.2), (0.55, 0.2 - SLOW_DOWN)]).await;
    let free = rig.send_drive(&drive_goal(0.1, 1.0)).await;
    assert!(free.accepted, "{:?}", free.refusal);
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn move_base_feedback_gives_what_remains_of_the_move() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    let mut goal = rig.send_move(&move_goal(1.0, 0.0, PI / 2.0, 5.0)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    rig.step_to(rig.at(0.25)).await;
    // The wheels roll the base 0.5 m straight ahead over five steps, 0.1 m a
    // step, within what they turn in a step at the base's limits, then hold.
    let turn = 0.1 / WHEEL_RADIUS_M;
    for _ in 0..5 {
        rig.wheels = [rig.wheels[0] + turn, rig.wheels[1] + turn];
        rig.step().await;
    }
    let mut seen: Vec<(f64, f64)> = Vec::new();
    while seen
        .last()
        .is_none_or(|(distance, _)| (distance - 0.5).abs() > 1e-9)
    {
        let feedback = tokio::time::timeout(ANSWER_BUDGET, goal.on_next_feedback())
            .await
            .expect("feedback within the budget")
            .expect("feedback receive");
        seen.push((feedback.distance_remaining_m, feedback.angle_remaining_rad));
    }
    assert_close(seen[0].0, 1.0, "the distance before the roll");
    assert!(
        seen.windows(2).all(|pair| pair[1].0 <= pair[0].0 + 1e-9),
        "what remains of the distance only shrinks as the base rolls: {seen:?}"
    );
    for (_, yaw) in &seen {
        assert_close(*yaw, PI / 2.0, "the yaw while the base rolls ahead");
    }
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn move_base_completes_at_its_planned_end_at_rest_within_its_tolerances() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    // A slow move to tolerances of 1 mm and 2 mrad: its legs' speeds, scaled
    // to 40 s, and the approach to each end lie below the deadbands, so the
    // move closes each leg at just above the deadband. The wheels follow
    // every setpoint, and the base rests within the tolerances well before
    // the planned end.
    let request = move_base::GoalRequestData {
        target: target(0.05, 0.0, 0.3),
        duration_s: 40.0,
        settle_position_tolerance_m: 0.001,
        settle_orientation_tolerance_rad: 0.002,
        settle_timeout_s: 5.0,
    };
    let goal = rig.send_move(&request).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    let planned_end = rig.at(request.duration_s);
    while rig.now_ns < planned_end - STEP_NS {
        rig.step_following().await;
    }
    // A step before its planned end the move still holds the base.
    let refused = refusal_of_drive(&rig, drive_goal(0.1, 1.0)).await;
    assert_eq!(
        refused.kind,
        drive_base::FailureKind::PreconditionFailed,
        "{}",
        refused.message
    );
    let result = rig.follow_until_result(&goal).await;
    let move_base::ResultOutcome::Completed(data) = result.outcome else {
        panic!("the move completes: {:?}", result.outcome);
    };
    let distance = (data.final_pose.position[0] - request.target.position[0])
        .hypot(data.final_pose.position[1] - request.target.position[1]);
    assert!(
        distance <= request.settle_position_tolerance_m,
        "the base rests {distance} m from the target"
    );
    let yaw = (yaw_of_quaternion(data.final_pose.orientation)
        - yaw_of_quaternion(request.target.orientation))
    .abs();
    assert!(
        yaw <= request.settle_orientation_tolerance_rad,
        "the base rests {yaw} rad from the target yaw"
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_robot_stop_ends_a_running_move_base_cancelled() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    let goal = rig.send_move(&move_goal(1.0, 0.5, PI / 2.0, 5.0)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    rig.step_to(rig.at(0.25)).await;
    // The leader streams to the base through the move and the stop: the move
    // drops its setpoints, and the stop holds the base against it.
    rig.lead([0.2, 0.0, 0.0], 0.0).await;
    let streamed = async {
        loop {
            let refused = refusal_of_drive(&rig, drive_goal(0.1, 1.0)).await;
            match refused.kind {
                drive_base::FailureKind::PreconditionFailed => continue,
                drive_base::FailureKind::LeaderStreaming => break,
                other => panic!("{other:?}: {}", refused.message),
            }
        }
    };
    tokio::time::timeout(ANSWER_BUDGET, streamed)
        .await
        .expect("the node takes the leader's setpoint");
    assert_eq!(
        rig.stop("the test stops it").await,
        vec!["base".to_string()]
    );
    let stopped_ns = rig.now_ns;
    while rig.now_ns < rig.at(1.25) {
        rig.lead([0.2, 0.0, 0.0], 0.0).await;
        rig.step().await;
        assert_eq!(
            follow_state(&mut rig).await,
            "robot_stop",
            "{} ns after the stop",
            rig.now_ns - stopped_ns
        );
    }
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    let move_base::ResultOutcome::Cancelled { reason, data } = result.outcome else {
        panic!("a stopped move ends cancelled: {:?}", result.outcome);
    };
    assert_eq!(reason, "stopped: the test stops it");
    // The wheels never moved, so the base rests where it started.
    assert_eq!(
        (
            data.final_pose.position,
            yaw_of_quaternion(data.final_pose.orientation)
        ),
        ([0.0, 0.0, 0.0], 0.0)
    );
    rig.shutdown().await;
}

// A stopped goal whose wheels go on drifting, faster than the rest velocity,
// as on a slope, ends cancelled at its settle timeout after the stop, and
// the base takes the next goal.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stopped_goal_whose_base_drifts_ends_cancelled_at_its_settle_timeout() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    let goal = rig.send_drive(&drive_goal(0.3, 5.0)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    rig.step_to(rig.at(0.3)).await;
    assert_eq!(
        rig.stop("the test stops it").await,
        vec!["base".to_string()]
    );
    // The wheels drift ahead 0.02 m/s past the rest velocity, at
    // REST_LINEAR_M_S + 0.02 m/s, whatever the setpoints say.
    let drift = (REST_LINEAR_M_S + 0.02) / WHEEL_RADIUS_M * DT;
    // drive_goal gives the base 1 s to rest: the goal still runs 0.9 s
    // after the stop, and has ended 1.1 s after it.
    let stopped_at = rig.now_ns;
    while rig.now_ns < stopped_at + 900_000_000 {
        rig.wheels = [rig.wheels[0] + drift, rig.wheels[1] + drift];
        rig.step().await;
    }
    let held = refusal_of_drive(&rig, drive_goal(0.1, 1.0)).await;
    assert_eq!(
        held.kind,
        drive_base::FailureKind::PreconditionFailed,
        "{}",
        held.message
    );
    while rig.now_ns < stopped_at + 1_100_000_000 {
        rig.wheels = [rig.wheels[0] + drift, rig.wheels[1] + drift];
        rig.step().await;
    }
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    let drive_base::ResultOutcome::Cancelled { reason, .. } = result.outcome else {
        panic!("a stopped drive ends cancelled: {:?}", result.outcome);
    };
    assert_eq!(reason, "stopped: the test stops it");
    let next = rig.send_drive(&drive_goal(0.1, 1.0)).await;
    assert!(
        next.accepted,
        "the base takes the next goal: {:?}",
        next.refusal
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stopped_goal_holds_the_base_until_it_rests() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    let goal = rig.send_drive(&drive_goal(0.3, 5.0)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    rig.step_to(rig.at(0.3)).await;
    assert_eq!(
        rig.stop("the test stops it").await,
        vec!["base".to_string()]
    );
    // A second stop ends no goal, the first having ended it, and finds no
    // leader streaming: it names nothing.
    let second = rig.stop("the test stops it").await;
    assert!(second.is_empty(), "{second:?}");
    // From 0.3 m/s the base slows at its deceleration limit, and the goal
    // holds the base until it rests, at 0.45 s.
    for (seconds, speed) in [(0.35, 0.3 - SLOW_DOWN), (0.4, 0.3 - 2.0 * SLOW_DOWN)] {
        assert_speeds(&mut rig, &[(seconds, speed)]).await;
        let refused = refusal_of_drive(&rig, drive_goal(0.1, 1.0)).await;
        assert_eq!(
            refused.kind,
            drive_base::FailureKind::PreconditionFailed,
            "{}",
            refused.message
        );
    }
    assert_speeds(&mut rig, &[(0.45, 0.0)]).await;
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    let drive_base::ResultOutcome::Cancelled { reason, .. } = result.outcome else {
        panic!("a stopped drive ends cancelled: {:?}", result.outcome);
    };
    assert_eq!(reason, "stopped: the test stops it");
    let next = rig.send_drive(&drive_goal(0.1, 1.0)).await;
    assert!(
        next.accepted,
        "a goal sent on the result finds the base free: {:?}",
        next.refusal
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_caller_cancel_ends_the_goal_cancelled_at_rest() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    let goal = rig.send_drive(&drive_goal(0.3, 5.0)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    rig.step_to(rig.at(0.3)).await;
    let refused_move = rig
        .send_move(&move_goal(1.0, 0.0, 0.0, 1.0))
        .await
        .refusal
        .expect("a move is refused while a drive runs");
    assert_eq!(
        refused_move.kind,
        move_base::FailureKind::PreconditionFailed
    );
    let cancel = goal
        .cancel_goal(ANSWER_BUDGET)
        .await
        .expect("the cancel answers");
    assert_eq!(cancel, drive_base::CancelState::Signalled);
    // From 0.3 m/s the base comes to rest at its deceleration limit, and
    // the goal ends then.
    assert_speeds(
        &mut rig,
        &[
            (0.35, 0.3 - SLOW_DOWN),
            (0.4, 0.3 - 2.0 * SLOW_DOWN),
            (0.45, 0.0),
        ],
    )
    .await;
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    let drive_base::ResultOutcome::Cancelled { reason, data } = result.outcome else {
        panic!("a cancelled drive ends cancelled: {:?}", result.outcome);
    };
    assert_eq!(reason, "cancelled by its caller");
    // The wheels never moved, so the base rests where it started.
    assert_eq!(
        (
            data.final_pose.position,
            yaw_of_quaternion(data.final_pose.orientation)
        ),
        ([0.0, 0.0, 0.0], 0.0)
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_base_rests_once_more_than_half_a_second_passes_after_the_last_setpoint() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    rig.lead_until_it_drives([0.2, 0.0, 0.0], 0.0).await;
    // Up at the acceleration limit, held through the step 0.5 s after the
    // setpoint, and from the next step down at the deceleration limit, to
    // rest.
    let held = 4.0 * SPEED_UP;
    let expected = [
        SPEED_UP,
        2.0 * SPEED_UP,
        3.0 * SPEED_UP,
        held,
        held,
        held,
        held,
        held,
        held,
        held,
        held - SLOW_DOWN,
        held - 2.0 * SLOW_DOWN,
    ];
    for (step, expected) in (1..).zip(expected) {
        let at = rig.origin_ns + step * STEP_NS;
        rig.step_to(at).await;
        let setpoint = rig.wheel_setpoint_at(at).await;
        assert_close(
            forward_speed(&setpoint),
            expected,
            &format!("the speed of step {step}"),
        );
    }
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_move_already_within_its_tolerances_completes_with_no_settle_time() {
    // The target is where the base rests: the plan takes no time, so the
    // planned end is the admission, and a settle timeout of 0 ends there.
    let mut rig = Rig::start([0.0, 0.0]).await;
    let goal = rig.send_move(&move_goal(0.0, 0.0, 0.0, 0.0)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    rig.step_to(rig.at(0.25)).await;
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    let move_base::ResultOutcome::Completed(data) = result.outcome else {
        panic!(
            "a move that rests on its target completes: {:?}",
            result.outcome
        );
    };
    assert_eq!(
        (
            data.final_pose.position,
            yaw_of_quaternion(data.final_pose.orientation)
        ),
        ([0.0, 0.0, 0.0], 0.0)
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn move_base_times_out_after_its_settle_timeout() {
    // Wheels held still: the base never reaches the target.
    let mut rig = Rig::start([0.0, 0.0]).await;
    let goal = rig.send_move(&move_goal(1.0, 0.0, 0.0, 0.625)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    // The fastest 1 m drive takes 2.375 s: 0.5 s up at 1 m/s^2, 0.25 s down
    // at 2 m/s^2 and the rest at 0.5 m/s. Its settle ends 0.625 s later, at
    // 3 s: the move drives the base at full speed through that step, which
    // fails it. From the next step the base comes to rest from 0.5 m/s at
    // its deceleration limit, by 3.5 s.
    assert_speeds(
        &mut rig,
        &[
            (3.0, MAX_LINEAR_SPEED_M_S),
            (3.05, MAX_LINEAR_SPEED_M_S - SLOW_DOWN),
        ],
    )
    .await;
    rig.step_to(rig.at(3.5)).await;
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    let move_base::ResultOutcome::Failed(failure) = result.outcome else {
        panic!("an unsettled move fails: {:?}", result.outcome);
    };
    assert_eq!(failure.kind, move_base::FailureKind::Timeout);
    assert_eq!(
        failure.message,
        "the base stands 1.000 m and 0.000 rad from the target at the end of its settle timeout"
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn odometry_follows_the_measured_wheels() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    let geometry =
        WheelGeometry::parse(WHEEL_SEPARATION_M, WHEEL_RADIUS_M, WHEEL_RADIUS_M).unwrap();
    let mut expected = Odometry::new(geometry, NonZeroUsize::MIN);
    expected.anchor_wheels(0.0, 0.0).unwrap();
    // The base stands: it rests, and nothing of a limb lies outside its
    // footprint, the base having no limb.
    let resting = rig.step().await;
    assert_eq!(resting.mode, "standing");
    assert!(resting.parts_within_footprint);
    // Straight ahead for 10 steps, then a left turn of more than half a turn
    // in place, so the heading wraps.
    let moves = std::iter::repeat_n([0.5, 0.5], 10).chain(std::iter::repeat_n([-0.5, 0.5], 40));
    let mut last: Option<BaseState> = None;
    for [left, right] in moves {
        rig.wheels = [rig.wheels[0] + left, rig.wheels[1] + right];
        expected.update_from_positions(rig.wheels[0], rig.wheels[1], DT);
        last = Some(rig.step().await);
    }
    let last = last.expect("the wheels moved");
    let heading = expected.heading();
    assert!(heading > PI, "the turn passes half a turn: {heading}");
    // The state of each step reports that step's measurement.
    for (what, actual, integrated) in [
        ("x", last.pose.position[0], expected.x()),
        ("y", last.pose.position[1], expected.y()),
        // The heading wrapped by whole turns into (-pi, pi].
        (
            "yaw",
            yaw_of_quaternion(last.pose.orientation),
            heading - 2.0 * PI * (heading / (2.0 * PI)).round(),
        ),
        (
            "linear_velocity's x",
            last.twist.linear_velocity[0],
            expected.linear(),
        ),
        (
            "yaw_rate",
            last.twist.angular_velocity[2],
            expected.angular(),
        ),
    ] {
        assert_close(actual, integrated, what);
    }
    assert_eq!(
        (
            last.pose.position[2],
            last.twist.linear_velocity[1],
            last.twist.linear_velocity[2]
        ),
        (0.0, 0.0, 0.0)
    );
    assert_eq!(last.mode, "moving");
    assert_eq!(ns(last.timestamp), rig.now_ns);
    // The odometry started before the wheels moved, and stays started.
    assert_eq!(
        last.odometry_start_timestamp,
        resting.odometry_start_timestamp
    );
    // The transform odom to base_footprint at the state's timestamp places
    // the base where the state does, its start the odometry's.
    let odom = rig.transforms_at(rig.now_ns, &ROBOT_FRAME).await;
    let [odom_transform] = odom.transforms.as_slice() else {
        panic!("one transform: {:?}", odom.transforms);
    };
    assert_eq!(odom_transform.frame, "odom");
    for (axis, (actual, expected)) in odom_transform
        .position
        .iter()
        .zip(last.pose.position)
        .enumerate()
    {
        assert_close(*actual, expected, &format!("odom's axis {axis}"));
    }
    assert_same_rotation(
        &odom_transform.orientation,
        rotation_about([0.0, 0.0, 1.0], yaw_of_quaternion(last.pose.orientation)),
        "odom's heading",
    );
    assert_eq!(odom.start_timestamp, last.odometry_start_timestamp);
    // drive_states carries the same measurement, with the same stamps, to
    // the leader, which the base would follow.
    let drive = rig.drive_state_at(rig.now_ns).await;
    assert_eq!(ns(drive.timestamp), rig.now_ns);
    assert_eq!(drive.follow_state, "following");
    assert_eq!(
        (
            drive.odometry_start_timestamp,
            drive.pose.position,
            yaw_of_quaternion(drive.pose.orientation),
            drive.twist.linear_velocity,
            drive.twist.angular_velocity[2],
            drive.mode.as_str(),
            drive.parts_within_footprint
        ),
        (
            last.odometry_start_timestamp,
            last.pose.position,
            yaw_of_quaternion(last.pose.orientation),
            last.twist.linear_velocity,
            last.twist.angular_velocity[2],
            last.mode.as_str(),
            last.parts_within_footprint
        )
    );
    rig.shutdown().await;
}

// The transforms of the wheel joints at each measurement: each wheel's link
// at its joint's origin, turned about its axle by its measured angle, and
// started with the node, so an odometry that starts over leaves their start.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_wheel_joints_transforms_follow_each_measurement_and_start_with_the_node() {
    let before_boot = SystemTime::now();
    let mut rig = Rig::start([0.0, 0.0]).await;
    rig.wheels = [0.5, -0.25];
    rig.now_ns += STEP_NS;
    let turned = rig.report_wheels().await;
    rig.tick().await;
    let origins = [[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]];
    for (wheel, (transform, origin)) in turned.transforms.iter().zip(origins).enumerate() {
        assert_eq!(transform.frame, "base_link", "{}", WHEEL_LINKS[wheel]);
        for (axis, (actual, expected)) in transform.position.iter().zip(origin).enumerate() {
            assert_close(
                *actual,
                expected,
                &format!("{} axis {axis}", WHEEL_LINKS[wheel]),
            );
        }
        assert_same_rotation(
            &transform.orientation,
            rotation_about([0.0, 1.0, 0.0], rig.wheels[wheel]),
            WHEEL_LINKS[wheel],
        );
    }
    let started = turned.start_timestamp;
    assert!(
        before_boot <= started && started <= SystemTime::now(),
        "the wheels' transforms start with the node"
    );
    // A measurement stamped before the last: the wheels' positions started
    // over, and so does the odometry; the wheels' transforms keep their
    // start.
    let before = rig.state_at(rig.now_ns).await;
    let earlier = rig.now_ns - STEP_NS;
    rig.wheels = [20.0, 20.0];
    rig.publish_wheels_stamped(earlier).await;
    let restarted = rig.transforms_at(earlier, &WHEEL_LINKS).await;
    assert_eq!(restarted.start_timestamp, started);
    let odom = rig.transforms_at(earlier, &ROBOT_FRAME).await;
    assert_ne!(odom.start_timestamp, before.odometry_start_timestamp);
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn held_wheels_wind_up_no_more_than_the_anchor_bound() {
    // Wheels held where they stand, as by a wall: each setpoint runs on from
    // the last while the measured angle lies within the anchor bound of it,
    // and starts from the measured angle past it, so no setpoint leads the
    // held wheel by more than the bound and one step.
    let measured = [1.0, -2.0];
    let mut rig = Rig::start(measured).await;
    let goal = rig.send_drive(&drive_goal(0.3, 2.0)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    let mut reanchored = false;
    let mut last_lead = 0.0;
    for step in 1..=20 {
        let at = START_NS + step * STEP_NS;
        rig.step_to(at).await;
        let setpoint = rig.wheel_setpoint_at(at).await;
        let wheels = setpoint
            .positions
            .iter()
            .zip(setpoint.velocities)
            .zip(measured);
        for (wheel, ((position, rate), measured)) in wheels.enumerate() {
            let lead = position - measured;
            assert!(
                lead <= ANCHOR_BOUND_RAD + rate * DT + 1e-9,
                "wheel {wheel} at step {step} leads by {lead} rad"
            );
            if wheel == 0 {
                reanchored |= last_lead > ANCHOR_BOUND_RAD && (lead - rate * DT).abs() < 1e-9;
                last_lead = lead;
            }
        }
    }
    assert!(
        reanchored,
        "a setpoint past the bound starts from the measured angle"
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_zero_velocity_holds_the_wheels_at_their_angle_of_the_stop_while_they_drift() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    let goal = rig.send_drive(&drive_goal(0.2, 0.3)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    // The drive completes at 0.5 s; the setpoint of that step is its last
    // one that moves the base.
    rig.step_to(rig.at(0.5)).await;
    // The wheels drift forward 0.02 rad a step, as on a slope, faster than
    // the deadband: the drive waits for the base to rest. The first setpoint
    // of zero velocity holds the wheels at their angle measured at that
    // step, and every one after it holds them there.
    let mut stopped_at = None;
    for step in 1..=4 {
        rig.wheels = [rig.wheels[0] + 0.02, rig.wheels[1] + 0.02];
        rig.step().await;
        let held = rig.wheel_setpoint_at(rig.now_ns).await;
        assert_eq!(held.velocities, [0.0, 0.0], "step {step} of the drift");
        let stop = *stopped_at.get_or_insert(rig.wheels);
        assert_eq!(held.positions, stop, "step {step} of the drift");
    }
    // The wheels hold. The base rests once every base_states of the last
    // 0.2 s reads it still: four steps after the drift the last drift step
    // still lies within them, and the drive holds the base.
    for step in 1..=4 {
        rig.step().await;
        let refused = refusal_of_drive(&rig, drive_goal(0.1, 1.0)).await;
        assert_eq!(
            refused.kind,
            drive_base::FailureKind::PreconditionFailed,
            "step {step} after the drift: {}",
            refused.message
        );
    }
    rig.step().await;
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    assert!(
        matches!(result.outcome, drive_base::ResultOutcome::Completed(_)),
        "{:?}",
        result.outcome
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_robot_stop_ends_every_base_goal() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    let idle = rig.stop("operator").await;
    assert!(idle.is_empty(), "no goal in flight, no leader: {idle:?}");

    let goal = rig.send_drive(&drive_goal(0.3, 5.0)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    rig.step_to(rig.at(0.2)).await;
    assert_eq!(rig.stop("operator").await, vec!["base".to_string()]);
    rig.step_to(rig.at(0.7)).await;
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    let drive_base::ResultOutcome::Cancelled { reason, .. } = result.outcome else {
        panic!("a stopped drive ends cancelled: {:?}", result.outcome);
    };
    assert_eq!(reason, "stopped: operator");
    // A goal after the stop runs as usual, as robot_stop states.
    let next = rig.send_drive(&drive_goal(0.3, 1.0)).await;
    assert!(
        next.accepted,
        "a goal after the stop runs: {:?}",
        next.refusal
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn drive_base_holds_its_velocity_for_its_hold_duration_from_reaching_it() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    let goal = rig.send_drive(&drive_goal(0.2, 0.3)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    // Reached at step 4, held for 0.3 s, and down from step 10, at rest at
    // step 11.
    let held = 4.0 * SPEED_UP;
    let expected = [
        SPEED_UP,
        2.0 * SPEED_UP,
        3.0 * SPEED_UP,
        held,
        held,
        held,
        held,
        held,
        held,
        held - SLOW_DOWN,
        held - 2.0 * SLOW_DOWN,
    ];
    for (step, expected) in (1..).zip(expected) {
        let at = START_NS + step * STEP_NS;
        rig.step_to(at).await;
        let setpoint = rig.wheel_setpoint_at(at).await;
        assert_close(
            forward_speed(&setpoint),
            expected,
            &format!("the speed of step {step}"),
        );
    }
    rig.step_to(rig.at(0.8)).await;
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    assert!(
        matches!(result.outcome, drive_base::ResultOutcome::Completed(_)),
        "{:?}",
        result.outcome
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_fails_when_the_wheels_fall_silent_and_the_ramp_continues_from_the_last_setpoint() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    let goal = rig.send_drive(&drive_goal(0.2, 5.0)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    // At its acceleration limit the base reaches 0.2 m/s in four steps.
    rig.step_to(rig.at(0.5)).await;
    let reached = rig.wheel_setpoint_at(rig.at(0.5)).await.positions[0];
    // The steps of the 0.5 s within which the wheels still report, then two
    // past it.
    let mut silent = Vec::new();
    for _ in 0..REPORTING_SILENT_STEPS + 2 {
        rig.step_silently().await;
        silent.push(rig.wheel_setpoint_at(rig.now_ns).await);
    }
    // While the wheels report, each setpoint at 0.2 m/s runs on from the
    // last while the measured angle, 0, lies within the anchor bound of it,
    // and starts from 0 past it. From the first silent step the goal fails
    // and the base slows at its deceleration limit, each setpoint continuing
    // from the last: one step at 0.1 m/s, then a hold at rest.
    let fresh = wheel_step(0.2);
    let reporting: Vec<f64> = std::iter::successors(Some(reached), |last| {
        Some(if *last <= ANCHOR_BOUND_RAD {
            last + fresh
        } else {
            fresh
        })
    })
    .skip(1)
    .take(REPORTING_SILENT_STEPS)
    .collect();
    let slowed = reporting[REPORTING_SILENT_STEPS - 1] + wheel_step(0.2 - SLOW_DOWN);
    let expected = reporting
        .into_iter()
        .map(|position| (0.2, position))
        .chain([(0.2 - SLOW_DOWN, slowed), (0.0, slowed)]);
    for (step, (setpoint, (speed, position))) in (1..).zip(silent.iter().zip(expected)) {
        assert_close(
            forward_speed(setpoint),
            speed,
            &format!("the speed of silent step {step}"),
        );
        for (wheel, actual) in setpoint.positions.iter().enumerate() {
            assert_close(
                *actual,
                position,
                &format!("wheel {wheel} at silent step {step}"),
            );
        }
    }
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    let drive_base::ResultOutcome::Failed(failure) = result.outcome else {
        panic!("a goal on silent wheels fails: {:?}", result.outcome);
    };
    assert_eq!(failure.kind, drive_base::FailureKind::ResourceUnavailable);
    let refused = refusal_of_drive(&rig, drive_goal(0.1, 1.0)).await;
    assert_eq!(refused.kind, drive_base::FailureKind::ResourceUnavailable);

    // At rest, the base sends nothing; the wheels report again, a goal runs,
    // and the next setpoint the base sends is that goal's first, anchored
    // on the measured angle.
    rig.step_silently().await;
    rig.step().await;
    let next = rig.send_drive(&drive_goal(0.1, 1.0)).await;
    assert!(next.accepted, "{:?}", next.refusal);
    rig.step().await;
    let resumed = rig.next_wheel_setpoint().await;
    assert_eq!(
        ns(resumed.timestamp),
        rig.now_ns,
        "a wheel setpoint between the rest and the next goal"
    );
    assert_close(
        forward_speed(&resumed),
        SPEED_UP,
        "the next goal's first speed",
    );
    for position in resumed.positions {
        assert_close(
            position,
            wheel_step(SPEED_UP),
            "the next goal's first wheel angle",
        );
    }
    rig.shutdown().await;
}

// The wheels' silence holds the base, which comes to rest at its
// deceleration limit. The leader streamed as the silence started, so the hold
// lasts on once the wheels report again, until more than 0.5 s pass with no
// setpoint, as drive_link states; the base takes none of the setpoints that
// arrive meanwhile.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_silence_holds_the_base_from_its_streaming_leader_until_the_leader_pauses() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    rig.lead_until_it_drives([0.2, 0.0, 0.0], 0.0).await;
    // The leader streams every step, and the base reaches 0.2 m/s in four.
    for _ in 0..4 {
        rig.lead([0.2, 0.0, 0.0], 0.0).await;
        rig.step().await;
    }
    // The steps of the 0.5 s within which the wheels still report, then two
    // past it, the leader streaming through them: from the first silent
    // step the base slows at its deceleration limit to rest.
    let mut speeds = Vec::new();
    for _ in 0..REPORTING_SILENT_STEPS + 2 {
        rig.lead([0.2, 0.0, 0.0], 0.0).await;
        rig.step_silently().await;
        speeds.push(forward_speed(&rig.wheel_setpoint_at(rig.now_ns).await));
    }
    let expected = std::iter::repeat_n(0.2, REPORTING_SILENT_STEPS).chain([0.2 - SLOW_DOWN, 0.0]);
    for (step, (speed, expected)) in (1..).zip(speeds.iter().zip(expected)) {
        assert_close(
            *speed,
            expected,
            &format!("the speed of silent step {step}"),
        );
    }
    // The wheels report again, the leader streaming on: drive_states names
    // the silence's hold.
    for step in 1..=3 {
        rig.lead([0.2, 0.0, 0.0], 0.0).await;
        rig.step().await;
        assert_eq!(follow_state(&mut rig).await, "base_mode", "step {step}");
    }
    // The leader pauses: the hold lasts through 0.5 s after its last
    // setpoint, sent at the step before the clock's, and ends after that.
    let last_sent = rig.now_ns - STEP_NS;
    while rig.now_ns < last_sent + LEADER_STREAMING_WINDOW_NS {
        rig.step().await;
        assert_eq!(follow_state(&mut rig).await, "base_mode", "in the pause");
    }
    loop {
        rig.step().await;
        if follow_state(&mut rig).await == "following" {
            break;
        }
        assert!(
            rig.now_ns < last_sent + 2 * LEADER_STREAMING_WINDOW_NS,
            "the hold outlasts the leader's pause"
        );
    }
    // The base followed none of the setpoints that came during the hold: its
    // next wheel setpoint that moves is the first step of the ramp toward the
    // leader's next setpoint.
    rig.lead_until_it_drives([0.2, 0.0, 0.0], 0.0).await;
    rig.step().await;
    let resumed = loop {
        let setpoint = rig.next_wheel_setpoint().await;
        if forward_speed(&setpoint) != 0.0 {
            break setpoint;
        }
    };
    assert_eq!(ns(resumed.timestamp), rig.origin_ns + STEP_NS);
    assert_close(
        forward_speed(&resumed),
        SPEED_UP,
        "the first speed after the hold",
    );
    rig.shutdown().await;
}

// A restarted backbone, a new process, starts its odometry at the origin on
// its first measurement: base_states gives a new odometry_start_timestamp,
// which the transform odom to base_footprint takes as its start.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restarted_backbone_starts_a_new_odometry() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    rig.wheels = [1.0, 1.0];
    let before = rig.step().await;
    rig.shutdown().await;
    let mut restarted = Rig::boot([3.0, 3.0]).await;
    restarted.now_ns = START_NS + 10 * STEP_NS;
    restarted.report_wheels().await;
    restarted.tick().await;
    let after_restart = restarted.state_at(restarted.now_ns).await;
    assert_ne!(
        after_restart.odometry_start_timestamp,
        before.odometry_start_timestamp
    );
    assert_eq!(
        (
            after_restart.pose.position,
            yaw_of_quaternion(after_restart.pose.orientation)
        ),
        ([0.0, 0.0, 0.0], 0.0)
    );
    let odom = restarted
        .transforms_at(restarted.now_ns, &ROBOT_FRAME)
        .await;
    assert_eq!(odom.start_timestamp, after_restart.odometry_start_timestamp);
    restarted.shutdown().await;
}

// Setpoints a leader sends while the wheels are silent start its stream:
// once the wheels report again, a goal is refused with leader_streaming
// through 0.5 s after the node took the last of them, and admitted after
// that. A drive that moves the base into the silence fails, and the node
// streams a wheel setpoint at each silent step until the base rests, so the
// leader sends at a step whose time the node has shown, and the node takes
// the setpoint at that time or later. It has taken it once a goal is refused
// for the stream.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn setpoints_sent_while_the_wheels_are_silent_start_a_stream_that_ends_more_than_half_a_second_after_the_last()
 {
    let mut rig = Rig::start([0.0, 0.0]).await;
    let drive = rig.send_drive(&drive_goal(0.2, 5.0)).await;
    assert!(drive.accepted, "{:?}", drive.refusal);
    for _ in 0..4 {
        rig.step_following().await;
    }
    for _ in 0..REPORTING_SILENT_STEPS + 1 {
        rig.step_silently().await;
        rig.wheel_setpoint_at(rig.now_ns).await;
    }
    let shown_ns = rig.now_ns;
    rig.lead([0.2, 0.0, 0.0], 0.0).await;
    rig.step().await;
    rig.until_the_leader_stream_refuses_a_goal().await;
    let taken_by_ns = rig.now_ns;
    // The node took the setpoint between `shown_ns` and `taken_by_ns`: the
    // stream refuses a goal 0.5 s after the earliest of those, and admits one
    // a step past 0.5 s after the latest.
    rig.step_to(shown_ns + LEADER_STREAMING_WINDOW_NS).await;
    let still = refusal_of_drive(&rig, drive_goal(0.1, 1.0)).await;
    assert_eq!(
        still.kind,
        drive_base::FailureKind::LeaderStreaming,
        "{}",
        still.message
    );
    rig.step_to(taken_by_ns + LEADER_STREAMING_WINDOW_NS + STEP_NS)
        .await;
    let free = rig.send_drive(&drive_goal(0.1, 1.0)).await;
    assert!(free.accepted, "the stream ended: {:?}", free.refusal);
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn odometry_takes_in_the_motion_measured_across_a_silence() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    let before = rig.step().await;
    for _ in 0..REPORTING_SILENT_STEPS + 2 {
        rig.step_silently().await;
    }
    // The wheels turned 1 rad through the silence: the base rolled one
    // wheel radius, and its odometry goes on from where it started.
    rig.wheels = [1.0, 1.0];
    let state = rig.step().await;
    assert_close(
        state.pose.position[0],
        WHEEL_RADIUS_M,
        "the roll across the silence",
    );
    assert_eq!(
        state.odometry_start_timestamp,
        before.odometry_start_timestamp
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn wheels_that_start_over_fail_the_goal_and_start_the_odometry_over() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    rig.wheels = [1.0, 1.0];
    let before = rig.step().await;
    let goal = rig.send_drive(&drive_goal(0.2, 5.0)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    rig.step().await;
    // A measurement stamped before the last one: the wheels' positions
    // started over, as after a restart of the simulator.
    rig.wheels = [20.0, 20.0];
    let earlier = rig.now_ns - 2 * STEP_NS;
    rig.publish_wheels_stamped(earlier).await;
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    let drive_base::ResultOutcome::Failed(failure) = result.outcome else {
        panic!("a goal across a start-over fails: {:?}", result.outcome);
    };
    assert_eq!(failure.kind, drive_base::FailureKind::ResourceUnavailable);
    assert_eq!(failure.message, "the base's odometry started over");
    let restarted = rig.state_at(earlier).await;
    assert_eq!(ns(restarted.timestamp), earlier);
    assert_ne!(
        restarted.odometry_start_timestamp,
        before.odometry_start_timestamp
    );
    assert_eq!(
        (
            restarted.pose.position,
            yaw_of_quaternion(restarted.pose.orientation)
        ),
        ([0.0, 0.0, 0.0], 0.0)
    );
    let odom = rig.transforms_at(earlier, &ROBOT_FRAME).await;
    assert_eq!(odom.start_timestamp, restarted.odometry_start_timestamp);
    // The new odometry integrates the measurement after it.
    rig.wheels = [20.5, 20.5];
    let rolled = rig.step().await;
    assert_close(
        rolled.pose.position[0],
        0.5 * WHEEL_RADIUS_M,
        "the roll after the start-over",
    );
    assert_eq!(
        rolled.odometry_start_timestamp,
        restarted.odometry_start_timestamp
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn wheels_that_start_over_at_a_later_stamp_fail_the_goal_and_start_the_odometry_over() {
    let mut rig = Rig::start([50.0, 50.0]).await;
    rig.wheels = [50.5, 50.5];
    let before = rig.step().await;
    let goal = rig.send_drive(&drive_goal(0.2, 5.0)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    rig.step().await;
    // The wheels report 0 a step later, farther from their last angles than
    // a wheel turns in a step within the base's limits: their positions
    // started over, as after a restart of the wheels' driver.
    rig.wheels = [0.0, 0.0];
    let restarted = rig.step().await;
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    let drive_base::ResultOutcome::Failed(failure) = result.outcome else {
        panic!("a goal across a start-over fails: {:?}", result.outcome);
    };
    assert_eq!(failure.kind, drive_base::FailureKind::ResourceUnavailable);
    assert_eq!(failure.message, "the base's odometry started over");
    assert_ne!(
        restarted.odometry_start_timestamp,
        before.odometry_start_timestamp
    );
    assert_eq!(
        (
            restarted.pose.position,
            yaw_of_quaternion(restarted.pose.orientation)
        ),
        ([0.0, 0.0, 0.0], 0.0)
    );
    let odom = rig.transforms_at(rig.now_ns, &ROBOT_FRAME).await;
    assert_eq!(odom.start_timestamp, restarted.odometry_start_timestamp);
    rig.shutdown().await;
}

// At each start of the odometry the limiter starts over at rest: the leader
// streams 0.2 m/s, the wheels start over, and the base holds them where they
// stand and sends nothing more until the leader's next setpoint, from which
// it speeds up from rest at its acceleration limit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn wheels_that_start_over_under_a_streaming_leader_speed_up_again_from_rest() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    rig.lead_until_it_drives([0.2, 0.0, 0.0], 0.0).await;
    assert_speeds(&mut rig, &[(0.2, 0.2)]).await;
    // The wheels report 20 rad a step later, farther from their last angles
    // than a wheel turns in a step within the base's limits: their positions
    // started over, the leader's setpoint still live.
    rig.wheels = [20.0, 20.0];
    rig.step().await;
    let hold = rig.next_wheel_setpoint().await;
    assert_eq!(
        (hold.positions, hold.velocities),
        ([20.0, 20.0], [0.0, 0.0]),
        "the start-over holds the wheels"
    );
    rig.lead_until_it_drives([0.2, 0.0, 0.0], 0.0).await;
    rig.step().await;
    // The first setpoint after the hold that moves the base is the first
    // step of the ramp from rest toward the leader's next setpoint.
    let resumed = loop {
        let setpoint = rig.next_wheel_setpoint().await;
        if forward_speed(&setpoint) != 0.0 {
            break setpoint;
        }
    };
    assert_eq!(ns(resumed.timestamp), rig.origin_ns + STEP_NS);
    assert_close(
        forward_speed(&resumed),
        SPEED_UP,
        "the first speed after the start-over",
    );
    rig.shutdown().await;
}

// A goal in flight when the wheels start over fails, and the hold stops the
// base where it stands: the base comes to rest from no goal, so the next goal
// is admitted at once, and its first setpoint, the next after the hold,
// speeds the base up from rest on the held angles.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn wheels_that_start_over_under_a_goal_take_the_next_goal_at_once_from_rest() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    let goal = rig.send_drive(&drive_goal(0.2, 5.0)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    assert_speeds(&mut rig, &[(0.2, 0.2)]).await;
    // The wheels report 20 rad at the time of the last step, farther from
    // their last angles than a wheel turns: their positions started over, and
    // the new odometry's first report goes out with the clock standing still.
    rig.wheels = [20.0, 20.0];
    rig.report_wheels().await;
    rig.state_at(rig.now_ns).await;
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    let drive_base::ResultOutcome::Failed(failure) = result.outcome else {
        panic!("a goal across a start-over fails: {:?}", result.outcome);
    };
    assert_eq!(failure.message, "the base's odometry started over");
    let hold = rig.next_wheel_setpoint().await;
    assert_eq!(
        (hold.positions, hold.velocities),
        ([20.0, 20.0], [0.0, 0.0])
    );
    let next = rig.send_drive(&drive_goal(0.1, 1.0)).await;
    assert!(
        next.accepted,
        "the held base comes to rest from no goal: {:?}",
        next.refusal
    );
    rig.step().await;
    let first = rig.next_wheel_setpoint().await;
    assert_eq!(
        ns(first.timestamp),
        rig.now_ns,
        "no wheel setpoint between the hold and the next goal's first"
    );
    assert_close(
        forward_speed(&first),
        SPEED_UP,
        "the next goal's first speed",
    );
    for position in first.positions {
        assert_close(
            position,
            20.0 + wheel_step(SPEED_UP),
            "the next goal's first wheel angle",
        );
    }
    rig.shutdown().await;
}

// The wheels start over at a measurement stamped a step past the node's
// clock, whose report goes out once the clock reads its stamp. Until then the
// robot's tree places by the odometry before: a move placed by it in that
// time fails as the loop takes it, before the base steers to old odometry
// coordinates in the new odometry.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_placed_by_the_odometry_before_a_start_over_fails_as_the_loop_takes_it() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    // The wheels roll the base one wheel radius ahead.
    rig.wheels = [1.0, 1.0];
    rig.step().await;
    rig.wheels = [20.0, 20.0];
    let later_ns = rig.now_ns + STEP_NS;
    rig.publish_wheels_stamped(later_ns).await;
    rig.transforms_at(later_ns, &WHEEL_LINKS).await;
    let goal = rig.send_move(&move_goal(1.2, 0.0, 0.0, 1.0)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    let move_base::ResultOutcome::Failed(failure) = result.outcome else {
        panic!(
            "a move placed by the odometry before fails: {:?}",
            result.outcome
        );
    };
    assert_eq!(failure.kind, move_base::FailureKind::ResourceUnavailable);
    assert_eq!(failure.message, "the base's odometry started over");
    // The new odometry's first report goes out once the clock reads its
    // stamp, and a goal placed by it runs.
    rig.now_ns = later_ns;
    rig.tick().await;
    rig.state_at(later_ns).await;
    let next = rig.send_move(&move_goal(1.2, 0.0, 0.0, 1.0)).await;
    assert!(next.accepted, "{:?}", next.refusal);
    rig.step().await;
    assert!(
        forward_speed(&rig.wheel_setpoint_at(rig.now_ns).await) > 0.0,
        "the move placed by the new odometry drives"
    );
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_clock_that_goes_back_fails_the_goal_brings_the_base_to_rest_and_starts_the_odometry_over()
 {
    let mut rig = Rig::start([0.0, 0.0]).await;
    // The wheels roll the base one wheel radius ahead.
    rig.wheels = [1.0, 1.0];
    let before = rig.step().await;
    let goal = rig.send_drive(&drive_goal(0.2, 5.0)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    rig.step().await;
    rig.step().await;
    let driving = rig.wheel_setpoint_at(rig.now_ns).await;
    // The clock goes back, as a restarted simulation's does: the running
    // goal fails.
    rig.now_ns = 2 * STEP_NS;
    rig.tick().await;
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    let drive_base::ResultOutcome::Failed(failure) = result.outcome else {
        panic!(
            "a goal across the clock going back fails: {:?}",
            result.outcome
        );
    };
    assert_eq!(failure.kind, drive_base::FailureKind::ResourceUnavailable);
    assert_eq!(failure.message, "the base's clock went back");
    // The base comes to rest through the limiter: each step's wheel setpoint
    // is no faster than the one before and slower by at most one step of
    // the deceleration, down to 0.
    let mut speeds = vec![forward_speed(&driving)];
    while speeds.last().is_some_and(|speed| *speed > 0.0) {
        assert!(speeds.len() < 100, "the base comes to rest: {speeds:?}");
        let next = rig.wheel_setpoint_at(rig.now_ns).await;
        speeds.push(forward_speed(&next));
        rig.step_silently().await;
    }
    assert!(
        speeds
            .windows(2)
            .all(|pair| pair[1] <= pair[0] && pair[0] - pair[1] <= SLOW_DOWN * (1.0 + 1e-9)),
        "{speeds:?}"
    );
    // The wheels report again at angles of their own: their measurement
    // starts the odometry over at the origin, with a start it never had.
    rig.wheels = [0.0, 0.0];
    rig.publish_wheels().await;
    let state = rig.state_at(rig.now_ns).await;
    assert_eq!(
        (
            state.pose.position,
            yaw_of_quaternion(state.pose.orientation)
        ),
        ([0.0, 0.0, 0.0], 0.0)
    );
    assert_ne!(
        state.odometry_start_timestamp,
        before.odometry_start_timestamp
    );
    // The new epoch holds the wheels where they stand; the next goal runs,
    // and its first setpoint, the next the base sends, anchors on the new
    // angles.
    let hold = rig.next_wheel_setpoint().await;
    assert_eq!((hold.positions, hold.velocities), ([0.0, 0.0], [0.0, 0.0]));
    let next = rig.send_drive(&drive_goal(0.1, 1.0)).await;
    assert!(next.accepted, "{:?}", next.refusal);
    rig.step().await;
    let first = rig.next_wheel_setpoint().await;
    assert_eq!(ns(first.timestamp), rig.now_ns);
    assert_close(
        forward_speed(&first),
        SPEED_UP,
        "the next goal's first speed",
    );
    for position in first.positions {
        assert_close(
            position,
            wheel_step(SPEED_UP),
            "the next goal's first wheel angle",
        );
    }
    rig.shutdown().await;
}

// After the clock went back, the leader's newest setpoint takes the new
// time: the leader streams until more than 0.5 s pass from it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_clock_that_goes_back_keeps_the_leaders_stream_from_its_new_time() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    rig.lead_until_it_drives([0.0, 0.0, 0.0], 0.0).await;
    let back_ns = 2 * STEP_NS;
    rig.now_ns = back_ns;
    rig.tick().await;
    // A goal is refused for the leader until the base starts its timing
    // over, and for the wheels, whose next measurement starts a new
    // odometry, from then on.
    let silent = async {
        loop {
            let refused = refusal_of_drive(&rig, drive_goal(0.1, 1.0)).await;
            match refused.kind {
                drive_base::FailureKind::LeaderStreaming => continue,
                drive_base::FailureKind::ResourceUnavailable => break,
                other => panic!("{other:?}: {}", refused.message),
            }
        }
    };
    tokio::time::timeout(ANSWER_BUDGET, silent)
        .await
        .expect("the base starts its timing over");
    rig.publish_wheels().await;
    rig.state_at(rig.now_ns).await;
    let refused = refusal_of_drive(&rig, drive_goal(0.1, 1.0)).await;
    assert_eq!(
        refused.kind,
        drive_base::FailureKind::LeaderStreaming,
        "the leader streams from the new time on: {}",
        refused.message
    );
    rig.step_to(back_ns + LEADER_STREAMING_WINDOW_NS + STEP_NS)
        .await;
    let goal = rig.send_drive(&drive_goal(0.1, 1.0)).await;
    assert!(
        goal.accepted,
        "more than 0.5 s after the new time, the leader streams no more: {:?}",
        goal.refusal
    );
    rig.shutdown().await;
}

// A goal that fails while the base moves leaves the base coming to rest
// from it: base_motion refuses a goal with precondition_failed until the
// limiter's command reaches 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_after_one_that_failed_is_refused_until_the_base_rests() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    // The wheels hold still: the move's settle ends at 3 s with the base at
    // full speed, which fails it, as in
    // move_base_times_out_after_its_settle_timeout.
    let goal = rig.send_move(&move_goal(1.0, 0.0, 0.0, 0.625)).await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    rig.step_to(rig.at(3.0)).await;
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    assert!(
        matches!(result.outcome, move_base::ResultOutcome::Failed(_)),
        "{:?}",
        result.outcome
    );
    let refused = refusal_of_drive(&rig, drive_goal(0.1, 1.0)).await;
    assert_eq!(
        refused.kind,
        drive_base::FailureKind::PreconditionFailed,
        "{}",
        refused.message
    );
    // Each step slows the base at its deceleration limit, and a goal is
    // refused until the step at which the limiter's command reaches 0.
    let mut steps = 0;
    loop {
        rig.step().await;
        let speed = forward_speed(&rig.wheel_setpoint_at(rig.now_ns).await);
        let next = rig.send_drive(&drive_goal(0.1, 1.0)).await;
        let Some(refused) = next.refusal else {
            assert_eq!(speed, 0.0, "a goal is admitted once the base rests");
            break;
        };
        assert_eq!(
            refused.kind,
            drive_base::FailureKind::PreconditionFailed,
            "{}",
            refused.message
        );
        steps += 1;
        assert!(steps < 100, "the base comes to rest");
    }
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn describe_base_answers_the_limits_and_rest_bounds_of_its_parameters_and_its_models_outline()
{
    let rig = Rig::start([0.0, 0.0]).await;
    let base = describe_base::poll(&rig.harness, ANSWER_BUDGET)
        .await
        .expect("describe_base answers");
    assert_eq!(
        (base.base_name.as_str(), base.odometry_source.as_str()),
        ("base", "measured")
    );
    // A differential base moves along no y: every limit of y is 0.
    let linear = &base.linear_limits;
    assert_eq!(
        [
            linear.min_velocity,
            linear.max_velocity,
            linear.max_acceleration,
            linear.max_deceleration,
            linear.deadband,
            linear.rest_velocity,
        ],
        [
            [-MAX_LINEAR_SPEED_M_S, 0.0],
            [MAX_LINEAR_SPEED_M_S, 0.0],
            [LINEAR_ACCELERATION_M_S2, 0.0],
            [LINEAR_DECELERATION_M_S2, 0.0],
            [LINEAR_DEADBAND_M_S, 0.0],
            [REST_LINEAR_M_S, 0.0],
        ]
    );
    let yaw = &base.angular_limits;
    assert_eq!(
        [
            yaw.min_rate,
            yaw.max_rate,
            yaw.max_acceleration,
            yaw.max_deceleration,
            yaw.deadband,
            yaw.rest_rate,
        ],
        [
            -MAX_ANGULAR_SPEED_RAD_S,
            MAX_ANGULAR_SPEED_RAD_S,
            ANGULAR_ACCELERATION_RAD_S2,
            ANGULAR_DECELERATION_RAD_S2,
            ANGULAR_DEADBAND_RAD_S,
            REST_ANGULAR_RAD_S,
        ]
    );
    // The model's outline: its chassis, 0.78 m by 0.58 m, which holds every
    // other part, in the robot frame at every time.
    assert_eq!(
        (base.footprint.frame.as_str(), base.footprint.timestamp),
        ("base_footprint", None)
    );
    let chassis = [[0.39, -0.29], [0.39, 0.29], [-0.39, 0.29], [-0.39, -0.29]];
    assert_eq!(base.footprint.points.len(), chassis.len());
    for (point, [x, y]) in base.footprint.points.iter().zip(chassis) {
        for (value, expected) in point.position.iter().zip([x, y, 0.0]) {
            assert!((value - expected).abs() < 1e-12, "{:?}", base.footprint);
        }
    }
    rig.shutdown().await;
}

// The base answers describe_base from its model before the robot's
// initializer reports the robot ready, and runs once it does.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_base_describes_itself_before_the_robot_is_ready_and_runs_once_it_is() {
    let mut rig = Rig::boot_unnamed([0.0, 0.0]).await;
    rig.name_model(MODEL)
        .await
        .expect("the node asks for the robot's model");
    rig.serve_description(None)
        .await
        .expect("the node asks for the robot's description");
    let base = describe_base::poll(&rig.harness, ANSWER_BUDGET)
        .await
        .expect("describe_base answers before the robot is ready");
    assert_eq!(base.base_name, "base");
    rig.ready.store(true, std::sync::atomic::Ordering::SeqCst);
    rig.report_first().await;
    rig.shutdown().await;
}

// A model whose URDF gives no differential base ends the node.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_model_with_no_base_ends_the_node() {
    let rig = Rig::boot_unnamed([0.0, 0.0]).await;
    rig.name_model("so101")
        .await
        .expect("the node asks for the robot's model");
    rig.serve_description(None)
        .await
        .expect("the node asks for the robot's description");
    tokio::time::timeout(
        ANSWER_BUDGET,
        rig.harness.node_runner().cancellation_token().cancelled(),
    )
    .await
    .expect("a model of no base ends the node");
    let reason = rig.fault.take().expect("the node raised its fault");
    assert_eq!(
        reason.to_string(),
        "the model the robot's initializer serves"
    );
    assert!(
        std::error::Error::source(reason.as_ref())
            .expect("the model's refusal")
            .to_string()
            .starts_with("model so101 is no differential base this node drives: "),
        "{reason:?}"
    );
    rig.shutdown().await;
}

// A robot whose initializer serves another description of the model ends
// the node: the base is modeled from the URDF the rest of the robot reads.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn another_description_of_the_model_ends_the_node() {
    let rig = Rig::boot_unnamed([0.0, 0.0]).await;
    rig.name_model(MODEL)
        .await
        .expect("the node asks for the robot's model");
    rig.serve_description(Some(&"0".repeat(64)))
        .await
        .expect("the node asks for the robot's description");
    tokio::time::timeout(
        ANSWER_BUDGET,
        rig.harness.node_runner().cancellation_token().cancelled(),
    )
    .await
    .expect("another description ends the node");
    let reason = rig.fault.take().expect("the node raised its fault");
    assert!(
        reason
            .to_string()
            .starts_with("the robot's initializer serves the description 0000"),
        "{reason}"
    );
    rig.shutdown().await;
}

/// The node's next base_states, in the order it published them.
async fn next_state(rig: &mut Rig) -> BaseState {
    tokio::time::timeout(
        ANSWER_BUDGET,
        rig.harness.emitted.base_state_base_states.next(),
    )
    .await
    .expect("the node publishes base_states")
    .expect("base_states receive")
    .expect("base_states open")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn base_states_repeat_the_newest_measurement_every_tenth_of_a_second_until_silence() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    // The wheels fall silent after their measurement at the start. While
    // they still report, every second step of 50 ms publishes that
    // measurement again, its stamp included; past 0.5 s the base publishes
    // nothing.
    for step in 1..=REPORTING_SILENT_STEPS + 2 {
        rig.step_silently().await;
        if step % 2 == 0 && step <= REPORTING_SILENT_STEPS {
            let repeat = next_state(&mut rig).await;
            assert_eq!(ns(repeat.timestamp), START_NS, "the repeat of step {step}");
        }
    }
    // The base's next base_states is the wheels' next measurement: no
    // repeat came between.
    rig.now_ns += STEP_NS;
    rig.report_wheels().await;
    rig.tick().await;
    let next = next_state(&mut rig).await;
    assert_eq!(ns(next.timestamp), rig.now_ns);
    // drive_states tells the leader the same: the first measurement, its
    // repeats, nothing through the silence, then the next measurement.
    let repeats = (1..=REPORTING_SILENT_STEPS)
        .filter(|step| step % 2 == 0)
        .count();
    let mut stamps = Vec::new();
    for _ in 0..repeats + 2 {
        let drive = tokio::time::timeout(ANSWER_BUDGET, rig.pairings.drive.drive_states.next())
            .await
            .expect("the node publishes drive_states")
            .expect("drive_states receive")
            .expect("drive_states open");
        stamps.push(ns(drive.timestamp));
    }
    let expected: Vec<i64> = std::iter::repeat_n(START_NS, repeats + 1)
        .chain([rig.now_ns])
        .collect();
    assert_eq!(stamps, expected);
    rig.shutdown().await;
}

// A robot stop while the leader streams names the base, brings it to rest and
// holds it until more than 0.5 s pass, from the stop on, with no setpoint, and
// a setpoint that arrives during the hold counts for that 0.5 s, as robot_stop
// and drive_link state; drive_states tells the leader the stop holds the
// base, and its setpoint after the hold drives the base again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_robot_stop_holds_the_base_until_more_than_half_a_second_passes_without_a_setpoint() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    rig.lead_until_it_drives([0.2, 0.0, 0.0], 0.0).await;
    // The leader's one setpoint drives the base up to 0.2 m/s by 0.2 s, and
    // its stream runs on.
    assert_speeds(&mut rig, &[(0.2, 0.2)]).await;
    assert_eq!(rig.stop("operator").await, vec!["base".to_string()]);
    let stopped_at = rig.now_ns;
    // The base slows to rest at its deceleration limit, and the hold lasts
    // through 0.5 s after the stop.
    let slowing = [0.2 - SLOW_DOWN, 0.0];
    for step in 1..=10 {
        rig.step().await;
        if let Some(speed) = slowing.get(step - 1) {
            let setpoint = rig.wheel_setpoint_at(rig.now_ns).await;
            assert_close(forward_speed(&setpoint), *speed, "the slowing base");
        }
        assert_eq!(follow_state(&mut rig).await, "robot_stop", "step {step}");
    }
    assert_eq!(rig.now_ns, stopped_at + LEADER_STREAMING_WINDOW_NS);
    rig.step().await;
    assert_eq!(follow_state(&mut rig).await, "following");
    // A stop with no goal and no stream names nothing, and holds the base.
    // The leader's setpoint 0.25 s into that hold, which the base drops,
    // makes it last until more than 0.5 s pass after the setpoint.
    assert!(rig.stop("operator").await.is_empty());
    let second_stop = rig.now_ns;
    rig.step_to(second_stop + 5 * STEP_NS).await;
    let held_setpoint = rig.now_ns;
    rig.lead([0.2, 0.0, 0.0], 0.0).await;
    rig.until_the_leader_stream_refuses_a_goal().await;
    while rig.now_ns < held_setpoint + LEADER_STREAMING_WINDOW_NS {
        rig.step().await;
        assert_eq!(
            follow_state(&mut rig).await,
            "robot_stop",
            "after the held setpoint"
        );
    }
    rig.step().await;
    assert_eq!(follow_state(&mut rig).await, "following");
    // The leader's next setpoint drives the base: it speeds up step by step.
    rig.lead_until_it_drives([0.2, 0.0, 0.0], 0.0).await;
    assert_speeds(&mut rig, &[(0.05, SPEED_UP), (0.1, 2.0 * SPEED_UP)]).await;
    assert_eq!(follow_state(&mut rig).await, "following");
    rig.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drive_base_whose_base_does_not_rest_by_its_settle_timeout_fails_with_timeout() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    let goal = rig
        .send_drive(&drive_base::GoalRequestData {
            settle_timeout_s: 0.1,
            ..drive_goal(0.2, 0.3)
        })
        .await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    // Reached at 0.2 s and held 0.3 s, the command comes to 0 at 0.55 s.
    // From then the wheels drift 0.02 rad a step, as on a slope: 0.04 m/s,
    // past the rest velocity, so the base does not rest by 0.65 s.
    rig.step_to(rig.at(0.5)).await;
    for _ in 0..3 {
        rig.wheels = [rig.wheels[0] + 0.02, rig.wheels[1] + 0.02];
        rig.step().await;
    }
    // The goal still held the base at the step of 0.65 s, which reports
    // before it judges the goal: the settle counts from 0.55 s.
    let at_failure = rig.drive_state_at(rig.at(0.65)).await;
    assert_eq!(at_failure.follow_state, "follower_goal");
    let result = goal.get_result(ANSWER_BUDGET).await.expect("a result");
    let drive_base::ResultOutcome::Failed(failure) = result.outcome else {
        panic!("a drive whose base moves on fails: {:?}", result.outcome);
    };
    assert_eq!(failure.kind, drive_base::FailureKind::Timeout);
    assert_eq!(
        failure.message,
        "the base still moves at x 0.040 m/s and yaw 0.000 rad/s 0.100 s after \
         the command reached 0"
    );
    rig.shutdown().await;
}

/// The place in the robot frame of `link`, a link the URDF's fixed joints hang
/// from the robot frame, through each fixed joint from the root down.
fn fixed_place(link: &str) -> Transform {
    let description = robot_description::model(MODEL)
        .expect("the bundled base")
        .description();
    let joints: Vec<_> = description
        .fixed_transforms()
        .iter()
        .map(|joint| joint.frame_transform().expect("a fixed joint"))
        .collect();
    let mut place = Transform::new([0.0; 3], [0.0, 0.0, 0.0, 1.0]).expect("the identity");
    let mut child = link.to_owned();
    while child != "base_footprint" {
        let joint = joints
            .iter()
            .find(|joint| joint.child().as_str() == child)
            .unwrap_or_else(|| panic!("a fixed joint to {child}"));
        let hung = joint.placement().transform();
        place = hung.compose(&place);
        child = joint.parent().as_str().to_owned();
    }
    place
}

// A target in the base's camera frame at the time of a base_states is placed in
// odom through the initializer's fixed joints and the odometry at that time:
// the base drives on after it, and the move ends where that placement puts the
// robot frame, not where the latest odometry would.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_target_in_the_camera_frame_at_a_state_time_is_placed_in_odom_by_the_tree() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    rig.publish_fixed_transforms().await;
    // The base turns on an arc, so the odometry at the state's time is a pose
    // of its own.
    for wheels in [[0.5, 0.8], [1.0, 1.6]] {
        rig.wheels = wheels;
        rig.step().await;
    }
    rig.wheels = [1.5, 2.4];
    let seen = rig.step().await;
    for wheels in [[2.0, 3.0], [2.5, 3.5]] {
        rig.wheels = wheels;
        rig.step().await;
    }
    // The target in odom, and the same pose seen from the camera as the base
    // stood at the state's time.
    let in_odom = Transform::new([0.6, 0.4, 0.0], quaternion_of_yaw(0.8)).expect("a pose");
    let camera = Transform::new(seen.pose.position, seen.pose.orientation)
        .expect("the odometry's pose")
        .compose(&fixed_place("base_camera_optical_frame"));
    let in_camera = camera.inverse().compose(&in_odom);
    let goal = rig
        .send_move(&helpers::move_to(
            peppygen::exposed_actions::base_motion::move_base::GoalTarget {
                frame: Some("base_camera_optical_frame".to_owned()),
                timestamp: Some(seen.timestamp),
                position: in_camera.origin(),
                orientation: in_camera.orientation(),
            },
            5.0,
        ))
        .await;
    assert!(goal.accepted, "{:?}", goal.refusal);
    let result = rig.follow_until_result(&goal).await;
    let move_base::ResultOutcome::Completed(data) = result.outcome else {
        panic!("the move completes: {:?}", result.outcome);
    };
    assert_eq!(data.final_pose.frame, "odom");
    let [x, y, _] = data.final_pose.position;
    assert!(
        (x - 0.6).hypot(y - 0.4) <= 0.02,
        "the base rests at ({x}, {y}), the target at (0.6, 0.4) in odom"
    );
    let heading = yaw_of_quaternion(data.final_pose.orientation);
    assert!(
        (heading - 0.8).abs() <= 0.03,
        "the base rests turned {heading} rad, the target 0.8 rad in odom"
    );
    rig.shutdown().await;
}

// A goal in a frame the robot's tree lacks is refused with unknown_frame, and a
// target at a time before the oldest odometry the tree holds with
// extrapolation, each when the goal arrives, before the base moves.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_goal_the_tree_cannot_place_is_refused_with_the_kind_of_the_lookup() {
    let rig = Rig::start([0.0, 0.0]).await;
    let nowhere = Some("nowhere".to_owned());
    let unplaced_move = rig
        .send_move_ticking(&helpers::move_to(
            peppygen::exposed_actions::base_motion::move_base::GoalTarget {
                frame: nowhere.clone(),
                ..target(1.0, 0.0, 0.0)
            },
            1.0,
        ))
        .await
        .refusal
        .expect("a target in no frame of the tree");
    assert_eq!(
        unplaced_move.kind,
        move_base::FailureKind::UnknownFrame,
        "{}",
        unplaced_move.message
    );
    assert!(
        unplaced_move.message.starts_with("the target: "),
        "{}",
        unplaced_move.message
    );
    let unplaced_drive = rig
        .send_drive_ticking(&drive_base::GoalRequestData {
            velocity: peppygen::exposed_actions::base_motion::drive_base::GoalVelocity {
                frame: nowhere,
                ..velocity(0.1, 0.0)
            },
            ..drive_goal(0.1, 1.0)
        })
        .await
        .refusal
        .expect("a velocity in no frame of the tree");
    assert_eq!(
        unplaced_drive.kind,
        drive_base::FailureKind::UnknownFrame,
        "{}",
        unplaced_drive.message
    );
    let before_the_odometry = rig
        .send_move_ticking(&helpers::move_to(
            peppygen::exposed_actions::base_motion::move_base::GoalTarget {
                timestamp: Some(
                    SystemTime::UNIX_EPOCH
                        + std::time::Duration::from_nanos((START_NS - STEP_NS) as u64),
                ),
                ..target(1.0, 0.0, 0.0)
            },
            1.0,
        ))
        .await
        .refusal
        .expect("a target before the odometry");
    assert_eq!(
        before_the_odometry.kind,
        move_base::FailureKind::Extrapolation,
        "{}",
        before_the_odometry.message
    );
    rig.shutdown().await;
}

// drive_link counts each 0.5 s from the arrival of setpoints on the node's
// clock. A simulation's clock reaches the node behind the leader's setpoints,
// so it can stand while they arrive and then leap: a leap of more than 0.5 s
// past the last arrival ends the stream at the leap, and the robot stop's hold
// with it, though the leader streamed on in its own time.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_clock_that_leaps_past_half_a_second_after_a_setpoint_ends_the_stream_and_its_hold() {
    let mut rig = Rig::start([0.0, 0.0]).await;
    rig.lead_until_it_drives([0.2, 0.0, 0.0], 0.0).await;
    assert_eq!(rig.stop("operator").await, vec!["base".to_string()]);
    for _ in 0..3 {
        rig.lead([0.2, 0.0, 0.0], 0.0).await;
        rig.step().await;
        assert_eq!(follow_state(&mut rig).await, "robot_stop");
    }
    // The clock leaps 0.6 s past the step of the last setpoint, the wheels
    // reporting at the new time.
    rig.now_ns += 12 * STEP_NS - STEP_NS;
    rig.step().await;
    assert_eq!(
        follow_state(&mut rig).await,
        "following",
        "the stream ended at the leap"
    );
    rig.shutdown().await;
}
