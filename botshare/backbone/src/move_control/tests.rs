use super::*;

use std::f64::consts::PI;

use control_core::time::nanoseconds;
use hegel::TestCase;
use hegel::generators as gs;

use crate::config::{AxisSpec, BaseAxes};
use crate::test_rig::{AXES, ROTATE_TO_HEADING_MIN_ANGLE_RAD};

/// The profiles of the rig's axes.
const LINEAR: AxisProfile = AxisProfile {
    positive_speed: 0.5,
    negative_speed: 0.5,
    acceleration: 1.0,
    deceleration: 2.0,
    deadband: 0.005,
};
const ANGULAR: AxisProfile = AxisProfile {
    positive_speed: 1.5,
    negative_speed: 1.5,
    acceleration: 3.0,
    deceleration: 4.0,
    deadband: 0.01,
};
const TOLERANCES: Tolerances = Tolerances {
    position: 0.02,
    yaw: 0.03,
};
/// The velocity limiter's rate (Hz) and its step (s, ns).
const FREQUENCY_HZ: f64 = 20.0;
const DT: f64 = 0.05;
const STEP_NS: i64 = 50_000_000;
/// How long after its planned end a move through the limiter comes to rest
/// within its tolerances at the latest (s).
const TAIL_S: f64 = 10.0;

fn pose(x: f64, y: f64, yaw: f64) -> PlanarPose {
    PlanarPose::new(x, y, yaw).expect("a finite pose")
}

/// The steering of the fastest move to `target` within [`TOLERANCES`].
fn fastest_move_to(target: PlanarPose) -> MoveController {
    let plan = plan_move(target, TOLERANCES.position, LINEAR, ANGULAR, 0.0);
    MoveController::new(
        target,
        plan,
        TOLERANCES,
        ROTATE_TO_HEADING_MIN_ANGLE_RAD,
        PlanarTwist::ZERO,
    )
}

#[test]
fn the_profiles_are_those_of_the_axes() {
    assert_eq!(AXES.x.profile(), LINEAR);
    assert_eq!(AXES.yaw.profile(), ANGULAR);
}

#[test]
fn a_long_move_cruises_between_its_ramps() {
    // 0.5 s up over 0.125 m, 0.25 s down over 0.0625 m, the rest at 0.5 m/s.
    let time = LINEAR.travel_time(2.0);
    assert!((time - (0.5 + 0.25 + (2.0 - 0.1875) / 0.5)).abs() < 1e-12);
}

#[test]
fn a_short_move_peaks_below_the_speed() {
    let distance = 0.06;
    let peak = (2.0 * distance * 1.0 * 2.0 / 3.0_f64).sqrt();
    assert!(peak < LINEAR.positive_speed);
    assert!((LINEAR.travel_time(distance) - (peak + peak / 2.0)).abs() < 1e-12);
}

#[test]
fn a_plan_is_floored_at_the_fastest_the_limits_allow() {
    let target = pose(1.0, 0.5, PI / 2.0);
    let fastest = plan_move(target, 0.02, LINEAR, ANGULAR, 0.0);
    assert_eq!(fastest.linear, LINEAR);
    let slower = plan_move(target, 0.02, LINEAR, ANGULAR, 3.0 * fastest.duration_s);
    assert!((slower.duration_s - 3.0 * fastest.duration_s).abs() < 1e-12);
    assert!((slower.linear.positive_speed - LINEAR.positive_speed / 3.0).abs() < 1e-12);
    assert!((slower.linear.negative_speed - LINEAR.negative_speed / 3.0).abs() < 1e-12);
    assert!((slower.angular.positive_speed - ANGULAR.positive_speed / 3.0).abs() < 1e-12);
    assert_eq!(
        (slower.linear.acceleration, slower.linear.deceleration),
        (LINEAR.acceleration, LINEAR.deceleration)
    );
}

// Each leg goes at the speed of its way: on a base that drives ahead at
// 0.5 m/s and backward at 0.05 m/s, a target 5 m ahead is driven to at
// 0.5 m/s, and a drive of 5 m backward takes the backward speed.
#[test]
fn each_leg_goes_at_the_speed_of_its_way() {
    let slow_back = AxisProfile {
        negative_speed: 0.05,
        ..LINEAR
    };
    let plan = plan_move(pose(5.0, 0.0, 0.0), 0.02, slow_back, ANGULAR, 0.0);
    assert!(
        (plan.duration_s - LINEAR.travel_time(5.0)).abs() < 1e-12,
        "{plan:?}"
    );
    let backward = slow_back.travel_time(-5.0);
    let at_back_speed = AxisProfile {
        positive_speed: 0.05,
        ..LINEAR
    };
    assert!((backward - at_back_speed.travel_time(5.0)).abs() < 1e-12);
    assert!(backward > 90.0, "{backward} s");
}

// A base that cannot drive backward turns toward a target it drove past,
// and commands no backward speed.
#[test]
fn a_base_that_cannot_drive_backward_turns_toward_a_target_it_passed() {
    let ahead_only = AxisProfile {
        negative_speed: 0.0,
        ..LINEAR
    };
    let target = pose(1.0, 0.0, 0.0);
    let plan = plan_move(target, TOLERANCES.position, ahead_only, ANGULAR, 0.0);
    let mut controller = MoveController::new(
        target,
        plan,
        TOLERANCES,
        ROTATE_TO_HEADING_MIN_ANGLE_RAD,
        PlanarTwist::ZERO,
    );
    let driving = controller.command(pose(0.0, 0.0, 0.0), DT);
    assert!(driving.linear_x() > 0.0, "{driving:?}");
    let past = controller.command(pose(1.1, 0.0, 0.0), DT);
    assert!(past.linear_x() >= 0.0, "{past:?}");
    assert!(
        matches!(controller.leg, Some(Leg::TurnToward(_))),
        "{:?}",
        controller.leg
    );
    let mut reversing = fastest_move_to(target);
    reversing.command(pose(0.0, 0.0, 0.0), DT);
    reversing.command(pose(1.1, 0.0, 0.0), DT);
    assert_eq!(
        reversing.leg,
        Some(Leg::Drive),
        "a base that reverses backs up"
    );
}

#[test]
fn a_target_within_the_position_tolerance_is_a_turn_in_place() {
    assert_eq!(legs(pose(0.01, 0.0, 1.0), 0.02), [0.0, 0.0, 1.0]);
}

// On steps of 0.05 s at 4 rad/s^2 the limiter slows a turn by 0.2 rad/s a
// step: 0.2 rad/s covers 0.01 rad in one step, 0.1 rad/s 0.005 rad, and
// 0.8, 0.6, 0.4 and 0.2 rad/s 0.1 rad.
#[test]
fn a_stopping_speed_stops_on_the_end_on_the_limiters_steps() {
    assert!((ANGULAR.stopping_speed(0.01, DT) - 0.2).abs() < 1e-12);
    assert!((ANGULAR.stopping_speed(0.005, DT) - 0.1).abs() < 1e-12);
    assert!((ANGULAR.stopping_speed(0.1, DT) - 0.8).abs() < 1e-12);
}

// An axis that takes the stopping speed of what remains each step, through
// the limiter's deceleration of deceleration x dt a step, ends on the end:
// each step's stopping speed lies within one step of the deceleration below
// the last, so the limiter follows it, and the last step covers what
// remained.
#[hegel::test]
fn an_axis_on_its_stopping_speed_stops_on_the_end(tc: TestCase) {
    let start: f64 = tc.draw(gs::floats().min_value(1e-6).max_value(5.0));
    let step = LINEAR.deceleration * DT;
    let mut remaining = start;
    let mut speed = LINEAR.stopping_speed(remaining, DT);
    for _ in 0..100_000 {
        let command = LINEAR.stopping_speed(remaining, DT);
        assert!(
            command >= speed - step * (1.0 + 1e-9),
            "{command} after {speed}"
        );
        speed = command;
        remaining -= speed * DT;
        if remaining <= 1e-12 * start {
            break;
        }
    }
    assert!(
        remaining.abs() <= 1e-9 * start,
        "{remaining} of {start} left"
    );
}

// Spin (spin.cpp:118-130): vel = sqrt(2 x rotational_acc_lim x
// remaining_yaw), clamped to [min_rotational_vel, max_rotational_vel],
// signed as the commanded turn; here the stopping speed on the limiter's
// steps, the leg's speed and the axis's deadband.
#[test]
fn the_turn_rate_is_nav2_spins() {
    assert_eq!(ANGULAR.turn_rate(Turning::Counterclockwise, 0.5, DT), 1.5);
    assert!((ANGULAR.turn_rate(Turning::Clockwise, 0.1, DT) + 0.8).abs() < 1e-12);
    // 1 urad over the step is 20 urad/s, below the 10 mrad/s deadband.
    assert_eq!(
        ANGULAR.turn_rate(Turning::Clockwise, 1e-6, DT),
        -0.01 * (1.0 + ABOVE_DEADBAND)
    );
}

// DriveOnHeading (drive_on_heading.hpp:140-164), dt = 1 / cycle_frequency:
// command_speed clamped to the last command plus or minus the acceleration
// and deceleration over dt, capped by the stopping speed, floored at
// minimum_speed; here the leg's speed, the stopping speed on the limiter's
// steps and the axis's deadband.
#[test]
fn the_drive_velocity_is_nav2_drive_on_headings() {
    // From rest 2 m ahead: up by 1 x 0.05.
    assert_eq!(LINEAR.drive_velocity(2.0, 0.0, DT), 0.05);
    // At the speed 4 cm ahead: 0.35, 0.25, 0.15 and 0.05 m/s a step cover
    // 4 cm.
    assert!((LINEAR.drive_velocity(0.04, 0.5, DT) - 0.35).abs() < 1e-12);
    // Past the speed, as after a scaled plan: down by 2 x 0.05.
    let slow = LINEAR.scaled(0.2);
    assert!((slow.drive_velocity(2.0, 0.5, DT) - 0.4).abs() < 1e-15);
    // From rest 1 m behind: back by 1 x 0.05.
    assert_eq!(LINEAR.drive_velocity(-1.0, 0.0, DT), -0.05);
    // Driving ahead at 0.3 m/s with the target 1 m behind: down by the
    // acceleration toward the reverse, 0.3 - 1 x 0.05.
    assert!((LINEAR.drive_velocity(-1.0, 0.3, DT) - 0.25).abs() < 1e-15);
    // At 0.3 m/s 4 mm ahead: the whole of it in one step, 0.08 m/s.
    assert!((LINEAR.drive_velocity(0.004, 0.3, DT) - 0.08).abs() < 1e-15);
    // 10 nm ahead: 0.2 um/s over the step, below the 5 mm/s deadband.
    assert_eq!(
        LINEAR.drive_velocity(1e-8, 0.0, DT),
        0.005 * (1.0 + ABOVE_DEADBAND)
    );
}

#[test]
fn a_target_behind_is_turned_toward_before_the_drive() {
    let mut controller = fastest_move_to(pose(-1.0, 0.0, 0.0));
    let first = controller.command(PlanarPose::IDENTITY, DT);
    assert_eq!(first.linear_x(), 0.0);
    assert_eq!(first.angular_z(), 1.5);
}

#[test]
fn a_target_ahead_within_the_yaw_tolerance_of_the_heading_is_driven_to_at_once() {
    // A bearing of 25 mrad, within the 30 mrad tolerance.
    let mut controller = fastest_move_to(pose(2.0, 0.05, 0.0));
    let first = controller.command(PlanarPose::IDENTITY, DT);
    assert_eq!((first.linear_x(), first.angular_z()), (0.05, 0.0));
}

#[test]
fn a_drive_whose_bearing_leaves_the_yaw_tolerance_turns_toward_the_target_again() {
    let mut controller = fastest_move_to(pose(2.0, 0.0, 0.0));
    controller.command(PlanarPose::IDENTITY, DT);
    // 1 m on, slipped 2 cm to the left: a bearing of 20 mrad, within the
    // 30 mrad tolerance, and the drive goes on, at 0.1 m/s.
    let aimed = controller.command(pose(1.0, 0.02, 0.0), DT);
    assert_eq!((aimed.linear_x(), aimed.angular_z()), (0.1, 0.0));
    // 4 cm to the left: the drive comes down by 2 m/s^2 over the step, to
    // 0, and the base then turns clockwise toward the target.
    let off = pose(1.0, 0.04, 0.0);
    assert_eq!(controller.command(off, DT), PlanarTwist::ZERO);
    let turning = controller.command(off, DT);
    assert_eq!(turning.linear_x(), 0.0);
    assert!(turning.angular_z() < 0.0);
    // The turn runs until the heading points at the target; its rate then
    // comes down to 0 by 4 rad/s^2 over each step, and the drive goes on.
    let bearing = (-0.04f64).atan2(1.0);
    let short = controller.command(pose(1.0, 0.04, bearing / 2.0), DT);
    assert!(short.angular_z() < 0.0);
    let aimed_again = pose(1.0, 0.04, bearing - 1e-5);
    let mut last = short;
    let resumed = loop {
        let command = controller.command(aimed_again, DT);
        if command.linear_x() != 0.0 {
            break command;
        }
        assert!(
            command.angular_z().abs() <= (last.angular_z().abs() - 0.2).max(0.0) + 1e-12,
            "{command:?} after {last:?}"
        );
        last = command;
    };
    assert_eq!(resumed.angular_z(), 0.0);
    assert!(resumed.linear_x() > 0.0);
}

#[test]
fn a_drive_runs_to_its_end_within_the_position_tolerance() {
    let mut controller = fastest_move_to(pose(1.0, 0.0, 0.5));
    controller.command(PlanarPose::IDENTITY, DT);
    // 1 cm short, within the 2 cm tolerance: the drive goes on to the
    // target before the turn to its yaw.
    assert!(controller.command(pose(0.99, 0.0, 0.0), DT).linear_x() > 0.0);
}

#[test]
fn a_drive_with_its_heading_free_turns_toward_a_target_past_the_rotate_to_heading_angle() {
    let free = Tolerances {
        position: 0.02,
        yaw: PI,
    };
    let target = pose(1.0, 0.0, 0.0);
    let plan = plan_move(target, free.position, LINEAR, ANGULAR, 0.0);
    let mut controller = MoveController::new(
        target,
        plan,
        free,
        ROTATE_TO_HEADING_MIN_ANGLE_RAD,
        PlanarTwist::ZERO,
    );
    assert!(controller.command(PlanarPose::IDENTITY, DT).linear_x() > 0.0);
    // Turned 0.78 rad off the target, within 0.785 rad: the drive goes on.
    let within = controller.command(pose(0.0, 0.0, 0.78), DT);
    assert_eq!(within.angular_z(), 0.0);
    assert!(within.linear_x() > 0.0);
    // 0.79 rad off: the drive comes down to 0, and the base turns toward the
    // target.
    let past = pose(0.0, 0.0, 0.79);
    assert_eq!(controller.command(past, DT), PlanarTwist::ZERO);
    let turning = controller.command(past, DT);
    assert_eq!(turning.linear_x(), 0.0);
    assert!(turning.angular_z() < 0.0);
}

#[test]
fn the_turn_to_the_yaw_runs_past_the_yaw_tolerance_to_its_end() {
    let target = pose(0.0, 0.0, 0.5);
    let mut controller = fastest_move_to(target);
    assert!(controller.command(PlanarPose::IDENTITY, DT).angular_z() > 0.0);
    // 10 mrad short of the yaw, within the 30 mrad tolerance: still turning.
    let near = controller.command(pose(0.0, 0.0, 0.49), DT);
    assert!(near.angular_z() > 0.0);
    // 1 mrad past it: the turn ends.
    assert_eq!(
        controller.command(pose(0.0, 0.0, 0.501), DT),
        PlanarTwist::ZERO
    );
    // Within the yaw tolerance the base stays.
    assert_eq!(
        controller.command(pose(0.0, 0.0, 0.52), DT),
        PlanarTwist::ZERO
    );
    // Past it, it turns back.
    assert!(controller.command(pose(0.0, 0.0, 0.54), DT).angular_z() < 0.0);
}

#[test]
fn a_heading_left_free_reaches_the_position_alone() {
    let free = Tolerances {
        position: 0.02,
        yaw: PI,
    };
    let target = pose(1.0, 0.0, 2.0);
    let plan = plan_move(target, free.position, LINEAR, ANGULAR, 0.0);
    let mut controller = MoveController::new(
        target,
        plan,
        free,
        ROTATE_TO_HEADING_MIN_ANGLE_RAD,
        PlanarTwist::ZERO,
    );
    assert_eq!(
        controller.command(pose(0.99, 0.0, 0.0), DT),
        PlanarTwist::ZERO
    );
    assert!(controller.within_tolerances(pose(0.99, 0.0, 0.0)));
}

#[test]
fn a_slow_plan_commands_just_above_the_deadband() {
    // 0.1 m in 60 s peaks at about 3.3 mm/s, below the 5 mm/s deadband.
    let target = pose(0.1, 0.0, 0.0);
    let plan = plan_move(target, TOLERANCES.position, LINEAR, ANGULAR, 60.0);
    let mut controller = MoveController::new(
        target,
        plan,
        TOLERANCES,
        ROTATE_TO_HEADING_MIN_ANGLE_RAD,
        PlanarTwist::ZERO,
    );
    assert_eq!(
        controller.command(PlanarPose::IDENTITY, DT),
        PlanarTwist::new(0.005 * (1.0 + ABOVE_DEADBAND), 0.0, 0.0).unwrap()
    );
}

// Every command of a drive keeps the leg's scaled speed and grows from the
// last by at most the acceleration over the step, or stands at the floor.
#[test]
fn a_drive_keeps_its_scaled_speed_and_the_acceleration() {
    let target = pose(2.0, 0.0, 0.0);
    let plan = plan_move(target, TOLERANCES.position, LINEAR, ANGULAR, 20.0);
    let mut controller = MoveController::new(
        target,
        plan,
        TOLERANCES,
        ROTATE_TO_HEADING_MIN_ANGLE_RAD,
        PlanarTwist::ZERO,
    );
    let dt = 0.02;
    let mut pose = PlanarPose::IDENTITY;
    let mut last = 0.0;
    for _ in 0..2000 {
        let command = controller.command(pose, dt).linear_x();
        assert!(command <= plan.linear.positive_speed + 1e-12, "{command}");
        assert!(
            command - last <= LINEAR.acceleration * dt + 1e-12 || command == LINEAR.floor(),
            "from {last} to {command}"
        );
        last = command;
        pose = self::pose(pose.x() + command * dt, pose.y(), pose.yaw());
    }
}

/// `pose` moved by `velocity` for `dt` (s): the turn, then the drive along
/// the new heading.
fn moved(pose: PlanarPose, velocity: PlanarTwist, dt: f64) -> PlanarPose {
    let yaw = pose.yaw() + velocity.angular_z() * dt;
    self::pose(
        pose.x() + velocity.linear_x() * dt * yaw.cos(),
        pose.y() + velocity.linear_x() * dt * yaw.sin(),
        normalize_angle(yaw),
    )
}

// A move to any target within reach, for any duration up to 60 s and
// tolerances down to 1 mm and 2 mrad, comes to rest within both tolerances
// by its planned end plus TAIL_S, on a base that executes what the velocity
// limiter gives.
#[hegel::test]
fn a_move_through_the_limiter_rests_within_its_tolerances(tc: TestCase) {
    let target = pose(
        tc.draw(gs::floats().min_value(-3.0).max_value(3.0)),
        tc.draw(gs::floats().min_value(-3.0).max_value(3.0)),
        tc.draw(gs::floats().min_value(-PI).max_value(PI)),
    );
    let duration_s: f64 = tc.draw(gs::floats().min_value(0.0).max_value(60.0));
    let tolerances = Tolerances {
        position: tc.draw(gs::floats().min_value(0.001).max_value(0.05)),
        yaw: tc.draw(gs::floats().min_value(0.002).max_value(0.05)),
    };
    // The rig's deadbands, or none, as the launcher's simulated base gives.
    let axes = match tc.draw(gs::booleans()) {
        true => AXES,
        false => BaseAxes {
            x: AxisSpec {
                deadband: 0.0,
                ..AXES.x
            },
            yaw: AxisSpec {
                deadband: 0.0,
                ..AXES.yaw
            },
        },
    };
    let plan = plan_move(
        target,
        tolerances.position,
        axes.x.profile(),
        axes.yaw.profile(),
        duration_s,
    );
    let mut controller = MoveController::new(
        target,
        plan,
        tolerances,
        ROTATE_TO_HEADING_MIN_ANGLE_RAD,
        PlanarTwist::ZERO,
    );
    let mut limiter = axes
        .limiter_settings(FREQUENCY_HZ)
        .expect("the test's axes")
        .start();
    let planned_end_ns = nanoseconds(plan.duration_s);
    let last_step = ((plan.duration_s + TAIL_S) / DT).ceil() as i64;
    let mut pose = PlanarPose::IDENTITY;
    for step in 1..=last_step {
        let now_ns = step * STEP_NS;
        limiter.take_setpoint(controller.command(pose, DT), now_ns);
        let executed = limiter.tick(now_ns).unwrap_or(PlanarTwist::ZERO);
        pose = moved(pose, executed, DT);
        if now_ns >= planned_end_ns
            && limiter.last_velocity() == PlanarTwist::ZERO
            && controller.within_tolerances(pose)
        {
            return;
        }
    }
    panic!(
        "the move to {target:?} in {duration_s} s within {tolerances:?} stands at {pose:?} \
         {TAIL_S} s after its planned end"
    );
}

// A move that starts while the limiter still carries the base ahead brings
// that speed down at the deceleration before its turn starts, as it brings
// down a leg of its own.
#[test]
fn a_move_started_rolling_ahead_winds_the_roll_down_before_it_turns() {
    let target = pose(0.0, 0.0, 1.57);
    let plan = plan_move(target, TOLERANCES.position, LINEAR, ANGULAR, 0.0);
    let rolling = PlanarTwist::new(0.3, 0.0, 0.0).unwrap();
    let mut controller = MoveController::new(
        target,
        plan,
        TOLERANCES,
        ROTATE_TO_HEADING_MIN_ANGLE_RAD,
        rolling,
    );
    let first = controller.command(PlanarPose::IDENTITY, DT);
    assert_eq!(first.angular_z(), 0.0, "{first:?}");
    assert!(
        (first.linear_x() - (0.3 - LINEAR.deceleration * DT)).abs() < 1e-12,
        "{first:?}"
    );
}
