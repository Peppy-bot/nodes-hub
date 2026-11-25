use super::*;

use std::f64::consts::PI;

use control_core::limiter::Limiter;
use control_core::time::nanoseconds;
use hegel::TestCase;
use hegel::generators as gs;

use crate::config::{AxisSpec, BaseAxes, SlowestSpeeds};
use crate::test_rig::{
    AXES, BOTSHARE_SIM_AXES, MIN_BACKWARD_SPEED_M_S, MIN_FORWARD_SPEED_M_S, MIN_TURN_RATE_RAD_S,
    ROTATE_TO_HEADING_MIN_ANGLE_RAD, botshare_sim_move_axes, move_axes,
};

/// The profiles of the rig's axes, floored at the rig's slowest speeds.
const LINEAR: AxisProfile = AxisProfile {
    positive: Way {
        speed: 0.5,
        floor: 0.02,
    },
    negative: Way {
        speed: 0.5,
        floor: 0.02,
    },
    acceleration: 1.0,
    deceleration: 2.0,
    deadband: 0.005,
};
const ANGULAR: AxisProfile = AxisProfile {
    positive: Way {
        speed: 1.5,
        floor: 0.03,
    },
    negative: Way {
        speed: 1.5,
        floor: 0.03,
    },
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
    MoveController::new(target, plan, TOLERANCES, ROTATE_TO_HEADING_MIN_ANGLE_RAD)
}

/// The command of `controller` at `pose` on a base whose limiter reached
/// `velocity`, the last command, at once; `velocity` takes the command.
fn command_reached(
    controller: &mut MoveController,
    pose: PlanarPose,
    velocity: &mut PlanarTwist,
) -> PlanarTwist {
    *velocity = controller.command(pose, *velocity, DT);
    *velocity
}

#[test]
fn the_profiles_are_those_of_the_axes() {
    let axes = move_axes(&AXES);
    assert_eq!(axes.linear, LINEAR);
    assert_eq!(axes.angular, ANGULAR);
}

// A floor is the node's slowest speed, raised to just above the deadband and
// held to the way's speed, as Spin clamps its rate (spin.cpp:125).
#[test]
fn a_floor_is_the_slowest_speed_above_the_deadband_within_the_ways_speed() {
    assert_eq!(leg_floor(0.4, 0.0, 0.5), 0.4);
    assert_eq!(leg_floor(0.001, 0.01, 1.5), 0.01 * (1.0 + ABOVE_DEADBAND));
    assert_eq!(leg_floor(0.6, 0.0, 0.5), 0.5);
    assert_eq!(leg_floor(0.1, 0.0, 0.0), 0.0);
    let slow_back = AxisSpec {
        min_velocity: -0.05,
        ..BOTSHARE_SIM_AXES.linear
    }
    .profile(0.1, 0.1);
    assert_eq!(
        (slow_back.positive.floor, slow_back.negative.floor),
        (0.1, 0.05)
    );
}

// At 3 rad/s^2 over steps of 0.05 s the limiter slows a turn by 0.15 rad/s a
// step: from 0.4 rad/s it turns at 0.25 and 0.10 rad/s, 0.0175 rad; from
// 0.3 rad/s at 0.15 rad/s, 0.0075 rad; from 0.15 rad/s it stops in the step.
#[test]
fn a_stopping_distance_is_the_limiters_steps_to_rest() {
    let angular = botshare_sim_move_axes().angular;
    let close = |speed: f64, distance: f64| {
        let stopping = angular.stopping_distance(speed, DT);
        assert!((stopping - distance).abs() < 1e-15, "{speed}: {stopping}");
    };
    close(0.4, 0.0175);
    close(0.3, 0.0075);
    close(0.15, 0.0);
    close(0.0, 0.0);
    close(-0.4, -0.0175);
}

// The limiter speeds up by 0.15 rad/s a step toward a faster command of the
// same way, and slows by 0.15 rad/s otherwise, through 0 toward a command
// the other way.
#[test]
fn a_next_speed_is_the_limiters_step() {
    let angular = botshare_sim_move_axes().angular;
    let close = |speed: f64, command: f64, next: f64| {
        let stepped = angular.next_speed(speed, command, DT);
        assert!(
            (stepped - next).abs() < 1e-15,
            "{speed} to {command}: {stepped}"
        );
    };
    close(0.0, 0.4, 0.15);
    close(0.3, 0.4, 0.4);
    close(0.4, 0.0, 0.25);
    close(-0.1, 0.4, 0.05);
}

// A turn that holds the 0.4 rad/s floor turns 0.02 rad a step and stops in
// 0.0175 rad more: it ends once 0.0275 rad or less remain, its stop landing
// within 0.01 rad of its end, half a step at the floor. Spin ends once less
// than 1e-6 rad remains (spin.cpp:119), and the stop from 0.4 rad/s, 0.0175
// rad, then lands past its end.
#[hegel::test]
fn a_turn_at_its_floor_ends_with_its_stop_within_half_a_step_of_its_end(tc: TestCase) {
    let angular = botshare_sim_move_axes().angular;
    let start: f64 = tc.draw(gs::floats().min_value(0.04).max_value(1.0));
    let mut remaining = start;
    while !angular.ends(remaining, 0.4, 0.4, DT) {
        remaining -= 0.4 * DT;
    }
    let landing = remaining - angular.stopping_distance(0.4, DT);
    assert!(remaining <= 0.0275 + 1e-12, "{remaining} of {start}");
    assert!(landing.abs() <= 0.01 + 1e-12, "{landing} of {start}");
}

// A leg on its stopping speed ends on its end: the limiter's stop from the
// speed it reached covers what remains. A millimetre more and the next
// stopping speed, about 0.6 rad/s, lands its stop on the end a step later.
#[test]
fn a_leg_on_its_stopping_speed_ends_with_its_stop_on_its_end() {
    let speed = 0.8;
    let ends = |remaining: f64| {
        let command = ANGULAR.turn_rate(Turning::Counterclockwise, remaining, DT);
        ANGULAR.ends(remaining, speed, command, DT)
    };
    let remaining = ANGULAR.stopping_distance(speed, DT);
    assert!(ends(remaining));
    assert!(!ends(remaining + 0.001));
}

// From rest the limiter's first step toward a floor turns at 0.15 rad/s,
// 0.0075 rad, and drives at 0.05 m/s, 2.5 mm, each stopping in the step: a
// turn at rest within half of it of its end ends there. The finest
// tolerances are the shortest drive and the shortest turn, which their sums
// land a few units in the last place above 2.5 mm and 7.5 mrad. The rig's
// base drives at its 0.02 m/s floor, 1 mm, and turns at its 0.03 rad/s, 1.5
// mrad. Each decimal is no finer than its finest and shows it, and a value
// past the slack below it is finer.
#[test]
fn the_finest_tolerances_take_and_show_the_decimals_of_the_shortest_drive_and_turn() {
    let taken = |finest: Tolerances, position: f64, yaw: f64| {
        for (finest, decimal) in [(finest.position, position), (finest.yaw, yaw)] {
            assert!(!finer_than(decimal, finest), "{decimal} of {finest}");
            assert!(
                finer_than(decimal * (1.0 - 2.0 * FINEST_SLACK), finest),
                "past the slack of {finest}"
            );
            assert_eq!(shown_finest(finest), decimal, "{finest}");
        }
    };
    taken(botshare_sim_move_axes().finest, 0.0025, 0.0075);
    taken(move_axes(&AXES).finest, 0.001, 0.0015);
    let rest = botshare_sim_move_axes().angular;
    assert!(rest.ends(0.00375, 0.0, 0.4, DT));
    assert!(!rest.ends(0.0038, 0.0, 0.4, DT));
}

// A finest shows rounded up to 9 significant digits from half the slack below
// it, a value no finer than it: 1.2345678941 shows as 1.23456790, above it,
// where its nearest 9 digits lie past the slack. 0, and a finest whose scale
// or rounding is not finite, show as themselves.
#[test]
fn a_finest_shows_as_nine_digits_no_finer_than_it() {
    let finest = 1.234_567_894_1;
    assert!(finer_than(1.234_567_89, finest));
    assert_eq!(shown_finest(finest), 1.234_567_9);
    assert!(!finer_than(shown_finest(finest), finest));
    assert_eq!(shown_finest(0.0), 0.0);
    assert_eq!(shown_finest(5e-324), 5e-324, "a scale past f64's range");
    assert!(shown_finest(f64::NAN).is_nan());
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
    assert!(peak < LINEAR.positive.speed);
    assert!((LINEAR.travel_time(distance) - (peak + peak / 2.0)).abs() < 1e-12);
}

#[test]
fn a_plan_is_floored_at_the_fastest_the_limits_allow() {
    let target = pose(1.0, 0.5, PI / 2.0);
    let fastest = plan_move(target, 0.02, LINEAR, ANGULAR, 0.0);
    assert_eq!(fastest.linear, LINEAR);
    let slower = plan_move(target, 0.02, LINEAR, ANGULAR, 3.0 * fastest.duration_s);
    assert!((slower.duration_s - 3.0 * fastest.duration_s).abs() < 1e-12);
    assert!((slower.linear.positive.speed - LINEAR.positive.speed / 3.0).abs() < 1e-12);
    assert!((slower.linear.negative.speed - LINEAR.negative.speed / 3.0).abs() < 1e-12);
    assert!((slower.angular.positive.speed - ANGULAR.positive.speed / 3.0).abs() < 1e-12);
    assert_eq!(
        (slower.linear.acceleration, slower.linear.deceleration),
        (LINEAR.acceleration, LINEAR.deceleration)
    );
    assert_eq!(
        (slower.linear.positive.floor, slower.angular.negative.floor),
        (LINEAR.positive.floor, ANGULAR.negative.floor)
    );
}

// Each leg goes at the speed of its way: on a base that drives ahead at
// 0.5 m/s and backward at 0.05 m/s, a target 5 m ahead is driven to at
// 0.5 m/s, and a drive of 5 m backward takes the backward speed.
#[test]
fn each_leg_goes_at_the_speed_of_its_way() {
    let slow_back = AxisProfile {
        negative: Way {
            speed: 0.05,
            ..LINEAR.negative
        },
        ..LINEAR
    };
    let plan = plan_move(pose(5.0, 0.0, 0.0), 0.02, slow_back, ANGULAR, 0.0);
    assert!(
        (plan.duration_s - LINEAR.travel_time(5.0)).abs() < 1e-12,
        "{plan:?}"
    );
    let backward = slow_back.travel_time(-5.0);
    let at_back_speed = AxisProfile {
        positive: Way {
            speed: 0.05,
            ..LINEAR.positive
        },
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
        negative: Way {
            speed: 0.0,
            ..LINEAR.negative
        },
        ..LINEAR
    };
    let target = pose(1.0, 0.0, 0.0);
    let plan = plan_move(target, TOLERANCES.position, ahead_only, ANGULAR, 0.0);
    let mut controller =
        MoveController::new(target, plan, TOLERANCES, ROTATE_TO_HEADING_MIN_ANGLE_RAD);
    let mut velocity = PlanarTwist::ZERO;
    let driving = command_reached(&mut controller, pose(0.0, 0.0, 0.0), &mut velocity);
    assert!(driving.linear_x() > 0.0, "{driving:?}");
    let past = command_reached(&mut controller, pose(1.1, 0.0, 0.0), &mut velocity);
    assert!(past.linear_x() >= 0.0, "{past:?}");
    assert!(
        matches!(controller.leg, Some(Leg::TurnToward(_))),
        "{:?}",
        controller.leg
    );
    let mut reversing = fastest_move_to(target);
    let mut velocity = PlanarTwist::ZERO;
    command_reached(&mut reversing, pose(0.0, 0.0, 0.0), &mut velocity);
    command_reached(&mut reversing, pose(1.1, 0.0, 0.0), &mut velocity);
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

// Spin (spin.cpp:124-125, 130): vel = sqrt(2 x rotational_acc_lim x
// remaining_yaw), clamped to [min_rotational_vel, max_rotational_vel],
// signed as the commanded turn; here the stopping speed on the limiter's
// steps, the leg's speed and the axis's floor.
#[test]
fn the_turn_rate_is_nav2_spins() {
    assert_eq!(ANGULAR.turn_rate(Turning::Counterclockwise, 0.5, DT), 1.5);
    assert!((ANGULAR.turn_rate(Turning::Clockwise, 0.1, DT) + 0.8).abs() < 1e-12);
    // 1 urad over the step is 20 urad/s, below the 30 mrad/s floor.
    assert_eq!(
        ANGULAR.turn_rate(Turning::Clockwise, 1e-6, DT),
        -ANGULAR.negative.floor
    );
    // Past its angle, at the floor.
    assert_eq!(
        ANGULAR.turn_rate(Turning::Counterclockwise, -0.1, DT),
        ANGULAR.positive.floor
    );
}

// DriveOnHeading (drive_on_heading.hpp:142-164), dt = 1 / cycle_frequency:
// command_speed clamped to the last command plus or minus the acceleration
// and deceleration over dt, capped by the stopping speed, floored at
// minimum_speed; here the leg's speed from the limiter's velocity, the
// stopping speed on the limiter's steps and the axis's floor.
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
    // 10 nm ahead: 0.2 um/s over the step, below the 20 mm/s floor.
    assert_eq!(LINEAR.drive_velocity(1e-8, 0.0, DT), LINEAR.positive.floor);
}

#[test]
fn a_target_behind_is_turned_toward_before_the_drive() {
    let mut controller = fastest_move_to(pose(-1.0, 0.0, 0.0));
    let first = controller.command(PlanarPose::IDENTITY, PlanarTwist::ZERO, DT);
    assert_eq!(first.linear_x(), 0.0);
    assert_eq!(first.angular_z(), 1.5);
}

#[test]
fn a_target_ahead_within_the_yaw_tolerance_of_the_heading_is_driven_to_at_once() {
    // A bearing of 25 mrad, within the 30 mrad tolerance.
    let mut controller = fastest_move_to(pose(2.0, 0.05, 0.0));
    let first = controller.command(PlanarPose::IDENTITY, PlanarTwist::ZERO, DT);
    assert_eq!((first.linear_x(), first.angular_z()), (0.05, 0.0));
}

#[test]
fn a_drive_whose_bearing_leaves_the_yaw_tolerance_turns_toward_the_target_again() {
    let mut controller = fastest_move_to(pose(2.0, 0.0, 0.0));
    let mut velocity = PlanarTwist::ZERO;
    command_reached(&mut controller, PlanarPose::IDENTITY, &mut velocity);
    // 1 m on, slipped 2 cm to the left: a bearing of 20 mrad, within the
    // 30 mrad tolerance, and the drive goes on, at 0.1 m/s.
    let aimed = command_reached(&mut controller, pose(1.0, 0.02, 0.0), &mut velocity);
    assert_eq!((aimed.linear_x(), aimed.angular_z()), (0.1, 0.0));
    // 4 cm to the left: the drive comes down by 2 m/s^2 over the step, to
    // 0, and the base then turns clockwise toward the target.
    let off = pose(1.0, 0.04, 0.0);
    assert_eq!(
        command_reached(&mut controller, off, &mut velocity),
        PlanarTwist::ZERO
    );
    let turning = command_reached(&mut controller, off, &mut velocity);
    assert_eq!(turning.linear_x(), 0.0);
    assert!(turning.angular_z() < 0.0);
    // The turn runs until the heading points at the target; its rate then
    // comes down to 0 by 4 rad/s^2 over each step, and the drive goes on.
    let bearing = (-0.04f64).atan2(1.0);
    let short = command_reached(
        &mut controller,
        pose(1.0, 0.04, bearing / 2.0),
        &mut velocity,
    );
    assert!(short.angular_z() < 0.0);
    let aimed_again = pose(1.0, 0.04, bearing - 1e-5);
    let mut last = short;
    let resumed = loop {
        let command = command_reached(&mut controller, aimed_again, &mut velocity);
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
    let mut velocity = PlanarTwist::ZERO;
    command_reached(&mut controller, PlanarPose::IDENTITY, &mut velocity);
    // 1 cm short, within the 2 cm tolerance: the drive goes on to the
    // target before the turn to its yaw.
    let near = command_reached(&mut controller, pose(0.99, 0.0, 0.0), &mut velocity);
    assert!(near.linear_x() > 0.0);
}

#[test]
fn a_drive_with_its_heading_free_turns_toward_a_target_past_the_rotate_to_heading_angle() {
    let free = Tolerances {
        position: 0.02,
        yaw: PI,
    };
    let target = pose(1.0, 0.0, 0.0);
    let plan = plan_move(target, free.position, LINEAR, ANGULAR, 0.0);
    let mut controller = MoveController::new(target, plan, free, ROTATE_TO_HEADING_MIN_ANGLE_RAD);
    let mut velocity = PlanarTwist::ZERO;
    let first = command_reached(&mut controller, PlanarPose::IDENTITY, &mut velocity);
    assert!(first.linear_x() > 0.0);
    // Turned 0.78 rad off the target, within 0.785 rad: the drive goes on.
    let within = command_reached(&mut controller, pose(0.0, 0.0, 0.78), &mut velocity);
    assert_eq!(within.angular_z(), 0.0);
    assert!(within.linear_x() > 0.0);
    // 0.79 rad off: the drive comes down to 0, and the base turns toward the
    // target.
    let past = pose(0.0, 0.0, 0.79);
    assert_eq!(
        command_reached(&mut controller, past, &mut velocity),
        PlanarTwist::ZERO
    );
    let turning = command_reached(&mut controller, past, &mut velocity);
    assert_eq!(turning.linear_x(), 0.0);
    assert!(turning.angular_z() < 0.0);
}

#[test]
fn the_turn_to_the_yaw_runs_past_the_yaw_tolerance_to_its_end() {
    let target = pose(0.0, 0.0, 0.5);
    let mut controller = fastest_move_to(target);
    let at_rest = PlanarTwist::ZERO;
    assert!(
        controller
            .command(PlanarPose::IDENTITY, at_rest, DT)
            .angular_z()
            > 0.0
    );
    // 10 mrad short of the yaw at rest, within the 30 mrad tolerance: still
    // turning.
    let near = controller.command(pose(0.0, 0.0, 0.49), at_rest, DT);
    assert!(near.angular_z() > 0.0);
    // 1 mrad past it: the turn ends.
    assert_eq!(
        controller.command(pose(0.0, 0.0, 0.501), near, DT),
        PlanarTwist::ZERO
    );
    // Within the yaw tolerance the base stays.
    assert_eq!(
        controller.command(pose(0.0, 0.0, 0.52), at_rest, DT),
        PlanarTwist::ZERO
    );
    // Past it, it turns back.
    assert!(
        controller
            .command(pose(0.0, 0.0, 0.54), at_rest, DT)
            .angular_z()
            < 0.0
    );
}

#[test]
fn a_heading_left_free_reaches_the_position_alone() {
    let free = Tolerances {
        position: 0.02,
        yaw: PI,
    };
    let target = pose(1.0, 0.0, 2.0);
    let plan = plan_move(target, free.position, LINEAR, ANGULAR, 0.0);
    let mut controller = MoveController::new(target, plan, free, ROTATE_TO_HEADING_MIN_ANGLE_RAD);
    assert_eq!(
        controller.command(pose(0.99, 0.0, 0.0), PlanarTwist::ZERO, DT),
        PlanarTwist::ZERO
    );
    assert!(controller.within_tolerances(pose(0.99, 0.0, 0.0)));
}

#[test]
fn a_slow_plan_commands_the_floor() {
    // 0.1 m in 60 s peaks at about 3.3 mm/s, below the 20 mm/s floor.
    let target = pose(0.1, 0.0, 0.0);
    let plan = plan_move(target, TOLERANCES.position, LINEAR, ANGULAR, 60.0);
    let mut controller =
        MoveController::new(target, plan, TOLERANCES, ROTATE_TO_HEADING_MIN_ANGLE_RAD);
    assert_eq!(
        controller.command(PlanarPose::IDENTITY, PlanarTwist::ZERO, DT),
        PlanarTwist::new(LINEAR.positive.floor, 0.0, 0.0).unwrap()
    );
}

// Every command of a drive keeps the leg's scaled speed and grows from the
// last by at most the acceleration over the step, or stands at the floor.
#[test]
fn a_drive_keeps_its_scaled_speed_and_the_acceleration() {
    let target = pose(2.0, 0.0, 0.0);
    let plan = plan_move(target, TOLERANCES.position, LINEAR, ANGULAR, 20.0);
    let mut controller =
        MoveController::new(target, plan, TOLERANCES, ROTATE_TO_HEADING_MIN_ANGLE_RAD);
    let dt = 0.02;
    let mut pose = PlanarPose::IDENTITY;
    let mut last = 0.0;
    for _ in 0..2000 {
        let velocity = PlanarTwist::new(last, 0.0, 0.0).expect("a finite command");
        let command = controller.command(pose, velocity, dt).linear_x();
        assert!(command <= plan.linear.positive.speed + 1e-12, "{command}");
        assert!(
            command - last <= LINEAR.acceleration * dt + 1e-12 || command == LINEAR.positive.floor,
            "from {last} to {command}"
        );
        last = command;
        pose = self::pose(pose.x() + command * dt, pose.y(), pose.yaw());
    }
}

/// `pose` moved by `velocity` for `dt` (s): the turn, then the drive along
/// the new heading, each axis standing still at a speed below its part of
/// `stuck_below`, [along x, of the turn].
fn moved(pose: PlanarPose, velocity: PlanarTwist, stuck_below: [f64; 2], dt: f64) -> PlanarPose {
    let moving = |value: f64, below: f64| if value.abs() < below { 0.0 } else { value };
    let linear = moving(velocity.linear_x(), stuck_below[0]);
    let yaw = pose.yaw() + moving(velocity.angular_z(), stuck_below[1]) * dt;
    self::pose(
        pose.x() + linear * dt * yaw.cos(),
        pose.y() + linear * dt * yaw.sin(),
        normalize_angle(yaw),
    )
}

/// A move the tests run through the limiter: to `target` within
/// `tolerances`, asked for `duration_s`, its drive turning toward the target
/// again past `rotate_to_heading_min_angle` (rad).
#[derive(Clone, Copy, Debug)]
struct LimitedMove {
    target: PlanarPose,
    duration_s: f64,
    tolerances: Tolerances,
    rotate_to_heading_min_angle: f64,
}

/// `run` planned and steered along `move_axes` through the limiter of
/// `axes`, a step of [`DT`] at a time, on a base that executes what the
/// limiter gives and stands still below `stuck_below` ([`moved`]).
struct LimitedRun {
    controller: MoveController,
    limiter: Limiter,
    planned_s: f64,
    stuck_below: [f64; 2],
    pose: PlanarPose,
    now_ns: i64,
}

impl LimitedRun {
    fn start(
        axes: &BaseAxes,
        move_axes: &MoveAxes,
        stuck_below: [f64; 2],
        run: LimitedMove,
    ) -> Self {
        let plan = plan_move(
            run.target,
            run.tolerances.position,
            move_axes.linear,
            move_axes.angular,
            run.duration_s,
        );
        Self {
            controller: MoveController::new(
                run.target,
                plan,
                run.tolerances,
                run.rotate_to_heading_min_angle,
            ),
            limiter: axes
                .limiter_settings(FREQUENCY_HZ)
                .expect("the test's axes")
                .start(),
            planned_s: plan.duration_s,
            stuck_below,
            pose: PlanarPose::IDENTITY,
            now_ns: 0,
        }
    }

    /// The steps from the start to `after_end_s` (s) past the planned end.
    fn steps_to(&self, after_end_s: f64) -> i64 {
        ((self.planned_s + after_end_s) / DT).ceil() as i64
    }

    /// Whether the run has not yet reached `after_end_s` (s) past the planned
    /// end.
    fn before(&self, after_end_s: f64) -> bool {
        self.now_ns < self.steps_to(after_end_s) * STEP_NS
    }

    /// One step: the move's command at the pose, the limiter at the velocity
    /// of its last step, and the base moved by what the limiter executes past
    /// its deadband, which this gives.
    fn step(&mut self) -> PlanarTwist {
        self.now_ns += STEP_NS;
        let command = self
            .controller
            .command(self.pose, self.limiter.last_velocity(), DT);
        self.limiter.take_setpoint(command, self.now_ns);
        let executed = self.limiter.tick(self.now_ns).unwrap_or(PlanarTwist::ZERO);
        self.pose = moved(self.pose, executed, self.stuck_below, DT);
        executed
    }

    /// Whether, at or after the planned end, the base rests within both
    /// tolerances.
    fn settled(&self) -> bool {
        self.now_ns >= nanoseconds(self.planned_s)
            && self.limiter.last_velocity() == PlanarTwist::ZERO
            && self.controller.within_tolerances(self.pose)
    }

    /// The pose of the base TAIL_S after the planned end: none once, at or
    /// after the planned end, the base rests within both tolerances.
    fn unsettled(mut self) -> Option<PlanarPose> {
        while self.before(TAIL_S) {
            self.step();
            if self.settled() {
                return None;
            }
        }
        Some(self.pose)
    }

    /// The first pose at which the base has stood outside the tolerances,
    /// executing 0 on both axes, for more than `idle_steps` steps in a row,
    /// up to `after_end_s` (s) past the planned end: none when no such
    /// stretch lasts longer, the run ending once, at or after the planned
    /// end, the base rests within both tolerances.
    fn idle(mut self, after_end_s: f64, idle_steps: u32) -> Option<PlanarPose> {
        let mut idle = 0;
        while self.before(after_end_s) {
            let executed = self.step();
            let outside = !self.controller.within_tolerances(self.pose);
            idle = if outside && executed == PlanarTwist::ZERO {
                idle + 1
            } else {
                0
            };
            if idle > idle_steps {
                return Some(self.pose);
            }
            if self.settled() {
                return None;
            }
        }
        None
    }
}

/// The pose of the base TAIL_S after the planned end of `run` through the
/// limiter ([`LimitedRun::unsettled`]).
fn unsettled_move(
    axes: &BaseAxes,
    move_axes: &MoveAxes,
    stuck_below: [f64; 2],
    run: LimitedMove,
) -> Option<PlanarPose> {
    LimitedRun::start(axes, move_axes, stuck_below, run).unsettled()
}

/// The rig's axes, with their deadbands or none, as the launcher's simulated
/// base gives.
fn any_rig_axes(tc: &TestCase) -> BaseAxes {
    match tc.draw(gs::booleans()) {
        true => AXES,
        false => BaseAxes {
            linear: AxisSpec {
                deadband: 0.0,
                ..AXES.linear
            },
            angular: AxisSpec {
                deadband: 0.0,
                ..AXES.angular
            },
        },
    }
}

/// A target within `reach` (m) along x and y of the robot frame, at any
/// heading.
fn target_within(tc: &TestCase, reach: f64) -> PlanarPose {
    pose(
        tc.draw(gs::floats().min_value(-reach).max_value(reach)),
        tc.draw(gs::floats().min_value(-reach).max_value(reach)),
        tc.draw(gs::floats().min_value(-PI).max_value(PI)),
    )
}

// A target 0.4 mm ahead and 1 mm to the left lies 1.08 mm away, outside the
// rig's finest position tolerance, 1 mm, the shortest drive, at a bearing of
// 1.19 rad, within a rotate-to-heading angle of 1.5 rad and a yaw tolerance
// of π, which leaves the heading free. A drive from rest toward a target
// within half the shortest drive ends where it starts, so the base turns
// toward the target, then drives to it.
#[test]
fn a_target_the_drive_cannot_close_from_rest_is_turned_toward() {
    let axes = move_axes(&AXES);
    let run = LimitedMove {
        target: pose(0.0004, 0.001, 0.0),
        duration_s: 0.0,
        tolerances: Tolerances {
            position: axes.finest.position,
            yaw: PI,
        },
        rotate_to_heading_min_angle: 1.5,
    };
    let mut limited = LimitedRun::start(&AXES, &axes, [0.0; 2], run);
    let first = limited.step();
    let unsettled = limited.unsettled();
    assert!(unsettled.is_none(), "{run:?} stands at {unsettled:?}");
    assert!(
        first.linear_x() == 0.0 && first.angular_z() > 0.0,
        "{first:?}"
    );
}

/// The rig's base turning at up to 1 rad/s, at 1 rad/s^2 each way, with a
/// deadband of 0.2 rad/s: the limiter's ramp from rest executes from its
/// fourth step, at 0.2 rad/s.
const DEADBAND_OF_FOUR_STEPS: BaseAxes = BaseAxes {
    angular: AxisSpec {
        min_velocity: -1.0,
        max_velocity: 1.0,
        max_acceleration: 1.0,
        max_deceleration: 1.0,
        deadband: 0.2,
    },
    ..AXES
};

// A stop and a ramp count only the steps at or above the deadband: from 0.4
// rad/s the stop executes at 0.35, 0.3, 0.25 and 0.2 rad/s, 0.055 rad, and
// the ramp from rest toward the floor first executes at 0.2 rad/s, so the
// shortest turn is 0.01 rad.
#[test]
fn a_stop_and_a_ramp_count_only_the_steps_at_or_above_the_deadband() {
    let axes = move_axes(&DEADBAND_OF_FOUR_STEPS);
    let angular = axes.angular;
    let stopping = angular.stopping_distance(0.4, DT);
    assert!((stopping - 0.055).abs() < 1e-15, "{stopping}");
    let first = angular
        .first_executing(0.0, angular.positive.floor, DT)
        .expect("a floor above the deadband");
    assert!((first - 0.2).abs() < 1e-15, "{first}");
    assert!((axes.finest.yaw - 0.01).abs() < 1e-15, "{axes:?}");
}

// A turn of 0.5 rad in place through a deadband of four steps of the
// acceleration, its floor raised to just above the deadband, within the
// finest yaw tolerance, 0.01 rad: its end counts only the steps that
// execute, so it ends where they land it, and the base rests within its
// tolerances.
#[test]
fn a_turn_through_a_deadband_of_several_steps_rests_within_its_tolerances() {
    let axes = move_axes(&DEADBAND_OF_FOUR_STEPS);
    let run = LimitedMove {
        target: pose(0.0, 0.0, 0.5),
        duration_s: 0.0,
        tolerances: axes.finest,
        rotate_to_heading_min_angle: ROTATE_TO_HEADING_MIN_ANGLE_RAD,
    };
    let unsettled = unsettled_move(&DEADBAND_OF_FOUR_STEPS, &axes, [0.0; 2], run);
    assert!(unsettled.is_none(), "{run:?} stands at {unsettled:?}");
}

// A turn at rest with 0.01 rad left, more than half the shortest turn through
// a deadband of four steps, ramps through three steps that execute 0 and
// lands on its end with its one step that executes, at 0.2 rad/s: the base
// rests within a yaw tolerance of half the shortest turn, which the test
// gives to see the landing.
#[test]
fn a_turn_from_rest_through_a_deadband_lands_on_its_end() {
    let axes = move_axes(&DEADBAND_OF_FOUR_STEPS);
    let run = LimitedMove {
        target: pose(0.0, 0.0, 0.01),
        duration_s: 0.0,
        tolerances: Tolerances {
            position: axes.finest.position,
            yaw: axes.finest.yaw / 2.0,
        },
        rotate_to_heading_min_angle: ROTATE_TO_HEADING_MIN_ANGLE_RAD,
    };
    let unsettled = unsettled_move(&DEADBAND_OF_FOUR_STEPS, &axes, [0.0; 2], run);
    assert!(unsettled.is_none(), "{run:?} stands at {unsettled:?}");
}

// A move to any target within reach, for any duration up to 60 s and
// tolerances down to 1 mm and 2 mrad, comes to rest within both tolerances
// by its planned end plus TAIL_S, on a base that executes what the velocity
// limiter gives.
#[hegel::test]
fn a_move_through_the_limiter_rests_within_its_tolerances(tc: TestCase) {
    let run = LimitedMove {
        target: target_within(&tc, 3.0),
        duration_s: tc.draw(gs::floats().min_value(0.0).max_value(60.0)),
        tolerances: Tolerances {
            position: tc.draw(gs::floats().min_value(0.001).max_value(0.05)),
            yaw: tc.draw(gs::floats().min_value(0.002).max_value(0.05)),
        },
        rotate_to_heading_min_angle: ROTATE_TO_HEADING_MIN_ANGLE_RAD,
    };
    let axes = any_rig_axes(&tc);
    if let Some(pose) = unsettled_move(&axes, &move_axes(&axes), [0.0; 2], run) {
        panic!("{run:?} stands at {pose:?} {TAIL_S} s after its planned end");
    }
}

// While the base stands outside its tolerances, it moves again within the
// steps its axes spend below their deadbands: the ramp of each from rest up
// to its deadband, the tail of each stop below it, and the steps a leg ends
// and the next starts on. For the rig's limits with deadbands from 0 up to
// four steps of the acceleration on each axis, floors from the rig's slowest
// speeds up to the speed, raised to just above the deadband ([`leg_floor`]),
// a target within reach, or within a few position
// tolerances of the robot frame, any duration up to 60 s, tolerances from the
// finest the axes take up to 5 cm and past a half turn, and a
// rotate-to-heading angle from the shortest turn up to a half turn. The move
// runs to its planned end plus TAIL_S and the time the slower floor of each
// axis takes over the legs the drive's aim adds past the plan: a half turn
// toward the target, a half turn back to its yaw, and the target's distance.
#[hegel::test(test_cases = 500)]
fn a_base_outside_its_tolerances_moves_again_past_its_deadbands(tc: TestCase) {
    let deadband = |spec: AxisSpec| {
        let most = 4.0 * spec.max_acceleration * DT;
        AxisSpec {
            deadband: tc.draw(gs::floats().min_value(0.0).max_value(most)),
            ..spec
        }
    };
    let axes = BaseAxes {
        linear: deadband(AXES.linear),
        angular: deadband(AXES.angular),
    };
    let floor = |spec: AxisSpec, slowest: f64| {
        tc.draw(gs::floats().min_value(slowest).max_value(spec.max_velocity))
    };
    let slowest = SlowestSpeeds {
        turn_rate: floor(axes.angular, MIN_TURN_RATE_RAD_S),
        forward: floor(axes.linear, MIN_FORWARD_SPEED_M_S),
        backward: floor(axes.linear, MIN_BACKWARD_SPEED_M_S),
    };
    let move_axes = axes.move_axes(slowest, DT);
    let finest = move_axes.finest;
    let tolerances = Tolerances {
        position: tc.draw(gs::floats().min_value(finest.position).max_value(0.05)),
        yaw: tc.draw(gs::floats().min_value(finest.yaw).max_value(3.5)),
    };
    let reach = match tc.draw(gs::booleans()) {
        true => 3.0,
        false => 4.0 * tolerances.position,
    };
    let run = LimitedMove {
        target: target_within(&tc, reach),
        duration_s: tc.draw(gs::floats().min_value(0.0).max_value(60.0)),
        tolerances,
        rotate_to_heading_min_angle: tc.draw(gs::floats().min_value(finest.yaw).max_value(PI)),
    };
    let slower_floor = |axis: AxisProfile| axis.positive.floor.min(axis.negative.floor);
    let distance = run.target.x().hypot(run.target.y());
    let after_end_s = TAIL_S
        + 2.0 * PI / slower_floor(move_axes.angular)
        + distance / slower_floor(move_axes.linear);
    let below_deadband = |axis: AxisProfile| {
        let steps = |rate: f64| (axis.deadband / (rate * DT)).ceil() as u32;
        steps(axis.acceleration) + steps(axis.deceleration)
    };
    let idle_steps = below_deadband(move_axes.linear) + below_deadband(move_axes.angular) + 2;
    let limited = LimitedRun::start(&axes, &move_axes, [0.0; 2], run);
    if let Some(pose) = limited.idle(after_end_s, idle_steps) {
        panic!(
            "{run:?} on {axes:?} floored at {slowest:?} stands at {pose:?} outside its \
             tolerances for more than {idle_steps} steps"
        );
    }
}

// On botshare_sim's floors, above a step of the deceleration, a move to any
// target within reach, for any duration up to 60 s and tolerances down to
// the finest the axes take, comes to rest within both tolerances by
// its planned end plus TAIL_S, on a base that executes what the limiter
// gives, or one that stands still below 0.01 m/s and 0.05 rad/s, the turn at
// which Isaac Sim's botshare base stands.
#[hegel::test]
fn a_move_on_floors_above_a_step_of_the_deceleration_rests_within_its_tolerances(tc: TestCase) {
    let axes = botshare_sim_move_axes();
    let run = LimitedMove {
        target: target_within(&tc, 3.0),
        duration_s: tc.draw(gs::floats().min_value(0.0).max_value(60.0)),
        tolerances: Tolerances {
            position: tc.draw(gs::floats().min_value(axes.finest.position).max_value(0.05)),
            yaw: tc.draw(gs::floats().min_value(axes.finest.yaw).max_value(0.05)),
        },
        rotate_to_heading_min_angle: ROTATE_TO_HEADING_MIN_ANGLE_RAD,
    };
    let stuck_below = match tc.draw(gs::booleans()) {
        true => [0.01, 0.05],
        false => [0.0; 2],
    };
    if let Some(pose) = unsettled_move(&BOTSHARE_SIM_AXES, &axes, stuck_below, run) {
        panic!(
            "{run:?}, the base stuck below {stuck_below:?}, stands at {pose:?} {TAIL_S} s after \
             its planned end"
        );
    }
}

// A move that starts while the limiter still carries the base ahead brings
// that speed down at the deceleration before its turn starts, as it brings
// down a leg of its own.
#[test]
fn a_move_started_rolling_ahead_winds_the_roll_down_before_it_turns() {
    let target = pose(0.0, 0.0, 1.57);
    let plan = plan_move(target, TOLERANCES.position, LINEAR, ANGULAR, 0.0);
    let rolling = PlanarTwist::new(0.3, 0.0, 0.0).unwrap();
    let mut controller =
        MoveController::new(target, plan, TOLERANCES, ROTATE_TO_HEADING_MIN_ANGLE_RAD);
    let first = controller.command(PlanarPose::IDENTITY, rolling, DT);
    assert_eq!(first.angular_z(), 0.0, "{first:?}");
    assert!(
        (first.linear_x() - (0.3 - LINEAR.deceleration * DT)).abs() < 1e-12,
        "{first:?}"
    );
}
