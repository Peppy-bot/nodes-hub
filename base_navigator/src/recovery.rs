//! The recoveries of a navigation, those of Nav2 1.5.2's default behavior tree
//! (nav2_bt_navigator navigate_to_pose_w_replanning_and_recovery.xml) in its
//! order: a clearing of the grid, a turn in place counterclockwise, a wait and
//! a back-up, each once; and the two motions the recoveries and the moves of
//! base_navigation share, a turn and a straight drive, each stepped at the
//! controller's steps.
//!
//! A [`Turn`] steps as Nav2's Spin does (nav2_behaviors plugins/spin.cpp
//! `onCycleUpdate`, `isCollisionFree`): it fails once its timeout has passed,
//! adds up the turn the odometry measures, and takes its rate from
//! navigation_core's port of Spin's law (spin.cpp:118-125), which ends the
//! turn within 1e-6 rad of its angle: the rate from which describe_base's
//! angular max_deceleration brings the base to rest at the end of the turn,
//! at least the larger of the slowest turn rate and the angular deadband and
//! at most the limit in the turn's direction. Each step it sweeps the padded
//! footprint turned at that rate a step at a time, up to the simulate-ahead
//! time and to less than the turn made so far, so the first step checks no
//! pose, and fails at the first pose that meets an occupied cell.
//!
//! A [`StraightDrive`] steps as Nav2's DriveOnHeading does
//! (include/nav2_behaviors/plugins/drive_on_heading.hpp `onCycleUpdate`,
//! `isCollisionFree`): it fails once its timeout has passed, is done once the
//! robot frame's origin stands its distance from where it started, and takes
//! its speed from navigation_core's port of DriveOnHeading's law
//! (drive_on_heading.hpp:142-164): its speed held within one step's change by
//! describe_base's max_acceleration and max_deceleration along x of its last
//! command, 0 before its first; then slowed to the speed from which that
//! max_deceleration brings the base to rest at its distance; then raised to
//! the larger of the slowest speed in its direction and the deadband along x.
//! The drive then commands at most the limit in its direction. Each step it
//! sweeps the padded footprint from where the base stands along its heading
//! at that speed a step at a time, up to the simulate-ahead time and to less
//! than the distance left.

use control_core::non_negative_finite::NonNegativeFinite;
use control_core::planar_pose::PlanarPose;
use control_core::positive_finite::PositiveFinite;
use control_core::twist::PlanarTwist;
use frame_tree::arithmetic::normalize_angle;
use navigation_core::PaddedFootprint;
use navigation_core::behaviors::{DriveOnHeadingParams, SpinParams};
use navigation_core::grid::CostGrid;

use crate::base::Base;
use crate::config::{MoveSettings, NavigatorConfig};

/// One recovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recovery {
    /// Empty the grid, as clear_grid does.
    Clear,
    /// Turn in place counterclockwise by the recovery angle.
    Turn,
    /// Stream zero velocity for the wait time.
    Wait,
    /// Drive backward by the back-up distance at the back-up speed.
    BackUp,
}

/// The recoveries of a navigation, in the order they run.
pub const RECOVERIES: [Recovery; 4] = [
    Recovery::Clear,
    Recovery::Turn,
    Recovery::Wait,
    Recovery::BackUp,
];

/// A turn's goal: the turn (rad), counterclockwise positive, and its timeout
/// (ns), none for no bound; a spin's, or the turn recovery's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinRequest {
    pub angle_rad: f64,
    pub timeout_ns: Option<i64>,
}

/// A straight drive's goal: the distance (m), the speed (m/s), and its
/// timeout (ns), none for no bound; a drive_on_heading's, a back_up's, or the
/// back-up recovery's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StraightRequest {
    pub distance_m: PositiveFinite,
    pub speed_m_s: PositiveFinite,
    pub timeout_ns: Option<i64>,
}

/// What one step of a turn or a straight drive gives.
#[derive(Clone, Debug, PartialEq)]
pub enum MotionStep {
    /// The command of this step.
    Command(PlanarTwist),
    /// The odometry measures the whole turn or the whole distance.
    Done,
    /// The sweep meets an occupied cell: why.
    Blocked(String),
    /// The timeout passed first: why.
    TimedOut(String),
}

/// How the motions step and sweep: the controller's step (s), and how many
/// steps a sweep checks at most, Nav2's `max_cycle_count`.
#[derive(Clone, Copy, Debug)]
pub struct Stepping {
    pub step_s: f64,
    pub sweep_steps: usize,
}

/// A turn in place by an angle, counterclockwise positive.
#[derive(Clone, Copy, Debug)]
pub struct Turn {
    angle_rad: f64,
    /// Spin's law for the base's limits in the turn's direction; none on a
    /// base whose angular axis has the range 0 to 0, which turns at no rate.
    law: Option<SpinParams>,
    stepping: Stepping,
    previous_yaw: f64,
    turned_rad: f64,
    timeout_end_ns: Option<i64>,
}

impl Turn {
    /// The turn of `request` for `base`, heading `yaw`, from `now_ns`, stepped
    /// and swept as `config` says.
    pub fn new(
        request: SpinRequest,
        base: &Base,
        config: &NavigatorConfig,
        yaw: f64,
        now_ns: i64,
    ) -> Self {
        let SpinRequest {
            angle_rad,
            timeout_ns,
        } = request;
        let (clockwise, counterclockwise) = base.angular.velocity_range();
        let fastest_rad_s = if angle_rad < 0.0 {
            clockwise.abs()
        } else {
            counterclockwise
        };
        let slowest_rad_s = config
            .moves
            .min_turn_rate_rad_s
            .get()
            .max(base.angular.deadband());
        let rate =
            |value: f64| NonNegativeFinite::parse(value).expect("a finite rate of 0 or above");
        let law = PositiveFinite::parse(base.angular.max_deceleration())
            .ok()
            .map(|deceleration| SpinParams {
                rotational_acc_lim_rad_s2: deceleration,
                min_rotational_vel_rad_s: rate(slowest_rad_s),
                max_rotational_vel_rad_s: rate(fastest_rad_s),
            });
        Self {
            angle_rad,
            law,
            stepping: config.stepping(),
            previous_yaw: yaw,
            turned_rad: 0.0,
            timeout_end_ns: timeout_ns.map(|timeout_ns| now_ns.saturating_add(timeout_ns)),
        }
    }

    /// The angle (rad) the odometry measured the base turn, counterclockwise
    /// positive, summed over the turn.
    pub fn turned_rad(&self) -> f64 {
        self.turned_rad
    }

    /// Take the odometry's heading `yaw` into the turn made so far.
    pub fn measure(&mut self, yaw: f64) {
        self.turned_rad += normalize_angle(yaw - self.previous_yaw);
        self.previous_yaw = yaw;
    }

    /// The step at `now_ns` of the base at `pose`, measured into the turn, on
    /// `grid`, with the padded footprint `footprint`.
    pub fn step(
        &self,
        pose: PlanarPose,
        now_ns: i64,
        grid: &CostGrid,
        footprint: &PaddedFootprint,
    ) -> MotionStep {
        if self.timeout_end_ns.is_some_and(|end_ns| now_ns > end_ns) {
            return MotionStep::TimedOut(format!(
                "the base turned {:.3} rad of {:.3} rad before the timeout passed",
                self.turned_rad, self.angle_rad
            ));
        }
        let remaining = self.angle_rad.abs() - self.turned_rad.abs();
        let Some(law) = &self.law else {
            // The base turns at no rate: a turn of 0 is done, and any other
            // commands 0 until its timeout.
            if remaining > 0.0 {
                return MotionStep::Command(PlanarTwist::ZERO);
            }
            return MotionStep::Done;
        };
        let Some(rate) = law.rate(self.angle_rad, remaining) else {
            return MotionStep::Done;
        };
        let step_s = self.stepping.step_s;
        let meets = (0..self.stepping.sweep_steps)
            .map(|k| rate * k as f64 * step_s)
            .take_while(|change| self.turned_rad.abs() - change.abs() > 0.0)
            .find(|change| {
                let turned = PlanarPose::new(pose.x(), pose.y(), pose.yaw() + change)
                    .expect("a finite heading turned by a finite angle");
                footprint.collides_at(grid, turned)
            });
        if let Some(change) = meets {
            return MotionStep::Blocked(format!(
                "the padded footprint meets an occupied cell {change:.3} rad into the turn ahead"
            ));
        }
        MotionStep::Command(PlanarTwist::new(0.0, 0.0, rate).expect("a finite turn rate"))
    }
}

/// The direction of a straight drive along the robot frame's x.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
}

impl Direction {
    /// The sign of a velocity along x in this direction.
    fn sign(self) -> f64 {
        match self {
            Self::Forward => 1.0,
            Self::Backward => -1.0,
        }
    }

    /// The point in odom `distance_m` from the robot frame's origin at `start`
    /// along its heading in this direction.
    pub fn end_of(self, start: PlanarPose, distance_m: f64) -> [f64; 2] {
        let (sin, cos) = start.yaw().sin_cos();
        let along = self.sign() * distance_m;
        [start.x() + along * cos, start.y() + along * sin]
    }

    /// The fastest speed (m/s) `base` executes in this direction, a
    /// magnitude.
    pub fn fastest_m_s(self, base: &Base) -> f64 {
        let (backward, forward) = base.linear_x.velocity_range();
        match self {
            Self::Forward => forward,
            Self::Backward => backward.abs(),
        }
    }

    /// The slowest speed (m/s) of the launch in this direction.
    fn slowest_m_s(self, moves: &MoveSettings) -> f64 {
        match self {
            Self::Forward => moves.min_forward_speed_m_s.get(),
            Self::Backward => moves.min_backward_speed_m_s.get(),
        }
    }
}

/// A drive along the robot frame's x, ahead or back, by a distance at a
/// speed.
#[derive(Clone, Copy, Debug)]
pub struct StraightDrive {
    direction: Direction,
    distance_m: f64,
    speed_m_s: f64,
    fastest_m_s: f64,
    /// DriveOnHeading's law for the base's limits along x.
    law: DriveOnHeadingParams,
    /// The velocity along x (m/s) of the drive's last command, 0 before its
    /// first.
    last_command_m_s: f64,
    stepping: Stepping,
    start: [f64; 2],
    timeout_end_ns: Option<i64>,
}

impl StraightDrive {
    /// The drive of `request` in `direction` for `base`, from `start` at
    /// `now_ns`, stepped and swept as `config` says.
    pub fn new(
        direction: Direction,
        request: StraightRequest,
        base: &Base,
        config: &NavigatorConfig,
        start: PlanarPose,
        now_ns: i64,
    ) -> Self {
        let limit = |value: f64| {
            PositiveFinite::parse(value)
                .expect("a limit along x of a base that drives forward, above 0")
        };
        let slowest_m_s = direction
            .slowest_m_s(&config.moves)
            .max(base.linear_x.deadband());
        Self {
            direction,
            distance_m: request.distance_m.get(),
            speed_m_s: request.speed_m_s.get(),
            fastest_m_s: direction.fastest_m_s(base),
            law: DriveOnHeadingParams {
                acceleration_limit_m_s2: limit(base.linear_x.max_acceleration()),
                deceleration_limit_m_s2: limit(base.linear_x.max_deceleration()),
                minimum_speed_m_s: NonNegativeFinite::parse(slowest_m_s)
                    .expect("a finite speed of 0 or above"),
                cycle_frequency_hz: config.follow.controller.controller_frequency_hz,
            },
            last_command_m_s: 0.0,
            stepping: config.stepping(),
            start: [start.x(), start.y()],
            timeout_end_ns: request
                .timeout_ns
                .map(|timeout_ns| now_ns.saturating_add(timeout_ns)),
        }
    }

    /// The drive, its speed at most `cap_m_s` whatever its slowest speed:
    /// the back-up recovery under a goal's max_speed_m_s.
    pub fn at_most(self, cap_m_s: f64) -> Self {
        Self {
            fastest_m_s: self.fastest_m_s.min(cap_m_s),
            ..self
        }
    }

    /// The distance (m) of the robot frame's origin at `pose` from where the
    /// drive started.
    pub fn traveled_m(&self, pose: PlanarPose) -> f64 {
        (pose.x() - self.start[0]).hypot(pose.y() - self.start[1])
    }

    /// The step at `now_ns` of the base at `pose`, on `grid`, with the padded
    /// footprint `footprint`; a command it gives is the drive's last.
    pub fn step(
        &mut self,
        pose: PlanarPose,
        now_ns: i64,
        grid: &CostGrid,
        footprint: &PaddedFootprint,
    ) -> MotionStep {
        let traveled = self.traveled_m(pose);
        if self.timeout_end_ns.is_some_and(|end_ns| now_ns > end_ns) {
            return MotionStep::TimedOut(format!(
                "the base drove {traveled:.3} m of {:.3} m before the timeout passed",
                self.distance_m
            ));
        }
        let remaining = self.distance_m - traveled;
        let sign = self.direction.sign();
        let Some(along) = self
            .law
            .speed(sign * self.speed_m_s, self.last_command_m_s, remaining)
        else {
            return MotionStep::Done;
        };
        let speed = along.abs().min(self.fastest_m_s);
        let (sin, cos) = pose.yaw().sin_cos();
        let step_s = self.stepping.step_s;
        let meets = (0..self.stepping.sweep_steps)
            .map(|k| speed * k as f64 * step_s)
            .take_while(|change| remaining - change > 0.0)
            .find(|change| {
                let ahead = PlanarPose::new(
                    pose.x() + sign * change * cos,
                    pose.y() + sign * change * sin,
                    pose.yaw(),
                )
                .expect("a finite pose moved a finite distance");
                footprint.collides_at(grid, ahead)
            });
        if let Some(change) = meets {
            return MotionStep::Blocked(format!(
                "the padded footprint meets an occupied cell {change:.3} m along the drive ahead"
            ));
        }
        self.last_command_m_s = sign * speed;
        MotionStep::Command(
            PlanarTwist::new(sign * speed, 0.0, 0.0).expect("a finite speed along x"),
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use control_core::non_negative_finite::NonNegativeFinite;
    use navigation_core::grid::{FREE, GridGeometry, LETHAL};

    use crate::base::tests::base;
    use crate::config::tests::config;

    /// A grid of 0.05 m cells 4 m on a side around the origin, the cells
    /// holding `lethal` occupied.
    pub(crate) fn grid(lethal: &[[f64; 2]]) -> CostGrid {
        let geometry =
            GridGeometry::new(80, 80, PositiveFinite::parse(0.05).unwrap(), [-2.0, -2.0]).unwrap();
        let mut costs = vec![FREE; geometry.cell_count()];
        for point in lethal {
            let cell = geometry.cell_of(*point).unwrap();
            costs[cell.y() as usize * 80 + cell.x() as usize] = LETHAL;
        }
        CostGrid::from_costs(geometry, costs).unwrap()
    }

    // A sweep checks as many steps as Nav2's max_cycle_count, the rate times
    // the simulate-ahead time: 6 for 0.3 s at 20 Hz and 40 for 2 s.
    #[test]
    fn a_sweep_checks_the_rate_times_the_simulate_ahead_time_in_steps() {
        let swept = |simulate_ahead_time_s| {
            let mut config = config();
            config.moves.simulate_ahead_time_s =
                NonNegativeFinite::parse(simulate_ahead_time_s).unwrap();
            config.stepping().sweep_steps
        };
        assert_eq!(swept(0.3), 6);
        assert_eq!(swept(2.0), 40);
    }

    fn at(x: f64, yaw: f64) -> PlanarPose {
        PlanarPose::new(x, 0.0, yaw).expect("a finite pose")
    }

    fn turn(angle_rad: f64, timeout_ns: Option<i64>) -> Turn {
        let request = SpinRequest {
            angle_rad,
            timeout_ns,
        };
        Turn::new(request, &base(2.0), &config(), 3.0, 0)
    }

    // A turn turns in its direction at the base's limit, slows into the end of
    // its turn, keeps adding up across the cut at pi, and is done once the
    // odometry has turned the whole angle.
    #[test]
    fn a_turn_turns_its_angle_and_is_done() {
        let footprint = base(2.0).padded_footprint;
        let free = grid(&[]);
        for angle in [1.57, -1.57] {
            let mut turn = turn(angle, None);
            let MotionStep::Command(first) = turn.step(at(0.0, 3.0), 0, &free, &footprint) else {
                panic!("a free turn turns");
            };
            assert_eq!(first.angular_z(), 1.5_f64.copysign(angle));
            let near_end = normalize_angle(3.0 + angle - 0.001_f64.copysign(angle));
            turn.measure(near_end);
            let MotionStep::Command(slow) = turn.step(at(0.0, near_end), 1, &free, &footprint)
            else {
                panic!("the turn has 0.001 rad to go");
            };
            assert!(
                (slow.angular_z().abs() - 0.4).abs() < 1e-12,
                "the slowest turn rate: {slow:?}"
            );
            let end = normalize_angle(3.0 + angle);
            turn.measure(end);
            assert!((turn.turned_rad() - angle).abs() < 1e-9);
            assert_eq!(
                turn.step(at(0.0, end), 2, &free, &footprint),
                MotionStep::Done
            );
        }
    }

    // The first step of a turn checks no pose, so a box under the
    // footprint's outline both where the base stands and 0.05 rad on blocks
    // the turn only once it has turned; past its timeout a turn fails.
    #[test]
    fn a_turn_beside_a_box_is_blocked_once_it_turned_and_times_out() {
        let footprint = base(2.0).padded_footprint;
        let box_cell = [-0.075, 0.225];
        for yaw in [3.0, 3.05] {
            assert!(
                footprint.collides_at(&grid(&[box_cell]), at(0.0, yaw)),
                "the outline covers the box at {yaw} rad"
            );
        }
        let beside = grid(&[box_cell]);
        let mut turn = turn(1.57, Some(1_000_000_000));
        assert!(matches!(
            turn.step(at(0.0, 3.0), 0, &beside, &footprint),
            MotionStep::Command(_)
        ));
        turn.measure(3.05);
        assert!(matches!(
            turn.step(at(0.0, 3.05), 50_000_000, &beside, &footprint),
            MotionStep::Blocked(_)
        ));
        let late = turn.step(at(0.0, 3.05), 1_000_000_001, &grid(&[]), &footprint);
        assert!(matches!(late, MotionStep::TimedOut(_)), "{late:?}");
    }

    fn drive(direction: Direction, distance_m: f64) -> StraightDrive {
        let request = StraightRequest {
            distance_m: PositiveFinite::parse(distance_m).unwrap(),
            speed_m_s: PositiveFinite::parse(0.3).unwrap(),
            timeout_ns: None,
        };
        StraightDrive::new(direction, request, &base(1.0), &config(), at(0.0, 0.0), 0)
    }

    /// The speeds (m/s) along x that `drive` commands at `steps` steps with
    /// the base at `pose`, on a free grid.
    fn commands(drive: &mut StraightDrive, pose: PlanarPose, steps: usize) -> Vec<f64> {
        let footprint = base(1.0).padded_footprint;
        let free = grid(&[]);
        (0..steps)
            .map(|_| match drive.step(pose, 0, &free, &footprint) {
                MotionStep::Command(command) => command.linear_x(),
                other => panic!("a free drive commands: {other:?}"),
            })
            .collect()
    }

    // A straight drive ramps from rest toward its speed, one step of
    // describe_base's 1 m/s^2 each, from the slowest speed, 0.1 m/s; at its
    // speed it slows to stop at its distance, sqrt(2 x 1 m/s^2 x 0.02 m) =
    // 0.2 m/s with 0.02 m to go, and is done at its distance.
    #[test]
    fn a_straight_drive_ramps_to_its_speed_slows_into_its_distance_and_is_done() {
        let footprint = base(1.0).padded_footprint;
        let free = grid(&[]);
        let mut back = drive(Direction::Backward, 1.0);
        let ramp = commands(&mut back, at(0.0, 0.0), 6);
        let expected = [-0.1, -0.15, -0.2, -0.25, -0.3, -0.3];
        assert!(
            ramp.iter()
                .zip(expected)
                .all(|(got, want)| (got - want).abs() < 1e-12),
            "{ramp:?}"
        );
        let MotionStep::Command(slowing) = back.step(at(-0.98, 0.0), 0, &free, &footprint) else {
            panic!("0.02 m to go");
        };
        assert!(
            (slowing.linear_x() + 0.2).abs() < 1e-12,
            "sqrt(2 x 1 m/s^2 x 0.02 m) = 0.2 m/s: {slowing:?}"
        );
        assert_eq!(
            back.step(at(-1.0, 0.0), 0, &free, &footprint),
            MotionStep::Done
        );
    }

    // A drive held at most to a cap commands no more than the cap, and
    // reaches it, below its speed and below its slowest speed alike.
    #[test]
    fn a_drive_at_most_a_cap_commands_no_more_than_the_cap() {
        for cap in [0.2, 0.05] {
            let mut capped = drive(Direction::Backward, 1.0).at_most(cap);
            let speeds = commands(&mut capped, at(0.0, 0.0), 6);
            assert!(
                speeds.iter().all(|speed| -speed <= cap) && speeds[5] == -cap,
                "{cap} m/s: {speeds:?}"
            );
        }
    }

    // A box ahead within the simulate-ahead time at the drive's speed
    // blocks a drive toward it once its ramp reaches that speed, and one
    // beyond its distance, or behind it, does not.
    #[test]
    fn a_box_ahead_blocks_a_drive_toward_it_within_its_distance() {
        let footprint = base(1.0).padded_footprint;
        let ahead = grid(&[[0.6, 0.0]]);
        let steps = |mut drive: StraightDrive| -> Vec<MotionStep> {
            (0..6)
                .map(|_| drive.step(at(0.0, 0.0), 0, &ahead, &footprint))
                .collect()
        };
        let toward = steps(drive(Direction::Forward, 1.0));
        assert!(
            matches!(toward.last(), Some(MotionStep::Blocked(_))),
            "{toward:?}"
        );
        for clear in [
            steps(drive(Direction::Forward, 0.1)),
            steps(drive(Direction::Backward, 1.0)),
        ] {
            assert!(
                clear
                    .iter()
                    .all(|step| matches!(step, MotionStep::Command(_))),
                "{clear:?}"
            );
        }
    }
}
