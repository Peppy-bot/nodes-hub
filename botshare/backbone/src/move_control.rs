//! move_base on a differential base: the turn toward the target, the drive
//! to it and the turn to the target yaw, planned against describe_base's limits,
//! then steered each step on the base's odometry through base_state's
//! limiter.
//!
//! The plan times each leg as the fastest move from rest to rest under the
//! axis's speed, acceleration and deceleration limits. A longer requested
//! duration scales each leg's speed by the fastest duration over the
//! requested one, and the planned end lies at the requested duration.
//!
//! A turn commands Nav2 1.5.2's Spin rate (nav2_behaviors plugins/spin.cpp,
//! `Spin::onCycleUpdate`, lines 124-125): the speed from which the
//! deceleration stops it on its angle, at most the leg's speed, and at least
//! the node's `min_turn_rate_rad_s`, Spin's min_rotational_vel, within the
//! axis's speed the way it turns. The drive commands Nav2 1.5.2's
//! DriveOnHeading velocity (nav2_behaviors include/nav2_behaviors/plugins/
//! drive_on_heading.hpp, `DriveOnHeading::onCycleUpdate`, lines 142-164):
//! angular 0, the leg's speed reached from the limiter's velocity by at most
//! acceleration x dt and deceleration x dt a step, at most the speed from
//! which the deceleration stops the base on the distance that remains along
//! the heading, and at least the node's `min_forward_speed_m_s` ahead and
//! `min_backward_speed_m_s` backward, DriveOnHeading's and BackUp's
//! minimum_speed, within the axis's speed that way. The drive ramps from the
//! limiter's velocity, where DriveOnHeading ramps from its own last command,
//! `last_vel_` (drive_on_heading.hpp:142, :175). That speed is
//! [`AxisProfile::stopping_speed`], Spin's and DriveOnHeading's
//! sqrt(2 x deceleration x remaining) taken on the limiter's steps. Each
//! floor is at least just above the axis's deadband, below which base_state's
//! limiter executes the axis as 0 ([`leg_floor`]).
//!
//! A leg ends at the stop nearest its end ([`AxisProfile::ends`]): at the
//! step at which the limiter, brought to 0 at the deceleration from its
//! velocity, stops the base on or past the end, or nearer it than the stop
//! from the next step that executes at the leg's command. A step executes at
//! or above the axis's deadband, and one below it moves nothing. A leg on its
//! stopping speed so ends with the stop landing on its end. Spin ends once
//! less than 1e-6 rad of its angle remains (spin.cpp:118-122), DriveOnHeading
//! once it has driven its distance (drive_on_heading.hpp:131-134), each
//! leaving the stop to the velocity smoother, which applies its deadband to
//! what it sends after it keeps the velocity it steps from, `last_cmd_`
//! (nav2_velocity_smoother/src/velocity_smoother.cpp:468, :471-481); the end
//! here counts only the steps that execute. A leg that holds its floor f
//! moves f x dt a step, and ends with its stop landing within f x dt / 2 of
//! its end. From rest, the shortest move a leg makes is the limiter's ramp
//! toward its floor up to its first step that executes, and the stop from
//! there ([`AxisProfile::shortest_move`]), so a leg at rest within half of
//! that of its end ends where it stands. A leg that ends with the base
//! outside the tolerances starts again from where the base stands. move_base
//! refuses a position tolerance below the shortest drive and a yaw tolerance
//! below the shortest turn ([`MoveAxes`]).
//!
//! Each leg runs to its end, and the next starts once the last has come to
//! 0. The drive starts when it runs from where the base stands, and the
//! base turns toward the target otherwise. The drive holds its aim: once the
//! target's bearing leaves the yaw tolerance, or the node's
//! `rotate_to_heading_min_angle_rad` when the tolerance is wider, the base
//! turns toward the target again and drives on.

use std::f64::consts::PI;

use control_core::chase::rate_limited;
use control_core::planar_pose::PlanarPose;
use control_core::twist::PlanarTwist;
use frame_tree::arithmetic::normalize_angle;

/// How far above its deadband a floored command stands, as a share of the
/// deadband: past the rounding of the limiter's sums, which can land a
/// command of exactly the deadband just below it. Each step's sum rounds by
/// at most half a unit in the last place of the axis's speed, 2^-53 of it,
/// and the error of a ramp grows by that at most each step, so this margin
/// covers 1e-6 x 2^53 (about 9e9) steps' roundings times the deadband over
/// the speed: for a deadband a thousandth of the speed, nine million steps,
/// two and a half hours of ramping at the limiter's fastest rate, 1 kHz.
const ABOVE_DEADBAND: f64 = 1e-6;

/// The slowest speed (a magnitude) a leg commands toward a way whose fastest
/// speed is `fastest`: `slowest`, the node's minimum that way, at least just
/// above the axis's `deadband` and at most `fastest`, as Spin clamps its rate
/// between min_rotational_vel and max_rotational_vel (spin.cpp:125).
pub fn leg_floor(slowest: f64, deadband: f64, fastest: f64) -> f64 {
    slowest.max(deadband * (1.0 + ABOVE_DEADBAND)).min(fastest)
}

/// One way an axis moves: the speed of a leg that way and its floor, the
/// slowest speed such a leg commands while it runs ([`leg_floor`]), each a
/// magnitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Way {
    pub speed: f64,
    pub floor: f64,
}

/// The limits of one axis of the base that move_base plans and steers
/// under: the way toward the axis's positive direction (ahead, or
/// counterclockwise) and the way toward its negative one, its acceleration
/// and deceleration, each above 0, and its deadband, below which base_state's
/// limiter executes the axis as 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisProfile {
    pub positive: Way,
    pub negative: Way,
    pub acceleration: f64,
    pub deceleration: f64,
    pub deadband: f64,
}

impl AxisProfile {
    /// The way toward the axis's positive direction when `positive`, and
    /// toward its negative one otherwise.
    fn way(&self, positive: bool) -> Way {
        if positive {
            self.positive
        } else {
            self.negative
        }
    }

    /// This profile with each way's speed times `scale`, its floors kept.
    fn scaled(&self, scale: f64) -> Self {
        let scaled = |way: Way| Way {
            speed: way.speed * scale,
            ..way
        };
        Self {
            positive: scaled(self.positive),
            negative: scaled(self.negative),
            ..*self
        }
    }

    /// The duration (s) of the fastest move over `distance` from rest to
    /// rest, toward the axis's negative direction when negative: a ramp up,
    /// a cruise at that way's speed when the distance allows one, and a
    /// ramp down.
    pub fn travel_time(&self, distance: f64) -> f64 {
        let Self {
            acceleration,
            deceleration,
            ..
        } = *self;
        let speed = self.way(distance >= 0.0).speed;
        let distance = distance.abs();
        if distance == 0.0 {
            return 0.0;
        }
        let ramps = speed * speed / (2.0 * acceleration) + speed * speed / (2.0 * deceleration);
        if distance >= ramps {
            return speed / acceleration + speed / deceleration + (distance - ramps) / speed;
        }
        let peak =
            (2.0 * distance * acceleration * deceleration / (acceleration + deceleration)).sqrt();
        peak / acceleration + peak / deceleration
    }

    /// The distance the axis covers, signed as `speed`, while base_state's
    /// limiter brings `speed` to rest, slowing it by one step of the
    /// deceleration, `a` = deceleration x `dt`, at each step of `dt` (s): a
    /// step at each of the magnitudes |speed| - a, |speed| - 2a, ... above 0
    /// that executes, at or above the deadband.
    fn stopping_distance(&self, speed: f64, dt: f64) -> f64 {
        let step = self.deceleration * dt;
        let magnitude = speed.abs();
        let above_zero = (magnitude / step).ceil() - 1.0;
        let executed = ((magnitude - self.deadband) / step).floor();
        let steps = above_zero.min(executed).max(0.0);
        speed.signum() * dt * steps * (magnitude - step * (steps + 1.0) / 2.0)
    }

    /// The speed base_state's limiter steps `speed` to over one step of `dt`
    /// (s) toward `command`, both signed the same way: by at most
    /// acceleration x `dt` while `command` is at least as fast and not the
    /// other way, and by at most deceleration x `dt` otherwise.
    fn next_speed(&self, speed: f64, command: f64, dt: f64) -> f64 {
        let rate = if command.abs() >= speed.abs() && command * speed >= 0.0 {
            self.acceleration
        } else {
            self.deceleration
        };
        rate_limited(speed, command, rate, dt)
    }

    /// The first speed base_state's limiter steps `speed` to toward
    /// `command`, over steps of `dt` (s), that executes, at or above the
    /// deadband, each step before it executing 0: none when the ramp comes
    /// to `command` without one. It looks as many steps ahead as the slower
    /// step of the acceleration and deceleration takes over |`speed`| and the
    /// deadband, by which the ramp has crossed the deadband or come to
    /// `command`.
    fn first_executing(&self, speed: f64, command: f64, dt: f64) -> Option<f64> {
        let slowest_step = self.acceleration.min(self.deceleration) * dt;
        let ramp = ((speed.abs() + self.deadband) / slowest_step).ceil() as usize + 1;
        std::iter::successors(Some(speed), |&last| {
            Some(self.next_speed(last, command, dt))
        })
        .skip(1)
        .take(ramp)
        .find(|next| next.abs() >= self.deadband)
    }

    /// Whether a leg along the axis ends at this step of `dt` (s), with
    /// `remaining` of it left, past its end when negative, base_state's
    /// limiter at `speed` and the leg commanding `command` were it to go on,
    /// each signed along the way the leg goes: the limiter's stop from
    /// `speed` lands on or past the end, or nearer it than the stop from the
    /// next step that executes at `command` lands past it, or no step at
    /// `command` executes ([`AxisProfile::first_executing`]). The leg so ends
    /// at the stop nearest its end of those it reaches a step at a time.
    fn ends(&self, remaining: f64, speed: f64, command: f64, dt: f64) -> bool {
        let short_now = remaining - self.stopping_distance(speed, dt);
        let Some(next) = self.first_executing(speed, command, dt) else {
            return true;
        };
        let short_next = remaining - next * dt - self.stopping_distance(next, dt);
        short_now <= (-short_next).max(0.0)
    }

    /// The shortest move a leg along the axis makes from rest over steps of
    /// `dt` (s), the longer of each way the axis moves: base_state's
    /// limiter's ramp toward the way's floor up to its first step that
    /// executes ([`AxisProfile::first_executing`]), and its stop from there.
    /// A leg that rests within half of its own way's shortest move of its end
    /// ends where it stands, and one that rests farther runs and moves the
    /// base ([`AxisProfile::ends`]), so half of this bounds both ways: a leg
    /// commands at least its floor, and a command above it is at most the
    /// stopping speed of what remains, whose move covers what remains. A way
    /// whose floor lies below the deadband never executes.
    fn shortest_move(&self, dt: f64) -> f64 {
        [self.positive, self.negative]
            .into_iter()
            .filter(|way| way.speed > 0.0)
            .filter_map(|way| self.first_executing(0.0, way.floor, dt))
            .map(|first| first * dt + self.stopping_distance(first, dt))
            .fold(0.0, f64::max)
    }

    /// The speed (a magnitude) from which base_state's limiter, slowing the
    /// axis by one step of the deceleration, `a` = deceleration x `dt`, at
    /// each step of `dt` (s), covers exactly `remaining`: the first of the
    /// speeds f + n a, ..., f + a, f, with 0 < f <= a, that cover it a step
    /// each. On the next step what remains takes the speed one `a` lower, so
    /// the limiter follows it and the axis stops on the end: Spin's and
    /// DriveOnHeading's sqrt(2 x deceleration x remaining), taken on the
    /// limiter's steps.
    fn stopping_speed(&self, remaining: f64, dt: f64) -> f64 {
        let step = self.deceleration * dt;
        // What remains in steps of `a` covered over one `dt`: the speeds n a,
        // ..., a cover n (n + 1) / 2 of them.
        let steps = remaining / (step * dt);
        let triangle = |n: f64| n * (n + 1.0) / 2.0;
        let mut n = (((1.0 + 8.0 * steps).sqrt() - 1.0) / 2.0).floor();
        while n > 0.0 && triangle(n) >= steps {
            n -= 1.0;
        }
        while triangle(n + 1.0) < steps {
            n += 1.0;
        }
        let last = step * (steps - triangle(n)) / (n + 1.0);
        last + n * step
    }

    /// Spin's rate (rad/s), signed, of a turn going `turning` with
    /// `remaining` (rad) of it left, past its angle when negative, over a
    /// step of `dt` (s).
    fn turn_rate(&self, turning: Turning, remaining: f64, dt: f64) -> f64 {
        let way = self.way(turning == Turning::Counterclockwise);
        let rate = self.stopping_speed(remaining.max(0.0), dt).min(way.speed);
        turning.sign() * rate.max(way.floor)
    }

    /// DriveOnHeading's velocity (m/s), signed, toward a target `ahead` (m)
    /// along the heading, behind the base when negative, base_state's
    /// limiter at `speed` (m/s), over a step of `dt` (s).
    fn drive_velocity(&self, ahead: f64, speed: f64, dt: f64) -> f64 {
        let forward = ahead >= 0.0;
        let way = self.way(forward);
        let direction = drive_direction(ahead);
        let (lowest, highest) = if forward {
            (
                speed - self.deceleration * dt,
                speed + self.acceleration * dt,
            )
        } else {
            (
                speed - self.acceleration * dt,
                speed + self.deceleration * dt,
            )
        };
        let ramped = (direction * way.speed).clamp(lowest, highest);
        let to_stop = self.stopping_speed(ahead.abs(), dt);
        let capped = if to_stop < ramped.abs() {
            direction * to_stop
        } else {
            ramped
        };
        if capped.abs() < way.floor {
            direction * way.floor
        } else {
            capped
        }
    }
}

/// The direction a drive toward a target `ahead` (m) along the heading goes:
/// 1 ahead, -1 back toward a target behind the base.
fn drive_direction(ahead: f64) -> f64 {
    if ahead >= 0.0 { 1.0 } else { -1.0 }
}

/// How far below the finest a value lies, as a share of the finest, and is
/// still taken: past the rounding of the sums that give the finest, which
/// lands botshare_sim's 2.5 mm at 0.0025000000000000005.
const FINEST_SLACK: f64 = 1e-9;
/// The significant digits a refusal shows the finest with.
const SHOWN_DIGITS: i32 = 9;
/// The longest shortest turn from rest (rad) the node takes ([`MoveAxes`]):
/// at it, (S_d / 2) / cos(1) is 0.93 S_d, 7% inside S_d, past the rounding
/// of the sums and so within the finest position tolerance the node takes,
/// S_d less [`FINEST_SLACK`] of it.
pub const LONGEST_SHORTEST_TURN: f64 = 2.0;

/// Whether `value` is finer than `finest`: below it by more than
/// [`FINEST_SLACK`] of it.
pub fn finer_than(value: f64, finest: f64) -> bool {
    value < finest - finest * FINEST_SLACK
}

/// `finest` as a refusal shows it: rounded up to [`SHOWN_DIGITS`]
/// significant digits from half [`FINEST_SLACK`] below it, a value no finer
/// than `finest` ([`finer_than`]), the other half of the slack covering the
/// rounding of the scaling. A finest of 0, or one whose scale or rounding is
/// not finite, shows as itself.
pub fn shown_finest(finest: f64) -> f64 {
    if finest == 0.0 {
        return finest;
    }
    let from = finest - finest * FINEST_SLACK / 2.0;
    let scale = 10f64.powi(SHOWN_DIGITS - 1 - from.log10().floor() as i32);
    let shown = (from * scale).ceil() / scale;
    if scale.is_finite() && shown.is_finite() {
        shown
    } else {
        finest
    }
}

/// The axes move_base plans and steers along, and the finest tolerances a
/// move along them takes, S_d and S_t: the shortest drive and the shortest
/// turn from rest ([`AxisProfile::shortest_move`]).
///
/// S_d and S_t count only the steps that execute, at or above the axis's
/// deadband, so from rest a drive runs and moves the base once its target
/// lies more than S_d / 2 ahead or behind, and a turn once more than S_t / 2
/// of it remains ([`AxisProfile::shortest_move`]). A move takes a
/// position tolerance of at least S_d and a yaw tolerance of at least S_t,
/// and its drive aims within the smaller of the yaw tolerance and the
/// rotate-to-heading angle, which the node holds to at least S_t. So a base
/// at rest outside its tolerances has a leg that runs. Within the position
/// tolerance, a heading outside the yaw tolerance lies more than S_t from the
/// target's, and the turn to the yaw runs. Outside it, a
/// target behind the base or past the aim lies at a bearing past S_t / 2,
/// and the turn toward it runs. A target ahead within the aim is driven to,
/// or turned toward, unless it lies at most S_d / 2 ahead at a bearing of at
/// most S_t / 2, at most (S_d / 2) / cos(S_t / 2) away: within the position
/// tolerance while S_t is at most 2 rad, the most the node takes
/// ([`LONGEST_SHORTEST_TURN`]).
///
/// A drive that lands past its target backs up to it on a base that drives
/// backward, and turns toward it again on one that does not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoveAxes {
    pub linear: AxisProfile,
    pub angular: AxisProfile,
    pub finest: Tolerances,
}

impl MoveAxes {
    /// The axes `linear` and `angular` under base_state's limiter stepping
    /// every `dt` (s).
    pub fn new(linear: AxisProfile, angular: AxisProfile, dt: f64) -> Self {
        Self {
            linear,
            angular,
            finest: Tolerances {
                position: linear.shortest_move(dt),
                yaw: angular.shortest_move(dt),
            },
        }
    }
}

/// How close a move has to come: the robot frame's origin within
/// `position` (m) of the target's, its heading within `yaw` (rad).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerances {
    pub position: f64,
    pub yaw: f64,
}

/// The planned motion of a move: the limits each leg runs under, the speed
/// of each scaled to the requested duration, and how long the motion takes
/// (s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MovePlan {
    pub linear: AxisProfile,
    pub angular: AxisProfile,
    pub duration_s: f64,
}

/// The legs of a move from the goal's frame's origin to `target`: the turn
/// toward it, the drive ahead to it and the turn to its yaw (rad, m, rad),
/// each turn the shorter way, clockwise when negative. A target within
/// `position_tolerance` of the origin is a turn in place.
fn legs(target: PlanarPose, position_tolerance: f64) -> [f64; 3] {
    let distance = target.x().hypot(target.y());
    if distance <= position_tolerance {
        return [0.0, 0.0, normalize_angle(target.yaw())];
    }
    let bearing = target.y().atan2(target.x());
    [bearing, distance, normalize_angle(target.yaw() - bearing)]
}

/// Plan the move to `target` under the base's linear and angular limits,
/// each leg at the speed of the way it goes, taking `requested_s` (s),
/// floored at the fastest the limits allow.
pub fn plan_move(
    target: PlanarPose,
    position_tolerance: f64,
    linear: AxisProfile,
    angular: AxisProfile,
    requested_s: f64,
) -> MovePlan {
    let [turn_toward, drive, turn_to_yaw] = legs(target, position_tolerance);
    let fastest = angular.travel_time(turn_toward)
        + linear.travel_time(drive)
        + angular.travel_time(turn_to_yaw);
    let duration_s = requested_s.max(fastest);
    let scale = if fastest > 0.0 {
        fastest / duration_s
    } else {
        1.0
    };
    MovePlan {
        linear: linear.scaled(scale),
        angular: angular.scaled(scale),
        duration_s,
    }
}

/// The way a turn goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Turning {
    Counterclockwise,
    Clockwise,
}

impl Turning {
    /// The way that turns through `angle` (rad), counterclockwise when
    /// positive.
    fn through(angle: f64) -> Self {
        if angle >= 0.0 {
            Self::Counterclockwise
        } else {
            Self::Clockwise
        }
    }

    fn sign(self) -> f64 {
        match self {
            Self::Counterclockwise => 1.0,
            Self::Clockwise => -1.0,
        }
    }

    /// What remains (rad) of a turn going this way with `error` (rad) to
    /// turn through: negative once the turn passes its angle.
    fn remaining(self, error: f64) -> f64 {
        self.sign() * error
    }
}

/// The leg a move is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leg {
    TurnToward(Turning),
    Drive,
    TurnToYaw(Turning),
    /// Within both tolerances of the target: the move commands rest.
    Arrived,
}

/// Where the target stands from a pose of the robot frame.
#[derive(Clone, Copy, Debug)]
struct Offset {
    /// From the robot frame's origin to the target's (m).
    distance: f64,
    /// The angle from the heading to the target's origin (rad).
    bearing: f64,
    /// How far ahead along the heading the target's origin stands (m),
    /// behind when negative.
    ahead: f64,
    /// The shorter angle from the heading to the target yaw (rad).
    yaw: f64,
}

impl Offset {
    fn of(target: PlanarPose, pose: PlanarPose) -> Self {
        let (dx, dy) = (target.x() - pose.x(), target.y() - pose.y());
        let [ahead, _] = pose.apply_inverse([target.x(), target.y()]);
        Self {
            distance: pose.distance_to(target),
            bearing: normalize_angle(dy.atan2(dx) - pose.yaw()),
            ahead,
            yaw: pose.yaw_to(target),
        }
    }
}

/// The steering of one move, in the goal's frame.
#[derive(Debug)]
pub struct MoveController {
    target: PlanarPose,
    plan: MovePlan,
    tolerances: Tolerances,
    /// The largest bearing (rad) at which the drive goes on toward its
    /// target, whatever the yaw tolerance, as Nav2's regulated pure pursuit
    /// turns in place toward its path past `rotate_to_heading_min_angle`.
    rotate_to_heading_min_angle: f64,
    /// The leg of the last command, none before the first.
    leg: Option<Leg>,
}

impl MoveController {
    pub fn new(
        target: PlanarPose,
        plan: MovePlan,
        tolerances: Tolerances,
        rotate_to_heading_min_angle: f64,
    ) -> Self {
        Self {
            target,
            plan,
            tolerances,
            rotate_to_heading_min_angle,
            leg: None,
        }
    }

    /// The distance (m) and the magnitude of the shorter angle (rad) from
    /// `pose` to the target.
    pub fn remaining(&self, pose: PlanarPose) -> (f64, f64) {
        (
            pose.distance_to(self.target),
            pose.yaw_to(self.target).abs(),
        )
    }

    /// Whether `pose` stands within both tolerances of the target.
    pub fn within_tolerances(&self, pose: PlanarPose) -> bool {
        let (distance, yaw) = self.remaining(pose);
        distance <= self.tolerances.position && yaw <= self.tolerances.yaw
    }

    /// Spin's rate (rad/s) of a turn going `turning` toward `angle` (rad)
    /// from the heading, base_state's limiter turning at `rate` (rad/s), over
    /// a step of `dt` (s): none once the turn ends ([`AxisProfile::ends`]).
    fn turn(&self, turning: Turning, angle: f64, rate: f64, dt: f64) -> Option<f64> {
        let angular = self.plan.angular;
        let remaining = turning.remaining(angle);
        let command = angular.turn_rate(turning, remaining, dt);
        let along = |value: f64| turning.sign() * value;
        (!angular.ends(remaining, along(rate), along(command), dt)).then_some(command)
    }

    /// DriveOnHeading's velocity (m/s) toward a target `ahead` (m) along the
    /// heading, behind the base when negative, base_state's limiter at
    /// `speed` (m/s), over a step of `dt` (s): none once the drive ends
    /// ([`AxisProfile::ends`]).
    fn drive(&self, ahead: f64, speed: f64, dt: f64) -> Option<f64> {
        let linear = self.plan.linear;
        let command = linear.drive_velocity(ahead, speed, dt);
        let along = |value: f64| drive_direction(ahead) * value;
        (!linear.ends(ahead.abs(), along(speed), along(command), dt)).then_some(command)
    }

    /// The leg the base is on with the target at `offset`, base_state's
    /// limiter at `velocity`, over a step of `dt` (s), and the leg's command
    /// along its axis: m/s along x on the drive and rad/s about z on a turn,
    /// 0 on a leg that ended and on arrival. A drive runs to its end
    /// ([`MoveController::drive`]), the target ahead, while its bearing lies
    /// within the yaw tolerance and the rotate-to-heading angle. Within the
    /// position tolerance: the turn to the yaw, which starts while the
    /// heading lies outside the yaw tolerance and runs to its end. Outside
    /// it: the turn toward the target, which runs to its end, and the drive,
    /// which starts when it runs with the target ahead and goes on while its
    /// bearing lies within the same bound, its rear toward a target it passed
    /// on a base that drives backward, while it runs and while the limiter
    /// brings its speed to 0 after its end.
    fn next_leg(&self, offset: Offset, velocity: PlanarTwist, dt: f64) -> (Leg, f64) {
        let tolerances = self.tolerances;
        let aim = tolerances.yaw.min(self.rotate_to_heading_min_angle);
        let turn = |leg: fn(Turning) -> Leg, turning: Turning, angle: f64| {
            self.turn(turning, angle, velocity.angular_z(), dt)
                .map(|command| (leg(turning), command))
        };
        let started_turn = |leg: fn(Turning) -> Leg, angle: f64| {
            let turning = Turning::through(angle);
            turn(leg, turning, angle).unwrap_or((leg(turning), 0.0))
        };
        let drive = self
            .drive(offset.ahead, velocity.linear_x(), dt)
            .map(|command| (Leg::Drive, command));
        let on_drive = self.leg == Some(Leg::Drive);
        let aimed_ahead = offset.ahead > 0.0 && offset.bearing.abs() <= aim;
        if on_drive
            && aimed_ahead
            && let Some(driving) = drive
        {
            return driving;
        }
        if offset.distance <= tolerances.position {
            if let Some(Leg::TurnToYaw(turning)) = self.leg
                && let Some(kept) = turn(Leg::TurnToYaw, turning, offset.yaw)
            {
                return kept;
            }
            if offset.yaw.abs() <= tolerances.yaw {
                return (Leg::Arrived, 0.0);
            }
            return started_turn(Leg::TurnToYaw, offset.yaw);
        }
        if let Some(Leg::TurnToward(turning)) = self.leg
            && let Some(kept) = turn(Leg::TurnToward, turning, offset.bearing)
        {
            return kept;
        }
        let steering = if offset.ahead >= 0.0 {
            offset.bearing
        } else {
            normalize_angle(offset.bearing + PI)
        };
        let drives_that_way = offset.ahead >= 0.0 || self.plan.linear.negative.speed > 0.0;
        if on_drive && steering.abs() <= aim && drives_that_way {
            if let Some(driving) = drive {
                return driving;
            }
            if velocity.linear_x() != 0.0 {
                return (Leg::Drive, 0.0);
            }
        }
        if aimed_ahead && let Some(driving) = drive {
            return driving;
        }
        started_turn(Leg::TurnToward, offset.bearing)
    }

    /// The velocity to command at `pose`, base_state's limiter at `velocity`,
    /// over a step of `dt` (s): zero once `pose` stands within both
    /// tolerances, and on the axis of a leg that ended. A leg on one axis
    /// starts once the limiter's velocity on the other has come to 0, the
    /// other brought down by its deceleration over each step meanwhile, so
    /// each leg runs on one axis through base_state's limiter, which scales
    /// the axes together.
    pub fn command(&mut self, pose: PlanarPose, velocity: PlanarTwist, dt: f64) -> PlanarTwist {
        let (leg, along) = self.next_leg(Offset::of(self.target, pose), velocity, dt);
        let (speed, rate) = (velocity.linear_x(), velocity.angular_z());
        let (linear_x, angular_z) = match leg {
            Leg::Drive => (along, 0.0),
            Leg::TurnToward(_) | Leg::TurnToYaw(_) => (0.0, along),
            Leg::Arrived => (0.0, 0.0),
        };
        let MovePlan {
            linear, angular, ..
        } = self.plan;
        let wound_down =
            |last: f64, axis: AxisProfile| rate_limited(last, 0.0, axis.deceleration, dt);
        let (linear_x, angular_z) = if angular_z != 0.0 && speed != 0.0 {
            (wound_down(speed, linear), 0.0)
        } else if linear_x != 0.0 && rate != 0.0 {
            (0.0, wound_down(rate, angular))
        } else {
            (linear_x, angular_z)
        };
        self.leg = Some(leg);
        PlanarTwist::new(linear_x, 0.0, angular_z).expect("a command of finite poses")
    }
}

#[cfg(test)]
mod tests;
