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
//! A turn follows Nav2 1.5.2's Spin (nav2_behaviors plugins/spin.cpp,
//! `Spin::onCycleUpdate`, lines 118-130): it turns the way it began at the
//! speed from which the deceleration stops it on its angle, at most the
//! leg's speed, and ends once the angle that remains falls below one limiter
//! step at the deadband, the finest turn a step makes, a turn past its angle
//! included. The drive follows Nav2 1.5.2's
//! DriveOnHeading (nav2_behaviors include/nav2_behaviors/plugins/
//! drive_on_heading.hpp, `DriveOnHeading::onCycleUpdate`, lines 140-164):
//! angular 0, and the leg's speed reached from the last command by at most
//! acceleration x dt and deceleration x dt a step, at most the speed from
//! which the deceleration stops the base on the distance that remains along
//! the heading. That speed is [`AxisProfile::stopping_speed`], Spin's and
//! DriveOnHeading's sqrt(2 x deceleration x remaining) taken on the
//! limiter's steps. Each leg runs to its end, and the next starts once the
//! last has come to 0. The drive holds its aim: once the target's
//! bearing leaves the yaw tolerance, or the node's
//! `rotate_to_heading_min_angle_rad` when the tolerance is wider, the base
//! turns toward the target again and drives on.
//! base_state's limiter executes an axis slower than its deadband as 0, so
//! while the base stands outside the tolerances, the axis of the leg it is on
//! commands at least just above its deadband, as Spin's min_rotational_vel
//! and DriveOnHeading's minimum_speed floor their commands.

use std::f64::consts::PI;

use control_core::chase::rate_limited;
use control_core::planar_pose::{PlanarPose, normalize_angle};
use control_core::twist::PlanarTwist;

/// How far above its deadband a floored command stands, as a share of the
/// deadband: past the rounding of the limiter's sums, which can land a
/// command of exactly the deadband just below it. Each step's sum rounds by
/// at most half a unit in the last place of the axis's speed, 2^-53 of it,
/// and the error of a ramp grows by that at most each step, so this margin
/// covers 1e-6 x 2^53 (about 9e9) steps' roundings times the deadband over
/// the speed: for a deadband a thousandth of the speed, nine million steps,
/// two and a half hours of ramping at the limiter's fastest rate, 1 kHz.
const ABOVE_DEADBAND: f64 = 1e-6;

/// The turn (rad) below which Nav2's Spin ends: it stops once less than this
/// remains (nav2_behaviors/plugins/spin.cpp:116, `remaining_yaw < 1e-6`).
const SPIN_DONE_RAD: f64 = 1e-6;

/// The limits of one axis of the base that move_base plans and steers
/// under: its speed toward the axis's positive direction (ahead, or
/// counterclockwise) and toward its negative one, magnitudes, its
/// acceleration and deceleration, each above 0, and its deadband, the speed
/// below which base_state's limiter executes the axis as 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisProfile {
    pub positive_speed: f64,
    pub negative_speed: f64,
    pub acceleration: f64,
    pub deceleration: f64,
    pub deadband: f64,
}

impl AxisProfile {
    /// The axis's speed toward its positive direction when `positive`, and
    /// toward its negative one otherwise.
    fn speed(&self, positive: bool) -> f64 {
        if positive {
            self.positive_speed
        } else {
            self.negative_speed
        }
    }

    /// This profile with each way's speed times `scale`.
    fn scaled(&self, scale: f64) -> Self {
        Self {
            positive_speed: self.positive_speed * scale,
            negative_speed: self.negative_speed * scale,
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
        let speed = self.speed(distance >= 0.0);
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

    /// The slowest speed (a magnitude) the axis commands while it closes an
    /// error: just above the deadband.
    fn floor(&self) -> f64 {
        self.deadband * (1.0 + ABOVE_DEADBAND)
    }

    /// The finest turn (rad) one step of `dt` (s) makes: a step at the
    /// deadband, and no less than Nav2's Spin leaves, [`SPIN_DONE_RAD`]. A
    /// turn with less than this left ends.
    fn finest_step(&self, dt: f64) -> f64 {
        (self.deadband * dt).max(SPIN_DONE_RAD)
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
    /// `remaining` (rad) of it left, over a step of `dt` (s).
    fn turn_rate(&self, turning: Turning, remaining: f64, dt: f64) -> f64 {
        let rate = self
            .stopping_speed(remaining, dt)
            .min(self.speed(turning == Turning::Counterclockwise));
        turning.sign() * rate.max(self.floor())
    }

    /// DriveOnHeading's velocity (m/s), signed, toward a target `ahead` (m)
    /// along the heading, behind the base when negative, after commanding
    /// `last` (m/s) one step of `dt` (s) ago.
    fn drive_velocity(&self, ahead: f64, last: f64, dt: f64) -> f64 {
        let forward = ahead >= 0.0;
        let direction = if forward { 1.0 } else { -1.0 };
        let (lowest, highest) = if forward {
            (last - self.deceleration * dt, last + self.acceleration * dt)
        } else {
            (last - self.acceleration * dt, last + self.deceleration * dt)
        };
        let ramped = (direction * self.speed(forward)).clamp(lowest, highest);
        let to_stop = self.stopping_speed(ahead.abs(), dt);
        let capped = if to_stop < ramped.abs() {
            direction * to_stop
        } else {
            ramped
        };
        if capped.abs() < self.floor() {
            direction * self.floor()
        } else {
            capped
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
    /// The last command: the limiter's velocity before the first.
    last: PlanarTwist,
}

impl MoveController {
    pub fn new(
        target: PlanarPose,
        plan: MovePlan,
        tolerances: Tolerances,
        rotate_to_heading_min_angle: f64,
        command: PlanarTwist,
    ) -> Self {
        Self {
            target,
            plan,
            tolerances,
            rotate_to_heading_min_angle,
            leg: None,
            last: command,
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

    /// The leg the base is on with the target at `offset`. A drive runs to
    /// its end, the target's foot on the heading reached, while its bearing
    /// lies within the yaw tolerance and the rotate-to-heading angle. Within the
    /// position tolerance: the turn to the yaw, which starts while the
    /// heading lies outside the yaw tolerance and runs to its end. Outside
    /// it: the turn toward the target, which runs to its end, and the drive,
    /// which starts with the target ahead and goes on while its bearing lies
    /// within the same bound, its rear toward a target it passed on a base
    /// that drives backward.
    fn next_leg(&self, offset: Offset, dt: f64) -> Leg {
        let tolerances = self.tolerances;
        let aim = tolerances.yaw.min(self.rotate_to_heading_min_angle);
        let turn_done = self.plan.angular.finest_step(dt);
        if self.leg == Some(Leg::Drive) && offset.ahead > 0.0 && offset.bearing.abs() <= aim {
            return Leg::Drive;
        }
        if offset.distance <= tolerances.position {
            return match self.leg {
                Some(Leg::TurnToYaw(turning)) if turning.remaining(offset.yaw) >= turn_done => {
                    Leg::TurnToYaw(turning)
                }
                _ if offset.yaw.abs() <= tolerances.yaw => Leg::Arrived,
                _ => Leg::TurnToYaw(Turning::through(offset.yaw)),
            };
        }
        let steering = if offset.ahead >= 0.0 {
            offset.bearing
        } else {
            normalize_angle(offset.bearing + PI)
        };
        match self.leg {
            Some(Leg::TurnToward(turning)) if turning.remaining(offset.bearing) >= turn_done => {
                Leg::TurnToward(turning)
            }
            Some(Leg::Drive)
                if steering.abs() <= aim
                    && (offset.ahead >= 0.0 || self.plan.linear.negative_speed > 0.0) =>
            {
                Leg::Drive
            }
            _ if offset.ahead > 0.0 && offset.bearing.abs() <= aim => Leg::Drive,
            _ => Leg::TurnToward(Turning::through(offset.bearing)),
        }
    }

    /// The velocity to command at `pose`, one step of `dt` (s) after the
    /// last command: zero once `pose` stands within both tolerances. A leg on
    /// one axis starts once the command of the other has come to 0, the
    /// other brought down by its deceleration over each step meanwhile, so
    /// each leg runs on one axis through base_state's limiter, which scales
    /// the axes together.
    pub fn command(&mut self, pose: PlanarPose, dt: f64) -> PlanarTwist {
        let offset = Offset::of(self.target, pose);
        let leg = self.next_leg(offset, dt);
        let MovePlan {
            linear, angular, ..
        } = self.plan;
        let (linear_x, angular_z) = match leg {
            Leg::TurnToward(turning) => (
                0.0,
                angular.turn_rate(turning, turning.remaining(offset.bearing), dt),
            ),
            Leg::Drive => (
                linear.drive_velocity(offset.ahead, self.last.linear_x(), dt),
                0.0,
            ),
            Leg::TurnToYaw(turning) => (
                0.0,
                angular.turn_rate(turning, turning.remaining(offset.yaw), dt),
            ),
            Leg::Arrived => (0.0, 0.0),
        };
        let wound_down =
            |last: f64, axis: AxisProfile| rate_limited(last, 0.0, axis.deceleration, dt);
        let (last_linear, last_angular) = (self.last.linear_x(), self.last.angular_z());
        let (linear_x, angular_z) = if angular_z != 0.0 && last_linear != 0.0 {
            (wound_down(last_linear, linear), 0.0)
        } else if linear_x != 0.0 && last_angular != 0.0 {
            (0.0, wound_down(last_angular, angular))
        } else {
            (linear_x, angular_z)
        };
        self.leg = Some(leg);
        self.last = PlanarTwist::new(linear_x, 0.0, angular_z).expect("a command of finite poses");
        self.last
    }
}

#[cfg(test)]
mod tests;
