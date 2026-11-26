//! The base the rig drives: drive_link's follower and base_state's base in
//! one, as botshare's backbone is both. While it follows, it commands the
//! newest setpoint the navigator streamed through base_state's limiter,
//! control_core's, from the step after the setpoint arrived; a hold drops each
//! setpoint and brings the base to rest, and so does the end of the stream,
//! 0.5 s after the newest setpoint, the limiter's velocity timeout. It moves
//! the robot frame by the velocity it executes, in the world and in its
//! odometry, and reports its mode by base_state's rest rule unless a test fixes
//! the mode.

use std::collections::VecDeque;

use control_core::drive_link::{FollowState, WINDOW_NS};
use control_core::limiter::{AxisLimits, AxisLimitsParams, Limiter, LimiterParams};
use control_core::non_negative_finite::NonNegativeFinite;
use control_core::positive_finite::PositiveFinite;
use control_core::twist::PlanarTwist;
use frame_tree::arithmetic::normalize_angle;

use super::{DT, STEP_NS};

/// The limiter's rate: a step every 50 ms, the rig's step.
const SMOOTHING_FREQUENCY_HZ: f64 = 20.0;

/// How many of the newest states base_state's rest rule reads: every state
/// stamped at most 0.2 s before the newest, the newest included, one each
/// step.
const REST_WINDOW_STATES: usize = 5;

/// What describe_base says of the base: the magnitudes of its velocity
/// limits along x and of its turn, forward and backward alike, how fast each
/// grows and falls, their deadbands, and the velocities at or below which it
/// counts as at rest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    pub max_linear_m_s: f64,
    pub linear_acceleration_m_s2: f64,
    pub linear_deceleration_m_s2: f64,
    pub linear_deadband_m_s: f64,
    pub max_angular_rad_s: f64,
    pub angular_acceleration_rad_s2: f64,
    pub angular_deceleration_rad_s2: f64,
    pub angular_deadband_rad_s: f64,
    pub rest_velocity_m_s: f64,
    pub rest_rate_rad_s: f64,
}

/// How the base reports its mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeReport {
    /// standing while it rests by base_state's rest rule, moving otherwise.
    ByRest,
    /// The mode a test sets, whatever the base does.
    Fixed(&'static str),
}

/// The robot frame on the floor: where it stands and its heading.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Planar {
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
}

impl Planar {
    /// This pose moved by `velocity`, in its own frame, for one step.
    fn moved(self, velocity: PlanarTwist) -> Self {
        let (sin, cos) = self.yaw.sin_cos();
        let (forward, leftward) = (velocity.linear_x(), velocity.linear_y());
        Self {
            x: self.x + (forward * cos - leftward * sin) * DT,
            y: self.y + (forward * sin + leftward * cos) * DT,
            yaw: normalize_angle(self.yaw + velocity.angular_z() * DT),
        }
    }

    /// The distance (m) from this pose's origin to `point`.
    pub fn distance_to(&self, point: [f64; 2]) -> f64 {
        (self.x - point[0]).hypot(self.y - point[1])
    }

    /// The pose `share` of the way from this pose to `other`, its heading
    /// turning the shorter way, as the robot's tree places the robot frame
    /// between two of its transforms.
    pub fn toward(self, other: Planar, share: f64) -> Self {
        Self {
            x: self.x + share * (other.x - self.x),
            y: self.y + share * (other.y - self.y),
            yaw: normalize_angle(self.yaw + share * normalize_angle(other.yaw - self.yaw)),
        }
    }
}

/// The limits of one axis of `max` (m/s or rad/s) each way.
fn axis(max: f64, acceleration: f64, deceleration: f64, deadband: f64) -> AxisLimits {
    let limit = |value: f64| NonNegativeFinite::parse(value).expect("the rig's limit");
    AxisLimits::parse(AxisLimitsParams {
        min_velocity: -max,
        max_velocity: max,
        max_acceleration: limit(acceleration),
        max_deceleration: limit(deceleration),
        deadband: limit(deadband),
    })
    .expect("the rig's axis")
}

/// base_state's limiter for `limits`, which drive_link's follower commands a
/// setpoint through.
fn limiter(limits: &Limits) -> Limiter {
    Limiter::parse(LimiterParams {
        linear_x: axis(
            limits.max_linear_m_s,
            limits.linear_acceleration_m_s2,
            limits.linear_deceleration_m_s2,
            limits.linear_deadband_m_s,
        ),
        linear_y: AxisLimits::STILL,
        angular_z: axis(
            limits.max_angular_rad_s,
            limits.angular_acceleration_rad_s2,
            limits.angular_deceleration_rad_s2,
            limits.angular_deadband_rad_s,
        ),
        smoothing_frequency_hz: PositiveFinite::parse(SMOOTHING_FREQUENCY_HZ)
            .expect("a positive rate"),
        velocity_timeout_s: NonNegativeFinite::parse(WINDOW_NS as f64 / 1e9).expect("a window"),
    })
    .expect("the rig's limiter")
}

/// The base, as the rig plays it.
pub struct MockBase {
    limits: Limits,
    limiter: Limiter,
    /// Where the robot frame stands in the world, which the cameras render
    /// from.
    pub truth: Planar,
    /// Where the base's odometry places the robot frame in odom.
    pub odometry: Planar,
    /// The velocity the base executes, in the robot frame.
    pub velocity: PlanarTwist,
    /// Whether the base executes the velocity it commands; one that executes
    /// none stands still and reports a velocity of 0.
    pub executes: bool,
    pub follow_state: FollowState,
    pub mode: ModeReport,
    pub parts_within_footprint: bool,
    /// When the odometry started (ns).
    pub odometry_start_ns: i64,
    /// The velocities of the newest states, newest last, for the rest rule.
    recent: VecDeque<PlanarTwist>,
}

impl MockBase {
    /// The base of `limits` at rest at the world's origin, its odometry
    /// started at `odometry_start_ns` there, following its leader.
    pub fn new(limits: Limits, odometry_start_ns: i64) -> Self {
        Self {
            limits,
            limiter: limiter(&limits),
            truth: Planar::default(),
            odometry: Planar::default(),
            velocity: PlanarTwist::ZERO,
            executes: true,
            follow_state: FollowState::Following,
            mode: ModeReport::ByRest,
            parts_within_footprint: true,
            odometry_start_ns,
            recent: VecDeque::from([PlanarTwist::ZERO; REST_WINDOW_STATES]),
        }
    }

    /// What describe_base says of the base, which drive_states carries too.
    pub fn limits(&self) -> Limits {
        self.limits
    }

    pub fn follows(&self) -> bool {
        self.follow_state == FollowState::Following
    }

    /// Take the setpoint `velocity` that arrived at `now_ns`: commanded from
    /// the next step while the base follows, dropped during a hold.
    pub fn take(&mut self, velocity: PlanarTwist, now_ns: i64) {
        if self.follows() {
            self.limiter.take_setpoint(velocity, now_ns);
        }
    }

    /// Move the base to the step at `now_ns`: a hold brings it to rest, the
    /// limiter steps, and the robot frame moves by the velocity executed.
    pub fn advance(&mut self, now_ns: i64) {
        if !self.follows() {
            self.limiter.take_setpoint(PlanarTwist::ZERO, now_ns);
        }
        let commanded = self.limiter.tick(now_ns).unwrap_or(PlanarTwist::ZERO);
        self.velocity = if self.executes {
            commanded
        } else {
            PlanarTwist::ZERO
        };
        self.truth = self.truth.moved(self.velocity);
        self.odometry = self.odometry.moved(self.velocity);
        self.recent.pop_front();
        self.recent.push_back(self.velocity);
    }

    /// Whether the base rests by base_state's rule: in each of the newest
    /// states of the last 0.2 s, its velocity within the rest bounds.
    pub fn rests(&self) -> bool {
        let limits = self.limits;
        self.recent.iter().all(|velocity| {
            velocity.linear_x().abs() <= limits.rest_velocity_m_s
                && velocity.linear_y().abs() <= limits.rest_velocity_m_s
                && velocity.angular_z().abs() <= limits.rest_rate_rad_s
        })
    }

    /// The mode the base reports.
    pub fn mode(&self) -> &'static str {
        match self.mode {
            ModeReport::Fixed(mode) => mode,
            ModeReport::ByRest if self.rests() => "standing",
            ModeReport::ByRest => "moving",
        }
    }

    /// The odometry slips: it places the robot frame `dx` and `dy` (m) along
    /// odom from where it stands, the base itself still.
    pub fn slip_odometry(&mut self, dx: f64, dy: f64) {
        self.odometry.x += dx;
        self.odometry.y += dy;
    }

    /// The odometry starts over at `now_ns`: odom becomes where the robot
    /// frame stands.
    pub fn start_odometry_over(&mut self, now_ns: i64) {
        assert_ne!(now_ns, self.odometry_start_ns, "a new start");
        self.odometry = Planar::default();
        self.odometry_start_ns = now_ns;
    }
}

/// How many steps `seconds` spans, rounded to the nearest step.
pub fn steps_in(seconds: f64) -> usize {
    (seconds * 1e9 / STEP_NS as f64).round() as usize
}
