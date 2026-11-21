//! What the node runs on: its parameters, parsed once at setup ([`BaseParameters`]),
//! the limits of the base's motion along x and of its yaw, as describe_base
//! answers them and base_state's limiter holds every velocity to, the rest
//! velocity, and the odometry's window; and, once the robot's initializer
//! names the robot's model, the model's base ([`BaseModel`]) with them
//! ([`BaseConfig`]): the wheel geometry and the odometry it starts from, the
//! rest bounds the geometry gives the rest velocity ([`RestBounds`]), and the
//! outline. A launch with a value the node cannot run on is refused at setup,
//! by parameter name.

use std::num::NonZeroUsize;
use std::time::{Duration, SystemTime};

use control_core::footprint::Footprint;
use control_core::limiter::{AxisLimits, AxisLimitsParams, Limiter, LimiterError, LimiterParams};
use control_core::non_negative_finite::{NonNegativeFinite, NotNonNegativeFinite};
use control_core::odometry::{Odometry, WheelGeometry};
use control_core::positive_finite::{NotPositiveFinite, PositiveFinite};
use control_core::time::{SecondsParameterError, duration_parameter, seconds, wait_parameter};
use peppygen::Parameters;

use crate::follower::LEADER_STREAMING_WINDOW_NS;
use crate::model::{BaseModel, BundledBase, ModelError};
use crate::move_control::AxisProfile;
use crate::transforms::JointTransforms;

/// A parameter the node cannot run on.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("parameter {name}")]
    NotPositive {
        name: &'static str,
        #[source]
        source: NotPositiveFinite,
    },
    #[error("parameter {axis}.{limit}")]
    NotNonNegative {
        axis: &'static str,
        limit: &'static str,
        #[source]
        source: NotNonNegativeFinite,
    },
    #[error("parameter {name}")]
    Axis {
        name: &'static str,
        #[source]
        source: LimiterError,
    },
    #[error("parameter x: move_base drives the base ahead, so give x a max_velocity above 0")]
    NoForward,
    #[error(
        "parameter angular: move_base turns the base the shorter way, so give angular a min_rate below 0 and a max_rate above 0"
    )]
    OneWayTurn,
    #[error("parameter smoothing_frequency_hz")]
    Smoothing(#[source] LimiterError),
    #[error("parameter odometry_velocity_window must be at least 1")]
    EmptyWindow,
    #[error(transparent)]
    Seconds(#[from] SecondsParameterError),
}

/// The limits of one axis of the base's motion, as describe_base answers
/// them (m/s and m/s^2 along x or y, rad/s and rad/s^2 for yaw): its range,
/// its acceleration, deceleration and deadband.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisSpec {
    pub min_velocity: f64,
    pub max_velocity: f64,
    pub max_acceleration: f64,
    pub max_deceleration: f64,
    pub deadband: f64,
}

impl AxisSpec {
    /// The axis a differential base does not move along, y: its range,
    /// acceleration, deceleration and deadband are 0.
    pub const STILL: AxisSpec = AxisSpec {
        min_velocity: 0.0,
        max_velocity: 0.0,
        max_acceleration: 0.0,
        max_deceleration: 0.0,
        deadband: 0.0,
    };

    /// The axis `name` as base_state's limiter takes it: control_core's
    /// limiter checks the range, the acceleration, the deceleration and the
    /// deadband.
    fn limits(&self, name: &'static str) -> Result<AxisLimits, ConfigError> {
        let non_negative = |limit, value| {
            NonNegativeFinite::parse(value).map_err(|source| ConfigError::NotNonNegative {
                axis: name,
                limit,
                source,
            })
        };
        AxisLimits::parse(AxisLimitsParams {
            min_velocity: self.min_velocity,
            max_velocity: self.max_velocity,
            max_acceleration: non_negative("max_acceleration", self.max_acceleration)?,
            max_deceleration: non_negative("max_deceleration", self.max_deceleration)?,
            deadband: non_negative("deadband", self.deadband)?,
        })
        .map_err(|source| ConfigError::Axis { name, source })
    }

    /// The fastest the axis moves each way, with its acceleration,
    /// deceleration and deadband: the profile move_base plans and steers a
    /// leg along it with.
    pub fn profile(&self) -> AxisProfile {
        AxisProfile {
            positive_speed: self.max_velocity,
            negative_speed: -self.min_velocity,
            acceleration: self.max_acceleration,
            deceleration: self.max_deceleration,
            deadband: self.deadband,
        }
    }
}

/// The axes a differential base moves along: ahead along x and about yaw.
/// It stands still along y, [`AxisSpec::STILL`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaseAxes {
    pub x: AxisSpec,
    pub yaw: AxisSpec,
}

impl BaseAxes {
    /// base_state's limiter of the base, stepping at `frequency_hz`: each
    /// axis within its limits, and to rest once the leader sends nothing for
    /// [`LEADER_STREAMING_WINDOW_NS`].
    pub fn limiter_settings(&self, frequency_hz: f64) -> Result<LimiterSettings, ConfigError> {
        let smoothing_frequency_hz =
            PositiveFinite::parse(frequency_hz).map_err(|source| ConfigError::NotPositive {
                name: "smoothing_frequency_hz",
                source,
            })?;
        let velocity_timeout_s = NonNegativeFinite::parse(seconds(LEADER_STREAMING_WINDOW_NS))
            .expect("drive_link's stream gap is a finite time of at least 0");
        LimiterSettings::parse(LimiterParams {
            linear_x: self.x.limits("x")?,
            linear_y: AxisLimits::STILL,
            angular_z: self.yaw.limits("angular")?,
            smoothing_frequency_hz,
            velocity_timeout_s,
        })
        .map_err(ConfigError::Smoothing)
    }
}

/// base_state's rest bounds of the base, as describe_base answers them: the
/// largest magnitude of the measured velocity along x (m/s) and of the
/// measured turn rate (rad/s) at which the base counts as at rest; along y,
/// which a differential base does not move along, 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestBounds {
    velocity: f64,
    rate: f64,
}

impl RestBounds {
    /// The rest bounds of a differential base on `wheels` whose wheels each
    /// measure at most `rest_velocity` (m/s) along the floor while the base
    /// stands. The odometry reads the velocity along x as the mean of the two
    /// wheel speeds and the turn rate as their difference over the wheel
    /// separation, so a reading that keeps each wheel within
    /// ±`rest_velocity` gives a velocity within `rest_velocity` and a turn
    /// rate within twice `rest_velocity` over the separation.
    fn of_wheels(rest_velocity: f64, wheels: &WheelGeometry) -> Self {
        Self {
            velocity: rest_velocity,
            rate: 2.0 * rest_velocity / wheels.wheel_separation_m.get(),
        }
    }

    /// The rest bound of each axis of base_states' velocity: x, y and yaw.
    pub fn bounds(&self) -> [f64; 3] {
        [self.velocity, 0.0, self.rate]
    }
}

/// base_state's limiter of the base as the node's parameters set it, which
/// control_core's [`Limiter::parse`] took at setup, and the period of its
/// steps on the node's clock (ns).
#[derive(Clone, Debug)]
pub struct LimiterSettings {
    params: LimiterParams,
    period_ns: i64,
}

impl LimiterSettings {
    fn parse(params: LimiterParams) -> Result<Self, LimiterError> {
        let period_ns = Limiter::parse(params.clone())?.period_ns();
        Ok(Self { params, period_ns })
    }

    /// A limiter of these settings at rest, before its first step.
    pub fn start(&self) -> Limiter {
        Limiter::parse(self.params.clone()).expect("settings the limiter took at setup parse again")
    }
}

/// The node's parameters, parsed.
#[derive(Debug)]
pub struct BaseParameters {
    pub axes: BaseAxes,
    /// x's rest_velocity (m/s): the largest speed each wheel measures along
    /// the floor while the base stands.
    rest_velocity: f64,
    pub limiter: LimiterSettings,
    /// One step of base_state's limiter on the node's clock (ns).
    pub step_ns: i64,
    odometry_window: NonZeroUsize,
    rotate_to_heading_min_angle_rad: f64,
    stop_answer_timeout: Duration,
    /// How long the robot's tree keeps each moving transform.
    pub cache_window: Duration,
    /// How long a placement waits for the robot's tree.
    pub lookup_timeout: Duration,
}

/// Everything the node runs on: its parameters and its model's base.
#[derive(Debug)]
pub struct BaseConfig {
    /// The bundled model the base is built from.
    pub model: &'static robot_description::Model,
    pub axes: BaseAxes,
    /// base_state's rest bounds, the turn rate's from the wheel geometry
    /// ([`RestBounds::of_wheels`]).
    pub rest: RestBounds,
    pub wheels: WheelGeometry,
    /// The odometry at the origin of the odometry frame, its rolling window
    /// parsed: the odometry every odometry frame begins from.
    pub odometry: Odometry,
    pub limiter: LimiterSettings,
    /// The outline's corners, counterclockwise from the front right one.
    pub footprint: Footprint,
    /// The wheel joints, whose transforms the node publishes, each started
    /// when the node started.
    pub joint_transforms: JointTransforms,
    /// One step of base_state's limiter on the node's clock (ns).
    pub step_ns: i64,
    /// The fastest a wheel turns (rad/s) within the base's limits, the
    /// fastest forward speed and turn rate together
    /// ([`WheelGeometry::fastest_wheel_rate`]).
    pub fastest_wheel_rate: f64,
    /// The largest bearing (rad) at which move_base's drive goes on toward
    /// its target, whatever the yaw tolerance.
    pub rotate_to_heading_min_angle_rad: f64,
    /// How long robot_stop's stop waits for the control loop.
    pub stop_answer_timeout: Duration,
}

impl BaseConfig {
    /// How far (rad) a moving wheel's measured angle may lie from its last
    /// setpoint for its next setpoint to start from that setpoint: the
    /// wheel's travel over one limiter step at the fastest wheel rate.
    pub fn anchor_bound_rad(&self) -> f64 {
        self.fastest_wheel_rate * seconds(self.step_ns)
    }

    /// The farthest (rad) a wheel turns between two measurements `dt_s`
    /// (s) apart: at the fastest wheel rate over that time, and over what it
    /// lags its setpoint at most, the anchor bound and one limiter step of
    /// travel.
    pub fn wheel_reach_rad(&self, dt_s: f64) -> f64 {
        self.fastest_wheel_rate * dt_s + 2.0 * self.anchor_bound_rad()
    }
}

fn positive(name: &'static str, value: f64) -> Result<f64, ConfigError> {
    PositiveFinite::parse(value)
        .map(PositiveFinite::get)
        .map_err(|source| ConfigError::NotPositive { name, source })
}

impl BaseParameters {
    pub fn parse(params: &Parameters) -> Result<Self, ConfigError> {
        let (x, angular) = (&params.x, &params.angular);
        let axes = BaseAxes {
            x: AxisSpec {
                min_velocity: x.min_velocity,
                max_velocity: x.max_velocity,
                max_acceleration: x.max_acceleration,
                max_deceleration: x.max_deceleration,
                deadband: x.deadband,
            },
            yaw: AxisSpec {
                min_velocity: angular.min_rate,
                max_velocity: angular.max_rate,
                max_acceleration: angular.max_acceleration,
                max_deceleration: angular.max_deceleration,
                deadband: angular.deadband,
            },
        };
        let rest_velocity = positive("x.rest_velocity", x.rest_velocity)?;
        // What move_base asks of the axes comes first, so an axis of range 0
        // to 0 is told what range to give before the limiter judges its other
        // limits.
        if axes.x.max_velocity <= 0.0 {
            return Err(ConfigError::NoForward);
        }
        if axes.yaw.min_velocity >= 0.0 || axes.yaw.max_velocity <= 0.0 {
            return Err(ConfigError::OneWayTurn);
        }
        let limiter = axes.limiter_settings(params.smoothing_frequency_hz)?;
        let step_ns = limiter.period_ns;
        let odometry_window = NonZeroUsize::new(params.odometry_velocity_window as usize)
            .ok_or(ConfigError::EmptyWindow)?;
        let rotate_to_heading_min_angle_rad = positive(
            "rotate_to_heading_min_angle_rad",
            params.rotate_to_heading_min_angle_rad,
        )?;
        let stop_answer_timeout =
            duration_parameter("stop_answer_timeout_s", params.stop_answer_timeout_s)?;
        let cache_window = duration_parameter("cache_window_s", params.cache_window_s)?;
        let lookup_timeout = wait_parameter("lookup_timeout_s", params.lookup_timeout_s)?;
        Ok(Self {
            axes,
            rest_velocity,
            limiter,
            step_ns,
            odometry_window,
            rotate_to_heading_min_angle_rad,
            stop_answer_timeout,
            cache_window,
            lookup_timeout,
        })
    }

    /// The node's configuration for a robot of `model`, the node started at
    /// `started` on the wall clock: these parameters with the model's base
    /// ([`BaseModel::of`]).
    pub fn configure(&self, model: &str, started: SystemTime) -> Result<BaseConfig, ModelError> {
        let BundledBase {
            model: bundled,
            base,
        } = BaseModel::of(model)?;
        let odometry = Odometry::new(base.wheels, self.odometry_window);
        let fastest = |axis: AxisSpec| axis.max_velocity.max(-axis.min_velocity);
        let fastest_wheel_rate = base
            .wheels
            .fastest_wheel_rate(fastest(self.axes.x), fastest(self.axes.yaw));
        Ok(BaseConfig {
            model: bundled,
            axes: self.axes,
            rest: RestBounds::of_wheels(self.rest_velocity, &base.wheels),
            wheels: base.wheels,
            odometry,
            limiter: self.limiter.clone(),
            footprint: base.footprint,
            joint_transforms: JointTransforms::new(base.description, base.wheel_joints, started),
            step_ns: self.step_ns,
            fastest_wheel_rate,
            rotate_to_heading_min_angle_rad: self.rotate_to_heading_min_angle_rad,
            stop_answer_timeout: self.stop_answer_timeout,
        })
    }
}

#[cfg(test)]
mod tests;
