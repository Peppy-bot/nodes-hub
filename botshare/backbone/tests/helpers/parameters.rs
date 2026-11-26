//! The parameters of the base the tests run, the one source of the harness
//! rig's (`tests/helpers/mod.rs`) and of the unit tests' (`src/test_rig.rs`):
//! limits with their accelerations apart from their decelerations, and a rest
//! velocity apart from every limit along x.

pub const MAX_LINEAR_SPEED_M_S: f64 = 0.5;
pub const MAX_ANGULAR_SPEED_RAD_S: f64 = 1.5;
/// The base's acceleration and deceleration limits, each pair apart, so a
/// ramp up tells itself from a ramp down: per step, 0.05 m/s up and 0.1 m/s
/// down, 0.15 rad/s up and 0.2 rad/s down.
pub const LINEAR_ACCELERATION_M_S2: f64 = 1.0;
pub const LINEAR_DECELERATION_M_S2: f64 = 2.0;
pub const ANGULAR_ACCELERATION_RAD_S2: f64 = 3.0;
pub const ANGULAR_DECELERATION_RAD_S2: f64 = 4.0;
pub const LINEAR_DEADBAND_M_S: f64 = 0.005;
pub const ANGULAR_DEADBAND_RAD_S: f64 = 0.01;
/// The rest velocity: botshare_sim's, botshare's measured odometry noise
/// floor (trans_stopped_velocity), below the slowest speed the tests drive
/// at, 0.1 m/s.
pub const REST_LINEAR_M_S: f64 = 0.02;
/// The limiter's rate: a step every 50 ms.
pub const SMOOTHING_FREQUENCY_HZ: f64 = 20.0;
/// Nav2's regulated pure pursuit's default `rotate_to_heading_min_angle`,
/// as the launcher gives it.
pub const ROTATE_TO_HEADING_MIN_ANGLE_RAD: f64 = 0.785;
/// The slowest turn rate and drive speeds a move commands, each above its
/// deadband and below one step of the acceleration: the shortest turn from
/// rest is 1.5 mrad and the shortest drive 1 mm, the finest tolerances a
/// move takes, no finer than the tests ask.
pub const MIN_TURN_RATE_RAD_S: f64 = 0.03;
pub const MIN_FORWARD_SPEED_M_S: f64 = 0.02;
pub const MIN_BACKWARD_SPEED_M_S: f64 = 0.02;
/// How long a stop waits for the control loop (s), as the launcher gives
/// it.
pub const STOP_ANSWER_TIMEOUT_S: f64 = 1.5;
/// How long the tests' tree keeps each moving transform (s): a stamp older
/// than this before the newest of its frame has no value.
pub const CACHE_WINDOW_S: f64 = 10.0;
/// How long a placement waits for the tests' tree (s): long enough for a
/// transform the test publishes before its goal to reach the tree.
pub const LOOKUP_TIMEOUT_S: f64 = 1.0;

/// The parameters of the base the tests run.
pub fn params() -> peppygen::Parameters {
    use peppygen::parameters::{angular_limits::AngularLimits, linear_limits::LinearLimits};
    peppygen::Parameters {
        linear_limits: LinearLimits {
            min_velocity: -MAX_LINEAR_SPEED_M_S,
            max_velocity: MAX_LINEAR_SPEED_M_S,
            max_acceleration: LINEAR_ACCELERATION_M_S2,
            max_deceleration: LINEAR_DECELERATION_M_S2,
            deadband: LINEAR_DEADBAND_M_S,
            rest_velocity: REST_LINEAR_M_S,
        },
        angular_limits: AngularLimits {
            min_rate: -MAX_ANGULAR_SPEED_RAD_S,
            max_rate: MAX_ANGULAR_SPEED_RAD_S,
            max_acceleration: ANGULAR_ACCELERATION_RAD_S2,
            max_deceleration: ANGULAR_DECELERATION_RAD_S2,
            deadband: ANGULAR_DEADBAND_RAD_S,
        },
        smoothing_frequency_hz: SMOOTHING_FREQUENCY_HZ,
        velocity_rolling_window_size: 1,
        rotate_to_heading_min_angle_rad: ROTATE_TO_HEADING_MIN_ANGLE_RAD,
        min_turn_rate_rad_s: MIN_TURN_RATE_RAD_S,
        min_forward_speed_m_s: MIN_FORWARD_SPEED_M_S,
        min_backward_speed_m_s: MIN_BACKWARD_SPEED_M_S,
        stop_answer_timeout_s: STOP_ANSWER_TIMEOUT_S,
        cache_window_s: CACHE_WINDOW_S,
        lookup_timeout_s: LOOKUP_TIMEOUT_S,
    }
}
