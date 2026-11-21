//! The parameters of the base the tests run, the one source of the harness
//! rig's (`tests/helpers/mod.rs`) and of the unit tests' (`src/test_rig.rs`):
//! limits with their accelerations apart from their decelerations.

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
/// The rest velocity, x's deadband.
pub const REST_LINEAR_M_S: f64 = LINEAR_DEADBAND_M_S;
/// The limiter's rate: a step every 50 ms.
pub const SMOOTHING_FREQUENCY_HZ: f64 = 20.0;
/// Nav2's regulated pure pursuit's default `rotate_to_heading_min_angle`,
/// as the launcher gives it.
pub const ROTATE_TO_HEADING_MIN_ANGLE_RAD: f64 = 0.785;
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
    use peppygen::parameters::{angular::Angular, x::X};
    peppygen::Parameters {
        x: X {
            min_velocity: -MAX_LINEAR_SPEED_M_S,
            max_velocity: MAX_LINEAR_SPEED_M_S,
            max_acceleration: LINEAR_ACCELERATION_M_S2,
            max_deceleration: LINEAR_DECELERATION_M_S2,
            deadband: LINEAR_DEADBAND_M_S,
            rest_velocity: REST_LINEAR_M_S,
        },
        angular: Angular {
            min_rate: -MAX_ANGULAR_SPEED_RAD_S,
            max_rate: MAX_ANGULAR_SPEED_RAD_S,
            max_acceleration: ANGULAR_ACCELERATION_RAD_S2,
            max_deceleration: ANGULAR_DECELERATION_RAD_S2,
            deadband: ANGULAR_DEADBAND_RAD_S,
        },
        smoothing_frequency_hz: SMOOTHING_FREQUENCY_HZ,
        odometry_velocity_window: 1,
        rotate_to_heading_min_angle_rad: ROTATE_TO_HEADING_MIN_ANGLE_RAD,
        stop_answer_timeout_s: STOP_ANSWER_TIMEOUT_S,
        cache_window_s: CACHE_WINDOW_S,
        lookup_timeout_s: LOOKUP_TIMEOUT_S,
    }
}
