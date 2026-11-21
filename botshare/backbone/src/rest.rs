//! base_state's rest: the base rests while its motion sensing is not silent
//! and each axis of the velocity of every base_states whose timestamp lies
//! within [`REST_WINDOW_NS`] of the newest lies within that axis's rest bound
//! of describe_base, rest_velocity along x and y and rest_rate for yaw, the
//! bound included. The window holds the base_states of one odometry epoch, in
//! the order of their timestamps; the control loop judges the silence.

use std::collections::VecDeque;

/// How far back from the newest base_states the rest looks (ns): base_state's
/// 0.2 s.
const REST_WINDOW_NS: i64 = 200_000_000;

/// The velocities along x, y and yaw (m/s, m/s, rad/s) of the base_states
/// within [`REST_WINDOW_NS`] of the newest, by timestamp (ns).
#[derive(Debug, Default)]
pub struct RestWindow {
    velocities: VecDeque<(i64, [f64; 3])>,
}

impl RestWindow {
    /// A base_states at `timestamp_ns`, no earlier than the newest, with
    /// `velocity` along x, y and yaw.
    pub fn note(&mut self, timestamp_ns: i64, velocity: [f64; 3]) {
        self.velocities.push_back((timestamp_ns, velocity));
        let oldest_kept_ns = timestamp_ns - REST_WINDOW_NS;
        while self
            .velocities
            .front()
            .is_some_and(|(stamp_ns, _)| *stamp_ns < oldest_kept_ns)
        {
            self.velocities.pop_front();
        }
    }

    /// Whether every velocity of the window lies within `bounds` on each
    /// axis, x, y and yaw; false for an empty window.
    pub fn rests(&self, bounds: [f64; 3]) -> bool {
        !self.velocities.is_empty()
            && self.velocities.iter().all(|(_, velocity)| {
                velocity
                    .iter()
                    .zip(bounds)
                    .all(|(value, bound)| value.abs() <= bound)
            })
    }

    /// Forget every base_states: the odometry started over.
    pub fn clear(&mut self) {
        self.velocities.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// x and y take 0.002 m/s and yaw 0.004 rad/s.
    const REST: [f64; 3] = [0.002, 0.002, 0.004];

    /// `ms` milliseconds, in nanoseconds.
    fn ms(ms: i64) -> i64 {
        ms * 1_000_000
    }

    #[test]
    fn an_empty_window_does_not_rest() {
        assert!(!RestWindow::default().rests(REST));
    }

    #[test]
    fn a_velocity_on_the_bound_rests_and_one_past_it_moves() {
        let mut window = RestWindow::default();
        window.note(ms(0), [-0.002, 0.0, 0.004]);
        assert!(window.rests(REST));
        window.note(ms(10), [0.0, 0.0, -0.0041]);
        assert!(!window.rests(REST));
    }

    #[test]
    fn a_motion_counts_until_its_timestamp_leaves_the_window() {
        let mut window = RestWindow::default();
        window.note(ms(0), [0.3, 0.0, 0.0]);
        window.note(ms(100), [0.0, 0.0, 0.0]);
        window.note(ms(200), [0.0, 0.0, 0.0]);
        assert!(!window.rests(REST), "0 ms lies within 0.2 s of 200 ms");
        window.note(ms(201), [0.0, 0.0, 0.0]);
        assert!(window.rests(REST));
    }

    #[test]
    fn a_cleared_window_does_not_rest() {
        let mut window = RestWindow::default();
        window.note(ms(0), [0.0; 3]);
        window.clear();
        assert!(!window.rests(REST));
    }
}
