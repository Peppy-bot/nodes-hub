//! The robot's measured positions as the planner keeps them: each arm's joints and each
//! gripper's opening at one instant, and the history of them that the map reads at a depth
//! frame's stamp, which may lie some tens of milliseconds before the frame arrives.

use std::time::{Duration, SystemTime};

use motion_planning_core::JointHistory;

use robot_description::PerArm;

/// Each of `K` arms' joint positions at one instant.
pub type ArmJoints<const N: usize, const K: usize> = PerArm<[f64; N], K>;

/// Each arm's joint positions and each gripper's opening at one instant.
#[derive(Debug, Clone, PartialEq)]
pub struct Measured<const N: usize, const K: usize> {
    pub arms: ArmJoints<N, K>,
    pub openings: PerArm<f64, K>,
}

/// The measurements of a robot of `K` arms over the history's span: each arm's joints, and
/// each gripper's opening, pushed together at each stamp.
pub struct History<const N: usize, const K: usize> {
    arms: PerArm<JointHistory<N>, K>,
    openings: PerArm<JointHistory<1>, K>,
}

impl<const N: usize, const K: usize> History<N, K> {
    /// A history that keeps the measurements of the last `span`.
    pub fn new(span: Duration) -> Self {
        Self {
            arms: PerArm::from_fn(|_| JointHistory::new(span)),
            openings: PerArm::from_fn(|_| JointHistory::new(span)),
        }
    }

    /// Adds the measurement at `at`, as [`JointHistory::push`] adds one.
    pub fn push(&mut self, at: SystemTime, measured: &Measured<N, K>) {
        for arm in PerArm::<(), K>::ids() {
            self.arms[arm].push(at, measured.arms[arm]);
            self.openings[arm].push(at, [measured.openings[arm]]);
        }
    }

    /// The newest measurement.
    pub fn latest(&self) -> Option<Measured<N, K>> {
        Some(Measured {
            arms: PerArm::from_fn(|arm| self.arms[arm].latest().copied()).transpose()?,
            openings: PerArm::from_fn(|arm| self.openings[arm].latest().map(|[opening]| *opening))
                .transpose()?,
        })
    }

    /// The instant the newest measurement was taken at.
    pub fn latest_at(&self) -> Option<SystemTime> {
        self.arms.iter().next().and_then(JointHistory::latest_at)
    }

    /// The measurement at `at`, interpolated between those around it; `None` when `at` lies
    /// outside the history.
    pub fn at(&self, at: SystemTime) -> Option<Measured<N, K>> {
        Some(Measured {
            arms: PerArm::from_fn(|arm| self.arms[arm].at(at)).transpose()?,
            openings: PerArm::from_fn(|arm| self.openings[arm].at(at).map(|[opening]| opening))
                .transpose()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured(base: f64, opening: f64) -> Measured<2, 2> {
        Measured {
            arms: PerArm::new([[base, base + 1.0], [base + 2.0, base + 3.0]]),
            openings: PerArm::new([opening, 1.0 - opening]),
        }
    }

    fn after(ms: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_millis(ms)
    }

    // Each value keeps its place through the history: the arms' joints and the grippers'
    // openings, the newest measurement as it was pushed.
    #[test]
    fn a_measurement_comes_back_from_the_history() {
        let mut history = History::new(Duration::from_secs(1));
        assert_eq!(history.latest(), None);
        assert_eq!(history.latest_at(), None);
        history.push(after(10), &measured(0.0, 0.25));
        history.push(after(20), &measured(10.0, 0.75));
        assert_eq!(history.latest(), Some(measured(10.0, 0.75)));
        assert_eq!(history.latest_at(), Some(after(20)));
    }

    // A stamp between two measurements reads each value interpolated between them.
    #[test]
    fn a_stamp_between_measurements_reads_between_them() {
        let mut history = History::new(Duration::from_secs(1));
        history.push(after(10), &measured(0.0, 0.25));
        history.push(after(20), &measured(10.0, 0.75));
        assert_eq!(history.at(after(15)), Some(measured(5.0, 0.5)));
        assert_eq!(history.at(after(5)), None);
    }
}
