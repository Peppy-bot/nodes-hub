//! The reports that wait for a publish: base_state publishes, once the
//! clock reads a measurement's stamp, the newest measurement whose stamp the
//! clock has reached, and every older one the clock reached with it goes
//! unreported. The queue holds the reports of one odometry epoch back to
//! [`SILENT_AFTER_NS`](crate::control::SILENT_AFTER_NS) before the newest.

use std::time::SystemTime;

use control_core::planar_pose::PlanarPose;

use crate::control::fresh_at;

/// One measurement of the base, as base_states and drive_states carry it:
/// when the wheels measured it (ns), its epoch's odometry_start_timestamp,
/// where the odometry places the robot frame, and the velocity it measures
/// (m/s, rad/s).
#[derive(Clone, Copy)]
pub struct StateReport {
    pub timestamp_ns: i64,
    pub odometry_start: SystemTime,
    pub pose: PlanarPose,
    pub linear: f64,
    pub angular: f64,
}

/// The reports of an epoch's measurements that no publish has taken.
#[derive(Default)]
pub struct PendingReports {
    reports: Vec<StateReport>,
}

impl PendingReports {
    /// Hold `report` for a publish to take. The reports that are not
    /// [`fresh_at`] its stamp leave.
    pub fn push(&mut self, report: StateReport) {
        self.reports
            .retain(|older| fresh_at(older.timestamp_ns, report.timestamp_ns));
        self.reports.push(report);
    }

    /// The newest report whose stamp the clock, at `now_ns`, has reached; it
    /// and every other report the clock has reached leave.
    pub fn take_newest_reached(&mut self, now_ns: i64) -> Option<StateReport> {
        let newest = self
            .reports
            .iter()
            .filter(|report| report.timestamp_ns <= now_ns)
            .max_by_key(|report| report.timestamp_ns)
            .copied();
        self.reports.retain(|report| report.timestamp_ns > now_ns);
        newest
    }

    /// Forget every report: the odometry started over.
    pub fn clear(&mut self) {
        self.reports.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::SILENT_AFTER_NS;

    /// A report of the base standing at the origin, stamped `timestamp_ns`.
    fn report(timestamp_ns: i64) -> StateReport {
        StateReport {
            timestamp_ns,
            odometry_start: SystemTime::UNIX_EPOCH,
            pose: PlanarPose::new(0.0, 0.0, 0.0).expect("a finite pose"),
            linear: 0.0,
            angular: 0.0,
        }
    }

    /// The stamps of the reports `pending` takes as the clock reads each of
    /// `times_ns` in turn.
    fn taken(pending: &mut PendingReports, times_ns: &[i64]) -> Vec<Option<i64>> {
        times_ns
            .iter()
            .map(|now_ns| {
                pending
                    .take_newest_reached(*now_ns)
                    .map(|report| report.timestamp_ns)
            })
            .collect()
    }

    #[test]
    fn a_report_waits_until_the_clock_reads_its_stamp() {
        let mut pending = PendingReports::default();
        pending.push(report(10));
        assert_eq!(taken(&mut pending, &[9, 10, 10]), [None, Some(10), None]);
    }

    #[test]
    fn the_clock_takes_the_newest_report_it_reached_and_the_older_ones_leave() {
        let mut pending = PendingReports::default();
        for stamp_ns in [10, 20, 30] {
            pending.push(report(stamp_ns));
        }
        assert_eq!(
            taken(&mut pending, &[25, 25, 30]),
            [Some(20), None, Some(30)]
        );
    }

    #[test]
    fn a_report_stamped_more_than_the_silence_before_the_newest_leaves() {
        let mut pending = PendingReports::default();
        pending.push(report(0));
        pending.push(report(SILENT_AFTER_NS));
        assert_eq!(
            taken(&mut pending, &[0]),
            [Some(0)],
            "on the bound it stays"
        );
        let mut pending = PendingReports::default();
        pending.push(report(0));
        pending.push(report(SILENT_AFTER_NS + 1));
        assert_eq!(
            taken(&mut pending, &[0]),
            [None],
            "past the bound it leaves"
        );
    }

    #[test]
    fn a_cleared_queue_takes_nothing() {
        let mut pending = PendingReports::default();
        pending.push(report(10));
        pending.clear();
        assert_eq!(taken(&mut pending, &[10]), [None]);
    }
}
