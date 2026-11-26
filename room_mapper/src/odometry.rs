//! The base's odometry, from base_state's base_states: the measurements of the current
//! odometry start and of the one before it, and where the robot frame stood in odom at the
//! times of a span that two measurements of one start enclose, interpolated as transforms
//! interpolates (tf2's linear blend of the position and slerp of the orientation,
//! frame_tree's `Transform::interpolated`).

use std::collections::VecDeque;

use frame_tree::Transform;

/// How long (ns) the measurements of a start are kept back from its newest: tf2's
/// default cache time, 10 s (tf2 buffer_core.h `DEFAULT_CACHE_TIME`), over which AMCL's
/// buffer carries an initial pose from its stamp.
pub const KEPT_NS: i64 = 10_000_000_000;

/// When an odometry started, base_state's odometry_start_timestamp (ns, the wall clock of
/// the machine that runs the base's node).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OdometryStart(pub i64);

/// One measurement of base_states: when it holds (ns, the clock of transforms) and where
/// it places base_footprint in odom.
#[derive(Debug, Clone, Copy)]
pub struct Measurement {
    pub at_ns: i64,
    pub odom_from_base: Transform,
}

/// The measurements kept of one odometry start, in time order.
#[derive(Debug, Clone)]
pub struct KeptStart(VecDeque<Measurement>);

impl KeptStart {
    /// Where the robot frame stood in odom at `at_ns`; none unless two measurements enclose
    /// the time.
    pub fn at(&self, at_ns: i64) -> Option<Transform> {
        let (first, last) = (self.0.front()?, self.0.back()?);
        (first.at_ns <= at_ns && at_ns <= last.at_ns).then(|| pose_at(&self.0, at_ns))
    }
}

/// Where the robot frame stood over a span of time, against the odometry.
#[derive(Debug, Clone)]
pub enum Span {
    /// Two measurements of `start` enclose the span: the robot frame in odom at each time
    /// asked, in the order asked.
    Enclosed {
        start: OdometryStart,
        poses: Vec<Transform>,
    },
    /// The newest start may yet enclose the span: no measurement of it at or after the
    /// span's end has arrived.
    Waiting,
    /// No start encloses the span, and none will: a newer start began within it or after
    /// it, or its measurements are older than the measurements kept.
    Never,
}

/// The measurements of the current odometry start and of the one before it.
#[derive(Debug, Default)]
pub struct Odometry {
    /// Each start with its measurements in time order, the older start first: at most two.
    starts: VecDeque<(OdometryStart, VecDeque<Measurement>)>,
}

impl Odometry {
    /// Takes `measurement` of `start`: true when it begins a new start, which ends the
    /// start before the current one. A measurement of an older start than the current
    /// one, or one no newer than its start's newest, is dropped.
    pub fn take(&mut self, start: OdometryStart, measurement: Measurement) -> bool {
        if let Some((_, measurements)) = self
            .starts
            .back_mut()
            .filter(|(current, _)| *current == start)
        {
            let newer = measurements
                .back()
                .is_none_or(|newest| measurement.at_ns > newest.at_ns);
            if newer {
                measurements.push_back(measurement);
                prune(measurements);
            }
            return false;
        }
        if self.starts.iter().any(|(kept, _)| *kept == start) {
            return false;
        }
        self.starts
            .push_back((start, VecDeque::from([measurement])));
        if self.starts.len() > 2 {
            self.starts.pop_front();
        }
        true
    }

    /// The newest measurement and its start.
    pub fn newest(&self) -> Option<(OdometryStart, Measurement)> {
        let (start, measurements) = self.starts.back()?;
        Some((*start, *measurements.back()?))
    }

    /// The measurements kept of `start`, which a reader holds without the odometry's lock;
    /// none for a start the odometry does not keep.
    pub fn kept_of(&self, start: OdometryStart) -> Option<KeptStart> {
        let (_, measurements) = self.starts.iter().find(|(kept, _)| *kept == start)?;
        Some(KeptStart(measurements.clone()))
    }

    /// Where the robot frame stood in odom at each of `times` (ns), once two
    /// measurements of one start enclose the earliest and the latest of them.
    pub fn span(&self, times: &[i64]) -> Span {
        let (Some(&earliest), Some(&latest)) = (times.iter().min(), times.iter().max()) else {
            return Span::Never;
        };
        let Some((newest_start, newest)) = self.starts.back() else {
            return Span::Waiting;
        };
        let begins_by = |measurements: &VecDeque<Measurement>| {
            measurements
                .front()
                .is_some_and(|first| first.at_ns <= earliest)
        };
        let ends_by = |measurements: &VecDeque<Measurement>| {
            measurements.back().is_some_and(|last| last.at_ns >= latest)
        };
        if begins_by(newest) {
            return if ends_by(newest) {
                enclosed(*newest_start, newest, times)
            } else {
                Span::Waiting
            };
        }
        // The newest start began after the earliest time, and the start before it, if one
        // is kept, takes no more measurements.
        match self.starts.iter().rev().nth(1) {
            Some((start, measurements)) if begins_by(measurements) && ends_by(measurements) => {
                enclosed(*start, measurements, times)
            }
            _ => Span::Never,
        }
    }
}

/// The span `times` within `measurements` of `start`, which enclose it.
fn enclosed(start: OdometryStart, measurements: &VecDeque<Measurement>, times: &[i64]) -> Span {
    Span::Enclosed {
        start,
        poses: times
            .iter()
            .map(|&at_ns| pose_at(measurements, at_ns))
            .collect(),
    }
}

/// The robot frame in odom at `at_ns`, between the two measurements that enclose it.
fn pose_at(measurements: &VecDeque<Measurement>, at_ns: i64) -> Transform {
    let after = measurements.partition_point(|measurement| measurement.at_ns < at_ns);
    let newer = measurements[after];
    if newer.at_ns == at_ns || after == 0 {
        return newer.odom_from_base;
    }
    let older = measurements[after - 1];
    let ratio = (at_ns - older.at_ns) as f64 / (newer.at_ns - older.at_ns) as f64;
    older
        .odom_from_base
        .interpolated(&newer.odom_from_base, ratio)
        .expect("a blend of two transforms is a transform")
}

/// Drops the measurements older than [`KEPT_NS`] before the newest, keeping the newest
/// one before that bound, so a time at the bound stays enclosed.
fn prune(measurements: &mut VecDeque<Measurement>) {
    let Some(newest) = measurements.back() else {
        return;
    };
    let bound = newest.at_ns - KEPT_NS;
    let before_bound = measurements.partition_point(|measurement| measurement.at_ns < bound);
    measurements.drain(..before_bound.saturating_sub(1));
}

#[cfg(test)]
mod tests {
    use super::*;
    use frame_tree::UnitQuaternion;

    const FIRST: OdometryStart = OdometryStart(1);
    const SECOND: OdometryStart = OdometryStart(2);

    /// base_footprint at `x` along odom's x, turned by `yaw`, at `at_ns`.
    fn measured(at_ns: i64, x: f64, yaw: f64) -> Measurement {
        Measurement {
            at_ns,
            odom_from_base: Transform::new(
                [x, 0.0, 0.0],
                UnitQuaternion::from_yaw(yaw).expect("a finite yaw"),
            )
            .expect("a transform"),
        }
    }

    fn placed(span: Span) -> (OdometryStart, Vec<(f64, f64)>) {
        let Span::Enclosed { start, poses } = span else {
            panic!("enclosed, got {span:?}");
        };
        let places = poses
            .iter()
            .map(|pose| (pose.position()[0], pose.orientation().yaw()))
            .collect();
        (start, places)
    }

    #[test]
    fn a_span_two_measurements_enclose_is_placed_between_them() {
        let mut odometry = Odometry::default();
        assert!(odometry.take(FIRST, measured(0, 0.0, 0.0)));
        assert!(!odometry.take(FIRST, measured(100, 1.0, 0.5)));
        let (start, places) = placed(odometry.span(&[0, 25, 100]));
        assert_eq!(start, FIRST);
        assert!((places[0].0 - 0.0).abs() < 1e-12 && (places[2].0 - 1.0).abs() < 1e-12);
        assert!((places[1].0 - 0.25).abs() < 1e-12, "{places:?}");
        assert!((places[1].1 - 0.125).abs() < 1e-12, "{places:?}");
    }

    #[test]
    fn a_span_past_the_newest_measurement_waits_for_one_after_it() {
        let mut odometry = Odometry::default();
        assert!(
            matches!(odometry.span(&[10, 20]), Span::Waiting),
            "no measurement yet"
        );
        odometry.take(FIRST, measured(0, 0.0, 0.0));
        odometry.take(FIRST, measured(15, 1.0, 0.0));
        assert!(matches!(odometry.span(&[10, 20]), Span::Waiting));
        odometry.take(FIRST, measured(30, 2.0, 0.0));
        assert_eq!(placed(odometry.span(&[10, 20])).0, FIRST);
    }

    #[test]
    fn the_start_before_a_new_one_places_a_span_it_encloses_and_none_across_the_restart() {
        let mut odometry = Odometry::default();
        odometry.take(FIRST, measured(0, 0.0, 0.0));
        odometry.take(FIRST, measured(100, 1.0, 0.0));
        assert!(odometry.take(SECOND, measured(150, 0.0, 0.0)));
        odometry.take(SECOND, measured(250, 1.0, 0.0));
        assert_eq!(placed(odometry.span(&[20, 80])).0, FIRST);
        assert_eq!(placed(odometry.span(&[160, 240])).0, SECOND);
        assert!(
            matches!(odometry.span(&[90, 160]), Span::Never),
            "across the restart"
        );
        assert!(
            matches!(odometry.span(&[120, 140]), Span::Never),
            "between the starts"
        );
    }

    #[test]
    fn a_measurement_out_of_order_or_of_an_ended_start_is_dropped() {
        let mut odometry = Odometry::default();
        odometry.take(FIRST, measured(100, 1.0, 0.0));
        assert!(!odometry.take(FIRST, measured(50, 9.0, 0.0)));
        odometry.take(SECOND, measured(200, 0.0, 0.0));
        assert!(!odometry.take(FIRST, measured(300, 9.0, 0.0)));
        assert_eq!(
            odometry.newest().map(|(start, m)| (start, m.at_ns)),
            Some((SECOND, 200))
        );
        assert!(odometry.take(OdometryStart(3), measured(400, 0.0, 0.0)));
        assert!(
            matches!(odometry.span(&[100, 100]), Span::Never),
            "a third start ends the first's measurements"
        );
    }

    #[test]
    fn measurements_older_than_the_kept_time_place_no_span() {
        let mut odometry = Odometry::default();
        let step = KEPT_NS / 4;
        for index in 0..=8 {
            odometry.take(FIRST, measured(index * step, index as f64, 0.0));
        }
        let newest = 8 * step;
        assert_eq!(placed(odometry.span(&[newest - KEPT_NS])).0, FIRST);
        // The newest measurement before the bound stays, and the ones before it go.
        assert_eq!(placed(odometry.span(&[newest - KEPT_NS - step])).0, FIRST);
        assert!(matches!(
            odometry.span(&[newest - KEPT_NS - 2 * step]),
            Span::Never
        ));
    }
}
