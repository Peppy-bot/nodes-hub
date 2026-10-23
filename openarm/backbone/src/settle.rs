//! How an arm move settles: the slack and the time its goal names, the
//! verdict the move gets from the planned end of its motion on, and how far a
//! joint target or a grasp point still stands from its target. The joint and
//! Cartesian moves are judged by this one rule.

use std::time::{Duration, Instant};

use srs_model::nalgebra::Isometry3;

use crate::types::{ARM_DOF, JointVec, PoseTolerance};

/// The slack a move's measured limb must stand within, judged from the
/// planned end of its motion on, and how long after that end it may stand
/// outside the slack before the move fails.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settle<T> {
    pub tolerance: T,
    pub timeout: Duration,
}

/// How a move at or past its planned end ends.
#[derive(Debug, PartialEq)]
pub enum Verdict {
    /// The measured limb stands within the tolerance of the target.
    Settled,
    /// The timeout passed with the limb still short of the target; the
    /// message says how far and how long after the planned end.
    Failed(String),
}

/// Judge a move at `now` whose motion ends at `planned_end`: nothing before
/// the planned end; from it on, settled once nothing stands short of the
/// target, failed once `timeout` has passed since the planned end with
/// `shortfall` still standing, and in flight in between.
pub fn verdict(
    shortfall: Option<String>,
    planned_end: Instant,
    now: Instant,
    timeout: Duration,
) -> Option<Verdict> {
    if now < planned_end {
        return None;
    }
    let since_end = now.duration_since(planned_end);
    match shortfall {
        None => Some(Verdict::Settled),
        Some(shortfall) if since_end >= timeout => Some(Verdict::Failed(format!(
            "still {shortfall}, {:.1} s after the planned end",
            since_end.as_secs_f64()
        ))),
        Some(_) => None,
    }
}

/// Where `measured` stands outside `tolerances` of `target`, at the joint
/// farthest outside its own tolerance; `None` with every joint within it.
pub fn joint_shortfall(
    measured: &JointVec,
    target: &JointVec,
    tolerances: &JointVec,
) -> Option<String> {
    // Scanned last to first, so the first of equally far joints is named,
    // as the so101 backbone names it.
    let (joint, error_rad) = (0..ARM_DOF)
        .rev()
        .map(|j| (j, (measured[j] - target[j]).abs()))
        .filter(|&(j, error_rad)| error_rad > tolerances[j])
        .max_by(|a, b| (a.1 - tolerances[a.0]).total_cmp(&(b.1 - tolerances[b.0])))?;
    Some(format!(
        "{error_rad:.3} rad from the target at joint {}",
        joint + 1
    ))
}

/// Where `measured` stands outside `tolerance` of `target`: the distance
/// between the grasp points and the angle between the orientations; `None`
/// within both.
pub fn pose_shortfall(
    measured: &Isometry3<f64>,
    target: &Isometry3<f64>,
    tolerance: &PoseTolerance,
) -> Option<String> {
    let error_m = (measured.translation.vector - target.translation.vector).norm();
    let error_rad = measured.rotation.angle_to(&target.rotation);
    (error_m > tolerance.position_m || error_rad > tolerance.orientation_rad).then(|| {
        format!(
            "{:.1} mm / {error_rad:.2} rad from the target",
            error_m * 1000.0
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use hegel::TestCase;
    use hegel::generators as gs;
    use srs_model::nalgebra::{Translation3, UnitQuaternion, Vector3};

    fn seconds(s: f64) -> Duration {
        Duration::from_secs_f64(s)
    }

    // Nothing is judged before the planned end; from it on, the verdict
    // settles on nothing short of the target, fails once the timeout has
    // passed with a shortfall still standing, and waits in between.
    #[hegel::test]
    fn the_verdict_settles_fails_or_waits_by_shortfall_and_time(tc: TestCase) {
        let short: bool = tc.draw(gs::booleans());
        let [since_end_s, timeout_s]: [f64; 2] =
            tc.draw(gs::arrays(gs::floats().min_value(-1.0).max_value(10.0)));
        let planned_end = Instant::now();
        let now = if since_end_s < 0.0 {
            planned_end - seconds(-since_end_s)
        } else {
            planned_end + seconds(since_end_s)
        };
        let shortfall = short.then(|| "8 mm / 0.02 rad from the target".to_string());
        let judged = verdict(
            shortfall.clone(),
            planned_end,
            now,
            seconds(timeout_s.max(0.0)),
        );
        if now < planned_end {
            assert_eq!(judged, None, "judged before the planned end");
            return;
        }
        let timed_out = now.duration_since(planned_end) >= seconds(timeout_s.max(0.0));
        match (shortfall, timed_out, judged) {
            (None, _, Some(Verdict::Settled)) => {}
            (Some(shortfall), true, Some(Verdict::Failed(message))) => {
                assert!(
                    message.starts_with(&format!("still {shortfall}, ")),
                    "{message}"
                );
                assert!(message.ends_with(" s after the planned end"), "{message}");
            }
            (Some(_), false, None) => {}
            (shortfall, timed_out, judged) => {
                panic!("shortfall {shortfall:?}, timed out {timed_out}: {judged:?}")
            }
        }
    }

    #[test]
    fn a_failed_verdict_names_the_time_since_the_planned_end() {
        let planned_end = Instant::now();
        assert_eq!(
            verdict(
                Some("8 mm / 0.02 rad from the target".to_string()),
                planned_end,
                planned_end + seconds(2.04),
                seconds(2.0)
            ),
            Some(Verdict::Failed(
                "still 8 mm / 0.02 rad from the target, 2.0 s after the planned end".to_string()
            ))
        );
    }

    #[test]
    fn equally_far_joints_name_the_first() {
        let measured = [0.0; ARM_DOF];
        let target = [0.2; ARM_DOF];
        let tolerances = [0.1; ARM_DOF];
        assert_eq!(
            joint_shortfall(&measured, &target, &tolerances),
            Some("0.200 rad from the target at joint 1".to_string())
        );
    }

    // The joint shortfall is empty exactly when every joint stands within its
    // tolerance, and otherwise names a joint no other joint stands farther
    // past its own tolerance than.
    #[hegel::test]
    fn the_joint_shortfall_names_the_joint_farthest_past_its_tolerance(tc: TestCase) {
        let measured: JointVec = tc.draw(gs::arrays(gs::floats().min_value(-3.0).max_value(3.0)));
        let target: JointVec = tc.draw(gs::arrays(gs::floats().min_value(-3.0).max_value(3.0)));
        let tolerances: JointVec = tc.draw(gs::arrays(gs::floats().min_value(1e-3).max_value(1.0)));
        let excess = |j: usize| (measured[j] - target[j]).abs() - tolerances[j];
        let worst_excess = (0..ARM_DOF).map(excess).fold(f64::MIN, f64::max);
        match joint_shortfall(&measured, &target, &tolerances) {
            None => assert!(
                worst_excess <= 0.0,
                "a joint stands {worst_excess} rad past its tolerance"
            ),
            Some(shortfall) => {
                let named = (0..ARM_DOF)
                    .find(|j| shortfall.ends_with(&format!(" at joint {}", j + 1)))
                    .unwrap_or_else(|| panic!("no joint named in {shortfall:?}"));
                assert!(
                    excess(named) > 0.0,
                    "joint {} stands within its tolerance",
                    named + 1
                );
                assert_eq!(excess(named), worst_excess, "{shortfall}");
                assert!(shortfall.starts_with(&format!(
                    "{:.3} rad from the target",
                    (measured[named] - target[named]).abs()
                )));
            }
        }
    }

    // The pose shortfall is empty exactly when the grasp point and the
    // orientation both stand within tolerance, and otherwise says how far
    // each stands.
    #[hegel::test]
    fn the_pose_shortfall_is_empty_exactly_within_both_tolerances(tc: TestCase) {
        let offset: [f64; 3] = tc.draw(gs::arrays(gs::floats().min_value(-0.02).max_value(0.02)));
        let turn: [f64; 3] = tc.draw(gs::arrays(gs::floats().min_value(-0.1).max_value(0.1)));
        let [position_m, orientation_rad]: [f64; 2] =
            tc.draw(gs::arrays(gs::floats().min_value(1e-4).max_value(0.1)));
        let target = Isometry3::from_parts(
            Translation3::new(0.4, 0.1, 0.8),
            UnitQuaternion::from_euler_angles(0.3, -0.2, 0.1),
        );
        let measured = Isometry3::from_parts(
            Translation3::from(target.translation.vector + Vector3::from(offset)),
            target.rotation * UnitQuaternion::from_euler_angles(turn[0], turn[1], turn[2]),
        );
        let error_m = (measured.translation.vector - target.translation.vector).norm();
        let error_rad = measured.rotation.angle_to(&target.rotation);
        let tolerance = PoseTolerance {
            position_m,
            orientation_rad,
        };
        let shortfall = pose_shortfall(&measured, &target, &tolerance);
        assert_eq!(
            shortfall.is_none(),
            error_m <= position_m && error_rad <= orientation_rad,
            "{error_m} m / {error_rad} rad against {tolerance:?}: {shortfall:?}"
        );
        if let Some(shortfall) = shortfall {
            assert_eq!(
                shortfall,
                format!(
                    "{:.1} mm / {error_rad:.2} rad from the target",
                    error_m * 1000.0
                )
            );
        }
    }
}
