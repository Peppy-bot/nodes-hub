//! The panel's joint ranges, resolved once from the robot's description: the
//! single source for the sliders' bounds (through the snapshot) and for
//! clamping incoming commands.

use robot_description::{ArmId, PerArm};
use srs_model::{ARM_DOF, JointVec};

use crate::robot::ARMS;

/// A second `init_limits` call, which would be a second description's ranges
/// arriving after the first were already handed out.
#[derive(Debug, thiserror::Error)]
#[error("init_limits must run exactly once")]
pub struct LimitsAlreadySet;

/// Each arm's joint ranges from the description plus the unitless gripper
/// opening range.
#[derive(Clone, Copy)]
struct JointLimits {
    gripper: [f64; 2],
    arms: PerArm<[[f64; 2]; ARM_DOF], ARMS>,
}

impl JointLimits {
    /// The panel's ranges of arms whose joints take `arms`.
    fn resolve(arms: PerArm<[[f64; 2]; ARM_DOF], ARMS>) -> Self {
        Self {
            gripper: [0.0, 1.0],
            arms,
        }
    }
}

static LIMITS: std::sync::OnceLock<JointLimits> = std::sync::OnceLock::new();

/// Resolve the panel's clamp/display ranges from each arm's joint limits, `arms`, the
/// description's: the ranges the backbone clamps into; the gripper is the unitless
/// opening fraction [0, 1] on every model. Must run before the UI serves, and exactly
/// once: a second call would be a second description's ranges arriving after the first
/// were already handed out.
pub fn init_limits(
    arms: PerArm<[[f64; 2]; ARM_DOF], ARMS>,
) -> std::result::Result<(), LimitsAlreadySet> {
    LIMITS
        .set(JointLimits::resolve(arms))
        .map_err(|_| LimitsAlreadySet)
}

fn joint_limits() -> &'static JointLimits {
    LIMITS
        .get()
        .expect("init_limits must run before the UI serves")
}

/// The per-joint `[min, max]` ranges (rad) of `arm`; commands clamp into them
/// and the sliders bound against them.
pub(crate) fn arm_limits(arm: ArmId) -> &'static [[f64; 2]; ARM_DOF] {
    &joint_limits().arms[arm]
}

/// The per-joint `[min, max]` ranges (rad) of every arm, the ranges
/// [`arm_limits`] gives per arm; the gesture bake keeps its tracks in them.
pub(crate) fn arm_ranges() -> &'static PerArm<[[f64; 2]; ARM_DOF], ARMS> {
    &joint_limits().arms
}

/// The gripper opening range `[closed, open]` as a fraction of full jaw travel;
/// the owner clamps gripper commands into it, the same single source the
/// sliders bound against.
pub(crate) fn gripper_limits() -> [f64; 2] {
    joint_limits().gripper
}

// Clamp each joint setpoint into its configured [min, max]. The single clamp
// path for every operator-driven arm command; the arm node clamps again on its own.
pub(crate) fn clamp_to_limits(joints: &mut JointVec, arm: ArmId) {
    for (j, &[lo, hi]) in joints.iter_mut().zip(arm_limits(arm).iter()) {
        *j = j.clamp(lo, hi);
    }
}

/// Tests have no main() to run init_limits, so resolve the v2 limits on
/// first use; concurrent tests settle benignly through get_or_init.
#[cfg(test)]
pub(crate) fn init_limits_for_tests() {
    LIMITS.get_or_init(|| {
        JointLimits::resolve(crate::test_support::driven("openarm_v2").joint_limits())
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_pins_each_joint_into_its_range() {
        init_limits_for_tests();
        for arm in PerArm::<(), ARMS>::ids() {
            let limits = arm_limits(arm);

            let mut high = [f64::INFINITY; ARM_DOF];
            clamp_to_limits(&mut high, arm);
            for (v, &[_, hi]) in high.iter().zip(limits.iter()) {
                assert_eq!(*v, hi);
            }

            let mut low = [f64::NEG_INFINITY; ARM_DOF];
            clamp_to_limits(&mut low, arm);
            for (v, &[lo, _]) in low.iter().zip(limits.iter()) {
                assert_eq!(*v, lo);
            }
        }
    }

    #[test]
    fn clamp_leaves_in_range_values_untouched() {
        init_limits_for_tests();
        for arm in PerArm::<(), ARMS>::ids() {
            let limits = arm_limits(arm);
            let mut mid = [0.0; ARM_DOF];
            for (m, &[lo, hi]) in mid.iter_mut().zip(limits.iter()) {
                *m = (lo + hi) / 2.0;
            }
            let before = mid;
            clamp_to_limits(&mut mid, arm);
            assert_eq!(mid, before);
        }
    }

    #[test]
    fn config_joint_limits_are_well_formed() {
        init_limits_for_tests();
        // Each range must be non-empty so clamp and the slider bounds are valid.
        for arm in PerArm::<(), ARMS>::ids() {
            for &[lo, hi] in arm_limits(arm).iter() {
                assert!(lo < hi, "joint range [{lo}, {hi}] must be non-empty");
            }
        }
        let [lo, hi] = gripper_limits();
        assert!(lo < hi, "gripper range [{lo}, {hi}] must be non-empty");
    }
}
