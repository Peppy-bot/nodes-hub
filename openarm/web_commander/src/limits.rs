//! The panel's joint ranges, resolved once from the generation's description:
//! the single source for the sliders' bounds (through the snapshot) and for
//! clamping incoming commands.

use openarm_description::{ARM_DOF, ArmPair, HardwareVersion, Side};
use srs_model::JointVec;

/// A second `init_limits` call, which would be a second generation's ranges
/// arriving after the first were already handed out.
#[derive(Debug, thiserror::Error)]
#[error("init_limits must run exactly once")]
pub struct LimitsAlreadySet;

/// Joint ranges from the generation's bundled URDF plus the unitless gripper
/// opening range.
#[derive(Clone, Copy)]
struct JointLimits {
    gripper: [f64; 2],
    arms: ArmPair<[[f64; 2]; ARM_DOF]>,
}

impl JointLimits {
    fn resolve(version: HardwareVersion) -> Self {
        Self {
            gripper: [0.0, 1.0],
            arms: ArmPair::from_fn(|side| version.joint_limits(side)),
        }
    }
}

static LIMITS: std::sync::OnceLock<JointLimits> = std::sync::OnceLock::new();

/// Resolve the panel's clamp/display ranges from the generation's description:
/// arm joints via its `joint_limits` (URDF limits with the elbow held off its
/// singularity floor, matching the backbone's clamp); the gripper is the
/// unitless opening fraction [0, 1] on every generation. Must run before the
/// UI serves, and exactly once: a second call would be a second generation's
/// ranges arriving after the first were already handed out.
pub fn init_limits(version: HardwareVersion) -> std::result::Result<(), LimitsAlreadySet> {
    LIMITS
        .set(JointLimits::resolve(version))
        .map_err(|_| LimitsAlreadySet)
}

fn joint_limits() -> &'static JointLimits {
    LIMITS
        .get()
        .expect("init_limits must run before the UI serves")
}

/// The per-joint `[min, max]` ranges (rad) of `side`'s arm; commands clamp into
/// them and the sliders bound against them.
pub(crate) fn arm_limits(side: Side) -> &'static [[f64; 2]; ARM_DOF] {
    &joint_limits().arms[side]
}

/// The per-joint `[min, max]` ranges (rad) of both arms, the ranges
/// [`arm_limits`] gives per side; the gesture bake keeps its tracks in them.
pub(crate) fn arm_ranges() -> &'static ArmPair<[[f64; 2]; ARM_DOF]> {
    &joint_limits().arms
}

/// The gripper opening range `[closed, open]` as a fraction of full jaw travel;
/// the owner clamps gripper commands into it, the same single source the
/// sliders bound against.
pub(crate) fn gripper_limits() -> [f64; 2] {
    joint_limits().gripper
}

// Clamp each joint setpoint into its configured [min, max]. The single clamp
// path for every operator-driven arm command; the arm clamps again on its side.
pub(crate) fn clamp_to_limits(joints: &mut JointVec, side: Side) {
    for (j, &[lo, hi]) in joints.iter_mut().zip(arm_limits(side).iter()) {
        *j = j.clamp(lo, hi);
    }
}

/// Tests have no main() to run init_limits, so resolve the v2 limits on
/// first use; concurrent tests settle benignly through get_or_init.
#[cfg(test)]
pub(crate) fn init_limits_for_tests() {
    LIMITS.get_or_init(|| JointLimits::resolve(HardwareVersion::V2));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_pins_each_joint_into_its_range() {
        init_limits_for_tests();
        for side in Side::BOTH {
            let limits = arm_limits(side);

            let mut high = [f64::INFINITY; ARM_DOF];
            clamp_to_limits(&mut high, side);
            for (v, &[_, hi]) in high.iter().zip(limits.iter()) {
                assert_eq!(*v, hi);
            }

            let mut low = [f64::NEG_INFINITY; ARM_DOF];
            clamp_to_limits(&mut low, side);
            for (v, &[lo, _]) in low.iter().zip(limits.iter()) {
                assert_eq!(*v, lo);
            }
        }
    }

    #[test]
    fn clamp_leaves_in_range_values_untouched() {
        init_limits_for_tests();
        for side in Side::BOTH {
            let limits = arm_limits(side);
            let mut mid = [0.0; ARM_DOF];
            for (m, &[lo, hi]) in mid.iter_mut().zip(limits.iter()) {
                *m = (lo + hi) / 2.0;
            }
            let before = mid;
            clamp_to_limits(&mut mid, side);
            assert_eq!(mid, before);
        }
    }

    #[test]
    fn config_joint_limits_are_well_formed() {
        init_limits_for_tests();
        // Each range must be non-empty so clamp and the slider bounds are valid.
        for side in Side::BOTH {
            for &[lo, hi] in arm_limits(side).iter() {
                assert!(lo < hi, "joint range [{lo}, {hi}] must be non-empty");
            }
        }
        let [lo, hi] = gripper_limits();
        assert!(lo < hi, "gripper range [{lo}, {hi}] must be non-empty");
    }
}
