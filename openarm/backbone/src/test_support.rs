//! What the unit tests share: the arms built from the bundled description,
//! the planner config and the planners of the tests, and a move slot claimed
//! as at a goal's admission.

use std::time::Duration;

use openarm_description::{ARM_DOF, HardwareVersion, Side};
use srs_model::chain_kinematics::{EeCaps, command_smoothing};
use srs_model::{Arm, JointVec, Limit};

use crate::move_slot::MoveSlot;
use crate::planner::{PlanConfig, Planner};

/// The launcher-default per-joint acceleration limits (peppy.json5).
const A_MAX: JointVec = [5.0; ARM_DOF];

/// The control period of the test planners.
const TEST_PERIOD: Duration = Duration::from_millis(10);

/// The `side` arm of `version`, built by `arm_model` from the bundled
/// description.
pub(crate) fn bundled_arm(version: HardwareVersion, side: Side) -> Arm {
    crate::arm_model(version, side).expect("build the arm from the bundled URDF")
}

/// The planner config of the tests: a 10 ms period, 10 rad/s on every
/// joint, the launcher's accelerations, a 1 m/s and 0.8 rad/s hand, and
/// joint limits of +-10 rad.
pub(crate) fn plan_config() -> PlanConfig {
    PlanConfig {
        cycle_period: TEST_PERIOD,
        smoothing: command_smoothing(TEST_PERIOD).unwrap(),
        max_joint_velocity_rad_s: [10.0; ARM_DOF],
        max_joint_acceleration_rad_s2: A_MAX,
        ee: EeCaps {
            linear_m_s: 1.0,
            angular_rad_s: 0.8,
        },
        limits: [Limit {
            lo: -10.0,
            hi: 10.0,
        }; ARM_DOF],
    }
}

/// A planner of the `side` arm of v1 hardware, within the arm's own joint
/// limits, holding `held`.
pub(crate) fn planner_holding(side: Side, held: JointVec) -> Planner {
    let model = bundled_arm(HardwareVersion::V1, side);
    let cfg = PlanConfig {
        limits: model.limits(),
        ..plan_config()
    };
    let mut planner = Planner::new(side, model, cfg);
    planner.commit(held);
    planner
}

/// A move slot claimed as an action handler claims it when it admits a goal.
pub(crate) fn claimed_move_slot() -> MoveSlot {
    let move_slot = MoveSlot::default();
    assert!(move_slot.claim(), "a new slot is free");
    move_slot
}
