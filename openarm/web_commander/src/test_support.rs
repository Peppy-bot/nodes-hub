//! Fixtures the unit tests share: the v2 arm models, the gesture roster baked
//! from them once, the production jog caps, the tests' governor controls, a
//! state built from both, a motor health reading at a given level, and the
//! error of a result poll that timed out.

use std::sync::OnceLock;

use openarm_description::HardwareVersion;

use crate::gestures::Registry;
use crate::limits::{arm_ranges, init_limits_for_tests};
use crate::pose::{ArmModels, JogCaps};
use crate::state::{GovernorSettings, HealthLevel, MotorHealthReading, UiState};

pub fn models() -> ArmModels {
    ArmModels::from_version(HardwareVersion::V2)
}

/// The roster baked from [`models`] against the panel's joint ranges, as
/// bringup bakes it (baking runs the IK rollouts once).
pub fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        init_limits_for_tests();
        Registry::bake(&models(), arm_ranges())
    })
}

/// The jog caps at the production defaults.
pub fn caps() -> JogCaps {
    JogCaps {
        joint_accel_rad_s2: 10.0,
        joint_velocity_rad_s: 1.0,
    }
}

/// The governor controls the tests run under, with collision avoidance on
/// or off.
pub fn governor(collision_enabled: bool) -> GovernorSettings {
    GovernorSettings {
        collision_enabled,
        d_stop: 0.005,
        d_safe: 0.02,
        max_ee_velocity_m_s: 0.25,
        max_gripper_rate_frac_s: 6.0,
    }
}

/// A fresh state at [`governor`] (collision avoidance on) and [`caps`].
pub fn ui_state() -> UiState {
    UiState::new(governor(true), caps())
}

pub fn reading(level: HealthLevel) -> MotorHealthReading {
    MotorHealthReading {
        level,
        effort_fraction_rated: Some(0.4),
        effort_fraction_rated_sustained: Some(0.3),
        effort_fraction_peak: Some(0.2),
        driver_temp_c: Some(41.0),
        winding_temp_c: Some(37.0),
    }
}

/// The error of a result poll that timed out while the producer stands.
pub fn retryable_result_error() -> peppygen::Error {
    peppygen::Error::ActionResultTimeout {
        instance_id: None,
        action_name: "scripted".to_string(),
    }
}
