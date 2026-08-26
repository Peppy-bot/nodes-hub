//! Fixtures shared by the backbone's integration tests.

/// Control rate for the tests: 20 ms cycle (Nyquist 25 Hz, above the 15 Hz
/// velocity-filter default) and an 80 ms stale limit, comfortably above the
/// 10 ms state pumps even on a loaded machine.
const CONTROL_RATE_HZ: u32 = 50;

/// The node's full parameter set (most have no schema default, so every test
/// passes them explicitly): v1 hardware, joints-mode upstream, the validated
/// collision band, and a permissive EE cap so streamed chases converge fast.
pub fn params() -> peppygen::Parameters {
    peppygen::Parameters {
        collision_governor_enabled: true,
        control_rate_hz: CONTROL_RATE_HZ,
        d_safe_m: 0.02,
        d_stop_m: 0.005,
        hardware_version: "v1".to_string(),
        max_ee_angular_velocity_rad_s: 0.8,
        max_ee_velocity_m_s: 2.0,
        max_gripper_rate_frac_s: 6.0,
        max_joint_velocity_rad_s_1: 16.754666,
        max_joint_velocity_rad_s_2: 16.754666,
        max_joint_velocity_rad_s_3: 5.445426,
        max_joint_velocity_rad_s_4: 5.445426,
        max_joint_velocity_rad_s_5: 20.943946,
        max_joint_velocity_rad_s_6: 20.943946,
        max_joint_velocity_rad_s_7: 20.943946,
        upstream_mode: "joints".to_string(),
        velocity_filter_cutoff_hz: 15.0,
    }
}
