//! Guarded servo for move_arm goals whose straight line no joint path can track
//! continuously (reaching them requires a branch change). A discrete IK walk
//! cannot cross the singular surface between branches, but the damped
//! resolved-rate law the operator's streaming jog runs passes through it: the
//! damping bounds the joint rates while the task error carries the arm across,
//! deviating from the line only where the geometry forces it and re-converging
//! beyond. This module runs that law against a reference that walks the line at
//! the end-effector speed cap, held back by a leash whenever the arm falls
//! behind, so a move_arm goal degrades to exactly the motion streaming produces
//! instead of a blind joint-space swing.
//!
//! The plan rolls the identical law out offline (closed-form steps, well under a
//! millisecond each) before accepting the goal: a goal that converges within
//! [`MAX_SERVO_S`] is accepted with that duration, one that does not is rejected
//! rather than started. That offline proof is the only reachability check the
//! servo needs, so the runtime just runs the law and trusts the plan, with
//! [`MAX_SERVO_S`] as its lone backstop.
//!
//! Tuning constants are anchored to MoveIt Servo's defaults (`servo_parameters.yaml`)
//! where the mechanism is the same: the smoothing cutoff and the convergence
//! tolerances. The singularity strategy is deliberately the opposite of MoveIt's,
//! which halts at a singularity (a plain pseudo-inverse with velocity scaled to zero
//! by the Jacobian condition number); this servo damps (DLS) to pass THROUGH one,
//! which is the reason the guarded servo exists, so it takes no condition-number
//! thresholds and its damping has no MoveIt analogue.

use std::time::Duration;

use control_core::filters::ButterworthFilter;
use control_core::servo::{ORIENTATION_TOLERANCE_RAD, POSITION_TOLERANCE_M};
use srs_model::nalgebra::{Isometry3, Rotation3, Vector3};
use srs_model::{Arm, DEFAULT_DLS_LAMBDA};

use crate::chase::rate_limited;
use crate::trajectory::{PlanLimits, interpolate_pose};
use crate::types::{ARM_DOF, JointVec};

/// The reference stops walking while the arm is farther than this from it, so a
/// wall crossing is ground through instead of the reference running away. Bespoke
/// to the leashed-reference law; no MoveIt analogue.
const LEASH_M: f64 = 0.05;
/// Hard ceiling on a servo move. The plan-time rollout runs at most this long; a
/// goal that has not converged by then is taken as unreachable and rejected, and
/// the runtime aborts a move still going past it (the rare case where the
/// governor holds the arm off its path indefinitely).
pub const MAX_SERVO_S: f64 = 30.0;

/// Cutoff (Hz) for the per-joint Butterworth smoothing of the servo's joint command,
/// so a reconfiguration through a singularity ramps rather than stepping - bounded
/// jerk on the real arm. Only the servo needs it; the line and joint tiers are
/// already quintic-smooth. MoveIt Servo's first-order default (`low_pass_filter_coeff`
/// = 1.5) is a -3 dB cutoff of `atan(1/1.5) / (pi * Ts)`; at the 100 Hz control rate
/// that is ~18.7 Hz, applied here to the steeper second-order filter so it smooths at
/// least as much.
const SERVO_SMOOTHING_CUTOFF_HZ: f64 = 18.7;

/// The per-joint command smoother for one control period, or a refusal when
/// the period cannot carry the cutoff (Nyquist at `0.5 / period` must sit
/// above it). Built once at setup, where a bad `control_rate_hz` can still be
/// refused; every [`ServoState`] then copies the value as proof it exists.
pub fn smoothing_for(
    control_period: Duration,
) -> Result<ButterworthFilter, control_core::filters::FilterError> {
    ButterworthFilter::from_cutoff(SERVO_SMOOTHING_CUTOFF_HZ, control_period.as_secs_f64())
}

/// The end-effector speed budget a Cartesian step runs under: the launcher's
/// linear cap and the angular slew cap.
#[derive(Clone, Copy)]
pub struct EeCaps {
    pub linear_m_s: f64,
    pub angular_rad_s: f64,
}

/// One damped resolved-rate step of the joints from `q` toward `target`, given
/// the end-effector pose `ee` already computed at `q` (every caller has it).
///
/// The task error is capped before it is resolved: position at the speed
/// budget, orientation at the slew budget. A target metres away produces the
/// same bounded step as one millimetres away, which is what lets the same law
/// serve a planned move and a live stream. Linear error has a 1 mm converged
/// deadband; rotation has none (it is the tracking floor of a live pose
/// stream).
pub fn rate_step_toward(
    model: &mut Arm,
    q: &JointVec,
    ee: &Isometry3<f64>,
    target: &Isometry3<f64>,
    max_joint_velocity_rad_s: &JointVec,
    caps: EeCaps,
    dt_s: f64,
) -> JointVec {
    let dp_world = target.translation.vector - ee.translation.vector;
    let dp_world = if dp_world.norm() > POSITION_TOLERANCE_M {
        dp_world * (caps.linear_m_s * dt_s / dp_world.norm()).min(1.0)
    } else {
        Vector3::zeros()
    };
    let rot_err: Rotation3<f64> = (target.rotation * ee.rotation.inverse()).to_rotation_matrix();
    let dw_world = rot_err
        .axis_angle()
        .map(|(axis, angle)| axis.into_inner() * angle.min(caps.angular_rad_s * dt_s))
        .unwrap_or_else(Vector3::zeros);
    model.rate_step(
        q,
        dp_world,
        dw_world,
        max_joint_velocity_rad_s,
        dt_s,
        DEFAULT_DLS_LAMBDA,
    )
}

/// One servo move's mutable state: where the reference is on the line, and the
/// per-joint output smoothing. The joint state lives with the caller (the planner's
/// commanded setpoint), which each tick's step advances.
pub struct ServoState {
    start: Isometry3<f64>,
    end: Isometry3<f64>,
    /// Reference progress along the line, 0..=1.
    reference_s: f64,
    /// Butterworth smoother per joint, bounding the jerk of the command. Run in both
    /// the runtime step and the plan-time rollout, so the validated duration already
    /// includes the (small) filter lag and the Q4 timeout stays honest.
    smoothing: [ButterworthFilter; ARM_DOF],
}

/// One tick's outcome.
pub enum ServoStep {
    /// Advanced: the new joint setpoint to command.
    Stepped(JointVec),
    /// Reached the goal pose within tolerance.
    Converged(JointVec),
}

impl ServoState {
    /// Distance (m) from `q`'s end-effector to the goal position, for timeout
    /// reporting.
    pub fn position_err_m(&self, model: &mut Arm, q: &JointVec) -> f64 {
        let ee_base = model.at(q).ee_pose();
        let ee = model.world_pose(&ee_base);
        (self.end.translation.vector - ee.translation.vector).norm()
    }

    pub fn new(start: Isometry3<f64>, end: Isometry3<f64>, smoothing: ButterworthFilter) -> Self {
        Self {
            start,
            end,
            reference_s: 0.0,
            smoothing: [smoothing; ARM_DOF],
        }
    }

    /// Advance one tick of `dt`: walk the reference (leashed to the arm), take
    /// one damped resolved-rate step of the joints toward it, and report whether
    /// the goal pose is reached.
    pub fn step(
        &mut self,
        model: &mut Arm,
        q: &JointVec,
        max_joint_velocity_rad_s: &JointVec,
        caps: EeCaps,
        dt: Duration,
    ) -> ServoStep {
        let dt_s = dt.as_secs_f64();
        let ee_base = model.at(q).ee_pose();
        let ee = model.world_pose(&ee_base);

        // Converged on the goal itself (not merely the reference)?
        let goal_pos_err = (self.end.translation.vector - ee.translation.vector).norm();
        let goal_rot_err = ee.rotation.angle_to(&self.end.rotation);
        if self.reference_s >= 1.0
            && goal_pos_err < POSITION_TOLERANCE_M
            && goal_rot_err < ORIENTATION_TOLERANCE_RAD
        {
            return ServoStep::Converged(*q);
        }

        // Walk the reference at the speed cap while the arm holds the leash; a
        // zero-length line (pure reorientation) starts fully advanced.
        let line_len = (self.end.translation.vector - self.start.translation.vector).norm();
        let reference = interpolate_pose(&self.start, &self.end, self.reference_s);
        let ref_pos_err = (reference.translation.vector - ee.translation.vector).norm();
        if line_len < POSITION_TOLERANCE_M {
            self.reference_s = 1.0;
        } else if ref_pos_err < LEASH_M {
            self.reference_s = (self.reference_s + caps.linear_m_s * dt_s / line_len).min(1.0);
        }
        let reference = interpolate_pose(&self.start, &self.end, self.reference_s);

        let next = rate_step_toward(
            model,
            q,
            &ee,
            &reference,
            max_joint_velocity_rad_s,
            caps,
            dt_s,
        );
        // Smooth each joint to bound the command's jerk through the reconfiguration,
        // then re-clamp the step to the joint velocity limit: a Butterworth overshoots
        // its input, so without this the smoothed command could exceed the limit
        // rate_step enforced on `next`. The clamp is the final safety stage, so the
        // commanded velocity always holds regardless of the filter transient.
        let smoothed = std::array::from_fn(|i| {
            let filtered = self.smoothing[i].filter(next[i]);
            rate_limited(q[i], filtered, max_joint_velocity_rad_s[i], dt_s)
        });
        ServoStep::Stepped(smoothed)
    }
}

/// Roll the servo law out offline at the control period per step: the plan-time
/// proof that the law reaches the pose, returning how long it took, or `None`
/// when it has not converged within [`MAX_SERVO_S`] (unreachable this way).
/// Deterministic and identical to the runtime law, so an accepted goal executes
/// the motion that was validated; a few thousand closed-form steps cost
/// milliseconds.
pub fn rollout(
    model: &mut Arm,
    start: &Isometry3<f64>,
    end: &Isometry3<f64>,
    seed: JointVec,
    limits: &PlanLimits,
) -> Option<f64> {
    let mut state = ServoState::new(*start, *end, limits.smoothing);
    let mut q = seed;
    let dt = limits.control_period;
    let steps = (MAX_SERVO_S / dt.as_secs_f64()).ceil() as usize;
    for k in 0..steps {
        match state.step(model, &q, limits.max_joint_velocity_rad_s, limits.ee, dt) {
            ServoStep::Stepped(next) => q = next,
            ServoStep::Converged(_) => return Some(k as f64 * dt.as_secs_f64()),
        }
    }
    None
}
