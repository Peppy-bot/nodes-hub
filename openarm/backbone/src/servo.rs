//! Guarded servo for move_arm goals whose straight line no joint path can track
//! continuously (reaching them requires a branch change). A discrete IK walk
//! cannot cross the singular surface between branches, but the damped
//! resolved-rate law the operator's streaming jog runs passes through it: the
//! damping bounds the joint rates while the task error carries the arm across,
//! deviating from the line only where the geometry forces it and re-converging
//! beyond. The law runs against a reference that walks the line at the
//! end-effector speed cap, held back by a leash whenever the arm falls behind, so
//! a move_arm goal degrades to exactly the motion streaming produces instead of a
//! blind joint-space swing.
//!
//! The plan rolls the identical law out offline (closed-form steps, well under a
//! millisecond each) before accepting the goal: a goal that converges within
//! [`MAX_SERVO_S`] is accepted with that duration, one that does not is rejected
//! rather than started. That offline proof is the only reachability check the
//! servo needs, so the runtime just runs the law and trusts the plan, with
//! [`MAX_SERVO_S`] as its lone backstop.
//!
//! The law itself is [`chain_kinematics::servo`], which knows nothing about an
//! SRS arm. What stays here is what is this robot's: the smoothing filter, the
//! move budget, the leash length, and the caller's [`PlanTolerance`], which the
//! law judges arrival against.
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
use srs_model::Arm;
use srs_model::chain_kinematics::{ServoLimits, ServoTolerances, Smoother};
use srs_model::nalgebra::Isometry3;

use crate::trajectory::PlanLimits;
use crate::types::{ARM_DOF, JointVec, PlanTolerance};

/// The end-effector speed budget a Cartesian step runs under: the launcher's
/// linear cap and the angular slew cap.
pub use srs_model::chain_kinematics::EeCaps;

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

/// One [`ButterworthFilter`] per joint, bounding the jerk of the servo's command.
/// Run in both the runtime step and the plan-time rollout, so the validated
/// duration already includes the (small) filter lag and the Q4 timeout stays
/// honest.
#[derive(Clone, Copy)]
pub struct JointSmoothing([ButterworthFilter; ARM_DOF]);

impl Smoother<ARM_DOF> for JointSmoothing {
    fn smooth(&mut self, q: &JointVec) -> JointVec {
        std::array::from_fn(|i| self.0[i].filter(q[i]))
    }
}

/// One servo move's state, and one tick's outcome: the generic law's, at this
/// arm's joint count and behind this arm's smoother.
pub type ServoState = srs_model::chain_kinematics::ServoState<ARM_DOF, JointSmoothing>;
pub type ServoStep = srs_model::chain_kinematics::ServoStep<ARM_DOF>;

/// A servo move from `start` to `end`, leashed and smoothed the way this arm is.
pub fn servo_from(
    start: Isometry3<f64>,
    end: Isometry3<f64>,
    smoothing: ButterworthFilter,
) -> ServoState {
    ServoState::new(start, end, LEASH_M, JointSmoothing([smoothing; ARM_DOF]))
}

impl PlanLimits<'_> {
    /// The plan's budgets as one servo tick's, so the rollout that accepts a goal
    /// and the runtime that executes it cannot be given different numbers. `dt`
    /// is the tick's own: the rollout steps at the nominal control period, the
    /// runtime at the period it measured, and each velocity-scales by what it used.
    /// `tolerance` is the caller's arrival slack, which decides convergence and
    /// nothing else.
    pub fn servo_at(&self, dt: Duration, tolerance: PlanTolerance) -> ServoLimits<ARM_DOF> {
        ServoLimits {
            max_joint_velocity: *self.max_joint_velocity_rad_s,
            ee: self.ee,
            tolerances: ServoTolerances {
                position_m: tolerance.position_m,
                orientation_rad: tolerance.orientation_rad,
            },
            dt_s: dt.as_secs_f64(),
        }
    }
}

/// Roll the servo law out offline at the control period per step: the plan-time
/// proof that the law reaches the pose, returning how long it took, or `None`
/// when it has not converged within [`MAX_SERVO_S`] (unreachable this way).
/// Deterministic and identical to the runtime law, so an accepted goal executes
/// the motion that was validated; a few thousand closed-form steps cost
/// milliseconds.
pub fn rollout(
    model: &Arm,
    start: &Isometry3<f64>,
    end: &Isometry3<f64>,
    seed: JointVec,
    limits: &PlanLimits,
    tolerance: PlanTolerance,
) -> Option<f64> {
    let mut state = servo_from(*start, *end, limits.smoothing);
    srs_model::chain_kinematics::rollout(
        model.chain(),
        &mut state,
        seed,
        &limits.servo_at(limits.control_period, tolerance),
        MAX_SERVO_S,
    )
}
