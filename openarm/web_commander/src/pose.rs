//! Cartesian jog for the panel: differential (resolved-rate) servoing only. The
//! joint vector stays the single source of truth: the UI reads a pose off forward
//! kinematics and writes joints back through damped-least-squares steps, so a pose
//! is only ever an input/display lens, never stored state. Exact point-to-point
//! pose moves are the backbone's move_arm action, not this module.
//!
//! Poses are in the world frame (matching the backbone's move_arm action) and
//! orientation is exposed as intrinsic-XYZ roll/pitch/yaw via nalgebra's
//! `euler_angles` round-trip for display; every orientation computation runs on
//! quaternions, so no control path ever interpolates euler components.

use std::sync::{Arc, Mutex};

use control_core::servo::{ORIENTATION_TOLERANCE_RAD, POSITION_TOLERANCE_M};
use openarm_description::HardwareVersion;
use srs_model::nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion, Vector3};
use srs_model::{Arm, ArmAnglePolicy, DEFAULT_DLS_LAMBDA};

use crate::state::{ARM_DOF, Side};

/// A world-frame end-effector pose as `[x, y, z, roll, pitch, yaw]` (metres,
/// radians): the form the panel edits and displays.
pub type Pose = [f64; 6];

/// A discrete move counts as "arrived" within this position / angle slack. The backbone
/// reports success once its setpoints finish streaming, which a governor stop satisfies
/// without the arm following, so the move handlers re-check the measured final pose
/// against these. One angle slack serves both joint error and orientation error.
pub const REACHED_POS_TOL_M: f64 = 0.02;
pub const REACHED_ANGLE_TOL_RAD: f64 = 5.0 * std::f64::consts::PI / 180.0; // 5 degrees

/// The two per-side arm models (FK/Jacobian/limits) behind mutexes, plus each
/// side's URDF joint velocity limits for step clamping.
///
/// Each lock is held only for a single synchronous FK or Jacobian call and never
/// across an `.await`. The owner task is the sole caller, so the locks are
/// uncontended; the mutex only provides the interior mutability an `Arm` needs
/// (`at` takes `&mut self`) behind `ArmModels`'s shared `&self` methods.
#[derive(Clone)]
pub struct ArmModels {
    left: Arc<Mutex<Arm>>,
    right: Arc<Mutex<Arm>>,
    left_velocity_limits: [f64; ARM_DOF],
    right_velocity_limits: [f64; ARM_DOF],
    // World-frame x/y/z reachable bounds per side, from the FK envelope (computed once
    // at construction); the panel sizes its position sliders to these.
    left_bounds: [[f64; 2]; 3],
    right_bounds: [[f64; 2]; 3],
}

impl ArmModels {
    /// Build both arm models from the generation's embedded description, mirroring
    /// the backbone's `arm_model` (same URDF, same elbow singularity floor) so the panel
    /// solves against the identical chain the backbone governs. Joint velocity limits
    /// come from the same URDF, so a jog step never demands more per tick than the
    /// backbone's chase can follow.
    pub fn from_version(version: HardwareVersion) -> Self {
        let mut left = build_arm(version, Side::Left);
        let mut right = build_arm(version, Side::Right);
        let left_bounds = workspace_aabb(&mut left);
        let right_bounds = workspace_aabb(&mut right);
        Self {
            left_velocity_limits: velocity_limits(version.urdf(), Side::Left),
            right_velocity_limits: velocity_limits(version.urdf(), Side::Right),
            left: Arc::new(Mutex::new(left)),
            right: Arc::new(Mutex::new(right)),
            left_bounds,
            right_bounds,
        }
    }

    /// World-frame end-effector pose of `joints` for `side`.
    pub fn ee_pose_world(&self, side: Side, joints: &[f64; ARM_DOF]) -> Pose {
        let mut model = self.get(side).lock().unwrap_or_else(|p| p.into_inner());
        let base = model.at(joints).ee_pose();
        decompose(&model.world_pose(&base))
    }

    /// World-frame end-effector orientation of `joints` as a quaternion `[x, y, z, w]`,
    /// for the panel's arcball (which composes orientation on quaternions, never euler).
    pub fn ee_quat_world(&self, side: Side, joints: &[f64; ARM_DOF]) -> [f64; 4] {
        let mut model = self.get(side).lock().unwrap_or_else(|p| p.into_inner());
        let base = model.at(joints).ee_pose();
        let q = model.world_pose(&base).rotation;
        [q.i, q.j, q.k, q.w]
    }

    /// Solve inverse kinematics for a world-frame end-effector pose (position in
    /// metres, orientation as a unit quaternion), seeded from `seed` so the branch
    /// nearest the current configuration is chosen. `None` when the pose is
    /// unreachable or admits no in-limit solution. Used to preview an Actions-mode
    /// pose move as joints; the backbone re-solves and plans the move itself.
    pub fn solve_ik(
        &self,
        side: Side,
        position: [f64; 3],
        rotation: UnitQuaternion<f64>,
        seed: &[f64; ARM_DOF],
    ) -> Option<[f64; ARM_DOF]> {
        self.solve_ik_with(side, position, rotation, ArmAnglePolicy::FromSeed, seed)
    }

    /// IK holding the arm angle at `psi` (elbow swivel pinned): the null-space jog
    /// re-solves the current end-effector pose at a stepped psi. `None` when no
    /// in-limit solution exists at that psi.
    fn solve_ik_fixed(
        &self,
        side: Side,
        position: [f64; 3],
        rotation: UnitQuaternion<f64>,
        psi: f64,
        seed: &[f64; ARM_DOF],
    ) -> Option<[f64; ARM_DOF]> {
        self.solve_ik_with(side, position, rotation, ArmAnglePolicy::Fixed(psi), seed)
    }

    fn solve_ik_with(
        &self,
        side: Side,
        position: [f64; 3],
        rotation: UnitQuaternion<f64>,
        arm_angle: ArmAnglePolicy,
        seed: &[f64; ARM_DOF],
    ) -> Option<[f64; ARM_DOF]> {
        let model = self.get(side).lock().unwrap_or_else(|p| p.into_inner());
        let world = Isometry3::from_parts(
            Translation3::new(position[0], position[1], position[2]),
            rotation,
        );
        let base_target = model.base_pose(&world);
        model.solve_ik(&base_target, arm_angle, seed).map(|s| s.q)
    }

    /// Current arm angle psi (elbow swivel, rad) of `joints` for `side`; `None` only at
    /// the straight-arm singularity, which the elbow floor keeps the arm off.
    pub fn arm_angle(&self, side: Side, joints: &[f64; ARM_DOF]) -> Option<f64> {
        let model = self.get(side).lock().unwrap_or_else(|p| p.into_inner());
        model.arm_angle(joints)
    }

    /// Scale the joint delta `to - from` so no joint exceeds its velocity budget over
    /// `dt_s`, preserving direction (the same clamp the resolved-rate step applies).
    fn velocity_clamp(
        &self,
        side: Side,
        from: &[f64; ARM_DOF],
        to: &[f64; ARM_DOF],
        dt_s: f64,
    ) -> [f64; ARM_DOF] {
        let budget = self.velocity_limits(side);
        let scale = (0..ARM_DOF)
            .map(|i| {
                let dq = (to[i] - from[i]).abs();
                let cap = budget[i] * dt_s;
                if dq > cap { cap / dq } else { 1.0 }
            })
            .fold(1.0_f64, f64::min);
        std::array::from_fn(|i| from[i] + (to[i] - from[i]) * scale)
    }

    /// One damped-least-squares joint step realizing a world-frame task increment
    /// at the configuration `joints`. The untasked half of the twist is commanded
    /// to zero, so a position step softly holds orientation (and an orientation
    /// step softly holds position); the damping trades that hold away only where
    /// the geometry forces it. The step is scaled so no joint exceeds its URDF
    /// velocity limit over `dt_s`, and clamped into position limits. `None` when
    /// limits pin the step (the free-space envelope boundary).
    fn resolved_rate_step(
        &self,
        side: Side,
        joints: &[f64; ARM_DOF],
        task: RateTask,
        dt_s: f64,
    ) -> Option<[f64; ARM_DOF]> {
        // The shared step (srs_model): rotates the task into the base frame,
        // damped pseudo-inverse, velocity-scaled against the same limits the
        // backbone chases under, clamped into position limits.
        let (dp_world, dw_world) = match task {
            RateTask::Linear(dx) => (dx, Vector3::zeros()),
            RateTask::Angular(dx) => (Vector3::zeros(), dx),
        };
        let q = {
            let mut model = self.get(side).lock().unwrap_or_else(|p| p.into_inner());
            model.rate_step(
                joints,
                dp_world,
                dw_world,
                self.velocity_limits(side),
                dt_s,
                DEFAULT_DLS_LAMBDA,
            )
        };
        let dx_world = match task {
            RateTask::Linear(dx) | RateTask::Angular(dx) => dx,
        };
        // Limits may have eaten the step: measure what actually moved along the
        // demanded direction and treat a mostly-pinned step as the boundary.
        let achieved = self.ee_pose_world(side, &q);
        let before = self.ee_pose_world(side, joints);
        let moved = match task {
            RateTask::Linear(_) => Vector3::new(
                achieved[0] - before[0],
                achieved[1] - before[1],
                achieved[2] - before[2],
            ),
            RateTask::Angular(_) => {
                let q_a = UnitQuaternion::from_euler_angles(achieved[3], achieved[4], achieved[5]);
                let q_b = UnitQuaternion::from_euler_angles(before[3], before[4], before[5]);
                let err = q_a * q_b.inverse();
                err.axis()
                    .map(|a| a.into_inner() * err.angle())
                    .unwrap_or_default()
            }
        };
        let progress = moved.dot(&dx_world) / dx_world.norm_squared().max(1e-12);
        (progress > MIN_STEP_PROGRESS).then_some(q)
    }

    fn get(&self, side: Side) -> &Arc<Mutex<Arm>> {
        match side {
            Side::Left => &self.left,
            Side::Right => &self.right,
        }
    }

    /// Per-joint URDF velocity limits (rad/s) for `side`: the budget the backbone's
    /// chase enforces, shared by the jog step clamp and the gesture bake asserts.
    pub(crate) fn velocity_limits(&self, side: Side) -> &[f64; ARM_DOF] {
        match side {
            Side::Left => &self.left_velocity_limits,
            Side::Right => &self.right_velocity_limits,
        }
    }

    /// Per-joint [lo, hi] position limits (rad) for `side`, from the same model the
    /// IK solves against (elbow singularity floor included).
    pub(crate) fn joint_ranges(&self, side: Side) -> [[f64; 2]; ARM_DOF] {
        let model = self.get(side).lock().unwrap_or_else(|p| p.into_inner());
        let lims = model.limits();
        std::array::from_fn(|i| [lims[i].lo, lims[i].hi])
    }

    /// World-frame x/y/z reachable bounds `[[min, max]; 3]` for `side`, so the panel
    /// sizes its position sliders to the arm's actual reach (correct per generation).
    pub fn pos_bounds(&self, side: Side) -> [[f64; 2]; 3] {
        match side {
            Side::Left => self.left_bounds,
            Side::Right => self.right_bounds,
        }
    }
}

/// A world-frame task increment for one resolved-rate step: a linear metre-step or
/// an angular axis-angle step (radians).
enum RateTask {
    Linear(Vector3<f64>),
    Angular(Vector3<f64>),
}

/// Angular jog rate (rad/s). The linear rate is the operator's live EE speed cap
/// (one knob governs both the jog and the backbone's enforcement); rotation has no
/// operator knob, so it jogs at this fixed, comfortable rate.
const JOG_ROT_RATE_RAD_S: f64 = 1.5;
/// Arm-angle (elbow swivel) jog rate (rad/s); fixed like the rotation rate, since the
/// null-space motion has no operator speed knob.
const ARM_ANGLE_RATE_RAD_S: f64 = 1.2;

/// A joint jog is converged once every joint is within this of the target and nearly
/// stopped; the jog then lands exactly on the target and retires.
const JOINT_JOG_CONVERGED_RAD: f64 = 1e-4;
const JOINT_JOG_STOP_RAD_S: f64 = 1e-3;

/// Per-tick Cartesian step caps, derived from the actual tick period and the
/// operator's live EE speed cap, so a different `command_rate_hz` or a retuned
/// speed knob changes the step size, never the speed.
#[derive(Clone, Copy)]
pub struct JogCaps {
    pub pos_step_m: f64,
    pub rot_step_rad: f64,
    pub arm_angle_step_rad: f64,
    /// Joint-jog acceleration limit (rad/s^2): the physics is integrated against `dt_s`,
    /// so this stays rate-valued. The whole jog is acceleration-limited (ramp up, brake to
    /// rest), so no separate speed cap is needed.
    pub joint_accel_rad_s2: f64,
    pub dt_s: f64,
}

impl JogCaps {
    /// Caps for one tick of length `dt` seconds at `max_ee_velocity_m_s`. The EE speed is
    /// the operator's governor knob (validated positive-finite at entry); the joint-jog
    /// acceleration is a node parameter, so a deployment can tune the jog feel without a
    /// rebuild. The clamps here only bound a degenerate combination.
    pub fn per_tick(dt_s: f64, max_ee_velocity_m_s: f64, joint_accel_rad_s2: f64) -> Self {
        Self {
            pos_step_m: (max_ee_velocity_m_s * dt_s).clamp(1e-5, 0.05),
            rot_step_rad: (JOG_ROT_RATE_RAD_S * dt_s).clamp(1e-4, 0.2),
            arm_angle_step_rad: (ARM_ANGLE_RATE_RAD_S * dt_s).clamp(1e-4, 0.2),
            joint_accel_rad_s2,
            dt_s,
        }
    }
}

/// Which component a Cartesian jog drives: `Position` tracks x/y/z (holding
/// orientation), `Orientation` tracks the hand frame (holding position), `ArmAngle`
/// swivels the elbow through the null space (holding the whole end-effector pose).
/// Whatever control the operator touches leads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JogMode {
    Position,
    Orientation,
    ArmAngle,
}

/// The operator's active drive for one arm, reconciled toward once per stream tick.
/// Joint sliders and world-frame controls both arm one of these, so the two spaces
/// share a single "what is this side being driven toward" slot and one advance path;
/// arming either clears the other, since the spaces must not fight.
#[derive(Clone, Copy, Debug)]
pub enum Jog {
    /// Joint sliders: walk the streamed target toward the slider one acceleration-limited
    /// step per tick (carrying `vel`, the per-joint jog velocity, across ticks), so the
    /// arm ramps smoothly instead of snapping. `target` refreshes as the operator drags;
    /// `vel` is preserved across those refreshes so a continuous drag keeps its momentum.
    Joints {
        target: [f64; ARM_DOF],
        vel: [f64; ARM_DOF],
    },
    /// World-frame controls: step the joints one resolved-rate increment per tick
    /// toward the target, since the command wire carries only joints and a pose jump
    /// would teleport or branch-flip the arm.
    Cartesian(CartesianJog),
}

/// An armed Cartesian jog: which component leads (`mode`), the desired world-frame
/// pose (used by `Position`/`Orientation`), and the desired arm angle (used by
/// `ArmAngle`); the field the active mode does not use is ignored.
#[derive(Clone, Copy, Debug)]
pub struct CartesianJog {
    pub mode: JogMode,
    pub desired: Pose,
    pub arm_angle: f64,
}

/// One jog tick's outcome.
pub enum JogStep {
    /// The commanded pose reached the desired pose; the jog is complete.
    Converged,
    /// Advanced one capped Cartesian step: the new joint target to stream.
    Stepped([f64; ARM_DOF]),
    /// The next step is pinned by limits or the envelope boundary: hold this tick.
    /// The jog stays armed, so pulling the desired pose back into reach resumes it.
    Blocked,
}

/// One joint-jog tick's outcome (mirrors [`JogStep`]; joints have no reach boundary, so
/// there is no `Blocked`).
pub enum JointJogStep {
    /// Every joint arrived and stopped; the setpoint lands exactly on the target and the
    /// jog retires.
    Converged([f64; ARM_DOF]),
    /// Advanced one acceleration-limited step: the new joint setpoint to stream and the
    /// carried per-joint velocity for the next tick.
    Stepped {
        joints: [f64; ARM_DOF],
        vel: [f64; ARM_DOF],
    },
}

/// Advance a joint jog one tick: walk each joint of `current` toward `target` under an
/// acceleration limit (`caps.joint_accel_rad_s2`), carrying `vel` so the velocity stays
/// continuous. The approach speed is set so the acceleration limit can brake to rest by
/// the target, and the position is clamped there so it never overshoots; the carried
/// velocity then bleeds to rest over the next few ticks under the same cap. Targets are
/// pre-clamped to joint limits by the caller, so a joint jog never blocks.
pub fn joint_jog_tick(
    current: &[f64; ARM_DOF],
    target: &[f64; ARM_DOF],
    vel: &[f64; ARM_DOF],
    caps: JogCaps,
) -> JointJogStep {
    let dt = caps.dt_s;
    let a_max = caps.joint_accel_rad_s2;
    let dv_max = a_max * dt;

    let next: [(f64, f64); ARM_DOF] = std::array::from_fn(|i| {
        let error = target[i] - current[i];
        // Approach speed set so a_max can brake to rest by the target (no overshoot): the
        // whole motion is acceleration-limited, ramping up then braking down.
        let v_brake = (2.0 * a_max * error.abs()).sqrt();
        let v_desired = error.signum() * v_brake;
        // |v_next - vel| <= dv_max, so the acceleration bound holds on every tick,
        // including the last: the velocity is never teleported to zero.
        let v_next = vel[i] + (v_desired - vel[i]).clamp(-dv_max, dv_max);
        let step = v_next * dt;
        // Land on the target only when this step is directed at it and would reach or
        // cross it (`step*error >= error^2`), clamping the position there so it never
        // overshoots. A mid-drag reversal (velocity still opposing `error`) fails this
        // test, so the joint decelerates and turns around instead of snapping backward.
        let joint = if step * error >= error * error {
            target[i]
        } else {
            current[i] + step
        };
        (joint, v_next)
    });
    let joints: [f64; ARM_DOF] = std::array::from_fn(|i| next[i].0);
    let vel: [f64; ARM_DOF] = std::array::from_fn(|i| next[i].1);

    let settled = (0..ARM_DOF).all(|i| {
        (target[i] - joints[i]).abs() < JOINT_JOG_CONVERGED_RAD
            && vel[i].abs() < JOINT_JOG_STOP_RAD_S
    });
    if settled {
        JointJogStep::Converged(*target)
    } else {
        JointJogStep::Stepped { joints, vel }
    }
}

const ARM_ANGLE_CONVERGED_RAD: f64 = 2e-3;
/// A resolved-rate step that achieves less than this fraction of its demanded
/// motion is pinned by joint limits: the envelope boundary in free space.
const MIN_STEP_PROGRESS: f64 = 0.2;

/// Advance one Cartesian jog tick for `side`: step the pose of `joints` one capped
/// increment toward `jog.desired` under `jog.mode`. Steps are velocity-clamped in
/// joint space, so the target walks toward the desired values and stops exactly where
/// reach or limits end.
pub fn jog_tick(
    models: &ArmModels,
    side: Side,
    joints: &[f64; ARM_DOF],
    jog: &CartesianJog,
    caps: JogCaps,
) -> JogStep {
    let current = models.ee_pose_world(side, joints);
    let step = match jog.mode {
        JogMode::Position => {
            let Some(dx) = position_step(&current, &jog.desired, caps.pos_step_m) else {
                return JogStep::Converged;
            };
            models.resolved_rate_step(side, joints, RateTask::Linear(dx), caps.dt_s)
        }
        JogMode::Orientation => {
            let Some(dw) = orientation_step(&current, &jog.desired, caps.rot_step_rad) else {
                return JogStep::Converged;
            };
            models.resolved_rate_step(side, joints, RateTask::Angular(dw), caps.dt_s)
        }
        JogMode::ArmAngle => {
            let Some(psi_cur) = models.arm_angle(side, joints) else {
                return JogStep::Blocked;
            };
            let err = jog.arm_angle - psi_cur;
            if err.abs() < ARM_ANGLE_CONVERGED_RAD {
                return JogStep::Converged;
            }
            let psi_step = psi_cur + err.clamp(-caps.arm_angle_step_rad, caps.arm_angle_step_rad);
            // Hold the current end-effector pose, re-solve at the stepped arm angle: a
            // pure null-space move (the elbow swivels, the hand stays put).
            let position = [current[0], current[1], current[2]];
            let q4 = models.ee_quat_world(side, joints);
            let rotation =
                UnitQuaternion::from_quaternion(Quaternion::new(q4[3], q4[0], q4[1], q4[2]));
            models
                .solve_ik_fixed(side, position, rotation, psi_step, joints)
                .map(|q_new| models.velocity_clamp(side, joints, &q_new, caps.dt_s))
        }
    };
    match step {
        Some(q) => JogStep::Stepped(q),
        None => JogStep::Blocked,
    }
}

/// One tick's world-frame linear step toward the desired position, capped in
/// norm; `None` once within the convergence slack.
fn position_step(current: &Pose, desired: &Pose, cap_m: f64) -> Option<Vector3<f64>> {
    let delta = Vector3::new(
        desired[0] - current[0],
        desired[1] - current[1],
        desired[2] - current[2],
    );
    let dist = delta.norm();
    if dist < POSITION_TOLERANCE_M {
        return None;
    }
    Some(delta * (cap_m.min(dist) / dist))
}

/// One tick's world-frame angular step (axis-angle vector) toward the desired
/// orientation, capped in angle; `None` once within the convergence slack. The
/// error is a quaternion geodesic, so it is chart-independent: no euler seam or
/// gimbal alias can inflate it.
fn orientation_step(current: &Pose, desired: &Pose, cap_rad: f64) -> Option<Vector3<f64>> {
    let q_cur = UnitQuaternion::from_euler_angles(current[3], current[4], current[5]);
    let q_des = UnitQuaternion::from_euler_angles(desired[3], desired[4], desired[5]);
    let err = q_des * q_cur.inverse();
    let angle = err.angle();
    if angle < ORIENTATION_TOLERANCE_RAD {
        return None;
    }
    let axis = err.axis()?;
    Some(axis.into_inner() * cap_rad.min(angle))
}

/// Build one arm model from the generation's embedded description, with the elbow
/// singularity floor applied and the URDF's tcp frame mounted (mirrors
/// `openarm_backbone`'s `arm_model`, so the panel's poses and the backbone's are the
/// same frame). A bad base link or tcp link aborts bringup, matching how the
/// backbone fails.
fn build_arm(version: HardwareVersion, side: Side) -> Arm {
    let base = version.base_link(side.description());
    Arm::from_urdf(version.urdf(), base)
        .unwrap_or_else(|e| panic!("build {} arm model from base '{base}': {e}", side.label()))
        .with_lower_floor(
            version.elbow_joint_index(),
            version.elbow_singularity_floor_rad(),
        )
        .with_tool_link(version.tcp_link(side.description()))
        .unwrap_or_else(|e| panic!("mount the {} gripper tcp frame: {e}", side.label()))
}

/// Per-joint velocity limits (rad/s) for `side`, j1..j7, from the bundled URDF:
/// the same numbers the backbone's chase enforces, so commander steps and backbone follow
/// capability agree by construction.
fn velocity_limits(urdf: &str, side: Side) -> [f64; ARM_DOF] {
    let robot = urdf_rs::read_from_string(urdf).expect("bundled URDF must parse");
    std::array::from_fn(|i| {
        let name = format!("openarm_{}_joint{}", side.label(), i + 1);
        let joint = robot
            .joints
            .iter()
            .find(|j| j.name == name)
            .unwrap_or_else(|| panic!("URDF missing joint {name}"));
        let v = joint.limit.velocity;
        assert!(
            v.is_finite() && v > 0.0,
            "URDF velocity limit for {name} must be positive"
        );
        v
    })
}

fn decompose(pose: &Isometry3<f64>) -> Pose {
    let t = pose.translation.vector;
    let (roll, pitch, yaw) = pose.rotation.euler_angles();
    [t.x, t.y, t.z, roll, pitch, yaw]
}

/// Euclidean distance (m) between two world-frame points.
pub fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Rotation angle (rad) between two orientations given as `[x, y, z, w]` quaternions;
/// robust to the double cover (q and -q are the same rotation, angle 0).
pub fn quat_angle(a: [f64; 4], b: [f64; 4]) -> f64 {
    let qa = UnitQuaternion::from_quaternion(Quaternion::new(a[3], a[0], a[1], a[2]));
    let qb = UnitQuaternion::from_quaternion(Quaternion::new(b[3], b[0], b[1], b[2]));
    qa.angle_to(&qb)
}

/// How far each world axis can be driven: `[[min, max]; 3]` (x, y, z) over every
/// in-limit configuration, so each bound is the best case for that axis with the
/// other two free. Runs once per side at construction to size the panel's position
/// sliders per generation rather than hardcoding them.
///
/// A grid coarse enough to afford (`N^7` poses) lands near the extremes but not on
/// them, so each of the six is refined by projected gradient ascent: the Jacobian's
/// linear rows are exactly the gradient of position with respect to the joints, and
/// the search space is a box, so stepping along the gradient and clamping converges
/// on the true extreme. The grid supplies the start, which keeps the ascent out of
/// the local maxima a single arbitrary start would find.
///
/// These bound reach, they do not map it: a target inside the box needs all three
/// axes together, so the backbone's IK can still refuse one.
fn workspace_aabb(arm: &mut Arm) -> [[f64; 2]; 3] {
    const N: usize = 3;
    let lims = arm.limits();
    let ranges: [(f64, f64); ARM_DOF] = std::array::from_fn(|i| (lims[i].lo, lims[i].hi));
    // Best grid pose per (axis, direction), as the ascent's start.
    let mut best = [[(f64::NEG_INFINITY, [0.0; ARM_DOF]); 2]; 3];
    for idx in 0..N.pow(ARM_DOF as u32) {
        let mut rem = idx;
        let q: [f64; ARM_DOF] = std::array::from_fn(|j| {
            let step = rem % N;
            rem /= N;
            let t = step as f64 / (N - 1) as f64;
            ranges[j].0 + t * (ranges[j].1 - ranges[j].0)
        });
        let base = arm.at(&q).ee_pose();
        let p = arm.world_pose(&base).translation.vector;
        for k in 0..3 {
            for (d, sign) in [1.0_f64, -1.0].into_iter().enumerate() {
                if sign * p[k] > best[k][d].0 {
                    best[k][d] = (sign * p[k], q);
                }
            }
        }
    }
    std::array::from_fn(|k| {
        let hi = ascend(arm, &lims, best[k][0].1, k, 1.0);
        let lo = ascend(arm, &lims, best[k][1].1, k, -1.0);
        [lo, hi]
    })
}

/// Push `q` along the joints that move world axis `axis` in `sign`'s direction,
/// clamped into `limits`, and return the extreme reached. Backtracks the step
/// whenever it overshoots, so it converges rather than oscillating at a bound.
fn ascend(
    arm: &mut Arm,
    limits: &[srs_model::Limit; ARM_DOF],
    start: [f64; ARM_DOF],
    axis: usize,
    sign: f64,
) -> f64 {
    // A step this small moves the end-effector well under a millimetre, which is
    // finer than the slider it sizes; the floor ends the walk.
    const STEP_FLOOR_RAD: f64 = 1e-4;
    const MAX_ITERS: usize = 200;
    let world_from_base = arm.base_from_world().inverse().rotation;
    let mut q = start;
    let mut step = 0.2_f64;
    let value = |arm: &mut Arm, q: &[f64; ARM_DOF]| {
        let base = arm.at(q).ee_pose();
        sign * arm.world_pose(&base).translation.vector[axis]
    };
    let mut current = value(arm, &q);
    for _ in 0..MAX_ITERS {
        if step < STEP_FLOOR_RAD {
            break;
        }
        let jacobian = arm.at(&q).jacobian();
        let gradient: [f64; ARM_DOF] = std::array::from_fn(|j| {
            let column = jacobian.column(j);
            let linear = Vector3::new(column[0], column[1], column[2]);
            sign * (world_from_base * linear)[axis]
        });
        let norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if norm < f64::EPSILON {
            break;
        }
        let candidate: [f64; ARM_DOF] = std::array::from_fn(|j| {
            (q[j] + step * gradient[j] / norm).clamp(limits[j].lo, limits[j].hi)
        });
        let next = value(arm, &candidate);
        if next > current {
            (q, current) = (candidate, next);
        } else {
            step *= 0.5;
        }
    }
    sign * current
}

#[cfg(test)]
mod tests {
    use super::*;

    // Representative joint-jog acceleration for the tests (production value is a param).
    const JOG_A: f64 = 10.0;

    fn models() -> ArmModels {
        ArmModels::from_version(HardwareVersion::V2)
    }

    // The caps a 100 Hz tick at the sim launcher's 0.5 m/s knob derives.
    fn test_caps() -> JogCaps {
        JogCaps::per_tick(0.01, 0.5, JOG_A)
    }

    fn cart(mode: JogMode, desired: Pose, arm_angle: f64) -> CartesianJog {
        CartesianJog {
            mode,
            desired,
            arm_angle,
        }
    }

    #[test]
    fn no_reachable_pose_falls_outside_the_bounds() {
        // The bound the panel must not get wrong is the tight one: a slider that
        // cannot dial a pose the arm can reach strands the operator. Sampling
        // cannot prove the extremes are attained (the ascent attains them by
        // construction, from a real in-limit configuration), but it does prove
        // nothing reachable sits outside them. Walks a golden-ratio lattice rather
        // than the grid the bounds were built from, so the poses are ones the
        // construction never saw.
        const GOLDEN: [f64; ARM_DOF] = [
            0.618_033_988_75,
            0.381_966_011_25,
            0.236_067_977_5,
            0.145_898_033_75,
            0.090_169_943_75,
            0.055_728_090_0,
            0.034_441_853_75,
        ];
        let m = models();
        for side in [Side::Left, Side::Right] {
            let bounds = m.pos_bounds(side);
            let limits = HardwareVersion::V2.joint_limits(side.description());
            for step in 0..5000 {
                let q: [f64; ARM_DOF] = std::array::from_fn(|i| {
                    let t = (GOLDEN[i] * (step + 1) as f64).fract();
                    limits[i][0] + t * (limits[i][1] - limits[i][0])
                });
                let p = m.ee_pose_world(side, &q);
                for k in 0..3 {
                    assert!(
                        p[k] >= bounds[k][0] && p[k] <= bounds[k][1],
                        "{side:?} axis {k}: reachable {} outside {:?}",
                        p[k],
                        bounds[k]
                    );
                }
            }
        }
    }

    #[test]
    fn workspace_bounds_are_sane_and_contain_home() {
        let m = models();
        for side in [Side::Left, Side::Right] {
            let b = m.pos_bounds(side);
            for [lo, hi] in b {
                assert!(lo < hi, "bound [{lo}, {hi}] must be non-empty");
            }
            // A known-reachable pose (home: zeros, elbow off its floor) sits inside.
            let home = m.ee_pose_world(side, &[0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]);
            for k in 0..3 {
                assert!(
                    b[k][0] <= home[k] && home[k] <= b[k][1],
                    "home axis {k} = {} outside bounds {:?}",
                    home[k],
                    b[k]
                );
            }
        }
    }

    fn geodesic(a: &Pose, b: &Pose) -> f64 {
        let q_a = UnitQuaternion::from_euler_angles(a[3], a[4], a[5]);
        let q_b = UnitQuaternion::from_euler_angles(b[3], b[4], b[5]);
        (q_a * q_b.inverse()).angle()
    }

    #[test]
    fn caps_scale_with_rate_and_speed() {
        // Same speed at 100 Hz vs 500 Hz: a 5x smaller step, the identical velocity.
        let hz100 = JogCaps::per_tick(0.01, 0.5, JOG_A);
        let hz500 = JogCaps::per_tick(0.002, 0.5, JOG_A);
        assert!((hz100.pos_step_m - 0.005).abs() < 1e-12);
        assert!((hz500.pos_step_m - 0.001).abs() < 1e-12);
        assert!((hz100.pos_step_m / 0.01 - hz500.pos_step_m / 0.002).abs() < 1e-12);
        // Retuning the operator knob rescales the step 1:1.
        assert!((JogCaps::per_tick(0.01, 0.25, JOG_A).pos_step_m - 0.0025).abs() < 1e-12);
        // The angular rate is the fixed jog constant, likewise per-tick.
        assert!((hz100.rot_step_rad - JOG_ROT_RATE_RAD_S * 0.01).abs() < 1e-12);
        // The joint-jog acceleration is the fixed SI rate, independent of dt.
        assert_eq!(hz100.joint_accel_rad_s2, JOG_A);
        assert_eq!(hz500.joint_accel_rad_s2, JOG_A);
    }

    #[test]
    fn joint_jog_tick_accelerates_from_rest_within_the_cap() {
        let caps = JogCaps::per_tick(0.01, 0.5, JOG_A);
        let current = [0.0; ARM_DOF];
        // A far target so every joint keeps accelerating.
        let target = [3.0; ARM_DOF];
        // From rest, one tick may add at most a_max*dt to the velocity, so the position
        // step is bounded by that: a small ramp, not a snap.
        let JointJogStep::Stepped { joints, vel } =
            joint_jog_tick(&current, &target, &current, caps)
        else {
            panic!("a far target does not converge in one tick");
        };
        for i in 0..ARM_DOF {
            assert!(
                vel[i] <= caps.joint_accel_rad_s2 * caps.dt_s + 1e-12,
                "accel-capped"
            );
            assert!(
                joints[i] > 0.0 && joints[i] < 0.05,
                "a small first step, not a snap"
            );
        }
    }

    #[test]
    fn joint_jog_tick_settles_on_the_target_without_overshoot() {
        let caps = JogCaps::per_tick(0.01, 0.5, JOG_A);
        let target = [0.4, -0.3, 0.2, 0.9, -0.5, 0.6, -0.2];
        let mut q = [0.0; ARM_DOF];
        let mut v = [0.0; ARM_DOF];
        let mut max_overshoot: f64 = 0.0;
        let mut converged = false;
        for _ in 0..5000 {
            match joint_jog_tick(&q, &target, &v, caps) {
                JointJogStep::Stepped { joints, vel } => {
                    for i in 0..ARM_DOF {
                        // Never crosses past the target (monotone approach, no overshoot).
                        let past = (joints[i] - target[i]) * target[i].signum();
                        max_overshoot = max_overshoot.max(past);
                    }
                    q = joints;
                    v = vel;
                }
                JointJogStep::Converged(joints) => {
                    assert_eq!(joints, target, "lands exactly on the target");
                    converged = true;
                    break;
                }
            }
        }
        assert!(
            converged,
            "the jog converges within a bounded number of ticks"
        );
        assert!(
            max_overshoot < 1e-6,
            "the ramp does not overshoot the target"
        );
    }

    #[test]
    fn joint_jog_tick_reversal_stays_within_the_acceleration_cap() {
        let caps = JogCaps::per_tick(0.01, 0.5, JOG_A);
        let dv_max = caps.joint_accel_rad_s2 * caps.dt_s;
        let up = [3.0; ARM_DOF];
        let mut q = [0.0; ARM_DOF];
        let mut v = [0.0; ARM_DOF];
        // Build up to cruise speed toward a far target.
        for _ in 0..20 {
            let JointJogStep::Stepped { joints, vel } = joint_jog_tick(&q, &up, &v, caps) else {
                unreachable!("a far target does not converge yet")
            };
            q = joints;
            v = vel;
        }
        assert!(v[0] > 2.0, "the joint is moving fast before the reversal");
        // Reverse mid-drag: re-target just behind the current position while still moving
        // forward. The old magnitude-only capture snapped backward and zeroed the velocity
        // here; the fix must decelerate under the cap instead.
        let down: [f64; ARM_DOF] = std::array::from_fn(|i| q[i] - 0.01);
        let mut worst_accel: f64 = 0.0;
        let mut max_overshoot: f64 = 0.0;
        let mut converged = false;
        for _ in 0..5000 {
            match joint_jog_tick(&q, &down, &v, caps) {
                JointJogStep::Stepped { joints, vel } => {
                    for i in 0..ARM_DOF {
                        worst_accel = worst_accel.max((vel[i] - v[i]).abs());
                        // Overshoot = dropping below the reversed target.
                        max_overshoot = max_overshoot.max(down[i] - joints[i]);
                    }
                    q = joints;
                    v = vel;
                }
                JointJogStep::Converged(joints) => {
                    assert_eq!(joints, down, "settles on the reversed target");
                    converged = true;
                    break;
                }
            }
        }
        assert!(converged, "the reversed jog converges");
        assert!(
            worst_accel <= dv_max + 1e-12,
            "velocity change per tick stays within the accel cap through the reversal ({worst_accel} > {dv_max})"
        );
        assert!(
            max_overshoot < 1e-6,
            "the reversal does not overshoot the target"
        );
    }

    #[test]
    fn velocity_limits_load_from_the_urdf() {
        let m = models();
        for side in [Side::Left, Side::Right] {
            let v = m.velocity_limits(side);
            assert!(v.iter().all(|x| x.is_finite() && *x > 0.0));
            // j3/j4 are the slow pair in both generations' datasheets; sanity-pin
            // that we read per-joint values, not one shared number.
            assert!(
                v[2] < v[0],
                "j3 ({}) should be slower than j1 ({})",
                v[2],
                v[0]
            );
        }
    }

    #[test]
    fn steps_never_exceed_joint_velocity_budgets() {
        // Demand an aggressive 5 cm/tick step; the returned joint delta must stay
        // inside every joint's URDF velocity budget for the tick.
        let m = models();
        let q0 = [0.3, 0.1, 0.2, 0.8, 0.3, 0.2, 0.15];
        let p0 = m.ee_pose_world(Side::Left, &q0);
        let caps = JogCaps {
            pos_step_m: 0.05,
            rot_step_rad: 0.2,
            arm_angle_step_rad: 0.2,
            joint_accel_rad_s2: JOG_A,
            dt_s: 0.01,
        };
        let desired = [p0[0] + 1.0, p0[1], p0[2], p0[3], p0[4], p0[5]];
        if let JogStep::Stepped(q) = jog_tick(
            &m,
            Side::Left,
            &q0,
            &cart(JogMode::Position, desired, 0.0),
            caps,
        ) {
            let budget = m.velocity_limits(Side::Left);
            for i in 0..ARM_DOF {
                let v = (q[i] - q0[i]).abs() / caps.dt_s;
                assert!(
                    v <= budget[i] * 1.0001,
                    "joint {i} at {v:.2} rad/s exceeds its {:.2} rad/s budget",
                    budget[i]
                );
            }
        } else {
            panic!("aggressive but reachable step must advance");
        }
    }

    #[test]
    fn position_jog_from_home_reaches_far_and_holds_orientation() {
        // The original from-home x drag scenario: the jog must walk deep into the
        // workspace with orientation softly held, then hold at the envelope.
        let m = models();
        let start = [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0];
        let p0 = m.ee_pose_world(Side::Left, &start);
        let desired = [p0[0] + 0.3, p0[1], p0[2], p0[3], p0[4], p0[5]];
        let mut q = start;
        for _ in 0..3000 {
            match jog_tick(
                &m,
                Side::Left,
                &q,
                &cart(JogMode::Position, desired, 0.0),
                test_caps(),
            ) {
                JogStep::Stepped(next) => q = next,
                JogStep::Blocked | JogStep::Converged => break,
            }
        }
        let p = m.ee_pose_world(Side::Left, &q);
        assert!(
            p[0] - p0[0] > 0.15,
            "must cover real distance, got {:.4}",
            p[0] - p0[0]
        );
        assert!(
            geodesic(&p, &p0) < 0.2,
            "orientation must hold softly, drifted {:.4} rad",
            geodesic(&p, &p0)
        );
    }

    #[test]
    fn orientation_jog_turns_the_hand_while_position_holds() {
        let m = models();
        let q0 = [0.3, 0.1, 0.2, 0.8, 0.3, 0.2, 0.15];
        let p0 = m.ee_pose_world(Side::Left, &q0);
        let desired = [p0[0], p0[1], p0[2], p0[3], p0[4], p0[5] + 0.3];
        let mut q = q0;
        for _ in 0..200 {
            match jog_tick(
                &m,
                Side::Left,
                &q,
                &cart(JogMode::Orientation, desired, 0.0),
                test_caps(),
            ) {
                JogStep::Stepped(next) => q = next,
                JogStep::Converged | JogStep::Blocked => break,
            }
        }
        let p = m.ee_pose_world(Side::Left, &q);
        assert!(geodesic(&p, &p0) > 0.15, "hand must actually turn");
        let drift =
            ((p[0] - p0[0]).powi(2) + (p[1] - p0[1]).powi(2) + (p[2] - p0[2]).powi(2)).sqrt();
        assert!(
            drift < 0.05,
            "position drift under an orientation jog stays small, got {drift:.4}"
        );
    }

    #[test]
    fn jog_converges_on_the_current_pose() {
        let m = models();
        let q = [0.3, 0.1, 0.2, 0.8, 0.3, 0.2, 0.15];
        let p = m.ee_pose_world(Side::Left, &q);
        for mode in [JogMode::Position, JogMode::Orientation] {
            assert!(matches!(
                jog_tick(&m, Side::Left, &q, &cart(mode, p, 0.0), test_caps()),
                JogStep::Converged
            ));
        }
    }

    #[test]
    fn arm_angle_jog_swivels_the_elbow_holding_the_ee_pose() {
        // Driving the arm angle moves psi toward the target while the end-effector
        // pose stays put: a pure null-space move (elbow swivels, hand holds).
        let m = models();
        let q0 = [0.3, 0.1, 0.2, 0.8, 0.3, 0.2, 0.15];
        let p0 = m.ee_pose_world(Side::Left, &q0);
        let psi0 = m
            .arm_angle(Side::Left, &q0)
            .expect("arm angle is defined off the straight-arm floor");
        let target_psi = psi0 + 0.3;
        let mut q = q0;
        for _ in 0..500 {
            match jog_tick(
                &m,
                Side::Left,
                &q,
                &cart(JogMode::ArmAngle, p0, target_psi),
                test_caps(),
            ) {
                JogStep::Stepped(next) => q = next,
                JogStep::Converged | JogStep::Blocked => break,
            }
        }
        let psi = m.arm_angle(Side::Left, &q).unwrap();
        let p = m.ee_pose_world(Side::Left, &q);
        assert!(
            (psi - target_psi).abs() < 0.05,
            "psi should reach the target, got {psi:.3} vs {target_psi:.3}"
        );
        let pos_drift =
            ((p[0] - p0[0]).powi(2) + (p[1] - p0[1]).powi(2) + (p[2] - p0[2]).powi(2)).sqrt();
        assert!(
            pos_drift < 0.01,
            "EE position must hold under an arm-angle jog, drifted {pos_drift:.4}"
        );
        assert!(
            geodesic(&p, &p0) < 0.05,
            "EE orientation must hold, drifted {:.4} rad",
            geodesic(&p, &p0)
        );
    }

    #[test]
    fn jog_blocks_at_the_envelope_and_never_converges_short() {
        // Far past reach: the jog must end Blocked (never Converged) and every
        // step along the way must be velocity-bounded.
        let m = models();
        let mut q = [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0];
        let p0 = m.ee_pose_world(Side::Left, &q);
        let desired = [p0[0] + 2.0, p0[1], p0[2], p0[3], p0[4], p0[5]];
        let budget = m.velocity_limits(Side::Left);
        for _ in 0..5000 {
            match jog_tick(
                &m,
                Side::Left,
                &q,
                &cart(JogMode::Position, desired, 0.0),
                test_caps(),
            ) {
                JogStep::Stepped(next) => {
                    for i in 0..ARM_DOF {
                        assert!((next[i] - q[i]).abs() / 0.01 <= budget[i] * 1.0001);
                    }
                    q = next;
                }
                JogStep::Blocked => return,
                JogStep::Converged => panic!("must not converge on an unreachable pose"),
            }
        }
        panic!("jog neither blocked nor converged within 5000 ticks");
    }
}
