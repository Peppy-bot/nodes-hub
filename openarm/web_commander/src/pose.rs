//! The arm models behind the panel: forward kinematics for the pose readouts,
//! inverse kinematics for the Actions-mode pose preview and the gesture bake,
//! the reachable workspace bounds the position sliders are sized to, and the
//! joint jog ramp a streamed setpoint walks toward its slider under. The joint
//! vector stays the single source of truth: a pose is a display and composition
//! lens, never stored state, and a pose move is the backbone's move_arm action.
//!
//! Poses are in the world frame (matching the backbone's move_arm action) and
//! orientation is exposed as intrinsic-XYZ roll/pitch/yaw via nalgebra's
//! `euler_angles` round-trip for display; the arcball composes orientation on
//! quaternions, so no euler component is ever interpolated.

use std::sync::{Arc, Mutex};

use control_core::sync::lock_ignoring_poison;
use openarm_description::{ARM_DOF, ArmPair, HardwareVersion, Side};
use srs_model::JointVec;
use srs_model::nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use srs_model::{Arm, ArmAnglePolicy};

/// A world-frame end-effector pose as `[x, y, z, roll, pitch, yaw]` (metres,
/// radians): the form the panel edits and displays.
pub type Pose = [f64; 6];

const DEG: f64 = std::f64::consts::PI / 180.0;

/// The slack the panel asks the backbone to plan a pose move within, sent as
/// `move_arm`'s `plan_*` tolerances; a goal it cannot plan this closely is refused.
pub const PLAN_POS_TOL_M: f64 = 0.005;
pub const PLAN_ANGLE_TOL_RAD: f64 = 5.0 * DEG;

/// The two per-side arm models (FK/IK/limits) behind mutexes, plus each
/// side's URDF joint velocity limits.
///
/// Each lock is held only for a single synchronous kinematics call and never
/// across an `.await`. The owner task is the sole caller, so the locks are
/// uncontended; the mutex only provides the interior mutability an `Arm` needs
/// (`at` takes `&mut self`) behind `ArmModels`'s shared `&self` methods.
#[derive(Clone)]
pub struct ArmModels {
    arms: ArmPair<Arc<Mutex<Arm>>>,
    velocity_limits: ArmPair<JointVec>,
    // World-frame x/y/z reachable bounds per side, from the FK envelope (computed once
    // at construction); the panel sizes its position sliders to these.
    bounds: ArmPair<[[f64; 2]; 3]>,
}

impl ArmModels {
    /// Build both arm models from the generation's embedded description, mirroring
    /// the backbone's `arm_model` (same URDF, same elbow singularity floor) so the panel
    /// solves against the identical chain the backbone governs. Joint velocity limits
    /// come from the same URDF: the budget the gesture bake asserts its trajectories
    /// against.
    pub fn from_version(version: HardwareVersion) -> Self {
        let mut arms = ArmPair::from_fn(|side| build_arm(version, side));
        let bounds = ArmPair::from_fn(|side| workspace_aabb(&mut arms[side]));
        Self {
            velocity_limits: ArmPair::from_fn(|side| velocity_limits(version.urdf(), side)),
            arms: arms.map(|_, arm| Arc::new(Mutex::new(arm))),
            bounds,
        }
    }

    /// World-frame end-effector pose of `joints` for `side`.
    pub fn ee_pose_world(&self, side: Side, joints: &JointVec) -> Pose {
        let model = lock_ignoring_poison(self.get(side));
        let base = model.at(joints).ee_pose();
        decompose(&model.world_pose(&base))
    }

    /// World-frame end-effector orientation of `joints` as a quaternion `[x, y, z, w]`,
    /// for the panel's arcball (which composes orientation on quaternions, never euler).
    pub fn ee_quat_world(&self, side: Side, joints: &JointVec) -> [f64; 4] {
        let model = lock_ignoring_poison(self.get(side));
        let base = model.at(joints).ee_pose();
        let q = model.world_pose(&base).rotation;
        [q.i, q.j, q.k, q.w]
    }

    /// Solve inverse kinematics for a world-frame end-effector pose (position in
    /// metres, orientation as a unit quaternion), seeded from `seed` so the branch
    /// nearest the current configuration is chosen. `None` when the pose is
    /// unreachable or admits no in-limit solution. Used to preview an Actions-mode
    /// pose move as joints and to bake the gestures' Cartesian curves; the
    /// backbone re-solves and plans a pose move itself.
    pub fn solve_ik(
        &self,
        side: Side,
        position: [f64; 3],
        rotation: UnitQuaternion<f64>,
        seed: &JointVec,
    ) -> Option<JointVec> {
        let model = lock_ignoring_poison(self.get(side));
        let world = Isometry3::from_parts(
            Translation3::new(position[0], position[1], position[2]),
            rotation,
        );
        let base_target = model.base_pose(&world);
        model
            .solve_ik(&base_target, ArmAnglePolicy::FromSeed, seed)
            .map(|s| s.q)
    }

    fn get(&self, side: Side) -> &Arc<Mutex<Arm>> {
        &self.arms[side]
    }

    /// Per-joint URDF velocity limits (rad/s) for `side`: the budget the backbone's
    /// chase enforces, which the gesture bake asserts its trajectories against.
    pub(crate) fn velocity_limits(&self, side: Side) -> &JointVec {
        &self.velocity_limits[side]
    }

    /// Per-joint [lo, hi] position limits (rad) for `side`, from the same model the
    /// IK solves against (elbow singularity floor included).
    pub(crate) fn joint_ranges(&self, side: Side) -> [[f64; 2]; ARM_DOF] {
        let model = lock_ignoring_poison(self.get(side));
        let lims = model.limits();
        std::array::from_fn(|i| [lims[i].lo, lims[i].hi])
    }

    /// World-frame x/y/z reachable bounds `[[min, max]; 3]` for `side`, so the panel
    /// sizes its position sliders to the arm's actual reach (correct per generation).
    pub fn pos_bounds(&self, side: Side) -> [[f64; 2]; 3] {
        self.bounds[side]
    }
}

/// A joint jog is converged once every joint is within this of the target and nearly
/// stopped; the jog then lands exactly on the target and retires.
const JOINT_JOG_CONVERGED_RAD: f64 = 1e-4;
const JOINT_JOG_STOP_RAD_S: f64 = 1e-3;

/// The rates a joint jog ramps under.
#[derive(Clone, Copy, Debug)]
pub struct JogCaps {
    /// Joint-jog acceleration limit (rad/s^2): the jog ramps up under it and brakes
    /// to rest by the target.
    pub joint_accel_rad_s2: f64,
    /// Joint-jog speed limit (rad/s): the approach speed of every joint is held at or
    /// under it, so a far slider target closes at this speed, not at the speed the
    /// braking distance alone would allow.
    pub joint_velocity_rad_s: f64,
}

/// The operator's active drive for one arm, reconciled toward once per stream tick:
/// the setpoint walks toward the slider `target` one acceleration-limited step per
/// tick (carrying `vel`, the per-joint jog velocity, across ticks), so the arm ramps
/// smoothly. `target` refreshes as the operator drags; `vel` is
/// preserved across those refreshes so a continuous drag keeps its momentum.
#[derive(Clone, Copy, Debug)]
pub struct JointJog {
    pub target: JointVec,
    pub vel: JointVec,
}

/// One joint-jog tick's outcome. Targets are pre-clamped to joint limits by the
/// caller, so a joint jog never blocks.
pub enum JointJogStep {
    /// Every joint arrived and stopped; the setpoint lands exactly on the target and the
    /// jog retires.
    Converged(JointVec),
    /// Advanced one acceleration-limited step: the new joint setpoint to stream and the
    /// carried per-joint velocity for the next tick.
    Stepped { joints: JointVec, vel: JointVec },
}

/// Advance a joint jog one command tick of `dt_s` seconds: walk each joint of `current`
/// toward `target` under an acceleration limit (`caps.joint_accel_rad_s2`) and a speed
/// limit (`caps.joint_velocity_rad_s`), carrying `vel` so the velocity stays continuous. The
/// approach speed is the lower of the speed limit and the speed the acceleration limit
/// can brake to rest from by the target, and the position is clamped there so it never
/// overshoots; the carried velocity then bleeds to rest over the next few ticks under
/// the same cap.
pub fn joint_jog_tick(
    current: &JointVec,
    target: &JointVec,
    vel: &JointVec,
    caps: JogCaps,
    dt_s: f64,
) -> JointJogStep {
    let a_max = caps.joint_accel_rad_s2;
    let dv_max = a_max * dt_s;

    let next: [(f64, f64); ARM_DOF] = std::array::from_fn(|i| {
        let error = target[i] - current[i];
        // Approach speed: what a_max can brake to rest from by the target (no overshoot),
        // held under the speed limit.
        let v_brake = (2.0 * a_max * error.abs()).sqrt();
        let v_desired = error.signum() * v_brake.min(caps.joint_velocity_rad_s);
        // |v_next - vel| <= dv_max, so the acceleration bound holds on every tick,
        // including the last: the velocity is never teleported to zero.
        let v_next = vel[i] + (v_desired - vel[i]).clamp(-dv_max, dv_max);
        let step = v_next * dt_s;
        // Land on the target only when this step is directed at it and would reach or
        // cross it (`step*error >= error^2`), clamping the position there so it never
        // overshoots. A mid-drag reversal (velocity still opposing `error`) fails this
        // test, so the joint decelerates and turns around.
        let joint = if step * error >= error * error {
            target[i]
        } else {
            current[i] + step
        };
        (joint, v_next)
    });
    let joints: JointVec = std::array::from_fn(|i| next[i].0);
    let vel: JointVec = std::array::from_fn(|i| next[i].1);

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

/// Build one arm model from the generation's embedded description, with the elbow
/// singularity floor applied and the URDF's tcp frame mounted (mirrors
/// `openarm_backbone`'s `arm_model`, so the panel's poses and the backbone's are the
/// same frame). A bad base link or tcp link aborts bringup, matching how the
/// backbone fails.
fn build_arm(version: HardwareVersion, side: Side) -> Arm {
    let base = version.base_link(side);
    Arm::from_urdf(version.urdf(), base)
        .unwrap_or_else(|e| panic!("build {side} arm model from base '{base}': {e}"))
        .with_lower_floor(
            version.elbow_joint_index(),
            version.elbow_singularity_floor_rad(),
        )
        .with_tool_link(version.tcp_link(side))
        .unwrap_or_else(|e| panic!("mount the {side} gripper tcp frame: {e}"))
}

/// Per-joint velocity limits (rad/s) for `side`, j1..j7, from the bundled URDF:
/// the same numbers the backbone's chase enforces, so the gesture bake and the
/// backbone's follow capability agree by construction.
fn velocity_limits(urdf: &str, side: Side) -> JointVec {
    let robot = urdf_rs::read_from_string(urdf).expect("bundled URDF must parse");
    std::array::from_fn(|i| {
        let name = format!("openarm_{side}_joint{}", i + 1);
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
        let q: JointVec = std::array::from_fn(|j| {
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
    start: JointVec,
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
    let value = |arm: &mut Arm, q: &JointVec| {
        let base = arm.at(q).ee_pose();
        sign * arm.world_pose(&base).translation.vector[axis]
    };
    let mut current = value(arm, &q);
    for _ in 0..MAX_ITERS {
        if step < STEP_FLOOR_RAD {
            break;
        }
        let jacobian = arm.at(&q).jacobian();
        let gradient: JointVec = std::array::from_fn(|j| {
            let column = jacobian.column(j);
            let linear = Vector3::new(column[0], column[1], column[2]);
            sign * (world_from_base * linear)[axis]
        });
        let norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if norm < f64::EPSILON {
            break;
        }
        let candidate: JointVec = std::array::from_fn(|j| {
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
    use crate::test_support::{caps, models};

    /// A 100 Hz command tick.
    const DT_S: f64 = 0.01;

    #[test]
    fn no_reachable_pose_falls_outside_the_bounds() {
        // The bound the panel must not get wrong is the tight one: a slider that
        // cannot dial a pose the arm can reach strands the operator. Sampling
        // cannot prove the extremes are attained (the ascent attains them by
        // construction, from a real in-limit configuration), but it does prove
        // nothing reachable sits outside them. Walks a golden-ratio lattice rather
        // than the grid the bounds were built from, so the poses are ones the
        // construction never saw.
        const GOLDEN: JointVec = [
            0.618_033_988_75,
            0.381_966_011_25,
            0.236_067_977_5,
            0.145_898_033_75,
            0.090_169_943_75,
            0.055_728_090_0,
            0.034_441_853_75,
        ];
        let m = models();
        for side in Side::BOTH {
            let bounds = m.pos_bounds(side);
            let limits = HardwareVersion::V2.joint_limits(side);
            for step in 0..5000 {
                let q: JointVec = std::array::from_fn(|i| {
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
        for side in Side::BOTH {
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

    #[test]
    fn solve_ik_re_solves_the_pose_of_a_configuration_near_its_seed() {
        // The Actions-mode preview and the gesture bake seed the solve from the
        // configuration they stand at, so the pose of that configuration solves
        // back to it.
        let m = models();
        let q = [0.3, 0.1, 0.2, 0.8, 0.3, 0.2, 0.15];
        let p = m.ee_pose_world(Side::Left, &q);
        let rotation = UnitQuaternion::from_euler_angles(p[3], p[4], p[5]);
        let solved = m
            .solve_ik(Side::Left, [p[0], p[1], p[2]], rotation, &q)
            .expect("a pose the arm stands at has a solution");
        for i in 0..ARM_DOF {
            assert!(
                (solved[i] - q[i]).abs() < 1e-3,
                "joint {i}: solved {} for seed {}",
                solved[i],
                q[i]
            );
        }
        assert!(
            m.solve_ik(Side::Left, [p[0] + 2.0, p[1], p[2]], rotation, &q)
                .is_none(),
            "two metres out of reach has no solution"
        );
    }

    #[test]
    fn joint_jog_tick_holds_a_far_approach_at_the_speed_limit() {
        let caps = caps();
        let mut current = [0.0; ARM_DOF];
        let mut vel = [0.0; ARM_DOF];
        // A target far enough that braking alone would allow sqrt(2 * 10 * 3) = 7.7 rad/s.
        let target = [3.0; ARM_DOF];
        let mut peak = 0.0f64;
        for _ in 0..200 {
            match joint_jog_tick(&current, &target, &vel, caps, DT_S) {
                JointJogStep::Stepped { joints, vel: v } => {
                    current = joints;
                    vel = v;
                }
                JointJogStep::Converged(_) => panic!("3 rad is not closed in 2 s at 1 rad/s"),
            }
            peak = peak.max(vel.iter().cloned().fold(0.0, f64::max));
        }
        assert!(
            peak <= caps.joint_velocity_rad_s + 1e-12,
            "speed held at the limit: {peak}"
        );
        assert!(
            peak > 0.99 * caps.joint_velocity_rad_s,
            "the limit is reached, not hovered under: {peak}"
        );
        // Two seconds at the limit after a 0.1 s ramp: 2 rad minus the ramp's shortfall.
        assert!(
            (1.9..2.0).contains(&current[0]),
            "travel at the limit: {}",
            current[0]
        );
    }

    #[test]
    fn joint_jog_tick_accelerates_from_rest_within_the_cap() {
        let caps = caps();
        let current = [0.0; ARM_DOF];
        // A far target so every joint keeps accelerating.
        let target = [3.0; ARM_DOF];
        // From rest, one tick may add at most a_max*dt to the velocity, so the position
        // step is bounded by that: a small ramp, not a snap.
        let JointJogStep::Stepped { joints, vel } =
            joint_jog_tick(&current, &target, &current, caps, DT_S)
        else {
            panic!("a far target does not converge in one tick");
        };
        for i in 0..ARM_DOF {
            assert!(
                vel[i] <= caps.joint_accel_rad_s2 * DT_S + 1e-12,
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
        let caps = caps();
        let target = [0.4, -0.3, 0.2, 0.9, -0.5, 0.6, -0.2];
        let mut q = [0.0; ARM_DOF];
        let mut v = [0.0; ARM_DOF];
        let mut max_overshoot: f64 = 0.0;
        let mut converged = false;
        for _ in 0..5000 {
            match joint_jog_tick(&q, &target, &v, caps, DT_S) {
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
        let caps = caps();
        let dv_max = caps.joint_accel_rad_s2 * DT_S;
        let up = [3.0; ARM_DOF];
        let mut q = [0.0; ARM_DOF];
        let mut v = [0.0; ARM_DOF];
        // Build up to cruise speed toward a far target.
        for _ in 0..20 {
            let JointJogStep::Stepped { joints, vel } = joint_jog_tick(&q, &up, &v, caps, DT_S)
            else {
                unreachable!("a far target does not converge yet")
            };
            q = joints;
            v = vel;
        }
        assert!(
            v[0] > 0.99 * caps.joint_velocity_rad_s,
            "the joint is at cruise speed before the reversal"
        );
        // Reverse mid-drag: re-target just behind the current position while still moving
        // forward. The joint must decelerate under the cap and turn around.
        let down: JointVec = std::array::from_fn(|i| q[i] - 0.01);
        let mut worst_accel: f64 = 0.0;
        let mut max_overshoot: f64 = 0.0;
        let mut converged = false;
        for _ in 0..5000 {
            match joint_jog_tick(&q, &down, &v, caps, DT_S) {
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
        for side in Side::BOTH {
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
}
