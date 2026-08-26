use std::time::{Duration, Instant};

use control_core::minimum_jerk::{quintic, velocity_limited_duration};
use srs_model::nalgebra::{Isometry3, Translation3};
use srs_model::{Arm, ArmAnglePolicy};

use crate::servo::EeCaps;
use crate::types::{ARM_DOF, JointVec};

/// Quintic minimum-jerk trajectory in joint space.
pub struct JointTrajectory {
    start: JointVec,
    end: JointVec,
    duration: Duration,
    pub motion_start: Instant,
}

impl JointTrajectory {
    pub fn new(
        start: JointVec,
        end: JointVec,
        max_velocity_rad_s: JointVec,
        requested_duration_secs: f64,
    ) -> Self {
        // Each joint moves Δq_i over the whole blend (Δs = 1), so its velocity
        // ratio is |Δq_i| / v_max_i; the slowest joint relative to its limit binds.
        // Mirrors ROS2 TOTG behaviour against per-joint URDF velocity limits.
        let peak_ratio = start
            .iter()
            .zip(end.iter())
            .zip(max_velocity_rad_s.iter())
            .map(|((s, e), v)| (e - s).abs() / v)
            .fold(0.0_f64, f64::max);
        // Both inputs are guaranteed upstream: the joint velocity limits are
        // asserted finite and positive at boot, and a move's requested duration
        // is refused at the action boundary unless it is finite and in range.
        let secs = velocity_limited_duration(peak_ratio, requested_duration_secs)
            .expect("velocity limits and requested duration are validated before planning");
        Self {
            start,
            end,
            duration: Duration::from_secs_f64(secs),
            motion_start: Instant::now(),
        }
    }

    /// Joint positions at time `now`, on the quintic blend from `start` to `end`.
    /// Holds at `end` once complete (and before `motion_start`, at `start`).
    pub fn sample(&self, now: Instant) -> JointVec {
        let t_total = self.duration.as_secs_f64();
        // Degenerate trajectory (start == end and requested_duration_secs == 0): hold at end.
        if t_total == 0.0 {
            return self.end;
        }
        let elapsed = now.duration_since(self.motion_start).as_secs_f64();
        let tau = (elapsed / t_total).clamp(0.0, 1.0);
        let (s, _) = quintic(tau);
        std::array::from_fn(|i| self.start[i] + (self.end[i] - self.start[i]) * s)
    }

    pub fn is_complete(&self, now: Instant) -> bool {
        now.duration_since(self.motion_start) >= self.duration
    }
}

/// Quintic minimum-jerk trajectory between two end-effector poses (world frame).
/// Position rides the quintic blend; orientation is slerped at the same blend
/// parameter, so position and orientation reach the goal together with zero
/// end-point velocity. Duration is sized up-front by [`plan_cartesian`]
/// so the IK'd path respects the per-joint velocity limits.
pub struct CartesianTrajectory {
    start: Isometry3<f64>,
    end: Isometry3<f64>,
    duration: Duration,
    pub motion_start: Instant,
}

impl CartesianTrajectory {
    pub fn new(start: Isometry3<f64>, end: Isometry3<f64>, duration_secs: f64) -> Self {
        Self {
            start,
            end,
            duration: Duration::from_secs_f64(duration_secs.max(0.0)),
            motion_start: Instant::now(),
        }
    }

    /// EE pose at blend parameter `s` on this trajectory's geometric path:
    /// position on the blend between start and end, orientation slerped at the
    /// same parameter. The runtime IK walk samples blends from
    /// [`subdivided_blends`] with this, so its resolution never falls below the
    /// plan's.
    pub fn sample_at_blend(&self, s: f64) -> Isometry3<f64> {
        interpolate_pose(&self.start, &self.end, s.clamp(0.0, 1.0))
    }

    /// Blend parameter `s in [0, 1]` at time `now` (the quintic of elapsed / total,
    /// 1 for a zero-duration trajectory): how far along its geometric path the
    /// trajectory is, fed to [`sample_at_blend`](Self::sample_at_blend). A steered
    /// line's per-tick elbow budget scales by the blend progressed between solves,
    /// mirroring the planner's per-sample cap.
    pub fn blend(&self, now: Instant) -> f64 {
        let t_total = self.duration.as_secs_f64();
        if t_total == 0.0 {
            return 1.0;
        }
        let elapsed = now.duration_since(self.motion_start).as_secs_f64();
        let (s, _) = quintic((elapsed / t_total).clamp(0.0, 1.0));
        s
    }

    pub fn is_complete(&self, now: Instant) -> bool {
        now.duration_since(self.motion_start) >= self.duration
    }
}

/// Path resolution for the up-front Cartesian velocity-limit sizing: the move's
/// geometric path is sampled this many segments and IK-solved at each to bound the
/// per-joint speed. Closed-form IK makes this sub-millisecond.
const CARTESIAN_PLAN_SAMPLES: usize = 100;

/// The plan's validation grid spacing in blend parameter: what one plan sample
/// spans. The runtime must never advance its IK walk coarser than this (see
/// [`subdivided_blends`]), or a short move's quintic could step past geometry the
/// plan validated cell by cell.
pub const CARTESIAN_PLAN_DS: f64 = 1.0 / CARTESIAN_PLAN_SAMPLES as f64;

/// The blend samples one runtime tick must IK-solve, walking from `prev`
/// (exclusive) to `next` (inclusive) in equal steps no wider than
/// [`CARTESIAN_PLAN_DS`]. A tick that progressed no more than one plan cell gets
/// the single sample `next` (including a zero-progress hold tick); a
/// short-duration move whose quintic outpaces the plan grid gets intermediate
/// samples, so the executed IK walk (and its per-step elbow budget) always runs
/// at least as fine as the walk that validated the line.
pub fn subdivided_blends(prev: f64, next: f64) -> impl Iterator<Item = f64> {
    let span = (next - prev).max(0.0);
    let steps = ((span / CARTESIAN_PLAN_DS).ceil() as usize).max(1);
    (1..=steps).map(move |k| prev + span * (k as f64 / steps as f64))
}

/// Arm-angle travel budget for a Cartesian move, in radians per unit blend
/// parameter: at each IK solve the elbow may step at most this far (scaled by the
/// blend progressed since the previous solve) toward higher manipulability, so a
/// move can swivel its elbow up to this much end to end to stay clear of singular
/// postures. The planner and the runtime derive their per-solve caps from the same
/// budget, so the planned dq/ds already includes the elbow motion the execution
/// performs.
pub const ARM_ANGLE_STEP_PER_BLEND_RAD: f64 = 2.0;

/// Largest joint step (rad) one plan sample of the line may demand and still be
/// tracked as a straight line. Above it the exact IK solution has jumped to
/// another branch (a genuine discontinuity: sampling finer does not shrink it),
/// which no continuous tracking can execute.
const MAX_LINE_STEP_RAD: f64 = 0.35;

/// A trackable straight line whose velocity-limited duration overruns the request
/// by more than this factor is treated as too slow to keep straight: the servo,
/// which reconfigures around the near-singular posture forcing the slowdown, is
/// preferred. Relative rather than an absolute cap so the tolerance scales with
/// the move - a long move gets proportional slack a short one does not, and a
/// short move is not handed a fixed multi-second budget it never needed. Near a
/// singularity the velocity-limited duration diverges, so the exact factor only
/// arbitrates the mild-graze regime; 2x keeps a line that is merely slowed while
/// sending a line that has to crawl to the servo.
const LINE_OVERRUN_FACTOR: f64 = 2.0;

/// Floor on the line-acceptance budget, covering a move requested with no
/// meaningful duration (`requested ~= 0`, "as fast as possible"), where
/// [`LINE_OVERRUN_FACTOR`] times the request degenerates to zero and would send
/// every slowed line - even a well-conditioned one - to the servo. A line whose
/// velocity-limited duration is within this many seconds is kept straight
/// regardless of the request: a few seconds of near-singular graze is honest
/// motion, not a stall.
const MIN_LINE_BUDGET_S: f64 = 3.0;

/// The longest velocity-limited duration a straight line may take and still be
/// kept over the servo: [`LINE_OVERRUN_FACTOR`] times the request, floored at
/// [`MIN_LINE_BUDGET_S`] so a near-zero request keeps a well-conditioned line
/// instead of collapsing to zero and forcing every slowed line to the servo.
fn line_duration_cap(requested_duration_secs: f64) -> f64 {
    (requested_duration_secs * LINE_OVERRUN_FACTOR).max(MIN_LINE_BUDGET_S)
}

/// The motion limits a Cartesian plan sizes and validates against: the same
/// values the runtime enforces, so acceptance and execution agree, plus the
/// control period the servo rollout steps at.
pub struct PlanLimits<'a> {
    pub max_joint_velocity_rad_s: &'a JointVec,
    pub ee: EeCaps,
    pub control_period: Duration,
    /// The servo's per-joint command smoother, built for `control_period` at
    /// setup: carrying it is the proof the period can run it.
    pub smoothing: control_core::filters::ButterworthFilter,
}

/// How an accepted move_arm goal executes, decided by [`plan_cartesian`].
pub enum CartesianPlan {
    /// The straight line is continuously trackable: track it over this duration,
    /// resolving the elbow the same way the plan did (`steer_elbow` on means the
    /// per-blend manipulability budget was needed to keep the line alive; off
    /// means the elbow holds its seed angle, the quiet default). `start_q` is the
    /// normalised start config the runtime must seed the line from (see [`LineWalk`]).
    Line {
        duration_s: f64,
        steer_elbow: bool,
        start_q: JointVec,
    },
    /// No line exists: every continuous tracking demands a branch jump, is
    /// untrackably slow, or leaves reach mid-path. Reach the pose with the
    /// guarded servo law instead (the streaming jog's damped resolved-rate
    /// follow, which crosses the singular surface a discrete branch choice
    /// cannot), validated by an offline rollout that took about this long.
    Servo { duration_s: f64 },
}

/// One policy's walk along the line: the velocity-sizing peak if the line is
/// continuously trackable under that policy.
struct LineWalk {
    /// Peak of `|dq_i/ds| / v_max_i` over the path: the binding joint/segment.
    peak_ratio: f64,
    /// The normalised start config: the k = 0 FromSeed solution. The runtime line
    /// must seed from this, not the raw setpoint, or plan and execution can follow
    /// different IK branches at a redundancy or joint-limit boundary.
    start_q: JointVec,
}

/// Walk the geometric path start->end at plan resolution, IK-solving each sample
/// seeded from the previous. `None` when the line is not continuously trackable
/// under `policy`: a per-sample joint step past [`MAX_LINE_STEP_RAD`] (a branch
/// jump) or a mid-path pose with no in-limit solution.
fn walk_line(
    model: &Arm,
    start: &Isometry3<f64>,
    end: &Isometry3<f64>,
    seed: JointVec,
    max_joint_velocity_rad_s: &JointVec,
    policy: ArmAnglePolicy,
) -> Option<LineWalk> {
    // Normalise the start onto the seed's own arm angle before walking: the raw seed
    // may not equal its FromSeed re-solve (the IK picks the nearest feasible arm
    // angle, which can differ at a redundancy or joint-limit boundary). The runtime
    // seeds the line from this same start_q, and steering (with its guards) accrues
    // from the first real motion sample - steering the elbow at the start, a
    // zero-motion point, would validate a branch the runtime never follows.
    let start_q = model
        .solve_ik(&model.base_pose(start), ArmAnglePolicy::FromSeed, &seed)?
        .q;
    let mut prev = start_q;
    let mut peak_ratio = 0.0_f64;
    for k in 1..=CARTESIAN_PLAN_SAMPLES {
        let pose = interpolate_pose(start, end, k as f64 * CARTESIAN_PLAN_DS);
        let sol = model.solve_ik(&model.base_pose(&pose), policy, &prev)?;
        for i in 0..ARM_DOF {
            let step = (sol.q[i] - prev[i]).abs();
            if step > MAX_LINE_STEP_RAD {
                return None;
            }
            peak_ratio = peak_ratio.max(step / CARTESIAN_PLAN_DS / max_joint_velocity_rad_s[i]);
        }
        prev = sol.q;
    }
    Some(LineWalk {
        peak_ratio,
        start_q,
    })
}

/// Plan a Cartesian move, preferring the quietest execution that works:
///
/// 1. **Held elbow** ([`ArmAnglePolicy::FromSeed`]): the elbow stays at its seed
///    angle, so an ordinary move never swings joints it does not need. Taken
///    whenever the line tracks at a sane duration.
/// 2. **Steered elbow** ([`ArmAnglePolicy::MaxManipulability`] under the shared
///    [`ARM_ANGLE_STEP_PER_BLEND_RAD`] budget): spends elbow motion only when
///    holding it would break the line (a singular graze inflating the duration,
///    or a limit wall the swivel can dodge).
/// 3. **Guarded servo** ([`crate::servo`]): no line exists at all (every
///    continuous tracking demands a branch jump or leaves reach). Follow a
///    leashed reference down the line with the damped resolved-rate law the
///    operator's streaming jog runs, deviating only where the geometry forces
///    it. The tier is accepted only if an offline rollout of the identical law
///    reaches the pose, so a servo move never starts blind.
///
/// A line tier is accepted when it tracks continuously and its velocity-limited
/// duration stays within [`LINE_OVERRUN_FACTOR`] times the request (floored at
/// [`MIN_LINE_BUDGET_S`] for a near-zero request). `None` when even the servo
/// cannot reach the pose. Bounding `dq/ds` numerically along the walk turns "respect every
/// joint velocity limit" into a minimum duration via
/// [`velocity_limited_duration`], the same sizing the joint trajectory does
/// analytically. Poses are in the world frame; IK runs in the arm base frame,
/// so each sample is converted with [`Arm::base_pose`].
pub fn plan_cartesian(
    model: &mut Arm,
    start: &Isometry3<f64>,
    end: &Isometry3<f64>,
    seed: JointVec,
    limits: &PlanLimits,
    requested_duration_secs: f64,
) -> Option<CartesianPlan> {
    let duration_cap = line_duration_cap(requested_duration_secs);
    let tiers = [
        (ArmAnglePolicy::FromSeed, false),
        (
            ArmAnglePolicy::MaxManipulability {
                max_step_rad: ARM_ANGLE_STEP_PER_BLEND_RAD * CARTESIAN_PLAN_DS,
            },
            true,
        ),
    ];
    // The EE speed cap is not enforced per tick on the move path (unlike Follow), so
    // size the duration to respect it up front alongside the joint limits: a short
    // requested duration must not drive the hand past either EE cap. Both axes
    // count, since a pure reorientation covers no distance at all.
    let travel_ratio =
        (end.translation.vector - start.translation.vector).norm() / limits.ee.linear_m_s;
    let turn_ratio = (end.rotation * start.rotation.inverse()).angle() / limits.ee.angular_rad_s;
    let ee_ratio = travel_ratio.max(turn_ratio);
    for (policy, steer_elbow) in tiers {
        let Some(walk) = walk_line(
            model,
            start,
            end,
            seed,
            limits.max_joint_velocity_rad_s,
            policy,
        ) else {
            continue;
        };
        // A budget that cannot size a duration cannot be planned as a line.
        let duration_s =
            velocity_limited_duration(walk.peak_ratio.max(ee_ratio), requested_duration_secs)
                .ok()?;
        if duration_s <= duration_cap {
            return Some(CartesianPlan::Line {
                duration_s,
                steer_elbow,
                start_q: walk.start_q,
            });
        }
    }
    // No line tracks: prove the servo law reaches the pose before accepting it.
    crate::servo::rollout(model, start, end, seed, limits)
        .map(|duration_s| CartesianPlan::Servo { duration_s })
}

/// Interpolate between two poses at blend parameter `s` ∈ [0,1]: position by a
/// straight-line lerp, orientation by slerp along the shorter arc (a 180°
/// reorientation interpolates smoothly along one of its two equal-length
/// geodesics). Shared by [`CartesianTrajectory`] (time-sampled, `s` from the
/// quintic) and [`plan_cartesian`] (geometric, `s` uniform). `try_slerp`
/// returns `None` only when the endpoint orientations are numerically identical
/// (quaternion |dot| ≈ 1 after its shortest-arc sign flip, a rotation gap of
/// microradians), so falling back to the goal orientation is exact there, not a
/// jump.
pub(crate) fn interpolate_pose(
    start: &Isometry3<f64>,
    end: &Isometry3<f64>,
    s: f64,
) -> Isometry3<f64> {
    let position = start.translation.vector.lerp(&end.translation.vector, s);
    let rotation = start
        .rotation
        .try_slerp(&end.rotation, s, 1e-6)
        .unwrap_or(end.rotation);
    Isometry3::from_parts(Translation3::from(position), rotation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use control_core::minimum_jerk::QUINTIC_PEAK_VELOCITY;
    use openarm_description::HardwareVersion;
    use srs_model::nalgebra::{Quaternion, Translation3};

    const V_MAX: JointVec = [1.0; ARM_DOF];
    const EPS: f64 = 1e-9;

    // The launcher-default per-joint velocity limits (peppy.json5), j1..j7.
    const V_MAX_V2: JointVec = [
        16.754666, 16.754666, 5.445426, 5.445426, 20.943946, 20.943946, 20.943946,
    ];

    fn v2_right_arm() -> Arm {
        crate::arm_model(HardwareVersion::V2, openarm_description::Side::Right)
            .expect("bundled v2 URDF builds")
    }

    fn world_pose(position: [f64; 3], quat_xyzw: [f64; 4]) -> Isometry3<f64> {
        Isometry3::from_parts(
            Translation3::new(position[0], position[1], position[2]),
            UnitQuaternion::from_quaternion(Quaternion::new(
                quat_xyzw[3],
                quat_xyzw[0],
                quat_xyzw[1],
                quat_xyzw[2],
            )),
        )
    }

    /// A field-recorded pose, in the chain-tip frame it was captured in, re-expressed
    /// at the tool the planner solves for. The recorded numbers stay verbatim and the
    /// arm runs the motion that was reported, which is what makes these field cases
    /// regressions rather than arbitrary coordinates. Reads the model's own mounted
    /// tool, so the conversion follows the description.
    fn recorded_tip_pose(model: &Arm, position: [f64; 3], quat_xyzw: [f64; 4]) -> Isometry3<f64> {
        world_pose(position, quat_xyzw) * model.tool()
    }

    /// World-frame chain-tip pose at `q`: the frame the field readouts below are
    /// recorded in.
    fn tip_world(model: &mut Arm, q: &JointVec) -> Isometry3<f64> {
        let tip = model.at(q).tip_pose();
        model.world_pose(&tip)
    }

    const TEST_EE_CAP_M_S: f64 = 0.5;
    const TEST_EE_CAP_RAD_S: f64 = 0.8;
    const TEST_DT: Duration = Duration::from_millis(10);

    fn v2_limits() -> PlanLimits<'static> {
        PlanLimits {
            max_joint_velocity_rad_s: &V_MAX_V2,
            ee: EeCaps {
                linear_m_s: TEST_EE_CAP_M_S,
                angular_rad_s: TEST_EE_CAP_RAD_S,
            },
            control_period: TEST_DT,
            smoothing: crate::servo::smoothing_for(TEST_DT).unwrap(),
        }
    }

    fn v1_limits() -> PlanLimits<'static> {
        PlanLimits {
            max_joint_velocity_rad_s: &V_MAX,
            ee: EeCaps {
                linear_m_s: TEST_EE_CAP_M_S,
                angular_rad_s: TEST_EE_CAP_RAD_S,
            },
            control_period: TEST_DT,
            smoothing: crate::servo::smoothing_for(TEST_DT).unwrap(),
        }
    }

    const READY: JointVec = [0.1537, 0.39547, -0.4808, 0.95, -0.0008, 0.0046, -0.0008];

    // The logged orientation of the field repro's poses.
    const REPRO_QUAT: [f64; 4] = [
        -0.06651768984258864,
        -0.5085494684876904,
        0.32535618836337443,
        0.7944156253074012,
    ];

    /// Run the servo law to convergence, asserting every step stays inside the
    /// per-joint velocity budget, and return the converged configuration.
    fn run_servo_to_convergence(
        model: &mut Arm,
        start: &Isometry3<f64>,
        end: &Isometry3<f64>,
        seed: JointVec,
    ) -> JointVec {
        let mut state = crate::servo::ServoState::new(
            *start,
            *end,
            crate::servo::smoothing_for(TEST_DT).unwrap(),
        );
        let mut q = seed;
        let steps = (crate::servo::MAX_SERVO_S / TEST_DT.as_secs_f64()).ceil() as usize;
        for _ in 0..steps {
            let caps = EeCaps {
                linear_m_s: TEST_EE_CAP_M_S,
                angular_rad_s: TEST_EE_CAP_RAD_S,
            };
            match state.step(model, &q, &V_MAX_V2, caps, TEST_DT) {
                crate::servo::ServoStep::Stepped(next) => {
                    for i in 0..ARM_DOF {
                        let v = (next[i] - q[i]).abs() / TEST_DT.as_secs_f64();
                        assert!(
                            v <= V_MAX_V2[i] * 1.0001,
                            "joint {i} at {v:.2} rad/s exceeds its budget"
                        );
                    }
                    q = next;
                }
                crate::servo::ServoStep::Converged(q) => {
                    let ee = model.at(&q).ee_pose();
                    let got = model.world_pose(&ee);
                    assert!(
                        (got.translation.vector - end.translation.vector).norm() < 2e-3,
                        "servo must land on the target position"
                    );
                    assert!(
                        got.rotation.angle_to(&end.rotation) < 1e-2,
                        "servo must land on the target orientation"
                    );
                    return q;
                }
            }
        }
        panic!("servo did not converge within the ceiling");
    }

    // The field repro: pulling the right arm straight back across the base
    // (world x +0.07 -> -0.18 at constant orientation). No continuous joint path
    // tracks that line at any arm angle (the exact IK solution jumps branches
    // mid-path), so the plan must fall through to the guarded servo, whose
    // damped law crosses the wall the way the streaming jog does.
    #[test]
    fn cross_body_pull_runs_the_guarded_servo() {
        let mut model = v2_right_arm();
        let start = recorded_tip_pose(
            &model,
            [0.0715597403410507, -0.179708420505458, 0.448631054180598],
            REPRO_QUAT,
        );
        let end = recorded_tip_pose(
            &model,
            [-0.178440259658949, -0.179708420505458, 0.448631054180598],
            REPRO_QUAT,
        );
        let seed = model
            .solve_ik(
                &model.base_pose(&start),
                srs_model::ArmAnglePolicy::FromSeed,
                &READY,
            )
            .expect("start pose reachable from ready")
            .q;
        let plan = plan_cartesian(&mut model, &start, &end, seed, &v2_limits(), 2.0)
            .expect("servo reaches the pose");
        let CartesianPlan::Servo { duration_s } = plan else {
            panic!("an untrackable line must fall through to the servo");
        };
        assert!(
            duration_s < crate::servo::MAX_SERVO_S,
            "servo rollout should finish inside the ceiling, took {duration_s:.1}s"
        );
        run_servo_to_convergence(&mut model, &start, &end, seed);
    }

    // The user-reported gap made a regression test: pulling x back to -0.2 from
    // Ready works in streaming, so the same intent fired as an action must reach
    // the pose too (via the servo when no line tracks), never a blind swing and
    // never a rejection.
    #[test]
    fn pull_from_ready_to_x_minus_02_reaches_via_servo() {
        let mut model = v2_right_arm();
        // The recorded pull is a tip coordinate, so -0.2 is placed there and carried
        // to the tool: the same physical motion, in the frame commanded.
        let start_tip = tip_world(&mut model, &READY);
        let mut end_tip = start_tip;
        end_tip.translation.vector.x = -0.2;
        let tool = model.tool();
        let (start, end) = (start_tip * tool, end_tip * tool);
        let plan = plan_cartesian(&mut model, &start, &end, READY, &v2_limits(), 2.0)
            .expect("the pull must be reachable, as streaming proves live");
        if let CartesianPlan::Servo { duration_s } = plan {
            assert!(duration_s < crate::servo::MAX_SERVO_S);
            run_servo_to_convergence(&mut model, &start, &end, READY);
        }
        // A Line verdict is equally acceptable: the pose is reached on the line.
    }

    // An easy short move must still plan as a line at exactly the requested
    // duration: the elbow steering must not inflate well-conditioned paths, and
    // trackable lines must never degrade to the servo.
    #[test]
    fn easy_move_plans_the_line_at_the_requested_duration() {
        let mut model = v2_right_arm();
        let start = {
            let ee = model.at(&READY).ee_pose();
            model.world_pose(&ee)
        };
        let mut end = start;
        end.translation.vector.z += 0.05;
        let plan = plan_cartesian(&mut model, &start, &end, READY, &v2_limits(), 2.0)
            .expect("small lift from ready is reachable");
        let CartesianPlan::Line {
            duration_s,
            steer_elbow,
            ..
        } = plan
        else {
            panic!("a trackable line must plan as a line");
        };
        assert!(
            (duration_s - 2.0).abs() < EPS,
            "easy move must stay at the request, got {duration_s:.3}s"
        );
        assert!(!steer_elbow, "an easy move must not spend the elbow budget");
    }

    // The steered elbow spends at most `ARM_ANGLE_STEP_PER_BLEND_RAD *
    // CARTESIAN_PLAN_DS` radians of arm angle per plan sample. That budget must
    // stay well inside the branch-jump guard `MAX_LINE_STEP_RAD`: if one elbow
    // step could induce a joint move near the guard, steering would trip the
    // very discontinuity check meant to catch real branch jumps. This pins the
    // worst-case (full-budget, either direction) induced joint step comfortably
    // under the guard from a well-conditioned pose, the justification for 2.0.
    #[test]
    fn elbow_budget_step_stays_under_the_branch_guard() {
        let mut model = v2_right_arm();
        let base = {
            let ee = model.at(&READY).ee_pose();
            model.base_pose(&model.world_pose(&ee))
        };
        let anchor = model
            .solve_ik(&base, ArmAnglePolicy::FromSeed, &READY)
            .expect("ready pose is reachable");
        let budget = ARM_ANGLE_STEP_PER_BLEND_RAD * CARTESIAN_PLAN_DS;
        for sign in [-1.0, 1.0] {
            let stepped = model
                .solve_ik(
                    &base,
                    ArmAnglePolicy::Fixed(anchor.arm_angle + sign * budget),
                    &anchor.q,
                )
                .expect("a full-budget elbow step from a well-conditioned pose is feasible");
            let induced = (0..ARM_DOF)
                .map(|i| (stepped.q[i] - anchor.q[i]).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                induced < 0.5 * MAX_LINE_STEP_RAD,
                "one elbow-budget step induced {induced:.4} rad, not comfortably under the \
                 {MAX_LINE_STEP_RAD} branch guard"
            );
        }
    }

    // A move that ends as pure rotation (position converges early, orientation
    // keeps slewing) must not trip the stall guard: rotational progress counts
    // as progress. Exercises the law directly with a large in-place
    // reorientation, where reference advance and position shrink both go quiet
    // while the wrist is still turning.
    //
    // In place means about the grasp point, so the wrist orbits it rather than
    // spinning where it is, and from Ready the elbow reaches its singularity floor
    // beyond ~0.8 rad. This turns as far as that posture allows; the servo path
    // under test is the same one either side of that bound.
    #[test]
    fn pure_reorientation_converges_in_the_servo_law() {
        let mut model = v2_right_arm();
        let start = {
            let ee = model.at(&READY).ee_pose();
            model.world_pose(&ee)
        };
        let mut end = start;
        end.rotation = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.7) * end.rotation;
        run_servo_to_convergence(&mut model, &start, &end, READY);
    }

    // The incident regression: after a servo move parks the arm at an unusual
    // arm angle, ordinary nudges from that posture must plan as quiet
    // held-elbow lines, not cascade into further wall crossings (the steered
    // walk's greedy optimizer can manufacture branch jumps on lines the held
    // walk tracks cleanly, so the quiet tier must be tried first).
    #[test]
    fn small_nudge_after_a_servo_move_stays_a_quiet_line() {
        let mut model = v2_right_arm();
        let start = recorded_tip_pose(
            &model,
            [0.0715597403410507, -0.179708420505458, 0.448631054180598],
            REPRO_QUAT,
        );
        let end = recorded_tip_pose(
            &model,
            [-0.178440259658949, -0.179708420505458, 0.448631054180598],
            REPRO_QUAT,
        );
        let seed = model
            .solve_ik(
                &model.base_pose(&start),
                srs_model::ArmAnglePolicy::FromSeed,
                &READY,
            )
            .expect("start pose reachable from ready")
            .q;
        let parked = run_servo_to_convergence(&mut model, &start, &end, seed);
        // From the parked posture, nudge 3 cm in +x: an ordinary move.
        let nudge_start = {
            let ee = model.at(&parked).ee_pose();
            model.world_pose(&ee)
        };
        let mut nudge_end = nudge_start;
        nudge_end.translation.vector.x += 0.03;
        let plan = plan_cartesian(
            &mut model,
            &nudge_start,
            &nudge_end,
            parked,
            &v2_limits(),
            2.0,
        )
        .expect("nudge from the parked posture is reachable");
        let CartesianPlan::Line {
            duration_s,
            steer_elbow,
            ..
        } = plan
        else {
            panic!("an ordinary nudge must stay a line, not servo");
        };
        assert!(!steer_elbow, "an ordinary nudge must hold the elbow");
        assert!(
            (duration_s - 2.0).abs() < EPS,
            "nudge stays at the request, got {duration_s:.3}s"
        );
    }

    #[test]
    fn subdivided_blends_match_plan_resolution() {
        // Within one plan cell (or zero progress): the single sample `next`.
        let one: Vec<f64> = subdivided_blends(0.42, 0.425).collect();
        assert_eq!(one, vec![0.425]);
        let hold: Vec<f64> = subdivided_blends(0.42, 0.42).collect();
        assert_eq!(hold, vec![0.42]);
        // A tick outpacing the grid subdivides evenly: last lands exactly on
        // `next`, and no step exceeds the plan spacing.
        let many: Vec<f64> = subdivided_blends(0.1, 0.1 + 3.7 * CARTESIAN_PLAN_DS).collect();
        assert_eq!(many.len(), 4);
        assert!(approx_eq(
            *many.last().unwrap(),
            0.1 + 3.7 * CARTESIAN_PLAN_DS
        ));
        let mut prev = 0.1;
        for s in many {
            assert!(s - prev <= CARTESIAN_PLAN_DS + EPS);
            prev = s;
        }
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn vec_approx_eq(a: &[f64], b: &[f64]) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y))
    }

    #[test]
    fn duration_floors_at_min() {
        let start = [0.0; ARM_DOF];
        let mut end = [0.0; ARM_DOF];
        end[0] = 0.01; // T_i = 1.875 * 0.01 = 0.01875 s, below the 100 ms floor
        let traj = JointTrajectory::new(start, end, V_MAX, 0.1);
        assert_eq!(traj.duration, Duration::from_millis(100));
    }

    #[test]
    fn duration_respects_larger_min() {
        let start = [0.0; ARM_DOF];
        let end = [0.1; ARM_DOF]; // would be ~0.1875 s at v_max=1
        let traj = JointTrajectory::new(start, end, V_MAX, 5.0);
        assert!(approx_eq(traj.duration.as_secs_f64(), 5.0));
    }

    #[test]
    fn duration_scales_with_largest_relative_motion() {
        let start = [0.0; ARM_DOF];
        let mut end = [0.0; ARM_DOF];
        end[0] = 1.0; // T_0 = 1.875
        end[3] = 0.5; // T_3 = 0.9375, slowest-relative joint wins
        let traj = JointTrajectory::new(start, end, V_MAX, 0.1);
        assert!(approx_eq(traj.duration.as_secs_f64(), 1.875));
    }

    #[test]
    fn boundary_at_tau_zero() {
        let start = [0.1; ARM_DOF];
        let end = [0.5; ARM_DOF];
        let traj = JointTrajectory::new(start, end, V_MAX, 0.1);
        assert!(vec_approx_eq(&traj.sample(traj.motion_start), &start));
    }

    #[test]
    fn boundary_at_tau_one() {
        let start = [0.0; ARM_DOF];
        let end = [0.5; ARM_DOF];
        let traj = JointTrajectory::new(start, end, V_MAX, 0.1);
        assert!(vec_approx_eq(
            &traj.sample(traj.motion_start + traj.duration),
            &end
        ));
    }

    #[test]
    fn holds_at_end_past_duration() {
        let start = [0.0; ARM_DOF];
        let end = [1.0; ARM_DOF];
        let traj = JointTrajectory::new(start, end, V_MAX, 0.1);
        let q = traj.sample(traj.motion_start + traj.duration + Duration::from_secs(5));
        assert!(vec_approx_eq(&q, &end));
    }

    #[test]
    fn joint_zero_duration_holds_at_end() {
        // start == end with a zero request: the degenerate t_total == 0 branch.
        let q = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7];
        let traj = JointTrajectory::new(q, q, V_MAX, 0.0);
        assert!(vec_approx_eq(&traj.sample(traj.motion_start), &q));
    }

    #[test]
    fn midpoint_position_is_halfway() {
        let start = [0.0; ARM_DOF];
        let end = [1.0; ARM_DOF];
        let traj = JointTrajectory::new(start, end, V_MAX, 0.1);
        let half = Duration::from_secs_f64(traj.duration.as_secs_f64() / 2.0);
        let q = traj.sample(traj.motion_start + half);
        assert!(q.iter().all(|v| approx_eq(*v, 0.5)));
    }

    #[test]
    fn is_complete_only_after_duration() {
        let q = [0.0; ARM_DOF];
        let traj = JointTrajectory::new(q, q, V_MAX, 0.1);
        assert!(!traj.is_complete(traj.motion_start));
        assert!(traj.is_complete(traj.motion_start + traj.duration));
        assert!(traj.is_complete(traj.motion_start + traj.duration + Duration::from_millis(1)));
    }

    // --- plan_cartesian (real arm model) ----------------------------------

    fn left_arm() -> Arm {
        let version = openarm_description::HardwareVersion::V1;
        crate::arm_model(version, openarm_description::Side::Left)
            .expect("build left arm from bundled URDF")
    }

    #[test]
    fn plan_cartesian_sizes_in_workspace_and_floors_at_request() {
        let mut arm = left_arm();
        let seed = [0.0, 0.3, 0.0, 0.8, 0.0, 0.5, 0.0];
        let ee = arm.at(&seed).ee_pose();
        let start = arm.world_pose(&ee);
        let mut goal = start;
        goal.translation.vector.z += 0.05; // a small reachable move

        let Some(CartesianPlan::Line { duration_s, .. }) =
            plan_cartesian(&mut arm, &start, &goal, seed, &v1_limits(), 0.0)
        else {
            panic!("an in-workspace move should plan a line");
        };
        assert!(
            duration_s > 0.0,
            "an in-workspace move should plan a positive duration"
        );
        // The request floors the velocity-limited duration.
        let Some(CartesianPlan::Line {
            duration_s: floored,
            ..
        }) = plan_cartesian(&mut arm, &start, &goal, seed, &v1_limits(), 5.0)
        else {
            panic!("reachable");
        };
        assert!(
            floored >= 5.0 - EPS,
            "duration must floor at the requested duration"
        );
    }

    // The move path does not cap end-effector speed per tick (unlike Follow), so the
    // line duration must be sized to respect it up front. A near-zero request would
    // otherwise let joint limits alone drive the hand past the cap; the duration must
    // floor at QUINTIC_PEAK_VELOCITY * line_length / max_ee_velocity.
    #[test]
    fn line_duration_respects_the_ee_speed_cap() {
        let mut arm = left_arm();
        let seed = [0.0, 0.3, 0.0, 0.8, 0.0, 0.5, 0.0];
        let ee = arm.at(&seed).ee_pose();
        let start = arm.world_pose(&ee);
        let mut goal = start;
        goal.translation.vector.z += 0.05;
        let line_len = 0.05;

        let cap = 0.05; // tight EE cap (m/s) so it binds the duration, not the joints
        let limits = PlanLimits {
            max_joint_velocity_rad_s: &V_MAX,
            ee: EeCaps {
                linear_m_s: cap,
                angular_rad_s: TEST_EE_CAP_RAD_S,
            },
            control_period: TEST_DT,
            smoothing: crate::servo::smoothing_for(TEST_DT).unwrap(),
        };
        let Some(CartesianPlan::Line { duration_s, .. }) =
            plan_cartesian(&mut arm, &start, &goal, seed, &limits, 0.0)
        else {
            panic!("a small lift should plan as a line");
        };
        assert!(
            duration_s >= QUINTIC_PEAK_VELOCITY * line_len / cap - EPS,
            "duration {duration_s:.3}s must respect the EE cap floor"
        );
    }

    // The plan must carry the normalized start config (the FromSeed re-solve of the
    // seed at the start pose) so the runtime seeds the line from the same branch the
    // plan validated. This seed does not round-trip through FromSeed, so start_q
    // differs from the raw seed - exactly the case the propagation guards against.
    #[test]
    fn line_plan_carries_the_normalized_start() {
        let mut arm = left_arm();
        let seed = [0.0, 0.3, 0.0, 0.8, 0.0, 0.5, 0.0];
        let ee = arm.at(&seed).ee_pose();
        let start = arm.world_pose(&ee);
        let mut goal = start;
        goal.translation.vector.z += 0.05;

        let Some(CartesianPlan::Line { start_q, .. }) =
            plan_cartesian(&mut arm, &start, &goal, seed, &v1_limits(), 0.0)
        else {
            panic!("a small lift should plan as a line");
        };
        let normalized = arm
            .solve_ik(&arm.base_pose(&start), ArmAnglePolicy::FromSeed, &seed)
            .expect("start reachable")
            .q;
        for i in 0..ARM_DOF {
            assert!(
                (start_q[i] - normalized[i]).abs() < 1e-9,
                "start_q must be the FromSeed normalization of the seed"
            );
        }
        // And for this seed the normalization actually moves the config, which is why
        // seeding the runtime from it (rather than the raw setpoint) matters.
        assert!(
            (0..ARM_DOF).any(|i| (start_q[i] - seed[i]).abs() > 1e-3),
            "this seed does not round-trip; start_q must differ from it"
        );
    }

    // The line-acceptance budget must scale with the request, not sit at a fixed
    // absolute. A long move gets proportional overrun slack (an absolute cap gave
    // a 20 s request zero tolerance, servoing the moment it slowed at all); a
    // near-zero "as fast as possible" request falls back to the floor so a
    // well-conditioned line is never forced to the servo by a zero budget.
    #[test]
    fn line_duration_cap_scales_with_request_and_floors_at_zero() {
        // Near-zero request: the floor governs, so a quick line is still allowed.
        assert!((line_duration_cap(0.0) - MIN_LINE_BUDGET_S).abs() < EPS);
        // Small request below the crossover (MIN_LINE_BUDGET_S / LINE_OVERRUN_FACTOR):
        // still the floor, never below it.
        assert!((line_duration_cap(1.0) - MIN_LINE_BUDGET_S).abs() < EPS);
        // Long request: proportional slack, strictly above the request (the old
        // absolute cap collapsed to exactly the request here, i.e. zero tolerance).
        let long = 20.0;
        assert!((line_duration_cap(long) - long * LINE_OVERRUN_FACTOR).abs() < EPS);
        assert!(line_duration_cap(long) > long);
    }

    #[test]
    fn plan_cartesian_rejects_an_unreachable_goal() {
        let mut arm = left_arm();
        let seed = [0.0, 0.3, 0.0, 0.8, 0.0, 0.5, 0.0];
        let ee = arm.at(&seed).ee_pose();
        let start = arm.world_pose(&ee);
        let mut unreachable = start;
        unreachable.translation.vector.x += 10.0; // 10 m away: no IK solution
        assert!(plan_cartesian(&mut arm, &start, &unreachable, seed, &v1_limits(), 0.0).is_none());
    }

    // --- CartesianTrajectory ---------------------------------------------

    use srs_model::nalgebra::{UnitQuaternion, Vector3};

    fn pose(x: f64, y: f64, z: f64, yaw: f64) -> Isometry3<f64> {
        let r = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw);
        Isometry3::from_parts(Translation3::new(x, y, z), r)
    }

    #[test]
    fn cartesian_180_degree_reorientation_interpolates_smoothly() {
        // Orientations 180° apart have quaternion dot 0, squarely inside slerp's
        // working range (its `None` case is |dot| ≈ 1, i.e. identical endpoint
        // orientations): the blend walks one geodesic continuously rather than
        // jumping to the goal orientation.
        let start = pose(0.0, 0.0, 0.0, 0.0);
        let end = pose(0.0, 0.0, 0.0, std::f64::consts::PI);
        let mut prev = start.rotation;
        for k in 1..=100 {
            let got = interpolate_pose(&start, &end, k as f64 / 100.0);
            let step = got.rotation.angle_to(&prev);
            assert!(step < 0.05, "rotation jumped {step} rad at sample {k}");
            prev = got.rotation;
        }
        assert!(prev.angle_to(&end.rotation) < EPS);
    }

    #[test]
    fn cartesian_boundary_at_tau_zero() {
        let start = pose(0.1, 0.2, 0.3, 0.2);
        let end = pose(0.5, -0.1, 0.4, 1.0);
        let traj = CartesianTrajectory::new(start, end, 2.0);
        let got = traj.sample_at_blend(traj.blend(traj.motion_start));
        assert!((got.translation.vector - start.translation.vector).norm() < EPS);
        assert!(got.rotation.angle_to(&start.rotation) < EPS);
    }

    #[test]
    fn cartesian_boundary_at_tau_one() {
        let start = pose(0.1, 0.2, 0.3, 0.2);
        let end = pose(0.5, -0.1, 0.4, 1.0);
        let traj = CartesianTrajectory::new(start, end, 2.0);
        let got = traj.sample_at_blend(traj.blend(traj.motion_start + traj.duration));
        assert!((got.translation.vector - end.translation.vector).norm() < EPS);
        assert!(got.rotation.angle_to(&end.rotation) < EPS);
    }

    #[test]
    fn cartesian_midpoint_position_is_halfway() {
        // s(0.5) = 0.5, so position and orientation are both at the halfway blend.
        let start = pose(0.0, 0.0, 0.0, 0.0);
        let end = pose(1.0, 2.0, -1.0, 1.0);
        let traj = CartesianTrajectory::new(start, end, 2.0);
        let half = Duration::from_secs_f64(traj.duration.as_secs_f64() / 2.0);
        let got = traj.sample_at_blend(traj.blend(traj.motion_start + half));
        let mid = start.translation.vector.lerp(&end.translation.vector, 0.5);
        assert!((got.translation.vector - mid).norm() < EPS);
        // Halfway in orientation: equal angle to both endpoints.
        let to_start = got.rotation.angle_to(&start.rotation);
        let to_end = got.rotation.angle_to(&end.rotation);
        assert!(approx_eq(to_start, to_end));
    }

    #[test]
    fn cartesian_holds_at_end_past_duration() {
        let start = pose(0.0, 0.0, 0.0, 0.0);
        let end = pose(0.3, 0.3, 0.3, 0.5);
        let traj = CartesianTrajectory::new(start, end, 1.0);
        let got = traj.sample_at_blend(
            traj.blend(traj.motion_start + traj.duration + Duration::from_secs(3)),
        );
        assert!((got.translation.vector - end.translation.vector).norm() < EPS);
        assert!(got.rotation.angle_to(&end.rotation) < EPS);
    }

    #[test]
    fn cartesian_zero_duration_holds_at_end() {
        let start = pose(0.0, 0.0, 0.0, 0.0);
        let end = pose(0.3, 0.3, 0.3, 0.5);
        let traj = CartesianTrajectory::new(start, end, 0.0);
        let got = traj.sample_at_blend(traj.blend(traj.motion_start));
        assert!((got.translation.vector - end.translation.vector).norm() < EPS);
        assert!(got.rotation.angle_to(&end.rotation) < EPS);
    }

    #[test]
    fn cartesian_is_complete_only_after_duration() {
        let p = pose(0.0, 0.0, 0.0, 0.0);
        let traj = CartesianTrajectory::new(p, p, 1.0);
        assert!(!traj.is_complete(traj.motion_start));
        assert!(traj.is_complete(traj.motion_start + traj.duration));
    }

    // The steered-elbow tier must actually be reachable: from the Ready seed
    // (EE at world (0.208, -0.212, 0.376), quat [x,y,z,w] =
    // [-0.0865, -0.4576, 0.2891, 0.8364]), a constant-orientation reach up and
    // across to this target has no continuously trackable held-elbow line (the
    // seed elbow drives a mid-path branch jump), but steering the elbow toward
    // higher manipulability keeps the line alive. Pins the middle tier: held
    // fails, steered wins.
    #[test]
    fn steered_elbow_tier_engages_on_a_graze() {
        let mut model = v2_right_arm();
        let seed = READY;
        // The graze is a tip coordinate, so the target is placed there and carried to
        // the tool: the same line through the same near-singular posture.
        let start_tip = tip_world(&mut model, &seed);
        let mut end_tip = start_tip;
        end_tip.translation.vector.x = 0.0325;
        end_tip.translation.vector.y = -0.3125;
        end_tip.translation.vector.z = 0.6900;
        let tool = model.tool();
        let (start, end) = (start_tip * tool, end_tip * tool);

        // Held elbow alone cannot track this straight line...
        assert!(
            walk_line(
                &model,
                &start,
                &end,
                seed,
                v2_limits().max_joint_velocity_rad_s,
                ArmAnglePolicy::FromSeed,
            )
            .is_none(),
            "held-elbow line should not track this graze"
        );
        // ...so the planner selects the steered line (and it stays under the cap).
        let Some(CartesianPlan::Line {
            steer_elbow: true, ..
        }) = plan_cartesian(&mut model, &start, &end, seed, &v2_limits(), 2.0)
        else {
            panic!("expected a steered-elbow line");
        };
    }

    // A pure reorientation covers no distance, so only the angular cap can size
    // it: with the linear term alone a short request would spin the hand as fast
    // as the joint limits allow.
    #[test]
    fn a_pure_reorientation_is_sized_by_the_angular_cap() {
        let mut model = v2_right_arm();
        let seed = READY;
        let start = tip_world(&mut model, &seed) * model.tool();
        let turn_rad = 0.6;
        let end = start
            * Isometry3::from_parts(
                Translation3::identity(),
                UnitQuaternion::from_axis_angle(&Vector3::z_axis(), turn_rad),
            );

        let Some(CartesianPlan::Line { duration_s, .. }) =
            plan_cartesian(&mut model, &start, &end, seed, &v2_limits(), 0.1)
        else {
            panic!("expected a line for a pure reorientation");
        };
        let peak_rad_s = QUINTIC_PEAK_VELOCITY * turn_rad / duration_s;
        assert!(
            peak_rad_s <= TEST_EE_CAP_RAD_S + 1e-9,
            "peak {peak_rad_s} rad/s exceeds the {TEST_EE_CAP_RAD_S} rad/s cap over {duration_s}s"
        );
    }
}
