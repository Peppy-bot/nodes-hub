//! The OpenArm models, the launcher's limits and the field-recorded poses the
//! plan regressions run on.

use std::time::{Duration, Instant};

use control_core::servo::{ORIENTATION_TOLERANCE_RAD, POSITION_TOLERANCE_M};
use openarm_backbone::arm_model;
use openarm_description::{ARM_DOF, HardwareVersion, Side};
use srs_model::chain_kinematics::{
    EeCaps, JointSmoothing, LEASH_M, MAX_SERVO_S, ServoState, ServoTolerances, ServoTrajectory,
    command_smoothing, rollout,
};
use srs_model::nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};
use srs_model::{Arm, ArmAnglePolicy, CartesianPlan, JointVec, PlanLimits};

pub const EPS: f64 = 1e-9;

/// The launcher-default per-joint acceleration limits (peppy.json5).
pub const A_MAX: JointVec = [5.0; ARM_DOF];

/// The launcher-default per-joint velocity limits (peppy.json5), j1..j7.
pub const V_MAX_V2: JointVec = [
    16.754666, 16.754666, 5.445426, 5.445426, 20.943946, 20.943946, 20.943946,
];

pub const V_MAX_V1: JointVec = [1.0; ARM_DOF];

const TEST_EE_CAP_M_S: f64 = 0.5;

const TEST_EE_CAP_RAD_S: f64 = 0.8;

pub const TEST_DT: Duration = Duration::from_millis(10);

pub const READY: JointVec = [0.1537, 0.39547, -0.4808, 0.95, -0.0008, 0.0046, -0.0008];

/// The logged orientation of the field repro's poses.
const REPRO_QUAT: [f64; 4] = [
    -0.06651768984258864,
    -0.5085494684876904,
    0.32535618836337443,
    0.7944156253074012,
];

/// The node's own slack, what a goal that states no preference is planned
/// against.
pub fn default_tolerance() -> ServoTolerances {
    ServoTolerances::new(POSITION_TOLERANCE_M, ORIENTATION_TOLERANCE_RAD)
        .expect("the node's default slack is reachable")
}

/// What the panel asks for: 5 mm and 5 degrees.
pub fn panel_tolerance() -> ServoTolerances {
    ServoTolerances::new(0.005, 5.0_f64.to_radians()).expect("the panel's tolerance is reachable")
}

pub fn v2_right_arm() -> Arm {
    arm_model(HardwareVersion::V2, Side::Right).expect("bundled v2 URDF builds")
}

pub fn v2_left_arm() -> Arm {
    arm_model(HardwareVersion::V2, Side::Left).expect("bundled v2 URDF builds")
}

pub fn v1_left_arm() -> Arm {
    arm_model(HardwareVersion::V1, Side::Left).expect("bundled v1 URDF builds")
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
/// regressions.
pub fn recorded_tip_pose(model: &Arm, position: [f64; 3], quat_xyzw: [f64; 4]) -> Isometry3<f64> {
    world_pose(position, quat_xyzw) * model.tool()
}

/// World-frame chain-tip pose at `q`: the frame the field readouts are recorded
/// in.
pub fn tip_world(model: &Arm, q: &JointVec) -> Isometry3<f64> {
    model.world_pose(&model.at(q).tip_pose())
}

pub fn limits_for(max_joint_velocity_rad_s: JointVec) -> PlanLimits {
    PlanLimits {
        max_joint_velocity_rad_s,
        max_joint_acceleration_rad_s2: A_MAX,
        ee: EeCaps {
            linear_m_s: TEST_EE_CAP_M_S,
            angular_rad_s: TEST_EE_CAP_RAD_S,
        },
        control_period: TEST_DT,
        smoothing: command_smoothing(TEST_DT).expect("a 100 Hz period carries the cutoff"),
    }
}

pub fn v2_limits() -> PlanLimits {
    limits_for(V_MAX_V2)
}

pub fn v1_limits() -> PlanLimits {
    limits_for(V_MAX_V1)
}

pub fn ready_pose(model: &Arm) -> Isometry3<f64> {
    model.world_pose(&model.at(&READY).ee_pose())
}

/// The configurations the servo law commands from `seed` to the pose `end`
/// from `start` at the v2 limits, within `tolerance`, or `None` when it does
/// not converge.
pub fn servo_rollout(
    model: &Arm,
    start: &Isometry3<f64>,
    end: &Isometry3<f64>,
    seed: JointVec,
    tolerance: ServoTolerances,
) -> Option<Vec<JointVec>> {
    let limits = v2_limits();
    let state = ServoState::new(*start, *end, LEASH_M, JointSmoothing::new(limits.smoothing));
    rollout(
        model.chain(),
        state,
        seed,
        &limits.servo_at(tolerance),
        MAX_SERVO_S,
    )
}

/// Roll the servo law out to convergence at the v2 limits, asserting every
/// step of its path stays inside the per-joint velocity budget and that the
/// pose it lands on honors `tolerance`, and return the converged
/// configuration.
pub fn roll_servo_to_convergence(
    model: &Arm,
    start: &Isometry3<f64>,
    end: &Isometry3<f64>,
    seed: JointVec,
    tolerance: ServoTolerances,
) -> JointVec {
    let configs = servo_rollout(model, start, end, seed, tolerance)
        .expect("servo did not converge within the ceiling");
    for step in configs.windows(2) {
        for i in 0..ARM_DOF {
            let v = (step[1][i] - step[0][i]).abs() / TEST_DT.as_secs_f64();
            assert!(
                v <= V_MAX_V2[i] * 1.0001,
                "joint {i} at {v:.2} rad/s exceeds its budget"
            );
        }
    }
    let landed = configs[configs.len() - 1];
    assert_lands_within(model, &landed, end, tolerance);
    landed
}

/// Asserts the hand at `q` lies within `tolerance` of `end` on both axes.
pub fn assert_lands_within(
    model: &Arm,
    q: &JointVec,
    end: &Isometry3<f64>,
    tolerance: ServoTolerances,
) {
    let got = model.world_pose(&model.at(q).ee_pose());
    let position_err = (got.translation.vector - end.translation.vector).norm();
    let orientation_err = got.rotation.angle_to(&end.rotation);
    assert!(
        position_err <= tolerance.position_m(),
        "landed {position_err:.6} m out, past the {:.6} m it was planned to",
        tolerance.position_m()
    );
    assert!(
        orientation_err <= tolerance.orientation_rad(),
        "landed {orientation_err:.6} rad out, past the {:.6} rad it was planned to",
        tolerance.orientation_rad()
    );
}

/// The field repro: the right arm pulled straight back across the base
/// (world x +0.07 -> -0.18 at constant orientation), as its start and end
/// poses and the configuration it starts from.
pub fn cross_body_pull(model: &Arm) -> (Isometry3<f64>, Isometry3<f64>, JointVec) {
    let start = recorded_tip_pose(
        model,
        [0.0715597403410507, -0.179708420505458, 0.448631054180598],
        REPRO_QUAT,
    );
    let end = recorded_tip_pose(
        model,
        [-0.178440259658949, -0.179708420505458, 0.448631054180598],
        REPRO_QUAT,
    );
    let seed = model
        .solve_ik(&model.base_pose(&start), ArmAnglePolicy::FromSeed, &READY)
        .expect("start pose reachable from ready")
        .q;
    (start, end, seed)
}

/// The duration and the first and last configurations a servo plan plays, read
/// off its trajectory at its start and a second past its end; panics on a line.
pub fn played_servo(plan: CartesianPlan) -> (f64, JointVec, JointVec) {
    let CartesianPlan::Servo(servo) = plan else {
        panic!("an untrackable line must fall through to the servo");
    };
    let duration = servo.duration;
    let start = Instant::now();
    let traj = ServoTrajectory::new(servo.path, duration, start);
    (
        duration.as_secs_f64(),
        traj.sample(start),
        traj.sample(start + duration + Duration::from_secs(1)),
    )
}
