//! The OpenArm models, the launcher's limits and the field-recorded poses the
//! plan regressions run on.

use std::time::{Duration, Instant};

use control_core::path_timing::{JointTrajectory, distinct_joint_waypoints};
use robot_description::states;
use srs_model::chain_kinematics::pose_miss;
use srs_model::nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};
use srs_model::{ARM_DOF, Arm, ArmAnglePolicy, JointVec};

/// How far (m, rad) a line's last solve may land from its goal: a descent's
/// convergence, as KDL's `CartToJnt` judges it.
const LANDING_SLACK: f64 = 1e-5;

/// The joint whose zero is the wrist's singular surface.
pub const WRIST_BEND: usize = 5;

/// The launcher-default per-joint acceleration limits (peppy.json5).
pub const A_MAX: JointVec = [0.5; ARM_DOF];

/// The launcher-default per-joint velocity limits (peppy.json5), j1..j7.
pub const V_MAX_V2: JointVec = [
    16.754666, 16.754666, 5.445426, 5.445426, 20.943946, 20.943946, 20.943946,
];

/// The arm of the description of the model `model` named `arm_name`, as the
/// backbone builds it.
fn arm_of(model: &str, arm_name: &str) -> Arm {
    let description = robot_description::model(model)
        .expect("a bundled model")
        .description();
    let arm = description
        .arms()
        .iter()
        .find(|arm| arm.name == arm_name)
        .expect("an arm of the model");
    Arm::from_description(&description, arm).expect("the bundled description builds")
}

/// The model the plan regressions run on.
pub const V2: &str = "openarm_v2";

/// The posture `state` of the arm `arm_name` of the model `model`, where the
/// backbone's posture moves put it.
pub fn posture(model: &str, arm_name: &str, state: &str) -> JointVec {
    let description = robot_description::model(model)
        .expect("a bundled model")
        .description();
    description
        .state(arm_name, state)
        .expect("each arm's posture")
        .try_into()
        .expect("a posture of each joint")
}

/// The posture of the right arm, as measured in the field, that the field reports
/// start from: the reported pull and the cross-body repro.
pub const REPORTED_START: JointVec = [0.1537, 0.39547, -0.4808, 0.95, -0.0008, 0.0046, -0.0008];

/// The left arm's mirror of [`REPORTED_START`]: the field-recorded grasp posture of
/// the reported lift.
pub const REPORTED_START_LEFT: JointVec =
    [-0.1537, -0.39547, 0.4808, 0.95, 0.0008, -0.0046, 0.0008];

/// The logged orientation of the field repro's poses.
const REPRO_QUAT: [f64; 4] = [
    -0.06651768984258864,
    -0.5085494684876904,
    0.32535618836337443,
    0.7944156253074012,
];

pub fn v2_right_arm() -> Arm {
    arm_of(V2, "right_arm")
}

pub fn v2_left_arm() -> Arm {
    arm_of(V2, "left_arm")
}

pub fn v1_left_arm() -> Arm {
    arm_of("openarm_v1", "left_arm")
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

/// The trajectory through `walk` at the launcher's v2 limits, resting at no sample
/// before the end, as fast as they allow, starting at `now`.
pub fn played(walk: &[JointVec], now: Instant) -> JointTrajectory<ARM_DOF> {
    JointTrajectory::through(
        &distinct_joint_waypoints(walk),
        &V_MAX_V2,
        &A_MAX,
        Duration::ZERO,
        now,
    )
}

/// The grasp pose of `model`, a right arm of [`V2`], at its Ready posture.
pub fn ready_pose(model: &Arm) -> Isometry3<f64> {
    model.world_pose(&model.at(&posture(V2, "right_arm", states::READY)).ee_pose())
}

/// Asserts the hand at `q` lies on `end`, to the solver's rounding, on both axes.
pub fn assert_lands_on(model: &Arm, q: &JointVec, end: &Isometry3<f64>) {
    let miss = pose_miss(&model.world_ee_pose(q), end);
    assert!(
        miss.position_m <= LANDING_SLACK && miss.orientation_rad <= LANDING_SLACK,
        "landed {miss:?} from the goal"
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
        .solve_ik(
            &model.base_pose(&start),
            ArmAnglePolicy::FromSeed,
            &REPORTED_START,
        )
        .expect("start pose reachable from the reported start")
        .q;
    (start, end, seed)
}
