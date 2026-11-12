//! What the unit tests share: the arms by their places, the bundled
//! descriptions, the arms and the postures built from them, the planner
//! config and the planners of the tests, the poses the tests hold the arms
//! at, each arm measured where its planner holds it, how far a point
//! stands from a segment, move slots claimed as at a goal's admission, the
//! goals of joint paths and where their shares report, a path guard that
//! lets every path through and one that holds every path short, the guard
//! the tests judge paths against, the stop of a judgment that runs to its
//! end, the paths of a step no joint path moves, and the node's parameter set
//! the integration tests boot the harness with.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use control_core::path_timing::RoundedPath;
use robot_description::{ArmId, Driven, PerArm, RobotDescription};
use srs_model::{ARM_DOF, Arm, JointVec, Limit};
use tokio::sync::mpsc;
use workspace_core::surface_block::ArmBodies;

use crate::arm_order::ArmOrder;
use crate::governor::{GovState, GuardAt, PathArms};
use crate::move_slot::MoveSlot;
use crate::planner::{
    JointPath, Measurement, PathGoal, PathGuard, PlanConfig, Planner, ShareReply, ShareReport,
};
use crate::robot::{self, ARMS};
use crate::settle::Settle;

#[path = "../tests/helpers/params.rs"]
mod params;

pub(crate) use params::params;

/// The per-joint acceleration limits of the test planners: above the node's
/// default (peppy.json5), so a move a test plays takes few ticks.
const A_MAX: JointVec = [5.0; ARM_DOF];

/// The control period of the test planners.
const TEST_PERIOD: Duration = Duration::from_millis(10);

/// The arms, in place order: the arm [`robot::LIMBS`] names first, then the second.
pub(crate) const LEFT: ArmId = ArmId::new(0);
pub(crate) const RIGHT: ArmId = ArmId::new(1);

/// The models the tests drive, by their robot_identity model ids.
pub(crate) const V1: &str = "openarm_v1";
pub(crate) const V2: &str = "openarm_v2";

/// [`V1`] and [`V2`], each held to this node's arms, read once per test binary.
static DRIVEN: LazyLock<[Driven<ARM_DOF, ARMS>; 2]> = LazyLock::new(|| {
    [V1, V2].map(|model| {
        robot::LIMBS
            .model(model)
            .expect("this node drives the model")
    })
});

/// `model`, one of [`V1`] and [`V2`], held to this node's arms.
pub(crate) fn driven(model: &str) -> &'static Driven<ARM_DOF, ARMS> {
    DRIVEN
        .iter()
        .find(|driven| driven.model().id() == model)
        .expect("the tests drive V1 and V2")
}

/// The bundled description of `model`, one of [`V1`] and [`V2`].
pub(crate) fn description(model: &str) -> &'static RobotDescription {
    driven(model).description()
}

/// The arm `arm` of `model`, built from its bundled description.
pub(crate) fn bundled_arm(model: &str, arm: ArmId) -> Arm {
    let description = description(model);
    Arm::from_description(description, &description.arms()[arm.place()])
        .expect("build the arm from the bundled URDF")
}

/// The arm bodies of `model`, from its bundled description and collision meshes.
pub(crate) fn arm_bodies(model: &str) -> ArmBodies {
    let meshes = driven(model).model().directory().join("meshes");
    ArmBodies::new(
        description(model),
        meshes.to_str().expect("the meshes path is UTF-8"),
    )
    .expect("the arm bodies of the bundled description")
}

/// Each arm's posture in the state `state` of `model`'s description.
pub(crate) fn postures(model: &str, state: &'static str) -> PerArm<JointVec, ARMS> {
    robot::postures_of(driven(model), state).expect("each arm's posture")
}

/// The planner config of the tests: a 10 ms period, 10 rad/s on every
/// joint, [`A_MAX`], a 1 m/s stream cap, joint limits of +-10 rad, and a start
/// tolerance of 0.01 rad.
pub(crate) fn plan_config() -> PlanConfig {
    PlanConfig {
        cycle_period: TEST_PERIOD,
        max_joint_velocity_rad_s: [10.0; ARM_DOF],
        max_joint_acceleration_rad_s2: A_MAX,
        stream_ee_velocity_m_s: 1.0,
        limits: [Limit {
            lo: -10.0,
            hi: 10.0,
        }; ARM_DOF],
        start_tolerance_rad: 0.01,
    }
}

/// An in-limit pose of v1's left arm, its elbow bent.
pub(crate) const POSE_TEST_Q: JointVec = [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];

/// Where both arms stand in the path tests: [`POSE_TEST_Q`], the right arm's
/// mirrored into its limits, whose second joint runs the other way.
pub(crate) const BOTH_HELD: PerArm<JointVec, ARMS> =
    PerArm::new([POSE_TEST_Q, [0.0, 0.8, 0.0, 1.2, 0.0, 0.0, 0.0]]);

/// A planner of the arm `arm` of [`V1`], within the arm's own joint limits,
/// holding `held`.
pub(crate) fn planner_holding(arm: ArmId, held: JointVec) -> Planner {
    let model = bundled_arm(V1, arm);
    let cfg = PlanConfig {
        limits: model.limits(),
        ..plan_config()
    };
    let mut planner = Planner::new(arm, model, cfg);
    planner.commit(held);
    planner
}

/// Each arm measured where its planner holds it.
pub(crate) fn measured_at_setpoints(planners: &PerArm<Planner, ARMS>) -> PerArm<Measurement, ARMS> {
    PerArm::from_fn(|arm| Ok(planners[arm].setpoint()))
}

/// How far `q` stands from the segment from `a` to `b`.
pub(crate) fn distance_to_segment(q: &[f64], a: &[f64], b: &[f64]) -> f64 {
    let run: Vec<f64> = a.iter().zip(b).map(|(a, b)| b - a).collect();
    let length_sq: f64 = run.iter().map(|d| d * d).sum();
    let t = if length_sq == 0.0 {
        0.0
    } else {
        let along: f64 = q
            .iter()
            .zip(a)
            .zip(&run)
            .map(|((q, a), r)| (q - a) * r)
            .sum();
        (along / length_sq).clamp(0.0, 1.0)
    };
    q.iter()
        .zip(a)
        .zip(&run)
        .map(|((q, a), r)| (q - a - t * r).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// A move slot claimed as an action handler claims it when it admits a goal.
pub(crate) fn claimed_move_slot() -> MoveSlot {
    let move_slot = MoveSlot::default();
    assert!(move_slot.claim(), "a new slot is free");
    move_slot
}

/// The move slots of both arms, those of `arms` claimed as the action's
/// admission claims them.
pub(crate) fn claimed(arms: &ArmOrder) -> PerArm<MoveSlot, ARMS> {
    let move_slots = PerArm::from_fn(|_| MoveSlot::default());
    for &arm in arms.ids() {
        assert!(move_slots[arm].claim(), "a new slot is free");
    }
    move_slots
}

/// Where the shares of a test path report, flagging the goal's cancel on
/// `cancelled`, and the reports they send.
pub(crate) fn share_reply(
    cancelled: &Arc<AtomicBool>,
) -> (ShareReply, mpsc::Receiver<ShareReport>) {
    let (done_tx, done_rx) = mpsc::channel(ARMS);
    let reply = ShareReply {
        done_tx,
        cancelled: cancelled.clone(),
        failed: Arc::new(AtomicBool::new(false)),
    };
    (reply, done_rx)
}

/// The settle of the joint moves of the tests, for `arms`: within 0.01 rad
/// in a second.
pub(crate) fn test_settle(arms: &ArmOrder) -> Settle<Vec<f64>> {
    Settle {
        tolerance: vec![0.01; arms.ids().len() * ARM_DOF],
        timeout: Duration::from_secs(1),
    }
}

/// The array in the order of `arms` of each named arm's joints in
/// `joints`.
pub(crate) fn array_of(arms: &ArmOrder, joints: &PerArm<JointVec, ARMS>) -> Vec<f64> {
    arms.ids().iter().flat_map(|&arm| joints[arm]).collect()
}

/// A goal of a path of `arms` through `waypoints` over `duration` at least,
/// with no expected start, settling under `settle`, reporting on `reply`,
/// holding the slot `slot_of` gives each of its arms.
pub(crate) fn path_goal_through<'a>(
    arms: &ArmOrder,
    waypoints: &[PerArm<JointVec, ARMS>],
    duration: Duration,
    settle: Settle<Vec<f64>>,
    reply: ShareReply,
    slot_of: impl Fn(ArmId) -> &'a MoveSlot,
) -> PathGoal {
    PathGoal {
        path: JointPath {
            arms: arms.clone(),
            waypoints: waypoints.iter().map(|w| array_of(arms, w)).collect(),
            duration,
        },
        expected_starts: vec![None; arms.ids().len()],
        settle,
        reply,
        move_slots: arms
            .ids()
            .iter()
            .map(|&arm| slot_of(arm).release_on_drop())
            .collect(),
    }
}

/// A guard that lets every path through: the governor's own tests check
/// paths against the robot's body.
pub(crate) struct OpenGuard;

impl PathGuard for OpenGuard {
    fn check(
        &mut self,
        _arms: PathArms,
        _guard: &GuardAt,
        _path: &RoundedPath,
        _stop: &AtomicBool,
    ) -> Result<(), String> {
        Ok(())
    }
}

/// The words of every failure of [`Refusing`].
pub(crate) const ARMS_WOULD_MEET: &str = "the arms would meet";

/// A guard that holds every path short, in the words of [`ARMS_WOULD_MEET`].
pub(crate) struct Refusing;

impl PathGuard for Refusing {
    fn check(
        &mut self,
        _arms: PathArms,
        _guard: &GuardAt,
        _path: &RoundedPath,
        _stop: &AtomicBool,
    ) -> Result<(), String> {
        Err(ARMS_WOULD_MEET.to_string())
    }
}

/// The guard the paths of the tests are judged against: where the arms are
/// held, the grippers open, and the guard disabled; the checkers of the tests
/// judge the path alone.
pub(crate) fn test_guard(planners: &PerArm<Planner, ARMS>) -> GuardAt {
    GuardAt::disabled(GovState::new(
        PerArm::from_fn(|arm| planners[arm].setpoint()),
        PerArm::new([1.0, 1.0]),
    ))
}

/// The stop of a judgment no test raises: the judgment runs to its end.
pub(crate) static NO_STOP: AtomicBool = AtomicBool::new(false);

/// The paths of a governed step that no arm takes along a joint path: the
/// governor shapes and holds each of its DOF on its own.
pub(crate) const NO_PATHS: &[PathArms] = &[];
