//! What the unit tests share: the arms built from the bundled description,
//! the planner config and the planners of the tests, the poses the tests hold
//! the arms at, each arm measured where its planner holds it, how far a point
//! stands from a segment, move slots claimed as at a goal's admission, the
//! goals of joint paths and where their shares report, a path guard that
//! lets every path through and one that holds every path short, the guard
//! the tests judge paths against, the stop of a judgment that runs to its
//! end, the paths of a step no joint path moves, and the node's parameter set
//! the integration tests boot the harness with.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use control_core::path_timing::RoundedPath;
use openarm_description::{ARM_DOF, ArmOrder, ArmPair, HardwareVersion, Side};
use srs_model::{Arm, JointVec, Limit};

use tokio::sync::mpsc;

use crate::governor::{GovState, GuardAt, PathArms};
use crate::move_slot::MoveSlot;
use crate::planner::{
    JointPath, Measurement, PathGoal, PathGuard, PlanConfig, Planner, ShareReply, ShareReport,
};
use crate::settle::Settle;

#[path = "../tests/helpers/params.rs"]
mod params;

pub(crate) use params::params;

/// The per-joint acceleration limits of the test planners: above the node's
/// default (peppy.json5), so a move a test plays takes few ticks.
const A_MAX: JointVec = [5.0; ARM_DOF];

/// The control period of the test planners.
const TEST_PERIOD: Duration = Duration::from_millis(10);

/// The `side` arm of `version`, built by `HardwareVersion::arm_model` from the bundled
/// description.
pub(crate) fn bundled_arm(version: HardwareVersion, side: Side) -> Arm {
    version
        .arm_model(side)
        .expect("build the arm from the bundled URDF")
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

/// An in-limit pose of v1's left arm: elbow off its 0.05 floor, j2 inside
/// [-3.32, 0.17].
pub(crate) const POSE_TEST_Q: JointVec = [0.0, -0.8, 0.0, 1.2, 0.0, 0.0, 0.0];

/// Where both arms stand in the path tests: [`POSE_TEST_Q`], the right arm's
/// mirrored into its limits, whose second joint runs the other way.
pub(crate) const BOTH_HELD: ArmPair<JointVec> = ArmPair {
    left: POSE_TEST_Q,
    right: [0.0, 0.8, 0.0, 1.2, 0.0, 0.0, 0.0],
};

/// A planner of the `side` arm of v1 hardware, within the arm's own joint
/// limits, holding `held`.
pub(crate) fn planner_holding(side: Side, held: JointVec) -> Planner {
    let model = bundled_arm(HardwareVersion::V1, side);
    let cfg = PlanConfig {
        limits: model.limits(),
        ..plan_config()
    };
    let mut planner = Planner::new(side, model, cfg);
    planner.commit(held);
    planner
}

/// Each arm measured where its planner holds it.
pub(crate) fn measured_at_setpoints(planners: &ArmPair<Planner>) -> ArmPair<Measurement> {
    ArmPair::from_fn(|side| Ok(planners[side].setpoint()))
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
pub(crate) fn claimed(arms: &ArmOrder) -> ArmPair<MoveSlot> {
    let move_slots = ArmPair::from_fn(|_| MoveSlot::default());
    for &side in arms.sides() {
        assert!(move_slots[side].claim(), "a new slot is free");
    }
    move_slots
}

/// Where the shares of a test path report, flagging the goal's cancel on
/// `cancelled`, and the reports they send.
pub(crate) fn share_reply(
    cancelled: &Arc<AtomicBool>,
) -> (ShareReply, mpsc::Receiver<ShareReport>) {
    let (done_tx, done_rx) = mpsc::channel(Side::BOTH.len());
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
        tolerance: vec![0.01; arms.sides().len() * ARM_DOF],
        timeout: Duration::from_secs(1),
    }
}

/// The array in the order of `arms` of each named arm's joints in
/// `joints`.
pub(crate) fn array_of(arms: &ArmOrder, joints: &ArmPair<JointVec>) -> Vec<f64> {
    arms.sides().iter().flat_map(|&side| joints[side]).collect()
}

/// A goal of a path of `arms` through `waypoints` over `duration` at least,
/// with no expected start, settling under `settle`, reporting on `reply`,
/// holding the slot `slot_of` gives each of its arms.
pub(crate) fn path_goal_through<'a>(
    arms: &ArmOrder,
    waypoints: &[ArmPair<JointVec>],
    duration: Duration,
    settle: Settle<Vec<f64>>,
    reply: ShareReply,
    slot_of: impl Fn(Side) -> &'a MoveSlot,
) -> PathGoal {
    PathGoal {
        path: JointPath {
            arms: arms.clone(),
            waypoints: waypoints.iter().map(|w| array_of(arms, w)).collect(),
            duration,
        },
        expected_starts: vec![None; arms.sides().len()],
        settle,
        reply,
        move_slots: arms
            .sides()
            .iter()
            .map(|&side| slot_of(side).release_on_drop())
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
pub(crate) fn test_guard(planners: &ArmPair<Planner>) -> GuardAt {
    GuardAt::disabled(GovState::new(
        ArmPair::from_fn(|side| planners[side].setpoint()),
        ArmPair::new(1.0, 1.0),
    ))
}

/// The stop of a judgment no test raises: the judgment runs to its end.
pub(crate) static NO_STOP: AtomicBool = AtomicBool::new(false);

/// The paths of a governed step that no arm takes along a joint path: the
/// governor shapes and holds each of its DOF on its own.
pub(crate) const NO_PATHS: &[PathArms] = &[];
