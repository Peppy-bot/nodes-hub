"""Action-layer behavior that the wire cannot provoke deterministically:
travel timeouts, walked-back admissions, per-server pending isolation, limb
naming, and failure-aware completion."""

import asyncio
import math
import time
from types import SimpleNamespace

import pytest
from conftest import WIDE_LIMITS, WIDE_REACH, FakeKinematics, make_config

from so101_backbone.actions import (
    MAX_REQUESTED_DURATION_S,
    ActionLayer,
    PendingSlot,
    PoseGoal,
    Rejection,
    _orientation_note,
)
from so101_backbone.coordinator import Coordinator


class FakeGoalContext:
    """Records the terminal call; cancel never fires unless asked."""

    def __init__(self):
        self.completed = None
        self.cancelled = None
        self._cancel = asyncio.Event()

    async def cancel_signal(self):
        await self._cancel.wait()

    async def complete(self, *args):
        self.completed = args

    async def complete_cancelled(self, *args):
        self.cancelled = args


def make_layer(kinematics=None, **config_overrides):
    config = make_config(**config_overrides)
    coordinator = Coordinator(config, kinematics or FakeKinematics(), WIDE_LIMITS, WIDE_REACH)
    return coordinator, ActionLayer(coordinator, config, WIDE_LIMITS)


async def installed_plan(coordinator):
    async def poll():
        # The coordinator installs a plan synchronously and owns no event to
        # wait on; adding one to production code for a test's benefit would
        # be the wrong trade, so this polls under a timeout.
        while coordinator._arm_plan is None:  # noqa: ASYNC110
            await asyncio.sleep(0.001)
        return coordinator._arm_plan

    return await asyncio.wait_for(poll(), timeout=2.0)


async def test_gripper_goal_times_out_instead_of_holding_the_slot():
    coordinator, layer = make_layer()
    coordinator.measured_gripper.set(0.0)
    plan = layer._admit_gripper_move(1.0, 0.0)
    # No control loop runs, so the plan can never complete: a wedged stream
    # mid-goal looks exactly like this.
    plan.timeout_s = 0.05
    ctx = FakeGoalContext()
    await layer.drive_gripper(ctx, plan)
    success, message, _opening, _action_time = ctx.completed
    assert success is False
    assert "travel budget" in message
    # The gripper slot is free again.
    assert coordinator.try_claim_gripper()


async def test_abandon_walks_back_a_claimed_admission():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    pending = PendingSlot()
    pending.plan = layer._admit_arm_move((0.1,) * 5, 0.0)
    # The accept failing on the wire is the case abandon exists for.
    plan = pending.plan
    layer.abandon(pending)
    assert plan.aborted
    assert coordinator.try_claim_arm(), "the arm slot must be released"


async def test_abandon_on_one_server_cannot_touch_anothers_admission():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    arm_pending = PendingSlot()
    arm_pending.plan = layer._admit_arm_move((0.1,) * 5, 0.0)
    other_server = PendingSlot()
    # Another server's delivery failure walks back only its own (empty) slot.
    layer.abandon(other_server)
    assert arm_pending.plan is not None
    assert not arm_pending.plan.aborted
    assert not coordinator.try_claim_arm(), "the admitted goal still holds its claim"


async def test_arm_and_gripper_claims_are_independent():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    coordinator.measured_gripper.set(0.0)
    layer._admit_arm_move((0.1,) * 5, 0.0)
    # An arm move in flight must not block a gripper goal, and vice versa.
    plan = layer._admit_gripper_move(1.0, 0.0)
    assert plan is not None
    with pytest.raises(Rejection, match="another arm goal"):
        layer._admit_arm_move((0.2,) * 5, 0.0)
    with pytest.raises(Rejection, match="another gripper goal"):
        layer._admit_gripper_move(0.5, 0.0)


async def test_failed_plan_completes_as_failure():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    plan = layer._admit_arm_move((0.1,) * 5, 0.0)
    coordinator._fail_arm_plan(plan, "follower state went stale mid-goal")
    ctx = FakeGoalContext()
    await layer.drive_arm(ctx, plan)
    success, message, _positions, _action_time = ctx.completed
    assert success is False
    assert "went stale mid-goal" in message
    assert coordinator.try_claim_arm()


async def test_none_plan_completes_as_failure_and_leaves_the_server_alive():
    _coordinator, layer = make_layer()
    ctx = FakeGoalContext()
    await layer.drive_arm(ctx, None)
    success, message, positions, _action_time = ctx.completed
    assert success is False
    assert "walked back" in message
    assert len(positions) == 5


def test_admission_rejects_before_claiming():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    with pytest.raises(Rejection, match="joint position limit"):
        layer._admit_arm_move((9.0,) * 5, 0.0)
    # A rejected goal never left the slot held.
    assert coordinator.try_claim_arm()


def test_pose_admission_rejects_busy_before_solving():
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    layer._admit_arm_move((0.1,) * 5, 0.0)
    with pytest.raises(Rejection, match="another arm goal"):
        layer._admit_pose_move((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0)
    # The expensive verified solve never ran for the doomed goal.
    assert kinematics.solve_calls == []


def test_unknown_limb_names_are_refused():
    from peppygen.exposed_actions.limb_motion import move_arm_joints as module

    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    request = SimpleNamespace(
        data=SimpleNamespace(arm_name="left_arm", joint_positions=[0.0] * 5, duration_s=0.0)
    )
    decision = layer.decide_arm_joints(module, PendingSlot())(request)
    assert not decision.accepted
    assert "unknown arm_name" in decision.reason
    assert coordinator.try_claim_arm()


def test_decide_rejects_any_admission_surprise_without_holding_the_slot():
    from peppygen.exposed_actions.limb_motion import move_arm_joints as module

    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)

    class Boom:
        @property
        def data(self):
            raise RuntimeError("admission surprise")

    decision = layer.decide_arm_joints(module, PendingSlot())(Boom())
    assert not decision.accepted
    assert "admission failed" in decision.reason
    assert coordinator.try_claim_arm(), "no surprise may leave the slot held"


async def test_pose_goal_solves_in_drive_and_streams_a_plan():
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = layer._admit_pose_move((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0)
    assert isinstance(goal, PoseGoal)
    assert kinematics.solve_calls == [], "admission must not pay the solve"

    async def run_plan_to_completion():
        while True:
            coordinator.measured_joints.set((0.0,) * 5)
            sample = coordinator.arm_tick(time.monotonic())
            if sample is not None:
                coordinator.arm_published(True, sample)
                if coordinator._arm_plan is None:
                    return
            await asyncio.sleep(0.001)

    ctx = FakeGoalContext()
    runner = asyncio.create_task(run_plan_to_completion())
    await layer.drive_pose(ctx, goal, kinematics, kinematics)
    runner.cancel()
    success, _message, _position, _orientation, _t = ctx.completed
    assert success is True
    assert ("solve",) == tuple(c[0] for c in kinematics.solve_calls)
    assert coordinator.try_claim_arm()


async def test_pose_move_duration_floors_at_the_linear_ee_cap():
    # FakeKinematics FK sits at (0.1, 0.0, 0.2); the goal is 1.0 m away, so
    # at 0.5 m/s the quintic floor is 15/8 * 1.0 / 0.5, above the joint floor.
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics, max_ee_velocity_m_s=0.5)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = layer._admit_pose_move((1.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0)
    ctx = FakeGoalContext()
    task = asyncio.create_task(layer.drive_pose(ctx, goal, kinematics, kinematics))
    plan = await installed_plan(coordinator)
    assert plan.profile.duration_s == pytest.approx(15.0 / 8.0 * 1.0 / 0.5)
    coordinator.abort_arm_plan(plan)
    await task


async def test_pose_move_duration_floors_at_the_angular_ee_cap():
    # A half turn from the FK's identity orientation at 1 rad/s floors the
    # duration at 15/8 * pi.
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics, max_ee_angular_velocity_rad_s=1.0)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = layer._admit_pose_move((0.1, 0.0, 0.2), (0.0, 0.0, 1.0, 0.0), 0.0, 0.0, 0.0)
    ctx = FakeGoalContext()
    task = asyncio.create_task(layer.drive_pose(ctx, goal, kinematics, kinematics))
    plan = await installed_plan(coordinator)
    assert plan.profile.duration_s == pytest.approx(15.0 / 8.0 * math.pi)
    coordinator.abort_arm_plan(plan)
    await task


async def test_unreachable_pose_completes_failed_and_releases():
    kinematics = FakeKinematics()
    kinematics.corrupted = True  # solve returns None
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = layer._admit_pose_move((5.0, 5.0, 5.0), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0)
    ctx = FakeGoalContext()
    await layer.drive_pose(ctx, goal, kinematics, kinematics)
    success, message, _position, _orientation, _t = ctx.completed
    assert success is False
    assert "no solution reaches this pose" in message
    # The refusal must say the search was local, so a caller knows to retry
    # from elsewhere rather than read it as the pose being out of reach.
    assert "local descent" in message
    assert coordinator.try_claim_arm()


async def test_abandoned_pose_admission_releases_the_claim():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    pending = PendingSlot()
    pending.plan = layer._admit_pose_move((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0)
    layer.abandon(pending)
    assert coordinator.try_claim_arm()


def test_absurd_duration_is_refused():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    with pytest.raises(Rejection, match="must not exceed"):
        layer._admit_arm_move((0.1,) * 5, MAX_REQUESTED_DURATION_S + 1.0)
    assert coordinator.try_claim_arm()


def test_gripper_effort_preference_is_admitted_and_ignored():
    coordinator, layer = make_layer()
    coordinator.measured_gripper.set(0.0)
    plan = layer._admit_gripper_move(0.5, 3.0)
    assert plan.target == 0.5


async def test_pose_solver_surprise_still_reaches_a_terminal_result():
    class BoomKinematics(FakeKinematics):
        def inverse_kinematics(self, seed, position, orientation):
            raise RuntimeError("solver surprise")

    kinematics = BoomKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = layer._admit_pose_move((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0)
    ctx = FakeGoalContext()
    await layer.drive_pose(ctx, goal, kinematics, kinematics)
    success, message, _position, _orientation, _t = ctx.completed
    assert success is False
    assert "pose solve failed" in message
    assert coordinator.try_claim_arm()


IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def quat_about_x(degrees: float):
    half = math.radians(degrees) / 2.0
    return (math.sin(half), 0.0, 0.0, math.cos(half))


def test_a_pose_result_says_how_far_the_orientation_landed_from_the_request():
    # Five joints underactuate three rotational degrees of freedom, so a move
    # can reach its position and still be turned well away from the pose that
    # was asked for. Reporting a bare success would hide exactly that.
    note = _orientation_note(quat_about_x(44.0), IDENTITY_QUAT)
    assert "44 degrees" in note
    assert "underactuate" in note


def test_an_orientation_that_landed_where_it_was_asked_earns_no_caveat():
    # A caveat on every result is a caveat nobody reads.
    assert _orientation_note(IDENTITY_QUAT, IDENTITY_QUAT) == ""
    assert _orientation_note(quat_about_x(0.5), IDENTITY_QUAT) == ""


async def test_a_turned_landing_carries_its_caveat_into_the_pose_result():
    # The caveat is only worth computing if it reaches the operator, so drive a
    # real pose goal whose arm lands turned away from the request and read the
    # message the action actually completes with.
    class TurnedKinematics(FakeKinematics):
        def forward_kinematics(self, positions_rad):
            return (0.1, 0.0, 0.2), quat_about_x(44.0)

    kinematics = TurnedKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = layer._admit_pose_move((0.1, 0.0, 0.2), IDENTITY_QUAT, 0.0, 0.0, 0.0)

    async def run_plan_to_completion():
        while True:
            coordinator.measured_joints.set((0.0,) * 5)
            sample = coordinator.arm_tick(time.monotonic())
            if sample is not None:
                coordinator.arm_published(True, sample)
                if coordinator._arm_plan is None:
                    return
            await asyncio.sleep(0.001)

    ctx = FakeGoalContext()
    runner = asyncio.create_task(run_plan_to_completion())
    await layer.drive_pose(ctx, goal, kinematics, kinematics)
    runner.cancel()
    success, message, _position, _orientation, _t = ctx.completed
    assert success is True
    assert "44 degrees" in message


async def test_the_callers_slack_reaches_the_solver():
    # The bug this guards: move_arm carried both tolerance fields and the
    # node dropped them, so a caller's stated slack changed nothing.
    kinematics = FakeKinematics()
    # Refused rather than solved: the bars are recorded before the solve
    # answers, and a refusal terminates without a control loop to run a plan.
    kinematics.corrupted = True
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = layer._admit_pose_move((0.1, 0.0, 0.2), IDENTITY_QUAT, 0.0, 0.004, 0.05)
    assert goal.position_tolerance_m == 0.004
    assert goal.orientation_tolerance_rad == 0.05
    await layer.drive_pose(FakeGoalContext(), goal, kinematics, kinematics)
    assert kinematics.bars == [(0.004, 0.05)]


async def test_a_wire_zero_asks_for_the_planners_own_bars():
    kinematics = FakeKinematics()
    # Refused rather than solved: the bars are recorded before the solve
    # answers, and a refusal terminates without a control loop to run a plan.
    kinematics.corrupted = True
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = layer._admit_pose_move((0.1, 0.0, 0.2), IDENTITY_QUAT, 0.0, 0.0, 0.0)
    assert goal.position_tolerance_m is None
    assert goal.orientation_tolerance_rad is None
    await layer.drive_pose(FakeGoalContext(), goal, kinematics, kinematics)
    assert kinematics.bars == [(None, None)]


@pytest.mark.parametrize("bad", [-0.001, math.nan, math.inf])
async def test_an_unusable_tolerance_is_rejected_rather_than_defaulted(bad):
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    with pytest.raises(Rejection):
        layer._admit_pose_move((0.1, 0.0, 0.2), IDENTITY_QUAT, 0.0, bad, 0.0)
    with pytest.raises(Rejection):
        layer._admit_pose_move((0.1, 0.0, 0.2), IDENTITY_QUAT, 0.0, 0.0, bad)
    assert coordinator.try_claim_arm(), "a rejected tolerance must not hold the slot"


async def test_a_refused_tolerance_names_the_bar_it_was_judged_against():
    kinematics = FakeKinematics()
    kinematics.corrupted = True
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = layer._admit_pose_move((0.1, 0.0, 0.2), IDENTITY_QUAT, 0.0, 0.004, 0.05)
    ctx = FakeGoalContext()
    await layer.drive_pose(ctx, goal, kinematics, kinematics)
    success, message, _p, _o, _t = ctx.completed
    assert success is False
    assert "4 mm" in message and "3 deg" in message


async def test_the_ee_caps_cannot_pin_the_slot_past_the_duration_ceiling():
    # A move admitted under the ceiling can still need longer than it once
    # the end-effector speed caps are applied; that must refuse, not stretch.
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics, max_ee_velocity_m_s=1e-4)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = layer._admit_pose_move((5.0, 0.0, 0.2), IDENTITY_QUAT, 0.0, 0.0, 0.0)
    ctx = FakeGoalContext()
    await layer.drive_pose(ctx, goal, kinematics, kinematics)
    success, message, _p, _o, _t = ctx.completed
    assert success is False
    assert "ceiling" in message
    assert coordinator.try_claim_arm(), "a refused move must release the arm"
