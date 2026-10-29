"""Action-layer behavior that the wire cannot provoke deterministically:
joint paths, how they are timed and where they rest, settles and their
timeouts, walked-back admissions, per-server pending isolation, limb naming,
refusals and failures with their kinds, and how each goal ends."""

import asyncio
import itertools
import math
import threading
import time
from types import SimpleNamespace

import pytest
from conftest import (
    MEASURED,
    START_TOLERANCE_RAD,
    FakeKinematics,
    gripper_step,
    joint_path_goal,
    make_config,
    make_layer,
    path_result_joints,
)
from control_core_py import joint_path, minimum_jerk
from control_core_py.joint_path import JOINT_PATH_CORNER_DEVIATION, MIN_JOINT_CHANGE_RAD
from peppygen.exposed_actions.limb_motion import execute_joint_path
from so101_description import limits as limits_mod
from so101_description.kinematics import Kinematics
from so101_description.model import KINEMATICS_URDF_PATH
from so101_description.postures import READY_POSITIONS_RAD
from so101_description.staleness import COMMAND_SILENCE_LIMIT_S

from so101_backbone.actions import (
    ARM_WORDS,
    GRIPPER_WORDS,
    MAX_REQUESTED_DURATION_S,
    WALKED_BACK,
    JointPathMove,
    PendingSlot,
    PoseMove,
    drive_taking,
)
from so101_backbone.coordinator import (
    FOLLOWER_SILENT,
    GRIPPER_STILL_WINDOW_S,
)
from so101_backbone.failures import FailureKind, Refusal

# The reason the generated goal context gives for each cause of a cancel
# signal, in peppy's words.
CANCEL_REASONS = {"caller": "cancelled by its caller", "sender_gone": "its sender is gone"}
# The reason of a goal that its caller cancelled.
CANCELLED_BY_CALLER = CANCEL_REASONS["caller"]

# The bars of the moves below, where a test sets no others: the plan and the
# settle of a pose move each allow 1 cm and 0.2 rad, a joint move 0.1 rad per
# joint, a gripper move 1 % of the jaw travel, and each limb has 2 s past the
# planned end to settle.
PLAN = (0.01, 0.2)
SETTLE = (0.01, 0.2, 2.0)
JOINT_TOLERANCES = (0.1,) * 5

# A path from MEASURED through four waypoints inside the joint limits, each
# leg turning by a right angle from the one before it.
CORNERED = (
    (0.5, 0.1, 0.2, 0.3, 0.4),
    (0.5, 0.6, 0.2, 0.3, 0.4),
    (0.0, 0.6, 0.7, 0.3, 0.4),
    (0.0, 0.1, 0.7, 0.8, 0.4),
)


def admit_joints(layer, target, duration_s=0.0, tolerances=JOINT_TOLERANCES, timeout_s=2.0):
    return layer._admit_arm_move(target, duration_s, tolerances, timeout_s)


def admit_path(layer, *waypoints, **fields):
    """Admit the execute_joint_path goal that joint_path_goal builds of
    `waypoints` and `fields`."""
    goal = joint_path_goal(*waypoints, **fields)
    return layer._admit_joint_path(goal.arms, goal.duration_s, goal.settle_timeout_s)


def admit_pose(layer, position, orientation, duration_s=0.0, plan=PLAN, settle=SETTLE):
    return layer._admit_pose_move(position, orientation, duration_s, *plan, *settle)


# The gripper moves below wait up to 2 s past the planned end of their travel
# for the jaws to stand still, where a test sets no other settle timeout.
GRIPPER_SETTLE_TIMEOUT_S = 2.0


def admit_gripper(layer, opening, max_effort=0.0, settle_timeout_s=GRIPPER_SETTLE_TIMEOUT_S):
    return layer._admit_gripper_move(opening, max_effort, settle_timeout_s)


class FakeGoalContext:
    """Records how the goal ended: its result when it completed, its kind and
    message when it failed, its reason and result when it ended cancelled.
    The cancel signal never fires unless asked."""

    def __init__(self):
        self.completed = None
        self.failed = None
        self.cancelled = None
        self._cancel = asyncio.Event()
        self._cause = None

    def request_cancel(self, cause="caller"):
        self._cause = cause
        self._cancel.set()

    async def cancel_signal(self):
        await self._cancel.wait()

    def cancel_cause(self):
        return self._cause

    def cancel_reason(self):
        return CANCEL_REASONS.get(self._cause)

    async def complete(self, *result):
        self.completed = result

    async def fail(self, kind, message):
        self.failed = (kind, message)

    async def complete_cancelled(self, reason, *result):
        self.cancelled = (reason, *result)


def refusal_of(raised) -> tuple[FailureKind, str]:
    """The kind and the reason of a Refusal that pytest.raises caught."""
    return raised.value.kind, str(raised.value)


async def installed_plan(coordinator):
    async def poll():
        # The coordinator installs a plan synchronously and owns no event to
        # wait on; adding one to production code for a test's benefit would
        # be the wrong trade, so this polls under a timeout.
        while coordinator._arm.plan is None:  # noqa: ASYNC110
            await asyncio.sleep(0.001)
        return coordinator._arm.plan

    return await asyncio.wait_for(poll(), timeout=2.0)


async def driving_path(coordinator, layer, move):
    """Start the drive of the admitted joint path `move`: the goal context,
    the drive task, which ends the goal when it runs next, and the plan the
    drive installed."""
    ctx = FakeGoalContext()
    task = asyncio.create_task(layer.drive_joint_path(ctx, move))
    return ctx, task, await installed_plan(coordinator)


async def drive_path_cancelled_at_once(layer, move) -> FakeGoalContext:
    """Drive the admitted joint path `move` for a goal that its caller
    cancels before the drive starts, and give the goal's context. A drive
    that ends the goal before it installs a plan ends it as for any goal; a
    drive that installs a plan ends it cancelled at once, with no control
    tick to wait for."""
    ctx = FakeGoalContext()
    ctx.request_cancel()
    await layer.drive_joint_path(ctx, move)
    return ctx


def run_gripper_plan_to_its_end(coordinator, plan, measured_at_end):
    """The control ticks that complete an admitted gripper plan from fully
    open: ticks at the control period until the command lands on the target,
    the next tick finds it there, the planned end, with the follower
    delivering `measured_at_end`, and a delivery of it a still window later
    completes the plan on that tick's delivery."""
    tick = 0
    while gripper_step(coordinator, 1.0, round(tick * 0.01, 2)) != plan.target:
        tick += 1
    planned_end = round((tick + 1) * 0.01, 2)
    gripper_step(coordinator, measured_at_end, planned_end)
    assert not plan.done.is_set(), "the jaws stand still only over a window"
    gripper_step(coordinator, measured_at_end, planned_end + GRIPPER_STILL_WINDOW_S + 0.01)
    assert plan.done.is_set()


def run_arm_plan_to_its_end(coordinator, plan):
    """Run the control ticks that end an admitted arm plan whose settle the
    follower meets, at explicit instants. The first sample is sent at 0 and
    the last sample at the end of the profile, and the follower receives the
    two. The measured joints are the caller's."""
    coordinator.arm_published(True, coordinator.arm_tick(0.0))
    coordinator.arm_published(True, coordinator.arm_tick(plan.profile.duration_s))
    assert plan.done.is_set()


async def test_a_gripper_move_completes_once_the_jaws_stand_still_past_the_planned_end(
    follower_never_stale,
):
    coordinator, layer = make_layer(max_gripper_rate_frac_s=1.0)
    coordinator.measured_gripper.set(1.0)
    plan = admit_gripper(layer, 0.5)
    assert plan.settle_timeout_s == 2.0
    # From fully open to 0.5 at 1.0 per second the command lands on the 50th
    # tick, at 0.49 s, and the tick after finds it there: the planned end,
    # at 0.50 s. The jaws stand at 0.52 from 0.25 s on, before it: they are
    # judged from the planned end on, so the move completes a still window
    # after it, at 0.75 s, and not before.
    for tick in range(75):
        now = round(tick * 0.01, 2)
        assert gripper_step(coordinator, 1.0 if now < 0.25 else 0.52, now) is not None
        assert not plan.done.is_set()
    assert gripper_step(coordinator, 0.52, 0.75) is not None
    assert plan.done.is_set()
    ctx = FakeGoalContext()
    await layer.drive_gripper(ctx, plan)
    settled, opening, _action_time = ctx.completed
    assert (settled, opening) == (True, 0.52)
    assert coordinator.try_claim_gripper()


async def test_a_close_onto_an_item_completes_once_the_jaws_stand_still_where_it_holds_them(
    follower_never_stale,
):
    coordinator, layer = make_layer(max_gripper_rate_frac_s=1.0)
    coordinator.measured_gripper.set(1.0)
    plan = admit_gripper(layer, 0.5)
    # An item holds the jaws at 0.8 on their way to 0.5: they stand still
    # there, the move completes, and final_opening is where the item holds
    # them.
    run_gripper_plan_to_its_end(coordinator, plan, measured_at_end=0.8)
    ctx = FakeGoalContext()
    await layer.drive_gripper(ctx, plan)
    settled, opening, _action_time = ctx.completed
    assert (settled, opening) == (True, 0.8)
    assert coordinator.try_claim_gripper()


async def test_the_settle_timeout_completes_a_move_whose_jaws_keep_moving(follower_never_stale):
    coordinator, layer = make_layer(max_gripper_rate_frac_s=1.0)
    coordinator.measured_gripper.set(1.0)
    plan = admit_gripper(layer, 0.5, settle_timeout_s=2.0)
    # The command lands on the 50th tick, at 0.49 s, and the tick after is
    # the planned end. Each delivery from then on lands 0.1 from the one
    # before, farther than the still band: the jaws never stand still, and
    # 2.0 s after the planned end the move completes wherever they are, not
    # settled.
    for tick in range(50):
        gripper_step(coordinator, 1.0, round(tick * 0.01, 2))
    for now, opening in ((0.5, 0.9), (1.0, 0.8), (1.5, 0.7), (2.0, 0.6), (2.49, 0.5)):
        gripper_step(coordinator, opening, now)
        assert not plan.done.is_set()
    gripper_step(coordinator, 0.4, 2.5)
    assert plan.done.is_set()
    ctx = FakeGoalContext()
    await layer.drive_gripper(ctx, plan)
    settled, opening, _action_time = ctx.completed
    assert (settled, opening) == (False, 0.4)
    assert coordinator.try_claim_gripper()


async def test_a_zero_settle_timeout_completes_the_gripper_move_at_its_planned_end(
    follower_never_stale,
):
    coordinator, layer = make_layer(max_gripper_rate_frac_s=1.0)
    coordinator.measured_gripper.set(1.0)
    plan = admit_gripper(layer, 0.5, settle_timeout_s=0.0)
    # The command lands on the 50th tick, at 0.49 s, with the jaws at 0.7;
    # the tick after is the planned end, and the timeout of 0 completes the
    # move there, not settled.
    for tick in range(50):
        gripper_step(coordinator, 1.0 if tick < 25 else 0.7, round(tick * 0.01, 2))
        assert not plan.done.is_set()
    gripper_step(coordinator, 0.55, 0.5)
    assert plan.done.is_set()
    ctx = FakeGoalContext()
    await layer.drive_gripper(ctx, plan)
    settled, opening, _action_time = ctx.completed
    assert (settled, opening) == (False, 0.55)


@pytest.mark.parametrize("bad", [-0.001, math.nan, math.inf, MAX_REQUESTED_DURATION_S + 0.001])
def test_a_gripper_settle_timeout_that_is_no_duration_is_refused(bad):
    coordinator, layer = make_layer()
    coordinator.measured_gripper.set(0.5)
    with pytest.raises(
        Refusal, match="settle_timeout_s must be a number of seconds from 0 to 60"
    ) as raised:
        admit_gripper(layer, 0.2, settle_timeout_s=bad)
    assert raised.value.kind is FailureKind.REJECTED
    assert coordinator.try_claim_gripper(), "a refused settle timeout must not hold the slot"
    coordinator.release_gripper()
    assert admit_gripper(layer, 0.2, settle_timeout_s=MAX_REQUESTED_DURATION_S) is not None


async def test_cancel_during_the_gripper_travel_ends_the_goal_cancelled(follower_never_stale):
    coordinator, layer = make_layer(max_gripper_rate_frac_s=1.0)
    coordinator.measured_gripper.set(1.0)
    plan = admit_gripper(layer, 0.5)
    # The first sample is out and the jaws are still on their way.
    gripper_step(coordinator, 1.0, 0.0)
    gripper_step(coordinator, 0.9, 0.25)
    ctx = FakeGoalContext()
    ctx.request_cancel()
    await layer.drive_gripper(ctx, plan)
    assert ctx.completed is None
    reason, settled, opening, _action_time = ctx.cancelled
    assert (reason, settled, opening) == (CANCELLED_BY_CALLER, False, 0.9)
    assert plan.aborted
    assert coordinator.try_claim_gripper()


async def test_cancel_after_the_gripper_move_ended_reports_the_finished_move(
    follower_never_stale,
):
    # The plan ended before the cancel could be served: the goal reports the
    # move it finished rather than a cancel that stopped nothing.
    coordinator, layer = make_layer(max_gripper_rate_frac_s=1.0)
    coordinator.measured_gripper.set(1.0)
    plan = admit_gripper(layer, 0.5)
    run_gripper_plan_to_its_end(coordinator, plan, measured_at_end=0.5)
    ctx = FakeGoalContext()
    ctx.request_cancel()
    await layer.drive_gripper(ctx, plan)
    assert ctx.cancelled is None
    settled, opening, _action_time = ctx.completed
    assert (settled, opening) == (True, 0.5)


async def test_abandon_walks_back_a_claimed_admission():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    pending = PendingSlot()
    pending.plan = admit_joints(layer, (0.1,) * 5, 0.0)
    # The accept failing on the wire is the case abandon exists for.
    plan = pending.plan
    layer.abandon(pending)
    assert plan.aborted
    assert coordinator.try_claim_arm(), "the arm slot must be released"


async def test_abandon_on_one_server_cannot_touch_anothers_admission():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    arm_pending = PendingSlot()
    arm_pending.plan = admit_joints(layer, (0.1,) * 5, 0.0)
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
    admit_joints(layer, (0.1,) * 5, 0.0)
    # An arm move in flight must not block a gripper goal, and vice versa.
    plan = admit_gripper(layer, 1.0, 0.0)
    assert plan is not None
    with pytest.raises(Refusal, match="arm is already executing a motion") as raised:
        admit_joints(layer, (0.2,) * 5, 0.0)
    assert raised.value.kind is FailureKind.PRECONDITION_FAILED
    with pytest.raises(Refusal, match="gripper is already executing a move") as raised:
        admit_gripper(layer, 0.5, 0.0)
    assert raised.value.kind is FailureKind.PRECONDITION_FAILED


def test_a_posture_does_not_claim_or_drive_the_gripper(follower_never_stale):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set(MEASURED)
    coordinator.measured_gripper.set(0.3)
    plan = layer._admit_posture(READY_POSITIONS_RAD, 0.0)
    # From its first sample to its last, the posture sends the gripper
    # nothing. The gripper stays free for its own goals.
    for now in (0.0, plan.profile.duration_s):
        coordinator.arm_published(True, coordinator.arm_tick(now))
        assert coordinator.gripper_tick(now) is None
    assert plan.done.is_set()
    assert coordinator.try_claim_gripper() is True


async def test_a_failed_plan_fails_the_goal_with_its_kind_and_no_result():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    plan = admit_joints(layer, (0.1,) * 5, 0.0)
    coordinator._fail_arm_plan(plan, FOLLOWER_SILENT)
    ctx = FakeGoalContext()
    await layer.drive_arm(ctx, plan)
    assert ctx.failed == (FailureKind.RESOURCE_UNAVAILABLE, "the follower stopped reporting")
    assert ctx.completed is None and ctx.cancelled is None
    assert coordinator.try_claim_arm()


async def test_a_failed_path_plan_fails_the_goal_with_its_kind_and_no_result(follower_never_stale):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    ctx, task, plan = await driving_path(coordinator, layer, admit_path(layer, (0.1,) * 5))
    coordinator._fail_arm_plan(plan, FOLLOWER_SILENT)
    await task
    assert ctx.failed == (FailureKind.RESOURCE_UNAVAILABLE, "the follower stopped reporting")
    assert ctx.completed is None and ctx.cancelled is None
    assert coordinator.try_claim_arm()


async def test_a_walked_back_admission_fails_the_goal_and_leaves_the_server_alive():
    _coordinator, layer = make_layer()
    ctx = FakeGoalContext()
    await drive_taking(PendingSlot(), layer.drive_arm)(ctx)
    assert ctx.failed == (FailureKind.ATTEMPT_FAILED, WALKED_BACK)
    assert ctx.completed is None and ctx.cancelled is None


async def test_an_arm_move_completes_once_the_measured_joints_are_within_their_tolerances(
    follower_never_stale,
):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    plan = admit_joints(layer, (0.5,) * 5, 0.0)
    end_s = plan.profile.duration_s
    coordinator.arm_published(True, coordinator.arm_tick(0.0))
    # At the planned end the follower still measures the start: the plan
    # keeps commanding its end sample.
    assert coordinator.arm_tick(end_s) == (0.5,) * 5
    coordinator.arm_published(True, (0.5,) * 5)
    assert not plan.done.is_set()
    # The arm arrives within 0.1 rad on every joint: the plan completes on
    # the delivery of that tick's end sample, and the result reports the
    # joints measured then.
    arrived = (0.45, 0.55, 0.5, 0.42, 0.58)
    coordinator.measured_joints.set(arrived)
    coordinator.arm_published(True, coordinator.arm_tick(end_s + 0.5))
    assert plan.done.is_set()
    ctx = FakeGoalContext()
    await layer.drive_arm(ctx, plan)
    positions, _action_time = ctx.completed
    assert positions == list(arrived)


async def test_a_joint_path_completes_once_the_measured_joints_are_within_their_tolerances(
    follower_never_stale,
):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    ctx, task, plan = await driving_path(coordinator, layer, admit_path(layer, (0.5,) * 5))
    end_s = plan.profile.duration_s
    coordinator.arm_published(True, coordinator.arm_tick(0.0))
    # At the planned end the follower still measures the start: the plan
    # keeps commanding its end sample.
    assert coordinator.arm_tick(end_s) == (0.5,) * 5
    coordinator.arm_published(True, (0.5,) * 5)
    assert not plan.done.is_set()
    # The arm arrives within 0.1 rad on every joint: the plan completes on
    # the delivery of that tick's end sample, and the result reports the
    # joints measured then.
    arrived = (0.45, 0.55, 0.5, 0.42, 0.58)
    coordinator.measured_joints.set(arrived)
    coordinator.arm_published(True, coordinator.arm_tick(end_s + 0.5))
    assert plan.done.is_set()
    await task
    arms, _action_time = ctx.completed
    assert path_result_joints(arms) == list(arrived)


async def test_an_arm_move_outside_its_tolerances_at_the_timeout_fails_with_timeout(
    follower_never_stale,
):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    plan = admit_joints(layer, (0.5,) * 5, 0.0, timeout_s=2.0)
    end_s = plan.profile.duration_s
    coordinator.arm_published(True, coordinator.arm_tick(0.0))
    # The follower never moves. Through the timeout the plan keeps
    # commanding its end sample; the tick the timeout runs out publishes
    # nothing and fails the plan.
    for since_end_s in (0.0, 1.0, 1.9):
        assert coordinator.arm_tick(end_s + since_end_s) == (0.5,) * 5
        coordinator.arm_published(True, (0.5,) * 5)
        assert not plan.done.is_set()
    assert coordinator.arm_tick(end_s + 2.0) is None
    assert plan.done.is_set()
    ctx = FakeGoalContext()
    await layer.drive_arm(ctx, plan)
    # The message says how far the arm still stands from the target and how
    # long after the planned end. A failed goal gives no result.
    assert ctx.failed == (
        FailureKind.TIMEOUT,
        "still 0.500 rad from the target at joint 1, 2.0 s after the planned end",
    )
    assert ctx.completed is None
    assert coordinator.try_claim_arm()


async def test_a_joint_path_outside_its_tolerances_at_the_timeout_fails_with_timeout(
    follower_never_stale,
):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    # A duration of 0.5 s, past the 0.3 s the caps need, puts the planned end
    # and each tick below on an instant a float holds exactly.
    ctx, task, plan = await driving_path(
        coordinator, layer, admit_path(layer, (0.5,) * 5, duration_s=0.5, settle_timeout_s=2.0)
    )
    end_s = plan.profile.duration_s
    assert end_s == 0.5
    coordinator.arm_published(True, coordinator.arm_tick(0.0))
    # The follower never moves. Through the timeout the plan keeps
    # commanding its end sample; the tick the timeout runs out publishes
    # nothing and fails the plan.
    for since_end_s in (0.0, 1.0, 1.9):
        assert coordinator.arm_tick(end_s + since_end_s) == (0.5,) * 5
        coordinator.arm_published(True, (0.5,) * 5)
        assert not plan.done.is_set()
    assert coordinator.arm_tick(end_s + 2.0) is None
    assert plan.done.is_set()
    await task
    # The message says how far the arm still stands from the target and how
    # long after the planned end. A failed goal gives no result.
    assert ctx.failed == (
        FailureKind.TIMEOUT,
        "still 0.500 rad from the target at joint 1, 2.0 s after the planned end",
    )
    assert ctx.completed is None
    assert coordinator.try_claim_arm()


async def test_a_stale_follower_reports_the_commanded_joints(follower_never_stale):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    plan = admit_joints(layer, (0.5,) * 5, 0.0)
    # The follower arrives on the target as the plan ends.
    coordinator.measured_joints.set((0.5,) * 5)
    run_arm_plan_to_its_end(coordinator, plan)
    # The follower sends no more state after it receives the last sample.
    # The last command of a finished plan is its end sample.
    coordinator.measured_joints.clear()
    ctx = FakeGoalContext()
    await layer.drive_arm(ctx, plan)
    positions, _action_time = ctx.completed
    assert positions == [0.5] * 5


async def test_a_stale_follower_reports_the_commanded_joints_of_a_path(follower_never_stale):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    ctx, task, plan = await driving_path(coordinator, layer, admit_path(layer, (0.5,) * 5))
    # The follower arrives on the target as the plan ends.
    coordinator.measured_joints.set((0.5,) * 5)
    run_arm_plan_to_its_end(coordinator, plan)
    # The follower sends no more state after it receives the last sample.
    # The last command of a finished plan is its end sample.
    coordinator.measured_joints.clear()
    await task
    arms, _action_time = ctx.completed
    assert path_result_joints(arms) == [0.5] * 5


async def test_a_posture_result_reports_the_grasp_point_of_the_measured_joints(
    follower_never_stale,
):
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set(MEASURED)
    plan = layer._admit_posture(READY_POSITIONS_RAD, 0.0)
    # The plan ends. The follower measures the arm at its start position.
    run_arm_plan_to_its_end(coordinator, plan)
    ctx = FakeGoalContext()
    await layer.drive_posture(ctx, plan, kinematics)
    position, orientation = kinematics.forward_kinematics(MEASURED)
    assert position != kinematics.forward_kinematics(READY_POSITIONS_RAD)[0]
    assert ctx.completed == (["arm"], list(position), list(orientation))
    assert coordinator.try_claim_arm()


async def test_a_posture_with_stale_follower_state_reports_empty_arrays(follower_never_stale):
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set(MEASURED)
    plan = layer._admit_posture(READY_POSITIONS_RAD, 0.0)
    run_arm_plan_to_its_end(coordinator, plan)
    # The follower sends no more state after it receives the last sample.
    # No measurement is fresh, thus the result gives no pose.
    coordinator.measured_joints.clear()
    ctx = FakeGoalContext()
    await layer.drive_posture(ctx, plan, kinematics)
    assert ctx.completed == ([], [], [])


def stop_every_move(layer, _ctx):
    assert layer.stop_moves("the operator asked") == ["arm"]


def cancel_the_goal(_layer, ctx):
    ctx.request_cancel()


@pytest.mark.parametrize(
    ("cut_the_move", "reason"),
    [(stop_every_move, "stopped: the operator asked"), (cancel_the_goal, CANCELLED_BY_CALLER)],
    ids=["stop", "cancel"],
)
async def test_a_stopped_posture_reports_the_measured_grasp_point(
    cut_the_move, reason, follower_never_stale
):
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set(MEASURED)
    plan = layer._admit_posture(READY_POSITIONS_RAD, 0.0)
    coordinator.arm_published(True, coordinator.arm_tick(0.0))
    # The move stops while the arm moves to the posture. The follower
    # measures the arm at that position.
    stopped_at = (0.0, 0.05, 0.1, 0.15, 0.2)
    coordinator.measured_joints.set(stopped_at)
    ctx = FakeGoalContext()
    cut_the_move(layer, ctx)
    await layer.drive_posture(ctx, plan, kinematics)
    position, orientation = kinematics.forward_kinematics(stopped_at)
    assert ctx.completed is None
    assert ctx.cancelled == (reason, ["arm"], list(position), list(orientation))
    assert plan.aborted
    assert coordinator.try_claim_arm()


def test_admission_refuses_before_claiming():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    with pytest.raises(Refusal, match="target out of joint limits") as raised:
        admit_joints(layer, (9.0,) * 5, 0.0)
    assert raised.value.kind is FailureKind.REJECTED
    # A refused goal never left the slot held.
    assert coordinator.try_claim_arm()


UNKNOWN_ARM = 'unknown arm_name: this robot\'s arm is "arm"'
ONE_ARM_ONCE = 'arms must name this robot\'s arm, "arm", once'


@pytest.mark.parametrize(
    ("fields", "refusal"),
    [
        ({"arm_names": ("left_arm",)}, UNKNOWN_ARM),
        ({"arm_names": ("arm", "left_arm")}, UNKNOWN_ARM),
        ({"arm_names": ("arm", "arm")}, ONE_ARM_ONCE),
        ({"arm_names": ()}, ONE_ARM_ONCE),
        ({"waypoints": ()}, "waypoints is empty: give at least one waypoint"),
        ({"waypoints": ((0.1,) * 5, (9.0,) * 5)}, "waypoint 2 out of joint limits"),
        (
            {"waypoints": ((0.1, 0.1, math.nan, 0.1, 0.1),)},
            "waypoint 1 has a non-finite joint position",
        ),
        (
            {"waypoints": ((0.1, 0.1, 0.1, 0.1, math.inf),)},
            "waypoint 1 has a non-finite joint position",
        ),
        (
            {"expected_start": (0.0, 0.0, 0.0, math.nan, 0.0)},
            "expected_start_joint_positions has a non-finite joint position",
        ),
        (
            {"expected_start": (0.0,) * 4},
            "expected_start_joint_positions has 4 joint positions: give one per joint of the "
            "arm, 5, or none to skip the check",
        ),
        (
            {"duration_s": MAX_REQUESTED_DURATION_S + 1.0},
            "duration_s must be a number of seconds from 0 to 60",
        ),
        (
            {"tolerances": (0.1, 0.1, 0.0, 0.1, 0.1)},
            "each settle_tolerances_rad entry must be a positive number",
        ),
        (
            {"settle_timeout_s": -0.001},
            "settle_timeout_s must be a number of seconds from 0 to 60",
        ),
    ],
    ids=[
        "unknown arm",
        "unknown arm beside the arm",
        "repeated arm",
        "no arm",
        "no waypoint",
        "outside the limits",
        "nan joint",
        "infinite joint",
        "nan expected start",
        "short expected start",
        "absurd duration",
        "zero settle tolerance",
        "negative settle timeout",
    ],
)
def test_a_joint_path_goal_this_robot_does_not_take_is_rejected(fields, refusal):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    waypoints = fields.get("waypoints", ((0.1,) * 5,))
    options = {field: value for field, value in fields.items() if field != "waypoints"}
    with pytest.raises(Refusal) as raised:
        admit_path(layer, *waypoints, **options)
    assert refusal_of(raised) == (FailureKind.REJECTED, refusal)
    assert coordinator.try_claim_arm(), "a refused goal never holds the slot"


@pytest.mark.parametrize("requested_s", [0.0, 3.0])
async def test_a_path_keeps_every_joint_within_its_velocity_and_acceleration_caps(
    requested_s, follower_never_stale
):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set(MEASURED)
    _ctx, task, plan = await driving_path(
        coordinator, layer, admit_path(layer, *CORNERED, duration_s=requested_s)
    )
    config = make_config()
    velocity_caps = config.max_joint_velocity_rad_s
    acceleration_caps = config.max_joint_acceleration_rad_s2
    # The path runs in the time its timing within the caps gives, or the
    # longer time the goal requests.
    fastest_s = joint_path.plan(
        (MEASURED, *CORNERED), 0.0, velocity_caps, acceleration_caps
    ).duration_s
    assert plan.profile.duration_s == max(requested_s, fastest_s)
    # Over a grid of instants, each joint's mean velocity and acceleration
    # between them stays within its cap.
    steps = 4000
    h = plan.profile.duration_s / steps
    samples = [plan.profile.sample(k * h) for k in range(steps + 1)]

    def rates(values):
        return [
            tuple((b - a) / h for a, b in zip(before, after, strict=True))
            for before, after in itertools.pairwise(values)
        ]

    velocities = rates(samples)
    for velocity in velocities:
        assert all(abs(v) <= cap * (1.0 + 1e-9) for v, cap in zip(velocity, velocity_caps))
    for acceleration in rates(velocities):
        assert all(
            abs(a) <= cap * (1.0 + 1e-6) for a, cap in zip(acceleration, acceleration_caps)
        )
    coordinator.abort_arm_plan(plan)
    await task


async def test_a_path_passes_its_inner_waypoints_without_stopping(follower_never_stale):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set(MEASURED)
    _ctx, task, plan = await driving_path(coordinator, layer, admit_path(layer, *CORNERED))
    assert all(cell.speed_in > 0.0 for cell in plan.profile.timed.cells[1:]), (
        "the path rests at its end alone"
    )
    # The control ticks of the whole move, at explicit instants; the follower
    # stays where the path started.
    ticks = 400
    trace = []
    for k in range(ticks + 1):
        command = coordinator.arm_tick(plan.profile.duration_s * k / ticks)
        coordinator.arm_published(True, command)
        trace.append(command)
    mean_step = plan.profile.path.length / ticks
    for inner in CORNERED[:-1]:
        nearest = min(range(1, ticks), key=lambda k: math.dist(trace[k], inner))
        # The arm cuts the corner within the contract's deviation, and it is
        # still moving there: its step is no fraction of the mean.
        assert math.dist(trace[nearest], inner) <= JOINT_PATH_CORNER_DEVIATION + mean_step
        assert math.dist(trace[nearest - 1], trace[nearest + 1]) / 2.0 > 0.25 * mean_step
    assert trace[-1] == CORNERED[-1]
    coordinator.abort_arm_plan(plan)
    await task


@pytest.mark.parametrize(
    "offset_rad", [0.0, 0.0009], ids=["exact repeat", "repeat within 0.001 rad"]
)
async def test_a_repeated_waypoint_stops_the_arm_on_it(offset_rad, follower_never_stale):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set(MEASURED)
    corner = CORNERED[0]
    # The waypoint after the corner repeats it: on it, or within
    # MIN_JOINT_CHANGE_RAD of it in every joint.
    repeat = tuple(p + offset_rad for p in corner)
    assert all(abs(r - c) < MIN_JOINT_CHANGE_RAD for r, c in zip(repeat, corner, strict=True))
    _ctx, task, plan = await driving_path(
        coordinator, layer, admit_path(layer, corner, repeat, *CORNERED[1:])
    )
    # The path rests on the repeated waypoint and at its end: one boundary of
    # its timing inside the path is passed at rest.
    [at_rest] = [cell for cell in plan.profile.timed.cells[1:] if cell.speed_in == 0.0]
    at_rest_s = at_rest.time_s
    coordinator.arm_published(True, coordinator.arm_tick(0.0))
    assert coordinator.arm_tick(at_rest_s) == pytest.approx(repeat, abs=1e-12)
    # A ten-thousandth of a second to either side the arm has barely moved,
    # where at its 2 rad/s cap it moves 2e-4 rad: it stands still on the
    # waypoint.
    for t in (at_rest_s - 1e-4, at_rest_s + 1e-4):
        assert math.dist(plan.profile.sample(t), repeat) < 1e-6
    coordinator.abort_arm_plan(plan)
    await task


async def test_a_path_that_turns_back_rests_at_the_turn_and_runs_back(follower_never_stale):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set(MEASURED)
    # Out 0.5 rad on joint 1 and straight back: the path turns back on itself
    # at the first waypoint.
    turn = CORNERED[0]
    _ctx, task, plan = await driving_path(coordinator, layer, admit_path(layer, turn, MEASURED))
    # The path rests at the turn and at its end: one boundary of its timing
    # inside the path is passed at rest.
    [at_rest] = [cell for cell in plan.profile.timed.cells[1:] if cell.speed_in == 0.0]
    at_rest_s = at_rest.time_s
    coordinator.arm_published(True, coordinator.arm_tick(0.0))
    assert coordinator.arm_tick(at_rest_s) == pytest.approx(turn, abs=1e-12)
    # A ten-thousandth of a second to either side the arm has barely moved,
    # where at its 2 rad/s cap it moves 2e-4 rad: it stands still at the turn.
    for t in (at_rest_s - 1e-4, at_rest_s + 1e-4):
        assert math.dist(plan.profile.sample(t), turn) < 1e-6
    # Joint 1 rises to the turn and falls back to the start, and no other
    # joint moves.
    ticks = 400
    trace = [plan.profile.sample(plan.profile.duration_s * k / ticks) for k in range(ticks + 1)]
    rest_tick = max(k for k in range(ticks + 1) if plan.profile.duration_s * k / ticks <= at_rest_s)
    joint_1 = [q[0] for q in trace]
    assert all(a <= b for a, b in itertools.pairwise(joint_1[: rest_tick + 1]))
    assert all(a >= b for a, b in itertools.pairwise(joint_1[rest_tick:]))
    assert all(q[1:] == MEASURED[1:] for q in trace)
    assert trace[-1] == MEASURED
    coordinator.abort_arm_plan(plan)
    await task


async def test_a_path_sent_with_a_duration_above_its_floor_runs_that_long(follower_never_stale):
    # At 0.41 rad/s per joint the path through CORNERED takes some 4 s at full
    # speed. Sent with duration_s 10, it runs 10 s on the control ticks'
    # clock.
    velocity_caps = (0.41,) * 5
    coordinator, layer = make_layer(max_joint_velocity_rad_s=velocity_caps)
    coordinator.measured_joints.set(MEASURED)
    fastest_s = joint_path.plan(
        (MEASURED, *CORNERED),
        0.0,
        velocity_caps,
        make_config().max_joint_acceleration_rad_s2,
    ).duration_s
    assert 3.5 < fastest_s < 4.5
    ctx, task, plan = await driving_path(
        coordinator, layer, admit_path(layer, *CORNERED, duration_s=10.0)
    )
    assert plan.profile.duration_s == pytest.approx(10.0)
    # The follower stands on the last waypoint from the start: the move
    # completes at its planned end, 10 s after its first tick, and no tick
    # before ends it.
    coordinator.measured_joints.set(CORNERED[-1])
    for t in (0.0, fastest_s, 9.99):
        coordinator.arm_published(True, coordinator.arm_tick(t))
        assert not plan.done.is_set()
    coordinator.arm_published(True, coordinator.arm_tick(10.0))
    assert plan.done.is_set()
    await task
    arms, _action_time = ctx.completed
    assert path_result_joints(arms) == list(CORNERED[-1])


async def test_an_arm_away_from_the_expected_start_fails_the_goal_before_it_moves(
    follower_never_stale,
):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set(MEASURED)
    # Joint 3 stands 0.08 rad from where the caller expects it, past the
    # start tolerance; every other joint stands on it. Admission accepts the
    # goal, and its drive fails it.
    expected = (0.0, 0.1, 0.28, 0.3, 0.4)
    assert START_TOLERANCE_RAD == 0.05
    move = admit_path(layer, CORNERED[0], expected_start=expected)
    assert isinstance(move, JointPathMove)
    ctx = await drive_path_cancelled_at_once(layer, move)
    assert ctx.failed == (
        FailureKind.PRECONDITION_FAILED,
        "arm stands 0.080 rad from expected_start_joint_positions at joint 3, past the "
        "0.050 rad start tolerance; plan the path from where the arm stands",
    )
    assert ctx.completed is None and ctx.cancelled is None
    assert coordinator._arm.plan is None, "nothing was installed"
    assert coordinator.try_claim_arm(), "the failed goal released the arm"
    coordinator.release_arm()
    # Within the tolerance on every joint the path runs from where the arm
    # stands.
    _ctx, task, plan = await driving_path(
        coordinator,
        layer,
        admit_path(layer, CORNERED[0], expected_start=tuple(p + 0.04 for p in MEASURED)),
    )
    assert plan.profile.start == MEASURED
    coordinator.abort_arm_plan(plan)
    await task


async def test_a_goal_refused_for_its_expected_start_commands_no_hold(follower_never_stale):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set(MEASURED)
    # The follower received a setpoint 0.02 rad from where it measures each
    # joint, and the caller expects joint 3 0.08 rad away, past the start
    # tolerance. The control loop ticks while the failure goes out.
    coordinator.arm_published(True, tuple(p + 0.02 for p in MEASURED))
    commanded_while_failing = []

    class FailingWhileTheLoopTicks(FakeGoalContext):
        async def fail(self, kind, message):
            commanded_while_failing.append(coordinator.arm_tick(0.0))
            await super().fail(kind, message)

    ctx = FailingWhileTheLoopTicks()
    away = (0.0, 0.1, 0.28, 0.3, 0.4)
    await layer.drive_joint_path(ctx, admit_path(layer, CORNERED[0], expected_start=away))
    assert ctx.failed[0] == FailureKind.PRECONDITION_FAILED
    # That tick commands nothing: the follower keeps the setpoint it
    # received last, and the refused goal never moved it.
    assert commanded_while_failing == [None]


async def test_an_empty_expected_start_is_not_checked(follower_never_stale):
    coordinator, layer = make_layer()
    # The arm stands 0.5 rad from the target on joint 1. An empty expected
    # start names no joints, so the path runs from where the arm stands.
    coordinator.measured_joints.set(MEASURED)
    move = admit_path(layer, CORNERED[0], expected_start=())
    assert move.goal.expected_start is None
    ctx, task, plan = await driving_path(coordinator, layer, move)
    assert (plan.profile.start, plan.profile.end) == (MEASURED, CORNERED[0])
    assert ctx.failed is None
    coordinator.abort_arm_plan(plan)
    await task


@pytest.mark.parametrize("expected_start", [(), MEASURED], ids=["unchecked", "checked"])
async def test_a_path_starts_from_the_measured_joints_when_the_last_setpoint_differs(
    expected_start, follower_never_stale
):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set(MEASURED)
    # The follower received a setpoint 0.02 rad from where it measures each
    # joint, within the start tolerance: the arm's anchor is that setpoint.
    last_setpoint = tuple(p + 0.02 for p in MEASURED)
    coordinator.arm_published(True, last_setpoint)
    assert coordinator.arm_anchor() == last_setpoint
    _ctx, task, plan = await driving_path(
        coordinator, layer, admit_path(layer, *CORNERED, expected_start=expected_start)
    )
    # The path runs from the measured joints: its first sample is where the
    # follower measures the arm.
    assert plan.profile.start == MEASURED
    assert coordinator.arm_tick(0.0) == MEASURED
    coordinator.abort_arm_plan(plan)
    await task


@pytest.mark.parametrize(
    ("measured", "start"),
    [(MEASURED, MEASURED), ((1.03, 0.1, 0.2, 0.3, 0.4), (1.0, 0.1, 0.2, 0.3, 0.4))],
    ids=["inside the limits", "past a limit"],
)
async def test_the_arm_holds_where_it_is_measured_while_its_path_is_planned(
    measured, start, monkeypatch, follower_never_stale
):
    # Each joint's limits are -1 and 1 rad. The follower measures `measured`,
    # past the upper limit of joint 1 by 0.03 rad in the second case, and it
    # received a setpoint 0.02 rad from there in each joint.
    limits = limits_mod.JointLimits(lower=(-1.0,) * 5, upper=(1.0,) * 5)
    coordinator, layer = make_layer(limits=limits)
    coordinator.measured_joints.set(measured)
    coordinator.arm_published(True, tuple(p + 0.02 for p in measured))
    plan = joint_path.plan
    commanded_while_planning = []

    def plan_while_the_control_loop_ticks(*arguments):
        commanded_while_planning.append(coordinator.arm_tick(0.0))
        return plan(*arguments)

    monkeypatch.setattr(joint_path, "plan", plan_while_the_control_loop_ticks)
    _ctx, task, installed = await driving_path(coordinator, layer, admit_path(layer, *CORNERED))
    # While the path was planned the arm was commanded to hold where the
    # follower measured it, clamped into the joint limits, and the path
    # starts there.
    assert commanded_while_planning == [start]
    assert installed.profile.start == start
    coordinator.abort_arm_plan(installed)
    await task


async def test_an_arm_that_moves_while_its_path_is_planned_fails_the_goal_and_installs_nothing(
    monkeypatch, follower_never_stale
):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set(MEASURED)
    plan = joint_path.plan

    def plan_while_the_arm_moves(*arguments):
        # Joint 3 moves 0.08 rad while the path is planned, past the start
        # tolerance.
        coordinator.measured_joints.set((0.0, 0.1, 0.28, 0.3, 0.4))
        return plan(*arguments)

    monkeypatch.setattr(joint_path, "plan", plan_while_the_arm_moves)
    assert START_TOLERANCE_RAD == 0.05
    ctx = await drive_path_cancelled_at_once(layer, admit_path(layer, *CORNERED))
    assert ctx.failed == (
        FailureKind.PRECONDITION_FAILED,
        "arm stands 0.080 rad at joint 3 from where it stood when the path was taken, "
        "past the 0.050 rad start tolerance; send the path again",
    )
    assert ctx.completed is None and ctx.cancelled is None
    assert coordinator._arm.plan is None, "nothing was installed"
    assert coordinator.try_claim_arm(), "the failed goal released the arm"


@pytest.mark.parametrize(
    "expected_start",
    [(), (1.03, 0.1, 0.2, 0.3, 0.4)],
    ids=["unchecked", "checked against the measured joints"],
)
async def test_an_arm_past_a_limit_by_more_than_the_start_tolerance_runs_its_path(
    expected_start, follower_never_stale
):
    # Each joint's limits are -1 and 1 rad. The follower measures joint 1
    # 0.03 rad past its upper limit, farther than the 0.02 rad start
    # tolerance, and the arm stands still while its path is planned. The
    # goal's expected start, when it names one, is the measured joints, and
    # the check compares it with them, as MoveIt does. Once the path is
    # planned, the measured joints, clamped as the start was, stand at the
    # start, and the path runs from the limit.
    limits = limits_mod.JointLimits(lower=(-1.0,) * 5, upper=(1.0,) * 5)
    coordinator, layer = make_layer(limits=limits, start_tolerance_rad=0.02)
    coordinator.measured_joints.set((1.03, 0.1, 0.2, 0.3, 0.4))
    ctx, task, plan = await driving_path(
        coordinator, layer, admit_path(layer, *CORNERED, expected_start=expected_start)
    )
    assert plan.profile.start == (1.0, 0.1, 0.2, 0.3, 0.4)
    assert ctx.failed is None
    coordinator.abort_arm_plan(plan)
    await task


async def test_both_start_checks_hold_the_configured_tolerance(monkeypatch, follower_never_stale):
    # At a start tolerance of 0.02 rad, the launchers' value: an expected
    # start 0.019 rad from where joint 3 measures runs and one 0.021 rad away
    # fails, and so does an arm that moves 0.021 rad at joint 3 while its path
    # is planned, where 0.019 rad runs.
    tolerance_rad = 0.02
    plan = joint_path.plan

    def off_at_joint_3(rad):
        return (*MEASURED[:2], MEASURED[2] + rad, *MEASURED[3:])

    def layer_measuring():
        coordinator, layer = make_layer(start_tolerance_rad=tolerance_rad)
        coordinator.measured_joints.set(MEASURED)
        return coordinator, layer

    coordinator, layer = layer_measuring()
    _ctx, task, installed = await driving_path(
        coordinator, layer, admit_path(layer, CORNERED[0], expected_start=off_at_joint_3(0.019))
    )
    coordinator.abort_arm_plan(installed)
    await task
    _coordinator, layer = layer_measuring()
    ctx = await drive_path_cancelled_at_once(
        layer, admit_path(layer, CORNERED[0], expected_start=off_at_joint_3(0.021))
    )
    assert ctx.failed == (
        FailureKind.PRECONDITION_FAILED,
        "arm stands 0.021 rad from expected_start_joint_positions at joint 3, past the "
        "0.020 rad start tolerance; plan the path from where the arm stands",
    )

    def moving_while_planned(coordinator, rad):
        def plan_while_the_arm_moves(*arguments):
            coordinator.measured_joints.set(off_at_joint_3(rad))
            return plan(*arguments)

        monkeypatch.setattr(joint_path, "plan", plan_while_the_arm_moves)

    coordinator, layer = layer_measuring()
    moving_while_planned(coordinator, 0.019)
    _ctx, task, installed = await driving_path(coordinator, layer, admit_path(layer, CORNERED[0]))
    coordinator.abort_arm_plan(installed)
    await task
    coordinator, layer = layer_measuring()
    moving_while_planned(coordinator, 0.021)
    ctx = await drive_path_cancelled_at_once(layer, admit_path(layer, CORNERED[0]))
    assert ctx.failed == (
        FailureKind.PRECONDITION_FAILED,
        "arm stands 0.021 rad at joint 3 from where it stood when the path was taken, "
        "past the 0.020 rad start tolerance; send the path again",
    )


@pytest.mark.parametrize("requested_s", [0.0, MAX_REQUESTED_DURATION_S])
async def test_a_path_past_the_ceiling_at_the_caps_fails_with_rejected_and_installs_nothing(
    requested_s, follower_never_stale
):
    # At 0.06 rad/s the out and back legs each take some 33 s, within the 60 s
    # ceiling; the path rests at the reversal and takes some 67 s, past it,
    # whatever duration the goal requests.
    velocity_caps = (0.06,) * 5
    acceleration_caps = make_config().max_joint_acceleration_rad_s2
    coordinator, layer = make_layer(max_joint_velocity_rad_s=velocity_caps)
    coordinator.measured_joints.set((0.0,) * 5)
    out = (2.0, 0.0, 0.0, 0.0, 0.0)
    out_s = joint_path.plan(((0.0,) * 5, out), 0.0, velocity_caps, acceleration_caps).duration_s
    out_and_back_s = joint_path.plan(
        ((0.0,) * 5, out, (0.0,) * 5), 0.0, velocity_caps, acceleration_caps
    ).duration_s
    assert out_s < MAX_REQUESTED_DURATION_S < out_and_back_s
    ctx = await drive_path_cancelled_at_once(
        layer, admit_path(layer, out, (0.0,) * 5, duration_s=requested_s)
    )
    assert ctx.failed == (
        FailureKind.REJECTED,
        f"the joint velocity and acceleration caps take {out_and_back_s:.1f} s to run this path, "
        "past the 60 s a move may take; split it into paths of fewer waypoints",
    )
    assert ctx.completed is None and ctx.cancelled is None
    assert coordinator._arm.plan is None, "nothing was installed"
    assert coordinator.try_claim_arm(), "the failed goal released the arm"
    coordinator.release_arm()
    # The out leg alone runs.
    _ctx, task, plan = await driving_path(
        coordinator, layer, admit_path(layer, out, duration_s=requested_s)
    )
    assert plan.profile.duration_s == max(requested_s, out_s)
    coordinator.abort_arm_plan(plan)
    await task


async def test_an_arm_goal_below_the_base_plane_is_admitted(follower_never_stale):
    # The SO-101 checks no collision. This target is inside the joint limits.
    # Its grasp point is 0.2 m below the plane that the base stands on, thus
    # through the surface below the robot. The backbone accepts the move and
    # runs it to its end.
    kinematics = Kinematics(KINEMATICS_URDF_PATH)
    limits = limits_mod.from_urdf(KINEMATICS_URDF_PATH)
    below_the_base = (0.0, 1.5, 0.0, 0.0, 0.0)
    assert limits.contains(below_the_base)
    position, _orientation = kinematics.forward_kinematics(below_the_base)
    assert position[2] < -0.15
    coordinator, layer = make_layer(kinematics, limits=limits)
    coordinator.measured_joints.set(READY_POSITIONS_RAD)
    plan = admit_joints(layer, below_the_base, 0.0)
    assert plan.profile.end == below_the_base
    coordinator.measured_joints.set(below_the_base)
    run_arm_plan_to_its_end(coordinator, plan)
    ctx = FakeGoalContext()
    await layer.drive_arm(ctx, plan)
    assert ctx.completed is not None


async def test_a_joint_path_below_the_base_plane_is_admitted(follower_never_stale):
    # The SO-101 checks no collision. This target is inside the joint limits.
    # Its grasp point is 0.2 m below the plane that the base stands on, thus
    # through the surface below the robot. The backbone accepts the move and
    # runs it to its end.
    kinematics = Kinematics(KINEMATICS_URDF_PATH)
    limits = limits_mod.from_urdf(KINEMATICS_URDF_PATH)
    below_the_base = (0.0, 1.5, 0.0, 0.0, 0.0)
    assert limits.contains(below_the_base)
    position, _orientation = kinematics.forward_kinematics(below_the_base)
    assert position[2] < -0.15
    coordinator, layer = make_layer(kinematics, limits=limits)
    coordinator.measured_joints.set(READY_POSITIONS_RAD)
    ctx, task, plan = await driving_path(coordinator, layer, admit_path(layer, below_the_base))
    assert plan.profile.end == below_the_base
    coordinator.measured_joints.set(below_the_base)
    run_arm_plan_to_its_end(coordinator, plan)
    await task
    assert ctx.completed is not None


def test_pose_admission_refuses_busy_before_solving():
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    admit_joints(layer, (0.1,) * 5, 0.0)
    with pytest.raises(Refusal) as raised:
        admit_pose(layer, (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    assert refusal_of(raised) == (FailureKind.PRECONDITION_FAILED, ARM_WORDS.busy)
    # The expensive verified solve never ran for the doomed goal.
    assert kinematics.solve_calls == []


def test_unknown_limb_names_are_refused():
    from peppygen.exposed_actions.limb_motion import move_arm_joints as module

    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    request = SimpleNamespace(
        data=SimpleNamespace(
            arm_name="left_arm",
            joint_positions=[0.0] * 5,
            duration_s=0.0,
            settle_tolerances_rad=[0.1] * 5,
            settle_timeout_s=2.0,
        )
    )
    decision = layer.decide_arm_joints(module, PendingSlot())(request)
    assert decision.refusal.kind is module.FailureKind.REJECTED
    assert "unknown arm_name" in decision.refusal.message
    assert coordinator.try_claim_arm()


async def test_a_decided_joint_path_runs_from_where_the_arm_stands_through_its_waypoints(
    follower_never_stale,
):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set(MEASURED)
    pending = PendingSlot()
    request = SimpleNamespace(data=joint_path_goal(*CORNERED, expected_start=MEASURED))
    decision = layer.decide_joint_path(pending)(request)
    assert decision.refusal is None
    _ctx, task, plan = await driving_path(coordinator, layer, pending.take())
    assert (plan.profile.start, plan.profile.end) == (MEASURED, CORNERED[-1])
    assert plan.settle.tolerances_rad == (0.1,) * 5
    coordinator.abort_arm_plan(plan)
    await task


def test_decide_refuses_any_admission_surprise_without_holding_the_slot():
    from peppygen.exposed_actions.limb_motion import move_arm_joints as module

    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)

    class Boom:
        @property
        def data(self):
            raise RuntimeError("admission surprise")

    decision = layer.decide_arm_joints(module, PendingSlot())(Boom())
    assert decision.refusal.kind is module.FailureKind.ATTEMPT_FAILED
    assert "admission failed" in decision.refusal.message
    assert coordinator.try_claim_arm(), "no surprise may leave the slot held"


def test_a_goal_on_a_limb_a_leader_streams_is_refused_with_leader_streaming():
    from peppygen.exposed_actions.limb_motion import move_arm, move_gripper
    from peppygen.exposed_actions.postures import move_to_home

    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    coordinator.measured_gripper.set(0.5)
    coordinator.leader_joints.set((0.0,) * 5)
    coordinator.leader_gripper.set(0.5)
    pose = SimpleNamespace(
        data=SimpleNamespace(
            arm_name="arm",
            position=[0.1, 0.0, 0.2],
            orientation=[0.0, 0.0, 0.0, 1.0],
            duration_s=0.0,
            plan_position_tolerance_m=0.01,
            plan_orientation_tolerance_rad=0.2,
            settle_position_tolerance_m=0.01,
            settle_orientation_tolerance_rad=0.2,
            settle_timeout_s=2.0,
        )
    )
    gripper = SimpleNamespace(
        data=SimpleNamespace(
            gripper_name="gripper", opening=0.2, max_effort=0.0, settle_timeout_s=2.0
        )
    )
    posture = SimpleNamespace(data=SimpleNamespace(duration_s=0.0))

    for decision, module, reason in (
        (layer.decide_pose(move_arm, PendingSlot())(pose), move_arm, ARM_WORDS.leader_streaming),
        (
            layer.decide_gripper(move_gripper, PendingSlot())(gripper),
            move_gripper,
            GRIPPER_WORDS.leader_streaming,
        ),
        (
            layer.decide_posture(move_to_home, READY_POSITIONS_RAD, PendingSlot())(posture),
            move_to_home,
            ARM_WORDS.leader_streaming,
        ),
    ):
        assert decision.refusal == module.Failure(module.FailureKind.LEADER_STREAMING, reason)
    # The refusals held no slot.
    assert coordinator.try_claim_arm() and coordinator.try_claim_gripper()


def test_a_joint_path_on_an_arm_a_leader_streams_is_refused_with_leader_streaming():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    coordinator.leader_joints.set((0.0,) * 5)
    request = SimpleNamespace(data=joint_path_goal((0.1,) * 5))
    decision = layer.decide_joint_path(PendingSlot())(request)
    assert decision.refusal == execute_joint_path.Failure(
        execute_joint_path.FailureKind.LEADER_STREAMING, ARM_WORDS.leader_streaming
    )
    assert coordinator.try_claim_arm(), "the refusal held no slot"


async def test_pose_goal_solves_in_drive_and_streams_a_plan():
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = admit_pose(layer, (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    assert isinstance(goal, PoseMove)
    assert kinematics.solve_calls == [], "admission must not pay the solve"

    async def run_plan_to_completion():
        while True:
            coordinator.measured_joints.set((0.0,) * 5)
            sample = coordinator.arm_tick(time.monotonic())
            if sample is not None:
                coordinator.arm_published(True, sample)
                if coordinator._arm.plan is None:
                    return
            await asyncio.sleep(0.001)

    ctx = FakeGoalContext()
    runner = asyncio.create_task(run_plan_to_completion())
    await layer.drive_pose(ctx, goal, kinematics, kinematics)
    runner.cancel()
    assert ctx.completed is not None
    assert ("solve",) == tuple(c[0] for c in kinematics.solve_calls)
    assert coordinator.try_claim_arm()


async def pose_move_at_the_end_of_its_plan(coordinator, layer, kinematics, measured_at_end):
    """Run the plan of a pose goal to its planned end, at explicit instants.
    The goal is the grasp point of kinematics.solution. The follower measures
    the arm at (0.0,) * 5 as the plan starts and at `measured_at_end` from
    the planned end on. Returns the goal context, the drive task, which
    completes the goal when it runs next, and the plan."""
    coordinator.measured_joints.set((0.0,) * 5)
    goal_position, _orientation = kinematics.forward_kinematics(kinematics.solution)
    move = admit_pose(layer, goal_position, (0.0, 0.0, 0.0, 1.0))
    ctx = FakeGoalContext()
    task = asyncio.create_task(layer.drive_pose(ctx, move, kinematics, kinematics))
    plan = await installed_plan(coordinator)
    assert plan.profile.end == kinematics.solution
    coordinator.arm_published(True, coordinator.arm_tick(0.0))
    coordinator.measured_joints.set(measured_at_end)
    coordinator.arm_published(True, coordinator.arm_tick(plan.profile.duration_s))
    return ctx, task, plan


async def test_a_pose_move_completes_once_the_measured_grasp_point_is_within_tolerance(
    follower_never_stale,
):
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    ctx, task, plan = await pose_move_at_the_end_of_its_plan(
        coordinator, layer, kinematics, measured_at_end=kinematics.solution
    )
    assert plan.done.is_set()
    await task
    position, _orientation, _t = ctx.completed
    assert position == list(kinematics.forward_kinematics(kinematics.solution)[0])


async def test_a_pose_move_outside_its_tolerance_at_the_timeout_fails_with_timeout(
    follower_never_stale,
):
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    # The follower stays where the plan started, 16 mm from the goal.
    ctx, task, plan = await pose_move_at_the_end_of_its_plan(
        coordinator, layer, kinematics, measured_at_end=(0.0,) * 5
    )
    assert not plan.done.is_set()
    assert coordinator.arm_tick(plan.profile.duration_s + 2.0) is None
    assert plan.done.is_set()
    await task
    assert ctx.failed == (
        FailureKind.TIMEOUT,
        "still 16.3 mm / 0.00 rad from the target, 2.0 s after the planned end",
    )
    assert ctx.completed is None
    assert coordinator.try_claim_arm()


async def test_a_stale_follower_reports_the_commanded_pose(follower_never_stale):
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    ctx, task, _plan = await pose_move_at_the_end_of_its_plan(
        coordinator, layer, kinematics, measured_at_end=kinematics.solution
    )
    # The follower sends no more state after it receives the last sample.
    # The last command of a finished plan is its end sample.
    coordinator.measured_joints.clear()
    await task
    position, _orientation, _t = ctx.completed
    assert position == list(kinematics.forward_kinematics(kinematics.solution)[0])


async def test_pose_move_duration_floors_at_the_linear_ee_cap():
    # FakeKinematics FK sits at (0.1, 0.0, 0.2); the goal is 1.0 m away, so
    # at 0.5 m/s the quintic floor is 15/8 * 1.0 / 0.5, above the joint floor.
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics, max_ee_velocity_m_s=0.5)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = admit_pose(layer, (1.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
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
    goal = admit_pose(layer, (0.1, 0.0, 0.2), (0.0, 0.0, 1.0, 0.0))
    ctx = FakeGoalContext()
    task = asyncio.create_task(layer.drive_pose(ctx, goal, kinematics, kinematics))
    plan = await installed_plan(coordinator)
    assert plan.profile.duration_s == pytest.approx(15.0 / 8.0 * math.pi)
    coordinator.abort_arm_plan(plan)
    await task


async def test_an_unreachable_pose_fails_with_unreachable_and_releases():
    kinematics = FakeKinematics()
    kinematics.corrupted = True  # solve returns None
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = admit_pose(layer, (5.0, 5.0, 5.0), (0.0, 0.0, 0.0, 1.0))
    ctx = FakeGoalContext()
    await layer.drive_pose(ctx, goal, kinematics, kinematics)
    kind, message = ctx.failed
    assert kind is FailureKind.UNREACHABLE
    assert message.startswith("goal pose not planned within ")
    # The failure must say the search was local, so a caller knows to retry
    # from elsewhere rather than read it as the pose being out of reach.
    assert "local descent" in message
    assert ctx.completed is None
    assert coordinator.try_claim_arm()


async def test_abandoned_pose_admission_releases_the_claim():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    pending = PendingSlot()
    pending.plan = admit_pose(layer, (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    layer.abandon(pending)
    assert coordinator.try_claim_arm()


async def test_an_abandoned_path_admission_releases_the_claim(follower_never_stale):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    pending = PendingSlot()
    pending.plan = admit_path(layer, (0.1,) * 5)
    layer.abandon(pending)
    assert coordinator.try_claim_arm()


def test_absurd_duration_is_refused():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    with pytest.raises(
        Refusal, match="duration_s must be a number of seconds from 0 to 60"
    ) as raised:
        admit_joints(layer, (0.1,) * 5, MAX_REQUESTED_DURATION_S + 1.0)
    assert raised.value.kind is FailureKind.REJECTED
    assert coordinator.try_claim_arm()


def test_gripper_effort_preference_is_admitted_and_ignored():
    coordinator, layer = make_layer()
    coordinator.measured_gripper.set(0.0)
    plan = admit_gripper(layer, 0.5, 3.0)
    assert plan.target == 0.5


async def test_pose_solver_surprise_still_reaches_a_terminal_result():
    class BoomKinematics(FakeKinematics):
        def inverse_kinematics(self, seed, position, orientation):
            raise RuntimeError("solver surprise")

    kinematics = BoomKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = admit_pose(layer, (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    ctx = FakeGoalContext()
    await layer.drive_pose(ctx, goal, kinematics, kinematics)
    kind, message = ctx.failed
    assert kind is FailureKind.ATTEMPT_FAILED
    assert "pose solve failed" in message
    assert coordinator.try_claim_arm()


IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


async def test_the_callers_slack_reaches_the_solver():
    # The bug this guards: move_arm carried both tolerance fields and the
    # node dropped them, so a caller's stated slack changed nothing.
    kinematics = FakeKinematics()
    # Refused rather than solved: the bars are recorded before the solve
    # answers, and a refusal terminates without a control loop to run a plan.
    kinematics.corrupted = True
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    move = admit_pose(
        layer, (0.1, 0.0, 0.2), IDENTITY_QUAT, plan=(0.004, 0.05), settle=(0.004, 0.05, 2.0)
    )
    assert (move.goal.position_tolerance_m, move.goal.orientation_tolerance_rad) == (0.004, 0.05)
    await layer.drive_pose(FakeGoalContext(), move, kinematics, kinematics)
    assert kinematics.bars == [(0.004, 0.05)]


@pytest.mark.parametrize("bad", [0.0, -0.001, math.nan, math.inf])
async def test_a_tolerance_that_is_no_bar_is_refused(bad):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    coordinator.measured_gripper.set(0.5)
    pose = ((0.1, 0.0, 0.2), IDENTITY_QUAT)
    for plan, settle, field in (
        ((bad, 0.2), SETTLE, "plan_position_tolerance_m"),
        ((0.01, bad), SETTLE, "plan_orientation_tolerance_rad"),
        (PLAN, (bad, 0.2, 2.0), "settle_position_tolerance_m"),
        (PLAN, (0.01, bad, 2.0), "settle_orientation_tolerance_rad"),
    ):
        with pytest.raises(Refusal, match=f"{field} must be a positive number") as raised:
            admit_pose(layer, *pose, plan=plan, settle=settle)
        assert raised.value.kind is FailureKind.REJECTED
    with pytest.raises(
        Refusal, match="each settle_tolerances_rad entry must be a positive number"
    ) as raised:
        admit_joints(layer, (0.1,) * 5, tolerances=(0.1, 0.1, bad, 0.1, 0.1))
    assert raised.value.kind is FailureKind.REJECTED
    assert coordinator.try_claim_arm(), "a refused tolerance must not hold the slot"


@pytest.mark.parametrize("bad", [-0.001, math.nan, math.inf, MAX_REQUESTED_DURATION_S + 1.0])
async def test_a_settle_timeout_that_is_no_duration_is_refused(bad):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    coordinator.measured_gripper.set(0.5)
    refusal = "settle_timeout_s must be a number of seconds from 0 to 60"
    with pytest.raises(Refusal, match=refusal):
        admit_pose(layer, (0.1, 0.0, 0.2), IDENTITY_QUAT, settle=(0.01, 0.2, bad))
    with pytest.raises(Refusal, match=refusal):
        admit_joints(layer, (0.1,) * 5, timeout_s=bad)
    assert coordinator.try_claim_arm()


def test_a_settle_tolerance_below_the_plan_tolerance_of_its_axis_is_refused():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    refusal = "settle_{0} must be at least plan_{0}: the plan may sit that far from the pose"
    with pytest.raises(Refusal, match=refusal.format("position_tolerance_m")):
        admit_pose(layer, (0.1, 0.0, 0.2), IDENTITY_QUAT, settle=(0.005, 0.2, 2.0))
    with pytest.raises(Refusal, match=refusal.format("orientation_tolerance_rad")):
        admit_pose(layer, (0.1, 0.0, 0.2), IDENTITY_QUAT, settle=(0.01, 0.1, 2.0))
    assert coordinator.try_claim_arm()


async def test_an_unreachable_failure_names_the_bar_it_was_judged_against():
    kinematics = FakeKinematics()
    kinematics.corrupted = True
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    move = admit_pose(
        layer, (0.1, 0.0, 0.2), IDENTITY_QUAT, plan=(0.004, 0.05), settle=(0.004, 0.05, 2.0)
    )
    ctx = FakeGoalContext()
    await layer.drive_pose(ctx, move, kinematics, kinematics)
    kind, message = ctx.failed
    assert kind is FailureKind.UNREACHABLE
    assert "4.0 mm" in message and "2.9 deg" in message


async def test_the_ee_caps_cannot_pin_the_slot_past_the_duration_ceiling():
    # A move admitted under the ceiling can still need longer than it once
    # the end-effector speed caps are applied; that has no plan, and must not
    # stretch.
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics, max_ee_velocity_m_s=1e-4)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = admit_pose(layer, (5.0, 0.0, 0.2), IDENTITY_QUAT)
    ctx = FakeGoalContext()
    await layer.drive_pose(ctx, goal, kinematics, kinematics)
    kind, message = ctx.failed
    assert kind is FailureKind.UNREACHABLE
    assert "ceiling" in message
    assert coordinator.try_claim_arm(), "a failed move must release the arm"


async def test_a_stopped_plan_ends_cancelled_with_the_stops_reason(follower_never_stale):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    plan = admit_joints(layer, (0.5,) * 5, 1.0)
    ctx = FakeGoalContext()
    driving = asyncio.create_task(layer.drive_arm(ctx, plan))
    await asyncio.sleep(0)

    assert layer.stop_moves("the operator asked") == ["arm"]
    await driving

    assert ctx.completed is None
    reason, positions, _action_time = ctx.cancelled
    assert reason == "stopped: the operator asked"
    assert positions == [0.0] * 5
    assert coordinator.try_claim_arm(), "the goal released the arm"


async def test_a_goal_whose_sender_is_gone_ends_cancelled_with_that_reason(
    follower_never_stale,
):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    plan = admit_joints(layer, (0.5,) * 5, 1.0)
    ctx = FakeGoalContext()
    ctx.request_cancel(cause="sender_gone")
    await layer.drive_arm(ctx, plan)

    reason, positions, _action_time = ctx.cancelled
    assert reason == "its sender is gone"
    assert positions == [0.0] * 5
    assert plan.aborted
    assert coordinator.try_claim_arm()


async def test_a_stop_mid_path_ends_the_goal_cancelled_where_the_arm_stands(
    follower_never_stale,
):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set(MEASURED)
    ctx, task, plan = await driving_path(
        coordinator, layer, admit_path(layer, *CORNERED, duration_s=1.0)
    )
    coordinator.arm_published(True, coordinator.arm_tick(0.0))
    midway = coordinator.arm_tick(plan.profile.duration_s / 2.0)
    coordinator.arm_published(True, midway)
    # The follower measures the arm where the path stands midway.
    coordinator.measured_joints.set(midway)

    assert layer.stop_moves("the operator asked") == ["arm"]
    await task

    assert ctx.completed is None
    reason, arms, _action_time = ctx.cancelled
    assert reason == "stopped: the operator asked"
    assert path_result_joints(arms) == list(midway)
    assert coordinator.try_claim_arm(), "the goal released the arm"


@pytest.mark.parametrize("cause", ["caller", "sender_gone"])
async def test_a_cancelled_path_ends_cancelled_with_the_reason_of_its_cause(
    cause, follower_never_stale
):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set(MEASURED)
    ctx, task, plan = await driving_path(
        coordinator, layer, admit_path(layer, *CORNERED, duration_s=1.0)
    )
    ctx.request_cancel(cause=cause)
    await task

    reason, arms, _action_time = ctx.cancelled
    assert reason == CANCEL_REASONS[cause]
    assert path_result_joints(arms) == list(MEASURED)
    assert plan.aborted
    assert coordinator.try_claim_arm()


async def test_a_stopped_gripper_plan_ends_cancelled_with_the_stops_reason(
    follower_never_stale,
):
    coordinator, layer = make_layer()
    coordinator.measured_gripper.set(0.0)
    plan = admit_gripper(layer, 1.0, 0.0)
    ctx = FakeGoalContext()
    driving = asyncio.create_task(layer.drive_gripper(ctx, plan))
    await asyncio.sleep(0)

    assert layer.stop_moves("") == ["gripper"]
    await driving

    reason, settled, _opening, _action_time = ctx.cancelled
    assert (reason, settled) == ("stopped", False)
    assert coordinator.try_claim_gripper()


async def test_a_stop_during_the_pose_solve_ends_the_goal_without_a_plan():
    class SlowKinematics(FakeKinematics):
        def __init__(self, layer_holder):
            super().__init__()
            self.layer_holder = layer_holder

        def inverse_kinematics(self, seed, position, orientation, **bars):
            # The stop lands while the solve runs off the event loop.
            self.layer_holder[0].stop_moves("the operator asked")
            return super().inverse_kinematics(seed, position, orientation, **bars)

    holder = []
    kinematics = SlowKinematics(holder)
    coordinator, layer = make_layer(kinematics)
    holder.append(layer)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = admit_pose(layer, (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    ctx = FakeGoalContext()
    await layer.drive_pose(ctx, goal, kinematics, kinematics)

    assert ctx.completed is None
    reason, _position, _orientation, _t = ctx.cancelled
    assert reason == "stopped: the operator asked"
    assert coordinator._arm.plan is None, "nothing was installed after the stop"
    assert coordinator.try_claim_arm()


async def test_a_stop_before_the_pose_drive_runs_ends_the_goal_as_stopped():
    # The stop lands between the goal's admission and its drive task: the
    # goal ends cancelled with the stop's reason, and nothing is solved.
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = admit_pose(layer, (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    assert layer.stop_moves("the operator asked") == []
    ctx = FakeGoalContext()
    await layer.drive_pose(ctx, goal, kinematics, kinematics)

    assert ctx.completed is None
    reason, position, _orientation, _t = ctx.cancelled
    assert reason == "stopped: the operator asked"
    assert position == list(kinematics.forward_kinematics((0.0,) * 5)[0])
    assert kinematics.solve_calls == []
    assert coordinator._arm.plan is None
    assert coordinator.try_claim_arm()


async def test_a_stop_before_the_path_drive_runs_ends_the_goal_as_stopped(follower_never_stale):
    # The stop lands between the goal's admission and its drive task: the
    # goal ends cancelled with the stop's reason and the measured joints, and
    # nothing is installed.
    coordinator, layer = make_layer()
    coordinator.measured_joints.set(MEASURED)
    move = admit_path(layer, *CORNERED)
    assert layer.stop_moves("the operator asked") == []
    ctx = await drive_path_cancelled_at_once(layer, move)

    assert ctx.completed is None and ctx.failed is None
    reason, arms, action_time = ctx.cancelled
    assert (reason, action_time) == ("stopped: the operator asked", 0.0)
    assert path_result_joints(arms) == list(MEASURED)
    assert coordinator._arm.plan is None
    assert coordinator.try_claim_arm()


async def test_a_stop_while_the_path_is_planned_off_the_loop_ends_the_goal_without_a_plan(
    monkeypatch, follower_never_stale
):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set(MEASURED)
    plan = joint_path.plan
    planning_threads = []

    def plan_while_a_stop_lands(*arguments):
        planning_threads.append(threading.get_ident())
        layer.stop_moves("the operator asked")
        return plan(*arguments)

    monkeypatch.setattr(joint_path, "plan", plan_while_a_stop_lands)
    ctx = await drive_path_cancelled_at_once(layer, admit_path(layer, *CORNERED))

    # The path was planned on a thread of its own, off the event loop.
    assert len(planning_threads) == 1
    assert planning_threads[0] != threading.get_ident()
    # The stop landed during the planning: the goal ends cancelled with the
    # stop's reason and where the arm stands, and nothing is installed.
    assert ctx.completed is None and ctx.failed is None
    reason, arms, action_time = ctx.cancelled
    assert (reason, action_time) == ("stopped: the operator asked", 0.0)
    assert path_result_joints(arms) == list(MEASURED)
    assert coordinator._arm.plan is None
    assert coordinator.try_claim_arm()


async def test_the_plan_check_answers_the_duration_the_joint_caps_floor():
    # The EE caps are transparent and the joint caps bind: the check answers
    # the duration the move's own trajectory takes.
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics, max_joint_velocity_rad_s=(2.0,) * 5)
    coordinator.measured_joints.set((0.0,) * 5)
    check = await layer.check_arm_move(
        "arm", (0.2, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, *PLAN, kinematics, kinematics
    )
    floored = minimum_jerk.plan((0.0,) * 5, kinematics.solution, 0.0, (2.0,) * 5).duration_s
    assert (check.reachable, check.reason) == (True, "")
    assert floored > 0.0
    assert check.duration_s == pytest.approx(floored)


async def test_a_zero_settle_timeout_fails_an_arm_move_short_of_its_target_at_the_planned_end(
    follower_never_stale,
):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    plan = admit_joints(layer, (0.5,) * 5, 0.0, timeout_s=0.0)
    coordinator.arm_published(True, coordinator.arm_tick(0.0))
    # The follower stands where it started as the plan ends: with no time
    # to settle, the move fails at the planned end.
    assert coordinator.arm_tick(plan.profile.duration_s) is None
    assert plan.done.is_set()
    ctx = FakeGoalContext()
    await layer.drive_arm(ctx, plan)
    assert ctx.failed == (
        FailureKind.TIMEOUT,
        "still 0.500 rad from the target at joint 1, 0.0 s after the planned end",
    )
    assert coordinator.try_claim_arm()


async def test_the_plan_check_answers_the_duration_and_installs_nothing():
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics, max_ee_velocity_m_s=0.1)
    coordinator.measured_joints.set((0.0,) * 5)

    check = await layer.check_arm_move(
        "arm", (0.2, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, *PLAN, kinematics, kinematics
    )

    assert (check.reachable, check.reason) == (True, "")
    assert check.duration_s > 0.0
    assert ("solve",) == tuple(c[0] for c in kinematics.solve_calls)
    assert coordinator._arm.plan is None
    assert coordinator.try_claim_arm(), "the check claims nothing"


async def test_the_plan_check_answers_unreachable_and_refuses_as_the_move_would():
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)

    kinematics.corrupted = True
    check = await layer.check_arm_move(
        "arm", (5.0, 5.0, 5.0), (0.0, 0.0, 0.0, 1.0), 0.0, *PLAN, kinematics, kinematics
    )
    assert (check.reachable, check.duration_s) == (False, 0.0)
    assert check.reason.startswith("goal pose not planned within ")

    kinematics.corrupted = False
    with pytest.raises(Refusal, match="unknown arm_name") as raised:
        await layer.check_arm_move(
            "wing", (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, *PLAN, kinematics, kinematics
        )
    assert raised.value.kind is FailureKind.REJECTED
    with pytest.raises(Refusal) as raised:
        await layer.check_arm_move(
            "arm", (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 0.0), 0.0, *PLAN, kinematics, kinematics
        )
    assert refusal_of(raised) == (
        FailureKind.REJECTED, "goal pose has an unnormalizable orientation"
    )

    # While an arm goal is in flight, the check is refused as a second goal is.
    admit_joints(layer, (0.5,) * 5, 1.0)
    with pytest.raises(Refusal) as raised:
        await layer.check_arm_move(
            "arm", (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, *PLAN, kinematics, kinematics
        )
    assert refusal_of(raised) == (FailureKind.PRECONDITION_FAILED, ARM_WORDS.busy)
    assert len(kinematics.solve_calls) == 1, "a refused check pays no solve"


async def test_the_plan_check_needs_fresh_follower_state():
    kinematics = FakeKinematics()
    _coordinator, layer = make_layer(kinematics)
    with pytest.raises(Refusal) as raised:
        await layer.check_arm_move(
            "arm", (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, *PLAN, kinematics, kinematics
        )
    assert refusal_of(raised) == (FailureKind.RESOURCE_UNAVAILABLE, ARM_WORDS.no_fresh_state)


async def test_a_solve_that_raises_fails_the_plan_check_with_attempt_failed():
    class BoomKinematics(FakeKinematics):
        def inverse_kinematics(self, seed, position, orientation, **bars):
            raise RuntimeError("solver surprise")

    kinematics = BoomKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    with pytest.raises(Refusal) as raised:
        await layer.check_arm_move(
            "arm", (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, *PLAN, kinematics, kinematics
        )
    assert refusal_of(raised) == (
        FailureKind.ATTEMPT_FAILED, "pose solve failed: RuntimeError('solver surprise')"
    )


def test_a_streaming_leader_keeps_the_limbs_until_its_stream_stops(monkeypatch):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    coordinator.measured_gripper.set(0.5)
    coordinator.leader_joints.set((0.0,) * 5)
    coordinator.leader_gripper.set(0.5)
    pose = ((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(Refusal) as raised:
        admit_joints(layer, (0.1,) * 5, 0.0)
    assert refusal_of(raised) == (FailureKind.LEADER_STREAMING, ARM_WORDS.leader_streaming)
    with pytest.raises(Refusal) as raised:
        admit_pose(layer, *pose)
    assert refusal_of(raised) == (FailureKind.LEADER_STREAMING, ARM_WORDS.leader_streaming)
    with pytest.raises(Refusal) as raised:
        admit_gripper(layer, 0.2, 0.0)
    assert refusal_of(raised) == (FailureKind.LEADER_STREAMING, GRIPPER_WORDS.leader_streaming)
    # The refusals held no slot.
    assert coordinator.try_claim_arm() and coordinator.try_claim_gripper()
    coordinator.release_arm()
    coordinator.release_gripper()
    # A goal boundary clears the leader holders; the leader still streams.
    with pytest.raises(Refusal, match="leader is streaming arm"):
        admit_joints(layer, (0.1,) * 5, 0.0)
    with pytest.raises(Refusal, match="leader is streaming gripper"):
        admit_gripper(layer, 0.2, 0.0)
    # The leader lets go: past the deadman, the limbs take goals again. The
    # follower state is refreshed so only the leader's age moves.
    later = time.monotonic() + COMMAND_SILENCE_LIMIT_S + 0.01
    monkeypatch.setattr(time, "monotonic", lambda: later)
    coordinator.measured_joints.set((0.0,) * 5)
    coordinator.measured_gripper.set(0.5)
    assert admit_pose(layer, *pose) is not None
    coordinator.release_arm()
    assert admit_joints(layer, (0.1,) * 5, 0.0) is not None
    assert admit_gripper(layer, 0.2, 0.0) is not None


async def test_the_move_check_refuses_in_the_goals_order():
    # No follower state and a streaming leader: the anchor is refused first,
    # by the check and by the goal alike.
    coordinator, layer = make_layer()
    coordinator.leader_joints.set((0.0,) * 5)
    pose = ((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(Refusal) as raised:
        await layer.check_arm_move("arm", *pose, 0.0, *PLAN, None, None)
    assert refusal_of(raised) == (FailureKind.RESOURCE_UNAVAILABLE, ARM_WORDS.no_fresh_state)
    with pytest.raises(Refusal) as raised:
        admit_pose(layer, *pose)
    assert refusal_of(raised) == (FailureKind.RESOURCE_UNAVAILABLE, ARM_WORDS.no_fresh_state)


async def test_the_move_check_gives_the_streaming_refusal_without_claiming():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    coordinator.leader_joints.set((0.0,) * 5)
    with pytest.raises(Refusal) as raised:
        await layer.check_arm_move(
            "arm", (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, *PLAN, None, None
        )
    assert refusal_of(raised) == (FailureKind.LEADER_STREAMING, ARM_WORDS.leader_streaming)
    assert coordinator.try_claim_arm()
