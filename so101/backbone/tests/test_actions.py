"""Action-layer behavior that the wire cannot provoke deterministically:
settles and their timeouts, walked-back admissions, per-server pending
isolation, limb naming, and failure-aware completion."""

import asyncio
import math
import time
from types import SimpleNamespace

import pytest
from conftest import (
    MEASURED,
    WIDE_LIMITS,
    WIDE_REACH,
    FakeKinematics,
    gripper_step,
    make_config,
)
from so101_description import limits as limits_mod
from so101_description.kinematics import Kinematics
from so101_description.model import KINEMATICS_URDF_PATH
from so101_description.postures import READY_POSITIONS_RAD

from so101_backbone.actions import (
    ARM_LEADER_STREAMING,
    NO_ANCHOR,
    MAX_REQUESTED_DURATION_S,
    READY_DONE,
    ActionLayer,
    PendingSlot,
    PoseMove,
    Rejection,
)
from so101_backbone.coordinator import (
    GRIPPER_STILL_WINDOW_S,
    STALE_LEADER_TIMEOUT_S,
    Coordinator,
)

STALE_NO_POSE = "no arm poses: arm stopped reporting its joints"
STALE_COMMANDED = "the result comes from the joints commanded last"
STALE_FOLLOWER = "the follower stopped reporting"

# The bars of the moves below, where a test sets no others: the plan and the
# settle of a pose move each allow 1 cm and 0.2 rad, a joint move 0.1 rad per
# joint, a gripper move 1 % of the jaw travel, and each limb has 2 s past the
# planned end to settle.
PLAN = (0.01, 0.2)
SETTLE = (0.01, 0.2, 2.0)
JOINT_TOLERANCES = (0.1,) * 5


def admit_joints(layer, target, duration_s=0.0, tolerances=JOINT_TOLERANCES, timeout_s=2.0):
    return layer._admit_arm_move(target, duration_s, tolerances, timeout_s)


def admit_pose(layer, position, orientation, duration_s=0.0, plan=PLAN, settle=SETTLE):
    return layer._admit_pose_move(position, orientation, duration_s, *plan, *settle)


# The gripper moves below wait up to 2 s past the planned end of their travel
# for the jaws to stand still, where a test sets no other settle timeout.
GRIPPER_SETTLE_TIMEOUT_S = 2.0


def admit_gripper(layer, opening, max_effort=0.0, settle_timeout_s=GRIPPER_SETTLE_TIMEOUT_S):
    return layer._admit_gripper_move(opening, max_effort, settle_timeout_s)


class FakeGoalContext:
    """Records the terminal call; cancel never fires unless asked."""

    def __init__(self):
        self.completed = None
        self.cancelled = None
        self._cancel = asyncio.Event()

    def request_cancel(self):
        self._cancel.set()

    async def cancel_signal(self):
        await self._cancel.wait()

    async def complete(self, *args):
        self.completed = args

    async def complete_cancelled(self, *args):
        self.cancelled = args


def make_layer(kinematics=None, limits=WIDE_LIMITS, **config_overrides):
    config = make_config(**config_overrides)
    coordinator = Coordinator(config, kinematics or FakeKinematics(), limits, WIDE_REACH)
    return coordinator, ActionLayer(coordinator, config, limits)


async def installed_plan(coordinator):
    async def poll():
        # The coordinator installs a plan synchronously and owns no event to
        # wait on; adding one to production code for a test's benefit would
        # be the wrong trade, so this polls under a timeout.
        while coordinator._arm_plan is None:  # noqa: ASYNC110
            await asyncio.sleep(0.001)
        return coordinator._arm_plan

    return await asyncio.wait_for(poll(), timeout=2.0)


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
    success, message, opening, _action_time = ctx.completed
    assert (success, message, opening) == (True, "move complete", 0.52)
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
    success, message, opening, _action_time = ctx.completed
    assert (success, message, opening) == (True, "move complete", 0.8)
    assert coordinator.try_claim_gripper()


async def test_the_settle_timeout_completes_a_move_whose_jaws_keep_moving(follower_never_stale):
    coordinator, layer = make_layer(max_gripper_rate_frac_s=1.0)
    coordinator.measured_gripper.set(1.0)
    plan = admit_gripper(layer, 0.5, settle_timeout_s=2.0)
    # The command lands on the 50th tick, at 0.49 s, and the tick after is
    # the planned end. Each delivery from then on lands 0.1 from the one
    # before, farther than the still band: the jaws never stand still, and
    # 2.0 s after the planned end the move completes wherever they are.
    for tick in range(50):
        gripper_step(coordinator, 1.0, round(tick * 0.01, 2))
    for now, opening in ((0.5, 0.9), (1.0, 0.8), (1.5, 0.7), (2.0, 0.6), (2.49, 0.5)):
        gripper_step(coordinator, opening, now)
        assert not plan.done.is_set()
    gripper_step(coordinator, 0.4, 2.5)
    assert plan.done.is_set()
    ctx = FakeGoalContext()
    await layer.drive_gripper(ctx, plan)
    success, message, opening, _action_time = ctx.completed
    assert success is True
    assert message == "move complete: the jaws had not stood still by the settle timeout, 2.0 s after the planned end"
    assert opening == 0.4
    assert coordinator.try_claim_gripper()


async def test_a_zero_settle_timeout_completes_the_gripper_move_at_its_planned_end(
    follower_never_stale,
):
    coordinator, layer = make_layer(max_gripper_rate_frac_s=1.0)
    coordinator.measured_gripper.set(1.0)
    plan = admit_gripper(layer, 0.5, settle_timeout_s=0.0)
    # The command lands on the 50th tick, at 0.49 s, with the jaws at 0.7;
    # the tick after is the planned end, and the timeout of 0 completes the
    # move there.
    for tick in range(50):
        gripper_step(coordinator, 1.0 if tick < 25 else 0.7, round(tick * 0.01, 2))
        assert not plan.done.is_set()
    gripper_step(coordinator, 0.55, 0.5)
    assert plan.done.is_set()
    ctx = FakeGoalContext()
    await layer.drive_gripper(ctx, plan)
    success, message, opening, _action_time = ctx.completed
    assert (success, opening) == (True, 0.55)
    assert message == "move complete: the jaws had not stood still by the settle timeout, 0.0 s after the planned end"


@pytest.mark.parametrize("bad", [-0.001, math.nan, math.inf, MAX_REQUESTED_DURATION_S + 0.001])
def test_a_gripper_settle_timeout_that_is_no_duration_is_refused(bad):
    coordinator, layer = make_layer()
    coordinator.measured_gripper.set(0.5)
    with pytest.raises(
        Rejection, match="settle_timeout_s must be a number of seconds from 0 to 60"
    ):
        admit_gripper(layer, 0.2, settle_timeout_s=bad)
    assert coordinator.try_claim_gripper(), "a refused settle timeout must not hold the slot"
    coordinator.release_gripper()
    assert admit_gripper(layer, 0.2, settle_timeout_s=MAX_REQUESTED_DURATION_S) is not None


async def test_cancel_during_the_gripper_travel_completes_cancelled(follower_never_stale):
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
    success, message, opening, _action_time = ctx.cancelled
    assert success is False
    assert message == "goal cancelled"
    assert opening == 0.9
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
    success, message, opening, _action_time = ctx.completed
    assert success is True
    assert message == "move complete"
    assert opening == 0.5


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
    with pytest.raises(Rejection, match="arm is already executing a motion"):
        admit_joints(layer, (0.2,) * 5, 0.0)
    with pytest.raises(Rejection, match="gripper is already executing a move"):
        admit_gripper(layer, 0.5, 0.0)


def test_a_posture_does_not_claim_or_drive_the_gripper(follower_never_stale):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set(MEASURED)
    coordinator.measured_gripper.set(0.3)
    plan = layer._admit_posture(READY_POSITIONS_RAD, 0.0, READY_DONE)
    # From its first sample to its last, the posture sends the gripper
    # nothing. The gripper stays free for its own goals.
    for now in (0.0, plan.profile.duration_s):
        coordinator.arm_published(True, coordinator.arm_tick(now))
        assert coordinator.gripper_tick(now) is None
    assert plan.done.is_set()
    assert coordinator.try_claim_gripper() is True


async def test_failed_plan_completes_as_failure():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    plan = admit_joints(layer, (0.1,) * 5, 0.0)
    coordinator._fail_arm_plan(plan, STALE_FOLLOWER)
    ctx = FakeGoalContext()
    await layer.drive_arm(ctx, plan)
    success, message, _positions, _action_time = ctx.completed
    assert success is False
    assert message.startswith(STALE_FOLLOWER)
    assert coordinator.try_claim_arm()


async def test_none_plan_completes_as_failure_and_leaves_the_server_alive():
    _coordinator, layer = make_layer()
    ctx = FakeGoalContext()
    await layer.drive_arm(ctx, None)
    success, message, positions, _action_time = ctx.completed
    assert success is False
    assert "walked back" in message
    assert len(positions) == 5


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
    success, message, positions, _action_time = ctx.completed
    assert (success, message) == (True, "trajectory complete")
    assert positions == list(arrived)


async def test_an_arm_move_outside_its_tolerances_at_the_timeout_fails_with_the_shortfall(
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
    success, message, positions, _action_time = ctx.completed
    assert success is False
    assert message == "still 0.500 rad from the target at joint 1, 2.0 s after the planned end"
    assert positions == [0.0] * 5
    assert coordinator.try_claim_arm()


async def test_a_stale_follower_reports_the_commanded_joints_with_a_note(follower_never_stale):
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
    success, message, positions, _action_time = ctx.completed
    assert (success, message) == (True, f"trajectory complete; {STALE_COMMANDED}")
    assert positions == [0.5] * 5


async def test_a_posture_result_reports_the_grasp_point_of_the_measured_joints(
    follower_never_stale,
):
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set(MEASURED)
    plan = layer._admit_posture(READY_POSITIONS_RAD, 0.0, READY_DONE)
    # The plan ends. The follower measures the arm at its start position.
    run_arm_plan_to_its_end(coordinator, plan)
    ctx = FakeGoalContext()
    await layer.drive_posture(ctx, plan, kinematics)
    position, orientation = kinematics.forward_kinematics(MEASURED)
    assert position != kinematics.forward_kinematics(READY_POSITIONS_RAD)[0]
    assert ctx.completed == (True, READY_DONE, ["arm"], list(position), list(orientation))
    assert coordinator.try_claim_arm()


async def test_a_posture_with_stale_follower_state_reports_empty_arrays(follower_never_stale):
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set(MEASURED)
    plan = layer._admit_posture(READY_POSITIONS_RAD, 0.0, READY_DONE)
    run_arm_plan_to_its_end(coordinator, plan)
    # The follower sends no more state after it receives the last sample.
    # No measurement is fresh, thus the result gives no pose and gives the
    # reason.
    coordinator.measured_joints.clear()
    ctx = FakeGoalContext()
    await layer.drive_posture(ctx, plan, kinematics)
    assert ctx.completed == (True, f"{READY_DONE}; {STALE_NO_POSE}", [], [], [])

    walked_back = FakeGoalContext()
    await layer.drive_posture(walked_back, None, kinematics)
    assert walked_back.completed == (
        False, f"goal admission was walked back; {STALE_NO_POSE}", [], [], [],
    )


async def test_a_walked_back_posture_reports_where_the_arm_stands(follower_never_stale):
    # The goal fails, and its result still gives the measured pose.
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set(MEASURED)
    ctx = FakeGoalContext()
    await layer.drive_posture(ctx, None, kinematics)
    position, orientation = kinematics.forward_kinematics(MEASURED)
    assert ctx.completed == (
        False, "goal admission was walked back", ["arm"], list(position), list(orientation),
    )


def stop_every_move(layer, _ctx):
    assert layer.stop_moves("the operator asked") == ["arm"]


def cancel_the_goal(_layer, ctx):
    ctx.request_cancel()


@pytest.mark.parametrize(
    ("cut_the_move", "message"),
    [(stop_every_move, "stopped: the operator asked"), (cancel_the_goal, "goal cancelled")],
    ids=["stop", "cancel"],
)
async def test_a_stopped_posture_reports_the_measured_grasp_point(
    cut_the_move, message, follower_never_stale
):
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set(MEASURED)
    plan = layer._admit_posture(READY_POSITIONS_RAD, 0.0, READY_DONE)
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
    assert ctx.cancelled == (False, message, ["arm"], list(position), list(orientation))
    assert plan.aborted
    assert coordinator.try_claim_arm()


def test_admission_rejects_before_claiming():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    with pytest.raises(Rejection, match="target out of joint limits"):
        admit_joints(layer, (9.0,) * 5, 0.0)
    # A rejected goal never left the slot held.
    assert coordinator.try_claim_arm()


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
    assert ctx.completed[0] is True


def test_pose_admission_rejects_busy_before_solving():
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    admit_joints(layer, (0.1,) * 5, 0.0)
    with pytest.raises(Rejection, match="arm is already executing a motion"):
        admit_pose(layer, (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
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
    goal = admit_pose(layer, (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    assert isinstance(goal, PoseMove)
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
    success, message, position, _orientation, _t = ctx.completed
    assert (success, message) == (True, "cartesian move complete")
    assert position == list(kinematics.forward_kinematics(kinematics.solution)[0])


async def test_a_pose_move_outside_its_tolerance_at_the_timeout_fails_with_the_shortfall(
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
    success, message, position, _orientation, _t = ctx.completed
    assert success is False
    assert message == "still 16.3 mm / 0.00 rad from the target, 2.0 s after the planned end"
    # The result gives the grasp point of the arm, not the goal.
    assert position == list(kinematics.forward_kinematics((0.0,) * 5)[0])
    assert coordinator.try_claim_arm()


async def test_a_stale_follower_reports_the_commanded_pose_with_a_note(follower_never_stale):
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    ctx, task, _plan = await pose_move_at_the_end_of_its_plan(
        coordinator, layer, kinematics, measured_at_end=kinematics.solution
    )
    # The follower sends no more state after it receives the last sample.
    # The last command of a finished plan is its end sample.
    coordinator.measured_joints.clear()
    await task
    success, message, position, _orientation, _t = ctx.completed
    assert (success, message) == (True, f"cartesian move complete; {STALE_COMMANDED}")
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


async def test_unreachable_pose_completes_failed_and_releases():
    kinematics = FakeKinematics()
    kinematics.corrupted = True  # solve returns None
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = admit_pose(layer, (5.0, 5.0, 5.0), (0.0, 0.0, 0.0, 1.0))
    ctx = FakeGoalContext()
    await layer.drive_pose(ctx, goal, kinematics, kinematics)
    success, message, _position, _orientation, _t = ctx.completed
    assert success is False
    assert message.startswith("goal pose not planned within ")
    # The refusal must say the search was local, so a caller knows to retry
    # from elsewhere rather than read it as the pose being out of reach.
    assert "local descent" in message
    assert coordinator.try_claim_arm()


async def test_abandoned_pose_admission_releases_the_claim():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    pending = PendingSlot()
    pending.plan = admit_pose(layer, (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    layer.abandon(pending)
    assert coordinator.try_claim_arm()


def test_absurd_duration_is_refused():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    with pytest.raises(Rejection, match="duration_s must be a number of seconds from 0 to 60"):
        admit_joints(layer, (0.1,) * 5, MAX_REQUESTED_DURATION_S + 1.0)
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
    success, message, _position, _orientation, _t = ctx.completed
    assert success is False
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
    for plan, settle in (
        ((bad, 0.2), SETTLE),
        ((0.01, bad), SETTLE),
        (PLAN, (bad, 0.2, 2.0)),
        (PLAN, (0.01, bad, 2.0)),
    ):
        with pytest.raises(Rejection, match="tolerance_(m|rad) must be a positive number"):
            admit_pose(layer, *pose, plan=plan, settle=settle)
    with pytest.raises(Rejection, match="each settle_tolerances_rad entry must be a positive number"):
        admit_joints(layer, (0.1,) * 5, tolerances=(0.1, 0.1, bad, 0.1, 0.1))
    assert coordinator.try_claim_arm(), "a refused tolerance must not hold the slot"


@pytest.mark.parametrize("bad", [-0.001, math.nan, math.inf, MAX_REQUESTED_DURATION_S + 1.0])
async def test_a_settle_timeout_that_is_no_duration_is_refused(bad):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    coordinator.measured_gripper.set(0.5)
    refusal = "settle_timeout_s must be a number of seconds from 0 to 60"
    with pytest.raises(Rejection, match=refusal):
        admit_pose(layer, (0.1, 0.0, 0.2), IDENTITY_QUAT, settle=(0.01, 0.2, bad))
    with pytest.raises(Rejection, match=refusal):
        admit_joints(layer, (0.1,) * 5, timeout_s=bad)
    assert coordinator.try_claim_arm()


def test_a_settle_tolerance_below_the_plan_tolerance_of_its_axis_is_refused():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    refusal = "settle_{0} must be at least plan_{0}: the plan may sit that far from the pose"
    with pytest.raises(Rejection, match=refusal.format("position_tolerance_m")):
        admit_pose(layer, (0.1, 0.0, 0.2), IDENTITY_QUAT, settle=(0.005, 0.2, 2.0))
    with pytest.raises(Rejection, match=refusal.format("orientation_tolerance_rad")):
        admit_pose(layer, (0.1, 0.0, 0.2), IDENTITY_QUAT, settle=(0.01, 0.1, 2.0))
    assert coordinator.try_claim_arm()


async def test_a_refused_tolerance_names_the_bar_it_was_judged_against():
    kinematics = FakeKinematics()
    kinematics.corrupted = True
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    move = admit_pose(
        layer, (0.1, 0.0, 0.2), IDENTITY_QUAT, plan=(0.004, 0.05), settle=(0.004, 0.05, 2.0)
    )
    ctx = FakeGoalContext()
    await layer.drive_pose(ctx, move, kinematics, kinematics)
    success, message, _p, _o, _t = ctx.completed
    assert success is False
    assert "4.0 mm" in message and "2.9 deg" in message


async def test_the_ee_caps_cannot_pin_the_slot_past_the_duration_ceiling():
    # A move admitted under the ceiling can still need longer than it once
    # the end-effector speed caps are applied; that must refuse, not stretch.
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics, max_ee_velocity_m_s=1e-4)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = admit_pose(layer, (5.0, 0.0, 0.2), IDENTITY_QUAT)
    ctx = FakeGoalContext()
    await layer.drive_pose(ctx, goal, kinematics, kinematics)
    success, message, _p, _o, _t = ctx.completed
    assert success is False
    assert "ceiling" in message
    assert coordinator.try_claim_arm(), "a refused move must release the arm"


async def test_a_stopped_plan_completes_cancelled_with_the_stops_message(follower_never_stale):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    plan = admit_joints(layer, (0.5,) * 5, 1.0)
    ctx = FakeGoalContext()
    driving = asyncio.create_task(layer.drive_arm(ctx, plan))
    await asyncio.sleep(0)

    assert layer.stop_moves("the operator asked") == ["arm"]
    await driving

    assert ctx.completed is None
    success, message, positions, _action_time = ctx.cancelled
    assert success is False
    assert message == "stopped: the operator asked"
    assert positions == [0.0] * 5
    assert coordinator.try_claim_arm(), "the goal released the arm"


async def test_a_stopped_gripper_plan_completes_cancelled_with_the_stops_message(
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

    success, message, _opening, _action_time = ctx.cancelled
    assert success is False
    assert message == "stopped"
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
    success, message, _position, _orientation, _t = ctx.cancelled
    assert success is False
    assert message == "stopped: the operator asked"
    assert coordinator._arm_plan is None, "nothing was installed after the stop"
    assert coordinator.try_claim_arm()


async def test_a_stop_before_the_pose_drive_runs_ends_the_goal_as_stopped():
    # The stop lands between the goal's admission and its drive task: the
    # goal ends as cancelled with the stop's message, and nothing is solved.
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)
    goal = admit_pose(layer, (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    assert layer.stop_moves("the operator asked") == []
    ctx = FakeGoalContext()
    await layer.drive_pose(ctx, goal, kinematics, kinematics)

    assert ctx.completed is None
    success, message, position, _orientation, _t = ctx.cancelled
    assert (success, message) == (False, "stopped: the operator asked")
    assert position == list(kinematics.forward_kinematics((0.0,) * 5)[0])
    assert coordinator._arm_plan is None
    assert coordinator.try_claim_arm()


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
    success, message, _positions, _action_time = ctx.completed
    assert success is False
    assert message == "still 0.500 rad from the target at joint 1, 0.0 s after the planned end"
    assert coordinator.try_claim_arm()


async def test_the_plan_check_answers_the_duration_and_installs_nothing():
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics, max_ee_velocity_m_s=0.1)
    coordinator.measured_joints.set((0.0,) * 5)

    success, message, duration_s = await layer.check_arm_move(
        "arm", (0.2, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, *PLAN, kinematics, kinematics
    )

    assert success is True
    assert duration_s > 0.0
    assert message == f"a plan reaches the pose in {duration_s:.3f} s"
    assert ("solve",) == tuple(c[0] for c in kinematics.solve_calls)
    assert coordinator._arm_plan is None
    assert coordinator.try_claim_arm(), "the check claims nothing"


async def test_the_plan_check_refuses_as_the_move_would():
    kinematics = FakeKinematics()
    coordinator, layer = make_layer(kinematics)
    coordinator.measured_joints.set((0.0,) * 5)

    kinematics.corrupted = True
    success, message, duration_s = await layer.check_arm_move(
        "arm", (5.0, 5.0, 5.0), (0.0, 0.0, 0.0, 1.0), 0.0, *PLAN, kinematics, kinematics
    )
    assert success is False
    assert message.startswith("goal pose not planned within ")
    assert duration_s == 0.0

    kinematics.corrupted = False
    success, message, _ = await layer.check_arm_move(
        "wing", (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, *PLAN, kinematics, kinematics
    )
    assert success is False
    assert "unknown arm_name" in message
    success, message, _ = await layer.check_arm_move(
        "arm", (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 0.0), 0.0, *PLAN, kinematics, kinematics
    )
    assert success is False
    assert message == "goal pose has an unnormalizable orientation"

    # While an arm goal is in flight, the check is refused as a second goal is.
    admit_joints(layer, (0.5,) * 5, 1.0)
    success, message, _ = await layer.check_arm_move(
        "arm", (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, *PLAN, kinematics, kinematics
    )
    assert success is False
    assert message == "arm is already executing a motion"
    assert len(kinematics.solve_calls) == 1, "a refused check pays no solve"


async def test_the_plan_check_needs_fresh_follower_state():
    kinematics = FakeKinematics()
    _coordinator, layer = make_layer(kinematics)
    success, message, _ = await layer.check_arm_move(
        "arm", (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, *PLAN, kinematics, kinematics
    )
    assert success is False
    assert message == "no fresh follower state to anchor the move from"


def test_a_streaming_leader_keeps_the_limbs_until_its_stream_stops(monkeypatch):
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    coordinator.measured_gripper.set(0.5)
    coordinator.leader_joints.set((0.0,) * 5)
    coordinator.leader_gripper.set(0.5)
    pose = ((0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(Rejection, match="leader is streaming arm"):
        admit_joints(layer, (0.1,) * 5, 0.0)
    with pytest.raises(Rejection, match="leader is streaming arm"):
        admit_pose(layer, *pose)
    with pytest.raises(Rejection, match="leader is streaming gripper"):
        admit_gripper(layer, 0.2, 0.0)
    # The refusals held no slot.
    assert coordinator.try_claim_arm() and coordinator.try_claim_gripper()
    coordinator.release_arm()
    coordinator.release_gripper()
    # A goal boundary clears the leader holders; the leader still streams.
    with pytest.raises(Rejection, match="leader is streaming arm"):
        admit_joints(layer, (0.1,) * 5, 0.0)
    with pytest.raises(Rejection, match="leader is streaming gripper"):
        admit_gripper(layer, 0.2, 0.0)
    # The leader lets go: past the deadman, the limbs take goals again. The
    # follower state is refreshed so only the leader's age moves.
    later = time.monotonic() + STALE_LEADER_TIMEOUT_S + 0.01
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
    ok, message, _duration_s = await layer.check_arm_move("arm", *pose, 0.0, *PLAN, None, None)
    assert (ok, message) == (False, NO_ANCHOR)
    with pytest.raises(Rejection, match=NO_ANCHOR):
        admit_pose(layer, *pose)


async def test_the_move_check_answers_the_streaming_refusal_without_claiming():
    coordinator, layer = make_layer()
    coordinator.measured_joints.set((0.0,) * 5)
    coordinator.leader_joints.set((0.0,) * 5)
    ok, message, duration_s = await layer.check_arm_move(
        "arm", (0.1, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), 0.0, *PLAN, None, None
    )
    assert (ok, message, duration_s) == (False, ARM_LEADER_STREAMING, 0.0)
    assert coordinator.try_claim_arm()
