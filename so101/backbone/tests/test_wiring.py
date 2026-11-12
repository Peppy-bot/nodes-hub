"""In-process harness tests: the node booted against mocked pairing peers,
covering the teleop path, the relays, the action surface and the workspace
services end to end."""

import asyncio
import contextlib
import math
import re
import time

import peppylib
import pytest
import robot_description_py
from conftest import (
    CLOSE,
    FAR,
    FAR_SHORT_BY_MORE_THAN,
    MAX_JOINT_ACCELERATION_RAD_S2,
    MEASURED,
    NEAR,
    START_TOLERANCE_RAD,
    assert_rectangle_inside_reach,
    joint_path_goal,
    path_result_joints,
)
from path_timing_py import JointTrajectory
from peppygen.exposed_actions.limb_motion import move_arm as move_arm_prod
from peppygen.exposed_actions.limb_motion import move_arm_joints as move_arm_joints_prod
from peppygen.exposed_actions.limb_motion import move_gripper as move_gripper_prod
from peppygen.exposed_actions.postures import move_to_home as move_to_home_prod
from peppygen.exposed_actions.postures import move_to_ready as move_to_ready_prod
from peppygen.fixtures import harness
from peppygen.fixtures.exposed_actions.limb_motion import (
    execute_joint_path as execute_joint_path_fx,
)
from peppygen.fixtures.exposed_actions.limb_motion import move_arm as move_arm_fx
from peppygen.fixtures.exposed_actions.limb_motion import (
    move_arm_joints as move_arm_joints_fx,
)
from peppygen.fixtures.exposed_actions.limb_motion import (
    move_gripper as move_gripper_fx,
)
from peppygen.fixtures.exposed_actions.postures import move_to_home as move_to_home_fx
from peppygen.fixtures.exposed_actions.postures import move_to_ready as move_to_ready_fx
from peppygen.exposed_services.limb_motion import check_arm_move as check_arm_move_prod
from peppygen.exposed_services.limb_motion import stop as stop_prod
from peppygen.exposed_services.workspace import check_positions as check_positions_prod
from peppygen.exposed_services.workspace import (
    describe_workspace as describe_workspace_prod,
)
from peppygen.fixtures.exposed_services.limb_motion import (
    check_arm_move as check_arm_move_fx,
)
from peppygen.fixtures.exposed_services.limb_motion import stop as stop_fx
from peppygen.fixtures.exposed_services.limb_state import (
    get_limb_names as get_limb_names_fx,
)
from peppygen.fixtures.exposed_services.workspace import (
    check_positions as check_positions_fx,
)
from peppygen.fixtures.exposed_services.workspace import (
    describe_workspace as describe_workspace_fx,
)
from peppygen.mock.deps.robot_identity import (
    GetDescriptionResponseData,
    GetIdentityResponseData,
)
from peppygen.paired_topics.arm import joint_states as arm_states_topic
from peppygen.paired_topics.leader_arm import joint_states as leader_joint_states_topic
from peppygen.paired_topics.gripper import gripper_states as gripper_states_topic
from peppygen.paired_topics.leader_arm import joint_setpoints as leader_setpoints_topic
from peppygen.paired_topics.leader_pose import pose_setpoints as leader_pose_topic
from peppygen.parameters import Parameters
from workspace_core_py import REACH_TOLERANCE

from so101_backbone import description_check, postures
from so101_backbone.__main__ import setup
from so101_backbone.coordinator import GRIPPER_STILL_WINDOW_S
from so101_backbone.kinematics import Kinematics
from so101_backbone.model import MODEL, urdf_path

TIMEOUT_S = 10.0
# The bound of an answer that measures reach: the first starts the worker
# process and measures a whole surface. A hang guard, not a measure of speed.
WORKSPACE_TIMEOUT_S = 60.0

# The grasp pose of MEASURED, solved once at import, outside the event loop of
# every test: building a solver takes longer than the follower's state stays
# fresh, and each test reads that state while its feeder runs.
MEASURED_POSE = Kinematics(urdf_path()).forward_kinematics(MEASURED)


def make_parameters(**overrides) -> Parameters:
    base = {
        "upstream_mode": "joints",
        "control_rate_hz": 100,
        "max_joint_velocity_rad_s_1": 8.0,
        "max_joint_velocity_rad_s_2": 8.0,
        "max_joint_velocity_rad_s_3": 8.0,
        "max_joint_velocity_rad_s_4": 8.0,
        "max_joint_velocity_rad_s_5": 8.0,
        "max_joint_acceleration_rad_s2_1": MAX_JOINT_ACCELERATION_RAD_S2,
        "max_joint_acceleration_rad_s2_2": MAX_JOINT_ACCELERATION_RAD_S2,
        "max_joint_acceleration_rad_s2_3": MAX_JOINT_ACCELERATION_RAD_S2,
        "max_joint_acceleration_rad_s2_4": MAX_JOINT_ACCELERATION_RAD_S2,
        "max_joint_acceleration_rad_s2_5": MAX_JOINT_ACCELERATION_RAD_S2,
        "max_stream_ee_velocity_m_s": 1e9,
        "max_gripper_rate_frac_s": 8.0,
        "start_tolerance_rad": START_TOLERANCE_RAD,
    }
    return Parameters(**{**base, **overrides})


async def answer_identity(h, model):
    """Answers the node's request for who the robot is with `model`."""
    responder = await h.mocks.deps.robot_identity.get_identity.next_request(TIMEOUT_S)
    await responder.respond(
        GetIdentityResponseData(robot="so101_under_test", model=model, core_node="harness")
    )


async def answer_description(h, digest):
    """Answers the node's request for the robot's description with the SO-101
    model's files and `digest`."""
    responder = await h.mocks.deps.robot_identity.get_description.next_request(TIMEOUT_S)
    await responder.respond(
        GetDescriptionResponseData(urdf=MODEL.urdf, srdf=MODEL.srdf, meshes=[], digest=digest)
    )


async def answer_as_the_initializer_of(h, model):
    """Answers every request of the node for who the robot is and for its
    description as the initializer of a robot of `model` does, until cancelled."""
    served = robot_description_py.model(model)
    identity = h.mocks.deps.robot_identity.get_identity
    description = h.mocks.deps.robot_identity.get_description

    async def identities():
        while True:
            responder = await identity.next_request(TIMEOUT_S)
            await responder.respond(
                GetIdentityResponseData(robot="robot_under_test", model=model, core_node="harness")
            )

    async def descriptions():
        while True:
            responder = await description.next_request(TIMEOUT_S)
            await responder.respond(
                GetDescriptionResponseData(
                    urdf=served.urdf, srdf=served.srdf, meshes=[], digest=served.digest
                )
            )

    await asyncio.gather(identities(), descriptions())


@contextlib.asynccontextmanager
async def started(parameters=None, **options):
    """The node booted on `parameters` (make_parameters() when None) and the
    harness `options`, its initializer serving the description this backbone
    was built with."""
    if parameters is None:
        parameters = make_parameters()
    async with harness.start(setup, parameters=parameters, **options) as h:
        await answer_identity(h, MODEL.id)
        await answer_description(h, MODEL.digest)
        yield h


async def setup_returned(h):
    while not h.setup_finished():
        await asyncio.sleep(0.01)


async def publish_measured(h, positions=MEASURED, opening=0.0):
    await h.mocks.pairings.arm.joint_states.publish(
        arm_states_topic.Message(
            timestamp=time.time(), positions=positions, velocities=[], efforts=[]
        )
    )
    await h.mocks.pairings.gripper.gripper_states.publish(
        gripper_states_topic.Message(
            timestamp=time.time(), opening=opening, effort=0.0, max_effort=0.0
        )
    )


async def keep_measured_fresh(h, positions=MEASURED, opening=0.0):
    while True:
        await publish_measured(h, positions, opening)
        await asyncio.sleep(0.01)


async def feeding_measured(h, positions=MEASURED, opening=0.0) -> asyncio.Task:
    """Keeps the follower's measured state fresh and returns the feeder once
    the node holds it. The node refuses a goal until it has adopted the
    follower's state as its anchor, and it relays a sample up the leader
    slots only after adopting it, so one relayed arm state and one relayed
    gripper state gate every goal."""
    feeder = asyncio.create_task(keep_measured_fresh(h, positions, opening))
    try:
        await asyncio.wait_for(h.mocks.pairings.leader_arm.joint_states.next(), TIMEOUT_S)
        await asyncio.wait_for(h.mocks.pairings.leader_gripper.gripper_states.next(), TIMEOUT_S)
    except BaseException:
        feeder.cancel()
        raise
    return feeder


async def test_the_joints_stream_passes_through_unchanged_and_relays_state():
    transparent = {f"max_joint_velocity_rad_s_{joint}": 1e9 for joint in range(1, 6)}
    async with started(parameters=make_parameters(**transparent)) as h:
        feeder = asyncio.create_task(keep_measured_fresh(h))
        try:
            # The relay carries the measured state back up the leader slot.
            relayed = await asyncio.wait_for(
                h.mocks.pairings.leader_arm.joint_states.next(), TIMEOUT_S
            )
            assert relayed.positions == pytest.approx(MEASURED)

            target = [p + 1.0 for p in MEASURED]
            stamps: list[float] = []
            streamer = asyncio.create_task(_stream_leader_target(h, target, stamps))
            try:
                first = await asyncio.wait_for(
                    h.mocks.pairings.arm.joint_setpoints.next(), TIMEOUT_S
                )
                # Pass-through: the leader's exact values go downstream (the
                # target lies inside the joint limits, and the joint velocity
                # limits and the EE cap are transparent here).
                assert first.positions == pytest.approx(target)
                assert first.efforts == []
                # And the downstream stamp is the leader's capture stamp,
                # not this node's publish time.
                assert any(
                    abs(first.timestamp - s) < 1e-6 for s in stamps
                )
            finally:
                streamer.cancel()
        finally:
            feeder.cancel()


async def test_the_backbone_checks_the_served_description_after_its_setup_returned():
    # Setup returns before the initializer answers: it may start after this node.
    async with harness.start(setup, parameters=make_parameters()) as h:
        await asyncio.wait_for(setup_returned(h), TIMEOUT_S)
        await answer_identity(h, MODEL.id)
        await answer_description(h, MODEL.digest)


async def test_a_backbone_whose_initializer_serves_another_description_ends():
    other = "the digest of another description"
    refusal = (
        f"serves the description {other} of so101, and this backbone was built with {MODEL.digest}"
    )
    async with harness.start(setup, parameters=make_parameters()) as h:
        # The check this test runs asks beside the one the node runs; the
        # initializer answers both with the other description.
        checking = asyncio.ensure_future(description_check.check(h.node_runner))
        await answer_identity(h, MODEL.id)
        await answer_identity(h, MODEL.id)
        await answer_description(h, other)
        await answer_description(h, other)
        with pytest.raises(ValueError, match=refusal):
            await asyncio.wait_for(checking, TIMEOUT_S)


async def test_a_backbone_whose_initializer_serves_another_model_ends_naming_both():
    other = "openarm_v2"
    refusal = (
        f"the robot's initializer serves model {other}, and this backbone drives so101: "
        "link robot_identity to the initializer of an so101 robot"
    )
    async with harness.start(setup, parameters=make_parameters()) as h:
        # The check this test runs asks beside the one the node runs; the
        # initializer of an OpenArm v2 answers both.
        checking = asyncio.ensure_future(description_check.check(h.node_runner))
        initializer = asyncio.ensure_future(answer_as_the_initializer_of(h, other))
        try:
            with pytest.raises(ValueError, match=refusal):
                await asyncio.wait_for(checking, TIMEOUT_S)
        finally:
            initializer.cancel()


async def test_the_limb_names_are_answered_before_any_snapshot():
    # No follower reports here, so no readout has been published: the names
    # and the joint count behind them come from the robot itself.
    async with started() as h:
        names = await get_limb_names_fx.poll(h, TIMEOUT_S)
        assert names.arm_names == ["arm"]
        assert names.joints_per_arm == [5]
        assert names.gripper_names == ["gripper"]


async def test_malformed_follower_state_neither_anchors_nor_relays():
    # Measured state is a trust boundary: a NaN vector must not become a
    # governor anchor, and must not be relayed upstream as if the follower
    # had said something coherent. Wrong-length vectors are the contract's
    # job now and cannot even be built; see the cardinality test below.
    async with started() as h:
        for positions in ([float("nan")] * 5,):
            await h.mocks.pairings.arm.joint_states.publish(
                arm_states_topic.Message(
                    timestamp=time.time(), positions=positions, velocities=[], efforts=[]
                )
            )
        await asyncio.sleep(0.2)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(h.mocks.pairings.leader_arm.joint_states.next(), 0.3)

        # A well-formed sample still gets through, so the drop was the guard
        # and not a dead relay.
        await publish_measured(h)
        relayed = await asyncio.wait_for(
            h.mocks.pairings.leader_arm.joint_states.next(), TIMEOUT_S
        )
        assert relayed.positions == pytest.approx(MEASURED)


async def test_backlog_replayed_follower_state_is_not_anchored_or_relayed():
    # Peppy subscriptions are per-producer FIFOs, so a backlog drained after
    # a stall arrives with fresh arrival stamps; only the producer's wire
    # stamp can tell it apart from current measurement.
    async with started() as h:
        await h.mocks.pairings.arm.joint_states.publish(
            arm_states_topic.Message(
                timestamp=time.time() - 60.0, positions=MEASURED, velocities=[], efforts=[]
            )
        )
        await asyncio.sleep(0.2)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(h.mocks.pairings.leader_arm.joint_states.next(), 0.3)


async def test_pose_mode_streams_downstream_under_the_leader_capture_stamp():
    # The pose path reads a different upstream slot, solves IK inside the
    # tick, and rate-steps the result; none of that may cost the leader's
    # capture stamp, or the follower's wire-age deadman judges this node's
    # publish time instead of when the headset sampled the hand.
    async with started(
        parameters=make_parameters(upstream_mode="pose"),
        leader_arm_vacant=True,
    ) as h:
        feeder = asyncio.create_task(keep_measured_fresh(h))
        stamps: list[float] = []

        async def stream_pose():
            while True:
                stamp = time.time()
                stamps.append(stamp)
                await h.mocks.pairings.leader_pose.pose_setpoints.publish(
                    leader_pose_topic.Message(
                        timestamp=stamp,
                        position=[0.15, 0.0, 0.2],
                        orientation=[0.0, 0.0, 0.0, 1.0],
                    )
                )
                await asyncio.sleep(0.02)

        streamer = asyncio.create_task(stream_pose())
        try:
            downstream = await asyncio.wait_for(
                h.mocks.pairings.arm.joint_setpoints.next(), TIMEOUT_S
            )
            assert len(downstream.positions) == 5
            assert downstream.efforts == []
            assert any(abs(downstream.timestamp - s) < 1e-6 for s in stamps), (
                "the pose stream published under this node's clock, not the leader's stamp"
            )
        finally:
            streamer.cancel()
            feeder.cancel()


async def _stream_leader_target(h, target, stamps: list[float]):
    while True:
        stamp = time.time()
        stamps.append(stamp)
        await h.mocks.pairings.leader_arm.joint_setpoints.publish(
            leader_setpoints_topic.Message(
                timestamp=stamp, positions=target, velocities=[], efforts=[]
            )
        )
        await asyncio.sleep(0.02)


async def test_move_arm_joints_completes_and_reports():
    async with started() as h:
        feeder = await feeding_measured(h)
        try:
            # The mocked follower keeps measuring MEASURED, within the 0.1 rad
            # settle tolerance of every joint's target, so the move completes
            # at its planned end.
            target = [p + 0.05 for p in MEASURED]
            goal = await move_arm_joints_fx.send_goal(
                h,
                move_arm_joints_prod.GoalRequestData(
                    arm_name="arm", joint_positions=target, duration_s=0.0,
                    settle_tolerances_rad=[0.1] * 5, settle_timeout_s=2.0,
                ),
                TIMEOUT_S,
            )
            assert goal.accepted
            result = await goal.get_result(TIMEOUT_S)
            assert result.status == move_arm_joints_fx.ResultStatus.COMPLETED
            assert result.data.final_joint_positions == pytest.approx(MEASURED)
            # The trajectory's last downstream setpoint lands on the target.
            last = None
            while True:
                try:
                    last = await asyncio.wait_for(
                        h.mocks.pairings.arm.joint_setpoints.next(), 0.5
                    )
                except TimeoutError:
                    break
            assert last is not None
            assert last.positions == pytest.approx(target)
        finally:
            feeder.cancel()


async def test_a_joint_path_completes_and_reports():
    async with started() as h:
        feeder = await feeding_measured(h)
        try:
            # The path turns at its inner waypoint and ends on the target. The
            # mocked follower keeps measuring MEASURED, within the 0.1 rad
            # settle tolerance of every joint's target, so the move completes
            # at its planned end.
            inner = [MEASURED[0] + 0.3, *MEASURED[1:]]
            target = [p + 0.05 for p in MEASURED]
            goal = await execute_joint_path_fx.send_goal(
                h, joint_path_goal(inner, target, expected_start=MEASURED), TIMEOUT_S
            )
            assert goal.accepted
            result = await goal.get_result(TIMEOUT_S)
            assert result.status == execute_joint_path_fx.ResultStatus.COMPLETED
            assert path_result_joints(result.data.arms) == pytest.approx(MEASURED)
            # The trajectory's last downstream setpoint lands on the target.
            last = None
            while True:
                try:
                    last = await asyncio.wait_for(
                        h.mocks.pairings.arm.joint_setpoints.next(), 0.5
                    )
                except TimeoutError:
                    break
            assert last is not None
            assert last.positions == pytest.approx(target)
        finally:
            feeder.cancel()


async def test_move_arm_joints_outside_its_tolerances_fails_with_timeout():
    async with started() as h:
        feeder = await feeding_measured(h)
        try:
            # The mocked follower never leaves MEASURED, 0.2 rad from every
            # joint's target, outside the 0.1 rad tolerance.
            goal = await move_arm_joints_fx.send_goal(
                h,
                move_arm_joints_prod.GoalRequestData(
                    arm_name="arm",
                    joint_positions=[p + 0.2 for p in MEASURED],
                    duration_s=0.0,
                    settle_tolerances_rad=[0.1] * 5,
                    settle_timeout_s=0.3,
                ),
                TIMEOUT_S,
            )
            assert goal.accepted
            result = await goal.get_result(TIMEOUT_S)
            assert result.status == move_arm_joints_fx.ResultStatus.FAILED
            # A failed goal gives no result.
            assert result.data is None
            assert result.failure.kind is move_arm_joints_fx.FailureKind.TIMEOUT
            # Every joint is 0.2 rad short; the one named is whichever float
            # sum lands a few ulps farther out.
            assert re.fullmatch(
                r"still 0\.200 rad from the target at joint [1-5], \d+\.\d s after the planned end",
                result.failure.message,
            ), result.failure.message
        finally:
            feeder.cancel()


async def test_a_joint_path_outside_its_tolerances_fails_with_timeout():
    async with started() as h:
        feeder = await feeding_measured(h)
        try:
            # The mocked follower never leaves MEASURED, 0.2 rad from every
            # joint's target, outside the 0.1 rad tolerance.
            goal = await execute_joint_path_fx.send_goal(
                h, joint_path_goal([p + 0.2 for p in MEASURED], settle_timeout_s=0.3), TIMEOUT_S
            )
            assert goal.accepted
            result = await goal.get_result(TIMEOUT_S)
            assert result.status == execute_joint_path_fx.ResultStatus.FAILED
            # A failed goal gives no result.
            assert result.data is None
            assert result.failure.kind is execute_joint_path_fx.FailureKind.TIMEOUT
            # Every joint is 0.2 rad short; the one named is whichever float
            # sum lands a few ulps farther out.
            assert re.fullmatch(
                r"still 0\.200 rad from the target at joint [1-5], \d+\.\d s after the planned end",
                result.failure.message,
            ), result.failure.message
        finally:
            feeder.cancel()


async def test_second_goal_is_refused_while_busy_and_a_cancel_ends_the_first():
    async with started() as h:
        # The in-flight goal outlives STALE_FOLLOWER_TIMEOUT_S, and a plan
        # failed for staleness would cancel before cancel_goal ever runs.
        feeder = await feeding_measured(h)
        try:
            slow = await move_arm_joints_fx.send_goal(
                h,
                move_arm_joints_prod.GoalRequestData(
                    arm_name="arm",
                    joint_positions=[p + 0.5 for p in MEASURED],
                    duration_s=5.0,
                    settle_tolerances_rad=[0.1] * 5,
                    settle_timeout_s=2.0,
                ),
                TIMEOUT_S,
            )
            assert slow.accepted
            second = await move_arm_joints_fx.send_goal(
                h,
                move_arm_joints_prod.GoalRequestData(
                    arm_name="arm", joint_positions=MEASURED, duration_s=0.0,
                    settle_tolerances_rad=[0.1] * 5, settle_timeout_s=2.0,
                ),
                TIMEOUT_S,
            )
            assert second.refusal == move_arm_joints_fx.Failure(
                move_arm_joints_fx.FailureKind.PRECONDITION_FAILED,
                "arm is already executing a motion",
            )
            cancel = await slow.cancel_goal(TIMEOUT_S)
            assert cancel.state == move_arm_joints_fx.CancelState.SIGNALLED
            result = await slow.get_result(TIMEOUT_S)
            assert result.status == move_arm_joints_fx.ResultStatus.CANCELLED
            assert result.cancel_reason == "cancelled by its caller"
            assert len(result.data.final_joint_positions) == 5
        finally:
            feeder.cancel()


async def test_a_goal_of_either_joint_member_is_refused_while_a_path_runs():
    async with started() as h:
        feeder = await feeding_measured(h)
        try:
            slow = await execute_joint_path_fx.send_goal(
                h, joint_path_goal([p + 0.5 for p in MEASURED], duration_s=5.0), TIMEOUT_S
            )
            assert slow.accepted
            # A second goal of either joint-space member finds the arm busy.
            second = await execute_joint_path_fx.send_goal(
                h, joint_path_goal(MEASURED), TIMEOUT_S
            )
            assert second.refusal == execute_joint_path_fx.Failure(
                execute_joint_path_fx.FailureKind.PRECONDITION_FAILED,
                "arm is already executing a motion",
            )
            joints = await move_arm_joints_fx.send_goal(
                h,
                move_arm_joints_prod.GoalRequestData(
                    arm_name="arm", joint_positions=MEASURED, duration_s=0.0,
                    settle_tolerances_rad=[0.1] * 5, settle_timeout_s=2.0,
                ),
                TIMEOUT_S,
            )
            assert joints.refusal == move_arm_joints_fx.Failure(
                move_arm_joints_fx.FailureKind.PRECONDITION_FAILED,
                "arm is already executing a motion",
            )
            cancel = await slow.cancel_goal(TIMEOUT_S)
            assert cancel.state == execute_joint_path_fx.CancelState.SIGNALLED
            result = await slow.get_result(TIMEOUT_S)
            assert result.status == execute_joint_path_fx.ResultStatus.CANCELLED
            assert result.cancel_reason == "cancelled by its caller"
            assert len(path_result_joints(result.data.arms)) == 5
        finally:
            feeder.cancel()


async def test_goal_without_follower_state_is_refused():
    async with started() as h:
        goal = await move_arm_joints_fx.send_goal(
            h,
            move_arm_joints_prod.GoalRequestData(
                arm_name="arm", joint_positions=MEASURED, duration_s=0.0,
                settle_tolerances_rad=[0.1] * 5, settle_timeout_s=2.0,
            ),
            TIMEOUT_S,
        )
        assert goal.refusal.kind is move_arm_joints_fx.FailureKind.RESOURCE_UNAVAILABLE
        assert "no fresh follower state" in goal.refusal.message


async def test_a_joint_path_without_follower_state_is_refused():
    async with started() as h:
        goal = await execute_joint_path_fx.send_goal(h, joint_path_goal(MEASURED), TIMEOUT_S)
        assert goal.refusal.kind is execute_joint_path_fx.FailureKind.RESOURCE_UNAVAILABLE
        assert "no fresh follower state" in goal.refusal.message


async def test_posture_and_gripper_actions():
    async with started() as h:
        # The mocked follower stands at Ready, so move_to_ready settles there.
        feeder = await feeding_measured(h, positions=postures.READY_POSITIONS_RAD)
        try:
            posture = await move_to_ready_fx.send_goal(
                h,
                move_to_ready_prod.GoalRequestData(duration_s=0.0),
                TIMEOUT_S,
            )
            assert posture.accepted
            result = await posture.get_result(TIMEOUT_S)
            assert result.status == move_to_ready_fx.ResultStatus.COMPLETED

            grip = await move_gripper_fx.send_goal(
                h,
                move_gripper_prod.GoalRequestData(
                    gripper_name="gripper", opening=0.5, max_effort=0.0, settle_timeout_s=2.0
                ),
                TIMEOUT_S,
            )
            assert grip.accepted
            grip_result = await grip.get_result(TIMEOUT_S)
            assert grip_result.status == move_gripper_fx.ResultStatus.COMPLETED
            # The mocked follower never moves its jaws: past the planned end
            # of its travel the goal finds them standing still and reports
            # where they stand.
            assert grip_result.data.settled
            assert grip_result.data.final_opening == 0.0
            assert grip_result.data.action_time >= GRIPPER_STILL_WINDOW_S
            # The ramp's last downstream opening is the target.
            last = None
            while True:
                try:
                    last = await asyncio.wait_for(
                        h.mocks.pairings.gripper.gripper_setpoints.next(), 0.5
                    )
                except TimeoutError:
                    break
            assert last is not None
            assert math.isclose(last.opening, 0.5)
        finally:
            feeder.cancel()


async def test_a_posture_result_names_the_arm_and_its_measured_grasp_point(
    follower_never_stale,
):
    # One scenario for two facts of the result of each posture move:
    # - the result names the arm;
    # - it gives the grasp point of the measured joints, not of the posture.
    # The mocked follower measures the arm 0.05 rad off the posture in every
    # joint, inside the posture's settle. follower_never_stale keeps a pause
    # of the event loop from making the measured joints stale.
    kinematics = Kinematics(urdf_path())
    async with started() as h:
        for production, fixture, posture in (
            (move_to_ready_prod, move_to_ready_fx, postures.READY_POSITIONS_RAD),
            (move_to_home_prod, move_to_home_fx, postures.HOME_POSITIONS_RAD),
        ):
            near = [p + 0.05 for p in posture]
            measured_position, measured_orientation = kinematics.forward_kinematics(near)
            feeder = await feeding_measured(h, positions=near)
            try:
                goal = await fixture.send_goal(
                    h,
                    production.GoalRequestData(duration_s=0.0),
                    TIMEOUT_S,
                )
                assert goal.accepted
                result = await goal.get_result(TIMEOUT_S)
                assert result.status == fixture.ResultStatus.COMPLETED
                assert result.data.arm_names == ["arm"]
                assert result.data.positions == pytest.approx(list(measured_position))
                assert result.data.orientations == pytest.approx(list(measured_orientation))
                posture_position, _orientation = kinematics.forward_kinematics(posture)
                assert math.dist(result.data.positions, posture_position) > 1e-3
            finally:
                feeder.cancel()


@contextlib.asynccontextmanager
async def obedient_follower(h, positions, opening):
    """Play a follower that stands on each setpoint it takes, from
    `positions` and `opening`, and measures the arm and the gripper there.
    Gives the setpoints it took, in the order it took them, each as
    ("arm", positions) or ("gripper", opening). The node holds the
    follower's state once this is entered."""
    stands = {"positions": list(positions), "opening": opening}
    taken = []

    async def measure():
        while True:
            await publish_measured(h, stands["positions"], stands["opening"])
            await asyncio.sleep(0.01)

    async def take_arm_setpoints():
        while True:
            setpoint = await h.mocks.pairings.arm.joint_setpoints.next()
            stands["positions"] = list(setpoint.positions)
            taken.append(("arm", stands["positions"]))

    async def take_gripper_setpoints():
        while True:
            setpoint = await h.mocks.pairings.gripper.gripper_setpoints.next()
            stands["opening"] = setpoint.opening
            taken.append(("gripper", setpoint.opening))

    tasks = [
        asyncio.create_task(play())
        for play in (measure, take_arm_setpoints, take_gripper_setpoints)
    ]
    try:
        await asyncio.wait_for(h.mocks.pairings.leader_arm.joint_states.next(), TIMEOUT_S)
        await asyncio.wait_for(h.mocks.pairings.leader_gripper.gripper_states.next(), TIMEOUT_S)
        yield taken
    finally:
        for task in tasks:
            task.cancel()


async def test_move_to_home_closes_the_gripper_before_the_arm_moves(follower_never_stale):
    # The mocked follower stands on each setpoint, from Ready with the
    # gripper fully open. The goal completes once the arm stands at Home, so
    # every setpoint that counts is taken when its result comes.
    async with started() as h:
        async with obedient_follower(h, postures.READY_POSITIONS_RAD, 1.0) as taken:
            goal = await move_to_home_fx.send_goal(
                h,
                move_to_home_prod.GoalRequestData(duration_s=0.0),
                TIMEOUT_S,
            )
            assert goal.accepted
            result = await goal.get_result(TIMEOUT_S)
            assert result.status == move_to_home_fx.ResultStatus.COMPLETED
            assert result.data.arm_names == ["arm"]

            limbs = [limb for limb, _setpoint in taken]
            assert "arm" in limbs, "move_to_home sent the arm no setpoint"
            arm_starts = limbs.index("arm")
            assert "gripper" not in limbs[arm_starts:], "the arm moved before the close ended"
            # The close ramps the gripper down from fully open to closed.
            openings = [setpoint for limb, setpoint in taken if limb == "gripper"]
            assert openings, "move_to_home sent the gripper no opening"
            assert openings == sorted(openings, reverse=True)
            assert openings[-1] == 0.0
            # The arm then leaves Ready and stands at Home, within the
            # tolerance of a posture move.
            arm_setpoints = [setpoint for limb, setpoint in taken if limb == "arm"]
            assert arm_setpoints[-1] == pytest.approx(postures.HOME_POSITIONS_RAD, abs=0.1)


async def test_a_posture_the_arm_does_not_reach_fails_with_timeout(follower_never_stale):
    # The mocked follower always measures MEASURED, as an arm that an object
    # holds where it stands: move_to_ready fails, naming the joint.
    async with started() as h:
        feeder = await feeding_measured(h)
        try:
            goal = await move_to_ready_fx.send_goal(
                h,
                move_to_ready_prod.GoalRequestData(duration_s=0.0),
                TIMEOUT_S,
            )
            assert goal.accepted
            result = await goal.get_result(TIMEOUT_S)
            assert result.status == move_to_ready_fx.ResultStatus.FAILED
            assert result.failure.kind is move_to_ready_fx.FailureKind.TIMEOUT
            assert " rad from the target at joint 5, " in result.failure.message
        finally:
            feeder.cancel()


async def test_nonzero_max_effort_is_accepted_and_ignored():
    # Contract semantics: 0 = no preference, >0 = a preference the
    # implementer may ignore; both the stream and the action run the
    # opening, so generic limb_motion clients work unmodified.
    async with started() as h:
        feeder = await feeding_measured(h)
        try:
            goal = await move_gripper_fx.send_goal(
                h,
                move_gripper_prod.GoalRequestData(
                    gripper_name="gripper", opening=0.5, max_effort=2.0, settle_timeout_s=2.0
                ),
                TIMEOUT_S,
            )
            assert goal.accepted
            result = await goal.get_result(TIMEOUT_S)
            assert result.status == move_gripper_fx.ResultStatus.COMPLETED
        finally:
            feeder.cancel()


async def test_a_gripper_goals_max_effort_never_reaches_the_follower():
    async with started() as h:
        feeder = await feeding_measured(h)
        try:
            goal = await move_gripper_fx.send_goal(
                h,
                move_gripper_prod.GoalRequestData(
                    gripper_name="gripper", opening=0.5, max_effort=2.0, settle_timeout_s=2.0
                ),
                TIMEOUT_S,
            )
            assert goal.accepted
            # No leader sends gripper setpoints. Thus the first setpoint that
            # the follower receives is a step of this goal to its opening.
            setpoint = await asyncio.wait_for(
                h.mocks.pairings.gripper.gripper_setpoints.next(), TIMEOUT_S
            )
            assert 0.0 < setpoint.opening <= 0.5
            assert setpoint.max_effort == 0.0
            result = await goal.get_result(TIMEOUT_S)
            assert result.status == move_gripper_fx.ResultStatus.COMPLETED
        finally:
            feeder.cancel()


async def test_malformed_pose_goal_is_refused_and_the_server_survives():
    async with started() as h:
        feeder = await feeding_measured(h)
        try:
            # A zero quaternion once killed the move_arm server permanently.
            bad = await move_arm_fx.send_goal(
                h,
                move_arm_prod.GoalRequestData(
                    arm_name="arm",
                    position=[0.1, 0.0, 0.2],
                    orientation=[0.0, 0.0, 0.0, 0.0],
                    duration_s=0.0,
                    settle_position_tolerance_m=0.01,
                    settle_orientation_tolerance_rad=0.5,
                    settle_timeout_s=2.0,
                ),
                TIMEOUT_S,
            )
            assert bad.refusal == move_arm_fx.Failure(
                move_arm_fx.FailureKind.REJECTED, "goal pose has an unnormalizable orientation"
            )
            # The server must still decide the next goal (reachability of the
            # target pose is not the point; a decision is).
            again = await move_arm_fx.send_goal(
                h,
                move_arm_prod.GoalRequestData(
                    arm_name="arm",
                    position=[0.0, 0.0, 0.42],
                    orientation=[0.0, 0.0, 0.0, 1.0],
                    duration_s=0.0,
                    settle_position_tolerance_m=0.01,
                    settle_orientation_tolerance_rad=0.5,
                    settle_timeout_s=2.0,
                ),
                TIMEOUT_S,
            )
            if not again.accepted:
                assert "unnormalizable" not in again.refusal.message
        finally:
            feeder.cancel()


async def test_out_of_limit_goal_is_refused():
    async with started() as h:
        await publish_measured(h)
        goal = await move_arm_joints_fx.send_goal(
            h,
            move_arm_joints_prod.GoalRequestData(
                arm_name="arm", joint_positions=[9.0] * 5, duration_s=0.0,
                settle_tolerances_rad=[0.1] * 5, settle_timeout_s=2.0,
            ),
            TIMEOUT_S,
        )
        assert goal.refusal == move_arm_joints_fx.Failure(
            move_arm_joints_fx.FailureKind.REJECTED,
            "the target puts joint 1 at 9.0 rad, outside its limits [-1.91986, 1.91986]",
        )


async def test_unknown_arm_name_is_refused():
    async with started() as h:
        await publish_measured(h)
        goal = await move_arm_joints_fx.send_goal(
            h,
            move_arm_joints_prod.GoalRequestData(
                arm_name="left_arm", joint_positions=MEASURED, duration_s=0.0,
                settle_tolerances_rad=[0.1] * 5, settle_timeout_s=2.0,
            ),
            TIMEOUT_S,
        )
        assert goal.refusal.kind is move_arm_joints_fx.FailureKind.REJECTED
        assert "unknown arm_name" in goal.refusal.message


async def test_a_joint_path_goal_this_robot_does_not_take_is_refused():
    async with started() as h:
        await publish_measured(h)
        for goal_data, refusal in (
            (
                joint_path_goal(MEASURED, [9.0] * 5),
                "waypoint 2 puts joint 1 at 9.0 rad, outside its limits [-1.91986, 1.91986]",
            ),
            (
                joint_path_goal(MEASURED, arm_names=["left_arm"]),
                'unknown arm_name: this robot\'s arm is "arm"',
            ),
            (
                joint_path_goal(MEASURED, arm_names=["arm", "arm"]),
                'arms must name this robot\'s arm, "arm", once',
            ),
            (
                joint_path_goal(MEASURED, arm_names=[]),
                'arms must name this robot\'s arm, "arm", once',
            ),
            # The manifest leaves the expected start free, since an empty one
            # is not checked: the handler refuses another length.
            (
                joint_path_goal(MEASURED, expected_start=[0.0] * 4),
                "expected_start_joint_positions has 4 joint positions: give one per joint of "
                "the arm, 5, or none to skip the check",
            ),
        ):
            goal = await execute_joint_path_fx.send_goal(h, goal_data, TIMEOUT_S)
            assert goal.refusal == execute_joint_path_fx.Failure(
                execute_joint_path_fx.FailureKind.REJECTED, refusal
            )


async def test_a_path_runs_on_its_expected_start_or_none_and_fails_away_from_it():
    async with started() as h:
        feeder = await feeding_measured(h)
        try:
            # Joint 2 stands 0.2 rad from where the caller expects it, past
            # the start tolerance: the goal is accepted, and fails with no
            # result.
            assert START_TOLERANCE_RAD < 0.2
            expected = [MEASURED[0], MEASURED[1] + 0.2, *MEASURED[2:]]
            target = [p + 0.05 for p in MEASURED]
            away = await execute_joint_path_fx.send_goal(
                h, joint_path_goal(target, expected_start=expected), TIMEOUT_S
            )
            assert away.accepted
            failed = await away.get_result(TIMEOUT_S)
            assert failed.status == execute_joint_path_fx.ResultStatus.FAILED
            assert failed.data is None
            assert failed.failure.kind is execute_joint_path_fx.FailureKind.PRECONDITION_FAILED
            assert failed.failure.message.startswith(
                "arm stands 0.200 rad from expected_start_joint_positions at joint 2"
            ), failed.failure.message
            # The arm stands on the expected start, or the goal names none,
            # which is not checked: the path runs.
            for expected_start in (MEASURED, []):
                on = await execute_joint_path_fx.send_goal(
                    h, joint_path_goal(target, expected_start=expected_start), TIMEOUT_S
                )
                assert on.accepted
                result = await on.get_result(TIMEOUT_S)
                assert result.status == execute_joint_path_fx.ResultStatus.COMPLETED
        finally:
            feeder.cancel()


async def test_a_path_past_the_ceiling_at_the_caps_fails_with_rejected():
    # At 0.03 rad/s a leg of 1 rad on joint 1 takes some 33 s, and the path
    # out and back some 67 s, past the 60 s ceiling: the goal is accepted,
    # and fails with no result.
    slow = {f"max_joint_velocity_rad_s_{joint}": 0.03 for joint in range(1, 6)}
    out = [MEASURED[0] + 1.0, *MEASURED[1:]]
    out_and_back_s = JointTrajectory.through(
        (MEASURED, out, MEASURED), (0.03,) * 5, (MAX_JOINT_ACCELERATION_RAD_S2,) * 5, 0.0
    ).duration_s
    assert out_and_back_s > 60.0
    async with started(parameters=make_parameters(**slow)) as h:
        feeder = await feeding_measured(h)
        try:
            goal = await execute_joint_path_fx.send_goal(
                h, joint_path_goal(out, MEASURED), TIMEOUT_S
            )
            assert goal.accepted
            result = await goal.get_result(TIMEOUT_S)
            assert result.status == execute_joint_path_fx.ResultStatus.FAILED
            assert result.data is None
            assert result.failure == execute_joint_path_fx.Failure(
                execute_joint_path_fx.FailureKind.REJECTED,
                f"the joint velocity and acceleration caps take {out_and_back_s:.1f} s to run "
                "this path, past the 60 s a move may take; split it into paths of fewer waypoints",
            )
        finally:
            feeder.cancel()


async def test_stale_timestamped_leader_input_is_dropped():
    async with started() as h:
        feeder = await feeding_measured(h)
        try:
            # Legal shape, ancient capture stamp: the age gate must drop it,
            # so nothing streams downstream.
            for _ in range(5):
                await h.mocks.pairings.leader_arm.joint_setpoints.publish(
                    leader_setpoints_topic.Message(
                        timestamp=time.time() - 60.0,
                        positions=[1.0] * 5,
                        velocities=[],
                        efforts=[],
                    )
                )
                await asyncio.sleep(0.02)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    h.mocks.pairings.arm.joint_setpoints.next(), 0.4
                )
        finally:
            feeder.cancel()


async def test_the_wire_refuses_a_joint_path_waypoint_of_the_wrong_width():
    """The manifest pins each waypoint of a joint path to five joints, so a
    goal with a waypoint of another width cannot be put on the wire. Asserted
    here for the reason the test below gives."""
    async with started() as h:
        with pytest.raises(ValueError, match="expected 5"):
            await execute_joint_path_fx.send_goal(
                h, joint_path_goal(MEASURED, [0.0] * 4), TIMEOUT_S
            )


async def test_the_wire_refuses_a_joint_vector_of_the_wrong_width():
    """The manifest pins joint_link positions to five, so a wrong-width
    vector cannot be put on the wire at all. Asserted here because a refine
    block is easy to drop from a manifest by accident, and its loss would be
    invisible: the node would simply go back to trusting whatever arrived."""
    with pytest.raises(ValueError, match="expected 5"):
        leader_joint_states_topic.build_message(
            timestamp=time.time(), positions=[0.0] * 4, velocities=[], efforts=[]
        )


async def test_a_stop_ends_the_moves_in_flight_and_a_later_goal_runs():
    # A gripper rate of 0.1 per second gives the full-travel gripper move
    # below a planned end 10 s out, so the stop finds it in flight.
    async with started(
        parameters=make_parameters(max_gripper_rate_frac_s=0.1)
    ) as h:
        feeder = await feeding_measured(h)
        try:
            idle = await stop_fx.poll(h, stop_prod.RequestData(reason="nothing yet"), TIMEOUT_S)
            assert idle.stopped == []

            slow = await move_arm_joints_fx.send_goal(
                h,
                move_arm_joints_prod.GoalRequestData(
                    arm_name="arm",
                    joint_positions=[p + 0.5 for p in MEASURED],
                    duration_s=5.0,
                    settle_tolerances_rad=[0.1] * 5,
                    settle_timeout_s=2.0,
                ),
                TIMEOUT_S,
            )
            assert slow.accepted
            jaws = await move_gripper_fx.send_goal(
                h,
                move_gripper_prod.GoalRequestData(
                    gripper_name="gripper", opening=1.0, max_effort=0.0, settle_timeout_s=2.0
                ),
                TIMEOUT_S,
            )
            assert jaws.accepted
            # Both plans stream before the stop: a setpoint of each is out.
            await asyncio.wait_for(h.mocks.pairings.arm.joint_setpoints.next(), TIMEOUT_S)
            await asyncio.wait_for(h.mocks.pairings.gripper.gripper_setpoints.next(), TIMEOUT_S)

            answer = await stop_fx.poll(
                h, stop_prod.RequestData(reason="the operator asked"), TIMEOUT_S
            )
            assert answer.stopped == ["arm", "gripper"]

            # Each goal the stop ended ends cancelled, with a reason that
            # names the stop and its reason, and with its result.
            result = await slow.get_result(TIMEOUT_S)
            assert result.status == move_arm_joints_fx.ResultStatus.CANCELLED
            assert result.cancel_reason == "stopped: the operator asked"
            assert len(result.data.final_joint_positions) == 5
            result = await jaws.get_result(TIMEOUT_S)
            assert result.status == move_gripper_fx.ResultStatus.CANCELLED
            assert result.cancel_reason == "stopped: the operator asked"
            assert not result.data.settled

            # The stop does not latch: a later goal runs to its end.
            later = await move_arm_joints_fx.send_goal(
                h,
                move_arm_joints_prod.GoalRequestData(
                    arm_name="arm", joint_positions=MEASURED, duration_s=0.0,
                    settle_tolerances_rad=[0.1] * 5, settle_timeout_s=2.0,
                ),
                TIMEOUT_S,
            )
            assert later.accepted
            result = await later.get_result(TIMEOUT_S)
            assert result.status == move_arm_joints_fx.ResultStatus.COMPLETED
        finally:
            feeder.cancel()


async def test_the_plan_check_answers_without_moving(follower_never_stale):
    # follower_never_stale keeps a pause of the event loop from making the
    # measured joints stale.
    position, orientation = MEASURED_POSE
    async with started() as h:
        feeder = await feeding_measured(h)
        try:
            check = check_arm_move_prod.RequestData(
                arm_name="arm",
                position=[position[0], position[1], position[2] + 0.02],
                orientation=list(orientation),
                duration_s=0.0,
            )
            answer = await check_arm_move_fx.poll(h, check, TIMEOUT_S)
            assert (answer.reachable, answer.reason) == (True, "")
            assert answer.duration_s > 0.0

            far = check_arm_move_prod.RequestData(
                arm_name="arm",
                position=[position[0] + 2.0, position[1], position[2]],
                orientation=list(orientation),
                duration_s=0.0,
            )
            answer = await check_arm_move_fx.poll(h, far, TIMEOUT_S)
            assert not answer.reachable
            assert answer.reason.startswith("goal pose not planned")
            assert answer.duration_s == 0.0

            # Nothing streamed downstream: the checks moved nothing.
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(h.mocks.pairings.arm.joint_setpoints.next(), 0.5)

            # A check that move_arm would refuse gets an error reply with the
            # kind of that refusal.
            unknown = check_arm_move_prod.RequestData(
                arm_name="left_arm",
                position=list(position),
                orientation=list(orientation),
                duration_s=0.0,
            )
            with pytest.raises(check_arm_move_fx.Failure) as raised:
                await check_arm_move_fx.poll(h, unknown, TIMEOUT_S)
            assert raised.value.kind is check_arm_move_fx.FailureKind.REJECTED
            assert "unknown arm_name" in raised.value.message
        finally:
            feeder.cancel()


async def test_a_move_arm_goal_with_no_plan_fails_with_unreachable(follower_never_stale):
    # follower_never_stale keeps a pause of the event loop from making the
    # measured joints stale.
    position, orientation = MEASURED_POSE
    async with started() as h:
        feeder = await feeding_measured(h)
        try:
            goal = await move_arm_fx.send_goal(
                h,
                move_arm_prod.GoalRequestData(
                    arm_name="arm",
                    position=[position[0] + 2.0, position[1], position[2]],
                    orientation=list(orientation),
                    duration_s=0.0,
                    settle_position_tolerance_m=0.01,
                    settle_orientation_tolerance_rad=0.5,
                    settle_timeout_s=2.0,
                ),
                TIMEOUT_S,
            )
            assert goal.accepted
            result = await goal.get_result(TIMEOUT_S)
            assert result.status == move_arm_fx.ResultStatus.FAILED
            assert result.data is None
            assert result.failure.kind is move_arm_fx.FailureKind.UNREACHABLE
            assert result.failure.message.startswith("goal pose not planned")
        finally:
            feeder.cancel()


async def test_a_streaming_leader_refuses_goals_and_checks_with_leader_streaming(
    follower_never_stale,
):
    # The node refuses for no fresh follower state before it looks at the
    # leader, so follower_never_stale keeps a pause of the event loop from
    # turning the refusal into RESOURCE_UNAVAILABLE.
    position, orientation = MEASURED_POSE
    async with started() as h:
        feeder = await feeding_measured(h)
        streamer = asyncio.create_task(_stream_leader_target(h, list(MEASURED), []))
        try:
            # The leader's target reached the follower: the node follows the
            # leader, which holds the arm while it streams.
            await asyncio.wait_for(h.mocks.pairings.arm.joint_setpoints.next(), TIMEOUT_S)

            posture = await move_to_ready_fx.send_goal(
                h, move_to_ready_prod.GoalRequestData(duration_s=0.0), TIMEOUT_S
            )
            assert posture.refusal.kind is move_to_ready_fx.FailureKind.LEADER_STREAMING
            assert posture.refusal.message.startswith("a leader is streaming arm commands")

            check = check_arm_move_prod.RequestData(
                arm_name="arm",
                position=list(position),
                orientation=list(orientation),
                duration_s=0.0,
            )
            with pytest.raises(check_arm_move_fx.Failure) as raised:
                await check_arm_move_fx.poll(h, check, TIMEOUT_S)
            assert raised.value.kind is check_arm_move_fx.FailureKind.LEADER_STREAMING
        finally:
            streamer.cancel()
            feeder.cancel()


async def test_a_streaming_leader_refuses_a_joint_path_with_leader_streaming():
    async with started() as h:
        feeder = await feeding_measured(h)
        streamer = asyncio.create_task(_stream_leader_target(h, list(MEASURED), []))
        try:
            # The leader's target reached the follower: the node follows the
            # leader, which holds the arm while it streams.
            await asyncio.wait_for(h.mocks.pairings.arm.joint_setpoints.next(), TIMEOUT_S)

            path = await execute_joint_path_fx.send_goal(h, joint_path_goal(MEASURED), TIMEOUT_S)
            assert path.refusal.kind is execute_joint_path_fx.FailureKind.LEADER_STREAMING
            assert path.refusal.message.startswith("a leader is streaming arm commands")
        finally:
            streamer.cancel()
            feeder.cancel()


async def test_the_workspace_answers_on_reach_alone_from_bringup():
    # No follower reports: the answers read the design, never the arm.
    async with started() as h:
        surface = await describe_workspace_fx.poll(
            h, describe_workspace_prod.RequestData(surface_height=0.0), WORKSPACE_TIMEOUT_S
        )
        assert surface.perception_camera == ""
        assert surface.workable
        assert surface.area >= 0.03
        assert_rectangle_inside_reach(surface)
        assert surface.view is None

        wire = [*NEAR, *CLOSE, *FAR]
        points = await check_positions_fx.poll(
            h, check_positions_prod.RequestData(positions=wire), WORKSPACE_TIMEOUT_S
        )
        assert points.perception_camera == ""
        assert not points.all_workable
        near, close, far = points.results
        for result, asked in zip(points.results, (NEAR, CLOSE, FAR), strict=True):
            assert result.position == pytest.approx(list(asked))
        assert (near.workable, near.reachable, near.arm, near.short_by) == (True, True, "arm", 0.0)
        assert (close.workable, close.reachable, close.arm) == (False, False, "")
        assert 0.0 <= close.short_by <= REACH_TOLERANCE
        assert (far.workable, far.reachable, far.arm) == (False, False, "")
        assert far.short_by > FAR_SHORT_BY_MORE_THAN
        assert {(p.in_view, p.view) for p in points.results} == {(False, "no_camera")}
        # Each point's message says that its view is not checked.
        assert all("view is not checked" in p.message for p in points.results)


async def test_the_workspace_gives_an_error_reply_to_what_is_not_a_request():
    async with started() as h:
        with pytest.raises(describe_workspace_fx.Failure) as raised:
            await describe_workspace_fx.poll(
                h, describe_workspace_prod.RequestData(surface_height=math.nan), TIMEOUT_S
            )
        assert raised.value == describe_workspace_fx.Failure(
            describe_workspace_fx.FailureKind.REJECTED, "surface_height must be a finite number"
        )

        with pytest.raises(check_positions_fx.Failure) as raised:
            await check_positions_fx.poll(
                h, check_positions_prod.RequestData(positions=[0.2, 0.0]), TIMEOUT_S
            )
        assert raised.value == check_positions_fx.Failure(
            check_positions_fx.FailureKind.REJECTED,
            "positions must hold 3 values (x, y, z) per point",
        )
