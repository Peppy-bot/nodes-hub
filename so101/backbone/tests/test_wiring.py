"""In-process harness tests: the node booted against mocked pairing peers,
covering the teleop path, the relays, the action surface and the workspace
services end to end."""

import asyncio
import math
import re
import time

import peppylib
import pytest
from conftest import (
    CLOSE,
    FAR,
    FAR_SHORT_BY_MORE_THAN,
    MEASURED,
    NEAR,
    assert_rectangle_inside_reach,
)
from peppygen.exposed_actions.limb_motion import move_arm as move_arm_prod
from peppygen.exposed_actions.limb_motion import move_arm_joints as move_arm_joints_prod
from peppygen.exposed_actions.limb_motion import move_gripper as move_gripper_prod
from peppygen.exposed_actions.postures import move_to_home as move_to_home_prod
from peppygen.exposed_actions.postures import move_to_ready as move_to_ready_prod
from peppygen.fixtures import harness
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
from peppygen.paired_topics.arm import joint_states as arm_states_topic
from peppygen.paired_topics.leader_arm import joint_states as upstream_states_topic
from peppygen.paired_topics.gripper import gripper_states as gripper_states_topic
from peppygen.paired_topics.leader_arm import joint_setpoints as leader_setpoints_topic
from peppygen.paired_topics.leader_pose import pose_setpoints as leader_pose_topic
from peppygen.parameters import Parameters
from so101_description import postures
from so101_description.kinematics import Kinematics
from so101_description.model import KINEMATICS_URDF_PATH
from workspace_core_py import REACH_TOLERANCE

from so101_backbone.__main__ import setup
from so101_backbone.coordinator import GRIPPER_STILL_WINDOW_S

TIMEOUT_S = 10.0
# The bound of an answer that measures reach: the first starts the worker
# process and measures a whole surface. A hang guard, not a measure of speed.
WORKSPACE_TIMEOUT_S = 60.0


def make_parameters(**overrides) -> Parameters:
    base = {
        "upstream_mode": "joints",
        "control_rate_hz": 100,
        "max_joint_velocity_rad_s_1": 8.0,
        "max_joint_velocity_rad_s_2": 8.0,
        "max_joint_velocity_rad_s_3": 8.0,
        "max_joint_velocity_rad_s_4": 8.0,
        "max_joint_velocity_rad_s_5": 8.0,
        "max_ee_velocity_m_s": 1e9,
        "max_ee_angular_velocity_rad_s": 1e9,
        "max_gripper_rate_frac_s": 8.0,
    }
    return Parameters(**{**base, **overrides})


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
    async with harness.start(setup, parameters=make_parameters()) as h:
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
                # Pass-through: the leader's exact values go downstream, no
                # clamp and no chase (the EE caps are transparent here).
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


async def test_the_limb_names_are_answered_before_any_snapshot():
    # No follower reports here, so no readout has been published: the names
    # and the joint count behind them come from the robot itself.
    async with harness.start(setup, parameters=make_parameters()) as h:
        names = await get_limb_names_fx.poll(h, TIMEOUT_S)
        assert names.arm_names == ["arm"]
        assert names.joints_per_arm == [5]
        assert names.gripper_names == ["gripper"]


async def test_malformed_follower_state_neither_anchors_nor_relays():
    # Measured state is a trust boundary: a NaN vector must not become a
    # governor anchor, and must not be relayed upstream as if the follower
    # had said something coherent. Wrong-length vectors are the contract's
    # job now and cannot even be built; see the cardinality test below.
    async with harness.start(setup, parameters=make_parameters()) as h:
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
    async with harness.start(setup, parameters=make_parameters()) as h:
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
    async with harness.start(
        setup,
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
    async with harness.start(setup, parameters=make_parameters()) as h:
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


async def test_move_arm_joints_outside_its_tolerances_fails_with_timeout():
    async with harness.start(setup, parameters=make_parameters()) as h:
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


async def test_second_goal_is_refused_while_busy_and_a_cancel_ends_the_first():
    async with harness.start(setup, parameters=make_parameters()) as h:
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


async def test_goal_without_follower_state_is_refused():
    async with harness.start(setup, parameters=make_parameters()) as h:
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


async def test_posture_and_gripper_actions():
    async with harness.start(setup, parameters=make_parameters()) as h:
        feeder = await feeding_measured(h)
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
    # One scenario for three facts of the result of each posture move:
    # - the result names the arm;
    # - it gives the grasp point of the measured joints;
    # - it completes when the arm did not arrive.
    # The mocked follower always measures MEASURED, so the arm never gets to
    # a posture. follower_never_stale keeps a pause of the event loop from
    # making the measured joints stale.
    kinematics = Kinematics(KINEMATICS_URDF_PATH)
    measured_position, measured_orientation = kinematics.forward_kinematics(MEASURED)
    async with harness.start(setup, parameters=make_parameters()) as h:
        feeder = await feeding_measured(h)
        try:
            for production, fixture, posture in (
                (move_to_ready_prod, move_to_ready_fx, postures.READY_POSITIONS_RAD),
                (move_to_home_prod, move_to_home_fx, postures.HOME_POSITIONS_RAD),
            ):
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
                assert math.dist(result.data.positions, posture_position) > 0.1
        finally:
            feeder.cancel()


async def test_nonzero_max_effort_is_accepted_and_ignored():
    # Contract semantics: 0 = no preference, >0 = a preference the
    # implementer may ignore; both the stream and the action run the
    # opening, so generic limb_motion clients work unmodified.
    async with harness.start(setup, parameters=make_parameters()) as h:
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
    async with harness.start(setup, parameters=make_parameters()) as h:
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
    async with harness.start(setup, parameters=make_parameters()) as h:
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
                    plan_position_tolerance_m=0.01,
                    plan_orientation_tolerance_rad=0.5,
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
                    plan_position_tolerance_m=0.01,
                    plan_orientation_tolerance_rad=0.5,
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
    async with harness.start(setup, parameters=make_parameters()) as h:
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
            move_arm_joints_fx.FailureKind.REJECTED, "target out of joint limits"
        )


async def test_unknown_arm_name_is_refused():
    async with harness.start(setup, parameters=make_parameters()) as h:
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


async def test_stale_timestamped_leader_input_is_dropped():
    async with harness.start(setup, parameters=make_parameters()) as h:
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


async def test_the_wire_refuses_a_joint_vector_of_the_wrong_width():
    """The manifest pins joint_link positions to five, so a wrong-width
    vector cannot be put on the wire at all. Asserted here because a refine
    block is easy to drop from a manifest by accident, and its loss would be
    invisible: the node would simply go back to trusting whatever arrived."""
    with pytest.raises(ValueError, match="expected 5"):
        upstream_states_topic.build_message(
            timestamp=time.time(), positions=[0.0] * 4, velocities=[], efforts=[]
        )


async def test_a_stop_ends_the_moves_in_flight_and_a_later_goal_runs():
    # A gripper rate of 0.1 per second gives the full-travel gripper move
    # below a planned end 10 s out, so the stop finds it in flight.
    async with harness.start(
        setup, parameters=make_parameters(max_gripper_rate_frac_s=0.1)
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


async def test_the_plan_check_answers_without_moving():
    async with harness.start(setup, parameters=make_parameters()) as h:
        feeder = await feeding_measured(h)
        try:
            from so101_description.kinematics import Kinematics
            from so101_description.model import KINEMATICS_URDF_PATH

            position, orientation = Kinematics(KINEMATICS_URDF_PATH).forward_kinematics(
                MEASURED
            )
            check = check_arm_move_prod.RequestData(
                arm_name="arm",
                position=[position[0], position[1], position[2] + 0.02],
                orientation=list(orientation),
                duration_s=0.0,
                plan_position_tolerance_m=0.01,
                plan_orientation_tolerance_rad=0.5,
            )
            answer = await check_arm_move_fx.poll(h, check, TIMEOUT_S)
            assert (answer.reachable, answer.reason) == (True, "")
            assert answer.duration_s > 0.0

            far = check_arm_move_prod.RequestData(
                arm_name="arm",
                position=[position[0] + 2.0, position[1], position[2]],
                orientation=list(orientation),
                duration_s=0.0,
                plan_position_tolerance_m=0.01,
                plan_orientation_tolerance_rad=0.5,
            )
            answer = await check_arm_move_fx.poll(h, far, TIMEOUT_S)
            assert not answer.reachable
            assert answer.reason.startswith("goal pose not planned within ")
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
                plan_position_tolerance_m=0.01,
                plan_orientation_tolerance_rad=0.5,
            )
            with pytest.raises(check_arm_move_fx.Failure) as raised:
                await check_arm_move_fx.poll(h, unknown, TIMEOUT_S)
            assert raised.value.kind is check_arm_move_fx.FailureKind.REJECTED
            assert "unknown arm_name" in raised.value.message
        finally:
            feeder.cancel()


async def test_a_move_arm_goal_with_no_plan_fails_with_unreachable():
    async with harness.start(setup, parameters=make_parameters()) as h:
        feeder = await feeding_measured(h)
        try:
            position, orientation = Kinematics(KINEMATICS_URDF_PATH).forward_kinematics(
                MEASURED
            )
            goal = await move_arm_fx.send_goal(
                h,
                move_arm_prod.GoalRequestData(
                    arm_name="arm",
                    position=[position[0] + 2.0, position[1], position[2]],
                    orientation=list(orientation),
                    duration_s=0.0,
                    plan_position_tolerance_m=0.01,
                    plan_orientation_tolerance_rad=0.5,
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
            assert result.failure.message.startswith("goal pose not planned within ")
        finally:
            feeder.cancel()


async def test_a_streaming_leader_refuses_goals_and_checks_with_leader_streaming():
    async with harness.start(setup, parameters=make_parameters()) as h:
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

            position, orientation = Kinematics(KINEMATICS_URDF_PATH).forward_kinematics(
                MEASURED
            )
            check = check_arm_move_prod.RequestData(
                arm_name="arm",
                position=list(position),
                orientation=list(orientation),
                duration_s=0.0,
                plan_position_tolerance_m=0.01,
                plan_orientation_tolerance_rad=0.5,
            )
            with pytest.raises(check_arm_move_fx.Failure) as raised:
                await check_arm_move_fx.poll(h, check, TIMEOUT_S)
            assert raised.value.kind is check_arm_move_fx.FailureKind.LEADER_STREAMING
        finally:
            streamer.cancel()
            feeder.cancel()


async def test_the_workspace_answers_on_reach_alone_from_bringup():
    # No follower reports: the answers read the design, never the arm.
    async with harness.start(setup, parameters=make_parameters()) as h:
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
    async with harness.start(setup, parameters=make_parameters()) as h:
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
