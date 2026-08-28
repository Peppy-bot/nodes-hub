"""In-process harness tests: the node booted against mocked pairing peers,
covering the teleop path, the relays, and the action surface end to end."""

import asyncio
import math
import time

import peppylib
import pytest
from peppygen.exposed_actions.limb_motion import move_arm as move_arm_prod
from peppygen.exposed_actions.limb_motion import move_arm_joints as move_arm_joints_prod
from peppygen.exposed_actions.limb_motion import move_gripper as move_gripper_prod
from peppygen.exposed_actions.postures import move_to_ready as move_to_ready_prod
from peppygen.fixtures import harness
from peppygen.fixtures.exposed_actions.limb_motion import move_arm as move_arm_fx
from peppygen.fixtures.exposed_actions.limb_motion import (
    move_arm_joints as move_arm_joints_fx,
)
from peppygen.fixtures.exposed_actions.limb_motion import (
    move_gripper as move_gripper_fx,
)
from peppygen.fixtures.exposed_actions.postures import move_to_ready as move_to_ready_fx
from peppygen.paired_topics.arm_link import joint_states as arm_states_topic
from peppygen.paired_topics.leader_arm import joint_states as upstream_states_topic
from peppygen.paired_topics.gripper_link import gripper_states as gripper_states_topic
from peppygen.paired_topics.leader_arm import joint_setpoints as leader_setpoints_topic
from peppygen.paired_topics.leader_pose import pose_setpoints as leader_pose_topic
from peppygen.parameters import Parameters

from so101_backbone.__main__ import setup

MEASURED = [0.0, 0.1, 0.2, 0.3, 0.4]
TIMEOUT_S = 10.0


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
    await h.mocks.pairings.arm_link.joint_states.publish(
        arm_states_topic.Message(
            timestamp=time.time(), positions=positions, velocities=[], efforts=[]
        )
    )
    await h.mocks.pairings.gripper_link.gripper_states.publish(
        gripper_states_topic.Message(
            timestamp=time.time(), opening=opening, effort=0.0, max_effort=0.0
        )
    )


async def keep_measured_fresh(h, positions=MEASURED, opening=0.0):
    while True:
        await publish_measured(h, positions, opening)
        await asyncio.sleep(0.01)


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
                    h.mocks.pairings.arm_link.joint_setpoints.next(), TIMEOUT_S
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


async def test_malformed_follower_state_neither_anchors_nor_relays():
    # Measured state is a trust boundary: a NaN vector must not become a
    # governor anchor, and must not be relayed upstream as if the follower
    # had said something coherent. Wrong-length vectors are the contract's
    # job now and cannot even be built; see the cardinality test below.
    async with harness.start(setup, parameters=make_parameters()) as h:
        for positions in ([float("nan")] * 5,):
            await h.mocks.pairings.arm_link.joint_states.publish(
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
        await h.mocks.pairings.arm_link.joint_states.publish(
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
                h.mocks.pairings.arm_link.joint_setpoints.next(), TIMEOUT_S
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
        feeder = asyncio.create_task(keep_measured_fresh(h))
        try:
            target = [p + 0.2 for p in MEASURED]
            goal = await move_arm_joints_fx.send_goal(
                h,
                move_arm_joints_prod.GoalRequestData(
                    arm_name="arm", joint_positions=target, duration_s=0.0
                ),
                peppylib.QoSProfile.Reliable,
                TIMEOUT_S,
            )
            assert goal.accepted
            result = await goal.get_result(TIMEOUT_S)
            assert result.status == move_arm_joints_fx.ResultStatus.COMPLETED
            assert result.data.success
            # The trajectory's last downstream setpoint lands on the target.
            last = None
            while True:
                try:
                    last = await asyncio.wait_for(
                        h.mocks.pairings.arm_link.joint_setpoints.next(), 0.5
                    )
                except TimeoutError:
                    break
            assert last is not None
            assert last.positions == pytest.approx(target)
        finally:
            feeder.cancel()


async def test_second_goal_is_rejected_while_busy():
    async with harness.start(setup, parameters=make_parameters()) as h:
        await publish_measured(h)
        # The in-flight goal outlives STALE_FOLLOWER_TIMEOUT_S, and a plan
        # failed for staleness would cancel before cancel_goal ever runs.
        feeder = asyncio.create_task(keep_measured_fresh(h))
        try:
            slow = await move_arm_joints_fx.send_goal(
                h,
                move_arm_joints_prod.GoalRequestData(
                    arm_name="arm",
                    joint_positions=[p + 0.5 for p in MEASURED],
                    duration_s=5.0,
                ),
                peppylib.QoSProfile.Reliable,
                TIMEOUT_S,
            )
            assert slow.accepted
            second = await move_arm_joints_fx.send_goal(
                h,
                move_arm_joints_prod.GoalRequestData(
                    arm_name="arm", joint_positions=MEASURED, duration_s=0.0
                ),
                peppylib.QoSProfile.Reliable,
                TIMEOUT_S,
            )
            assert not second.accepted
            assert "in flight" in second.reason
            cancel = await slow.cancel_goal(TIMEOUT_S)
            assert cancel.state == move_arm_joints_fx.CancelState.SIGNALLED
            result = await slow.get_result(TIMEOUT_S)
            assert result.status == move_arm_joints_fx.ResultStatus.CANCELLED
            assert not result.data.success
        finally:
            feeder.cancel()


async def test_goal_without_follower_state_is_rejected():
    async with harness.start(setup, parameters=make_parameters()) as h:
        goal = await move_arm_joints_fx.send_goal(
            h,
            move_arm_joints_prod.GoalRequestData(
                arm_name="arm", joint_positions=MEASURED, duration_s=0.0
            ),
            peppylib.QoSProfile.Reliable,
            TIMEOUT_S,
        )
        assert not goal.accepted
        assert "no fresh follower state" in goal.reason


async def test_posture_and_gripper_actions():
    async with harness.start(setup, parameters=make_parameters()) as h:
        feeder = asyncio.create_task(keep_measured_fresh(h))
        try:
            posture = await move_to_ready_fx.send_goal(
                h,
                move_to_ready_prod.GoalRequestData(duration_s=0.0),
                peppylib.QoSProfile.Reliable,
                TIMEOUT_S,
            )
            assert posture.accepted
            result = await posture.get_result(TIMEOUT_S)
            assert result.status == move_to_ready_fx.ResultStatus.COMPLETED
            assert result.data.success

            grip = await move_gripper_fx.send_goal(
                h,
                move_gripper_prod.GoalRequestData(
                    gripper_name="gripper", opening=0.5, max_effort=0.0
                ),
                peppylib.QoSProfile.Reliable,
                TIMEOUT_S,
            )
            assert grip.accepted
            grip_result = await grip.get_result(TIMEOUT_S)
            assert grip_result.status == move_gripper_fx.ResultStatus.COMPLETED
            assert grip_result.data.success
            # The ramp's last downstream opening is the target.
            last = None
            while True:
                try:
                    last = await asyncio.wait_for(
                        h.mocks.pairings.gripper_link.gripper_setpoints.next(), 0.5
                    )
                except TimeoutError:
                    break
            assert last is not None
            assert math.isclose(last.opening, 0.5)
        finally:
            feeder.cancel()


async def test_nonzero_max_effort_is_accepted_and_ignored():
    # Contract semantics: 0 = no preference, >0 = a preference the
    # implementer may ignore; both the stream and the action run the
    # opening, so generic limb_motion clients work unmodified.
    async with harness.start(setup, parameters=make_parameters()) as h:
        feeder = asyncio.create_task(keep_measured_fresh(h))
        try:
            goal = await move_gripper_fx.send_goal(
                h,
                move_gripper_prod.GoalRequestData(
                    gripper_name="gripper", opening=0.5, max_effort=2.0
                ),
                peppylib.QoSProfile.Reliable,
                TIMEOUT_S,
            )
            assert goal.accepted
            result = await goal.get_result(TIMEOUT_S)
            assert result.data.success
        finally:
            feeder.cancel()


async def test_malformed_pose_goal_is_rejected_and_the_server_survives():
    async with harness.start(setup, parameters=make_parameters()) as h:
        feeder = asyncio.create_task(keep_measured_fresh(h))
        try:
            # A zero quaternion once killed the move_arm server permanently.
            bad = await move_arm_fx.send_goal(
                h,
                move_arm_prod.GoalRequestData(
                    arm_name="arm",
                    position=[0.1, 0.0, 0.2],
                    orientation=[0.0, 0.0, 0.0, 0.0],
                    duration_s=0.0,
                    plan_position_tolerance_m=0.0,
                    plan_orientation_tolerance_rad=0.0,
                ),
                peppylib.QoSProfile.Reliable,
                TIMEOUT_S,
            )
            assert not bad.accepted
            assert "unit quaternion" in bad.reason
            # The server must still decide the next goal (reachability of the
            # target pose is not the point; a decision is).
            again = await move_arm_fx.send_goal(
                h,
                move_arm_prod.GoalRequestData(
                    arm_name="arm",
                    position=[0.0, 0.0, 0.42],
                    orientation=[0.0, 0.0, 0.0, 1.0],
                    duration_s=0.0,
                    plan_position_tolerance_m=0.0,
                    plan_orientation_tolerance_rad=0.0,
                ),
                peppylib.QoSProfile.Reliable,
                TIMEOUT_S,
            )
            if not again.accepted:
                assert "unit quaternion" not in again.reason
        finally:
            feeder.cancel()


async def test_out_of_limit_goal_is_rejected():
    async with harness.start(setup, parameters=make_parameters()) as h:
        await publish_measured(h)
        goal = await move_arm_joints_fx.send_goal(
            h,
            move_arm_joints_prod.GoalRequestData(
                arm_name="arm", joint_positions=[9.0] * 5, duration_s=0.0
            ),
            peppylib.QoSProfile.Reliable,
            TIMEOUT_S,
        )
        assert not goal.accepted
        assert "joint position limit" in goal.reason


async def test_unknown_arm_name_is_refused():
    async with harness.start(setup, parameters=make_parameters()) as h:
        await publish_measured(h)
        goal = await move_arm_joints_fx.send_goal(
            h,
            move_arm_joints_prod.GoalRequestData(
                arm_name="left_arm", joint_positions=MEASURED, duration_s=0.0
            ),
            peppylib.QoSProfile.Reliable,
            TIMEOUT_S,
        )
        assert not goal.accepted
        assert "unknown arm_name" in goal.reason


async def test_stale_timestamped_leader_input_is_dropped():
    async with harness.start(setup, parameters=make_parameters()) as h:
        feeder = asyncio.create_task(keep_measured_fresh(h))
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
                    h.mocks.pairings.arm_link.joint_setpoints.next(), 0.4
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
