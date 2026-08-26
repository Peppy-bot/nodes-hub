"""In-process harness tests: the node booted against mocked pairing peers and
a fake serial bus, covering both wire directions, health, alerts, readiness."""

import asyncio
import math
import time

import pytest
from conftest import FakeHardware
from peppygen.fixtures import harness
from peppygen.fixtures.exposed_services.ready import is_ready as is_ready_fx
from peppygen.paired_topics.arm import joint_setpoints as arm_setpoints_topic
from peppygen.paired_topics.gripper import gripper_setpoints as gripper_setpoints_topic
from peppygen.parameters import Parameters
from so101_description.units import GRIPPER_NAME, JOINT_NAMES

from so101_follower.__main__ import make_setup

TIMEOUT_S = 10.0

PARAMETERS = Parameters(
    device_path="/dev/so101_follower_test",
    calibration_dir="/tmp/so101_calibration",
    robot_id="wiring_test",
    control_rate_hz=200,
    state_rate_hz=100,
)


async def wait_for(predicate, timeout_s=TIMEOUT_S):
    """Block until the predicate holds, or fail the test. Raising rather than
    returning false keeps a caller from waiting on a condition and silently
    accepting that it never arrived."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"condition never held within {timeout_s}s")


async def test_measured_state_flows_out_in_wire_units():
    fake = FakeHardware()
    fake.positions = dict.fromkeys(JOINT_NAMES, 30.0) | {GRIPPER_NAME: 40.0}
    async with harness.start(make_setup(lambda config: fake), parameters=PARAMETERS) as h:
        state = await asyncio.wait_for(h.mocks.pairings.arm.joint_states.next(), TIMEOUT_S)
        assert state.positions == pytest.approx([math.radians(30.0)] * 5)
        assert state.velocities == []
        assert state.efforts == []

        gripper = await asyncio.wait_for(
            h.mocks.pairings.gripper.gripper_states.next(), TIMEOUT_S
        )
        assert gripper.opening == pytest.approx(0.4)

        health = await asyncio.wait_for(
            h.emitted.motor_health_motor_health.next(), TIMEOUT_S
        )
        assert len(health.level) == 6
        assert list(health.winding_temp_c) == pytest.approx([30.0] * 6)

        response = await is_ready_fx.poll(h, TIMEOUT_S)
        assert response.ready


async def test_held_health_readings_keep_the_stamp_of_the_read_that_produced_them():
    # Republishing held readings under a fresh stamp would hide a stalling
    # bus from anyone judging reading age off the wire.
    fake = FakeHardware()
    async with harness.start(make_setup(lambda config: fake), parameters=PARAMETERS) as h:
        await asyncio.wait_for(h.emitted.motor_health_motor_health.next(), TIMEOUT_S)
        fake.fail_health_reads = 10_000
        # Drain the samples still carrying live reads from before the tap.
        for _ in range(3):
            held = await asyncio.wait_for(
                h.emitted.motor_health_motor_health.next(), TIMEOUT_S
            )
        again = await asyncio.wait_for(h.emitted.motor_health_motor_health.next(), TIMEOUT_S)
        # The stamp is reconstructed from the capture instant each publish, so
        # it jitters by well under a microsecond; a publish-time stamp would
        # instead advance by the full health period.
        assert again.timestamp == pytest.approx(held.timestamp, abs=1e-3), (
            "held readings were restamped as if freshly read"
        )


async def test_a_latched_servo_fault_reaches_the_alert_stream():
    fake = FakeHardware()
    async with harness.start(make_setup(lambda config: fake), parameters=PARAMETERS) as h:
        await asyncio.wait_for(h.emitted.motor_health_motor_health.next(), TIMEOUT_S)
        fake.faults = (0, 32, 0, 0, 0, 0)  # overload latched on shoulder_lift
        deadline = asyncio.get_event_loop().time() + TIMEOUT_S
        while asyncio.get_event_loop().time() < deadline:
            alert = await asyncio.wait_for(h.emitted.alerts_alerts.next(), TIMEOUT_S)
            if "shoulder_lift" in alert.source:
                break
        assert "shoulder_lift" in alert.source
        assert alert.severity > 0
        assert alert.kind == "motor_condition"
        assert "overload" in alert.message


async def test_setpoints_reach_the_bus_in_device_units():
    fake = FakeHardware()
    async with harness.start(make_setup(lambda config: fake), parameters=PARAMETERS) as h:
        await h.mocks.pairings.arm.joint_setpoints.publish(
            arm_setpoints_topic.Message(
                timestamp=time.time(),
                positions=[math.radians(15.0)] * 5,
                velocities=[],
                efforts=[],
            )
        )
        await h.mocks.pairings.gripper.gripper_setpoints.publish(
            gripper_setpoints_topic.Message(timestamp=time.time(), opening=0.5, max_effort=0.0)
        )
        await wait_for(
            lambda: any(GRIPPER_NAME in goals for goals in fake.written_goals)
        )
        merged = fake.written_goals[-1]
        assert merged[JOINT_NAMES[0]] == pytest.approx(15.0)
        assert merged[GRIPPER_NAME] == pytest.approx(50.0)


async def test_effort_carrying_setpoints_are_dropped_and_never_move():
    fake = FakeHardware()
    async with harness.start(make_setup(lambda config: fake), parameters=PARAMETERS) as h:
        await h.mocks.pairings.arm.joint_setpoints.publish(
            arm_setpoints_topic.Message(
                timestamp=time.time(),
                positions=[0.0] * 5,
                velocities=[],
                efforts=[1.0] * 5,
            )
        )
        # Rejection is a logged refusal, not an alert (openarm's policy);
        # the proof is that nothing ever reaches the bus ...
        await asyncio.sleep(0.3)
        assert fake.written_goals == []
        await h.mocks.pairings.arm.joint_setpoints.publish(
            arm_setpoints_topic.Message(
                timestamp=time.time(), positions=[0.2] * 5, velocities=[], efforts=[]
            )
        )
        # ... while a conforming follow-up still does.
        await wait_for(lambda: len(fake.written_goals) > 0)


async def test_stale_timestamped_setpoints_never_reach_the_bus():
    fake = FakeHardware()
    async with harness.start(make_setup(lambda config: fake), parameters=PARAMETERS) as h:
        # Stamped far in the past: legal shape, but the age gate must drop it.
        await h.mocks.pairings.arm.joint_setpoints.publish(
            arm_setpoints_topic.Message(
                timestamp=time.time() - 60.0, positions=[0.1] * 5, velocities=[], efforts=[]
            )
        )
        await asyncio.sleep(0.3)
        assert fake.written_goals == []


def test_lerobot_follower_constructs(tmp_path):
    """The real hardware seam must at least construct: this is the layer no
    fake covers, and a str/Path mismatch here once broke every launch."""
    from so101_follower.device import LerobotFollower

    follower = LerobotFollower(
        device_path="/dev/null",
        calibration_dir=str(tmp_path),
        robot_id="smoke",
    )
    assert set(follower._robot.bus.motors) == {
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"
    }


async def test_dead_bus_revokes_readiness():
    fake = FakeHardware()
    async with harness.start(
        make_setup(lambda _config: fake), parameters=PARAMETERS
    ) as h:
        response = await is_ready_fx.poll(h, TIMEOUT_S)
        assert response.ready
        # The bus dies mid-session: state reads stop landing. Readiness is
        # live evidence, not a bringup latch, so it must flip false.
        fake.fail_position_reads = 10_000_000
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            response = await is_ready_fx.poll(h, TIMEOUT_S)
            if not response.ready:
                break
            await asyncio.sleep(0.2)
        assert not response.ready
