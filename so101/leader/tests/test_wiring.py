"""In-process harness tests: the leader booted against mocked pairing peers
and a fake serial bus, streaming positions-only setpoints in wire units."""

import asyncio
import math

import pytest
from conftest import FakeHardware
from peppygen.paired_topics.arm import joint_setpoints as arm_wire
from peppygen.fixtures import harness
from peppygen.parameters import Parameters
from so101_description.units import GRIPPER_NAME, JOINT_NAMES

from so101_leader.__main__ import make_setup

TIMEOUT_S = 10.0

PARAMETERS = Parameters(
    device_path="/dev/so101_leader_test",
    calibration_dir="/tmp/so101_calibration",
    teleop_id="wiring_test",
    command_rate_hz=100,
    stale_timeout_s=0.25,
)


async def test_streams_positions_only_in_radians():
    fake = FakeHardware()
    fake.positions = dict.fromkeys(JOINT_NAMES, 45.0) | {GRIPPER_NAME: 25.0}
    async with harness.start(make_setup(lambda config: fake), parameters=PARAMETERS) as h:
        setpoints = await asyncio.wait_for(
            h.mocks.pairings.arm.joint_setpoints.next(), TIMEOUT_S
        )
        assert setpoints.positions == pytest.approx([math.radians(45.0)] * 5)
        assert setpoints.velocities == []
        assert setpoints.efforts == []

        gripper = await asyncio.wait_for(
            h.mocks.pairings.gripper.gripper_setpoints.next(), TIMEOUT_S
        )
        assert gripper.opening == pytest.approx(0.25)
        assert gripper.max_effort == 0.0


async def test_deadman_goes_silent_when_reads_fail():
    fake = FakeHardware()
    async with harness.start(make_setup(lambda config: fake), parameters=PARAMETERS) as h:
        # Streaming while healthy...
        await asyncio.wait_for(h.mocks.pairings.arm.joint_setpoints.next(), TIMEOUT_S)
        # ...then every read fails: the sample ages out and the wire goes
        # silent within the deadman window.
        fake.fail_reads = 10_000_000
        await asyncio.sleep(PARAMETERS.stale_timeout_s * 2)
        while True:  # drain anything published before the gate closed
            try:
                await asyncio.wait_for(h.mocks.pairings.arm.joint_setpoints.next(), 0.05)
            except TimeoutError:
                break
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(h.mocks.pairings.arm.joint_setpoints.next(), 0.5)


def test_lerobot_leader_constructs(tmp_path):
    """The real hardware seam must at least construct: this is the layer no
    fake covers, and a str/Path mismatch here once broke every launch."""
    from so101_leader.device import LerobotLeader

    leader = LerobotLeader(
        device_path="/dev/null", calibration_dir=str(tmp_path), teleop_id="smoke"
    )
    assert "gripper" in leader._teleop.bus.motors


def test_the_wire_refuses_a_joint_vector_of_the_wrong_width():
    """The manifest pins joint_link positions to five. Asserted here because
    a refine block is easy to drop from a manifest by accident, and its loss
    would be invisible: this node would go back to putting whatever width it
    was handed onto the wire."""
    with pytest.raises(ValueError, match="expected 5"):
        arm_wire.build_message(
            timestamp=0.0, positions=[0.0] * 4, velocities=[], efforts=[]
        )
