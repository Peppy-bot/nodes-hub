"""Boots the node in-process under the generated test harness, with the one
parameter its manifest requires, and checks that it comes up and shuts down
cleanly."""

from peppygen.fixtures import harness
from peppygen.parameters import Parameters

from openarm_ai_brain.__main__ import setup


async def test_node_boots_and_shuts_down_cleanly():
    params = Parameters.from_dict({"gripper_names": "left_gripper,right_gripper"})
    async with harness.start(setup, parameters=params) as h:
        assert h.instance_id
