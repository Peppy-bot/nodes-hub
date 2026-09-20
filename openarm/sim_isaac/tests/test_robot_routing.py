"""Which robot a limb pair belongs to, and what that decides.

Every robot's limbs reach the engine on their own pairs, and the copy a pair
carries is the robot that attached under that name. These cover the two ways
that is read: a setpoint arriving on a pair, and a robot asking whether it is
ready to be driven.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robots" / "openarm"))

_LIMB_SLOTS = ["left_arm", "right_arm", "left_gripper", "right_gripper"]
_CAMERA_SLOTS = ["wrist_left", "wrist_right", "chest"]
_TOPICS = [
    "joint_setpoints",
    "joint_states",
    "gripper_setpoints",
    "gripper_states",
    "video_stream",
    "depth_stream",
    "stream_info",
    "geometry",
]


class _Peer:
    """A peer as peppylib reports it: the wire address of one end of a pair."""

    def __init__(self, instance: str) -> None:
        self.instance = instance

    def __eq__(self, other) -> bool:
        return isinstance(other, _Peer) and other.instance == self.instance

    def __hash__(self) -> int:
        return hash(self.instance)


class _Member:
    """One pair of a slot: the peer it reaches and the copy that peer runs
    as, which is the robot it belongs to."""

    def __init__(self, instance: str, copy) -> None:
        self.info = _Peer(instance)
        self.copy = copy


def _install_runtime_fakes() -> None:
    """The smallest module tree sim_topics imports. Each slot module records
    the pairs the test gave it."""
    peppylib = types.ModuleType("peppylib")
    peppylib.NodeRunner = object
    peppylib.TopicPublisher = object
    peppylib.PeerPublisher = object
    peppylib_clock = types.ModuleType("peppylib.clock")
    peppylib_clock.ClockPublisher = type("ClockPublisher", (), {})
    peppylib.clock = peppylib_clock

    peppygen = types.ModuleType("peppygen")
    peppygen.__path__ = []
    peppygen_clock = types.ModuleType("peppygen.clock")
    peppygen_clock.now_ns = lambda: 0
    peppygen.clock = peppygen_clock
    paired = types.ModuleType("peppygen.paired_topics")
    emitted = types.ModuleType("peppygen.emitted_topics")
    emitted_objects = types.ModuleType("peppygen.emitted_topics.objects")
    emitted_objects.object_states = types.ModuleType(
        "peppygen.emitted_topics.objects.object_states"
    )
    # The scene's own object stream, which the clock tests drive; this module
    # never publishes one, and only needs the name its guard is reported by.
    emitted_objects.object_states.TOPIC_NAME = "object_states"
    emitted.objects = emitted_objects
    peppygen.emitted_topics = emitted
    modules = {
        "peppylib": peppylib,
        "peppylib.clock": peppylib_clock,
        "peppygen": peppygen,
        "peppygen.clock": peppygen_clock,
        "peppygen.paired_topics": paired,
        "peppygen.emitted_topics": emitted,
        "peppygen.emitted_topics.objects": emitted_objects,
    }
    for slot in (*_LIMB_SLOTS, *_CAMERA_SLOTS):
        slot_module = types.ModuleType(f"peppygen.paired_topics.{slot}")
        for topic in _TOPICS:
            topic_module = types.ModuleType(f"peppygen.paired_topics.{slot}.{topic}")
            topic_module.LINK_ID = slot
            topic_module.members = []
            topic_module.peers = (
                lambda _runner, module=topic_module: list(module.members)
            )
            setattr(slot_module, topic, topic_module)
        setattr(paired, slot, slot_module)
        modules[f"peppygen.paired_topics.{slot}"] = slot_module
    sys.modules.update(modules)


_install_runtime_fakes()

import sim_topics  # noqa: E402  (needs the fakes above)
from robots import Caller, Limbs, Registry  # noqa: E402  pylint: disable=C0413

LIMBS = Limbs(arm_names=("left", "right"), arm_joints=(7, 7), gripper_names=("left", "right"))


def _slot_modules() -> dict:
    """Every generated module of each limb and camera slot, read from the
    tables sim_topics itself holds, so the fakes patched here are the ones it
    calls whatever else a test session imported first."""
    by_slot: dict[str, list] = {}
    tables = (
        sim_topics._ARM_SLOTS,
        sim_topics._GRIPPER_SLOTS,
        sim_topics._COLOR_CAMERA_SLOTS,
        sim_topics._RGBD_CAMERA_SLOTS,
    )
    for table in tables:
        for modules in table.values():
            for module in modules:
                by_slot.setdefault(module.LINK_ID, []).append(module)
    return by_slot


def _pair(slot: str, *members: _Member) -> None:
    """Gives one slot the pairs it holds. A slot's pairs are the slot's, so
    every module of it reads the same set, exactly as the generated
    `peers()` does."""
    for module in _slot_modules()[slot]:
        module.members = list(members)


def _clear_pairs() -> None:
    """Empties every slot and (re)attaches the accessor that reads it."""
    for modules in _slot_modules().values():
        for module in modules:
            module.members = []
            module.peers = lambda _runner, module=module: list(module.members)


@pytest.fixture(name="io")
def io_fixture():
    _clear_pairs()
    loop = asyncio.new_event_loop()
    robots = Registry()
    io = sim_topics.SimTopicIO(node_runner=object(), loop=loop, robots=robots)
    yield io, robots
    loop.close()


def _setpoints_of(slot: str):
    """The module a setpoint arrives on for this slot."""
    return _slot_modules()[slot][0]


class TestWhichRobotAPairBelongsTo:
    def test_a_pair_belongs_to_the_copy_it_carries(self, io):
        bridge, robots = io
        robots.admit("alpha", "openarm_v2", Caller("cn", "alpha_init"), 0.0)
        robots.admit("bravo", "openarm_v2", Caller("cn", "bravo_init"), 0.0)
        _pair("left_arm", _Member("alpha_backbone", "alpha"),
            _Member("bravo_backbone", "bravo"),
        )

        module = _setpoints_of("left_arm")
        assert bridge._robot_of(module, _Peer("bravo_backbone")) == "bravo"
        assert bridge._peer_of(module, "alpha") == _Peer("alpha_backbone")

    def test_a_pair_with_no_copy_belongs_to_the_only_robot_in_the_scene(self, io):
        bridge, robots = io
        robots.admit("solo", "openarm_v2", Caller("cn", "solo_init"), 0.0)
        _pair("left_arm", _Member("backbone_inst", None))

        module = _setpoints_of("left_arm")
        assert bridge._robot_of(module, _Peer("backbone_inst")) == "solo"

    def test_a_pair_with_no_copy_names_no_robot_in_a_fleet(self, io):
        bridge, robots = io
        robots.admit("alpha", "openarm_v2", Caller("cn", "alpha_init"), 0.0)
        robots.admit("bravo", "openarm_v2", Caller("cn", "bravo_init"), 0.0)
        _pair("left_arm", _Member("backbone_inst", None))

        module = _setpoints_of("left_arm")
        assert bridge._robot_of(module, _Peer("backbone_inst")) is None

    def test_a_peer_the_slot_does_not_hold_belongs_to_no_robot(self, io):
        bridge, _ = io
        module = _setpoints_of("left_arm")
        assert bridge._robot_of(module, _Peer("stranger")) is None
        assert bridge._peer_of(module, "alpha") is None


class TestReadiness:
    def test_a_robot_is_driveable_only_with_every_limb_paired(self, io):
        bridge, _ = io
        for slot in _LIMB_SLOTS[:3]:
            _pair(slot, _Member(f"alpha_{slot}", "alpha"))

        assert bridge.robots_with_any_limb() == {"alpha"}
        assert bridge.robots_with_every_limb() == set()

        _pair("right_gripper", _Member("alpha_right_gripper", "alpha"))
        assert bridge.robots_with_every_limb() == {"alpha"}

    def test_each_robot_is_judged_on_its_own_pairs(self, io):
        bridge, _ = io
        for slot in _LIMB_SLOTS:
            members = [_Member(f"alpha_{slot}", "alpha")]
            if slot != "right_arm":
                members.append(_Member(f"bravo_{slot}", "bravo"))
            _pair(slot, *members)

        assert bridge.robots_with_every_limb() == {"alpha"}
        assert bridge.robots_with_any_limb() == {"alpha", "bravo"}

    def test_a_camera_is_rendered_only_for_the_robot_that_pairs_one(self, io):
        bridge, _ = io
        _pair("wrist_left", _Member("alpha_wrist_left", "alpha"))

        assert bridge.camera_robots() == {"alpha"}


class TestSetpoints:
    def test_a_setpoint_is_kept_per_limb_of_each_robot(self, io):
        bridge, _ = io
        bridge._command_slot(bridge._arm_cmd, "left", "alpha").set(([0.5], []))
        bridge._command_slot(bridge._arm_cmd, "left", "bravo").set(([0.9], []))

        assert bridge.latest_arm_command("alpha", "left") == ([0.5], [])
        assert bridge.latest_arm_command("bravo", "left") == ([0.9], [])
        assert bridge.latest_arm_command("alpha", "right") is None

    def test_a_robot_that_left_takes_its_setpoints_with_it(self, io):
        bridge, _ = io
        bridge._command_slot(bridge._arm_cmd, "left", "alpha").set(([0.5], []))
        bridge._command_slot(bridge._gripper_cmd, "left", "alpha").set((0.25, 0.0))
        bridge._command_slot(bridge._arm_cmd, "left", "bravo").set(([0.9], []))

        bridge.forget("alpha")

        assert bridge.latest_arm_command("alpha", "left") is None
        assert bridge.latest_gripper_command("alpha", "left") is None
        assert bridge.latest_arm_command("bravo", "left") == ([0.9], [])
