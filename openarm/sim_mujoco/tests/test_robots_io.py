"""Who may join this engine's scene: one robot at a time, of a model the
engine carries, standing where the scene puts it, and staying for as long as
it holds a limb pair."""

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robots" / "openarm"))


class _Decision:
    """What `_admit` answers: the decision, and what it carried."""

    def __init__(self, accepted: bool, payload) -> None:
        self.accepted = accepted
        self.payload = payload


class _GoalDecision:
    @staticmethod
    def accept(response) -> _Decision:
        return _Decision(True, response)

    @staticmethod
    def reject(reason: str) -> _Decision:
        return _Decision(False, reason)


class _GoalResponse:
    def __init__(self, arm_names, arm_joints, gripper_names) -> None:
        self.arm_names = arm_names
        self.arm_joints = arm_joints
        self.gripper_names = gripper_names


# The typed transport is not under test; the module only needs its names.
for _name in (
    "peppygen",
    "peppygen.exposed_actions",
    "peppygen.exposed_actions.robots",
    "peppygen.exposed_services",
    "peppygen.exposed_services.robots",
):
    sys.modules.setdefault(_name, ModuleType(_name))
_attach = ModuleType("attach")
_attach.GoalDecision = _GoalDecision
_attach.GoalResponse = _GoalResponse
sys.modules["peppygen.exposed_actions.robots"].attach = _attach
sys.modules["peppygen.exposed_services.robots"].is_ready = ModuleType("is_ready")

from head_camera import Pack  # noqa: E402  pylint: disable=C0413
from robots import Caller, Limbs, Registry  # noqa: E402  pylint: disable=C0413
from robots_io import Ending, RobotsIO  # noqa: E402  pylint: disable=C0413
from scenes import Catalogue  # noqa: E402  pylint: disable=C0413

LIMBS = Limbs(arm_names=("left", "right"), arm_joints=(7, 7), gripper_names=("left", "right"))
# The head camera pack the catalogue carries; standing never reads it here.
HEAD_CAMERA_PACK = Pack(directory=Path("/staged/head_camera"), body_position=(0.0315, 0.0, 0.743))


def _robots_io(tmp_path: Path, renders: bool = False) -> RobotsIO:
    """A RobotsIO with only the parts admitting a robot touches, over a
    catalogue of two scenes that exist."""
    scenes = {}
    for model in ("openarm_v1", "openarm_v2"):
        path = tmp_path / f"{model}.xml"
        path.write_text("<mujoco/>")
        scenes[model] = path
    io = RobotsIO.__new__(RobotsIO)
    io._catalogue = Catalogue(scenes, head_camera_pack=HEAD_CAMERA_PACK)
    io._robots = Registry()
    io._limbs = LIMBS
    io._renders = renders
    io._loop = SimpleNamespace(time=lambda: 0.0)
    io._handovers = {}
    return io


def _request(robot: str = "alpha", model: str = "openarm_v2", placement=None, instance: str = "alpha_init_inst"):
    return SimpleNamespace(
        core_node="sim16",
        instance_id=instance,
        data=SimpleNamespace(robot=robot, model=model, placement=placement),
    )


def test_a_robot_of_a_carried_model_is_admitted_with_its_limbs(tmp_path):
    io = _robots_io(tmp_path)
    decision = io._admit(_request())
    assert decision.accepted
    assert (decision.payload.arm_names, decision.payload.arm_joints, decision.payload.gripper_names) == (
        ["left", "right"],
        [7, 7],
        ["left", "right"],
    )
    assert io._robots.of_name("alpha").model == "openarm_v2"


def test_a_model_the_engine_does_not_carry_is_refused_with_what_it_does(tmp_path):
    io = _robots_io(tmp_path)
    decision = io._admit(_request(model="openarm_v9"))
    assert not decision.accepted
    assert "unknown model 'openarm_v9'" in decision.payload
    assert "openarm_v1, openarm_v2" in decision.payload


def test_a_placement_is_refused_because_the_scene_places_the_robot(tmp_path):
    io = _robots_io(tmp_path)
    decision = io._admit(_request(placement=SimpleNamespace(position=[1.0, 0.0, 0.0], yaw=0.0)))
    assert not decision.accepted
    assert "placement { auto: true }" in decision.payload
    assert io._robots.robots() == []


def test_a_second_robot_is_refused_while_one_stands(tmp_path):
    io = _robots_io(tmp_path)
    assert io._admit(_request()).accepted
    decision = io._admit(_request(robot="bravo", instance="bravo_init_inst"))
    assert not decision.accepted
    assert "stands one robot" in decision.payload
    assert "'alpha' of alpha_init_inst@sim16 stands already" in decision.payload


def test_the_name_is_free_again_once_the_robot_left(tmp_path):
    io = _robots_io(tmp_path)
    assert io._admit(_request()).accepted
    io._robots.release("alpha")
    assert io._admit(_request(robot="bravo", instance="bravo_init_inst")).accepted


def test_a_rendering_engine_stands_the_model_its_rig_mounts_on(tmp_path):
    io = _robots_io(tmp_path, renders=True)
    decision = io._admit(_request(model="openarm_v1"))
    assert not decision.accepted
    assert "renders the camera rig of openarm_v2" in decision.payload
    assert io._admit(_request(model="openarm_v2")).accepted


def test_a_robot_with_no_name_is_refused(tmp_path):
    decision = _robots_io(tmp_path)._admit(_request(robot=""))
    assert not decision.accepted
    assert "the copy it runs as" in decision.payload


def test_another_caller_naming_the_standing_robot_is_refused(tmp_path):
    """The name is the sharper answer than the engine's capacity: it names
    the instance whose robot stands under it."""
    io = _robots_io(tmp_path)
    assert io._admit(_request()).accepted
    decision = io._admit(_request(instance="other_init_inst"))
    assert not decision.accepted
    assert "already stands as 'alpha'" in decision.payload


def test_a_copy_re_registering_its_own_robot_is_admitted(tmp_path):
    """Its initializer died and came back: the same instance attaching the
    robot it stands is accepted, and the robot is never stood again."""
    io = _robots_io(tmp_path)
    assert io._admit(_request()).accepted
    io._robots.stand("alpha", LIMBS, now_s=1.0)

    decision = io._admit(_request())

    assert decision.accepted
    assert decision.payload.arm_names == ["left", "right"]
    assert list(io._robots.standing()) == ["alpha"], "one entity, not two"
    assert io._handover("alpha").is_set() is False, "the new goal's own signal is fresh"


def test_a_re_registration_naming_another_model_is_refused(tmp_path):
    io = _robots_io(tmp_path)
    assert io._admit(_request()).accepted
    io._robots.stand("alpha", LIMBS, now_s=1.0)
    decision = io._admit(_request(model="openarm_v1"))
    assert not decision.accepted
    assert "stands 'alpha' as openarm_v2" in decision.payload


class _Pairs:
    """The robots holding a limb pair, as the engine reads its pairs."""

    def __init__(self, paired) -> None:
        self.paired = set(paired)

    def robots_with_any_limb(self) -> set:
        return set(self.paired)


def _leasing(tmp_path: Path, now_s: float, paired) -> RobotsIO:
    """A RobotsIO whose robot `alpha` last renewed its lease at 0 s, read at
    `now_s` with the lease this scene gives."""
    io = _robots_io(tmp_path)
    io._robots.admit("alpha", "openarm_v2", Caller(core_node="sim16", instance_id="alpha_init_inst"), 0.0)
    io._loop = SimpleNamespace(time=lambda: now_s)
    io._io = _Pairs(paired)
    io._lease_s = 2.0
    return io


class _Answers:
    """The answers a goal context takes."""

    def __init__(self):
        self.answers = []

    async def complete(self, success, message):
        self.answers.append(("completed", success, message))

    async def complete_cancelled(self, success, message):
        self.answers.append(("cancelled", success, message))


def _ending_io(tmp_path: Path) -> RobotsIO:
    io = _robots_io(tmp_path)
    io._stands = Mock(spec=["unstand"])
    io._io = Mock(spec=["forget"])
    return io


class _Goal(_Answers):
    """A goal context carrying `request`."""

    def __init__(self, request):
        super().__init__()
        self._request = request

    def request(self):
        return self._request


@pytest.mark.parametrize(
    ("model", "pack"), [("openarm_v2", HEAD_CAMERA_PACK), ("openarm_v1", None)]
)
def test_a_stand_seats_the_head_camera_its_model_draws(tmp_path, model, pack):
    """The scene is asked to stand a v2 robot with the catalogue's head
    camera pack and a v1 robot with none."""
    io = _robots_io(tmp_path)
    io._robots.admit("alpha", model, Caller(core_node="sim16", instance_id="alpha_init_inst"), 0.0)
    refused = concurrent.futures.Future()
    refused.set_exception(RuntimeError("the test scene stands nothing"))
    io._stands = Mock(spec=["stand"])
    io._stands.stand.return_value = refused
    goal = _Goal(_request(model=model))

    asyncio.run(io._stay(goal))

    io._stands.stand.assert_called_once_with(
        tmp_path / f"{model}.xml", "alpha", head_camera_pack=pack
    )
    assert goal.answers == [
        ("completed", False, "the scene could not stand the robot: the test scene stands nothing")
    ]


class _Unreachable(_Answers):
    """A goal context whose holder cannot be told anything."""

    async def publish_feedback(self, standing):
        raise RuntimeError("the holder is away")


def test_a_robot_whose_holder_cannot_be_told_it_stands_is_taken_out(tmp_path):
    io = _ending_io(tmp_path)
    taken_out = concurrent.futures.Future()
    taken_out.set_result(None)
    io._stands.unstand.return_value = taken_out
    goal = _Unreachable()

    assert asyncio.run(io._standing(goal, "alpha")) is False

    io._stands.unstand.assert_called_once_with()
    io._io.forget.assert_called_once_with("alpha")
    assert goal.answers == [
        ("completed", False, "the robot's holder could not be told it stands: the holder is away")
    ]


def test_a_stay_ending_with_the_engine_asks_nothing_of_the_stopping_scene(tmp_path):
    io = _ending_io(tmp_path)
    goal = _Answers()

    asyncio.run(io._take_out(goal, "alpha", Ending.STOPPED, "the engine stopped"))

    io._stands.unstand.assert_not_called()
    io._io.forget.assert_called_once_with("alpha")
    assert goal.answers == [("completed", False, "the engine stopped")]


def test_a_stay_that_lapsed_takes_the_robot_out_of_the_scene_first(tmp_path):
    io = _ending_io(tmp_path)
    taken_out = concurrent.futures.Future()
    taken_out.set_result(None)
    io._stands.unstand.return_value = taken_out
    goal = _Answers()

    asyncio.run(io._take_out(goal, "alpha", Ending.LAPSED, "its pairs were gone for the lease"))

    io._stands.unstand.assert_called_once_with()
    io._io.forget.assert_called_once_with("alpha")
    assert goal.answers == [("completed", False, "its pairs were gone for the lease")]


def test_a_robot_still_paired_keeps_its_place_however_stale_its_lease(tmp_path):
    """The watcher's tick renews a lease from the pairs it reads, and a node
    loop too busy to run it leaves the lease stale: a robot still holding a
    pair is not taken out for that."""
    io = _leasing(tmp_path, now_s=10.0, paired={"alpha"})
    robot = io._robots.of_name("alpha")

    assert not io._lapsed(robot)
    assert robot.last_paired_s == 10.0


def test_a_robot_holding_no_pair_for_the_lease_lapses(tmp_path):
    io = _leasing(tmp_path, now_s=10.0, paired=set())

    assert io._lapsed(io._robots.of_name("alpha"))
