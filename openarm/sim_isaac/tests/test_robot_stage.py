"""Standing a robot on the stage, what happens when it cannot be stood, and
how a robot leaves it.

Standing lets go of the stage before it changes it: the views stop reading and
the timeline stops. Whatever the change does, the stage has to be taken up
again, or the robots already on it are frozen for good. A robot leaves when
its goal is cancelled, or when it holds no limb pair for the lease.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robots" / "openarm"))

# The typed transport is not under test; the module only needs its names.
for _name in (
    "peppygen",
    "peppygen.exposed_actions",
    "peppygen.exposed_actions.robots",
    "peppygen.exposed_services",
    "peppygen.exposed_services.robots",
):
    sys.modules.setdefault(_name, ModuleType(_name))
sys.modules["peppygen.exposed_actions.robots"].attach = ModuleType("attach")
sys.modules["peppygen.exposed_services.robots"].is_ready = ModuleType("is_ready")

from edits import Edits  # noqa: E402  pylint: disable=C0413
from robots import Caller, Registry  # noqa: E402  pylint: disable=C0413
from robots_io import Ending, RobotsIO  # noqa: E402  pylint: disable=C0413


def _robots_io(world):
    """A RobotsIO with only the parts standing a robot touches."""
    io = RobotsIO.__new__(RobotsIO)
    io._world = world
    io._robots = Mock(spec=["renew", "release"])
    io._io = Mock(spec=["forget"])
    io._loop = Mock()
    io._loop.time.return_value = 0.0
    io._bind = Mock()
    io._unbind_stage = Mock()
    io.binds_with(io._bind, io._unbind_stage)
    return io


def test_a_robot_that_stands_leaves_the_stage_taken_up():
    world = Mock(spec=["add", "remove"])
    io = _robots_io(world)

    io.stand("alpha", "openarm_v2", Mock())

    world.add.assert_called_once()
    io._bind.assert_called_once_with()
    io._unbind_stage.assert_called_once_with()


def test_a_robot_the_stage_refuses_leaves_it_taken_up_anyway():
    """A duplicate name, a model with no stage, or any USD error raises out of
    World.add, after the stage has already been let go of."""
    world = Mock(spec=["add", "remove"])
    world.add.side_effect = ValueError("alpha already stands in the stage")
    io = _robots_io(world)

    with pytest.raises(ValueError, match="already stands"):
        io.stand("alpha", "openarm_v2", Mock())

    io._bind.assert_called_once_with()
    world.remove.assert_not_called()


def test_a_robot_the_engine_cannot_resolve_is_taken_back_out():
    world = Mock(spec=["add", "remove"])
    io = _robots_io(world)
    io._bind.side_effect = [RuntimeError("joints are not the ones driven"), None]

    with pytest.raises(RuntimeError, match="joints"):
        io.stand("bravo", "openarm_v1", Mock())

    world.remove.assert_called_once_with("bravo")
    assert io._bind.call_count == 2


def test_a_robot_that_cannot_be_taken_out_leaves_the_stage_taken_up():
    world = Mock(spec=["add", "remove"])
    world.remove.side_effect = KeyError("charlie")
    io = _robots_io(world)

    with pytest.raises(KeyError):
        io.unstand("charlie")

    io._bind.assert_called_once_with()


def test_a_name_is_free_the_moment_nothing_stands_under_it():
    """Taking the robot out and giving its name back are one change, so a
    robot that comes straight back is neither refused its own name nor stood
    twice."""
    world = Mock(spec=["add", "remove"])
    io = _robots_io(world)
    order = []
    world.remove.side_effect = lambda *_: order.append("left the stage")
    io._robots.release.side_effect = lambda *_: order.append("name given back")
    io._io.forget.side_effect = lambda *_: order.append("setpoints dropped")

    io._rebind = Mock(side_effect=lambda: order.append("scene resolved"))

    io.unstand("alpha")

    assert order == [
        "left the stage",
        "name given back",
        "setpoints dropped",
        "scene resolved",
    ]


def test_a_removal_that_fails_keeps_the_name():
    """The robot is still standing, so its name stays taken."""
    world = Mock(spec=["add", "remove"])
    world.remove.side_effect = KeyError("alpha")
    io = _robots_io(world)

    with pytest.raises(KeyError):
        io.unstand("alpha")

    io._robots.release.assert_not_called()


class _Goal:
    """The parts of an attach goal a robot's stand reads and answers."""

    def __init__(self):
        self.leave = asyncio.Event()
        self.answers = []
        self.feedback = []

    async def cancel_signal(self):
        await self.leave.wait()

    async def publish_feedback(self, standing):
        self.feedback.append(standing)

    async def complete(self, success, message):
        self.answers.append(("completed", success, message))

    async def complete_cancelled(self, success, message):
        self.answers.append(("cancelled", success, message))


def _joining_io():
    """A RobotsIO with one robot admitted and waiting for the thread that
    steps the scene, which takes nothing up until a test drains the edits."""
    io = RobotsIO.__new__(RobotsIO)
    io._edits = Edits()
    io._placements = {"alpha": Mock()}
    io._handovers = {}
    io._admit_lock = threading.Lock()
    io._robots = Mock(spec=["release", "stand"])
    io._limbs = Mock()
    io._loop = Mock()
    io._loop.time.return_value = 0.0
    return io


def test_a_robot_that_leaves_before_the_stage_takes_it_up_never_stands():
    """Its stand still waits for the thread that steps the scene, so the
    stand is withdrawn and the name given back at once: a copy removed and
    joined straight back finds its name free."""
    io = _joining_io()

    async def leave_while_waiting():
        goal = _Goal()
        standing = asyncio.create_task(io._stand(goal, "alpha", "openarm_v2"))
        await asyncio.sleep(0)
        goal.leave.set()
        return goal, await asyncio.wait_for(standing, 5)

    goal, stood = asyncio.run(leave_while_waiting())

    assert stood is False
    assert goal.answers == [("cancelled", True, "the robot left before it stood")]
    io._robots.release.assert_called_once_with("alpha")
    io._robots.stand.assert_not_called()
    assert "alpha" not in io._placements
    assert io._edits.drain() == 0


def test_a_robot_that_leaves_while_the_stage_stands_it_is_stood_first():
    """A stand the thread has taken up runs to its end; the stay that follows
    sees the robot leave and takes it out."""
    io = _joining_io()
    taken_up, finish = threading.Event(), threading.Event()

    def stand(*_):
        taken_up.set()
        finish.wait(5)

    io.stand = stand

    async def leave_while_standing():
        goal = _Goal()
        standing = asyncio.create_task(io._stand(goal, "alpha", "openarm_v2"))
        await asyncio.sleep(0)
        stage = threading.Thread(target=io._edits.drain)
        stage.start()
        await asyncio.get_running_loop().run_in_executor(None, taken_up.wait, 5)
        goal.leave.set()
        await asyncio.sleep(0.05)
        finish.set()
        stood = await asyncio.wait_for(standing, 5)
        stage.join(5)
        return goal, stood

    goal, stood = asyncio.run(leave_while_standing())

    assert stood is True
    assert goal.answers == []
    assert goal.feedback == [], "the stay tells the holder, once, that it stands"
    io._robots.stand.assert_called_once()
    io._robots.release.assert_not_called()


def _done():
    """A future the thread that steps the scene has already answered."""
    future = concurrent.futures.Future()
    future.set_result(None)
    return future


class _UnreachableGoal(_Goal):
    """A goal whose holder cannot be told anything."""

    async def publish_feedback(self, standing):
        raise RuntimeError("the holder is away")


def test_a_robot_whose_holder_cannot_be_told_it_stands_is_taken_out():
    io = _joining_io()
    io._edits = Mock(spec=["submit"])
    io._edits.submit.return_value = _done()
    goal = _UnreachableGoal()

    assert asyncio.run(io._standing(goal, "alpha")) is False

    io._edits.submit.assert_called_once()
    assert goal.answers == [
        ("completed", False, "the robot's holder could not be told it stands: the holder is away")
    ]


def test_a_stay_ending_with_the_engine_asks_nothing_of_the_stopping_stage():
    io = _joining_io()
    io._edits = Mock(spec=["submit"])
    goal = _Goal()

    asyncio.run(io._take_out(goal, "alpha", Ending.STOPPED, "the engine stopped"))

    io._edits.submit.assert_not_called()
    io._robots.release.assert_called_once_with("alpha")
    assert "alpha" not in io._placements
    assert goal.answers == [("completed", False, "the engine stopped")]


def test_a_stay_that_lapsed_takes_the_robot_off_the_stage_first():
    io = _joining_io()
    io._edits = Mock(spec=["submit"])
    io._edits.submit.return_value = _done()
    goal = _Goal()

    asyncio.run(io._take_out(goal, "alpha", Ending.LAPSED, "its pairs were gone for the lease"))

    io._edits.submit.assert_called_once()
    assert goal.answers == [("completed", False, "its pairs were gone for the lease")]


class _Pairs:
    """The robots holding a limb pair, as the engine reads its pairs."""

    def __init__(self, paired):
        self.paired = set(paired)

    def robots_with_any_limb(self):
        return set(self.paired)


def _leasing(now_s, paired):
    """A RobotsIO whose robot `alpha` last renewed its lease at 0 s, read at
    `now_s` with the lease this scene gives."""
    io = RobotsIO.__new__(RobotsIO)
    io._robots = Registry()
    io._robots.admit("alpha", "openarm_v2", Caller(core_node="sim16", instance_id="alpha_init_inst"), 0.0)
    io._loop = Mock()
    io._loop.time.return_value = now_s
    io._io = _Pairs(paired)
    io._lease_s = 2.0
    return io


def test_a_robot_still_paired_keeps_its_place_however_stale_its_lease():
    """The watcher's tick renews a lease from the pairs it reads, and a node
    loop the stage kept busy leaves the lease stale: a robot still holding a
    pair is not taken out for that."""
    io = _leasing(now_s=10.0, paired={"alpha"})
    robot = io._robots.of_name("alpha")

    assert not io._lapsed(robot)
    assert robot.last_paired_s == 10.0


def test_a_robot_holding_no_pair_for_the_lease_lapses():
    io = _leasing(now_s=10.0, paired=set())

    assert io._lapsed(io._robots.of_name("alpha"))
