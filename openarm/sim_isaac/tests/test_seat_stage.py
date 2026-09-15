"""Standing a robot on the stage, and what happens when it cannot be stood.

Standing lets go of the stage before it changes it: the views stop reading and
the timeline stops. Whatever the change does, the stage has to be taken up
again, or the robots already on it are frozen for good.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
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
sys.modules["peppygen.exposed_services.robots"].command = ModuleType("command")

from seat_io import SeatIO  # noqa: E402  pylint: disable=C0413


def _seat_io(world):
    """A SeatIO with only the parts standing a robot touches."""
    io = SeatIO.__new__(SeatIO)
    io._world = world
    io._seats = Mock(spec=["renew"])
    io._loop = Mock()
    io._loop.time.return_value = 0.0
    io._bind = Mock()
    io._unbind_stage = Mock()
    io.binds_with(io._bind, io._unbind_stage)
    return io


def test_a_robot_that_stands_leaves_the_stage_taken_up():
    world = Mock(spec=["add", "remove"])
    io = _seat_io(world)

    io.stand("alpha", "openarm_v2", Mock())

    world.add.assert_called_once()
    io._bind.assert_called_once_with()
    io._unbind_stage.assert_called_once_with()


def test_a_robot_the_stage_refuses_leaves_it_taken_up_anyway():
    """A duplicate name, a model with no stage, or any USD error raises out of
    World.add, after the stage has already been let go of."""
    world = Mock(spec=["add", "remove"])
    world.add.side_effect = ValueError("alpha already stands in the stage")
    io = _seat_io(world)

    with pytest.raises(ValueError, match="already stands"):
        io.stand("alpha", "openarm_v2", Mock())

    io._bind.assert_called_once_with()
    world.remove.assert_not_called()


def test_a_robot_the_engine_cannot_resolve_is_taken_back_out():
    world = Mock(spec=["add", "remove"])
    io = _seat_io(world)
    io._bind.side_effect = [RuntimeError("joints are not the ones driven"), None]

    with pytest.raises(RuntimeError, match="joints"):
        io.stand("bravo", "openarm_v1", Mock())

    world.remove.assert_called_once_with("bravo")
    assert io._bind.call_count == 2


def test_a_robot_that_cannot_be_taken_out_leaves_the_stage_taken_up():
    world = Mock(spec=["add", "remove"])
    world.remove.side_effect = KeyError("charlie")
    io = _seat_io(world)

    with pytest.raises(KeyError):
        io.unstand("charlie")

    io._bind.assert_called_once_with()
