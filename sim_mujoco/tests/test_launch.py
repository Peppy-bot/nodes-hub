"""The node's setup binds the viewer, announces the address it took, and
hands the server to the thread that steps the scene.

What the setup starts is faked: each step it takes is recorded, and a test
makes one of them fail.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import viewer_fakes

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime the engine imports)

_ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(_ENGINE_DIR))

from viewer_server import ViewerServer  # noqa: E402  pylint: disable=C0413

# The port a launcher prefers for the viewer in these tests, other than the
# manifest's default, so a setup that ignores its parameter fails them.
PREFERRED_PORT = 8311
# The port the viewer took, other than the one the launcher prefers, so a
# setup that announces its parameter fails them.
BOUND_PORT = 49731

_PARAMS = types.SimpleNamespace(
    viewer_host="127.0.0.1",
    viewer_port=PREFERRED_PORT,
    cameras_enabled=False,
    state_rate_hz=100,
    robot_pair_grace_ms=2000,
)


class _Refused(Exception):
    """What a step of the setup fails with."""


@pytest.fixture(name="node")
def node_fixture(monkeypatch):
    """The node's launch module over what it starts, faked. `steps` holds
    each step the setup took, in order, and the step named `failing` raises
    `_Refused`."""
    node = types.SimpleNamespace(steps=[], failing=None, viser=None, drawn_on=None)

    def take(step: str) -> None:
        node.steps.append(step)
        if step == node.failing:
            raise _Refused(step)

    def serve(host, port):
        node.served_on = (host, port)
        take("bind")
        node.viser = viewer_fakes.Viser(host, BOUND_PORT)
        return ViewerServer(node.viser)

    class _Topics:
        def __init__(self, *_args) -> None:
            pass

        async def start(self) -> None:
            take("topics")

    class _Contract:
        def __init__(self, *_args) -> None:
            pass

        async def start(self) -> None:
            take("contract")

        def binds_with(self, _rebind, _unbind) -> None:
            pass

    class _Launcher:
        def __init__(self, _world, _edits, _stop, _io, _state_rate_hz, viewer_server) -> None:
            node.drawn_on = viewer_server

        def run(self) -> None:
            take("run")

        def rebind(self) -> None:
            pass

        def unbind(self) -> None:
            pass

    node.runner = Mock(spec=["announce_endpoint", "on_shutdown"])
    node.runner.announce_endpoint.side_effect = lambda *_binding: take("announce")
    node.runner.on_shutdown.side_effect = lambda _hook: take("shutdown hook")

    # The module configures the process's logging as it is imported, which is
    # the node's to do and not a suite's.
    monkeypatch.setattr(logging, "basicConfig", Mock())
    spec = importlib.util.spec_from_file_location(
        "_sim_mujoco_launch_under_test", _ENGINE_DIR / "launch.py"
    )
    node.module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node.module)
    monkeypatch.setattr(node.module, "MujocoModels", types.SimpleNamespace(read=object))
    monkeypatch.setattr(node.module, "ViewerServer", types.SimpleNamespace(serve=serve))
    monkeypatch.setattr(node.module, "SimTopicIO", _Topics)
    monkeypatch.setattr(node.module, "RobotsIO", _Contract)
    monkeypatch.setattr(node.module, "SimLauncher", _Launcher)
    return node


def _set_up(node) -> list:
    """Runs the node's setup, and then the thread that steps the scene to
    its end. Answers the tasks the setup returned."""

    async def run() -> list:
        tasks = await node.module._run_sim(_PARAMS, node.runner)  # pylint: disable=W0212
        await asyncio.gather(*tasks)
        return tasks

    return asyncio.run(run())


def test_setup_announces_the_viewer_at_the_address_it_bound(node):
    """The daemon renders the URLs an operator opens from the announcement,
    so it carries the port the viewer took, not the one it prefers."""
    _set_up(node)

    assert node.served_on == ("127.0.0.1", PREFERRED_PORT)
    node.runner.announce_endpoint.assert_called_once_with("viewer", "http", "127.0.0.1", BOUND_PORT)


def test_the_viewer_is_bound_before_it_is_announced_and_before_a_robot_can_attach(node):
    _set_up(node)

    assert node.steps == ["bind", "announce", "topics", "contract", "shutdown hook", "run"]


def test_every_scene_is_drawn_on_the_server_the_node_announced(node):
    _set_up(node)

    assert node.drawn_on.viser is node.viser
    assert not node.viser.stopped, "the thread that steps the scene stops the server it draws on"


def test_the_thread_that_steps_the_scene_ends_with_the_stop_set(node):
    """The stop is what ends the loops that read the scene, so it is set
    however the thread ended."""
    [stepping] = _set_up(node)

    assert stepping.done()
    assert node.module._stop.is_set()  # pylint: disable=W0212


@pytest.mark.parametrize("failing", ["announce", "topics", "contract", "shutdown hook"])
def test_a_setup_that_fails_once_the_viewer_is_bound_stops_the_viewer(node, failing):
    """The thread that stops the viewer's server never starts, so the setup
    gives the address back itself."""
    node.failing = failing

    with pytest.raises(_Refused, match=failing):
        _set_up(node)

    assert node.viser.stopped
    assert "run" not in node.steps


def test_a_viewer_that_cannot_bind_fails_the_setup_before_the_node_exposes_anything(node):
    node.failing = "bind"

    with pytest.raises(_Refused, match="bind"):
        _set_up(node)

    assert node.steps == ["bind"]
