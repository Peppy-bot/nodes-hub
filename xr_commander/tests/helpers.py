import asyncio
import time
from types import SimpleNamespace

import numpy as np

from peppygen.fixtures import harness
from peppygen.parameters import Parameters

from xr_commander.frames import Pose

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
POSE = Pose(np.zeros(3), IDENTITY)


def default_parameters(**overrides) -> SimpleNamespace:
    """The full raw launch-parameter set, valid by default."""
    values = {
        "command_rate_hz": 1000,
        "https_port": 4443,
        "https_host": "0.0.0.0",
        "motion_scale": 1.0,
        "gripper_open_fraction": 1.0,
        "posture_move_duration_s": 2.0,
        "stale_timeout_s": 10.0,
        "view_max_width": 640,
        "status_panel_enabled": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def launch_parameters(**overrides) -> Parameters:
    """`default_parameters` as the typed shape the generated harness seeds."""
    return Parameters(**vars(default_parameters(**overrides)))


async def idle_setup(_params, _node_runner):
    """A setup that starts nothing: the real `setup` raises TLS, an HTTPS
    server, and a WebXR session, none of which belongs in a test. Harness
    tests boot the node's runtime (ephemeral router, generated mocks, seeded
    slots) under this no-op and drive the production functions directly
    against the real runner and real generated modules."""
    return []


def boot(**kwargs):
    """The generated harness under `idle_setup`: awaitable or an async
    context manager. Keyword arguments are `harness.start`'s: per-slot mock
    counts (`<link_id>_instances`) and the usual overrides."""
    kwargs.setdefault("parameters", launch_parameters())
    return harness.start(idle_setup, **kwargs)


async def eventually(predicate, timeout=10.0, interval=0.005, message="condition"):
    """Poll until `predicate()` is truthy; AssertionError past `timeout`.

    The bounded-wait primitive for harness tests: wire delivery is ordered
    but not instant, so state assertions converge instead of sleeping blind.
    """
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"{message}: not met within {timeout}s")
        await asyncio.sleep(interval)


class RecordingSink:
    """A FrameSink that keeps every frame it was handed."""

    def __init__(self):
        self.frames = []

    def put_frame(self, frame):
        self.frames.append(frame)


class FakeSubscription:
    """Yields queued items, then pends forever like the generated one.

    Kept for the pure bus-plumbing tests (`messages`/`next_message`), which
    exercise receive-loop edge cases no real subscription can be scripted
    into (mid-stream exceptions, self-cancelling receives)."""

    def __init__(self, items):
        self._items = list(items)

    async def next(self):
        if self._items:
            return self._items.pop(0)
        await asyncio.Event().wait()


class FakeToken:
    """A cancellation token the test fires by hand.

    Kept deliberately alongside the harness: the production tasks accept any
    token with the CancellationToken shape, and a test must be able to end
    one drain without cancelling the runner's real token (which would
    converge the whole node mid-test). The pure loop tests have no runner at
    all."""

    def __init__(self):
        self._event = asyncio.Event()

    def cancel(self):
        self._event.set()

    def is_cancelled(self):
        return self._event.is_set()

    async def cancelled(self):
        await self._event.wait()
