import asyncio
from types import SimpleNamespace

import numpy as np

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
        "camera_names": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeTopic:
    """A consumed-topic module surface: hands out one prebuilt subscription."""

    def __init__(self, subscription):
        self._subscription = subscription

    async def subscribe(self, _runner):
        return self._subscription


class RecordingSink:
    """A FrameSink that keeps every frame it was handed."""

    def __init__(self):
        self.frames = []

    def put_frame(self, frame):
        self.frames.append(frame)


class FakeSubscription:
    """Yields queued items, then pends forever like the generated one."""

    def __init__(self, items):
        self._items = list(items)

    async def next(self):
        if self._items:
            return self._items.pop(0)
        await asyncio.Event().wait()


class FakeToken:
    """A cancellation token the test fires by hand."""

    def __init__(self):
        self._event = asyncio.Event()

    def cancel(self):
        self._event.set()

    def is_cancelled(self):
        return self._event.is_set()

    async def cancelled(self):
        await self._event.wait()
