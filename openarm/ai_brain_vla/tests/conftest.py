"""Shared test pieces: the node's parameters, fake backends, and a fake
goal context, so the core and the handlers run in plain tests with no
router, no robot and no model."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np
import pytest

from peppygen.parameters import Parameters

from openarm_ai_brain_vla.ports import Box, CancelToken, Gripper, Item, Outcome, Pose

PARAMS = {
    "gripper_names": "left_gripper,right_gripper",
    "perception_backend": "none",
    "perception_model": "",
    "manipulation_backend": "none",
    "camera_fovy_deg": 90.0,
    # At the origin, looking along world -Z with +Y up: the identity pose.
    "camera_pose": "0 0 0 0 0 0 1",
}


@pytest.fixture
def params() -> Parameters:
    return Parameters.from_dict(dict(PARAMS))


class FakeDetector:
    """Returns the boxes the test hands it, and records the vocabulary."""

    name = "fake"

    def __init__(self, boxes: Optional[list[Box]] = None) -> None:
        self.boxes = boxes or []
        self.loaded: Optional[str] = None
        self.vocabulary: list[str] = []

    @property
    def available(self) -> bool:
        return True

    def load(self, model: str) -> None:
        self.loaded = model

    def set_vocabulary(self, phrases: Sequence[str]) -> None:
        self.vocabulary = list(phrases)

    def detect(self, image: np.ndarray) -> list[Box]:
        return list(self.boxes)


class FakeManipulator:
    """Succeeds at once unless told to fail or to wait for a stop."""

    name = "fake"

    def __init__(self, *, fail: str = "", wait_for_stop: bool = False) -> None:
        self.fail = fail
        self.wait_for_stop = wait_for_stop
        self.calls: list[tuple] = []
        self.stopped = 0
        self.robot = None

    @property
    def available(self) -> bool:
        return True

    async def start(self, robot) -> None:
        self.robot = robot

    async def _run(self, cancel: CancelToken, final_position=None) -> Outcome:
        if self.wait_for_stop:
            await cancel.wait()
            cancel.check()
        if self.fail:
            return Outcome(False, self.fail)
        return Outcome(True, "", final_position)

    async def grab(self, item: Item, gripper: Gripper, max_effort: float, cancel: CancelToken) -> Outcome:
        self.calls.append(("grab", item.item_id, gripper.name, max_effort))
        return await self._run(cancel)

    async def drop(self, gripper: Gripper, cancel: CancelToken) -> Outcome:
        self.calls.append(("drop", gripper.name))
        return await self._run(cancel)

    async def place(self, gripper: Gripper, pose: Pose, cancel: CancelToken) -> Outcome:
        self.calls.append(("place", gripper.name, pose.position))
        return await self._run(cancel, final_position=(pose.position[0], pose.position[1], pose.position[2] + 0.01))

    async def stop(self) -> None:
        self.stopped += 1


class FakeCtx:
    """Stands in for a generated GoalContext: holds the goal, records the
    completion, and lets a test cancel the goal the way a caller would."""

    def __init__(self, **data) -> None:
        self._request = SimpleNamespace(data=SimpleNamespace(**data))
        self._cancel = asyncio.Event()
        self.completed: Optional[dict] = None
        self.cancelled: Optional[dict] = None

    def request(self):
        return self._request

    def goal_id(self) -> str:
        return "goal"

    async def cancel_signal(self) -> None:
        await self._cancel.wait()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()

    async def complete(self, **fields) -> None:
        assert self.completed is None and self.cancelled is None, "completed twice"
        self.completed = fields

    async def complete_cancelled(self, **fields) -> None:
        assert self.completed is None and self.cancelled is None, "completed twice"
        self.cancelled = fields


def rgb_frame(width: int = 16, height: int = 12):
    """A colour message of the given size, plain grey."""
    from peppygen.consumed_topics.camera import video_stream

    return video_stream.Message(
        header=video_stream.MessageHeader(timestamp=1.0, frame_id=1, align_mode="depth_to_color"),
        encoding="rgb8",
        width=width,
        height=height,
        frame=bytes([128]) * (width * height * 3),
    )


def depth_frame(depth_m: float, width: int = 16, height: int = 12, unit: float = 0.001):
    """A z16 depth message of the given size, `depth_m` everywhere."""
    from peppygen.consumed_topics.camera import depth_stream

    value = int(round(depth_m / unit))
    return depth_stream.Message(
        header=depth_stream.MessageHeader(timestamp=1.0, frame_id=1, align_mode="depth_to_color"),
        encoding="z16",
        width=width,
        height=height,
        frame=np.full((height, width), value, dtype="<u2").tobytes(),
    )
