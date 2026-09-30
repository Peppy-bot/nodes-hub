"""Shared test pieces: the node's parameters, fake backends, a fake goal
context, and the camera's messages, so the core and the handlers run in
plain tests with no router, no robot and no model."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np
import pytest

from peppygen.parameters import Parameters

from openarm_ai_brain_vla.perception import weights
from openarm_ai_brain_vla.ports import Box, CancelToken, Coverage, Deadline, Gripper, Item, Outcome, Pose

PARAMS = {
    "gripper_names": "left_gripper,right_gripper",
    "perception_backend": "none",
    "perception_model": "",
    "perception_gallery": "",
    "perception_confidence": 0.0,
    "manipulation_backend": "none",
    # At the origin, looking along world -Z with +Y up: the identity pose.
    "camera_pose": "0 0 0 0 0 0 1",
}

# A deadline no search reaches, and one every search has passed.
NEVER = Deadline(at=float("inf"), budget_s=float("inf"))
PASSED = Deadline(at=0.0, budget_s=1.0)


@pytest.fixture
def params() -> Parameters:
    return Parameters.from_dict(dict(PARAMS))


@pytest.fixture
def no_network(monkeypatch):
    """Fails the test at any reach for the network, a name lookup or a
    connection: what runs under it downloads nothing."""
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError(f"reached for the network: {args!r}")

    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture
def staged_weights(tmp_path, monkeypatch):
    """The two models on the machine: a directory of the test's own, which
    the node is told to keep its weights in, holds the directory of each
    model, so a load downloads nothing."""
    directory = tmp_path / "weights"
    for source in weights.SOURCES:
        (directory / source.directory_name).mkdir(parents=True)
    monkeypatch.setenv(weights.WEIGHTS_DIRECTORY_VARIABLE, str(directory))
    return directory


class FakeToken:
    """Stands in for the node's cancellation token."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    async def cancelled(self) -> None:
        await self._event.wait()

    def cancel(self) -> None:
        self._event.set()


class FakeDetector:
    """Returns the boxes the test hands it, and records the vocabulary and
    the deadlines it was given. Its scans name every label unless the test
    hands it a coverage."""

    name = "fake"

    def __init__(self, boxes: Optional[list[Box]] = None, coverage: Coverage = Coverage(every_label=True)) -> None:
        self.boxes = boxes or []
        self.coverage = coverage
        self.loaded: Optional[tuple[str, str]] = None
        self.vocabulary: list[str] = []
        self.deadlines: list[Deadline] = []
        self.min_confidence = 0.0

    @property
    def available(self) -> bool:
        return True

    def load(self, model: str, gallery: str) -> None:
        self.loaded = (model, gallery)

    def set_vocabulary(self, phrases: Sequence[str]) -> None:
        self.vocabulary = list(phrases)

    def detect(self, image: np.ndarray, deadline: Deadline) -> list[Box]:
        self.deadlines.append(deadline)
        return list(self.boxes)

    def scan_coverage(self) -> Coverage:
        return self.coverage


class FakeManipulator:
    """Succeeds at once unless told to fail or to wait for a stop. `begun`
    resolves when a sequence has reached it, so a test knows the sequence
    is running without waiting on the clock."""

    name = "fake"

    def __init__(self, *, fail: str = "", wait_for_stop: bool = False) -> None:
        self.fail = fail
        self.wait_for_stop = wait_for_stop
        self.calls: list[tuple] = []
        self.stopped = 0
        self.robot = None
        self._began = asyncio.Event()

    @property
    def available(self) -> bool:
        return True

    async def start(self, robot) -> None:
        self.robot = robot

    async def begun(self) -> None:
        await self._began.wait()
        self._began.clear()

    async def _run(self, cancel: CancelToken, final_position=None) -> Outcome:
        self._began.set()
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
    """Stands in for the generated GoalContext of `action`: holds the goal,
    records the completion after checking that its fields are the ones the
    action's result carries, and lets a test cancel the goal the way a
    caller would."""

    def __init__(self, action, **data) -> None:
        self.action = action
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
        self._record(fields)
        self.completed = fields

    async def complete_cancelled(self, **fields) -> None:
        self._record(fields)
        self.cancelled = fields

    def _record(self, fields: dict) -> None:
        assert self.completed is None and self.cancelled is None, "completed twice"
        expected = set(inspect.signature(self.action.GoalContext.complete).parameters) - {"self"}
        assert set(fields) == expected, f"{self.action.__name__} completed with {sorted(fields)}, its result carries {sorted(expected)}"


def answer_the_camera(h, *, colour=None) -> None:
    """The camera mocks of a harness answer what the brain asks at start:
    the depth unit, and the geometry once per answer in `colour`, the
    known colour intrinsics when none are given."""
    from peppygen.consumed_services.camera import depth_stream_info
    from peppygen.consumed_services.geometry import get_depth_intrinsics

    h.mocks.deps.camera.depth_stream_info.enqueue_response(
        depth_stream_info.ResponseData(width=16, height=12, frames_per_second=15, encoding="z16", depth_unit=0.001)
    )
    for answer in colour if colour is not None else (colour_intrinsics(),):
        h.mocks.deps.geometry.get_color_intrinsics.enqueue_response(answer)
        h.mocks.deps.geometry.get_depth_intrinsics.enqueue_response(
            get_depth_intrinsics.ResponseData(
                success=True, message="", width=16, height=12, fx=6.0, fy=6.0, cx=8.0, cy=6.0, distortion_model="none", distortion=[],
                depth_model="z", min_depth_m=0.1, max_depth_m=10.0, align_mode="depth_to_color",
            )
        )


def colour_intrinsics(success: bool = True, message: str = ""):
    """The colour intrinsics of a 16 x 12 camera with a 90 degree lens, or
    the answer of a camera that does not know them yet."""
    from peppygen.consumed_services.geometry import get_color_intrinsics

    if not success:
        return get_color_intrinsics.ResponseData(
            success=False, message=message, width=0, height=0, fx=0.0, fy=0.0, cx=0.0, cy=0.0, distortion_model="", distortion=[]
        )
    return get_color_intrinsics.ResponseData(
        success=True, message="", width=16, height=12, fx=6.0, fy=6.0, cx=8.0, cy=6.0, distortion_model="none", distortion=[]
    )


def rgb_frame(width: int = 16, height: int = 12, *, frame_id: int = 1, align_mode: str = "depth_to_color"):
    """A colour message of the given size, plain grey."""
    from peppygen.consumed_topics.camera import video_stream

    return video_stream.Message(
        header=video_stream.MessageHeader(timestamp=1.0, frame_id=frame_id, align_mode=align_mode),
        encoding="rgb8",
        width=width,
        height=height,
        frame=bytes([128]) * (width * height * 3),
    )


def depth_frame(depth_m: float, width: int = 16, height: int = 12, unit: float = 0.001, *, frame_id: int = 1, align_mode: str = "depth_to_color"):
    """A z16 depth message of the given size, `depth_m` everywhere."""
    from peppygen.consumed_topics.camera import depth_stream

    value = int(round(depth_m / unit))
    return depth_stream.Message(
        header=depth_stream.MessageHeader(timestamp=1.0, frame_id=frame_id, align_mode=align_mode),
        encoding="z16",
        width=width,
        height=height,
        frame=np.full((height, width), value, dtype="<u2").tobytes(),
    )
