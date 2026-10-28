"""Shared test pieces: the node's parameters, fake models on brain_core's
ports, an enrolment shipped from a test's own folder, and the camera's
messages, so the node's parts run in plain tests with no router, no robot
and no model."""

from __future__ import annotations

import asyncio
from typing import Optional, Sequence

import numpy as np
import pytest

from brain_core.ports import Box, CancelToken, Coverage, Deadline, Gripper, Item, Outcome, Pose
from peppygen.parameters import Parameters

from ai_brain_vla.perception import gallery, weights

PARAMS = {
    "limbs": "left_arm:left_gripper,right_arm:right_gripper",
    "perception_model": "none",
    "manipulation_model": "none",
    "perception_gallery": "",
    "camera_name": "chest",
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


@pytest.fixture
def shipped_folder(tmp_path, monkeypatch):
    """The folder the node ships its enrolments from, a test's own and
    empty: an enrolment written in it under a name is one the node ships
    under that name."""
    folder = tmp_path / "enrolments"
    folder.mkdir()
    monkeypatch.setattr(gallery, "SHIPPED_DIRECTORY", folder)
    return folder


class FakeDetector:
    """Returns the boxes the test hands it, and records the phrases of each
    search. Its scans name every label."""

    name = "fake"

    def __init__(self, boxes: Optional[list[Box]] = None) -> None:
        self.boxes = boxes or []
        self.phrases: list[list[str]] = []

    def why_unavailable(self) -> str:
        return ""

    async def detect(self, image: np.ndarray, phrases: Sequence[str], deadline: Deadline) -> list[Box]:
        self.phrases.append(list(phrases))
        return list(self.boxes)

    def scan_coverage(self) -> Coverage:
        return Coverage(every_label=True)


class FakeManipulator:
    """Succeeds at once unless told to wait for a stop. `begun` resolves when
    a sequence has reached it."""

    name = "fake"

    def __init__(self, *, wait_for_stop: bool = False) -> None:
        self.wait_for_stop = wait_for_stop
        self.calls: list[tuple] = []
        self.stopped = 0
        self._began = asyncio.Event()

    def why_unavailable(self) -> str:
        return ""

    async def start(self, senses) -> None:
        return None

    async def begun(self) -> None:
        await self._began.wait()
        self._began.clear()

    async def _run(self, cancel: CancelToken, final_position=None) -> Outcome:
        self._began.set()
        if self.wait_for_stop:
            await cancel.wait()
            cancel.check()
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


def answer_the_camera(h, *, colour: bool = True, mounts=None) -> None:
    """The camera and robot mocks of a harness answer what the brain asks
    at start: the depth unit, the depth intrinsics, the known colour
    intrinsics unless `colour` is false and the test answers them itself,
    and where the camera stands once per answer in `mounts`, the chest
    camera at the origin looking along the robot's +Z when none are
    given."""
    from peppygen.consumed_services.camera import depth_stream_info
    from peppygen.consumed_services.geometry import get_depth_intrinsics

    h.mocks.deps.camera.depth_stream_info.enqueue_response(
        depth_stream_info.ResponseData(width=16, height=12, frames_per_second=15, encoding="z16", depth_unit=0.001)
    )
    if colour:
        h.mocks.deps.geometry.get_color_intrinsics.enqueue_response(colour_intrinsics())
    h.mocks.deps.geometry.get_depth_intrinsics.enqueue_response(
        get_depth_intrinsics.ResponseData(
            width=16, height=12, fx=6.0, fy=6.0, cx=8.0, cy=6.0, distortion_model="none", distortion=[],
            depth_model="z", min_depth_m=0.1, max_depth_m=10.0, align_mode="depth_to_color",
        )
    )
    for answer in mounts if mounts is not None else (camera_poses(),):
        h.mocks.deps.camera_mounts.get_camera_poses.enqueue_response(answer)


def camera_poses(names=("chest",)):
    """The robot's response: every camera of `names` at the origin, its
    optical frame the robot frame's."""
    from peppygen.consumed_services.camera_mounts import get_camera_poses

    return get_camera_poses.ResponseData(
        timestamp=7.0, camera_names=list(names),
        positions=[0.0, 0.0, 0.0] * len(names), orientations=[0.0, 0.0, 0.0, 1.0] * len(names),
        carried_by=[""] * len(names),
    )


def colour_intrinsics():
    """The colour intrinsics of a 16 x 12 camera with a 90 degree lens."""
    from peppygen.consumed_services.geometry import get_color_intrinsics

    return get_color_intrinsics.ResponseData(
        width=16, height=12, fx=6.0, fy=6.0, cx=8.0, cy=6.0, distortion_model="none", distortion=[]
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
