"""Shared test pieces: the node's parameters, fake models on brain_core's
ports, an enrolment shipped from a test's own folder, and the camera's
messages, so the node's parts run in plain tests with no router, no robot
and no model."""

from __future__ import annotations

import asyncio
from typing import Optional, Sequence
from weakref import WeakKeyDictionary

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
    "lookup_timeout_s": 0.5,
}

# The robot's cameras, each with its frame and its place in the robot frame
# at every time: the chest at the robot's origin looking along +Z.
ROBOT_FRAME = "base_link"
CAMERAS = {"chest": ("chest_color_optical_frame", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])}
# How long the answerer of the lookups waits for the next one: longer than a
# test runs, so it stops when the harness stops its mocks.
LOOKUP_WAIT_S = 3600.0
# The answerer of each harness's lookups, kept while it runs.
_ANSWERERS: WeakKeyDictionary = WeakKeyDictionary()

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

    def starting(self) -> str:
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

    def starting(self) -> str:
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


def answer_the_camera(h, *, colour: bool = True, sensors=None) -> None:
    """The camera and robot mocks of a harness answer what the brain asks:
    at start, the depth unit, the depth intrinsics, the known colour
    intrinsics unless `colour` is false and the test answers them itself,
    and the robot's sensors once per answer in `sensors`, the cameras of
    CAMERAS when none are given; then the place of each camera of CAMERAS in
    the robot frame for every lookup the brain makes."""
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
    for answer in sensors if sensors is not None else (sensors_described(),):
        h.mocks.deps.sensor_mounts.describe_sensors.enqueue_response(answer)
    _ANSWERERS[h] = asyncio.get_running_loop().create_task(_answer_lookups(h.mocks.deps.transform_lookup.lookup_transform))


def sensors_described(names=tuple(CAMERAS)):
    """The robot's describe_sensors response: each camera of `names` an
    rgbd camera fixed to the base, its frame `<name>_color_optical_frame`."""
    from peppygen.consumed_services.sensor_mounts import describe_sensors

    return describe_sensors.ResponseData(
        robot_frame=ROBOT_FRAME,
        sensor_names=list(names),
        sensor_kinds=["rgbd_camera"] * len(names),
        sensor_frames=[f"{name}_color_optical_frame" for name in names],
        carried_by=[""] * len(names),
    )


async def _answer_lookups(service) -> None:
    """Answers each lookup of a camera's frame in the robot frame with its
    place of CAMERAS, and any other with the failure `unknown_frame`, until
    the harness stops its mocks."""
    from peppygen.consumed_services.transform_lookup import lookup_transform

    places = {frame: (position, orientation) for frame, position, orientation in CAMERAS.values()}
    while True:
        try:
            request, responder = await service.next_request(LOOKUP_WAIT_S)
        except Exception:  # pylint: disable=broad-exception-caught
            # The harness stopped its mocks.
            return
        if request.target_frame != ROBOT_FRAME or request.source_frame not in places:
            await responder.respond_failure(
                lookup_transform.FailureKind.UNKNOWN_FRAME, f"no transform names {request.source_frame}"
            )
            continue
        position, orientation = places[request.source_frame]
        await responder.respond(lookup_transform.ResponseData(position=position, orientation=orientation, timestamp=request.time))


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
