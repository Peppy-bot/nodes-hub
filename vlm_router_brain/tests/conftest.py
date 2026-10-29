"""Shared test pieces: a scripted provider, a catalog of the test's own,
and the records of one logger, so the router's parts run with no network,
no key and no model."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Iterator, Optional

import pytest

from vlm_router_brain.catalog import Catalog, Route, parse_catalog
from vlm_router_brain.provider import Reply

NOTHING = '{"boxes": []}'


@dataclass(frozen=True)
class Call:
    route: Route
    images: tuple[bytes, ...]
    prompt: str
    schema: object
    timeout_s: float

    @property
    def image(self) -> bytes:
        """The one picture of a search or a probe."""
        (image,) = self.images
        return image


class FakeProvider:
    """Answers each call with the next of `replies`, a Reply or an
    exception to raise, and `{"boxes": []}` once they run out; records
    every call. With `gate` set, each call waits until the test opens it;
    `called` resolves once a call has begun."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls: list[Call] = []
        self.gate: Optional[asyncio.Event] = None
        self.called = asyncio.Event()

    async def ask(self, route: Route, images, prompt: str, schema: object, timeout_s: float) -> Reply:
        self.calls.append(Call(route, tuple(images), prompt, schema, timeout_s))
        self.called.set()
        if self.gate is not None:
            await self.gate.wait()
        reply = self.replies.pop(0) if self.replies else Reply(NOTHING)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def answer(*boxes: tuple[str, list[int]]) -> Reply:
    """A reply holding `boxes`, each (label, box_2d)."""
    return Reply(json.dumps({"boxes": [{"label": label, "box_2d": box} for label, box in boxes]}), 100, 20, 5, 1.5)


def catalog_of(*models: dict) -> Catalog:
    return parse_catalog({"models": list(models)})


def model(name: str, *routes: str, floor: float = 10.0) -> dict:
    """A perception model of the catalog with `routes`, each "provider:model"."""
    return {
        "name": name,
        "roles": ["perception"],
        "coordinates": "yx_0_1000",
        "routes": [{"route": route, "min_call_timeout_s": floor, "reasoning_effort": "medium"} for route in routes],
    }


# One model on Google's API, one reached through OpenRouter first and
# Google's API second.
TEST_CATALOG = catalog_of(
    model("fast", "gemini:gemini-fast"),
    model("careful", "openrouter:lab/careful", "gemini:gemini-careful"),
)
BOTH_KEYS = {"GEMINI_API_KEY": "test-gemini-key", "OPENROUTER_API_KEY": "test-openrouter-key"}


@contextmanager
def records_of(caplog, name: str, level: int = logging.INFO) -> Iterator[pytest.LogCaptureFixture]:
    """`caplog` holding the records of the logger `name` from `level` up,
    whatever `configure_logging` set on it."""
    logger = logging.getLogger(name)
    propagated, previous = logger.propagate, logger.level
    logger.propagate = False
    logger.setLevel(level)
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous)
        logger.propagate = propagated


# --- what the aligner's tests need: a limb_states message and a frame, as
# brain_core's own tests build them.

@dataclass
class LimbStatesMessage:
    """A limb_states message, as the generated module gives it."""

    timestamp: float
    arm_names: list[str] = field(default_factory=list)
    joints_per_arm: list[int] = field(default_factory=list)
    joint_positions: list[float] = field(default_factory=list)
    positions: list[float] = field(default_factory=list)
    orientations: list[float] = field(default_factory=list)
    gripper_names: list[str] = field(default_factory=list)
    gripper_openings: list[float] = field(default_factory=list)


# Where each grasp point stands at rest in a snapshot.
LEFT_AT_REST = ((0.4, 0.2, 0.9), (0.0, 1.0, 0.0, 0.0))
RIGHT_AT_REST = ((0.4, -0.2, 0.9), (0.0, 1.0, 0.0, 0.0))


def limb_states_message(timestamp: float, *, left_arm=LEFT_AT_REST, **openings: float) -> LimbStatesMessage:
    """A snapshot of the two-armed robot of seven joints per arm, every
    joint at 0, the left grasp point at `left_arm`, a position and an
    orientation, and the right one at rest, with the grippers `openings`
    names, by name."""
    (left_position, left_orientation), (right_position, right_orientation) = left_arm, RIGHT_AT_REST
    return LimbStatesMessage(
        timestamp=timestamp,
        arm_names=["left_arm", "right_arm"],
        joints_per_arm=[7, 7],
        joint_positions=[0.0] * 14,
        positions=[*left_position, *right_position],
        orientations=[*left_orientation, *right_orientation],
        gripper_names=list(openings),
        gripper_openings=list(openings.values()),
    )


def rgb_frame(width: int = 16, height: int = 12, *, frame_id: int = 1, align_mode: str = "depth_to_color"):
    """A video_stream message of the given size, plain grey."""
    return SimpleNamespace(
        header=SimpleNamespace(timestamp=1.0, frame_id=frame_id, align_mode=align_mode),
        encoding="rgb8",
        width=width,
        height=height,
        frame=bytes([128]) * (width * height * 3),
    )
