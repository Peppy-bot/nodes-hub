"""Shared test pieces: a scripted provider, a catalog of the test's own,
the records of one logger, and no OpenAI-compatible API from the shell, so
the router's parts run with no network, no key and no model. A test that
calls an API calls the fake one of fake_api.py, on 127.0.0.1, with a key
of the test's own (`through_api`)."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Iterator, Optional

import pytest

from fake_api import FakeApi

from vlm_router_brain import provider as provider_module
from vlm_router_brain.catalog import Catalog, Route, load_catalog, parse_catalog
from vlm_router_brain.openai_compatible import API_BASE_VARIABLE, API_KEY_VARIABLE
from vlm_router_brain.provider import Reply

NOTHING = '{"boxes": []}'


@pytest.fixture(autouse=True)
def no_api_from_the_shell(monkeypatch):
    """No test reads an OpenAI-compatible API from the shell that runs the
    suite: a test that wants one sets it."""
    monkeypatch.delenv(API_BASE_VARIABLE, raising=False)
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)


# The key the tests give the fake API: it passes peppy's value filter and
# is no real key.
API_KEY = "sentinel-api-key-0451"
# The provider keys that the node or a library could read on its own.
PROVIDER_KEY_VARIABLES = ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")


def sentinel_of(variable: str) -> str:
    """The value a test gives the provider key `variable`, which no request
    to the fake API may carry."""
    return f"sentinel-{variable.lower().replace('_', '-')}"


@pytest.fixture(scope="module")
def module_api() -> Iterator[FakeApi]:
    """One fake OpenAI-compatible API for the whole test module."""
    api = FakeApi()
    yield api
    api.close()


@pytest.fixture
def fake_api(module_api: FakeApi) -> FakeApi:
    """The module's fake API, serving the shipped catalog's models, with no
    request and no answer left by an earlier test."""
    module_api.reset(entry.name for entry in load_catalog().entries)
    return module_api


@dataclass
class ThroughApi:
    api: FakeApi
    #: The provider of each provider client the node made: none may be.
    provider_clients: list[str]


@pytest.fixture
def through_api(monkeypatch, fake_api: FakeApi) -> ThroughApi:
    """The node's environment names the fake API, with API_KEY, and holds a
    sentinel in each provider key; 127.0.0.1 goes through no proxy.
    AnyLLM.create, which makes a provider's client, records each call and
    fails it."""
    monkeypatch.setenv(API_BASE_VARIABLE, fake_api.base)
    monkeypatch.setenv(API_KEY_VARIABLE, API_KEY)
    for variable in PROVIDER_KEY_VARIABLES:
        monkeypatch.setenv(variable, sentinel_of(variable))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    made: list[str] = []

    def create(provider, **kwargs):
        made.append(provider)
        raise RuntimeError(f"a client of the provider {provider} was made")

    monkeypatch.setattr(provider_module.AnyLLM, "create", staticmethod(create))
    return ThroughApi(fake_api, made)


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
class GraspPoint:
    """A grasp-point pose of a limb_states message."""

    position: list[float]
    orientation: list[float]


@dataclass
class LimbStatesMessage:
    """A limb_states message, as the generated module gives it."""

    timestamp: float
    frame: str = "base_link"
    arm_names: list[str] = field(default_factory=list)
    joints_per_arm: list[int] = field(default_factory=list)
    joint_positions: list[float] = field(default_factory=list)
    grasp_points: list[GraspPoint] = field(default_factory=list)
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
        grasp_points=[GraspPoint(list(left_position), list(left_orientation)), GraspPoint(list(right_position), list(right_orientation))],
        gripper_names=list(openings),
        gripper_openings=list(openings.values()),
    )


def rgb_frame(
    width: int = 16, height: int = 12, *, capture_index: int = 1, align_mode: str = "depth_to_color", frame: str = "chest_color_optical_frame"
):
    """A video_stream message of the given size, plain grey, that names
    `frame`."""
    return SimpleNamespace(
        header=SimpleNamespace(timestamp=1.0, frame=frame, capture_index=capture_index, align_mode=align_mode),
        encoding="rgb8",
        width=width,
        height=height,
        data=bytes([128]) * (width * height * 3),
    )
