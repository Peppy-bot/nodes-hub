"""The node's runtime, faked for the suites that import the engine's transport.

peppylib and the peppygen modules generated from peppy.json5 exist only inside
the node's image, so importing this module installs the smallest module tree
the engine imports, once, ahead of the engine's own imports. Every suite that
needs the runtime imports it, so they all meet the same modules whichever of
them a session collects first.

The pairing modules follow the node's four slots. A slot's pairs are the
slot's, so every topic module of a slot reads the same members, exactly as the
generated `peers()` does. A setpoint module hands the engine the arrivals a
test queued and then ends its subscription, and every publisher records the
peer each payload was addressed to, so nothing here waits on a clock.
"""

from __future__ import annotations

import asyncio
import enum
import sys
import types
from dataclasses import dataclass
from typing import Optional

SLOT_TOPICS = {
    "arms": ("joint_setpoints", "joint_states"),
    "grippers": ("gripper_setpoints", "gripper_states"),
    "rgb_cameras": ("video_stream", "stream_info", "geometry"),
    "rgbd_cameras": ("video_stream", "depth_stream", "stream_info", "geometry"),
}

# The core node every fake instance runs on.
CORE_NODE = "sim16"


@dataclass(frozen=True)
class Producer:
    core_node: str
    instance_id: str


@dataclass(frozen=True)
class Peer:
    """A peer as peppylib reports it: the instance at the other end of a pair,
    and the link the pair comes from on its side."""

    producer: Producer
    peer_link_id: str


@dataclass(frozen=True)
class Member:
    """One pair of a slot: the peer it reaches and the copy that peer runs
    as."""

    info: Peer
    copy: Optional[str]


def member(instance: str, link: str, copy: Optional[str]) -> Member:
    return Member(info=Peer(producer=Producer(CORE_NODE, instance), peer_link_id=link), copy=copy)


class Publisher:
    """A publisher: records each payload, with the peer a pairing publish was
    addressed to."""

    def __init__(self) -> None:
        self.sent: list[tuple[Peer, object]] = []
        self.published: list = []

    async def publish_to(self, peer: Peer, payload) -> None:
        self.sent.append((peer, payload))

    async def publish(self, payload) -> None:
        self.published.append(payload)


class Subscription:
    """A slot's setpoint subscription: hands over what the test queued, in
    order, then ends."""

    def __init__(self, arrivals: list) -> None:
        self._arrivals = arrivals

    async def next(self):
        return self._arrivals.pop(0) if self._arrivals else None


class FailureKind(str, enum.Enum):
    """The failure kinds of attach: the kinds of peppy, as the generated
    module names them."""

    ATTEMPT_FAILED = "attempt_failed"
    TIMEOUT = "timeout"
    PRECONDITION_FAILED = "precondition_failed"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Failure:
    """Why a goal was refused: a kind of attach and a message."""

    kind: FailureKind
    message: str


class _GoalDecision:
    """What admitting a goal answers, as the generated GoalDecision holds it:
    the response of an accepted goal, or the failure of a refused one."""

    def __init__(self, response, refusal: Optional[Failure]) -> None:
        self.response = response
        self.refusal = refusal

    @staticmethod
    def accept(response) -> "_GoalDecision":
        return _GoalDecision(response, None)

    @staticmethod
    def refuse(kind, message: str) -> "_GoalDecision":
        return _GoalDecision(None, Failure(FailureKind(kind), message))


# Every attach action an engine exposed, in the order it exposed them.
ATTACH_HANDLES: list = []


class _AttachHandle:
    """The attach action an engine exposes. It records the sender grace it
    is given, and the sender grace it had when the engine first asked it for
    a goal. Its goal stream is closed, as it is when the node shuts down."""

    def __init__(self) -> None:
        self.sender_grace_s: Optional[float] = None
        self.sender_grace_s_when_served: Optional[float] = None

    @classmethod
    async def expose(cls, _node_runner) -> "_AttachHandle":
        handle = cls()
        ATTACH_HANDLES.append(handle)
        return handle

    async def set_sender_grace(self, seconds: float) -> None:
        self.sender_grace_s = seconds

    async def handle_goal_next_request(self, _admit) -> None:
        self.sender_grace_s_when_served = self.sender_grace_s
        return None


async def _no_readiness_request(_runner, _handler) -> None:
    """The readiness service of an engine nobody asks: it waits for a request
    that never comes."""
    await asyncio.Event().wait()


@dataclass(frozen=True)
class _GoalResponse:
    arm_names: list
    arm_joints: list
    gripper_names: list


@dataclass(frozen=True)
class _ReadyResponse:
    ready: bool


class _ClockPublisher:
    """No deployment names a test's engine the publisher of a clock domain."""

    @staticmethod
    async def for_node(_node_runner) -> None:
        return None


def _topic_module(slot_module: types.ModuleType, slot: str, topic: str) -> types.ModuleType:
    module = types.ModuleType(f"peppygen.paired_topics.{slot}.{topic}")
    module.LINK_ID = slot
    module.arrivals = []
    module.publisher = Publisher()
    module.peers = lambda _runner: list(slot_module.members)
    # A payload is the fields it was built from, so a test reads them back.
    module.build_message = lambda *fields: fields
    module.MessageHeader = lambda **fields: types.SimpleNamespace(**fields)

    async def subscribe(_runner) -> Subscription:
        return Subscription(module.arrivals)

    async def declare_publisher(_runner) -> Publisher:
        return module.publisher

    module.subscribe = subscribe
    module.declare_publisher = declare_publisher
    return module


def _object_states_module() -> types.ModuleType:
    """The scene's own stream of spawned objects, which belongs to no pair."""
    module = types.ModuleType("peppygen.emitted_topics.objects.object_states")
    module.TOPIC_NAME = "object_states"
    module.publisher = Publisher()
    module.build_message = lambda timestamp, objects: (timestamp, objects)
    module.MessageObjectsItem = lambda **fields: fields

    async def declare_publisher(_runner) -> Publisher:
        return module.publisher

    module.declare_publisher = declare_publisher
    return module


def _install() -> None:
    peppylib = types.ModuleType("peppylib")
    peppylib.NodeRunner = object
    peppylib.PeerPublisher = object
    peppylib.TopicPublisher = object
    peppylib_clock = types.ModuleType("peppylib.clock")
    peppylib_clock.ClockPublisher = _ClockPublisher
    peppylib.clock = peppylib_clock

    peppygen = types.ModuleType("peppygen")
    peppygen.__path__ = []
    peppygen_clock = types.ModuleType("peppygen.clock")
    peppygen_clock.now_ns = lambda: 0

    async def init(_runner) -> None:
        return None

    peppygen_clock.init = init
    peppygen.clock = peppygen_clock

    attach = types.ModuleType("peppygen.exposed_actions.robots.attach")
    attach.ActionHandle = _AttachHandle
    attach.FailureKind = FailureKind
    attach.GoalDecision = _GoalDecision
    attach.GoalResponse = _GoalResponse
    is_ready = types.ModuleType("peppygen.exposed_services.robots.is_ready")
    is_ready.Response = _ReadyResponse
    is_ready.handle_next_request = _no_readiness_request
    actions_robots = types.ModuleType("peppygen.exposed_actions.robots")
    actions_robots.attach = attach
    services_robots = types.ModuleType("peppygen.exposed_services.robots")
    services_robots.is_ready = is_ready

    emitted = types.ModuleType("peppygen.emitted_topics")
    emitted_objects = types.ModuleType("peppygen.emitted_topics.objects")
    emitted_objects.object_states = _object_states_module()
    emitted.objects = emitted_objects

    paired = types.ModuleType("peppygen.paired_topics")
    modules = {
        "peppylib": peppylib,
        "peppylib.clock": peppylib_clock,
        "peppygen": peppygen,
        "peppygen.clock": peppygen_clock,
        "peppygen.exposed_actions": types.ModuleType("peppygen.exposed_actions"),
        "peppygen.exposed_actions.robots": actions_robots,
        "peppygen.exposed_services": types.ModuleType("peppygen.exposed_services"),
        "peppygen.exposed_services.robots": services_robots,
        "peppygen.emitted_topics": emitted,
        "peppygen.emitted_topics.objects": emitted_objects,
        "peppygen.emitted_topics.objects.object_states": emitted_objects.object_states,
        "peppygen.paired_topics": paired,
    }
    for slot, topics in SLOT_TOPICS.items():
        slot_module = types.ModuleType(f"peppygen.paired_topics.{slot}")
        slot_module.members = []
        for topic in topics:
            setattr(slot_module, topic, _topic_module(slot_module, slot, topic))
        setattr(paired, slot, slot_module)
        modules[f"peppygen.paired_topics.{slot}"] = slot_module
    sys.modules.update(modules)


_install()


def _slot(slot: str) -> types.ModuleType:
    return sys.modules[f"peppygen.paired_topics.{slot}"]


def topic(slot: str, name: str) -> types.ModuleType:
    """The generated module of one topic of a slot."""
    return getattr(_slot(slot), name)


def pair(slot: str, *members: Member) -> None:
    """Gives one slot the pairs it holds."""
    _slot(slot).members = list(members)


def reset() -> None:
    """Empties every slot, its queued arrivals and what it published."""
    for slot, topics in SLOT_TOPICS.items():
        pair(slot)
        for name in topics:
            module = topic(slot, name)
            module.arrivals = []
            module.publisher = Publisher()
