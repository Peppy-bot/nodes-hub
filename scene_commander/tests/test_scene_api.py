"""Drive the commander's HTTP API against a fake scene provider, without Peppy."""

import asyncio
import collections
import enum
import importlib.util
import ipaddress
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pyjson5
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

_NODE = Path(__file__).resolve().parents[1]
_SOURCE = _NODE / "src" / "scene_commander" / "__main__.py"
_MANIFEST = _NODE / "peppy.json5"
_ACTIONS = ("apply_force", "clear_scene", "load_scene", "move_object", "move_robot", "remove_object", "spawn_object")
# The services of every slot a launch may leave vacant, by link id.
_OPTIONAL_LINKS = {
    "lighting": (
        "get_lighting", "set_light_intensity", "set_light_color", "set_light_position", "set_light_direction",
        "set_light_cone", "set_light_orientation", "reset_lighting",
    ),
    "materials": ("get_materials", "set_material_color", "set_material_finish", "reset_materials"),
    "cameras": (
        "get_cameras", "describe_camera", "set_camera_exposure", "set_camera_white_balance", "set_camera_gain",
        "set_camera_brightness", "set_camera_contrast", "reset_camera",
    ),
}
_BUSY = "Isaac is still discovering its asset catalogue"
_NOT_READY = "Isaac has not loaded its stage yet"
_NO_SCENE = "no scene is loaded"
_NOT_ATTACHED = "the camera is not attached to its simulation yet"
_SCENE = {"asset_id": "scene/full_warehouse", "display_name": "Full Warehouse", "kind": "scene", "category": "Scenes"}
# A scene whose load runs long enough to report progress.
_FACTORY = "scene/factory_conveyor"
_LOAD = {"asset_id": _FACTORY, "scale": 1.0}
_LOAD_SUMMARY = f"load_scene(asset_id={_FACTORY}, scale=1.0)"
_ROBOTS = [
    {"robot": "alpha", "model": "openarm_v2", "position": [0.0, 0.0, 0.0], "yaw": 0.0, "attached": True},
    {"robot": "bravo", "model": "openarm_v1", "position": [0.0, -1.5, 0.0], "yaw": 1.25, "attached": True},
]

# A snapshot's capture time, as the generated binding decodes it: epoch seconds.
_CAPTURED = 1_757_944_800.25
# The simulation instance serving the scene, its lighting and its materials.
_SIMULATION = ("alpha", "alpha_isaac")
_SUN = {
    "id": "sun", "kind": "directional", "label": "Sun",
    "properties": {
        "illuminance_lux": {"unit": "lx", "min": 0.0, "max": 200000.0, "default": 1000.0, "value": 1000.0},
        "color_rgb": {"unit": "linear rgb", "default": [1.0, 1.0, 1.0], "value": [1.0, 0.95, 0.9]},
        "direction": {"unit": "unit vector", "default": [0.0, 0.0, -1.0], "value": [0.3, 0.0, -0.95]},
    },
}
_STEEL = {
    "id": "steel", "label": "Brushed steel",
    "scope": {"owner": "scene", "surfaces": 3, "bodies": ["shelf_1", "shelf_2"]},
    "properties": {
        "color_rgb": {"unit": "linear rgb", "default": [0.6, 0.6, 0.6], "value": [0.6, 0.6, 0.6]},
        "metallic": {"unit": "ratio", "min": 0.0, "max": 1.0, "default": 1.0, "value": 1.0},
        "roughness": {"unit": "ratio", "min": 0.0, "max": 1.0, "default": 0.3, "value": 0.3},
    },
}
_PROFILE = {
    "device": "c920", "label": "Logitech C920", "encoding": "mjpeg",
    "reference": {"source": "v4l2 on 2026-09-01", "captured_on": "2026-09-01"},
    "controls": {
        "exposure": {
            "supported": True, "modes": ["auto", "manual"], "unit": "us", "min": 100, "max": 33000,
            "default_mode": "auto", "default_value": 8000, "mode": "auto", "value": 8300,
        },
        "gain": {"supported": False},
    },
}


# A scenario that hangs fails its test instead of hanging the run; no
# assertion reads it.
LIVENESS_TIMEOUT_S = 10


class Status(enum.IntEnum):
    """How a goal ended, as the generated ResultStatus names it."""

    COMPLETED = 0
    CANCELLED = 1
    ABANDONED = 2
    EXPIRED = 3


class CancelState(enum.IntEnum):
    """How the simulator took a cancel, as the generated CancelState names it."""

    SIGNALLED = 0
    ALREADY_TERMINAL = 1
    UNKNOWN = 2


class FakeClock:
    """Virtual time, which only the test moves on.

    It stands in for the commander's silence timer and for the time a result
    wait may take. Every event of the commander's and the simulator's is
    stamped in order, so a test can wait until the commander waits again.
    """

    def __init__(self):
        self.now = 0.0
        # The stamp of the latest event.
        self.latest = 0
        self._sleepers = []
        self._next_change = asyncio.Event()

    def stamp(self):
        """Stamp one event and wake whoever waits for a state."""
        self.latest += 1
        self.wake()
        return self.latest

    def wake(self):
        """Wake whoever waits for a state, so each checks it again."""
        self._next_change.set()
        self._next_change = asyncio.Event()

    async def next_change(self):
        await self._next_change.wait()

    async def sleep(self, seconds):
        sleeper = SimpleNamespace(
            deadline=self.now + seconds, armed_at=self.stamp(), woken=asyncio.get_running_loop().create_future(),
        )
        self._sleepers.append(sleeper)
        try:
            await sleeper.woken
        finally:
            if sleeper in self._sleepers:
                self._sleepers.remove(sleeper)

    def advance(self, seconds):
        """Move time on, ending every sleep it reaches."""
        self.now += seconds
        for sleeper in [sleeper for sleeper in self._sleepers if sleeper.deadline <= self.now]:
            self._sleepers.remove(sleeper)
            if not sleeper.woken.done():
                sleeper.woken.set_result(None)
        self.stamp()

    def armed_after(self, stamp):
        """Whether a sleep still running began after the event stamped `stamp`."""
        return any(sleeper.armed_at > stamp and not sleeper.woken.done() for sleeper in self._sleepers)

    async def until(self, state):
        while not state():
            await self.next_change()


class FakeGoal:
    """One goal as the simulator runs it, and the handle the commander holds on it.

    The test plays the simulator: it reports progress and ends the goal. The
    feedback stream raises RuntimeError at its end and ConnectionError when the
    simulator is gone, as the binding does. A result wait answers once the goal
    has ended and raises TimeoutError once the clock passes its timeout, as the
    binding does; for a simulator that is gone it answers ABANDONED then. A
    cancel answers the next reply the test queued in cancel_replies, a
    CancelState or an exception it raises, and SIGNALLED when none is queued.
    """

    def __init__(self, name, clock, accepted, reason):
        self.name = name
        self.accepted = accepted
        self.reason = reason
        # The timeout of every cancel and every result wait the commander asked for.
        self.cancels = []
        self.result_waits = []
        self.cancel_replies = collections.deque()
        self._clock = clock
        self._feedback = collections.deque()
        self._result = None
        self._gone = False
        self._reading = False
        self._pushed = 0
        self._taken = 0
        self._taken_at = 0
        self._cancelled_at = 0
        self._waiting_for_result = False
        # The latest event a result wait has checked the time and the result against.
        self._result_checked_at = 0

    def report(self, bytes_fetched, files_ready, building=False):
        self.report_message(SimpleNamespace(bytes_fetched=bytes_fetched, files_ready=files_ready, building=building))

    def report_message(self, message):
        """Send one feedback message, whatever its shape."""
        self._push(message)

    def end(self, status, data):
        """End the goal: its result stands, then its feedback stream ends."""
        self.end_without_stream_end(status, data)
        self.end_without_result()

    def end_without_stream_end(self, status, data):
        """End the goal and lose the message that ends its feedback stream."""
        self._result = SimpleNamespace(status=status, data=data)
        self._clock.stamp()

    def end_without_result(self):
        """End the feedback stream of a goal whose result never comes."""
        self._push(RuntimeError(f"action '{self.name}' feedback stream closed"))

    def complete(self, message, **fields):
        self.end(Status.COMPLETED, SimpleNamespace(success=True, message=message, **fields))

    def vanish(self):
        """The simulator's process is gone: the stream breaks without an end."""
        self._gone = True
        self._push(ConnectionError(f"action '{self.name}' producer is gone"))

    async def settled(self):
        """Wait until the commander waits on this goal again, having seen
        every event so far: its result wait is over or has checked the latest
        event, or it took every message and waits for the next one under a
        silence window it armed since."""
        await self._clock.until(lambda: (
            (self.result_waits and not self._waiting_for_result)
            or (self._waiting_for_result and self._result_checked_at == self._clock.latest)
            or (
                self._taken == self._pushed
                and self._reading
                and self._clock.armed_after(max(self._taken_at, self._cancelled_at))
            )
        ))

    def _push(self, item):
        self._feedback.append(item)
        self._pushed += 1
        self._clock.stamp()

    async def on_next_feedback_message(self):
        self._reading = True
        self._clock.stamp()
        try:
            await self._clock.until(lambda: self._feedback)
        finally:
            self._reading = False
        item = self._feedback.popleft()
        self._taken += 1
        self._taken_at = self._clock.stamp()
        if isinstance(item, Exception):
            raise item
        return item

    async def get_result(self, timeout):
        self.result_waits.append(timeout)
        deadline = self._clock.now + timeout
        self._waiting_for_result = True
        try:
            while self._result is None and self._clock.now < deadline:
                # Tell a settling test this wait has seen the latest event;
                # waking only on news keeps two waits from waking each other.
                if self._result_checked_at != self._clock.latest:
                    self._result_checked_at = self._clock.latest
                    self._clock.wake()
                await self._clock.next_change()
        finally:
            self._waiting_for_result = False
            self._clock.stamp()
        if self._result is not None:
            return self._result
        if self._gone:
            return SimpleNamespace(status=Status.ABANDONED, data=None)
        raise TimeoutError(f"action '{self.name}' (alpha_waldo) has timed out waiting for result")

    async def cancel_goal(self, timeout):
        self.cancels.append(timeout)
        self._cancelled_at = self._clock.stamp()
        reply = self.cancel_replies.popleft() if self.cancel_replies else CancelState.SIGNALLED
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(state=reply)


@dataclass(frozen=True)
class FakeProducer:
    """A ProducerRef: the instance serving a slot, equal to any other ref of that instance."""

    core_node: str
    instance_id: str


class FakeService(ModuleType):
    """A consumed service: poll() answers the next queued response, the last one forever.

    Its producers are its link's, shared by every service of that link; every
    request polled is kept with the producer it went to.
    """

    def __init__(self, link, name):
        super().__init__(f"{link.__name__}.{name}")
        self.link = link
        self.answers = []
        self.requests = []
        self.Request = lambda **fields: SimpleNamespace(**fields)

    def bound_producer(self, node_runner):
        return self.link.producers[0] if self.link.producers else None

    def bound_producers(self, node_runner):
        return list(self.link.producers)

    async def poll(self, node_runner, producer, *request, timeout):
        assert producer in self.link.producers, f"{self.__name__} polled on {producer!r}, which its slot does not bind"
        self.requests.append((producer, request[0] if request else None))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(data=answer)


def _link(name, services, producers=()):
    """One consumed link's module: its services, bound to `producers`."""
    link = ModuleType(f"peppygen.consumed_services.{name}")
    link.producers = list(producers)
    for service in services:
        setattr(link, service, FakeService(link, service))
    return link


class FakeAction(ModuleType):
    """A consumed action: fire_goal() records the goal and hands out a FakeGoal.

    A goal ends with `result` as soon as it is fired, unless the test lets it
    run (ends_at_once false) and plays the simulator through it. A test that
    sets `admission` to an event holds every goal fired at admission until it
    sets the event.
    """

    def __init__(self, name, clock):
        super().__init__(f"peppygen.consumed_actions.simulation.{name}")
        self.ResultStatus = Status
        self.CancelState = CancelState
        self.GoalRequest = lambda **fields: SimpleNamespace(**fields)
        # Every goal request fired, and the admission timeout of each.
        self.goals = []
        self.admissions = []
        self.handles = []
        self.accepted = True
        self.reason = ""
        self.ends_at_once = True
        self.admission = None
        self.result = SimpleNamespace(
            status=Status.COMPLETED,
            data=SimpleNamespace(success=True, message=f"{name} done", object_id="obj_1"),
        )
        self._clock = clock
        action = self

        class ActionHandle:
            @staticmethod
            async def fire_goal(node_runner, producer, *goal, timeout, feedback_qos):
                assert feedback_qos == "standard"
                action.goals.append(goal[0] if goal else None)
                action.admissions.append(timeout)
                clock.stamp()
                if action.admission is not None:
                    await action.admission.wait()
                handle = FakeGoal(name, clock, action.accepted, action.reason)
                if action.ends_at_once:
                    handle.end(action.result.status, action.result.data)
                action.handles.append(handle)
                clock.stamp()
                return handle

        self.ActionHandle = ActionHandle

    def bound_producer(self, node_runner):
        return "producer"

    async def fired(self, index=0):
        """The goal fired `index`-th, once the commander has fired it."""
        await self._clock.until(lambda: len(self.handles) > index)
        return self.handles[index]


@pytest.fixture
def commander(monkeypatch):
    services = _link("simulation", ["get_assets_list", "get_robots_list"], producers=["producer"])
    services.get_assets_list.answers = [_catalogue(_SCENE)]
    services.get_robots_list.answers = [SimpleNamespace(
        success=True,
        message="2 robots standing",
        robots=[SimpleNamespace(**robot) for robot in _ROBOTS],
    )]
    objects = _link("objects", ["get_object_states"], producers=["producer"])
    objects.get_object_states.answers = [_snapshot()]
    # The optional links start vacant; a test binds what its launch has.
    optional = {name: _link(name, members) for name, members in _OPTIONAL_LINKS.items()}
    for name, members in _OPTIONAL_LINKS.items():
        for member in members:
            getattr(optional[name], member).answers = [_done(member)]
    optional["lighting"].get_lighting.answers = [_lighting(_SUN)]
    optional["materials"].get_materials.answers = [_materials(_STEEL)]
    optional["cameras"].get_cameras.answers = [_cameras(
        _listed("alpha", "wrist_left", "rgb", 1280, 720, 30), _listed("bravo", "chest", "rgbd", 640, 480, 15),
    )]
    optional["cameras"].describe_camera.answers = [_profile()]
    clock = FakeClock()
    actions = ModuleType("peppygen.consumed_actions.simulation")
    for name in _ACTIONS:
        setattr(actions, name, FakeAction(name, clock))
    peppygen = ModuleType("peppygen")
    peppygen.NodeBuilder = Mock()
    peppygen.NodeRunner = object
    peppylib = ModuleType("peppylib")
    peppylib.QoSProfile = SimpleNamespace(Standard="standard")
    for name, module in {
        "peppylib": peppylib,
        "peppygen": peppygen,
        "peppygen.consumed_actions": ModuleType("peppygen.consumed_actions"),
        "peppygen.consumed_actions.simulation": actions,
        "peppygen.consumed_services": ModuleType("peppygen.consumed_services"),
        "peppygen.consumed_services.simulation": services,
        "peppygen.consumed_services.objects": objects,
        **{link.__name__: link for link in optional.values()},
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_scene_commander_under_test", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    def bind_lighting():
        optional["lighting"].producers.append(FakeProducer(*_SIMULATION))

    def bind_materials():
        optional["materials"].producers.append(FakeProducer(*_SIMULATION))

    def bind_cameras():
        """Link the simulation's cameras here, the way it links its lighting and materials."""
        producer = FakeProducer(*_SIMULATION)
        optional["cameras"].producers.append(producer)
        return producer

    return SimpleNamespace(
        module=module, services=services, objects=objects, actions=actions, clock=clock,
        bind_lighting=bind_lighting, bind_materials=bind_materials, bind_cameras=bind_cameras, **optional,
    )


class FakeNodeRunner:
    """The runner's endpoint announcements, held to the rules peppylib holds
    them to: each announced label is one the manifest declares, announced once
    with an IP literal, and the seal that ends setup finds every declared label
    announced.
    """

    def __init__(self):
        self.declared = pyjson5.loads(_MANIFEST.read_text())["execution"].get("endpoints", {})
        self.announced = []

    def announce_endpoint(self, label, scheme, host, port, path=""):
        if label not in self.declared:
            raise ValueError(f"endpoint `{label}` is not declared in the manifest")

        if label in self._announced_labels():
            raise ValueError(f"endpoint `{label}` is already announced")

        try:
            ipaddress.ip_address(host)

        except ValueError as error:
            raise ValueError(f"endpoint `{label}` host `{host}` is not an IP literal") from error

        self.announced.append((label, scheme, host, port, path))

    def seal_endpoints(self):
        unannounced = sorted(self.declared.keys() - self._announced_labels())

        if unannounced:
            raise ValueError(f"endpoints {unannounced} are declared but setup returned without announcing them")

    def _announced_labels(self):
        return {label for label, *_ in self.announced}


@pytest.fixture
def node_runner():
    return FakeNodeRunner()


class _FakeListener:
    """A bound socket, for the wiring around it; the real one is in test_http_port."""

    def __init__(self, host, port):
        self._name = (host, port)

    def getsockname(self):
        return self._name


async def _serve_nothing(app, listener):
    """A server that stops at once, for the setups that do not read what it serves."""
    return asyncio.create_task(asyncio.sleep(0))


def _parameters(http_port, http_host="0.0.0.0"):
    return SimpleNamespace(http_host=http_host, http_port=http_port)


def _busy():
    return SimpleNamespace(success=False, message=_BUSY, assets_json="[]")


def _catalogue(*assets):
    return SimpleNamespace(success=True, message=f"{len(assets)} assets available", assets_json=json.dumps(list(assets)))


def _record(object_id, asset_id, physics, mass, scale, position, orientation, linear_velocity, angular_velocity):
    """One spawned object, with the field names of the generated response item."""
    return SimpleNamespace(
        object_id=object_id, asset_id=asset_id, physics=physics, mass=mass, scale=scale, position=position,
        orientation=orientation, linear_velocity=linear_velocity, angular_velocity=angular_velocity,
    )


def _snapshot(*records):
    return SimpleNamespace(
        success=True, message=f"{len(records)} objects", timestamp=_CAPTURED, objects=list(records),
    )


def _no_object_state():
    # An unavailable answer carries no snapshot: zero time, no objects.
    return SimpleNamespace(success=False, message=_NOT_READY, timestamp=0.0, objects=[])


def _done(name):
    """A setter's answer when a test does not queue its own: success, no effective values."""
    return SimpleNamespace(success=True, message=f"{name} done")


def _set(message, **current):
    return SimpleNamespace(success=True, message=message, **current)


def _refused(message, **current):
    return SimpleNamespace(success=False, message=message, **current)


def _lighting(*lights):
    return SimpleNamespace(success=True, message=f"{len(lights)} lights", lighting_json=json.dumps(_lighting_json(*lights)))


def _lighting_json(*lights):
    return {"scene": _SCENE["asset_id"], "ambient_lux": 120.0, "fill_lux": 40.0, "lights": list(lights)}


def _no_lighting():
    return SimpleNamespace(success=False, message=_NO_SCENE, lighting_json="")


def _materials(*materials):
    return SimpleNamespace(
        success=True, message=f"{len(materials)} materials", materials_json=json.dumps({"materials": list(materials)}),
    )


def _listed(robot, camera, kind, width, height, frames_per_second):
    """One camera as the simulation lists it."""
    return SimpleNamespace(
        robot=robot, camera=camera, kind=kind, width=width, height=height,
        frames_per_second=frames_per_second, encoding="rgb8",
    )


def _cameras(*listed):
    return SimpleNamespace(success=True, message=f"{len(listed)} cameras rendered", cameras=list(listed))


def _profile():
    return SimpleNamespace(success=True, message=f"profile of {_PROFILE['label']}", profile_json=json.dumps(_PROFILE))


def _no_profile():
    return SimpleNamespace(success=False, message=_NOT_ATTACHED, profile_json="")


def _requests(commander, *links):
    """Every request the services of `links` were polled with, by link and service name."""
    return {
        f"{link.__name__.rsplit('.', 1)[-1]}.{name}": getattr(link, name).requests
        for link in links
        for name in _OPTIONAL_LINKS[link.__name__.rsplit(".", 1)[-1]]
        if getattr(link, name).requests
    }


def _node_log(commander, caplog):
    return [(r.levelno, r.getMessage(), r.exc_info) for r in caplog.records if r.name == commander.module.logger.name]


class _PanelServer(TestServer):
    """The test server, run as the node runs its panel: a page that goes
    away leaves its handler running, and the handler finds out when it
    writes to it."""

    async def _make_runner(self, handler_cancellation, **options):
        return web.AppRunner(self.app, access_log=None, **options)


def _app(commander):
    """The commander's app, its silence timer on the test's clock."""
    return commander.module._build_app(object(), silence=commander.clock.sleep)


def _call(commander, scenario, app=None):
    async def run():
        async with asyncio.timeout(LIVENESS_TIMEOUT_S):
            async with TestClient(_PanelServer(app or _app(commander))) as client:
                return await scenario(client)

    return asyncio.run(run())


async def _get(client, path):
    response = await client.get(path)
    return response.status, await response.json()


async def _post(client, path, payload):
    response = await client.post(path, json=payload)
    return response.status, await response.json()


async def _stream(client, path, payload):
    """The status, the content type and the lines of an answer read to its
    end: each JSON line of a stream, or the one JSON body of a plain answer."""
    response = await client.post(path, json=payload)
    lines = [json.loads(line) for line in (await response.text()).splitlines() if line]
    return response.status, response.content_type, lines


def _handlers_returned(commander, app):
    """The paths whose handlers returned, in order, each stamped on the clock."""
    returned = []

    @web.middleware
    async def record(request, handler):
        try:
            return await handler(request)
        finally:
            returned.append(request.path)
            commander.clock.stamp()

    app.middlewares.append(record)
    return returned


def _transports(commander, app):
    """The connection of each request, kept as its handler starts, so a test
    can close the page's end of it."""
    transports = []

    @web.middleware
    async def keep(request, handler):
        transports.append(request.transport)
        commander.clock.stamp()
        return await handler(request)

    app.middlewares.append(keep)
    return transports


def _page_that_closes(commander):
    """The commander's app, the connection of each request, which the test
    closes as a page that goes away, and the paths whose handlers returned."""
    app = _app(commander)
    return app, _transports(commander, app), _handlers_returned(commander, app)


async def _load_handler_returned(commander, returned, answer):
    """Wait until the handler of the closed page's load has returned, and
    let its request end."""
    await commander.clock.until(lambda: "/api/scene/load" in returned)
    await asyncio.gather(answer, return_exceptions=True)


def test_assets_answer_503_with_the_provider_reason_until_the_catalogue_arrives(commander, caplog):
    caplog.set_level(logging.INFO)
    commander.services.get_assets_list.answers = [_busy(), _busy(), _catalogue(_SCENE)]

    async def scenario(client):
        return [await _get(client, "/api/assets") for _ in range(3)]

    first, second, third = _call(commander, scenario)
    assert first == (503, {"success": False, "message": _BUSY})
    assert second == first
    assert third == (200, {"success": True, "assets": [_SCENE], "count": 1})
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"Scene provider has no catalogue: {_BUSY}", None),
        (logging.INFO, "Scene provider catalogue ready: 1 assets", None),
    ]


def test_an_empty_scene_is_a_snapshot_with_no_objects_and_its_capture_time(commander, caplog):
    caplog.set_level(logging.INFO)

    assert _call(commander, lambda client: _get(client, "/api/objects")) == (
        200, {"success": True, "objects": [], "count": 0, "timestamp": _CAPTURED},
    )
    assert _node_log(commander, caplog) == [
        (logging.INFO, "Scene provider object state ready: 0 runtime objects", None),
    ]


def test_objects_answer_503_with_the_provider_reason_while_it_has_no_object_state(commander, caplog):
    caplog.set_level(logging.INFO)
    commander.objects.get_object_states.answers = [_no_object_state(), _no_object_state(), _snapshot()]

    async def scenario(client):
        return [await _get(client, "/api/objects") for _ in range(3)]

    first, second, third = _call(commander, scenario)
    assert first == (503, {"success": False, "message": _NOT_READY})
    assert second == first
    assert third == (200, {"success": True, "objects": [], "count": 0, "timestamp": _CAPTURED})
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"Scene provider has no object state: {_NOT_READY}", None),
        (logging.INFO, "Scene provider object state ready: 0 runtime objects", None),
    ]


def test_object_records_reach_the_page_with_every_field_as_spawned(commander):
    commander.objects.get_object_states.answers = [_snapshot(
        _record("obj_1", "props/blocks/red_block", "dynamic", 0.2, 1.5,
                [0.5, 0.0, 0.81], [0.0, 0.0, 0.38, 0.92], [0.1, 0.0, -0.2], [0.0, 1.5, 0.0]),
        _record("obj_2", "object/warehouse_cage", "static", 40.0, 1.0,
                [2.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        _record("obj_3", "object/warehouse_carton", "none", 2.0, 0.5,
                [1.0, -1.0, 0.4], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    )]

    status, body = _call(commander, lambda client: _get(client, "/api/objects"))

    assert status == 200
    assert body == {
        "success": True,
        "count": 3,
        "timestamp": _CAPTURED,
        "objects": [
            {
                "object_id": "obj_1", "asset_id": "props/blocks/red_block", "physics": "dynamic",
                "mass": 0.2, "scale": 1.5, "position": [0.5, 0.0, 0.81], "orientation": [0.0, 0.0, 0.38, 0.92],
                "linear_velocity": [0.1, 0.0, -0.2], "angular_velocity": [0.0, 1.5, 0.0],
            },
            {
                "object_id": "obj_2", "asset_id": "object/warehouse_cage", "physics": "static",
                "mass": 40.0, "scale": 1.0, "position": [2.0, 1.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0],
                "linear_velocity": [0.0, 0.0, 0.0], "angular_velocity": [0.0, 0.0, 0.0],
            },
            {
                "object_id": "obj_3", "asset_id": "object/warehouse_carton", "physics": "none",
                "mass": 2.0, "scale": 0.5, "position": [1.0, -1.0, 0.4], "orientation": [0.0, 0.0, 0.0, 1.0],
                "linear_velocity": [0.0, 0.0, 0.0], "angular_velocity": [0.0, 0.0, 0.0],
            },
        ],
    }


@pytest.mark.parametrize(("path", "service"), [
    ("/api/assets", lambda commander: commander.services.get_assets_list),
    ("/api/objects", lambda commander: commander.objects.get_object_states),
], ids=["assets", "objects"])
def test_provider_transport_failures_are_server_errors_logged_in_one_line(commander, caplog, path, service):
    caplog.set_level(logging.INFO)
    service(commander).answers = [TimeoutError("the service timed out")]

    assert _call(commander, lambda client: _get(client, path)) == (
        500, {"success": False, "message": "the service timed out"},
    )
    assert _node_log(commander, caplog) == [
        (logging.WARNING, f"GET {path} failed: the service timed out", None),
    ]


def test_load_scene_fires_the_goal_and_streams_the_provider_answer(commander, caplog):
    caplog.set_level(logging.INFO)
    load = commander.actions.load_scene
    load.result.data.message = "Loaded scene scene/full_warehouse"

    status, content_type, lines = _call(commander, lambda client: _stream(
        client, "/api/scene/load", {"asset_id": "scene/full_warehouse", "scale": 1.5},
    ))

    assert (status, content_type) == (200, "application/x-ndjson")
    assert lines == [{"goal": "1"}, {"success": True, "message": "Loaded scene scene/full_warehouse"}]
    assert load.goals == [SimpleNamespace(asset_id="scene/full_warehouse", scale=1.5)]
    # Admission keeps its whole bound; the result is asked for once the
    # stream has ended, when the simulator answers it at once.
    assert load.admissions == [60.0]
    assert load.handles[0].result_waits == [10.0]
    assert _node_log(commander, caplog) == [
        (logging.INFO, "load_scene(asset_id=scene/full_warehouse, scale=1.5): Loaded scene scene/full_warehouse", None),
    ]


def test_a_rejected_load_answers_400_with_the_reason_and_opens_no_stream(commander, caplog):
    caplog.set_level(logging.INFO)
    load = commander.actions.load_scene
    load.accepted = False
    load.reason = "the simulator is shutting down"

    assert _call(commander, lambda client: _stream(client, "/api/scene/load", _LOAD)) == (
        400, "application/json", [{"success": False, "message": "load_scene rejected: the simulator is shutting down"}],
    )
    assert _node_log(commander, caplog) == [
        (logging.WARNING, "POST /api/scene/load failed: load_scene rejected: the simulator is shutting down", None),
    ]


@pytest.mark.parametrize(("payload", "message"), [
    ({"scale": 1.0}, "asset_id must be a non-empty string"),
    ({"asset_id": "", "scale": 1.0}, "asset_id must be a non-empty string"),
    ({"asset_id": 7, "scale": 1.0}, "asset_id must be a non-empty string"),
    ({"asset_id": _FACTORY, "scale": "large"}, "scale must be a finite number"),
    ({"asset_id": _FACTORY, "scale": True}, "scale must be a finite number"),
])
def test_a_load_naming_no_scene_or_no_scale_is_refused_before_the_simulator(commander, payload, message):
    assert _call(commander, lambda client: _stream(client, "/api/scene/load", payload)) == (
        400, "application/json", [{"success": False, "message": message}],
    )
    assert commander.actions.load_scene.goals == []


def test_a_load_naming_no_scale_loads_the_scene_as_authored(commander):
    status, _, _ = _call(commander, lambda client: _stream(client, "/api/scene/load", {"asset_id": _FACTORY}))

    assert status == 200
    assert commander.actions.load_scene.goals == [SimpleNamespace(asset_id=_FACTORY, scale=1.0)]


def _abandon(action):
    action.result.status = Status.ABANDONED


def _fail(action):
    action.result.data = SimpleNamespace(success=False, message="Unknown asset_id: scene/none")


def _no_data(action):
    action.result.data = None


@pytest.mark.parametrize(("outcome", "message"), [
    (_abandon, "load_scene did not complete: ABANDONED"),
    (_fail, "Unknown asset_id: scene/none"),
    (_no_data, "load_scene completed without result data"),
], ids=["abandoned", "failed", "no-data"])
def test_a_load_that_fails_once_accepted_ends_its_stream_with_the_reason(commander, caplog, outcome, message):
    caplog.set_level(logging.INFO)
    outcome(commander.actions.load_scene)

    assert _call(commander, lambda client: _stream(
        client, "/api/scene/load", {"asset_id": "scene/none", "scale": 1.0},
    )) == (200, "application/x-ndjson", [{"goal": "1"}, {"success": False, "message": message}])
    assert _node_log(commander, caplog) == [
        (logging.WARNING, f"POST /api/scene/load failed: {message}", None),
    ]


def test_spawn_object_reports_the_minted_object_id(commander, caplog):
    caplog.set_level(logging.INFO)
    spawn = commander.actions.spawn_object
    spawn.result.data.object_id = "obj_42"

    status, _, lines = _call(commander, lambda client: _stream(client, "/api/objects/spawn", {
        "asset_id": "props/blocks/red_block", "position": [0.5, 0, 0.8], "physics": "dynamic", "mass": 0.2,
    }))

    assert (status, lines) == (
        200, [{"goal": "1"}, {"success": True, "message": "spawn_object done", "object_id": "obj_42"}],
    )
    # A spawn naming no yaw stands as authored; the panel turns an object by
    # its yaw alone and names no orientation.
    assert spawn.goals == [SimpleNamespace(
        asset_id="props/blocks/red_block", position=[0.5, 0.0, 0.8], yaw=0.0, orientation=None, scale=1.0,
        physics="dynamic", mass=0.2,
    )]
    assert _node_log(commander, caplog) == [(
        logging.INFO,
        "spawn_object(asset_id=props/blocks/red_block, position=[0.5, 0.0, 0.8], yaw=0.0, physics=dynamic, "
        "mass=0.2): spawn_object done",
        None,
    )]


def test_a_spawn_carries_the_yaw_it_names(commander):
    status, _, _ = _call(commander, lambda client: _stream(client, "/api/objects/spawn", {
        "asset_id": "props/furniture/desk", "position": [1.0, 0.0, 0.0], "yaw": -1.5,
    }))

    assert status == 200
    assert commander.actions.spawn_object.goals[-1].yaw == -1.5


def test_a_spawn_naming_only_its_asset_and_position_takes_the_defaults(commander):
    status, _, _ = _call(commander, lambda client: _stream(client, "/api/objects/spawn", {
        "asset_id": "props/furniture/desk", "position": [1, 0, 0],
    }))

    assert status == 200
    assert commander.actions.spawn_object.goals == [SimpleNamespace(
        asset_id="props/furniture/desk", position=[1.0, 0.0, 0.0], yaw=0.0, orientation=None, scale=1.0,
        physics="none", mass=0.1,
    )]


_PHYSICS_REFUSAL = 'physics must be "dynamic", "static" or "none"'


@pytest.mark.parametrize(("fields", "message"), [
    ({"asset_id": None}, "asset_id must be a non-empty string"),
    ({"asset_id": ""}, "asset_id must be a non-empty string"),
    ({"asset_id": 7}, "asset_id must be a non-empty string"),
    ({"position": None}, "position must be 3 finite numbers"),
    ({"position": [1, 2]}, "position must be 3 finite numbers"),
    ({"position": [1, "2", 3]}, "position must be 3 finite numbers"),
    ({"yaw": "north"}, "yaw must be a finite number"),
    ({"yaw": None}, "yaw must be a finite number"),
    ({"yaw": [0.0]}, "yaw must be a finite number"),
    ({"yaw": True}, "yaw must be a finite number"),
    ({"scale": "large"}, "scale must be a finite number"),
    ({"scale": True}, "scale must be a finite number"),
    ({"physics": 3}, _PHYSICS_REFUSAL),
    ({"physics": "floating"}, _PHYSICS_REFUSAL),
    ({"physics": None}, _PHYSICS_REFUSAL),
    ({"mass": None}, "mass must be a finite number"),
    ({"mass": "heavy"}, "mass must be a finite number"),
])
def test_a_spawn_with_a_field_the_contract_does_not_take_is_refused_before_the_simulation(commander, fields, message):
    payload = {"asset_id": "props/furniture/desk", "position": [1.0, 0.0, 0.0], **fields}

    assert _call(commander, lambda client: _stream(client, "/api/objects/spawn", payload)) == (
        400, "application/json", [{"success": False, "message": message}],
    )
    assert commander.actions.spawn_object.goals == []


def test_a_spawn_naming_no_asset_is_refused_before_the_simulation(commander):
    assert _call(commander, lambda client: _stream(client, "/api/objects/spawn", {"position": [1.0, 0.0, 0.0]})) == (
        400, "application/json", [{"success": False, "message": "asset_id must be a non-empty string"}],
    )
    assert commander.actions.spawn_object.goals == []


def test_invalid_input_is_a_400_that_never_reaches_the_provider(commander, caplog):
    caplog.set_level(logging.INFO)

    assert _call(commander, lambda client: _post(
        client, "/api/objects/spawn", {"asset_id": "props/blocks/red_block", "position": [1, 2]},
    )) == (400, {"success": False, "message": "position must be 3 finite numbers"})
    assert commander.actions.spawn_object.goals == []
    assert _node_log(commander, caplog) == [
        (logging.WARNING, "POST /api/objects/spawn failed: position must be 3 finite numbers", None),
    ]


def test_setup_tolerates_a_provider_without_catalogue_but_not_an_unreachable_one(
    commander, node_runner, caplog, monkeypatch,
):
    caplog.set_level(logging.INFO)
    served = []
    # The socket reports a different port from the one configured, which is
    # what a fallback looks like: the log must follow the socket.
    listener = _FakeListener("127.0.0.1", 9100)

    monkeypatch.setattr(commander.module.listen, "bind_listener", lambda *bound: listener)

    async def fake_server(app, bound):
        served.append((bound, {route.resource.canonical for route in app.router.routes()}))
        return asyncio.create_task(asyncio.sleep(0))

    monkeypatch.setattr(commander.module.listen, "start_serving", fake_server)
    commander.services.get_assets_list.answers = [_busy()]

    async def run():
        await asyncio.gather(*await commander.module.setup(_parameters(9000, "127.0.0.1"), node_runner))

    asyncio.run(run())
    [(bound, routes)] = served
    assert bound is listener
    assert routes >= {"/", "/api/assets", "/api/scene/load"}
    assert _node_log(commander, caplog) == [
        (logging.INFO, "Scene commander starting", None),
        # The socket is taken before the provider is waited on.
        (logging.INFO, "Scene panel bound at 127.0.0.1:9100", None),
        (logging.INFO, "Capabilities: scene manipulation only", None),
        (logging.INFO, f"Scene provider has no catalogue: {_BUSY}", None),
        (logging.INFO, "Scene provider object state ready: 0 runtime objects", None),
    ]

    commander.objects.get_object_states.answers = [TimeoutError("get_object_states timed out")]
    with pytest.raises(TimeoutError, match="get_object_states timed out"):
        asyncio.run(commander.module.setup(_parameters(9000, "127.0.0.1"), FakeNodeRunner()))
    assert len(served) == 1


def test_setup_tolerates_a_provider_without_object_state_and_the_page_asks_again(
    commander, node_runner, caplog, monkeypatch,
):
    caplog.set_level(logging.INFO)
    served = []
    monkeypatch.setattr(commander.module.listen, "bind_listener", lambda *bound: _FakeListener("127.0.0.1", 9000))

    async def fake_server(app, bound):
        served.append(app)
        return asyncio.create_task(asyncio.sleep(0))

    monkeypatch.setattr(commander.module.listen, "start_serving", fake_server)
    commander.objects.get_object_states.answers = [_no_object_state()]

    async def run():
        await asyncio.gather(*await commander.module.setup(_parameters(9000, "127.0.0.1"), node_runner))

    asyncio.run(run())
    [app] = served
    # The served page reads the same unavailable state, which is logged once.
    assert _call(commander, lambda client: _get(client, "/api/objects"), app) == (
        503, {"success": False, "message": _NOT_READY},
    )
    assert _node_log(commander, caplog)[3:] == [
        (logging.INFO, "Scene provider catalogue ready: 1 assets", None),
        (logging.INFO, f"Scene provider has no object state: {_NOT_READY}", None),
    ]


def test_setup_refuses_a_launch_it_cannot_serve_and_starts_no_server(commander, node_runner, monkeypatch):
    # A node that never binds must fail its launch, not stand ready with
    # nothing listening.
    served = []
    monkeypatch.setattr(commander.module.listen, "start_serving", lambda *started: served.append(started))

    with pytest.raises(ValueError, match="http_host"):
        asyncio.run(commander.module.setup(_parameters(9000, http_host="localhost"), node_runner))

    assert served == [], "a commander that cannot serve must start no server"
    assert node_runner.announced == [], "a commander that cannot serve must announce no panel"


def test_the_manifest_declares_the_panel_as_a_web_page(node_runner):
    # A page endpoint is what the launch lists under `Web pages:`.
    assert {label: endpoint["kind"] for label, endpoint in node_runner.declared.items()} == {"panel": "page"}


def test_setup_announces_the_panel_on_the_address_its_socket_took(commander, node_runner, monkeypatch):
    # The socket reports a different port from the one configured, which is
    # what a fallback looks like: the announcement must follow the socket.
    monkeypatch.setattr(commander.module.listen, "bind_listener", lambda *bound: _FakeListener("0.0.0.0", 9100))
    monkeypatch.setattr(commander.module.listen, "start_serving", _serve_nothing)

    async def run():
        await asyncio.gather(*await commander.module.setup(_parameters(9000), node_runner))

    asyncio.run(run())
    # A panel on every interface is announced on the wildcard, which the
    # daemon renders as one URL per address of the machine.
    assert node_runner.announced == [("panel", "http", "0.0.0.0", 9100, "")]
    # The check the runtime makes once setup returns.
    node_runner.seal_endpoints()


def test_http_server_keeps_browser_requests_out_of_the_log(commander, monkeypatch):
    runners = []
    sites = []
    started = asyncio.Event()

    class FakeRunner:
        def __init__(self, app, **options):
            runners.append((app, options))
            self.cleaned = False

        async def setup(self):
            pass

        async def cleanup(self):
            self.cleaned = True

    class FakeSite:
        def __init__(self, runner, sock):
            sites.append((runner, sock))

        async def start(self):
            started.set()

    listener = _FakeListener("127.0.0.1", 8766)
    monkeypatch.setattr(commander.module.web, "AppRunner", FakeRunner)
    monkeypatch.setattr(commander.module.web, "SockSite", FakeSite)

    async def run():
        task = await commander.module.listen.start_serving(
            commander.module._build_app(object()), listener,
        )
        assert started.is_set(), "serving starts before the node is reported ready"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    [(app, options)] = runners
    assert options == {"access_log": None}
    assert {route.resource.canonical for route in app.router.routes()} >= {"/", "/api/assets", "/api/scene/load"}
    [(runner, sock)] = sites
    assert sock is listener
    assert runner.cleaned



def test_the_robot_listing_names_what_the_scene_stands(commander):
    status, body = _call(commander, lambda client: _get(client, "/api/robots"))

    assert status == 200
    assert body["success"] is True
    assert body["count"] == 2
    assert [robot["robot"] for robot in body["robots"]] == ["alpha", "bravo"]
    # Each stands where the simulation lists it, facing the way it lists.
    assert body["robots"] == _ROBOTS


def test_a_move_carries_the_robot_it_addresses_and_the_way_it_faces(commander):
    status, _ = _call(commander, lambda client: _post(
        client, "/api/robot/move", {"robot": "bravo_init_inst", "position": [1.0, -1.5, 0.0], "yaw": 1.25}
    ))

    assert status == 200
    goal = commander.actions.move_robot.goals[-1]
    assert goal.robot == "bravo_init_inst"
    assert goal.position == [1.0, -1.5, 0.0]
    assert goal.yaw == 1.25


def test_a_move_naming_no_robot_is_refused_before_the_simulation(commander):
    status, _ = _call(commander, lambda client: _post(
        client, "/api/robot/move", {"position": [1.0, 0.0, 0.0], "yaw": 0.0}
    ))

    assert status == 400
    assert commander.actions.move_robot.goals == []


@pytest.mark.parametrize("payload", [
    {"robot": "bravo", "position": [1.0, 0.0, 0.0]},
    {"robot": "bravo", "position": [1.0, 0.0, 0.0], "yaw": "north"},
    {"robot": "bravo", "position": [1.0, 0.0, 0.0], "yaw": None},
])
def test_a_move_naming_no_yaw_is_refused_before_the_simulation(commander, payload):
    assert _call(commander, lambda client: _post(client, "/api/robot/move", payload)) == (
        400, {"success": False, "message": "yaw must be a finite number"},
    )
    assert commander.actions.move_robot.goals == []


# ---------------------------------------------------------------------------
# Loads and spawns: followed by their progress, whatever the link speed
# ---------------------------------------------------------------------------


def _running(action):
    """Let every goal of `action` run until the test ends it."""
    action.ends_at_once = False
    return action


def _cancelled(message):
    return SimpleNamespace(success=False, message=message)


_SPAWN = {"asset_id": "props/furniture/desk", "position": [1.0, 0.0, 0.0], "physics": "static"}
_SPAWN_SUMMARY = "spawn_object(asset_id=props/furniture/desk, position=[1.0, 0.0, 0.0], yaw=0.0, physics=static, mass=0.1)"
# The goals followed by their progress: the action, its route, a payload and
# the summary its log lines open with.
_FOLLOWED = [
    ("load_scene", "/api/scene/load", _LOAD, _LOAD_SUMMARY),
    ("spawn_object", "/api/objects/spawn", _SPAWN, _SPAWN_SUMMARY),
]
_FOLLOWED_IDS = [goal[0] for goal in _FOLLOWED]


async def _start(client, action, path, payload):
    """Post a load or a spawn, and return its answer still streaming and its
    goal once the commander follows it."""
    answer = asyncio.create_task(_stream(client, path, payload))
    goal = await action.fired()
    await goal.settled()
    return answer, goal


async def _window_passes(commander, client, goal):
    """One silence window passes: the first has the commander cancel the
    goal, the second ends its follow."""
    commander.clock.advance(commander.module.PROGRESS_TIMEOUT_S)


async def _stall(commander, client, goal):
    """One silence window passes, and the commander waits on the goal again."""
    await _window_passes(commander, client, goal)
    await goal.settled()


async def _cancel_from_the_page(commander, client, goal):
    await _post(client, "/api/goals/1/cancel", {})


# Who cancels a load, and why its log line says it did.
_CANCELLERS = [(_window_passes, "no progress for 60 s"), (_cancel_from_the_page, "the page asked")]
_CANCELLER_IDS = ["stall", "page"]


def test_a_load_that_keeps_reporting_progress_past_one_window_completes(commander, caplog):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)
    clock = commander.clock

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        # One report every 40 s: the load runs well past one window, and is
        # never a whole window without progress.
        for bytes_fetched, files_ready, building in (
            (40_000_000, 10, False), (90_000_000, 30, False), (123_400_000, 41, True),
        ):
            clock.advance(40)
            goal.report(bytes_fetched, files_ready, building)
            await goal.settled()
        clock.advance(40)
        goal.complete(f"Loaded scene {_FACTORY}")
        return await answer, goal

    (status, _, lines), goal = _call(commander, scenario)

    assert clock.now == 160
    assert (status, lines[-1]) == (200, {"success": True, "message": f"Loaded scene {_FACTORY}"})
    assert goal.cancels == []
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"{_LOAD_SUMMARY}: Loaded scene {_FACTORY}", None),
    ]


def test_the_stream_carries_the_goal_id_then_each_progress_report_then_the_result(commander):
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        for bytes_fetched, files_ready, building in ((0, 12, False), (5_000_000, 13, False), (5_000_000, 41, True)):
            goal.report(bytes_fetched, files_ready, building)
            await goal.settled()
        goal.complete(f"Loaded scene {_FACTORY}")
        return await answer

    assert _call(commander, scenario) == (200, "application/x-ndjson", [
        {"goal": "1"},
        {"progress": {"bytes_fetched": 0, "files_ready": 12, "building": False}},
        {"progress": {"bytes_fetched": 5_000_000, "files_ready": 13, "building": False}},
        {"progress": {"bytes_fetched": 5_000_000, "files_ready": 41, "building": True}},
        {"success": True, "message": f"Loaded scene {_FACTORY}"},
    ])


@pytest.mark.parametrize(("name", "path", "payload", "summary"), _FOLLOWED, ids=_FOLLOWED_IDS)
def test_a_silent_goal_is_cancelled_after_one_window_and_reported(commander, caplog, name, path, payload, summary):
    caplog.set_level(logging.INFO)
    action = _running(getattr(commander.actions, name))

    async def scenario(client):
        answer, goal = await _start(client, action, path, payload)
        goal.report(0, 3)
        await goal.settled()
        await _stall(commander, client, goal)
        cancelled_in_the_window = list(goal.cancels)
        goal.end(Status.CANCELLED, _cancelled("Cancelled while fetching"))
        return await answer, goal, cancelled_in_the_window

    (status, _, lines), goal, cancelled_in_the_window = _call(commander, scenario)

    message = f"{name} made no progress for 60 s; it was cancelled"
    assert cancelled_in_the_window == [10.0]
    assert (status, lines[-1]) == (200, {"success": False, "message": message, "cancelled_by": "commander"})
    assert goal.cancels == [10.0]
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"{summary}: no progress for 60 s; cancel SIGNALLED", None),
        (logging.WARNING, f"POST {path} failed: {message}", None),
    ]


def test_a_load_that_completes_after_the_cancel_is_reported_as_success(commander, caplog):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        goal.report(368_000_000, 135, building=True)
        await goal.settled()
        # The build outlasts a window; the cancel comes too late to stop it.
        await _stall(commander, client, goal)
        goal.complete(f"Loaded scene {_FACTORY}")
        return await answer, goal

    (status, _, lines), goal = _call(commander, scenario)

    assert (status, lines[-1]) == (200, {"success": True, "message": f"Loaded scene {_FACTORY}"})
    assert goal.cancels == [10.0]
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"{_LOAD_SUMMARY}: no progress for 60 s; cancel SIGNALLED", None),
        (logging.INFO, f"{_LOAD_SUMMARY}: Loaded scene {_FACTORY}", None),
    ]


@pytest.mark.parametrize(("name", "path", "payload", "summary", "cancel", "why", "end"), [
    (
        "load_scene", "/api/scene/load", _LOAD, _LOAD_SUMMARY, _stall, "no progress for 60 s",
        {"success": True, "message": f"Loaded scene {_FACTORY}"},
    ),
    (
        "spawn_object", "/api/objects/spawn", _SPAWN, _SPAWN_SUMMARY, _cancel_from_the_page, "the page asked",
        {"success": True, "message": "Spawned props/furniture/desk", "object_id": "obj_7"},
    ),
], ids=["load-after-stall", "spawn-after-page-cancel"])
def test_a_goal_the_simulator_built_despite_its_cancel_is_reported_as_success(
    commander, caplog, name, path, payload, summary, cancel, why, end,
):
    caplog.set_level(logging.INFO)
    action = _running(getattr(commander.actions, name))

    async def scenario(client):
        answer, goal = await _start(client, action, path, payload)
        await cancel(commander, client, goal)
        # The provider builds what it started, and ends the goal cancelled
        # with success.
        goal.end(Status.CANCELLED, SimpleNamespace(**end))
        return await answer, goal

    (status, _, lines), goal = _call(commander, scenario)

    assert (status, lines[-1]) == (200, end)
    assert goal.cancels == [10.0]
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"{summary}: {why}; cancel SIGNALLED", None),
        (logging.INFO, f"{summary}: {end['message']}", None),
    ]


@pytest.mark.parametrize(("name", "path", "payload", "summary"), _FOLLOWED, ids=_FOLLOWED_IDS)
def test_a_goal_whose_simulator_answers_no_cancel_ends_one_window_later(commander, caplog, name, path, payload, summary):
    caplog.set_level(logging.INFO)
    action = _running(getattr(commander.actions, name))

    async def scenario(client):
        answer, goal = await _start(client, action, path, payload)
        await _stall(commander, client, goal)
        await _window_passes(commander, client, goal)
        return await answer, goal

    (status, _, lines), goal = _call(commander, scenario)

    message = f"{name} made no progress for 60 s; a cancel was sent and the simulator did not answer"
    assert (status, lines[-1]) == (200, {"success": False, "message": message})
    assert goal.cancels == [10.0]
    # A goal that never ended has no result to ask for.
    assert goal.result_waits == []
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"{summary}: no progress for 60 s; cancel SIGNALLED", None),
        (logging.WARNING, f"POST {path} failed: {message}", None),
    ]


def test_a_load_that_reports_progress_after_its_stall_cancel_ends_at_the_next_silent_window(commander, caplog):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        await _stall(commander, client, goal)
        # Progress after the cancel does not undo it: the next silent window
        # ends the follow.
        goal.report(2_000_000, 5)
        await goal.settled()
        await _window_passes(commander, client, goal)
        return await answer, goal

    (_, _, lines), goal = _call(commander, scenario)

    message = "load_scene made no progress for 60 s; a cancel was sent and the simulator did not answer"
    assert lines == [
        {"goal": "1"},
        {"progress": {"bytes_fetched": 2_000_000, "files_ready": 5, "building": False}},
        {"success": False, "message": message},
    ]
    assert goal.cancels == [10.0]
    assert goal.result_waits == []


def test_a_load_cancelled_from_the_page_that_stays_silent_ends_two_windows_later(commander, caplog):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        await _cancel_from_the_page(commander, client, goal)
        # The first silent window finds the page's cancel delivered and sends
        # nothing more; the second ends the follow.
        await _stall(commander, client, goal)
        await _window_passes(commander, client, goal)
        return await answer, goal

    (_, _, lines), goal = _call(commander, scenario)

    message = "load_scene made no progress for 60 s; a cancel was sent and the simulator did not answer"
    assert lines[-1] == {"success": False, "message": message}
    assert goal.cancels == [10.0]
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"{_LOAD_SUMMARY}: the page asked; cancel SIGNALLED", None),
        (logging.WARNING, f"POST /api/scene/load failed: {message}", None),
    ]


@pytest.mark.parametrize(("cancel", "why"), _CANCELLERS, ids=_CANCELLER_IDS)
def test_a_cancel_that_finds_the_load_ended_asks_for_its_result_at_once(commander, caplog, cancel, why):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        # The load has ended, and the message that ends its stream is lost.
        goal.end_without_stream_end(Status.COMPLETED, SimpleNamespace(success=True, message=f"Loaded scene {_FACTORY}"))
        goal.cancel_replies.append(CancelState.ALREADY_TERMINAL)
        await cancel(commander, client, goal)
        return await answer, goal

    (_, _, lines), goal = _call(commander, scenario)

    assert lines[-1] == {"success": True, "message": f"Loaded scene {_FACTORY}"}
    assert goal.cancels == [10.0]
    assert goal.result_waits == [10.0]
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"{_LOAD_SUMMARY}: {why}; cancel ALREADY_TERMINAL", None),
        (logging.INFO, f"{_LOAD_SUMMARY}: Loaded scene {_FACTORY}", None),
    ]


@pytest.mark.parametrize(("cancel", "why"), _CANCELLERS, ids=_CANCELLER_IDS)
def test_a_cancel_of_a_load_the_simulator_does_not_know_ends_the_load_at_once(commander, caplog, cancel, why):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        goal.cancel_replies.append(CancelState.UNKNOWN)
        await cancel(commander, client, goal)
        return await answer, goal

    (_, _, lines), goal = _call(commander, scenario)

    message = "load_scene: the simulator does not know the goal"
    assert lines[-1] == {"success": False, "message": message}
    assert goal.result_waits == []
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"{_LOAD_SUMMARY}: {why}; cancel UNKNOWN", None),
        (logging.WARNING, f"POST /api/scene/load failed: {message}", None),
    ]


_UNDELIVERED = [
    TimeoutError("service 'cancel' timed out"),
    ConnectionError("service 'cancel' is unreachable"),
]
_UNDELIVERED_IDS = ["timeout", "unreachable"]


@pytest.mark.parametrize("failure", _UNDELIVERED, ids=_UNDELIVERED_IDS)
def test_a_page_cancel_that_is_not_delivered_answers_502_and_the_next_one_is_sent(commander, caplog, failure):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        goal.cancel_replies.append(failure)
        lost = await _post(client, "/api/goals/1/cancel", {})
        delivered = await _post(client, "/api/goals/1/cancel", {})
        goal.end(Status.CANCELLED, _cancelled("Cancelled while fetching"))
        return lost, delivered, await answer, goal

    lost, delivered, (_, _, lines), goal = _call(commander, scenario)

    assert lost == (502, {"success": False, "message": f"the cancel of load_scene was not delivered: {failure}"})
    assert delivered == (200, {"success": True, "message": "the simulator answered the cancel of load_scene: SIGNALLED"})
    assert goal.cancels == [10.0, 10.0]
    assert lines[-1] == {"success": False, "message": "load_scene was cancelled from the page", "cancelled_by": "page"}
    assert _node_log(commander, caplog) == [
        (logging.WARNING, f"{_LOAD_SUMMARY}: the page asked; the cancel was not delivered: {failure}", None),
        (logging.INFO, f"{_LOAD_SUMMARY}: the page asked; cancel SIGNALLED", None),
        (logging.WARNING, "POST /api/scene/load failed: load_scene was cancelled from the page", None),
    ]


def test_a_stall_after_a_page_cancel_that_was_not_delivered_sends_the_cancel_again(commander, caplog):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        goal.cancel_replies.append(TimeoutError("service 'cancel' timed out"))
        lost = await _post(client, "/api/goals/1/cancel", {})
        await _stall(commander, client, goal)
        goal.end(Status.CANCELLED, _cancelled("Cancelled while fetching"))
        return lost, await answer, goal

    (lost_status, _), (_, _, lines), goal = _call(commander, scenario)

    assert lost_status == 502
    assert goal.cancels == [10.0, 10.0]
    assert lines[-1] == {
        "success": False, "message": "load_scene made no progress for 60 s; it was cancelled", "cancelled_by": "commander",
    }


# The stall cancel that does not reach the simulator, and the log line it leaves.
_LOST_STALL_CANCEL = _UNDELIVERED[0]
_LOST_STALL_CANCEL_LOG = (
    logging.WARNING, f"{_LOAD_SUMMARY}: no progress for 60 s; the cancel was not delivered: {_LOST_STALL_CANCEL}", None,
)
# How a stall cancel sent again fares: the reply of the simulator, the end
# of the load it gives, and the log line of the cancel.
_SENT_AGAIN = [
    (
        CancelState.SIGNALLED,
        "the cancel was sent again and the simulator took it",
        (logging.INFO, f"{_LOAD_SUMMARY}: no progress for 60 s again; cancel SIGNALLED", None),
    ),
    *(
        (
            failure,
            f"the cancel was sent again and was not delivered: {failure}",
            (logging.WARNING, f"{_LOAD_SUMMARY}: no progress for 60 s again; the cancel was not delivered: {failure}", None),
        )
        for failure in _UNDELIVERED
    ),
]
_SENT_AGAIN_IDS = ["taken", *(f"not-delivered-{failure}" for failure in _UNDELIVERED_IDS)]


@pytest.mark.parametrize(("reply", "outcome", "cancel_log"), _SENT_AGAIN, ids=_SENT_AGAIN_IDS)
def test_a_stall_cancel_that_is_not_delivered_is_sent_again_once_as_the_next_window_ends(
    commander, caplog, reply, outcome, cancel_log,
):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        goal.cancel_replies.extend([_LOST_STALL_CANCEL, reply])
        await _stall(commander, client, goal)
        await _window_passes(commander, client, goal)
        return await answer, goal

    (_, _, lines), goal = _call(commander, scenario)

    message = f"load_scene made no progress for 60 s; {outcome}"
    assert lines[-1] == {"success": False, "message": message}
    # Sent again once, and never a third time: the follow ends.
    assert goal.cancels == [10.0, 10.0]
    assert goal.result_waits == []
    assert _node_log(commander, caplog) == [
        _LOST_STALL_CANCEL_LOG,
        cancel_log,
        (logging.WARNING, f"POST /api/scene/load failed: {message}", None),
    ]


def test_a_stall_cancel_that_is_not_delivered_is_sent_again_once_new_progress_falls_silent(commander, caplog):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)
    clock = commander.clock
    fetched = (1_000_000, 2_000_000, 3_000_000)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        goal.cancel_replies.append(_LOST_STALL_CANCEL)
        await _stall(commander, client, goal)
        # The link comes back for a while: the simulator reports progress,
        # then its fetch hangs again.
        for bytes_fetched in fetched:
            clock.advance(20)
            goal.report(bytes_fetched, 1)
            await goal.settled()
        await _window_passes(commander, client, goal)
        return await answer, goal

    (_, _, lines), goal = _call(commander, scenario)

    message = "load_scene made no progress for 60 s; the cancel was sent again and the simulator took it"
    assert clock.now == 180
    assert lines == [
        {"goal": "1"},
        *({"progress": {"bytes_fetched": bytes_fetched, "files_ready": 1, "building": False}} for bytes_fetched in fetched),
        {"success": False, "message": message},
    ]
    assert goal.cancels == [10.0, 10.0]
    assert _node_log(commander, caplog) == [
        _LOST_STALL_CANCEL_LOG,
        (logging.INFO, f"{_LOAD_SUMMARY}: no progress for 60 s again; cancel SIGNALLED", None),
        (logging.WARNING, f"POST /api/scene/load failed: {message}", None),
    ]


@pytest.mark.parametrize(("reply", "end", "result_waits"), [
    (CancelState.ALREADY_TERMINAL, {"success": True, "message": f"Loaded scene {_FACTORY}"}, [10.0]),
    (CancelState.UNKNOWN, {"success": False, "message": "load_scene: the simulator does not know the goal"}, []),
], ids=["ended", "unknown"])
def test_a_stall_cancel_sent_again_that_finds_the_load_over_ends_it_with_what_the_simulator_knows(
    commander, caplog, reply, end, result_waits,
):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        goal.cancel_replies.extend([_LOST_STALL_CANCEL, reply])
        await _stall(commander, client, goal)
        # The load ends, and the message that ends its stream is lost.
        goal.end_without_stream_end(Status.COMPLETED, SimpleNamespace(success=True, message=f"Loaded scene {_FACTORY}"))
        await _window_passes(commander, client, goal)
        return await answer, goal

    (_, _, lines), goal = _call(commander, scenario)

    assert lines[-1] == end
    assert goal.cancels == [10.0, 10.0]
    assert goal.result_waits == result_waits
    assert _node_log(commander, caplog)[:2] == [
        _LOST_STALL_CANCEL_LOG,
        (logging.INFO, f"{_LOAD_SUMMARY}: no progress for 60 s again; cancel {reply.name}", None),
    ]


def test_a_load_whose_stall_cancel_was_not_delivered_is_followed_to_its_real_end(commander, caplog):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        goal.cancel_replies.append(ConnectionError("service 'cancel' is unreachable"))
        await _stall(commander, client, goal)
        goal.complete(f"Loaded scene {_FACTORY}")
        return await answer

    (_, _, lines) = _call(commander, scenario)

    assert lines[-1] == {"success": True, "message": f"Loaded scene {_FACTORY}"}


def test_a_simulator_that_is_gone_ends_the_load_at_once_without_asking_for_its_result(commander, caplog):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        # The result is asked for only once the feedback stream has ended.
        assert goal.result_waits == []
        goal.report(1_000_000, 2)
        await goal.settled()
        goal.vanish()
        return await answer, goal

    (status, _, lines), goal = _call(commander, scenario)

    assert (status, lines[-1]) == (200, {"success": False, "message": "load_scene ended: the simulator is gone"})
    assert goal.result_waits == []
    assert goal.cancels == []
    assert _node_log(commander, caplog) == [
        (logging.WARNING, "POST /api/scene/load failed: load_scene ended: the simulator is gone", None),
    ]


def test_a_load_whose_result_does_not_come_after_its_stream_ended_reports_the_timeout(commander, caplog):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        goal.end_without_result()
        await goal.settled()
        commander.clock.advance(10)
        return await answer, goal

    (status, _, lines), goal = _call(commander, scenario)

    message = "action 'load_scene' (alpha_waldo) has timed out waiting for result"
    assert (status, lines[-1]) == (200, {"success": False, "message": message})
    assert goal.result_waits == [10.0]
    assert _node_log(commander, caplog) == [(logging.WARNING, f"POST /api/scene/load failed: {message}", None)]


def test_a_progress_message_the_commander_cannot_read_ends_the_stream_and_keeps_its_traceback(commander, caplog):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        goal.report_message(SimpleNamespace(files_ready=1, building=False))
        return await answer

    status, _, lines = _call(commander, scenario)

    message = "'types.SimpleNamespace' object has no attribute 'bytes_fetched'"
    assert (status, lines) == (200, [{"goal": "1"}, {"success": False, "message": message}])
    [(level, text, exc_info)] = _node_log(commander, caplog)
    assert (level, text) == (logging.WARNING, f"POST /api/scene/load failed: {message}")
    assert isinstance(exc_info[1], AttributeError)


@pytest.mark.parametrize(("data", "message"), [
    (
        _cancelled("load_scene(scene/factory_conveyor) was replaced by load_scene(scene/full_warehouse)"),
        "load_scene was cancelled by the simulator: "
        "load_scene(scene/factory_conveyor) was replaced by load_scene(scene/full_warehouse)",
    ),
    (None, "load_scene was cancelled by the simulator"),
], ids=["with-reason", "without-reason"])
def test_a_load_the_simulator_cancelled_itself_is_reported_as_cancelled_by_it(commander, caplog, data, message):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        # A newer load replaces this one.
        goal.end(Status.CANCELLED, data)
        return await answer, goal

    (status, _, lines), goal = _call(commander, scenario)

    assert (status, lines[-1]) == (200, {"success": False, "message": message, "cancelled_by": "simulator"})
    assert goal.cancels == []
    assert _node_log(commander, caplog) == [(logging.WARNING, f"POST /api/scene/load failed: {message}", None)]


def test_the_cancel_route_cancels_the_load_its_stream_names_once(commander, caplog):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)

    async def scenario(client):
        answer, goal = await _start(client, load, "/api/scene/load", _LOAD)
        first = await _post(client, "/api/goals/1/cancel", {})
        again = await _post(client, "/api/goals/1/cancel", {})
        goal.end(Status.CANCELLED, _cancelled("Cancelled while fetching"))
        streamed = await answer
        after_the_end = await _post(client, "/api/goals/1/cancel", {})
        return first, again, streamed, after_the_end, goal

    first, again, (status, _, lines), after_the_end, goal = _call(commander, scenario)

    assert first == (200, {"success": True, "message": "the simulator answered the cancel of load_scene: SIGNALLED"})
    assert again == first
    assert goal.cancels == [10.0]
    assert status == 200
    assert lines[0] == {"goal": "1"}
    assert lines[-1] == {"success": False, "message": "load_scene was cancelled from the page", "cancelled_by": "page"}
    assert after_the_end == (404, {"success": False, "message": "no running goal 1"})
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"{_LOAD_SUMMARY}: the page asked; cancel SIGNALLED", None),
        (logging.WARNING, "POST /api/scene/load failed: load_scene was cancelled from the page", None),
        (logging.WARNING, "POST /api/goals/1/cancel failed: no running goal 1", None),
    ]


def test_a_page_that_closes_its_stream_cancels_the_goal(commander, caplog):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)
    app, transports, returned = _page_that_closes(commander)

    async def scenario(client):
        answer = asyncio.create_task(client.post("/api/scene/load", json=_LOAD))
        goal = await load.fired()
        await goal.settled()
        # The page goes away; the next line has nowhere to go.
        transports[0].close()
        goal.report(2_000_000, 4)
        await goal.settled()
        cancelled_on_the_write = list(goal.cancels)
        goal.end(Status.CANCELLED, _cancelled("Cancelled while fetching"))
        await _load_handler_returned(commander, returned, answer)
        return goal, cancelled_on_the_write

    goal, cancelled_on_the_write = _call(commander, scenario, app)

    assert cancelled_on_the_write == [10.0]
    assert goal.cancels == [10.0]
    assert returned == ["/api/scene/load"]
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"{_LOAD_SUMMARY}: the page closed its stream; cancel SIGNALLED", None),
        (logging.WARNING, "POST /api/scene/load failed: load_scene was cancelled from the page", None),
    ]


def test_a_page_that_closes_before_its_load_is_admitted_cancels_it_on_the_first_line(commander, caplog):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)
    app, transports, returned = _page_that_closes(commander)

    async def scenario(client):
        load.admission = asyncio.Event()
        answer = asyncio.create_task(client.post("/api/scene/load", json=_LOAD))
        await commander.clock.until(lambda: load.goals)
        # The page goes away while the simulator admits its load; the goal
        # line has nowhere to go.
        transports[0].close()
        load.admission.set()
        goal = await load.fired()
        await goal.settled()
        cancelled_on_the_goal_line = list(goal.cancels)
        goal.end(Status.CANCELLED, _cancelled("Cancelled while fetching"))
        await _load_handler_returned(commander, returned, answer)
        return goal, cancelled_on_the_goal_line

    goal, cancelled_on_the_goal_line = _call(commander, scenario, app)

    assert cancelled_on_the_goal_line == [10.0]
    assert goal.cancels == [10.0]
    assert returned == ["/api/scene/load"]
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"{_LOAD_SUMMARY}: the page closed its stream; cancel SIGNALLED", None),
        (logging.WARNING, "POST /api/scene/load failed: load_scene was cancelled from the page", None),
    ]


def test_a_page_that_closes_while_its_load_is_silent_leaves_the_cancel_to_the_silence_window(commander, caplog):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)
    app, transports, returned = _page_that_closes(commander)

    async def scenario(client):
        answer = asyncio.create_task(client.post("/api/scene/load", json=_LOAD))
        goal = await load.fired()
        await goal.settled()
        # The page goes away while nothing is written to it.
        transports[0].close()
        await _stall(commander, client, goal)
        goal.end(Status.CANCELLED, _cancelled("Cancelled while fetching"))
        await _load_handler_returned(commander, returned, answer)
        return goal

    goal = _call(commander, scenario, app)

    message = "load_scene made no progress for 60 s; it was cancelled"
    assert goal.cancels == [10.0]
    assert returned == ["/api/scene/load"]
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"{_LOAD_SUMMARY}: no progress for 60 s; cancel SIGNALLED", None),
        (logging.WARNING, f"POST /api/scene/load failed: {message}", None),
        (logging.INFO, f"{_LOAD_SUMMARY}: the page closed its stream before its result", None),
    ]


def test_a_page_that_aborts_its_stream_then_asks_for_the_cancel_has_its_load_cancelled_once(commander, caplog):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)
    app, transports, returned = _page_that_closes(commander)

    async def scenario(client):
        answer = asyncio.create_task(client.post("/api/scene/load", json=_LOAD))
        goal = await load.fired()
        await goal.settled()
        # The page aborts its stream of the load, which frees its connection,
        # then asks for the cancel on another.
        transports[0].close()
        cancelled = await _post(client, "/api/goals/1/cancel", {})
        # The next line finds the page gone; the cancel stands.
        goal.report(2_000_000, 4)
        await goal.settled()
        goal.end(Status.CANCELLED, _cancelled("Cancelled while fetching"))
        await _load_handler_returned(commander, returned, answer)
        return cancelled, goal

    cancelled, goal = _call(commander, scenario, app)

    assert cancelled == (200, {"success": True, "message": "the simulator answered the cancel of load_scene: SIGNALLED"})
    assert goal.cancels == [10.0]
    assert returned == ["/api/goals/1/cancel", "/api/scene/load"]
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"{_LOAD_SUMMARY}: the page asked; cancel SIGNALLED", None),
        (logging.WARNING, "POST /api/scene/load failed: load_scene was cancelled from the page", None),
    ]


@pytest.mark.parametrize(("end", "end_log"), [
    (
        lambda goal: goal.complete(f"Loaded scene {_FACTORY}"),
        (logging.INFO, f"{_LOAD_SUMMARY}: Loaded scene {_FACTORY}", None),
    ),
    (
        FakeGoal.vanish,
        (logging.WARNING, "POST /api/scene/load failed: load_scene ended: the simulator is gone", None),
    ),
], ids=["completed", "simulator-gone"])
def test_a_page_that_closes_before_the_result_of_its_load_cancels_nothing(commander, caplog, end, end_log):
    caplog.set_level(logging.INFO)
    load = _running(commander.actions.load_scene)
    app, transports, returned = _page_that_closes(commander)

    async def scenario(client):
        answer = asyncio.create_task(client.post("/api/scene/load", json=_LOAD))
        goal = await load.fired()
        await goal.settled()
        # The page goes away while the load builds in silence, and the load
        # then ends: its result is the first line with nowhere to go.
        transports[0].close()
        end(goal)
        await _load_handler_returned(commander, returned, answer)
        return goal

    goal = _call(commander, scenario, app)

    assert goal.cancels == []
    assert returned == ["/api/scene/load"]
    assert _node_log(commander, caplog) == [
        end_log,
        (logging.INFO, f"{_LOAD_SUMMARY}: the page closed its stream before its result", None),
    ]


def test_a_spawn_streams_its_progress_and_reports_the_minted_object_id(commander, caplog):
    caplog.set_level(logging.INFO)
    spawn = _running(commander.actions.spawn_object)

    async def scenario(client):
        answer, goal = await _start(client, spawn, "/api/objects/spawn", _SPAWN)
        goal.report(1_200_000, 2)
        await goal.settled()
        goal.report(1_900_000, 3, building=True)
        await goal.settled()
        goal.complete("Spawned props/furniture/desk", object_id="obj_7")
        return await answer

    assert _call(commander, scenario) == (200, "application/x-ndjson", [
        {"goal": "1"},
        {"progress": {"bytes_fetched": 1_200_000, "files_ready": 2, "building": False}},
        {"progress": {"bytes_fetched": 1_900_000, "files_ready": 3, "building": True}},
        {"success": True, "message": "Spawned props/furniture/desk", "object_id": "obj_7"},
    ])
    assert spawn.admissions == [60.0]


def test_spawns_in_flight_together_are_each_followed_and_cancelled_by_their_own_id(commander):
    spawn = _running(commander.actions.spawn_object)
    desk = {"asset_id": "props/furniture/desk", "position": [1.0, 0.0, 0.0]}
    block = {"asset_id": "props/blocks/red_block", "position": [0.5, 0.0, 0.8]}

    async def scenario(client):
        desk_answer = asyncio.create_task(_stream(client, "/api/objects/spawn", desk))
        desk_goal = await spawn.fired(0)
        await desk_goal.settled()
        block_answer = asyncio.create_task(_stream(client, "/api/objects/spawn", block))
        block_goal = await spawn.fired(1)
        await block_goal.settled()
        cancelled = await _post(client, "/api/goals/2/cancel", {})
        block_goal.end(Status.CANCELLED, _cancelled("Cancelled while fetching"))
        desk_goal.complete("Spawned props/furniture/desk", object_id="obj_1")
        return cancelled, await desk_answer, await block_answer, desk_goal, block_goal

    cancelled, desk_answer, block_answer, desk_goal, block_goal = _call(commander, scenario)

    assert cancelled == (200, {"success": True, "message": "the simulator answered the cancel of spawn_object: SIGNALLED"})
    assert (desk_goal.cancels, block_goal.cancels) == ([], [10.0])
    assert desk_answer[2] == [{"goal": "1"}, {"success": True, "message": "Spawned props/furniture/desk", "object_id": "obj_1"}]
    assert block_answer[2] == [
        {"goal": "2"},
        {"success": False, "message": "spawn_object was cancelled from the page", "cancelled_by": "page"},
    ]


_MOVE_ROBOT = {"robot": "alpha", "position": [0, 0, 0], "yaw": 0}
_MOVE_ROBOT_SUMMARY = "move_robot(robot=alpha, position=[0.0, 0.0, 0.0], yaw=0.0)"
# The actions that report no progress: each is bounded as a whole goal.
_WHOLE_GOALS = [
    ("clear_scene", "/api/scene/clear", {}, "clear_scene()"),
    (
        "move_object", "/api/objects/move", {"object_id": "obj_1", "position": [1, 0, 0]},
        "move_object(object_id=obj_1, position=[1.0, 0.0, 0.0])",
    ),
    ("remove_object", "/api/objects/remove", {"object_id": "obj_1"}, "remove_object(object_id=obj_1)"),
    (
        "apply_force", "/api/objects/force", {"object_id": "obj_1", "force": [0, 0, 20]},
        "apply_force(object_id=obj_1, force=[0.0, 0.0, 20.0], duration_s=0.5)",
    ),
    ("move_robot", "/api/robot/move", _MOVE_ROBOT, _MOVE_ROBOT_SUMMARY),
]


async def _whole_goal_bound_passes(commander, goal):
    """The whole-goal bound of `goal` passes once the commander waits for its result."""
    await commander.clock.until(lambda: goal.result_waits)
    commander.clock.advance(commander.module.ACTION_TIMEOUT_S)


@pytest.mark.parametrize(("name", "path", "payload", "summary"), _WHOLE_GOALS, ids=[goal[0] for goal in _WHOLE_GOALS])
def test_an_action_without_progress_is_cancelled_once_its_whole_goal_bound_expires(
    commander, caplog, name, path, payload, summary,
):
    caplog.set_level(logging.INFO)
    action = _running(getattr(commander.actions, name))

    async def scenario(client):
        answer = asyncio.create_task(_post(client, path, payload))
        goal = await action.fired()
        await _whole_goal_bound_passes(commander, goal)
        return await answer, goal

    answer, goal = _call(commander, scenario)

    message = f"{name} did not end within 60 s; a cancel was sent"
    assert answer == (400, {"success": False, "message": message})
    assert action.admissions == [60.0]
    assert goal.result_waits == [60.0]
    assert goal.cancels == [10.0]
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"{summary}: no result within 60 s; cancel SIGNALLED", None),
        (logging.WARNING, f"POST {path} failed: {message}", None),
    ]


def test_an_action_whose_bound_cancel_finds_it_ended_answers_its_result(commander, caplog):
    caplog.set_level(logging.INFO)
    move = _running(commander.actions.move_robot)

    async def scenario(client):
        answer = asyncio.create_task(_post(client, "/api/robot/move", _MOVE_ROBOT))
        goal = await move.fired()
        goal.cancel_replies.append(CancelState.ALREADY_TERMINAL)
        await _whole_goal_bound_passes(commander, goal)
        # The goal ended as its bound expired; its result is asked for again.
        await commander.clock.until(lambda: len(goal.result_waits) == 2)
        goal.complete("alpha moved")
        return await answer, goal

    answer, goal = _call(commander, scenario)

    assert answer == (200, {"success": True, "message": "alpha moved"})
    assert goal.result_waits == [60.0, 10.0]
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"{_MOVE_ROBOT_SUMMARY}: no result within 60 s; cancel ALREADY_TERMINAL", None),
        (logging.INFO, f"{_MOVE_ROBOT_SUMMARY}: alpha moved", None),
    ]


@pytest.mark.parametrize(("reply", "message", "cancel_log"), [
    (
        CancelState.UNKNOWN,
        "move_robot: the simulator does not know the goal",
        (logging.INFO, f"{_MOVE_ROBOT_SUMMARY}: no result within 60 s; cancel UNKNOWN", None),
    ),
    (
        _UNDELIVERED[0],
        f"move_robot did not end within 60 s; the cancel was not delivered: {_UNDELIVERED[0]}",
        (
            logging.WARNING,
            f"{_MOVE_ROBOT_SUMMARY}: no result within 60 s; the cancel was not delivered: {_UNDELIVERED[0]}",
            None,
        ),
    ),
], ids=["unknown", "not-delivered"])
def test_an_action_whose_bound_cancel_does_not_take_says_why(commander, caplog, reply, message, cancel_log):
    caplog.set_level(logging.INFO)
    move = _running(commander.actions.move_robot)

    async def scenario(client):
        answer = asyncio.create_task(_post(client, "/api/robot/move", _MOVE_ROBOT))
        goal = await move.fired()
        goal.cancel_replies.append(reply)
        await _whole_goal_bound_passes(commander, goal)
        return await answer, goal

    answer, goal = _call(commander, scenario)

    assert answer == (400, {"success": False, "message": message})
    assert goal.result_waits == [60.0]
    assert goal.cancels == [10.0]
    assert _node_log(commander, caplog) == [cancel_log, (logging.WARNING, f"POST /api/robot/move failed: {message}", None)]


# ---------------------------------------------------------------------------
# Lighting, materials and cameras: what the launch bound decides the panels
# ---------------------------------------------------------------------------


def test_with_only_the_scene_bound_the_capabilities_are_absent_and_their_routes_answer_404(commander, caplog):
    caplog.set_level(logging.INFO)

    async def scenario(client):
        return (
            await _get(client, "/api/capabilities"),
            await _get(client, "/api/lighting"),
            await _post(client, "/api/lighting/intensity", {"light_id": "sun", "value": 1200}),
            await _post(client, "/api/lighting/reset", {}),
            await _get(client, "/api/materials"),
            await _post(client, "/api/materials/color", {"material_id": "steel", "color": [1, 1, 1]}),
            await _get(client, "/api/cameras"),
            await _post(client, "/api/cameras/alpha/wrist_left/gain", {"value": 1}),
        )

    capabilities, lighting, intensity, reset, materials, color, cameras, gain = _call(commander, scenario)
    assert capabilities == (200, {"success": True, "lighting": False, "materials": False, "cameras": False})
    assert lighting == (404, {"success": False, "message": "lighting is not bound in this launch"})
    assert intensity == lighting
    assert reset == lighting
    assert materials == (404, {"success": False, "message": "materials are not bound in this launch"})
    assert color == materials
    assert cameras == (404, {"success": False, "message": "cameras are not bound in this launch"})
    assert gain == cameras
    # No provider of a vacant slot is ever called.
    assert _requests(commander, commander.lighting, commander.materials, commander.cameras) == {}
    assert [level for level, _, _ in _node_log(commander, caplog)] == [logging.WARNING] * 7


def test_the_page_carries_the_capability_cards_hidden_until_the_capabilities_say_otherwise(commander):
    async def scenario(client):
        response = await client.get("/")
        return response.status, await response.text()

    status, html = _call(commander, scenario)

    assert status == 200
    # Each card is in the page, hidden; its content is markup the script
    # renders from what the provider lists.
    for card, heading in (("lightingCard", "Lighting"), ("materialsCard", "Materials"), ("camerasCard", "Cameras")):
        assert f'<section class="card" id="{card}" hidden>\n<h2>{heading}</h2>' in html
    assert 'api("/api/capabilities")' in html
    assert "if (anyCapability()) {" in html
    for renderer in (
        "function renderTargets(panel)", "function renderCameras()",
        "async function applyProperty(name, index, route)", "async function applyCamera(index, name)",
        "async function resetPanel(name)", "async function resetCamera(index)",
    ):
        assert renderer in html
    assert 'document.visibilityState === "visible"' in html
    assert "const PANEL_REFRESH_MS = 3000;" in html


def test_bound_lighting_reaches_the_page_parsed_and_its_state_is_logged_once(commander, caplog):
    caplog.set_level(logging.INFO)
    commander.bind_lighting()
    commander.lighting.get_lighting.answers = [_no_lighting(), _no_lighting(), _lighting(_SUN), _lighting(_SUN)]

    async def scenario(client):
        capabilities = await _get(client, "/api/capabilities")
        return capabilities, [await _get(client, "/api/lighting") for _ in range(4)]

    capabilities, (first, second, third, fourth) = _call(commander, scenario)
    assert capabilities == (200, {"success": True, "lighting": True, "materials": False, "cameras": False})
    assert first == (503, {"success": False, "message": _NO_SCENE})
    assert second == first
    assert third == (200, {"success": True, "lighting": _lighting_json(_SUN)})
    assert fourth == third
    assert commander.lighting.get_lighting.requests == [(FakeProducer(*_SIMULATION), None)] * 4
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"Scene provider has no lighting: {_NO_SCENE}", None),
        (logging.INFO, "Scene provider lighting ready: 1 lights", None),
    ]


@pytest.mark.parametrize(("route", "payload", "sent", "answer", "summary"), [
    (
        "intensity", {"light_id": "sun", "value": 1200},
        SimpleNamespace(light_id="sun", value=1200.0),
        _set("Sun at 1200 lx", current_value=1200.0),
        "set_light_intensity(light_id=sun, value=1200.0)",
    ),
    (
        "color", {"light_id": "sun", "color": [1, 0.9, 0.8]},
        SimpleNamespace(light_id="sun", color=[1.0, 0.9, 0.8]),
        _set("Sun coloured", current_color=[1.0, 0.9, 0.8]),
        "set_light_color(light_id=sun, color=[1.0, 0.9, 0.8])",
    ),
    (
        "position", {"light_id": "lamp", "position": [1, 2, 3]},
        SimpleNamespace(light_id="lamp", position=[1.0, 2.0, 3.0]),
        _set("Lamp moved", current_position=[1.0, 2.0, 3.0]),
        "set_light_position(light_id=lamp, position=[1.0, 2.0, 3.0])",
    ),
    (
        "direction", {"light_id": "sun", "direction": [0, 0, -1]},
        SimpleNamespace(light_id="sun", direction=[0.0, 0.0, -1.0]),
        _set("Sun turned", current_direction=[0.0, 0.0, -1.0]),
        "set_light_direction(light_id=sun, direction=[0.0, 0.0, -1.0])",
    ),
    (
        "cone", {"light_id": "lamp", "inner_angle": 0.2, "outer_angle": 0.5},
        SimpleNamespace(light_id="lamp", inner_angle=0.2, outer_angle=0.5),
        _set("Lamp cone set", current_inner_angle=0.2, current_outer_angle=0.5),
        "set_light_cone(light_id=lamp, inner_angle=0.2, outer_angle=0.5)",
    ),
    (
        "orientation", {"light_id": "sky", "orientation": [0, 0, 0, 1]},
        SimpleNamespace(light_id="sky", orientation=[0.0, 0.0, 0.0, 1.0]),
        _set("Sky turned", current_orientation=[0.0, 0.0, 0.0, 1.0]),
        "set_light_orientation(light_id=sky, orientation=[0.0, 0.0, 0.0, 1.0])",
    ),
], ids=["intensity", "color", "position", "direction", "cone", "orientation"])
def test_a_light_setter_posts_the_request_and_answers_the_effective_value(
    commander, caplog, route, payload, sent, answer, summary,
):
    caplog.set_level(logging.INFO)
    commander.bind_lighting()
    service = getattr(commander.lighting, f"set_light_{route}")
    service.answers = [answer]

    status, body = _call(commander, lambda client: _post(client, f"/api/lighting/{route}", payload))

    current = {key: value for key, value in vars(answer).items() if key.startswith("current_")}
    assert (status, body) == (200, {"success": True, "message": answer.message, **current})
    assert service.requests == [(FakeProducer(*_SIMULATION), sent)]
    assert _node_log(commander, caplog) == [(logging.INFO, f"{summary}: {answer.message}", None)]


def test_reset_lighting_calls_the_provider_once_and_answers_its_message(commander, caplog):
    caplog.set_level(logging.INFO)
    commander.bind_lighting()
    commander.lighting.reset_lighting.answers = [_set("Lighting of scene/full_warehouse restored")]

    assert _call(commander, lambda client: _post(client, "/api/lighting/reset", {})) == (
        200, {"success": True, "message": "Lighting of scene/full_warehouse restored"},
    )
    assert commander.lighting.reset_lighting.requests == [(FakeProducer(*_SIMULATION), None)]
    assert _node_log(commander, caplog) == [
        (logging.INFO, "reset_lighting(): Lighting of scene/full_warehouse restored", None),
    ]


def test_a_refused_setter_answers_400_with_the_reason_and_what_stands(commander, caplog):
    caplog.set_level(logging.INFO)
    commander.bind_lighting()
    commander.bind_cameras()
    commander.lighting.set_light_intensity.answers = [
        _refused("value 500000 is above the max of 200000 lx", current_value=1000.0),
    ]
    commander.cameras.set_camera_white_balance.answers = [
        _refused("temperature 12000 K is outside 2000-8000 K", current_temperature=4500),
    ]

    async def scenario(client):
        return (
            await _post(client, "/api/lighting/intensity", {"light_id": "sun", "value": 500000}),
            await _post(client, "/api/cameras/alpha/wrist_left/white_balance", {"mode": "manual", "temperature": 12000}),
        )

    intensity, white_balance = _call(commander, scenario)
    assert intensity == (
        400, {"success": False, "message": "value 500000 is above the max of 200000 lx", "current_value": 1000.0},
    )
    assert white_balance == (
        400, {"success": False, "message": "temperature 12000 K is outside 2000-8000 K", "current_temperature": 4500},
    )
    assert _node_log(commander, caplog) == [
        (logging.WARNING, "POST /api/lighting/intensity failed: value 500000 is above the max of 200000 lx", None),
        (
            logging.WARNING,
            "POST /api/cameras/alpha/wrist_left/white_balance failed: temperature 12000 K is outside 2000-8000 K",
            None,
        ),
    ]


@pytest.mark.parametrize(("path", "payload", "message"), [
    ("/api/lighting/intensity", {"light_id": "sun", "value": "bright"}, "value must be a finite number"),
    ("/api/lighting/intensity", {"light_id": "sun", "value": float("nan")}, "value must be a finite number"),
    ("/api/lighting/intensity", {"light_id": "sun", "value": float("inf")}, "value must be a finite number"),
    ("/api/lighting/intensity", {"light_id": "sun", "value": True}, "value must be a finite number"),
    ("/api/lighting/intensity", {"light_id": "", "value": 1}, "light_id must be a non-empty string"),
    ("/api/lighting/intensity", {"value": 1}, "light_id must be a non-empty string"),
    ("/api/lighting/color", {"light_id": "sun", "color": [1, 0.9]}, "color must be 3 finite numbers"),
    ("/api/lighting/color", {"light_id": "sun", "color": [1, 0.9, None]}, "color must be 3 finite numbers"),
    ("/api/lighting/position", {"light_id": "lamp", "position": "1,2,3"}, "position must be 3 finite numbers"),
    ("/api/lighting/direction", {"light_id": "sun", "direction": [0, 0, 1, 0]}, "direction must be 3 finite numbers"),
    ("/api/lighting/cone", {"light_id": "lamp", "inner_angle": 0.2}, "outer_angle must be a finite number"),
    ("/api/lighting/orientation", {"light_id": "sky", "orientation": [0, 0, 0, "1"]}, "orientation must be 4 finite numbers"),
    ("/api/materials/color", {"material_id": "steel", "color": [0.5, 0.5, 0.5, 1]}, "color must be 3 finite numbers"),
    ("/api/materials/finish", {"material_id": "steel", "metallic": True, "roughness": 0.3}, "metallic must be a finite number"),
    ("/api/materials/finish", {"material_id": 7, "metallic": 0.5, "roughness": 0.3}, "material_id must be a non-empty string"),
    ("/api/cameras/alpha/wrist_left/exposure", {"mode": "sometimes", "value": 8000}, 'mode must be "auto" or "manual"'),
    ("/api/cameras/alpha/wrist_left/exposure", {"value": 8000}, 'mode must be "auto" or "manual"'),
    ("/api/cameras/alpha/wrist_left/exposure", {"mode": "manual", "value": 8000.5}, "value must be a whole number"),
    ("/api/cameras/alpha/wrist_left/white_balance", {"mode": "manual"}, "temperature must be a finite number"),
    ("/api/cameras/bravo/chest/gain", {"value": "12"}, "value must be a finite number"),
    ("/api/cameras/bravo/chest/brightness", {"value": 1e400}, "value must be a finite number"),
    ("/api/cameras/bravo/chest/contrast", {"value": [1]}, "value must be a finite number"),
])
def test_invalid_capability_input_answers_400_and_never_reaches_a_provider(commander, caplog, path, payload, message):
    caplog.set_level(logging.INFO)
    commander.bind_lighting()
    commander.bind_materials()
    commander.bind_cameras()

    assert _call(commander, lambda client: _post(client, path, payload)) == (400, {"success": False, "message": message})
    assert _requests(commander, commander.lighting, commander.materials, commander.cameras) == {}
    assert _node_log(commander, caplog) == [(logging.WARNING, f"POST {path} failed: {message}", None)]


def test_an_unknown_property_or_control_is_404_and_reaches_no_provider(commander):
    commander.bind_lighting()
    commander.bind_materials()
    commander.bind_cameras()

    async def scenario(client):
        return (
            await _post(client, "/api/lighting/temperature", {"light_id": "sun", "value": 1}),
            await _post(client, "/api/materials/opacity", {"material_id": "steel", "value": 1}),
            await _post(client, "/api/cameras/alpha/wrist_left/zoom", {"value": 1}),
        )

    assert _call(commander, scenario) == (
        (404, {"success": False, "message": "unknown lighting property: temperature"}),
        (404, {"success": False, "message": "unknown material property: opacity"}),
        (404, {"success": False, "message": "unknown camera control: zoom"}),
    )
    assert _requests(commander, commander.lighting, commander.materials, commander.cameras) == {}


def test_bound_materials_reach_the_page_and_their_setters_address_the_material(commander, caplog):
    caplog.set_level(logging.INFO)
    commander.bind_materials()
    color = commander.materials.set_material_color
    color.answers = [_set("Brushed steel coloured", current_color=[0.2, 0.2, 0.8])]
    finish = commander.materials.set_material_finish
    finish.answers = [_set("Brushed steel finished", current_metallic=0.5, current_roughness=0.7)]
    reset = commander.materials.reset_materials
    reset.answers = [_set("Materials restored")]

    async def scenario(client):
        return (
            await _get(client, "/api/capabilities"),
            await _get(client, "/api/materials"),
            await _post(client, "/api/materials/color", {"material_id": "steel", "color": [0.2, 0.2, 0.8]}),
            await _post(client, "/api/materials/finish", {"material_id": "steel", "metallic": 0.5, "roughness": 0.7}),
            await _post(client, "/api/materials/reset", {}),
        )

    capabilities, materials, coloured, finished, restored = _call(commander, scenario)
    assert capabilities == (200, {"success": True, "lighting": False, "materials": True, "cameras": False})
    assert materials == (200, {"success": True, "materials": {"materials": [_STEEL]}})
    assert coloured == (200, {"success": True, "message": "Brushed steel coloured", "current_color": [0.2, 0.2, 0.8]})
    assert finished == (
        200, {"success": True, "message": "Brushed steel finished", "current_metallic": 0.5, "current_roughness": 0.7},
    )
    assert restored == (200, {"success": True, "message": "Materials restored"})
    producer = FakeProducer(*_SIMULATION)
    assert color.requests == [(producer, SimpleNamespace(material_id="steel", color=[0.2, 0.2, 0.8]))]
    assert finish.requests == [(producer, SimpleNamespace(material_id="steel", metallic=0.5, roughness=0.7))]
    assert reset.requests == [(producer, None)]
    assert _node_log(commander, caplog) == [
        (logging.INFO, "Scene provider materials ready: 1 materials", None),
        (logging.INFO, "set_material_color(material_id=steel, color=[0.2, 0.2, 0.8]): Brushed steel coloured", None),
        (logging.INFO, "set_material_finish(material_id=steel, metallic=0.5, roughness=0.7): Brushed steel finished", None),
        (logging.INFO, "reset_materials(): Materials restored", None),
    ]


def test_materials_answer_503_with_the_provider_reason_until_a_scene_is_loaded(commander, caplog):
    caplog.set_level(logging.INFO)
    commander.bind_materials()
    commander.materials.get_materials.answers = [
        SimpleNamespace(success=False, message=_NO_SCENE, materials_json=""), _materials(_STEEL),
    ]

    async def scenario(client):
        return [await _get(client, "/api/materials") for _ in range(2)]

    assert _call(commander, scenario) == [
        (503, {"success": False, "message": _NO_SCENE}),
        (200, {"success": True, "materials": {"materials": [_STEEL]}}),
    ]
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"Scene provider has no materials: {_NO_SCENE}", None),
        (logging.INFO, "Scene provider materials ready: 1 materials", None),
    ]


def test_the_simulation_lists_every_robots_cameras_and_each_control_names_its_camera(commander, caplog):
    caplog.set_level(logging.INFO)
    simulation = commander.bind_cameras()
    commander.cameras.set_camera_exposure.answers = [
        _set("alpha/wrist_left exposure set to 8000", current_value=8000),
        _set("bravo/chest exposure set to auto", current_value=8300),
    ]
    commander.cameras.set_camera_white_balance.answers = [
        _set("bravo/chest white balance set to 4500", current_temperature=4500),
    ]
    commander.cameras.set_camera_gain.answers = [_set("alpha/wrist_left gain set to 12", current_value=12)]
    commander.cameras.set_camera_brightness.answers = [_set("bravo/chest brightness set to 100", current_value=100)]
    commander.cameras.set_camera_contrast.answers = [_set("alpha/wrist_left contrast set to 40", current_value=40)]
    commander.cameras.reset_camera.answers = [_set("alpha/wrist_left restored to its device defaults")]

    async def scenario(client):
        return (
            await _get(client, "/api/capabilities"),
            await _get(client, "/api/cameras"),
            await _post(client, "/api/cameras/alpha/wrist_left/exposure", {"mode": "manual", "value": 8000}),
            await _post(client, "/api/cameras/bravo/chest/exposure", {"mode": "auto", "value": 0}),
            await _post(client, "/api/cameras/bravo/chest/white_balance", {"mode": "manual", "temperature": 4500}),
            await _post(client, "/api/cameras/alpha/wrist_left/gain", {"value": 12}),
            await _post(client, "/api/cameras/bravo/chest/brightness", {"value": 100}),
            await _post(client, "/api/cameras/alpha/wrist_left/contrast", {"value": 40}),
            await _post(client, "/api/cameras/alpha/wrist_left/reset", {}),
        )

    capabilities, cameras, *answers = _call(commander, scenario)
    assert capabilities == (200, {"success": True, "lighting": False, "materials": False, "cameras": True})
    # Every robot's cameras, as the simulation lists them, each with the
    # profile the simulation describes for it.
    assert cameras == (200, {"success": True, "message": "2 cameras rendered", "count": 2, "cameras": [
        {
            "id": "alpha/wrist_left", "robot": "alpha", "camera": "wrist_left", "kind": "rgb",
            "info": {"width": 1280, "height": 720, "frames_per_second": 30, "encoding": "rgb8"},
            "profile": _PROFILE, "profile_message": "profile of Logitech C920",
        },
        {
            "id": "bravo/chest", "robot": "bravo", "camera": "chest", "kind": "rgbd",
            "info": {"width": 640, "height": 480, "frames_per_second": 15, "encoding": "rgb8"},
            "profile": _PROFILE, "profile_message": "profile of Logitech C920",
        },
    ]})
    assert answers == [
        (200, {"success": True, "message": "alpha/wrist_left exposure set to 8000", "current_value": 8000}),
        (200, {"success": True, "message": "bravo/chest exposure set to auto", "current_value": 8300}),
        (200, {"success": True, "message": "bravo/chest white balance set to 4500", "current_temperature": 4500}),
        (200, {"success": True, "message": "alpha/wrist_left gain set to 12", "current_value": 12}),
        (200, {"success": True, "message": "bravo/chest brightness set to 100", "current_value": 100}),
        (200, {"success": True, "message": "alpha/wrist_left contrast set to 40", "current_value": 40}),
        (200, {"success": True, "message": "alpha/wrist_left restored to its device defaults"}),
    ]
    # Every call went to the simulation, naming the camera by its robot and
    # slot.
    assert _requests(commander, commander.cameras) == {
        "cameras.get_cameras": [(simulation, None)],
        "cameras.describe_camera": [
            (simulation, SimpleNamespace(robot="alpha", camera="wrist_left")),
            (simulation, SimpleNamespace(robot="bravo", camera="chest")),
        ],
        "cameras.set_camera_exposure": [
            (simulation, SimpleNamespace(robot="alpha", camera="wrist_left", mode="manual", value=8000)),
            (simulation, SimpleNamespace(robot="bravo", camera="chest", mode="auto", value=0)),
        ],
        "cameras.set_camera_white_balance": [
            (simulation, SimpleNamespace(robot="bravo", camera="chest", mode="manual", temperature=4500)),
        ],
        "cameras.set_camera_gain": [(simulation, SimpleNamespace(robot="alpha", camera="wrist_left", value=12))],
        "cameras.set_camera_brightness": [(simulation, SimpleNamespace(robot="bravo", camera="chest", value=100))],
        "cameras.set_camera_contrast": [(simulation, SimpleNamespace(robot="alpha", camera="wrist_left", value=40))],
        "cameras.reset_camera": [(simulation, SimpleNamespace(robot="alpha", camera="wrist_left"))],
    }
    assert _node_log(commander, caplog) == [
        (logging.INFO, "Camera alpha/wrist_left profile ready: 1 controls", None),
        (logging.INFO, "Camera bravo/chest profile ready: 1 controls", None),
        (
            logging.INFO,
            "set_camera_exposure(robot=alpha, camera=wrist_left, mode=manual, value=8000): "
            "alpha/wrist_left exposure set to 8000",
            None,
        ),
        (
            logging.INFO,
            "set_camera_exposure(robot=bravo, camera=chest, mode=auto, value=0): bravo/chest exposure set to auto",
            None,
        ),
        (
            logging.INFO,
            "set_camera_white_balance(robot=bravo, camera=chest, mode=manual, temperature=4500): "
            "bravo/chest white balance set to 4500",
            None,
        ),
        (logging.INFO, "set_camera_gain(robot=alpha, camera=wrist_left, value=12): alpha/wrist_left gain set to 12", None),
        (
            logging.INFO,
            "set_camera_brightness(robot=bravo, camera=chest, value=100): bravo/chest brightness set to 100",
            None,
        ),
        (
            logging.INFO,
            "set_camera_contrast(robot=alpha, camera=wrist_left, value=40): alpha/wrist_left contrast set to 40",
            None,
        ),
        (
            logging.INFO,
            "reset_camera(robot=alpha, camera=wrist_left): alpha/wrist_left restored to its device defaults",
            None,
        ),
    ]


def test_a_camera_the_simulation_does_not_render_is_refused_with_its_reason(commander, caplog):
    caplog.set_level(logging.INFO)
    commander.bind_cameras()
    commander.cameras.set_camera_gain.answers = [
        _refused("robot 'charlie' renders no camera 'wrist_left'; get_cameras lists the cameras rendered",
                 current_value=-1),
    ]
    commander.cameras.get_cameras.answers = [_cameras()]

    async def scenario(client):
        return (
            await _get(client, "/api/cameras"),
            await _post(client, "/api/cameras/charlie/wrist_left/gain", {"value": 12}),
        )

    cameras, gain = _call(commander, scenario)
    # No robot standing holds a camera pair: the list is empty, not refused.
    assert cameras == (200, {"success": True, "message": "0 cameras rendered", "cameras": [], "count": 0})
    assert gain == (400, {
        "success": False,
        "message": "robot 'charlie' renders no camera 'wrist_left'; get_cameras lists the cameras rendered",
        "current_value": -1,
    })


def test_a_profile_the_camera_cannot_give_yet_is_reported_with_its_reason(commander, caplog):
    caplog.set_level(logging.INFO)
    commander.bind_cameras()
    commander.cameras.get_cameras.answers = [_cameras(_listed("alpha", "chest", "rgbd", 640, 480, 15))]
    commander.cameras.describe_camera.answers = [_no_profile(), _no_profile(), _profile()]

    async def scenario(client):
        return [await _get(client, "/api/cameras") for _ in range(3)]

    first, second, third = _call(commander, scenario)
    assert first[0] == 200
    assert first[1]["cameras"] == [{
        "id": "alpha/chest", "robot": "alpha", "camera": "chest", "kind": "rgbd",
        "info": {"width": 640, "height": 480, "frames_per_second": 15, "encoding": "rgb8"},
        "profile": None, "profile_message": _NOT_ATTACHED,
    }]
    assert second == first
    assert third[1]["cameras"][0]["profile"] == _PROFILE
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"Camera alpha/chest has no profile: {_NOT_ATTACHED}", None),
        (logging.INFO, "Camera alpha/chest profile ready: 1 controls", None),
    ]


def test_a_camera_transport_failure_is_a_server_error_logged_in_one_line(commander, caplog):
    caplog.set_level(logging.INFO)
    commander.bind_cameras()
    commander.cameras.get_cameras.answers = [TimeoutError("get_cameras timed out")]

    assert _call(commander, lambda client: _get(client, "/api/cameras")) == (
        500, {"success": False, "message": "get_cameras timed out"},
    )
    assert _node_log(commander, caplog) == [
        (logging.WARNING, "GET /api/cameras failed: get_cameras timed out", None),
    ]


def test_setup_logs_the_bound_capabilities_and_calls_none_of_them(commander, node_runner, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    commander.bind_lighting()
    commander.bind_materials()
    commander.bind_cameras()
    monkeypatch.setattr(commander.module.listen, "bind_listener", lambda *bound: _FakeListener("127.0.0.1", 9000))
    monkeypatch.setattr(commander.module.listen, "start_serving", _serve_nothing)

    async def run():
        await asyncio.gather(*await commander.module.setup(_parameters(9000, "127.0.0.1"), node_runner))

    asyncio.run(run())
    assert _node_log(commander, caplog) == [
        (logging.INFO, "Scene commander starting", None),
        (logging.INFO, "Scene panel bound at 127.0.0.1:9000", None),
        (logging.INFO, "Capabilities: lighting, materials, cameras", None),
        (logging.INFO, "Scene provider catalogue ready: 1 assets", None),
        (logging.INFO, "Scene provider object state ready: 0 runtime objects", None),
    ]
    # The slots say what is bound; the providers are asked once the page asks.
    assert _requests(commander, commander.lighting, commander.materials, commander.cameras) == {}
