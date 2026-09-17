"""Drive the commander's HTTP API against a fake scene provider, without Peppy."""

import asyncio
import enum
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from aiohttp.test_utils import TestClient, TestServer

_SOURCE = Path(__file__).resolve().parents[1] / "src" / "scene_commander" / "__main__.py"
_ACTIONS = ("apply_force", "clear_scene", "load_scene", "move_object", "move_robot", "remove_object", "spawn_object")
# The services of every slot a launch may leave vacant, by link id.
_OPTIONAL_LINKS = {
    "lighting": (
        "get_lighting", "set_light_intensity", "set_light_color", "set_light_position", "set_light_direction",
        "set_light_cone", "set_light_orientation", "reset_lighting",
    ),
    "materials": ("get_materials", "set_material_color", "set_material_finish", "reset_materials"),
    "color_cameras": ("video_stream_info", "set_exposure", "set_white_balance", "set_gain", "set_brightness", "set_contrast"),
    "rgbd_cameras": (
        "video_stream_info", "set_color_exposure", "set_color_white_balance", "set_color_gain", "set_color_brightness",
        "set_color_contrast",
    ),
    "camera_profiles": ("get_camera_profile", "reset_camera"),
}
_BUSY = "Isaac is still discovering its asset catalogue"
_NOT_READY = "Isaac has not loaded its stage yet"
_NO_SCENE = "no scene is loaded"
_NOT_ATTACHED = "the camera is not attached to its simulation yet"
_SCENE = {"asset_id": "scene/full_warehouse", "display_name": "Full Warehouse", "kind": "scene", "category": "Scenes"}
_ROBOTS = [
    {"robot": "alpha_init_inst", "model": "openarm_v2", "position": [0.0, 0.0, 0.0], "attached": True},
    {"robot": "bravo_init_inst", "model": "openarm_v1", "position": [0.0, -1.5, 0.0], "attached": True},
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


class Status(enum.Enum):
    COMPLETED = "completed"
    ABORTED = "aborted"


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
    """A consumed action: fire_goal() records the goal and get_result() answers."""

    def __init__(self, name):
        super().__init__(f"peppygen.consumed_actions.simulation.{name}")
        self.ResultStatus = Status
        self.GoalRequest = lambda **fields: SimpleNamespace(**fields)
        self.goals = []
        self.accepted = True
        self.reason = ""
        self.result = SimpleNamespace(
            status=Status.COMPLETED,
            data=SimpleNamespace(success=True, message=f"{name} done", object_id="obj_1"),
        )
        action = self

        class ActionHandle:
            @staticmethod
            async def fire_goal(node_runner, producer, *goal, timeout, feedback_qos):
                assert feedback_qos == "standard"
                action.goals.append(goal[0] if goal else None)
                return SimpleNamespace(accepted=action.accepted, reason=action.reason, get_result=action.get_result)

        self.ActionHandle = ActionHandle

    def bound_producer(self, node_runner):
        return "producer"

    async def get_result(self, timeout):
        return self.result


@pytest.fixture
def commander(monkeypatch):
    services = _link("simulation", ["get_assets_list", "get_robots_list"], producers=["producer"])
    services.get_assets_list.answers = [_catalogue(_SCENE)]
    services.get_robots_list.answers = [SimpleNamespace(
        success=True, message="2 robots standing", robots_json=json.dumps(_ROBOTS)
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
    optional["color_cameras"].video_stream_info.answers = [_stream(1280, 720, 30, "rgb8")]
    optional["rgbd_cameras"].video_stream_info.answers = [_stream(640, 480, 15, "rgb8")]
    optional["camera_profiles"].get_camera_profile.answers = [_profile()]
    actions = ModuleType("peppygen.consumed_actions.simulation")
    for name in _ACTIONS:
        setattr(actions, name, FakeAction(name))
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

    def bind_camera(instance_id, kind, profile=True):
        """Link a camera relay of `kind` here and, with `profile`, the profile the same instance serves."""
        producer = FakeProducer("alpha", instance_id)
        optional["color_cameras" if kind == "rgb" else "rgbd_cameras"].producers.append(producer)
        if profile:
            optional["camera_profiles"].producers.append(producer)
        return producer

    return SimpleNamespace(
        module=module, services=services, objects=objects, actions=actions,
        bind_lighting=bind_lighting, bind_materials=bind_materials, bind_camera=bind_camera, **optional,
    )


class _FakeListener:
    """A bound socket, for the wiring around it; the real one is in test_http_port."""

    def __init__(self, host, port):
        self._name = (host, port)

    def getsockname(self):
        return self._name


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


def _stream(width, height, frames_per_second, encoding):
    return SimpleNamespace(width=width, height=height, frames_per_second=frames_per_second, encoding=encoding)


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


def _call(commander, scenario, app=None):
    async def run():
        async with TestClient(TestServer(app or commander.module._build_app(object()))) as client:
            return await scenario(client)

    return asyncio.run(run())


async def _get(client, path):
    response = await client.get(path)
    return response.status, await response.json()


async def _post(client, path, payload):
    response = await client.post(path, json=payload)
    return response.status, await response.json()


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


def test_load_scene_fires_the_goal_and_logs_the_provider_answer(commander, caplog):
    caplog.set_level(logging.INFO)
    load = commander.actions.load_scene
    load.result.data.message = "Loaded scene scene/full_warehouse"

    status, body = _call(commander, lambda client: _post(
        client, "/api/scene/load", {"asset_id": "scene/full_warehouse", "scale": 1.5},
    ))

    assert (status, body) == (200, {"success": True, "message": "Loaded scene scene/full_warehouse"})
    assert load.goals == [SimpleNamespace(asset_id="scene/full_warehouse", scale=1.5)]
    assert _node_log(commander, caplog) == [
        (logging.INFO, "load_scene(asset_id=scene/full_warehouse, scale=1.5): Loaded scene scene/full_warehouse", None),
    ]


def _reject(action):
    action.accepted = False
    action.reason = "another goal is running"


def _abort(action):
    action.result.status = Status.ABORTED


def _fail(action):
    action.result.data = SimpleNamespace(success=False, message="Unknown asset_id: scene/none")


def _no_data(action):
    action.result.data = None


@pytest.mark.parametrize(("outcome", "message"), [
    (_reject, "load_scene rejected: another goal is running"),
    (_abort, "load_scene did not complete: ABORTED"),
    (_fail, "Unknown asset_id: scene/none"),
    (_no_data, "load_scene completed without result data"),
], ids=["rejected", "aborted", "failed", "no-data"])
def test_refused_goals_answer_400_with_the_reason_in_one_log_line(commander, caplog, outcome, message):
    caplog.set_level(logging.INFO)
    outcome(commander.actions.load_scene)

    assert _call(commander, lambda client: _post(
        client, "/api/scene/load", {"asset_id": "scene/none", "scale": 1.0},
    )) == (400, {"success": False, "message": message})
    assert _node_log(commander, caplog) == [
        (logging.WARNING, f"POST /api/scene/load failed: {message}", None),
    ]


def test_spawn_object_reports_the_minted_object_id(commander, caplog):
    caplog.set_level(logging.INFO)
    spawn = commander.actions.spawn_object
    spawn.result.data.object_id = "obj_42"

    status, body = _call(commander, lambda client: _post(client, "/api/objects/spawn", {
        "asset_id": "props/blocks/red_block", "position": [0.5, 0, 0.8], "physics": "dynamic", "mass": 0.2,
    }))

    assert (status, body) == (200, {"success": True, "message": "spawn_object done", "object_id": "obj_42"})
    assert spawn.goals == [SimpleNamespace(
        asset_id="props/blocks/red_block", position=[0.5, 0.0, 0.8], scale=1.0, physics="dynamic", mass=0.2,
    )]
    assert _node_log(commander, caplog) == [(
        logging.INFO,
        "spawn_object(asset_id=props/blocks/red_block, position=[0.5, 0.0, 0.8], physics=dynamic, mass=0.2): "
        "spawn_object done",
        None,
    )]


def test_invalid_input_is_a_400_that_never_reaches_the_provider(commander, caplog):
    caplog.set_level(logging.INFO)

    assert _call(commander, lambda client: _post(
        client, "/api/objects/spawn", {"asset_id": "props/blocks/red_block", "position": [1, 2]},
    )) == (400, {"success": False, "message": "position must be 3 finite numbers"})
    assert commander.actions.spawn_object.goals == []
    assert _node_log(commander, caplog) == [
        (logging.WARNING, "POST /api/objects/spawn failed: position must be 3 finite numbers", None),
    ]


def test_setup_tolerates_a_provider_without_catalogue_but_not_an_unreachable_one(commander, caplog, monkeypatch):
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
        await asyncio.gather(*await commander.module.setup(_parameters(9000, "127.0.0.1"), object()))

    asyncio.run(run())
    [(bound, routes)] = served
    assert bound is listener
    assert routes >= {"/", "/api/assets", "/api/scene/load"}
    assert _node_log(commander, caplog) == [
        (logging.INFO, "Scene commander starting", None),
        # The bound address reaches the operator here and nowhere else, and
        # the socket is taken before the provider is waited on.
        (logging.INFO, "Scene panel at http://127.0.0.1:9100 (bound 127.0.0.1:9100)", None),
        (logging.INFO, "Capabilities: scene manipulation only", None),
        (logging.INFO, f"Scene provider has no catalogue: {_BUSY}", None),
        (logging.INFO, "Scene provider object state ready: 0 runtime objects", None),
    ]

    commander.objects.get_object_states.answers = [TimeoutError("get_object_states timed out")]
    with pytest.raises(TimeoutError, match="get_object_states timed out"):
        asyncio.run(commander.module.setup(_parameters(9000, "127.0.0.1"), object()))
    assert len(served) == 1


def test_setup_tolerates_a_provider_without_object_state_and_the_page_asks_again(commander, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    served = []
    monkeypatch.setattr(commander.module.listen, "bind_listener", lambda *bound: _FakeListener("127.0.0.1", 9000))

    async def fake_server(app, bound):
        served.append(app)
        return asyncio.create_task(asyncio.sleep(0))

    monkeypatch.setattr(commander.module.listen, "start_serving", fake_server)
    commander.objects.get_object_states.answers = [_no_object_state()]

    async def run():
        await asyncio.gather(*await commander.module.setup(_parameters(9000, "127.0.0.1"), object()))

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


def test_setup_refuses_a_launch_it_cannot_serve_and_starts_no_server(commander, monkeypatch):
    # A node that never binds must fail its launch, not stand ready with
    # nothing listening.
    served = []
    monkeypatch.setattr(commander.module.listen, "start_serving", lambda *started: served.append(started))

    with pytest.raises(ValueError, match="http_host"):
        asyncio.run(commander.module.setup(_parameters(9000, http_host="localhost"), object()))

    assert served == [], "a commander that cannot serve must start no server"


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
    assert [robot["robot"] for robot in body["robots"]] == [
        "alpha_init_inst",
        "bravo_init_inst",
    ]


def test_a_move_carries_the_robot_it_addresses(commander):
    status, _ = _call(commander, lambda client: _post(
        client, "/api/robot/move", {"robot": "bravo_init_inst", "position": [1.0, -1.5, 0.0]}
    ))

    assert status == 200
    goal = commander.actions.move_robot.goals[-1]
    assert goal.robot == "bravo_init_inst"
    assert goal.position == [1.0, -1.5, 0.0]


def test_a_move_naming_no_robot_is_refused_before_the_simulation(commander):
    status, _ = _call(commander, lambda client: _post(
        client, "/api/robot/move", {"position": [1.0, 0.0, 0.0]}
    ))

    assert status == 400
    assert commander.actions.move_robot.goals == []


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
            await _post(client, "/api/cameras/alpha_wrist_left/gain", {"value": 1}),
        )

    capabilities, lighting, intensity, reset, materials, color, cameras, gain = _call(commander, scenario)
    assert capabilities == (200, {"success": True, "lighting": False, "materials": False, "cameras": []})
    assert lighting == (404, {"success": False, "message": "lighting is not bound in this launch"})
    assert intensity == lighting
    assert reset == lighting
    assert materials == (404, {"success": False, "message": "materials are not bound in this launch"})
    assert color == materials
    assert cameras == (200, {"success": True, "cameras": [], "count": 0})
    assert gain == (404, {"success": False, "message": "no camera alpha_wrist_left is bound in this launch"})
    # No provider of a vacant slot is ever called.
    assert _requests(commander, commander.lighting, commander.materials, commander.color_cameras,
                     commander.rgbd_cameras, commander.camera_profiles) == {}
    assert [level for level, _, _ in _node_log(commander, caplog)] == [logging.WARNING] * 6


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
    assert capabilities == (200, {"success": True, "lighting": True, "materials": False, "cameras": []})
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
    commander.bind_camera("alpha_wrist_left", "rgb")
    commander.lighting.set_light_intensity.answers = [
        _refused("value 500000 is above the max of 200000 lx", current_value=1000.0),
    ]
    commander.color_cameras.set_white_balance.answers = [
        _refused("temperature 12000 K is outside 2000-8000 K", current_temperature=4500),
    ]

    async def scenario(client):
        return (
            await _post(client, "/api/lighting/intensity", {"light_id": "sun", "value": 500000}),
            await _post(client, "/api/cameras/alpha_wrist_left/white_balance", {"mode": "manual", "temperature": 12000}),
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
            "POST /api/cameras/alpha_wrist_left/white_balance failed: temperature 12000 K is outside 2000-8000 K",
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
    ("/api/cameras/alpha_wrist_left/exposure", {"mode": "sometimes", "value": 8000}, 'mode must be "auto" or "manual"'),
    ("/api/cameras/alpha_wrist_left/exposure", {"value": 8000}, 'mode must be "auto" or "manual"'),
    ("/api/cameras/alpha_wrist_left/exposure", {"mode": "manual", "value": 8000.5}, "value must be a whole number"),
    ("/api/cameras/alpha_wrist_left/white_balance", {"mode": "manual"}, "temperature must be a finite number"),
    ("/api/cameras/alpha_chest/gain", {"value": "12"}, "value must be a finite number"),
    ("/api/cameras/alpha_chest/brightness", {"value": 1e400}, "value must be a finite number"),
    ("/api/cameras/alpha_chest/contrast", {"value": [1]}, "value must be a finite number"),
])
def test_invalid_capability_input_answers_400_and_never_reaches_a_provider(commander, caplog, path, payload, message):
    caplog.set_level(logging.INFO)
    commander.bind_lighting()
    commander.bind_materials()
    commander.bind_camera("alpha_wrist_left", "rgb")
    commander.bind_camera("alpha_chest", "rgbd")

    assert _call(commander, lambda client: _post(client, path, payload)) == (400, {"success": False, "message": message})
    assert _requests(commander, commander.lighting, commander.materials, commander.color_cameras,
                     commander.rgbd_cameras, commander.camera_profiles) == {}
    assert _node_log(commander, caplog) == [(logging.WARNING, f"POST {path} failed: {message}", None)]


def test_an_unknown_property_or_control_is_404_and_reaches_no_provider(commander):
    commander.bind_lighting()
    commander.bind_materials()
    commander.bind_camera("alpha_wrist_left", "rgb")

    async def scenario(client):
        return (
            await _post(client, "/api/lighting/temperature", {"light_id": "sun", "value": 1}),
            await _post(client, "/api/materials/opacity", {"material_id": "steel", "value": 1}),
            await _post(client, "/api/cameras/alpha_wrist_left/zoom", {"value": 1}),
        )

    assert _call(commander, scenario) == (
        (404, {"success": False, "message": "unknown lighting property: temperature"}),
        (404, {"success": False, "message": "unknown material property: opacity"}),
        (404, {"success": False, "message": "unknown camera control: zoom"}),
    )
    assert _requests(commander, commander.lighting, commander.materials, commander.color_cameras) == {}


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
    assert capabilities == (200, {"success": True, "lighting": False, "materials": True, "cameras": []})
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


def test_cameras_list_their_stream_and_profile_and_route_each_control_by_kind(commander, caplog):
    caplog.set_level(logging.INFO)
    wrist = commander.bind_camera("alpha_wrist_left", "rgb")
    chest = commander.bind_camera("alpha_chest", "rgbd")
    commander.color_cameras.set_exposure.answers = [_set("Wrist exposure 8000 us", current_value=8000)]
    commander.rgbd_cameras.set_color_exposure.answers = [_set("Chest exposure auto", current_value=8300)]
    commander.rgbd_cameras.set_color_white_balance.answers = [_set("Chest white balance 4500 K", current_temperature=4500)]
    commander.color_cameras.set_gain.answers = [_set("Wrist gain 12", current_value=12)]
    commander.rgbd_cameras.set_color_brightness.answers = [_set("Chest brightness 100", current_value=100)]
    commander.color_cameras.set_contrast.answers = [_set("Wrist contrast 40", current_value=40)]
    commander.camera_profiles.reset_camera.answers = [_set("Wrist camera reset")]

    async def scenario(client):
        return (
            await _get(client, "/api/capabilities"),
            await _get(client, "/api/cameras"),
            await _post(client, "/api/cameras/alpha_wrist_left/exposure", {"mode": "manual", "value": 8000}),
            await _post(client, "/api/cameras/alpha_chest/exposure", {"mode": "auto", "value": 0}),
            await _post(client, "/api/cameras/alpha_chest/white_balance", {"mode": "manual", "temperature": 4500}),
            await _post(client, "/api/cameras/alpha_wrist_left/gain", {"value": 12}),
            await _post(client, "/api/cameras/alpha_chest/brightness", {"value": 100}),
            await _post(client, "/api/cameras/alpha_wrist_left/contrast", {"value": 40}),
            await _post(client, "/api/cameras/alpha_wrist_left/reset", {}),
            await _post(client, "/api/cameras/alpha_head/gain", {"value": 1}),
        )

    capabilities, cameras, *answers = _call(commander, scenario)
    assert capabilities == (200, {"success": True, "lighting": False, "materials": False, "cameras": [
        {"id": "alpha_wrist_left", "kind": "rgb", "profile": True},
        {"id": "alpha_chest", "kind": "rgbd", "profile": True},
    ]})
    assert cameras == (200, {"success": True, "count": 2, "cameras": [
        {
            "id": "alpha_wrist_left", "kind": "rgb",
            "info": {"width": 1280, "height": 720, "frames_per_second": 30, "encoding": "rgb8"},
            "profile": _PROFILE, "profile_message": "profile of Logitech C920",
        },
        {
            "id": "alpha_chest", "kind": "rgbd",
            "info": {"width": 640, "height": 480, "frames_per_second": 15, "encoding": "rgb8"},
            "profile": _PROFILE, "profile_message": "profile of Logitech C920",
        },
    ]})
    assert answers == [
        (200, {"success": True, "message": "Wrist exposure 8000 us", "current_value": 8000}),
        (200, {"success": True, "message": "Chest exposure auto", "current_value": 8300}),
        (200, {"success": True, "message": "Chest white balance 4500 K", "current_temperature": 4500}),
        (200, {"success": True, "message": "Wrist gain 12", "current_value": 12}),
        (200, {"success": True, "message": "Chest brightness 100", "current_value": 100}),
        (200, {"success": True, "message": "Wrist contrast 40", "current_value": 40}),
        (200, {"success": True, "message": "Wrist camera reset"}),
        (404, {"success": False, "message": "no camera alpha_head is bound in this launch"}),
    ]
    # Each control went to the service set of the camera's kind, on that
    # camera's instance, and nowhere else.
    assert _requests(commander, commander.color_cameras, commander.rgbd_cameras, commander.camera_profiles) == {
        "color_cameras.video_stream_info": [(wrist, None)],
        "color_cameras.set_exposure": [(wrist, SimpleNamespace(mode="manual", value=8000))],
        "color_cameras.set_gain": [(wrist, SimpleNamespace(value=12))],
        "color_cameras.set_contrast": [(wrist, SimpleNamespace(value=40))],
        "rgbd_cameras.video_stream_info": [(chest, None)],
        "rgbd_cameras.set_color_exposure": [(chest, SimpleNamespace(mode="auto", value=0))],
        "rgbd_cameras.set_color_white_balance": [(chest, SimpleNamespace(mode="manual", temperature=4500))],
        "rgbd_cameras.set_color_brightness": [(chest, SimpleNamespace(value=100))],
        "camera_profiles.get_camera_profile": [(wrist, None), (chest, None)],
        "camera_profiles.reset_camera": [(wrist, None)],
    }
    assert _node_log(commander, caplog) == [
        (logging.INFO, "Camera alpha_wrist_left profile ready: 1 controls", None),
        (logging.INFO, "Camera alpha_chest profile ready: 1 controls", None),
        (logging.INFO, "set_exposure(camera=alpha_wrist_left, mode=manual, value=8000): Wrist exposure 8000 us", None),
        (logging.INFO, "set_color_exposure(camera=alpha_chest, mode=auto, value=0): Chest exposure auto", None),
        (
            logging.INFO,
            "set_color_white_balance(camera=alpha_chest, mode=manual, temperature=4500): Chest white balance 4500 K",
            None,
        ),
        (logging.INFO, "set_gain(camera=alpha_wrist_left, value=12): Wrist gain 12", None),
        (logging.INFO, "set_color_brightness(camera=alpha_chest, value=100): Chest brightness 100", None),
        (logging.INFO, "set_contrast(camera=alpha_wrist_left, value=40): Wrist contrast 40", None),
        (logging.INFO, "reset_camera(camera=alpha_wrist_left): Wrist camera reset", None),
        (logging.WARNING, "POST /api/cameras/alpha_head/gain failed: no camera alpha_head is bound in this launch", None),
    ]


def test_a_camera_without_a_profile_lists_its_stream_and_has_no_reset(commander, caplog):
    caplog.set_level(logging.INFO)
    wrist = commander.bind_camera("alpha_wrist_left", "rgb", profile=False)
    commander.color_cameras.set_gain.answers = [_set("Wrist gain 12", current_value=12)]

    async def scenario(client):
        return (
            await _get(client, "/api/capabilities"),
            await _get(client, "/api/cameras"),
            await _post(client, "/api/cameras/alpha_wrist_left/gain", {"value": 12}),
            await _post(client, "/api/cameras/alpha_wrist_left/reset", {}),
        )

    capabilities, cameras, gain, reset = _call(commander, scenario)
    assert capabilities[1]["cameras"] == [{"id": "alpha_wrist_left", "kind": "rgb", "profile": False}]
    assert cameras == (200, {"success": True, "count": 1, "cameras": [{
        "id": "alpha_wrist_left", "kind": "rgb",
        "info": {"width": 1280, "height": 720, "frames_per_second": 30, "encoding": "rgb8"},
        "profile": None, "profile_message": "no camera profile is bound for alpha_wrist_left in this launch",
    }]})
    # The controls are the camera contract's own: they work without a profile.
    assert gain == (200, {"success": True, "message": "Wrist gain 12", "current_value": 12})
    assert reset == (404, {"success": False, "message": "no camera profile is bound for alpha_wrist_left in this launch"})
    assert _requests(commander, commander.color_cameras, commander.camera_profiles) == {
        "color_cameras.video_stream_info": [(wrist, None)],
        "color_cameras.set_gain": [(wrist, SimpleNamespace(value=12))],
    }
    assert _node_log(commander, caplog) == [
        (logging.INFO, "set_gain(camera=alpha_wrist_left, value=12): Wrist gain 12", None),
        (
            logging.WARNING,
            "POST /api/cameras/alpha_wrist_left/reset failed: "
            "no camera profile is bound for alpha_wrist_left in this launch",
            None,
        ),
    ]


def test_a_profile_the_camera_cannot_give_yet_is_reported_with_its_reason(commander, caplog):
    caplog.set_level(logging.INFO)
    commander.bind_camera("alpha_chest", "rgbd")
    commander.camera_profiles.get_camera_profile.answers = [_no_profile(), _no_profile(), _profile()]

    async def scenario(client):
        return [await _get(client, "/api/cameras") for _ in range(3)]

    first, second, third = _call(commander, scenario)
    assert first[0] == 200
    assert first[1]["cameras"] == [{
        "id": "alpha_chest", "kind": "rgbd",
        "info": {"width": 640, "height": 480, "frames_per_second": 15, "encoding": "rgb8"},
        "profile": None, "profile_message": _NOT_ATTACHED,
    }]
    assert second == first
    assert third[1]["cameras"][0]["profile"] == _PROFILE
    assert _node_log(commander, caplog) == [
        (logging.INFO, f"Camera alpha_chest has no profile: {_NOT_ATTACHED}", None),
        (logging.INFO, "Camera alpha_chest profile ready: 1 controls", None),
    ]


def test_a_camera_transport_failure_is_a_server_error_logged_in_one_line(commander, caplog):
    caplog.set_level(logging.INFO)
    commander.bind_camera("alpha_wrist_left", "rgb")
    commander.color_cameras.video_stream_info.answers = [TimeoutError("video_stream_info timed out")]

    assert _call(commander, lambda client: _get(client, "/api/cameras")) == (
        500, {"success": False, "message": "video_stream_info timed out"},
    )
    assert _node_log(commander, caplog) == [
        (logging.WARNING, "GET /api/cameras failed: video_stream_info timed out", None),
    ]


def test_setup_logs_the_bound_capabilities_and_calls_none_of_them(commander, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    commander.bind_lighting()
    commander.bind_materials()
    commander.bind_camera("alpha_wrist_left", "rgb")
    commander.bind_camera("alpha_chest", "rgbd", profile=False)
    monkeypatch.setattr(commander.module.listen, "bind_listener", lambda *bound: _FakeListener("127.0.0.1", 9000))

    async def fake_server(app, bound):
        return asyncio.create_task(asyncio.sleep(0))

    monkeypatch.setattr(commander.module.listen, "start_serving", fake_server)

    async def run():
        await asyncio.gather(*await commander.module.setup(_parameters(9000, "127.0.0.1"), object()))

    asyncio.run(run())
    assert _node_log(commander, caplog) == [
        (logging.INFO, "Scene commander starting", None),
        (logging.INFO, "Scene panel at http://127.0.0.1:9000 (bound 127.0.0.1:9000)", None),
        (logging.INFO, "Capabilities: lighting, materials, cameras: alpha_wrist_left (rgb), alpha_chest (rgbd)", None),
        (logging.INFO, "Scene provider catalogue ready: 1 assets", None),
        (logging.INFO, "Scene provider object state ready: 0 runtime objects", None),
    ]
    # The slots say what is bound; the providers are asked once the page asks.
    assert _requests(commander, commander.lighting, commander.materials, commander.color_cameras,
                     commander.rgbd_cameras, commander.camera_profiles) == {}
