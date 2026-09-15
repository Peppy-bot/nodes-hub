"""Drive the commander's HTTP API against a fake scene provider, without Peppy."""

import asyncio
import enum
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from aiohttp.test_utils import TestClient, TestServer

_SOURCE = Path(__file__).resolve().parents[1] / "src" / "scene_commander" / "__main__.py"
_ACTIONS = ("apply_force", "clear_scene", "load_scene", "move_object", "move_robot", "remove_object", "spawn_object")
_BUSY = "Isaac is still discovering its asset catalogue"
_NOT_READY = "Isaac has not loaded its stage yet"
_SCENE = {"asset_id": "scene/full_warehouse", "display_name": "Full Warehouse", "kind": "scene", "category": "Scenes"}
# A snapshot's capture time, as the generated binding decodes it: epoch seconds.
_CAPTURED = 1_757_944_800.25


class Status(enum.Enum):
    COMPLETED = "completed"
    ABORTED = "aborted"


class FakeService(ModuleType):
    """A consumed service: poll() answers the next queued response, the last one forever."""

    def __init__(self, link, name):
        super().__init__(f"peppygen.consumed_services.{link}.{name}")
        self.answers = []

    def bound_producer(self, node_runner):
        return "producer"

    async def poll(self, node_runner, producer, timeout):
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(data=answer)


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
    services = ModuleType("peppygen.consumed_services.simulation")
    services.get_assets_list = FakeService("simulation", "get_assets_list")
    services.get_assets_list.answers = [_catalogue(_SCENE)]
    objects = ModuleType("peppygen.consumed_services.objects")
    objects.get_object_states = FakeService("objects", "get_object_states")
    objects.get_object_states.answers = [_snapshot()]
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
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_scene_commander_under_test", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, services=services, objects=objects, actions=actions)


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
    )) == (400, {"success": False, "message": "position must be [x, y, z]"})
    assert commander.actions.spawn_object.goals == []
    assert _node_log(commander, caplog) == [
        (logging.WARNING, "POST /api/objects/spawn failed: position must be [x, y, z]", None),
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
    assert _node_log(commander, caplog)[2:] == [
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
