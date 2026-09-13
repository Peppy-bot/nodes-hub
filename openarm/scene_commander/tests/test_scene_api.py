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

_SOURCE = Path(__file__).resolve().parents[1] / "src" / "openarm_scene_commander" / "__main__.py"
_ACTIONS = ("apply_force", "clear_scene", "load_scene", "move_object", "move_robot", "remove_object", "spawn_object")
_BUSY = "Isaac is still discovering its asset catalogue"
_SCENE = {"asset_id": "scene/full_warehouse", "display_name": "Full Warehouse", "kind": "scene", "category": "Scenes"}


class Status(enum.Enum):
    COMPLETED = "completed"
    ABORTED = "aborted"


class FakeService(ModuleType):
    """A consumed service: poll() answers the next queued response, the last one forever."""

    def __init__(self, name):
        super().__init__(f"peppygen.consumed_services.simulation.{name}")
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
    services.get_assets_list = FakeService("get_assets_list")
    services.get_assets_list.answers = [_catalogue(_SCENE)]
    services.get_objects_list = FakeService("get_objects_list")
    services.get_objects_list.answers = [SimpleNamespace(success=True, message="0 runtime objects", objects_json="[]")]
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
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_scene_commander_under_test", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, services=services, actions=actions)


def _busy():
    return SimpleNamespace(success=False, message=_BUSY, assets_json="[]")


def _catalogue(*assets):
    return SimpleNamespace(success=True, message=f"{len(assets)} assets available", assets_json=json.dumps(list(assets)))


def _node_log(commander, caplog):
    return [(r.levelno, r.getMessage(), r.exc_info) for r in caplog.records if r.name == commander.module.logger.name]


def _call(commander, scenario):
    async def run():
        app = commander.module._build_app(object(), commander.module._CatalogueWatch())
        async with TestClient(TestServer(app)) as client:
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
        (logging.INFO, f"Scene provider has no catalogue yet: {_BUSY}", None),
        (logging.INFO, "Scene provider catalogue ready: 1 assets", None),
    ]


def test_provider_transport_failures_are_server_errors_logged_in_one_line(commander, caplog):
    caplog.set_level(logging.INFO)
    commander.services.get_assets_list.answers = [TimeoutError("get_assets_list timed out")]

    assert _call(commander, lambda client: _get(client, "/api/assets")) == (
        500, {"success": False, "message": "get_assets_list timed out"},
    )
    assert _node_log(commander, caplog) == [
        (logging.WARNING, "GET /api/assets failed: get_assets_list timed out", None),
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

    async def fake_server(node_runner, host, port, catalogue):
        served.append((host, port, type(catalogue)))

    monkeypatch.setattr(commander.module, "_run_http_server", fake_server)
    commander.services.get_assets_list.answers = [_busy()]

    async def run():
        await asyncio.gather(*await commander.module.setup({"http_port": 9000}, object()))

    asyncio.run(run())
    assert served == [("0.0.0.0", 9000, commander.module._CatalogueWatch)]
    assert _node_log(commander, caplog) == [
        (logging.INFO, "OpenArm scene commander starting", None),
        (logging.INFO, f"Scene provider has no catalogue yet: {_BUSY}", None),
        (logging.INFO, "Scene provider reachable: 0 runtime objects", None),
    ]

    commander.services.get_objects_list.answers = [TimeoutError("get_objects_list timed out")]
    with pytest.raises(TimeoutError, match="get_objects_list timed out"):
        asyncio.run(commander.module.setup({}, object()))
    assert len(served) == 1


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
        def __init__(self, runner, host, port):
            sites.append((runner, host, port))

        async def start(self):
            started.set()

    monkeypatch.setattr(commander.module.web, "AppRunner", FakeRunner)
    monkeypatch.setattr(commander.module.web, "TCPSite", FakeSite)

    async def run():
        task = asyncio.create_task(commander.module._run_http_server(
            object(), "127.0.0.1", 8766, commander.module._CatalogueWatch(),
        ))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    [(app, options)] = runners
    assert options == {"access_log": None}
    assert {route.resource.canonical for route in app.router.routes()} >= {"/", "/api/assets", "/api/scene/load"}
    [(runner, host, port)] = sites
    assert (host, port) == ("127.0.0.1", 8766)
    assert runner.cleaned
