"""The scene_manipulation and object_state provider: what get_assets_list answers
before and after discovery, what load_scene and clear_scene do to spawned
objects, and what get_object_states answers: nothing until a capture, then
the latest capture under its own stamp, every completed edit included."""

import importlib
import importlib.util
import json
import logging
import sys
from concurrent.futures import Future
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest

_ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
_ACTIONS = ("apply_force", "clear_scene", "load_scene", "move_object", "move_robot", "remove_object", "spawn_object")
_SERVICES = {
    "scene": ("get_assets_list", "get_robots_list"),
    "objects": ("get_object_states",),
}


class _Robot(SimpleNamespace):
    """A robot as world.py reports it: the name it stands under, its model,
    and where."""

    def prim(self):
        return f"/World/{self.instance}"


class _World:
    """The stage's robots, as SceneActionIO reads them."""

    def __init__(self, robots):
        self._robots = list(robots)

    def robots(self):
        return list(self._robots)


def _world(*robots):
    """The stage's robots, each as (instance, model, position, yaw)."""
    return _World(
        _Robot(
            instance=instance,
            model=model,
            placement=SimpleNamespace(position=position, yaw=yaw),
        )
        for instance, model, position, yaw in robots
    )


class FakeStamps:
    """SimTopicIO's capture stamp; None while the engine cannot stamp yet."""

    def __init__(self):
        self.now_s = 10.0

    def capture_timestamp_s(self):
        return self.now_s


class FakeReader:
    """The engine side of a capture: each spawned object where the fake stage
    has it, at rest, or a failure when the stage has lost it."""

    def __init__(self, stage, record):
        self._stage = stage
        self._record = record
        self.failure = None
        self.reads = 0

    def read(self, spawned):
        self.reads += 1
        if self.failure is not None:
            raise RuntimeError(self.failure)
        return [
            self._record(
                object_id=obj["object_id"], asset_id=obj["asset_id"], physics=obj["physics"],
                mass=obj["mass"], scale=obj["scale"], position=tuple(self._stage[obj["object_id"]]),
                orientation=(0.0, 0.0, 0.0, 1.0), linear_velocity=(0.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0),
            )
            for obj in spawned
        ]


@pytest.fixture
def provider(monkeypatch):
    actions = ModuleType("peppygen.exposed_actions.scene")
    for name in _ACTIONS:
        setattr(actions, name, ModuleType(f"{actions.__name__}.{name}"))
    modules = {
        "peppygen": ModuleType("peppygen"),
        "peppygen.exposed_actions": ModuleType("peppygen.exposed_actions"),
        "peppygen.exposed_actions.scene": actions,
        "peppygen.exposed_services": ModuleType("peppygen.exposed_services"),
    }
    for link, names in _SERVICES.items():
        services = ModuleType(f"peppygen.exposed_services.{link}")
        for name in names:
            module = ModuleType(f"{services.__name__}.{name}")
            module.Response = lambda **fields: SimpleNamespace(**fields)
            module.ResponseObjectsItem = lambda **fields: SimpleNamespace(**fields)
            # The generated record of one robot in a get_robots_list answer.
            module.ResponseRobotsItem = lambda **fields: SimpleNamespace(**fields)
            setattr(services, name, module)
        modules[services.__name__] = services
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(str(_ENGINE_DIR))
    spec = importlib.util.spec_from_file_location("_scene_actions_under_test", _ENGINE_DIR / "scene_actions.py")
    module = importlib.util.module_from_spec(spec)
    # Its dataclass resolves postponed annotations through sys.modules.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    stamps = FakeStamps()
    world = _world(("alpha", "openarm_v2", (0.0, 0.0, 0.0), 0.0))
    io = module.SceneActionIO(object(), Mock(), stamps, world)
    # The stage by object_id: where the launcher put each object.
    stage = {}
    reader = FakeReader(stage, importlib.import_module("object_state").ObjectRecord)
    io._object_reader = reader
    launcher = Mock()
    launcher._runtime_spawn_isaac_asset.side_effect = lambda command: stage.__setitem__(command["name"], command["position"])
    launcher._runtime_move_object.side_effect = lambda command: stage.__setitem__(command["name"], command["position"])
    launcher._runtime_remove.side_effect = lambda command: stage.pop(command["name"], None)
    return SimpleNamespace(
        module=module, io=io, launcher=launcher, stamps=stamps, reader=reader,
        stage=stage, world=world,
    )


_CATALOGUE = {
    "scene/full_warehouse": {
        "asset_id": "scene/full_warehouse", "display_name": "Full Warehouse", "kind": "scene",
        "path": "Isaac/Environments/Simple_Warehouse/full_warehouse.usd", "category": "Scenes",
    },
    "props/blocks/red_block": {
        "asset_id": "props/blocks/red_block", "display_name": "red block", "kind": "object",
        "path": "Isaac/Props/Blocks/red_block.usd", "category": "Blocks",
    },
    "props/blocks/blue_block": {
        "asset_id": "props/blocks/blue_block", "display_name": "blue block", "kind": "object",
        "path": "Isaac/Props/Blocks/blue_block.usd", "category": "Blocks",
    },
}


def test_assets_are_refused_with_a_reason_until_discovery_hands_them_over(provider):
    response = provider.io._handle_get_assets(None)
    assert response.success is False
    assert "still discovering its asset catalogue" in response.message
    assert json.loads(response.assets_json) == []

    provider.io.set_assets({})
    response = provider.io._handle_get_assets(None)
    assert response.success is True
    assert response.message == "0 assets available"
    assert json.loads(response.assets_json) == []

    provider.io.set_assets(_CATALOGUE)
    response = provider.io._handle_get_assets(None)
    assert response.success is True
    assert [a["asset_id"] for a in json.loads(response.assets_json)] == [
        "props/blocks/blue_block", "props/blocks/red_block", "scene/full_warehouse",
    ]
    assert all("path" not in a for a in json.loads(response.assets_json))


def _spawn(provider, asset_id="props/blocks/red_block"):
    result = provider.io._execute(provider.launcher, "spawn_object", {
        "asset_id": asset_id, "position": [0.5, 0.0, 0.8], "yaw": 0.0, "scale": 1.0, "physics": "dynamic",
        "mass": 0.1,
    })
    assert result["success"]
    return result["object_id"]


def _spawned_ids(provider):
    return [record.object_id for record in provider.io.capture_object_states().objects]


def test_load_scene_removes_spawned_objects_then_replaces_the_scene(provider):
    provider.io.set_assets(_CATALOGUE)
    first, second = _spawn(provider), _spawn(provider)
    provider.launcher.reset_mock()

    result = provider.io._execute(provider.launcher, "load_scene", {"asset_id": "scene/full_warehouse", "scale": 2.0})

    assert result == {"success": True, "message": "Loaded scene scene/full_warehouse"}
    assert provider.launcher.mock_calls == [
        call._runtime_remove({"name": first}),
        call._runtime_remove({"name": second}),
        call._runtime_load_isaac_scene({
            "path": "Isaac/Environments/Simple_Warehouse/full_warehouse.usd", "scale": [2.0, 2.0, 2.0],
        }),
    ]
    assert _spawned_ids(provider) == []


def test_clear_scene_removes_spawned_objects_then_the_scene(provider):
    provider.io.set_assets(_CATALOGUE)
    object_id = _spawn(provider)
    provider.launcher.reset_mock()

    result = provider.io._execute(provider.launcher, "clear_scene", {})

    assert result == {"success": True, "message": "Runtime scene cleared"}
    assert provider.launcher.mock_calls == [
        call._runtime_remove({"name": object_id}),
        call._runtime_clear_scene(),
    ]
    assert _spawned_ids(provider) == []


@pytest.mark.parametrize(("asset_id", "message"), [
    ("scene/missing", "Unknown asset_id: scene/missing"),
    ("props/blocks/red_block", "Asset is not a scene: props/blocks/red_block"),
])
def test_load_scene_rejects_unknown_or_non_scene_assets_before_touching_the_stage(provider, asset_id, message):
    provider.io.set_assets(_CATALOGUE)
    object_id = _spawn(provider)
    provider.launcher.reset_mock()

    with pytest.raises(ValueError, match=message):
        provider.io._execute(provider.launcher, "load_scene", {"asset_id": asset_id, "scale": 1.0})
    assert provider.launcher.mock_calls == []
    assert _spawned_ids(provider) == [object_id]


def test_the_listing_names_every_robot_the_stage_stands(provider):
    listed = provider.io._handle_get_robots(None).robots

    assert len(listed) == 1
    assert (listed[0].robot, listed[0].model) == ("alpha", "openarm_v2")
    assert listed[0].position == [0.0, 0.0, 0.0]
    assert listed[0].yaw == 0.0
    assert listed[0].attached


def test_the_listing_follows_the_robots_that_join(provider):
    provider.io._world = _world(
        ("alpha", "openarm_v2", (0.0, 0.0, 0.0), 0.0),
        ("bravo", "openarm_v1", (0.0, -1.5, 0.0), 1.25),
    )

    listed = provider.io._handle_get_robots(None).robots

    assert [r.robot for r in listed] == ["alpha", "bravo"]
    assert [r.attached for r in listed] == [True, True]
    assert listed[1].model == "openarm_v1"
    assert listed[1].position == [0.0, -1.5, 0.0]
    # Each faces the way it was placed.
    assert [r.yaw for r in listed] == [0.0, 1.25]


def test_moving_a_robot_by_name_moves_that_robot_to_face_the_yaw_it_names(provider):
    provider.io._world = _world(
        ("alpha", "openarm_v2", (0.0, 0.0, 0.0), 0.0),
        ("bravo", "openarm_v1", (0.0, -1.5, 0.0), 1.25),
    )

    result = provider.io._execute(
        provider.launcher, "move_robot", {"robot": "bravo", "position": [1.0, -2.0, 0.0], "yaw": -0.5}
    )

    assert result["success"] is True
    assert provider.launcher.mock_calls == [
        call._runtime_move_robot_root({"robot": "bravo", "position": [1.0, -2.0, 0.0], "yaw": -0.5})
    ]


def test_a_robot_the_stage_does_not_stand_is_refused_before_the_stage(provider):
    with pytest.raises(ValueError, match="no robot stands as 'ghost'"):
        provider.io._execute(
            provider.launcher, "move_robot", {"robot": "ghost", "position": [1.0, 0.0, 0.0], "yaw": 0.0}
        )

    assert provider.launcher.mock_calls == []


@pytest.mark.parametrize("yaw", [float("nan"), float("inf"), float("-inf")])
def test_a_move_to_a_yaw_that_is_no_number_is_refused_before_the_stage(provider, yaw):
    with pytest.raises(ValueError, match="yaw must be a finite number of radians"):
        provider.io._execute(
            provider.launcher, "move_robot", {"robot": "alpha", "position": [1.0, 0.0, 0.0], "yaw": yaw}
        )

    assert provider.launcher.mock_calls == []


def test_a_spawn_reaches_the_stage_with_the_yaw_it_names(provider):
    provider.io.set_assets(_CATALOGUE)

    result = provider.io._execute(provider.launcher, "spawn_object", {
        "asset_id": "props/blocks/red_block", "position": [0.5, 0.0, 0.8], "yaw": 1.5, "scale": 1.0,
        "physics": "static", "mass": 0.1,
    })

    assert result["success"]
    (spawned,) = provider.launcher._runtime_spawn_isaac_asset.call_args.args
    assert (spawned["name"], spawned["position"], spawned["yaw"]) == (result["object_id"], [0.5, 0.0, 0.8], 1.5)


@pytest.mark.parametrize("yaw", [float("nan"), float("inf"), float("-inf")])
def test_a_spawn_at_a_yaw_that_is_no_number_is_refused_and_mints_nothing(provider, yaw):
    provider.io.set_assets(_CATALOGUE)

    with pytest.raises(ValueError, match="yaw must be a finite number of radians"):
        provider.io._execute(provider.launcher, "spawn_object", {
            "asset_id": "props/blocks/red_block", "position": [0.5, 0.0, 0.8], "yaw": yaw, "scale": 1.0,
            "physics": "static", "mass": 0.1,
        })

    assert provider.launcher.mock_calls == []
    assert _spawned_ids(provider) == []


def test_it_owns_exactly_the_objects_it_spawned_and_has_not_removed(provider):
    provider.io.set_assets(_CATALOGUE)
    first, second = _spawn(provider), _spawn(provider)
    assert (provider.io.owns(first), provider.io.owns(second), provider.io.owns("MyObject")) == (True, True, False)

    provider.io._execute(provider.launcher, "remove_object", {"object_id": first})
    assert (provider.io.owns(first), provider.io.owns(second)) == (False, True)

    provider.io._execute(provider.launcher, "clear_scene", {})
    assert not provider.io.owns(second)


def _assert_unavailable(response, reason):
    assert response.success is False
    assert reason in response.message
    # Unavailable is not an empty scene: nothing rides along.
    assert (response.timestamp, response.objects) == (0.0, [])


def test_object_state_is_unavailable_with_a_reason_until_a_stamped_capture(provider):
    _assert_unavailable(provider.io._handle_get_object_states(None), "has not captured its object state yet")

    # A publisher that has not stepped cannot stamp: no snapshot, no read.
    provider.stamps.now_s = None
    assert provider.io.capture_object_states() is None
    provider.io.process_pending(provider.launcher)
    _assert_unavailable(provider.io._handle_get_object_states(None), "has not captured its object state yet")
    assert provider.reader.reads == 0


def test_the_first_frame_answers_an_empty_scene_before_any_stream_tick(provider):
    provider.io.process_pending(provider.launcher)

    response = provider.io._handle_get_object_states(None)
    assert (response.success, response.message) == (True, "0 objects")
    assert (response.timestamp, response.objects) == (10.0, [])
    assert provider.launcher.mock_calls == []

    # A snapshot exists, so later frames with nothing queued leave it alone.
    provider.io.process_pending(provider.launcher)
    assert provider.reader.reads == 1


def test_records_carry_what_each_object_was_spawned_with_in_spawn_order(provider):
    provider.io.set_assets(_CATALOGUE)
    spawned = [
        ("props/blocks/red_block", "dynamic", 0.25, 1.5),
        ("props/blocks/blue_block", "static", 2.0, 0.5),
        ("props/blocks/red_block", "none", 0.1, 3.0),
    ]
    object_ids = []
    for index, (asset_id, physics, mass, scale) in enumerate(spawned):
        result = provider.io._execute(provider.launcher, "spawn_object", {
            "asset_id": asset_id, "position": [float(index), 0.0, 0.8], "yaw": 0.0, "scale": scale,
            "physics": physics, "mass": mass,
        })
        object_ids.append(result["object_id"])

    provider.io.capture_object_states()
    objects = provider.io._handle_get_object_states(None).objects

    assert [vars(item) for item in objects] == [
        {
            "object_id": object_id, "asset_id": asset_id, "physics": physics, "mass": mass, "scale": scale,
            "position": [float(index), 0.0, 0.8], "orientation": [0.0, 0.0, 0.0, 1.0],
            "linear_velocity": [0.0, 0.0, 0.0], "angular_velocity": [0.0, 0.0, 0.0],
        }
        for index, (object_id, (asset_id, physics, mass, scale)) in enumerate(zip(object_ids, spawned))
    ]


def test_the_answer_carries_its_capture_stamp_never_the_request_time(provider):
    captured = provider.io.capture_object_states()
    assert captured.timestamp_s == 10.0

    provider.stamps.now_s = 99.0
    assert [provider.io._handle_get_object_states(None).timestamp for _ in range(2)] == [10.0, 10.0]

    # The stream publishes the capture the service answers, and the next
    # capture replaces both.
    assert provider.io.capture_object_states().timestamp_s == 99.0
    assert provider.io._handle_get_object_states(None).timestamp == 99.0


def _submit(provider, operation, payload):
    """Queue a command as a goal would, and record what get_object_states
    answers at the instant the goal completes."""
    future = Future()
    seen = []
    future.add_done_callback(lambda done: seen.append((done.result(), provider.io._handle_get_object_states(None))))
    provider.io._pending.put(provider.module._PendingCommand(operation=operation, payload=payload, future=future))
    return seen


def test_a_read_once_an_edit_completes_observes_it(provider):
    provider.io.set_assets(_CATALOGUE)
    provider.io.process_pending(provider.launcher)

    provider.stamps.now_s = 11.0
    spawned = _submit(provider, "spawn_object", {
        "asset_id": "props/blocks/red_block", "position": [0.5, 0.0, 0.8], "yaw": 0.0, "scale": 1.0,
        "physics": "dynamic", "mass": 0.1,
    })
    provider.io.process_pending(provider.launcher)
    ((result, response),) = spawned
    object_id = result["object_id"]
    assert [(item.object_id, item.position) for item in response.objects] == [(object_id, [0.5, 0.0, 0.8])]
    assert response.timestamp == 11.0

    provider.stamps.now_s = 12.0
    moved = _submit(provider, "move_object", {"object_id": object_id, "position": [0.2, 0.3, 0.9]})
    removed = _submit(provider, "remove_object", {"object_id": object_id})
    provider.io.process_pending(provider.launcher)
    ((_, after_move),) = moved
    ((_, after_remove),) = removed
    assert [item.position for item in after_move.objects] == [[0.2, 0.3, 0.9]]
    assert after_remove.objects == []
    assert (after_move.timestamp, after_remove.timestamp) == (12.0, 12.0)


def test_a_failed_goal_still_completes_after_a_capture(provider):
    provider.io.set_assets(_CATALOGUE)
    failed = _submit(provider, "remove_object", {"object_id": "obj_missing"})
    provider.io.process_pending(provider.launcher)

    ((result, response),) = failed
    assert result == {"success": False, "message": "Unknown object_id: obj_missing"}
    assert (response.success, response.objects) == (True, [])


def test_a_failed_capture_is_unavailable_with_its_reason_rather_than_an_older_snapshot(provider, caplog):
    provider.io.process_pending(provider.launcher)
    provider.reader.failure = "PhysX simulates no rigid body at /World/RuntimeObjects/obj_1"

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert provider.io.capture_object_states() is None

    _assert_unavailable(
        provider.io._handle_get_object_states(None),
        "Isaac could not capture its object state: PhysX simulates no rigid body at /World/RuntimeObjects/obj_1",
    )
    assert len([r for r in caplog.records if "could not capture" in r.message]) == 1, "one line per failure"

    # Frames with nothing queued do not retry: a read that keeps failing
    # would otherwise rebuild the physics view every frame.
    reads = provider.reader.reads
    provider.io.process_pending(provider.launcher)
    assert provider.reader.reads == reads

    # The next state tick's capture brings it back.
    provider.reader.failure = None
    provider.stamps.now_s = 20.0
    provider.io.capture_object_states()
    assert provider.io._handle_get_object_states(None).timestamp == 20.0
