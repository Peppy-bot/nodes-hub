"""The object reader: which engine source each spawned object reads from, the
x, y, z, w orientation PhysX hands over, velocities moved from the centre of
mass to the object's origin, and when the rigid-body view is created again.
The engine calls are faked; everything under test is pure python."""

import importlib
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

_ROBOT_DIR = Path(__file__).resolve().parents[1] / "robots" / "openarm"
_HALF_TURN = 2 ** -0.5  # a 90 degree yaw is (0, 0, sin 45, cos 45) in x, y, z, w


def _spawned(object_id, physics, mass=0.5, scale=2.0):
    return {"object_id": object_id, "asset_id": f"props/{object_id}", "physics": physics, "mass": mass, "scale": scale}


class FakeBodyView:
    """What create_rigid_body_view returns, one row per body in prim_paths order."""

    def __init__(self, bodies):
        self.prim_paths = list(bodies)
        rows = list(bodies.values())
        self._transforms = np.array([row["transform"] for row in rows])
        self._velocities = np.array([row["velocity"] for row in rows])
        self._coms = np.array([row.get("com", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) for row in rows])

    def get_transforms(self):
        return Mock(numpy=lambda: self._transforms)

    def get_velocities(self):
        return Mock(numpy=lambda: self._velocities)

    def get_coms(self):
        return Mock(numpy=lambda: self._coms)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.syspath_prepend(str(_ROBOT_DIR))
    module = importlib.import_module("object_state")
    engine = Mock()
    engine.authored = {}
    engine.views = []
    engine.bodies = {}
    engine.trace = []

    def create_body_view(paths):
        engine.trace.append(("view", tuple(paths)))
        view = FakeBodyView({path: engine.bodies[path] for path in paths if path in engine.bodies})
        engine.views.append(view)
        return view

    monkeypatch.setattr(module, "_flush_physics_changes", lambda: engine.trace.append("flush"))
    monkeypatch.setattr(module, "_create_body_view", create_body_view)
    monkeypatch.setattr(module, "_authored_world_pose", lambda path: engine.authored[path])
    engine.module = module
    engine.reader = module.IsaacObjectReader()
    return engine


def test_static_and_visual_only_objects_report_their_authored_pose_at_rest(engine):
    engine.authored = {
        "/World/RuntimeObjects/obj_table": ((1.0, 0.0, 0.4), (0.0, 0.0, _HALF_TURN, _HALF_TURN)),
        "/World/RuntimeObjects/obj_marker": ((0.2, 0.3, 0.0), (0.0, 0.0, 0.0, 1.0)),
    }

    records = engine.reader.read([_spawned("obj_table", "static", mass=3.0), _spawned("obj_marker", "none", scale=0.5)])

    assert [record.fields() for record in records] == [
        {
            "object_id": "obj_table", "asset_id": "props/obj_table", "physics": "static", "mass": 3.0, "scale": 2.0,
            "position": [1.0, 0.0, 0.4], "orientation": [0.0, 0.0, _HALF_TURN, _HALF_TURN],
            "linear_velocity": [0.0, 0.0, 0.0], "angular_velocity": [0.0, 0.0, 0.0],
        },
        {
            "object_id": "obj_marker", "asset_id": "props/obj_marker", "physics": "none", "mass": 0.5, "scale": 0.5,
            "position": [0.2, 0.3, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0],
            "linear_velocity": [0.0, 0.0, 0.0], "angular_velocity": [0.0, 0.0, 0.0],
        },
    ]
    assert engine.trace == [], "nothing dynamic, so PhysX is not asked"


def test_dynamic_objects_read_their_live_pose_and_velocities_from_physx(engine):
    engine.bodies = {
        "/World/RuntimeObjects/obj_b": {"transform": [0.4, 0.1, 0.02, 0.0, 0.0, 0.0, 1.0], "velocity": [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]},
        "/World/RuntimeObjects/obj_a": {"transform": [0.5, 0.0, 0.8, 0.0, 0.0, _HALF_TURN, _HALF_TURN], "velocity": [0.1, 0.2, 0.3, 0.0, 0.0, 1.5]},
    }
    engine.authored = {"/World/RuntimeObjects/obj_c": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))}

    records = engine.reader.read([_spawned("obj_a", "dynamic"), _spawned("obj_c", "static"), _spawned("obj_b", "dynamic")])

    # The view lists bodies in its own order; records keep the registry's.
    assert [record.object_id for record in records] == ["obj_a", "obj_c", "obj_b"]
    a, _, b = records
    assert a.position == (0.5, 0.0, 0.8)
    assert a.orientation == (0.0, 0.0, _HALF_TURN, _HALF_TURN), "PhysX's x, y, z, w order is the contract's"
    assert a.linear_velocity == (0.1, 0.2, 0.3)
    assert a.angular_velocity == (0.0, 0.0, 1.5)
    assert (b.position, b.linear_velocity) == ((0.4, 0.1, 0.02), (0.0, 0.0, -1.0))
    assert (a.mass, a.scale, a.physics) == (0.5, 2.0, "dynamic")
    # Pending edits reach the backend before the view reads it.
    assert engine.trace == ["flush", ("view", ("/World/RuntimeObjects/obj_a", "/World/RuntimeObjects/obj_b"))]


def test_the_linear_velocity_is_moved_from_the_centre_of_mass_to_the_origin(engine):
    # Spinning at 2 rad/s about z, yawed 90 degrees, with the centre of mass
    # 0.1 m along the body's x: the origin sits 0.1 m along world -y of the
    # centre of mass, so it moves at w x r = (0.2, 0, 0) on top of it.
    engine.bodies = {
        "/World/RuntimeObjects/obj_spin": {
            "transform": [0.0, 0.0, 0.5, 0.0, 0.0, _HALF_TURN, _HALF_TURN],
            "velocity": [0.0, 0.0, -0.1, 0.0, 0.0, 2.0],
            "com": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
    }

    (record,) = engine.reader.read([_spawned("obj_spin", "dynamic")])

    assert record.linear_velocity == pytest.approx((0.2, 0.0, -0.1))
    assert record.angular_velocity == (0.0, 0.0, 2.0)


def test_the_view_is_kept_until_the_dynamic_set_changes_or_a_removal_invalidates_it(engine):
    for object_id in ("obj_a", "obj_b"):
        engine.bodies[f"/World/RuntimeObjects/{object_id}"] = {
            "transform": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "velocity": [0.0] * 6,
        }
    one, two = [_spawned("obj_a", "dynamic")], [_spawned("obj_a", "dynamic"), _spawned("obj_b", "dynamic")]

    engine.reader.read(one)
    engine.reader.read(one)
    assert len(engine.views) == 1
    engine.reader.read(two)
    assert len(engine.views) == 2
    engine.reader.invalidate()
    engine.reader.read(two)
    assert len(engine.views) == 3
    assert engine.trace.count("flush") == 4, "every read flushes, a kept view included"


def test_a_dynamic_object_physx_does_not_simulate_fails_the_read_and_drops_the_view(engine):
    engine.bodies = {"/World/RuntimeObjects/obj_a": {"transform": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "velocity": [0.0] * 6}}
    spawned = [_spawned("obj_a", "dynamic"), _spawned("obj_ghost", "dynamic")]

    with pytest.raises(RuntimeError, match="PhysX simulates no rigid body at /World/RuntimeObjects/obj_ghost"):
        engine.reader.read(spawned)

    engine.bodies["/World/RuntimeObjects/obj_ghost"] = engine.bodies["/World/RuntimeObjects/obj_a"]
    assert [record.object_id for record in engine.reader.read(spawned)] == ["obj_a", "obj_ghost"]
    assert len(engine.views) == 2, "the failed view is not reused"


def test_nothing_spawned_reads_nothing(engine):
    assert engine.reader.read([]) == []
    assert engine.trace == []
