"""Tests for the lidar sensor around sim_robot_core's LidarSweeper, whose
own tests cover when each ray is due, the pose it takes at its time, its
range and the scans a sweep drops: the frame the sensor sweeps from, the
scans it publishes, what its teardown drops and keeps, the pair it asks
about, the rays that pass the lidar's own body, and where a model's lidar is
mounted on its robot's stage. Fakes stand in for PhysX's scene; its own
behavior is exercised only in a live engine."""

import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from control_core_py.geometry import Placement, UnitQuaternion, quaternion_of_yaw
from sim_robot_core.lidars import NO_RETURN_MARK_M, LidarConfig, LidarSweep

_ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(_ENGINE_DIR))
sys.path.insert(0, str(_ENGINE_DIR / "exts"))

import _world  # noqa: E402
import isaac_models  # noqa: E402
import lidar_sensor  # noqa: E402
from entries import entry  # noqa: E402
from lidar_sensor import IsaacLidarSensor, LidarMount, PhysxLidarScene, mount_of  # noqa: E402

_BODY = "/World/delta/base_footprint"
# Four rays a quarter turn apart at 10 Hz: one ray every 25 ms, one scan
# every 100 ms.
_RAYS = 4
_INCREMENT = math.pi / 2
_RATE_HZ = 10


def _lidar(
    rays=_RAYS, increment=_INCREMENT, min_angle=-math.pi, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)
):
    """A lidar on lidar_link, sweeping from -pi as the botshare base's does
    unless the test says otherwise, measuring from 0.05 m to 40 m."""
    return LidarConfig(
        name="lidar",
        frame="lidar_frame",
        parent_link="lidar_link",
        pos=pos,
        quat_wxyz=quat,
        sweep=LidarSweep(
            min_angle_rad=min_angle, angle_increment_rad=increment, rays=rays, scan_rate_hz=_RATE_HZ
        ),
        min_range_m=0.05,
        max_range_m=40.0,
    )


def _known(*lidars, **engine):
    """A model carrying `lidars`, as this engine knows it."""
    rig = entry("rig", lidars=tuple(lidars))
    raw = {"stage": "rig/rig.usd", "articulation_root": ".", **engine}
    return isaac_models.stood("rig", rig, isaac_models.parse("rig", raw), {}, {}, "base_footprint")


def _at(x=0.0, y=0.0, z=0.0, yaw=0.0) -> Placement:
    return Placement((x, y, z), UnitQuaternion(quaternion_of_yaw(yaw)))


class FakeScene:
    """Stands in for PhysX: where the body stands, which the test moves,
    and each ray cast, which meets the distance `meets` gives its index."""

    def __init__(self, meets=lambda ray: None):
        self.body = _at()
        self.casts = []
        self._meets = meets

    def body_placement(self):
        return self.body

    def cast(self, origin, direction, max_range_m):
        self.casts.append((origin, direction, max_range_m))
        return self._meets(len(self.casts) - 1)


class FakeIO:
    def __init__(self, delivers=True):
        self.scans = []
        self.delivers = delivers
        self.viewed = True
        self.asked = []
        self.stamped = []

    def lidar_viewed(self, robot, name):
        self.asked.append((robot, name))
        return self.viewed

    def timestamp_at_s(self, instant_s, now_s):
        self.stamped.append((instant_s, now_s))
        # A clock domain's publisher stamps an instant as itself.
        return instant_s

    def publish_scan(self, robot, lidar, timestamp_s, capture_index, ranges):
        self.scans.append((robot, lidar.frame, timestamp_s, capture_index, list(ranges)))
        return self.delivers


def _sensor(lidar=None, scene=None, frame_in_body=Placement.IDENTITY, io=None):
    lidar = lidar or _lidar()
    sensor = IsaacLidarSensor("delta", "/World/delta", _known(lidar), lidar, io or FakeIO())
    sensor._mount = LidarMount(_BODY, frame_in_body)
    sensor._scene = scene or FakeScene()
    sensor._ready = True
    return sensor


def _moving(sensor, *times_s):
    """Steps `sensor` at each of `times_s`, its body standing at x = the
    step's time, so a ray cast from x = t was cast at t."""
    for now_s in times_s:
        sensor._scene.body = _at(x=now_s)
        sensor.step(now_s)


def _cast_times(scene) -> list:
    return [origin[0] for origin, _, _ in scene.casts]


class TestTheSweep:
    def test_each_ray_leaves_the_frame_the_body_carries_along_its_angle_turned_into_world(self):
        # The body stands at (1, 2, 0) facing +y, and the frame sits 0.3 m
        # ahead of it and 0.1 m up.
        scene = FakeScene()
        scene.body = _at(1.0, 2.0, 0.0, yaw=math.pi / 2)
        sensor = _sensor(scene=scene, frame_in_body=Placement((0.3, 0.0, 0.1), UnitQuaternion.IDENTITY))

        sensor.step(0.0)
        sensor.step(0.2)

        # The first ray points at -pi in the frame, the ones after it a quarter
        # turn counterclockwise each.
        origins = [origin for origin, _, _ in scene.casts[:_RAYS]]
        directions = [direction for _, direction, _ in scene.casts[:_RAYS]]
        assert origins == [pytest.approx((1.0, 2.3, 0.1))] * _RAYS
        assert directions == [
            pytest.approx((0.0, -1.0, 0.0), abs=1e-12),
            pytest.approx((1.0, 0.0, 0.0), abs=1e-12),
            pytest.approx((0.0, 1.0, 0.0), abs=1e-12),
            pytest.approx((-1.0, 0.0, 0.0), abs=1e-12),
        ]

    def test_each_scan_goes_out_naming_the_lidars_frame_stamped_at_its_ray_0(self):
        sensor = _sensor(scene=FakeScene(lambda ray: 2.5))

        _moving(sensor, 0.01, 0.13, 0.2)

        assert sensor._io.scans == [("delta", "lidar_frame", pytest.approx(0.1), 0, [2.5] * _RAYS)]
        # Ray 0's engine time, stamped on the step that completed the scan.
        assert sensor._io.stamped == [(pytest.approx(0.1), 0.2)]

    def test_a_scan_starts_only_while_the_lidars_own_pair_views_it(self):
        io = FakeIO()
        io.viewed = False
        sensor = _sensor(io=io)

        _moving(sensor, 0.0, 0.29)

        assert sensor._scene.casts == []
        assert set(io.asked) == {("delta", "lidar")}

    def test_a_publish_the_transport_refuses_still_counts_its_capture(self):
        sensor = _sensor(io=FakeIO(delivers=False))

        _moving(sensor, 0.0, 0.29)
        sensor._io.delivers = True
        _moving(sensor, 0.39)

        assert [index for _, _, _, index, _ in sensor._io.scans] == [0, 1, 2]


class TestATeardown:
    def test_lets_go_of_the_bodys_view_and_drops_the_scan_in_progress(self):
        scene = FakeScene()
        sensor = _sensor(scene=scene)
        _moving(sensor, 0.01, 0.13)

        # unbind tears every sweep down before the stage changes, and so does
        # a pair that goes; the next step it sweeps on sets it up again.
        sensor.teardown()
        assert not sensor.is_ready
        assert (sensor._mount, sensor._scene) == (None, None)
        sensor._mount, sensor._scene, sensor._ready = LidarMount(_BODY, Placement.IDENTITY), scene, True
        _moving(sensor, 0.24, 0.41)

        # The first step after it only takes where the frame stands, so no ray
        # is cast across it: the scan of 0.2 s is never begun.
        stamps = [timestamp_s for _, _, timestamp_s, _, _ in sensor._io.scans]
        assert stamps == pytest.approx([0.3])
        assert _cast_times(scene)[2:] == pytest.approx([0.3, 0.325, 0.35, 0.375, 0.4])

    def test_keeps_the_capture_index_counting_on(self):
        scene = FakeScene()
        sensor = _sensor(scene=scene)
        _moving(sensor, 0.0, 0.29)

        sensor.teardown()
        sensor._mount, sensor._scene, sensor._ready = LidarMount(_BODY, Placement.IDENTITY), scene, True
        _moving(sensor, 0.33, 0.49)

        assert [(timestamp_s, index) for _, _, timestamp_s, index, _ in sensor._io.scans] == [
            pytest.approx((0.1, 0)),
            pytest.approx((0.2, 1)),
            pytest.approx((0.4, 2)),
        ]
        assert sensor._io.scans[-1][-1] == [NO_RETURN_MARK_M] * _RAYS


class TestTheBodysOwnCollidersArePassed:
    """PhysX's scene query, faked: each ray meets the hits the test lays
    along it, in no particular order, as raycast_all reports them."""

    @pytest.fixture(name="physx")
    def physx_fixture(self, monkeypatch):
        query = SimpleNamespace(hits=[], rays=[])

        def raycast_all(origin, direction, distance, report):
            query.rays.append((origin, direction, distance))
            for hit in query.hits:
                if not report(hit):
                    break
            return bool(query.hits)

        query.raycast_all = raycast_all
        carb = ModuleType("carb")
        carb.Float3 = lambda x, y, z: (x, y, z)
        physx = ModuleType("omni.physx")
        physx.get_physx_scene_query_interface = lambda: query
        omni = ModuleType("omni")
        omni.physx = physx
        monkeypatch.setitem(sys.modules, "carb", carb)
        monkeypatch.setitem(sys.modules, "omni", omni)
        monkeypatch.setitem(sys.modules, "omni.physx", physx)
        return query

    @staticmethod
    def _hit(body: str, distance: float):
        return SimpleNamespace(rigid_body=body, collision=f"{body}/collisions/geom_0", distance=distance)

    def test_a_ray_passes_the_body_that_carries_the_lidar_and_meets_the_nearest_other(self, physx):
        scene = PhysxLidarScene(_BODY, view=None)
        physx.hits = [
            self._hit("/World/RuntimeObjects/obj_wall", 3.0),
            # The lidar's own housing, which the ray starts inside.
            self._hit(_BODY, 0.0),
            # The robot's own wheel, which is another body of it.
            self._hit("/World/delta/base_footprint/left_wheel_link", 0.4),
        ]

        assert scene.cast((0.0, 0.0, 0.2), (1.0, 0.0, 0.0), 40.0) == 0.4
        assert physx.rays == [((0.0, 0.0, 0.2), (1.0, 0.0, 0.0), 40.0)]

    def test_a_ray_that_meets_only_its_own_body_meets_nothing(self, physx):
        scene = PhysxLidarScene(_BODY, view=None)
        physx.hits = [self._hit(_BODY, 0.0), self._hit(_BODY, 0.05)]

        assert scene.cast((0.0, 0.0, 0.2), (0.0, 1.0, 0.0), 40.0) is None

    def test_the_body_stands_where_physx_reads_it(self, physx, monkeypatch):
        view = SimpleNamespace(
            prim_paths=[_BODY],
            get_transforms=lambda: SimpleNamespace(
                numpy=lambda: np.array([[1.0, 2.0, 0.1, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]])
            ),
        )
        monkeypatch.setattr(lidar_sensor, "create_body_view", lambda paths: view)

        placement = PhysxLidarScene.of(_BODY).body_placement()

        assert placement.position == pytest.approx((1.0, 2.0, 0.1))
        assert placement.orientation.yaw() == pytest.approx(math.pi / 2)

    def test_a_body_physx_does_not_simulate_yet_has_no_scene(self, physx, monkeypatch):
        monkeypatch.setattr(lidar_sensor, "create_body_view", lambda paths: SimpleNamespace(prim_paths=[]))

        assert PhysxLidarScene.of(_BODY) is None


class TestWhereAModelsLidarIsMounted:
    """Against real USD: a lidar's frame is fixed in the rigid body that
    carries the prim of its parent link, on its own robot's stage."""

    @pytest.fixture(autouse=True)
    def _usd(self):
        pytest.importorskip("pxr", reason="USD Python wheels are unavailable on this platform")

    @staticmethod
    def _stage(links: dict, bodies=()):
        """A stage of Xforms at `links`, each path to its translation and its
        yaw, with PhysicsRigidBodyAPI on each of `bodies`."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        for path, (translation, yaw) in links.items():
            xform = UsdGeom.Xformable(UsdGeom.Xform.Define(stage, path))
            xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
            w, x, y, z = UnitQuaternion(quaternion_of_yaw(yaw)).wxyz
            xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        for path in bodies:
            UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(path))
        return stage

    def test_a_link_fixed_to_a_body_mounts_its_lidar_in_that_body(self):
        # The botshare base's layout: base_link and lidar_link hang from
        # base_footprint, the one body of its chassis.
        stage = self._stage(
            {
                "/World/delta": ((0.0, 0.0, 0.0), 0.0),
                _BODY: ((0.0, 0.0, 0.0), 0.0),
                f"{_BODY}/base_link": ((0.0, 0.0, 0.05), 0.0),
                f"{_BODY}/base_link/lidar_link": ((0.33, 0.0, 0.124), math.pi),
            },
            bodies=(_BODY,),
        )
        lidar = _lidar(pos=(0.01, 0.0, 0.02))

        mount = mount_of(stage.GetPrimAtPath("/World/delta"), _known(lidar), lidar)

        assert mount.body_path == _BODY
        # The frame's 0.01 m along lidar_link's x points back along the
        # body's, lidar_link being turned half a turn.
        assert mount.frame_in_body.position == pytest.approx((0.32, 0.0, 0.194))
        assert abs(mount.frame_in_body.orientation.yaw()) == pytest.approx(math.pi)

    def test_a_link_that_is_a_body_mounts_its_lidar_in_itself(self):
        link = "/World/delta/lidar_link"
        stage = self._stage(
            {"/World/delta": ((0.0, 0.0, 0.0), 0.0), link: ((0.2, 0.0, 0.3), 0.0)}, bodies=(link,)
        )
        quarter = UnitQuaternion(quaternion_of_yaw(math.pi / 2))
        lidar = _lidar(pos=(0.0, 0.0, 0.02), quat=quarter.wxyz)

        mount = mount_of(stage.GetPrimAtPath("/World/delta"), _known(lidar), lidar)

        assert mount.body_path == link
        assert mount.frame_in_body.position == pytest.approx((0.0, 0.0, 0.02))
        assert mount.frame_in_body.orientation.yaw() == pytest.approx(math.pi / 2)

    def test_a_link_no_rigid_body_carries_is_refused_naming_what_to_give(self):
        stage = self._stage(
            {"/World/delta": ((0.0, 0.0, 0.0), 0.0), "/World/delta/lidar_link": ((0.0, 0.0, 0.0), 0.0)}
        )
        lidar = _lidar()

        with pytest.raises(RuntimeError) as refused:
            mount_of(stage.GetPrimAtPath("/World/delta"), _known(lidar), lidar)

        assert str(refused.value) == (
            "lidar 'lidar' hangs from 'lidar_link', and no rigid body carries its prim "
            "/World/delta/lidar_link: give that prim or one above it under /World/delta "
            "PhysicsRigidBodyAPI"
        )

    def test_a_parent_link_the_robots_stage_lacks_is_refused_with_what_it_has(self):
        stage = self._stage({"/World/delta": ((0.0, 0.0, 0.0), 0.0), _BODY: ((0.0, 0.0, 0.0), 0.0)})
        lidar = _lidar()

        with pytest.raises(RuntimeError) as refused:
            mount_of(stage.GetPrimAtPath("/World/delta"), _known(lidar), lidar)

        message = str(refused.value)
        assert "lidar 'lidar' parent link 'lidar_link' is not under /World/delta as 'lidar_link'" in message
        assert "base_footprint" in message

    def test_the_botshare_bases_lidar_hangs_from_lidar_link_and_names_lidar_frame(self):
        (lidar,) = _world.known("botshare_base").entry.lidars

        assert (lidar.name, lidar.frame, lidar.parent_link) == ("lidar", "lidar_frame", "lidar_link")
        assert lidar.sweep.rays == 3240
