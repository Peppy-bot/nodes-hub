"""The MuJoCo half of the lidar path: where a lidar's frame stands, what a
ray cast with mj_ray meets and passes, which pair a sweep asks for, how a
sweep is handed across a scene composed again, and what a scan carries to
the transport. When each ray falls due and the scans a sweep completes and
drops are sim_robot_core's LidarSweeper, tested there. The scenes are real
(tiny) MJCF composed and compiled by MuJoCo, and the transport is a fake that
keeps what the sensor read and published.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import pytest
from control_core_py.geometry import quaternion_of_yaw, wxyz_of_xyzw
from entries import entry, known as known_model, placement_at
from sim_robot_core.lidars import NO_RETURN_MARK_M, UNMEASURED_MARK_M, LidarConfig, LidarSweep

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime the bridge imports)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

import mujoco_models  # noqa: E402  pylint: disable=C0413
from bridge_extension import MujocoBridgeExtension  # noqa: E402  pylint: disable=C0413
from exts.lidar_sensor import MujocoLidarSensor  # noqa: E402  pylint: disable=C0413
from world import World  # noqa: E402  pylint: disable=C0413

ROBOT = "charlo"
_STATE_RATE_HZ = 50
# How high the lidar's frame stands over the post's foot (m).
_SCAN_HEIGHT_M = 1.0
# How close a measured range comes to the geometry's (m).
_RANGE_TOLERANCE_M = 1e-9
# Ten scans a second: a scan starts every 100 ms of the engine clock.
_PERIOD_NS = 100_000_000

# The lidar of the post: eight rays from +x, an eighth of a turn apart, from
# 0.1 m to 10 m, ten scans a second, on the post's head.
_LIDAR = LidarConfig(
    name="lidar",
    frame="lidar_frame",
    parent_link="head",
    pos=(0.0, 0.0, 0.0),
    quat_wxyz=(1.0, 0.0, 0.0, 0.0),
    sweep=LidarSweep(min_angle_rad=0.0, angle_increment_rad=math.pi / 4, rays=8, scan_rate_hz=10),
    min_range_m=0.1,
    max_range_m=10.0,
)

# A post whose head carries the lidar inside a housing that encloses its
# frame. The post's own body carries a beacon in the scan plane at 135
# degrees and a collision hull, which no camera draws, at 315 degrees.
_BEACON_RADIUS_M = 0.1
_POST = f"""<mujoco model="lidar_post">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="post">
      <geom type="cylinder" size="0.03 0.45" pos="0 0 0.45"/>
      <geom name="beacon" type="sphere" size="{_BEACON_RADIUS_M}" pos="-0.5 0.5 {_SCAN_HEIGHT_M}"/>
      <geom name="hull" type="box" size="0.05 0.05 0.05" pos="0.5 -0.5 {_SCAN_HEIGHT_M}" group="3"/>
      <body name="head" pos="0 0 {_SCAN_HEIGHT_M}">
        <geom name="housing" type="cylinder" size="0.04 0.03"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""

# The walls standing around a post at the origin, each a box of the scene by
# its centre and half sizes: the face of the first at x 2 m, ahead of ray 0;
# the face of the second at y 1.5 m, from x 0.5 m to 3 m, which ray 1 meets
# and ray 2 passes beside; and the face of the third at y -0.05 m, nearer
# than the lidar's nearest range, ahead of ray 6.
_FRONT_WALL_X_M = 2.0
_SIDE_WALL_Y_M = 1.5
_NEAR_WALL_Y_M = -0.05
_WALLS = (
    ((_FRONT_WALL_X_M + 0.05, 0.0, _SCAN_HEIGHT_M), (0.05, 0.5, 0.5)),
    ((1.75, _SIDE_WALL_Y_M + 0.05, _SCAN_HEIGHT_M), (1.25, 0.05, 0.5)),
    ((0.0, _NEAR_WALL_Y_M - 0.1, _SCAN_HEIGHT_M), (0.03, 0.1, 0.5)),
)


class FakeIO:
    """The transport as the lidar sensor and the bridge meet it: whether a
    viewer holds a lidar's pair, which pairs were asked for, and every scan
    published, each stamped at the engine instant of its ray 0. The post has
    no limb and no base, so the bridge reads no setpoint."""

    def __init__(self, viewed: bool = True) -> None:
        self.setpoints = None
        self.viewed = viewed
        # (robot, lidar) of each pair asked for.
        self.asked: list[tuple[str, str]] = []
        # (robot, lidar, timestamp, capture index, ranges) of each scan.
        self.scans: list[tuple] = []
        # (engine instant, engine clock of its step) of each stamp asked for.
        self.stamps: list[tuple[float, float]] = []

    def lidar_viewed(self, robot, name) -> bool:
        self.asked.append((robot, name))
        return self.viewed

    def timestamp_at_s(self, instant_s, now_s) -> float:
        self.stamps.append((instant_s, now_s))
        return instant_s

    def publish_scan(self, robot, lidar, timestamp_s, capture_index, ranges) -> bool:
        self.scans.append((robot, lidar, timestamp_s, capture_index, ranges))
        return True

    def record_engine_time(self, _engine_time_s) -> None:
        return None

    def publish_clock_tick(self) -> None:
        return None

    def publish_object_states(self, _robots) -> None:
        return None


@pytest.fixture(name="assets", autouse=True)
def assets_fixture(tmp_path, monkeypatch):
    """Where the post's MJCF is read from, so its scene stands as a baked
    one does."""
    monkeypatch.setattr(mujoco_models, "ASSETS_DIR", tmp_path)
    return tmp_path


def _post(lidar: LidarConfig = _LIDAR, link_bodies=None):
    scene = mujoco_models.ASSETS_DIR / "lidar_post.xml"
    scene.write_text(_POST, encoding="utf-8")
    return known_model(
        "lidar_post", entry("lidar_post", lidars=(lidar,)), scene, "post", link_bodies=link_bodies
    )


def _room(known, placement=None):
    """The post standing among the walls, composed and compiled, its
    kinematics computed, and the world it stands in."""
    world = World(renders=False)
    world.add(ROBOT, known, placement or world.free_spot())
    stage = world.compose()
    for centre, half_sizes in _WALLS:
        wall = stage.worldbody.add_geom()
        wall.type = mujoco.mjtGeom.mjGEOM_BOX
        wall.pos = list(centre)
        wall.size = list(half_sizes)
    model = stage.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, world


def _sensor(io: FakeIO):
    """The post among the walls, with the sensor that sweeps its lidar."""
    model, data, world = _room(_post())
    return model, data, MujocoLidarSensor(model, data, world.robots(), io)


def _step(model, data, sensor: MujocoLidarSensor, steps: int) -> None:
    """`steps` physics steps on from where the scene stands, each swept
    after."""
    for _ in range(steps):
        mujoco.mj_step(model, data)
        sensor.after_step(float(data.time))


def _bridge(io: FakeIO, time_base_s: float = 0.0, carried=None) -> MujocoBridgeExtension:
    """The post among the walls stood as the launcher stands a scene, its
    engine clock running on from `time_base_s`, with what the scene before
    it handed on."""
    model, data, world = _room(_post())
    bridge = MujocoBridgeExtension(
        model, data, io, world, _STATE_RATE_HZ, renders=False,
        time_base_s=time_base_s, carried_sensors=carried,
    )
    bridge.startup()
    return bridge


def _first_ranges() -> list[float]:
    """The ranges of the first scan of the post among the walls: it starts
    at 100 ms and its last ray falls due 87.5 ms later, within 100 steps."""
    io = FakeIO()
    _step(*_sensor(io), 100)
    (_, _, _, _, ranges), *_ = io.scans
    return ranges


def _assert_range(actual: float, expected: float, what: str) -> None:
    assert actual == pytest.approx(expected, abs=_RANGE_TOLERANCE_M), what


class TestWhatARayMeets:
    def test_a_lidar_measures_the_walls_around_it_at_their_distances(self):
        ranges = _first_ranges()

        assert len(ranges) == _LIDAR.sweep.rays
        _assert_range(ranges[0], _FRONT_WALL_X_M, "0 degrees, the front wall")
        _assert_range(ranges[1], _SIDE_WALL_Y_M * math.sqrt(2.0), "45 degrees, the side wall")
        assert ranges[6] == UNMEASURED_MARK_M, "270 degrees, the near wall"

    def test_a_ray_that_meets_nothing_gives_no_return(self):
        """At 90 degrees beside the side wall, at 180 and at 225 degrees."""
        ranges = _first_ranges()

        assert [ranges[2], ranges[4], ranges[5]] == [NO_RETURN_MARK_M] * 3

    def test_a_ray_passes_the_body_that_carries_the_lidar(self):
        """The housing around the lidar's frame stands 4 cm from it on every
        ray, nearer than the lidar's nearest range, so a ray that met it
        gives 0: the one ray that gives 0 meets the near wall."""
        ranges = _first_ranges()

        assert [ray for ray, measured in enumerate(ranges) if measured == UNMEASURED_MARK_M] == [6]

    def test_a_ray_meets_the_other_bodies_of_its_own_robot(self):
        ranges = _first_ranges()

        _assert_range(
            ranges[3], 0.5 * math.sqrt(2.0) - _BEACON_RADIUS_M, "135 degrees, the post's beacon"
        )

    def test_a_ray_passes_the_geoms_no_camera_draws(self):
        """At 315 degrees the post's collision hull, in a group the cameras
        leave out, stands in the ray's way."""
        assert _first_ranges()[7] == NO_RETURN_MARK_M


class TestWhereTheLidarStands:
    def test_its_frame_is_its_pose_on_the_body_its_parent_link_is_mapped_to(self):
        """A lidar 10 cm ahead of its link, turned a quarter turn on it, on
        a robot placed at (3, 4) facing +y: its frame stands 10 cm along the
        robot's heading from the head, facing -x."""
        turned = replace(
            _LIDAR,
            parent_link="head_link",
            pos=(0.1, 0.0, 0.0),
            quat_wxyz=wxyz_of_xyzw(quaternion_of_yaw(math.pi / 2)),
        )
        known = _post(turned, link_bodies={"head_link": "head"})
        model, data, world = _room(known, placement_at(3.0, 4.0, yaw=math.pi / 2))
        sensor = MujocoLidarSensor(model, data, world.robots(), FakeIO())
        (lidar,) = sensor._lidars  # pylint: disable=W0212

        frame = sensor._frame(lidar)  # pylint: disable=W0212

        assert lidar.body == model.body(f"{ROBOT}/head").id
        assert frame.position == pytest.approx((3.0, 4.1, _SCAN_HEIGHT_M), abs=1e-12)
        assert frame.orientation.xyzw == pytest.approx(quaternion_of_yaw(math.pi), abs=1e-12)

    def test_a_lidar_whose_parent_link_is_no_body_of_the_mjcf_is_refused_naming_its_bodies(self):
        lost = replace(_LIDAR, parent_link="mast_link")
        model, data, world = _room(_post(lost))

        with pytest.raises(RuntimeError) as refused:
            MujocoLidarSensor(model, data, world.robots(), FakeIO())

        assert str(refused.value) == (
            "lidar 'lidar' parent link 'mast_link' is not a body of lidar_post.xml; "
            "bodies: ['head', 'post']"
        )


class TestTheSweep:
    def test_a_lidar_is_swept_for_the_pair_its_robot_holds_of_it(self):
        """Over 0.3 s, as the scans of 100 ms and 200 ms fall due, the sweep
        asks for the pair of the robot's own lidar, and while none is held
        casts no ray and publishes nothing."""
        io = FakeIO(viewed=False)
        model, data, sensor = _sensor(io)
        casts = []
        sensor._cast = lambda *ray: casts.append(ray)  # pylint: disable=W0212

        _step(model, data, sensor, 150)

        assert (io.scans, casts) == ([], [])
        assert set(io.asked) == {(ROBOT, "lidar")}

    def test_a_scene_composed_again_drops_the_scan_in_progress_and_counts_on(self):
        """The scene goes at 250 ms, the scan of 200 ms half cast; the engine
        clock carries onto the scene composed next, which sweeps from the
        first whole period after its first step, the capture index counting
        on."""
        io = FakeIO()
        first = _bridge(io)
        for _ in range(125):
            first.step()
        second = _bridge(io, time_base_s=first.engine_time_s(), carried=first.shutdown())
        for _ in range(100):
            second.step()

        starts_ns = [round(timestamp_s * 1e9) for _, _, timestamp_s, _, _ in io.scans]
        assert starts_ns == [_PERIOD_NS, 3 * _PERIOD_NS]
        assert [capture_index for _, _, _, capture_index, _ in io.scans] == [0, 1]

    def test_a_robot_that_stands_again_counts_its_scans_from_0(self):
        """The scene composed next stands no robot, and the one after it
        stands the post again."""
        io = FakeIO()
        first = _bridge(io)
        for _ in range(100):
            first.step()
        empty_world = World(renders=False)
        model = empty_world.compose().compile()
        empty = MujocoBridgeExtension(
            model, mujoco.MjData(model), io, empty_world, _STATE_RATE_HZ, renders=False,
            time_base_s=first.engine_time_s(), carried_sensors=first.shutdown(),
        )
        again = _bridge(io, time_base_s=empty.engine_time_s(), carried=empty.shutdown())
        for _ in range(100):
            again.step()

        assert [capture_index for _, _, _, capture_index, _ in io.scans] == [0, 0]


class TestWhatAScanCarries:
    def test_a_scan_is_its_robots_lidars_stamped_at_its_ray_0_on_the_engine_clock(self):
        """On an engine clock that runs on from 10 s, the first scan starts
        at 10.1 s and its last ray falls due at 10.1875 s, which the
        geometry of the step that ends at 10.19 s is the first to reach."""
        io = FakeIO()
        bridge = _bridge(io, time_base_s=10.0)
        for _ in range(100):
            bridge.step()

        ((robot, lidar, timestamp_s, capture_index, ranges),) = io.scans
        ((ray_0_s, engine_now_s),) = io.stamps
        assert (robot, lidar, capture_index, len(ranges)) == (ROBOT, _LIDAR, 0, _LIDAR.sweep.rays)
        assert isinstance(ranges, list)
        assert timestamp_s == ray_0_s == pytest.approx(10.1, abs=1e-9)
        assert engine_now_s == pytest.approx(10.19, abs=1e-9)
