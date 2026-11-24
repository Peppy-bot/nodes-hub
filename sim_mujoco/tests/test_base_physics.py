"""The botshare base on the wheels of its base, as this engine stands it from
the release of Waldo's catalogue: its wheels and its tires as
robot_description's URDF gives them, driven as its base node drives them, a
wheel setpoint at each step of the node's limiter held until the next. Its
wheels drive it straight as the differential mix of their measured turns
predicts, and through a turn as the mix predicts on a track within its
tires' width of its own; standing at rate 0, its wheel states give the base
node a velocity at rest; lifted off the floor, its wheels spin while its
body stays. Every outcome is read after a step count, so none depends on the
host's speed. The bounds are Waldo's (private-nodes-hub
waldo/crates/sim_mujoco/src/testing/base_physics.rs), both engines standing
one model of the base."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import pytest
import robot_description_py
from control_core_py import geometry
from control_core_py.geometry import xyzw_of_wxyz, yaw_of_quaternion
from entries import staged_release
from sim_robot_core.setpoints import JointSetpoint

import runtime_fakes  # noqa: F401  pylint: disable=W0611  (installs the runtime the bridge imports)

import mujoco_models  # noqa: E402  pylint: disable=C0413
from bridge_extension import MujocoBridgeExtension  # noqa: E402  pylint: disable=C0413
from mujoco_models import MujocoModels  # noqa: E402  pylint: disable=C0413
from world import World  # noqa: E402  pylint: disable=C0413

MODEL = "botshare_base"
ROBOT = "india"
# The base node's limiter rate in the simulation launchers (launchers-hub
# botshare/fragments/botshare_sim.json5, smoothing_frequency_hz):
# one wheel setpoint each step of it.
LIMITER_RATE_HZ = 20
# The base's rest velocity in the same launchers (linear_limits.rest_velocity,
# m/s), the speed each wheel measures at most while the base stands, and its odometry
# window there (velocity_rolling_window_size): each velocity the node reports is
# the mean over that many of its wheel states.
REST_VELOCITY_M_S = 0.02
ODOMETRY_WINDOW = 10
# This engine's state rate in the simulation launchers (launchers-hub
# simulation/fragments/mujoco.json5, state_rate_hz): the rate its wheel states
# go out at.
WHEEL_STATE_RATE_HZ = 100
# How long the base settles at rate 0 before a drive, as Waldo's tests settle it.
SETTLE_S = 0.5
# Half the side of the block a lifted base rests its belly on (m): Waldo's
# lifted-base block (base_physics.rs BLOCK_HALF_M), the fixture cube scaled
# to 0.24 m wide, which fits under the belly clear of the wheels and the
# casters.
BLOCK_HALF_M = 0.12


@pytest.fixture(name="release", scope="module", autouse=True)
def release_fixture():
    release = staged_release(mujoco_models.ASSETS_DIR)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mujoco_models, "ASSETS_DIR", release)
        yield release


@dataclass(frozen=True)
class Wheels:
    """The base's wheels as its description gives them: their radius and
    track (m), and the width of their tires (m), each tire its link's one
    collision cylinder."""

    radius_m: float
    separation_m: float
    tire_width_m: float

    @staticmethod
    def described() -> Wheels:
        base = robot_description_py.model(MODEL).description().differential_base()
        assert (base.left.radius_m, base.left.width_m) == (base.right.radius_m, base.right.width_m), (
            "the base's two wheels are alike"
        )
        return Wheels(
            radius_m=base.left.radius_m,
            separation_m=base.separation_m,
            tire_width_m=base.left.width_m,
        )


WHEELS = Wheels.described()
# The base's rest rate (rad/s), as its node derives it from the rest velocity:
# the turn the odometry reads with each wheel at the rest velocity the other
# way, twice it over the separation (control_core's
# WheelGeometry::turn_rate_at_opposite_wheel_speeds).
REST_RATE_RAD_S = 2.0 * REST_VELOCITY_M_S / WHEELS.separation_m


@dataclass(frozen=True)
class Pose:
    """Where the base stands on the floor: its position (m) and its heading
    (rad)."""

    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Odometry:
    """What the differential mix of the wheels' measured turns predicts of
    the base: where it carries it, and the length of the path."""

    pose: Pose
    path_m: float

    def advanced(self, left: float, right: float, track_m: float) -> Odometry:
        """This odometry carried on by a turn of `left` and `right` (rad) of
        the wheels, on a track of `track_m`."""
        ahead = WHEELS.radius_m * (left + right) / 2.0
        turned = WHEELS.radius_m * (right - left) / track_m
        heading = self.pose.yaw + turned / 2.0
        pose = Pose(
            x=self.pose.x + ahead * math.cos(heading),
            y=self.pose.y + ahead * math.sin(heading),
            yaw=self.pose.yaw + turned,
        )
        return Odometry(pose=pose, path_m=self.path_m + abs(ahead))


class _BaseSetpoint:
    """The setpoints as the bridge reads them while a base node drives the
    wheels: the latest setpoint of the base, which a step of its limiter
    sends before its window ends."""

    def __init__(self) -> None:
        self.latest: JointSetpoint | None = None

    def base(self, _robot, _now_s):
        return self.latest

    def take_base_release(self, _robot) -> bool:
        return False

    def arm(self, _robot, _arm):
        return None

    def gripper(self, _robot, _gripper):
        return None


class _Driver:
    """The transport as the bridge meets it while a base node drives the
    wheels: the base's setpoint, nothing published, and no viewer of its
    lidar, which therefore casts no ray."""

    def __init__(self) -> None:
        self.setpoints = _BaseSetpoint()

    def publish_object_states(self, _robots) -> None:
        return None

    def record_engine_time(self, _engine_time_s) -> None:
        return None

    def publish_clock_tick(self) -> None:
        return None

    def publish_base_states(self, _robot, _positions, _velocities) -> None:
        return None

    def lidar_viewed(self, _robot, _lidar) -> bool:
        return False


@dataclass
class Drive:
    """What a drive of the wheels gives: where the base stood as it began,
    the wheels' measured turns over each physics step (rad, left then right),
    and their angles at each wheel state."""

    start: Pose
    turns: list[tuple[float, float]]
    angles: list[tuple[float, float]]

    def odometry_on(self, track_m: float) -> Odometry:
        odometry = Odometry(pose=self.start, path_m=0.0)
        for left, right in self.turns:
            odometry = odometry.advanced(left, right, track_m)
        return odometry


def _belly_height_m(known) -> float:
    """The lowest face of the base link's collision box above the robot
    frame (m), as the base's MJCF places it standing level: the belly Waldo's
    lifted base rests on (base_physics.rs Chassis::belly_height_m)."""
    world = World(renders=False)
    world.add(ROBOT, known, world.free_spot())
    model = world.compose().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    box = model.geom(f"{ROBOT}/base_link")
    assert int(box.type[0]) == int(mujoco.mjtGeom.mjGEOM_BOX), "the base link collides as a box"
    assert list(data.geom_xmat[box.id]) == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0, 1]), "standing level"
    return float(data.geom_xpos[box.id][2] - box.size[2])


def _lay_block(stage) -> None:
    """A static block at the origin, BLOCK_HALF_M each way of its centre, its
    top BLOCK_HALF_M above the floor."""
    block = stage.worldbody.add_geom()
    block.type = mujoco.mjtGeom.mjGEOM_BOX
    block.size = [BLOCK_HALF_M] * 3


class DrivenBase:
    """The base standing alone as `india` at the origin, its wheels driven as
    its base node drives them: on the floor, or, `lifted`, with its belly on
    a block under it, its wheels and its casters clear of the floor."""

    def __init__(self, lifted: bool = False) -> None:
        known = MujocoModels.read().of(MODEL, robot_description_py.model(MODEL).description())
        height_m = BLOCK_HALF_M - _belly_height_m(known) if lifted else 0.0
        self.world = World(renders=False)
        self.world.add(ROBOT, known, geometry.Pose.parse((0.0, 0.0, height_m), geometry.IDENTITY))
        stage = self.world.compose()
        if lifted:
            _lay_block(stage)
        self.model = stage.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.driver = _Driver()
        self.bridge = MujocoBridgeExtension(
            self.model, self.data, self.driver, self.world, WHEEL_STATE_RATE_HZ, renders=False,
            time_base_s=0.0,
        )
        self.bridge.startup(posture_for=self.world.robots())
        self.timestep_s = float(self.model.opt.timestep)
        self.wheels = [
            int(self.model.jnt_qposadr[self.model.joint(f"{ROBOT}/{joint}").id])
            for joint in known.entry.wheel_joints()
        ]
        self.free = int(self.model.jnt_qposadr[self.model.joint(ROBOT).id])
        self.drive(SETTLE_S, lambda _t: (0.0, 0.0))

    def _steps(self, seconds: float) -> int:
        return round(seconds / self.timestep_s)

    def angles(self) -> tuple[float, float]:
        left, right = self.wheels
        return float(self.data.qpos[left]), float(self.data.qpos[right])

    def pose(self) -> Pose:
        at = self.free
        heading = yaw_of_quaternion(xyzw_of_wxyz(self.data.qpos[at + 3 : at + 7]))
        return Pose(x=float(self.data.qpos[at]), y=float(self.data.qpos[at + 1]), yaw=heading)

    def height_m(self) -> float:
        """How high the robot frame stands above the floor (m)."""
        return float(self.data.qpos[self.free + 2])

    def drive(self, seconds: float, rates) -> Drive:
        """Steps for `seconds`, each step of the node's limiter sending the
        wheel rates `rates` gives for the time since the drive began, with
        the wheels' measured angles, held until its next step."""
        per_setpoint = self._steps(1.0 / LIMITER_RATE_HZ)
        per_state = self._steps(1.0 / WHEEL_STATE_RATE_HZ)
        drive = Drive(start=self.pose(), turns=[], angles=[self.angles()])
        for step in range(self._steps(seconds)):
            before = self.angles()
            if step % per_setpoint == 0:
                self.driver.setpoints.latest = JointSetpoint.parse_wheels(
                    before, rates(step * self.timestep_s), 2
                )
            self.bridge.step()
            after = self.angles()
            drive.turns.append((after[0] - before[0], after[1] - before[1]))
            if (step + 1) % per_state == 0:
                drive.angles.append(after)
        return drive


def ramped(left: float, right: float):
    """Wheel rates (rad/s) that ramp from rest to `left` and `right` over the
    first second."""
    return lambda t: (left * min(t, 1.0), right * min(t, 1.0))


def plan_distance(one: Pose, other: Pose) -> float:
    return math.hypot(one.x - other.x, one.y - other.y)


def test_the_wheels_drive_the_base_straight_as_far_as_they_roll():
    """Driven straight for 3 s, up to 5 rad/s (0.5 m/s), the base goes as far
    as its wheels roll: it stands within 0.5 % of the distance from where
    their measured turns predict, facing within 1 mrad of their heading."""
    india = DrivenBase()
    odometry = india.drive(3.0, ramped(5.0, 5.0)).odometry_on(WHEELS.separation_m)
    reached = india.pose()
    assert reached.x > 0.9, f"it drove ahead: {reached}"
    gap = plan_distance(odometry.pose, reached)
    assert gap < 0.005 * odometry.path_m, f"{gap} m from {odometry.pose} after {odometry.path_m} m"
    assert abs(math.remainder(odometry.pose.yaw - reached.yaw, math.tau)) < 1e-3, reached


def test_the_wheels_turn_the_base_as_their_differential_mix_predicts():
    """Driven through a left turn for 4 s, its wheels up to 2 and 4 rad/s, the
    base turns more than 1.2 rad as the differential mix of their measured
    turns predicts on a track within its tires' width of its URDF's: each
    tire meets the floor across its width, and its edges scrub as the base
    turns. On the track whose mix turns it as far as it turned, the mix
    stands it from where it stands within the straight drive's 0.5 % of its
    path and the turn of half a tire's width."""
    india = DrivenBase()
    drive = india.drive(4.0, ramped(2.0, 4.0))
    reached = india.pose()
    mixed = drive.odometry_on(WHEELS.separation_m).pose.yaw - drive.start.yaw
    assert mixed > 1.2, f"the wheels turned it left: {mixed} rad"
    # A mix turns the base by the inverse of its track.
    turned = reached.yaw - drive.start.yaw
    track = WHEELS.separation_m * mixed / turned
    assert abs(track - WHEELS.separation_m) <= WHEELS.tire_width_m, (
        f"it turned {turned} rad, as a mix of {mixed} rad on its {WHEELS.separation_m} m track "
        f"turns it on a {track} m track"
    )
    odometry = drive.odometry_on(track)
    gap = plan_distance(odometry.pose, reached)
    assert gap < 0.005 * odometry.path_m + WHEELS.tire_width_m / 2.0 * turned, (
        f"{gap} m from {odometry.pose} along {odometry.path_m} m through {turned} rad"
    )


def test_a_standing_base_reports_a_velocity_at_rest():
    """Standing for 10 s, its wheels driven at rate 0, the base gives at
    every wheel state a velocity within its rest velocity and rest rate, as
    its node computes it: the mean over its odometry window of the
    differential mix of the wheels' rates."""
    india = DrivenBase()
    angles = india.drive(10.0, lambda _t: (0.0, 0.0)).angles
    window_s = ODOMETRY_WINDOW / WHEEL_STATE_RATE_HZ
    moving = []
    for state in range(ODOMETRY_WINDOW, len(angles)):
        (left_then, right_then), (left, right) = angles[state - ODOMETRY_WINDOW], angles[state]
        linear = WHEELS.radius_m * ((left - left_then) + (right - right_then)) / 2.0 / window_s
        angular = WHEELS.radius_m * ((right - right_then) - (left - left_then)) / WHEELS.separation_m / window_s
        if abs(linear) > REST_VELOCITY_M_S or abs(angular) > REST_RATE_RAD_S:
            moving.append((state / WHEEL_STATE_RATE_HZ, linear, angular))
    assert moving == [], "the time (s), linear (m/s) and angular (rad/s) velocity of each state past rest"


def test_a_base_lifted_off_the_floor_spins_its_wheels_while_its_body_stays():
    """On a block narrower than its track, under its belly, the base's
    wheels clear the floor: driven 2 s up to 5 rad/s they turn far enough to
    roll it over half a metre, and its body moves less than a millimetre and
    turns less than a milliradian, Waldo's bounds of the same drive."""
    india = DrivenBase(lifted=True)
    start, start_height_m = india.pose(), india.height_m()

    odometry = india.drive(2.0, ramped(5.0, 5.0)).odometry_on(WHEELS.separation_m)

    assert odometry.path_m > 0.5, f"the wheels rolled {odometry.path_m} m"
    reached = india.pose()
    moved = math.hypot(plan_distance(start, reached), india.height_m() - start_height_m)
    assert moved < 1e-3, f"the body moved {moved} m: {start}, then {reached}"
    assert abs(math.remainder(reached.yaw - start.yaw, math.tau)) < 1e-3, reached
