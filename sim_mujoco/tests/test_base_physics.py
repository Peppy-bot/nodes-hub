"""The botshare base on the wheels of its base, as this engine stands it from
the release of Waldo's catalogue: its wheels and its tires as
robot_description's URDF gives them, driven as its base node drives them, a
wheel setpoint at each step of the node's limiter held until the next. Its
wheels drive it straight as the differential mix of their measured turns
predicts, and through a turn as the mix predicts on a track within its
tires' width of its own; standing at rate 0, its wheel states give the base
node a velocity at rest. Every outcome is read after a step count, so none
depends on the host's speed. The bounds are Waldo's (private-nodes-hub
waldo/crates/sim_mujoco/src/testing/base_physics.rs), both engines standing
one model of the base."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import pytest
import robot_description_py
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
# botshare_base/fragments/botshare_base_sim.json5, smoothing_frequency_hz):
# one wheel setpoint each step of it.
LIMITER_RATE_HZ = 20
# The base's rest velocity and rest rate in the same launchers (x.rest_velocity
# in m/s, angular.rest_rate in rad/s), and its odometry window: each velocity
# the node reports is the mean over that many of its 100 Hz wheel states.
REST_VELOCITY_M_S = 0.005
REST_RATE_RAD_S = 0.01
WHEEL_STATE_RATE_HZ = 100
ODOMETRY_WINDOW = 10
# How long the base settles at rate 0 before a drive, as Waldo's tests settle it.
SETTLE_S = 0.5


@pytest.fixture(name="release", scope="module", autouse=True)
def release_fixture():
    release = staged_release(mujoco_models.ASSETS_DIR)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mujoco_models, "ASSETS_DIR", release)
        yield release


@dataclass(frozen=True)
class Wheels:
    """The base's wheels as its URDF gives them: their radius and track (m),
    and the width of their tires (m), each tire its link's one collision
    cylinder."""

    radius_m: float
    separation_m: float
    tire_width_m: float

    @staticmethod
    def described() -> Wheels:
        description = robot_description_py.model(MODEL).description()
        base = description.differential_base()
        assert base.left.radius_m == base.right.radius_m, "the base's two wheels are alike"
        urdf = ET.fromstring(description.urdf)
        child = {joint.get("name"): joint.find("child").get("link") for joint in urdf.iter("joint")}
        links = {link.get("name"): link for link in urdf.iter("link")}
        widths = {
            float(links[child[wheel.joint]].find("collision/geometry/cylinder").get("length"))
            for wheel in (base.left, base.right)
        }
        assert len(widths) == 1, "the base's two tires are alike"
        return Wheels(
            radius_m=base.left.radius_m, separation_m=base.separation_m, tire_width_m=widths.pop()
        )


WHEELS = Wheels.described()


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


class _Driver:
    """The transport as the bridge meets it while a base node drives the
    wheels: the latest setpoint of the base, and nothing published."""

    def __init__(self) -> None:
        self.base: JointSetpoint | None = None

    def latest_base_command(self, _robot):
        return self.base

    def latest_arm_command(self, _robot, _arm):
        return None

    def latest_gripper_command(self, _robot, _gripper):
        return None

    def record_engine_time(self, _engine_time_s) -> None:
        return None

    def publish_clock_tick(self) -> None:
        return None

    def publish_base_states(self, _robot, _positions, _velocities) -> None:
        return None


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


class DrivenBase:
    """The base standing alone as `india` at the origin, its wheels driven as
    its base node drives them."""

    def __init__(self) -> None:
        known = MujocoModels.read().of(MODEL, robot_description_py.model(MODEL).description())
        self.world = World(renders=False)
        self.world.add(ROBOT, known, self.world.free_spot())
        self.model = self.world.compose().compile()
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
                self.driver.base = JointSetpoint.parse(before, rates(step * self.timestep_s), 2, "wheels")
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
