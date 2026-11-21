"""The entries of the robots these suites stand: what a description gives a
robot (its arms, its grippers with where each finger closes and opens, its
base, its cameras and its start posture), and what the release of Waldo's
catalogue gives its MuJoCo model, written out for a fake robot of a test; and
where a test stands a robot."""

from __future__ import annotations

import sys
from pathlib import Path

from control_core_py.geometry import Pose, quaternion_of_yaw
from robot_description_py import FingerSpan
from sim_robot_core.cameras import CameraConfig
from sim_robot_core.models import Arm, Base, Gripper, ModelEntry
from sim_robot_core.pairs import BASE_NAME

_ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

import importlib.util  # noqa: E402  pylint: disable=C0413
import tempfile  # noqa: E402  pylint: disable=C0413

from catalogue_robots import ROBOTS_FILE, CatalogueRobot, ServoGains  # noqa: E402  pylint: disable=C0413
from mujoco_models import MujocoModel, parse, stood  # noqa: E402  pylint: disable=C0413

_NODES_HUB = Path(__file__).resolve().parents[2]


def _load_script(path: Path):
    """A script of sim_base_images as a module, registered under its name, as
    its dataclasses resolve their annotations through sys.modules."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


waldo_catalogue = _load_script(_NODES_HUB / "sim_base_images" / "waldo_catalogue.py")


def staged_release(baked: Path) -> Path:
    """The pinned release of Waldo's catalogue: the one baked at `baked`, or,
    where none is baked, the one these suites stage in the machine's
    temporary directory, under the pinned digest, which the first run stages
    over the network. Either way every file is checked against the pin."""
    if (baked / ROBOTS_FILE).is_file():
        release = baked
    else:
        release = Path(tempfile.gettempdir()) / f"sim_mujoco-release-{waldo_catalogue.RELEASE_DIGEST}"
    waldo_catalogue.fetch(release)
    return release


def entry(
    model: str,
    arms: dict | None = None,
    grippers: dict | None = None,
    cameras: tuple[CameraConfig, ...] = (),
    start_posture: dict | None = None,
    wheels: tuple[str, ...] = (),
) -> ModelEntry:
    """The entry of a robot of `model`: `arms` maps each arm to its joints,
    `grippers` each gripper to its fingers, each finger a joint that moves of
    its own with the position it closes at and the one it opens at, and
    `wheels` are the joints of its base's wheels, left then right, none for a
    robot without a base."""
    return ModelEntry(
        model=model,
        arms=tuple(Arm(name=name, joints=tuple(joints)) for name, joints in (arms or {}).items()),
        grippers=tuple(
            Gripper(
                name=name,
                joints=tuple(fingers),
                spans=tuple(FingerSpan(closed, opened) for closed, opened in fingers.values()),
                mimics=(None,) * len(fingers),
            )
            for name, fingers in (grippers or {}).items()
        ),
        cameras=cameras,
        start_posture=dict(start_posture or {}),
        base=Base(name=BASE_NAME, wheels=tuple(wheels)) if wheels else None,
    )


def placement_at(x: float, y: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> Pose:
    """A robot's placement at (x, y, z) in world, turned `yaw` about +z."""
    return Pose.parse((x, y, z), quaternion_of_yaw(yaw))


def camera(name: str, parent_link: str, pos, quat_wxyz, fovy_deg: float, width: int, height: int, fps: int, depth=None) -> CameraConfig:
    return CameraConfig(
        name=name,
        frame=f"{name}_optical_frame",
        parent_link=parent_link,
        pos=tuple(pos),
        quat_wxyz=tuple(quat_wxyz),
        fovy_deg=fovy_deg,
        width=width,
        height=height,
        fps=fps,
        depth=depth,
    )


def known(
    model: str,
    robot: ModelEntry,
    scene: Path,
    mount: str,
    *,
    engine: dict | None = None,
    link_bodies: dict | None = None,
    gravity_compensation: bool = False,
    arm_gains: dict | None = None,
    gripper_gains: dict | None = None,
) -> MujocoModel:
    """A robot of `model` that `robot` describes, as this engine stands it:
    its MJCF the file `scene`, whose body `mount` carries it, rolling exactly
    when `robot` has a base, with what the release gives (`arm_gains` maps an
    arm to its (kp, kd) per joint, `gripper_gains` an arm to its gripper's
    (kp, kd)) and this engine's entry `engine`."""
    release = CatalogueRobot(
        model=model,
        file=scene,
        mount_body=mount,
        rolls=robot.base is not None,
        link_bodies=dict(link_bodies or {}),
        gravity_compensation=gravity_compensation,
        arm_gains={arm: tuple(ServoGains(kp, kd) for kp, kd in gains) for arm, gains in (arm_gains or {}).items()},
        gripper_gains={arm: ServoGains(*gains) for arm, gains in (gripper_gains or {}).items()},
    )
    return stood(robot, release, parse(model, engine or {}))


# A base on two wheels, each run by the rate loop of a velocity and an
# integrated-velocity actuator whose force the joint caps, and a caster that
# takes no friction: the shape of the botshare base's MJCF, small.
WHEEL_CAP_NM = 4.0
WHEEL_INTEGRAL_GAIN = 20.0
WHEEL_RADIUS_M = 0.05
WHEELED = f"""<mujoco model="wheeled">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <default>
    <default class="wheel">
      <joint type="hinge" axis="0 1 0" actuatorfrcrange="-{WHEEL_CAP_NM} {WHEEL_CAP_NM}"/>
    </default>
  </default>
  <worldbody>
    <body name="chassis">
      <geom type="box" size="0.2 0.15 0.04" pos="0 0 0.1" mass="5"/>
      <geom type="sphere" size="0.025" pos="0.15 0 0.025" condim="1" priority="2" mass="0.1"/>
      <geom type="sphere" size="0.025" pos="-0.15 0 0.025" condim="1" priority="2" mass="0.1"/>
      <body name="left_wheel_link" pos="0 0.2 {WHEEL_RADIUS_M}">
        <joint name="left_wheel" class="wheel"/>
        <geom type="cylinder" size="{WHEEL_RADIUS_M} 0.02" quat="0.7071068 0.7071068 0 0" mass="0.5"/>
      </body>
      <body name="right_wheel_link" pos="0 -0.2 {WHEEL_RADIUS_M}">
        <joint name="right_wheel" class="wheel"/>
        <geom type="cylinder" size="{WHEEL_RADIUS_M} 0.02" quat="0.7071068 0.7071068 0 0" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="left_wheel_proportional" joint="left_wheel" kv="10"/>
    <intvelocity name="left_wheel_integral" joint="left_wheel" kp="{WHEEL_INTEGRAL_GAIN}"/>
    <velocity name="right_wheel_proportional" joint="right_wheel" kv="10"/>
    <intvelocity name="right_wheel_integral" joint="right_wheel" kp="{WHEEL_INTEGRAL_GAIN}"/>
  </actuator>
</mujoco>"""
WHEELED_MOUNT = "chassis"
WHEELED_ENTRY = entry("wheeled", wheels=("left_wheel", "right_wheel"))


def wheeled(directory: Path, xml: str = WHEELED) -> MujocoModel:
    """The wheeled base as this engine stands it, its MJCF `xml` written into
    `directory`."""
    scene = directory / "wheeled.xml"
    scene.write_text(xml)
    return known("wheeled", WHEELED_ENTRY, scene, WHEELED_MOUNT)
