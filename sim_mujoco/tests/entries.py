"""The entries of the robots these suites stand: what a description gives a
robot (its arms, its grippers with where each finger closes and opens, its
cameras and its start posture), and what the release of Waldo's catalogue
gives its MuJoCo model, written out for a fake robot of a test."""

from __future__ import annotations

import sys
from pathlib import Path

from sim_robot_core.cameras import CameraConfig
from sim_robot_core.models import Arm, FingerSpan, Gripper, ModelEntry

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
) -> ModelEntry:
    """The entry of a robot of `model`: `arms` maps each arm to its joints,
    and `grippers` each gripper to its fingers, each finger a joint that
    moves of its own with the position it closes at and the one it opens
    at."""
    return ModelEntry(
        model=model,
        arms=tuple(Arm(name=name, joints=tuple(joints)) for name, joints in (arms or {}).items()),
        grippers=tuple(
            Gripper(
                name=name,
                joints=tuple(fingers),
                spans=tuple(
                    FingerSpan(closed=closed, travel=opened - closed)
                    for closed, opened in fingers.values()
                ),
                mimics=(None,) * len(fingers),
            )
            for name, fingers in (grippers or {}).items()
        ),
        cameras=cameras,
        start_posture=dict(start_posture or {}),
    )


def camera(name: str, parent_link: str, pos, quat_wxyz, fovy_deg: float, width: int, height: int, fps: int, depth=None) -> CameraConfig:
    return CameraConfig(
        name=name,
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
    its MJCF the file `scene`, whose body `mount` carries it, with what the
    release gives (`arm_gains` maps an arm to its (kp, kd) per joint,
    `gripper_gains` an arm to its gripper's (kp, kd)) and this engine's entry
    `engine`."""
    release = CatalogueRobot(
        model=model,
        file=scene,
        mount_body=mount,
        link_bodies=dict(link_bodies or {}),
        gravity_compensation=gravity_compensation,
        arm_gains={arm: tuple(ServoGains(kp, kd) for kp, kd in gains) for arm, gains in (arm_gains or {}).items()},
        gripper_gains={arm: ServoGains(*gains) for arm, gains in (gripper_gains or {}).items()},
    )
    return stood(robot, release, parse(model, engine or {}))
