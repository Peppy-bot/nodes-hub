"""The models this engine stands, one entry per model under models/.

What a robot of a model is made of (its limbs, joints, cameras and start
posture) is sim_robot_core's entry of the same name, shared with every
engine. What is here is what MuJoCo alone knows about it: the MJCF it loads,
how the robot's URDF links map onto that file's bodies, its servo gains, and
where the file is corrected to the robot's description. A model with no
entry here is refused, and none falls back to another's.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sim_robot_core.models import EngineModel, ModelEntry, Models

ENTRIES_DIR = Path(__file__).parent / "models"

# Where the base image bakes every model's files.
ASSETS_DIR = Path(
    os.environ.get("PEPPY_ROBOT_ASSETS_DIR", str(Path(__file__).parent / "assets" / "robots"))
)

_ENTRY_KEYS = frozenset(
    {
        "scene",
        "world_links",
        "link_bodies",
        "arm_gains",
        "gravity_compensation",
        "camera_lights",
        "head_camera",
        "joint_ranges",
        "site_poses",
    }
)
_GAIN_KEYS = frozenset({"kp", "kd"})
_SITE_POSE_KEYS = frozenset({"pos", "quat_wxyz"})


@dataclass(frozen=True)
class ArmGains:
    """MIT servo gains, one per joint of an arm, applied to every arm of the
    model."""

    kp: tuple[float, ...]
    kd: tuple[float, ...]


@dataclass(frozen=True)
class SitePose:
    """A site's pose in its body's frame."""

    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class MujocoModel:
    """One model this engine stands."""

    entry: ModelEntry
    # The robot's MJCF, relative to the baked assets. It is the whole world
    # the robot stands in.
    scene: str
    # URDF links the MJCF compiler folds into the world body, so whatever
    # hangs from one hangs from the world.
    world_links: frozenset
    # URDF link name to the MJCF body that is that link, where they differ.
    link_bodies: dict
    # None for a model driven by its file's own actuator model.
    arm_gains: Optional[ArmGains]
    gravity_compensation: bool
    # Whether rendering adds the engine's light rig, for a scene that ships
    # no light of its own.
    camera_lights: bool
    # Whether the robot draws the OpenArm v2 head camera pack.
    head_camera: bool
    # Where the file is corrected to the robot's description.
    joint_ranges: dict
    site_poses: dict

    @property
    def model(self) -> str:
        return self.entry.model

    def scene_path(self) -> Path:
        """The MJCF on disk."""
        path = ASSETS_DIR / self.scene
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: the models are baked into the container image"
            )
        return path

    def body_of(self, link: str) -> str:
        """The MJCF body a URDF link is."""
        return self.link_bodies.get(link, link)

    def actuator_params(self) -> dict:
        """The gains of every arm joint, the same per-joint gains on each
        arm."""
        gains = self.arm_gains
        arms = self.entry.arms
        return {
            "joint_names": self.entry.arm_joints() if gains else [],
            "kp": list(gains.kp) * len(arms) if gains else [],
            "kd": list(gains.kd) * len(arms) if gains else [],
        }


def _fail(model: str, reason: str) -> RuntimeError:
    return RuntimeError(f"models/{model}.json5: {reason}")


def _reject_unknown_keys(model: str, obj: dict, allowed: frozenset, what: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise _fail(model, f"{what} has unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numbers(model: str, value, length: int, what: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length or not all(map(_is_number, value)):
        raise _fail(model, f"{what} must be {length} finite numbers, got {value!r}")
    return tuple(float(v) for v in value)


def _flag(model: str, raw: dict, key: str) -> bool:
    value = raw.get(key, False)
    if not isinstance(value, bool):
        raise _fail(model, f"{key} must be true or false, got {value!r}")
    return value


def _names(model: str, value, what: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(name, str) and name for name in value):
        raise _fail(model, f"{what} must be a list of names, got {value!r}")
    return value


def _arm_gains(model: str, entry: ModelEntry, raw) -> Optional[ArmGains]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _fail(model, f"arm_gains must be an object, got {raw!r}")
    _reject_unknown_keys(model, raw, _GAIN_KEYS, "arm_gains")
    joint_counts = set(entry.arm_joint_counts())
    if len(joint_counts) != 1:
        raise _fail(model, f"arm_gains serve arms of one joint count, and these have {sorted(joint_counts)}")
    (joints,) = joint_counts
    return ArmGains(
        kp=_numbers(model, raw.get("kp"), joints, "arm_gains.kp"),
        kd=_numbers(model, raw.get("kd"), joints, "arm_gains.kd"),
    )


def _link_bodies(model: str, raw) -> dict:
    if not isinstance(raw, dict) or not all(
        isinstance(body, str) and body for body in raw.values()
    ):
        raise _fail(model, f"link_bodies must map link names to body names, got {raw!r}")
    return dict(raw)


def _joint_ranges(model: str, entry: ModelEntry, raw) -> dict:
    if not isinstance(raw, dict):
        raise _fail(model, f"joint_ranges must map joints to [lower, upper], got {raw!r}")
    unknown = sorted(set(raw) - set(entry.joints()))
    if unknown:
        raise _fail(model, f"joint_ranges names joints no limb moves: {unknown}")
    ranges = {}
    for joint, limits in raw.items():
        lower, upper = _numbers(model, limits, 2, f"joint_ranges.{joint}")
        if not lower < upper:
            raise _fail(model, f"joint_ranges.{joint} [{lower}, {upper}] is not a range")
        ranges[joint] = (lower, upper)
    return ranges


def _site_poses(model: str, raw) -> dict:
    if not isinstance(raw, dict):
        raise _fail(model, f"site_poses must map sites to a pose, got {raw!r}")
    poses = {}
    for site, pose in raw.items():
        if not isinstance(pose, dict):
            raise _fail(model, f"site_poses.{site} must be an object, got {pose!r}")
        _reject_unknown_keys(model, pose, _SITE_POSE_KEYS, f"site_poses.{site}")
        quat = _numbers(model, pose.get("quat_wxyz"), 4, f"site_poses.{site}.quat_wxyz")
        norm = math.sqrt(sum(v * v for v in quat))
        if not math.isclose(norm, 1.0, abs_tol=1e-3):
            raise _fail(model, f"site_poses.{site}.quat_wxyz norm {norm} is not 1")
        poses[site] = SitePose(
            pos=_numbers(model, pose.get("pos"), 3, f"site_poses.{site}.pos"), quat_wxyz=quat
        )
    return poses


def parse(known: EngineModel) -> MujocoModel:
    """This engine's entry of one model, refusing anything it does not spell
    out."""
    entry, raw = known.entry, known.engine
    model = entry.model
    _reject_unknown_keys(model, raw, _ENTRY_KEYS, "the entry")
    scene = raw.get("scene")
    if not isinstance(scene, str) or not scene:
        raise _fail(model, f"scene must name the model's MJCF, got {scene!r}")
    return MujocoModel(
        entry=entry,
        scene=scene,
        world_links=frozenset(_names(model, raw.get("world_links", []), "world_links")),
        link_bodies=_link_bodies(model, raw.get("link_bodies", {})),
        arm_gains=_arm_gains(model, entry, raw.get("arm_gains")),
        gravity_compensation=_flag(model, raw, "gravity_compensation"),
        camera_lights=_flag(model, raw, "camera_lights"),
        head_camera=_flag(model, raw, "head_camera"),
        joint_ranges=_joint_ranges(model, entry, raw.get("joint_ranges", {})),
        site_poses=_site_poses(model, raw.get("site_poses", {})),
    )


class MujocoModels:
    """The models this engine stands, by the id a robot attaches with."""

    def __init__(self, models: Models) -> None:
        self._models = models
        self._parsed = {name: parse(models.of(name)) for name in models.names()}

    @staticmethod
    def read(entries_dir: Path = ENTRIES_DIR) -> "MujocoModels":
        return MujocoModels(Models.read(entries_dir))

    def names(self) -> list[str]:
        return self._models.names()

    def of(self, model: str) -> MujocoModel:
        """The model a robot attaches as; an id this engine has no entry for
        is refused naming the ones it stands."""
        self._models.of(model)
        return self._parsed[model]
