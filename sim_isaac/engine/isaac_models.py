"""The models this engine stands, one entry per model under models/.

What a robot of a model is made of (its limbs, joints, cameras and start
posture) is sim_robot_core's entry of the same name, shared with every
engine. What is here is what Isaac Sim alone knows about it: the USD stage it
references, where that stage keeps its articulation root and its links, the
gains and effort ceilings of its PhysX drives, and whether its weight is
compensated. A model with no entry here is refused, and none falls back to
another's.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

from sim_robot_core.models import Arm, EngineModel, Gripper, ModelEntry, Models

ENTRIES_DIR = Path(__file__).parent / "models"

# Where the base image bakes every model's files, one directory per robot.
ASSETS_DIR = Path(
    os.environ.get("PEPPY_ROBOT_ASSETS_DIR", str(Path(__file__).parent / "assets" / "robots"))
)

# The articulation root of a stage whose default prim carries it: the prim
# the robot is referenced under.
ROBOT_PRIM = "."

_ENTRY_KEYS = frozenset(
    {
        "stage",
        "articulation_root",
        "link_prims",
        "arm_gains",
        "gripper_gains",
        "gravity_compensation",
        "head_camera",
    }
)
_GAIN_KEYS = frozenset({"kp", "kd", "max_efforts"})


@dataclass(frozen=True)
class DriveGains:
    """PhysX drive gains and effort ceilings, one per joint of a limb, applied
    to every limb of that kind the model has. The articulation view takes
    them per radian and per metre: N*m/rad and N*m*s/rad on a revolute joint,
    N/m and N*s/m on a prismatic one."""

    kp: tuple[float, ...]
    kd: tuple[float, ...]
    max_efforts: tuple[float, ...]


@dataclass(frozen=True)
class IsaacModel:
    """One model this engine stands."""

    entry: ModelEntry
    # The robot's USD, relative to the baked assets. It is referenced under a
    # prim of the robot's own name, so its default prim is the robot's prim.
    stage: str
    # Where the stage keeps its articulation root, relative to the robot's
    # prim: ROBOT_PRIM for a stage whose default prim carries it.
    articulation_root: str
    # URDF link name to the name of the prim that is that link under the
    # robot's prim, where they differ.
    link_prims: dict
    # None for a limb driven by the gains its stage authors.
    arm_gains: Optional[DriveGains]
    gripper_gains: Optional[DriveGains]
    gravity_compensation: bool
    # Whether the robot draws the OpenArm v2 head camera pack.
    head_camera: bool

    @property
    def model(self) -> str:
        return self.entry.model

    def stage_path(self) -> Path:
        """The USD on disk."""
        path = ASSETS_DIR / self.stage
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: the models are baked into the container image"
            )
        return path

    def articulation_path(self, robot_prim: str) -> str:
        """The prim the engine's views read this model's articulation at, for
        a robot referenced under `robot_prim`."""
        if self.articulation_root == ROBOT_PRIM:
            return robot_prim
        return f"{robot_prim}/{self.articulation_root}"

    def prim_of(self, link: str) -> str:
        """The name of the prim a URDF link is."""
        return self.link_prims.get(link, link)

    def arm_params(self, arm: Arm) -> dict:
        """What one arm's actuator controller applies to its drives."""
        return _drive_params(arm.joints, self.arm_gains)

    def gripper_params(self, gripper: Gripper) -> dict:
        """What one gripper's actuator controller applies to its drives."""
        return _drive_params(gripper.joints, self.gripper_gains)


def _drive_params(joints: tuple[str, ...], gains: Optional[DriveGains]) -> dict:
    return {
        "joint_names": list(joints),
        "kp": list(gains.kp) if gains else [],
        "kd": list(gains.kd) if gains else [],
        "max_efforts": list(gains.max_efforts) if gains else [],
    }


def _fail(model: str, reason: str) -> RuntimeError:
    return RuntimeError(f"models/{model}.json5: {reason}")


def _reject_unknown_keys(model: str, obj: dict, allowed: frozenset, what: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise _fail(model, f"{what} has unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _gain_values(model: str, value, length: int, what: str) -> tuple[float, ...]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or not all(_is_number(v) and v >= 0.0 for v in value)
    ):
        raise _fail(model, f"{what} must be {length} finite numbers, none negative, got {value!r}")
    return tuple(float(v) for v in value)


def _flag(model: str, raw: dict, key: str) -> bool:
    value = raw.get(key, False)
    if not isinstance(value, bool):
        raise _fail(model, f"{key} must be true or false, got {value!r}")
    return value


def _relative_prim_path(model: str, value, what: str) -> str:
    """A prim path under the robot's prim, as `a/b`."""
    if not isinstance(value, str) or not value:
        raise _fail(model, f"{what} must be a prim path under the robot's prim, got {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise _fail(model, f"{what} must be a prim path under the robot's prim, got {value!r}")
    return value


def _stage(model: str, value) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(model, f"stage must name the model's USD, got {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise _fail(model, f"stage must stay under the baked assets, got {value!r}")
    return value


def _articulation_root(model: str, value) -> str:
    if value == ROBOT_PRIM:
        return ROBOT_PRIM
    return _relative_prim_path(model, value, "articulation_root")


def _link_prims(model: str, raw) -> dict:
    if not isinstance(raw, dict) or not all(
        isinstance(prim, str) and prim and "/" not in prim for prim in raw.values()
    ):
        raise _fail(model, f"link_prims must map link names to prim names, got {raw!r}")
    return dict(raw)


def _gains(model: str, key: str, joint_counts: list[int], limbs: str, raw) -> Optional[DriveGains]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _fail(model, f"{key} must be an object, got {raw!r}")
    _reject_unknown_keys(model, raw, _GAIN_KEYS, key)
    counts = set(joint_counts)
    if len(counts) != 1:
        raise _fail(model, f"{key} serve {limbs} of one joint count, and these have {sorted(counts)}")
    (joints,) = counts
    return DriveGains(
        kp=_gain_values(model, raw.get("kp"), joints, f"{key}.kp"),
        kd=_gain_values(model, raw.get("kd"), joints, f"{key}.kd"),
        max_efforts=_gain_values(model, raw.get("max_efforts"), joints, f"{key}.max_efforts"),
    )


def parse(known: EngineModel) -> IsaacModel:
    """This engine's entry of one model, refusing anything it does not spell
    out."""
    entry, raw = known.entry, known.engine
    model = entry.model
    _reject_unknown_keys(model, raw, _ENTRY_KEYS, "the entry")
    return IsaacModel(
        entry=entry,
        stage=_stage(model, raw.get("stage")),
        articulation_root=_articulation_root(model, raw.get("articulation_root")),
        link_prims=_link_prims(model, raw.get("link_prims", {})),
        arm_gains=_gains(
            model, "arm_gains", entry.arm_joint_counts(), "arms", raw.get("arm_gains")
        ),
        gripper_gains=_gains(
            model,
            "gripper_gains",
            [len(gripper.joints) for gripper in entry.grippers],
            "grippers",
            raw.get("gripper_gains"),
        ),
        gravity_compensation=_flag(model, raw, "gravity_compensation"),
        head_camera=_flag(model, raw, "head_camera"),
    )


class IsaacModels:
    """The models this engine stands, by the id a robot attaches with."""

    def __init__(self, models: Models) -> None:
        self._models = models
        self._parsed = {name: parse(models.of(name)) for name in models.names()}

    @staticmethod
    def read(entries_dir: Path = ENTRIES_DIR) -> "IsaacModels":
        return IsaacModels(Models.read(entries_dir))

    def names(self) -> list[str]:
        return self._models.names()

    def of(self, model: str) -> IsaacModel:
        """The model a robot attaches as; an id this engine has no entry for
        is refused naming the ones it stands."""
        self._models.of(model)
        return self._parsed[model]
