"""The models this engine stands, one entry per model under models/.

What a robot of a model is made of (its limbs, joints, cameras and start
posture) comes from the description its attach goal carries, through
sim_robot_core's entry of the same model, shared with every engine, and so
does the effort ceiling of each of its PhysX drives, the effort limit the
description gives the drive's joint. What is here is what Isaac Sim alone
knows about the model: the USD stage it references, where that stage keeps
its articulation root and its links, the gains of its PhysX drives, and
whether its weight is compensated. The engine reads every entry here when it
starts, and holds one to a robot's description when the robot attaches. A
model with no entry here is refused, and none falls back to another's.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

from robot_description_py import RobotDescription
from sim_robot_core.errors import ModelEntryError
from sim_robot_core.models import Arm, Gripper, ModelEntry, Models

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
_GAIN_KEYS = frozenset({"kp", "kd"})


@dataclass(frozen=True)
class DriveGains:
    """PhysX drive gains, one per driven joint of a limb, applied to every
    limb of that kind the model has: each joint of an arm, and each finger of
    a gripper that moves of its own. The articulation view takes them per
    radian and per metre: N*m/rad and N*m*s/rad on a revolute joint, N/m and
    N*s/m on a prismatic one."""

    kp: tuple[float, ...]
    kd: tuple[float, ...]


@dataclass(frozen=True)
class IsaacModel:
    """One model this engine stands, as a robot of it stands: what the robot
    is made of, from its description, and what this engine's entry says of
    the model."""

    entry: ModelEntry
    # Joint name to the effort limit the description gives the joint, N*m or
    # N: the ceiling of the joint's drive.
    effort_limits: dict
    # Joint name to the lower and upper position limit the description gives
    # the joint, rad or m: what the stage's joint of that name holds.
    position_limits: dict
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
        return _drive_params(arm.joints, self.arm_gains, self.effort_limits)

    def gripper_params(self, gripper: Gripper) -> dict:
        """What one gripper's actuator controller applies to the drives of
        its fingers that move of their own; every other finger follows one of
        them and has no drive."""
        return _drive_params(gripper.driven_joints(), self.gripper_gains, self.effort_limits)


def _drive_params(joints: tuple[str, ...], gains: Optional[DriveGains], effort_limits: dict) -> dict:
    """What one limb's actuator controller applies to the drives of `joints`:
    `gains`, each drive capped at its joint's effort limit in
    `effort_limits`, or nothing for a limb driven by the gains its stage
    authors."""
    if gains is None:
        return {"joint_names": list(joints), "kp": [], "kd": [], "max_efforts": []}
    return {
        "joint_names": list(joints),
        "kp": list(gains.kp),
        "kd": list(gains.kd),
        "max_efforts": [effort_limits[joint] for joint in joints],
    }


def _fail(model: str, reason: str) -> ModelEntryError:
    return ModelEntryError(f"models/{model}.json5: {reason}")


def _reject_unknown_keys(model: str, obj: dict, allowed: frozenset, what: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise _fail(model, f"{what} has unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _gain_values(model: str, value, what: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value or not all(_is_number(v) and v >= 0.0 for v in value):
        raise _fail(model, f"{what} must be a list of finite numbers, none negative, got {value!r}")
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


def _gains(model: str, key: str, raw) -> Optional[DriveGains]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _fail(model, f"{key} must be an object, got {raw!r}")
    _reject_unknown_keys(model, raw, _GAIN_KEYS, key)
    gains = DriveGains(
        kp=_gain_values(model, raw.get("kp"), f"{key}.kp"),
        kd=_gain_values(model, raw.get("kd"), f"{key}.kd"),
    )
    if len(gains.kp) != len(gains.kd):
        raise _fail(model, f"{key} give {len(gains.kp)} kp and {len(gains.kd)} kd")
    return gains


def _gains_fit(
    model: str, key: str, joint_counts: list[int], limbs: str, unit: str, gains: Optional[DriveGains]
) -> None:
    """Refuses `gains` that do not serve every limb of a kind: one gain per
    driven joint, `joint_counts` of them on each limb, counted in `unit`."""
    if gains is None:
        return
    counts = sorted(set(joint_counts))
    if counts != [len(gains.kp)]:
        raise _fail(
            model,
            f"{key} serve {limbs} of {len(gains.kp)} {unit}, and the description's {limbs} have {counts}",
        )


def parse(model: str, raw: dict) -> dict:
    """This engine's entry of one model, refusing anything it does not spell
    out: every field of an IsaacModel but the robot's entry."""
    _reject_unknown_keys(model, raw, _ENTRY_KEYS, "the entry")
    return {
        "stage": _stage(model, raw.get("stage")),
        "articulation_root": _articulation_root(model, raw.get("articulation_root")),
        "link_prims": _link_prims(model, raw.get("link_prims", {})),
        "arm_gains": _gains(model, "arm_gains", raw.get("arm_gains")),
        "gripper_gains": _gains(model, "gripper_gains", raw.get("gripper_gains")),
        "gravity_compensation": _flag(model, raw, "gravity_compensation"),
        "head_camera": _flag(model, raw, "head_camera"),
    }


def _effort_limits_fit(model: str, joints: list[str], gains: Optional[DriveGains], effort_limits: dict) -> None:
    """Refuses an effort limit of `effort_limits` that caps a drive of
    `joints` at nothing: one that is not a positive finite number, on a joint
    whose drive `gains` set."""
    if gains is None:
        return
    for joint in joints:
        limit = effort_limits[joint]
        if not (math.isfinite(limit) and limit > 0.0):
            raise ModelEntryError(
                f"model {model}: the description gives joint '{joint}' the effort limit {limit}, "
                "and its drive is capped at it: give the joint's URDF <limit> a positive effort"
            )


def stood(
    model: str, entry: ModelEntry, fields: dict, effort_limits: dict, position_limits: dict
) -> IsaacModel:
    """The model `fields` of this engine's entry describe, as the robot that
    `entry` describes stands: the entry's gains serve its arms and the
    fingers of its grippers that move of their own, `effort_limits`, the
    effort limit the robot's description gives each of its joints, cap their
    drives, and `position_limits`, the position limits it gives each, are
    what the stage's joints hold."""
    _gains_fit(model, "arm_gains", entry.arm_joint_counts(), "arms", "joints", fields["arm_gains"])
    _gains_fit(
        model,
        "gripper_gains",
        [len(gripper.driven_joints()) for gripper in entry.grippers],
        "grippers",
        "fingers that move of their own",
        fields["gripper_gains"],
    )
    driven_fingers = [joint for gripper in entry.grippers for joint in gripper.driven_joints()]
    _effort_limits_fit(model, entry.arm_joints(), fields["arm_gains"], effort_limits)
    _effort_limits_fit(model, driven_fingers, fields["gripper_gains"], effort_limits)
    return IsaacModel(
        entry=entry, effort_limits=effort_limits, position_limits=position_limits, **fields
    )


class IsaacModels:
    """The models this engine stands, by the id a robot attaches with."""

    def __init__(self, models: Models) -> None:
        self._models = models
        self._parsed = {name: parse(name, models.of(name).engine) for name in models.names()}

    @staticmethod
    def read(entries_dir: Path = ENTRIES_DIR) -> "IsaacModels":
        return IsaacModels(Models.read(entries_dir))

    def names(self) -> list[str]:
        return self._models.names()

    def draws_head_camera(self) -> bool:
        """Whether a model of this engine draws the OpenArm v2 head camera."""
        return any(fields["head_camera"] for fields in self._parsed.values())

    def of(self, model: str, description: RobotDescription) -> IsaacModel:
        """The model a robot attaches as, as the robot `description`
        describes stands. An id this engine has no entry for, and a
        description the entry does not fit, are refused with ModelEntryError
        naming why."""
        entry = self._models.of(model).entry(description)
        limits = {joint: description.joint_limits(joint) for joint in entry.joints()}
        effort_limits = {joint: limit.effort for joint, limit in limits.items()}
        position_limits = {joint: (limit.lower, limit.upper) for joint, limit in limits.items()}
        return stood(model, entry, self._parsed[model], effort_limits, position_limits)
