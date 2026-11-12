"""The models this engine stands, one entry per model under models/.

What a robot of a model is made of (its limbs, joints, cameras and start
posture) comes from the description its attach goal carries, through
sim_robot_core's entry of the same model, shared with every engine. Its
MuJoCo model (the MJCF it is attached from, the body that carries it, the
body of each URDF link the file names otherwise, its weight compensation and
its servo gains) is the release of Waldo's catalogue the base image bakes
(catalogue_robots), which Waldo stands its robots from too. What is here is
what this engine alone decides about a model: the light rig and the floor.
The engine reads every entry here when it starts, and holds one to a robot's
description when the robot attaches. A model with no entry here is refused,
and none falls back to another's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import catalogue_robots
from catalogue_robots import CatalogueRobot, ServoGains
from robot_description_py import RobotDescription
from sim_robot_core.errors import ModelEntryError
from sim_robot_core.models import ModelEntry, Models

ENTRIES_DIR = Path(__file__).parent / "models"

# Where the base image bakes every model's files.
ASSETS_DIR = Path(
    os.environ.get("PEPPY_ROBOT_ASSETS_DIR", str(Path(__file__).parent / "assets" / "robots"))
)

_ENTRY_KEYS = frozenset({"camera_lights", "floor"})


@dataclass(frozen=True)
class Servo:
    """The position servo a joint is driven through: its gains, and whether
    its damping is raised to critical against the joint's inertia, as it is
    on an arm joint (the real gearbox damps what the simulated joint does
    not)."""

    gains: ServoGains
    critical: bool


@dataclass(frozen=True)
class MujocoModel:
    """One model this engine stands, as a robot of it stands: what the robot
    is made of, from its description, its MuJoCo model, from the release,
    and what this engine's entry says of the model."""

    entry: ModelEntry
    robot: CatalogueRobot
    # Whether rendering adds the engine's light rig, for a scene that ships
    # no light of its own.
    camera_lights: bool
    # Whether the robot works against a floor, which the scene lays once for
    # every robot standing that asks for one.
    floor: bool

    @property
    def model(self) -> str:
        return self.entry.model

    def scene_path(self) -> Path:
        """The MJCF on disk."""
        path = self.robot.file
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: the models are baked into the container image"
            )
        return path

    def body_of(self, link: str) -> str:
        """The MJCF body a URDF link is."""
        return self.robot.body_of(link)

    def servos(self) -> dict[str, Servo]:
        """The position servo of each joint the release gives gains for: each
        arm joint, in its group's order, and each finger joint of a gripper
        that has gains."""
        servos = {}
        for arm in self.entry.arms:
            for joint, gains in zip(arm.joints, self.robot.arm_gains.get(arm.name, ())):
                servos[joint] = Servo(gains=gains, critical=True)
        # A description gives each arm its gripper, in the arms' order.
        for arm, gripper in zip(self.entry.arms, self.entry.grippers):
            finger = self.robot.gripper_gains.get(arm.name)
            if finger is not None:
                servos.update(
                    {joint: Servo(gains=finger, critical=False) for joint in gripper.joints}
                )
        return servos

    def actuator_params(self) -> dict:
        """The gains of each arm joint, which the actuator controller writes
        again once the scene is compiled, its damping raised to critical
        against the joint's inertia there."""
        arm = {joint: servo.gains for joint, servo in self.servos().items() if servo.critical}
        return {
            "joint_names": list(arm),
            "kp": [gains.kp for gains in arm.values()],
            "kd": [gains.kd for gains in arm.values()],
        }


def _fail(model: str, reason: str) -> ModelEntryError:
    return ModelEntryError(f"models/{model}.json5: {reason}")


def _reject_unknown_keys(model: str, obj: dict, allowed: frozenset, what: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise _fail(model, f"{what} has unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _flag(model: str, raw: dict, key: str) -> bool:
    value = raw.get(key, False)
    if not isinstance(value, bool):
        raise _fail(model, f"{key} must be true or false, got {value!r}")
    return value


def parse(model: str, raw: dict) -> dict:
    """This engine's entry of one model, refusing anything it does not spell
    out: every field of a MujocoModel but the robot's entry and its MuJoCo
    model."""
    _reject_unknown_keys(model, raw, _ENTRY_KEYS, "the entry")
    return {
        "camera_lights": _flag(model, raw, "camera_lights"),
        "floor": _flag(model, raw, "floor"),
    }


def robot_fail(robot: CatalogueRobot, reason: str) -> ModelEntryError:
    return ModelEntryError(f"{catalogue_robots.ROBOTS_FILE}: robot '{robot.model}': {reason}")


def stood(entry: ModelEntry, robot: CatalogueRobot, fields: dict) -> MujocoModel:
    """The model `robot` and `fields` of this engine's entry describe, as the
    robot that `entry` describes stands: the release's gains name the arms of
    the description, with one gain per joint."""
    arms = {arm.name: arm for arm in entry.arms}
    for name in sorted(set(robot.arm_gains) | set(robot.gripper_gains)):
        if name not in arms:
            raise robot_fail(
                robot, f"gains name the arm '{name}', and the description's arms are {sorted(arms)}"
            )
    for name, gains in robot.arm_gains.items():
        if len(gains) != len(arms[name].joints):
            raise robot_fail(
                robot,
                f"arm '{name}' has gains for {len(gains)} joints, and its group moves "
                f"{len(arms[name].joints)}",
            )
    return MujocoModel(entry=entry, robot=robot, **fields)


class MujocoModels:
    """The models this engine stands, by the id a robot attaches with."""

    def __init__(self, models: Models, robots: dict[str, CatalogueRobot]) -> None:
        self._models = models
        self._robots = robots
        self._parsed = {name: parse(name, models.of(name).engine) for name in models.names()}

    @staticmethod
    def read(entries_dir: Path = ENTRIES_DIR, assets_dir: Path | None = None) -> MujocoModels:
        models = Models.read(entries_dir)
        return MujocoModels(models, catalogue_robots.read(assets_dir or ASSETS_DIR, models.names()))

    def names(self) -> list[str]:
        return self._models.names()

    def of(self, model: str, description: RobotDescription) -> MujocoModel:
        """The model a robot attaches as, as the robot `description`
        describes stands. An id this engine has no entry for, and a
        description the entry does not fit, are refused with ModelEntryError
        naming why."""
        known = self._models.of(model)
        return stood(known.entry(description), self._robots[model], self._parsed[model])
