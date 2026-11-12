"""Each robot's MuJoCo model as the release of Waldo's catalogue gives it,
which the base image bakes under the assets directory
(sim_base_images/waldo_catalogue.py): the MJCF a robot is attached from, the
body of it that carries the whole robot, the body of each URDF link the MJCF
names otherwise, whether the robot's weight is compensated, and the servo
gains of its arms and grippers. Waldo stands its robots from the same
release, so both engines stand one model of each robot.

The robots this engine stands are read strictly: each names the description
its MJCF stands for, the URDF and SRDF the release ships beside it, which are
the files of the model of the same id in robot_description; a field the
engine reads that is missing or of another type, and a description file that
differs from the model's, are refused naming the robot.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import robot_description_py
from sim_robot_core.errors import ModelEntryError

ROBOTS_FILE = "robots.json"


@dataclass(frozen=True)
class ServoGains:
    """The gains of one position servo: stiffness `kp` (N m/rad) and damping
    `kd` (N m s/rad)."""

    kp: float
    kd: float


@dataclass(frozen=True)
class CatalogueRobot:
    """One robot model of the release."""

    model: str
    # The MJCF the robot is attached from: its wrapper over its tree.
    file: Path
    # The body of the MJCF that carries the whole robot, which is what a
    # scene attaches.
    mount_body: str
    # URDF link to the MJCF body that is that link, where they differ.
    link_bodies: dict
    gravity_compensation: bool
    # Per arm group of the description: the gains of each joint, in the
    # group's order. An arm without gains runs its MJCF's own actuators.
    arm_gains: dict
    # Per arm group: the gains of the finger actuators of its gripper.
    gripper_gains: dict

    def body_of(self, link: str) -> str:
        """The MJCF body a URDF link is."""
        return self.link_bodies.get(link, link)


def _fail(model: str, reason: str) -> ModelEntryError:
    return ModelEntryError(f"{ROBOTS_FILE}: robot '{model}': {reason}")


def _object(model: str, value, what: str) -> dict:
    if not isinstance(value, dict):
        raise _fail(model, f"{what} must be an object, got {value!r}")
    return value


def _name(model: str, value, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(model, f"{what} must be a name, got {value!r}")
    return value


def _servo(model: str, kp, kd, what: str) -> ServoGains:
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (kp, kd))
    if not numbers or not (math.isfinite(kp) and kp > 0.0 and math.isfinite(kd) and kd >= 0.0):
        raise _fail(
            model, f"{what} gains need kp > 0 and kd >= 0, both finite, not kp {kp!r} and kd {kd!r}"
        )
    return ServoGains(kp=float(kp), kd=float(kd))


def _arm(model: str, raw) -> tuple[str, tuple, ServoGains | None]:
    """An arm of the `ik` block: its group's name, the gains of each of its
    joints (none for an arm that runs its file's actuators), and the gains of
    its gripper's fingers (none for a gripper that runs its file's
    actuator)."""
    if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
        raise _fail(model, f"an arm of ik names its arm group, got {raw!r}")
    name = raw["name"]
    per_joint = ()
    gains = raw.get("gains")
    if gains is not None:
        kp = gains.get("kp") if isinstance(gains, dict) else None
        kd = gains.get("kd") if isinstance(gains, dict) else None
        if not isinstance(kp, list) or not isinstance(kd, list) or len(kp) != len(kd) or not kp:
            raise _fail(model, f"arm '{name}' gains need one kd per kp, got {gains!r}")
        per_joint = tuple(
            _servo(model, p, d, f"arm '{name}' joint {i}") for i, (p, d) in enumerate(zip(kp, kd))
        )
    gripper = _object(model, raw.get("gripper", {}), f"arm '{name}' gripper")
    finger_gains = gripper.get("gains")
    finger = None
    if finger_gains is not None:
        if not isinstance(finger_gains, dict):
            raise _fail(model, f"arm '{name}' gripper gains need kp and kd, got {finger_gains!r}")
        finger = _servo(
            model, finger_gains.get("kp"), finger_gains.get("kd"), f"arm '{name}' gripper"
        )
    return name, per_joint, finger


def _scene(model: str, raw: dict) -> str:
    """The MJCF a robot is attached from: its wrapper, or, for a robot the
    release wraps in none, its source's scene."""
    wrapper = raw.get("wrapper")
    if wrapper is not None:
        return _name(model, wrapper, "wrapper")
    source = _object(model, raw.get("source"), "source")
    if not isinstance(source.get("scene"), str) or not source["scene"]:
        raise _fail(model, "names neither a wrapper nor a source scene")
    return source["scene"]


def _description(directory: Path, model: str, raw: dict) -> None:
    """Holds the description a robot's MJCF stands for, the URDF and SRDF the
    release ships beside it, to the files of the model of the same id in
    robot_description, which is what makes a URDF link's body (linkBodies)
    and an arm group's gains (ik) the description's. A robot that names no
    description, a model robot_description does not bundle, and a file that
    differs from the model's are refused."""
    description = raw.get("description")
    if not isinstance(description, dict) or not all(
        isinstance(description.get(file), str) and description[file] for file in ("urdf", "srdf")
    ):
        raise _fail(
            model,
            f"names no description, the URDF and SRDF its MJCF stands for, got {description!r}: "
            "pin a release of Waldo's catalogue whose robots name theirs",
        )
    try:
        bundled = robot_description_py.model(model)
    except ValueError as unknown:
        raise _fail(model, f"robot_description bundles no such model: {unknown}") from unknown
    differing = [
        description[file]
        for file, text in (("urdf", bundled.urdf), ("srdf", bundled.srdf))
        if (directory / model / description[file]).read_bytes() != text.encode()
    ]
    if differing:
        raise _fail(
            model,
            f"the release ships another {' and '.join(differing)} than robot_description's "
            f"{model}: pack the catalogue again with Waldo's catalogue.mjs pack at the "
            "public-peppy-libs commit sim_base_images/requirements.mujoco.txt pins, and pin "
            "that release in sim_base_images/waldo_catalogue.py",
        )


def _robot(directory: Path, model: str, raw: dict) -> CatalogueRobot:
    _description(directory, model, raw)
    mount = _object(model, raw.get("mount"), "mount")
    if not isinstance(mount.get("body"), str) or not mount["body"]:
        raise _fail(model, "names no mount body, the body a scene attaches the robot by")
    link_bodies = raw.get("linkBodies", {})
    if not isinstance(link_bodies, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and k and v for k, v in link_bodies.items()
    ):
        raise _fail(model, f"linkBodies must map link names to body names, got {link_bodies!r}")
    compensated = raw.get("gravityCompensation", False)
    if not isinstance(compensated, bool):
        raise _fail(model, f"gravityCompensation must be true or false, got {compensated!r}")
    ik = _object(model, raw.get("ik", {}), "ik")
    ik_arms = ik.get("arms", [])
    if not isinstance(ik_arms, list):
        raise _fail(model, f"ik.arms must be a list of the robot's arm groups, got {ik_arms!r}")
    arms = [_arm(model, arm) for arm in ik_arms]
    return CatalogueRobot(
        model=model,
        file=directory / model / _scene(model, raw),
        mount_body=mount["body"],
        link_bodies=dict(link_bodies),
        gravity_compensation=compensated,
        arm_gains={name: joints for name, joints, _ in arms if joints},
        gripper_gains={name: finger for name, _, finger in arms if finger is not None},
    )


def _robot_id(raw) -> str:
    model = raw.get("id") if isinstance(raw, dict) else None
    if not isinstance(model, str) or not model:
        raise ModelEntryError(f"{ROBOTS_FILE}: a robot names no id, got {raw!r}")
    return model


def read(directory: Path, models: list[str]) -> dict[str, CatalogueRobot]:
    """The robot model of the release staged at `directory` of each of
    `models`, the models this engine stands, by its model id. A model the
    release does not stand is refused, naming the ones it does."""
    path = directory / ROBOTS_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing: the base image bakes the robots' models from the release "
            "sim_base_images/waldo_catalogue.py pins"
        )
    listed = json.loads(path.read_text())
    if not isinstance(listed, list):
        raise ModelEntryError(f"{path} must list the release's robots, got {listed!r}")
    by_id = {_robot_id(raw): raw for raw in listed}
    missing = sorted(set(models) - set(by_id))
    if missing:
        raise ModelEntryError(
            f"this engine stands {missing}, which the release of the assets does not: it "
            f"stands {sorted(by_id)}"
        )
    return {model: _robot(directory, model, by_id[model]) for model in models}
