#!/usr/bin/env python3
"""The scene the engine steps, and the robots standing in it.

The scene opens empty and every robot is attached into it under a prefix of
its own name, so two robots of the same model keep their own joints,
actuators and cameras, and taking one out is composing the scene again
without it. What is attached is its model's own MJCF, corrected and fitted
out as its entry asks; where it stands is its placement.

MuJoCo compiles a scene whole, so a robot joining or leaving makes a new
model out of the robots standing then. The thread that steps the scene
carries the survivors' state onto it, which is why every name here is the
one the compiled model carries: `alpha/left_joint1` is alpha's, whatever
else stands.

A model's `<option>` and `<statistic>` stay behind when it is attached, and
MuJoCo keeps the scene's own. The scene therefore takes the settings its
models share, and a model that asks for others is refused.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Optional

import head_camera
import numpy as np
from exts.camera_sensor import add_cameras, add_light_rig, widen_offscreen
from mujoco_models import MujocoModel

logger = logging.getLogger(__name__)

# The empty scene robots join. A model that works against a floor asks for
# one in its entry, and the scene lays it once for every robot that does.
_STAGE_XML = '<mujoco model="stage"><compiler angle="radian"/><worldbody/></mujoco>'

# Side of the square the engine parks robots on when one asks for no
# placement of its own, in metres: far enough apart that no two robots of
# the models this engine stands can touch. Isaac Sim parks them on the same
# lattice, so a fleet stands alike on both.
SPOT_PITCH_M = 1.5

# What stands between a robot's name and the name of anything it carries,
# in the composed scene: `alpha/left_joint1` is alpha's.
PREFIX_SEPARATOR = "/"

# How far the lattice is walked before a fleet is told the scene is full.
_MAX_RINGS = 64

# The floor a scene lays once for the robots that work against one, and the light
# that shows it: the plane through the origin, with a grid every half metre,
# lit from above as upstream's own scenes light theirs.
_FLOOR_GEOM = "floor"
_FLOOR_LIGHT = "floor_light"
_FLOOR_GRID_M = 0.5
_FLOOR_LIGHT_POS = (0.0, 0.0, 3.5)


@dataclass(frozen=True)
class Placement:
    """Where a robot's base stands: a world-frame position in metres and a
    rotation about +z in radians."""

    position: tuple[float, float, float]
    yaw: float

    @staticmethod
    def of(position, yaw: float) -> "Placement":
        """The placement a caller asked for, refusing anything the scene
        cannot take: a robot attached at NaN compiles and simulates
        nothing."""
        values = tuple(float(value) for value in position)
        if len(values) != 3:
            raise ValueError(f"a placement is 3 coordinates, got {len(values)}")
        for value in (*values, float(yaw)):
            if not math.isfinite(value):
                raise ValueError("a placement is finite in every coordinate")
        return Placement(position=values, yaw=float(yaw))

    def quat_wxyz(self) -> tuple[float, float, float, float]:
        """The rotation as MuJoCo writes it: a quaternion about +z."""
        half = self.yaw / 2.0
        return (math.cos(half), 0.0, 0.0, math.sin(half))


@dataclass(frozen=True)
class Robot:
    """One robot in the scene: the name it stands under, the model it is,
    and where."""

    instance: str
    known: MujocoModel
    placement: Placement

    @property
    def model(self) -> str:
        return self.known.model

    @property
    def prefix(self) -> str:
        """What every name of this robot carries in the composed scene."""
        return f"{self.instance}{PREFIX_SEPARATOR}"

    def scene_name(self, name: str) -> str:
        """The name a joint, actuator or camera of this robot has in the
        composed scene."""
        return f"{self.prefix}{name}"


def name_in_the_scene(instance: str) -> str:
    """The name a robot stands under. Every name it answers to in the scene
    is this name and the separator, so a name carrying the separator names
    two robots at once."""
    if not instance:
        raise ValueError("a robot stands under the name of the copy it runs as")
    if PREFIX_SEPARATOR in instance:
        plain = instance.replace(PREFIX_SEPARATOR, "_")
        raise ValueError(
            f"a robot stands under a name carrying no {PREFIX_SEPARATOR!r}, and "
            f"'{instance}' carries one: join under a name without it, "
            f"`peppy stack join LAUNCHER -i {plain}`"
        )
    return instance


class World:
    """The robots standing in the scene, and the scene they compose.

    Standing a robot records it and composes the scene again; taking one out
    composes it without that robot. Both leave the compiled model behind, so
    the engine builds its views on the new one."""

    def __init__(self, head_camera_pack: Optional[head_camera.Pack], renders: bool) -> None:
        # The pack a model that draws the head camera seats on its pedestal,
        # or None when no model of this engine draws it.
        self._head_camera_pack = head_camera_pack
        # Whether the robots' cameras are rendered: the rig is its model's.
        self._renders = renders
        self._robots: dict[str, Robot] = {}
        # Robots are stood and taken out on the thread that steps the scene,
        # and listed on the one serving the contracts, so neither reads this
        # while the other changes it. Reentrant, because the lookups below
        # are written in terms of each other.
        self._lock = threading.RLock()

    @property
    def renders(self) -> bool:
        """Whether the scene renders the cameras of the robots standing in
        it, which is what the engine was launched to do or not."""
        return self._renders

    def robots(self) -> list[Robot]:
        """Every robot in the scene, in the order they joined."""
        with self._lock:
            return list(self._robots.values())

    def free_spot(
        self, promised: "tuple[Placement, ...]" = (), leaving: Optional[str] = None
    ) -> Placement:
        """A spot no robot stands on and none has been promised, walked out
        from the origin on a square lattice, so a fleet fills the floor. A
        robot admitted a moment ago has a spot and does not stand on it yet,
        so its placement is passed in here."""
        for ring in range(0, _MAX_RINGS):
            square = [
                (row, column)
                for row in range(-ring, ring + 1)
                for column in range(-ring, ring + 1)
                if max(abs(row), abs(column)) == ring
            ]
            # Along the axes before the diagonals, so a small fleet stands in
            # a cross around the origin.
            for row, column in sorted(square, key=lambda spot: (abs(spot[0]) + abs(spot[1]), spot)):
                spot = Placement.of((row * SPOT_PITCH_M, column * SPOT_PITCH_M, 0.0), 0.0)
                if not self.occupied(spot, promised, leaving):
                    return spot
        raise ValueError(
            "every spot this scene lays out is taken: take a robot out with "
            "`peppy stack remove NAME` before standing another"
        )

    def standing_within(
        self, placement: Placement, leaving: Optional[str] = None
    ) -> Optional[Robot]:
        """The robot standing within a spot of this placement, if one does.
        The robot named by `leaving` is on its way out, so the spot it stands
        on is free for whoever is asking."""
        return next(
            (
                robot
                for robot in self.robots()
                if robot.instance != leaving and within_one_spot(placement, robot.placement)
            ),
            None,
        )

    def occupied(
        self,
        placement: Placement,
        promised: "tuple[Placement, ...]" = (),
        leaving: Optional[str] = None,
    ) -> bool:
        """Whether a robot stands within a spot of this placement, or has
        been promised one there. The robot named by `leaving` is on its way
        out, so the spot it stands on is free."""
        standing = (
            robot.placement for robot in self.robots() if robot.instance != leaving
        )
        return any(within_one_spot(placement, other) for other in (*standing, *promised))

    def add(self, instance: str, known: MujocoModel, placement: Placement) -> Robot:
        """Records a robot of the model `known` as standing. Admission has
        already found the caller a name and a spot, so this raises on
        either. The scene is composed again by whoever steps it."""
        instance = name_in_the_scene(instance)
        with self._lock:
            if instance in self._robots:
                raise ValueError(
                    f"'{instance}' is still in the scene, which a robot that could not be "
                    "taken out is until this simulation restarts: `peppy stack list` says "
                    "whether its copy still runs"
                )
            robot = Robot(instance=instance, known=known, placement=placement)
            self._robots[instance] = robot
        logger.info(
            "robot '%s' (%s) joins at %s", instance, known.model, list(placement.position)
        )
        return robot

    def remove(self, instance: str) -> None:
        """Takes a robot out of the scene."""
        with self._lock:
            robot = self._robots.pop(instance, None)
        if robot is None:
            return
        logger.info("robot '%s' leaves the scene", instance)

    def compose(self):
        """The spec of the scene as it stands: the settings its models
        share, the light rig and offscreen buffer its cameras need, and
        every robot attached under its own prefix at its own placement."""
        import mujoco  # pylint: disable=C0415

        robots = self.robots()
        stage = mujoco.MjSpec.from_string(_STAGE_XML)
        children = [(robot, self._model_spec(robot)) for robot in robots]
        settings = _shared_settings(children)
        if settings is not None:
            stage.option, stage.stat = settings
        if any(robot.known.floor for robot in robots):
            _lay_floor(stage)
        if self._renders:
            self._fit_out_rendering(stage, robots)
        for robot, child in children:
            frame = stage.worldbody.add_frame()
            frame.pos = list(robot.placement.position)
            frame.quat = list(robot.placement.quat_wxyz())
            stage.attach(child, prefix=robot.prefix, frame=frame)
        return stage

    def _fit_out_rendering(self, stage, robots: list[Robot]) -> None:
        """What rendering needs of the scene as a whole: an
        offscreen buffer wide enough for every camera standing, and the
        light rig, once, for a scene holding a model that asks for one."""
        widen_offscreen(stage, [camera for robot in robots for camera in robot.known.entry.cameras])
        if any(robot.known.camera_lights for robot in robots):
            add_light_rig(stage)

    def _model_spec(self, robot: Robot):
        """One robot's model as it is attached: its MJCF corrected to the
        robot's description, weight compensated where its entry asks, with
        the head camera it draws and the cameras of its own rig when the
        engine renders."""
        known = robot.known
        spec = compile_spec(known)
        if known.head_camera:
            if self._head_camera_pack is None:
                raise RuntimeError(f"{known.model} draws the head camera, and no pack was staged")
            head_camera.attach(spec, self._head_camera_pack)
            logger.info(
                "robot '%s' draws its head camera from %s",
                robot.instance,
                self._head_camera_pack.directory,
            )
        if self._renders and known.entry.cameras:
            add_cameras(spec, known, known.scene_path())
        return spec


def within_one_spot(one: Placement, other: Placement) -> bool:
    """Whether two placements are too close for a robot to stand on both. It
    is the distance between them that counts: two robots closer together than
    the lattice leaves them resolve their overlap by throwing each other,
    wherever the lattice's own lines happen to fall."""
    return math.dist(one.position, other.position) < SPOT_PITCH_M


def _lay_floor(stage) -> None:
    """The floor the robots that ask for one work against, laid once for the
    scene: a model brings its robot, and the ground every robot stands on is
    the scene's. It is where the models put theirs, the plane through the
    origin, and it is lit so a viewer sees it."""
    import mujoco  # pylint: disable=C0415

    floor = stage.worldbody.add_geom()
    floor.name = _FLOOR_GEOM
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, _FLOOR_GRID_M]
    light = stage.worldbody.add_light()
    light.name = _FLOOR_LIGHT
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.pos = list(_FLOOR_LIGHT_POS)
    light.dir = [0.0, 0.0, -1.0]


def _shared_settings(children: "list[tuple[Robot, object]]"):
    """The `<option>` and `<statistic>` every standing model asks for, or
    None while nothing stands. MuJoCo drops a model's settings when it is
    attached and steps it under the scene's, so two models that ask for
    different ones are refused here: what a scene runs is what its models
    asked for."""
    if not children:
        return None
    first_robot, first_spec = children[0]
    for robot, spec in children[1:]:
        differences = _settings_differences(first_spec, spec)
        if differences:
            raise RuntimeError(
                f"the models '{first_robot.model}' and '{robot.model}' ask for different "
                f"simulation settings ({', '.join(differences)}), and a scene steps one set "
                f"of them: give both models the same settings, or stand the {robot.model} "
                "in a simulation of its own"
            )
    return first_spec.option, first_spec.stat


def _settings_differences(one, other) -> list[str]:
    """Which `<option>` and `<statistic>` fields two specs disagree on."""
    return [
        f"{group}.{field}"
        for group in ("option", "stat")
        for field in _settings_fields(getattr(one, group))
        if not _same(
            getattr(getattr(one, group), field), getattr(getattr(other, group), field)
        )
    ]


def _same(one, other) -> bool:
    """Whether two settings values are the same one. A statistic a model
    leaves out reads as NaN, which is MuJoCo's way of saying it works the
    value out at compile: two models that both leave it out ask for the same
    thing, and NaN is equal to nothing, itself included."""
    first, second = np.asarray(one), np.asarray(other)
    if first.dtype.kind == "f" and second.dtype.kind == "f":
        return np.array_equal(first, second, equal_nan=True)
    return np.array_equal(first, second)


def _settings_fields(settings) -> list[str]:
    """Every value of one settings group, whatever MuJoCo's build carries."""
    return [
        name
        for name in dir(settings)
        if not name.startswith("_") and not callable(getattr(settings, name))
    ]


def compile_spec(known: MujocoModel):
    """The spec of a model's MJCF, with the joint ranges, site poses and
    solver settings its entry corrects to the scene the robot stands in, and
    the robot's weight compensated where its entry asks for it. A correction
    naming a joint or a site the file lacks is refused, so an upstream rename
    is caught at the first stand."""
    import mujoco  # pylint: disable=C0415

    spec = mujoco.MjSpec.from_file(str(known.scene_path()))
    if known.solver is not None:
        spec.option = known.solver
    for name, (lower, upper) in known.joint_ranges.items():
        joint = spec.joint(name)
        if joint is None:
            raise RuntimeError(f"{known.model}: joint_ranges names '{name}', not in {known.scene}")
        joint.range = [lower, upper]
        for actuator in spec.actuators:
            if actuator.target == name and _ctrl_limited(actuator):
                actuator.ctrlrange = [lower, upper]
    for name, pose in known.site_poses.items():
        site = spec.site(name)
        if site is None:
            raise RuntimeError(f"{known.model}: site_poses names '{name}', not in {known.scene}")
        site.pos = list(pose.pos)
        site.quat = list(pose.quat_wxyz)
    if known.gravity_compensation:
        _compensate_gravity(spec, known)
    return spec


def _ctrl_limited(actuator) -> bool:
    """Whether an actuator's control is clamped to its ctrlrange: its file
    says so, or leaves it to the compiler, which clamps the ones that carry a
    range."""
    import mujoco  # pylint: disable=C0415

    if actuator.ctrllimited == mujoco.mjtLimited.mjLIMITED_AUTO:
        return actuator.ctrlrange[0] < actuator.ctrlrange[1]
    return actuator.ctrllimited == mujoco.mjtLimited.mjLIMITED_TRUE


def _compensate_gravity(spec, known: MujocoModel) -> None:
    """Mirror the real driver's in-process gravity feedforward: MuJoCo's body
    gravcomp applies an exact counter-gravity force per body, every step,
    inside the engine. Set on every body the robot's joints move (the real
    arm and gripper drivers both feedforward): the bodies that carry one of
    the model's joints, and everything below them. It goes on the spec
    because MuJoCo compensates only a model compiled with a compensated body,
    so a gravcomp written to the compiled model is never applied. Coriolis is
    intentionally not compensated, negligible at teleop speeds."""
    import mujoco  # pylint: disable=C0415

    compensated = set()
    for name in known.entry.joints():
        joint = spec.joint(name)
        if joint is None:
            raise RuntimeError(
                f"{known.model}: gravity compensation names joint '{name}', not in {known.scene}"
            )
        moved = joint.parent
        for body in (moved, *moved.find_all(mujoco.mjtObj.mjOBJ_BODY)):
            body.gravcomp = 1.0
            compensated.add(body.name)
    logger.info(f"gravity compensation enabled on {len(compensated)} robot bodies")
