#!/usr/bin/env python3
"""The stage the engine simulates: the robot it opens on, and the robots
that take a seat in it.

The robot the engine stands is the stage, which leaves its prims exactly
where its USD puts them. A seat's robot is referenced under /World as a
prim of its own name, so two robots of the same model keep separate
articulations, and taking one out is removing that prim.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Stages of the models this engine ships, baked into the base image under
# PEPPY_ROBOT_ASSETS_DIR.
BAKED_STAGES = {
    "openarm_v1": "openarm_bimanual.usd",
    "openarm_v2": "openarm_bimanual_v2.usd",
}

ASSETS_DIR = Path(
    os.environ.get("PEPPY_ROBOT_ASSETS_DIR", str(Path(__file__).parent / "openarm" / "assets"))
)

# Where a seat's robot is referenced, and the prim the engine's own robot
# stands at, which is the default prim of every model's stage.
WORLD_PRIM = "/World"
STANDING_PRIM = "/openarm"
# The name a scene command addresses the robot the engine stands by. A seat's
# robot answers to the name of the attachment that took it; this one has none
# of its own.
STANDING_NAME = "openarm"

# Side of the square the engine parks robots on when a seat asks for no
# placement of its own: far enough apart that two OpenArms cannot touch.
SPOT_PITCH_M = 1.5

# How brightly an empty stage is lit, in the units UsdLux reads: enough to
# read a robot's shape without washing out its materials.
DOME_LIGHT_INTENSITY = 1000.0


class Catalogue:
    """The models this engine can stand, by the id a seat attaches with."""

    def __init__(self, stages: dict[str, Path]) -> None:
        self._stages = dict(stages)

    @staticmethod
    def baked() -> "Catalogue":
        """The stages of the container image, under PEPPY_ROBOT_ASSETS_DIR."""
        return Catalogue({model: ASSETS_DIR / file for model, file in BAKED_STAGES.items()})

    def models(self) -> list[str]:
        return sorted(self._stages)

    def scene(self, model: str) -> Path:
        """The USD of a model. An id this engine does not carry names what it
        does, so a launcher's typo says what to write instead."""
        if model not in self._stages:
            raise ValueError(
                f"unknown model {model!r}: this engine stands {', '.join(self.models())}"
            )
        path = self._stages[model]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: the stages are baked into the container image"
            )
        return path


@dataclass(frozen=True)
class Placement:
    """Where a robot's base stands: a world-frame position in metres and a
    rotation about +z in radians."""

    position: tuple[float, float, float]
    yaw: float

    @staticmethod
    def of(position, yaw: float) -> "Placement":
        """The placement a caller asked for, refusing anything the stage
        cannot take: a prim placed at NaN reports no error and simulates
        nothing."""
        values = tuple(float(value) for value in position)
        if len(values) != 3:
            raise ValueError(f"a placement is 3 coordinates, got {len(values)}")
        for value in (*values, float(yaw)):
            if not math.isfinite(value):
                raise ValueError("a placement is finite in every coordinate")
        return Placement(position=values, yaw=float(yaw))

    def degrees(self) -> float:
        """The rotation as USD writes it, in degrees about +z."""
        return math.degrees(self.yaw)


@dataclass(frozen=True)
class Robot:
    """One robot in the stage: the name it stands under, the catalogue model
    it is, and where. The robot the engine stands has an empty instance and
    keeps the prims of its own stage."""

    instance: str
    model: str
    placement: Placement

    def prim(self) -> str:
        """The prim this robot's articulation lives under."""
        return f"{WORLD_PRIM}/{self.instance}" if self.instance else STANDING_PRIM


class World:
    """The stage as opened, and the robots it holds. Standing a robot adds a
    prim referencing its model; taking one out removes that prim. Both leave
    the views on the articulations invalid, so the engine resolves the stage
    again around them."""

    def __init__(self, catalogue: Catalogue, standing: Robot | None) -> None:
        self._catalogue = catalogue
        self._standing = standing
        self._seated: dict[str, Robot] = {}

    def catalogue(self) -> Catalogue:
        return self._catalogue

    def robots(self) -> list[Robot]:
        """Every robot in the stage, the one it stands first."""
        standing = [self._standing] if self._standing else []
        return standing + list(self._seated.values())

    def seated(self) -> list[Robot]:
        return list(self._seated.values())

    def standing(self) -> Robot | None:
        return self._standing

    def holds(self, instance: str) -> bool:
        return instance in self._seated

    def free_spot(self) -> Placement:
        """A spot no robot stands on, walked out from the origin on a square
        lattice so a fleet fills the floor around the stage's own robot."""
        taken = {self._rounded(robot.placement.position) for robot in self.robots()}
        for ring in range(0, 64):
            square = [
                (row, column)
                for row in range(-ring, ring + 1)
                for column in range(-ring, ring + 1)
                if max(abs(row), abs(column)) == ring
            ]
            # Along the axes before the diagonals, so a small fleet stands in
            # a cross around the stage's own robot.
            for row, column in sorted(square, key=lambda spot: (abs(spot[0]) + abs(spot[1]), spot)):
                spot = (row * SPOT_PITCH_M, column * SPOT_PITCH_M, 0.0)
                if self._rounded(spot) not in taken:
                    return Placement.of(spot, 0.0)
        raise RuntimeError("the stage has no free spot left")

    def occupied(self, placement: Placement) -> bool:
        """Whether a robot already stands within a spot of this placement."""
        wanted = self._rounded(placement.position)
        return any(self._rounded(robot.placement.position) == wanted for robot in self.robots())

    def open(self) -> None:
        """Opens the stage: the robot the engine stands, or an empty world
        holding whichever robots take a seat. A stage of its own carries the
        physics the robots are simulated by, and the light they are seen by."""
        import omni.usd  # pylint: disable=C0415
        from pxr import UsdGeom, UsdLux, UsdPhysics  # pylint: disable=C0415

        context = omni.usd.get_context()
        if self._standing:
            stage_path = self._catalogue.scene(self._standing.model)
            logger.info("Loading stage: %s", stage_path)
            context.open_stage(str(stage_path))
        else:
            logger.info("Opening an empty stage; robots arrive by taking a seat")
            context.new_stage()
            # The models are authored z-up in metres. A stage that says
            # otherwise stands every robot on its side.
            stage = context.get_stage()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        stage = context.get_stage()
        UsdGeom.Xform.Define(stage, WORLD_PRIM)
        if not any(prim.GetTypeName() == "PhysicsScene" for prim in stage.Traverse()):
            UsdPhysics.Scene.Define(stage, f"{WORLD_PRIM}/physicsScene")
        # A robot's own stage carries the light it was authored under; an empty
        # one carries none, and renders black however many robots take a seat.
        if not any(prim.IsA(UsdLux.DomeLight) for prim in stage.Traverse()):
            dome = UsdLux.DomeLight.Define(stage, f"{WORLD_PRIM}/dome_light")
            dome.CreateIntensityAttr(DOME_LIGHT_INTENSITY)

    def add(self, instance: str, model: str, placement: Placement) -> Robot:
        """Stands a robot in the stage. Admission has already found the caller
        a name and a spot, so this raises on either."""
        if instance in self._seated:
            raise ValueError(f"{instance} already stands in the stage")
        if not instance:
            raise ValueError("a seat's robot stands under the name of its attachment")
        robot = Robot(instance=instance, model=model, placement=placement)
        stage_path = self._catalogue.scene(model)
        self._reference(robot, stage_path)
        self._seated[instance] = robot
        logger.info(
            "robot '%s' (%s) joins at %s",
            instance,
            model,
            list(placement.position),
        )
        return robot

    def remove(self, instance: str) -> None:
        """Takes a seat's robot out of the stage."""
        robot = self._seated.pop(instance, None)
        if robot is None:
            return
        import omni.usd  # pylint: disable=C0415

        omni.usd.get_context().get_stage().RemovePrim(robot.prim())
        logger.info("robot '%s' leaves the stage", instance)

    @staticmethod
    def _reference(robot: Robot, stage_path: Path) -> None:
        """Puts a model's stage under this robot's prim, at its placement.

        The reference brings the model's own transform ops along, and these
        models orient by quaternion, which XformCommonAPI does not carry: it
        reports itself unusable on such a prim and writes nothing, leaving
        every robot stacked on the origin. The placement is written into the
        ops the prim actually has."""
        import omni.usd  # pylint: disable=C0415
        from pxr import Gf, UsdGeom  # pylint: disable=C0415

        stage = omni.usd.get_context().get_stage()
        prim = stage.DefinePrim(robot.prim(), "Xform")
        prim.GetReferences().AddReference(str(stage_path))

        xform = UsdGeom.Xformable(prim)
        ops = {op.GetOpType(): op for op in xform.GetOrderedXformOps()}

        translate = ops.get(UsdGeom.XformOp.TypeTranslate) or xform.AddTranslateOp()
        translate.Set(Gf.Vec3d(*robot.placement.position))

        half = robot.placement.yaw / 2.0
        axis = (0.0, 0.0, math.sin(half))
        orient = ops.get(UsdGeom.XformOp.TypeOrient)
        if orient is None:
            xform.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(*axis)))
        elif orient.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            orient.Set(Gf.Quatd(math.cos(half), Gf.Vec3d(*axis)))
        else:
            orient.Set(Gf.Quatf(math.cos(half), Gf.Vec3f(*axis)))

    @staticmethod
    def _rounded(position) -> tuple[int, int, int]:
        """A placement as a spot on the lattice, so a robot that asked for a
        spot by hand still counts as standing on it."""
        return tuple(int(round(value / SPOT_PITCH_M)) for value in position)
