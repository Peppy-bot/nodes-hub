#!/usr/bin/env python3
"""The stage the engine simulates, and the robots standing in it.

The stage opens empty and each robot is referenced under /World as a prim of
its own name, so two robots of the same model keep separate articulations,
and taking one out is removing that prim.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import head_camera

logger = logging.getLogger(__name__)

# The head camera assembly seats on the v2 pedestal, so a robot standing as
# that model draws it and a v1 has none (see head_camera.py).
HEAD_CAMERA_MODELS = frozenset({"openarm_v2"})

# Stages of the models this engine ships, baked into the base image under
# PEPPY_ROBOT_ASSETS_DIR.
BAKED_STAGES = {
    "openarm_v1": "openarm_bimanual.usd",
    "openarm_v2": "openarm_bimanual_v2.usd",
}

ASSETS_DIR = Path(
    os.environ.get("PEPPY_ROBOT_ASSETS_DIR", str(Path(__file__).parent / "openarm" / "assets"))
)

# Where each robot of the scene is referenced, under the name it stands as.
WORLD_PRIM = "/World"

# Side of the square the engine parks robots on when one asks for no
# placement of its own: far enough apart that two OpenArms cannot touch.
SPOT_PITCH_M = 1.5

# How brightly an empty stage is lit, in the units UsdLux reads: enough to
# read a robot's shape without washing out its materials.
DOME_LIGHT_INTENSITY = 1000.0

# The physics pipeline the stage simulates on, authored on its physics scene
# as Isaac Sim's own assets author it: CPU dynamics with the MBP broadphase,
# the pipeline Isaac Sim pairs with a CPU device. That is the device the
# engine's articulation views read every robot's state from and write its
# targets to on every step.
PHYSX_SCENE_API = "PhysxSceneAPI"
GPU_DYNAMICS = "physxScene:enableGPUDynamics"
BROADPHASE = "physxScene:broadphaseType"
CPU_BROADPHASE = "MBP"


class Catalogue:
    """The models this engine can stand, by the id a robot attaches with,
    and the head camera pack the models carrying one draw."""

    def __init__(self, stages: dict[str, Path], *, head_camera_pack: head_camera.Pack) -> None:
        self._stages = dict(stages)
        self._head_camera_pack = head_camera_pack

    @staticmethod
    def baked(*, head_camera_pack: head_camera.Pack) -> "Catalogue":
        """The stages of the container image, under PEPPY_ROBOT_ASSETS_DIR."""
        return Catalogue(
            {model: ASSETS_DIR / file for model, file in BAKED_STAGES.items()},
            head_camera_pack=head_camera_pack,
        )

    def models(self) -> list[str]:
        return sorted(self._stages)

    def head_camera_pack(self, model: str) -> head_camera.Pack | None:
        """The head camera pack a robot standing as `model` draws, or None
        for a model whose robot has no head camera."""
        self._known(model)
        return self._head_camera_pack if model in HEAD_CAMERA_MODELS else None

    def scene(self, model: str) -> Path:
        """The USD of a model."""
        path = self._known(model)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: the stages are baked into the container image"
            )
        return path

    def _known(self, model: str) -> Path:
        """The stage file of a model this engine carries. An id it does not
        carry names what it does, so a launcher's typo says what to write
        instead."""
        if model not in self._stages:
            raise ValueError(
                f"unknown model {model!r}: this engine stands {', '.join(self.models())}"
            )
        return self._stages[model]


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
    it is, and where."""

    instance: str
    model: str
    placement: Placement

    def prim(self) -> str:
        """The prim this robot's articulation lives under."""
        return f"{WORLD_PRIM}/{self.instance}"


class World:
    """The stage as opened, and the robots it holds. Standing a robot adds a
    prim referencing its model; taking one out removes that prim. Both leave
    the views on the articulations invalid, so the engine resolves the stage
    again around them."""

    def __init__(self, catalogue: Catalogue) -> None:
        self._catalogue = catalogue
        self._robots: dict[str, Robot] = {}
        # Robots are stood and taken out on the thread that steps the scene,
        # and listed on the one serving the contracts, so the two of them do
        # not read this while the other is changing it. Reentrant, because
        # the lookups below are written in terms of each other.
        self._lock = threading.RLock()

    def catalogue(self) -> Catalogue:
        return self._catalogue

    def robots(self) -> list[Robot]:
        """Every robot in the stage, in the order they joined."""
        with self._lock:
            return list(self._robots.values())

    def free_spot(self, promised: "tuple[Placement, ...]" = ()) -> Placement:
        """A spot no robot stands on and none has been promised, walked out
        from the origin on a square lattice, so a fleet fills the floor. A
        robot admitted a moment ago has a spot and does not stand on it yet,
        so its placement is passed in here."""
        taken = {self._rounded(robot.placement.position) for robot in self.robots()}
        taken |= {self._rounded(placement.position) for placement in promised}
        for ring in range(0, 64):
            square = [
                (row, column)
                for row in range(-ring, ring + 1)
                for column in range(-ring, ring + 1)
                if max(abs(row), abs(column)) == ring
            ]
            # Along the axes before the diagonals, so a small fleet stands in
            # a cross around the origin.
            for row, column in sorted(square, key=lambda spot: (abs(spot[0]) + abs(spot[1]), spot)):
                spot = (row * SPOT_PITCH_M, column * SPOT_PITCH_M, 0.0)
                if self._rounded(spot) not in taken:
                    return Placement.of(spot, 0.0)
        raise RuntimeError("the stage has no free spot left")

    def occupied(
        self, placement: Placement, promised: "tuple[Placement, ...]" = ()
    ) -> bool:
        """Whether a robot stands within a spot of this placement, or has
        been promised one there."""
        wanted = self._rounded(placement.position)
        standing = (robot.placement for robot in self.robots())
        return any(
            self._rounded(other.position) == wanted
            for other in (*standing, *promised)
        )

    def open(self) -> None:
        """Opens the empty stage the robots join, which carries the physics
        they are simulated by and the light they are seen by."""
        import omni.usd  # pylint: disable=C0415
        from pxr import UsdGeom, UsdLux  # pylint: disable=C0415

        context = omni.usd.get_context()
        logger.info("Opening an empty stage; robots arrive by attaching")
        context.new_stage()
        # The models are authored z-up in metres. A stage that says otherwise
        # stands every robot on its side.
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, WORLD_PRIM)
        World._physics_scene(stage)
        # An empty stage carries no light of its own, and renders black
        # however many robots join it.
        if not any(prim.IsA(UsdLux.DomeLight) for prim in stage.Traverse()):
            dome = UsdLux.DomeLight.Define(stage, f"{WORLD_PRIM}/dome_light")
            dome.CreateIntensityAttr(DOME_LIGHT_INTENSITY)

    @staticmethod
    def _physics_scene(stage):
        """The stage's physics scene, defined under /World when the stage has
        none, simulating on the CPU pipeline."""
        from pxr import Sdf, UsdPhysics  # pylint: disable=C0415

        scene = next((prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)), None)
        if scene is None:
            scene = UsdPhysics.Scene.Define(stage, f"{WORLD_PRIM}/physicsScene").GetPrim()
        scene.AddAppliedSchema(PHYSX_SCENE_API)
        scene.CreateAttribute(GPU_DYNAMICS, Sdf.ValueTypeNames.Bool, False).Set(False)
        scene.CreateAttribute(
            BROADPHASE, Sdf.ValueTypeNames.Token, False, Sdf.VariabilityUniform
        ).Set(CPU_BROADPHASE)
        return scene

    def add(self, instance: str, model: str, placement: Placement) -> Robot:
        """Stands a robot in the stage. Admission has already found the caller
        a name and a spot, so this raises on either."""
        with self._lock:
            if instance in self._robots:
                raise ValueError(f"{instance} already stands in the stage")
        if not instance:
            raise ValueError("a robot stands under the name of the copy it runs as")
        robot = Robot(instance=instance, model=model, placement=placement)
        stage_path = self._catalogue.scene(model)
        try:
            self._reference(robot, stage_path, self._catalogue.head_camera_pack(model))
        except Exception:
            # What the reference authored before it failed comes off the
            # stage, so a robot that cannot join leaves the stage as it was.
            World._unreference(robot)
            raise
        with self._lock:
            self._robots[instance] = robot
        logger.info(
            "robot '%s' (%s) joins at %s",
            instance,
            model,
            list(placement.position),
        )
        return robot

    def move(self, name: str, position, yaw: float) -> Robot:
        """Puts a robot somewhere else in the stage, turned `yaw` radians
        about +z. The record moves with the prim, so what the scene reports
        and where the next robot is given room both follow it."""
        robot = next((one for one in self.robots() if one.instance == name), None)
        if robot is None:
            raise KeyError(f"no robot stands as {name!r} in this stage")
        moved = Robot(
            instance=robot.instance,
            model=robot.model,
            placement=Placement.of(position, yaw),
        )

        import omni.usd  # pylint: disable=C0415

        prim = omni.usd.get_context().get_stage().GetPrimAtPath(moved.prim())
        if not prim.IsValid():
            raise RuntimeError(
                f"the prim {moved.prim()} of robot {name!r} does not exist"
            )
        World.place(prim, moved.placement.position, moved.placement.yaw)

        with self._lock:
            self._robots[moved.instance] = moved
        logger.info(
            "robot '%s' moves to %s at a yaw of %s rad",
            name,
            list(moved.placement.position),
            moved.placement.yaw,
        )
        return moved

    def remove(self, instance: str) -> None:
        """Takes a robot out of the stage."""
        with self._lock:
            robot = self._robots.pop(instance, None)
        if robot is None:
            return
        World._unreference(robot)
        logger.info("robot '%s' leaves the stage", instance)

    @staticmethod
    def _unreference(robot: Robot) -> None:
        """Removes a robot's prim, and everything under it, from the stage."""
        import omni.usd  # pylint: disable=C0415

        omni.usd.get_context().get_stage().RemovePrim(robot.prim())

    @staticmethod
    def _reference(
        robot: Robot, stage_path: Path, head_camera_pack: head_camera.Pack | None
    ) -> None:
        """Puts a model's stage under this robot's prim, at its placement,
        and seats on it the head camera its model draws.

        The reference brings the model's own transform ops along, and these
        models orient by quaternion, which XformCommonAPI does not carry: it
        reports itself unusable on such a prim and writes nothing, leaving
        every robot stacked on the origin. The placement is written into the
        ops the prim actually has."""
        import omni.usd  # pylint: disable=C0415

        stage = omni.usd.get_context().get_stage()
        prim = stage.DefinePrim(robot.prim(), "Xform")
        prim.GetReferences().AddReference(str(stage_path))

        World.place(prim, robot.placement.position, robot.placement.yaw)

        if head_camera_pack is not None:
            body = head_camera.attach(stage, robot.prim(), head_camera_pack)
            logger.info(
                "robot '%s' draws its head camera at %s from %s",
                robot.instance,
                body,
                head_camera_pack.directory,
            )

    @staticmethod
    def place(prim, position, yaw) -> None:
        """Stands a prim, a robot or a spawned object, at `position` turned
        `yaw` radians about +z. USD applies an xform's ops in the order they
        are listed and a referenced model brings its own, so the order is
        pinned here: the prim turns about its own origin and is then moved.
        Left to the order the reference happened to leave, a robot asked for
        (1, 0, 0) at a quarter turn stands at (0, 1, 0)."""
        from pxr import Gf, UsdGeom  # pylint: disable=C0415

        xform = UsdGeom.Xformable(prim)
        ops = {op.GetOpType(): op for op in xform.GetOrderedXformOps()}

        translate = ops.get(UsdGeom.XformOp.TypeTranslate) or xform.AddTranslateOp()
        translate.Set(Gf.Vec3d(*position))

        half = yaw / 2.0
        axis = (0.0, 0.0, math.sin(half))
        orient = ops.get(UsdGeom.XformOp.TypeOrient) or xform.AddOrientOp()
        # USD refuses a quaternion of the wrong width outright, and a
        # referenced model brings whichever its author chose.
        turn = {
            UsdGeom.XformOp.PrecisionDouble: lambda: Gf.Quatd(
                math.cos(half), Gf.Vec3d(*axis)
            ),
            UsdGeom.XformOp.PrecisionHalf: lambda: Gf.Quath(
                math.cos(half), Gf.Vec3h(*axis)
            ),
        }.get(orient.GetPrecision(), lambda: Gf.Quatf(math.cos(half), Gf.Vec3f(*axis)))
        orient.Set(turn())

        # Whatever else the reference carried keeps its place after these two.
        rest = [
            op
            for op in xform.GetOrderedXformOps()
            if op.GetOpType()
            not in (UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeOrient)
        ]
        xform.SetXformOpOrder([translate, orient, *rest])

    @staticmethod
    def _rounded(position) -> tuple[int, int, int]:
        """A placement as a spot on the lattice, so a robot that asked for a
        spot by hand still counts as standing on it."""
        return tuple(int(round(value / SPOT_PITCH_M)) for value in position)
