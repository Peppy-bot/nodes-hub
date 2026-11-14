#!/usr/bin/env python3
"""The stage the engine simulates, and the robots standing in it.

The stage opens empty and each robot is referenced under /World as a prim of
its own name, so two robots of the same model keep separate articulations,
and taking one out is removing that prim. What is referenced, and what is
authored on it as the robot joins, is its model's own entry.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Optional

import head_camera
from isaac_models import IsaacModel

logger = logging.getLogger(__name__)

# Where each robot of the scene is referenced, under the name it stands as.
WORLD_PRIM = "/World"

# Side of the square the engine parks robots on when one asks for no
# placement of its own: far enough apart that no two robots of the models
# this engine stands can touch.
SPOT_PITCH_M = 1.5

# How far the lattice is walked before a fleet is told the stage is full.
_MAX_RINGS = 64

# What stands between a robot's name and the stage path it is a prim of:
# `/World/alpha` is alpha's.
PRIM_SEPARATOR = "/"

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

# Where a joint's drive and its PhysX joint state are authored, by the kind
# of joint: USD writes a revolute joint's positions in degrees and a
# prismatic joint's in the stage's own length unit, which is the metre.
JOINT_STATE_API = "PhysxJointStateAPI"
ANGULAR = "angular"
LINEAR = "linear"

# How far a stage's joint limit may stand from the description's, in the
# description's unit, rad or m: a stage keeps a limit in single precision, a
# revolute joint's in degrees, 3.2e-7 rad off at most on the OpenArm bundles.
LIMIT_TOLERANCE = 1e-6

# The PhysX schema that binds one joint's position to another's, applied
# once per degree of freedom under an instance name that names its axis,
# "rot" and the joint's axis for a revolute and a prismatic joint alike, and
# the prefix of its attributes. It holds position + gearing * reference
# position + offset = 0, its offset in the joint's own unit.
MIMIC_JOINT_API = "PhysxMimicJointAPI"
MIMIC_JOINT = "physxMimicJoint"

# The PhysX rigid body schema and its attribute that takes a link out of
# gravity's reach.
PHYSX_RIGID_BODY_API = "PhysxRigidBodyAPI"
DISABLE_GRAVITY = "physxRigidBody:disableGravity"


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

    def orientation(self) -> tuple[float, float, float, float]:
        """The rotation as a unit quaternion (x, y, z, w)."""
        return yaw_orientation(self.yaw)


def yaw_orientation(yaw: float) -> tuple[float, float, float, float]:
    """The turn of `yaw` radians about +z as a unit quaternion (x, y, z, w):
    the one rule every heading becomes an orientation by, a robot's and a
    spawned object's alike."""
    half = yaw / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _joints_named(root, names: set[str]) -> dict:
    """The joint prims under `root` whose names are among `names`, by name."""
    from pxr import Usd, UsdPhysics  # pylint: disable=C0415

    return {
        prim.GetName(): prim
        for prim in Usd.PrimRange(root)
        if prim.IsA(UsdPhysics.Joint) and prim.GetName() in names
    }


def _degree_of_freedom(joint, known: IsaacModel, name: str):
    """The kind of the one degree of freedom of `joint`, the joint `name` of
    the model `known`, and the conversion of a position from the
    description's unit to the stage's: degrees on a revolute joint, the metre
    on a prismatic one. Any other joint is refused."""
    from pxr import UsdPhysics  # pylint: disable=C0415

    if joint.IsA(UsdPhysics.RevoluteJoint):
        return ANGULAR, math.degrees
    if joint.IsA(UsdPhysics.PrismaticJoint):
        return LINEAR, float
    raise RuntimeError(
        f"the {known.model} entry moves '{name}', which {known.stage} makes a "
        f"{joint.GetTypeName()}: a robot's limbs move revolute and prismatic joints"
    )


def _moving_joints(root) -> dict:
    """The revolute and prismatic joint prims under `root`, by name."""
    from pxr import Usd, UsdPhysics  # pylint: disable=C0415

    return {
        prim.GetName(): prim
        for prim in Usd.PrimRange(root)
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
    }


def _stage_limits(prim) -> tuple[float, float]:
    """The lower and upper limit of the revolute or prismatic joint `prim`,
    in rad or m."""
    from pxr import UsdPhysics  # pylint: disable=C0415

    if prim.IsA(UsdPhysics.RevoluteJoint):
        joint = UsdPhysics.RevoluteJoint(prim)
        lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
        return math.radians(lower), math.radians(upper)
    joint = UsdPhysics.PrismaticJoint(prim)
    return float(joint.GetLowerLimitAttr().Get()), float(joint.GetUpperLimitAttr().Get())


def name_in_the_stage(instance: str) -> str:
    """The name a robot stands under. Every robot is a prim of the stage
    under this name, so a name carrying the separator nests one robot inside
    another's path."""
    if not instance:
        raise ValueError("a robot stands under the name of the copy it runs as")
    if PRIM_SEPARATOR in instance:
        plain = instance.replace(PRIM_SEPARATOR, "_")
        raise ValueError(
            f"a robot stands under a name carrying no {PRIM_SEPARATOR!r}, and "
            f"'{instance}' carries one: join under a name without it, "
            f"`peppy stack join LAUNCHER -i {plain}`"
        )
    return instance


def within_one_spot(one: Placement, other: Placement) -> bool:
    """Whether two placements are too close for a robot to stand on both. It
    is the distance between them that counts: two robots closer together than
    the lattice leaves them resolve their overlap by throwing each other,
    wherever the lattice's own lines happen to fall."""
    return math.dist(one.position, other.position) < SPOT_PITCH_M


@dataclass(frozen=True)
class Robot:
    """One robot in the stage: the name it stands under, the model it is,
    and where."""

    instance: str
    known: IsaacModel
    placement: Placement

    @property
    def model(self) -> str:
        return self.known.model

    def prim(self) -> str:
        """The prim this robot's stage is referenced under."""
        return f"{WORLD_PRIM}/{self.instance}"

    def articulation(self) -> str:
        """The prim the engine's views read this robot's articulation at."""
        return self.known.articulation_path(self.prim())


class World:
    """The stage as opened, and the robots it holds. Standing a robot adds a
    prim referencing its model; taking one out removes that prim. Both leave
    the views on the articulations invalid, so the engine resolves the stage
    again around them."""

    def __init__(self, head_camera_pack: Optional[head_camera.Pack]) -> None:
        # The pack a model that draws the head camera seats on its pedestal,
        # or None when no model of this engine draws it.
        self._head_camera_pack = head_camera_pack
        self._robots: dict[str, Robot] = {}
        # Robots are stood and taken out on the thread that steps the scene,
        # and listed on the one serving the contracts, so the two of them do
        # not read this while the other is changing it. Reentrant, because
        # the lookups below are written in terms of each other.
        self._lock = threading.RLock()

    def robots(self) -> list[Robot]:
        """Every robot in the stage, in the order they joined."""
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
            "every spot this stage lays out is taken: take a robot out with "
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

    def add(self, instance: str, known: IsaacModel, placement: Placement) -> Robot:
        """Stands a robot of the model `known` in the stage. Admission has
        already found the caller a name and a spot, so this raises on
        either."""
        instance = name_in_the_stage(instance)
        with self._lock:
            if instance in self._robots:
                raise ValueError(
                    f"'{instance}' is still on the stage, which a robot that could not be "
                    "taken out is until this simulation restarts: `peppy stack list` says "
                    "whether its copy still runs"
                )
        robot = Robot(instance=instance, known=known, placement=placement)
        try:
            self._reference(robot, self._head_camera_pack)
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
            known.model,
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
            known=robot.known,
            placement=Placement.of(position, yaw),
        )

        import omni.usd  # pylint: disable=C0415

        prim = omni.usd.get_context().get_stage().GetPrimAtPath(moved.prim())
        if not prim.IsValid():
            raise RuntimeError(
                f"the prim {moved.prim()} of robot {name!r} does not exist"
            )
        World.place(prim, moved.placement.position, moved.placement.orientation())

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
    def _reference(robot: Robot, head_camera_pack: Optional[head_camera.Pack]) -> None:
        """Puts a model's stage under this robot's prim, at its placement,
        holds the stage to the robot's description, and authors on it what
        its model's entry asks for: the posture the robot starts in, each finger that mimics another bound to it, its
        links out of gravity's reach, and the head camera on its pedestal.

        The reference brings the model's own transform ops along, and these
        models orient by quaternion, which XformCommonAPI does not carry: it
        reports itself unusable on such a prim and writes nothing, leaving
        every robot stacked on the origin. The placement is written into the
        ops the prim actually has."""
        import omni.usd  # pylint: disable=C0415

        known = robot.known
        stage = omni.usd.get_context().get_stage()
        prim = stage.DefinePrim(robot.prim(), "Xform")
        prim.GetReferences().AddReference(str(known.stage_path()))

        World.place(prim, robot.placement.position, robot.placement.orientation())
        World._hold_to_description(prim, known)
        World._start_in_posture(prim, known)
        World._couple_followers(prim, known)
        if known.gravity_compensation:
            compensated = World._compensate_gravity(prim)
            logger.info(
                "robot '%s' stands with gravity compensated on %d links",
                robot.instance,
                compensated,
            )

        if known.head_camera:
            if head_camera_pack is None:
                raise RuntimeError(
                    f"{known.model} draws the head camera, and no pack was staged"
                )
            head_camera.check_chest_camera(head_camera_pack.rig, known.entry)
            body = head_camera.attach(stage, robot.prim(), head_camera_pack)
            logger.info(
                "robot '%s' draws its head camera at %s from %s",
                robot.instance,
                body,
                head_camera_pack.directory,
            )

    @staticmethod
    def _hold_to_description(root, known: IsaacModel) -> None:
        """Refuses a stage whose joints are not the description's: each joint
        a limb of the robot moves is a revolute or prismatic joint of the
        stage with the description's limits, within LIMIT_TOLERANCE; a finger
        that follows another holds the description's limits within its own,
        the joint it follows placing it; and every other revolute or
        prismatic joint of the stage is locked, its two limits one, as a
        joint the description fixes."""
        moving = _moving_joints(root)
        followers = {
            joint for gripper in known.entry.grippers for joint, _ in gripper.followers()
        }
        differences = []
        for name, (lower, upper) in known.position_limits.items():
            prim = moving.get(name)
            if prim is None:
                differences.append(f"'{name}' is no revolute or prismatic joint of it")
                continue
            low, high = _stage_limits(prim)
            if name in followers:
                held = low <= lower + LIMIT_TOLERANCE and high >= upper - LIMIT_TOLERANCE
            else:
                held = abs(low - lower) <= LIMIT_TOLERANCE and abs(high - upper) <= LIMIT_TOLERANCE
            if not held:
                differences.append(
                    f"'{name}' moves over [{low:.6g}, {high:.6g}] where the description "
                    f"gives [{lower:.6g}, {upper:.6g}]"
                )
        for name, prim in moving.items():
            if name in known.position_limits:
                continue
            low, high = _stage_limits(prim)
            if high - low > LIMIT_TOLERANCE:
                differences.append(
                    f"'{name}' moves over [{low:.6g}, {high:.6g}], and the description moves no joint "
                    "of that name"
                )
        if differences:
            raise RuntimeError(
                f"{known.stage} is not the {known.model} its description describes: "
                + "; ".join(differences)
                + ". Pin the asset bundle built for this description in "
                "sim_base_images/isaac_assets.env and rebuild the base image: "
                "bash sim_base_images/build_base_images.sh --isaac-only"
            )

    @staticmethod
    def _start_in_posture(root, known: IsaacModel) -> None:
        """Puts the robot in the posture its model starts in, and holds it
        there: each joint's PhysX state, which is where the simulation starts
        it once the timeline plays with the robot on the stage, and the
        target of the drive of each joint a motor drives, so the arm stands
        still until its first setpoint. A finger that follows another has no
        drive: its coupling holds it. A posture naming a joint the stage lacks
        is refused, so a stage that names its joints otherwise is caught at
        the first stand."""
        posture = known.entry.start_posture
        followers = {
            joint for gripper in known.entry.grippers for joint, _ in gripper.followers()
        }
        from pxr import Sdf, UsdPhysics  # pylint: disable=C0415

        joints = _joints_named(root, set(posture))
        missing = sorted(set(posture) - set(joints))
        if missing:
            raise RuntimeError(
                f"the {known.model} entry starts joints not in {known.stage}: {missing}"
            )
        for name, position in posture.items():
            joint = joints[name]
            kind, in_stage_unit = _degree_of_freedom(joint, known, name)
            value = in_stage_unit(position)
            joint.AddAppliedSchema(f"{JOINT_STATE_API}:{kind}")
            joint.CreateAttribute(
                f"state:{kind}:physics:position", Sdf.ValueTypeNames.Float, False
            ).Set(value)
            if name not in followers:
                UsdPhysics.DriveAPI.Apply(joint, kind).CreateTargetPositionAttr().Set(value)

    @staticmethod
    def _couple_followers(root, known: IsaacModel) -> None:
        """Binds each finger joint that follows another, as the robot's
        description gives it, to the joint it follows with PhysX's mimic
        joint, so one drive moves both: the description's position =
        multiplier * followed position + offset is PhysX's relationship with
        gearing -multiplier and offset -offset, in the stage's unit. A
        follower and a joint it follows that the stage lacks, or that are of
        two kinds, are refused."""
        followers = {
            joint: mimic
            for gripper in known.entry.grippers
            for joint, mimic in gripper.followers()
        }
        from pxr import Sdf, UsdPhysics  # pylint: disable=C0415

        named = set(followers) | {mimic.joint for mimic in followers.values()}
        joints = _joints_named(root, named)
        missing = sorted(named - set(joints))
        if missing:
            raise RuntimeError(
                f"the {known.model} entry couples finger joints not in {known.stage}: {missing}"
            )
        for name, mimic in followers.items():
            joint = joints[name]
            kind, in_stage_unit = _degree_of_freedom(joint, known, name)
            followed_kind, _ = _degree_of_freedom(joints[mimic.joint], known, mimic.joint)
            if kind != followed_kind:
                raise RuntimeError(
                    f"the {known.model} entry has '{name}' follow '{mimic.joint}', and "
                    f"{known.stage} makes one {kind} and the other {followed_kind}: a finger "
                    "follows a joint of its own kind"
                )
            axis = (
                UsdPhysics.RevoluteJoint(joint).GetAxisAttr().Get()
                if kind == ANGULAR
                else UsdPhysics.PrismaticJoint(joint).GetAxisAttr().Get()
            )
            instance = f"rot{axis}"
            joint.AddAppliedSchema(f"{MIMIC_JOINT_API}:{instance}")
            prefix = f"{MIMIC_JOINT}:{instance}"
            joint.CreateAttribute(f"{prefix}:gearing", Sdf.ValueTypeNames.Float, False).Set(
                -mimic.multiplier
            )
            joint.CreateAttribute(f"{prefix}:offset", Sdf.ValueTypeNames.Float, False).Set(
                -in_stage_unit(mimic.offset)
            )
            joint.CreateRelationship(f"{prefix}:referenceJoint", False).SetTargets(
                [joints[mimic.joint].GetPath()]
            )

    @staticmethod
    def _compensate_gravity(root) -> int:
        """Mirrors the real driver's in-process gravity feedforward by
        disabling gravity on every rigid-body link of the robot, so PhysX
        behaves as if an exact counter-gravity force were applied each step
        (the real arm and gripper drivers both feedforward). Coriolis is
        intentionally not compensated. Returns how many links it reached."""
        from pxr import Sdf, Usd, UsdPhysics  # pylint: disable=C0415

        compensated = 0
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            prim.AddAppliedSchema(PHYSX_RIGID_BODY_API)
            prim.CreateAttribute(DISABLE_GRAVITY, Sdf.ValueTypeNames.Bool, False).Set(True)
            compensated += 1
        return compensated

    @staticmethod
    def place(prim, position, orientation=None) -> None:
        """Stands a prim, a robot or a spawned object, at `position` turned to
        `orientation`, a unit quaternion (x, y, z, w), or turned as it stands
        when none is given. USD applies an xform's ops in the order they are
        listed and a referenced model brings its own, so the order is pinned
        here: the prim turns about its own origin and is then moved. Left to
        the order the reference happened to leave, a robot asked for
        (1, 0, 0) at a quarter turn stands at (0, 1, 0)."""
        from pxr import Gf, UsdGeom  # pylint: disable=C0415

        xform = UsdGeom.Xformable(prim)
        ops = {op.GetOpType(): op for op in xform.GetOrderedXformOps()}

        translate = ops.get(UsdGeom.XformOp.TypeTranslate) or xform.AddTranslateOp()
        translate.Set(Gf.Vec3d(*position))

        orient = ops.get(UsdGeom.XformOp.TypeOrient)
        if orientation is not None:
            x, y, z, w = (float(value) for value in orientation)
            orient = orient or xform.AddOrientOp()
            # USD refuses a quaternion of the wrong width outright, and a
            # referenced model brings whichever its author chose.
            turn = {
                UsdGeom.XformOp.PrecisionDouble: lambda: Gf.Quatd(w, Gf.Vec3d(x, y, z)),
                UsdGeom.XformOp.PrecisionHalf: lambda: Gf.Quath(w, Gf.Vec3h(x, y, z)),
            }.get(orient.GetPrecision(), lambda: Gf.Quatf(w, Gf.Vec3f(x, y, z)))
            orient.Set(turn())

        # Whatever else the reference carried keeps its place after these.
        placing = [translate] if orient is None else [translate, orient]
        rest = [
            op
            for op in xform.GetOrderedXformOps()
            if op.GetOpType()
            not in (UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeOrient)
        ]
        xform.SetXformOpOrder([*placing, *rest])
