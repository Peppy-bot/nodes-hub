#!/usr/bin/env python3
"""The stage the engine simulates, and the robots standing in it.

The stage opens empty but for its floor, which it keeps unless the scene's
own floor lies at z 0 under every robot standing (World.scene_floor_holds),
and each robot is referenced under /World as a prim of its own name, so two
robots of the same model keep separate articulations, and taking one out is
removing that prim. What is referenced, and what is authored on it as the
robot joins, is its model's own entry. A robot that rolls stands free on the
floor, and its record follows where its wheels carry it.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, replace
from typing import Optional

from control_core_py.geometry import Point, Pose, PoseTarget, UnitQuaternion, wxyz_of_xyzw
from sim_robot_core.object_state import AT_REST, RobotRecord
from sim_robot_core.world import rolled_placement

import head_camera
from isaac_models import IsaacModel
from wheel_loop import INTEGRAL_GAIN_ATTRIBUTE, RateLoopGains

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

# The floor of a stage whose scene brings none of its own: a plane collider
# at z 0 of world, which a robot that rolls rests its wheels on.
GROUND_PRIM = f"{WORLD_PRIM}/ground"

# How far from z 0 a scene's floor may lie under a robot for the stage's
# floor to come up (m). The floors of the catalogue's scenes that lie at z 0
# lie within 0.03 mm of it, and the nearest floor off z 0, curved_grid's, lies
# 7.0 mm under it.
FLOOR_TOLERANCE_M = 1e-3

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


# What a wheel's drive that is not its rate loop asks for.
_REBUILD_THE_BUNDLE = (
    "Pin the asset bundle scripts/build_botshare_base.py builds in sim_base_images/isaac_assets.env"
)


def _same_limit(stage: float, described: float) -> bool:
    """Whether a stage's joint limit is the description's, within
    LIMIT_TOLERANCE: the same infinity on a joint that turns without end."""
    return stage == described or abs(stage - described) <= LIMIT_TOLERANCE


def _rolling_differences(root, known: IsaacModel) -> list[str]:
    """How the stage of a robot that rolls is not one: its robot frame, the
    description's root link, is the rigid body no joint moves, which the
    articulation is rooted at and the robot is read at, and no joint holds it
    to the world."""
    from pxr import Usd, UsdPhysics  # pylint: disable=C0415

    joints = [UsdPhysics.Joint(prim) for prim in Usd.PrimRange(root) if prim.IsA(UsdPhysics.Joint)]
    frame = known.prim_of(known.root_link)
    differences = [
        f"'{joint.GetPrim().GetName()}' holds it to the world, and a robot that rolls stands free"
        for joint in joints
        if not joint.GetBody0Rel().GetTargets()
    ]
    body = known.link_prim(root, known.root_link)
    if body is None or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        differences.append(f"its robot frame '{known.root_link}' is no rigid body of it")
    differences.extend(
        f"'{joint.GetPrim().GetName()}' hangs its robot frame '{known.root_link}' from another body, "
        "and the robot is read at the root of its articulation"
        for joint in joints
        if any(target.name == frame for target in joint.GetBody1Rel().GetTargets())
    )
    return differences


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


def within_one_spot(one: Pose, other: Pose) -> bool:
    """Whether two placements are too close for a robot to stand on both. It
    is the distance between them that counts: two robots closer together than
    the lattice leaves them resolve their overlap by throwing each other,
    wherever the lattice's own lines happen to fall."""
    return math.dist(one.position.position, other.position.position) < SPOT_PITCH_M


@dataclass(frozen=True)
class Robot:
    """One robot in the stage: the name it stands under, the model it is,
    where: the pose of its base in world, whose orientation turns about +z
    alone, how fast its robot frame moves in world at its origin, at rest
    for a robot fixed to the world, and the rate loop of each wheel of its
    base as its stage authors it, left then right, none for a robot that
    stands where it is placed."""

    instance: str
    known: IsaacModel
    placement: Pose
    linear_velocity: tuple[float, float, float] = AT_REST
    angular_velocity: tuple[float, float, float] = AT_REST
    wheel_loops: tuple[RateLoopGains, ...] = ()
    # The pose of its robot frame in world as the articulation's root last
    # read it, tilted as its body is; None while the robot frame stands at
    # its placement: a robot fixed to the world, and a robot that rolls
    # until a step reads it and from each move until the next.
    rolled_frame: Optional[Pose] = None

    @property
    def model(self) -> str:
        return self.known.model

    def record(self) -> RobotRecord:
        """Its record on object_state: the pose of its robot frame in world,
        its tilt included, with its velocities there."""
        frame = self.placement if self.rolled_frame is None else self.rolled_frame
        return RobotRecord(
            robot=self.instance,
            model=self.model,
            position=frame.position.position,
            orientation=frame.orientation.xyzw,
            linear_velocity=self.linear_velocity,
            angular_velocity=self.angular_velocity,
        )

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
        # The robots that roll and were moved since the thread that steps the
        # scene last set them down at rest.
        self._set_down: set[str] = set()
        # Whether an edit changed the scene or a placement since the floor
        # was last judged (World.take_floor_judgment).
        self._floor_due = False
        # The engine time of the step whose roll last recorded the robots
        # (s), the instant their records hold at; None before the first.
        self._recorded_at_s: Optional[float] = None
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
        self, promised: "tuple[Pose, ...]" = (), leaving: Optional[str] = None
    ) -> Pose:
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
                spot = Pose(
                    Point((row * SPOT_PITCH_M, column * SPOT_PITCH_M, 0.0)), UnitQuaternion.IDENTITY
                )
                if not self.occupied(spot, promised, leaving):
                    return spot
        raise ValueError(
            "every spot this stage lays out is taken: take a robot out with "
            "`peppy stack remove NAME` before standing another"
        )

    def standing_within(
        self, placement: Pose, leaving: Optional[str] = None
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
        placement: Pose,
        promised: "tuple[Pose, ...]" = (),
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
        World.lay_floor(stage)
        # An empty stage carries no light of its own, and renders black
        # however many robots join it.
        if not any(prim.IsA(UsdLux.DomeLight) for prim in stage.Traverse()):
            dome = UsdLux.DomeLight.Define(stage, f"{WORLD_PRIM}/dome_light")
            dome.CreateIntensityAttr(DOME_LIGHT_INTENSITY)

    @staticmethod
    def lay_floor(stage):
        """The floor of the stage: a plane collider at z 0 of world, facing
        +z, which a robot that rolls rests its wheels on. Hydra draws no
        plane, and it stands as a guide."""
        from pxr import UsdGeom, UsdPhysics  # pylint: disable=C0415

        ground = UsdGeom.Plane.Define(stage, GROUND_PRIM)
        ground.CreateAxisAttr(UsdGeom.Tokens.z)
        ground.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        return ground.GetPrim()

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

    def add(self, instance: str, known: IsaacModel, placement: Pose) -> Robot:
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
            robot = replace(robot, wheel_loops=self._reference(robot, self._head_camera_pack))
        except Exception:
            # What the reference authored before it failed comes off the
            # stage, so a robot that cannot join leaves the stage as it was.
            World._unreference(robot)
            raise
        with self._lock:
            self._robots[instance] = robot
        import omni.usd  # pylint: disable=C0415

        self.lay_floor_until_judged(omni.usd.get_context().get_stage())
        logger.info(
            "robot '%s' (%s) joins at %s",
            instance,
            known.model,
            list(placement.position.position),
        )
        return robot

    def move(self, name: str, target: PoseTarget) -> Robot:
        """Puts a robot somewhere else in the stage: its base at the
        target's position, turned to the target's orientation, a turn about
        +z, or keeping the heading it has when the target gives none. The
        record moves with the prim, so what the scene reports and where the
        next robot is given room both follow it."""
        robot = next((one for one in self.robots() if one.instance == name), None)
        if robot is None:
            raise KeyError(f"no robot stands as {name!r} in this stage")
        heading = robot.placement.orientation if target.orientation is None else target.orientation
        moved = replace(
            robot,
            placement=Pose(target.position, heading),
            linear_velocity=AT_REST,
            angular_velocity=AT_REST,
            rolled_frame=None,
        )

        import omni.usd  # pylint: disable=C0415

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(moved.prim())
        if not prim.IsValid():
            raise RuntimeError(
                f"the prim {moved.prim()} of robot {name!r} does not exist"
            )
        World.place(prim, moved.placement.position.position, moved.placement.orientation.xyzw)

        with self._lock:
            self._robots[moved.instance] = moved
            if moved.known.rolls:
                self._set_down.add(moved.instance)
        self.lay_floor_until_judged(stage)
        logger.info(
            "robot '%s' moves to %s turned to %s",
            name,
            list(moved.placement.position.position),
            list(moved.placement.orientation.xyzw),
        )
        return moved

    def roll(
        self,
        stamp_s: float,
        rolled: dict[str, tuple[Pose, tuple[float, float, float], tuple[float, float, float]]],
    ) -> None:
        """Records one step at the engine time `stamp_s`: where the wheels of
        each robot that rolls carried it, by name, as (frame, linear velocity,
        angular velocity), `frame` the pose of its robot frame in world, tilted
        as its body is, and how fast it moves there. Each robot's record, and
        its placement, that position facing the heading its wheels turned it
        to, where the next robot is given room from, change together with the
        stamp, so a listing pairs them (World.listing). A robot that left is
        not recorded again."""
        with self._lock:
            for instance, (frame, linear_velocity, angular_velocity) in rolled.items():
                robot = self._robots.get(instance)
                if robot is None:
                    continue
                self._robots[instance] = replace(
                    robot,
                    placement=rolled_placement(frame.position.position, frame.orientation.xyzw),
                    linear_velocity=linear_velocity,
                    angular_velocity=angular_velocity,
                    rolled_frame=frame,
                )
            self._recorded_at_s = stamp_s

    def listing(self) -> tuple[Optional[float], list[Robot]]:
        """Every robot in the stage, in the order they joined, with the
        engine time of the step whose roll recorded them (s), None before the
        first: the instant their records hold at."""
        with self._lock:
            return self._recorded_at_s, list(self._robots.values())

    def take_set_down(self, reading: set[str]) -> set[str]:
        """The robots of `reading`, the robots whose views read, that roll and
        were moved since they were last taken: each starts at rest where its
        move stood it. A moved robot whose views do not read yet is taken by
        the first step that reads them."""
        with self._lock:
            moved = self._set_down & reading
            self._set_down = self._set_down - moved
        return moved

    def lay_floor_until_judged(self, stage) -> None:
        """Lays the stage's floor where it is not, and asks for the floor to
        be judged again (take_floor_judgment). A load, a clear, a join and a
        move each stand robots on a floor no judgment has seen, so the stage's
        floor holds them until the scene's own floor is found to."""
        if not stage.GetPrimAtPath(GROUND_PRIM).IsValid():
            World.lay_floor(stage)
        with self._lock:
            self._floor_due = True

    def take_floor_judgment(self) -> bool:
        """Whether a robot joined, left or was moved, or the scene changed,
        since the last call: the floor is judged again, once."""
        with self._lock:
            due, self._floor_due = self._floor_due, False
        return due

    def floor_spots(self) -> list[tuple[float, float]]:
        """Where on the floor the floor rule looks: under each robot
        standing, or under the origin while none stands (x, y in world)."""
        spots = [tuple(robot.placement.position.position[:2]) for robot in self.robots()]
        return spots or [(0.0, 0.0)]

    @staticmethod
    def scene_floor_holds(heights: list[Optional[float]]) -> bool:
        """Whether the scene's floor holds every robot at z 0: under each of
        the floor spots a floor is found (a height, None for none), and each
        lies within FLOOR_TOLERANCE_M of z 0."""
        return bool(heights) and all(
            height is not None and abs(height) <= FLOOR_TOLERANCE_M for height in heights
        )

    def stand_where_rolled(self) -> None:
        """Stands the prim of every robot that rolls where its wheels carried
        it, and its record there at rest. A timeline that stops sets every
        body back on its prim, still, so a robot that rolled stays where it
        went when the stage is edited, and its record holds as the stage does
        until a step reads it again."""
        import omni.usd  # pylint: disable=C0415

        stage = omni.usd.get_context().get_stage()
        with self._lock:
            rolling = [robot for robot in self._robots.values() if robot.known.rolls]
            for robot in rolling:
                World.place(
                    stage.GetPrimAtPath(robot.prim()),
                    robot.placement.position.position,
                    robot.placement.orientation.xyzw,
                )
            self._robots.update(
                {
                    robot.instance: replace(
                        robot, linear_velocity=AT_REST, angular_velocity=AT_REST, rolled_frame=None
                    )
                    for robot in rolling
                }
            )

    def remove(self, instance: str) -> None:
        """Takes a robot out of the stage."""
        with self._lock:
            robot = self._robots.pop(instance, None)
            self._set_down.discard(instance)
            if robot is not None:
                self._floor_due = True
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
        robot: Robot, head_camera_pack: Optional[head_camera.Pack]
    ) -> tuple[RateLoopGains, ...]:
        """Puts a model's stage under this robot's prim, at its placement,
        holds the stage to the robot's description, and authors on it what
        its model's entry asks for: the posture the robot starts in, each finger that mimics another bound to it, its
        links out of gravity's reach, and the head camera on its pedestal.
        Returns the rate loop of each wheel of its base, as its stage
        authors it.

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

        World.place(prim, robot.placement.position.position, robot.placement.orientation.xyzw)
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
        return World._wheel_loops(prim, known)

    @staticmethod
    def _wheel_loops(root, known: IsaacModel) -> tuple[RateLoopGains, ...]:
        """The rate loop of each wheel of the robot's base, left then right,
        as its stage authors it: a velocity drive, no stiffness, the
        proportional gain as its damping, per degree as USD writes an
        angular drive's, and the motor's torque as its maximum force, and the
        integral gain as the joint's INTEGRAL_GAIN_ATTRIBUTE. None for a
        robot with no base. A wheel whose drive has a stiffness, or whose
        damping, maximum force or integral gain is no positive finite number,
        is refused."""
        from pxr import UsdPhysics  # pylint: disable=C0415

        base = known.entry.base
        if base is None:
            return ()
        joints = _joints_named(root, set(base.wheels))
        loops = []
        for wheel in base.wheels:
            prim = joints.get(wheel)
            drive = (
                UsdPhysics.DriveAPI(prim, ANGULAR)
                if prim is not None and prim.HasAPI(UsdPhysics.DriveAPI, ANGULAR)
                else None
            )
            if drive is None or drive.GetStiffnessAttr().Get() != 0.0:
                raise RuntimeError(
                    f"{known.stage} drives the wheel '{wheel}' with no velocity drive: a wheel's drive "
                    f"has no stiffness. {_REBUILD_THE_BUNDLE}"
                )
            values = {
                "damping": drive.GetDampingAttr().Get(),
                "maximum force": drive.GetMaxForceAttr().Get(),
                INTEGRAL_GAIN_ATTRIBUTE: prim.GetAttribute(INTEGRAL_GAIN_ATTRIBUTE).Get(),
            }
            unusable = {
                name: value
                for name, value in values.items()
                if value is None or not math.isfinite(value) or value <= 0.0
            }
            if unusable:
                raise RuntimeError(
                    f"{known.stage} gives the wheel '{wheel}' {unusable}: its rate loop takes a positive "
                    f"number of each. {_REBUILD_THE_BUNDLE}"
                )
            loops.append(
                RateLoopGains(
                    proportional_gain=math.degrees(values["damping"]),
                    integral_gain=float(values[INTEGRAL_GAIN_ATTRIBUTE]),
                    max_torque=float(values["maximum force"]),
                )
            )
        return tuple(loops)

    @staticmethod
    def _hold_to_description(root, known: IsaacModel) -> None:
        """Refuses a stage whose joints are not the description's: each joint
        a limb or the base of the robot moves is a revolute or prismatic
        joint of the stage with the description's limits, within
        LIMIT_TOLERANCE, none on a wheel that turns without end; a finger that
        follows another holds the description's limits within its own, the
        joint it follows placing it; every other revolute or prismatic joint
        of the stage is locked, its two limits one, as a joint the
        description fixes; and a robot that rolls stands free, rooted at its
        robot frame."""
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
                held = _same_limit(low, lower) and _same_limit(high, upper)
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
        if known.rolls:
            differences.extend(_rolling_differences(root, known))
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
            w, x, y, z = (float(value) for value in wxyz_of_xyzw(orientation))
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
