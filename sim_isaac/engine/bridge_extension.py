#!/usr/bin/env python3
# pylint: disable=R0902,C0413
"""IsaacBridgeExtension owns the per-step bridge for the Isaac stage. Isaac's
sim_app.update() advances physics on the main thread; each step this
extension applies the latest setpoint of every robot's limbs, reads the
measured joint and gripper state, and (throttled to state_rate_hz) publishes
Isaac's own timeline clock followed by the state it stamps: each robot's
joint and gripper states and a snapshot of every spawned object.

Every robot is commanded through its own limb pairs and publishes its state
back on them; the copy a pair carries is the robot it belongs to, and the
limbs it is driven by are its own model's. Transport is typed peppygen via
SimTopicIO; there is no JSON and no raw peppylib on the path.
"""
from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sim_robot_core.cameras import FramePacer
from sim_robot_core.models import FingerSpan, finger_span
from sim_robot_core.registry import Registry

from world import Robot, World
from exts import IsaacActuatorCtrl, IsaacArticulation, IsaacCameraSensor, IsaacGripperSensor

if TYPE_CHECKING:  # the transport, which carries the node runtime with it
    from sim_topics import SimTopicIO

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArmSetpoint:
    """One arm's commanded joint positions, and the velocities to reach them
    with when the command carries any."""

    positions: tuple[float, ...]
    velocities: tuple[float, ...]


@dataclass(frozen=True)
class GripperSetpoint:
    """One gripper's commanded opening fraction and the force it may apply;
    0 leaves the model's own limit as the only one."""

    opening: float
    max_effort: float


class RobotLimbs:
    """One robot's limbs on the stage, as its model's entry names them: the
    view that reads its articulation, a controller per limb, and the sensor
    that reads each gripper. Isaac registers a view under a name, so every
    view here carries the name of the robot it belongs to, and a limb's the
    name of its limb, which no other limb of its model shares."""

    def __init__(self, robot: Robot) -> None:
        self.robot = robot
        # What this robot's model is made of: its limbs under the names its
        # pairs carry, and the joints each one moves.
        self.entry = robot.known.entry
        known = robot.known
        prim = robot.articulation()
        view = robot.instance or "standing"
        self.articulation = IsaacArticulation(prim, name=f"articulation_{view}")
        # One actuator controller per limb: the gains and effort ceilings of
        # the model's entry are applied to that limb's PhysX drives at setup,
        # and the limb's commands are written through it. A limb whose model
        # carries no gains keeps the drives its stage authors.
        self.arm_actuators = {
            arm.name: IsaacActuatorCtrl(
                prim,
                joint_names=list(arm.joints),
                params=known.arm_params(arm),
                name=f"{arm.name}_{view}",
            )
            for arm in self.entry.arms
        }
        self.gripper_actuators = {
            gripper.name: IsaacActuatorCtrl(
                prim,
                joint_names=list(gripper.joints),
                params=known.gripper_params(gripper),
                name=f"{gripper.name}_{view}",
            )
            for gripper in self.entry.grippers
        }
        self.gripper_sensors = {
            gripper.name: IsaacGripperSensor(prim, finger_joints=list(gripper.joints))
            for gripper in self.entry.grippers
        }
        self.joint_index: dict[str, int] = {}
        # Each finger joint's closed position and signed travel to fully
        # open, read from the articulation at setup; commanded opening
        # fractions scale onto it.
        self.spans: dict[str, list[FingerSpan]] = {}
        # Last force limit written per gripper, so the cap is not re-sent per tick.
        self.applied_effort: dict[str, float] = {}
        self.ready = False

    def exts(self) -> list:
        return [
            self.articulation,
            *self.arm_actuators.values(),
            *self.gripper_actuators.values(),
            *self.gripper_sensors.values(),
        ]

    def setup(self) -> bool:
        """Initialise every view against the live stage. Returns True once
        they are all reading; safe to call every step until it does."""
        if self.ready:
            return True
        if not all(ext.setup() for ext in self.exts()):
            return False
        self.joint_index = {
            name: index for index, name in enumerate(self.articulation.get_joint_names())
        }
        # Every joint of this robot has to be on its articulation: a limb
        # whose joints are missing takes no command and publishes no state, so
        # the engine refuses to run and names them.
        missing = sorted(
            {name for name in self.entry.joints() if name not in self.joint_index}
        )
        if missing:
            raise RuntimeError(
                f"the {self.entry.model} entry names joints not on the articulation of "
                f"'{self.robot.instance or self.robot.model}': {missing}"
            )
        limits = self.articulation.get_joint_limits()
        if limits is None:
            return False
        lower, upper = limits
        for gripper in self.entry.grippers:
            self.spans[gripper.name] = [
                finger_span(
                    name,
                    lower[self.joint_index[name]],
                    upper[self.joint_index[name]],
                    gripper.closed_at,
                )
                for name in gripper.joints
            ]
        self.ready = True
        return True

    def teardown(self) -> None:
        for ext in self.exts():
            ext.teardown()
        self.joint_index = {}
        self.spans = {}
        self.applied_effort = {}
        self.ready = False


class IsaacBridgeExtension:
    """Drives every robot on the stage from its own command stream and
    publishes the state of each.

    A robot's views cannot initialise until the stage has loaded and the
    timeline is playing, which races both the bridge's construction and every
    robot that joins later, so setup is retried on the steps that follow.
    """

    def __init__(
        self,
        world: World,
        io: "SimTopicIO",
        robots: Registry,
        objects,
        state_rate_hz: int,
        renders: bool,
    ) -> None:
        self._world = world
        self._io = io
        self._robots = robots
        # Captures the spawned objects' snapshot each state tick publishes.
        self._objects = objects
        # State publishes ride an absolute grid at state_rate_hz, evaluated once
        # per rendered frame: physics advances inside sim_app.update(), so a
        # frame is the finest cadence there is, and a request at or above the
        # frame rate publishes every frame.
        if state_rate_hz <= 0:
            raise ValueError(f"state_rate_hz must be positive, got {state_rate_hz}")
        self._state_pacer = FramePacer(state_rate_hz)
        # Whether the robots' cameras are rendered. A rig is its robot's
        # model's, mounted on each robot holding a camera pair: a robot with
        # no camera pair, or of a model with no camera, renders nothing and
        # pays for nothing.
        self._renders = renders
        self._camera_sensors: dict[str, IsaacCameraSensor] = {}
        self._limbs: dict[str, RobotLimbs] = {}
        self._standing_reported = False

    def bind(self) -> None:
        """Builds a view of every robot on the stage, and a camera rig on
        each robot holding a camera pair. The views read on the steps that
        follow, once the stage is playing again. Every caller lets go of the
        views it held first."""
        assert not self._limbs, "bind takes the stage up after unbind let it go"
        self._limbs = {
            robot.instance: RobotLimbs(robot) for robot in self._world.robots()
        }
        self._reconcile_rigs()

    def resolve(self, update, steps: int) -> None:
        """Initialises every robot's views on the live stage, running
        `update` until they all read or `steps` run out. A robot whose
        model names a joint its articulation lacks raises here, inside the
        stand that put it on the stage, so that stand takes it back out."""
        for _ in range(steps):
            if all(limbs.setup() for limbs in self._limbs.values()):
                return
            update()

    def _reconcile_rigs(self) -> None:
        """Mounts the rig of its model on each robot on the stage that holds
        a camera pair and drops the rig of one that holds none: a robot's
        relays pair in after it stands, and a relay that stops takes its pair
        with it."""
        if not self._renders:
            return
        rigged = self._io.camera_robots()
        on_stage = {robot.instance for robot in self._world.robots()}
        for instance, sensor in list(self._camera_sensors.items()):
            if instance not in rigged or instance not in on_stage:
                sensor.teardown()
                del self._camera_sensors[instance]
        for robot in self._world.robots():
            if robot.instance not in rigged or robot.instance in self._camera_sensors:
                continue
            if not robot.known.entry.cameras:
                continue
            self._camera_sensors[robot.instance] = IsaacCameraSensor(
                robot.instance, robot.prim(), robot.known, self._io
            )

    def unbind(self) -> None:
        """Drops every view. A view of a prim that is edited under it stops
        reading, so this runs before the stage changes, and the bind after
        the change takes the views again. The camera rigs stay: their render
        products ride the robots' prims, and the next bind drops the rigs of
        robots that left."""
        for limbs in self._limbs.values():
            limbs.teardown()
        self._limbs = {}
        gc.collect()

    @property
    def is_ready(self) -> bool:
        """Whether every robot on the stage is being read."""
        return bool(self._limbs) and all(limbs.ready for limbs in self._limbs.values())

    def step(self) -> None:
        """Physics has already advanced in sim_app.update(); apply the latest
        commands and, when the state grid is due, publish measured state."""
        # The timeline is Isaac's own clock, advanced by sim_app.update(), so
        # a stopped timeline stops advancing it. Recorded ahead of every
        # stamp of this step, the camera captures included, and ahead of the
        # setup gate: an object-state capture after a scene edit stamps from
        # it while the articulation views are being created again.
        self._io.record_engine_time(self._engine_time_s())
        ready = [limbs for limbs in self._limbs.values() if limbs.setup()]
        if not ready:
            return

        self._apply_commands(ready)
        for sensor in self._camera_sensors.values():
            if sensor.setup():
                sensor.step()

        if not self._state_pacer.take_if_due(time.monotonic()):
            return
        # The camera pairs are read once per state tick: a rig follows its
        # pair within a tick of it forming or dissolving.
        self._reconcile_rigs()
        # Published before the state it stamps; a stopped timeline stops the
        # domain's clock with it.
        self._io.publish_clock_tick()
        self._publish_state(ready)

    def _engine_time_s(self) -> float:
        """Isaac's simulated time, in seconds since the timeline started."""
        import omni.timeline  # pylint: disable=C0415

        return float(omni.timeline.get_timeline_interface().get_current_time())

    def _apply_commands(self, ready: list[RobotLimbs]) -> None:
        standing = self._robots.standing()
        for limbs in ready:
            if limbs.robot.instance in standing:
                self._apply(limbs, *self._setpoints(limbs))

    def _setpoints(self, limbs: RobotLimbs):
        """One robot's latest setpoints, from the pairs its backbone leads,
        limb by limb of its model. A limb nothing has commanded holds where
        it stands."""
        robot = limbs.robot.instance
        arms = []
        for arm in limbs.entry.arms:
            command = self._io.latest_arm_command(robot, arm.name)
            if command is None or len(command[0]) != len(arm.joints):
                arms.append(None)
                continue
            positions, velocities = command
            arms.append(
                ArmSetpoint(
                    positions=tuple(positions),
                    velocities=(
                        tuple(velocities) if len(velocities) == len(arm.joints) else ()
                    ),
                )
            )
        grippers = []
        for gripper in limbs.entry.grippers:
            command = self._io.latest_gripper_command(robot, gripper.name)
            grippers.append(
                None
                if command is None
                else GripperSetpoint(opening=command[0], max_effort=command[1])
            )
        return arms, grippers

    def _apply(self, limbs: RobotLimbs, arms, grippers) -> None:
        """Writes one robot's setpoints into its drives. A limb with no
        setpoint keeps the last target written for it."""
        for arm, setpoint in zip(limbs.entry.arms, arms):
            if setpoint is None:
                continue
            velocities = (
                dict(zip(arm.joints, setpoint.velocities)) if setpoint.velocities else None
            )
            limbs.arm_actuators[arm.name].write_targets(
                dict(zip(arm.joints, setpoint.positions)), velocities
            )

        for gripper, setpoint in zip(limbs.entry.grippers, grippers):
            if setpoint is None:
                continue
            actuator = limbs.gripper_actuators[gripper.name]
            # Re-applied only on change: the ceiling write is a model-wide
            # articulation call, not a per-tick target.
            if limbs.applied_effort.get(gripper.name) != setpoint.max_effort:
                # Recorded only once written, so a not-ready tick retries.
                if actuator.set_force_limit(list(gripper.joints), setpoint.max_effort):
                    limbs.applied_effort[gripper.name] = setpoint.max_effort
            # Map the opening fraction onto each finger's own span, so the same
            # command drives prismatic fingers, revolute ones and a single jaw.
            actuator.write_targets(
                {
                    name: span.position(setpoint.opening)
                    for name, span in zip(gripper.joints, limbs.spans[gripper.name])
                }
            )

    def _publish_state(self, ready: list[RobotLimbs]) -> None:
        standing = self._robots.standing()
        for limbs in ready:
            if limbs.robot.instance not in standing:
                continue
            states = limbs.articulation.get_joint_states()
            if states is None:
                continue
            positions, velocities = states
            arms = []
            for arm in limbs.entry.arms:
                rows = [limbs.joint_index[name] for name in arm.joints]
                arms.append(([positions[row] for row in rows], [velocities[row] for row in rows]))
            grippers = [
                (
                    self._gripper_state(limbs, gripper.name),
                    limbs.applied_effort.get(gripper.name, 0.0),
                )
                for gripper in limbs.entry.grippers
            ]
            self._publish_pairs(limbs, arms, grippers)

        # A full snapshot of the spawned objects rides the same tick; it is
        # the one get_object_states answers until the next capture.
        snapshot = self._objects.capture_object_states()
        if snapshot is not None:
            self._io.publish_object_states(snapshot)

    def _publish_pairs(self, limbs: RobotLimbs, arms, grippers) -> None:
        robot = limbs.robot.instance
        for arm, (positions, velocities) in zip(limbs.entry.arms, arms):
            self._io.publish_arm_states(robot, arm.name, positions, velocities)
        for gripper, (reading, _cap) in zip(limbs.entry.grippers, grippers):
            if reading is not None:
                self._io.publish_gripper_states(robot, gripper.name, reading[0])

    @staticmethod
    def _gripper_state(limbs: RobotLimbs, name: str):
        """One gripper's opening and the force its fingers are applying.
        Opening is the mean per-finger travel fraction, the inverse of the
        mapping a command takes."""
        reading = limbs.gripper_sensors[name].get_gripper_state()
        spans = limbs.spans.get(name, [])
        if not reading or len(reading["positions"]) != len(spans):
            return None
        fractions = [span.opening(q) for q, span in zip(reading["positions"], spans)]
        forces = reading["applied_forces"]
        return (
            sum(fractions) / len(fractions),
            sum(forces) / len(forces) if forces else 0.0,
        )

    def shutdown(self) -> None:
        logger.info("IsaacBridgeExtension shutting down.")
        for sensor in self._camera_sensors.values():
            sensor.teardown()
        self.unbind()
