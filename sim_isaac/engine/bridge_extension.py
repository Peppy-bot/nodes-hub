#!/usr/bin/env python3
# pylint: disable=R0902,C0413
"""IsaacBridgeExtension owns the per-step bridge for the Isaac stage. Isaac's
sim_app.update() advances physics on the main thread; each step this
extension applies the latest setpoint of every robot's limbs and base, runs
the rate loop of each wheel (wheel_loop), records where the wheels carried
each robot that rolls, reads the measured joint and gripper state, and
(throttled to state_rate_hz) publishes Isaac's own timeline clock followed
by the state it stamps: each robot's joint, wheel and gripper states and a
snapshot of every spawned object and standing robot.

Every robot is commanded through its own limb and base pairs and publishes
its state back on them; the copy a pair carries is the robot it belongs to,
and the parts it is driven by are its own model's. Transport is typed
peppygen via SimTopicIO; there is no JSON and no raw peppylib on the path.
"""
from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from robot_description_py import FingerSpan
from sim_robot_core.cameras import FramePacer
from sim_robot_core.registry import Registry
from sim_robot_core.setpoints import JointSetpoint
from sim_robot_core.world import rolled_placement

from wheel_loop import WheelLoop
from world import Robot, World
from exts import IsaacActuatorCtrl, IsaacArticulation, IsaacCameraSensor, IsaacGripperSensor

if TYPE_CHECKING:  # the transport, which carries the node runtime with it
    from sim_topics import SimTopicIO

logger = logging.getLogger(__name__)


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
        # One actuator controller per limb: the gains of the model's entry,
        # each drive capped at its joint's effort limit in the description,
        # are applied to that limb's PhysX drives at setup, and the limb's
        # commands are written through it. A gripper's controller drives its
        # fingers that move of their own; every other finger follows one of
        # them through the coupling the stage authors as the robot joins. A
        # limb whose model carries no gains keeps the drives its stage
        # authors.
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
                joint_names=list(gripper.driven_joints()),
                params=known.gripper_params(gripper),
                name=f"{gripper.name}_{view}",
            )
            for gripper in self.entry.grippers
        }
        self.gripper_sensors = {
            gripper.name: IsaacGripperSensor(prim, finger_joints=list(gripper.joints))
            for gripper in self.entry.grippers
        }
        # The controller of the base's wheels, each on the velocity drive its
        # stage authors, which it keeps; none for a robot that stands where
        # it is placed.
        base = self.entry.base
        self.base_actuator: Optional[IsaacActuatorCtrl] = (
            None
            if base is None
            else IsaacActuatorCtrl(
                prim,
                joint_names=list(base.wheels),
                params={"joint_names": list(base.wheels), "kp": [], "kd": [], "max_efforts": []},
                name=f"{base.name}_{view}",
            )
        )
        self.joint_index: dict[str, int] = {}
        # Where the gripper's closed and open states put each finger joint,
        # as the robot's description gives them; measured positions read
        # back as an opening fraction through it.
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
            *([] if self.base_actuator is None else [self.base_actuator]),
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
        for gripper in self.entry.grippers:
            self.spans[gripper.name] = list(gripper.spans)
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
        # The rate loop of each wheel of each robot that rolls, left then
        # right, kept while the stage is edited so the angle a wheel reports
        # runs on.
        self._wheels: dict[str, tuple[WheelLoop, ...]] = {}
        # The engine time of the last step (s), which a wheel's loop sums its
        # rate over.
        self._last_step_s: Optional[float] = None
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
        self._wheels = {
            name: loops for name, loops in self._wheels.items() if name in self._limbs
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

    def hold_through_stop(self) -> None:
        """Readies the robots that roll for a timeline that stops, which sets
        every body and joint back on its stage: each is stood where its
        wheels carried it, so the stage holds it there, and its wheels' loops
        read the stage afresh once it plays again."""
        self._world.stand_where_rolled()
        self._wheels = {
            name: tuple(loop.let_go() for loop in loops) for name, loops in self._wheels.items()
        }

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
        now_s = self._engine_time_s()
        self._io.record_engine_time(now_s)
        elapsed_s = 0.0 if self._last_step_s is None else now_s - self._last_step_s
        self._last_step_s = now_s
        ready = [limbs for limbs in self._limbs.values() if limbs.setup()]
        if not ready:
            return

        self._set_down_moved()
        self._apply_commands(ready, elapsed_s)
        # Every step, so each capture and listing is of the instant it is
        # stamped with.
        self._roll(ready)
        for sensor in self._camera_sensors.values():
            if sensor.setup():
                sensor.step()

        if not self._state_pacer.take_if_due(time.monotonic()):
            return
        # The camera and base pairs are read once per state tick: a rig
        # follows its pair within a tick of it forming or dissolving, and a
        # base whose pair is gone holds its wheels at rate 0.
        self._io.keep_base_pairs()
        self._reconcile_rigs()
        # Published before the state it stamps; a stopped timeline stops the
        # domain's clock with it.
        self._io.publish_clock_tick()
        self._publish_state(ready)

    def _engine_time_s(self) -> float:
        """Isaac's simulated time, in seconds since the timeline started."""
        import omni.timeline  # pylint: disable=C0415

        return float(omni.timeline.get_timeline_interface().get_current_time())

    def _apply_commands(self, ready: list[RobotLimbs], elapsed_s: float) -> None:
        standing = self._robots.standing()
        for limbs in ready:
            if limbs.robot.instance not in standing:
                continue
            self._apply(limbs, *self._setpoints(limbs))
            if limbs.base_actuator is not None:
                self._drive_base(limbs, elapsed_s)

    def _drive_base(self, limbs: RobotLimbs, elapsed_s: float) -> None:
        """Runs each wheel's rate loop over the `elapsed_s` since the last
        step and writes its drive's target velocity: the rate of the latest
        setpoint of the robot's base, or rate 0 while it has none."""
        robot = limbs.robot.instance
        wheels = limbs.entry.base.wheels
        gains = limbs.robot.wheel_loops
        states = limbs.articulation.get_joint_states()
        if states is None:
            return
        positions, _ = states
        setpoint = self._io.latest_base_command(robot)
        rates = (0.0,) * len(wheels) if setpoint is None else setpoint.rates()
        loops = tuple(
            loop.advanced(positions[limbs.joint_index[wheel]], elapsed_s, gain).driven_at(rate)
            for loop, wheel, rate, gain in zip(
                self._wheels.get(robot, (WheelLoop(),) * len(wheels)), wheels, rates, gains, strict=True
            )
        )
        self._wheels[robot] = loops
        limbs.base_actuator.write_velocity_targets(
            {
                wheel: loop.velocity_target(gain)
                for wheel, loop, gain in zip(wheels, loops, gains, strict=True)
            }
        )

    def _roll(self, ready: list[RobotLimbs]) -> None:
        """Records where the wheels carried each robot that rolls, its robot
        frame as the articulation's root reads it, and how fast it moves."""
        for limbs in ready:
            if not limbs.robot.known.rolls:
                continue
            state = limbs.articulation.get_root_state()
            if state is None:
                continue
            position, orientation, linear, angular = state
            self._world.roll(
                limbs.robot.instance, rolled_placement(position, orientation), linear, angular
            )

    def _set_down_moved(self) -> None:
        """Sets each robot that rolls and was moved since the last step down
        at rest where the move stood it: its root and its joints still, the
        integral of each wheel's loop cleared, and its base's setpoint
        dropped, so its wheels hold at rate 0 until its leader sends again."""
        for name in self._world.take_set_down():
            limbs = self._limbs.get(name)
            robot = next((one for one in self._world.robots() if one.instance == name), None)
            if limbs is None or robot is None:
                continue
            limbs.articulation.set_down_at(
                robot.placement.position.position, robot.placement.orientation.quaternion
            )
            self._wheels[name] = tuple(loop.let_go() for loop in self._wheels.get(name, ()))
            self._io.drop_base_command(name)
            logger.info("robot '%s' is set down at rest where it was moved", name)

    def _setpoints(self, limbs: RobotLimbs):
        """One robot's latest setpoints, from the pairs its backbone leads,
        limb by limb of its model. A limb nothing has commanded holds where
        it stands."""
        robot = limbs.robot.instance
        arms: list[Optional[JointSetpoint]] = [
            self._io.latest_arm_command(robot, arm.name) for arm in limbs.entry.arms
        ]
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
                if actuator.set_force_limit(list(gripper.driven_joints()), setpoint.max_effort):
                    limbs.applied_effort[gripper.name] = setpoint.max_effort
            # Map the opening fraction onto each driven finger's own span, so
            # the same command drives prismatic fingers, revolute ones and a
            # single jaw.
            actuator.write_targets(
                {name: span.position_at(setpoint.opening) for name, span in gripper.driven()}
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
            self._publish_base(limbs, velocities)

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

    def _publish_base(self, limbs: RobotLimbs, velocities: list[float]) -> None:
        """The wheels' angles, as their loops count them, and their measured
        rates, back on the base's pair. A wheel whose loop has not read the
        stage yet reports nothing."""
        base = limbs.entry.base
        loops = self._wheels.get(limbs.robot.instance)
        if base is None or loops is None or any(loop.read is None for loop in loops):
            return
        self._io.publish_base_states(
            limbs.robot.instance,
            base.name,
            [loop.angle for loop in loops],
            [velocities[limbs.joint_index[wheel]] for wheel in base.wheels],
        )

    @staticmethod
    def _gripper_state(limbs: RobotLimbs, name: str):
        """One gripper's opening and the force its fingers are applying.
        Opening is the mean per-finger travel fraction, the inverse of the
        mapping a command takes."""
        reading = limbs.gripper_sensors[name].get_gripper_state()
        spans = limbs.spans.get(name, [])
        if not reading or len(reading["positions"]) != len(spans):
            return None
        fractions = [span.opening_at(q) for q, span in zip(reading["positions"], spans)]
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
