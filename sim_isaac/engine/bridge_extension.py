#!/usr/bin/env python3
# pylint: disable=R0902,C0413
"""IsaacBridgeExtension owns the per-step bridge for the Isaac stage. Isaac's
sim_app.update() advances physics on the main thread; each step this
extension applies the latest setpoint of every robot's limbs and base, runs
the rate loop of each wheel (wheel_loop), records where the wheels carried
each robot that rolls, reads the measured joint and gripper state, and
(throttled to state_rate_hz) publishes the engine clock, the time Isaac's
timeline played (`_engine_time_s`), followed by the state it stamps: each
robot's joint, wheel and gripper states and a snapshot of every spawned
object and standing robot.

Every robot is commanded through its own limb and base pairs and publishes
its state back on them; the copy a pair carries is the robot it belongs to,
and the parts it is driven by are its own model's. A base's wheels run at the
rates of its pair's latest setpoint, and at rate 0 while it holds none; once
a reading of the pairs finds that pair gone, or the setpoint outlives its
leader's window on the engine's clock, they are held where they stand, the
integral of each wheel's loop cleared (sim_robot_core.setpoints). Transport is
typed peppygen via SimTopicIO; there is no JSON and no raw peppylib on the
path.
"""
from __future__ import annotations

import gc
import logging
import time
from typing import TYPE_CHECKING, Callable, Optional

from control_core_py.geometry import Pose
from robot_description_py import FingerSpan
from sim_robot_core.cameras import FramePacer
from sim_robot_core.registry import Registry
from sim_robot_core.setpoints import GripperSetpoint, JointSetpoint

from wheel_loop import WheelLoop
from world import Robot, World
from exts import IsaacActuatorCtrl, IsaacArticulation, IsaacCameraSensor, IsaacGripperSensor

if TYPE_CHECKING:  # the transport, which carries the node runtime with it
    from sim_topics import SimTopicIO

logger = logging.getLogger(__name__)


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
        # The time the timeline played before its last stop, and its last
        # read since (s), None until a step reads it: the engine clock is
        # their sum.
        self._timeline_base_s = 0.0
        self._timeline_read_s: Optional[float] = None
        self._standing_reported = False
        # Whether a bind took views that no resolve has read since
        # (views_to_read).
        self._views_to_read = False

    def bind(self) -> None:
        """Builds a view of every robot on the stage, and a camera rig on
        each robot holding a camera pair. The rigs follow their pairs before
        the views are built: a rig's drop removes its render product's prim,
        and Isaac's Articulation drops its PhysX view on the removal of any
        prim. The views read once resolve runs, or on the steps that follow,
        once the stage is playing again. Every caller lets go of the views it
        held first."""
        assert not self._limbs, "bind takes the stage up after unbind let it go"
        self._drop_rigs(self._rigs_to_drop())
        self._mount_rigs()
        self._limbs = {
            robot.instance: RobotLimbs(robot) for robot in self._world.robots()
        }
        self._wheels = {
            name: loops for name, loops in self._wheels.items() if name in self._limbs
        }
        self._views_to_read = True

    def resolve(self, update, steps: int) -> None:
        """Initialises every robot's views on the live stage, running
        `update` until they all read or `steps` run out. A robot whose
        model names a joint its articulation lacks raises here, inside the
        stand that put it on the stage, so that stand takes it back out."""
        self._views_to_read = False
        for _ in range(steps):
            if all(limbs.setup() for limbs in self._limbs.values()):
                return
            update()

    @property
    def views_to_read(self) -> bool:
        """Whether a bind took views that no resolve has read since."""
        return self._views_to_read

    def remove_from_stage(self, remove: Callable[[], None]) -> None:
        """Runs `remove`, which takes prims off the live stage, between
        letting go of every view and taking them again, and drops the object
        reader's rigid-body view, which the next capture creates again.
        Isaac's Articulation drops its PhysX view on the removal of any prim,
        its own or another's. No update runs inside it; the loop resolves the
        views before its next step (views_to_read). A robot that joins or
        leaves lets go of the views around a stopped timeline
        (SimLauncher.unbind and rebind)."""
        self.unbind()
        remove()
        self.bind()
        self._objects.invalidate_physics_views()

    def _reconcile_rigs(self) -> None:
        """Mounts the rig of its model on each robot on the stage that holds
        a camera pair and drops the rig of one that holds none: a robot's
        relays pair in after it stands, and a relay that stops takes its pair
        with it. A drop is a removal (remove_from_stage)."""
        dropped = self._rigs_to_drop()
        if dropped:
            self.remove_from_stage(lambda: self._drop_rigs(dropped))
        self._mount_rigs()

    def _rigs_to_drop(self) -> list[str]:
        """The robots whose rig stands while they hold no camera pair or
        have left the stage."""
        if not self._camera_sensors:
            return []
        rigged = self._io.camera_robots()
        on_stage = {robot.instance for robot in self._world.robots()}
        return [
            instance for instance in self._camera_sensors
            if instance not in rigged or instance not in on_stage
        ]

    def _drop_rigs(self, instances: list[str]) -> None:
        for instance in instances:
            self._camera_sensors.pop(instance).teardown()

    def _mount_rigs(self) -> None:
        """Mounts the rig of its model on each robot on the stage that holds
        a camera pair and has none."""
        if not self._renders:
            return
        rigged = self._io.camera_robots()
        for robot in self._world.robots():
            if robot.instance not in rigged or robot.instance in self._camera_sensors:
                continue
            if not robot.known.entry.cameras:
                continue
            self._camera_sensors[robot.instance] = IsaacCameraSensor(
                robot.instance, robot.prim(), robot.known, self._io
            )

    def hold_through_stop(self) -> None:
        """Readies the robots that roll and the engine clock for a timeline
        that stops, which sets every body and joint back on its stage and its
        time back to its start: each robot that rolls is stood where its
        wheels carried it, so the stage holds it there, its wheels' loops read
        the stage afresh once it plays again, and the time the timeline played
        is added to the time it played before each stop."""
        self._world.stand_where_rolled()
        self._wheels = {
            name: tuple(loop.let_go() for loop in loops) for name, loops in self._wheels.items()
        }
        self._timeline_base_s += self._timeline_time_s()
        self._timeline_read_s = None

    def unbind(self) -> None:
        """Drops every view. A view of a prim that is edited under it stops
        reading, so this runs before the stage changes, and the bind after
        the change takes the views again. The camera rigs stay: their render
        products ride the robots' prims, and the next bind drops the rigs of
        robots that left or hold no camera pair."""
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
        # The engine clock runs on the timeline, which sim_app.update()
        # advances, so a stopped timeline stops it. Recorded ahead of every
        # stamp of this step, the camera captures included, and ahead of the
        # setup gate.
        now_s = self._engine_time_s()
        self._io.record_engine_time(now_s)
        elapsed_s = 0.0 if self._last_step_s is None else now_s - self._last_step_s
        self._last_step_s = now_s
        ready = [limbs for limbs in self._limbs.values() if limbs.setup()]
        if not ready:
            # No robot stands, or no view an edit took again reads yet: the
            # world records this step's instant with each robot's record as
            # it stands, so an object-state capture now is stamped with the
            # instant it reads its objects at.
            self._world.roll(self._io.timestamp_s(), {})
            return

        self._set_down_moved(ready)
        self._apply_commands(ready, now_s, elapsed_s)
        # Every step, so each capture and listing is of the instant it is
        # stamped with.
        self._roll(ready, self._io.timestamp_s())
        for sensor in self._camera_sensors.values():
            if sensor.setup():
                sensor.step()

        if not self._state_pacer.take_if_due(time.monotonic()):
            return
        # The camera and base pairs are read on each state tick: a base
        # whose pair is gone holds its wheels where they stand, and a rig
        # follows its pair within a tick of it forming or dissolving.
        self._io.setpoints.keep_paired_bases()
        # Published before the state it stamps; a stopped timeline stops the
        # domain's clock with it.
        self._io.publish_clock_tick()
        self._publish_state(ready)
        # After the publish: a rig's drop lets every view go and takes them
        # again, and the loop resolves them before the next step.
        self._reconcile_rigs()

    def _engine_time_s(self) -> float:
        """Isaac's simulated time, in seconds since the engine started: the
        time the timeline played before each stop, plus where it reads now.
        The engine stops the timeline for each robot that joins or leaves,
        once hold_through_stop added the time it played, and plays it again
        before the next step. A read earlier than the last one, a stop
        the engine did not make, adds the last read to the time played
        before."""
        read_s = self._timeline_time_s()
        if self._timeline_read_s is not None and read_s < self._timeline_read_s:
            self._timeline_base_s += self._timeline_read_s
        self._timeline_read_s = read_s
        return self._timeline_base_s + read_s

    @staticmethod
    def _timeline_time_s() -> float:
        """Where Isaac's timeline reads now (s)."""
        import omni.timeline  # pylint: disable=C0415

        return float(omni.timeline.get_timeline_interface().get_current_time())

    def _apply_commands(self, ready: list[RobotLimbs], now_s: float, elapsed_s: float) -> None:
        standing = self._robots.standing()
        for limbs in ready:
            if limbs.robot.instance not in standing:
                continue
            self._apply(limbs, *self._setpoints(limbs))
            if limbs.base_actuator is not None:
                self._drive_base(limbs, now_s, elapsed_s)

    def _drive_base(self, limbs: RobotLimbs, now_s: float, elapsed_s: float) -> None:
        """Runs each wheel's rate loop over the `elapsed_s` since the last
        step and writes its drive's target velocity: the rate of the setpoint
        the robot's base follows at `now_s`, or rate 0 while it has none.
        Wheels released since the last step are held where they stand: each
        loop's integral cleared, anchored at its read of this step."""
        robot = limbs.robot.instance
        wheels = limbs.entry.base.wheels
        gains = limbs.robot.wheel_loops
        states = limbs.articulation.get_joint_states()
        if states is None:
            return
        positions, _ = states
        setpoint = self._io.setpoints.base(robot, now_s)
        if self._io.setpoints.take_base_release(robot):
            self._let_go_of_wheels(robot)
        rates = (0.0,) * len(wheels) if setpoint is None else setpoint.velocities
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

    def _roll(self, ready: list[RobotLimbs], stamp_s: float) -> None:
        """Records the step the engine stamps `stamp_s`: where the wheels
        carried each robot that rolls, its robot frame as the articulation's
        root reads it, and how fast it moves."""
        rolled = {}
        for limbs in ready:
            if not limbs.robot.known.rolls:
                continue
            state = limbs.articulation.get_root_state()
            if state is None:
                continue
            position, orientation, linear, angular = state
            rolled[limbs.robot.instance] = (Pose.parse(position, orientation), linear, angular)
        self._world.roll(stamp_s, rolled)

    def _set_down_moved(self, ready: list[RobotLimbs]) -> None:
        """Sets each robot of `ready` that rolls and was moved since it was
        last set down at rest where the move stood it: its root and its joints
        still, the integral of each wheel's loop cleared, and its base's
        setpoint dropped, so its wheels hold at rate 0 until its leader sends
        again. A moved robot whose views do not read yet is set down by the
        first step that reads them; one whose set-down is not written
        (IsaacArticulation.set_down_at) keeps its loops and its setpoint."""
        reading = {limbs.robot.instance: limbs for limbs in ready}
        for name in self._world.take_set_down(set(reading)):
            limbs = reading[name]
            robot = next((one for one in self._world.robots() if one.instance == name), None)
            if robot is None:
                continue
            if not limbs.articulation.set_down_at(
                robot.placement.position.position, robot.placement.orientation.xyzw
            ):
                continue
            self._let_go_of_wheels(name)
            self._io.setpoints.drop_base(name)
            logger.info("robot '%s' is set down at rest where it was moved", name)

    def _let_go_of_wheels(self, robot: str) -> None:
        """Clears the integral of each wheel's loop of `robot`, the loop
        anchored afresh at its next read. A robot whose wheels no step read
        yet has no loop to clear."""
        loops = self._wheels.get(robot)
        if loops is not None:
            self._wheels[robot] = tuple(loop.let_go() for loop in loops)

    def _setpoints(self, limbs: RobotLimbs):
        """One robot's latest setpoints, from the pairs its backbone leads,
        limb by limb of its model. A limb nothing has commanded holds where
        it stands."""
        robot = limbs.robot.instance
        setpoints = self._io.setpoints
        arms: list[Optional[JointSetpoint]] = [
            setpoints.arm(robot, arm.name) for arm in limbs.entry.arms
        ]
        grippers: list[Optional[GripperSetpoint]] = [
            setpoints.gripper(robot, gripper.name) for gripper in limbs.entry.grippers
        ]
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
