#!/usr/bin/env python3
# pylint: disable=R0902,C0413
"""IsaacBridgeExtension owns the per-step bridge for the Isaac stage. Isaac's
sim_app.update() advances physics on the main thread; each step this
extension applies the latest setpoint of every robot's limbs, reads the
measured joint and gripper state, and (throttled to state_rate_hz) publishes
Isaac's own timeline clock followed by the state it stamps.

The robot the engine stands is commanded through the limb pairings and
publishes on them; a robot in a seat is commanded through its seat and its
state goes back as that seat's feedback. Transport is typed peppygen via
SimTopicIO and SeatIO; there is no JSON and no raw peppylib on the path.
"""
from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyjson5

from camera_common import CameraConfig, FramePacer
from seats import ArmSetpoint, GripperSetpoint, Registry
from world import Robot, World
from exts import IsaacActuatorCtrl, IsaacArticulation, IsaacCameraSensor, IsaacGripperSensor

if TYPE_CHECKING:  # the transport, which carries the node runtime with it
    from sim_topics import SimTopicIO

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "sim_bridge.json5"


def _finger_travel_from_range(joint_name: str, lo: float, hi: float) -> float:
    """Signed full-open travel of a finger joint from its limit range (prismatic
    meters or revolute radians; the right side's revolute fingers open toward
    negative angles). Closed (0) must lie within the range; the signed travel is
    lo + hi, which cancels any symmetric slack an importer added around the
    nominal 0..travel range (e.g. Isaac's mimic-joint margin)."""
    if not (lo <= 0.0 <= hi):
        raise RuntimeError(
            f"finger joint '{joint_name}' range ({lo}, {hi}) does not contain the"
            " closed pose (0)"
        )
    travel = lo + hi
    if abs(travel) <= 1e-9:
        raise RuntimeError(
            f"finger joint '{joint_name}' range ({lo}, {hi}) has no usable travel"
        )
    return travel


@dataclass(frozen=True)
class Limb:
    """One arm or gripper of a model: the name every array of the seat
    contract orders by, and the joints it moves in the robot's own stage."""

    name: str
    joints: tuple[str, ...]


@dataclass(frozen=True)
class Layout:
    """What a robot of this engine is made of, read from sim_bridge.json5.
    Every model the catalogue carries has these limbs under these names."""

    arms: tuple[Limb, ...]
    grippers: tuple[Limb, ...]
    arm_gains: dict
    gripper_gains: dict

    @staticmethod
    def read(path: Path = _CONFIG_PATH) -> "Layout":
        config = pyjson5.loads(path.read_text())
        return Layout(
            arms=tuple(
                Limb(name=arm["name"], joints=tuple(arm["joints"])) for arm in config["arms"]
            ),
            grippers=tuple(
                Limb(name=gripper["name"], joints=tuple(gripper["fingers"]))
                for gripper in config["grippers"]
            ),
            arm_gains=dict(config.get("arm_gains", {})),
            gripper_gains=dict(config.get("gripper_gains", {})),
        )

    def arm_names(self) -> list[str]:
        return [arm.name for arm in self.arms]

    def arm_joint_counts(self) -> list[int]:
        return [len(arm.joints) for arm in self.arms]

    def gripper_names(self) -> list[str]:
        return [gripper.name for gripper in self.grippers]

    def joints_of(self) -> list[str]:
        """Every joint a robot of this engine moves."""
        return [joint for limb in (*self.arms, *self.grippers) for joint in limb.joints]

    def arm_params(self, joints: tuple[str, ...]) -> dict:
        return {
            "joint_names": list(joints),
            "kp": list(self.arm_gains.get("kp", [])),
            "kd": list(self.arm_gains.get("kd", [])),
            "max_efforts": list(self.arm_gains.get("max_efforts", [])),
            "gravity_compensation": self.arm_gains.get("gravity_compensation", False),
        }

    def gripper_params(self, joints: tuple[str, ...]) -> dict:
        return {
            "joint_names": list(joints),
            "kp": list(self.gripper_gains.get("kp", [])),
            "kd": list(self.gripper_gains.get("kd", [])),
            "max_efforts": list(self.gripper_gains.get("max_efforts", [])),
        }


class RobotLimbs:
    """One robot's limbs on the stage: the view that reads its articulation,
    a controller per limb, and the sensor that reads each gripper. Isaac
    registers a view under a name, so every view here carries the name of the
    robot it belongs to."""

    def __init__(self, robot: Robot, layout: Layout) -> None:
        self.robot = robot
        self.layout = layout
        prim = robot.prim()
        view = robot.instance or "standing"
        self.articulation = IsaacArticulation(prim, name=f"articulation_{view}")
        # One actuator controller per limb: the MIT gains and torque caps are
        # applied to that limb's PhysX drives at setup, and the limb's
        # commands are written through it. Finger joints use explicit PhysX
        # position-drive gains so they hold their commanded opening.
        self.arm_actuators = {
            arm.name: IsaacActuatorCtrl(
                prim,
                joint_names=list(arm.joints),
                params=layout.arm_params(arm.joints),
                name=f"arm_{arm.name}_{view}",
            )
            for arm in layout.arms
        }
        self.gripper_actuators = {
            gripper.name: IsaacActuatorCtrl(
                prim,
                joint_names=list(gripper.joints),
                params=layout.gripper_params(gripper.joints),
                name=f"gripper_{gripper.name}_{view}",
            )
            for gripper in layout.grippers
        }
        self.gripper_sensors = {
            gripper.name: IsaacGripperSensor(prim, finger_joints=list(gripper.joints))
            for gripper in layout.grippers
        }
        self.joint_index: dict[str, int] = {}
        self.travels: dict[str, list[float]] = {}
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
            {name for name in self.layout.joints_of() if name not in self.joint_index}
        )
        if missing:
            raise RuntimeError(
                f"sim_bridge.json5 references joints not on the articulation of "
                f"'{self.robot.instance or self.robot.model}': {missing}"
            )
        limits = self.articulation.get_joint_limits()
        if limits is None:
            return False
        lower, upper = limits
        for gripper in self.layout.grippers:
            self.travels[gripper.name] = [
                _finger_travel_from_range(
                    name, lower[self.joint_index[name]], upper[self.joint_index[name]]
                )
                for name in gripper.joints
            ]
        self.ready = True
        return True

    def teardown(self) -> None:
        for ext in self.exts():
            ext.teardown()
        self.joint_index = {}
        self.travels = {}
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
        seats: Registry,
        seat_io,
        layout: Layout,
        state_rate_hz: int,
        cameras: list[CameraConfig],
    ) -> None:
        self._world = world
        self._io = io
        self._seats = seats
        self._seat_io = seat_io
        self._layout = layout
        # State publishes ride an absolute grid at state_rate_hz, evaluated once
        # per rendered frame: physics advances inside sim_app.update(), so a
        # frame is the finest cadence there is, and a request at or above the
        # frame rate publishes every frame.
        if state_rate_hz <= 0:
            raise ValueError(f"state_rate_hz must be positive, got {state_rate_hz}")
        self._state_pacer = FramePacer(state_rate_hz)
        # The rig mounts on the links of the robot the engine stands, which
        # the launch has already made a condition of rendering cameras.
        self._camera_sensor = (
            IsaacCameraSensor(world.standing().prim(), cameras, io) if cameras else None
        )
        self._limbs: dict[str, RobotLimbs] = {}
        self._standing_reported = False

    def bind(self) -> None:
        """Builds a view of every robot on the stage. The views read on the
        steps that follow, once the stage is playing again."""
        self._limbs = {
            robot.instance: RobotLimbs(robot, self._layout) for robot in self._world.robots()
        }

    def unbind(self) -> None:
        """Drops every view. A view of a prim that is edited under it stops
        reading, so this runs before the stage changes."""
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
        ready = [limbs for limbs in self._limbs.values() if limbs.setup()]
        if not ready:
            return

        self._apply_commands(ready)
        # The timeline is Isaac's own clock, advanced by sim_app.update(), so
        # a stopped timeline stops advancing it. Recorded ahead of every
        # stamp of this step, the camera captures included.
        self._io.record_engine_time(self._engine_time_s())
        if self._camera_sensor is not None and self._camera_sensor.setup():
            self._camera_sensor.step()

        if not self._state_pacer.take_if_due(time.monotonic()):
            return
        # Published before the state it stamps; a stopped timeline stops the
        # fleet's time with it.
        self._io.publish_sim_time()
        self._publish_state(ready)

    def _engine_time_s(self) -> float:
        """Isaac's simulated time, in seconds since the timeline started."""
        import omni.timeline  # pylint: disable=C0415

        return float(omni.timeline.get_timeline_interface().get_current_time())

    def _apply_commands(self, ready: list[RobotLimbs]) -> None:
        seats = self._seats.standing()
        for limbs in ready:
            seat = seats.get(limbs.robot.instance)
            if seat is None:
                self._apply(limbs, *self._pairing_setpoints(limbs))
                continue
            setpoints = seat.take()
            if setpoints is not None:
                self._apply(limbs, setpoints.arms, setpoints.grippers)

    def _pairing_setpoints(self, limbs: RobotLimbs):
        """The setpoints of the robot the engine stands, from the limb slots
        its relays lead. A slot nothing has commanded holds its limb where it
        stands, as an uncommanded seat does."""
        arms = []
        for arm in self._layout.arms:
            command = self._io.latest_arm_command(arm.name)
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
        for gripper in self._layout.grippers:
            command = self._io.latest_gripper_command(gripper.name)
            grippers.append(
                None
                if command is None
                else GripperSetpoint(opening=command[0], max_effort=command[1])
            )
        return arms, grippers

    def _apply(self, limbs: RobotLimbs, arms, grippers) -> None:
        """Writes one robot's setpoints into its drives. A limb with no
        setpoint keeps the last target written for it."""
        for arm, setpoint in zip(self._layout.arms, arms):
            if setpoint is None:
                continue
            velocities = (
                dict(zip(arm.joints, setpoint.velocities)) if setpoint.velocities else None
            )
            limbs.arm_actuators[arm.name].write_targets(
                dict(zip(arm.joints, setpoint.positions)), velocities
            )

        for gripper, setpoint in zip(self._layout.grippers, grippers):
            if setpoint is None:
                continue
            actuator = limbs.gripper_actuators[gripper.name]
            # Re-applied only on change: the ceiling write is a model-wide
            # articulation call, not a per-tick target.
            if limbs.applied_effort.get(gripper.name) != setpoint.max_effort:
                # Recorded only once written, so a not-ready tick retries.
                if actuator.set_force_limit(list(gripper.joints), setpoint.max_effort):
                    limbs.applied_effort[gripper.name] = setpoint.max_effort
            # Map the opening fraction onto each finger's own signed travel, so
            # the same command drives prismatic (v1) and revolute (v2) fingers.
            actuator.write_targets(
                {
                    name: travel * setpoint.opening
                    for name, travel in zip(gripper.joints, limbs.travels[gripper.name])
                }
            )

    def _publish_state(self, ready: list[RobotLimbs]) -> None:
        seats = self._seats.standing()
        for limbs in ready:
            states = limbs.articulation.get_joint_states()
            if states is None:
                continue
            positions, velocities = states
            arms = []
            for arm in self._layout.arms:
                rows = [limbs.joint_index[name] for name in arm.joints]
                arms.append(([positions[row] for row in rows], [velocities[row] for row in rows]))
            grippers = [
                self._gripper_state(limbs, gripper.name) for gripper in self._layout.grippers
            ]
            seat = seats.get(limbs.robot.instance)
            if seat is None:
                self._publish_pairings(arms, grippers)
            elif self._seat_io is not None:
                self._seat_io.publish_state(seat, self._io.timestamp_s(), arms, grippers)

    def _publish_pairings(self, arms, grippers) -> None:
        for arm, (positions, velocities) in zip(self._layout.arms, arms):
            self._io.publish_arm_states(arm.name, positions, velocities)
        for gripper, reading in zip(self._layout.grippers, grippers):
            if reading is not None:
                self._io.publish_gripper_states(gripper.name, reading[0])

    @staticmethod
    def _gripper_state(limbs: RobotLimbs, name: str):
        """One gripper's opening and the force its fingers are applying.
        Opening is the mean per-finger travel fraction, the inverse of the
        mapping a command takes."""
        reading = limbs.gripper_sensors[name].get_gripper_state()
        travels = limbs.travels.get(name, [])
        if not reading or len(reading["positions"]) != len(travels):
            return None
        fractions = [q / travel for q, travel in zip(reading["positions"], travels)]
        forces = reading["applied_forces"]
        return (
            sum(fractions) / len(fractions),
            sum(forces) / len(forces) if forces else 0.0,
        )

    def shutdown(self) -> None:
        logger.info("IsaacBridgeExtension shutting down.")
        if self._camera_sensor is not None:
            self._camera_sensor.teardown()
        self.unbind()
