#!/usr/bin/env python3
# pylint: disable=C0413
"""MujocoBridgeExtension owns the physics tick of the scene. Each step it
applies the latest setpoint of every limb of every robot standing, advances
physics, and (throttled to state_rate_hz) publishes MuJoCo's own clock
followed by the measured joint and gripper state it stamps.

Every robot is driven through its own limb pairs and publishes its state
back on them. A robot's joints, actuators and cameras carry its own prefix in
the scene, which is what tells one robot's limbs from another's when two
robots of the same model stand together. Transport is typed peppygen via
SimTopicIO; there is no JSON and no raw peppylib on the path.
"""
from __future__ import annotations

import logging
import time

from sim_robot_core.cameras import FramePacer
from sim_robot_core.models import FingerSpan, finger_span

from exts import (
    MujocoActuatorCtrl,
    MujocoArticulation,
    MujocoCameraSensor,
    MujocoGripperSensor,
    home_inertia,
)
from exts.camera_sensor import Rig
from sim_topics import SimTopicIO
from world import Robot

logger = logging.getLogger(__name__)


class _RobotLimbs:
    """One standing robot's own views of the scene: the actuators its limbs
    are driven through, the sensor each gripper is read from, the span each
    finger travels, and where its joints sit in the scene's state."""

    def __init__(self, robot: Robot, model, data) -> None:
        self.robot = robot
        self.entry = robot.known.entry
        self._model = model
        self._data = data
        # One actuator controller per robot: it resolves that robot's
        # actuators by their scene names, applies the model's MIT gains to
        # its arm joints when its entry carries some, and leaves every other
        # actuator on its MJCF defaults.
        self.actuator = MujocoActuatorCtrl(model, data, params=self._gains())
        self.gripper_sensors: dict[str, MujocoGripperSensor] = {}
        # Each finger joint's closed position and signed travel to fully
        # open, read from the model at setup; commanded opening fractions
        # scale onto it.
        self.finger_spans: dict[str, list[FingerSpan]] = {}
        # Last force limit written per gripper, so the cap is not re-sent per
        # tick.
        self.applied_effort: dict[str, float] = {}
        # Where each joint of this robot sits in the scene's joint order.
        self.joint_index: dict[str, int] = {}

    def scene_name(self, joint: str) -> str:
        return self.robot.scene_name(joint)

    def _gains(self) -> dict:
        params = self.robot.known.actuator_params()
        return {
            **params,
            "joint_names": [self.scene_name(joint) for joint in params["joint_names"]],
        }

    def setup(self, joint_index: dict[str, int], inertia: "list[float]") -> None:
        """Resolves this robot in the scene just compiled. A joint its entry
        names that the scene lacks is refused here, with its name."""
        missing = sorted(
            {
                joint
                for joint in self.entry.joints()
                if self.scene_name(joint) not in joint_index
            }
        )
        if missing:
            raise RuntimeError(
                f"the {self.entry.model} entry names joints not in the scene standing "
                f"'{self.robot.instance}': {missing}"
            )
        self.joint_index = {
            joint: joint_index[self.scene_name(joint)] for joint in self.entry.joints()
        }
        if not self.actuator.setup(inertia):
            raise RuntimeError(f"MujocoActuatorCtrl setup failed for '{self.robot.instance}'")
        self.actuator.require_force_limited(
            [self.scene_name(joint) for joint in self.entry.finger_joints()]
        )
        for gripper in self.entry.grippers:
            fingers = [self.scene_name(joint) for joint in gripper.joints]
            sensor = MujocoGripperSensor(self._model, self._data, finger_joints=fingers)
            if not sensor.setup():
                raise RuntimeError(
                    f"MujocoGripperSensor setup failed for gripper '{gripper.name}' of "
                    f"'{self.robot.instance}'"
                )
            self.gripper_sensors[gripper.name] = sensor
            self.finger_spans[gripper.name] = [
                self._finger_span(joint, gripper.closed_at) for joint in fingers
            ]

    def _finger_span(self, scene_joint: str, closed_at: str) -> FingerSpan:
        import mujoco  # pylint: disable=C0415

        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, scene_joint)
        lower, upper = (float(value) for value in self._model.jnt_range[jid])
        return finger_span(scene_joint, lower, upper, closed_at)

    def start_in_posture(self) -> None:
        """Puts this robot in the posture its model starts in, and holds it
        there: the joints are placed and their actuators target the same
        positions, so the arm stands still until its first setpoint. A model
        whose entry names no posture starts where its file puts it."""
        import mujoco  # pylint: disable=C0415

        posture = self.entry.start_posture
        if not posture:
            return
        placed = {self.scene_name(joint): position for joint, position in posture.items()}
        for joint, position in placed.items():
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            self._data.qpos[self._model.jnt_qposadr[jid]] = position
        self.actuator.write_targets(placed)

    def teardown(self) -> None:
        self.actuator.teardown()
        for sensor in self.gripper_sensors.values():
            sensor.teardown()


class MujocoBridgeExtension:
    """Drives every robot standing in the scene from its own command streams
    and publishes its state on its own pairs."""

    def __init__(
        self,
        model,
        data,
        io: SimTopicIO,
        robots: "list[Robot]",
        state_rate_hz: int,
        renders: bool,
        time_base_s: float,
        camera_counters=None,
    ) -> None:
        self._model = model
        self._data = data
        self._io = io
        self._robots = list(robots)
        # State publishes ride an absolute grid at state_rate_hz: serializing
        # every reader at the ~500 Hz physics tick saturates the single sim
        # thread. Writers and the physics step still run every tick.
        if state_rate_hz <= 0:
            raise ValueError(f"state_rate_hz must be positive, got {state_rate_hz}")
        self._state_pacer = FramePacer(state_rate_hz)
        # The engine clock runs on from where the previous scene left it, so
        # the fleet's time never goes back when a robot joins or leaves.
        self._time_base_s = time_base_s
        self._articulation = MujocoArticulation(model, data)
        self._limbs: dict[str, _RobotLimbs] = {
            robot.instance: _RobotLimbs(robot, model, data) for robot in self._robots
        }
        # The cameras of every robot whose model carries a rig, rendered off
        # this thread when the engine renders at all.
        self._camera_sensor = (
            MujocoCameraSensor(model, self._rigs(), io, camera_counters)
            if renders and self._rigs()
            else None
        )

    def _rigs(self) -> "list[Rig]":
        return [
            Rig(
                robot=robot.instance,
                prefix=robot.prefix,
                cameras=tuple(robot.known.entry.cameras),
            )
            for robot in self._robots
            if robot.known.entry.cameras
        ]

    def startup(self, posture_for: "list[Robot]" = ()) -> None:
        """Resolves every robot in the scene and starts rendering. The robots
        of `posture_for` are put in the posture their model starts in; the
        others carried their state onto this scene and keep it."""
        if not self._articulation.setup():
            raise RuntimeError("MujocoArticulation setup failed")
        joint_index = {
            name: index for index, name in enumerate(self._articulation.get_joint_names())
        }
        inertia = home_inertia(self._model)
        for limbs in self._limbs.values():
            limbs.setup(joint_index, inertia)
        for robot in posture_for:
            self._limbs[robot.instance].start_in_posture()
        if posture_for:
            import mujoco  # pylint: disable=C0415

            mujoco.mj_forward(self._model, self._data)
        if self._camera_sensor is not None:
            self._camera_sensor.start()
        logger.info(
            "MujocoBridgeExtension ready for %s",
            ", ".join(
                f"'{limbs.robot.instance}' ({limbs.entry.model}) with "
                f"{len(limbs.entry.arms)} arm(s), {len(limbs.entry.grippers)} gripper(s)"
                for limbs in self._limbs.values()
            )
            or "an empty scene",
        )

    def engine_time_s(self) -> float:
        """The engine clock: the scenes before this one, plus MuJoCo's own
        time in this one, which mj_step advances."""
        return self._time_base_s + float(self._data.time)

    def step(self) -> None:
        import mujoco  # pylint: disable=C0415

        self._apply_commands()
        mujoco.mj_step(self._model, self._data)
        # `data.time` is MuJoCo's own clock, advanced by mj_step above, so a
        # paused engine stops advancing it. Recorded ahead of every stamp of
        # this step, the camera snapshot included.
        self._io.record_engine_time(self.engine_time_s())
        if self._camera_sensor is not None:
            self._camera_sensor.snapshot(self._data.qpos)
            self._camera_sensor.raise_if_failed()

        if not self._state_pacer.take_if_due(time.monotonic()):
            return
        # Published before the state it stamps; a stopped engine stops the
        # domain's clock with it.
        self._io.publish_clock_tick()
        self._publish_state()

    def _apply_commands(self) -> None:
        for limbs in self._limbs.values():
            robot = limbs.robot.instance
            for arm in limbs.entry.arms:
                command = self._io.latest_arm_command(robot, arm.name)
                if command is None:
                    continue
                positions, velocities = command
                if len(positions) != len(arm.joints):
                    continue
                joints = [limbs.scene_name(joint) for joint in arm.joints]
                velocity_values = (
                    dict(zip(joints, velocities)) if len(velocities) == len(joints) else None
                )
                limbs.actuator.write_targets(dict(zip(joints, positions)), velocity_values)

            for gripper in limbs.entry.grippers:
                command = self._io.latest_gripper_command(robot, gripper.name)
                if command is None:
                    continue
                opening, max_effort = command
                joints = [limbs.scene_name(joint) for joint in gripper.joints]
                # Re-applied only on change: the cap is a model write, not a
                # per-tick target.
                if limbs.applied_effort.get(gripper.name) != max_effort:
                    # Recorded only once written, so a not-ready tick retries.
                    if limbs.actuator.set_force_limit(joints, max_effort):
                        limbs.applied_effort[gripper.name] = max_effort
                # Map the opening fraction onto each finger's own span, so the
                # same command drives prismatic fingers, revolute ones and a
                # single jaw.
                spans = limbs.finger_spans[gripper.name]
                limbs.actuator.write_targets(
                    {name: span.position(opening) for name, span in zip(joints, spans)}
                )

    def _publish_state(self) -> None:
        states = self._articulation.get_joint_states()
        for limbs in self._limbs.values():
            robot = limbs.robot.instance
            if states is not None:
                positions, velocities = states
                for arm in limbs.entry.arms:
                    indices = [limbs.joint_index[joint] for joint in arm.joints]
                    self._io.publish_arm_states(
                        robot,
                        arm.name,
                        [positions[index] for index in indices],
                        [velocities[index] for index in indices],
                    )

            for name, sensor in limbs.gripper_sensors.items():
                data = sensor.get_gripper_state()
                spans = limbs.finger_spans[name]
                if data and len(data["positions"]) == len(spans):
                    # Opening = mean per-finger travel fraction, the inverse of
                    # the command mapping above.
                    fractions = [
                        span.opening(position)
                        for position, span in zip(data["positions"], spans)
                    ]
                    self._io.publish_gripper_states(robot, name, sum(fractions) / len(fractions))

    def shutdown(self) -> dict:
        """Stops every view this scene reads the model through, and says what
        each standing camera counted, so the scene composed next carries on
        from there."""
        logger.info(
            "MujocoBridgeExtension shutting down: %s",
            ", ".join(f"'{name}'" for name in self._limbs) or "an empty scene",
        )
        counters: dict = {}

        def stop_rendering() -> None:
            nonlocal counters
            if self._camera_sensor is None:
                return
            self._camera_sensor.stop()
            counters = self._camera_sensor.counters()

        for close in (
            stop_rendering,
            self._articulation.teardown,
            *(limbs.teardown for limbs in self._limbs.values()),
        ):
            try:
                close()
            except Exception:  # pylint: disable=W0718
                logger.exception("a view of this scene did not close")
        return counters
