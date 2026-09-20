#!/usr/bin/env python3
# pylint: disable=C0413
"""MujocoBridgeExtension owns the physics tick of one robot's scene. Each
step it applies the latest setpoint of every limb, advances physics, and
(throttled to state_rate_hz) publishes MuJoCo's own clock followed by the
measured joint and gripper state it stamps. The robot is commanded through
its own limb pairs and publishes its state back on them. Transport is typed
peppygen via SimTopicIO; there is no JSON and no raw peppylib on the path.
"""
from __future__ import annotations

import logging
import time

from sim_robot_core.cameras import FramePacer
from sim_robot_core.models import FingerSpan, finger_span

from mujoco_models import MujocoModel
from sim_topics import SimTopicIO
from exts import (
    MujocoActuatorCtrl,
    MujocoArticulation,
    MujocoCameraSensor,
    MujocoGripperSensor,
)

logger = logging.getLogger(__name__)


class MujocoBridgeExtension:
    """Drives one robot's scene from its command streams and publishes its
    state on its own pairs."""

    def __init__(
        self,
        model,
        data,
        io: SimTopicIO,
        robot: str,
        known: MujocoModel,
        state_rate_hz: int,
        renders: bool,
        time_base_s: float,
    ) -> None:
        self._model = model
        self._data = data
        self._io = io
        # The robot this scene stands: every setpoint read and every state
        # published is that robot's, on its own pairs.
        self._robot = robot
        # What this robot's model is made of: its limbs under the names its
        # pairs carry, and the joints each one moves.
        self._entry = known.entry
        # State publishes ride an absolute grid at state_rate_hz: serializing
        # every reader at the ~500 Hz physics tick saturates the single sim
        # thread. Writers and the physics step still run every tick.
        if state_rate_hz <= 0:
            raise ValueError(f"state_rate_hz must be positive, got {state_rate_hz}")
        self._state_pacer = FramePacer(state_rate_hz)
        # The engine clock runs on from where the previous scene left it, so
        # the fleet's time never goes back when a robot leaves and another
        # stands.
        self._time_base_s = time_base_s
        # Each finger joint's closed position and signed travel to fully open,
        # read from the model at setup; commanded opening fractions scale
        # onto it.
        self._finger_spans: dict[str, list[FingerSpan]] = {}
        # Last force limit written per gripper, so the cap is not re-sent per tick.
        self._applied_effort: dict[str, float] = {}

        self._articulation = MujocoArticulation(model, data)
        # One actuator controller for the whole robot: it resolves every actuator
        # by joint name, applies the model's MIT gains to the arm joints when
        # its entry carries some, and leaves every other actuator on its MJCF
        # defaults.
        self._actuator = MujocoActuatorCtrl(model, data, params=known.actuator_params())
        self._gripper_sensors: dict[str, MujocoGripperSensor] = {}
        # The cameras of the robot's model were attached to the scene at
        # compile time when the engine renders, so a model with none, or an
        # engine that renders none, has no sensor.
        cameras = list(known.entry.cameras) if renders else []
        self._camera_sensor = (
            MujocoCameraSensor(model, cameras, io, robot) if cameras else None
        )
        self._joint_index: dict[str, int] = {}

    def startup(self) -> None:
        if not self._articulation.setup():
            raise RuntimeError("MujocoArticulation setup failed")
        self._joint_index = {
            name: i for i, name in enumerate(self._articulation.get_joint_names())
        }
        # A joint of the model's entry that its MJCF lacks is refused here,
        # with its name.
        missing = sorted({n for n in self._entry.joints() if n not in self._joint_index})
        if missing:
            raise RuntimeError(
                f"the {self._entry.model} entry names joints not in its MuJoCo model: {missing}"
            )
        if not self._actuator.setup():
            raise RuntimeError("MujocoActuatorCtrl setup failed")
        self._actuator.require_force_limited(self._entry.finger_joints())
        for gripper in self._entry.grippers:
            sensor = MujocoGripperSensor(
                self._model, self._data, finger_joints=list(gripper.joints)
            )
            if not sensor.setup():
                raise RuntimeError(f"MujocoGripperSensor setup failed for gripper '{gripper.name}'")
            self._gripper_sensors[gripper.name] = sensor
            self._finger_spans[gripper.name] = [
                self._finger_span(name, gripper.closed_at) for name in gripper.joints
            ]
        self._start_in_posture()
        if self._camera_sensor is not None:
            self._camera_sensor.start()
        logger.info(
            f"MujocoBridgeExtension ready for '{self._robot}' ({self._entry.model}) with "
            f"{len(self._entry.arms)} arm(s), {len(self._entry.grippers)} gripper(s)"
        )

    def _finger_span(self, joint_name: str, closed_at: str) -> FingerSpan:
        import mujoco  # pylint: disable=C0415

        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        lo, hi = (float(v) for v in self._model.jnt_range[jid])
        return finger_span(joint_name, lo, hi, closed_at)

    def _start_in_posture(self) -> None:
        """Puts the robot in the posture its model starts in, and holds it
        there: the joints are placed and their actuators target the same
        positions, so the arm stands still until its first setpoint. A model
        whose entry names no posture starts where its file puts it."""
        import mujoco  # pylint: disable=C0415

        posture = self._entry.start_posture
        if not posture:
            return
        for joint, position in posture.items():
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            self._data.qpos[self._model.jnt_qposadr[jid]] = position
        self._actuator.write_targets(dict(posture))
        mujoco.mj_forward(self._model, self._data)

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
        for arm in self._entry.arms:
            command = self._io.latest_arm_command(self._robot, arm.name)
            if command is None:
                continue
            positions, velocities = command
            if len(positions) != len(arm.joints):
                continue
            velocity_values = (
                dict(zip(arm.joints, velocities)) if len(velocities) == len(arm.joints) else None
            )
            self._actuator.write_targets(dict(zip(arm.joints, positions)), velocity_values)

        for gripper in self._entry.grippers:
            command = self._io.latest_gripper_command(self._robot, gripper.name)
            if command is None:
                continue
            opening, max_effort = command
            # Re-applied only on change: the cap is a model write, not a
            # per-tick target.
            if self._applied_effort.get(gripper.name) != max_effort:
                # Recorded only once written, so a not-ready tick retries.
                if self._actuator.set_force_limit(list(gripper.joints), max_effort):
                    self._applied_effort[gripper.name] = max_effort
            # Map the opening fraction onto each finger's own span, so the same
            # command drives prismatic fingers, revolute ones and a single jaw.
            spans = self._finger_spans[gripper.name]
            self._actuator.write_targets(
                {name: span.position(opening) for name, span in zip(gripper.joints, spans)}
            )

    def _publish_state(self) -> None:
        states = self._articulation.get_joint_states()
        if states is not None:
            positions, velocities = states
            for arm in self._entry.arms:
                indices = [self._joint_index[name] for name in arm.joints]
                self._io.publish_arm_states(
                    self._robot,
                    arm.name,
                    [positions[i] for i in indices],
                    [velocities[i] for i in indices],
                )

        for name, sensor in self._gripper_sensors.items():
            data = sensor.get_gripper_state()
            spans = self._finger_spans[name]
            if data and len(data["positions"]) == len(spans):
                # Opening = mean per-finger travel fraction, the inverse of the
                # command mapping above.
                fractions = [span.opening(q) for q, span in zip(data["positions"], spans)]
                self._io.publish_gripper_states(
                    self._robot, name, sum(fractions) / len(fractions)
                )

    def shutdown(self) -> None:
        logger.info(f"MujocoBridgeExtension for '{self._robot}' shutting down.")
        if self._camera_sensor is not None:
            self._camera_sensor.stop()
        self._articulation.teardown()
        self._actuator.teardown()
        for sensor in self._gripper_sensors.values():
            sensor.teardown()
