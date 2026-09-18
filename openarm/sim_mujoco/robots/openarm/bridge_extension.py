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
from dataclasses import dataclass
from pathlib import Path

import pyjson5

from camera_common import CameraConfig, FramePacer
from sim_topics import SimTopicIO
from exts import (
    MujocoActuatorCtrl,
    MujocoArticulation,
    MujocoCameraSensor,
    MujocoGripperSensor,
)

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
    """One arm or gripper of a model: the name the engine lists a robot's
    limbs under, and the joints it moves in the scene."""

    name: str
    joints: tuple[str, ...]


@dataclass(frozen=True)
class Layout:
    """What a robot of this engine is made of, read from sim_bridge.json5.
    Every scene the catalogue carries has these limbs under these names."""

    arms: tuple[Limb, ...]
    grippers: tuple[Limb, ...]
    arm_gains: dict

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

    def actuator_params(self) -> dict:
        """The MIT gains of every arm joint, the same per-joint gains (j1..j7)
        on each arm."""
        return {
            "joint_names": [joint for arm in self.arms for joint in arm.joints],
            "kp": list(self.arm_gains.get("kp", [])) * len(self.arms),
            "kd": list(self.arm_gains.get("kd", [])) * len(self.arms),
            "gravity_compensation": self.arm_gains.get("gravity_compensation", False),
        }


class MujocoBridgeExtension:
    """Drives one robot's scene from its command streams and publishes its
    state on its own pairs."""

    def __init__(
        self,
        model,
        data,
        io: SimTopicIO,
        robot: str,
        layout: Layout,
        state_rate_hz: int,
        cameras: list[CameraConfig],
        time_base_s: float,
    ) -> None:
        self._model = model
        self._data = data
        self._io = io
        # The robot this scene stands: every setpoint read and every state
        # published is that robot's, on its own pairs.
        self._robot = robot
        self._layout = layout
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
        # Signed full-open travel per finger joint, read from the model at
        # setup; commanded opening fractions scale onto it.
        self._gripper_travels: dict[str, list[float]] = {}
        # Last force limit written per gripper, so the cap is not re-sent per tick.
        self._applied_effort: dict[str, float] = {}

        self._articulation = MujocoArticulation(model, data)
        # One actuator controller for the whole robot: it resolves every actuator
        # by joint name, applies the MIT gains to the arm joints, and leaves the
        # finger joints on their MJCF defaults.
        self._actuator = MujocoActuatorCtrl(model, data, params=layout.actuator_params())
        self._gripper_sensors: dict[str, MujocoGripperSensor] = {}
        # The cameras were attached to this model at compile time, so an empty
        # list here means the scene carries none to render.
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
        # A sim_bridge.json5 joint the model lacks is refused here, with its
        # name.
        missing = sorted({n for n in self._layout.joints_of() if n not in self._joint_index})
        if missing:
            raise RuntimeError(
                f"sim_bridge.json5 references joints not in the MuJoCo model: {missing}"
            )
        if not self._actuator.setup():
            raise RuntimeError("MujocoActuatorCtrl setup failed")
        self._actuator.require_force_limited(
            [joint for gripper in self._layout.grippers for joint in gripper.joints]
        )
        for gripper in self._layout.grippers:
            sensor = MujocoGripperSensor(
                self._model, self._data, finger_joints=list(gripper.joints)
            )
            if not sensor.setup():
                raise RuntimeError(f"MujocoGripperSensor setup failed for gripper '{gripper.name}'")
            self._gripper_sensors[gripper.name] = sensor
            self._gripper_travels[gripper.name] = [
                self._finger_travel(name) for name in gripper.joints
            ]
        if self._camera_sensor is not None:
            self._camera_sensor.start()
        logger.info(
            f"MujocoBridgeExtension ready for '{self._robot}' with "
            f"{len(self._layout.arms)} arm(s), {len(self._layout.grippers)} gripper(s)"
        )

    def _finger_travel(self, joint_name: str) -> float:
        import mujoco  # pylint: disable=C0415

        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        lo, hi = (float(v) for v in self._model.jnt_range[jid])
        return _finger_travel_from_range(joint_name, lo, hi)

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
        for arm in self._layout.arms:
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

        for gripper in self._layout.grippers:
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
            # Map the opening fraction onto each finger's own signed travel, so
            # the same command drives prismatic (v1) and revolute (v2) fingers.
            travels = self._gripper_travels[gripper.name]
            self._actuator.write_targets(
                {name: travel * opening for name, travel in zip(gripper.joints, travels)}
            )

    def _publish_state(self) -> None:
        states = self._articulation.get_joint_states()
        if states is not None:
            positions, velocities = states
            for arm in self._layout.arms:
                indices = [self._joint_index[name] for name in arm.joints]
                self._io.publish_arm_states(
                    self._robot,
                    arm.name,
                    [positions[i] for i in indices],
                    [velocities[i] for i in indices],
                )

        for name, sensor in self._gripper_sensors.items():
            data = sensor.get_gripper_state()
            travels = self._gripper_travels[name]
            if data and len(data["positions"]) == len(travels):
                # Opening = mean per-finger travel fraction, the inverse of the
                # command mapping above.
                fractions = [q / t for q, t in zip(data["positions"], travels)]
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
