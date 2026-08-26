#!/usr/bin/env python3
# pylint: disable=R0902,C0413
"""IsaacBridgeExtension owns the per-step bridge for the openarm Isaac scene.
Isaac's sim_app.update() advances physics on the main thread; each step this
extension applies the latest sim-passthrough setpoint per side, reads the
measured joint and gripper state, and (throttled to state_rate_hz) publishes it.
Transport is typed peppygen via SimTopicIO; there is no JSON and no raw
peppylib on the path.
"""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import pyjson5

from camera_common import load_camera_configs, validate_camera_slots
from sim_topics import COLOR_CAMERA_SLOT_NAMES, RGBD_CAMERA_SLOT_NAMES, SimTopicIO
from exts import IsaacActuatorCtrl, IsaacArticulation, IsaacCameraSensor, IsaacGripperSensor

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent / "config"
_CONFIG_PATH = _CONFIG_DIR / "sim_bridge.json5"
_CAMERAS_CONFIG_PATH = _CONFIG_DIR / "cameras.json5"


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
# The articulation root prim in the loaded USD stage. Every ext (state read,
# actuator write, gripper sensor) targets this one articulation. The robot USD's
# defaultPrim is /openarm (the launcher only adds /World/defaultDomeLight at
# runtime; the robot itself is not reparented under /World).
_ROOT_ARTICULATION_PRIM = "/openarm"


class IsaacBridgeExtension:
    """Drives the engine from the typed command streams and publishes state.

    Articulation setup is deferred to the first step() that succeeds: the
    Articulation views cannot initialise until the USD stage has loaded and the
    timeline is playing, which races the bridge's construction. Until every ext
    is ready, step() is a no-op except for the setup retry.
    """

    def __init__(self, io: SimTopicIO, state_rate_hz: int, cameras_enabled: bool) -> None:
        self._io = io
        # Telemetry is throttled to state_rate_hz: serializing every reader at
        # the physics tick saturates the single sim thread. Writers and the
        # physics step still run every tick.
        if state_rate_hz <= 0:
            raise ValueError(f"state_rate_hz must be positive, got {state_rate_hz}")
        # Signed full-open travel per finger joint, read from the articulation's
        # DOF limits at setup; commanded opening fractions scale onto it.
        self._gripper_travels: dict[int, list[float]] = {}
        # Last force limit written per gripper, so the cap is not re-sent per tick.
        self._applied_effort: dict[int, float] = {}
        self._telemetry_period_s = 1.0 / state_rate_hz
        self._last_publish_s = 0.0

        cfg = pyjson5.loads(_CONFIG_PATH.read_text())
        self._arms: list[dict] = cfg["arms"]
        self._grippers: list[dict] = cfg["grippers"]
        self._gains: dict = cfg.get("arm_gains", {})
        self._gripper_gains: dict = cfg.get("gripper_gains", {})

        self._articulation = IsaacArticulation(_ROOT_ARTICULATION_PRIM)
        # One actuator controller per arm side: the MIT gains and torque caps
        # are applied to that side's PhysX drives at setup, and the side's
        # commands are written through it. Finger joints use explicit PhysX
        # position-drive gains so they hold their commanded opening in simulation.
        self._arm_actuators: dict[int, IsaacActuatorCtrl] = {
            arm["arm_id"]: IsaacActuatorCtrl(
                _ROOT_ARTICULATION_PRIM,
                joint_names=arm["joints"],
                params=self._actuator_params(arm["joints"]),
            )
            for arm in self._arms
        }
        self._gripper_actuators: dict[int, IsaacActuatorCtrl] = {
            gripper["gripper_id"]: IsaacActuatorCtrl(
                _ROOT_ARTICULATION_PRIM,
                joint_names=gripper["fingers"],
                params=self._gripper_actuator_params(gripper["fingers"]),
            )
            for gripper in self._grippers
        }
        self._gripper_sensors: dict[int, IsaacGripperSensor] = {
            gripper["gripper_id"]: IsaacGripperSensor(
                _ROOT_ARTICULATION_PRIM, finger_joints=gripper["fingers"]
            )
            for gripper in self._grippers
        }
        self._camera_sensor = None
        if cameras_enabled:
            cameras = load_camera_configs(_CAMERAS_CONFIG_PATH)
            validate_camera_slots(cameras, COLOR_CAMERA_SLOT_NAMES, RGBD_CAMERA_SLOT_NAMES)
            self._camera_sensor = IsaacCameraSensor(_ROOT_ARTICULATION_PRIM, cameras, io)
        self._joint_index: dict[str, int] = {}
        self._ready: bool = False

    def _gripper_actuator_params(self, joints: list[str]) -> dict:
        gains = self._gripper_gains
        return {
            "joint_names": joints,
            "kp": list(gains.get("kp", [])),
            "kd": list(gains.get("kd", [])),
            "max_efforts": list(gains.get("max_efforts", [])),
        }

    def _actuator_params(self, joints: list[str]) -> dict:
        return {
            "joint_names": joints,
            "kp": list(self._gains.get("kp", [])),
            "kd": list(self._gains.get("kd", [])),
            "max_efforts": list(self._gains.get("max_efforts", [])),
            "gravity_compensation": self._gains.get("gravity_compensation", False),
        }

    def _all_exts(self) -> list:
        return [
            self._articulation,
            *self._arm_actuators.values(),
            *self._gripper_actuators.values(),
            *self._gripper_sensors.values(),
            *([self._camera_sensor] if self._camera_sensor is not None else []),
        ]

    def _try_setup(self) -> bool:
        """Attempt to initialise every ext against the live stage. Returns True
        once all are ready; safe to call every step until it succeeds."""
        if self._ready:
            return True
        if not all(ext.setup() for ext in self._all_exts()):
            return False
        self._joint_index = {
            name: i for i, name in enumerate(self._articulation.get_joint_names())
        }
        # Fail loudly on a sim_bridge.json5 typo: a joint the articulation doesn't
        # have would otherwise silently drop that side's commands + telemetry.
        configured = [j for arm in self._arms for j in arm["joints"]] + [
            f for g in self._grippers for f in g["fingers"]
        ]
        missing = sorted({n for n in configured if n not in self._joint_index})
        if missing:
            raise RuntimeError(
                f"sim_bridge.json5 references joints not in the Isaac articulation: {missing}"
            )
        limits = self._articulation.get_joint_limits()
        if limits is None:
            return False
        lower, upper = limits
        for gripper in self._grippers:
            travels = [
                _finger_travel_from_range(
                    name, lower[self._joint_index[name]], upper[self._joint_index[name]]
                )
                for name in gripper["fingers"]
            ]
            self._gripper_travels[gripper["gripper_id"]] = travels
        self._ready = True
        logger.info(
            f"IsaacBridgeExtension ready with {len(self._arms)} arm(s), "
            f"{len(self._grippers)} gripper(s)"
        )
        return True

    @property
    def is_ready(self) -> bool:
        return self._ready

    def step(self) -> None:
        """Physics has already advanced in sim_app.update(); apply the latest
        commands and (throttled) publish measured state."""
        if not self._try_setup():
            return

        self._apply_commands()
        if self._camera_sensor is not None:
            self._camera_sensor.step()

        now = time.monotonic()
        if now - self._last_publish_s < self._telemetry_period_s:
            return
        self._last_publish_s = now
        self._publish_state()

    def _apply_commands(self) -> None:
        for arm in self._arms:
            command = self._io.latest_arm_command(arm["arm_id"])
            if command is None:
                continue
            positions, velocities = command
            joints = arm["joints"]
            if len(positions) != len(joints):
                continue
            velocity_values = (
                dict(zip(joints, velocities)) if len(velocities) == len(joints) else None
            )
            self._arm_actuators[arm["arm_id"]].write_targets(
                dict(zip(joints, positions)), velocity_values
            )

        for gripper in self._grippers:
            command = self._io.latest_gripper_command(gripper["gripper_id"])
            if command is None:
                continue
            opening, max_effort = command
            # Re-applied only on change: the ceiling write is a model-wide
            # articulation call, not a per-tick target.
            if self._applied_effort.get(gripper["gripper_id"]) != max_effort:
                # Recorded only once written, so a not-ready tick retries.
                if self._gripper_actuators[gripper["gripper_id"]].set_force_limit(
                    gripper["fingers"], max_effort
                ):
                    self._applied_effort[gripper["gripper_id"]] = max_effort
            # Map the opening fraction onto each finger's own signed travel, so
            # the same command drives prismatic (v1) and revolute (v2) fingers.
            travels = self._gripper_travels[gripper["gripper_id"]]
            self._gripper_actuators[gripper["gripper_id"]].write_targets(
                {
                    name: travel * opening
                    for name, travel in zip(gripper["fingers"], travels)
                }
            )

    def _publish_state(self) -> None:
        states = self._articulation.get_joint_states()
        if states is not None:
            positions, velocities = states
            for arm in self._arms:
                indices = [self._joint_index.get(name) for name in arm["joints"]]
                if any(i is None for i in indices):
                    continue
                self._io.publish_arm_states(
                    arm["arm_id"],
                    [positions[i] for i in indices],
                    [velocities[i] for i in indices],
                )

        for gripper_id, sensor in self._gripper_sensors.items():
            data = sensor.get_gripper_state()
            travels = self._gripper_travels[gripper_id]
            if data and len(data["positions"]) == len(travels):
                # Opening = mean per-finger travel fraction, the inverse of the
                # command mapping above.
                fractions = [q / t for q, t in zip(data["positions"], travels)]
                self._io.publish_gripper_states(
                    gripper_id, sum(fractions) / len(fractions)
                )

    def shutdown(self) -> None:
        logger.info("IsaacBridgeExtension shutting down.")
        for ext in self._all_exts():
            ext.teardown()
        gc.collect()
