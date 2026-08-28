"""The device loop: targets in, state and health snapshots out.

Built on control_core_py's DeviceThread: one OS thread owns the serial bus for the
process lifetime, and the asyncio side exchanges data through latest-wins
slots. A failed read leaves the last snapshot standing with its old capture
stamp, so staleness is visible downstream instead of papered over.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

from control_core_py.device import DeviceThread, LatestSlot
from control_core_py.runtime import Latch, RateMeter
from so101_description.device import connect_calibrated, read_positions_validated
from so101_description.units import GRIPPER_NAME, JOINT_NAMES, MOTOR_NAMES

from so101_follower.health import HEALTH_RATE_HZ, describe_faults

# Setpoints whose wire stamp sits this far from now are refused on arrival:
# a replayed backlog or a clock fault is not a current command.
STALE_WIRE_TIMEOUT_S = 0.25

# Setpoints that arrived longer ago than this stop being written; the servo
# PID holds the last goal, the deadman for a commander that goes silent (the
# leader has no engage switch). The two gates compose rather than overlap, so
# a command that passed the wire gate at its limit can still be written for
# this long after: the worst-case stop is their sum.
STALE_SETPOINT_TIMEOUT_S = 0.25

# Present_Load arrives sign-decoded by lerobot (direction bit already folded
# into the sign); the magnitude is per-mille of stall torque.
_LOAD_PER_MILLE = 1000.0


# One health read: temps_c, |load| stall fractions, torque_enabled, fault_bits.
HealthReadings = tuple[
    tuple[float, ...], tuple[float, ...], tuple[bool, ...], tuple[int, ...]
]


@dataclass(frozen=True)
class StateSnapshot:
    positions_deg: tuple[float, ...]
    gripper_percent: float
    captured_monotonic: float


@dataclass(frozen=True)
class HealthSnapshot:
    temps_c: tuple[float, ...]
    load_fractions: tuple[float, ...]
    # Torque_Enable per motor: False on a motor the firmware or a brownout
    # has silently disabled while the node believes it is driving.
    torque_enabled: tuple[bool, ...]
    # Status register (fault latch) per motor: nonzero when the servo's own
    # protection has tripped (overload keeps Torque_Enable at 1 while
    # cutting output, so this is the only register that shows it).
    fault_bits: tuple[int, ...]
    # Health-read failures in a row; state reads keep their own evidence
    # (the state snapshot's age) and never touch this.
    consecutive_missed_reads: int
    # When the readings were captured (advances only on a good read).
    readings_captured_monotonic: float
    # When the device thread last refreshed this snapshot (thread liveness).
    captured_monotonic: float


class FollowerHardware(Protocol):
    """The narrow seam tests fake: everything below it touches the serial bus."""

    def connect(self) -> None: ...
    def read_positions(self) -> dict[str, float]: ...
    def write_goals(self, goals_by_motor: dict[str, float]) -> None: ...
    def read_health(self) -> HealthReadings: ...
    def disconnect(self) -> None: ...


class LerobotFollower:
    """FollowerHardware over lerobot's SOFollower. Imported lazily so the
    policy modules and their tests never load torch or the serial stack."""

    def __init__(self, device_path: str, calibration_dir: str, robot_id: str):
        from pathlib import Path

        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

        self._robot = SOFollower(
            SOFollowerRobotConfig(
                port=device_path,
                id=robot_id,
                # lerobot requires a Path here; a str dies in its mkdir.
                calibration_dir=Path(calibration_dir),
                # lerobot's default: no per-cycle jump clamp, no extra
                # position read per write.
                max_relative_target=None,
                # Pinned explicitly: the whole unit model rides on the
                # degrees frame, and shutdown safety on the torque release.
                use_degrees=True,
                disable_torque_on_disconnect=True,
            )
        )

    def connect(self) -> None:
        connect_calibrated(self._robot, "arm")

    def read_positions(self) -> dict[str, float]:
        observation = self._robot.get_observation()
        return {key.removesuffix(".pos"): value for key, value in observation.items()}

    def write_goals(self, goals_by_motor: dict[str, float]) -> None:
        # The servo EPROM position limits, written by calibration, are the
        # travel guard.
        self._robot.send_action({f"{motor}.pos": value for motor, value in goals_by_motor.items()})

    def read_health(self) -> HealthReadings:
        retries = self._robot.config.num_read_retries
        bus = self._robot.bus
        temps = bus.sync_read("Present_Temperature", normalize=False, num_retry=retries)
        loads = bus.sync_read("Present_Load", normalize=False, num_retry=retries)
        torque = bus.sync_read("Torque_Enable", normalize=False, num_retry=retries)
        status = bus.sync_read("Status", normalize=False, num_retry=retries)
        return (
            tuple(float(temps[motor]) for motor in MOTOR_NAMES),
            tuple(abs(float(loads[motor])) / _LOAD_PER_MILLE for motor in MOTOR_NAMES),
            tuple(bool(torque[motor]) for motor in MOTOR_NAMES),
            tuple(int(status[motor]) for motor in MOTOR_NAMES),
        )

    def disconnect(self) -> None:
        # Releases the arm (disable_torque_on_disconnect is pinned true).
        # Idempotent: the calibration-failure path disconnects before
        # raising, and the bringup cleanup disconnects again.
        if self._robot.is_connected:
            self._robot.disconnect()


class DeviceLoop(DeviceThread):
    """The follower's hardware loop: write fresh targets, read state, and
    read health decimated off the motion path."""

    stop_warning = "device thread did not stop; torque may still be enabled"

    def __init__(self, hardware: FollowerHardware, control_rate_hz: int):
        super().__init__(hardware, 1.0 / control_rate_hz, thread_name="so101-device")
        self._rate = RateMeter("device loop", control_rate_hz)
        # Floored so the actual read rate never lands below HEALTH_RATE_HZ
        # on odd ratios.
        self._health_every_ticks = control_rate_hz // HEALTH_RATE_HZ
        self._stale_timeout_s = STALE_SETPOINT_TIMEOUT_S

        self._arm_target = LatestSlot()
        self._gripper_target = LatestSlot()
        self._state = LatestSlot()
        self._health = LatestSlot()
        self._tick_count = 0
        self._write_failing = Latch()
        self._read_failing = Latch()
        self._missed_health_reads = 0
        self._last_health_readings: HealthReadings = ((), (), (), ())
        self._readings_captured_monotonic = 0.0

    def submit_arm_target(self, positions_deg: tuple[float, ...]) -> None:
        self._arm_target.put(positions_deg)

    def submit_gripper_target(self, percent: float) -> None:
        self._gripper_target.put(percent)

    def latest_state(self) -> StateSnapshot | None:
        return self._state.get()

    def latest_health(self) -> HealthSnapshot | None:
        return self._health.get()

    def _verify_first_read(self) -> None:
        self._read_state(raise_on_error=True)
        # Enable-and-confirm: connect's torque writes are acknowledged, but
        # an ack does not prove the servo stays enabled (its own protection
        # can re-disable it), and a disabled motor still answers position
        # reads. A launch must not report ready around a limp joint.
        temps_c, load_fractions, torque_enabled, fault_bits = self._read_health_validated()
        disabled = [name for name, on in zip(MOTOR_NAMES, torque_enabled, strict=True) if not on]
        if disabled:
            raise RuntimeError(f"torque did not enable at bringup: {', '.join(disabled)}")
        # A latched servo fault (e.g. overload) survives reconnects and cuts
        # output while Torque_Enable stays 1; refuse to report ready over it.
        faulted = [
            f"{name} ({describe_faults(bits)})"
            for name, bits in zip(MOTOR_NAMES, fault_bits, strict=True)
            if bits != 0
        ]
        if faulted:
            raise RuntimeError(f"servo faults latched at bringup: {', '.join(faulted)}")
        self._last_health_readings = (temps_c, load_fractions, torque_enabled, fault_bits)
        self._readings_captured_monotonic = time.monotonic()

    def _tick(self) -> None:
        self._rate.tick()
        self._write_fresh_targets()
        self._read_state()
        if self._tick_count % self._health_every_ticks == 0:
            self._read_health()
        self._tick_count += 1

    def _write_fresh_targets(self) -> None:
        goals: dict[str, float] = {}
        arm = self._arm_target.fresh(self._stale_timeout_s)
        if arm is not None:
            goals.update(zip(JOINT_NAMES, arm, strict=True))
        gripper = self._gripper_target.fresh(self._stale_timeout_s)
        if gripper is not None:
            goals[GRIPPER_NAME] = gripper
        if not goals:
            # Silence is the hold: no goal write, the servo PID keeps the last.
            return
        try:
            self._hardware.write_goals(goals)
            self._write_failing.clear()
        except Exception as e:
            self._write_failing.trip(f"goal write failing: {e!r}")

    def _read_state(self, raise_on_error: bool = False) -> None:
        try:
            positions_deg, gripper_percent = read_positions_validated(self._hardware)
            snapshot = StateSnapshot(
                positions_deg=positions_deg,
                gripper_percent=gripper_percent,
                captured_monotonic=time.monotonic(),
            )
        except Exception as e:
            if raise_on_error:
                raise
            # No fabrication: the last snapshot stands and its age shows. The
            # edge is logged, matching the leader's read loop, so a bus going
            # intermittent is visible before the watchdog's exit horizon.
            self._read_failing.trip(f"state read failing: {e!r}")
            return
        self._read_failing.clear()
        self._state.put(snapshot)

    def _read_health_validated(self) -> HealthReadings:
        """One health read, refused unless every scalar is finite: a corrupt
        reading must not poison the sustained-load EWMA or compare as nominal
        downstream."""
        temps_c, load_fractions, torque_enabled, fault_bits = self._hardware.read_health()
        if not all(math.isfinite(v) for v in (*temps_c, *load_fractions)):
            raise ValueError("non-finite health read")
        return temps_c, load_fractions, torque_enabled, fault_bits

    def _read_health(self) -> None:
        try:
            temps_c, load_fractions, torque_enabled, fault_bits = self._read_health_validated()
            self._last_health_readings = (temps_c, load_fractions, torque_enabled, fault_bits)
            self._readings_captured_monotonic = time.monotonic()
            self._missed_health_reads = 0
        except Exception:
            # Keep the last good readings standing under their old capture
            # stamp; the missed-read counter decides when they stop counting
            # as evidence.
            self._missed_health_reads += 1
            temps_c, load_fractions, torque_enabled, fault_bits = self._last_health_readings
        self._health.put(
            HealthSnapshot(
                temps_c=temps_c,
                load_fractions=load_fractions,
                torque_enabled=torque_enabled,
                fault_bits=fault_bits,
                consecutive_missed_reads=self._missed_health_reads,
                readings_captured_monotonic=self._readings_captured_monotonic,
                captured_monotonic=time.monotonic(),
            )
        )
