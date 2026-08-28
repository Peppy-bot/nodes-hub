"""The reader loop: the leader's positions out, nothing in.

Built on control_core_py's DeviceThread. Read-only: the leader is passive (torque
disabled by lerobot's SOLeader), so the thread polls positions and keeps the
newest sample in a latest-wins slot. A failed read leaves the last sample
standing with its old stamp, so the publish side's staleness gate is the
deadman.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from control_core_py.device import DeviceThread, LatestSlot
from control_core_py.runtime import Latch, RateMeter, log
from so101_description.device import connect_calibrated, read_positions_validated

# Sustained read failure before a reconnect attempt, and the pace of further
# attempts: USB serial adapters drop and re-enumerate under the same udev
# symlink, and the staleness deadman keeps the follower safe meanwhile.
RECONNECT_AFTER_S = 1.0
RECONNECT_BACKOFF_S = 5.0


@dataclass(frozen=True)
class LeaderSample:
    positions_deg: tuple[float, ...]
    gripper_percent: float
    captured_monotonic: float


class LeaderHardware(Protocol):
    """The narrow seam tests fake: everything below it touches the serial bus."""

    def connect(self) -> None: ...
    def read_positions(self) -> dict[str, float]: ...
    def disconnect(self) -> None: ...


class LerobotLeader:
    """LeaderHardware over lerobot's SOLeader, imported lazily."""

    def __init__(self, device_path: str, calibration_dir: str, teleop_id: str):
        self._device_path = device_path
        self._calibration_dir = calibration_dir
        self._teleop_id = teleop_id
        self._teleop = self._build()

    def _build(self):
        from pathlib import Path

        from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

        return SOLeader(
            SOLeaderTeleopConfig(
                port=self._device_path,
                id=self._teleop_id,
                # lerobot requires a Path here; a str dies in its mkdir.
                calibration_dir=Path(self._calibration_dir),
            )
        )

    def _fresh_teleop(self):
        """A connectable teleop. A dead fd fails lerobot's graceful
        disconnect before the port-close clears is_connected, and its
        already-connected guard then blocks every reconnect; a stale
        instance is rebuilt so the re-enumerated device opens cleanly."""
        if not self._teleop.is_connected:
            return self._teleop
        return self._build()

    def connect(self) -> None:
        self._teleop = self._fresh_teleop()
        connect_calibrated(self._teleop, "leader arm")

    def read_positions(self) -> dict[str, float]:
        action = self._teleop.get_action()
        return {key.removesuffix(".pos"): value for key, value in action.items()}

    def disconnect(self) -> None:
        # Idempotent: the calibration-failure path disconnects before
        # raising, and the bringup cleanup disconnects again.
        if not self._teleop.is_connected:
            return
        try:
            self._teleop.disconnect()
        except Exception:
            # The graceful path flushes and disables torque before closing,
            # and a dead fd fails those first; close the port directly so
            # the handle is not left claimed.
            self._close_port_quietly()

    def _close_port_quietly(self) -> None:
        try:
            self._teleop.bus.port_handler.closePort()
        except Exception:
            # A vanished device cannot be released; connect() rebuilds.
            pass


class DeviceLoop(DeviceThread):
    """The leader's read loop: one validated position poll per period."""

    stop_warning = "reader thread did not stop cleanly"

    def __init__(self, hardware: LeaderHardware, read_rate_hz: int):
        super().__init__(hardware, 1.0 / read_rate_hz, thread_name="so101-leader")
        self._rate = RateMeter("read loop", read_rate_hz)
        self._sample = LatestSlot()
        self._read_failing = Latch()
        self._reconnect_failing = Latch()
        self._last_success_monotonic = 0.0
        self._last_reconnect_monotonic = 0.0

    def fresh_sample(self, stale_timeout_s: float) -> LeaderSample | None:
        """The newest sample, or None once it is older than the deadman gate."""
        return self._sample.fresh(stale_timeout_s)

    def _verify_first_read(self) -> None:
        self._read_once(raise_on_error=True)
        self._last_success_monotonic = time.monotonic()

    def _tick(self) -> None:
        self._rate.tick()
        self._read_once()
        self._maybe_reconnect()

    def _maybe_reconnect(self) -> None:
        now = time.monotonic()
        if now - self._last_success_monotonic <= RECONNECT_AFTER_S:
            return
        if now - self._last_reconnect_monotonic <= RECONNECT_BACKOFF_S:
            return
        self._last_reconnect_monotonic = now
        try:
            self._hardware.disconnect()
        except Exception:
            # A dead fd refuses the release too; the reconnect below is the
            # recovery either way.
            pass
        try:
            self._hardware.connect()
        except Exception as e:
            self._reconnect_failing.trip(f"reconnect failing: {e!r}")
            return
        self._reconnect_failing.clear()
        log("leader bus reconnected")

    def _read_once(self, raise_on_error: bool = False) -> None:
        try:
            positions_deg, gripper_percent = read_positions_validated(self._hardware)
            sample = LeaderSample(
                positions_deg=positions_deg,
                gripper_percent=gripper_percent,
                captured_monotonic=time.monotonic(),
            )
        except Exception as e:
            if raise_on_error:
                raise
            # The stale gate downstream is the deadman; nothing to fabricate.
            # Logged once, so an unplugged leader leaves evidence beyond
            # wire silence.
            self._read_failing.trip(f"read failing: {e!r}")
            return
        self._read_failing.clear()
        self._last_success_monotonic = time.monotonic()
        self._sample.put(sample)
