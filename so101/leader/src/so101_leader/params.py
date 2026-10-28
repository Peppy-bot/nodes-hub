"""Launch parameters parsed once into an immutable config."""

from __future__ import annotations

from dataclasses import dataclass

from control_core_py.params import require_non_empty, require_positive, require_rate
from so101_description.device import BUS_MAX_RATE_HZ


@dataclass(frozen=True)
class Config:
    device_path: str
    calibration_dir: str
    teleop_id: str
    command_rate_hz: int
    stale_timeout_s: float


def parse(params) -> Config:
    # The bus ceiling; the stack reads the leader at 60 Hz.
    command_rate_hz = require_rate("command_rate_hz", params.command_rate_hz, BUS_MAX_RATE_HZ)
    stale_timeout_s = require_positive("stale_timeout_s", params.stale_timeout_s)
    if stale_timeout_s < 2.0 / command_rate_hz:
        # Publish ticks are phase-independent of read ticks, so a healthy
        # sample's age approaches one full period; anything under two flaps.
        raise ValueError("stale_timeout_s must be at least two read periods")
    return Config(
        device_path=require_non_empty("device_path", params.device_path),
        calibration_dir=require_non_empty("calibration_dir", params.calibration_dir),
        teleop_id=require_non_empty("teleop_id", params.teleop_id),
        command_rate_hz=command_rate_hz,
        stale_timeout_s=stale_timeout_s,
    )
