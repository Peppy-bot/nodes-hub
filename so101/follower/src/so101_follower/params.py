"""Launch parameters parsed once into an immutable config."""

from __future__ import annotations

from dataclasses import dataclass

from control_core_py.params import require_non_empty, require_rate

from so101_follower.health import HEALTH_RATE_HZ

# The fastest this node drives the bus: six STS3215s share one 1 Mbaud
# half-duplex serial line, and a goal write plus a position read must both
# fit in every cycle; the stack runs at 60 Hz.
MAX_RATE_HZ = 1000


@dataclass(frozen=True)
class Config:
    device_path: str
    calibration_dir: str
    robot_id: str
    control_rate_hz: int
    state_rate_hz: int


def _require_control_rate(value) -> int:
    """The control loop also carries the health read, decimated to
    HEALTH_RATE_HZ; below that rate the decimation interval would floor to
    zero ticks."""
    rate = require_rate("control_rate_hz", value, MAX_RATE_HZ)
    if rate < HEALTH_RATE_HZ:
        raise ValueError(
            f"control_rate_hz must be at least the health rate ({HEALTH_RATE_HZ} Hz); got {rate}"
        )
    return rate


def parse(params) -> Config:
    return Config(
        device_path=require_non_empty("device_path", params.device_path),
        calibration_dir=require_non_empty("calibration_dir", params.calibration_dir),
        robot_id=require_non_empty("robot_id", params.robot_id),
        control_rate_hz=_require_control_rate(params.control_rate_hz),
        state_rate_hz=require_rate("state_rate_hz", params.state_rate_hz, MAX_RATE_HZ),
    )
