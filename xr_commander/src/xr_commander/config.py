"""Launch parameters, parsed once into the shapes the rest of the node uses.

The only place a raw parameter is read. A bad launch fails here, naming the
parameter, not later as a NaN on the wire.
"""

from __future__ import annotations

import ipaddress
import math
from dataclasses import dataclass


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _positive(name: str, value: float) -> float:
    value = _finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def _fraction(name: str, value: float) -> float:
    value = _positive(name, value)
    if value > 1.0:
        raise ValueError(f"{name} must be at most 1, got {value!r}")
    return value


def _non_negative(name: str, value: float) -> float:
    value = _finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value


def _integer(name: str, value) -> int:
    """An exact int or whole-valued float; anything else (fractional, boolean,
    string, non-numeric) is refused rather than coerced."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if not value.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _int_in(name: str, value, lo: int, hi: int) -> int:
    value = _integer(name, value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..={hi}, got {value}")
    return value


def _port(name: str, value) -> int:
    return _int_in(name, value, 1, 65535)


def _host(name: str, value) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    try:
        return ipaddress.ip_address(str(value))
    except ValueError:
        raise ValueError(f"{name} is not an IP address: {value!r}") from None


# Command-rate ceiling: far above any tracked-input rate, and low enough
# that a stream can never starve the event loop.
_MAX_COMMAND_RATE_HZ = 1000
# Wider than any sensor this node should be asked to relay.
_MAX_VIEW_WIDTH_PX = 8192


def _flag(name: str, value) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false, got {value!r}")
    return value


@dataclass(frozen=True)
class Settings:
    command_rate_hz: int
    https_port: int
    https_host: ipaddress.IPv4Address | ipaddress.IPv6Address
    motion_scale: float
    # Opening commanded at a released trigger; a pull spans [0, this].
    gripper_open_fraction: float
    # Requested duration of the posture moves; 0 = fastest.
    posture_move_duration_s: float
    stale_timeout_s: float
    # Widest frame (px) relayed into the headset; 0 = sensor-native.
    view_max_width: int
    # Whether the headset shows the node's status panel.
    status_panel_enabled: bool

    @property
    def tick_period_s(self) -> float:
        return 1.0 / self.command_rate_hz


def from_parameters(params) -> Settings:
    """Parse the launch parameters; ValueError on anything the node cannot run with."""
    return Settings(
        command_rate_hz=_int_in(
            "command_rate_hz", params.command_rate_hz, 1, _MAX_COMMAND_RATE_HZ
        ),
        https_port=_port("https_port", params.https_port),
        https_host=_host("https_host", params.https_host),
        motion_scale=_positive("motion_scale", params.motion_scale),
        gripper_open_fraction=_fraction(
            "gripper_open_fraction", params.gripper_open_fraction
        ),
        posture_move_duration_s=_non_negative(
            "posture_move_duration_s", params.posture_move_duration_s
        ),
        stale_timeout_s=_positive("stale_timeout_s", params.stale_timeout_s),
        view_max_width=_int_in(
            "view_max_width", params.view_max_width, 0, _MAX_VIEW_WIDTH_PX
        ),
        status_panel_enabled=_flag("status_panel_enabled", params.status_panel_enabled),
    )
