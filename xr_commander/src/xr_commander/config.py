"""Launch parameters, parsed once into the shapes the rest of the node uses.

The only place a raw parameter is read. A bad launch fails here, naming the
parameter, not later as a NaN on the wire.
"""

from __future__ import annotations

import ipaddress
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


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


# Command-rate ceiling: far above the ~90 Hz headset and 100 Hz followers,
# and low enough that a stream can never starve the event loop.
_MAX_COMMAND_RATE_HZ = 1000


def _camera_names(raw: str) -> Mapping[str, str]:
    """Parse `camera_names` ("instance=view,instance=view") into a mapping.

    View names must be unique: they become the headset's track ids, and two
    tracks with one id would silently drop a camera.
    """
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    names: dict[str, str] = {}
    for entry in entries:
        instance, sep, view = (p.strip() for p in entry.partition("="))
        if not sep or not instance or not view:
            raise ValueError(
                f"camera_names entry must be 'instance_id=view_name', got {entry!r}"
            )
        if instance in names:
            raise ValueError(f"camera_names maps {instance!r} twice")
        names[instance] = view
    views = list(names.values())
    if len(set(views)) != len(views):
        raise ValueError(f"camera_names view names must be unique, got {views}")
    return MappingProxyType(names)


@dataclass(frozen=True)
class Settings:
    command_rate_hz: int
    https_port: int
    https_host: ipaddress.IPv4Address | ipaddress.IPv6Address
    motion_scale: float
    # Opening commanded at a released trigger; a pull spans [0, this].
    gripper_open_fraction: float
    # Requested duration of the ready move; 0 = fastest.
    ready_move_duration_s: float
    stale_timeout_s: float
    # Camera instance id -> headset view name; see peppy.json5 for the wire
    # syntax and the reserved wrist view names.
    camera_names: Mapping[str, str]

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
        ready_move_duration_s=_non_negative(
            "ready_move_duration_s", params.ready_move_duration_s
        ),
        stale_timeout_s=_positive("stale_timeout_s", params.stale_timeout_s),
        camera_names=_camera_names(str(params.camera_names)),
    )
