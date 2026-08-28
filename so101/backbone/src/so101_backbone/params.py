"""Launch parameters parsed once into an immutable config."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from control_core_py.params import require_positive, require_rate
from so101_description.units import NUM_JOINTS


class UpstreamMode(Enum):
    JOINTS = "joints"
    POSE = "pose"


# The fastest the coordination loop is driven. This node touches no serial
# bus of its own; the ceiling is the downstream follower's, since setpoints
# it cannot write are setpoints not worth computing. The stack runs at 60 Hz.
MAX_RATE_HZ = 1000


@dataclass(frozen=True)
class Config:
    upstream_mode: UpstreamMode
    control_rate_hz: int
    # Sizes action trajectories (postures, move_arm) and bounds the
    # machine-generated pose stream's per-tick joint steps. Never shapes
    # the joints-led teleop stream, which only the EE governor may limit.
    max_joint_velocity_rad_s: tuple[float, ...]
    max_ee_velocity_m_s: float
    max_ee_angular_velocity_rad_s: float
    max_gripper_rate_frac_s: float


def parse(params) -> Config:
    try:
        mode = UpstreamMode(params.upstream_mode)
    except ValueError:
        raise ValueError(
            f'upstream_mode must be "joints" or "pose", got {params.upstream_mode!r}'
        ) from None
    control_rate_hz = require_rate("control_rate_hz", params.control_rate_hz, MAX_RATE_HZ)

    velocity_caps = tuple(
        require_positive(
            f"max_joint_velocity_rad_s_{i + 1}",
            getattr(params, f"max_joint_velocity_rad_s_{i + 1}"),
        )
        for i in range(NUM_JOINTS)
    )
    return Config(
        upstream_mode=mode,
        control_rate_hz=control_rate_hz,
        max_joint_velocity_rad_s=velocity_caps,
        max_ee_velocity_m_s=require_positive("max_ee_velocity_m_s", params.max_ee_velocity_m_s),
        max_ee_angular_velocity_rad_s=require_positive(
            "max_ee_angular_velocity_rad_s", params.max_ee_angular_velocity_rad_s
        ),
        max_gripper_rate_frac_s=require_positive(
            "max_gripper_rate_frac_s", params.max_gripper_rate_frac_s
        ),
    )
