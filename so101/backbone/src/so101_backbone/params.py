"""Launch parameters parsed once into an immutable config."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from control_core_py.params import require_positive, require_rate
from so101.device import BUS_MAX_RATE_HZ

from so101_backbone.model import ARM
from so101_backbone.placement import MAX_LOOKUP_TIMEOUT_S


class UpstreamMode(Enum):
    JOINTS = "joints"
    POSE = "pose"


@dataclass(frozen=True)
class Config:
    upstream_mode: UpstreamMode
    control_rate_hz: int
    # Sizes action trajectories (postures, joint moves, joint paths,
    # move_arm) and bounds every stream's per-tick joint steps.
    max_joint_velocity_rad_s: tuple[float, ...]
    # Times action trajectories with the velocity caps: the peak acceleration
    # (rad/s²) of each joint.
    max_joint_acceleration_rad_s2: tuple[float, ...]
    max_stream_ee_velocity_m_s: float
    max_gripper_rate_frac_s: float
    # The farthest (rad) any measured joint may stand from a joint path's
    # expected start as the path starts, and, clamped into the joint limits,
    # from where the arm holds once the path is planned, before the goal
    # fails.
    start_tolerance_rad: float
    # How long (s) the placement of a command waits for the robot's tree,
    # from 0 to MAX_LOOKUP_TIMEOUT_S.
    lookup_timeout_s: float
    # How long (s) robot_stop's stop waits for the control loop to take and
    # answer it before it answers attempt_failed.
    stop_answer_timeout_s: float


def parse(params) -> Config:
    try:
        mode = UpstreamMode(params.upstream_mode)
    except ValueError:
        raise ValueError(
            f'upstream_mode must be "joints" or "pose", got {params.upstream_mode!r}'
        ) from None
    # This node touches no serial bus of its own; the ceiling is the
    # downstream follower's bus, since setpoints it cannot write are
    # setpoints not worth computing. The stack runs at 60 Hz.
    control_rate_hz = require_rate("control_rate_hz", params.control_rate_hz, BUS_MAX_RATE_HZ)

    velocity_caps = _per_joint(params, "max_joint_velocity_rad_s")
    acceleration_caps = _per_joint(params, "max_joint_acceleration_rad_s2")
    return Config(
        upstream_mode=mode,
        control_rate_hz=control_rate_hz,
        max_joint_velocity_rad_s=velocity_caps,
        max_joint_acceleration_rad_s2=acceleration_caps,
        max_stream_ee_velocity_m_s=require_positive("max_stream_ee_velocity_m_s", params.max_stream_ee_velocity_m_s),
        max_gripper_rate_frac_s=require_positive(
            "max_gripper_rate_frac_s", params.max_gripper_rate_frac_s
        ),
        start_tolerance_rad=require_positive("start_tolerance_rad", params.start_tolerance_rad),
        lookup_timeout_s=_lookup_timeout(params.lookup_timeout_s),
        stop_answer_timeout_s=require_positive(
            "stop_answer_timeout_s", params.stop_answer_timeout_s
        ),
    )


def _lookup_timeout(value) -> float:
    """The `lookup_timeout_s` parameter: from 0 to MAX_LOOKUP_TIMEOUT_S, the
    wait transform_lookup's lookup_transform takes, or a ValueError naming
    it, so the node stops at its start."""
    timeout_s = float(value)
    if not 0.0 <= timeout_s <= MAX_LOOKUP_TIMEOUT_S:
        raise ValueError(
            f"lookup_timeout_s is {value}: set it from 0 to {MAX_LOOKUP_TIMEOUT_S:g} s, "
            "the wait transform_lookup's lookup_transform takes"
        )
    return timeout_s


def _per_joint(params, prefix: str) -> tuple[float, ...]:
    """The values of the parameters `prefix`_1 onward, one per joint in the
    arm's joint order, each positive and finite."""
    return tuple(
        require_positive(f"{prefix}_{i + 1}", getattr(params, f"{prefix}_{i + 1}"))
        for i in range(len(ARM.joints))
    )
