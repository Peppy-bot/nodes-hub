"""The entries of the robots these suites stand: what a description gives a
robot (its arms, its grippers with where each finger closes and opens, its
cameras and its start posture), written out for a fake robot of a test."""

from __future__ import annotations

from sim_robot_core.cameras import CameraConfig
from sim_robot_core.models import Arm, FingerSpan, Gripper, ModelEntry


def entry(
    model: str,
    arms: dict | None = None,
    grippers: dict | None = None,
    cameras: tuple[CameraConfig, ...] = (),
    start_posture: dict | None = None,
) -> ModelEntry:
    """The entry of a robot of `model`: `arms` maps each arm to its joints,
    and `grippers` each gripper to its fingers, each finger a joint that
    moves of its own with the position it closes at and the one it opens
    at."""
    return ModelEntry(
        model=model,
        arms=tuple(Arm(name=name, joints=tuple(joints)) for name, joints in (arms or {}).items()),
        grippers=tuple(
            Gripper(
                name=name,
                joints=tuple(fingers),
                spans=tuple(
                    FingerSpan(closed=closed, travel=opened - closed)
                    for closed, opened in fingers.values()
                ),
                mimics=(None,) * len(fingers),
            )
            for name, fingers in (grippers or {}).items()
        ),
        cameras=cameras,
        start_posture=dict(start_posture or {}),
    )

