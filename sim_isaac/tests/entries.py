"""The entries of the robots these suites stand: what a description gives a
robot (its arms, its grippers with where each finger closes and opens, the
wheels of its base, its cameras and its start posture), written out for a
fake robot of a test."""

from __future__ import annotations

from robot_description_py import FingerSpan
from sim_robot_core.cameras import CameraConfig
from sim_robot_core.models import Arm, Base, Gripper, ModelEntry
from sim_robot_core.pairs import BASE_NAME


def entry(
    model: str,
    arms: dict | None = None,
    grippers: dict | None = None,
    cameras: tuple[CameraConfig, ...] = (),
    start_posture: dict | None = None,
    wheels: tuple[str, ...] = (),
) -> ModelEntry:
    """The entry of a robot of `model`: `arms` maps each arm to its joints,
    `grippers` each gripper to its fingers, each finger a joint that moves
    of its own with the position it closes at and the one it opens at, and
    `wheels` are the joints of its base's wheels, left then right, none for
    a robot without a base."""
    return ModelEntry(
        model=model,
        arms=tuple(Arm(name=name, joints=tuple(joints)) for name, joints in (arms or {}).items()),
        grippers=tuple(
            Gripper(
                name=name,
                joints=tuple(fingers),
                spans=tuple(FingerSpan(closed, opened) for closed, opened in fingers.values()),
                mimics=(None,) * len(fingers),
            )
            for name, fingers in (grippers or {}).items()
        ),
        cameras=cameras,
        start_posture=dict(start_posture or {}),
        base=Base(name=BASE_NAME, wheels=tuple(wheels)) if wheels else None,
    )

