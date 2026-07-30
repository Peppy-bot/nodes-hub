"""What an XR frame is, independent of how it arrives.

teleop_xr has already converted every pose to forward-left-up; the mapping
assumes the operator shares the robot's facing.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from xr_commander.frames import Pose

# WebXR `xr-standard` gamepad indices (oculus-touch-v3).
TRIGGER_BUTTON = 0
SQUEEZE_BUTTON = 1
# The face button under the thumb: A on the right hand, X on the left.
PRIMARY_BUTTON = 4


@dataclass(frozen=True)
class HandSample:
    """One controller's contribution to a frame."""

    pose: Pose
    # The deadman: grip button held.
    squeezing: bool
    # Analog trigger, 0 released to 1 fully pulled.
    trigger: float
    # A/X held this frame; released when the gamepad does not carry it.
    primary_button: bool = False


@dataclass(frozen=True)
class XrFrame:
    """One XR update; a hand absent from `hands` was not tracked."""

    # Local monotonic arrival time: staleness is judged on our clock, not the
    # headset's, which is what may have died.
    received_monotonic_s: float
    hands: Mapping[str, HandSample]

    def hand(self, handedness: str) -> HandSample | None:
        return self.hands.get(handedness)


class FrameSource(Protocol):
    """Hands back the most recent XR frame, or None before the first."""

    def latest(self) -> XrFrame | None: ...


def fresh_sample(
    source: FrameSource, handedness: str, stale_timeout_s: float
) -> HandSample | None:
    """This hand's sample if the frame carrying it is recent enough.

    Past the timeout the hands are unknown, not unchanged: a frozen headset is
    still reporting whatever button it last held.
    """
    frame = source.latest()
    if frame is None:
        return None
    if time.monotonic() - frame.received_monotonic_s > stale_timeout_s:
        return None
    return frame.hand(handedness)


def gripper_opening(trigger: float, open_fraction: float) -> float:
    """The opening a trigger commands: released rests at `open_fraction`,
    fully pulled is closed, linear between."""
    return open_fraction * (1.0 - trigger)


def _pressed(buttons: list, index: int) -> bool:
    """A button's held state, False when the gamepad does not carry it."""
    if len(buttons) <= index:
        return False
    return bool(buttons[index].get("pressed", False))


def _parse_pose(pose_data: dict) -> Pose:
    position = pose_data["position"]
    orientation = pose_data["orientation"]
    return Pose.from_xyz_quat(
        [position["x"], position["y"], position["z"]],
        [orientation["x"], orientation["y"], orientation["z"], orientation["w"]],
    )


def parse_hands(devices: list[object]) -> dict[str, HandSample]:
    """The tracked controllers in one frame, keyed by handedness.

    A device missing its grip pose or gamepad, or carrying an unusable pose,
    is simply untracked this frame: partial frames are normal in XR.
    """
    hands: dict[str, HandSample] = {}
    for device in devices:
        if not isinstance(device, dict) or device.get("role") != "controller":
            continue
        handedness = device.get("handedness")
        if handedness not in ("left", "right"):
            continue
        # Browser-supplied JSON: any shape violation inside one device drops
        # that device, never the frame.
        try:
            pose_data = device.get("gripPose")
            gamepad = device.get("gamepad")
            if not pose_data or not gamepad:
                continue
            buttons = gamepad.get("buttons") or []
            if len(buttons) <= max(TRIGGER_BUTTON, SQUEEZE_BUTTON):
                continue
            pose = _parse_pose(pose_data)
            trigger = float(buttons[TRIGGER_BUTTON].get("value", 0.0))
            squeezing = _pressed(buttons, SQUEEZE_BUTTON)
            primary = _pressed(buttons, PRIMARY_BUTTON)
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        if not math.isfinite(trigger):
            trigger = 0.0
        hands[handedness] = HandSample(
            pose=pose,
            squeezing=squeezing,
            trigger=min(max(trigger, 0.0), 1.0),
            primary_button=primary,
        )
    return hands
