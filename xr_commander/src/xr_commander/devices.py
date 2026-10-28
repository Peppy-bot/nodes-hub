"""What an XR frame is, independent of how it arrives.

teleop_xr has already converted every pose to forward-left-up; the mapping
assumes the operator shares the robot's facing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from xr_commander.bus import REAL_CLOCK, Clock
from xr_commander.frames import Pose


class Hand(StrEnum):
    """A controller's handedness: the closed set WebXR reports; anything
    else is untracked."""

    LEFT = "left"
    RIGHT = "right"


# The controller whose face buttons move the robot to its postures.
POSTURE_HAND = Hand.RIGHT
# The controller whose face buttons drive the recorder.
RECORDER_HAND = Hand.LEFT

# WebXR `xr-standard` gamepad indices (oculus-touch-v3).
TRIGGER_BUTTON = 0
SQUEEZE_BUTTON = 1
# The face buttons under the thumb: lower is A on the right hand (X on the
# left), upper is B on the right hand (Y on the left).
PRIMARY_BUTTON = 4
SECONDARY_BUTTON = 5


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
    # B/Y held this frame; released when the gamepad does not carry it.
    secondary_button: bool = False


@dataclass(frozen=True)
class XrFrame:
    """One XR update; a hand absent from `hands` was not tracked."""

    # Local monotonic arrival time: staleness is judged on our clock, not the
    # headset's, which is what may have died.
    received_monotonic_s: float
    hands: Mapping[Hand, HandSample]

    def __post_init__(self) -> None:
        # Frozen in depth: the frame crosses threads, so the mapping must not
        # alias a caller's mutable dict.
        object.__setattr__(self, "hands", MappingProxyType(dict(self.hands)))

    def hand(self, handedness: Hand) -> HandSample | None:
        return self.hands.get(handedness)


class FrameSource(Protocol):
    """Hands back the most recent XR frame, or None before the first."""

    def latest(self) -> XrFrame | None: ...


def fresh_frame(
    source: FrameSource, stale_timeout_s: float, clock: Clock = REAL_CLOCK
) -> XrFrame | None:
    """The newest XR frame if it is recent enough.

    Past the timeout the hands are unknown, not unchanged: a frozen headset is
    still reporting whatever button it last held.
    """
    frame = source.latest()
    if frame is None:
        return None
    if clock.monotonic() - frame.received_monotonic_s > stale_timeout_s:
        return None
    return frame


def fresh_sample(
    source: FrameSource,
    handedness: Hand,
    stale_timeout_s: float,
    clock: Clock = REAL_CLOCK,
) -> HandSample | None:
    """This hand's sample if the frame carrying it is recent enough."""
    frame = fresh_frame(source, stale_timeout_s, clock)
    return frame.hand(handedness) if frame else None


def primary_pressed(sample: HandSample) -> bool:
    return sample.primary_button


def secondary_pressed(sample: HandSample) -> bool:
    return sample.secondary_button


class RisingEdge:
    """A button's presses, read tick by tick. A press is a rise against an
    observed release: a button already down on the first tracked frame is
    not a press, and neither is one held across a stale gap."""

    def __init__(self) -> None:
        # None until the button has been observed at all.
        self._last: bool | None = None

    def step(self, held: bool | None) -> bool:
        """Whether this tick's reading is a press. `held` is None while the
        hand is untracked, which keeps the last observed reading."""
        reading = self._last if held is None else held
        rising = reading and self._last is False
        self._last = reading
        return bool(rising)


def gripper_opening(trigger: float, open_fraction: float) -> float:
    """The opening a trigger commands: released rests at `open_fraction`,
    fully pulled is closed, linear between."""
    return open_fraction * (1.0 - trigger)


def _pressed(buttons: list, index: int) -> bool:
    """A button's held state, False when the gamepad does not carry it.

    Only the JSON boolean counts: any other type in browser-supplied JSON
    (`"false"`, 1) reads as released, because a squeeze must never be
    conjured out of a malformed payload."""
    if len(buttons) <= index:
        return False
    return buttons[index].get("pressed") is True


def _parse_pose(pose_data: dict) -> Pose:
    position = pose_data["position"]
    orientation = pose_data["orientation"]
    return Pose.from_xyz_quat(
        [position["x"], position["y"], position["z"]],
        [orientation["x"], orientation["y"], orientation["z"], orientation["w"]],
    )


def _parse_hand(value: object) -> Hand | None:
    try:
        return Hand(value)
    except ValueError:
        return None


def parse_hands(devices: list[object]) -> dict[Hand, HandSample]:
    """The tracked controllers in one frame, keyed by handedness.

    A device missing its grip pose or gamepad, or carrying an unusable pose,
    is simply untracked this frame: partial frames are normal in XR.
    """
    hands: dict[Hand, HandSample] = {}
    for device in devices:
        if not isinstance(device, dict) or device.get("role") != "controller":
            continue
        handedness = _parse_hand(device.get("handedness"))
        if handedness is None:
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
            secondary = _pressed(buttons, SECONDARY_BUTTON)
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        if not math.isfinite(trigger):
            trigger = 0.0
        hands[handedness] = HandSample(
            pose=pose,
            squeezing=squeezing,
            trigger=min(max(trigger, 0.0), 1.0),
            primary_button=primary,
            secondary_button=secondary,
        )
    return hands
