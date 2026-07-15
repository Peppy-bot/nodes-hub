"""G1 preset arm-gesture vocabulary and action-id mapping.

Parse a wire gesture name into a known gesture once at the boundary; the handler
looks up its SDK action id. Ids match the G1ArmActionClient action_map.
"""

from __future__ import annotations


class UnknownGesture(ValueError):
    """Raised when a wire string does not name a known arm gesture."""


# Gesture name -> G1ArmActionClient action id.
GESTURE_ACTION_ID: dict[str, int] = {
    "release_arm": 99,
    "two_hand_kiss": 11,
    "left_kiss": 12,
    "right_kiss": 13,
    "hands_up": 15,
    "clap": 17,
    "high_five": 18,
    "hug": 19,
    "heart": 20,
    "right_heart": 21,
    "reject": 22,
    "right_hand_up": 23,
    "x_ray": 24,
    "face_wave": 25,
    "high_wave": 26,
    "shake_hand": 27,
}


def action_id_for(gesture: str) -> int:
    """Return the SDK action id for a gesture name, or raise UnknownGesture."""
    key = gesture.strip().lower()
    try:
        return GESTURE_ACTION_ID[key]
    except KeyError as exc:
        known = ", ".join(GESTURE_ACTION_ID)
        raise UnknownGesture(f"unknown gesture {gesture!r}; expected one of: {known}") from exc
