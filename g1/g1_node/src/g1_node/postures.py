"""Posture vocabulary, FSM ordering guard, and LocoClient method mapping.

Parse a wire posture string into a `Posture` once at the boundary; everything
downstream works with the enum. The ordering guard encodes the SDK's hard
requirement (damp before standing before locomotion) as data, so an out-of-order
goal is rejected rather than sent to the robot.
"""

from __future__ import annotations

from enum import Enum


class Posture(Enum):
    DAMP = "damp"
    SQUAT_TO_STAND = "squat_to_stand"
    LIE_TO_STAND = "lie_to_stand"
    START = "start"
    SIT = "sit"
    STAND_TO_SQUAT = "stand_to_squat"
    HIGH_STAND = "high_stand"
    LOW_STAND = "low_stand"
    ZERO_TORQUE = "zero_torque"


class UnknownPosture(ValueError):
    """Raised when a wire string does not name a known posture."""


class TransitionRejected(ValueError):
    """Raised when a parsed posture is not reachable from the current state."""


def parse_posture(raw: str) -> Posture:
    """Parse a wire string into a Posture, or raise UnknownPosture."""
    try:
        return Posture(raw.strip().lower())
    except ValueError as exc:
        known = ", ".join(p.value for p in Posture)
        raise UnknownPosture(f"unknown posture {raw!r}; expected one of: {known}") from exc


# Standing configurations: any of these means the robot is upright and can take
# a locomotion, sit, squat, or stance-height transition next.
_STANDING: frozenset[Posture] = frozenset(
    {
        Posture.SQUAT_TO_STAND,
        Posture.LIE_TO_STAND,
        Posture.HIGH_STAND,
        Posture.LOW_STAND,
        Posture.START,
    }
)

# Which postures may immediately precede a target. `None` means "any state,
# including a fresh boot" (Damp is always safe). Encodes the SDK's ordering:
# damp -> stand -> {locomotion, sit, squat, stance height}.
_PRECONDITIONS: dict[Posture, frozenset[Posture] | None] = {
    Posture.DAMP: None,
    Posture.ZERO_TORQUE: frozenset({Posture.DAMP}),
    Posture.SQUAT_TO_STAND: frozenset({Posture.DAMP}),
    Posture.LIE_TO_STAND: frozenset({Posture.DAMP}),
    Posture.START: _STANDING,
    Posture.SIT: _STANDING,
    Posture.STAND_TO_SQUAT: _STANDING,
    Posture.HIGH_STAND: _STANDING,
    Posture.LOW_STAND: _STANDING,
}


def is_transition_allowed(current: Posture | None, target: Posture) -> bool:
    """True if `target` may be commanded from `current` (None = fresh boot)."""
    allowed_from = _PRECONDITIONS[target]
    if allowed_from is None:
        return True
    return current in allowed_from


def plan_transition(current: Posture | None, raw: str) -> Posture:
    """Parse and order-check a wire posture in one step.

    Returns the target posture, or raises UnknownPosture / TransitionRejected
    (both ValueError) so the caller can reject the goal with the message.
    """
    target = parse_posture(raw)
    if not is_transition_allowed(current, target):
        origin = current.value if current is not None else "boot"
        raise TransitionRejected(f"cannot go {origin} -> {target.value}")
    return target


# Posture -> LocoClient method name. Centralized so a method-name mismatch found
# against a real robot is a one-line fix. Note the SDK has no StandUp(): standing
# up is Squat2StandUp() or Lie2StandUp() depending on the start pose.
POSTURE_METHOD: dict[Posture, str] = {
    Posture.DAMP: "Damp",
    Posture.SQUAT_TO_STAND: "Squat2StandUp",
    Posture.LIE_TO_STAND: "Lie2StandUp",
    Posture.START: "Start",
    Posture.SIT: "Sit",
    Posture.STAND_TO_SQUAT: "StandUp2Squat",
    Posture.HIGH_STAND: "HighStand",
    Posture.LOW_STAND: "LowStand",
    Posture.ZERO_TORQUE: "ZeroTorque",
}


# Best-effort FSM id each posture settles into, used for the action result when
# the live GetFsmId poll has not yet reported. Start -> 200 (active locomotion).
POSTURE_FSM_ID: dict[Posture, int] = {
    Posture.DAMP: 1,
    Posture.ZERO_TORQUE: 0,
    Posture.SQUAT_TO_STAND: 4,
    Posture.LIE_TO_STAND: 4,
    Posture.START: 200,
    Posture.SIT: 3,
    Posture.STAND_TO_SQUAT: 2,
    Posture.HIGH_STAND: 4,
    Posture.LOW_STAND: 4,
}
