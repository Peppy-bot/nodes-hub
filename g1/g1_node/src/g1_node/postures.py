"""Posture vocabulary, FSM ordering guard, and LocoClient method mapping.

Parse a wire posture string into a `Posture` once at the boundary; everything
downstream works with the enum. The ordering guard encodes the SDK's hard
requirement (Damp before StandUp before Start) as data, so an out-of-order goal
is rejected rather than sent to the robot.
"""

from __future__ import annotations

from enum import Enum


class Posture(Enum):
    DAMP = "damp"
    STAND_UP = "stand_up"
    START = "start"
    SIT = "sit"
    LOW_STAND = "low_stand"
    HIGH_STAND = "high_stand"
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


# Which postures may immediately precede a given target. `None` means "any state,
# including a fresh boot" (Damp is always safe; the settled stands/sit are only
# reachable from a powered stand). Encodes the SDK's Damp -> StandUp -> Start rule.
_PRECONDITIONS: dict[Posture, frozenset[Posture] | None] = {
    Posture.DAMP: None,
    Posture.ZERO_TORQUE: frozenset({Posture.DAMP}),
    Posture.STAND_UP: frozenset({Posture.DAMP}),
    Posture.START: frozenset({Posture.STAND_UP}),
    Posture.SIT: frozenset({Posture.STAND_UP, Posture.START}),
    Posture.LOW_STAND: frozenset({Posture.STAND_UP, Posture.START}),
    Posture.HIGH_STAND: frozenset({Posture.STAND_UP, Posture.START}),
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
# against a real robot is a one-line fix, not a scattered edit.
POSTURE_METHOD: dict[Posture, str] = {
    Posture.DAMP: "Damp",
    Posture.STAND_UP: "StandUp",
    Posture.START: "Start",
    Posture.SIT: "Sit",
    Posture.LOW_STAND: "LowStand",
    Posture.HIGH_STAND: "HighStand",
    Posture.ZERO_TORQUE: "ZeroTorque",
}


# Best-effort FSM id each posture settles into, used for the action result until
# the robot's own state channel reports the live id. Values follow the SDK's
# documented ids (Start -> 200); the sim backends reuse this map for parity.
POSTURE_FSM_ID: dict[Posture, int] = {
    Posture.DAMP: 1,
    Posture.ZERO_TORQUE: 0,
    Posture.STAND_UP: 4,
    Posture.START: 200,
    Posture.SIT: 3,
    Posture.LOW_STAND: 4,
    Posture.HIGH_STAND: 4,
}
