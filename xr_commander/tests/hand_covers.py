"""The interruptions a grip's tick reads and the let-gos the world notes, as
the traces' deadman covers the hands by them: a hand
whose squeeze an interruption covers counts for no grip until it reads
released."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from xr_commander.devices import Hand
from xr_commander.grip_world import (
    ROBOT_NOTE,
    Effect,
    Engaged,
    Lingering,
    Note,
    Phase,
    Pins,
    Reading,
    Requesting,
)

# The robot row's note at a let-go of a robot the world held: `let go of
# <robot>: <why>`.
LET_GO_NOTE = re.compile(r"let go of (?P<robot>[^:]+): ")


@dataclass(frozen=True)
class Choice:
    """The chosen robot the grips' ticks saw last, and whether one ever
    was."""

    seen: str | None = None
    had_chosen: bool = False


def pins_lost(phase: Phase, peers: Pins, pins: Pins) -> bool:
    """Whether a tick that reads `peers` on the grip's slots and the chosen
    robot's `pins` loses a pin of the grip's `phase`: a pin of an engaged or
    lingering phase left its slot, or a requesting phase's tick reads no
    pin."""
    match phase:
        case Engaged(pins=driven) | Lingering(pins=driven):
            return not driven <= peers
        case Requesting():
            return not pins
    return False


def tick_covers(
    grip_hands: Iterable[Hand],
    readings: Mapping[Hand, Reading],
    chosen: str | None,
    choice: Choice,
    *,
    down: bool,
    lost: bool,
) -> tuple[frozenset[Hand], Choice]:
    """The hands a tick of a grip with `grip_hands` covers, and the choice
    after it: each hand of the grip that reads a squeeze while the stack is
    halted or the goals do not run (`down`); both hands at a change of the
    chosen robot after one was chosen; and the grip's hands when the tick
    loses a pin of its phase (`lost`)."""
    hands = frozenset(grip_hands)
    squeezed = frozenset(h for h in hands if readings[h] is Reading.SQUEEZE)
    covered = squeezed if down else frozenset()
    if lost:
        covered |= hands
    if chosen == choice.seen:
        return covered, choice
    if choice.had_chosen:
        covered |= frozenset(Hand)
    return covered, Choice(chosen, choice.had_chosen or chosen is not None)


def noted_let_go(effect: Effect) -> str | None:
    """The robot whose let-go `effect` notes, None for any other effect: the
    world sets the robot row's note `let go of <robot>: <why>` in the step
    that lets go of a robot it held."""
    if not isinstance(effect, Note) or effect.target != ROBOT_NOTE:
        return None
    noted = LET_GO_NOTE.match(effect.text)
    return None if noted is None else noted["robot"]


def let_go_covers(robot: str, choice: Choice) -> frozenset[Hand]:
    """The hands a let-go of `robot` covers: both hands when `robot` is the
    chosen robot the grips' ticks saw last, else none."""
    return frozenset(Hand) if robot == choice.seen else frozenset()
