"""The rule of a grip stream on one slot, as the stream traces hold the
streams to it. Each send goes to the pin of the grip's phase
after its tick's step. A target drives that pin, and only while the phase is
engaged. At the first tick with no target after driving, a burst of the value
held then starts, to the pin last driven, when there is a value; each tick of
the burst sends it while its pair is in the slot, and the burst ends after
its send once it sent `BURST_VALUES` values and the stream's burst time
passed since its first tick, or at the tick its pair left. The stream drives
another pin only once its burst to the pin it drove ended or that pin left.
A pair first in delivered order since it joined ahead gets nothing until an
engagement takes it."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

from xr_commander.grip_world import Engaged, Phase, Pins, Slot, driven_pins, pin_on

# The values a burst sends at least, and the time it runs at least (s), on
# the host's clock.
BURST_VALUES = 3
BURST_S = 0.25

# What a release burst holds: an arm's pose, or a base's velocity.
Held = TypeVar("Held")


@dataclass(frozen=True)
class Burst(Generic[Held]):
    """A release burst: its pair, the value it holds, its first tick on the
    host's clock, and the values it sent."""

    pin: Hashable
    value: Held
    started_s: float
    sent: int = 0


@dataclass(frozen=True)
class StreamTick(Generic[Held]):
    """One tick of a stream by the rule: the stream after it, what it sends,
    None for silence, whether that is a burst value, and each clause of the
    rule the tick's phase and target break."""

    model: StreamModel[Held]
    send: tuple[Hashable, Held] | None
    burst_sent: bool
    broken: tuple[str, ...]


@dataclass(frozen=True)
class StreamModel(Generic[Held]):
    """A grip stream on `slot` by the rule, whose bursts run `burst_s` at
    least: the pin last driven, whether its last tick drove it, its burst,
    and the pairs that joined ahead."""

    slot: Slot
    burst_s: float
    driven: Hashable | None = None
    driving: bool = False
    burst: Burst[Held] | None = None
    joined_ahead: frozenset[Hashable] = frozenset()

    def __post_init__(self) -> None:
        assert self.driven is not None or not self.driving, "a target has a pin"
        assert self.burst is None or not self.driving, (
            "a burst starts once the driving stops"
        )
        assert self.burst is None or self.burst.pin == self.driven, (
            "the burst goes to the pin last driven"
        )

    def with_join_ahead(self, peer: Hashable) -> StreamModel[Held]:
        """The stream once `peer` joined first in delivered order."""
        return replace(self, joined_ahead=self.joined_ahead | {peer})

    def engaged(self, pins: Pins) -> StreamModel[Held]:
        """The stream once its grip engaged on `pins`."""
        taken = frozenset(pin.peer for pin in pins if pin.slot is self.slot)
        return replace(self, joined_ahead=self.joined_ahead - taken)

    def stepped(
        self,
        phase: Phase,
        *,
        target: Held | None,
        held: Held | None,
        peers: frozenset[Hashable],
        now_s: float,
    ) -> StreamTick[Held]:
        """The tick whose phase after its step is `phase`, with `target` to
        drive, `held` for a burst and `peers` on the slot, at `now_s`."""
        pin = pin_on(driven_pins(phase), self.slot)
        broken = self._target_broken(phase, pin, target, held, peers)
        if target is not None and isinstance(phase, Engaged) and pin is not None:
            driving = replace(self, driven=pin, driving=True, burst=None)
            return StreamTick(driving, (pin, target), False, broken)
        burst = self.burst
        if self.driving:
            burst = None if held is None else Burst(self.driven, held, now_s)
        resting = replace(self, driving=False, burst=None)
        if burst is None or burst.pin not in peers:
            return StreamTick(resting, None, False, broken)
        sent = burst.sent + 1
        ended = sent >= BURST_VALUES and now_s - burst.started_s >= self.burst_s
        model = resting if ended else replace(resting, burst=replace(burst, sent=sent))
        send = (burst.pin, burst.value)
        return StreamTick(model, send, True, broken + self._send_broken(send, pin))

    def _target_broken(
        self,
        phase: Phase,
        pin: Hashable | None,
        target: Held | None,
        held: Held | None,
        peers: frozenset[Hashable],
    ) -> tuple[str, ...]:
        """The clauses a tick's target breaks: it drives only an engaged
        phase's pin, and another pin than the one driven only once the burst
        to that one ended or that one left."""
        if target is None:
            return ()
        if not isinstance(phase, Engaged) or pin is None:
            return (f"the pins: a target {target!r} with the phase {phase}",)
        due = self.burst is not None or (self.driving and held is not None)
        old = self.driven
        if old is not None and old != pin and old in peers and due:
            return (
                f"the pins: the stream drives {pin!r} while its burst to {old!r},"
                f" a pair of the slot, runs or is due",
            )
        return self._send_broken((pin, target), pin)

    def _send_broken(
        self, send: tuple[Hashable, Held], pin: Hashable | None
    ) -> tuple[str, ...]:
        """The clauses a send breaks: it goes to the phase's pin, and to no
        pair that joined ahead."""
        broken: tuple[str, ...] = ()
        if send[0] != pin:
            broken += (f"the pins: {send!r} goes past the pin {pin!r}",)
        if send[0] in self.joined_ahead:
            broken += (f"the pins: {send!r} goes to a pair that joined ahead",)
        return broken
