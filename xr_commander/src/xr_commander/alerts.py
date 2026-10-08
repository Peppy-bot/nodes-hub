"""Operator alerts, listed on the status panel.

The headset page is teleop_xr's, with a fixed server-to-client vocabulary and
no channel for our text, so the alerts reach the operator through the status
panel track alongside the camera views.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache

import cv2

from xr_commander.bus import CancellationToken, Latch, log, messages, ticks

# Severity encoding of the alert contract: warning < critical < fault. A
# listed alert is active; a set that omits one clears it.
WARNING = 1
CRITICAL = 2
FAULT = 3

_LEVEL_LABELS = {WARNING: "WARNING", CRITICAL: "CRITICAL", FAULT: "FAULT"}

# A producer's set ages out this long after arrival: three times the
# contract's 2000 ms cadence ceiling, so a set survives two dropped
# re-publishes but not a producer that went quiet. An aged-out set means the
# producer's conditions are unknown, not cleared; only an omitted entry says
# cleared.
ALERT_STALE_AFTER_MS = 6000

# How often the slot's membership is read, so a producer that leaves takes
# its alerts with it sooner than the aging window would.
MEMBERSHIP_POLL_S = 0.5


@dataclass(frozen=True)
class Alert:
    """One active alert: its severity and operator text."""

    severity: int
    text: str


@dataclass(frozen=True)
class _ProducerSet:
    """One producer's active set and when it arrived."""

    alerts: tuple[Alert, ...]
    received_monotonic_s: float


class ActiveAlerts:
    """Each producer's latest alert set, loop-confined: the listener and the
    membership poll write, and the panel reads.

    A producer is addressed by the (core node, instance id) pair the wire
    gives, which is unique across the mesh where an instance id alone is
    unique only within one stack. A producer's set replaces its own entry and
    nobody else's, so no producer can clear another's alert through the wire
    strings.

    A set that omits an alert clears it. A set goes when its producer leaves
    the slot, and ages out after `ALERT_STALE_AFTER_MS` if the producer stops
    re-publishing. An entry with no identity, a severity outside the
    contract, or an identity the same message already carries is dropped and
    the rest of the set stands: a producer's one bad entry must not blank the
    motors it reported correctly.
    """

    def __init__(
        self,
        *,
        bound_now: Callable[[], Iterable[tuple[str, str]]],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._bound_now = bound_now
        self._monotonic = monotonic
        self._by_producer: dict[tuple[str, str], _ProducerSet] = {}
        self._bound: set[tuple[str, str]] = set(bound_now())

    @property
    def producers_bound(self) -> bool:
        """Whether anything is wired to the alert slot.

        The slot is zero_or_more, so an unwired stack receives nothing and
        looks exactly like a healthy one. A surface that renders silence has
        to be able to say which of the two it is showing. Follows the slot,
        so a producer that joins or leaves after start-up moves it.
        """
        return bool(self._bound)

    def replace(self, producer: tuple[str, str], active: Iterable[object]) -> list[str]:
        """Take `producer`'s set from the wire message's `active` entries.

        Answers the reasons any entry was dropped, for the caller to log.
        """
        alerts: list[Alert] = []
        seen: set[tuple[str, str]] = set()
        refused: list[str] = []
        for item in active:
            source, kind = item.source, item.kind
            if not source or not kind:
                refused.append("alert identity needs a source and a kind")
                continue
            if item.severity not in _LEVEL_LABELS:
                refused.append(f"undefined severity {item.severity}")
                continue
            if (source, kind) in seen:
                refused.append(f"{source!r} {kind!r} listed twice")
                continue
            seen.add((source, kind))
            alerts.append(
                Alert(
                    severity=item.severity,
                    text=f"{source.upper()} {_LEVEL_LABELS[item.severity]}: {item.message}",
                )
            )
        self._by_producer[producer] = _ProducerSet(
            alerts=tuple(alerts), received_monotonic_s=self._monotonic()
        )
        return refused

    def retain(self, producers: Iterable[tuple[str, str]]) -> None:
        """Keep the sets of the producers the slot holds, and record them."""
        self._bound = set(producers)
        self._by_producer = {
            producer: held
            for producer, held in self._by_producer.items()
            if producer in self._bound
        }

    def active(self) -> tuple[Alert, ...]:
        """Every live alert, worst first, ordered so equal severities keep a
        stable place on the panel between re-draws.

        A producer that stopped re-publishing is purged here, so its last set
        cannot stay on screen as current.
        """
        oldest = self._monotonic() - ALERT_STALE_AFTER_MS / 1000.0
        self._by_producer = {
            producer: held
            for producer, held in self._by_producer.items()
            if held.received_monotonic_s >= oldest
        }
        # Ordered by severity, then by producer, then by the order the
        # producer listed its own alerts in. Never by the rendered text,
        # which carries a measurement that moves between re-publishes.
        ordered = sorted(
            (
                (-alert.severity, producer, index, alert)
                for producer, held in self._by_producer.items()
                for index, alert in enumerate(held.alerts)
            ),
            key=lambda row: row[:3],
        )
        return tuple(row[3] for row in ordered)


# Below this fraction of the intended scale the glyphs stop surviving VP8,
# which blurs fine strokes. Past it the text is truncated rather than shrunk
# further: a readable prefix naming the joint beats an unreadable smear of
# the whole message.
_MIN_SCALE_FRACTION = 0.55
_ELLIPSIS = "..."


# Memoized: the status panel refits its rows on every redraw, and the
# truncation branch measures once per dropped character. The result is
# pure in the arguments, so a re-drawn row costs one cache lookup.
@lru_cache(maxsize=256)
def fit_text(
    text: str, font: int, scale: float, thickness: int, available_px: int
) -> tuple[str, float]:
    """`text` and the scale to draw it at, fitted inside `available_px`.

    Shrinks first, then truncates once shrinking would cost legibility.
    """
    if available_px <= 0:
        return "", scale
    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= available_px:
        return text, scale
    floor = scale * _MIN_SCALE_FRACTION
    if cv2.getTextSize(text, font, floor, thickness)[0][0] > available_px:
        # Even the legibility floor cannot hold the whole text: keep the
        # floor and drop characters off the end instead.
        kept = text
        while kept and (
            cv2.getTextSize(kept + _ELLIPSIS, font, floor, thickness)[0][0]
            > available_px
        ):
            kept = kept[:-1]
        return (kept + _ELLIPSIS if kept else ""), floor
    # The whole text fits at the floor but not at full scale, so the largest
    # fitting scale lies between them. Rendered widths are a stair-stepped
    # function of scale (glyph widths round up), which defeats proportional
    # estimation near the boundary; bisection pins it in a dozen passes.
    lo, hi = floor, scale
    for _ in range(12):
        mid = (lo + hi) / 2.0
        if cv2.getTextSize(text, font, mid, thickness)[0][0] <= available_px:
            lo = mid
        else:
            hi = mid
    return text, lo



async def drain_alerts(
    node_runner,
    topic_module,
    active: ActiveAlerts,
    token: CancellationToken,
) -> None:
    """Keep `active` at every producer's newest alert set."""
    try:
        subscription = await topic_module.subscribe(node_runner)
    except Exception as e:
        # Loud and fail-safe, matching the camera drains: no alerts rather
        # than a task that dies and takes its failure with it.
        log(f"alerts subscribe failed: {e!r}")
        return
    # Latched per producer: a producer's refusal logs once, until a set from
    # that same producer parses clean.
    unusable: dict[tuple[str, str], Latch] = {}
    async for producer, message in messages(subscription, token, "alerts"):
        key = (producer.core_node, producer.instance_id)
        latch = unusable.get(key)
        if latch is None:
            latch = unusable[key] = Latch()
        try:
            refused = active.replace(key, message.active)
        except Exception as e:
            latch.trip(f"alert set unusable from {key[1]}: {e!r}")
            continue
        if refused:
            latch.trip(f"alert entry dropped from {key[1]}: {refused[0]}")
        else:
            latch.clear()
    log("alerts stream ended")


async def follow_producers(
    node_runner,
    topic_module,
    active: ActiveAlerts,
    token: CancellationToken,
    *,
    period_s: float = MEMBERSHIP_POLL_S,
) -> None:
    """Keep `active` at the producers the alert slot holds.

    The generated slot API answers who is bound and offers no change stream,
    so this polls it. It runs in its own task, so the panel's frame loop
    never pays for the call or carries its failure, and a producer that
    leaves without publishing again is noticed before its set ages out.
    """
    failing = Latch()
    async for _ in ticks(period_s, token):
        try:
            active.retain(
                (p.core_node, p.instance_id)
                for p in topic_module.bound_producers(node_runner)
            )
            failing.clear()
        except BaseException as e:  # a slot the manifest lost raises through pyo3
            failing.trip(f"alerts bound set unreadable: {e!r}")
    log("alerts membership poll ended")
