"""Operator alerts, listed on the status panel.

The headset page is teleop_xr's, with a fixed server-to-client vocabulary and
no channel for our text, so the alerts reach the operator through the status
panel track alongside the camera views.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import cv2

from xr_commander.bus import CancellationToken, Latch, log, messages

# Severity encoding of the alert contract: 0 clears, then warning < critical
# < fault.
CLEAR = 0
WARNING = 1
CRITICAL = 2
FAULT = 3

_LEVEL_LABELS = {WARNING: "WARNING", CRITICAL: "CRITICAL", FAULT: "FAULT"}

# Alerts age out this long after arrival: 3x the contract's 2000 ms re-emit
# ceiling, so an alert survives two dropped re-emits but not a dead producer.
# An aged-out alert means the producer went quiet and its condition is
# unknown, not cleared; only a severity-0 message clears.
ALERT_STALE_AFTER_MS = 6000

@dataclass(frozen=True)
class Alert:
    """One active alert: its severity, operator text, and arrival time."""

    severity: int
    text: str
    received_monotonic_s: float


class ActiveAlerts:
    """Latest alert per (producer, source, kind), loop-confined: the listener
    writes and the panel reads, with no await between a read and its purge.
    The producer is the transport-authenticated instance, so no producer can
    replace or clear another's alert through the wire strings.

    A severity-0 message removes its entry, so recovery clears the alert as
    directly as it was raised. Severities above the contract's ceiling are
    refused (ValueError), as is an alert with no identity, matching the
    commander's boundary so a malformed producer can neither outrank a
    genuine fault nor render as an unnamed one.
    """

    def __init__(self, *, producers_bound: bool = True) -> None:
        self._producers_bound = producers_bound
        self._by_identity: dict[tuple[str, str, str], Alert] = {}

    @property
    def producers_bound(self) -> bool:
        """Whether anything is wired to the alert slot.

        The slot is zero_or_more, so an unwired stack receives nothing and
        looks exactly like a healthy one. A surface that renders silence has
        to be able to say which of the two it is showing.
        """
        return self._producers_bound

    def update(
        self, producer: str, source: str, kind: str, severity: int, message: str
    ) -> None:
        if not source or not kind:
            raise ValueError("alert identity needs a source and a kind")
        if severity != CLEAR and severity not in _LEVEL_LABELS:
            raise ValueError(f"undefined severity {severity}")
        if severity == CLEAR:
            self._by_identity.pop((producer, source, kind), None)
            return
        self._by_identity[(producer, source, kind)] = Alert(
            severity=severity,
            text=f"{source.upper()} {_LEVEL_LABELS[severity]}: {message}",
            received_monotonic_s=time.monotonic(),
        )

    def active(self) -> tuple[Alert, ...]:
        """Every live alert, worst first, ordered so equal severities keep a
        stable place on the panel rather than swapping between re-draws.

        Stale entries are purged here: a producer that stopped re-emitting
        cannot leave its alert on screen, and the map stays bounded when
        sources vary.
        """
        now = time.monotonic()
        self._by_identity = {
            identity: alert
            for identity, alert in self._by_identity.items()
            if now - alert.received_monotonic_s <= ALERT_STALE_AFTER_MS / 1000.0
        }
        # Ordered by severity then by identity, never by the rendered text:
        # the text carries a live measurement the producer re-emits, so
        # sorting on it swaps two equal-severity rows whenever a reading
        # ticks.
        return tuple(
            alert
            for _identity, alert in sorted(
                self._by_identity.items(),
                key=lambda item: (-item[1].severity, item[0]),
            )
        )


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
    """Keep `active` at every producer's newest alerts."""
    try:
        subscription = await topic_module.subscribe(node_runner)
    except Exception as e:
        # Loud and fail-safe, matching the camera drains: no alerts rather
        # than a task that dies and takes its failure with it.
        log(f"alerts subscribe failed: {e!r}")
        return
    # Latched per producer: a malformed producer repeats every re-emit, and
    # a shared latch would be cleared by any other producer's good message,
    # so one bad arm would log on every re-emit forever.
    unusable: dict[str, Latch] = {}
    async for producer, message in messages(subscription, token, "alerts"):
        latch = unusable.get(producer.instance_id)
        if latch is None:
            latch = unusable[producer.instance_id] = Latch()
        try:
            active.update(
                producer.instance_id,
                message.source,
                message.kind,
                message.severity,
                message.message,
            )
            latch.clear()
        except Exception as e:
            latch.trip(f"alert unusable from {producer.instance_id}: {e!r}")
    log("alerts stream ended")
