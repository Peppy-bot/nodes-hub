"""Operator alerts, drawn onto the status panel and the camera tracks.

The headset page is teleop_xr's, with a fixed server-to-client vocabulary and
no channel for our text, so the only surfaces reaching the operator's eyes are
the video tracks. Every active alert is listed on the status panel, which the
operator can look away from; a banner burned across the camera frames cannot
be looked away from, so it is reserved for the severities that must interrupt.
Where that line falls is the caller's to set, because a stack that publishes
no status panel has nowhere else to say it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

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

# Banner geometry as a fraction of frame height, so every camera resolution
# gets the same visual weight.
_BANNER_HEIGHT_FRACTION = 0.14
_TEXT_HEIGHT_FRACTION = 0.075
_MARGIN_FRACTION = 0.02

# BGR banner colors per severity: amber for a warning, red above it.
_BANNER_COLORS = {
    WARNING: (16, 116, 168),
    CRITICAL: (34, 126, 230),
    FAULT: (53, 53, 155),
}
_TEXT_COLOR = (255, 255, 255)


@dataclass(frozen=True)
class Alert:
    """One active alert: its severity, operator text, and arrival time."""

    severity: int
    text: str
    received_monotonic_s: float


class ActiveAlerts:
    """Latest alert per (source, kind), loop-confined: the listener writes and
    the panel and every camera drain read, with no await between a read and
    its purge.

    A severity-0 message removes its entry, so recovery clears the alert as
    directly as it was raised. Severities above the contract's ceiling are
    refused (ValueError), as is an alert with no identity, matching the
    commander's boundary so a malformed producer can neither outrank a
    genuine fault nor render as an unnamed one.

    `banner_from` is the lowest severity worth interrupting the video for.
    """

    def __init__(self, *, banner_from: int, producers_bound: bool = True) -> None:
        if banner_from not in _LEVEL_LABELS:
            raise ValueError(f"undefined banner severity {banner_from}")
        self._banner_from = banner_from
        self._producers_bound = producers_bound
        self._by_identity: dict[tuple[str, str], Alert] = {}

    @property
    def producers_bound(self) -> bool:
        """Whether anything is wired to the alert slot.

        The slot is zero_or_more, so an unwired stack receives nothing and
        looks exactly like a healthy one. A surface that renders silence has
        to be able to say which of the two it is showing.
        """
        return self._producers_bound

    def update(self, source: str, kind: str, severity: int, message: str) -> None:
        if not source or not kind:
            raise ValueError("alert identity needs a source and a kind")
        if severity != CLEAR and severity not in _LEVEL_LABELS:
            raise ValueError(f"undefined severity {severity}")
        if severity == CLEAR:
            self._by_identity.pop((source, kind), None)
            return
        self._by_identity[(source, kind)] = Alert(
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

    def banner(self) -> tuple[str, int] | None:
        """The text and severity to burn into the video, or None while nothing
        active is severe enough to interrupt it.

        Names the worst alert and counts the rest. On a stack with no status
        panel this is the operator's only channel, so one dead joint and a
        whole dead arm must not read the same.
        """
        live = self.active()
        if not live or live[0].severity < self._banner_from:
            return None
        others = len(live) - 1
        text = live[0].text if not others else f"{live[0].text}  (+{others} more)"
        return text, live[0].severity


# Below this fraction of the intended scale the glyphs stop surviving VP8,
# which blurs fine strokes. Past it the text is truncated rather than shrunk
# further: a readable prefix naming the joint beats an unreadable smear of
# the whole message.
_MIN_SCALE_FRACTION = 0.55
_ELLIPSIS = "..."


# Memoized: draw_banner refits the same text every camera frame while an
# alert stands, and the truncation branch measures once per dropped
# character. The result is pure in the arguments, so a standing banner
# costs one cache lookup per frame instead.
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


def draw_banner(frame: np.ndarray, text: str, severity: int) -> np.ndarray:
    """`frame` with the alert banner across its top, drawn in place.

    The frame is the caller's own decoded array (never the recycled wire
    buffer), so drawing in place is safe and avoids a copy per frame.
    """
    height, width = frame.shape[:2]
    banner_height = max(1, int(height * _BANNER_HEIGHT_FRACTION))
    cv2.rectangle(
        frame, (0, 0), (width, banner_height), _BANNER_COLORS[severity], thickness=-1
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = cv2.getFontScaleFromHeight(
        font, max(1, int(height * _TEXT_HEIGHT_FRACTION)), thickness=2
    )
    margin = max(1, int(width * _MARGIN_FRACTION))
    text, scale = fit_text(text, font, scale, 2, width - 2 * margin)
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, 2)
    origin = ((width - text_width) // 2, (banner_height + text_height) // 2)
    cv2.putText(frame, text, origin, font, scale, _TEXT_COLOR, 2, cv2.LINE_AA)
    return frame


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
                message.source, message.kind, message.severity, message.message
            )
            latch.clear()
        except Exception as e:
            latch.trip(f"alert unusable from {producer.instance_id}: {e!r}")
    log("alerts stream ended")
