"""The in-headset status panel: node state drawn onto a synthetic track.

A headset has no console, so anything the node only logs is invisible to the
operator wearing it. This carries the state that decides whether a hand can
drive at all, the alerts the robot is raising about itself, and each motor's
health. teleop_xr renders every declared view as a floating panel, so nothing
in the web stack needs to know this one is not a camera.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from xr_commander import motor_health
from xr_commander.alerts import (
    CRITICAL,
    FAULT,
    WARNING,
    ActiveAlerts,
    Alert,
    fit_text,
)
from xr_commander.bus import CancellationToken, Latch, ticks
from xr_commander.clutch import HandClutch
from xr_commander.config import Settings
from xr_commander.devices import FrameSource, fresh_frame
from xr_commander.publish import LatestPose
from xr_commander.record import RecorderStatus
from xr_commander.video import FrameSink

TRACK_ID = "status"

# The track's timestamps advance a fixed 1/30 s per delivered frame, so the
# panel is published at that rate whatever it says; the content is re-drawn
# only when it changes, which is a few times a session.
_PUBLISH_HZ = 30.0

# The headset stretches this track into its fixed panel quad whatever the
# frame shape, so the canvas keeps the camera tracks' 16:9 aspect and the
# sections budget its height; the type stays large because VP8 blurs fine
# strokes at arm's length.
_WIDTH = 1024
_HEIGHT = 576
_MARGIN_X = 36
_STATE_X = 340
_TITLE_BASELINE = 48
_LINK_BASELINE = 92
_FIRST_ROW_BASELINE = 148
_ROW_STEP = 44

# One row per hand plus the recorder's. The alert section sits below the space
# they can occupy, so it does not shift when the recorder binds mid-session.
_MAX_STATUS_ROWS = 3
_RULE_Y = _FIRST_ROW_BASELINE + _MAX_STATUS_ROWS * _ROW_STEP - 20
_RULE_HEIGHT = 2
_ALERTS_TITLE_BASELINE = _RULE_Y + 36
_FIRST_ALERT_BASELINE = _ALERTS_TITLE_BASELINE + 34
_ALERT_STEP = 32
# What the alert band holds; further alerts collapse into the last line.
_MAX_ALERT_LINES = 2
# Room below a baseline for the glyphs that hang under it.
_DESCENDER = 12

# The motor health section sits below the alert band's tallest case.
_MOTORS_RULE_Y = (
    _FIRST_ALERT_BASELINE + (_MAX_ALERT_LINES - 1) * _ALERT_STEP + _DESCENDER + 8
)
_MOTORS_TITLE_BASELINE = _MOTORS_RULE_Y + 30
_FIRST_HEALTH_BASELINE = _MOTORS_TITLE_BASELINE + 32
_HEALTH_STEP = 30
# A collapsed line per component plus a detail row; further rows fold into
# the last line.
_MAX_HEALTH_LINES = 5

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TITLE_SCALE = 1.1
_BODY_SCALE = 0.9
_ALERT_SCALE = 0.7
_HEALTH_SCALE = 0.7
_STROKE = 2

_BACKGROUND = (24, 24, 24)
_HEADING = (210, 210, 210)
_RULE = (70, 70, 70)
# BGR: green drives, amber waits, grey rests, red records.
_DRIVING = (120, 220, 120)
_WAITING = (60, 200, 250)
_RESTING = (170, 170, 170)
_RECORDING = (70, 70, 235)
_CRITICAL_RED = (70, 70, 235)
_FAULT_RED = (90, 90, 255)
# Magenta for a producer or motor gone quiet: its state is unknown, which is
# neither healthy grey nor alert red.
_NOT_REPORTING_MAGENTA = (230, 120, 230)
# Alert text by severity: amber warns, red demands a stop.
_ALERT_COLOURS = {WARNING: _WAITING, CRITICAL: _CRITICAL_RED, FAULT: _FAULT_RED}
# Health rows by level, sharing the alert palette from warning up; a None
# level is a source-name heading.
_HEALTH_COLOURS = {
    motor_health.NOMINAL: _RESTING,
    motor_health.WARNING: _WAITING,
    motor_health.CRITICAL: _CRITICAL_RED,
    motor_health.FAULT: _FAULT_RED,
    motor_health.NOT_REPORTING: _NOT_REPORTING_MAGENTA,
    None: _HEADING,
}

assert (
    _FIRST_HEALTH_BASELINE + (_MAX_HEALTH_LINES - 1) * _HEALTH_STEP + _DESCENDER
    <= _HEIGHT
)


@dataclass(frozen=True)
class HandSource:
    """Everything the panel reads for one hand, in display order.

    Carried together so a hand cannot be described from another hand's clutch,
    and so an unbound arm is a state rather than a silent wait. arm_paired is
    read live each snapshot: at setup time the daemon may still be
    establishing the launcher's pairs, so a boot-time read races.
    """

    handedness: str
    clutch: HandClutch
    measured: LatestPose
    arm_paired: Callable[[], bool]


@dataclass(frozen=True)
class Row:
    """One hand's row: what it is doing, and the colour that says so."""

    label: str
    state: str
    colour: tuple[int, int, int]


@dataclass(frozen=True)
class PanelLine:
    """One list line as the panel says it, in the colour of its meaning."""

    text: str
    colour: tuple[int, int, int]


@dataclass(frozen=True)
class PanelState:
    """Everything drawn this tick; compared to decide whether to re-draw."""

    headset_live: bool
    hands: tuple[Row, ...]
    # Worst first, capped at what the canvas holds.
    alerts: tuple[PanelLine, ...] = ()
    # Whether anything is wired to the alert slot at all.
    alerts_bound: bool = True
    # Per-motor health in source order, capped at what the canvas holds.
    health: tuple[PanelLine, ...] = ()
    # Whether anything is wired to the motor_health slot at all.
    health_bound: bool = True
    # The recorder's row; None when no recorder is bound.
    recorder: Row | None = None
    # The task line as drawn; None when no recorder is bound. With one bound
    # it always draws, so an unset label is visible rather than absent.
    task: str | None = None


def alert_lines(active: Sequence[Alert]) -> tuple[PanelLine, ...]:
    """The alert section's lines, worst first.

    Past what the canvas holds the last line counts the remainder instead of
    dropping it silently, so the panel never understates how much is wrong.
    """
    if len(active) <= _MAX_ALERT_LINES:
        return tuple(PanelLine(a.text, _ALERT_COLOURS[a.severity]) for a in active)
    shown = active[: _MAX_ALERT_LINES - 1]
    hidden = active[_MAX_ALERT_LINES - 1 :]
    return tuple(PanelLine(a.text, _ALERT_COLOURS[a.severity]) for a in shown) + (
        PanelLine(f"+{len(hidden)} more", _ALERT_COLOURS[hidden[0].severity]),
    )


def health_lines(rows: Sequence[motor_health.HealthRow]) -> tuple[PanelLine, ...]:
    """The motor section's lines, in the store's source order.

    Past what the canvas holds the last line counts the remainder, wearing
    the highest hidden level so a buried fault still colours the panel.
    """
    lines = tuple(PanelLine(row.text, _HEALTH_COLOURS[row.level]) for row in rows)
    if len(lines) <= _MAX_HEALTH_LINES:
        return lines
    hidden = rows[_MAX_HEALTH_LINES - 1 :]
    worst = max((row.level for row in hidden if row.level is not None), default=None)
    return lines[: _MAX_HEALTH_LINES - 1] + (
        PanelLine(f"+{len(hidden)} more", _HEALTH_COLOURS[worst]),
    )


def hand_line(
    label: str,
    *,
    arm_bound: bool,
    tracked: bool,
    squeezing: bool,
    follower_fresh: bool,
    engaged: bool,
) -> Row:
    """Read one hand's row off the same conditions the clutch gates on, in the
    order the operator hits them: a bound arm, a controller, a grip, a
    follower, a snapshot. The earliest unmet one is what to fix.
    """
    if not arm_bound:
        return Row(label, "no arm bound", _RESTING)
    if not tracked:
        return Row(label, "no controller", _RESTING)
    if not squeezing:
        return Row(label, "grip released", _RESTING)
    if not follower_fresh:
        return Row(label, "waiting for arm", _WAITING)
    if not engaged:
        return Row(label, "re-anchoring", _WAITING)
    return Row(label, "driving", _DRIVING)


def recorder_row(status: RecorderStatus | None) -> Row | None:
    """The recorder's row: red while an episode runs, amber while it saves;
    None hides the row.

    Outcomes (saved, finished, refused) flash on the row for a few seconds;
    recording state stays visible through them.
    """
    if status is None:
        return None
    note = status.note()
    if status.saving:
        # The frame counter freezes here; without this the row reads as a
        # stuck recording for however long the save takes.
        return Row("REC", "saving...", _WAITING)
    if status.recording:
        state = f"recording {status.frames}"
        return Row("REC", f"{state}, {note}" if note else state, _RECORDING)
    if note:
        return Row("REC", note, _WAITING)
    if status.finish_held:
        return Row("REC", "finishing...", _WAITING)
    return Row("REC", "idle", _RESTING)


def task_line(label: str | None) -> str:
    """The task line for a bound recorder. An unset label reads as missing
    rather than showing the placeholder like a chosen one, so the operator
    can tell an unnamed session from a named one at a glance."""
    return f"task: {label}" if label is not None else "task: NOT SET (unnamed)"


def snapshot(
    session: FrameSource,
    hands: Sequence[HandSource],
    stale_timeout_s: float,
    alerts: ActiveAlerts,
    health: motor_health.MotorHealthReports,
    recorder: RecorderStatus | None = None,
    task: str | None = None,
) -> PanelState:
    """This tick's panel content, read from the sources the streams read."""
    frame = fresh_frame(session, stale_timeout_s)
    rows = []
    for hand in hands:
        sample = frame.hand(hand.handedness) if frame else None
        rows.append(
            hand_line(
                hand.handedness.upper(),
                arm_bound=hand.arm_paired(),
                tracked=sample is not None,
                squeezing=bool(sample and sample.squeezing),
                follower_fresh=hand.measured.fresh(stale_timeout_s) is not None,
                engaged=hand.clutch.engaged,
            )
        )
    recorder_line = recorder_row(recorder)
    assert len(rows) + (1 if recorder_line else 0) <= _MAX_STATUS_ROWS
    return PanelState(
        headset_live=frame is not None,
        hands=tuple(rows),
        alerts=alert_lines(alerts.active()),
        alerts_bound=alerts.producers_bound,
        health=health_lines(motor_health.health_rows(health.by_instance())),
        health_bound=health.producers_bound,
        recorder=recorder_line,
        # The recorder's presence is what decides the line exists; the label's
        # own absence is a state drawn on it.
        task=task_line(task) if recorder is not None else None,
    )


def _text(
    frame: np.ndarray,
    body: str,
    origin: tuple[int, int],
    scale: float,
    colour: tuple[int, int, int],
) -> None:
    cv2.putText(frame, body, origin, _FONT, scale, colour, _STROKE, cv2.LINE_AA)


def _fitted(body: str, scale: float, width: int) -> tuple[str, float]:
    """`body` and the scale to draw it at, fitted inside the margins.

    An alert that runs off the edge loses the joint or the number the operator
    needs, so long messages shrink before they lose their tail.
    """
    return fit_text(body, _FONT, scale, _STROKE, width)


def render(state: PanelState) -> np.ndarray:
    """The panel as one BGR frame, ready for the encoder."""
    frame = np.full((_HEIGHT, _WIDTH, 3), _BACKGROUND, dtype=np.uint8)
    _text(frame, "xr_commander", (_MARGIN_X, _TITLE_BASELINE), _TITLE_SCALE, _HEADING)
    live = state.headset_live
    link, colour = ("headset live", _DRIVING) if live else ("headset stale", _WAITING)
    _text(frame, link, (_MARGIN_X, _LINK_BASELINE), _BODY_SCALE, colour)
    # Session-wide context like the link state, so it shares that line rather
    # than a status row: the row band is full at _MAX_STATUS_ROWS, and the
    # sections below are laid out from it.
    if state.task is not None:
        fitted, scale = _fitted(state.task, _BODY_SCALE, _WIDTH - _MARGIN_X - _STATE_X)
        _text(frame, fitted, (_STATE_X, _LINK_BASELINE), scale, _HEADING)
    rows = state.hands + ((state.recorder,) if state.recorder else ())
    for index, row in enumerate(rows):
        baseline = _FIRST_ROW_BASELINE + index * _ROW_STEP
        _text(frame, row.label, (_MARGIN_X, baseline), _BODY_SCALE, _HEADING)
        _text(frame, row.state, (_STATE_X, baseline), _BODY_SCALE, row.colour)
    _render_alerts(frame, state.alerts, state.alerts_bound)
    _render_health(frame, state.health, state.health_bound)
    return frame


def _render_alerts(
    frame: np.ndarray, alerts: tuple[PanelLine, ...], bound: bool
) -> None:
    """The alert section, headed even when empty so a quiet robot is legible
    as quiet rather than as a panel that stopped drawing.

    An empty section says which kind of quiet it is. The slot is
    zero_or_more, so a stack that wired nothing receives nothing and would
    otherwise be indistinguishable from a robot with nothing to report.
    """
    cv2.rectangle(
        frame,
        (_MARGIN_X, _RULE_Y),
        (_WIDTH - _MARGIN_X, _RULE_Y + _RULE_HEIGHT),
        _RULE,
        thickness=-1,
    )
    _text(frame, "ALERTS", (_MARGIN_X, _ALERTS_TITLE_BASELINE), _BODY_SCALE, _HEADING)
    if not alerts:
        empty, colour = ("none", _RESTING) if bound else ("not wired", _WAITING)
        _text(frame, empty, (_STATE_X, _ALERTS_TITLE_BASELINE), _BODY_SCALE, colour)
        return
    usable = _WIDTH - 2 * _MARGIN_X
    for index, line in enumerate(alerts):
        baseline = _FIRST_ALERT_BASELINE + index * _ALERT_STEP
        body, scale = _fitted(line.text, _ALERT_SCALE, usable)
        _text(frame, body, (_MARGIN_X, baseline), scale, line.colour)


def _render_health(
    frame: np.ndarray, health: tuple[PanelLine, ...], bound: bool
) -> None:
    """The motor health section, headed even when empty like the alerts.

    An empty section with producers wired reads "no reports" in the quiet
    colour: the contract mandates a report per 500 ms, so a wired slot that
    has heard nothing is itself a finding, where "not wired" is a launcher
    choice. Sources never received draw no rows at all.
    """
    cv2.rectangle(
        frame,
        (_MARGIN_X, _MOTORS_RULE_Y),
        (_WIDTH - _MARGIN_X, _MOTORS_RULE_Y + _RULE_HEIGHT),
        _RULE,
        thickness=-1,
    )
    _text(frame, "MOTORS", (_MARGIN_X, _MOTORS_TITLE_BASELINE), _BODY_SCALE, _HEADING)
    if not health:
        empty, colour = (
            ("no reports", _NOT_REPORTING_MAGENTA) if bound else ("not wired", _WAITING)
        )
        _text(frame, empty, (_STATE_X, _MOTORS_TITLE_BASELINE), _BODY_SCALE, colour)
        return
    usable = _WIDTH - 2 * _MARGIN_X
    for index, line in enumerate(health):
        baseline = _FIRST_HEALTH_BASELINE + index * _HEALTH_STEP
        body, scale = _fitted(line.text, _HEALTH_SCALE, usable)
        _text(frame, body, (_MARGIN_X, baseline), scale, line.colour)


async def stream_status(
    sink: FrameSink,
    *,
    session: FrameSource,
    hands: Sequence[HandSource],
    settings: Settings,
    token: CancellationToken,
    alerts: ActiveAlerts,
    health: motor_health.MotorHealthReports,
    recorder: RecorderStatus | None = None,
    read_task: Callable[[], str] | None = None,
) -> None:
    """Publish the panel until shutdown, re-drawing only on a change.

    Latched: a failing render would repeat every frame, and the panel going
    dark is not worth killing a session over.
    """
    failing = Latch()
    drawn: PanelState | None = None
    frame: np.ndarray | None = None
    async for _ in ticks(1.0 / _PUBLISH_HZ, token):
        try:
            task = read_task() if read_task is not None else None
            state = snapshot(
                session, hands, settings.stale_timeout_s, alerts, health, recorder, task
            )
            if state != drawn or frame is None:
                frame, drawn = render(state), state
            sink.put_frame(frame)
            failing.clear()
        except Exception as e:
            failing.trip(f"status panel frame failed: {e!r}")
