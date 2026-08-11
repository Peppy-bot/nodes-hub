"""The in-headset status panel: node state drawn onto a synthetic track.

A headset has no console, so anything the node only logs is invisible to the
operator wearing it. This carries the state that decides whether a hand can
drive at all. teleop_xr renders every declared view as a floating panel, so
nothing in the web stack needs to know this one is not a camera.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

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

# The headset shows this at arm's length and VP8 blurs fine strokes, so the
# canvas stays small and the type large rather than dense.
_WIDTH = 512
_HEIGHT = 256
_MARGIN_X = 18
_STATE_X = 170
_TITLE_BASELINE = 44
_LINK_BASELINE = 88
_FIRST_ROW_BASELINE = 150
_ROW_STEP = 46

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TITLE_SCALE = 0.9
_BODY_SCALE = 0.7
_STROKE = 2

_BACKGROUND = (24, 24, 24)
_HEADING = (210, 210, 210)
# BGR: green drives, amber waits, grey rests, red records.
_DRIVING = (120, 220, 120)
_WAITING = (60, 200, 250)
_RESTING = (170, 170, 170)
_RECORDING = (70, 70, 235)


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
class PanelState:
    """Everything drawn this tick; compared to decide whether to re-draw."""

    headset_live: bool
    hands: tuple[Row, ...]
    # The recorder's row; None when no recorder is bound.
    recorder: Row | None = None


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


def snapshot(
    session: FrameSource,
    hands: Sequence[HandSource],
    stale_timeout_s: float,
    recorder: RecorderStatus | None = None,
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
    return PanelState(
        headset_live=frame is not None,
        hands=tuple(rows),
        recorder=recorder_row(recorder),
    )


def _text(
    frame: np.ndarray,
    body: str,
    origin: tuple[int, int],
    scale: float,
    colour: tuple[int, int, int],
) -> None:
    cv2.putText(frame, body, origin, _FONT, scale, colour, _STROKE, cv2.LINE_AA)


def render(state: PanelState) -> np.ndarray:
    """The panel as one BGR frame, ready for the encoder."""
    frame = np.full((_HEIGHT, _WIDTH, 3), _BACKGROUND, dtype=np.uint8)
    _text(frame, "xr_commander", (_MARGIN_X, _TITLE_BASELINE), _TITLE_SCALE, _HEADING)
    live = state.headset_live
    link, colour = ("headset live", _DRIVING) if live else ("headset stale", _WAITING)
    _text(frame, link, (_MARGIN_X, _LINK_BASELINE), _BODY_SCALE, colour)
    rows = state.hands + ((state.recorder,) if state.recorder else ())
    for index, row in enumerate(rows):
        baseline = _FIRST_ROW_BASELINE + index * _ROW_STEP
        _text(frame, row.label, (_MARGIN_X, baseline), _BODY_SCALE, _HEADING)
        _text(frame, row.state, (_STATE_X, baseline), _BODY_SCALE, row.colour)
    return frame


async def stream_status(
    sink: FrameSink,
    *,
    session: FrameSource,
    hands: Sequence[HandSource],
    settings: Settings,
    token: CancellationToken,
    recorder: RecorderStatus | None = None,
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
            state = snapshot(session, hands, settings.stale_timeout_s, recorder)
            if state != drawn or frame is None:
                frame, drawn = render(state), state
            sink.put_frame(frame)
            failing.clear()
        except Exception as e:
            failing.trip(f"status panel frame failed: {e!r}")
