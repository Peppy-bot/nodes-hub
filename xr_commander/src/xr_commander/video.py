"""Camera frames off the peppy bus and into the headset's WebRTC tracks.

The headset shows the chosen robot's cameras, one panel per camera role, the
id a camera's copy wrote for it (`wrist_left`, `chest`). A robot that pairs
one arm shows its one `wrist` camera on the wrist panel of the hand driving
that arm. When the choice changes the panels change with it, on the pages in
session too. Every shown panel is fed the chosen robot's newest frame at
`TRACK_HZ`, steadily. Frames are handed to teleop_xr as BGR: its track
converts BGR-to-RGB before encoding.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import ModuleType
from typing import Protocol

import cv2
import numpy as np

from xr_commander.bus import (
    CancellationToken,
    Latch,
    latch_for,
    log,
    messages,
    producer_key,
    subscribe_or_log,
    ticks,
)
from xr_commander.devices import Hand
from xr_commander.robots import PairedRobots


class _Encoding(StrEnum):
    """The color encodings the rgb_camera / rgbd_camera contracts declare. A
    camera announcing any other encoding is refused by name."""

    MJPEG = "mjpeg"
    RGB8 = "rgb8"
    BGR8 = "bgr8"
    YUYV = "yuyv"


# The raw encodings, by the bytes each pixel takes.
_BYTES_PER_PIXEL = {_Encoding.RGB8: 3, _Encoding.BGR8: 3, _Encoding.YUYV: 2}


def _parse_encoding(encoding: str) -> _Encoding | None:
    """`encoding` as an _Encoding, or None for one the panels do not decode."""
    try:
        return _Encoding(encoding)
    except ValueError:
        return None


# A camera that has delivered nothing for this long gets its track blanked: in
# the headset a frozen frame is indistinguishable from live video.
_CAMERA_SILENT_S = 2.0
_SILENCE_POLL_S = 0.5
# How often the chosen robot's panels are read off the camera slots.
_FOLLOW_PERIOD_S = 0.2
_NO_SIGNAL_HEIGHT = 180
_NO_SIGNAL_WIDTH = 320

# The rate teleop_xr's camera tracks play at: each frame is stamped 1/30 s
# after the one before (aiortc's video clock), so every panel is fed its
# newest frame at exactly this rate, whatever rate its camera publishes at.
TRACK_HZ = 30
# The role of the one wrist camera of a robot with one arm.
_LONE_WRIST_ROLE = "wrist"
# The panels teleop_xr anchors to the controllers, by hand.
_WRIST_PANELS = {Hand.LEFT: "wrist_left", Hand.RIGHT: "wrist_right"}


class FrameSink(Protocol):
    """Where decoded frames go (teleop_xr's ExternalVideoSource, in practice)."""

    def put_frame(self, frame: np.ndarray) -> None: ...


def decode_to_bgr(encoding: str, width: int, height: int, data: bytes) -> np.ndarray:
    """One contract frame as an HxWx3 BGR array; ValueError when undecodable.

    Raw encodings are checked against their declared geometry; mjpeg carries
    its own, so only the decode itself vouches for it.
    """
    parsed = _parse_encoding(encoding)
    if parsed is _Encoding.MJPEG:
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("mjpeg frame failed to decode")
        return frame

    stride = _BYTES_PER_PIXEL.get(parsed)
    if stride is None:
        raise ValueError(f"unsupported encoding {encoding!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"degenerate frame geometry {width}x{height}")
    expected = width * height * stride
    if len(data) != expected:
        raise ValueError(
            f"{encoding} frame is {len(data)} bytes, expected {expected} "
            f"for {width}x{height}"
        )

    raw = np.frombuffer(data, dtype=np.uint8).reshape(height, width, stride)
    try:
        if parsed is _Encoding.BGR8:
            # Copy: a frombuffer view aliases the wire buffer, which is
            # recycled while the encoder still holds the frame.
            return raw.copy()
        if parsed is _Encoding.RGB8:
            return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUY2)
    except cv2.error as e:
        raise ValueError(f"{encoding} frame failed to convert: {e}") from None


@dataclass(frozen=True)
class CameraSource:
    """One bound camera: the robot it belongs to, None for a camera the
    launcher deploys outside any copy, and its role, the id its copy's
    fragment wrote for it (its instance id outside a copy)."""

    robot: str | None
    role: str


def camera_source(member) -> CameraSource:
    """A bound member's camera, by robot and role."""
    if member.copy is None:
        return CameraSource(robot=None, role=member.producer.instance_id)
    return CameraSource(robot=member.copy.name, role=member.copy.instance_id)


def panel_of(
    source: CameraSource, arm_sides: Callable[[str], frozenset[Hand]]
) -> str | None:
    """The panel a camera fills: its role, or for the `wrist` camera of a
    robot that pairs one arm, the wrist panel of the hand driving that arm.
    None while the camera's robot pairs no arm with the headset: the headset
    has nothing of it to show."""
    if source.robot is None:
        return source.role
    sides = arm_sides(source.robot)
    if not sides:
        return None
    if source.role == _LONE_WRIST_ROLE and len(sides) == 1:
        (side,) = sides
        return _WRIST_PANELS[side]
    return source.role


def shows_for(source: CameraSource, chosen_robot: str | None) -> bool:
    """Whether a camera's frames reach its role's panel: the chosen robot's
    cameras do, and a camera outside any copy belongs to every robot."""
    return source.robot is None or source.robot == chosen_robot


def _panel_for(source: CameraSource, robots: PairedRobots) -> str | None:
    """The panel a camera fills now, None while it is not the chosen
    robot's."""
    if not shows_for(source, robots.chosen()):
        return None
    return panel_of(source, robots.arm_sides)


@dataclass(frozen=True)
class LastFrame:
    """The newest frame a panel was given: whose camera, and when (monotonic
    seconds)."""

    source: CameraSource
    monotonic_s: float


@dataclass(frozen=True)
class _Shown:
    """A frame waiting on a panel, and the camera it came from; None for a
    frame of the node's own, which every robot shows."""

    frame: np.ndarray
    source: CameraSource | None


class CameraPanels:
    """The headset's camera panels, each made the first time a camera fills
    it, and the ones the chosen robot fills.

    `show_views` tells the session the shown panels whenever they change; a
    name the session reserves (the status panel's) is refused, once.
    """

    def __init__(
        self,
        make_sink: Callable[[], FrameSink],
        show_views: Callable[[Mapping[str, FrameSink]], None],
        reserved: frozenset[str],
    ) -> None:
        self._make_sink = make_sink
        self._show_views = show_views
        self._reserved = reserved
        self._sinks: dict[str, FrameSink] = {}
        self._newest: dict[str, _Shown] = {}
        self._shown: tuple[str, ...] = ()
        self._refused = Latch()

    def ensure(self, name: str) -> bool:
        """Make the panel `name` if this is its first camera; False for a
        reserved name."""
        if name in self._reserved:
            self._refused.trip(
                f"camera instance {name!r} clashes with the status panel's name; "
                "its frames are dropped. Rename the camera instance"
            )
            return False
        if name not in self._sinks:
            self._sinks[name] = self._make_sink()
        return True

    def show_only(self, names: Iterable[str]) -> None:
        """Show the panels `names` alone, telling the session when the set
        changes."""
        shown = tuple(sorted(name for name in set(names) if self.ensure(name)))
        if shown == self._shown:
            return
        self._shown = shown
        self._show_views({name: self._sinks[name] for name in shown})
        log(f"camera panels shown: {', '.join(shown) or 'none'}")

    def shown(self) -> tuple[str, ...]:
        """The panels shown, by name."""
        return self._shown

    def show(
        self, name: str, frame: np.ndarray, source: CameraSource | None = None
    ) -> None:
        """Show `frame`, from `source`'s camera, on the panel `name` from the
        next feed on."""
        if self.ensure(name):
            self._newest[name] = _Shown(frame, source)

    def feed(self, chosen_robot: str | None) -> None:
        """Hand every shown panel its newest frame once, while that frame
        is the chosen robot's."""
        for name in self._shown:
            newest = self._newest.get(name)
            if newest is None:
                continue
            if newest.source is None or shows_for(newest.source, chosen_robot):
                self._sinks[name].put_frame(newest.frame)


async def feed_panels(
    panels: CameraPanels,
    chosen_robot: Callable[[], str | None],
    token: CancellationToken,
) -> None:
    """Feed every shown panel at `TRACK_HZ` until shutdown."""
    async for _ in ticks(1.0 / TRACK_HZ, token):
        panels.feed(chosen_robot())


def chosen_panels(
    node_runner, slot_modules: Sequence[ModuleType], robots: PairedRobots
) -> set[str]:
    """The panels the chosen robot's bound cameras fill."""
    sources = (
        camera_source(member)
        for module in slot_modules
        for member in module.bound_members(node_runner)
    )
    return {
        name for source in sources if (name := _panel_for(source, robots)) is not None
    }


async def follow_camera_panels(
    node_runner,
    slot_modules: Sequence[ModuleType],
    panels: CameraPanels,
    robots: PairedRobots,
    token: CancellationToken,
    period_s: float = _FOLLOW_PERIOD_S,
) -> None:
    """Show the chosen robot's panels as its cameras and arms pair and as
    the choice changes, so a camera that joins late has a panel before its
    first frame, and a silent one reads NO SIGNAL."""
    async for _ in ticks(period_s, token):
        panels.show_only(chosen_panels(node_runner, slot_modules, robots))


def shrink_to_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """The frame at most `max_width` wide, aspect kept; 0 disables.

    Headset panels render small at arm's length, so encoding sensor-native
    video wastes cores on the control path; consumers that need full
    resolution (the recorder) bind the cameras directly and never see this.
    """
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    height = max(1, round(frame.shape[0] * scale))
    return cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)


def _decode_and_shrink(message, view_max_width: int) -> np.ndarray:
    return shrink_to_width(
        decode_to_bgr(message.encoding, message.width, message.height, message.frame),
        view_max_width,
    )


async def drain_frames(
    node_runner,
    topic_module,
    panels: CameraPanels,
    robots: PairedRobots,
    token: CancellationToken,
    label: str,
    view_max_width: int = 0,
    health: dict[str, LastFrame] | None = None,
) -> None:
    """Route every frame of the chosen robot's cameras on one slot onto the
    camera's panel; the other robots' frames are dropped.

    Frames are handled in arrival order; the subscription's bounded queue is
    the only backpressure. Decode runs in a worker thread so a heavy frame
    never stalls the command streams sharing the loop. An undecodable frame is
    logged once and dropped, never stopping the other producers. `health`
    (panel to its newest frame) is stamped per delivered frame for the
    silence watchdog.
    """
    subscription = await subscribe_or_log(node_runner, topic_module, label)
    if subscription is None:
        # Loud and fail-safe: the affected panels simply never show video.
        return
    # Latched per producer: a bad encoding repeats every frame.
    unusable: dict[tuple[str, str], Latch] = {}
    async for producer, message in messages(subscription, token, label):
        # The set follows the stack's joins, so it is read per frame.
        sources = {
            producer_key(m.producer): camera_source(m)
            for m in topic_module.bound_members(node_runner)
        }
        key = producer_key(producer)
        source = sources.get(key)
        name = _panel_for(source, robots) if source is not None else None
        if name is None:
            continue
        latch = latch_for(unusable, key)
        try:
            frame = await asyncio.to_thread(_decode_and_shrink, message, view_max_width)
            if _panel_for(source, robots) != name:
                # The choice changed while the frame decoded.
                continue
            panels.show(name, frame, source)
            if health is not None:
                health[name] = LastFrame(source, time.monotonic())
            latch.clear()
        except Exception as e:
            latch.trip(f"{label} {producer.instance_id} frame unusable: {e!r}")


def no_signal_frame(track_id: str) -> np.ndarray:
    """The frame a silent camera's track shows instead of its last image."""
    frame = np.zeros((_NO_SIGNAL_HEIGHT, _NO_SIGNAL_WIDTH, 3), dtype=np.uint8)
    cv2.putText(
        frame, "NO SIGNAL", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
    )
    cv2.putText(
        frame, track_id, (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1
    )
    return frame


def shows_live_video(
    last: LastFrame | None,
    chosen_robot: str | None,
    now: float,
    timed_from: float,
    silent_after_s: float,
) -> bool:
    """Whether a panel shows its chosen robot's camera: a frame of that
    camera came within `silent_after_s`. A panel holding no frame of the
    chosen robot is timed from `timed_from`."""
    if last is None or not shows_for(last.source, chosen_robot):
        return now - timed_from <= silent_after_s
    return now - last.monotonic_s <= silent_after_s


async def watch_camera_silence(
    panels: CameraPanels,
    health: Mapping[str, LastFrame],
    chosen_robot: Callable[[], str | None],
    token: CancellationToken,
    silent_after_s: float = _CAMERA_SILENT_S,
    poll_s: float = _SILENCE_POLL_S,
) -> None:
    """Show NO SIGNAL on a shown panel whose chosen robot's camera stopped
    delivering, or never did, and say so once.

    A panel is timed from when it was shown or the chosen robot last
    changed, whichever is later.
    """
    chosen = chosen_robot()
    shown_since: dict[str, float] = {}
    dark: set[str] = set()
    async for _ in ticks(poll_s, token):
        now = time.monotonic()
        current = chosen_robot()
        if current != chosen:
            chosen = current
            shown_since = dict.fromkeys(shown_since, now)
        shown_since = {name: shown_since.get(name, now) for name in panels.shown()}
        for name, since in shown_since.items():
            live = shows_live_video(
                health.get(name), chosen, now, since, silent_after_s
            )
            if not live and name not in dark:
                dark.add(name)
                log(f"camera panel {name} went silent; showing NO SIGNAL")
                panels.show(name, no_signal_frame(name))
            elif live and name in dark:
                dark.discard(name)
                log(f"camera panel {name} recovered")
