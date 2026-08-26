"""Typed peppygen pairing IO for the openarm sim.

Bridges the physics thread (sync, runs the engine step in an executor) to the
node_runner's asyncio loop, where peppygen pairing pub/sub lives. The engine
plays the follower role of every limb's joint_link / gripper_link pairing, one
slot per limb, so there is no id demux: consume tasks on the loop each hold one
generated `subscribe()` subscription (gap-free, in-order) and keep the latest
governed setpoint per limb in thread-safe slots; the physics thread reads those
and publishes stamped measured state back up the same pair. The engine also
plays the camera role of one sim_rgb_camera_link / sim_rgbd_camera_link
pairing per robot-mounted camera, handing finished frames to the same loop and
dropping a frame instead of queueing when its stream's previous publish is
still in flight. Every hop is a generated peppygen pairing topic: no JSON, no
raw peppylib.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from typing import Optional

import peppylib
from peppygen import clock
from peppygen.paired_topics.chest import depth_stream as chest_depth
from peppygen.paired_topics.chest import stream_info as chest_info
from peppygen.paired_topics.chest import video_stream as chest_video
from peppygen.paired_topics.left_arm import joint_setpoints as left_arm_setpoints
from peppygen.paired_topics.left_arm import joint_states as left_arm_states
from peppygen.paired_topics.left_gripper import gripper_setpoints as left_gripper_setpoints
from peppygen.paired_topics.left_gripper import gripper_states as left_gripper_states
from peppygen.paired_topics.right_arm import joint_setpoints as right_arm_setpoints
from peppygen.paired_topics.right_arm import joint_states as right_arm_states
from peppygen.paired_topics.right_gripper import gripper_setpoints as right_gripper_setpoints
from peppygen.paired_topics.right_gripper import gripper_states as right_gripper_states
from peppygen.paired_topics.wrist_left import stream_info as wrist_left_info
from peppygen.paired_topics.wrist_left import video_stream as wrist_left_video
from peppygen.paired_topics.wrist_right import stream_info as wrist_right_info
from peppygen.paired_topics.wrist_right import video_stream as wrist_right_video

logger = logging.getLogger(__name__)

# Left = 0, right = 1: the slot layout mirrors the arm_id / gripper_id
# convention the rest of the stack uses for sides.
_ARM_SLOTS = {0: (left_arm_setpoints, left_arm_states), 1: (right_arm_setpoints, right_arm_states)}
_GRIPPER_SLOTS = {
    0: (left_gripper_setpoints, left_gripper_states),
    1: (right_gripper_setpoints, right_gripper_states),
}
# Camera slots are keyed by slot name (the camera's identity end to end: pairing
# link_id here, relay instance_id and dataset key downstream).
_COLOR_CAMERA_SLOTS = {
    "wrist_left": (wrist_left_video, wrist_left_info),
    "wrist_right": (wrist_right_video, wrist_right_info),
}
_RGBD_CAMERA_SLOTS = {
    "chest": (chest_video, chest_depth, chest_info),
}
# The slot names as sets, for validating a camera config against the manifest.
COLOR_CAMERA_SLOT_NAMES = frozenset(_COLOR_CAMERA_SLOTS)
RGBD_CAMERA_SLOT_NAMES = frozenset(_RGBD_CAMERA_SLOTS)

# Publisher and guard keys, spelled once: a publisher is keyed by
# (camera, topic) and a guard by (camera, surface).
_VIDEO_STREAM = "video_stream"
_DEPTH_STREAM = "depth_stream"
_STREAM_INFO = "stream_info"
_FRAMES_SURFACE = "frames"
_INFO_SURFACE = "info"


class _LatestSlot:
    """Thread-safe latest-wins single value, written on the loop and read on the
    physics thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = None

    def set(self, value) -> None:
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value


# How long one publish batch may stay in flight before the surface holding it
# is reported. Well past any real batch at these frame rates, so reaching it
# means the publish is not going to complete on its own.
_PUBLISH_STALL_S = 5.0


class _PublishGuard:
    """At most one in-flight publish batch per camera surface: acquired on the
    render thread when a capture is scheduled, released on the loop when every
    publish task of the batch finishes. A camera that can't keep up drops
    whole captures instead of piling publish tasks onto the loop, and an rgbd
    color + depth pair shares one guard so a pair is never half-dropped."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy = False
        self._acquired_s = 0.0
        self._stall_reported = False

    def try_acquire(self, surface: str) -> bool:
        """True when the caller now owns the surface. A publish future that
        never completes would hold the guard forever, and every later batch
        would be refused with nothing in the log to say the stream had gone
        dark, so a refusal past the stall bound is reported once per stall."""
        now_s = time.monotonic()
        with self._lock:
            if not self._busy:
                self._busy = True
                self._acquired_s = now_s
                self._stall_reported = False
                return True
            stalled_s = now_s - self._acquired_s
            report = stalled_s > _PUBLISH_STALL_S and not self._stall_reported
            if report:
                self._stall_reported = True
        if report:
            logger.error(
                f"{surface}: a publish has been in flight for {stalled_s:.1f}s; "
                "this surface drops every frame until it completes"
            )
        return False

    def release(self) -> None:
        with self._lock:
            self._busy = False
            self._stall_reported = False


class SimTopicIO:
    """Owns the typed pairing publishers + setpoint-consume tasks on the node
    loop, and exposes thread-safe accessors the physics thread calls each step."""

    def __init__(self, node_runner: peppylib.NodeRunner, loop: asyncio.AbstractEventLoop) -> None:
        self._node_runner = node_runner
        self._loop = loop
        self._arm_pubs: dict[int, peppylib.TopicPublisher] = {}
        self._gripper_pubs: dict[int, peppylib.TopicPublisher] = {}
        self._arm_cmd = {side: _LatestSlot() for side in _ARM_SLOTS}
        self._gripper_cmd = {side: _LatestSlot() for side in _GRIPPER_SLOTS}
        # Camera publishers and their in-flight guards, keyed by (slot, topic).
        self._camera_pubs: dict[tuple[str, str], peppylib.TopicPublisher] = {}
        self._camera_guards: dict[tuple[str, str], _PublishGuard] = {}
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Declare publishers and spawn the setpoint-consume loops. Runs on the
        node loop before the sim thread starts. Publishing while a slot is
        unpaired is a legal no-op, so bringup order never matters."""
        # State timestamps read the daemon-resolved clock, the same source every
        # follower uses, so consumers age samples on one timeline.
        await clock.init(self._node_runner)
        for side, (_, states) in _ARM_SLOTS.items():
            self._arm_pubs[side] = await states.declare_publisher(self._node_runner)
        for side, (_, states) in _GRIPPER_SLOTS.items():
            self._gripper_pubs[side] = await states.declare_publisher(self._node_runner)
        for name, (video, info) in _COLOR_CAMERA_SLOTS.items():
            await self._declare_camera_publishers(
                name, [(_VIDEO_STREAM, video), (_STREAM_INFO, info)]
            )
        for name, (video, depth, info) in _RGBD_CAMERA_SLOTS.items():
            await self._declare_camera_publishers(
                name, [(_VIDEO_STREAM, video), (_DEPTH_STREAM, depth), (_STREAM_INFO, info)]
            )
        self._tasks = [
            asyncio.create_task(self._consume_arm(mod, side))
            for side, (mod, _) in _ARM_SLOTS.items()
        ] + [
            asyncio.create_task(self._consume_gripper(mod, side))
            for side, (mod, _) in _GRIPPER_SLOTS.items()
        ]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        # Let the cancellations land so the consume loops exit before teardown.
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _consume_arm(self, topic, side: int) -> None:
        subscription = await topic.subscribe(self._node_runner)
        while True:
            try:
                pair = await subscription.next()
                if pair is None:
                    return
            except asyncio.CancelledError:
                return
            except Exception as exc:
                # A corrupt frame is dropped and logged rather than killing
                # this consume task; the pause keeps a persistent fault from
                # hot-spinning the loop.
                logger.warning(f"{topic.LINK_ID} setpoint consume error: {exc}")
                await asyncio.sleep(0.1)
                continue
            _peer, msg = pair
            # Drop a poisoned setpoint rather than writing NaN/Inf into the sim.
            if not all(math.isfinite(v) for v in msg.positions) or not all(
                math.isfinite(v) for v in msg.velocities
            ):
                logger.warning(f"dropping non-finite arm setpoint on {topic.LINK_ID}")
                continue
            if self._arm_cmd[side].get() is None:
                logger.info(f"first arm setpoint on {topic.LINK_ID}")
            self._arm_cmd[side].set((msg.positions, msg.velocities))

    async def _consume_gripper(self, topic, side: int) -> None:
        subscription = await topic.subscribe(self._node_runner)
        while True:
            try:
                pair = await subscription.next()
                if pair is None:
                    return
            except asyncio.CancelledError:
                return
            except Exception as exc:
                # A corrupt frame is dropped and logged rather than killing
                # this consume task; the pause keeps a persistent fault from
                # hot-spinning the loop.
                logger.warning(f"{topic.LINK_ID} setpoint consume error: {exc}")
                await asyncio.sleep(0.1)
                continue
            _peer, msg = pair
            if not (
                math.isfinite(msg.opening)
                and math.isfinite(msg.max_effort)
                and msg.max_effort >= 0.0
            ):
                logger.warning(f"dropping unusable gripper setpoint on {topic.LINK_ID}")
                continue
            # max_effort caps the finger drive effort in engine units; 0
            # (unset on the wire) leaves the engine's own force ceiling.
            if self._gripper_cmd[side].get() is None:
                logger.info(f"first gripper setpoint on {topic.LINK_ID}")
            self._gripper_cmd[side].set((msg.opening, msg.max_effort))

    # --- called from the physics thread ---

    def latest_arm_command(self, arm_id: int) -> Optional[tuple[list[float], list[float]]]:
        slot = self._arm_cmd.get(arm_id)
        return slot.get() if slot is not None else None

    def latest_gripper_command(self, gripper_id: int) -> Optional[tuple[float, float]]:
        slot = self._gripper_cmd.get(gripper_id)
        return slot.get() if slot is not None else None

    def publish_arm_states(self, arm_id: int, positions: list[float], velocities: list[float]) -> None:
        pub = self._arm_pubs.get(arm_id)
        if pub is not None:
            # Efforts are empty: the engine measures no joint torques.
            payload = _ARM_SLOTS[arm_id][1].build_message(
                clock.now_ns() / 1e9, positions, velocities, []
            )
            self._schedule_publish(pub, payload)

    def publish_gripper_states(self, gripper_id: int, opening: float, force: float = 0.0) -> None:
        pub = self._gripper_pubs.get(gripper_id)
        if pub is not None:
            # The engine torque rides as the pairing effort; the ceiling is 0
            # (no effort control).
            payload = _GRIPPER_SLOTS[gripper_id][1].build_message(
                clock.now_ns() / 1e9, opening, force, 0.0
            )
            self._schedule_publish(pub, payload)

    def camera_timestamp_s(self) -> float:
        """Capture timestamp on the daemon-resolved clock, taken once per capture so
        an rgbd color + depth pair shares one timestamp."""
        return clock.now_ns() / 1e9

    def publish_color_frame(
        self,
        name: str,
        timestamp_s: float,
        frame_id: int,
        encoding: str,
        width: int,
        height: int,
        frame: bytes,
    ) -> bool:
        video, _ = _COLOR_CAMERA_SLOTS[name]
        payload = video.build_message(
            video.MessageHeader(timestamp=timestamp_s, frame_id=frame_id),
            encoding,
            width,
            height,
            frame,
        )
        return self._publish_guarded(name, _FRAMES_SURFACE, [(_VIDEO_STREAM, payload)])

    def publish_color_stream_info(
        self, name: str, width: int, height: int, frames_per_second: int, encoding: str
    ) -> None:
        _, info = _COLOR_CAMERA_SLOTS[name]
        payload = info.build_message(width, height, frames_per_second, encoding)
        self._publish_guarded(name, _INFO_SURFACE, [(_STREAM_INFO, payload)])

    def publish_rgbd_frames(
        self,
        name: str,
        timestamp_s: float,
        frame_id: int,
        align_mode: str,
        color: tuple[str, int, int, bytes],
        depth: tuple[str, int, int, bytes],
    ) -> bool:
        """One rgbd capture: color and depth (each an (encoding, width,
        height, frame) tuple) publish as a single guarded batch sharing the
        timestamp and frame_id, so a pair is dropped or delivered whole."""
        video, depth_mod, _ = _RGBD_CAMERA_SLOTS[name]
        color_encoding, color_width, color_height, color_frame = color
        depth_encoding, depth_width, depth_height, depth_frame = depth
        color_payload = video.build_message(
            video.MessageHeader(timestamp=timestamp_s, frame_id=frame_id, align_mode=align_mode),
            color_encoding,
            color_width,
            color_height,
            color_frame,
        )
        depth_payload = depth_mod.build_message(
            depth_mod.MessageHeader(timestamp=timestamp_s, frame_id=frame_id, align_mode=align_mode),
            depth_encoding,
            depth_width,
            depth_height,
            depth_frame,
        )
        return self._publish_guarded(
            name,
            _FRAMES_SURFACE,
            [(_VIDEO_STREAM, color_payload), (_DEPTH_STREAM, depth_payload)],
        )

    def publish_rgbd_stream_info(
        self,
        name: str,
        width: int,
        height: int,
        frames_per_second: int,
        encoding: str,
        depth_width: int,
        depth_height: int,
        depth_encoding: str,
        depth_unit: float,
    ) -> None:
        _, _, info = _RGBD_CAMERA_SLOTS[name]
        payload = info.build_message(
            width,
            height,
            frames_per_second,
            encoding,
            depth_width,
            depth_height,
            depth_encoding,
            depth_unit,
        )
        self._publish_guarded(name, _INFO_SURFACE, [(_STREAM_INFO, payload)])

    async def _declare_camera_publishers(
        self, name: str, topics: list[tuple[str, object]]
    ) -> None:
        for topic_name, module in topics:
            self._camera_pubs[(name, topic_name)] = await module.declare_publisher(
                self._node_runner
            )
        for surface in (_FRAMES_SURFACE, _INFO_SURFACE):
            self._camera_guards[(name, surface)] = _PublishGuard()

    def _publish_guarded(
        self, name: str, surface: str, topic_payloads: list[tuple[str, bytes]]
    ) -> bool:
        """False when this surface's previous batch is still in flight, so the
        caller knows the sample never reached a consumer."""
        guard = self._camera_guards[(name, surface)]
        if not guard.try_acquire(f"{name} {surface}"):
            return False
        publishes = [
            (self._camera_pubs[(name, topic_name)], payload)
            for topic_name, payload in topic_payloads
        ]

        def _publish() -> None:
            # Runs on the loop, so the counter needs no lock; the guard is
            # released once every task of the batch has finished, and on any
            # synchronous failure, so no path can wedge the stream.
            state = {"remaining": len(publishes)}

            def _finish_one() -> None:
                state["remaining"] -= 1
                if state["remaining"] == 0:
                    guard.release()

            for index, (pub, payload) in enumerate(publishes):
                try:
                    task = asyncio.ensure_future(pub.publish(payload))
                except BaseException:
                    # This publish and every unscheduled one after it are over.
                    for _ in range(len(publishes) - index):
                        _finish_one()
                    raise

                def _done(finished: asyncio.Task) -> None:
                    _finish_one()
                    _log_publish_error(finished)

                task.add_done_callback(_done)

        try:
            self._loop.call_soon_threadsafe(_publish)
        except RuntimeError:
            # Loop closed during shutdown; drop the sample.
            guard.release()
            return False
        return True

    def _schedule_publish(self, publisher: peppylib.TopicPublisher, payload: bytes) -> None:
        # Hand the publish to the node loop and return immediately; the physics
        # thread must never block on messaging.
        def _publish() -> None:
            task = asyncio.ensure_future(publisher.publish(payload))
            task.add_done_callback(_log_publish_error)

        try:
            self._loop.call_soon_threadsafe(_publish)
        except RuntimeError:
            # Loop closed during shutdown; drop the sample.
            pass


def _log_publish_error(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning(f"publish failed: {exc}")
