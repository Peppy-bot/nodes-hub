"""Typed peppygen pairing IO for the openarm sim.

Bridges the physics thread (sync, runs the engine step in an executor) to the
node_runner's asyncio loop, where peppygen pairing pub/sub lives. The engine
plays the follower role of every limb's joint_link / gripper_link pairing, one
slot per limb and one pair per robot: consume tasks on the loop each hold one
generated `subscribe()` subscription (gap-free, in-order) and keep the latest
governed setpoint per limb of each robot in thread-safe slots; the physics
thread reads those and publishes each robot's stamped measured state back on
its own pair. A pair carries the copy its robot attached under, which is what
tells one robot's limbs from another's. The engine also plays the camera
role of one sim_rgb_camera_link / sim_rgbd_camera_link pairing per
robot-mounted camera, handing finished frames to the same loop and dropping a
frame whose stream's previous publish is still in flight. Every hop is a
generated peppygen pairing topic.

Beside the pairings, the engine emits object_state's object_states topic: a
full snapshot of every spawned object on the state tick, dropped instead of
queued while the previous one is still in flight, as a camera frame is.

When the deployment names this instance the publisher of a clock domain, the
physics thread records its engine clock each step (`record_engine_time`, ahead
of every stamp of that step) and this node publishes it on each telemetry tick
(`publish_clock_tick`). The domain's granularity therefore follows
`state_rate_hz` while this engine's own stamps stay step-fresh. A publisher
reads back the instant it committed, so the stamps come straight from the
recorded engine clock.
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
from peppylib.clock import ClockPublisher
from peppygen.emitted_topics.objects import object_states
from peppygen.paired_topics.chest import depth_stream as chest_depth
from peppygen.paired_topics.chest import geometry as chest_geometry
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
from peppygen.paired_topics.wrist_left import geometry as wrist_left_geometry
from peppygen.paired_topics.wrist_left import stream_info as wrist_left_info
from peppygen.paired_topics.wrist_left import video_stream as wrist_left_video
from peppygen.paired_topics.wrist_right import geometry as wrist_right_geometry
from peppygen.paired_topics.wrist_right import stream_info as wrist_right_info
from peppygen.paired_topics.wrist_right import video_stream as wrist_right_video

logger = logging.getLogger(__name__)

# The clock topic reserves zero for "no tick observed yet", so the smallest
# instant an engine can carry is one nanosecond. A timeline still sitting at
# zero is a real instant rather than a fault, so it is floored to that
# minimum: the states this engine stamps and the domain's ticks then carry
# the same instant instead of differing by the wire's own clamp.
_MIN_ENGINE_TIME_S = 1e-9

# Limb slots are keyed by the name the model gives the limb, which is the
# name the engine lists a robot's limbs under when it attaches and the name
# sim_bridge.json5 writes.
_ARM_SLOTS = {
    "left": (left_arm_setpoints, left_arm_states),
    "right": (right_arm_setpoints, right_arm_states),
}
_GRIPPER_SLOTS = {
    "left": (left_gripper_setpoints, left_gripper_states),
    "right": (right_gripper_setpoints, right_gripper_states),
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
# The geometry topic of every camera slot, colour and rgbd alike.
_CAMERA_GEOMETRY = {
    "wrist_left": wrist_left_geometry,
    "wrist_right": wrist_right_geometry,
    "chest": chest_geometry,
}
# The slot names as sets, for validating a camera config against the manifest.
COLOR_CAMERA_SLOT_NAMES = frozenset(_COLOR_CAMERA_SLOTS)
RGBD_CAMERA_SLOT_NAMES = frozenset(_RGBD_CAMERA_SLOTS)

# Publisher and guard keys, spelled once: a publisher is keyed by
# (slot, topic) and a guard by (robot, slot, surface).
_VIDEO_STREAM = "video_stream"
_DEPTH_STREAM = "depth_stream"
_STREAM_INFO = "stream_info"
_GEOMETRY = "geometry"
_FRAMES_SURFACE = "frames"
_INFO_SURFACE = "info"
# Geometry goes out on the same tick as the stream info, so it takes a guard
# of its own: sharing the info one would drop whichever was scheduled second.
_GEOMETRY_SURFACE = "geometry"


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

# A rendered image is an ideal pinhole: the camera_geometry contract's "none".
_RENDERED_DISTORTION = "none"


class _PublishGuard:
    """At most one in-flight publish batch per surface (a camera's frames or
    stream info, the object-state stream): acquired on the physics thread
    when a sample is scheduled, released on the loop when every publish task
    of the batch finishes. A surface that can't keep up drops whole samples
    instead of piling publish tasks onto the loop, and an rgbd color + depth
    pair shares one guard so a pair is never half-dropped."""

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

    def __init__(
        self,
        node_runner: peppylib.NodeRunner,
        loop: asyncio.AbstractEventLoop,
        robots,
    ) -> None:
        self._node_runner = node_runner
        self._loop = loop
        # The robots in the scene, which is what a pair's copy names.
        self._robots = robots
        self._arm_pubs: dict[str, peppylib.PeerPublisher] = {}
        self._gripper_pubs: dict[str, peppylib.PeerPublisher] = {}
        # The latest setpoint of one limb of one robot, keyed by (limb, robot).
        self._arm_cmd: dict[tuple[str, str], _LatestSlot] = {}
        self._gripper_cmd: dict[tuple[str, str], _LatestSlot] = {}
        self._cmd_lock = threading.Lock()
        # Camera publishers and their in-flight guards, keyed by (slot, topic)
        # and (robot, slot, surface): a publisher reaches every robot's camera
        # on its slot, so the guard is per robot.
        self._camera_pubs: dict[tuple[str, str], peppylib.PeerPublisher] = {}
        self._camera_guards: dict[tuple[str, str, str], _PublishGuard] = {}
        # The object_states publisher and its in-flight guard: the scene's
        # objects are one stream.
        self._object_states_pub: Optional[peppylib.TopicPublisher] = None
        self._object_states_guard = _PublishGuard()
        self._tasks: list[asyncio.Task] = []
        # Set in start() when the deployment names this instance the
        # publisher of a clock domain.
        self._clock_publisher: Optional[ClockPublisher] = None
        # The engine clock this instance stamps its own state with. Written
        # and read on the physics thread only, while _clock_publisher is set.
        self._engine_time_s: Optional[float] = None
        # Clock publish state, node loop only: whether a publish is in
        # flight, the newest tick that arrived while one was, and whether the
        # last completed publish failed (logged once per transition).
        self._clock_in_flight = False
        self._clock_latched_ns: Optional[int] = None
        self._clock_publish_failed = False

    async def start(self) -> None:
        """Declare publishers and spawn the setpoint-consume loops. Runs on the
        node loop before the sim thread starts. Publishing while a slot is
        unpaired is a legal no-op, so bringup order never matters."""
        # State timestamps read this instance's bound clock, the one its
        # consumers read too, so samples age on one timeline.
        await clock.init(self._node_runner)
        # A deployment names one instance of a clock domain its publisher, and
        # holding a publisher is that answer.
        self._clock_publisher = await ClockPublisher.for_node(self._node_runner)
        if self._clock_publisher is not None:
            logger.info(f"publishing clock domain {self._clock_publisher.domain}")
        for side, (_, states) in _ARM_SLOTS.items():
            self._arm_pubs[side] = await states.declare_publisher(self._node_runner)
        for side, (_, states) in _GRIPPER_SLOTS.items():
            self._gripper_pubs[side] = await states.declare_publisher(self._node_runner)
        self._object_states_pub = await object_states.declare_publisher(self._node_runner)
        for name, (video, info) in _COLOR_CAMERA_SLOTS.items():
            await self._declare_camera_publishers(
                name,
                [
                    (_VIDEO_STREAM, video),
                    (_STREAM_INFO, info),
                    (_GEOMETRY, _CAMERA_GEOMETRY[name]),
                ],
            )
        for name, (video, depth, info) in _RGBD_CAMERA_SLOTS.items():
            await self._declare_camera_publishers(
                name,
                [
                    (_VIDEO_STREAM, video),
                    (_DEPTH_STREAM, depth),
                    (_STREAM_INFO, info),
                    (_GEOMETRY, _CAMERA_GEOMETRY[name]),
                ],
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

    async def _consume_arm(self, topic, side: str) -> None:
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
            peer, msg = pair
            robot = self._robot_of(topic, peer)
            if robot is None:
                continue
            # Drop a poisoned setpoint rather than writing NaN/Inf into the sim.
            if not all(math.isfinite(v) for v in msg.positions) or not all(
                math.isfinite(v) for v in msg.velocities
            ):
                logger.warning(
                    f"dropping non-finite arm setpoint for '{robot}' on {topic.LINK_ID}"
                )
                continue
            slot = self._command_slot(self._arm_cmd, side, robot)
            if slot.get() is None:
                logger.info(f"first arm setpoint for '{robot}' on {topic.LINK_ID}")
            slot.set((msg.positions, msg.velocities))

    async def _consume_gripper(self, topic, side: str) -> None:
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
            peer, msg = pair
            robot = self._robot_of(topic, peer)
            if robot is None:
                continue
            if not (
                math.isfinite(msg.opening)
                and math.isfinite(msg.max_effort)
                and msg.max_effort >= 0.0
            ):
                logger.warning(
                    f"dropping unusable gripper setpoint for '{robot}' on {topic.LINK_ID}"
                )
                continue
            # max_effort caps the finger drive effort in engine units; 0
            # (unset on the wire) leaves the engine's own force ceiling.
            slot = self._command_slot(self._gripper_cmd, side, robot)
            if slot.get() is None:
                logger.info(f"first gripper setpoint for '{robot}' on {topic.LINK_ID}")
            slot.set((msg.opening, msg.max_effort))

    # --- the robot a pair belongs to ---

    def _robot_of(self, module, peer) -> Optional[str]:
        """The robot whose limb this pair drives: the copy the pair carries,
        which is the name that robot attached under. A pair with no copy is a
        robot launched outside one, and outside a copy there is one robot in
        the scene for it to belong to."""
        for member in module.peers(self._node_runner):
            if member.info == peer:
                return member.copy or self._robots.sole_name()
        return None

    def _peer_of(self, module, robot: str):
        """The pair this robot's limb reaches, or None while it holds none."""
        for member in module.peers(self._node_runner):
            if (member.copy or self._robots.sole_name()) == robot:
                return member.info
        return None

    def _command_slot(self, slots: dict, limb: str, robot: str) -> "_LatestSlot":
        with self._cmd_lock:
            return slots.setdefault((limb, robot), _LatestSlot())

    def paired_robots(self) -> dict[str, set[str]]:
        """The robots holding a pair on each limb slot, by limb slot. Read on
        the node loop and on the physics thread, which is what keeps a robot
        in the scene."""
        held: dict[str, set[str]] = {}
        for side, (setpoints, _) in (*_ARM_SLOTS.items(), *_GRIPPER_SLOTS.items()):
            names = set()
            for member in setpoints.peers(self._node_runner):
                name = member.copy or self._robots.sole_name()
                if name is not None:
                    names.add(name)
            held[setpoints.LINK_ID] = names
        return held

    def robots_with_every_limb(self) -> set[str]:
        """The robots holding a pair on every limb slot: their setpoints
        reach the engine and their state reaches them, which is what a robot
        asks about when it asks whether it is ready."""
        held = list(self.paired_robots().values())
        if not held:
            return set()
        return set.intersection(*held)

    def robots_with_any_limb(self) -> set[str]:
        """Every robot holding at least one limb pair."""
        held = list(self.paired_robots().values())
        return set().union(*held) if held else set()

    def camera_robots(self) -> set[str]:
        """The robots holding at least one camera pair: the engine renders a
        rig for those and for no others, so a stack that pairs no camera pays
        for none."""
        names = set()
        for video, *_ in (*_COLOR_CAMERA_SLOTS.values(), *_RGBD_CAMERA_SLOTS.values()):
            for member in video.peers(self._node_runner):
                name = member.copy or self._robots.sole_name()
                if name is not None:
                    names.add(name)
        return names

    # --- called from the physics thread ---

    def latest_arm_command(
        self, robot: str, arm: str
    ) -> Optional[tuple[list[float], list[float]]]:
        slot = self._arm_cmd.get((arm, robot))
        return slot.get() if slot is not None else None

    def latest_gripper_command(self, robot: str, gripper: str) -> Optional[tuple[float, float]]:
        slot = self._gripper_cmd.get((gripper, robot))
        return slot.get() if slot is not None else None

    def forget(self, robot: str) -> None:
        """Drops what a robot that left the scene had commanded, so a robot
        rejoining under the same name starts from its model's pose."""
        with self._cmd_lock:
            for slots in (self._arm_cmd, self._gripper_cmd):
                for key in [key for key in slots if key[1] == robot]:
                    del slots[key]

    def record_engine_time(self, engine_time_s: float) -> None:
        """Adopt this step's engine clock for every stamp this engine emits.
        Called from the physics thread once physics has advanced, ahead of
        every state and camera stamp of the step. Records while this instance
        publishes a clock domain; a consumer stamps from its bound clock."""
        if self._clock_publisher is None:
            return
        if not math.isfinite(engine_time_s) or engine_time_s < 0.0:
            raise ValueError(
                f"the engine clock produced a non-publishable instant: {engine_time_s!r}"
            )
        self._engine_time_s = max(engine_time_s, _MIN_ENGINE_TIME_S)

    def publish_clock_tick(self) -> None:
        """Publish the recorded engine clock on the telemetry cadence. At most
        one publish is in flight; a tick arriving while one is out is latched
        and sent when it completes, so a slow send can delay the domain's clock
        but never reorder it.

        Called from the physics thread, which must never block on messaging,
        so the publish is handed to the node loop like every other."""
        if self._clock_publisher is None:
            return
        engine_time_s = self._engine_time_s
        if engine_time_s is None:
            raise RuntimeError(
                "a domain's publisher must record an engine step before publishing"
            )
        time_ns = int(engine_time_s * 1e9)
        try:
            self._loop.call_soon_threadsafe(self._publish_clock_tick_on_loop, time_ns)
        except RuntimeError:
            # Loop closed during shutdown; the tick is dropped with the rest.
            pass

    def _publish_clock_tick_on_loop(self, time_ns: int) -> None:
        if self._clock_in_flight:
            self._clock_latched_ns = time_ns
            return
        self._clock_in_flight = True
        task = asyncio.ensure_future(self._clock_publisher.publish(time_ns))
        task.add_done_callback(self._on_clock_tick_published)

    def _on_clock_tick_published(self, task: asyncio.Task) -> None:
        """Publish failures are latched, not repeated: one error line when the
        domain stops being published, one line when it recovers, never a
        warning per tick."""
        self._clock_in_flight = False
        if not task.cancelled():
            error = task.exception()
            if error is not None and not self._clock_publish_failed:
                self._clock_publish_failed = True
                logger.error(f"the clock domain is not being published: {error}")
            elif error is None and self._clock_publish_failed:
                self._clock_publish_failed = False
                logger.info("the clock domain is being published again")
        latched_ns = self._clock_latched_ns
        self._clock_latched_ns = None
        if latched_ns is not None:
            self._publish_clock_tick_on_loop(latched_ns)

    def timestamp_s(self) -> float:
        """The instant this engine stamps its own state with: its engine clock
        while it publishes a clock domain, its bound clock otherwise. A
        publisher's stamp is the instant it committed, which is the instant
        its tick carries."""
        if self._clock_publisher is None:
            return clock.now_ns() / 1e9
        engine_time_s = self._engine_time_s
        if engine_time_s is None:
            raise RuntimeError(
                "a domain's publisher must record an engine step before stamping"
            )
        return engine_time_s

    def publish_arm_states(
        self, robot: str, arm: str, positions: list[float], velocities: list[float]
    ) -> None:
        pub = self._arm_pubs.get(arm)
        if pub is not None:
            # Efforts are empty: the engine measures no joint torques.
            payload = _ARM_SLOTS[arm][1].build_message(
                self.timestamp_s(), positions, velocities, []
            )
            self._schedule_publish(pub, _ARM_SLOTS[arm][0], robot, payload)

    def publish_gripper_states(
        self, robot: str, gripper: str, opening: float, force: float = 0.0
    ) -> None:
        pub = self._gripper_pubs.get(gripper)
        if pub is not None:
            # The engine torque rides as the pairing effort; the ceiling is 0
            # (no effort control).
            payload = _GRIPPER_SLOTS[gripper][1].build_message(
                self.timestamp_s(), opening, force, 0.0
            )
            self._schedule_publish(pub, _GRIPPER_SLOTS[gripper][0], robot, payload)

    def publish_color_frame(
        self,
        robot: str,
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
        return self._publish_guarded(
            robot, name, _FRAMES_SURFACE, [(_VIDEO_STREAM, payload)]
        )

    def publish_color_stream_info(
        self,
        robot: str,
        name: str,
        width: int,
        height: int,
        frames_per_second: int,
        encoding: str,
    ) -> None:
        _, info = _COLOR_CAMERA_SLOTS[name]
        payload = info.build_message(width, height, frames_per_second, encoding)
        self._publish_guarded(robot, name, _INFO_SURFACE, [(_STREAM_INFO, payload)])

    def publish_rgbd_frames(
        self,
        robot: str,
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
            robot,
            name,
            _FRAMES_SURFACE,
            [(_VIDEO_STREAM, color_payload), (_DEPTH_STREAM, depth_payload)],
        )

    def publish_rgbd_stream_info(
        self,
        robot: str,
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
        self._publish_guarded(robot, name, _INFO_SURFACE, [(_STREAM_INFO, payload)])

    def capture_timestamp_s(self) -> Optional[float]:
        """The instant of a state capture taken now, on the timeline the
        joint states are stamped on. None while this engine publishes a clock
        domain and has not recorded a step: such a capture has no instant yet,
        so it is not a snapshot."""
        if self._clock_publisher is not None and self._engine_time_s is None:
            return None
        return self.timestamp_s()

    def publish_object_states(self, snapshot) -> bool:
        """Publish one object-state snapshot, stamped with the instant it was
        captured. False when the previous snapshot is still in flight: this
        one is dropped, and the next replaces it."""
        pub = self._object_states_pub
        if pub is None:
            return False
        payload = object_states.build_message(
            snapshot.timestamp_s,
            [object_states.MessageObjectsItem(**record.fields()) for record in snapshot.objects],
        )
        return self._publish_batch(
            self._object_states_guard, object_states.TOPIC_NAME, [(pub, payload)]
        )

    def publish_color_geometry(self, robot: str, name: str, color) -> None:
        """Where a colour camera's pixels point: `color` is the pinhole model
        of its stream (width, height, fx, fy, cx, cy), rendered and so without
        distortion."""
        payload = _CAMERA_GEOMETRY[name].build_message(
            width=color.width,
            height=color.height,
            fx=color.fx,
            fy=color.fy,
            cx=color.cx,
            cy=color.cy,
            distortion_model=_RENDERED_DISTORTION,
            distortion=[],
        )
        self._publish_guarded(robot, name, _GEOMETRY_SURFACE, [(_GEOMETRY, payload)])

    def publish_rgbd_geometry(
        self,
        robot: str,
        name: str,
        color,
        depth,
        depth_model: str,
        min_depth_m: float,
        max_depth_m: float,
        align_mode: str,
        depth_to_color_position: tuple[float, float, float],
        depth_to_color_orientation: tuple[float, float, float, float],
    ) -> None:
        """Where an rgbd camera's pixels point and how its depth sits against
        its colour: `color` and `depth` are the pinhole models of the two
        streams, each at its own published size."""
        payload = _CAMERA_GEOMETRY[name].build_message(
            width=color.width,
            height=color.height,
            fx=color.fx,
            fy=color.fy,
            cx=color.cx,
            cy=color.cy,
            distortion_model=_RENDERED_DISTORTION,
            distortion=[],
            depth_width=depth.width,
            depth_height=depth.height,
            depth_fx=depth.fx,
            depth_fy=depth.fy,
            depth_cx=depth.cx,
            depth_cy=depth.cy,
            depth_distortion_model=_RENDERED_DISTORTION,
            depth_distortion=[],
            depth_model=depth_model,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            align_mode=align_mode,
            depth_to_color_position=list(depth_to_color_position),
            depth_to_color_orientation=list(depth_to_color_orientation),
        )
        self._publish_guarded(robot, name, _GEOMETRY_SURFACE, [(_GEOMETRY, payload)])

    async def _declare_camera_publishers(
        self, name: str, topics: list[tuple[str, object]]
    ) -> None:
        for topic_name, module in topics:
            self._camera_pubs[(name, topic_name)] = await module.declare_publisher(
                self._node_runner
            )

    def _camera_guard(self, robot: str, name: str, surface: str) -> "_PublishGuard":
        with self._cmd_lock:
            return self._camera_guards.setdefault(
                (robot, name, surface), _PublishGuard()
            )

    def _publish_guarded(
        self, robot: str, name: str, surface: str, topic_payloads: list[tuple[str, bytes]]
    ) -> bool:
        """One robot's camera batch, behind that robot's own guard on this
        surface. False when the robot holds no pair on this camera's slot, so
        there is nothing to publish to."""
        camera = _COLOR_CAMERA_SLOTS.get(name) or _RGBD_CAMERA_SLOTS[name]
        peer = self._peer_of(camera[0], robot)
        if peer is None:
            return False
        publishes = [
            (self._camera_pubs[(name, topic_name)], payload)
            for topic_name, payload in topic_payloads
        ]
        return self._publish_batch(
            self._camera_guard(robot, name, surface),
            f"'{robot}' {name} {surface}",
            publishes,
            peer=peer,
        )

    def _publish_batch(
        self,
        guard: _PublishGuard,
        surface: str,
        publishes: list[tuple[peppylib.TopicPublisher, bytes]],
        peer=None,
    ) -> bool:
        """False when this surface's previous batch is still in flight, so the
        caller knows the sample never reached a consumer. `peer` is the pair a
        pairing publisher addresses; the scene's own streams have none."""
        if not guard.try_acquire(surface):
            return False

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
                    task = asyncio.ensure_future(
                        pub.publish_to(peer, payload) if peer is not None else pub.publish(payload)
                    )
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

    def _schedule_publish(
        self,
        publisher: peppylib.PeerPublisher,
        module,
        robot: str,
        payload: bytes,
    ) -> None:
        # Hand the publish to the node loop and return immediately; the physics
        # thread must never block on messaging. The pair this robot's limb
        # reaches is read on the loop, so a publish to a robot whose pair ended
        # between the step and the publish is dropped.
        def _publish() -> None:
            peer = self._peer_of(module, robot)
            if peer is None:
                return
            task = asyncio.ensure_future(publisher.publish_to(peer, payload))
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
