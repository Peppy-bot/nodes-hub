"""Typed peppygen pairing IO for the MuJoCo simulation.

Bridges the physics thread (sync, runs the engine step in an executor) to the
node_runner's asyncio loop, where peppygen pairing pub/sub lives. The engine
plays the follower role of every limb's joint_link / gripper_link pairing and
of every base's joint_link pairing on three slots, `arms`, `grippers` and
`bases`, each holding any number of pairs: one consume task per slot holds a
generated `subscribe()` subscription (gap-free, in-order) and hands each
setpoint to sim_robot_core's `Setpoints`, which keeps the latest of each
limb and base of each robot; the physics thread reads those and publishes
the robot's stamped measured state back on the pair of the same limb or
base. A pair carries the copy its robot attached under and the link it comes
from on the robot's side, which is what tells one robot's parts from
another's and one part from the next (sim_robot_core.pairs), and a pair of a
robot the engine does not stand is left alone. The engine also plays the
camera role of the sim_rgb_camera_link / sim_rgbd_camera_link pairings on
`rgb_cameras` and `rgbd_cameras`, one pair per rendered camera named after
its relay, handing finished frames to the same loop and dropping a frame
whose stream's previous publish is still in flight. Every hop is a generated
peppygen pairing topic.

Beside the pairings, the engine implements object_state: the bridge captures
one record per standing robot on each telemetry tick, stamped as that tick's
states are, the object_states topic publishes each capture, and
get_object_states answers the latest one. Its scene spawns no object, so a
capture lists none.

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
from functools import partial
from typing import Optional

import peppylib
from control_core_py.geometry import Pose
from peppygen import clock
from peppygen.emitted_topics.objects import object_states
from peppygen.exposed_services.objects import get_object_states
from peppylib.clock import ClockPublisher
from peppygen.paired_topics.arms import joint_setpoints as arm_setpoints
from peppygen.paired_topics.arms import joint_states as arm_states
from peppygen.paired_topics.bases import joint_setpoints as base_setpoints
from peppygen.paired_topics.bases import joint_states as base_states
from peppygen.paired_topics.grippers import gripper_setpoints
from peppygen.paired_topics.grippers import gripper_states
from peppygen.paired_topics.rgb_cameras import geometry as rgb_geometry
from peppygen.paired_topics.rgb_cameras import stream_info as rgb_info
from peppygen.paired_topics.rgb_cameras import video_stream as rgb_video
from peppygen.paired_topics.rgbd_cameras import depth_stream as rgbd_depth
from peppygen.paired_topics.rgbd_cameras import geometry as rgbd_geometry
from peppygen.paired_topics.rgbd_cameras import stream_info as rgbd_info
from peppygen.paired_topics.rgbd_cameras import video_stream as rgbd_video
from sim_robot_core.cameras import DISTORTION_MODEL
from sim_robot_core.object_state import ObjectStateSnapshot, RobotRecord
from sim_robot_core.pairs import (
    ARMS,
    BASE_NAME,
    BASES,
    GRIPPERS,
    RGB_CAMERAS,
    RGBD_CAMERAS,
    Held,
    PairTable,
)
from sim_robot_core.setpoints import Setpoints
from sim_robot_core.world import WORLD

logger = logging.getLogger(__name__)

# The topic each driven slot's setpoints arrive on.
_SETPOINT_TOPICS = (
    (ARMS, arm_setpoints),
    (GRIPPERS, gripper_setpoints),
    (BASES, base_setpoints),
)

# The clock topic reserves zero for "no tick observed yet", so the smallest
# instant an engine can carry is one nanosecond. A timeline still sitting at
# zero is a real instant rather than a fault, so it is floored to that
# minimum: the states this engine stamps and the domain's ticks then carry
# the same instant instead of differing by the wire's own clamp.
_MIN_ENGINE_TIME_S = 1e-9

# Publisher and guard keys, spelled once: a publisher is keyed by
# (slot, topic) and a guard by (robot, camera, surface).
_VIDEO_STREAM = "video_stream"
_DEPTH_STREAM = "depth_stream"
_STREAM_INFO = "stream_info"
_GEOMETRY = "geometry"
_FRAMES_SURFACE = "frames"
_INFO_SURFACE = "info"
# Geometry goes out on the same tick as the stream info, so it takes a guard
# of its own: sharing the info one would drop whichever was scheduled second.
_GEOMETRY_SURFACE = "geometry"


# What get_object_states answers until the first capture: not an empty scene.
_NOT_CAPTURED = (
    "MuJoCo has not captured its object state yet; it captures on each state tick once "
    "the scene steps"
)
_PRECONDITION_FAILED = "precondition_failed"


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
    loop, and the setpoints they take, which the physics thread reads each
    step (`setpoints`)."""

    def __init__(
        self,
        node_runner: peppylib.NodeRunner,
        loop: asyncio.AbstractEventLoop,
        robots,
    ) -> None:
        self._node_runner = node_runner
        self._loop = loop
        # The robots in the scene, of which a capture of object_state lists
        # the ones that stand.
        self._robots = robots
        # The five slots, read into robots, limbs, bases and cameras.
        self._pairs = PairTable(
            {
                ARMS: lambda: arm_setpoints.peers(node_runner),
                GRIPPERS: lambda: gripper_setpoints.peers(node_runner),
                BASES: lambda: base_setpoints.peers(node_runner),
                RGB_CAMERAS: lambda: rgb_video.peers(node_runner),
                RGBD_CAMERAS: lambda: rgbd_video.peers(node_runner),
            },
            robots.sole_name,
        )
        self._arm_pub: Optional[peppylib.PeerPublisher] = None
        self._gripper_pub: Optional[peppylib.PeerPublisher] = None
        self._base_pub: Optional[peppylib.PeerPublisher] = None
        # The latest setpoint of each limb and base of each robot, whose
        # models give each limb and base its joint count.
        self.setpoints = Setpoints(self._pairs, robots)
        # Held while a camera's guard is made, on the thread that renders.
        self._guards_lock = threading.Lock()
        # Camera publishers and their in-flight guards, keyed by (slot, topic)
        # and (robot, camera, surface); a slot's publisher reaches every
        # camera paired on it, so the guard is per robot and camera.
        self._camera_pubs: dict[tuple[str, str], peppylib.PeerPublisher] = {}
        self._camera_guards: dict[tuple[str, str, str], _PublishGuard] = {}
        # The object_states publisher and its in-flight guard, and the latest
        # capture, which get_object_states answers: written on the physics
        # thread, read on the loop.
        self._object_states_pub: Optional[peppylib.TopicPublisher] = None
        self._object_states_guard = _PublishGuard()
        self._snapshot: Optional[ObjectStateSnapshot] = None
        self._snapshot_lock = threading.Lock()
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
        node loop before the sim thread starts. A publish to a robot holding
        no pair is dropped, so bringup order never matters."""
        # State timestamps read this instance's bound clock, the one its
        # consumers read too, so samples age on one timeline.
        await clock.init(self._node_runner)
        # A deployment names one instance of a clock domain its publisher, and
        # holding a publisher is that answer.
        self._clock_publisher = await ClockPublisher.for_node(self._node_runner)
        if self._clock_publisher is not None:
            logger.info(f"publishing clock domain {self._clock_publisher.domain}")
        self._arm_pub = await arm_states.declare_publisher(self._node_runner)
        self._gripper_pub = await gripper_states.declare_publisher(self._node_runner)
        self._base_pub = await base_states.declare_publisher(self._node_runner)
        self._object_states_pub = await object_states.declare_publisher(self._node_runner)
        await self._declare_camera_publishers(
            RGB_CAMERAS,
            [(_VIDEO_STREAM, rgb_video), (_STREAM_INFO, rgb_info), (_GEOMETRY, rgb_geometry)],
        )
        await self._declare_camera_publishers(
            RGBD_CAMERAS,
            [
                (_VIDEO_STREAM, rgbd_video),
                (_DEPTH_STREAM, rgbd_depth),
                (_STREAM_INFO, rgbd_info),
                (_GEOMETRY, rgbd_geometry),
            ],
        )
        # The consume tasks end with their subscriptions or the node's
        # cancellation; the service runs for the life of the node.
        runner = self._node_runner
        token = runner.cancellation_token()
        self._tasks = [
            *(
                asyncio.create_task(
                    self.setpoints.consume(slot, partial(topic.subscribe, runner), token)
                )
                for slot, topic in _SETPOINT_TOPICS
            ),
            asyncio.create_task(self._serve_object_states()),
        ]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        # Let the cancellations land so the loops exit before teardown.
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _serve_object_states(self) -> None:
        """Answers get_object_states for the life of the node. When the
        service stops, its request stream ended or it could not listen, the
        log says why and it listens again a second later."""
        while True:
            try:
                await get_object_states.handle_requests(
                    self._node_runner, self._answer_object_states
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # pylint: disable=W0718
                logger.exception("get_object_states service failed")
            else:
                logger.warning("get_object_states service: its request stream ended")
            await asyncio.sleep(1.0)

    def _answer_object_states(self, _request) -> get_object_states.Response:
        """The latest capture, in world. No capture yet is not an empty
        scene: it gives an error reply with `precondition_failed`, as
        object_state:v1 says of an implementer that is not ready."""
        with self._snapshot_lock:
            snapshot = self._snapshot
        if snapshot is None:
            raise get_object_states.Failure(_PRECONDITION_FAILED, _NOT_CAPTURED)
        return get_object_states.Response(
            frame=WORLD,
            timestamp=snapshot.timestamp_s,
            objects=[],
            robots=[
                get_object_states.ResponseRobotsItem(**record.fields())
                for record in snapshot.robots
            ],
        )

    def held_by(self, robot: str) -> Held:
        """What `robot` holds on the five slots: the limbs and the base its
        pairs drive and the cameras they stream. Read on the node loop, which
        is what keeps a robot in the scene and what its readiness asks."""
        return self._pairs.held_by(robot)

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
        but never reorder it. The tick carries the instant that the stamps of
        its step carry on the wire, so a consumer whose clock reads the tick
        has reached every stamp of the step.

        Called from the physics thread, which must never block on messaging,
        so the publish is handed to the node loop like every other."""
        if self._clock_publisher is None:
            return
        engine_time_s = self._engine_time_s
        if engine_time_s is None:
            raise RuntimeError(
                "a domain's publisher must record an engine step before publishing"
            )
        time_ns = _wire_ns(engine_time_s)
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
        if self._arm_pub is None:
            return
        # Efforts are empty: the engine measures no joint torques.
        payload = arm_states.build_message(self.timestamp_s(), positions, velocities, [])
        self._schedule_publish(self._arm_pub, ARMS, robot, arm, payload)

    def publish_base_states(
        self, robot: str, positions: list[float], velocities: list[float]
    ) -> None:
        """The measured angles and rates of `robot`'s wheels, left then
        right, on its base's pair."""
        if self._base_pub is None:
            return
        # Efforts are empty, as an arm's are: the engine measures no joint
        # torques.
        payload = base_states.build_message(self.timestamp_s(), positions, velocities, [])
        self._schedule_publish(self._base_pub, BASES, robot, BASE_NAME, payload)

    def publish_object_states(self, robots: list[RobotRecord]) -> None:
        """Captures object_state as the robots stand now, stamped as this
        tick's states are: the capture get_object_states answers from here
        on, published on the object_states stream unless the previous one is
        still in flight. Of `robots`, the records of the robots in the
        scene, it lists each the registry has stood, as object_state lists a
        robot from the feedback that reports it standing. Called on the
        physics thread."""
        standing = self._robots.standing()
        snapshot = ObjectStateSnapshot(
            timestamp_s=self.timestamp_s(),
            objects=(),
            robots=tuple(record for record in robots if record.robot in standing),
        )
        with self._snapshot_lock:
            self._snapshot = snapshot
        if self._object_states_pub is None:
            return
        payload = object_states.build_message(
            WORLD,
            snapshot.timestamp_s,
            [],
            [object_states.MessageRobotsItem(**record.fields()) for record in snapshot.robots],
        )
        self._publish_batch(
            self._object_states_guard,
            object_states.TOPIC_NAME,
            [(self._object_states_pub, payload)],
        )

    def publish_gripper_states(
        self, robot: str, gripper: str, opening: float, force: float = 0.0
    ) -> None:
        if self._gripper_pub is None:
            return
        # The engine torque rides as the pairing effort; the ceiling is 0
        # (no effort control).
        payload = gripper_states.build_message(self.timestamp_s(), opening, force, 0.0)
        self._schedule_publish(self._gripper_pub, GRIPPERS, robot, gripper, payload)

    def publish_color_frame(
        self,
        robot: str,
        name: str,
        timestamp_s: float,
        frame: str,
        capture_index: int,
        encoding: str,
        width: int,
        height: int,
        data: bytes,
    ) -> bool:
        """One colour picture, its header naming `frame`, the camera's
        frame, its capture time and its capture counter."""
        payload = rgb_video.build_message(
            rgb_video.MessageHeader(
                timestamp=timestamp_s, frame=frame, capture_index=capture_index
            ),
            encoding,
            width,
            height,
            data,
        )
        return self._publish_guarded(
            RGB_CAMERAS, robot, name, _FRAMES_SURFACE, [(_VIDEO_STREAM, payload)]
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
        payload = rgb_info.build_message(width, height, frames_per_second, encoding)
        self._publish_guarded(
            RGB_CAMERAS, robot, name, _INFO_SURFACE, [(_STREAM_INFO, payload)]
        )

    def publish_rgbd_frames(
        self,
        robot: str,
        name: str,
        timestamp_s: float,
        frame: str,
        capture_index: int,
        align_mode: str,
        color: tuple[str, int, int, bytes],
        depth: tuple[str, int, int, bytes],
    ) -> bool:
        """One rgbd capture: color and depth (each an (encoding, width,
        height, data) tuple) publish as a single guarded batch whose headers
        share `frame`, the camera's frame, the timestamp and the
        capture_index, so a pair is dropped or delivered whole."""
        color_encoding, color_width, color_height, color_data = color
        depth_encoding, depth_width, depth_height, depth_data = depth
        color_payload = rgbd_video.build_message(
            rgbd_video.MessageHeader(
                timestamp=timestamp_s,
                frame=frame,
                capture_index=capture_index,
                align_mode=align_mode,
            ),
            color_encoding,
            color_width,
            color_height,
            color_data,
        )
        depth_payload = rgbd_depth.build_message(
            rgbd_depth.MessageHeader(
                timestamp=timestamp_s,
                frame=frame,
                capture_index=capture_index,
                align_mode=align_mode,
            ),
            depth_encoding,
            depth_width,
            depth_height,
            depth_data,
        )
        return self._publish_guarded(
            RGBD_CAMERAS,
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
        payload = rgbd_info.build_message(
            width,
            height,
            frames_per_second,
            encoding,
            depth_width,
            depth_height,
            depth_encoding,
            depth_unit,
        )
        self._publish_guarded(
            RGBD_CAMERAS, robot, name, _INFO_SURFACE, [(_STREAM_INFO, payload)]
        )

    def publish_color_geometry(self, robot: str, name: str, color) -> None:
        """Where a colour camera's pixels point: `color` is the pinhole model
        of its stream (width, height, fx, fy, cx, cy), rendered and so without
        distortion."""
        payload = rgb_geometry.build_message(
            width=color.width,
            height=color.height,
            fx=color.fx,
            fy=color.fy,
            cx=color.cx,
            cy=color.cy,
            distortion_model=DISTORTION_MODEL,
            distortion=[],
        )
        self._publish_guarded(
            RGB_CAMERAS, robot, name, _GEOMETRY_SURFACE, [(_GEOMETRY, payload)]
        )

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
        frame: str,
        depth_to_color: Pose,
    ) -> None:
        """Where an rgbd camera's pixels point and how its depth sits against
        its colour: `color` and `depth` are the pinhole models of the two
        streams, each at its own published size, and `depth_to_color` the
        pose of the depth optical frame in `frame`, the camera's frame, which
        holds at every time, so the record holds no timestamp."""
        payload = rgbd_geometry.build_message(
            width=color.width,
            height=color.height,
            fx=color.fx,
            fy=color.fy,
            cx=color.cx,
            cy=color.cy,
            distortion_model=DISTORTION_MODEL,
            distortion=[],
            depth_width=depth.width,
            depth_height=depth.height,
            depth_fx=depth.fx,
            depth_fy=depth.fy,
            depth_cx=depth.cx,
            depth_cy=depth.cy,
            depth_distortion_model=DISTORTION_MODEL,
            depth_distortion=[],
            depth_model=depth_model,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            align_mode=align_mode,
            depth_to_color=rgbd_geometry.MessageDepthToColor(
                frame=frame,
                position=list(depth_to_color.position.position),
                orientation=list(depth_to_color.orientation.quaternion),
            ),
        )
        self._publish_guarded(
            RGBD_CAMERAS, robot, name, _GEOMETRY_SURFACE, [(_GEOMETRY, payload)]
        )

    async def _declare_camera_publishers(
        self, slot: str, topics: list[tuple[str, object]]
    ) -> None:
        for topic_name, module in topics:
            self._camera_pubs[(slot, topic_name)] = await module.declare_publisher(
                self._node_runner
            )

    def _camera_guard(self, robot: str, name: str, surface: str) -> "_PublishGuard":
        with self._guards_lock:
            return self._camera_guards.setdefault((robot, name, surface), _PublishGuard())

    def _publish_guarded(
        self,
        slot: str,
        robot: str,
        name: str,
        surface: str,
        topic_payloads: list[tuple[str, bytes]],
    ) -> bool:
        """One camera's batch, behind that robot's own guard on this surface.
        False when the robot holds no pair for this camera on `slot`, so
        there is nothing to publish to."""
        peer = self._pairs.peer_of(slot, robot, name)
        if peer is None:
            return False
        publishes = [
            (self._camera_pubs[(slot, topic_name)], payload)
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
        slot: str,
        robot: str,
        limb: str,
        payload: bytes,
    ) -> None:
        # Hand the publish to the node loop and return immediately; the physics
        # thread must never block on messaging. The pair this limb reaches is
        # read on the loop, so a publish to a robot whose pair ended between
        # the step and the publish is dropped.
        def _publish() -> None:
            peer = self._pairs.peer_of(slot, robot, limb)
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


def _wire_ns(time_s: float) -> int:
    """The instant a stamp of `time_s` seconds carries on the wire, in
    nanoseconds: the whole seconds and the nearest nanosecond of the rest, as
    peppylib encodes the timestamp of every message the engine builds."""
    stamp = peppylib.encoding.convert_time(time_s)
    return stamp.sec * peppylib.encoding.NANOS_PER_SEC + stamp.nsec
