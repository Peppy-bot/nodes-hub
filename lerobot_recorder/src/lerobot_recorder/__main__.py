"""Wiring: parse parameters, discover the cameras, run one drain task per
observed link and per camera stream into the latest-value cache, and serve
record_episode.

Drain rule: a drain task only decodes and caches; anything slow (dataset
writes, uploads) lives elsewhere, so a stalled sink can never block message
reception.
"""

from __future__ import annotations

import asyncio
import datetime
import json
from pathlib import Path

import peppygen.clock
from peppygen import NodeBuilder, NodeRunner
from peppygen.consumed_topics.color_cameras import (
    video_stream as color_cameras_video_stream,
)
from peppygen.consumed_topics.rgbd_cameras import (
    depth_stream as rgbd_cameras_depth_stream,
)
from peppygen.consumed_topics.rgbd_cameras import (
    video_stream as rgbd_cameras_video_stream,
)
from peppygen.exposed_services import finish_session as finish_session_svc
from peppygen.exposed_services import resume_session as resume_session_svc
from peppygen.parameters import Parameters

from . import plan as plan_mod
from . import recording, storage
from .episode import Recorder
from .recording import Cache, CameraFrame, stamp_to_ns
from .sink import Sink

# Backoff after a failed receive or service request, so a persistently
# failing endpoint logs at a readable rate instead of spinning.
RECEIVE_RETRY_S = 0.1
SERVICE_RETRY_S = 1.0


async def _drain(topic_module, route, token, label: str) -> None:
    """Route every (source, message) from one held subscription into the
    cache; the generated subscription fans in every member bound to the slot,
    tagging each message with the member that sent it. One bad message never
    kills the drain: a decode or route error is logged and the next message is
    served. The drain returns when the node shuts down."""
    subscription = await topic_module.subscribe(token.node_runner)
    cancelled = asyncio.ensure_future(token.cancelled())
    try:
        while not token.is_cancelled():
            receive = asyncio.ensure_future(subscription.next())
            await asyncio.wait([cancelled, receive], return_when=asyncio.FIRST_COMPLETED)
            if not receive.done():
                receive.cancel()
                return
            try:
                received = receive.result()
            except Exception as e:
                print(f"[recorder] {label} receive error: {e!r}")
                await asyncio.sleep(RECEIVE_RETRY_S)
                continue
            if received is None:
                return  # subscription closed: node shutting down
            tag, message = received
            try:
                route(tag, message)
            except Exception as e:
                # One bad message must not kill the drain for the stream.
                print(f"[recorder] {label} route error: {e!r}")
    finally:
        cancelled.cancel()


class _DrainToken:
    """The cancellation token plus the runner _drain needs to subscribe."""

    def __init__(self, node_runner: NodeRunner):
        self.node_runner = node_runner
        self._token = node_runner.cancellation_token()

    def is_cancelled(self) -> bool:
        return self._token.is_cancelled()

    async def cancelled(self) -> None:
        await self._token.cancelled()


def _route_link(cache: Cache, kind: plan_mod.LinkKind, parse_sample):
    def route(source, message):
        sample = parse_sample(kind, message)
        if sample is None:
            return
        # The observed member set is live, but the recorded shape is the plan
        # captured at startup, so a member that joined since then has no slot
        # to write into and its messages are not recorded.
        key = plan_mod.source_key(source)
        if key in cache.links:
            cache.links[key] = sample

    return route


def _link_drains(token: _DrainToken, cache: Cache) -> list:
    return [
        _drain(topic_module, _route_link(cache, kind, parse), token, topic_module.LINK_ID)
        for kind, observed, commanded in plan_mod.limb_slots()
        for topic_module, parse in (
            (observed, recording.state_sample),
            (commanded, recording.action_sample),
        )
    ]


def _route_frames(slots: list[CameraFrame | None], index: dict[plan_mod.ProducerKey, int]):
    def route(producer, m):
        slot = index.get(plan_mod.producer_key(producer))
        if slot is None:
            return
        stamp = stamp_to_ns(m.header.stamp)
        if stamp is None:
            return
        slots[slot] = CameraFrame(
            encoding=m.encoding,
            width=m.width,
            height=m.height,
            data=m.frame,
            stamp_ns=stamp,
        )

    return route


def _recorded_links(plan: plan_mod.RecordingPlan) -> dict:
    """Which source fed which dataset dimension, as the session records it.
    Keys are full source identities: the label alone drops core_node, and two
    sources differing only there would collapse into one entry and hide a
    swap between them."""
    links = {
        "state_links": {"/".join(e.key): e.feature_key for e in plan.state},
        "action_links": {"/".join(e.key): e.feature_key for e in plan.action},
    }
    assert len(links["state_links"]) == len(plan.state)
    assert len(links["action_links"]) == len(plan.action)
    return links


def _check_same_links(session_dir: Path, plan: plan_mod.RecordingPlan) -> None:
    """Refuse a session whose limbs were fed by different sources than this
    launch binds. The dataset's own features cannot catch a swap: limbs are
    named after their follower, so relaunching with the leaders bound in the
    other order keeps every name and shape identical while mirroring the two
    arms' setpoints into each other's action columns. Every session this node
    opens records its links, so a session without them is not one of ours."""
    provenance = session_dir / "session.json"
    try:
        recorded = json.loads(provenance.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"session.json unreadable: {e}") from e
    for field, bound in _recorded_links(plan).items():
        was = recorded.get(field)
        if was is None:
            raise ValueError(f"session.json records no {field}; not a session this node wrote")
        if was != bound:
            raise ValueError(
                f"session recorded {field} {was}, this launch binds {bound}; "
                f"appending would cross the limbs already in the dataset"
            )


def _resolve_existing_session(
    target: storage.StorageTarget, name: str, plan: plan_mod.RecordingPlan
) -> Path:
    """The one path where an operator-supplied name reaches the filesystem:
    validated before it touches a path, then checked against the session's
    recorded provenance."""
    storage.validate_session_name(name)
    session_dir = target.root / name
    if not session_dir.is_dir():
        raise ValueError(f"no session named {name} under this storage target")
    _check_same_links(session_dir, plan)
    return session_dir


def _write_session_json(session_dir: Path, params: Parameters, plan: plan_mod.RecordingPlan) -> None:
    provenance = {
        "robot_type": params.robot_type,
        "fps": params.fps,
        "storage_root": params.storage_root,
        "s3_uri": params.s3_uri,
        **_recorded_links(plan),
        "color_cameras": ["/".join(e.key) for e in plan.color_cameras],
        "rgbd_cameras": ["/".join(e.key) for e in plan.rgbd_cameras],
    }
    (session_dir / "session.json").write_text(json.dumps(provenance, indent=2) + "\n")


async def setup(params: Parameters, node_runner: NodeRunner) -> list[asyncio.Task]:
    if params.fps <= 0:
        raise ValueError("fps must be positive")
    # add_frame runs on the event loop, so the library must encode images on
    # its own threads: at zero it writes every frame's PNG inline and stalls
    # the sampler and the drains behind it.
    if params.image_writer_threads <= 0:
        raise ValueError("image_writer_threads must be positive")
    target = storage.parse_storage(params.storage_root, params.s3_uri)
    storage.ensure_mounted(target)
    await asyncio.to_thread(storage.probe_credentials, target)
    if target.s3 is not None:
        # The full resolved destination, so a wrong or missing endpoint
        # environment is visible at launch instead of at the first upload.
        print(
            f"[recorder] mirroring to {storage.resolved_endpoint()}, "
            f"bucket {target.s3.bucket!r}, prefix {target.s3.prefix!r}"
        )

    # Staleness ages producer stamps against this daemon-resolved clock (sim
    # time under a simulated clock); producers stamp from the same clock.
    await peppygen.clock.init(node_runner)

    # The boot config seeds every observer slot with the launch's membership,
    # so the robot's shape is readable right here in setup.
    plan = plan_mod.discover(node_runner, plan_mod.read_limb_sources(node_runner))
    print(
        f"[recorder] {len(plan.state)} state link(s), "
        f"{len(plan.color_cameras)} color camera(s), {len(plan.rgbd_cameras)} rgbd camera(s)"
    )

    def session_handles(session_dir: Path, session_name: str):
        sink = Sink(
            root=session_dir / "dataset",
            repo_id=f"{plan_mod.sanitize_key(params.robot_type)}/session",
            robot_type=params.robot_type,
            fps=params.fps,
            image_writer_threads=params.image_writer_threads,
        )
        mirror = (
            storage.Mirror(dest=target.s3, session_dir=session_dir, session_name=session_name)
            if target.s3 is not None
            else None
        )
        return session_dir, sink, mirror

    def open_session():
        session_name = datetime.datetime.now(datetime.UTC).strftime(storage.SESSION_NAME_FORMAT)
        session_dir = target.root / session_name
        # exist_ok=False: two sessions opened in the same second must not
        # interleave into one directory.
        session_dir.mkdir(parents=True, exist_ok=False)
        print(f"[recorder] session at {session_dir}")
        _write_session_json(session_dir, params, plan)
        return session_handles(session_dir, session_name)

    def open_existing(name: str):
        return session_handles(_resolve_existing_session(target, name, plan), name)

    session_dir, sink, mirror = open_session()
    cache = Cache.for_plan(plan)
    recorder = Recorder(
        plan,
        cache,
        sink,
        session_dir,
        params.max_staleness_s,
        params.min_remaining_disk_bytes,
        mirror,
        open_session,
        open_existing,
    )
    recorder_task = asyncio.create_task(recorder.run(node_runner))
    warmup_task = asyncio.create_task(recorder.warm_up(node_runner))
    finish_task = asyncio.create_task(
        _serve(node_runner, finish_session_svc, _finish_session_handler(recorder), "finish_session")
    )
    resume_task = asyncio.create_task(
        _serve(
            node_runner,
            resume_session_svc,
            _resume_session_handler(node_runner, recorder),
            "resume_session",
        )
    )

    # One hook so the order is guaranteed: the in-flight episode saves and
    # completes, in-flight uploads land, the writers close (parquet footers),
    # and only then does the final mirror pass upload the now-valid files.
    async def close_session():
        warmup_task.cancel()
        finish_task.cancel()
        resume_task.cancel()
        # Awaited, not just cancelled: a warm-up mid-create or an in-flight
        # finish or resume must be out of the recorder before finalize runs
        # under it.
        await asyncio.gather(warmup_task, finish_task, resume_task, return_exceptions=True)
        try:
            await recorder_task
        except Exception as e:
            print(f"[recorder] recorder task failed: {e!r}")
        await recorder.close_for_shutdown()

    node_runner.on_shutdown(close_session)

    token = _DrainToken(node_runner)
    camera_drains = [
        _drain(topic_module, _route_frames(slots, index), token, label)
        for topic_module, slots, index, label in (
            (color_cameras_video_stream, cache.color, plan.color_index, "color camera"),
            (rgbd_cameras_video_stream, cache.rgbd_video, plan.rgbd_index, "rgbd camera"),
            (rgbd_cameras_depth_stream, cache.rgbd_depth, plan.rgbd_index, "rgbd depth"),
        )
    ]
    return [
        asyncio.create_task(coro)
        for coro in _link_drains(token, cache) + camera_drains
    ] + [recorder_task]


async def _serve(node_runner: NodeRunner, svc, handler, label: str) -> None:
    """Answer one exposed service until the node shuts down. A handler that
    raises must not end the loop: the service would then never answer again."""
    token = node_runner.cancellation_token()
    while not token.is_cancelled():
        try:
            await svc.handle_next_request(node_runner, handler)
        except Exception as e:
            # CancelledError is a BaseException, so shutdown's cancel passes.
            print(f"[recorder] {label} service error: {e!r}")
            await asyncio.sleep(SERVICE_RETRY_S)


def _resume_session_handler(node_runner: NodeRunner, recorder: Recorder):
    async def handler(request):
        session, episodes, error = await recorder.resume_session(
            request.data.session, node_runner
        )
        return resume_session_svc.Response(session=session, episodes=episodes, error=error)

    return handler


def _finish_session_handler(recorder: Recorder):
    async def handler(_request):
        session, error = await recorder.finish_session()
        return finish_session_svc.Response(session=session, error=error)

    return handler


def main():
    NodeBuilder().run(setup)


if __name__ == "__main__":
    main()
