import asyncio
import contextlib
import time
from types import SimpleNamespace

import numpy as np

import peppygen.clock
from peppygen.fixtures import harness
from peppygen.parameters import Parameters

from xr_commander.frames import Pose

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
POSE = Pose(np.zeros(3), IDENTITY)


def default_parameters(**overrides) -> SimpleNamespace:
    """The full raw launch-parameter set, valid by default."""
    values = {
        "command_rate_hz": 1000,
        "https_port": 4443,
        "https_host": "0.0.0.0",
        "motion_scale": 1.0,
        "gripper_open_fraction": 1.0,
        "posture_move_duration_s": 2.0,
        "stale_timeout_s": 10.0,
        "view_max_width": 640,
        "status_panel_enabled": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


async def idle_setup(_params, _node_runner):
    """A setup that starts nothing: the real `setup` raises TLS, an HTTPS
    server, and a WebXR session, none of which belongs in a test. Harness
    tests boot the node's runtime (ephemeral router, generated mocks, seeded
    slots) under this no-op and drive the production functions directly
    against the real runner and real generated modules."""
    return []


def asgi_request(
    app, method: str, path: str, *, body=b"", headers: dict | None = None
) -> SimpleNamespace:
    """One request through an ASGI app, without a web client library.

    The node ships no HTTP client, and these routes are worth testing where
    they are mounted: their ordering against the frontend mount is the hazard.
    `body` takes a list of chunks to arrive as a streamed body rather than one
    delivery.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "https",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (name.lower().encode(), value.encode())
            for name, value in (headers or {}).items()
        ],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 4443),
    }
    sent = []
    pending = list(body) if isinstance(body, list) else [body]

    async def receive():
        if not pending:
            return {"type": "http.disconnect"}
        chunk = pending.pop(0)
        return {"type": "http.request", "body": chunk, "more_body": bool(pending)}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    payload = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    return SimpleNamespace(
        status=start["status"],
        headers={key.decode(): value.decode() for key, value in start["headers"]},
        body=payload.decode("utf-8", "replace"),
    )


def boot(**kwargs):
    """The generated harness under `idle_setup`: awaitable or an async
    context manager. Keyword arguments are `harness.start`'s: per-slot mock
    counts (`<link_id>_instances`) and the usual overrides."""
    kwargs.setdefault("parameters", Parameters(**vars(default_parameters())))
    return harness.start(idle_setup, **kwargs)


@contextlib.asynccontextmanager
async def boot_clocked(**kwargs):
    """[`boot`] with the node's clock initialized: for tests that await
    time-based node behavior. Shuts down on exit, like `boot`'s context
    form."""
    h = await boot(**kwargs)
    try:
        await peppygen.clock.init(h.node_runner)
        yield h
    finally:
        await h.shutdown()


async def eventually(predicate, message="condition"):
    """Poll until `predicate()` is truthy; AssertionError past 10s.

    The bounded-wait primitive for harness tests: wire delivery is ordered
    but not instant, so state assertions converge instead of sleeping blind.
    """
    timeout = 10.0
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"{message}: not met within {timeout}s")
        await asyncio.sleep(0.005)


async def press_until_goal(session, mock_action, release, press):
    """Release-then-press cycles until the node fires: buttons re-arm only
    once the previous goal's watcher finished, so the retry absorbs that tail
    without sleeping blind. `release`/`press` are the two `session.press`
    kwargs dicts."""
    deadline = time.monotonic() + 10.0
    while True:
        session.press(**release)
        await asyncio.sleep(0.03)
        session.press(**press)
        try:
            return await mock_action.next_goal(0.5)
        except TimeoutError:
            if time.monotonic() >= deadline:
                raise


async def tap_x(session, *, settle=0.05):
    """One observed X press edge: release, let the tracker see it, press.
    The sleep spans the tracker's frame cadence, not a node wait."""
    session.press(x=False)
    await asyncio.sleep(settle)
    session.press(x=True)


async def stop_and_save(session, active, cancelled_result, *, settle=0.05):
    """The second X press of an episode: stop-and-save, not a fresh goal —
    cancels the active goal and completes it with the recorder's verdict."""
    session.press(x=False)
    await asyncio.sleep(settle)
    session.press(x=True)
    await asyncio.wait_for(active.cancel_signal(), 10.0)
    await active.complete_cancelled(cancelled_result)


class RecordingSink:
    """A FrameSink that keeps every frame it was handed."""

    def __init__(self):
        self.frames = []

    def put_frame(self, frame):
        self.frames.append(frame)


class FakeSubscription:
    """Yields queued items, then pends forever like the generated one.

    Kept for the pure bus-plumbing tests (`messages`/`next_message`), which
    exercise receive-loop edge cases no real subscription can be scripted
    into (mid-stream exceptions, self-cancelling receives)."""

    def __init__(self, items):
        self._items = list(items)

    async def next(self):
        if self._items:
            return self._items.pop(0)
        await asyncio.Event().wait()


@contextlib.asynccontextmanager
async def running_drain(make_drain):
    """One drain task for the body's duration: a hand-fired token cancels it
    on exit (never the runner's real token, which would converge the whole
    node), and the task is awaited bounded so a hung drain fails the test
    instead of leaking."""
    token = FakeToken()
    drain = asyncio.create_task(make_drain(token))
    try:
        yield drain
    finally:
        token.cancel()
        await asyncio.wait_for(drain, 5.0)


class FakeToken:
    """A cancellation token the test fires by hand.

    Kept deliberately alongside the harness: the production tasks accept any
    token with the CancellationToken shape, and a test must be able to end
    one drain without cancelling the runner's real token (which would
    converge the whole node mid-test). The pure loop tests have no runner at
    all."""

    def __init__(self):
        self._event = asyncio.Event()

    def cancel(self):
        self._event.set()

    def is_cancelled(self):
        return self._event.is_set()

    async def cancelled(self):
        await self._event.wait()
