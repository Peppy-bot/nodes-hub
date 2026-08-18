import asyncio
from types import SimpleNamespace

import numpy as np

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


class FakeTopic:
    """A consumed-topic module surface: hands out one prebuilt subscription."""

    def __init__(self, subscription):
        self._subscription = subscription

    async def subscribe(self, _runner):
        return self._subscription


class RecordingSink:
    """A FrameSink that keeps every frame it was handed."""

    def __init__(self):
        self.frames = []

    def put_frame(self, frame):
        self.frames.append(frame)


class FakeSubscription:
    """Yields queued items, then pends forever like the generated one."""

    def __init__(self, items):
        self._items = list(items)

    async def next(self):
        if self._items:
            return self._items.pop(0)
        await asyncio.Event().wait()


class ClosingSubscription:
    """Yields queued items then closes, so a drain loop ends on its own."""

    def __init__(self, items):
        self._items = list(items)

    async def next(self):
        return self._items.pop(0) if self._items else None


class FakeToken:
    """A cancellation token the test fires by hand."""

    def __init__(self):
        self._event = asyncio.Event()

    def cancel(self):
        self._event.set()

    def is_cancelled(self):
        return self._event.is_set()

    async def cancelled(self):
        await self._event.wait()
