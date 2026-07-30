import asyncio

import numpy as np

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])


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
