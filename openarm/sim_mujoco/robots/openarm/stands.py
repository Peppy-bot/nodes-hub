"""The handoff between the node loop, where a robot attaches, and the thread
that steps the scene, which is the only one that may load a model or let
one go.

A stand asks the thread to load a robot's scene and step it; an unstand
asks it to let the standing scene go. Each is answered through a future the
node loop awaits, once the thread has done it. The thread reads stands while
it is idle and unstands between steps.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from head_camera import Pack


@dataclass
class Stand:
    """One request to stand a robot: the scene file to load, the robot it
    is loaded for, and the future its answer goes to."""

    scene: Path
    robot: str
    # The head camera pack this robot's model draws, or None for a model
    # whose robot has no head camera.
    head_camera_pack: Optional[Pack]
    future: Future


class Stands:
    """The requests waiting for the thread that steps the scene."""

    def __init__(self) -> None:
        self._stands: queue.Queue[Stand] = queue.Queue()
        self._unstand: Optional[Future] = None
        self._lock = threading.Lock()

    # --- the node loop ---

    def stand(
        self, scene: Path, robot: str, *, head_camera_pack: Optional[Pack]
    ) -> Future:
        """Asks the thread to load `scene`, seat the head camera of
        `head_camera_pack` on it where the model draws one, and step it for
        `robot`. The future is done once the scene is stepping, or carries
        why it could not be loaded."""
        future: Future = Future()
        self._stands.put(
            Stand(scene=scene, robot=robot, head_camera_pack=head_camera_pack, future=future)
        )
        return future

    def unstand(self) -> Future:
        """Asks the thread to let the standing scene go. The future is done
        once nothing stands, which is at once when nothing does. One unstand
        is pending at a time: a second asker shares the first's answer."""
        with self._lock:
            if self._unstand is None:
                self._unstand = Future()
            return self._unstand

    # --- the thread that steps the scene ---

    def next_stand(self, timeout_s: float) -> Optional[Stand]:
        """The next stand to make, or None when none arrived in time."""
        try:
            return self._stands.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def unstand_pending(self) -> bool:
        with self._lock:
            return self._unstand is not None

    def take_unstand(self) -> Optional[Future]:
        """The pending unstand, taken so it is answered exactly once."""
        with self._lock:
            future, self._unstand = self._unstand, None
            return future

    def cancel_all(self, reason: str) -> None:
        """Fails everything still waiting, for an engine that is stopping."""
        pending = self.take_unstand()
        if pending is not None and pending.set_running_or_notify_cancel():
            pending.set_exception(RuntimeError(reason))
        while True:
            try:
                stand = self._stands.get_nowait()
            except queue.Empty:
                return
            if stand.future.set_running_or_notify_cancel():
                stand.future.set_exception(RuntimeError(reason))
