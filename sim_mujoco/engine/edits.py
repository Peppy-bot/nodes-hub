#!/usr/bin/env python3
"""The changes to the scene that the thread stepping it has to make.

Standing a robot composes and compiles the scene again, which no other
thread may do while physics reads it, so the contract server hands the work
over here and waits for the answer.
"""

from __future__ import annotations

import queue
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable


@dataclass
class Edit:
    """One change to the scene, run on the thread that steps it."""

    work: Callable[[], object]
    future: Future


class Edits:
    """The changes waiting for the thread that steps the scene."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Edit] = queue.Queue()

    def submit(self, work) -> Future:
        """Hands work to the sim thread. The caller awaits the future."""
        future: Future = Future()
        self._queue.put(Edit(work=work, future=future))
        return future

    def drain(self) -> int:
        """Runs everything waiting, on the calling thread, and answers with
        how many changes were made."""
        made = 0
        while True:
            try:
                edit = self._queue.get_nowait()
            except queue.Empty:
                return made
            if not edit.future.set_running_or_notify_cancel():
                continue
            try:
                edit.future.set_result(edit.work())
                made += 1
            except Exception as error:  # pylint: disable=W0718
                edit.future.set_exception(error)

    def cancel_all(self, reason: str) -> None:
        """Fails everything still waiting, for a scene that is shutting down."""
        while True:
            try:
                edit = self._queue.get_nowait()
            except queue.Empty:
                return
            if edit.future.set_running_or_notify_cancel():
                edit.future.set_exception(RuntimeError(reason))
