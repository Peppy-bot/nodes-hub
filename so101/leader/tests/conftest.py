"""Shared fakes and a wait: a scripted LeaderHardware with injectable
failures, and a blocking wait on a condition."""

from __future__ import annotations

import time

import pytest
from so101_description.units import MOTOR_NAMES


class FakeHardware:
    def __init__(self):
        self.connected = False
        self.disconnected = False
        self.connect_count = 0
        self.positions = dict.fromkeys(MOTOR_NAMES, 0.0)
        self.fail_connect: Exception | None = None
        self.fail_reads = 0
        # Reads fail until a fresh connect, the unplug/replug shape.
        self.dead_until_reconnect = False

    def connect(self):
        if self.fail_connect is not None:
            raise self.fail_connect
        if self.connected and not self.disconnected:
            # lerobot's guard: connecting an already-open port raises.
            raise RuntimeError("already connected")
        self.connected = True
        self.disconnected = False
        self.connect_count += 1
        self.dead_until_reconnect = False

    def read_positions(self):
        if self.dead_until_reconnect:
            raise OSError("dead until reconnect")
        if self.fail_reads > 0:
            self.fail_reads -= 1
            raise OSError("scripted read failure")
        return dict(self.positions)

    def disconnect(self):
        self.disconnected = True


@pytest.fixture
def fake_hardware():
    return FakeHardware()


def wait_for(predicate, timeout_s=2.0):
    """Block until the predicate holds, or fail the test."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"condition never held within {timeout_s}s")
