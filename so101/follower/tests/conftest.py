"""Shared fakes and waits: a scripted FollowerHardware with injectable
failures, and a blocking and an async wait on a condition."""

from __future__ import annotations

import asyncio
import time

import pytest
from so101_description.units import MOTOR_NAMES


class FakeHardware:
    """FollowerHardware that records writes and serves scripted reads."""

    def __init__(self):
        self.connected = False
        self.disconnected = False
        self.written_goals: list[dict[str, float]] = []
        self.positions = dict.fromkeys(MOTOR_NAMES, 0.0)
        self.temps = tuple(30.0 for _ in MOTOR_NAMES)
        self.loads = tuple(0.1 for _ in MOTOR_NAMES)
        self.torque = tuple(True for _ in MOTOR_NAMES)
        self.faults = tuple(0 for _ in MOTOR_NAMES)
        self.fail_connect: Exception | None = None
        # Independent failure taps: state and health reads fail separately,
        # exactly as a single bad register would on hardware.
        self.fail_position_reads = 0
        self.fail_health_reads = 0

    def connect(self):
        if self.fail_connect is not None:
            raise self.fail_connect
        self.connected = True

    def read_positions(self):
        if self.fail_position_reads > 0:
            self.fail_position_reads -= 1
            raise OSError("scripted read failure")
        return dict(self.positions)

    def write_goals(self, goals_by_motor):
        self.written_goals.append(dict(goals_by_motor))

    def read_health(self):
        if self.fail_health_reads > 0:
            self.fail_health_reads -= 1
            raise OSError("scripted read failure")
        return self.temps, self.loads, self.torque, self.faults

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


async def wait_for_async(predicate, timeout_s=10.0):
    """Wait on the event loop until the predicate holds, or fail the test."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"condition never held within {timeout_s}s")
