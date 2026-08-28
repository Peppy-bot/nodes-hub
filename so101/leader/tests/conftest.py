"""Shared fakes: a scripted LeaderHardware with injectable failures."""

from __future__ import annotations

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
