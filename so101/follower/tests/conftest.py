"""Shared fakes: a scripted FollowerHardware with injectable failures."""

from __future__ import annotations

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
