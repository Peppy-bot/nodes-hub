import time

import pytest

from so101_leader.device import DeviceLoop


def wait_for(predicate, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_bringup_failure_raises(fake_hardware):
    fake_hardware.fail_connect = RuntimeError("leader arm is not calibrated")
    loop = DeviceLoop(fake_hardware, read_rate_hz=200)
    with pytest.raises(RuntimeError, match="not calibrated"):
        loop.start()


def test_fresh_sample_flows_and_goes_stale_when_reads_fail(fake_hardware):
    loop = DeviceLoop(fake_hardware, read_rate_hz=200)
    loop.start()
    try:
        assert wait_for(lambda: loop.fresh_sample(0.1) is not None)
        # Every read now fails: the sample must age out of the deadman gate.
        fake_hardware.fail_reads = 10_000
        assert wait_for(lambda: loop.fresh_sample(0.1) is None)
    finally:
        loop.stop()
    assert fake_hardware.disconnected


def test_sample_carries_device_units(fake_hardware):
    fake_hardware.positions = {
        "shoulder_pan": 10.0,
        "shoulder_lift": -20.0,
        "elbow_flex": 30.0,
        "wrist_flex": 0.0,
        "wrist_roll": 5.0,
        "gripper": 42.0,
    }
    loop = DeviceLoop(fake_hardware, read_rate_hz=200)
    loop.start()
    try:
        assert wait_for(lambda: loop.fresh_sample(0.5) is not None)
        sample = loop.fresh_sample(0.5)
        assert sample.positions_deg == (10.0, -20.0, 30.0, 0.0, 5.0)
        assert sample.gripper_percent == 42.0
    finally:
        loop.stop()


def test_unplugged_leader_reconnects_and_streams_again(fake_hardware, monkeypatch):
    from so101_leader import device as device_mod

    # Reconnect waits out a window longer than the deadman, so the outage
    # is observable before recovery.
    monkeypatch.setattr(device_mod, "RECONNECT_AFTER_S", 0.15)
    monkeypatch.setattr(device_mod, "RECONNECT_BACKOFF_S", 0.02)
    loop = DeviceLoop(fake_hardware, read_rate_hz=200)
    loop.start()
    try:
        assert wait_for(lambda: loop.fresh_sample(0.1) is not None)
        fake_hardware.dead_until_reconnect = True
        assert wait_for(lambda: loop.fresh_sample(0.1) is None)
        # The device thread reconnects on its own; samples flow again.
        assert wait_for(lambda: fake_hardware.connect_count >= 2)
        assert wait_for(lambda: loop.fresh_sample(0.1) is not None)
    finally:
        loop.stop()


def test_lerobot_leader_rebuilds_a_stale_teleop_before_reconnecting(tmp_path):
    from so101_leader.device import LerobotLeader

    leader = LerobotLeader("/dev/null", str(tmp_path), "leader")
    # A dead fd leaves lerobot's connected flag standing; the wrapper must
    # hand connect a fresh instance instead of tripping the guard.
    leader._teleop.bus.port_handler.is_open = True
    assert leader._fresh_teleop() is not leader._teleop
    leader._teleop.bus.port_handler.is_open = False
    assert leader._fresh_teleop() is leader._teleop
