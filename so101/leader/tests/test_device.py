import pytest
from conftest import wait_for
from so101_description.units import MOTOR_NAMES

from so101_leader.device import DeviceLoop


def test_bringup_failure_raises(fake_hardware):
    fake_hardware.fail_connect = RuntimeError("leader arm is not calibrated")
    loop = DeviceLoop(fake_hardware, read_rate_hz=200)
    with pytest.raises(RuntimeError, match="not calibrated"):
        loop.start()


def test_fresh_sample_flows_and_goes_stale_when_reads_fail(fake_hardware):
    loop = DeviceLoop(fake_hardware, read_rate_hz=200)
    loop.start()
    try:
        wait_for(lambda: loop.fresh_sample(0.1) is not None)
        # Every read now fails: the sample must age out of the deadman gate.
        fake_hardware.fail_reads = 10_000
        wait_for(lambda: loop.fresh_sample(0.1) is None)
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
        wait_for(lambda: loop.fresh_sample(0.5) is not None)
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
        wait_for(lambda: loop.fresh_sample(0.1) is not None)
        fake_hardware.dead_until_reconnect = True
        wait_for(lambda: loop.fresh_sample(0.1) is None)
        # The device thread reconnects on its own; samples flow again.
        wait_for(lambda: fake_hardware.connect_count >= 2)
        wait_for(lambda: loop.fresh_sample(0.1) is not None)
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


class ScriptedTeleop:
    """The lerobot teleop surface the leader calls, answering one fixed
    action."""

    def __init__(self, action):
        self._action = action
        self.is_connected = False
        self.is_calibrated = True

    def connect(self, calibrate):
        self.is_connected = True

    def get_action(self):
        return dict(self._action)

    def disconnect(self):
        self.is_connected = False


def test_two_channels_naming_one_motor_fail_the_launch(tmp_path):
    from so101_leader.device import LerobotLeader

    leader = LerobotLeader("/dev/null", str(tmp_path), "leader")
    teleop = ScriptedTeleop(
        {**{f"{motor}.pos": 0.0 for motor in MOTOR_NAMES}, "wrist_flex": 1.0}
    )
    leader._teleop = teleop
    loop = DeviceLoop(leader, read_rate_hz=200)
    with pytest.raises(ValueError, match="two channels name motor wrist_flex"):
        loop.start()
    assert not loop.ready()
    # The refused bringup releases the bus.
    assert not teleop.is_connected


def test_lerobot_leader_constructs(tmp_path):
    """The real hardware seam must at least construct: this is the layer no
    fake covers, and a str/Path mismatch here once broke every launch."""
    from so101_leader.device import LerobotLeader

    leader = LerobotLeader(
        device_path="/dev/null", calibration_dir=str(tmp_path), teleop_id="smoke"
    )
    assert "gripper" in leader._teleop.bus.motors
