from types import SimpleNamespace

import pytest

from so101_follower import params as params_mod

GOOD = {
    "device_path": "/dev/so101_follower",
    "calibration_dir": "/var/lib/so101/calibration",
    "robot_id": "follower",
    "control_rate_hz": 60,
    "state_rate_hz": 60,
}


def parse(**overrides):
    return params_mod.parse(SimpleNamespace(**{**GOOD, **overrides}))


def test_good_config_parses():
    config = parse()
    assert config.robot_id == "follower"
    assert config.control_rate_hz == 60


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device_path", ""),
        ("calibration_dir", ""),
        ("robot_id", ""),
        ("control_rate_hz", 0),
        ("state_rate_hz", 0),
        ("control_rate_hz", 2000),  # above the loop ceiling
        # Below the health rate the read decimation would floor to zero ticks.
        ("control_rate_hz", 4),
        ("state_rate_hz", 2000),
    ],
)
def test_bad_values_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        parse(**{field: value})
