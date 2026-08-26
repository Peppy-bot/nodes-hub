from types import SimpleNamespace

import pytest

from so101_leader import params as params_mod

GOOD = {
    "device_path": "/dev/so101_leader",
    "calibration_dir": "/var/lib/so101/calibration",
    "teleop_id": "leader",
    "command_rate_hz": 60,
    "stale_timeout_s": 0.25,
}


def parse(**overrides):
    return params_mod.parse(SimpleNamespace(**{**GOOD, **overrides}))


def test_good_config_parses():
    assert parse().teleop_id == "leader"


def test_a_whole_number_timeout_reaches_the_config_as_a_float():
    # A whole-number f64 launch parameter arrives as a Python int, and an
    # int-typed float leaks into isinstance-checking APIs downstream; the
    # parse must carry the coercion through, not merely check the value.
    assert isinstance(parse(stale_timeout_s=1).stale_timeout_s, float)


@pytest.mark.parametrize(
    "field,value",
    [
        ("device_path", ""),
        ("calibration_dir", ""),
        ("teleop_id", ""),
        ("command_rate_hz", 0),
        ("command_rate_hz", 2000),  # above the loop ceiling
        ("stale_timeout_s", 0.0),
        ("stale_timeout_s", 0.03),  # below two read periods at 60 Hz
    ],
)
def test_bad_config_is_refused(field, value):
    # The refusal must name the field it refused, or a check firing for an
    # unrelated reason would pass as if the intended one still existed.
    with pytest.raises(ValueError, match=field):
        parse(**{field: value})
