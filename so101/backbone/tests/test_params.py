from types import SimpleNamespace

import pytest

from so101_backbone import params as params_mod
from so101_backbone.params import MAX_RATE_HZ, UpstreamMode

GOOD = {
    "upstream_mode": "joints",
    "control_rate_hz": 60,
    "max_joint_velocity_rad_s_1": 2.0,
    "max_joint_velocity_rad_s_2": 2.0,
    "max_joint_velocity_rad_s_3": 2.0,
    "max_joint_velocity_rad_s_4": 2.0,
    "max_joint_velocity_rad_s_5": 2.0,
    "max_ee_velocity_m_s": 1.5,
    "max_ee_angular_velocity_rad_s": 8.0,
    "max_gripper_rate_frac_s": 10.0,
}


def parse(**overrides):
    return params_mod.parse(SimpleNamespace(**{**GOOD, **overrides}))


def test_good_config_parses():
    config = parse()
    assert config.upstream_mode is UpstreamMode.JOINTS
    assert config.max_joint_velocity_rad_s == (2.0,) * 5


def test_pose_mode_parses():
    assert parse(upstream_mode="pose").upstream_mode is UpstreamMode.POSE


@pytest.mark.parametrize(
    "field,value",
    [
        ("upstream_mode", "cartesian"),
        ("control_rate_hz", 0),
        ("max_joint_velocity_rad_s_3", 0.0),
        ("max_gripper_rate_frac_s", -1.0),
        ("max_ee_velocity_m_s", 0.0),
        ("max_ee_angular_velocity_rad_s", -2.0),
        # The ceiling, not just the floor: one past MAX_RATE_HZ, so raising
        # the bound without revisiting this row cannot pass unnoticed.
        ("control_rate_hz", MAX_RATE_HZ + 1),
    ],
)
def test_bad_config_is_refused(field, value):
    with pytest.raises(ValueError):
        parse(**{field: value})
