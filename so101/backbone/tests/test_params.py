from types import SimpleNamespace

import pytest
from so101.device import BUS_MAX_RATE_HZ

from so101_backbone import params as params_mod
from so101_backbone.params import UpstreamMode

GOOD = {
    "upstream_mode": "joints",
    "control_rate_hz": 60,
    "max_joint_velocity_rad_s_1": 2.0,
    "max_joint_velocity_rad_s_2": 2.0,
    "max_joint_velocity_rad_s_3": 2.0,
    "max_joint_velocity_rad_s_4": 2.0,
    "max_joint_velocity_rad_s_5": 2.0,
    "max_joint_acceleration_rad_s2_1": 38.96,
    "max_joint_acceleration_rad_s2_2": 38.96,
    "max_joint_acceleration_rad_s2_3": 38.96,
    "max_joint_acceleration_rad_s2_4": 38.96,
    "max_joint_acceleration_rad_s2_5": 38.96,
    "max_stream_ee_velocity_m_s": 1.5,
    "max_gripper_rate_frac_s": 10.0,
    "start_tolerance_rad": 0.05,
}


def parse(**overrides):
    return params_mod.parse(SimpleNamespace(**{**GOOD, **overrides}))


def test_good_config_parses():
    config = parse()
    assert config.upstream_mode is UpstreamMode.JOINTS
    assert config.max_joint_velocity_rad_s == (2.0,) * 5


def test_the_joint_path_parameters_parse():
    config = parse()
    assert config.max_joint_acceleration_rad_s2 == (38.96,) * 5
    assert config.start_tolerance_rad == 0.05


def test_pose_mode_parses():
    assert parse(upstream_mode="pose").upstream_mode is UpstreamMode.POSE


@pytest.mark.parametrize(
    "field,value",
    [
        ("upstream_mode", "cartesian"),
        ("control_rate_hz", 0),
        ("max_joint_velocity_rad_s_3", 0.0),
        ("max_gripper_rate_frac_s", -1.0),
        ("max_stream_ee_velocity_m_s", 0.0),
        # The ceiling, not just the floor: one past BUS_MAX_RATE_HZ, so raising
        # the bound without revisiting this row cannot pass unnoticed.
        ("control_rate_hz", BUS_MAX_RATE_HZ + 1),
    ],
)
def test_bad_config_is_refused(field, value):
    with pytest.raises(ValueError):
        parse(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_joint_acceleration_rad_s2_2", 0.0),
        ("max_joint_acceleration_rad_s2_5", float("inf")),
        ("start_tolerance_rad", 0.0),
        ("start_tolerance_rad", float("nan")),
    ],
)
def test_a_bad_joint_path_parameter_is_refused(field, value):
    with pytest.raises(ValueError):
        parse(**{field: value})
