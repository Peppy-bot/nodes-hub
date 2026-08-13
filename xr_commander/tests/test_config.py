import math
from types import SimpleNamespace

import pytest

from xr_commander import config

BASE = {
    "command_rate_hz": 100,
    "https_port": 4443,
    "https_host": "0.0.0.0",
    "motion_scale": 1.0,
    "gripper_open_fraction": 1.0,
    "posture_move_duration_s": 2.0,
    "stale_timeout_s": 0.25,
    "view_max_width": 640,
    "status_panel_enabled": True,
}


def params(**overrides):
    return SimpleNamespace(**{**BASE, **overrides})


def test_the_defaults_parse_and_derive_the_tick_period():
    settings = config.from_parameters(params())
    assert settings.tick_period_s == pytest.approx(0.01)
    assert settings.motion_scale == pytest.approx(1.0)


@pytest.mark.parametrize("rate", [0, -1])
def test_a_non_positive_command_rate_is_refused(rate):
    with pytest.raises(ValueError, match="command_rate_hz"):
        config.from_parameters(params(command_rate_hz=rate))


@pytest.mark.parametrize(
    "field",
    [
        "motion_scale",
        "gripper_open_fraction",
        "stale_timeout_s",
    ],
)
@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_every_bound_must_be_finite_and_positive(field, bad):
    with pytest.raises(ValueError, match=field):
        config.from_parameters(params(**{field: bad}))


def test_an_open_fraction_beyond_the_wire_range_is_refused():
    with pytest.raises(ValueError, match="gripper_open_fraction"):
        config.from_parameters(params(gripper_open_fraction=1.1))


def test_a_reduced_open_fraction_is_accepted():
    settings = config.from_parameters(params(gripper_open_fraction=0.67))
    assert settings.gripper_open_fraction == pytest.approx(0.67)


def test_posture_duration_allows_zero_for_fastest_and_refuses_the_rest():
    assert config.from_parameters(
        params(posture_move_duration_s=0.0)
    ).posture_move_duration_s == pytest.approx(0.0)
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="posture_move_duration_s"):
            config.from_parameters(params(posture_move_duration_s=bad))


@pytest.mark.parametrize("bad", [1, 0, "true", None])
def test_a_non_boolean_panel_flag_is_refused(bad):
    with pytest.raises(ValueError, match="status_panel_enabled"):
        config.from_parameters(params(status_panel_enabled=bad))


def test_the_panel_can_be_turned_off():
    assert not config.from_parameters(
        params(status_panel_enabled=False)
    ).status_panel_enabled


@pytest.mark.parametrize("port", [0, 65536, -1])
def test_an_impossible_https_port_is_refused(port):
    with pytest.raises(ValueError, match="https_port"):
        config.from_parameters(params(https_port=port))


@pytest.mark.parametrize(
    "bad", [2.5, True, "abc", "100", b"4443", math.nan, math.inf, None]
)
def test_a_non_integer_rate_is_refused_not_coerced(bad):
    with pytest.raises(ValueError, match="command_rate_hz"):
        config.from_parameters(params(command_rate_hz=bad))


def test_a_whole_valued_float_rate_is_accepted():
    assert config.from_parameters(params(command_rate_hz=100.0)).command_rate_hz == 100


def test_a_rate_that_would_starve_the_loop_is_refused():
    assert config.from_parameters(params(command_rate_hz=1000)).command_rate_hz == 1000
    with pytest.raises(ValueError, match="command_rate_hz"):
        config.from_parameters(params(command_rate_hz=1001))


def test_a_non_ip_bind_address_is_refused():
    with pytest.raises(ValueError, match="https_host"):
        config.from_parameters(params(https_host="robot.local"))


def test_view_max_width_bounds_and_zero_disable():
    assert config.from_parameters(params(view_max_width=0)).view_max_width == 0
    assert config.from_parameters(params(view_max_width=8192)).view_max_width == 8192
    for bad in (-1, 8193, 1.5):
        with pytest.raises(ValueError, match="view_max_width"):
            config.from_parameters(params(view_max_width=bad))


def test_the_task_label_is_not_a_launch_parameter():
    # It comes from the operator's page only, so a launcher naming one is a
    # stale label waiting to record under.
    assert not hasattr(config.from_parameters(params()), "task")
