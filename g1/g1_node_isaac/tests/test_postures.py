import pytest

from g1_node_isaac.postures import UnknownPosture, sim_posture


def test_damp_and_zero_torque_go_limp():
    assert sim_posture("damp") == (1, False)
    assert sim_posture("zero_torque") == (0, False)


def test_standing_postures_hold_and_report_fsm():
    assert sim_posture("start") == (200, True)
    assert sim_posture("squat_to_stand") == (4, True)
    assert sim_posture("sit") == (3, True)


def test_case_insensitive():
    assert sim_posture("  Start ") == (200, True)


def test_unknown_posture_rejected():
    with pytest.raises(UnknownPosture):
        sim_posture("moonwalk")
