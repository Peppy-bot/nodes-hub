import pytest

from g1_node.postures import (
    POSTURE_FSM_ID,
    POSTURE_METHOD,
    Posture,
    TransitionRejected,
    UnknownPosture,
    is_transition_allowed,
    parse_posture,
    plan_transition,
)


def test_parse_posture_accepts_every_known_value():
    for posture in Posture:
        assert parse_posture(posture.value) is posture


def test_parse_posture_is_case_and_space_insensitive():
    assert parse_posture("  Stand_Up ") is Posture.STAND_UP


def test_parse_posture_rejects_unknown():
    with pytest.raises(UnknownPosture):
        parse_posture("moonwalk")


def test_method_and_fsm_maps_cover_every_posture():
    # A missing entry would KeyError only at runtime against the robot.
    for posture in Posture:
        assert posture in POSTURE_METHOD
        assert posture in POSTURE_FSM_ID


def test_ordering_damp_always_allowed_including_boot():
    assert is_transition_allowed(None, Posture.DAMP)
    assert is_transition_allowed(Posture.START, Posture.DAMP)


def test_ordering_enforces_damp_before_standup_before_start():
    # From a fresh boot only damp is reachable.
    assert not is_transition_allowed(None, Posture.STAND_UP)
    assert not is_transition_allowed(None, Posture.START)
    # The documented sequence.
    assert is_transition_allowed(Posture.DAMP, Posture.STAND_UP)
    assert is_transition_allowed(Posture.STAND_UP, Posture.START)
    # Skipping stand_up is rejected.
    assert not is_transition_allowed(Posture.DAMP, Posture.START)


def test_plan_transition_returns_target_on_valid_sequence():
    assert plan_transition(Posture.DAMP, "stand_up") is Posture.STAND_UP


def test_plan_transition_raises_on_unknown():
    with pytest.raises(UnknownPosture):
        plan_transition(Posture.DAMP, "nope")


def test_plan_transition_raises_on_out_of_order():
    with pytest.raises(TransitionRejected):
        plan_transition(None, "start")
