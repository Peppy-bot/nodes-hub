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
    assert parse_posture("  Squat_To_Stand ") is Posture.SQUAT_TO_STAND


def test_parse_posture_rejects_unknown():
    with pytest.raises(UnknownPosture):
        parse_posture("moonwalk")


def test_standup_maps_to_the_real_sdk_methods():
    # The SDK has no StandUp(); standing up is Squat2StandUp / Lie2StandUp.
    assert POSTURE_METHOD[Posture.SQUAT_TO_STAND] == "Squat2StandUp"
    assert POSTURE_METHOD[Posture.LIE_TO_STAND] == "Lie2StandUp"
    assert "StandUp" not in POSTURE_METHOD.values()


def test_method_and_fsm_maps_cover_every_posture():
    for posture in Posture:
        assert posture in POSTURE_METHOD
        assert posture in POSTURE_FSM_ID


def test_ordering_damp_always_allowed_including_boot():
    assert is_transition_allowed(None, Posture.DAMP)
    assert is_transition_allowed(Posture.START, Posture.DAMP)


def test_ordering_enforces_damp_before_stand_before_locomotion():
    # From a fresh boot only damp is reachable.
    assert not is_transition_allowed(None, Posture.SQUAT_TO_STAND)
    assert not is_transition_allowed(None, Posture.START)
    # damp -> stand -> start.
    assert is_transition_allowed(Posture.DAMP, Posture.SQUAT_TO_STAND)
    assert is_transition_allowed(Posture.SQUAT_TO_STAND, Posture.START)
    # Locomotion straight from damp (skipping the stand) is rejected.
    assert not is_transition_allowed(Posture.DAMP, Posture.START)


def test_sit_and_stance_require_standing():
    for target in (Posture.SIT, Posture.STAND_TO_SQUAT, Posture.HIGH_STAND, Posture.LOW_STAND):
        assert not is_transition_allowed(Posture.DAMP, target)
        assert is_transition_allowed(Posture.LIE_TO_STAND, target)


def test_plan_transition_returns_target_on_valid_sequence():
    assert plan_transition(Posture.DAMP, "squat_to_stand") is Posture.SQUAT_TO_STAND


def test_plan_transition_raises_on_unknown():
    with pytest.raises(UnknownPosture):
        plan_transition(Posture.DAMP, "nope")


def test_plan_transition_raises_on_out_of_order():
    with pytest.raises(TransitionRejected):
        plan_transition(None, "start")
