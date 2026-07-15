import json

import pytest

from g1_arm_node.backend import ArmActionClientBackend
from g1_arm_node.gestures import GESTURE_ACTION_ID, UnknownGesture, action_id_for


class FakeArm:
    """Stands in for the SDK G1ArmActionClient."""

    def __init__(self):
        self.executed: list[int] = []

    def ExecuteAction(self, action_id):
        self.executed.append(action_id)
        return (0, None)

    def GetActionList(self):
        return (0, [{"name": "clap", "id": 17}])


def _arm_backend(fake):
    backend = object.__new__(ArmActionClientBackend)
    backend._arm = fake
    return backend


def test_gesture_names_map_to_ids():
    assert action_id_for("shake_hand") == 27
    assert action_id_for("  High_Five ") == 18
    assert action_id_for("release_arm") == 99


def test_unknown_gesture_raises():
    with pytest.raises(UnknownGesture):
        action_id_for("floss")


def test_every_gesture_id_is_unique():
    ids = list(GESTURE_ACTION_ID.values())
    assert len(ids) == len(set(ids))


def test_execute_action_dispatches_and_normalizes_code():
    fake = FakeArm()
    backend = _arm_backend(fake)
    code = backend.execute_action(19)
    assert fake.executed == [19]
    assert code == 0


def test_get_action_list_returns_json():
    backend = _arm_backend(FakeArm())
    parsed = json.loads(backend.get_action_list())
    assert parsed == [{"name": "clap", "id": 17}]
