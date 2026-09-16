"""The rules in State, each one a test: gripper resolution, the holding
flag, item ids across scans, and the best match for a description."""

import pytest

from openarm_ai_brain_vla.ports import Detection, Refusal
from openarm_ai_brain_vla.state import State, arm_of, best_match


def make_state() -> State:
    return State(["left_gripper", "right_gripper"])


def test_arm_names_follow_the_backbone_side_naming():
    assert arm_of("left_gripper") == "left_arm"
    assert arm_of("right_gripper") == "right_arm"
    assert make_state().grippers["left_gripper"].arm == "left_arm"


def test_gripper_names_must_be_present_and_distinct():
    with pytest.raises(ValueError):
        State([""])
    with pytest.raises(ValueError):
        State(["a", "a"])


def test_named_gripper_for_a_grab_must_exist_and_be_free():
    state = make_state()
    with pytest.raises(Refusal, match="unknown gripper 'claw'"):
        state.gripper_for_grab("claw", None)
    left = state.gripper_for_grab("left_gripper", None)
    state.set_held(left, "cup_1")
    with pytest.raises(Refusal, match="already holds item 'cup_1'"):
        state.gripper_for_grab("left_gripper", None)


def test_an_empty_name_picks_a_free_gripper_on_the_items_side():
    state = make_state()
    assert state.gripper_for_grab("", (0.5, 0.2, 0.7)).name == "left_gripper"
    assert state.gripper_for_grab("", (0.5, -0.2, 0.7)).name == "right_gripper"
    state.set_held(state.grippers["right_gripper"], "cup_1")
    assert state.gripper_for_grab("", (0.5, -0.2, 0.7)).name == "left_gripper"
    state.set_held(state.grippers["left_gripper"], "cup_2")
    with pytest.raises(Refusal, match="no gripper is free"):
        state.gripper_for_grab("", (0.5, 0.0, 0.7))


def test_the_holder_rules_for_drop_and_place():
    state = make_state()
    with pytest.raises(Refusal, match="no gripper holds an item"):
        state.holder("")
    with pytest.raises(Refusal, match="holds nothing"):
        state.holder("left_gripper")
    state.set_held(state.grippers["left_gripper"], "cup_1")
    assert state.holder("").name == "left_gripper"
    state.set_held(state.grippers["right_gripper"], "cup_2")
    with pytest.raises(Refusal, match="more than one gripper"):
        state.holder("")
    assert state.holder("right_gripper").name == "right_gripper"
    assert state.clear_held(state.grippers["right_gripper"]) == "cup_2"
    assert state.snapshot() == (["left_gripper", "right_gripper"], [True, False], ["cup_1", ""])


def test_a_scan_keeps_ids_for_items_that_stayed_and_drops_the_ones_that_left():
    state = make_state()
    first = state.remember(
        [Detection("cup", (0.50, 0.10, 0.70), 0.9), Detection("cup", (0.50, -0.20, 0.70), 0.8), Detection("banana", (0.40, 0.0, 0.70), 0.7)],
        now_ns=1,
        complete=True,
    )
    assert [item.item_id for item in first] == ["cup_1", "cup_2", "banana_1"]
    # The first cup moved 2 cm, the second one is gone, a bowl appeared.
    second = state.remember(
        [Detection("cup", (0.52, 0.10, 0.70), 0.95), Detection("bowl", (0.60, 0.0, 0.70), 0.6)],
        now_ns=2,
        complete=True,
    )
    assert [item.item_id for item in second] == ["cup_1", "bowl_1"]
    assert set(state.items) == {"cup_1", "bowl_1"}
    assert state.items["cup_1"].position == (0.52, 0.10, 0.70)
    # Beyond the match radius the same label is a new item.
    third = state.remember([Detection("cup", (0.80, 0.10, 0.70), 0.9)], now_ns=3, complete=True)
    assert third[0].item_id == "cup_3"


def test_a_scan_never_drops_an_item_a_gripper_holds():
    state = make_state()
    state.remember([Detection("cup", (0.5, 0.1, 0.7), 0.9)], now_ns=1, complete=True)
    state.set_held(state.grippers["left_gripper"], "cup_1")
    state.remember([], now_ns=2, complete=True)
    assert "cup_1" in state.items


def test_an_identify_search_refreshes_without_dropping_the_rest():
    state = make_state()
    state.remember([Detection("cup", (0.5, 0.1, 0.7), 0.9), Detection("banana", (0.4, 0.0, 0.7), 0.7)], now_ns=1, complete=True)
    found = state.remember([Detection("cup", (0.51, 0.1, 0.7), 0.9)], now_ns=2, complete=False)
    assert found[0].item_id == "cup_1"
    assert set(state.items) == {"cup_1", "banana_1"}


def test_unknown_items_are_refused_and_pose_grabs_get_an_id():
    state = make_state()
    with pytest.raises(Refusal, match="unknown item 'cup_9'"):
        state.item("cup_9")
    item = state.mint_from_pose((0.5, 0.0, 0.7), None, now_ns=1)
    assert item.item_id == "item_1"
    assert state.item("item_1") is item


def test_labels_become_clean_id_stems():
    state = make_state()
    item = state.remember([Detection("Cheez-It cracker box", (0.5, 0.0, 0.7), 0.9)], now_ns=1, complete=True)[0]
    assert item.item_id == "cheez_it_cracker_box_1"
    assert item.label == "Cheez-It cracker box"


def test_best_match_prefers_the_exact_label_then_a_word_then_confidence():
    cup = Detection("cup", (0, 0, 0), 0.6)
    red_cup = Detection("red cup", (0, 0, 0), 0.5)
    bowl = Detection("bowl", (0, 0, 0), 0.9)
    assert best_match([cup, red_cup, bowl], "red cup") is red_cup
    assert best_match([cup, bowl], "cup") is cup
    assert best_match([red_cup, bowl], "cup") is red_cup
    assert best_match([cup, bowl], "") is bowl
    assert best_match([cup, bowl], "spoon") is None
    assert best_match([], "cup") is None
