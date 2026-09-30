"""The rules in State, each one a test: gripper resolution, the holding
flag, item ids across looks and runs, and the best match for a description."""

import pytest

from openarm_ai_brain_vla.ports import Coverage, Detection, Refusal
from openarm_ai_brain_vla.state import State, arm_of, best_match, new_run_token
from openarm_ai_brain_vla.words import content_words, named_by, normalised

# A scan that can name anything, and a search by description, which covers nothing.
EVERY = Coverage(every_label=True)
NOTHING = Coverage()


def make_state(run_token: str = "t0") -> State:
    return State(["left_gripper", "right_gripper"], run_token)


def test_arm_names_follow_the_backbone_side_naming():
    assert arm_of("left_gripper") == "left_arm"
    assert arm_of("right_gripper") == "right_arm"
    assert make_state().grippers["left_gripper"].arm == "left_arm"


def test_gripper_names_must_be_present_and_distinct():
    with pytest.raises(ValueError):
        State([""], "t0")
    with pytest.raises(ValueError):
        State(["a", "a"], "t0")


def test_the_run_token_must_be_letters_and_digits():
    for token in ("", "a-b", "a b", "a_b"):
        with pytest.raises(ValueError, match="run_token must be letters and digits"):
            State(["left_gripper"], token)
    token = new_run_token()
    assert len(token) == 6 and int(token, 16) >= 0
    assert State(["left_gripper"], token).run_token == token


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
        coverage=EVERY,
    )
    assert [item.item_id for item in first] == ["cup_1-t0", "cup_2-t0", "banana_1-t0"]
    # The first cup moved 2 cm, the second one is gone, a bowl appeared.
    second = state.remember(
        [Detection("cup", (0.52, 0.10, 0.70), 0.95), Detection("bowl", (0.60, 0.0, 0.70), 0.6)],
        now_ns=2,
        coverage=EVERY,
    )
    assert [item.item_id for item in second] == ["cup_1-t0", "bowl_1-t0"]
    assert set(state.items) == {"cup_1-t0", "bowl_1-t0"}
    assert state.items["cup_1-t0"].position == (0.52, 0.10, 0.70)
    # Beyond the match radius the same label is a new item, and a dropped
    # number is never minted again.
    third = state.remember([Detection("cup", (0.80, 0.10, 0.70), 0.9)], now_ns=3, coverage=EVERY)
    assert third[0].item_id == "cup_3-t0"
    assert set(state.items) == {"cup_3-t0"}


def test_a_scan_never_drops_an_item_a_gripper_holds():
    state = make_state()
    state.remember([Detection("cup", (0.5, 0.1, 0.7), 0.9)], now_ns=1, coverage=EVERY)
    state.set_held(state.grippers["left_gripper"], "cup_1-t0")
    state.remember([], now_ns=2, coverage=EVERY)
    assert "cup_1-t0" in state.items


def test_an_identify_search_refreshes_without_dropping_the_rest():
    state = make_state()
    state.remember([Detection("cup", (0.5, 0.1, 0.7), 0.9), Detection("banana", (0.4, 0.0, 0.7), 0.7)], now_ns=1, coverage=EVERY)
    found = state.remember([Detection("cup", (0.51, 0.1, 0.7), 0.8)], now_ns=2, coverage=NOTHING)
    assert found[0].item_id == "cup_1-t0"
    assert found[0].position == (0.51, 0.1, 0.7) and found[0].confidence == 0.8 and found[0].seen_at_ns == 2
    assert set(state.items) == {"cup_1-t0", "banana_1-t0"}


def test_a_search_under_other_words_keeps_the_scans_id_and_label():
    state = make_state()
    vocabulary_scan = Coverage(frozenset({"mustard bottle", "apple"}))
    scanned = state.remember([Detection("mustard bottle", (0.5, 0.0, 0.7), 0.9)], now_ns=1, coverage=vocabulary_scan)[0]
    # The words route labels the same bottle with the caller's description.
    found = state.remember([Detection("yellow bottle", (0.51, 0.0, 0.7), 0.96)], now_ns=2, coverage=NOTHING)[0]
    assert found.item_id == scanned.item_id == "mustard_bottle_1-t0"
    assert found.label == "mustard bottle"
    rescanned = state.remember([Detection("mustard bottle", (0.5, 0.0, 0.7), 0.9)], now_ns=3, coverage=vocabulary_scan)[0]
    assert rescanned.item_id == "mustard_bottle_1-t0"
    assert set(state.items) == {"mustard_bottle_1-t0"}


def test_a_scan_under_other_words_keeps_the_searchs_id_and_takes_its_label():
    state = make_state()
    found = state.remember([Detection("red apple", (0.5, 0.1, 0.7), 1.0)], now_ns=1, coverage=NOTHING)[0]
    assert found.item_id == "red_apple_1-t0"
    # An open-vocabulary scan names the same apple in its own words.
    scanned = state.remember([Detection("apple", (0.51, 0.1, 0.7), 1.0)], now_ns=2, coverage=EVERY)[0]
    assert scanned.item_id == "red_apple_1-t0" and scanned.label == "apple"
    # Under a label the scan names, the apple is dropped once it is gone.
    state.remember([], now_ns=3, coverage=Coverage(frozenset({"apple"})))
    assert state.items == {}


def test_items_sharing_a_place_keep_their_own_ids():
    state = make_state()
    first = state.remember(
        [Detection("bowl", (0.50, 0.00, 0.70), 0.9), Detection("apple", (0.51, 0.01, 0.72), 0.8)], now_ns=1, coverage=EVERY
    )
    assert [item.item_id for item in first] == ["bowl_1-t0", "apple_1-t0"]
    # The apple now comes first and sits nearer the bowl's last position
    # than the bowl does: the label decides between the two.
    second = state.remember(
        [Detection("apple", (0.50, 0.00, 0.71), 0.95), Detection("bowl", (0.52, 0.00, 0.70), 0.9)], now_ns=2, coverage=EVERY
    )
    assert [item.item_id for item in second] == ["apple_1-t0", "bowl_1-t0"]
    assert [item.label for item in second] == ["apple", "bowl"]


def test_a_scan_drops_only_the_unseen_items_it_could_name():
    state = make_state()
    vocabulary_scan = Coverage(frozenset({"cup", "bowl"}))
    state.remember([Detection("cup", (0.5, 0.1, 0.7), 0.9)], now_ns=1, coverage=vocabulary_scan)
    # Only the words route finds a blue ball: the scan's vocabulary has no such name.
    state.remember([Detection("blue ball", (0.4, -0.1, 0.7), 0.9)], now_ns=2, coverage=NOTHING)
    assert set(state.items) == {"cup_1-t0", "blue_ball_1-t0"}
    state.remember([], now_ns=3, coverage=vocabulary_scan)
    assert set(state.items) == {"blue_ball_1-t0"}
    state.remember([], now_ns=4, coverage=EVERY)
    assert state.items == {}


def test_no_detection_matches_an_item_a_gripper_holds():
    state = make_state()
    state.remember([Detection("cup", (0.5, 0.1, 0.7), 0.9)], now_ns=1, coverage=EVERY)
    state.set_held(state.grippers["left_gripper"], "cup_1-t0")
    # Another cup stands where the held one was grabbed.
    seen = state.remember([Detection("cup", (0.5, 0.1, 0.7), 0.9)], now_ns=2, coverage=EVERY)
    assert seen[0].item_id == "cup_2-t0"
    assert set(state.items) == {"cup_1-t0", "cup_2-t0"}


def test_a_placed_item_keeps_its_id_where_it_was_put_and_a_dropped_one_stays_where_it_was_grabbed():
    state = make_state()
    state.remember([Detection("cup", (0.5, 0.1, 0.7), 0.9), Detection("bowl", (0.4, -0.1, 0.7), 0.9)], now_ns=1, coverage=EVERY)
    left, right = state.grippers["left_gripper"], state.grippers["right_gripper"]
    state.set_held(left, "cup_1-t0")
    assert state.clear_held_placed(left, (0.6, -0.2, 0.75), (0.0, 0.0, 0.0, 1.0)) == "cup_1-t0"
    assert not left.holding
    assert state.items["cup_1-t0"].position == (0.6, -0.2, 0.75)
    assert state.items["cup_1-t0"].orientation == (0.0, 0.0, 0.0, 1.0)
    state.set_held(right, "bowl_1-t0")
    assert state.clear_held(right) == "bowl_1-t0"
    assert state.items["bowl_1-t0"].position == (0.4, -0.1, 0.7)
    seen = state.remember([Detection("cup", (0.61, -0.2, 0.74), 0.9)], now_ns=2, coverage=EVERY)
    assert seen[0].item_id == "cup_1-t0"


def test_ids_carry_the_run_token_so_an_earlier_runs_id_is_refused():
    before, after = make_state("aaaaaa"), make_state("bbbbbb")
    apple = Detection("apple", (0.5, 0.0, 0.7), 0.9)
    old = before.remember([apple], now_ns=1, coverage=EVERY)[0]
    new = after.remember([apple], now_ns=1, coverage=EVERY)[0]
    assert (old.item_id, new.item_id) == ("apple_1-aaaaaa", "apple_1-bbbbbb")
    with pytest.raises(Refusal, match="unknown item 'apple_1-aaaaaa'"):
        after.item(old.item_id)


def test_unknown_items_are_refused_and_pose_grabs_get_an_id():
    state = make_state()
    with pytest.raises(Refusal, match="unknown item 'cup_9-t0'"):
        state.item("cup_9-t0")
    with pytest.raises(Refusal, match="unknown item 'cup_9-t0'"):
        state.item_to_grab("cup_9-t0")
    item = state.mint_from_pose((0.5, 0.0, 0.7), None, now_ns=1)
    assert item.item_id == "item_1-t0"
    assert state.item("item_1-t0") is item
    assert state.item_to_grab("item_1-t0") is item


def test_an_item_a_gripper_holds_cannot_be_grabbed_again():
    state = make_state()
    state.remember([Detection("cup", (0.5, 0.1, 0.7), 0.9), Detection("bowl", (0.4, -0.1, 0.7), 0.9)], now_ns=1, coverage=EVERY)
    assert state.holder_of("cup_1-t0") is None
    state.set_held(state.grippers["left_gripper"], "cup_1-t0")
    assert state.holder_of("cup_1-t0") is state.grippers["left_gripper"]
    with pytest.raises(Refusal, match="item 'cup_1-t0' is held by gripper 'left_gripper'"):
        state.item_to_grab("cup_1-t0")
    assert state.item_to_grab("bowl_1-t0").item_id == "bowl_1-t0"
    state.clear_held(state.grippers["left_gripper"])
    assert state.item_to_grab("cup_1-t0").item_id == "cup_1-t0"


def test_labels_become_clean_id_stems():
    state = make_state()
    item = state.remember([Detection("Cheez-It cracker box", (0.5, 0.0, 0.7), 0.9)], now_ns=1, coverage=EVERY)[0]
    assert item.item_id == "cheez_it_cracker_box_1-t0"
    assert item.label == "Cheez-It cracker box"


def test_best_match_prefers_the_exact_label_then_a_word_then_confidence():
    cup = Detection("cup", (0, 0, 0), 0.6)
    red_cup = Detection("red cup", (0, 0, 0), 0.5)
    bowl = Detection("bowl", (0, 0, 0), 0.9)
    assert best_match([cup, red_cup, bowl], "red cup") is red_cup
    assert best_match([cup, bowl], "cup") is cup
    assert best_match([red_cup, bowl], "cup") is red_cup
    assert best_match([cup, red_cup, bowl], "the cup please") is cup
    assert best_match([cup, bowl], "") is bowl
    assert best_match([cup, bowl], "spoon") is None
    assert best_match([], "cup") is None


def test_best_match_never_matches_part_of_a_word_or_a_stopword():
    thermos = Detection("thermos", (0, 0, 0), 0.9)
    candle = Detection("candle", (0, 0, 0), 0.8)
    feather = Detection("feather", (0, 0, 0), 0.7)
    seen = [thermos, candle, feather]
    assert best_match(seen, "the coffee can please") is None
    assert best_match(seen, "can") is None
    assert best_match(seen, "the mug") is None
    assert best_match(seen, "the") is None
    assert best_match(seen, "the candle") is candle


def test_a_description_names_a_label_by_whole_words_stopwords_aside():
    labels = ["coffee can", "cracker box", "candle", "red coffee can"]
    assert named_by("coffee can", labels) == [0]
    assert named_by("Coffee_Can", labels) == [0]
    assert named_by("the coffee can please", labels) == [0, 3]
    assert named_by("can", labels) == [0, 3]
    assert named_by("red can", labels) == [3]
    assert named_by("box can", labels) == []
    assert named_by("coffee tin", labels) == []
    assert named_by("cand", labels) == []
    assert named_by("the", labels) == [] and named_by("  ", labels) == []
    assert normalised("  Red_Coffee   Can ") == "red coffee can"
    assert content_words("the Red coffee can, please") == {"red", "coffee", "can,"}
