"""The prompts a search asks, and an answer read against the schema, then
placed in the picture's pixels."""

import json

import pytest

from vlm_router_brain.parsing import UNREADABLE, Answer, UnreadableAnswer, labelled, read_answer, yx_0_1000_to_pixels
from vlm_router_brain.prompts import IDENTIFY, SCAN, plan_for


def test_nothing_visible_is_a_valid_answer_with_no_boxes():
    assert read_answer('{"boxes": []}').boxes == []


@pytest.mark.parametrize("text", [
    "",
    "I cannot see a mustard bottle.",
    '```json\n{"boxes": []}\n```',
    '[{"label": "cup", "box_2d": [1, 2, 3, 4]}]',
    '{"boxes": [{"label": "cup", "y": 1, "x": 2, "y2": 3, "x2": 4}]}',
    '{"boxes": [{"box_2d": [1, 2, 3, 4]}]}',
    '{"boxes": [{"label": "cup", "box_2d": [1, 2, 3]}]}',
    '{"boxes": [{"label": "cup", "box_2d": [1, 2, 3, 4, 5]}]}',
    '{"boxes": [{"label": "cup", "box_2d": ["a", 2, 3, 4]}]}',
])
def test_an_answer_not_of_the_schemas_shape_is_unreadable(text):
    with pytest.raises(UnreadableAnswer) as raised:
        read_answer(text)
    assert raised.value.message == UNREADABLE == "the model's answer could not be read"
    assert raised.value.text == text


def test_the_schema_is_the_one_step_0_asked_every_route_for():
    schema = Answer.model_json_schema()
    assert schema["required"] == ["boxes"]
    box = schema["$defs"]["Box"]
    assert set(box["properties"]) == {"label", "box_2d"} and box["properties"]["box_2d"]["items"] == {"type": "integer"}


def test_yx_0_1000_becomes_pixels_clipped_to_the_picture():
    # The mustard bottle of step 0 on the 1280 x 720 chest frame.
    assert yx_0_1000_to_pixels([462, 312, 630, 382], 1280, 720) == pytest.approx((399.36, 332.64, 488.96, 453.6))
    assert yx_0_1000_to_pixels([-50, 900, 500, 1100], 1000, 500) == (900.0, 0.0, 1000.0, 250.0)
    # Less than a pixel left is no box.
    assert yx_0_1000_to_pixels([100, 100, 100, 300], 1280, 720) is None


def boxes(*pairs) -> Answer:
    return Answer.model_validate_json(json.dumps({"boxes": [{"label": label, "box_2d": [1, 2, 3, 4]} for label in pairs]}))


def test_an_identify_search_names_every_box_its_item():
    kept = labelled(boxes("yellow bottle", ""), "the mustard bottle")
    assert [label for label, _ in kept] == ["the mustard bottle", "the mustard bottle"]


def test_a_scan_keeps_the_models_own_label_in_lower_case_and_drops_an_empty_one():
    kept = labelled(boxes(" Red Mug ", "  "), None)
    assert [label for label, _ in kept] == ["red mug"]


def test_the_phrases_choose_the_prompt():
    identify = plan_for(["  the mustard bottle "])
    assert (identify.mode, identify.force_label) == (IDENTIFY, "the mustard bottle")
    assert identify.prompt.startswith("Find the mustard bottle in this image")
    assert '{"boxes": [{"label": "the mustard bottle", "box_2d": [<y_min>, <x_min>, <y_max>, <x_max>]}]}' in identify.prompt
    # A blank phrase is no phrase: the search is a scan.
    for scan in (plan_for([]), plan_for([" "])):
        assert (scan.mode, scan.force_label) == (SCAN, None)
        assert scan.prompt.startswith("Detect every distinct object")
    for plan in (identify, scan):
        assert 'Return {"boxes": []}' in plan.prompt and "normalized to 0-1000" in plan.prompt


def test_a_search_for_more_than_one_item_is_an_error_naming_the_limit():
    with pytest.raises(ValueError) as raised:
        plan_for(["cup", " ", "bowl"])
    assert str(raised.value) == "vlm_router_brain searches for one item at a time; this search names 2: 'cup', 'bowl'"
