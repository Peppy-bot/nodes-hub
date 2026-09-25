"""The gemini_er backend around its API: the prompts, the repair of the
model's JSON, the merging of answers, the key lookup and the search plan,
all against a fake API. The real client is never constructed here."""

import json

import numpy as np
import pytest

from openarm_ai_brain_vla.perception import make_detector
from openarm_ai_brain_vla.perception.gemini_er import (
    IDENTIFY,
    KEY_ENV,
    LIST,
    MODEL,
    OPEN,
    Answer,
    GeminiErDetector,
    api_key,
    identify_prompt,
    list_prompt,
    merge_samples,
    parse_boxes,
    plan_for,
)


class FakeApi:
    """Answers every question with the texts it was given, in order, and
    keeps what it was asked."""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.prompts: list[str] = []
        self.images: list[bytes] = []

    def ask(self, image: bytes, prompt: str) -> Answer:
        self.images.append(image)
        self.prompts.append(prompt)
        return Answer(self.texts.pop(0) if self.texts else "[]", input_tokens=1000, output_tokens=20, thought_tokens=100, latency_s=1.5)


def frame(width=200, height=100):
    return np.full((height, width, 3), 90, dtype=np.uint8)


def test_the_prompts_are_the_studys():
    assert identify_prompt("the Spam").startswith("Find the Spam in this image and return its bounding box.\n")
    assert '"label": "the Spam"' in identify_prompt("the Spam")
    assert "Return [] if it is not visible in this image." in identify_prompt("the Spam")
    prompt = list_prompt(["mug", "banana"])
    assert 'Allowed labels, to be used exactly as written: "mug", "banana".' in prompt
    assert "Do not include objects that are not in the list." in prompt


def test_the_plan_follows_the_vocabulary():
    assert plan_for([]).mode == OPEN
    assert plan_for(["", "  "]).mode == OPEN
    one = plan_for([" mug "])
    assert (one.mode, one.force_label, one.allowed) == (IDENTIFY, "mug", None)
    many = plan_for(["mug", "banana"])
    assert (many.mode, many.allowed, many.force_label) == (LIST, ("mug", "banana"), None)


def test_boxes_are_scaled_from_the_thousandths_and_clipped():
    text = json.dumps([{"label": "mug", "y": 100, "x": 250, "y2": 500, "x2": 750}, {"label": "mug", "y": -50, "x": 900, "y2": 1200, "x2": 1300}])
    boxes, unmatched = parse_boxes(text, 200, 100, allowed=["mug"])
    assert unmatched == []
    assert [(label, box.round(1).tolist()) for label, box in boxes] == [("mug", [50.0, 10.0, 150.0, 50.0]), ("mug", [180.0, 0.0, 200.0, 100.0])]


def test_a_label_outside_the_list_is_dropped_and_reported():
    text = json.dumps([{"label": "Banana", "y": 0, "x": 0, "y2": 500, "x2": 500}, {"label": "spoon", "y": 0, "x": 0, "y2": 500, "x2": 500}])
    boxes, unmatched = parse_boxes(text, 100, 100, allowed=["banana", "mug"])
    assert [label for label, _ in boxes] == ["banana"]
    assert unmatched == ["spoon"]


def test_the_asked_item_owns_every_box_whatever_the_label():
    text = json.dumps([{"label": "cup", "y": 0, "x": 0, "y2": 500, "x2": 500}])
    boxes, _ = parse_boxes(text, 100, 100, force_label="the mug")
    assert [label for label, _ in boxes] == ["the mug"]


def test_the_models_irregular_json_is_repaired():
    fenced = '```json\n[{"label": "mug", "y": 0, "x": 0, "y2": 500, "x2": 500}]\n```'
    assert len(parse_boxes(fenced, 100, 100, force_label="mug")[0]) == 1
    trailing = '[{"label": "mug", "y": 0, "x": 0, "y2": 500, "x2": 500}] and that is all]'
    assert len(parse_boxes(trailing, 100, 100, force_label="mug")[0]) == 1
    bare_numbers = 'The box is "y": 10, "x": 20, "y2": 300, "x2": 400.'
    boxes, _ = parse_boxes(bare_numbers, 1000, 1000, force_label="mug")
    assert boxes[0][1].tolist() == [20.0, 10.0, 400.0, 300.0]
    box_2d = json.dumps([{"label": "mug", "box_2d": [10, 20, 300, 400]}])
    assert parse_boxes(box_2d, 1000, 1000, force_label="mug")[0][0][1].tolist() == [20.0, 10.0, 400.0, 300.0]
    single = json.dumps({"label": "mug", "y": 0, "x": 0, "y2": 500, "x2": 500})
    assert len(parse_boxes(single, 100, 100, force_label="mug")[0]) == 1
    keyed = json.dumps([{"the sponge": [10, 20, 300, 400]}])
    assert len(parse_boxes(keyed, 1000, 1000, force_label="the sponge")[0]) == 1
    assert parse_boxes("I cannot see it.", 100, 100, force_label="mug") == ([], ["<unparseable>"])
    assert parse_boxes("I cannot see it.", 100, 100, allowed=["mug"]) == ([], ["<unparseable>"])


def test_a_point_or_a_degenerate_box_is_not_a_detection():
    point = json.dumps([{"label": "mug", "point": [500, 500]}, {"label": "mug", "y": 500, "x": 500, "y2": 500, "x2": 501}])
    assert parse_boxes(point, 100, 100, force_label="mug")[0] == []


def test_the_open_prompt_keeps_the_models_own_words():
    text = json.dumps([{"label": "Red Mug", "y": 0, "x": 0, "y2": 500, "x2": 500}, {"label": "", "y": 0, "x": 0, "y2": 500, "x2": 500}])
    boxes, unmatched = parse_boxes(text, 100, 100)
    assert [label for label, _ in boxes] == ["red mug"]
    assert unmatched == []


def test_answers_agreeing_on_a_box_are_one_item_with_their_share_as_confidence():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([1.0, 1.0, 11.0, 11.0])
    far = np.array([50.0, 50.0, 60.0, 60.0])
    merged = merge_samples([[("mug", a)], [("mug", b), ("mug", far)], [("mug", a)]])
    by_conf = sorted(merged, key=lambda box: -box.confidence)
    assert [round(box.confidence, 2) for box in by_conf] == [1.0, 0.33]
    assert (round(by_conf[0].x0, 2), round(by_conf[0].x1, 2)) == (0.33, 10.33)
    assert by_conf[1].x0 == 50.0


def test_one_answer_makes_every_box_confidence_one():
    merged = merge_samples([[("mug", np.array([0.0, 0.0, 10.0, 10.0])), ("banana", np.array([20.0, 0.0, 30.0, 10.0]))]])
    assert sorted((box.label, box.confidence) for box in merged) == [("banana", 1.0), ("mug", 1.0)]


def test_the_key_comes_from_the_environment_then_the_file(tmp_path, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    file = tmp_path / "key"
    assert api_key(file=file) == ""
    file.write_text("from-file\n")
    assert api_key(file=file) == "from-file"
    monkeypatch.setenv(KEY_ENV, " from-env ")
    assert api_key(file=file) == "from-env"


def test_loading_without_a_key_refuses_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    detector = GeminiErDetector()
    with pytest.raises(RuntimeError, match=KEY_ENV):
        detector.load("")
    assert not detector.available
    assert detector.detect(frame()) == []


def test_the_registry_builds_the_backend_and_the_model_id_defaults():
    detector = make_detector("gemini_er")
    assert detector.name == "gemini_er"
    fake = GeminiErDetector(api=FakeApi())
    fake.load("  ")
    assert fake.model == MODEL and fake.available
    fake.load("gemini-robotics-er-3")
    assert fake.model == "gemini-robotics-er-3"


def test_a_scan_asks_for_everything_and_keeps_the_models_labels():
    api = FakeApi(json.dumps([{"label": "red mug", "y": 100, "x": 100, "y2": 500, "x2": 300}]))
    detector = GeminiErDetector(api=api)
    detector.load("")
    detector.set_vocabulary([])
    boxes = detector.detect(frame(200, 100))
    assert api.prompts[0].startswith("Detect every distinct object")
    assert api.images[0][:3] == b"\xff\xd8\xff"  # a JPEG
    assert [(b.label, b.confidence, b.x0, b.y0, b.x1, b.y1) for b in boxes] == [("red mug", 1.0, 20.0, 10.0, 60.0, 50.0)]


def test_an_identify_search_asks_one_question_and_names_its_boxes():
    api = FakeApi(json.dumps([{"label": "cup", "y": 0, "x": 0, "y2": 1000, "x2": 500}]))
    detector = GeminiErDetector(api=api)
    detector.load("")
    detector.set_vocabulary(["the mug"])
    boxes = detector.detect(frame(200, 100))
    assert api.prompts == [identify_prompt("the mug")]
    assert [(b.label, b.x1) for b in boxes] == [("the mug", 100.0)]
    assert detector.calls == 1 and detector.spent_usd > 0.0


def test_a_scan_for_named_items_lists_them_and_drops_the_rest():
    api = FakeApi(json.dumps([
        {"label": "mug", "y": 0, "x": 0, "y2": 500, "x2": 500},
        {"label": "spoon", "y": 0, "x": 500, "y2": 500, "x2": 1000},
    ]))
    detector = GeminiErDetector(api=api)
    detector.load("")
    detector.set_vocabulary(["mug", "banana"])
    boxes = detector.detect(frame())
    assert api.prompts == [list_prompt(["mug", "banana"])]
    assert [b.label for b in boxes] == ["mug"]
