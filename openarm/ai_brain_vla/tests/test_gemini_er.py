"""The gemini_er backend around its API: the prompts, the repair of the
model's JSON, the merging of answers, the key lookup and the search plan,
all against a fake API. The real client is never constructed here."""

import json

import numpy as np
import pytest

from conftest import NEVER, PASSED
from openarm_ai_brain_vla.perception import gemini_er
from openarm_ai_brain_vla.perception import make_detector
from openarm_ai_brain_vla.perception.gemini_er import (
    IDENTIFY,
    KEY_ENV,
    LIST,
    MIN_CALL_TIMEOUT_S,
    MIN_CONFIDENCE,
    MODEL,
    OPEN,
    Answer,
    ApiFailure,
    GeminiErDetector,
    api_key,
    identify_prompt,
    list_prompt,
    merge_samples,
    parse_boxes,
    plan_for,
)
from openarm_ai_brain_vla.ports import Coverage, SearchTimeout


class FakeApi:
    """Answers every question with the texts it was given, in order, a
    failure where a text is an ApiFailure, and keeps what it was asked and
    the deadline each call got."""

    def __init__(self, *texts) -> None:
        self.texts = list(texts)
        self.prompts: list[str] = []
        self.images: list[bytes] = []
        self.timeouts: list[float] = []

    def ask(self, image: bytes, prompt: str, timeout_s: float) -> Answer:
        self.images.append(image)
        self.prompts.append(prompt)
        self.timeouts.append(timeout_s)
        text = self.texts.pop(0) if self.texts else "[]"
        if isinstance(text, ApiFailure):
            raise text
        return Answer(text, input_tokens=1000, output_tokens=20, thought_tokens=100, latency_s=1.5)


class Budget:
    """A deadline with a fixed amount of time left, never passed."""

    def __init__(self, remaining_s: float) -> None:
        self._remaining_s = remaining_s

    def check(self) -> None:
        return None

    def remaining_s(self) -> float:
        return self._remaining_s


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
        detector.load("", "")
    assert not detector.available
    assert detector.detect(frame(), NEVER) == []


def test_the_registry_builds_the_backend_and_the_model_id_defaults():
    detector = make_detector("gemini_er")
    assert detector.name == "gemini_er"
    assert detector.min_confidence == MIN_CONFIDENCE == 0.5
    fake = GeminiErDetector(api=FakeApi())
    fake.load("  ", "")
    assert fake.model == MODEL and fake.available
    fake.load("gemini-robotics-er-3", "")
    assert fake.model == "gemini-robotics-er-3"


def test_a_gallery_fails_the_load_since_the_backend_reads_none(tmp_path):
    # Refused before the key is even looked for.
    detector = GeminiErDetector()
    with pytest.raises(ValueError, match="gemini_er takes no enrolment gallery"):
        detector.load("", str(tmp_path))
    assert not detector.available


def test_a_scan_covers_every_label_once_the_client_is_open():
    assert GeminiErDetector().scan_coverage() == Coverage()
    assert GeminiErDetector(api=FakeApi()).scan_coverage() == Coverage(every_label=True)


def test_a_scan_asks_for_everything_and_keeps_the_models_labels():
    api = FakeApi(json.dumps([{"label": "red mug", "y": 100, "x": 100, "y2": 500, "x2": 300}]))
    detector = GeminiErDetector(api=api)
    detector.load("", "")
    detector.set_vocabulary([])
    boxes = detector.detect(frame(200, 100), NEVER)
    assert api.prompts[0].startswith("Detect every distinct object")
    assert api.images[0][:3] == b"\xff\xd8\xff"  # a JPEG
    assert [(b.label, b.confidence, b.x0, b.y0, b.x1, b.y1) for b in boxes] == [("red mug", 1.0, 20.0, 10.0, 60.0, 50.0)]


def test_an_identify_search_asks_one_question_and_names_its_boxes():
    api = FakeApi(json.dumps([{"label": "cup", "y": 0, "x": 0, "y2": 1000, "x2": 500}]))
    detector = GeminiErDetector(api=api)
    detector.load("", "")
    detector.set_vocabulary(["the mug"])
    boxes = detector.detect(frame(200, 100), NEVER)
    assert api.prompts == [identify_prompt("the mug")]
    assert [(b.label, b.x1) for b in boxes] == [("the mug", 100.0)]
    assert detector.calls == 1 and detector.spent_usd > 0.0


def test_a_scan_for_named_items_lists_them_and_drops_the_rest():
    api = FakeApi(json.dumps([
        {"label": "mug", "y": 0, "x": 0, "y2": 500, "x2": 500},
        {"label": "spoon", "y": 0, "x": 500, "y2": 500, "x2": 1000},
    ]))
    detector = GeminiErDetector(api=api)
    detector.load("", "")
    detector.set_vocabulary(["mug", "banana"])
    boxes = detector.detect(frame(), NEVER)
    assert api.prompts == [list_prompt(["mug", "banana"])]
    assert [b.label for b in boxes] == ["mug"]


MUG = json.dumps([{"label": "mug", "y": 0, "x": 0, "y2": 500, "x2": 500}])


def test_a_call_gets_the_budget_left_never_under_the_apis_least_deadline():
    api = FakeApi(MUG, MUG)
    detector = GeminiErDetector(api=api)
    detector.load("", "")
    detector.set_vocabulary(["mug"])
    detector.detect(frame(), Budget(30.0))
    detector.detect(frame(), Budget(3.0))
    assert api.timeouts == [30.0, MIN_CALL_TIMEOUT_S]


def test_a_search_whose_budget_is_gone_stops_before_the_call():
    api = FakeApi(MUG)
    detector = GeminiErDetector(api=api)
    detector.load("", "")
    detector.set_vocabulary(["mug"])
    with pytest.raises(SearchTimeout, match="did not finish within 1 s"):
        detector.detect(frame(), PASSED)
    assert api.prompts == [] and detector.calls == 0


def test_a_refused_call_is_asked_again_only_with_budget_for_the_wait(monkeypatch):
    monkeypatch.setattr(gemini_er, "RETRY_WAIT_S", 0.0)
    # Room for another call: the 429 is asked again and answered.
    api = FakeApi(ApiFailure(429, "quota"), MUG)
    detector = GeminiErDetector(api=api)
    detector.load("", "")
    detector.set_vocabulary(["mug"])
    assert [b.label for b in detector.detect(frame(), Budget(30.0))] == ["mug"]
    assert len(api.prompts) == 2 and detector.calls == 1
    # No room: the failure is the search's, and the bill stops at one call.
    api = FakeApi(ApiFailure(503, "overloaded"), MUG)
    detector = GeminiErDetector(api=api)
    detector.load("", "")
    detector.set_vocabulary(["mug"])
    with pytest.raises(ApiFailure, match="the API answered 503: overloaded"):
        detector.detect(frame(), Budget(0.0))
    assert len(api.prompts) == 1
    # A refusal that will not change with a retry is never asked again.
    api = FakeApi(ApiFailure(400, "bad request"), MUG)
    detector = GeminiErDetector(api=api)
    detector.load("", "")
    detector.set_vocabulary(["mug"])
    with pytest.raises(ApiFailure, match="400"):
        detector.detect(frame(), Budget(30.0))
    assert len(api.prompts) == 1
    assert ApiFailure(429, "").retryable and ApiFailure(500, "").retryable and not ApiFailure(404, "").retryable


def test_the_confidence_is_the_share_of_answers_and_the_floor_applies_to_it(monkeypatch):
    monkeypatch.setattr(gemini_er, "SAMPLES", 2)
    api = FakeApi(MUG, "[]")
    detector = GeminiErDetector(api=api)
    detector.load("", "")
    detector.set_vocabulary(["mug"])
    boxes = detector.detect(frame(), NEVER)
    assert [(b.label, b.confidence) for b in boxes] == [("mug", 0.5)]
    api = FakeApi(MUG, "[]")
    detector = GeminiErDetector(api=api)
    detector.load("", "")
    detector.min_confidence = 0.6
    detector.set_vocabulary(["mug"])
    assert detector.detect(frame(), NEVER) == []
