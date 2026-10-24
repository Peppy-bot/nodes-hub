"""The detector over a scripted provider: one question per search with the
frame as a JPEG, boxes in pixels with confidence 0, the deadline and the
one retry, and the line each call leaves in the log."""

import io
import logging
import time

import numpy as np
import pytest
from conftest import FakeProvider, TEST_CATALOG, answer, records_of
from PIL import Image

from brain_core.ports import Coverage, Deadline, FailureKind, SearchTimeout

from vlm_router_brain.parsing import UnreadableAnswer
from vlm_router_brain.provider import ProviderFailure, Reply
from vlm_router_brain.vlm_detector import PROBE_PROMPT, RETRY_WAIT_S, VlmDetector, probe

FAST = TEST_CATALOG.get("fast")
FRAME = np.full((720, 1280, 3), 90, dtype=np.uint8)
# A deadline no search reaches.
FAR = Deadline(at=time.monotonic() + 1000.0, budget_s=1000.0)


class Waits:
    """Stands in for asyncio.sleep: records each wait and returns at once."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def detector_on(provider: FakeProvider, waits: Waits = None) -> VlmDetector:
    return VlmDetector(FAST, FAST.routes[0], provider, sleep=waits or Waits())


def retryable(message="slow down") -> ProviderFailure:
    return ProviderFailure("fast", "RateLimitError", message, retryable=True)


async def test_an_identify_search_asks_one_question_with_the_whole_frame_and_reports_no_confidence():
    provider = FakeProvider(answer(("the mustard bottle", [462, 312, 630, 382]), ("bottle", [0, 0, 100, 100])))
    detector = detector_on(provider)
    boxes = await detector.detect(FRAME, ["the mustard bottle"], FAR)
    [call] = provider.calls
    assert call.prompt.startswith("Find the mustard bottle in this image")
    assert call.route is FAST.routes[0]
    with Image.open(io.BytesIO(call.image)) as sent:
        assert (sent.format, sent.size) == ("JPEG", (1280, 720))
    # Every box an identify search returns is its item, in pixels.
    assert [box.label for box in boxes] == ["the mustard bottle", "the mustard bottle"]
    assert [box.confidence for box in boxes] == [0.0, 0.0]
    assert (boxes[0].x0, boxes[0].y0, boxes[0].x1, boxes[0].y1) == pytest.approx((399.36, 332.64, 488.96, 453.6))
    assert detector.why_unavailable() == "" and detector.scan_coverage() == Coverage(every_label=True)
    assert detector.name == "fast"


async def test_nothing_visible_is_no_box_and_a_box_under_a_pixel_is_dropped():
    assert await detector_on(FakeProvider(Reply('{"boxes": []}'))).detect(FRAME, [], FAR) == []
    flat = answer(("cup", [100, 100, 100, 400]))
    assert await detector_on(FakeProvider(flat)).detect(FRAME, [], FAR) == []


async def test_a_search_for_two_items_fails_before_any_call():
    provider = FakeProvider()
    with pytest.raises(ValueError, match="^vlm_router_brain searches for one item at a time; this search names 2: 'cup', 'bowl'$"):
        await detector_on(provider).detect(FRAME, ["cup", "bowl"], FAR)
    assert provider.calls == []


async def test_a_retryable_failure_is_asked_again_once_after_the_wait():
    waits = Waits()
    provider = FakeProvider(retryable(), answer(("cup", [0, 0, 100, 100])))
    boxes = await detector_on(provider, waits).detect(FRAME, ["cup"], FAR)
    assert [box.label for box in boxes] == ["cup"]
    assert len(provider.calls) == 2 and waits.waits == [RETRY_WAIT_S]


async def test_a_second_retryable_failure_fails_the_search():
    provider = FakeProvider(retryable("first"), retryable("second"), answer(("cup", [0, 0, 100, 100])))
    with pytest.raises(ProviderFailure, match="^fast failed: RateLimitError: second$"):
        await detector_on(provider).detect(FRAME, ["cup"], FAR)
    assert len(provider.calls) == 2


async def test_no_retry_when_the_budget_has_no_room_for_the_wait():
    waits = Waits()
    provider = FakeProvider(retryable(), answer(("cup", [0, 0, 100, 100])))
    short = Deadline(at=time.monotonic() + RETRY_WAIT_S / 2, budget_s=RETRY_WAIT_S / 2)
    with pytest.raises(ProviderFailure, match="slow down"):
        await detector_on(provider, waits).detect(FRAME, ["cup"], short)
    assert len(provider.calls) == 1 and waits.waits == []


async def test_a_failure_that_may_not_be_asked_again_fails_at_once():
    waits = Waits()
    lost = ProviderFailure("fast", "ProviderError", "connection reset", retryable=False)
    provider = FakeProvider(lost, answer(("cup", [0, 0, 100, 100])))
    with pytest.raises(ProviderFailure, match="^fast failed: ProviderError: connection reset$"):
        await detector_on(provider, waits).detect(FRAME, ["cup"], FAR)
    assert len(provider.calls) == 1 and waits.waits == []


async def test_a_call_gets_the_budget_left_but_never_less_than_the_routes_floor():
    provider = FakeProvider()
    await detector_on(provider).detect(FRAME, [], FAR)
    # The budget left, above the route's 10 s floor however long the
    # tests took to come here.
    assert 10.0 < provider.calls[0].timeout_s < 1000.0
    # Two seconds left: the call still gets the route's 10 s.
    await detector_on(provider).detect(FRAME, [], Deadline(at=time.monotonic() + 2.0, budget_s=2.0))
    assert provider.calls[1].timeout_s == 10.0


async def test_a_search_whose_budget_is_gone_before_a_call_makes_none():
    provider = FakeProvider()
    with pytest.raises(SearchTimeout):
        await detector_on(provider).detect(FRAME, [], Deadline(at=0.0, budget_s=4.0))
    assert provider.calls == []


async def test_an_unreadable_answer_fails_the_search_and_its_text_goes_to_the_log(caplog):
    provider = FakeProvider(Reply("I see a yellow bottle on the left."))
    with records_of(caplog, "vlm_router_brain.vlm_detector") as log:
        with pytest.raises(UnreadableAnswer) as raised:
            await detector_on(provider).detect(FRAME, ["the mustard bottle"], FAR)
    assert raised.value.message == "the model's answer could not be read"
    assert raised.value.kind is FailureKind.ATTEMPT_FAILED
    [unreadable] = [record for record in log.records if record.levelno == logging.WARNING]
    assert unreadable.getMessage() == "fast: the answer could not be read: 'I see a yellow bottle on the left.'"


async def test_each_call_logs_its_route_tokens_and_latency_and_no_cost(caplog):
    provider = FakeProvider(answer(("cup", [0, 0, 100, 100])))
    with records_of(caplog, "vlm_router_brain.vlm_detector") as log:
        await detector_on(provider).detect(FRAME, ["cup"], FAR)
    assert [record.getMessage() for record in log.records] == [
        "fast: identify call through gemini:gemini-fast, 100 in / 20 out / 5 reasoning tokens, 1.50 s"
    ]
    assert "$" not in log.text and "cost" not in log.text


async def test_the_probe_asks_a_small_picture_for_nothing_and_reads_the_answer():
    provider = FakeProvider(Reply('{"boxes": []}'))
    route = FAST.routes[0]
    await probe(provider, route)
    [call] = provider.calls
    assert call.prompt == PROBE_PROMPT and call.timeout_s == 30.0
    with Image.open(io.BytesIO(call.image)) as sent:
        assert sent.size == (64, 64)
    with pytest.raises(UnreadableAnswer):
        await probe(FakeProvider(Reply("hello")), route)
    with pytest.raises(ProviderFailure, match="no such model"):
        await probe(FakeProvider(ProviderFailure("fast", "ModelNotFoundError", "no such model", retryable=False)), route)
