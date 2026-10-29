"""The perception backend of a catalog model: brain_core's `Detector` over
one route of it.

A search asks one question with one picture: the colour frame at its full
resolution as a JPEG at quality 95, and the prompt its phrases choose
(prompts.py). The answer is read against the schema (parsing.py) and its
boxes placed in the frame's pixels. A hosted model gives no confidence, so
every box reports 0, as the contracts ask of an implementer with none of
its own; brain_core then takes the first matching box in the answer's
order.

A search keeps its deadline the way every call of this node does
(`calls.ask_once_more`): a call gets the budget left, never under the
route's `min_call_timeout_s`, the least deadline the provider takes; a
call that failed is asked again once, after `RETRY_WAIT_S`, when the
failure is one that comes back at once (`ProviderFailure.retryable`) and
the budget has room for the wait; and a search whose budget is gone
before a call stops there, so the core fails it with `timeout`. Each call
logs its route, its tokens and its latency, and no cost.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Sequence

import numpy as np

from brain_core.ports import Box, Coverage, Deadline

from .calls import RETRY_WAIT_S, Sleep, ask_once_more
from .catalog import CatalogEntry, Route
from .parsing import Answer, UnreadableAnswer, labelled, read_answer, yx_0_1000_to_pixels
from .prompts import NOTHING, plan_for
from .provider import Provider, Reply

__all__ = ["RETRY_WAIT_S", "VlmDetector", "jpeg_bytes", "log_call", "probe"]

logger = logging.getLogger(__name__)

JPEG_QUALITY = 95
# The longest piece of an unreadable answer the log keeps.
ANSWER_LOG_CHARS = 500
# The converter of each coordinate format the catalog names.
CONVERTERS = {"yx_0_1000": yx_0_1000_to_pixels}
# The switch probe: a small grey picture and a question whose one right
# answer is "nothing", within at least this long.
PROBE_SIDE_PX = 64
PROBE_PROMPT = f"This picture is a connection test and shows nothing. Answer {NOTHING}."
PROBE_TIMEOUT_S = 30.0


def jpeg_bytes(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(out, format="JPEG", quality=quality)
    return out.getvalue()


class VlmDetector:
    def __init__(self, entry: CatalogEntry, route: Route, provider: Provider, *, sleep: Sleep = asyncio.sleep) -> None:
        self.name = entry.name
        self.route = route
        self._provider = provider
        self._to_pixels = CONVERTERS[entry.coordinates]
        self._sleep = sleep

    def why_unavailable(self) -> str:
        return ""

    def scan_coverage(self) -> Coverage:
        """A scan asks for every object in the model's own words, so it can
        name any label."""
        return Coverage(every_label=True)

    async def detect(self, image: np.ndarray, phrases: Sequence[str], deadline: Deadline) -> list[Box]:
        height, width = image.shape[:2]
        plan = plan_for(phrases)
        reply = await ask_once_more(
            self._provider, self.route, [jpeg_bytes(image)], plan.prompt, Answer, deadline, name=self.name, sleep=self._sleep
        )
        log_call(self.name, plan.mode, self.route, reply)
        try:
            answer = read_answer(reply.text)
        except UnreadableAnswer:
            logger.warning("%s: the answer could not be read: %r", self.name, reply.text[:ANSWER_LOG_CHARS])
            raise
        boxes: list[Box] = []
        for label, box in labelled(answer, plan.force_label):
            pixels = self._to_pixels(box.box_2d, width, height)
            if pixels is not None:
                boxes.append(Box(label, 0.0, *pixels))
        return boxes



async def probe(provider: Provider, route: Route) -> None:
    """One call that proves the route reaches the model, the model takes a
    picture, and its answer reads against the schema. A ProviderFailure or
    an UnreadableAnswer says why not."""
    picture = jpeg_bytes(np.full((PROBE_SIDE_PX, PROBE_SIDE_PX, 3), 128, dtype=np.uint8))
    reply = await provider.ask(route, [picture], PROBE_PROMPT, Answer, max(PROBE_TIMEOUT_S, route.min_call_timeout_s))
    log_call(route.model_name, "probe", route, reply)
    try:
        read_answer(reply.text)
    except UnreadableAnswer:
        logger.warning("%s: the probe's answer could not be read: %r", route.model_name, reply.text[:ANSWER_LOG_CHARS])
        raise


def log_call(model_name: str, mode: str, route: Route, reply: Reply) -> None:
    """The line each call to a model leaves: its route, its tokens and its
    latency, and no cost."""
    logger.info(
        "%s: %s call through %s, %d in / %d out / %d reasoning tokens, %.2f s",
        model_name, mode, route.spelled, reply.input_tokens, reply.output_tokens, reply.reasoning_tokens, reply.latency_s,
    )
