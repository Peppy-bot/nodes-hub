"""The step model of a catalog model: the loop's `StepModel` over one route
of it, the model behind the alignment loop.

A step asks one question with the turn's pictures, in their order, and
the prompt `prompt.step_prompt` renders from the turn: the goal, the state,
the earlier steps and the actions. The answer is held to the flat schema
the loop builds from its tools (`loop.answer_schema`), and comes back as
text for the loop to read. A step has `STEP_TIMEOUT_S` for its call, never
under the route's floor, with the one retry every call of this node gets
(`calls.ask_once_more`). Each call logs its route, its tokens and its
latency.

`probe_alignment` is the switch probe of a manipulation model: one call
with a grey picture whose one right answer is `{"action": "give_up"}`,
held to the actions' schema, so a switch to a model that cannot be reached
or that does not answer an action fails before the role takes it.
"""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np

from brain_core.ports import Deadline

from ...calls import Sleep, ask_once_more
from ...catalog import CatalogEntry, Route
from ...parsing import UnreadableAnswer
from ...provider import Provider
from ...vlm_detector import ANSWER_LOG_CHARS, PROBE_SIDE_PX, PROBE_TIMEOUT_S, jpeg_bytes, log_call
from .aligner import ACTIONS
from .loop import Turn, answer_schema
from .prompt import step_prompt

logger = logging.getLogger(__name__)

STEP_TIMEOUT_S = 60.0
PROBE_PROMPT = 'This picture is a connection test and shows nothing. Answer {"action": "give_up"}.'


class _Named:
    """A tool as the probe's schema needs it: a name and no fields."""

    description = ""
    fields: dict[str, str] = {}

    def __init__(self, name: str) -> None:
        self.name = name


PROBE_SCHEMA = answer_schema([_Named(action) for action in ACTIONS])


class RouterStepModel:
    def __init__(self, entry: CatalogEntry, route: Route, provider: Provider, *, sleep: Sleep = asyncio.sleep) -> None:
        self.name = entry.name
        self.route = route
        self._provider = provider
        self._sleep = sleep

    async def next_step(self, turn: Turn) -> str:
        deadline = Deadline.after(max(STEP_TIMEOUT_S, self.route.min_call_timeout_s))
        reply = await ask_once_more(
            self._provider, self.route, [picture.jpeg for picture in turn.pictures], step_prompt(turn), answer_schema(turn.tools),
            deadline, name=self.name, sleep=self._sleep,
        )
        log_call(self.name, f"align step {turn.step}", self.route, reply)
        return reply.text


async def probe_alignment(provider: Provider, route: Route) -> None:
    """One call that proves the route reaches the model, the model takes a
    picture, and its answer names an action. A ProviderFailure or an
    UnreadableAnswer says why not."""
    picture = jpeg_bytes(np.full((PROBE_SIDE_PX, PROBE_SIDE_PX, 3), 128, dtype=np.uint8))
    reply = await provider.ask(route, [picture], PROBE_PROMPT, PROBE_SCHEMA, max(PROBE_TIMEOUT_S, route.min_call_timeout_s))
    log_call(route.model_name, "probe", route, reply)
    try:
        action = json.loads(reply.text).get("action")
    except (json.JSONDecodeError, AttributeError):
        action = None
    if action not in ACTIONS:
        logger.warning("%s: the probe's answer names no action: %r", route.model_name, reply.text[:ANSWER_LOG_CHARS])
        raise UnreadableAnswer(reply.text)
