"""The step model over a scripted provider: a turn becomes one call with
its pictures, the prompt and the loop's schema; the retry rule is the one
every call gets; the probe of a manipulation model asks for an action."""

import json
import logging

import pytest
from conftest import TEST_CATALOG, FakeProvider, records_of
from test_vlm_detector import Waits, retryable

from vlm_router_brain.calls import RETRY_WAIT_S
from vlm_router_brain.manipulators.vlm.aligner import ACTIONS
from vlm_router_brain.manipulators.vlm.loop import Picture, Turn, answer_schema
from vlm_router_brain.manipulators.vlm.prompt import step_prompt
from vlm_router_brain.manipulators.vlm.step_model import PROBE_PROMPT, PROBE_SCHEMA, STEP_TIMEOUT_S, RouterStepModel, probe_alignment
from vlm_router_brain.parsing import UnreadableAnswer
from vlm_router_brain.provider import ProviderFailure, Reply

FAST = TEST_CATALOG.get("fast")


class MoveTool:
    name = "move"
    description = "moves the gripper sideways by dx and dy and turns it by roll_deg"
    fields = {"dx": "number", "dy": "number", "roll_deg": "number"}


class ReadyTool:
    name = "ready"
    description = "the gripper is centred: it goes down"
    fields: dict = {}


def turn(step: int = 1, history=()) -> Turn:
    return Turn(
        goal="Align the gripper above the cube.", tools=(MoveTool(), ReadyTool()), history=tuple(history),
        state="The grasp point is +0.0 cm in x and +0.0 cm in y from the cube's seen centre.",
        pictures=(Picture("wrist", b"\xff\xd8wrist"), Picture("chest", b"\xff\xd8chest")), step=step, max_steps=5,
    )


def model_on(provider: FakeProvider, waits: Waits = None) -> RouterStepModel:
    return RouterStepModel(FAST, FAST.routes[0], provider, sleep=waits or Waits())


def test_the_prompt_carries_the_goal_the_state_the_history_and_each_action_with_its_fields():
    text = step_prompt(turn(step=2, history=['{"action": "move", "dx": 1, "dy": 0, "roll_deg": 45} -> moved: ...']))
    assert text.startswith("Align the gripper above the cube.\n\nNow: step 2 of 5. The grasp point is +0.0 cm")
    assert "Earlier steps:\n{\"action\": \"move\", \"dx\": 1, \"dy\": 0, \"roll_deg\": 45} -> moved: ...\n" in text
    assert text.endswith(
        "Answer with one JSON object, one of:\n"
        '{"action": "move", "dx": <number>, "dy": <number>, "roll_deg": <number>}\n'
        "  moves the gripper sideways by dx and dy and turns it by roll_deg\n"
        '{"action": "ready"}\n'
        "  the gripper is centred: it goes down"
    )
    assert "Earlier steps:\nnone\n" in step_prompt(turn())


async def test_a_step_is_one_call_with_the_pictures_in_order_the_prompt_and_the_schema(caplog):
    provider = FakeProvider(Reply('{"action": "ready"}', 2800, 40, 30, 9.4))
    with records_of(caplog, "vlm_router_brain.vlm_detector") as log:
        text = await model_on(provider).next_step(turn())
    assert text == '{"action": "ready"}'
    [call] = provider.calls
    assert call.images == (b"\xff\xd8wrist", b"\xff\xd8chest")
    assert call.prompt == step_prompt(turn())
    assert call.schema == answer_schema(turn().tools) and call.schema["json_schema"]["strict"] is False
    assert call.route is FAST.routes[0] and call.timeout_s == pytest.approx(STEP_TIMEOUT_S, abs=0.5)
    assert [record.getMessage() for record in log.records] == [
        "fast: align step 1 call through gemini:gemini-fast, 2800 in / 40 out / 30 reasoning tokens, 9.40 s"
    ]


async def test_a_failure_that_may_be_asked_again_is_asked_once_more_and_another_fails_the_step():
    waits = Waits()
    provider = FakeProvider(retryable(), Reply('{"action": "ready"}'))
    assert await model_on(provider, waits).next_step(turn()) == '{"action": "ready"}'
    assert len(provider.calls) == 2 and waits.waits == [RETRY_WAIT_S]
    provider = FakeProvider(ProviderFailure("fast", "AuthenticationError", "bad key", retryable=False))
    with pytest.raises(ProviderFailure, match="fast failed: AuthenticationError: bad key"):
        await model_on(provider).next_step(turn())


async def test_the_probe_asks_for_an_action_with_a_grey_picture_and_reads_the_answer():
    provider = FakeProvider(Reply('{"action": "give_up"}'))
    await probe_alignment(provider, FAST.routes[0])
    [call] = provider.calls
    assert call.prompt == PROBE_PROMPT and call.schema is PROBE_SCHEMA and call.image.startswith(b"\xff\xd8")
    assert PROBE_SCHEMA["json_schema"]["schema"]["properties"]["action"]["enum"] == list(ACTIONS)
    for text in ('{"boxes": []}', "not json", '{"action": "fly"}', "[1]"):
        with pytest.raises(UnreadableAnswer):
            await probe_alignment(FakeProvider(Reply(text)), FAST.routes[0])


async def test_the_probe_logs_its_call_and_an_answer_it_cannot_read(caplog):
    with records_of(caplog, "vlm_router_brain.manipulators.vlm.step_model") as log, pytest.raises(UnreadableAnswer):
        await probe_alignment(FakeProvider(Reply(json.dumps({"action": "fly"}))), FAST.routes[0])
    assert [record.levelno for record in log.records] == [logging.WARNING]
    assert "the probe's answer names no action" in log.records[0].getMessage()
