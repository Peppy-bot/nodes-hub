"""The tool loop against a scripted step model and two fake tools: what
each answer does, what the history carries, how it ends, and the schema it
builds from the tools."""

import asyncio
import json
from dataclasses import dataclass

import pytest

from vlm_router_brain.manipulators.vlm.loop import UNREADABLE, LoopEnd, Observation, Picture, ToolLoop, ToolResult, Turn, answer_schema
from brain_core.ports import Cancelled, CancelToken


class FakeStepModel:
    """Answers the texts the test scripted, in order, and records every
    turn. With no text left it waits until the test lets it go."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.turns: list[Turn] = []
        self.release = asyncio.Event()

    async def next_step(self, turn: Turn) -> str:
        self.turns.append(turn)
        if not self.answers:
            await self.release.wait()
            return '{"action": "done"}'
        return self.answers.pop(0)


@dataclass(frozen=True)
class Stepped:
    by: float


class StepTool:
    name = "step"
    description = "moves by x"
    fields = {"x": "number"}

    def __init__(self) -> None:
        self.ran: list[Stepped] = []

    def parse(self, fields) -> Stepped:
        try:
            by = float(fields["x"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("x must be a number") from None
        return Stepped(by)

    async def run(self, arguments: Stepped, cancel: CancelToken) -> ToolResult:
        self.ran.append(arguments)
        return ToolResult(f"stepped by {arguments.by:g}")


class DoneTool:
    name = "done"
    description = "ends"
    fields: dict = {}

    def parse(self, fields) -> None:
        return None

    async def run(self, arguments, cancel: CancelToken) -> ToolResult:
        return ToolResult("ended", ends=True, value="finished")


def observing(*states: str):
    """An observer that gives `states` in turn, each with one picture,
    and counts its calls."""
    calls = {"count": 0}
    remaining = list(states)

    async def observe() -> Observation:
        calls["count"] += 1
        state = remaining.pop(0) if remaining else "later"
        return Observation(state, (Picture("wrist", b"jpeg"),))

    observe.calls = calls  # type: ignore[attr-defined]
    return observe


def loop(model, *, max_steps: int = 5, observe=None) -> tuple[ToolLoop, StepTool, DoneTool]:
    step, done = StepTool(), DoneTool()
    return ToolLoop(model, [step, done], "the goal", max_steps, observe or observing("first")), step, done


async def test_done_at_once_ends_in_one_step_with_the_tools_value():
    model = FakeStepModel('{"action": "done"}')
    the_loop, step, _ = loop(model)
    end = await the_loop.run(CancelToken())
    assert end == LoopEnd("finished", 1, False)
    assert step.ran == [] and len(model.turns) == 1
    turn = model.turns[0]
    assert (turn.goal, turn.step, turn.max_steps, turn.state, turn.history) == ("the goal", 1, 5, "first", ())
    assert turn.pictures == (Picture("wrist", b"jpeg"),) and [tool.name for tool in turn.tools] == ["step", "done"]


async def test_a_step_then_done_runs_the_tool_once_and_the_next_turn_reads_the_history():
    model = FakeStepModel('{"action": "step", "x": 2.5}', '{"action": "done"}')
    the_loop, step, _ = loop(model, observe=observing("first", "second"))
    end = await the_loop.run(CancelToken())
    assert end == LoopEnd("finished", 2, False)
    assert len(step.ran) == 1 and step.ran[0].by == 2.5
    second = model.turns[1]
    assert second.step == 2 and second.state == "second"
    assert second.history == ('{"action": "step", "x": 2.5} -> stepped by 2.5',)


@pytest.mark.parametrize("answer, why", [
    ("not json at all", "not JSON"),
    ("[1, 2]", "not a JSON object"),
    ('{"x": 1}', "no action"),
    ('{"action": 3}', "no action"),
    ('{"action": "fly"}', "no tool is named 'fly'; the tools are step, done"),
    ('{"action": "step"}', "x must be a number"),
    ('{"action": "step", "x": "far"}', "x must be a number"),
])
async def test_an_answer_the_loop_cannot_read_runs_nothing_counts_and_is_fed_back(answer, why):
    model = FakeStepModel(answer, '{"action": "done"}')
    the_loop, step, _ = loop(model)
    end = await the_loop.run(CancelToken())
    assert end.steps == 2 and step.ran == []
    (line,) = model.turns[1].history
    assert line.startswith(answer[:40]) and f"{UNREADABLE}: {why}" in line


async def test_five_steps_with_no_end_run_out():
    model = FakeStepModel(*['{"action": "step", "x": 1}'] * 5)
    the_loop, step, _ = loop(model, max_steps=5)
    end = await the_loop.run(CancelToken())
    assert end == LoopEnd(None, 5, True)
    assert len(step.ran) == 5 and len(model.turns) == 5
    assert len(model.turns[4].history) == 4


async def test_the_history_of_step_n_holds_every_earlier_answer_and_observation_in_order():
    model = FakeStepModel('{"action": "step", "x": 1}', "junk", '{"action": "step", "x": 3}', '{"action": "done"}')
    the_loop, _, _ = loop(model)
    await the_loop.run(CancelToken())
    assert model.turns[3].history == (
        '{"action": "step", "x": 1} -> stepped by 1',
        f"junk -> {UNREADABLE}: not JSON (Expecting value)",
        '{"action": "step", "x": 3} -> stepped by 3',
    )


async def test_a_long_answer_is_cut_in_the_history():
    long = json.dumps({"action": "step", "x": 1, "note": "n" * 400})
    model = FakeStepModel(long, '{"action": "done"}')
    the_loop, _, _ = loop(model)
    await the_loop.run(CancelToken())
    (line,) = model.turns[1].history
    shown = line.split(" -> ")[0]
    assert len(shown) == 200 and shown.endswith("...")


async def test_a_cancel_during_the_models_call_raises_cancelled_without_waiting_for_the_answer():
    model = FakeStepModel()  # no answer: it waits until released
    the_loop, _, _ = loop(model)
    cancel = CancelToken()
    running = asyncio.ensure_future(the_loop.run(cancel))
    for _ in range(10):
        await asyncio.sleep(0)
    assert len(model.turns) == 1 and not running.done()
    cancel.cancel("aborted: test", signalled=False)
    with pytest.raises(Cancelled):
        await running
    assert not model.release.is_set()


async def test_the_observer_runs_once_per_step_before_the_model():
    observe = observing("a", "b", "c")
    model = FakeStepModel('{"action": "step", "x": 1}', '{"action": "step", "x": 1}', '{"action": "done"}')
    the_loop, _, _ = loop(model, observe=observe)
    await the_loop.run(CancelToken())
    assert observe.calls["count"] == 3
    assert [turn.state for turn in model.turns] == ["a", "b", "c"]


def test_the_schema_is_one_flat_object_with_the_tools_names_and_fields_and_no_strictness():
    schema = answer_schema([StepTool(), DoneTool()])
    assert schema["type"] == "json_schema"
    inner = schema["json_schema"]
    assert inner["strict"] is False and inner["name"] == "step"
    assert inner["schema"] == {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": ["step", "done"]}, "x": {"type": "number"}},
        "required": ["action"],
    }


def test_two_tools_giving_one_field_different_types_are_refused():
    class Other:
        name = "other"
        description = ""
        fields = {"x": "string"}

        def parse(self, fields):
            return None

        async def run(self, arguments, cancel):
            return ToolResult("")

    with pytest.raises(ValueError, match="field 'x' is a number for one tool and a string for 'other'"):
        answer_schema([StepTool(), Other()])


def test_two_tools_of_one_name_and_a_loop_with_no_step_are_refused():
    with pytest.raises(ValueError, match="two tools are named 'step'"):
        ToolLoop(FakeStepModel(), [StepTool(), StepTool()], "g", 1, observing())
    with pytest.raises(ValueError, match="at least one step"):
        ToolLoop(FakeStepModel(), [StepTool()], "g", 0, observing())
