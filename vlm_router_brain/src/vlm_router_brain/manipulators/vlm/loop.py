"""A loop that asks a model for one action per step, over a fixed set of
tools, and knows nothing of what the tools do.

Each step: `observe` gives the state as text and the pictures of the
moment; the loop builds a `Turn` from the goal, the tools, the history of
the earlier steps and that observation; the `StepModel` answers with the
text of one JSON object; the loop reads its `action`, finds the tool of
that name, lets the tool parse its own fields and runs it; one line goes
into the history, the answer and what the tool observed. The loop ends
when a tool's result ends it, or when `max_steps` steps have run.

Every step counts, a bad one too: an answer that is not a JSON object, an
action no tool has, or a field a tool refuses runs nothing, and its reason
goes into the history as `nothing was done: the answer could not be read:
<why>`, so the model reads what it got wrong and cannot loop on it for
free.

A cancel (`CancelToken`) cuts the model's call short: the loop waits on
the answer under `until(cancel.wait(), ...)`, then raises `Cancelled` at
its next check. What the model raises (a provider failure, an
`ExpectedFailure`) is raised out of `run`.

The answer's schema is one flat JSON object, made from the tools
(`answer_schema`): `action` is one of their names, and each tool's fields
are optional properties beside it, so one schema serves every provider. It
asks for no strictness: OpenAI's strict output would want every property
required. The tools parse plain JSON into their own values; nothing here
depends on a schema library.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Optional, Protocol, Sequence

from brain_core.ports import CancelToken
from brain_core.waiting import until

logger = logging.getLogger(__name__)

UNREADABLE = "nothing was done: the answer could not be read"
# The longest piece of an answer a history line keeps.
ANSWER_CHARS = 200


@dataclass(frozen=True)
class Picture:
    """One picture a step shows the model: its name, the one the goal's
    text calls it by, and its JPEG bytes."""

    name: str
    jpeg: bytes


@dataclass(frozen=True)
class Observation:
    """What one step sees before it asks: the state as text and the
    pictures of the moment."""

    state: str
    pictures: tuple[Picture, ...] = ()


@dataclass(frozen=True)
class ToolResult:
    """What a tool reports: `observation` goes into the history line, and
    `ends` with `value` end the loop."""

    observation: str
    ends: bool = False
    value: object = None


class Tool(Protocol):
    """One action the model can take. `fields` names its arguments and
    their JSON types ("number", "string", "boolean", "integer"), the
    properties of the answer's schema; `description` tells the model when
    to take it."""

    name: str
    description: str
    fields: Mapping[str, str]

    def parse(self, fields: Mapping[str, object]) -> object:
        """The tool's own value for the answer's fields, or a ValueError
        naming the field that is missing or wrong."""
        ...

    async def run(self, arguments: object, cancel: CancelToken) -> ToolResult: ...


@dataclass(frozen=True)
class Turn:
    """What the model gets for one step."""

    goal: str
    tools: tuple[Tool, ...]
    history: tuple[str, ...]
    state: str
    pictures: tuple[Picture, ...]
    step: int
    max_steps: int


class StepModel(Protocol):
    """The model behind the loop: one answer, the text of one JSON object,
    per turn. A provider failure is an `ExpectedFailure`."""

    async def next_step(self, turn: Turn) -> str: ...


@dataclass(frozen=True)
class LoopEnd:
    """How a loop ended: the value of the tool that ended it, how many
    steps ran, and whether the steps ran out with no end."""

    value: object
    steps: int
    out_of_steps: bool


class ToolLoop:
    def __init__(
        self,
        model: StepModel,
        tools: Sequence[Tool],
        goal: str,
        max_steps: int,
        observe: Callable[[], Awaitable[Observation]],
    ) -> None:
        if max_steps < 1:
            raise ValueError("a loop runs at least one step")
        self.model = model
        self.tools = tuple(tools)
        self.goal = goal
        self.max_steps = max_steps
        self.observe = observe
        self._by_name = _tools_by_name(self.tools)
        # Checked once: the schema is built from the tools' fields.
        answer_schema(self.tools)

    async def run(self, cancel: CancelToken) -> LoopEnd:
        history: list[str] = []
        for step in range(1, self.max_steps + 1):
            observation = await self.observe()
            cancel.check()
            turn = Turn(self.goal, self.tools, tuple(history), observation.state, observation.pictures, step, self.max_steps)
            text = await until(cancel.wait(), self.model.next_step(turn))
            cancel.check()
            assert text is not None
            tool, fields, problem = _read(text, self._by_name)
            if tool is None:
                self._record(history, step, f"{_shown(text)} -> {UNREADABLE}: {problem}")
                continue
            try:
                arguments = tool.parse(fields)
            except ValueError as error:
                self._record(history, step, f"{_shown(text)} -> {UNREADABLE}: {error}")
                continue
            result = await tool.run(arguments, cancel)
            cancel.check()
            self._record(history, step, f"{_shown(text)} -> {result.observation}")
            if result.ends:
                return LoopEnd(result.value, step, False)
        return LoopEnd(None, self.max_steps, True)

    def _record(self, history: list[str], step: int, line: str) -> None:
        """One line per step, in the history the model reads and in the
        log: the answer and what came of it."""
        history.append(line)
        logger.info("step %d of %d: %s", step, self.max_steps, line)


def answer_schema(tools: Sequence[Tool]) -> dict[str, Any]:
    """The flat schema of an answer, in the form `response_format` takes:
    `action` is one of the tools' names, and each tool's fields are
    optional properties beside it. A ValueError names a field two tools
    give different types."""
    properties: dict[str, Any] = {"action": {"type": "string", "enum": [tool.name for tool in tools]}}
    for tool in tools:
        for field, kind in tool.fields.items():
            if field == "action":
                raise ValueError(f"tool '{tool.name}' names a field 'action', the name of the tool itself")
            known = properties.get(field)
            if known is not None and known != {"type": kind}:
                raise ValueError(f"field '{field}' is a {known['type']} for one tool and a {kind} for '{tool.name}'")
            properties[field] = {"type": kind}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "step",
            "strict": False,
            "schema": {"type": "object", "properties": properties, "required": ["action"]},
        },
    }


def _tools_by_name(tools: Sequence[Tool]) -> dict[str, Tool]:
    by_name: dict[str, Tool] = {}
    for tool in tools:
        if tool.name in by_name:
            raise ValueError(f"two tools are named '{tool.name}'")
        by_name[tool.name] = tool
    return by_name


def _read(text: str, by_name: Mapping[str, Tool]) -> tuple[Optional[Tool], Mapping[str, object], str]:
    """The tool an answer names and the answer's other fields, or, for an
    answer the loop cannot read, no tool and why."""
    try:
        answer = json.loads(text)
    except json.JSONDecodeError as error:
        return None, {}, f"not JSON ({error.msg})"
    if not isinstance(answer, dict):
        return None, {}, "not a JSON object"
    action = answer.get("action")
    if not isinstance(action, str):
        return None, {}, "no action"
    tool = by_name.get(action)
    if tool is None:
        return None, {}, f"no tool is named '{action}'; the tools are {', '.join(by_name)}"
    return tool, {key: value for key, value in answer.items() if key != "action"}, ""


def _shown(text: str) -> str:
    """An answer on one line, at most ANSWER_CHARS long, for the history."""
    line = " ".join(text.split())
    return line if len(line) <= ANSWER_CHARS else line[: ANSWER_CHARS - 3] + "..."
