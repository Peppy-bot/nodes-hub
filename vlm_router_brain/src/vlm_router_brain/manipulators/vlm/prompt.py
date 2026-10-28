"""The question of one step of the alignment loop, rendered from the turn
the loop built: the goal, the state, the earlier steps and the actions as
JSON objects with their fields."""

from __future__ import annotations


def step_prompt(turn) -> str:
    """The question of one step of a tool loop (`loop.Turn`): the goal, the
    step's state, the earlier steps, and the actions as JSON objects with
    their fields, each with the words that say when to take it."""
    lines = [turn.goal, "", f"Now: step {turn.step} of {turn.max_steps}. {turn.state}", "", "Earlier steps:"]
    lines += list(turn.history) or ["none"]
    lines += ["", "Answer with one JSON object, one of:"]
    for tool in turn.tools:
        fields = "".join(f', "{name}": <{kind}>' for name, kind in tool.fields.items())
        lines.append(f'{{"action": "{tool.name}"{fields}}}')
        lines.append(f"  {tool.description}")
    return "\n".join(lines)
