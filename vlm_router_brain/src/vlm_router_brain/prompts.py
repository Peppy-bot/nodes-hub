"""The question each search asks a model, in the yx_0_1000 format: a JSON
object whose `boxes` each hold a label and `box_2d`, [y_min, x_min, y_max,
x_max] normalised to 0-1000, and `{"boxes": []}` when nothing asked for is
visible. The schema the answer is held to is `parsing.Answer`.

Two prompts, chosen by the phrases of the search. identify_item names one
item or none, and scan_items none, so a search names at most one phrase:

- an identify search names one item: "Find the mustard bottle", and every
  box that comes back is that item. This is the question the September
  2026 perception study measured;
- a scan, no phrase, asks for every object on the table, labelled in the
  model's own words.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

IDENTIFY = "identify"
SCAN = "scan"

ANSWER_FORMAT = '{"boxes": [{"label": <label>, "box_2d": [<y_min>, <x_min>, <y_max>, <x_max>]}, ...]}'
NOTHING = '{"boxes": []}'


def identify_prompt(phrase: str) -> str:
    """One item; every box in the answer is it."""
    return (
        f"Find {phrase} in this image and return its bounding box.\n"
        "The answer should follow the JSON format:\n"
        f'{{"boxes": [{{"label": "{phrase}", "box_2d": [<y_min>, <x_min>, <y_max>, <x_max>]}}]}}\n'
        "where coordinates are normalized to 0-1000. One entry per instance you can see. "
        f"Return bounding boxes only, never points. Return {NOTHING} if it is not visible in this image."
    )


def scan_prompt() -> str:
    """Every object in front of the robot, named in the model's own words."""
    return (
        "Detect every distinct object resting on the table or floor in front of the robot and return bounding boxes.\n"
        'Label each one with two or three plain English words naming what it is, such as "red mug" or "cardboard box".\n'
        "The answer should follow the JSON format:\n"
        f"{ANSWER_FORMAT}\n"
        "where coordinates are normalized to 0-1000. One entry per object. "
        f"Leave out the robot, its arms and grippers, the table and the floor. Return {NOTHING} if there is no object."
    )


@dataclass(frozen=True)
class Plan:
    """One search: the prompt, and how the answer's labels are read. With
    `force_label` every box is that item; without it the model's own label
    stands."""

    mode: str
    prompt: str
    force_label: Optional[str] = None


def plan_for(phrases: Sequence[str]) -> Plan:
    """The scan for no phrase, the identify search for one. More than one
    phrase is a ValueError: the router searches for one item at a time."""
    wanted = [p.strip() for p in phrases if p.strip()]
    if not wanted:
        return Plan(SCAN, scan_prompt())
    if len(wanted) > 1:
        named = ", ".join(f"'{p}'" for p in wanted)
        raise ValueError(f"vlm_router_brain searches for one item at a time; this search names {len(wanted)}: {named}")
    return Plan(IDENTIFY, identify_prompt(wanted[0]), force_label=wanted[0])
