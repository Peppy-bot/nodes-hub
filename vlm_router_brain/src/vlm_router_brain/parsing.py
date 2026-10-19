"""A model's answer read against the schema it was asked for, then turned
into boxes in the picture's pixels.

Every route asks for `Answer` as its structured output, so an answer is
JSON of that shape or it is unreadable: `read_answer` validates it with
`Answer.model_validate_json` and raises `UnreadableAnswer` otherwise, with
no repair. `{"boxes": []}` is a valid answer, "nothing of what was asked
is visible", and an unreadable one fails the search: the robot never acts
on an answer the brain could not read.
"""

from __future__ import annotations

from typing import Optional, Sequence

from pydantic import BaseModel, ValidationError

from brain_core.ports import ExpectedFailure

UNREADABLE = "the model's answer could not be read"


class Box(BaseModel):
    label: str
    # [y_min, x_min, y_max, x_max], 0-1000.
    box_2d: list[int]


class Answer(BaseModel):
    boxes: list[Box]


class UnreadableAnswer(ExpectedFailure):
    """An answer that is not JSON of the `Answer` shape. The text goes to
    the log, never to the goal's result."""

    def __init__(self, text: str) -> None:
        super().__init__(UNREADABLE)
        self.text = text


def read_answer(text: str) -> Answer:
    try:
        answer = Answer.model_validate_json(text)
    except ValidationError:
        raise UnreadableAnswer(text) from None
    if any(len(box.box_2d) != 4 for box in answer.boxes):
        raise UnreadableAnswer(text)
    return answer


def yx_0_1000_to_pixels(box_2d: Sequence[int], width: int, height: int) -> Optional[tuple[float, float, float, float]]:
    """A yx_0_1000 box as x0, y0, x1, y1 in the pixels of a `width` x
    `height` picture, clipped to it; none when less than a pixel is left."""
    y0, x0, y1, x1 = box_2d
    left = min(max(x0 / 1000 * width, 0.0), width)
    top = min(max(y0 / 1000 * height, 0.0), height)
    right = min(max(x1 / 1000 * width, 0.0), width)
    bottom = min(max(y1 / 1000 * height, 0.0), height)
    if right - left < 1 or bottom - top < 1:
        return None
    return (left, top, right, bottom)


def labelled(answer: Answer, force_label: Optional[str]) -> list[tuple[str, Box]]:
    """Each box of the answer with the label it keeps. With `force_label`
    every box is that item; without it the model's own label stands, in
    lower case, and a box with no label is dropped."""
    if force_label is not None:
        return [(force_label, box) for box in answer.boxes]
    return [(box.label.strip().lower(), box) for box in answer.boxes if box.label.strip()]
