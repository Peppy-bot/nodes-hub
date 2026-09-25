"""Gemini Robotics-ER through Google's API: `perception_backend: "gemini_er"`.

The API comparison of the September 2026 perception study (yolo_world_eval,
row 7): asked one plain question per item the way identify_item asks, it
put 76.5% of the Waldo queries and 63.1% of the Isaac ones on the right
item, named the wrong item for about 1% and boxed an absent item for 2%,
and answered "not visible" for the rest, at about 1.8 s and $0.004 a
question at its low thinking level; the node asks at medium. It needs no GPU, no weights and no gallery: it understood "the
Spam" and "the jug" from the words alone. It needs the network, is billed
per question and returns no confidence. The study ranked it last for the
robot on those grounds; it stands here as the backend that works with
nothing set up.

The frame goes to the model as a JPEG with one prompt, in the format the
robotics spatial-reasoning guide documents: a JSON list of labels with
boxes normalised to 0-1000, `[]` when nothing asked for is visible. Three
prompts, chosen by the vocabulary the core sets:

- an identify search names one item: the study's question, "Find the
  mustard bottle", and every box that comes back is that item;
- a scan for named items lists them: the study's list prompt, one call for
  all of them, boxes labelled outside the list dropped;
- a scan for everything asks for every object on the table, labelled in
  the model's own words. This is the one prompt the study did not run.

The model's JSON is irregular (stray brackets, a bare label, a point
instead of a box), so it is repaired the way the study repaired it. A
generative model gives no confidence: with `SAMPLES` answers a frame, boxes
of one label overlapping at IoU 0.5 across answers are one item whose
confidence is the share of answers that returned it, so one answer, the
study's setting, makes every box confidence 1.

The API key is read from `KEY_ENV` in the node's environment, else from
`KEY_FILE` under the daemon user's home, which the container sees. `peppy
stack launch` forwards the launching shell's environment to the nodes it
starts on that machine, so an exported key reaches the node without
touching a launcher file; a machine the node is placed on holds the key
in the file. The key is never logged. Token usage and the list-price cost
of every call are logged so the bill is visible.

`perception_model` is the model id; empty is `MODEL`. Everything around
the API is plain code tested with a fake; google-genai is imported by
`load` alone.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

from ..ports import Box

logger = logging.getLogger(__name__)

MODEL = "gemini-robotics-er-2-preview"
# One level above the study's "low", the cheapest that answered with boxes:
# more thought tokens a question, so more cost and latency than the study
# measured, for answers that reason longer about what is in view.
THINKING = "medium"
KEY_ENV = "GEMINI_API_KEY"
KEY_FILE = Path("~/.config/openarm_ai_brain_vla/gemini_api_key")
# Answers a frame is asked for; the study's identify runs used one.
SAMPLES = 1
MATCH_IOU = 0.5
# One call must fit inside the core's search timeout (10 s) with a retry.
CALL_TIMEOUT_S = 8.0
ATTEMPTS = 2
RETRY_WAIT_S = 1.0
# The frame travels as a JPEG: a tenth of the PNG the study sent, no
# visible difference at this quality.
JPEG_QUALITY = 95
# List prices per million tokens at the time of the study; thoughts bill as
# output. Only for the cost the log shows.
PRICE_IN_PER_M = 2.00
PRICE_OUT_PER_M = 10.00

BOX_KEYS = ("y", "x", "y2", "x2")

IDENTIFY = "identify"
LIST = "list"
OPEN = "open"


def identify_prompt(phrase: str) -> str:
    """The study's question for one item; every box in the answer is it."""
    return (
        f"Find {phrase} in this image and return its bounding box.\n"
        "The answer should follow the JSON format:\n"
        f'[{{"label": "{phrase}", "y": <y_min>, "x": <x_min>, "y2": <y_max>, "x2": <x_max>}}]\n'
        "where coordinates are normalized to 0-1000. One entry per instance you can see. "
        "Return bounding boxes only, never points. Return [] if it is not visible in this image."
    )


def list_prompt(phrases: Sequence[str]) -> str:
    """The study's list prompt: boxes of the named items only, in one call."""
    labels = ", ".join(f'"{p}"' for p in phrases)
    return (
        "Detect every object in this image that is one of the following labels and return bounding boxes.\n"
        f"Allowed labels, to be used exactly as written: {labels}.\n"
        "The answer should follow the JSON format:\n"
        '[{"label": <label>, "y": <y_min>, "x": <x_min>, "y2": <y_max>, "x2": <x_max>}, ...]\n'
        "where coordinates are normalized to 0-1000. One entry per visible object. "
        "Return [] if none of the listed objects is visible. Do not include objects that are not in the list."
    )


def open_prompt() -> str:
    """Every object in front of the robot, named in the model's own words:
    what a scan with no vocabulary asks. Not a prompt the study measured."""
    return (
        "Detect every distinct object resting on the table or floor in front of the robot and return bounding boxes.\n"
        'Label each one with two or three plain English words naming what it is, such as "red mug" or "cardboard box".\n'
        "The answer should follow the JSON format:\n"
        '[{"label": <label>, "y": <y_min>, "x": <x_min>, "y2": <y_max>, "x2": <x_max>}, ...]\n'
        "where coordinates are normalized to 0-1000. One entry per object. "
        "Leave out the robot, its arms and grippers, the table and the floor. Return [] if there is no object."
    )


@dataclass(frozen=True)
class Plan:
    """One search: the prompt, and how the answer's labels are read. With
    `force_label` every box is that item; with `allowed` a box keeps the
    listed phrase it names and any other is dropped; with neither the
    model's own label stands."""

    mode: str
    prompt: str
    allowed: Optional[tuple[str, ...]] = None
    force_label: Optional[str] = None


def plan_for(phrases: Sequence[str]) -> Plan:
    wanted = [p.strip() for p in phrases if p.strip()]
    if not wanted:
        return Plan(OPEN, open_prompt())
    if len(wanted) == 1:
        return Plan(IDENTIFY, identify_prompt(wanted[0]), force_label=wanted[0])
    return Plan(LIST, list_prompt(wanted), allowed=tuple(wanted))


def api_key(env: str = KEY_ENV, file: Path = KEY_FILE) -> str:
    """The key from the environment, else from the file; empty when neither
    has one."""
    key = os.environ.get(env, "").strip()
    if key:
        return key
    path = file.expanduser()
    if path.is_file():
        return path.read_text().strip()
    return ""


def jpeg_bytes(image: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(out, format="JPEG", quality=quality)
    return out.getvalue()


def parse_boxes(
    text: str,
    width: int,
    height: int,
    allowed: Optional[Sequence[str]] = None,
    force_label: Optional[str] = None,
) -> tuple[list[tuple[str, np.ndarray]], list[str]]:
    """(label, xyxy pixel box) per entry of the model's answer, and the
    labels it used that were not allowed. The study's repair: a fenced or
    trailing-text answer is decoded from its first bracket; when the label
    is known from the question, four numbers in any wrapping are a box."""
    body = text.strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body)
    try:
        items = json.loads(body)
    except json.JSONDecodeError:
        start = body.find("[")
        try:
            items, _ = json.JSONDecoder().raw_decode(body[start:]) if start >= 0 else (None, 0)
        except json.JSONDecodeError:
            items = None
        if items is None and force_label is not None:
            m = re.search(r'"y"\s*:\s*(\d+).*?"x"\s*:\s*(\d+).*?"y2"\s*:\s*(\d+).*?"x2"\s*:\s*(\d+)', body, re.S)
            if not m:
                m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", body)
            if m:
                items = [{"label": "", "y": m.group(1), "x": m.group(2), "y2": m.group(3), "x2": m.group(4)}]
        if items is None:
            return [], ["<unparseable>"]
    if isinstance(items, dict):
        items = [items]
    by_lower = {p.lower(): p for p in allowed} if allowed is not None else None
    boxes: list[tuple[str, np.ndarray]] = []
    unmatched: list[str] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if all(k in item for k in BOX_KEYS):
            try:
                y0, x0, y1, x1 = (float(item[k]) for k in BOX_KEYS)
            except (TypeError, ValueError):
                continue
        elif isinstance(item.get("box_2d"), list) and len(item["box_2d"]) == 4:
            y0, x0, y1, x1 = (float(v) for v in item["box_2d"])
        elif force_label is not None and len(item) == 1 and isinstance(next(iter(item.values())), list) and len(next(iter(item.values()))) == 4:
            y0, x0, y1, x1 = (float(v) for v in next(iter(item.values())))
        else:
            continue
        if force_label is not None:
            name = force_label
        elif by_lower is not None:
            name = by_lower.get(label.lower(), "")
            if not name:
                unmatched.append(label.lower())
                continue
        else:
            name = label.lower()
            if not name:
                continue
        box = np.array([x0 / 1000 * width, y0 / 1000 * height, x1 / 1000 * width, y1 / 1000 * height])
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            continue
        boxes.append((name, box))
    return boxes, unmatched


def iou(a: np.ndarray, b: np.ndarray) -> float:
    lt, rb = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    inter = float(np.clip(rb - lt, 0, None).prod())
    union = float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def merge_samples(per_sample: Sequence[Sequence[tuple[str, np.ndarray]]], iou_threshold: float = MATCH_IOU) -> list[Box]:
    """Boxes of one label overlapping across answers are one item: its box
    is their mean and its confidence the share of answers that returned it.
    Every cluster is kept, so two items under one label stay two; the
    core merges what overlaps."""
    n = len(per_sample)
    clusters: dict[str, list[dict]] = {}
    for s, boxes in enumerate(per_sample):
        for label, box in boxes:
            for c in clusters.setdefault(label, []):
                if s not in c["samples"] and iou(c["mean"], box) > iou_threshold:
                    c["members"].append(box)
                    c["samples"].add(s)
                    c["mean"] = np.mean(c["members"], axis=0)
                    break
            else:
                clusters[label].append({"members": [box], "samples": {s}, "mean": box.copy()})
    out: list[Box] = []
    for label, cs in clusters.items():
        for c in cs:
            x0, y0, x1, y1 = (float(v) for v in c["mean"])
            out.append(Box(label=label, confidence=len(c["samples"]) / n, x0=x0, y0=y0, x1=x1, y1=y1))
    return out


@dataclass(frozen=True)
class Answer:
    """One answer of the API: its text and what it cost."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    latency_s: float = 0.0

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * PRICE_IN_PER_M / 1e6 + (self.output_tokens + self.thought_tokens) * PRICE_OUT_PER_M / 1e6


class Asks(Protocol):
    def ask(self, image: bytes, prompt: str) -> Answer: ...


class GeminiApi:
    """The google-genai client behind `ask`. Imports the library at
    construction and nowhere else."""

    def __init__(self, key: str, model: str, thinking: str = THINKING, timeout_s: float = CALL_TIMEOUT_S) -> None:
        from google import genai
        from google.genai import errors, types

        self.types = types
        self.errors = errors
        self.model = model
        self.thinking = thinking
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))

    def ask(self, image: bytes, prompt: str) -> Answer:
        types = self.types
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking.upper()),
            response_mime_type="application/json",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = [types.Part.from_bytes(data=image, mime_type="image/jpeg"), prompt]
        for attempt in range(1, ATTEMPTS + 1):
            t0 = time.perf_counter()
            try:
                response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
            except self.errors.APIError as error:
                retry = error.code == 429 or 500 <= (error.code or 0) < 600
                if not retry or attempt == ATTEMPTS:
                    raise
                time.sleep(RETRY_WAIT_S)
                continue
            usage = response.usage_metadata
            return Answer(
                text=response.text or "",
                input_tokens=(usage.prompt_token_count or 0) if usage else 0,
                output_tokens=(usage.candidates_token_count or 0) if usage else 0,
                thought_tokens=(usage.thoughts_token_count or 0) if usage else 0,
                latency_s=time.perf_counter() - t0,
            )
        raise RuntimeError("unreachable")


class GeminiErDetector:
    name = "gemini_er"

    def __init__(self, api: Optional[Asks] = None) -> None:
        self._api = api
        self._vocabulary: list[str] = []
        self.model = MODEL
        self.spent_usd = 0.0
        self.calls = 0

    @property
    def available(self) -> bool:
        return self._api is not None

    def load(self, model: str) -> None:
        """Finds the key and opens the client. No key, or no library, fails
        the load with the reason every search is then refused with."""
        self.model = model.strip() or MODEL
        if self._api is not None:
            return
        key = api_key()
        if not key:
            raise RuntimeError(
                f"no Gemini API key: export {KEY_ENV} in the shell peppy launches from, or write it to {KEY_FILE}"
            )
        try:
            self._api = GeminiApi(key, self.model)
        except ImportError as error:
            raise RuntimeError(f"the gemini_er backend needs google-genai (the node's gemini extra): {error}") from error
        logger.info("gemini_er: %s, key from %s", self.model, KEY_ENV if os.environ.get(KEY_ENV, "").strip() else KEY_FILE)

    def set_vocabulary(self, phrases: Sequence[str]) -> None:
        self._vocabulary = [p.strip() for p in phrases if p.strip()]

    def detect(self, image: np.ndarray) -> list[Box]:
        if self._api is None:
            return []
        height, width = image.shape[:2]
        plan = plan_for(self._vocabulary)
        data = jpeg_bytes(image)
        per_sample: list[list[tuple[str, np.ndarray]]] = []
        for _ in range(SAMPLES):
            answer = self._api.ask(data, plan.prompt)
            self._account(plan, answer)
            boxes, unmatched = parse_boxes(answer.text, width, height, plan.allowed, plan.force_label)
            if unmatched:
                logger.info("gemini_er: labels outside the list dropped: %s", ", ".join(sorted(set(unmatched))))
            per_sample.append(boxes)
        return merge_samples(per_sample, MATCH_IOU)

    def _account(self, plan: Plan, answer: Answer) -> None:
        self.calls += 1
        self.spent_usd += answer.cost_usd
        logger.info(
            "gemini_er: %s call, %d in / %d out / %d thought tokens, %.2f s, about $%.4f, $%.3f over %d calls",
            plan.mode, answer.input_tokens, answer.output_tokens, answer.thought_tokens,
            answer.latency_s, answer.cost_usd, self.spent_usd, self.calls,
        )
