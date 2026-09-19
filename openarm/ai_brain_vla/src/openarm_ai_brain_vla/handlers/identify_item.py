"""identify_item: the one item best matching a description, with an id
the manipulation actions accept."""

from __future__ import annotations

from ..ports import Refusal
from ..sequencer import PERCEPTION
from ..state import best_match

ZERO = dict(item_id="", label="", position=[0.0, 0.0, 0.0], orientation=None, confidence=0.0)


async def run(brain, ctx) -> None:
    goal = ctx.request().data

    async def body(job) -> dict:
        description = goal.description.strip()
        phrases = [description] if description else []
        detections = await brain.perceiver.scan(phrases, job.cancel, goal.timeout_s)
        best = best_match(detections, description)
        if best is None:
            if description:
                raise Refusal(f"no item matches '{description}'")
            raise Refusal("no item in view")
        # A search is not a full scan: it refreshes what it found and
        # drops nothing, so the other items keep their ids.
        item = brain.state.remember([best], brain.now(), complete=False)[0]
        return dict(
            item_id=item.item_id,
            label=item.label,
            position=list(item.position),
            orientation=list(item.orientation) if item.orientation is not None else None,
            confidence=item.confidence,
        )

    await brain.run_guarded(ctx, PERCEPTION, "identify_item", body, ZERO)
