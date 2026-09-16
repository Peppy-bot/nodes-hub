"""scan_items: every item in view, each with an id, in one result."""

from __future__ import annotations

from ..sequencer import PERCEPTION

ZERO = dict(item_ids=[], labels=[], positions=[], confidences=[])


async def run(brain, ctx) -> None:
    goal = ctx.request().data

    async def body(job) -> dict:
        detections = await brain.perceiver.scan([], job.cancel, goal.timeout_s)
        items = brain.state.remember(detections, brain.now(), complete=True)
        return dict(
            item_ids=[item.item_id for item in items],
            labels=[item.label for item in items],
            positions=[coordinate for item in items for coordinate in item.position],
            confidences=[item.confidence for item in items],
        )

    await brain.run_guarded(ctx, PERCEPTION, "scan_items", body, ZERO)
