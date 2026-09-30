"""scan_items: every item in view, each with an id and its region in the
picture the scan looked at, in one result."""

from __future__ import annotations

from ..sequencer import PERCEPTION

ZERO = dict(
    item_ids=[], labels=[], positions=[], confidences=[], regions=[], camera="", image_width=0, image_height=0,
    frame_timestamp=0.0,
)

# The region of an item the scan reports without having seen it.
NO_REGION = (0.0, 0.0, 0.0, 0.0)


async def run(brain, ctx) -> None:
    goal = ctx.request().data

    async def body(job) -> dict:
        look = await brain.perceiver.scan([], job.cancel, goal.timeout_s)
        items = brain.state.remember(look.detections, brain.now(), coverage=brain.perceiver.detector.scan_coverage())
        return dict(
            item_ids=[item.item_id for item in items],
            labels=[item.label for item in items],
            positions=[coordinate for item in items for coordinate in item.position],
            confidences=[item.confidence for item in items],
            regions=[corner for item in items for corner in (item.region or NO_REGION)],
            camera=brain.params.camera_name,
            image_width=look.image_width,
            image_height=look.image_height,
            frame_timestamp=look.frame_timestamp,
        )

    await brain.run_guarded(ctx, PERCEPTION, "scan_items", body, ZERO)
