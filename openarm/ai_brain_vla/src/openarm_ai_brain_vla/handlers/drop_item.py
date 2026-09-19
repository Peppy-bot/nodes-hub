"""drop_item: open a gripper where it is and release what it holds."""

from __future__ import annotations

from ..ports import Refusal
from ..sequencer import MANIPULATION

ZERO = dict(gripper_name="", item_id="")


async def run(brain, ctx) -> None:
    goal = ctx.request().data

    async def body(job) -> dict:
        state = brain.state
        gripper = state.holder(goal.gripper_name)
        if not brain.manipulator.available:
            raise Refusal(f"no manipulation backend: manipulation_backend is '{brain.manipulator.name}'")
        outcome = await brain.manipulator.drop(gripper, job.cancel)
        job.cancel.check()
        if not outcome.success:
            raise Refusal(outcome.message or "the release failed")
        item_id = state.clear_held(gripper)
        return dict(gripper_name=gripper.name, item_id=item_id)

    await brain.run_guarded(ctx, MANIPULATION, "drop_item", body, ZERO)
